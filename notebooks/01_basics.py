# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 · Encrypted analytics basics: Snowflake computes, only you can read
#
# ```
#  LAPTOP (data owner)                                   SNOWFLAKE (standard warehouse, Anaconda pyfhel 3.4.2)
#  ┌──────────────────────────────┐   ciphertexts      ┌───────────────────────────────────────────────┐
#  │ keys/  secret keys (local)   │ ─────────────────▶ │ EMPLOYEES_ENC   plaintext dims + BINARY cts    │
#  │ encrypt salary, bonus, …     │   public params    │ HE_KEYS         public parameters only         │
#  │                              │ ─────────────────▶ │ HE_SUM · HE_ADD · HE_SUB · HE_MUL_PLAIN · …    │
#  │ decrypt results + verify     │ ◀───────────────── │ GROUP BY / WHERE on plaintext columns          │
#  └──────────────────────────────┘ encrypted results  └───────────────────────────────────────────────┘
# ```
#
# | Scheme | Columns | Arithmetic |
# |---|---|---|
# | **BFV** | `salary`, `bonus`, `deductions` (USD) | exact integers; results must stay within ±t/2 ≈ ±8.6·10⁹ |
# | **CKKS** | `hourly_rate`, `hours`, `perf_score`, `tenure`, `remote_flag` | approximate reals (~1e-6 relative at depth 1) |
#
# Every number below is computed by Snowflake **on ciphertexts**, returned **encrypted**, decrypted here,
# and checked against the plaintext ground truth, which never leaves the laptop.

# %%
import numpy as np
import pandas as pd

from he_snowflake import data
from he_snowflake.config import get_session, load_config
from he_snowflake.crypto import bfv_range, decrypt, noise_budget
from he_snowflake.deploy import deploy
from he_snowflake.keys import load_all
from he_snowflake.load import load
from he_snowflake.verify import Checks, run_encrypted

pd.set_option("display.float_format", "{:,.4f}".format)
cfg = load_config()
session = get_session(cfg, create_schema=True)
keys = load_all(cfg.key_dir, max_packed_values=cfg.n_employees)
BFV, CKKS = keys["bfv"], keys["ckks"]
emp = data.employees(cfg.n_employees, cfg.seed)      # plaintext ground truth (local only)
checks = Checks()
print(session.sql("SELECT CURRENT_ROLE() AS role, CURRENT_DATABASE() AS db, CURRENT_SCHEMA() AS schema").collect()[0])

# %% [markdown]
# ## 1. Deploy functions and load encrypted data
# `deploy` creates the tables and the homomorphic functions; `load` encrypts every sensitive value locally and
# uploads only ciphertexts plus public parameters. Set `LOAD = False` to reuse what is already in Snowflake.

# %%
LOAD = True
if LOAD:
    deploy(session)
    load(session, cfg, keys)
emp.head(5)

# %% [markdown]
# ## 2. What Snowflake actually holds

# %%
session.sql("""
SELECT emp_id, department, region,
       LENGTH(salary_ct)                            AS salary_ct_bytes,
       HEX_ENCODE(SUBSTR(salary_ct, 4097, 12))      AS salary_ct_sample,
       LENGTH(hourly_rate_ct)                       AS rate_ct_bytes,
       HEX_ENCODE(SUBSTR(hourly_rate_ct, 4097, 12)) AS rate_ct_sample
FROM EMPLOYEES_ENC ORDER BY emp_id LIMIT 5""").to_pandas()

# %%
session.sql("""
SELECT key_id, scheme, mode, LENGTH(context) AS public_context_bytes,
       COALESCE(LENGTH(rotation_keys), 0) AS public_rotation_key_bytes
FROM HE_KEYS ORDER BY key_id""").to_pandas()

# %% [markdown]
# Row-mode keys upload **only the ~100-byte public context** (ring size, moduli, scale). There is no secret key,
# no public key and no evaluation key in Snowflake: it can add and multiply ciphertexts, but it can neither
# decrypt nor even encrypt. Packed mode (notebook 03) additionally uploads public rotation keys.

# %% [markdown]
# ## 3. SUM, COUNT, AVG over whole encrypted columns

# %%
SQL = """
SELECT COUNT(*) AS n,
       HE_SUM(k.context, e.salary_ct)     AS salary_sum,
       HE_SUM(k.context, e.bonus_ct)      AS bonus_sum,
       HE_SUM(k.context, e.deductions_ct) AS deductions_sum
FROM EMPLOYEES_ENC e JOIN HE_KEYS k ON k.key_id = e.bfv_key_id"""
raw = session.sql(SQL).collect()[0]
print(f"encrypted total salary returned by Snowflake: {len(raw['SALARY_SUM']):,} bytes, "
      f"noise budget left {noise_budget(BFV, raw['SALARY_SUM'])} bits")
r, secs = run_encrypted(session, SQL, {"SALARY_SUM": BFV, "BONUS_SUM": BFV, "DEDUCTIONS_SUM": BFV})
r = r.iloc[0]
print(f"query {secs:.1f}s")
for col, plain in (("SALARY_SUM", "salary_usd"), ("BONUS_SUM", "bonus_usd"), ("DEDUCTIONS_SUM", "deductions_usd")):
    checks.check(f"SUM({plain})", r[col], emp[plain].sum())
checks.check("AVG(salary_usd)", r.SALARY_SUM / r.N, emp.salary_usd.mean(), rel_tol=1e-12)

# %% [markdown]
# ## 4. GROUP BY and ROLLUP on plaintext dimensions

# %%
SQL = """
SELECT COALESCE(e.department, '<ALL>') AS department, COUNT(*) AS headcount,
       HE_SUM(k.context, e.salary_ct) AS payroll
FROM EMPLOYEES_ENC e JOIN HE_KEYS k ON k.key_id = e.bfv_key_id
GROUP BY ROLLUP (e.department) ORDER BY department"""
dept, secs = run_encrypted(session, SQL, {"PAYROLL": BFV})
truth = emp.groupby("department").salary_usd.agg(["count", "sum"])
truth.loc["<ALL>"] = [len(emp), emp.salary_usd.sum()]
dept = dept.set_index("DEPARTMENT")
dept["payroll (plaintext)"] = truth["sum"]
dept["avg salary"] = dept.PAYROLL / dept.HEADCOUNT
dept["share of total"] = dept.PAYROLL / dept.loc["<ALL>", "PAYROLL"]
checks.frame("payroll by department (ROLLUP)", dept, "PAYROLL", "payroll (plaintext)")
print(f"query {secs:.1f}s")
dept

# %% [markdown]
# ## 5. Plaintext filters: `WHERE` works on non-encrypted columns

# %%
SQL = """
SELECT COUNT(*) AS n, HE_SUM(k.context, e.salary_ct) AS payroll
FROM EMPLOYEES_ENC e JOIN HE_KEYS k ON k.key_id = e.bfv_key_id
WHERE e.region = 'EMEA' AND e.hire_year >= 2020"""
r, _ = run_encrypted(session, SQL, {"PAYROLL": BFV})
sub = emp[(emp.region == "EMEA") & (emp.hire_year >= 2020)]
checks.check("SUM(salary) WHERE region=EMEA AND hire_year>=2020", r.PAYROLL[0], sub.salary_usd.sum())
checks.check("COUNT(*) same filter", r.N[0], len(sub))

# %% [markdown]
# ## 6. Encrypted ± encrypted: gross pay and net pay
# `net = salary + bonus − deductions`, row by row on ciphertexts, then summed per department.

# %%
SQL = """
SELECT e.department,
       HE_SUM(k.context, HE_ADD(k.context, e.salary_ct, e.bonus_ct)) AS gross,
       HE_SUM(k.context, HE_SUB(k.context, HE_ADD(k.context, e.salary_ct, e.bonus_ct), e.deductions_ct)) AS net
FROM EMPLOYEES_ENC e JOIN HE_KEYS k ON k.key_id = e.bfv_key_id
GROUP BY e.department ORDER BY e.department"""
pay, secs = run_encrypted(session, SQL, {"GROSS": BFV, "NET": BFV})
pay = pay.set_index("DEPARTMENT")
pay["gross (plaintext)"] = (emp.salary_usd + emp.bonus_usd).groupby(emp.department).sum()
pay["net (plaintext)"] = (emp.salary_usd + emp.bonus_usd - emp.deductions_usd).groupby(emp.department).sum()
checks.frame("gross pay", pay, "GROSS", "gross (plaintext)")
checks.frame("net pay", pay, "NET", "net (plaintext)")
pay

# %% [markdown]
# ## 7. Constant arithmetic: deductions, sign flips, raises, linear combinations
# BFV is exact integer arithmetic, so percentages are applied in hundredths and divided after decryption:
# * flat deduction: `salary − 1,200` (health premium)
# * negation: `−deductions` (book deductions as a credit)
# * 3% raise + $1,000 bonus: `salary·103 + 1000·100` → `/100`
# * employer cost: `1.08·salary + bonus + 1,200` → `108·salary + 100·bonus + 120,000` → `/100`

# %%
SQL = """
SELECT HE_SUM(k.context, HE_ADD_PLAIN(k.context, e.salary_ct, -1200))                     AS after_premium,
       HE_SUM(k.context, HE_NEG(k.context, e.deductions_ct))                              AS neg_deductions,
       HE_SUM(k.context, HE_ADD_PLAIN(k.context, HE_MUL_PLAIN(k.context, e.salary_ct, 103), 100000)) AS raise_x100,
       HE_SUM(k.context, HE_ADD_PLAIN(k.context,
                         HE_ADD(k.context, HE_MUL_PLAIN(k.context, e.salary_ct, 108),
                                           HE_MUL_PLAIN(k.context, e.bonus_ct, 100)), 120000)) AS employer_cost_x100
FROM EMPLOYEES_ENC e JOIN HE_KEYS k ON k.key_id = e.bfv_key_id"""
r, secs = run_encrypted(session, SQL, {c: BFV for c in ("AFTER_PREMIUM", "NEG_DEDUCTIONS", "RAISE_X100", "EMPLOYER_COST_X100")})
r = r.iloc[0]
checks.check("SUM(salary - 1200)", r.AFTER_PREMIUM, (emp.salary_usd - 1200).sum())
checks.check("SUM(-deductions)", r.NEG_DEDUCTIONS, -emp.deductions_usd.sum())
checks.check("SUM(salary*1.03 + 1000)", r.RAISE_X100 / 100, (emp.salary_usd * 1.03 + 1000).sum(), abs_tol=1e-6)
checks.check("SUM(1.08*salary + bonus + 1200)", r.EMPLOYER_COST_X100 / 100,
             (1.08 * emp.salary_usd + emp.bonus_usd + 1200).sum(), abs_tol=1e-6)
print(f"query {secs:.1f}s | largest intermediate {r.EMPLOYER_COST_X100:,} vs BFV limit ±{bfv_range(BFV):,}")

# %% [markdown]
# ## 8. Weighted average with plaintext weights (FTE-weighted salary)
# `Σ(fte_pct · salary) / Σ fte_pct`: the weights are plaintext, the salaries stay encrypted.

# %%
SQL = """
SELECT SUM(e.fte_pct) AS w_sum,
       HE_SUM(k.context, HE_MUL_PLAIN(k.context, e.salary_ct, e.fte_pct)) AS weighted
FROM EMPLOYEES_ENC e JOIN HE_KEYS k ON k.key_id = e.bfv_key_id"""
r, _ = run_encrypted(session, SQL, {"WEIGHTED": BFV})
checks.check("FTE-weighted avg salary", r.WEIGHTED[0] / r.W_SUM[0],
             (emp.fte_pct * emp.salary_usd).sum() / emp.fte_pct.sum(), rel_tol=1e-12)

# %% [markdown]
# ## 9. CKKS decimals: averages, uplift, and currency conversion via a plaintext join

# %%
SQL = """
SELECT e.region, COUNT(*) AS n, MAX(f.usd_to_local) AS fx,
       HE_SUM(k.context, e.hourly_rate_ct)                                  AS rate_sum,
       HE_SUM(k.context, HE_MUL_PLAIN(k.context, e.hourly_rate_ct, 1.05))   AS uplift_sum,
       HE_SUM(k.context, HE_MUL_PLAIN(k.context, e.hourly_rate_ct, f.usd_to_local)) AS local_sum
FROM EMPLOYEES_ENC e
JOIN HE_KEYS k  ON k.key_id = e.ckks_key_id
JOIN FX_RATES f ON f.region = e.region
GROUP BY e.region ORDER BY e.region"""
fx, secs = run_encrypted(session, SQL, {"RATE_SUM": CKKS, "UPLIFT_SUM": CKKS, "LOCAL_SUM": CKKS})
fx = fx.set_index("REGION")
g = emp.groupby("region").hourly_rate
fx["avg rate USD"] = fx.RATE_SUM / fx.N
fx["avg rate +5%"] = fx.UPLIFT_SUM / fx.N
fx["avg rate local ccy"] = fx.LOCAL_SUM / fx.N
fx["avg rate local (plaintext)"] = g.mean() * pd.Series(data.FX_RATES)
checks.frame("avg hourly rate", fx.assign(t=g.mean()), "avg rate USD", "t", rel_tol=1e-6)
checks.frame("avg hourly rate +5%", fx.assign(t=g.mean() * 1.05), "avg rate +5%", "t", rel_tol=1e-6)
checks.frame("avg hourly rate in local currency", fx, "avg rate local ccy", "avg rate local (plaintext)", rel_tol=1e-6)
fx.drop(columns=["RATE_SUM", "UPLIFT_SUM", "LOCAL_SUM"])

# %% [markdown]
# ## 10. Limits: BFV wraps around beyond ±t/2
# Multiplying total payroll by 1,000 exceeds t/2. Snowflake happily computes it; the decrypted value is
# the true result modulo t. Keep BFV results inside the range, or move the computation to CKKS.

# %%
SQL = """
SELECT HE_MUL_PLAIN(MAX(k.context), HE_SUM(k.context, e.salary_ct), 1000) AS times_1000
FROM EMPLOYEES_ENC e JOIN HE_KEYS k ON k.key_id = e.bfv_key_id"""
r, _ = run_encrypted(session, SQL, {"TIMES_1000": BFV})
true = int(emp.salary_usd.sum()) * 1000
t = BFV.he.t
print(f"true {true:,} | decrypted {r.TIMES_1000[0]:,} | true mod t (signed) {((true + t // 2) % t) - t // 2:,} | limit ±{t // 2:,}")

# %%
checks.summary()
