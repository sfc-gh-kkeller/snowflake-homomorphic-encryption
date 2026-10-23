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
# # 02 · Advanced encrypted analytics: products, statistics, SUMIF, ML scoring, time series
#
# Builds on the data loaded by notebook 01. CKKS ciphertexts here have **two multiplicative levels**
# (`q = [60, 35, 35, 60]`, scale 2^35), enough for products of three encrypted values or a degree-3
# polynomial. No relinearization key is uploaded: products stay as 3- or 4-polynomial ciphertexts, which
# Snowflake can still add up and the laptop can still decrypt.

# %%
import numpy as np
import json

import pandas as pd

from he_snowflake import data
from he_snowflake.config import get_session, load_config
from he_snowflake.keys import load_all
from he_snowflake.verify import Checks, run_encrypted

pd.set_option("display.float_format", "{:,.6f}".format)
cfg = load_config()
session = get_session(cfg)
keys = load_all(cfg.key_dir, max_packed_values=cfg.n_employees)
BFV, CKKS = keys["bfv"], keys["ckks"]
emp = data.employees(cfg.n_employees, cfg.seed)
rev = data.monthly_revenue(cfg.seed)
checks = Checks()
CKKS_TOL = 1e-5   # relative, depth <= 2

# %% [markdown]
# ## 1. Encrypted × encrypted: labor cost, SUMPRODUCT, hours-weighted average rate
# `labor = rate × hours` per row (both encrypted); the weighted average `Σ(rate·hours) / Σ hours` has an
# encrypted numerator *and* an encrypted denominator.

# %%
SQL = """
SELECT e.region,
       HE_SUM(k.context, HE_MUL(k.context, e.hourly_rate_ct, e.hours_ct)) AS labor_cost,
       HE_SUM(k.context, e.hours_ct)                                      AS hours
FROM EMPLOYEES_ENC e JOIN HE_KEYS k ON k.key_id = e.ckks_key_id
GROUP BY e.region ORDER BY e.region"""
lab, secs = run_encrypted(session, SQL, {"LABOR_COST": CKKS, "HOURS": CKKS})
lab = lab.set_index("REGION")
lab["weighted avg rate"] = lab.LABOR_COST / lab.HOURS
grp = emp.assign(l=emp.hourly_rate * emp.hours_logged).groupby("region")
lab["labor (plaintext)"] = grp.l.sum()
lab["weighted avg rate (plaintext)"] = grp.l.sum() / grp.hours_logged.sum()
checks.frame("labor cost = SUM(rate*hours)", lab, "LABOR_COST", "labor (plaintext)", rel_tol=CKKS_TOL)
checks.frame("hours-weighted avg rate", lab, "weighted avg rate", "weighted avg rate (plaintext)", rel_tol=CKKS_TOL)
print(f"query {secs:.1f}s")
lab

# %% [markdown]
# ## 2. Statistics from encrypted sufficient statistics
# Snowflake returns encrypted `Σx, Σx², Σy, Σy², Σxy` (x = hourly rate, y = performance score). Everything
# else is plaintext algebra on the decrypted sums: mean, variance, standard deviation, covariance, Pearson
# correlation, and the least-squares line `perf ≈ a + b·rate`.

# %%
SQL = """
SELECT COUNT(*) AS n,
       HE_SUM(k.context, e.hourly_rate_ct)                              AS sx,
       HE_SUM(k.context, HE_SQUARE(k.context, e.hourly_rate_ct))        AS sxx,
       HE_SUM(k.context, e.perf_score_ct)                               AS sy,
       HE_SUM(k.context, HE_SQUARE(k.context, e.perf_score_ct))         AS syy,
       HE_SUM(k.context, HE_MUL(k.context, e.hourly_rate_ct, e.perf_score_ct)) AS sxy
FROM EMPLOYEES_ENC e JOIN HE_KEYS k ON k.key_id = e.ckks_key_id"""
s, secs = run_encrypted(session, SQL, {c: CKKS for c in ("SX", "SXX", "SY", "SYY", "SXY")})
s = s.iloc[0]
n = s.N
var_x, var_y = s.SXX / n - (s.SX / n) ** 2, s.SYY / n - (s.SY / n) ** 2
cov = s.SXY / n - (s.SX / n) * (s.SY / n)
slope = cov / var_x
stats = {
    "mean(rate)": (s.SX / n, emp.hourly_rate.mean()),
    "var_pop(rate)": (var_x, emp.hourly_rate.var(ddof=0)),
    "var_samp(rate)": (var_x * n / (n - 1), emp.hourly_rate.var(ddof=1)),
    "stddev(rate)": (np.sqrt(var_x), emp.hourly_rate.std(ddof=0)),
    "covar_pop(rate, perf)": (cov, np.cov(emp.hourly_rate, emp.perf_score, ddof=0)[0, 1]),
    "corr(rate, perf)": (cov / np.sqrt(var_x * var_y), emp.hourly_rate.corr(emp.perf_score)),
    "regr_slope(perf ~ rate)": (slope, np.polyfit(emp.hourly_rate, emp.perf_score, 1)[0]),
    "regr_intercept(perf ~ rate)": (s.SY / n - slope * s.SX / n, np.polyfit(emp.hourly_rate, emp.perf_score, 1)[1]),
}
for label, (got, exp) in stats.items():
    checks.check(label, got, exp, rel_tol=1e-4)
print(f"query {secs:.1f}s")

# %% [markdown]
# ## 3. Encrypted SUMIF / COUNTIF: the condition itself is secret
# `remote_flag` is an **encrypted** 0/1 column. Multiplying by it keeps qualifying rows and zeroes the others,
# and Snowflake cannot tell which rows qualified. `office = 1 − flag` is computed homomorphically too.
#
# * COUNTIF(remote) = Σ flag
# * SUMIF(remote, rate) = Σ flag·rate  → average remote rate
# * SUMIF(remote, rate·hours) = Σ flag·rate·hours  (three encrypted factors, depth 2)

# %%
SQL = """
SELECT e.department,
       HE_SUM(k.context, e.remote_flag_ct)                                         AS remote_n,
       HE_SUM(k.context, HE_MUL(k.context, e.remote_flag_ct, e.hourly_rate_ct))    AS remote_rate_sum,
       HE_SUM(k.context, HE_MUL(k.context, HE_MUL(k.context, e.hourly_rate_ct, e.hours_ct), e.remote_flag_ct)) AS remote_labor,
       HE_SUM(k.context, HE_MUL(k.context, HE_ADD_PLAIN(k.context, HE_NEG(k.context, e.remote_flag_ct), 1),
                                           e.hourly_rate_ct))                      AS office_rate_sum
FROM EMPLOYEES_ENC e JOIN HE_KEYS k ON k.key_id = e.ckks_key_id
GROUP BY e.department ORDER BY e.department"""
sif, secs = run_encrypted(session, SQL, {c: CKKS for c in ("REMOTE_N", "REMOTE_RATE_SUM", "REMOTE_LABOR", "OFFICE_RATE_SUM")})
sif = sif.set_index("DEPARTMENT")
g = emp.assign(r=emp.remote_flag, rr=emp.remote_flag * emp.hourly_rate,
               rl=emp.remote_flag * emp.hourly_rate * emp.hours_logged,
               orr=(1 - emp.remote_flag) * emp.hourly_rate).groupby("department")
sif["avg remote rate"] = sif.REMOTE_RATE_SUM / sif.REMOTE_N
checks.frame("COUNTIF(remote)", sif.assign(t=g.r.sum()), "REMOTE_N", "t", abs_tol=1e-4)
checks.frame("SUMIF(remote, rate)", sif.assign(t=g.rr.sum()), "REMOTE_RATE_SUM", "t", rel_tol=CKKS_TOL, abs_tol=1e-3)
checks.frame("SUMIF(remote, rate*hours)", sif.assign(t=g.rl.sum()), "REMOTE_LABOR", "t", rel_tol=CKKS_TOL, abs_tol=1e-2)
checks.frame("SUMIF(office, rate)", sif.assign(t=g.orr.sum()), "OFFICE_RATE_SUM", "t", rel_tol=CKKS_TOL, abs_tol=1e-3)
print(f"query {secs:.1f}s")
sif

# %% [markdown]
# ## 4. ML scoring on encrypted features
# **Linear model** with plaintext weights (the model owner's weights could equally be encrypted):
# `score = 0.4·perf + 0.05·tenure + 0.002·rate − 1.5`. Per-row encrypted scores come back, and the laptop
# decrypts and ranks them.
#
# **Logistic risk** via a degree-3 sigmoid approximation, σ(z) ≈ 0.5 + 0.197z − 0.004z³ on z ∈ [−4, 4], with
# z = 2·(3 − perf). The affine map is folded into the polynomial's coefficients on the client, so Snowflake
# evaluates a single cubic in the encrypted `perf` (`HE_POLY`, depth 2).

# %%
W = {"perf": 0.4, "tenure": 0.05, "rate": 0.002, "bias": -1.5}
SQL = f"""
SELECT e.emp_id,
       HE_ADD_PLAIN(k.context,
           HE_ADD(k.context,
               HE_ADD(k.context, HE_MUL_PLAIN(k.context, e.perf_score_ct, {W['perf']}),
                                 HE_MUL_PLAIN(k.context, e.tenure_ct, {W['tenure']})),
               HE_MUL_PLAIN(k.context, e.hourly_rate_ct, {W['rate']})),
           {W['bias']}) AS score
FROM EMPLOYEES_ENC e JOIN HE_KEYS k ON k.key_id = e.ckks_key_id ORDER BY e.emp_id"""
sc, secs = run_encrypted(session, SQL, {"SCORE": CKKS})
plain_score = W["perf"] * emp.perf_score + W["tenure"] * emp.tenure_years + W["rate"] * emp.hourly_rate + W["bias"]
sc["plaintext"] = plain_score.values
checks.check("linear score, max abs error over all rows", (sc.SCORE - sc.plaintext).abs().max(), 0.0, abs_tol=1e-4)
top = sc.nlargest(5, "SCORE")
print(f"query {secs:.1f}s | top-5 by decrypted score == top-5 by plaintext score:",
      list(top.EMP_ID) == list(sc.nlargest(5, "plaintext").EMP_ID))
top

# %%
# sigmoid(z) ~ c0 + c1 z + c3 z^3 with z = A*perf + B  ->  expand into a cubic in perf (client side)
c0, c1, c3 = 0.5, 0.197, -0.004
A, B = -2.0, 6.0
poly = np.polynomial.Polynomial([c0, c1, 0.0, c3])(np.polynomial.Polynomial([B, A]))
coeffs = [float(c) for c in poly.coef]
print("coefficients sent to HE_POLY:", [round(c, 6) for c in coeffs])
SQL = """
SELECT e.department, COUNT(*) AS n,
       HE_SUM(k.context, HE_POLY(k.context, e.perf_score_ct, PARSE_JSON(?)::ARRAY)) AS risk_sum
FROM EMPLOYEES_ENC e JOIN HE_KEYS k ON k.key_id = e.ckks_key_id
GROUP BY e.department ORDER BY e.department"""
risk, secs = run_encrypted(session, SQL, {"RISK_SUM": CKKS}, params=[str(coeffs)])
risk = risk.set_index("DEPARTMENT")
risk["avg risk (approx sigmoid)"] = risk.RISK_SUM / risk.N
z = A * emp.perf_score + B
risk["avg risk (same poly, plaintext)"] = (c0 + c1 * z + c3 * z ** 3).groupby(emp.department).mean()
risk["avg risk (true sigmoid)"] = (1 / (1 + np.exp(-z))).groupby(emp.department).mean()
checks.frame("avg attrition risk (HE_POLY)", risk, "avg risk (approx sigmoid)", "avg risk (same poly, plaintext)", abs_tol=2e-3)
print(f"query {secs:.1f}s")
risk

# %% [markdown]
# ## 5. Time series: running totals, month-over-month deltas, moving sums
# * **YTD revenue** per region with the `HE_RUNNING_SUM` UDTF (`PARTITION BY region ORDER BY month`)
# * **MoM delta** = `HE_SUB(revenue, LAG(revenue))`: `LAG` moves ciphertexts between rows; Snowflake never sees values
# * **3-month moving sum** = `revenue + LAG(1) + LAG(2)`

# %%
SQL = """
SELECT r.region, t.row_key AS month, t.running_ct AS ytd
FROM REVENUE_MONTHLY_ENC r JOIN HE_KEYS k ON k.key_id = r.bfv_key_id,
     TABLE(HE_RUNNING_SUM(k.context, r.revenue_ct, TO_VARCHAR(r.month))
           OVER (PARTITION BY r.region ORDER BY r.month)) t
ORDER BY r.region, month"""
ytd, secs_ytd = run_encrypted(session, SQL, {"YTD": BFV})

SQL = """
WITH x AS (
  SELECT r.region, r.month, k.context AS ctx, r.revenue_ct AS cur,
         LAG(r.revenue_ct, 1) OVER (PARTITION BY r.region ORDER BY r.month) AS prev1,
         LAG(r.revenue_ct, 2) OVER (PARTITION BY r.region ORDER BY r.month) AS prev2
  FROM REVENUE_MONTHLY_ENC r JOIN HE_KEYS k ON k.key_id = r.bfv_key_id)
SELECT region, TO_VARCHAR(month) AS month, cur AS revenue,
       HE_SUB(ctx, cur, prev1)                      AS mom_delta,
       HE_ADD(ctx, HE_ADD(ctx, cur, prev1), prev2)  AS moving_3m
FROM x ORDER BY region, month"""
ts, secs = run_encrypted(session, SQL, {"REVENUE": BFV, "MOM_DELTA": BFV, "MOVING_3M": BFV})
ts = ts.merge(ytd, on=["REGION", "MONTH"])
ts["MoM %"] = ts.MOM_DELTA / (ts.REVENUE - ts.MOM_DELTA) * 100

p = rev.assign(MONTH=rev.month.astype(str), REGION=rev.region).sort_values(["REGION", "MONTH"])
g = p.groupby("REGION").revenue_usd
p["ytd"], p["mom"], p["m3"] = g.cumsum(), g.diff(), g.rolling(3).sum().reset_index(level=0, drop=True)
ts = ts.merge(p[["REGION", "MONTH", "ytd", "mom", "m3"]], on=["REGION", "MONTH"])
checks.frame("YTD revenue (UDTF)", ts.set_index(["REGION", "MONTH"]), "YTD", "ytd")
checks.frame("MoM delta (LAG)", ts.dropna(subset=["mom"]).set_index(["REGION", "MONTH"]), "MOM_DELTA", "mom")
checks.frame("3-month moving sum", ts.dropna(subset=["m3"]).set_index(["REGION", "MONTH"]), "MOVING_3M", "m3")
print(f"queries {secs_ytd:.1f}s + {secs:.1f}s")
ts[ts.REGION == "EMEA"][["MONTH", "REVENUE", "YTD", "MOM_DELTA", "MoM %", "MOVING_3M"]]

# %% [markdown]
# ## 6. What the server can learn: structure, never values
# `HE_INFO` exposes what any server-side code can see: ciphertext size, polynomial count, level and scale. It
# depends on *which operations ran*, not on the data.

# %%
info = session.sql("""
SELECT 1 AS step, 'fresh rate'                AS expr, HE_INFO(k.context, e.hourly_rate_ct) AS info
FROM EMPLOYEES_ENC e JOIN HE_KEYS k ON k.key_id = e.ckks_key_id WHERE e.emp_id = 1
UNION ALL
SELECT 2, 'rate * 1.05', HE_INFO(k.context, HE_MUL_PLAIN(k.context, e.hourly_rate_ct, 1.05))
FROM EMPLOYEES_ENC e JOIN HE_KEYS k ON k.key_id = e.ckks_key_id WHERE e.emp_id = 1
UNION ALL
SELECT 3, 'rate * hours', HE_INFO(k.context, HE_MUL(k.context, e.hourly_rate_ct, e.hours_ct))
FROM EMPLOYEES_ENC e JOIN HE_KEYS k ON k.key_id = e.ckks_key_id WHERE e.emp_id = 1
UNION ALL
SELECT 4, 'rate * hours * flag', HE_INFO(k.context, HE_MUL(k.context, HE_MUL(k.context, e.hourly_rate_ct, e.hours_ct), e.remote_flag_ct))
FROM EMPLOYEES_ENC e JOIN HE_KEYS k ON k.key_id = e.ckks_key_id WHERE e.emp_id = 1
UNION ALL
SELECT 5, 'cubic poly(perf)', HE_INFO(k.context, HE_POLY(k.context, e.perf_score_ct, [0.5, 0.1, 0.01, 0.001]))
FROM EMPLOYEES_ENC e JOIN HE_KEYS k ON k.key_id = e.ckks_key_id WHERE e.emp_id = 1
ORDER BY step
""").to_pandas()
pd.json_normalize(info.INFO.map(json.loads)).set_index(info.EXPR)[["polys", "level", "levels_left", "scale_bits", "bytes"]]

# %%
summary = checks.summary()
summary[["check", "decrypted", "plaintext", "rel_err", "ok"]]
