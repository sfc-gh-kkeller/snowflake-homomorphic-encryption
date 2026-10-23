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
# # 03 · Packed (SIMD) mode: a whole column in one ciphertext
#
# BFV (n=4096) has 4,096 slots and CKKS (n=8192) has 4,096. Row mode puts one value in slot 0 of each ciphertext;
# **packed mode puts the entire column into a single ciphertext**, so storage and compute shrink by orders of
# magnitude. Every homomorphic operation then acts on all slots at once.
#
# Summing *inside* a ciphertext needs slot **rotations**, so packed keys export **public rotation (Galois)
# keys** to Snowflake: still no secret material, but more public key material than row mode (which uploads
# only the ~100-byte context). Packed keys are separate keys, so row-mode data never gets rotation keys.
#
# | | row mode (01/02) | packed mode (this notebook) |
# |---|---|---|
# | ciphertexts per column | one per row | one |
# | server-side material | public context | public context + public rotation keys |
# | per-row filters / GROUP BY | yes (plaintext dims) | no: slots are positional |

# %%
import json
import time

import numpy as np
import pandas as pd

from he_snowflake import data
from he_snowflake.config import get_session, load_config
from he_snowflake.crypto import decrypt_packed_sum, decrypt_vector
from he_snowflake.keys import load_all
from he_snowflake.verify import Checks

pd.set_option("display.float_format", "{:,.4f}".format)
cfg = load_config()
session = get_session(cfg)
keys = load_all(cfg.key_dir, max_packed_values=cfg.n_employees)
BFV_P, CKKS_P = keys["bfv_packed"], keys["ckks_packed"]
emp = data.employees(cfg.n_employees, cfg.seed).sort_values("emp_id")
N = len(emp)
checks = Checks()

# %% [markdown]
# ## 1. Storage: one ciphertext versus one per row

# %%
sizes = session.sql("""
SELECT 1 AS o, 'row mode: salary column'  AS layout, COUNT(*) AS ciphertexts, SUM(LENGTH(salary_ct)) AS bytes FROM EMPLOYEES_ENC
UNION ALL
SELECT 2, 'packed: salary column', COUNT(*), SUM(LENGTH(ct)) FROM PACKED_COLUMNS WHERE column_name = 'SALARY_USD'
UNION ALL
SELECT 3, 'packed: public rotation keys (bfv)', 1, LENGTH(rotation_keys) FROM HE_KEYS WHERE mode = 'packed' AND scheme = 'bfv'
UNION ALL
SELECT 4, 'packed: public rotation keys (ckks)', 1, LENGTH(rotation_keys) FROM HE_KEYS WHERE mode = 'packed' AND scheme = 'ckks'
ORDER BY o
""").to_pandas().drop(columns="O")
sizes["MB"] = sizes.BYTES / 1e6
sizes

# %% [markdown]
# ## 2. SUM and AVG of the whole column via rotations (BFV, exact)
# `HE_PACKED_SUM` runs log₂(window) rotate-and-add steps; the total lands in slot `window − 1`.

# %%
PACKED = """
FROM PACKED_COLUMNS p JOIN HE_KEYS k ON k.key_id = p.key_id WHERE p.column_name = '{col}'"""
blob = session.sql("SELECT HE_PACKED_SUM(k.context, k.rotation_keys, p.ct, p.n_values)"
                   + PACKED.format(col="SALARY_USD")).collect()[0][0]
total = decrypt_packed_sum(BFV_P, blob, N)
checks.check("packed SUM(salary)", total, emp.salary_usd.sum())
checks.check("packed AVG(salary)", total / N, emp.salary_usd.mean(), rel_tol=1e-12)

# %% [markdown]
# ## 3. Elementwise operations on all rows at once
# A single `HE_MUL_PLAIN` raises *every* salary by 3%; a single `HE_ADD` adds the bonus vector to the salary
# vector. The results stay packed: decrypt the vector, or sum it with `HE_PACKED_SUM`.

# %%
SQL = """
WITH s AS (SELECT k.context ctx, k.rotation_keys rk, p.ct, p.n_values n FROM PACKED_COLUMNS p
           JOIN HE_KEYS k ON k.key_id = p.key_id WHERE p.column_name = 'SALARY_USD'),
     b AS (SELECT p.ct FROM PACKED_COLUMNS p WHERE p.column_name = 'BONUS_USD')
SELECT HE_MUL_PLAIN(s.ctx, s.ct, 103)                         AS raised_x100,
       HE_ADD(s.ctx, s.ct, b.ct)                              AS total_comp,
       HE_PACKED_SUM(s.ctx, s.rk, HE_ADD(s.ctx, s.ct, b.ct), s.n) AS total_comp_sum
FROM s, b"""
r = session.sql(SQL).collect()[0]
raised = decrypt_vector(BFV_P, r["RAISED_X100"], N) / 100
comp = decrypt_vector(BFV_P, r["TOTAL_COMP"], N)
checks.check("elementwise salary*1.03, max abs error over all rows", np.abs(raised - emp.salary_usd * 1.03).max(), 0, abs_tol=1e-9)
checks.check("elementwise salary+bonus, max abs error", np.abs(comp - (emp.salary_usd + emp.bonus_usd)).max(), 0)
checks.check("packed SUM(salary+bonus)", decrypt_packed_sum(BFV_P, r["TOTAL_COMP_SUM"], N), (emp.salary_usd + emp.bonus_usd).sum())
pd.DataFrame({"emp_id": emp.emp_id.values[:5], "salary*1.03 (decrypted)": raised[:5],
              "salary+bonus (decrypted)": comp[:5]})

# %% [markdown]
# ## 4. Dot product with a plaintext vector (CKKS): FTE-weighted rate in one call
# The weights come from a plaintext column, aggregated in SQL into an ordered `ARRAY`; the rates stay encrypted.

# %%
SQL = """
WITH w AS (SELECT ARRAY_AGG(fte_pct / 100) WITHIN GROUP (ORDER BY emp_id) AS weights FROM EMPLOYEES_ENC)
SELECT HE_PACKED_DOT(k.context, k.rotation_keys, p.ct, w.weights) AS dot,
       HE_PACKED_SUM(k.context, k.rotation_keys, p.ct, p.n_values) AS total
FROM PACKED_COLUMNS p JOIN HE_KEYS k ON k.key_id = p.key_id, w
WHERE p.column_name = 'HOURLY_RATE'"""
r = session.sql(SQL).collect()[0]
dot = decrypt_packed_sum(CKKS_P, r["DOT"], N)
fte = emp.fte_pct / 100
checks.check("packed dot(fte, rate)", dot, (fte * emp.hourly_rate).sum(), rel_tol=1e-6)
checks.check("FTE-weighted avg rate", dot / fte.sum(), (fte * emp.hourly_rate).sum() / fte.sum(), rel_tol=1e-6)
checks.check("packed SUM(rate)", decrypt_packed_sum(CKKS_P, r["TOTAL"], N), emp.hourly_rate.sum(), rel_tol=1e-6)

# %% [markdown]
# ## 5. Encrypted × encrypted, elementwise: every employee's labor cost in one multiplication
# The product of two packed ciphertexts has three polynomials. Rotating it would need a relinearization key,
# so this demo decrypts the product vector instead of summing it server-side.

# %%
SQL = """
SELECT HE_MUL(k.context, r.ct, h.ct) AS labor
FROM PACKED_COLUMNS r JOIN PACKED_COLUMNS h ON h.column_name = 'HOURS_LOGGED'
JOIN HE_KEYS k ON k.key_id = r.key_id WHERE r.column_name = 'HOURLY_RATE'"""
labor = decrypt_vector(CKKS_P, session.sql(SQL).collect()[0][0], N)
plain = (emp.hourly_rate * emp.hours_logged).to_numpy()
checks.check("elementwise rate*hours, max relative error", np.max(np.abs(labor - plain) / plain), 0, abs_tol=1e-5)
checks.check("SUM(rate*hours) after client-side sum", labor.sum(), plain.sum(), rel_tol=1e-6)

# %% [markdown]
# ## 6. Row mode versus packed mode: same answer, different cost

# %%
t0 = time.time()
session.sql("SELECT HE_SUM(k.context, e.salary_ct) FROM EMPLOYEES_ENC e JOIN HE_KEYS k ON k.key_id = e.bfv_key_id").collect()
row_s = time.time() - t0
t0 = time.time()
session.sql("SELECT HE_PACKED_SUM(k.context, k.rotation_keys, p.ct, p.n_values)" + PACKED.format(col="SALARY_USD")).collect()
packed_s = time.time() - t0
pd.DataFrame({"mode": ["row: HE_SUM over N ciphertexts", "packed: HE_PACKED_SUM over 1 ciphertext"],
              "seconds (incl. round trip)": [row_s, packed_s]})

# %%
checks.summary()
