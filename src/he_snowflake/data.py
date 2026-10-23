"""Deterministic synthetic data. The plaintext frames are the ground truth and never leave the laptop."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEPARTMENTS = {  # salary band (USD), headcount share
    "Engineering": ((120_000, 240_000), 0.35),
    "Sales": ((80_000, 190_000), 0.20),
    "Finance": ((90_000, 180_000), 0.15),
    "Marketing": ((70_000, 160_000), 0.15),
    "Support": ((55_000, 110_000), 0.15),
}
REGIONS = {"AMER": 0.5, "EMEA": 0.3, "APJ": 0.2}
FX_RATES = {"AMER": 1.0, "EMEA": 0.92, "APJ": 151.4}   # USD -> local currency (illustrative, plaintext)

# Encrypted columns and the scheme used for each
BFV_COLUMNS = ["salary_usd", "bonus_usd", "deductions_usd"]
CKKS_COLUMNS = ["hourly_rate", "hours_logged", "perf_score", "tenure_years", "remote_flag"]


def employees(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dept = rng.choice(list(DEPARTMENTS), n, p=[v[1] for v in DEPARTMENTS.values()])
    salary = np.array([rng.integers(*DEPARTMENTS[d][0]) for d in dept])
    hire_year = rng.integers(2012, 2026, n)
    rate = rng.uniform(25, 180, n).round(2)
    # performance correlates with hourly rate (so correlation/regression demos have signal)
    perf = (1 + 4 * np.clip(0.65 * (rate - 25) / 155 + 0.35 * rng.uniform(0, 1, n), 0, 1)).round(2)
    return pd.DataFrame({
        "emp_id": np.arange(1, n + 1),
        "department": dept,
        "region": rng.choice(list(REGIONS), n, p=list(REGIONS.values())),
        "hire_year": hire_year,
        "fte_pct": rng.choice([50, 60, 80, 100], n, p=[0.1, 0.1, 0.2, 0.6]),
        "salary_usd": salary,
        "bonus_usd": (salary * rng.uniform(0.03, 0.20, n)).astype(int),
        "deductions_usd": (salary * rng.uniform(0.08, 0.15, n)).astype(int) + 1_200,
        "hourly_rate": rate,
        "hours_logged": rng.uniform(10, 60, n).round(2),
        "perf_score": perf,
        "tenure_years": (2026 - hire_year + rng.uniform(0, 1, n)).round(1),
        "remote_flag": (rng.uniform(0, 1, n) < 0.4).astype(int),
    })


def monthly_revenue(seed: int, year: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    base = {"AMER": 4_200_000, "EMEA": 2_600_000, "APJ": 1_500_000}
    rows = []
    for region, b in base.items():
        for m in range(1, 13):
            season = 1 + 0.15 * np.sin((m - 3) / 12 * 2 * np.pi)
            rows.append({"region": region, "month": pd.Timestamp(year, m, 1).date(),
                         "revenue_usd": int(b * season * rng.uniform(0.85, 1.15))})
    return pd.DataFrame(rows)


def fx_rates() -> pd.DataFrame:
    return pd.DataFrame({"region": list(FX_RATES), "usd_to_local": list(FX_RATES.values())})
