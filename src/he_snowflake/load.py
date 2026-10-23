"""Encrypt the synthetic data on the client and load ciphertexts + public parameters into Snowflake.

    pixi run load
"""

from __future__ import annotations

import base64
import json
import time

import pandas as pd

from . import data
from .config import get_session, load_config
from .crypto import encrypt, encrypt_vector
from .keys import HEKey, load_all


def _b64(b: bytes | None):
    return None if b is None else base64.b64encode(b).decode()


def upload(session, df: pd.DataFrame, table: str, binary_cols: list[str], variant_cols: tuple = ()) -> int:
    """write_pandas sends Python bytes as VARIANT, so ship base64 text to a temp table and convert in SQL."""
    stage = f"{table}_STAGE"
    up = df.copy()
    up.columns = [c.upper() for c in up.columns]
    for c in binary_cols:
        up[c.upper()] = up[c.upper()].map(_b64)
    session.write_pandas(up, stage, auto_create_table=True, overwrite=True, table_type="temporary")
    exprs = []
    for c in up.columns:
        if c.lower() in binary_cols:
            exprs.append(f'TO_BINARY("{c}", \'BASE64\')')
        elif c.lower() in variant_cols:
            exprs.append(f'PARSE_JSON("{c}")')
        else:
            exprs.append(f'"{c}"')
    cols = ", ".join(up.columns)
    session.sql(f"INSERT INTO {table} ({cols}) SELECT {', '.join(exprs)} FROM {stage}").collect()
    session.sql(f"DROP TABLE IF EXISTS {stage}").collect()
    return len(df)


def register_keys(session, keys: dict[str, HEKey]) -> int:
    df = pd.DataFrame([{
        "key_id": k.key_id, "scheme": k.scheme, "mode": k.mode, "params": json.dumps(k.params),
        "context": k.context_bytes(), "rotation_keys": k.rotation_key_bytes(),
    } for k in keys.values()])
    session.sql("DELETE FROM HE_KEYS").collect()
    return upload(session, df, "HE_KEYS", ["context", "rotation_keys"], variant_cols=("params",))


def encrypt_employees(emp: pd.DataFrame, bfv: HEKey, ckks: HEKey) -> pd.DataFrame:
    out = emp[["emp_id", "department", "region", "hire_year", "fte_pct"]].copy()
    out["bfv_key_id"] = bfv.key_id
    for col, ct in (("salary_usd", "salary_ct"), ("bonus_usd", "bonus_ct"), ("deductions_usd", "deductions_ct")):
        out[ct] = emp[col].map(lambda v: encrypt(bfv, v))
    out["ckks_key_id"] = ckks.key_id
    for col, ct in (("hourly_rate", "hourly_rate_ct"), ("hours_logged", "hours_ct"), ("perf_score", "perf_score_ct"),
                    ("tenure_years", "tenure_ct"), ("remote_flag", "remote_flag_ct")):
        out[ct] = emp[col].map(lambda v: encrypt(ckks, v))
    return out


def encrypt_revenue(rev: pd.DataFrame, bfv: HEKey) -> pd.DataFrame:
    out = rev[["region", "month"]].copy()
    out["bfv_key_id"] = bfv.key_id
    out["revenue_ct"] = rev.revenue_usd.map(lambda v: encrypt(bfv, v))
    return out


def encrypt_packed(emp: pd.DataFrame, bfv_p: HEKey, ckks_p: HEKey) -> pd.DataFrame:
    emp = emp.sort_values("emp_id")
    rows = [("EMPLOYEES", "salary_usd", bfv_p), ("EMPLOYEES", "bonus_usd", bfv_p),
            ("EMPLOYEES", "hourly_rate", ckks_p), ("EMPLOYEES", "hours_logged", ckks_p)]
    return pd.DataFrame([{"table_name": t, "column_name": c.upper(), "key_id": k.key_id, "n_values": len(emp),
                          "ct": encrypt_vector(k, emp[c].to_numpy())} for t, c, k in rows])


def load(session, cfg, keys: dict[str, HEKey] | None = None, verbose: bool = True) -> dict:
    keys = keys or load_all(cfg.key_dir, max_packed_values=cfg.n_employees)
    emp, rev = data.employees(cfg.n_employees, cfg.seed), data.monthly_revenue(cfg.seed)
    stats = {}
    t0 = time.time()
    enc_emp = encrypt_employees(emp, keys["bfv"], keys["ckks"])
    enc_rev = encrypt_revenue(rev, keys["bfv"])
    enc_packed = encrypt_packed(emp, keys["bfv_packed"], keys["ckks_packed"])
    stats["encrypt_s"] = round(time.time() - t0, 1)
    ct_cols = [c for c in enc_emp.columns if c.endswith("_ct")]
    stats["employee_ciphertext_mb"] = round(sum(enc_emp[c].map(len).sum() for c in ct_cols) / 1e6, 1)

    t0 = time.time()
    register_keys(session, keys)
    session.sql("TRUNCATE TABLE EMPLOYEES_ENC").collect()
    upload(session, enc_emp, "EMPLOYEES_ENC", ct_cols)
    session.sql("TRUNCATE TABLE REVENUE_MONTHLY_ENC").collect()
    upload(session, enc_rev, "REVENUE_MONTHLY_ENC", ["revenue_ct"])
    session.sql("TRUNCATE TABLE PACKED_COLUMNS").collect()
    upload(session, enc_packed, "PACKED_COLUMNS", ["ct"])
    session.sql("TRUNCATE TABLE FX_RATES").collect()
    session.write_pandas(data.fx_rates().rename(columns=str.upper), "FX_RATES", overwrite=False)
    stats["upload_s"] = round(time.time() - t0, 1)
    stats["rows"] = {t: session.table(t).count() for t in
                     ("HE_KEYS", "EMPLOYEES_ENC", "REVENUE_MONTHLY_ENC", "PACKED_COLUMNS", "FX_RATES")}
    if verbose:
        print(json.dumps(stats, indent=2))
    return stats


def main():
    cfg = load_config()
    session = get_session(cfg)
    load(session, cfg)


if __name__ == "__main__":
    main()
