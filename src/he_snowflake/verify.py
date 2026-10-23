"""Compare decrypted results with plaintext ground truth and keep a scoreboard for the notebooks."""

from __future__ import annotations

import time

import pandas as pd

from .crypto import decrypt


class Checks:
    """Collects (label, decrypted, expected, tolerance) results; `summary()` renders a table."""

    def __init__(self):
        self.rows = []

    def check(self, label: str, got, expected, rel_tol: float = 0.0, abs_tol: float = 0.0) -> None:
        got, expected = float(got), float(expected)
        err = abs(got - expected)
        ok = err <= max(abs_tol, rel_tol * abs(expected))
        self.rows.append({"check": label, "decrypted": got, "plaintext": expected, "abs_err": err,
                          "rel_err": err / abs(expected) if expected else err, "ok": ok})
        print(f"{'OK      ' if ok else 'MISMATCH'} {label}: decrypted={got:,.6f} plaintext={expected:,.6f} "
              f"abs_err={err:.2e}")

    def frame(self, label: str, df: pd.DataFrame, got_col: str, exp_col: str, rel_tol=0.0, abs_tol=0.0) -> None:
        for idx, r in df.iterrows():
            self.check(f"{label} [{idx}]", r[got_col], r[exp_col], rel_tol, abs_tol)

    @property
    def all_ok(self) -> bool:
        return all(r["ok"] for r in self.rows)

    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        print(f"{int(df.ok.sum())}/{len(df)} checks passed")
        return df


def run_encrypted(session, sql: str, decrypt_cols: dict, params=None):
    """Run SQL in Snowflake, then decrypt {column: HEKey} on the client. Returns (DataFrame, seconds)."""
    t0 = time.time()
    df = session.sql(sql, params=params).to_pandas()
    secs = time.time() - t0
    for col, key in decrypt_cols.items():
        df[col] = df[col].map(lambda b, k=key: decrypt(k, b))
    return df, secs
