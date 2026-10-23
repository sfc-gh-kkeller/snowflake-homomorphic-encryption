"""Demo configuration: config.toml (gitignored) -> config.example.toml, overridable by HE_* env vars."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    connection_name: str
    role: str
    warehouse: str
    database: str
    schema: str
    n_employees: int
    seed: int
    key_dir: Path

    @property
    def fq_schema(self) -> str:
        return f"{self.database}.{self.schema}"


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        path = ROOT / "config.toml"
        if not path.exists():
            path = ROOT / "config.example.toml"
    raw = tomllib.loads(Path(path).read_text())
    sf, demo = raw.get("snowflake", {}), raw.get("demo", {})

    def pick(env: str, value, cast=str):
        return cast(os.environ[env]) if os.environ.get(env) else cast(value)

    key_dir = Path(pick("HE_KEY_DIR", demo.get("key_dir", "keys")))
    return Config(
        connection_name=pick("HE_CONNECTION", sf.get("connection_name", "")),
        role=pick("HE_ROLE", sf.get("role", "")),
        warehouse=pick("HE_WAREHOUSE", sf.get("warehouse", "")),
        database=pick("HE_DATABASE", sf.get("database", "HE_DEMO")),
        schema=pick("HE_SCHEMA", sf.get("schema", "PYFHEL")),
        n_employees=pick("HE_N_EMPLOYEES", demo.get("n_employees", 100), int),
        seed=pick("HE_SEED", demo.get("seed", 42), int),
        key_dir=key_dir if key_dir.is_absolute() else ROOT / key_dir,
    )


def get_session(cfg: Config, create_schema: bool = False):
    """Snowpark session using the configured connection, role, warehouse and schema."""
    from snowflake.snowpark import Session

    builder = Session.builder
    if cfg.connection_name:
        builder = builder.config("connection_name", cfg.connection_name)
    session = builder.create()
    if cfg.role:
        session.use_role(cfg.role)
    if cfg.warehouse:
        session.use_warehouse(cfg.warehouse)
    if create_schema:
        session.sql(f"CREATE DATABASE IF NOT EXISTS {cfg.database}").collect()
        session.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.fq_schema}").collect()
    session.use_schema(cfg.fq_schema)
    return session
