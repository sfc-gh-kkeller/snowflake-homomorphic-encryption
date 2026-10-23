"""Create the demo schema, tables and homomorphic functions in Snowflake.

    pixi run deploy                 # tables + functions
    pixi run deploy --functions     # functions only (keeps loaded data)
    pixi run clean-snowflake        # DROP the demo database (asks for confirmation)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ROOT, get_session, load_config

HANDLER_SOURCE = (Path(__file__).parent / "udf_handlers.py").read_text()
RUNTIME = """LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('pyfhel==3.4.2', 'numpy==1.26.4')"""

# name -> (signature, returns, handler)
SCALAR_FUNCTIONS = {
    "HE_ADD":        ("(ctx BINARY, a BINARY, b BINARY)", "BINARY", "he_add"),
    "HE_SUB":        ("(ctx BINARY, a BINARY, b BINARY)", "BINARY", "he_sub"),
    "HE_NEG":        ("(ctx BINARY, a BINARY)", "BINARY", "he_neg"),
    "HE_ADD_PLAIN":  ("(ctx BINARY, a BINARY, k FLOAT)", "BINARY", "he_add_plain"),
    "HE_MUL_PLAIN":  ("(ctx BINARY, a BINARY, k FLOAT)", "BINARY", "he_mul_plain"),
    "HE_MUL":        ("(ctx BINARY, a BINARY, b BINARY)", "BINARY", "he_mul"),
    "HE_SQUARE":     ("(ctx BINARY, a BINARY)", "BINARY", "he_square"),
    "HE_POLY":       ("(ctx BINARY, a BINARY, coeffs ARRAY)", "BINARY", "he_poly"),
    "HE_INFO":       ("(ctx BINARY, a BINARY)", "VARIANT", "he_info"),
    "HE_PACKED_SUM": ("(ctx BINARY, rot_keys BINARY, a BINARY, n_values INT)", "BINARY", "he_packed_sum"),
    "HE_PACKED_DOT": ("(ctx BINARY, rot_keys BINARY, a BINARY, weights ARRAY)", "BINARY", "he_packed_dot"),
}


def function_ddl() -> list[str]:
    body = f"AS $$\n{HANDLER_SOURCE}\n$$"
    ddl = [f"CREATE OR REPLACE FUNCTION {name}{sig}\nRETURNS {ret}\n{RUNTIME}\nHANDLER = '{handler}'\n{body}"
           for name, (sig, ret, handler) in SCALAR_FUNCTIONS.items()]
    ddl.append(f"CREATE OR REPLACE AGGREGATE FUNCTION HE_SUM(ctx BINARY, ct BINARY)\nRETURNS BINARY\n{RUNTIME}\n"
               f"HANDLER = 'HeSum'\n{body}")
    ddl.append(f"CREATE OR REPLACE FUNCTION HE_RUNNING_SUM(ctx BINARY, ct BINARY, row_key STRING)\n"
               f"RETURNS TABLE (row_key STRING, running_ct BINARY)\n{RUNTIME}\nHANDLER = 'HeRunningSum'\n{body}")
    return ddl


def table_ddl() -> list[str]:
    text = "\n".join(line for line in (ROOT / "sql" / "10_tables.sql").read_text().splitlines()
                     if not line.lstrip().startswith("--"))
    return [s.strip() for s in text.split(";") if s.strip()]


def deploy(session, tables: bool = True, functions: bool = True) -> list[str]:
    created = []
    if tables:
        for stmt in table_ddl():
            session.sql(stmt).collect()
        created.append("tables")
    if functions:
        for stmt in function_ddl():
            session.sql(stmt).collect()
        created.append("functions")
    return created


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--functions", action="store_true", help="only (re)create functions")
    ap.add_argument("--drop", action="store_true", help="drop the demo database")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt for --drop")
    args = ap.parse_args()
    cfg = load_config()
    if args.drop:
        if not args.yes and input(f"DROP DATABASE {cfg.database}? type the name to confirm: ") != cfg.database:
            print("aborted")
            return
        session = get_session(cfg)
        session.sql(f"DROP DATABASE IF EXISTS {cfg.database}").collect()
        print(f"dropped {cfg.database}")
        return
    session = get_session(cfg, create_schema=True)
    done = deploy(session, tables=not args.functions, functions=True)
    print(f"deployed {', '.join(done)} to {cfg.fq_schema}")
    print(session.sql("SHOW USER FUNCTIONS LIKE 'HE_%' IN SCHEMA").select('"name"', '"arguments"').to_pandas())


if __name__ == "__main__":
    main()
