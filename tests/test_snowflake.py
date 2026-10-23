"""Integration test against a live Snowflake account (opt-in: `pixi run test-snowflake`).

Deploys into <schema>_TEST with a small dataset so the demo schema is untouched, then checks a few
end-to-end queries (encrypted in Python -> computed in Snowflake -> decrypted in Python).
"""

import dataclasses

import pytest

from he_snowflake import data
from he_snowflake.config import get_session, load_config
from he_snowflake.crypto import decrypt, decrypt_packed_sum
from he_snowflake.deploy import deploy
from he_snowflake.keys import load_all
from he_snowflake.load import load

pytestmark = pytest.mark.snowflake


@pytest.fixture(scope="module")
def env():
    cfg = load_config()
    cfg = dataclasses.replace(cfg, schema=f"{cfg.schema}_TEST", n_employees=16)
    session = get_session(cfg, create_schema=True)
    deploy(session)
    keys = load_all(cfg.key_dir, max_packed_values=cfg.n_employees)
    load(session, cfg, keys, verbose=False)
    yield cfg, session, keys, data.employees(cfg.n_employees, cfg.seed)
    session.sql(f"DROP SCHEMA IF EXISTS {cfg.fq_schema}").collect()
    session.close()


def test_sum_net_pay_by_department(env):
    cfg, s, keys, emp = env
    df = s.sql("""
        SELECT e.department,
               HE_SUM(k.context, HE_SUB(k.context, HE_ADD(k.context, e.salary_ct, e.bonus_ct), e.deductions_ct)) AS net_ct
        FROM EMPLOYEES_ENC e JOIN HE_KEYS k ON k.key_id = e.bfv_key_id
        GROUP BY e.department""").to_pandas()
    truth = (emp.salary_usd + emp.bonus_usd - emp.deductions_usd).groupby(emp.department).sum()
    for _, r in df.iterrows():
        assert decrypt(keys["bfv"], r.NET_CT) == truth[r.DEPARTMENT]


def test_encrypted_sumif(env):
    cfg, s, keys, emp = env
    blob = s.sql("""
        SELECT HE_SUM(k.context, HE_MUL(k.context, HE_MUL(k.context, e.hourly_rate_ct, e.hours_ct), e.remote_flag_ct))
        FROM EMPLOYEES_ENC e JOIN HE_KEYS k ON k.key_id = e.ckks_key_id""").collect()[0][0]
    exp = float((emp.hourly_rate * emp.hours_logged * emp.remote_flag).sum())
    assert abs(decrypt(keys["ckks"], blob) - exp) <= 1e-4 * exp + 1e-3


def test_running_total_udtf(env):
    cfg, s, keys, _ = env
    rev = data.monthly_revenue(cfg.seed)
    df = s.sql("""
        SELECT r.region, t.row_key AS month, t.running_ct
        FROM REVENUE_MONTHLY_ENC r JOIN HE_KEYS k ON k.key_id = r.bfv_key_id,
             TABLE(HE_RUNNING_SUM(k.context, r.revenue_ct, TO_VARCHAR(r.month))
                   OVER (PARTITION BY r.region ORDER BY r.month)) t""").to_pandas()
    ytd = rev.assign(ytd=rev.groupby("region").revenue_usd.cumsum(), m=rev.month.astype(str))
    exp = {(r.region, r.m): r.ytd for r in ytd.itertuples()}
    assert len(df) == len(rev)
    for r in df.itertuples():
        assert decrypt(keys["bfv"], r.RUNNING_CT) == exp[(r.REGION, r.MONTH)]


def test_packed_sum(env):
    cfg, s, keys, emp = env
    blob = s.sql("""
        SELECT HE_PACKED_SUM(k.context, k.rotation_keys, p.ct, p.n_values)
        FROM PACKED_COLUMNS p JOIN HE_KEYS k ON k.key_id = p.key_id
        WHERE p.column_name = 'SALARY_USD'""").collect()[0][0]
    assert decrypt_packed_sum(keys["bfv_packed"], blob, cfg.n_employees) == int(emp.salary_usd.sum())
