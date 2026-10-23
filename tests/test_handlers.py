"""Offline tests: every Snowflake handler, run locally exactly as the server runs it (public context only)."""

import pickle
import random

import numpy as np
import pytest

from he_snowflake import data, deploy
from he_snowflake import udf_handlers as h
from he_snowflake.crypto import bfv_range, decrypt, decrypt_packed_sum, decrypt_vector, encrypt, encrypt_vector, noise_budget

CKKS_REL = 1e-5      # depth <= 2 at scale 2^35


def ctx(key):
    return key.context_bytes()


def close(got, exp, rel=CKKS_REL, abs_=1e-6):
    assert abs(got - exp) <= max(abs_, rel * abs(exp)), f"{got} vs {exp}"


# ----------------------------------------------------------------------------- server sees public material only

def test_server_context_has_no_keys(keys):
    he = h._he(ctx(keys["ckks"]))
    assert he.is_secret_key_empty() and he.is_public_key_empty()
    assert len(ctx(keys["bfv"])) < 200 and len(ctx(keys["ckks"])) < 200


# ----------------------------------------------------------------------------- BFV: exact integers

def test_bfv_add_sub_neg(keys):
    k = keys["bfv"]
    a, b = encrypt(k, 187_500), encrypt(k, 42_310)
    assert decrypt(k, h.he_add(ctx(k), a, b)) == 229_810
    assert decrypt(k, h.he_sub(ctx(k), a, b)) == 145_190
    assert decrypt(k, h.he_sub(ctx(k), b, a)) == -145_190
    assert decrypt(k, h.he_neg(ctx(k), a)) == -187_500


def test_bfv_plain_ops_and_linear_combination(keys):
    k = keys["bfv"]
    s, bonus = encrypt(k, 150_000), encrypt(k, 12_000)
    assert decrypt(k, h.he_add_plain(ctx(k), s, -1_200)) == 148_800
    assert decrypt(k, h.he_mul_plain(ctx(k), s, 103)) == 15_450_000
    # employer cost in cents-of-a-dollar units: 108*salary + 100*bonus + 120000
    lin = h.he_add_plain(ctx(k), h.he_add(ctx(k), h.he_mul_plain(ctx(k), s, 108), h.he_mul_plain(ctx(k), bonus, 100)), 120_000)
    assert decrypt(k, lin) == 108 * 150_000 + 100 * 12_000 + 120_000
    assert noise_budget(k, lin) > 0


def test_bfv_overflow_wraps(keys):
    k = keys["bfv"]
    t_half = bfv_range(k)
    big = h.he_mul_plain(ctx(k), encrypt(k, 10_000_000), 1_000)   # 1e10 > t/2 ~ 8.6e9
    assert decrypt(k, big) != 10_000_000_000
    assert abs(decrypt(k, big)) <= t_half


def test_bfv_poly_degree1_only(keys):
    k = keys["bfv"]
    x = encrypt(k, 500)
    assert decrypt(k, h.he_poly(ctx(k), x, [7, 3])) == 1_507
    with pytest.raises(ValueError):
        h.he_poly(ctx(k), x, [0, 1, 1])


def test_multiply_by_zero_is_rejected(keys):
    for name in ("bfv", "ckks"):
        k = keys[name]
        with pytest.raises(ValueError):
            h.he_mul_plain(ctx(k), encrypt(k, 5), 0)


# ----------------------------------------------------------------------------- CKKS: approximate reals

def test_ckks_basic_ops(keys):
    k = keys["ckks"]
    a, b = encrypt(k, 112.37), encrypt(k, 37.5)
    close(decrypt(k, h.he_add(ctx(k), a, b)), 149.87)
    close(decrypt(k, h.he_sub(ctx(k), a, b)), 74.87)
    close(decrypt(k, h.he_add_plain(ctx(k), a, 0.13)), 112.5)
    close(decrypt(k, h.he_mul_plain(ctx(k), a, 1.05)), 112.37 * 1.05)
    close(decrypt(k, h.he_mul(ctx(k), a, b)), 112.37 * 37.5)
    close(decrypt(k, h.he_square(ctx(k), a)), 112.37 ** 2)


def test_ckks_depth_two_chains(keys):
    k = keys["ckks"]
    rate, hours, flag = encrypt(k, 142.5), encrypt(k, 38.25), encrypt(k, 1.0)
    p = h.he_mul(ctx(k), h.he_mul(ctx(k), rate, hours), flag)          # level 2
    close(decrypt(k, p), 142.5 * 38.25)
    # level is restored after deserialization and exposed via HE_INFO
    info = h.he_info(ctx(k), p)
    assert info["level"] == 2 and info["levels_left"] == 0 and info["polys"] == 4
    # mixing levels: (rate*hours) * 0.9 then + rate (level 0) still aligns
    q = h.he_mul_plain(ctx(k), h.he_mul(ctx(k), rate, hours), 0.9)
    close(decrypt(k, q), 142.5 * 38.25 * 0.9)


@pytest.mark.parametrize("coeffs", [[0.5, 0.197, 0.0, -0.004], [1.0, 0.0, 2.0], [0.0, 0.0, 0.0, 1.0], [3.0, -2.0]])
def test_ckks_poly(keys, coeffs):
    # depth-2 CKKS at scale 2^35: ~1e-4..1e-3 absolute noise on O(1..64) results (documented in README)
    k = keys["ckks"]
    for v in (-4.0, -0.7, 0.0, 1.3, 4.0):
        exp = sum(c * v ** i for i, c in enumerate(coeffs))
        close(decrypt(k, h.he_poly(ctx(k), encrypt(k, v), coeffs)), exp, rel=1e-4, abs_=2e-3)


# ----------------------------------------------------------------------------- UDAF + UDTF semantics

def _simulate_udaf(ctx_b, blobs, n_workers=4, seed=0):
    """Mimic Snowflake: rows split across workers, states pickled, merged in arbitrary order."""
    rng = random.Random(seed)
    parts = [[] for _ in range(n_workers)]
    for b in blobs:
        parts[rng.randrange(n_workers)].append(b)
    states = []
    for part in parts:
        agg = h.HeSum()
        for b in part:
            agg.accumulate(ctx_b, b)
        states.append(pickle.loads(pickle.dumps(agg.aggregate_state)))
    rng.shuffle(states)
    final = h.HeSum()
    for st in states:
        final.merge(st)
    return final.finish()


def test_he_sum_exact_bfv(keys):
    k = keys["bfv"]
    vals = np.random.default_rng(1).integers(55_000, 250_000, 60)
    blobs = [encrypt(k, v) for v in vals] + [None]
    assert decrypt(k, _simulate_udaf(ctx(k), blobs)) == int(vals.sum())


def test_he_sum_ckks_products(keys):
    k = keys["ckks"]
    rng = np.random.default_rng(2)
    r, hrs = rng.uniform(25, 180, 40), rng.uniform(10, 60, 40)
    blobs = [h.he_mul(ctx(k), encrypt(k, a), encrypt(k, b)) for a, b in zip(r, hrs)]
    close(decrypt(k, _simulate_udaf(ctx(k), blobs)), float((r * hrs).sum()))


def test_he_sum_empty_and_mixed_keys(keys):
    assert h.HeSum().finish() is None
    agg = h.HeSum()
    agg.accumulate(ctx(keys["bfv"]), encrypt(keys["bfv"], 1))
    with pytest.raises(ValueError):
        agg.accumulate(ctx(keys["ckks"]), encrypt(keys["ckks"], 1.0))


def test_running_sum_udtf(keys):
    k = keys["bfv"]
    vals = [100, -40, 250, 7]
    udtf = h.HeRunningSum()
    out = [row for i, v in enumerate(vals) for row in udtf.process(ctx(k), encrypt(k, v), f"m{i}")]
    assert [r[0] for r in out] == ["m0", "m1", "m2", "m3"]
    assert [decrypt(k, r[1]) for r in out] == list(np.cumsum(vals))


# ----------------------------------------------------------------------------- packed (SIMD)

def test_packed_sum_bfv(keys):
    k = keys["bfv_packed"]
    vals = np.random.default_rng(3).integers(55_000, 250_000, 100)
    ct = encrypt_vector(k, vals)
    s = h.he_packed_sum(ctx(k), k.rotation_key_bytes(), ct, len(vals))
    assert decrypt_packed_sum(k, s, len(vals)) == int(vals.sum())
    raised = h.he_mul_plain(ctx(k), ct, 103)                              # elementwise on all slots
    assert list(decrypt_vector(k, raised, 5)) == list(vals[:5] * 103)


def test_packed_dot_ckks(keys):
    k = keys["ckks_packed"]
    rng = np.random.default_rng(4)
    x, w = rng.uniform(25, 180, 100), rng.uniform(0, 1, 100)
    ct = encrypt_vector(k, x)
    close(decrypt_packed_sum(k, h.he_packed_dot(ctx(k), k.rotation_key_bytes(), ct, list(w)), 100), float(x @ w))
    close(decrypt_packed_sum(k, h.he_packed_sum(ctx(k), k.rotation_key_bytes(), ct, 100), 100), float(x.sum()))


def test_packed_requires_rotation_keys_and_ckks_for_dot(keys):
    kb, kc = keys["bfv_packed"], keys["ckks_packed"]
    with pytest.raises(ValueError):
        h.he_packed_dot(ctx(kb), kb.rotation_key_bytes(), encrypt_vector(kb, [1, 2, 3]), [1, 1, 1])
    assert keys["bfv"].rotation_key_bytes() is None and kc.rotation_key_bytes() is not None
    a, b = encrypt_vector(kc, [1.0, 2.0]), encrypt_vector(kc, [3.0, 4.0])
    with pytest.raises(ValueError):                                       # 3-poly ct x ct cannot be rotated
        h.he_packed_sum(ctx(kc), kc.rotation_key_bytes(), h.he_mul(ctx(kc), a, b), 2)


# ----------------------------------------------------------------------------- deploy + data plumbing

def test_every_deployed_handler_exists():
    for _, (_, _, handler) in deploy.SCALAR_FUNCTIONS.items():
        assert callable(getattr(h, handler))
    ddl = deploy.function_ddl()
    assert len(ddl) == len(deploy.SCALAR_FUNCTIONS) + 2
    assert all("pyfhel==3.4.2" in s and "numpy==1.26.4" in s for s in ddl)
    assert len(deploy.table_ddl()) == 5


def test_data_is_deterministic():
    a, b = data.employees(30, 7), data.employees(30, 7)
    assert a.equals(b)
    assert set(a.remote_flag.unique()) <= {0, 1}
    assert a.salary_usd.sum() * 103 < 8.5e9                              # raise demo stays inside BFV range
