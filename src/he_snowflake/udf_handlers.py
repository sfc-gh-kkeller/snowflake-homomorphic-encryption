"""Single source of truth for every Snowflake UDF / UDAF / UDTF handler.

`deploy.py` embeds this whole file as the inline handler of each function, and the offline tests import it
directly, so the code that runs in Snowflake is exactly the code that is unit-tested.

Server-side constraints this module respects:
* Only the public encryption context (and, for packed mode, public rotation keys) is ever available.
  No secret key, no public key -> decryption and fresh encryption are impossible by construction.
* Pyfhel keeps a ciphertext's modulus level on the Python object and loses it on deserialization, so
  `_ct` recovers it from the serialized size (SEAL stores one RNS limb per remaining prime).
* Some SEAL errors abort the process instead of raising, so known failure modes are checked up front.

Runtime: Snowflake Anaconda `pyfhel==3.4.2` (SEAL 4.1) with `numpy==1.26.4`. Imports must stay limited
to the standard library, numpy and Pyfhel.
"""

import hashlib

import numpy as np
from Pyfhel import Pyfhel, PyCtxt

_CACHE = {}


def _he(ctx, rot_keys=None):
    """Pyfhel instance holding only public material, cached per (context, rotation keys)."""
    ctx = bytes(ctx)
    rot = bytes(rot_keys) if rot_keys is not None else b""
    key = hashlib.sha256(ctx).digest() + hashlib.sha256(rot).digest()
    he = _CACHE.get(key)
    if he is None:
        he = Pyfhel()
        he.from_bytes_context(ctx)
        if rot:
            he.from_bytes_rotate_key(rot)
        _CACHE[key] = he
    return he


def _is_ckks(he):
    return he.scheme.name == "ckks"


def _ct(he, blob):
    """Deserialize a ciphertext and restore its CKKS modulus level from its size."""
    c = PyCtxt(pyfhel=he, bytestring=bytes(blob))
    if _is_ckks(he):
        limbs = round(c.sizeof_ciphertext("none") / (c.size() * he.n * 8))
        c.mod_level = (len(he.qi_sizes) - 1) - limbs
    return c


def _out(c):
    return c.to_bytes(compr_mode="zstd")


def _slots(he):
    return he.get_nSlots()


def _match_level(he, obj, ct):
    """Mod-switch a CKKS plaintext/ciphertext down to `ct`'s level (never touches scales)."""
    while obj.mod_level < ct.mod_level:
        he.mod_switch_to_next(obj)
    return obj


def _plain_at(he, ct, value, scale=None):
    """Plaintext constant broadcast to all slots, at `ct`'s level (CKKS: at `scale`, default ct.scale)."""
    if _is_ckks(he):
        pt = he.encodeFrac(np.full(_slots(he), float(value)), scale=ct.scale if scale is None else scale)
        return _match_level(he, pt, ct)
    return he.encodeInt(np.full(_slots(he), int(round(value)), dtype=np.int64))


def _require_nonzero(value, op):
    if value == 0:
        raise ValueError(f"{op}: multiplying by 0 yields a transparent ciphertext; handle zero factors in SQL")


def _mul_plain(he, ct, value, scale=None):
    """ct * constant. CKKS: constant encoded at `scale` (default: context scale), then rescaled."""
    _require_nonzero(value, "HE_MUL_PLAIN")
    if _is_ckks(he):
        pt = _plain_at(he, ct, value, scale=scale or _base_scale(ct))
        out = he.multiply_plain(ct, pt, in_new_ctxt=True)
        he.rescale_to_next(out)
        return out
    return he.multiply_plain(ct, _plain_at(he, ct, value), in_new_ctxt=True)


def _base_scale(ct):
    """Nominal power-of-two scale of a CKKS ciphertext (exact for fresh ciphertexts)."""
    return 2.0 ** round(float(np.log2(ct.scale)))


def _mul(he, a, b):
    """ct * ct without relinearization (result has one more polynomial).

    CKKS: levels are aligned by mod-switching only. Pyfhel's `*` would also round slightly different
    scales to the nearest power of two, which costs ~1e-5 relative precision; SEAL tracks the exact
    product scale itself, so no rounding is needed.
    """
    if not _is_ckks(he):
        return a * b
    a, b = PyCtxt(copy_ctxt=a), PyCtxt(copy_ctxt=b)
    _match_level(he, a, b)
    _match_level(he, b, a)
    out = he.multiply(a, b, in_new_ctxt=True)
    he.rescale_to_next(out)
    return out


# ----------------------------------------------------------------------------- scalar UDFs

def he_add(ctx, a, b):
    if ctx is None or a is None or b is None:
        return None
    he = _he(ctx)
    return _out(_ct(he, a) + _ct(he, b))


def he_sub(ctx, a, b):
    if ctx is None or a is None or b is None:
        return None
    if bytes(a) == bytes(b):
        raise ValueError("HE_SUB: a - a yields a transparent ciphertext")
    he = _he(ctx)
    return _out(_ct(he, a) - _ct(he, b))


def he_neg(ctx, a):
    if ctx is None or a is None:
        return None
    he = _he(ctx)
    return _out(-_ct(he, a))


def he_add_plain(ctx, a, k):
    if ctx is None or a is None or k is None:
        return None
    he = _he(ctx)
    c = _ct(he, a)
    return _out(he.add_plain(c, _plain_at(he, c, k), in_new_ctxt=True))


def he_mul_plain(ctx, a, k):
    if ctx is None or a is None or k is None:
        return None
    he = _he(ctx)
    return _out(_mul_plain(he, _ct(he, a), k))


def he_mul(ctx, a, b):
    if ctx is None or a is None or b is None:
        return None
    he = _he(ctx)
    return _out(_mul(he, _ct(he, a), _ct(he, b)))


def he_square(ctx, a):
    if ctx is None or a is None:
        return None
    he = _he(ctx)
    c = _ct(he, a)
    return _out(_mul(he, c, c))


def he_poly(ctx, a, coeffs):
    """c0 + c1*x + c2*x^2 + c3*x^3 on an encrypted x (CKKS, two levels; BFV: degree <= 1).

    All non-constant terms are brought to the same level *and the same exact scale* (s1^2 / q2) before
    they are added, so no scale rounding error is introduced.
    """
    if ctx is None or a is None or coeffs is None:
        return None
    c0, c1, c2, c3 = (list(map(float, coeffs)) + [0.0, 0.0, 0.0, 0.0])[:4]
    if len(coeffs) > 4:
        raise ValueError("HE_POLY: degree <= 3 supported")
    he = _he(ctx)
    x = _ct(he, a)
    if not _is_ckks(he):
        if c2 or c3:
            raise ValueError("HE_POLY: BFV parameters here support degree <= 1 (no ct x ct budget)")
        if not c1:
            raise ValueError("HE_POLY: polynomial has no x term")
        return _out(he.add_plain(_mul_plain(he, x, c1), _plain_at(he, x, c0), in_new_ctxt=True))
    if not (c1 or c2 or c3):
        raise ValueError("HE_POLY: polynomial has no x term")

    s = x.scale
    terms = []
    x2 = _mul(he, x, x) if (c2 or c3) else None      # level 1, scale s1 = s^2/q1
    s1 = x2.scale if x2 is not None else None
    if c1:
        t1 = _mul_plain(he, x, c1, scale=s)          # level 1, scale s1
        if s1 is None:
            s1 = t1.scale
        terms.append(_mul_plain(he, t1, 1.0, scale=s1))
    if c2:
        terms.append(_mul_plain(he, x2, c2, scale=s1))
    if c3:
        terms.append(_mul(he, x2, _mul_plain(he, x, c3, scale=s)))
    acc = terms[0]
    for t in terms[1:]:
        acc = acc + t
    if c0:
        acc = he.add_plain(acc, _plain_at(he, acc, c0), in_new_ctxt=True)
    return _out(acc)


def he_info(ctx, a):
    """Structural metadata visible to the server. Never the value."""
    if ctx is None or a is None:
        return None
    he = _he(ctx)
    c = _ct(he, a)
    info = {"scheme": he.scheme.name, "n": he.n, "polys": c.size(), "bytes": len(a)}
    if _is_ckks(he):
        info["level"] = c.mod_level
        info["levels_left"] = len(he.qi_sizes) - 2 - c.mod_level
        info["scale_bits"] = round(float(np.log2(c.scale)), 2)
    return info


# ----------------------------------------------------------------------------- packed (SIMD) mode

def packed_window(n_values):
    """Power-of-two window used by packed sums; the total ends up in slot window-1."""
    w = 1
    while w < n_values:
        w *= 2
    return w


def _require_rotatable(he, c, op):
    if he.is_rotate_key_empty():
        raise ValueError(f"{op}: rotation keys missing")
    if c.size() != 2:
        raise ValueError(f"{op}: rotations need a 2-polynomial ciphertext; ct x ct results would need "
                         "relinearization keys first")


def he_packed_sum(ctx, rot_keys, a, n_values):
    """Sum of the first n_values slots via log2(window) rotations. Result in slot packed_window(n)-1."""
    if ctx is None or rot_keys is None or a is None or n_values is None:
        return None
    he = _he(ctx, rot_keys)
    c = _ct(he, a)
    _require_rotatable(he, c, "HE_PACKED_SUM")
    return _out(he.cumul_add(c, in_new_ctxt=True, n_elements=packed_window(int(n_values))))


def he_packed_dot(ctx, rot_keys, a, weights):
    """sum_i w_i * x_i for a CKKS-packed column and a plaintext weight vector."""
    if ctx is None or rot_keys is None or a is None or weights is None:
        return None
    he = _he(ctx, rot_keys)
    if not _is_ckks(he):
        raise ValueError("HE_PACKED_DOT: use CKKS (an arbitrary BFV plaintext vector exceeds the noise budget)")
    w = np.zeros(_slots(he))
    w[: len(weights)] = np.asarray(weights, dtype=np.float64)
    if not w.any():
        raise ValueError("HE_PACKED_DOT: all-zero weights")
    c = _ct(he, a)
    _require_rotatable(he, c, "HE_PACKED_DOT")
    pt = _match_level(he, he.encodeFrac(w, scale=_base_scale(c)), c)
    prod = he.multiply_plain(c, pt, in_new_ctxt=True)
    he.rescale_to_next(prod)
    return _out(he.cumul_add(prod, in_new_ctxt=True, n_elements=packed_window(len(weights))))


# ----------------------------------------------------------------------------- aggregate (UDAF)

class HeSum:
    """Homomorphic SUM. State = (context bytes, serialized partial sum) so it pickles across nodes."""

    def __init__(self):
        self._ctx = None
        self._acc = None

    @property
    def aggregate_state(self):
        return (self._ctx, None if self._acc is None else self._acc.to_bytes())

    def _add(self, ctx, ct_bytes):
        ctx = bytes(ctx)
        if self._ctx is None:
            self._ctx = ctx
        elif ctx != self._ctx:
            raise ValueError("HE_SUM: ciphertexts under different keys/contexts cannot be added")
        c = _ct(_he(self._ctx), ct_bytes)
        if self._acc is None:
            self._acc = c
        else:
            self._acc += c

    def accumulate(self, ctx, ct):
        if ctx is not None and ct is not None:
            self._add(ctx, ct)

    def merge(self, other_state):
        other_ctx, other_acc = other_state
        if other_acc is not None:
            self._add(other_ctx, other_acc)

    def finish(self):
        return None if self._acc is None else _out(self._acc)


# ----------------------------------------------------------------------------- table function (UDTF)

class HeRunningSum:
    """Encrypted running total within a partition: OVER (PARTITION BY ... ORDER BY ...)."""

    def __init__(self):
        self._acc = None

    def process(self, ctx, ct, row_key):
        if ctx is None or ct is None:
            yield (row_key, None)
            return
        he = _he(ctx)
        c = _ct(he, ct)
        self._acc = c if self._acc is None else self._acc + c
        yield (row_key, _out(self._acc))
