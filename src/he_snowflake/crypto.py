"""Client-side encrypt / decrypt helpers (require the secret key)."""

from __future__ import annotations

import numpy as np
from Pyfhel import PyCtxt

from .keys import HEKey
from .udf_handlers import packed_window


def encrypt(key: HEKey, value) -> bytes:
    """Encrypt one value (row mode) into slot 0."""
    if key.scheme == "bfv":
        c = key.he.encryptInt(np.array([int(value)], dtype=np.int64))
    else:
        c = key.he.encryptFrac(np.array([float(value)], dtype=np.float64))
    return c.to_bytes(compr_mode="zstd")


def encrypt_vector(key: HEKey, values) -> bytes:
    """Encrypt a whole column into one ciphertext (packed / SIMD mode)."""
    values = np.asarray(values)
    if len(values) > key.he.get_nSlots():
        raise ValueError(f"{len(values)} values > {key.he.get_nSlots()} slots")
    if key.scheme == "bfv":
        c = key.he.encryptInt(values.astype(np.int64))
    else:
        c = key.he.encryptFrac(values.astype(np.float64))
    return c.to_bytes(compr_mode="zstd")


def _load(key: HEKey, blob) -> PyCtxt:
    return PyCtxt(pyfhel=key.he, bytestring=bytes(blob))


def decrypt_vector(key: HEKey, blob, n: int | None = None) -> np.ndarray:
    c = _load(key, blob)
    v = key.he.decryptInt(c) if key.scheme == "bfv" else key.he.decryptFrac(c)
    return v if n is None else v[:n]


def decrypt(key: HEKey, blob, slot: int = 0):
    if blob is None:
        return None
    v = decrypt_vector(key, blob)[slot]
    return int(v) if key.scheme == "bfv" else float(v)


def decrypt_packed_sum(key: HEKey, blob, n_values: int):
    """Packed sums land in slot packed_window(n)-1 (see udf_handlers.he_packed_sum)."""
    return decrypt(key, blob, slot=packed_window(n_values) - 1)


def noise_budget(key: HEKey, blob) -> int:
    """Remaining BFV noise budget in bits (0 = undecryptable). Needs the secret key."""
    if key.scheme != "bfv":
        raise ValueError("noise budget is a BFV concept; CKKS tracks levels instead")
    return key.he.noise_level(_load(key, blob))


def bfv_range(key: HEKey) -> int:
    """Largest |value| a BFV result can hold before it wraps modulo t."""
    return key.he.t // 2
