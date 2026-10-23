"""Client-side key management. Secret keys never leave `key_dir`; only public material is exported."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from Pyfhel import Pyfhel

from .udf_handlers import packed_window

# Validated parameter sets (see README "Parameters"):
#  bfv   exact integers mod t (t_bits=34 -> results must stay within +-8.6e9), additions + plaintext
#        multiplications. No ct x ct: the noise budget of n=4096 is exhausted by one multiplication.
#  ckks  approximate reals, scale 2^35, two multiplicative levels (ct x ct x ct, degree-3 polynomials).
#        Values after two multiplications must stay within ~+-1.6e7.
PARAMS = {
    "bfv": dict(scheme="bfv", n=4096, t_bits=34, qi_sizes=[36, 36, 37]),
    "ckks": dict(scheme="ckks", n=8192, scale=2**35, qi_sizes=[60, 35, 35, 60]),
}
# Packed (SIMD) keys use the same parameters but are separate keys that also export *public* rotation keys,
# so the row-mode keys never have any evaluation key uploaded.
KEYS = {
    "bfv": ("bfv", "row"),
    "ckks": ("ckks", "row"),
    "bfv_packed": ("bfv", "packed"),
    "ckks_packed": ("ckks", "packed"),
}


@dataclass
class HEKey:
    name: str
    mode: str          # "row" | "packed"
    params: dict
    he: Pyfhel         # holds the secret key: client side only
    key_id: str
    rot_steps: list[int]

    @property
    def scheme(self) -> str:
        return self.params["scheme"]

    def context_bytes(self) -> bytes:
        return self.he.to_bytes_context()

    def rotation_key_bytes(self) -> bytes | None:
        return self.he.to_bytes_rotate_key(compr_mode="zstd") if self.rot_steps else None


def _rot_steps(max_values: int) -> list[int]:
    # Pyfhel's cumul_add rotates by -1, -2, -4, ... up to window/2
    w = packed_window(max_values)
    return [-(2**i) for i in range(w.bit_length() - 1)]


def load_or_create(key_dir: Path, name: str, max_packed_values: int = 1024, regenerate: bool = False) -> HEKey:
    scheme, mode = KEYS[name]
    params = PARAMS[scheme]
    d = Path(key_dir) / name
    meta_path = d / "meta.json"
    steps = _rot_steps(max_packed_values) if mode == "packed" else []
    he = Pyfhel()

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    reuse = meta is not None and not regenerate and meta["params"] == params and set(steps) <= set(meta["rot_steps"])
    if reuse:
        he.load_context(str(d / "context"))
        he.load_public_key(str(d / "pub.key"))
        he.load_secret_key(str(d / "sec.key"))
        if meta["rot_steps"]:
            he.load_rotate_key(str(d / "rot.key"))
        steps = meta["rot_steps"]
    else:
        d.mkdir(parents=True, exist_ok=True)
        he.contextGen(**params)
        he.keyGen()
        he.save_context(str(d / "context"))
        he.save_public_key(str(d / "pub.key"))
        he.save_secret_key(str(d / "sec.key"))
        if steps:
            he.rotateKeyGen(steps)
            he.save_rotate_key(str(d / "rot.key"))
    key_id = f"{name}-" + hashlib.sha256(he.to_bytes_context() + he.to_bytes_public_key()).hexdigest()[:12]
    if not reuse:
        meta_path.write_text(json.dumps({"key_id": key_id, "params": params, "rot_steps": steps}, indent=2))
    return HEKey(name=name, mode=mode, params=params, he=he, key_id=key_id, rot_steps=steps)


def load_all(key_dir: Path, max_packed_values: int = 1024, regenerate: bool = False) -> dict[str, HEKey]:
    return {n: load_or_create(key_dir, n, max_packed_values, regenerate) for n in KEYS}
