# Vendored Pyfhel 3.4.2: provenance and patches

`vendor/Pyfhel-3.4.2/` is the [Pyfhel](https://github.com/ibarrond/Pyfhel) 3.4.2 source distribution from PyPI
(sha256 `fa41664f8815b27f6347821f6363a7516fa84f50570816d8f292f2c22f88f88c`) with exactly the modifications
below. `scripts/refresh_vendor.sh` reproduces the tree from upstream and verifies the checksum and the SEAL
commit. Everything else is upstream, unmodified.

| # | Change | Files |
|---|---|---|
| 1 | Bundled Microsoft SEAL **4.0.0 replaced by SEAL v4.1.2** (commit `119dc32e135cb89c1062076a69310d4413ebc824`), unmodified apart from removing `.git/` | `Pyfhel/backend/SEAL/` |
| 2 | `SEAL_USE_ZLIB = 'OFF'` (was `'ON'`) | `pyproject.toml`, `[cpplibraries.SEAL.cmake_opts]` |
| 3 | Removed the generated `Pyfhel.egg-info/` | |

## Why

1. **Ciphertext compatibility with Snowflake.** Snowflake's Anaconda channel builds `pyfhel 3.4.2` against
   SEAL 4.1. SEAL writes its `major.minor` version into every serialized object and refuses to load objects
   from another version: ciphertexts produced by the PyPI sdist (SEAL 4.0) fail in Snowflake with
   `RuntimeError: incompatible version`. Pyfhel 3.4.2 compiles unchanged against SEAL 4.1.2.
2. **macOS build.** SEAL's bundled zlib 1.2.x fails to compile against recent macOS SDKs (`fdopen` macro
   clash in `zutil.h`). Pyfhel's `setup.py` ignores the failed CMake build and produces a wheel *without*
   libseal, which then fails at import with `symbol not found ... seal13parms_id_zero`. With zlib off, SEAL
   still compresses with Zstandard, which is Pyfhel's default (`compr_mode="zstd"`), so serialized objects stay
   interoperable.

## Build requirements

SEAL's CMake fetches Zstandard and Microsoft GSL from GitHub at build time (`FetchContent`), so the first
`pixi install` needs network access to github.com. The pixi environment provides the compiler toolchain
(`cxx-compiler`, `cmake<4`, `make`, `git`, and `llvm-openmp` on macOS) plus `cython<3.1`; Cython 3.1 changed
C++ enum handling and fails on `PyPoly.pyx`.

## Licenses

* Pyfhel: Apache License 2.0, see `Pyfhel-3.4.2/LICENSE.txt`. Upstream relicensed from GPLv3 to Apache-2.0 in
  July 2023; the 3.4.2 README badge and `pyproject.toml` classifier still say GPLv3, but the shipped
  `LICENSE.txt` and the upstream repository are Apache-2.0.
* Microsoft SEAL: MIT License, see `Pyfhel-3.4.2/Pyfhel/backend/SEAL/LICENSE`.
