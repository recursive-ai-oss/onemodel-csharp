"""Shared helpers for the tools scripts: locations, hashing, the model manifest."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

HOME = Path(os.environ.get("ONEMODEL_HOME", Path.home() / "onemodel"))
CACHE = Path(os.environ.get("ONEMODEL_CACHE", HOME / "cache"))
MODELS = CACHE / "models"
REFERENCE = CACHE / "reference"
BACKEND_TEST = CACHE / "onnx-backend-test"
MANIFEST = Path(__file__).with_name("models.json")


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_manifest() -> dict:
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)


def save_manifest(m: dict) -> None:
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as f:
        json.dump(m, f, indent=2, sort_keys=False)
        f.write("\n")


def machine_id() -> str:
    """Short arch tag used to separate per-machine outputs (reference outputs differ by ISA)."""
    return f"{platform.system().lower()}-{platform.machine().lower()}"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def seeded_rng(*keys):
    """numpy Generator seeded from a stable hash of the keys (never Python's randomized hash())."""
    import numpy as np

    seed = int.from_bytes(hashlib.sha256("/".join(map(str, keys)).encode()).digest()[:8], "little")
    return np.random.Generator(np.random.PCG64(seed))
