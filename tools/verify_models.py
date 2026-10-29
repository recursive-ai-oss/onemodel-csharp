"""Check every model file in the cache against the SHA-256 recorded in models.json.

    verify_models.py          exit status 1 if anything is missing or different
"""
from __future__ import annotations

import sys

from common import MODELS, load_manifest, sha256_file


def main() -> int:
    bad = 0
    for name, m in load_manifest()["models"].items():
        for key, f in m["files"].items():
            p = MODELS / f["path"]
            if not p.exists():
                print(f"MISSING {name}/{key}")
                bad += 1
                continue
            h = sha256_file(p)
            ok = h == f["sha256"]
            bad += not ok
            print(f"{'ok     ' if ok else 'DIFFERS'} {name}/{key} {h}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
