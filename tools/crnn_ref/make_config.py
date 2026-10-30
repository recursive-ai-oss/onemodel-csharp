#!/usr/bin/env python3
"""Writes the fine-tune config used by the reference run: a source config with chosen training sources and validation sets removed.

    make_config.py --src CONFIG.yaml --dst OUT.yaml --drop-source NAME ... --drop-set NAME ... --steps N --batch B

Variables such as ${NAME} are left in place; the trainer expands them from the environment. Validation sets whose directory does
not exist (checked after expansion) are dropped as well, and the guard entries naming dropped sets are removed.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--drop-source", action="append", default=[])
    ap.add_argument("--drop-set", action="append", default=[])
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.src).read_text(encoding="utf-8"))
    names = {s["name"] for s in cfg["sources"]}
    for n in a.drop_source:
        assert n in names, n
    cfg["sources"] = [s for s in cfg["sources"] if s["name"] not in a.drop_source]
    ev = {}
    for k, v in cfg["extra_val"].items():
        if k in a.drop_set:
            continue
        if not os.path.isdir(os.path.expandvars(v)):
            print(f"dropping {k}: {os.path.expandvars(v)} does not exist")
            continue
        ev[k] = v
    cfg["extra_val"] = ev
    tc = cfg["train"]
    tc["steps"], tc["batch"] = a.steps, a.batch
    for g in ("guard", "guard_wer"):
        tc[g] = {k: v for k, v in tc.get(g, {}).items() if k in ev or k == "val"}
    Path(a.dst).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


if __name__ == "__main__":
    main()
