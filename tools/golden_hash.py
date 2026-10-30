#!/usr/bin/env python3
"""Golden-hash runner: run one command under several configurations and compare output hashes.

The command prints lines of the form `<name> <64 hex chars>` (a SHA-256 per output); other lines
are ignored. It is run once per configuration row and the table shows, per output name, each
row's hash prefix and whether all rows agree.

    golden_hash.py --cmd "dotnet app.dll" [--aot-cmd ./app-aot] [--row NAME:K=V,K=V:COMMAND ...]
                   [--save rows.json] [--merge other.json ...]
    golden_hash.py --table-only --merge a.json --merge b.json

Built-in rows (the commands see ONEMODEL_THREADS=1 or unset = all cores):
  x64 / generic : jit-default, jit-1thread, jit-no-avx512 (DOTNET_EnableAVX512=0),
                  jit-no-avx2 (DOTNET_EnableAVX2=0), jit-no-hwintrinsic (DOTNET_EnableHWIntrinsic=0)
  arm64         : jit-default, jit-1thread, jit-no-dotprod (DOTNET_EnableArm64Dp=0),
                  jit-no-hwintrinsic
  with --aot-cmd: aot-default, aot-1thread
Extra rows: --row NAME:ENV:COMMAND, e.g. a browser-WASM host: --row wasm::"node run-wasm.mjs".
--save writes this machine's results; --merge adds results saved on other machines (their row names
carry the machine tag) so one table covers x64, arm64 and WASM.
Exit status 1 when any output differs between rows (use --allow-diff to always exit 0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys

LINE = re.compile(r"^(\S+)\s+([0-9a-f]{64})\s*$")


def run_row(cmd: str, env_extra: dict[str, str], timeout: int) -> dict[str, str] | str:
    env = dict(os.environ)
    env.pop("ONEMODEL_THREADS", None)
    env.update(env_extra)
    p = subprocess.run(shlex.split(cmd), env=env, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        return f"exit {p.returncode}: {p.stderr.strip()[:200]}"
    out = {}
    for ln in p.stdout.splitlines():
        m = LINE.match(ln)
        if m:
            out[m.group(1)] = m.group(2)
    return out or "no hash lines in output"


def builtin_rows(cmd: str, aot: str | None) -> list[tuple[str, dict, str]]:
    arm = platform.machine().lower() in ("arm64", "aarch64")
    rows = [("jit-default", {}, cmd), ("jit-1thread", {"ONEMODEL_THREADS": "1"}, cmd)]
    if arm:
        rows.append(("jit-no-dotprod", {"DOTNET_EnableArm64Dp": "0"}, cmd))
    else:
        rows += [("jit-no-avx512", {"DOTNET_EnableAVX512": "0"}, cmd),
                 ("jit-no-avx2", {"DOTNET_EnableAVX2": "0"}, cmd)]
    rows.append(("jit-no-hwintrinsic", {"DOTNET_EnableHWIntrinsic": "0"}, cmd))
    if aot:
        rows += [("aot-default", {}, aot), ("aot-1thread", {"ONEMODEL_THREADS": "1"}, aot)]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cmd")
    ap.add_argument("--table-only", action="store_true", help="only merge and print saved results")
    ap.add_argument("--aot-cmd")
    ap.add_argument("--row", action="append", default=[])
    ap.add_argument("--save")
    ap.add_argument("--merge", action="append", default=[])
    ap.add_argument("--tag", default=f"{platform.system().lower()}-{platform.machine().lower()}")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--allow-diff", action="store_true")
    ap.add_argument("--markdown", help="write a per-group table (one row per model, or per case set) to this file")
    ap.add_argument("--no-builtin", action="store_true", help="run only the --row entries")
    ap.add_argument("--quiet", action="store_true", help="print only the per-group summary, not every output")
    a = ap.parse_args()

    if not a.cmd and not a.table_only and not a.no_builtin:
        ap.error("--cmd is required unless --table-only")
    rows = [] if (a.table_only or a.no_builtin) else builtin_rows(a.cmd, a.aot_cmd)
    for spec in a.row:
        name, env, cmd = spec.split(":", 2)
        rows.append((name, dict(kv.split("=", 1) for kv in env.split(",") if kv), cmd or a.cmd))

    results: dict[str, dict[str, str] | str] = {}
    for name, env, cmd in rows:
        results[f"{a.tag}/{name}"] = run_row(cmd, env, a.timeout)
    if a.save:
        with open(a.save, "w", encoding="utf-8", newline="\n") as f:
            json.dump(results, f, indent=1)
            f.write("\n")
    for path in a.merge:
        with open(path, encoding="utf-8") as f:
            results.update(json.load(f))

    cols = list(results)
    good = {c: r for c, r in results.items() if isinstance(r, dict)}
    names = sorted({n for r in good.values() for n in r})
    # A row named *wasm* may run only a subset of the cases (the interpreter is slow); it is compared on the names it has.
    partial = {c for c in good if "wasm" in c}
    w = max([len(n) for n in names] + [6])
    differs = False
    verdict: dict[str, bool] = {}
    for n in names:
        vals = [good[c].get(n) for c in good if not (c in partial and n not in good[c])]
        same = len(set(vals)) == 1 and None not in vals
        verdict[n] = same
        differs |= not same
        if not a.quiet or not same:
            cells = [(good[c].get(n) or "-")[:12] if c in good else "FAILED" for c in cols]
            print(f"{n:<{w}} | " + " | ".join(f"{x:<14}" for x in cells) + f" | {'equal' if same else 'UNEQUAL'}")
    print("columns: " + " | ".join(cols))
    for c, r in results.items():
        if isinstance(r, str):
            print(f"row {c}: {r}")
            differs = True
    groups = sorted({n.split("/")[0] for n in names})
    lines = ["| group | outputs | " + " | ".join(cols) + " | verdict |", "|---|---|" + "---|" * (len(cols) + 1)]
    for g in groups:
        gn = [n for n in names if n.split("/")[0] == g]
        cells = []
        for c in cols:
            r = good.get(c)
            if r is None:
                cells.append("FAILED")
                continue
            have = [n for n in gn if n in r]
            if not have:
                cells.append("-")
                continue
            h = hashlib.sha256("\n".join(f"{n} {r[n]}" for n in have).encode()).hexdigest()[:12]
            cells.append(h + (f" ({len(have)})" if len(have) != len(gn) else ""))
        ok = all(verdict[n] for n in gn)
        lines.append(f"| {g} | {len(gn)} | " + " | ".join(cells) + f" | {'equal' if ok else '**UNEQUAL**'} |")
    print("\n".join(lines))
    if a.markdown:
        with open(a.markdown, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
    return 0 if (a.allow_diff or not differs) else 1


if __name__ == "__main__":
    sys.exit(main())
