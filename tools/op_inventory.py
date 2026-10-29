"""List the operators a set of ONNX files use: op type, domain, opset, node count, attribute names, and which are
not in the runtime's operator table.

    op_inventory.py [--table src/OneModel/Graph/Operators] model.onnx [model.onnx ...]

With --table, the operator table sources are scanned for `Name = "Op"` style registrations (the string literals passed
to the operator definitions) and ops absent from them are marked MISSING. Weights are not loaded.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import onnx


def known_ops(table_dir: Path) -> set[str]:
    names: set[str] = set()
    for f in table_dir.rglob("*.cs"):
        text = f.read_text(encoding="utf-8")
        # Operator definitions name their ONNX operator with a string literal on a line mentioning OpDef/Name.
        for m in re.finditer(r'(?:OpDef|Name\s*=|Op\s*=)[^;\n]*?"([A-Z][A-Za-z0-9]+)"', text):
            names.add(m.group(1))
        for m in re.finditer(r'\bnew\s*\(\s*"([A-Z][A-Za-z0-9]+)"', text):
            names.add(m.group(1))
    return names


def walk(graph, acc):
    for n in graph.node:
        e = acc[(n.domain or "", n.op_type)]
        e["count"] += 1
        e["attrs"].update(a.name for a in n.attribute)
        for a in n.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                walk(a.g, acc)
            elif a.type == onnx.AttributeProto.GRAPHS:
                for g in a.graphs:
                    walk(g, acc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table")
    ap.add_argument("models", nargs="+")
    args = ap.parse_args()
    known = known_ops(Path(args.table)) if args.table else None
    total = defaultdict(lambda: {"count": 0, "attrs": set(), "models": set()})
    for path in args.models:
        m = onnx.load(path, load_external_data=False)
        opsets = {o.domain or "": o.version for o in m.opset_import}
        acc = defaultdict(lambda: {"count": 0, "attrs": set()})
        walk(m.graph, acc)
        print(f"== {Path(path).name}: opset {opsets}, {len(acc)} distinct ops")
        for k, v in acc.items():
            t = total[k]
            t["count"] += v["count"]
            t["attrs"] |= v["attrs"]
            t["models"].add(Path(path).stem)
    print("\n== union")
    print("| domain | op | nodes | attributes | models | in table |")
    print("|---|---|---|---|---|---|")
    for (dom, op), v in sorted(total.items(), key=lambda kv: kv[0][1]):
        status = "" if known is None else ("yes" if op in known else "MISSING")
        print(f"| {dom or 'ai.onnx'} | {op} | {v['count']} | {', '.join(sorted(v['attrs']))} | "
              f"{', '.join(sorted(v['models']))} | {status} |")


if __name__ == "__main__":
    main()
