"""List the operator types a set of ONNX models uses: type, count, attribute names seen and the opset of the default domain.

    model_ops.py MODEL.onnx [MODEL.onnx ...] [--only OP,OP,...]

Prints one table per model and one summary across all of them. Reads only the model graphs (external weight data is not loaded).
"""
from __future__ import annotations

import argparse
import collections
import sys

import onnx


def walk(graph):
    for n in graph.node:
        yield n
        for a in n.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                yield from walk(a.g)
            elif a.type == onnx.AttributeProto.GRAPHS:
                for g in a.graphs:
                    yield from walk(g)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = {x for x in a.only.split(",") if x}
    total = collections.defaultdict(lambda: {"count": 0, "models": set(), "attrs": set()})
    for path in a.models:
        m = onnx.load(path, load_external_data=False)
        opset = next((o.version for o in m.opset_import if o.domain in ("", "ai.onnx")), 0)
        rows = collections.defaultdict(lambda: {"count": 0, "attrs": set()})
        for n in walk(m.graph):
            key = (n.domain or "", n.op_type)
            rows[key]["count"] += 1
            rows[key]["attrs"].update(x.name for x in n.attribute)
        print(f"== {path} (opset {opset}, {len(rows)} op types)")
        for (dom, op), r in sorted(rows.items()):
            if only and op not in only:
                continue
            print(f"  {dom + '.' if dom else ''}{op:24s} x{r['count']:<5d} attrs: {', '.join(sorted(r['attrs'])) or '-'}")
            t = total[(dom, op)]
            t["count"] += r["count"]
            t["models"].add(path.rsplit("/", 1)[-1])
            t["attrs"].update(r["attrs"])
    print("== all models")
    for (dom, op), t in sorted(total.items()):
        print(f"  {dom + '.' if dom else ''}{op:24s} x{t['count']:<5d} in {len(t['models'])} models; attrs: {', '.join(sorted(t['attrs'])) or '-'}")


if __name__ == "__main__":
    sys.exit(main())
