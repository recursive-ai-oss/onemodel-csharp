"""Export the multilingual encoders used to check fine-tuning an imported graph.

    export_finetune_models.py [--record] [--force] [--ops] [name ...]    (names: xlm_roberta_base mmbert_base)

Same method as export_models.py (pinned Hugging Face revision, weights hash checked against models.json,
TorchScript-based exporter at opset 17). Each model is exported with inputs (input_ids, attention_mask)
and the last hidden state as output, from the eager-attention implementation, and the weights are kept
as float32 initializers so the graph can be fine-tuned. --ops prints the operator histogram of the export.
"""
from __future__ import annotations

import argparse
import collections
import sys

import export_models as em
from common import MODELS, load_manifest, save_manifest


def export_encoder(name: str, entry: dict, opset: int, record: bool, force: bool) -> None:
    import torch
    from transformers import AutoModel

    out = MODELS / name / f"{name}.onnx"
    d = em.hf_weights(entry["source"], record)
    if out.exists() and not force:
        return em.finish(name, "fp32", out, entry, record)
    m = AutoModel.from_pretrained(d, attn_implementation="eager", dtype=torch.float32).eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]

    ids = torch.ones(2, 16, dtype=torch.int64)
    mask = torch.ones(2, 16, dtype=torch.int64)
    ax = {"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
          "last_hidden_state": {0: "batch", 1: "seq"}}
    em.export(Wrap(m), (ids, mask), out, ["input_ids", "attention_mask"], ["last_hidden_state"], ax, opset)
    em.finish(name, "fp32", out, entry, record)


def histogram(name: str) -> None:
    import onnx

    m = onnx.load(str(MODELS / name / f"{name}.onnx"), load_external_data=False)
    c = collections.Counter(n.op_type for n in m.graph.node)
    print(name, "nodes", len(m.graph.node), "initializers", len(m.graph.initializer))
    for op, k in sorted(c.items()):
        print(f"  {op:28s} {k}")


EXPORTERS = {"xlm_roberta_base": export_encoder, "mmbert_base": export_encoder}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--record", action="store_true", help="write hashes into models.json")
    ap.add_argument("--force", action="store_true", help="re-export even if the file exists")
    ap.add_argument("--ops", action="store_true", help="print each exported graph's operator histogram")
    a = ap.parse_args()
    man = load_manifest()
    for n in a.names or list(EXPORTERS):
        EXPORTERS[n](n, man["models"][n], man["onnxOpset"], a.record, a.force)
        if a.ops:
            histogram(n)
        if a.record:
            save_manifest(man)


if __name__ == "__main__":
    sys.exit(main())
