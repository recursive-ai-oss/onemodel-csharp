#!/usr/bin/env python3
"""PyTorch reference for fine-tuning an exported encoder: forward output, gradients of a fixed loss, and a five-step AdamW loss curve.

    finetune_reference.py <name> [--steps 5] [--lr 1e-5]        (names: xlm_roberta_base mmbert_base, see export_finetune_models.py)

Loads the pinned Hugging Face weights in float32 (eager attention, evaluation mode, so dropout is off), tokenizes a few multilingual
sentences into a padded batch of two with an attention mask, and takes

    loss = mean((mean_over_positions(hidden) @ P - target)^2)        P [d, 8] and target [2, 8] seeded

Step 0 saves the forward output and the gradients of the embedding tables, the first layer, the last layer and the final norm, keyed
by the name of the matching initializer in the exported ONNX file (found by comparing values; a weight the exporter transposed is
saved transposed). Then AdamW (betas 0.9/0.999, eps 1e-8, weight decay 0.01) runs <steps> steps and the loss before each update is
saved. Output: <cache>/finetune/<name>/ref.bin + ref.json (same layout as finetune_oracle.py). PyTorch is used as a black box.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

import export_models as em
from common import CACHE, MODELS, load_manifest, log

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "東京は日本の首都であり、世界最大の都市圏のひとつです。",
]
SEED = 20261105


def wanted(hf_name: str, layers: int) -> bool:
    n = hf_name.removeprefix("m.")
    keys = ("embeddings.", "final_norm.", f"layer.0.", f"layers.0.", f"layer.{layers - 1}.", f"layers.{layers - 1}.")
    return any(k in n for k in keys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    a = ap.parse_args()
    torch.set_num_threads(8)
    entry = load_manifest()["models"][a.name]
    d = em.hf_weights(entry["source"], record=False)
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModel.from_pretrained(d, attn_implementation="eager", dtype=torch.float32).eval()
    # The ONNX export has a plain Gather, which has no padding row: its gradient reaches every row it read. PyTorch's
    # embedding with padding_idx instead zeroes the padding row's gradient, so clear padding_idx to compare like with like.
    for mod in model.modules():
        if isinstance(mod, torch.nn.Embedding):
            mod.padding_idx = None
    enc = tok(SENTENCES, padding=True, return_tensors="pt")
    ids, mask = enc["input_ids"].to(torch.int64), enc["attention_mask"].to(torch.int64)
    log("batch", tuple(ids.shape), "real tokens per row", mask.sum(1).tolist())
    width = model.config.hidden_size
    rng = np.random.default_rng(SEED)
    proj = torch.tensor((rng.standard_normal((width, 8)) * 0.05).astype("float32"))
    target = torch.tensor(rng.standard_normal((2, 8)).astype("float32"))

    def loss_fn():
        h = model(input_ids=ids, attention_mask=mask, return_dict=False)[0]
        return ((h.mean(dim=1) @ proj - target) ** 2).mean(), h

    import onnx
    from onnx import numpy_helper

    onx = onnx.load(str(MODELS / a.name / f"{a.name}.onnx"))
    params = dict(model.named_parameters())
    layers = model.config.num_hidden_layers
    # ONNX initializer name -> (hf parameter name, transposed)
    found: dict[str, tuple[str, bool]] = {}
    by_shape: dict[tuple, list[str]] = {}
    for k, p in params.items():
        by_shape.setdefault(tuple(p.shape), []).append(k)
        if p.ndim == 2:
            by_shape.setdefault(tuple(p.T.shape), []).append(k)
    for init in onx.graph.initializer:
        arr = numpy_helper.to_array(init)
        for k in by_shape.get(tuple(arr.shape), []):
            p = params[k].detach().numpy()
            if p.shape == arr.shape and np.array_equal(p, arr):
                found[init.name] = (k, False)
                break
            if p.ndim == 2 and p.T.shape == arr.shape and np.array_equal(p.T, arr):
                found[init.name] = (k, True)
                break
    del onx
    log("initializers matched to parameters:", len(found))

    out = {"in.input_ids": ids.float().numpy(), "in.attention_mask": mask.float().numpy(), "in.P": proj.numpy(), "in.target": target.numpy()}
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    losses = []
    for step in range(a.steps):
        opt.zero_grad()
        loss, h = loss_fn()
        loss.backward()
        losses.append(loss.item())
        if step == 0:
            out["out.hidden"] = h.detach().numpy()
            for init_name, (hf, transposed) in found.items():
                if wanted(hf, layers) and params[hf].grad is not None:
                    g = params[hf].grad
                    out["grad." + init_name] = (g.T if transposed else g).detach().numpy()
        opt.step()
        log("step", step, "loss", losses[-1])
    out["out.losses"] = np.array(losses, dtype=np.float32)
    dest = CACHE / "finetune" / a.name
    dest.mkdir(parents=True, exist_ok=True)
    chunks, manifest, offset = [], {"torch": torch.__version__, "seed": SEED, "lr": a.lr, "tensors": {}}, 0
    for k, v in out.items():
        arr = np.ascontiguousarray(v, dtype="<f4")
        manifest["tensors"][k] = {"shape": list(arr.shape), "offset": offset}
        chunks.append(arr.reshape(-1))
        offset += arr.size
    with open(dest / "ref.bin", "wb") as f:
        for c in chunks:
            c.tofile(f)
    with open(dest / "ref.json", "w") as f:
        json.dump(manifest, f)
    log("wrote", dest, "floats", offset)


if __name__ == "__main__":
    sys.exit(main())
