"""Reference outputs from ONNX Runtime for every model and variant in the manifest.

    make_reference.py [--threads 1] [name ...]

For each model file and batch size (1 and the model's N) this writes, under
<cache>/reference/<os-arch>/<model>/<variant>/b<batch>/ :

    inputs/<name>.npy      seeded inputs (batch 1 is sample 0 of batch N)
    outputs/<name>.npy     ONNX Runtime outputs, ORT_ENABLE_ALL, intra-op threads = --threads
    index.json             names, dtypes, shapes, SHA-256 of every file, ORT version, session options

and appends a row per output to <cache>/reference/<os-arch>/noise.csv comparing ORT_ENABLE_ALL
against ORT_DISABLE_ALL: the noise floor ONNX Runtime itself has between its optimisation levels.
.npy (format 1.0, little-endian, C order) is used because it is trivial to read from any language.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from common import MODELS, REFERENCE, load_manifest, log, machine_id, seeded_rng, sha256_file

VOCAB = {"bert_base": 30522, "minilm_l6": 30522, "whisper_small": 51865}


def conv_inputs(model: str, n: int) -> dict:
    x = np.stack([seeded_rng(model, "input", i).standard_normal((3, 224, 224), dtype=np.float32)
                  for i in range(n)])
    return {"input": x}


def encoder_inputs(model: str, n: int, seq: int) -> dict:
    ids = np.zeros((n, seq), np.int64)
    mask = np.zeros((n, seq), np.int64)
    for i in range(n):
        r = seeded_rng(model, "tokens", i)
        length = int(r.integers(seq // 2, seq + 1)) if i else seq  # sample 0 is unpadded
        ids[i, :length] = r.integers(1000, 30000, length)
        mask[i, :length] = 1
    return {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}


def inputs_for(model: str, variant: str, n: int, entry: dict) -> dict:
    if variant.startswith("encoder_") and model == "whisper_small":
        x = np.stack([0.5 * seeded_rng(model, "mel", i).standard_normal((80, 3000), dtype=np.float32)
                      for i in range(n)])
        return {"input_features": x}
    if variant.startswith("decoder_") and model == "whisper_small":
        ids = np.stack([seeded_rng(model, "dec-ids", i).integers(0, VOCAB[model], entry["decoderLen"])
                        for i in range(n)]).astype(np.int64)
        hs = np.stack([0.5 * seeded_rng(model, "dec-hs", i).standard_normal((1500, 768), dtype=np.float32)
                       for i in range(n)])
        return {"input_ids": ids, "encoder_hidden_states": hs}
    if entry["source"]["kind"] == "torchvision":
        return conv_inputs(model, n)
    return encoder_inputs(model, n, entry["seqLen"])


def run(path: Path, feeds: dict, level, threads: int) -> dict:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = level
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    names = [o.name for o in s.get_outputs()]
    return dict(zip(names, s.run(names, feeds)))


def diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """(max abs, max abs / max|b|, max elementwise rel with floor 1e-3 * max|b|)."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    d = np.abs(a64 - b64)
    scale = max(float(np.abs(b64).max()), 1e-30)
    return float(d.max()), float(d.max() / scale), float((d / np.maximum(np.abs(b64), 1e-3 * scale)).max())


def main() -> None:
    import onnxruntime as ort

    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    man = load_manifest()
    names = a.names or [n for n, m in man["models"].items() if m["files"]]
    root = REFERENCE / machine_id()
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in names:
        entry = man["models"][model]
        for variant, f in entry["files"].items():
            path = MODELS / f["path"]
            for n in sorted({1, entry["batch"]}):
                feeds = inputs_for(model, variant, n, entry)
                log(f"{model}/{variant} batch {n}")
                ref = run(path, feeds, ort.GraphOptimizationLevel.ORT_ENABLE_ALL, a.threads)
                raw = run(path, feeds, ort.GraphOptimizationLevel.ORT_DISABLE_ALL, a.threads)
                d = root / model / variant / f"b{n}"
                index = {"model": model, "variant": variant, "batch": n, "modelSha256": f["sha256"],
                         "onnxruntime": ort.__version__, "optimization": "ORT_ENABLE_ALL",
                         "intraOpThreads": a.threads, "interOpThreads": 1, "inputs": {}, "outputs": {}}
                for kind, arrs in (("inputs", feeds), ("outputs", ref)):
                    (d / kind).mkdir(parents=True, exist_ok=True)
                    for k, v in arrs.items():
                        v = np.ascontiguousarray(v)
                        p = d / kind / f"{k}.npy"
                        np.save(p, v)
                        index[kind][k] = {"dtype": str(v.dtype), "shape": list(v.shape),
                                          "sha256": sha256_file(p)}
                with open(d / "index.json", "w", encoding="utf-8", newline="\n") as fh:
                    json.dump(index, fh, indent=2)
                    fh.write("\n")
                for k in ref:
                    if np.issubdtype(ref[k].dtype, np.floating):
                        mabs, mscaled, mrel = diff(ref[k], raw[k])
                        rows.append([model, variant, n, k, f"{mabs:.3e}", f"{mscaled:.3e}", f"{mrel:.3e}"])
    out = root / "noise.csv"
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        if new:
            w.writerow(["model", "variant", "batch", "output", "max_abs", "max_abs_over_max_ref", "max_rel_floored"])
        w.writerows(rows)
    log(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
