"""Export the two OCR reference models (a CRNN recogniser and a DBNet-style detector) to ONNX.

    export_oneocr.py [--record]

Inputs (arguments or environment variables; defaults under $ONEMODEL_CACHE/oneocr/):
    --training-dir  / ONEOCR_TRAINING   read-only copy of the models' training code (models/, export/)
    --crnn-onm      / ONEOCR_CRNN_ONM   recogniser weights container (fp16 .onm)
    --dbnet-onm     / ONEOCR_DBNET_ONM  detector weights container
    --crnn-fixtures / ONEOCR_CRNN_FIXTURES, --dbnet-fixtures / ONEOCR_DBNET_FIXTURES
                                        per-layer fp32 reference tensors written by the training code

Steps: check the container hashes against models.json; build the fp32 PyTorch modules from the
containers (batch norms already folded); export with torch.onnx at the manifest opset, dynamic
width (recogniser) and dynamic height and width (detector); compare the PyTorch module and then
ONNX Runtime running the exported file against the fixtures; write ONNX Runtime reference outputs
(ORT_ENABLE_ALL, 1 thread) in the layout make_reference.py uses.

Error metrics, against the fixture tensor `ref` (the fixtures' own fp32-vs-fp64 floors use the same):
    max_abs   max |out - ref|
    max_rel   max |out - ref| / |ref| over elements with |ref| > 1e-3 (recogniser) or
              |ref| > 1e-3 * max|ref| (detector)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np

from common import CACHE, MODELS, REFERENCE, load_manifest, log, machine_id, save_manifest, sha256_file

SRC = CACHE / "oneocr"
CRNN_BATCH_IDS = ["val-00000104", "val-00001356", "val-00000050", "val-00000223",
                  "val-00000728", "val-00002515", "val-00002219", "val-00000539"]  # widths 77..128
CRNN_B1_IDS = ["val-00000965", "val-00000050", "val-00000539", "val-00002930", "val-00002703"]
DBNET_B1_IDS = ["tile0", "wide"]


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default))


def load_training_code(training_dir: Path):
    """Imports the container reader and model definitions from the (read-only) training tree.

    The detector's reference module lives in a file that also imports post-processing helpers
    (image and SciPy libraries) it does not use for the forward pass; those thresholds are stubbed
    so the forward pass can be imported with torch and NumPy alone.
    """
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(training_dir))
    stub = types.ModuleType("eval.detect")
    stub.EXPAND_LAYOUT = stub.LAYOUT_THRESHOLD = stub.TEXT_THRESHOLD = 0.0
    sys.modules["eval.detect"] = stub
    from export.container import read_container
    from export.dbnet import FoldedReference
    from models.crnn import load_folded

    return read_container, load_folded, FoldedReference


def read_f32(path: Path, shape) -> np.ndarray:
    return np.fromfile(path, dtype="<f4").reshape(shape)


def metrics(out: np.ndarray, ref: np.ndarray, floor_mode: str) -> tuple[float, float]:
    d = np.abs(out.astype(np.float64) - ref.astype(np.float64))
    r = np.abs(ref.astype(np.float64))
    thr = 1e-3 if floor_mode == "abs" else 1e-3 * float(r.max())
    m = r > thr
    return float(d.max()), float((d[m] / r[m]).max()) if m.any() else 0.0


def op_counts(path: Path) -> dict[str, int]:
    import onnx

    g = onnx.load(str(path), load_external_data=False).graph
    return dict(sorted(Counter(n.op_type for n in g.node).items()))


def ort_session(path: Path, threads=1):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def write_reference(model: str, label: str, sha: str, feeds: dict, outs: dict, batch: int) -> None:
    import onnxruntime as ort

    d = REFERENCE / machine_id() / model / "fp32" / f"b{label}"
    index = {"model": model, "variant": "fp32", "batch": batch, "modelSha256": sha,
             "onnxruntime": ort.__version__, "optimization": "ORT_ENABLE_ALL",
             "intraOpThreads": 1, "interOpThreads": 1, "inputs": {}, "outputs": {}}
    for kind, arrs in (("inputs", feeds), ("outputs", outs)):
        (d / kind).mkdir(parents=True, exist_ok=True)
        for k, v in arrs.items():
            v = np.ascontiguousarray(v)
            p = d / kind / f"{k}.npy"
            np.save(p, v)
            index[kind][k] = {"dtype": str(v.dtype), "shape": list(v.shape), "sha256": sha256_file(p)}
    (d / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8", newline="\n")


def pin(entry: dict, key: str, path: Path, record: bool, extra: dict) -> str:
    f = entry["files"].setdefault(key, {"path": str(path.relative_to(MODELS))})
    h = sha256_file(path)
    if record or not f.get("sha256"):
        f.update(sha256=h, bytes=path.stat().st_size, **extra)
    elif f["sha256"] != h:
        log(f"NOTE {key}: export differs from the pinned file ({h} vs {f['sha256']}); copy the pinned file")
        f["localSha256"] = h
    return f["sha256"]


def do_crnn(a, man, read_container, load_folded, record):
    import onnx
    import torch

    entry = man["models"]["crnn"]
    src = entry["source"]
    if sha256_file(a.crnn_onm) != src["weightsSha256"]:
        raise SystemExit("recogniser container hash differs from the manifest")
    manifest, raw = read_container(a.crnn_onm)
    net = load_folded(manifest["config"], {k: torch.from_numpy(v) for k, v in raw.items()})
    out = MODELS / "crnn" / "crnn_v0.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(2, 1, 32, 64)
    with torch.no_grad():
        torch.onnx.export(net, (x,), str(out), input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input": {0: "batch", 3: "width"}, "logits": {0: "batch", 1: "steps"}},
                          opset_version=man["onnxOpset"], dynamo=False)
    onnx.checker.check_model(str(out))
    sha = pin(entry, "fp32", out, record, {"opset": man["onnxOpset"], "ops": op_counts(out)})

    fx = json.loads((a.crnn_fixtures / "fixtures.json").read_text())
    sess = ort_session(out)
    worst = {"torch": [0.0, 0.0], "ort": [0.0, 0.0]}
    byid = {}
    for line in fx["lines"]:
        layers = {l["name"]: l for l in line["layers"]}
        inp = read_f32(a.crnn_fixtures / layers["input"]["file"], layers["input"]["shape"])[None]  # [1,1,32,W]
        ref = read_f32(a.crnn_fixtures / layers["fc"]["file"], layers["fc"]["shape"])
        with torch.no_grad():
            t = net(torch.from_numpy(inp))[0].numpy()
        o = sess.run(["logits"], {"input": inp})[0][0]
        byid[line["id"]] = inp
        for k, v in (("torch", t), ("ort", o)):
            ma, mr = metrics(v, ref, "abs")
            worst[k] = [max(worst[k][0], ma), max(worst[k][1], mr)]
    floor = fx["fp32_vs_fp64_floor"]["fc"]
    log(f"crnn vs fixtures over {len(fx['lines'])} lines: torch max_abs {worst['torch'][0]:.3e} max_rel {worst['torch'][1]:.3e}; "
        f"ort max_abs {worst['ort'][0]:.3e} max_rel {worst['ort'][1]:.3e}; "
        f"fixtures' fp32-vs-fp64 floor (fc) max_abs {floor['max_abs']:.3e} max_rel {floor['max_rel_above_1e-3']:.3e}")
    entry["files"]["fp32"]["fixtureCheck"] = {
        "lines": len(fx["lines"]), "torchMaxAbs": worst["torch"][0], "torchMaxRel": worst["torch"][1],
        "ortMaxAbs": worst["ort"][0], "ortMaxRel": worst["ort"][1],
        "fixtureFloorMaxAbs": floor["max_abs"], "fixtureFloorMaxRel": floor["max_rel_above_1e-3"]}

    for i in CRNN_B1_IDS:
        write_reference("crnn", f"1-{i}", sha, {"input": byid[i]}, {"logits": sess.run(["logits"], {"input": byid[i]})[0]}, 1)
    batch = np.ones((8, 1, 32, 128), np.float32)  # padded with white (normalised 1.0) on the right
    for n, i in enumerate(CRNN_BATCH_IDS):
        batch[n, :, :, : byid[i].shape[3]] = byid[i][0]
    write_reference("crnn", "8", sha, {"input": batch}, {"logits": sess.run(["logits"], {"input": batch})[0]}, 8)


def do_dbnet(a, man, read_container, FoldedReference, record):
    import onnx
    import torch

    entry = man["models"]["dbnet"]
    src = entry["source"]
    if sha256_file(a.dbnet_onm) != src["weightsSha256"]:
        raise SystemExit("detector container hash differs from the manifest")
    manifest, raw = read_container(a.dbnet_onm)
    cfg = manifest["config"]
    tensors = {k: torch.from_numpy(v) for k, v in raw.items()}

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.names = {}
            for i, (k, v) in enumerate(tensors.items()):
                self.register_buffer(f"w{i}", v)  # buffer names cannot hold dots
                self.names[k] = f"w{i}"

        def forward(self, x):
            o = FoldedReference(cfg, {k: getattr(self, b) for k, b in self.names.items()}).forward(x)
            return o["prob"], o["layout"], o["orient"]

    net = Net().eval()
    out = MODELS / "dbnet" / "dbnet_lite_v1.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(1, 1, 64, 96)
    with torch.no_grad():
        torch.onnx.export(net, (x,), str(out), input_names=["input"], output_names=["prob", "layout", "orient"],
                          dynamic_axes={"input": {2: "height", 3: "width"}, "prob": {2: "h2", 3: "w2"},
                                        "layout": {2: "h8", 3: "w8"}},
                          opset_version=man["onnxOpset"], dynamo=False)
    onnx.checker.check_model(str(out))
    sha = pin(entry, "fp32", out, record, {"opset": man["onnxOpset"], "ops": op_counts(out)})

    fx = json.loads((a.dbnet_fixtures / "fixtures.json").read_text())
    sess = ort_session(out)
    names = ("prob", "layout", "orient")
    worst = {k: {"torch": [0.0, 0.0], "ort": [0.0, 0.0]} for k in names}
    inputs = {}
    for s in fx["samples"]:
        layers = {l["name"]: l for l in s["layers"]}
        inp = read_f32(a.dbnet_fixtures / layers["input"]["file"], layers["input"]["shape"])[None]
        inputs[s["id"]] = inp
        with torch.no_grad():
            t = [v.numpy() for v in net(torch.from_numpy(inp))]
        o = sess.run(list(names), {"input": inp})
        for k, tv, ov in zip(names, t, o):
            ref = read_f32(a.dbnet_fixtures / layers[k]["file"], layers[k]["shape"])
            for who, v in (("torch", tv), ("ort", ov)):
                ma, mr = metrics(v.reshape(ref.shape), ref, "rel-of-max")
                worst[k][who] = [max(worst[k][who][0], ma), max(worst[k][who][1], mr)]
    chk = {}
    for k, w in worst.items():
        fl = fx["fp32_vs_fp64_floor"][k]
        log(f"dbnet {k} vs fixtures over {len(fx['samples'])} inputs: torch max_abs {w['torch'][0]:.3e} max_rel {w['torch'][1]:.3e}; "
            f"ort max_abs {w['ort'][0]:.3e} max_rel {w['ort'][1]:.3e}; fixtures' floor max_abs {fl['max_abs']:.3e} "
            f"max_rel {fl['max_rel_above_1e-3_of_max']:.3e}")
        chk[k] = {"torchMaxAbs": w["torch"][0], "torchMaxRel": w["torch"][1], "ortMaxAbs": w["ort"][0],
                  "ortMaxRel": w["ort"][1], "fixtureFloorMaxAbs": fl["max_abs"],
                  "fixtureFloorMaxRel": fl["max_rel_above_1e-3_of_max"]}
    entry["files"]["fp32"]["fixtureCheck"] = {"inputs": len(fx["samples"]), **chk}
    for i in DBNET_B1_IDS:
        o = sess.run(list(names), {"input": inputs[i]})
        write_reference("dbnet", f"1-{i}", sha, {"input": inputs[i]}, dict(zip(names, o)), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--training-dir", type=Path, default=env_path("ONEOCR_TRAINING", SRC / "training"))
    ap.add_argument("--crnn-onm", type=Path, default=env_path("ONEOCR_CRNN_ONM", SRC / "crnn-v0-0.7.3.onm"))
    ap.add_argument("--dbnet-onm", type=Path, default=env_path("ONEOCR_DBNET_ONM", SRC / "dbnet-lite-v1-0.3.2.onm"))
    ap.add_argument("--crnn-fixtures", type=Path, default=env_path("ONEOCR_CRNN_FIXTURES", SRC / "crnn-parity"))
    ap.add_argument("--dbnet-fixtures", type=Path, default=env_path("ONEOCR_DBNET_FIXTURES", SRC / "dbnet-parity"))
    a = ap.parse_args()
    man = load_manifest()
    read_container, load_folded, FoldedReference = load_training_code(a.training_dir)
    do_crnn(a, man, read_container, load_folded, a.record)
    do_dbnet(a, man, read_container, FoldedReference, a.record)
    if a.record:
        save_manifest(man)


if __name__ == "__main__":
    sys.exit(main())
