"""Fetch pinned weights and export the reference models to ONNX (fp32), and derive fp16 variants.

    export_models.py [--record] [--force] [name ...]
    export_models.py --fp16 [--record] [--force] [name ...]    (default names: minilm_l6 resnet50)

Weights come from pinned URLs / Hugging Face revisions and are checked against the SHA-256s in
models.json. Exports go to <cache>/models/<name>/. With --record the resulting hashes are written
back into models.json; without it they are compared with the recorded ones and a difference is
reported (exports are not guaranteed to be byte-identical across machines: the manifest records
the canonical file and other machines copy it).
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import common
from common import MODELS, load_manifest, log, save_manifest, sha256_file


def download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        log("downloading", url)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    return dest


def check_or_record(entry: dict, key: str, value: str, record: bool, what: str) -> None:
    old = entry.get(key, "")
    if not old or record:
        entry[key] = value
    elif old != value:
        raise SystemExit(f"{what}: hash mismatch (manifest {old}, got {value})")


def hf_weights(src: dict, record: bool) -> Path:
    from huggingface_hub import snapshot_download

    d = snapshot_download(
        src["repo"],
        revision=src["revision"],
        allow_patterns=[src["weightsFile"], "*.json", "*.txt"],
    )
    p = Path(d) / src["weightsFile"]
    check_or_record(src, "weightsSha256", sha256_file(p), record, f"{src['repo']} weights")
    return Path(d)


def torchvision_weights(src: dict, record: bool) -> Path:
    from common import CACHE

    p = download(src["weightsUrl"], CACHE / "weights" / src["weightsUrl"].rsplit("/", 1)[-1])
    check_or_record(src, "weightsSha256", sha256_file(p), record, src["weightsName"])
    return p


def export(model, args, path: Path, input_names, output_names, dynamic_axes, opset: int) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        # The TorchScript-based exporter yields plain, widely supported graphs (LayerNormalization
        # at opset 17, no function wrappers), which is what the runtime's importer is tested on.
        torch.onnx.export(
            model, args, str(path), input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes, opset_version=opset, dynamo=False,
        )


def finish(name: str, key: str, path: Path, entry: dict, record: bool) -> None:
    import onnx

    onnx.checker.check_model(str(path))
    files = entry.setdefault("files", {})
    f = files.setdefault(key, {"path": str(path.relative_to(MODELS))})
    h = sha256_file(path)
    if record or not f.get("sha256"):
        f["sha256"] = h
        f["bytes"] = path.stat().st_size
    elif f["sha256"] != h:
        log(f"NOTE {name}/{key}: export differs from the recorded canonical file ({h} vs {f['sha256']}); "
            "copy the canonical file instead of using this one")
        f["localSha256"] = h
    log(f"{name}/{key}: {path.name} {path.stat().st_size} bytes sha256 {h}")


def export_tv(name: str, entry: dict, opset: int, record: bool, force: bool) -> None:
    import torch
    import torchvision

    out = MODELS / name / f"{name}.onnx"
    w = torchvision_weights(entry["source"], record)
    if out.exists() and not force:
        return finish(name, "fp32", out, entry, record)
    ctor = {"resnet50": torchvision.models.resnet50,
            "mobilenet_v3_large": torchvision.models.mobilenet_v3_large}[name]
    m = ctor(weights=None)
    m.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
    m.eval()
    x = torch.zeros(1, 3, 224, 224)
    export(m, (x,), out, ["input"], ["logits"], {"input": {0: "batch"}, "logits": {0: "batch"}}, opset)
    finish(name, "fp32", out, entry, record)


def export_bert(name: str, entry: dict, opset: int, record: bool, force: bool) -> None:
    import torch
    from transformers import AutoModel

    out = MODELS / name / f"{name}.onnx"
    d = hf_weights(entry["source"], record)
    if out.exists() and not force:
        return finish(name, "fp32", out, entry, record)
    m = AutoModel.from_pretrained(d, attn_implementation="eager").eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids, return_dict=False)[0]

    ids = torch.ones(2, 16, dtype=torch.int64)
    ax = {"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
          "token_type_ids": {0: "batch", 1: "seq"}, "last_hidden_state": {0: "batch", 1: "seq"}}
    export(Wrap(m), (ids, ids, torch.zeros_like(ids)), out,
           ["input_ids", "attention_mask", "token_type_ids"], ["last_hidden_state"], ax, opset)
    finish(name, "fp32", out, entry, record)


def export_whisper(name: str, entry: dict, opset: int, record: bool, force: bool) -> None:
    import torch
    from transformers import WhisperForConditionalGeneration

    enc_out = MODELS / name / f"{name}_encoder.onnx"
    dec_out = MODELS / name / f"{name}_decoder.onnx"
    d = hf_weights(entry["source"], record)
    if enc_out.exists() and dec_out.exists() and not force:
        finish(name, "encoder_fp32", enc_out, entry, record)
        return finish(name, "decoder_fp32", dec_out, entry, record)
    m = WhisperForConditionalGeneration.from_pretrained(d, attn_implementation="eager").eval()

    class Enc(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.e = m.model.encoder

        def forward(self, input_features):
            return self.e(input_features=input_features, return_dict=False)[0]

    class Dec(torch.nn.Module):
        """Decoder without a key/value cache: full-sequence logits, enough to time and compare."""

        def __init__(self, m):
            super().__init__()
            self.d, self.head = m.model.decoder, m.proj_out

        def forward(self, input_ids, encoder_hidden_states):
            # Explicit positions keep the exporter off the packed-sequence detection (aten::diff).
            pos = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            h = self.d(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states,
                       attention_mask=torch.ones_like(input_ids), position_ids=pos, cache_position=pos[0], use_cache=False, return_dict=False)[0]
            return self.head(h)

    mel = torch.zeros(1, 80, 3000)
    export(Enc(m), (mel,), enc_out, ["input_features"], ["encoder_hidden_states"],
           {"input_features": {0: "batch"}, "encoder_hidden_states": {0: "batch"}}, opset)
    ids = torch.ones(1, 8, dtype=torch.int64)
    hs = torch.zeros(1, 1500, m.config.d_model)
    export(Dec(m), (ids, hs), dec_out, ["input_ids", "encoder_hidden_states"], ["logits"],
           {"input_ids": {0: "batch", 1: "seq"}, "encoder_hidden_states": {0: "batch"},
            "logits": {0: "batch", 1: "seq"}}, opset)
    finish(name, "encoder_fp32", enc_out, entry, record)
    finish(name, "decoder_fp32", dec_out, entry, record)


FP16_MAX = 65504.0
# Operator inputs that must stay float32 (ONNX types them separately from the data).
KEEP_FP32 = {("Resize", 1), ("Resize", 2), ("Upsample", 1)}


def to_fp16(src: Path, dst: Path) -> None:
    """Rewrites a float32 model with float16 weights and float16 activations.

    Float graph inputs and outputs keep their float32 type, with a Cast at each end, so the same inputs
    drive both files. Finite values beyond the float16 range (the -3.4e38 of an attention mask) are
    clamped to the largest finite float16 instead of becoming infinity, which would turn 0 * -inf into NaN.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(str(src))
    g = m.graph
    FLOAT, HALF = TensorProto.FLOAT, TensorProto.FLOAT16

    def half_tensor(t):
        a = numpy_helper.to_array(t).astype(np.float32)
        a = np.where(np.isfinite(a), np.clip(a, -FP16_MAX, FP16_MAX), a).astype(np.float16)
        return numpy_helper.from_array(a, t.name)

    keep = {(n.input[i]) for n in g.node for i in range(len(n.input)) if (n.op_type, i) in KEEP_FP32}
    for i, t in enumerate(list(g.initializer)):
        if t.data_type == FLOAT and t.name not in keep:
            g.initializer[i].CopyFrom(half_tensor(t))
    for n in g.node:
        if n.op_type in ("Cast",):
            for a in n.attribute:
                if a.name == "to" and a.i == FLOAT:
                    a.i = HALF
        for a in n.attribute:
            if a.type == onnx.AttributeProto.TENSOR and a.t.data_type == FLOAT and n.op_type in ("Constant", "ConstantOfShape"):
                a.t.CopyFrom(half_tensor(a.t))
    del g.value_info[:]

    head, tail = [], []
    renames = {}
    for v in g.input:
        if v.type.tensor_type.elem_type == FLOAT and v.name not in {t.name for t in g.initializer}:
            renames[v.name] = v.name + "__half"
            head.append(helper.make_node("Cast", [v.name], [renames[v.name]], to=HALF, name="cast_in_" + v.name))
    for n in g.node:
        for i, x in enumerate(n.input):
            if x in renames:
                n.input[i] = renames[x]
    produced = {o: n for n in g.node for o in n.output}
    for v in g.output:
        if v.type.tensor_type.elem_type == FLOAT and v.name in produced:
            inner = v.name + "__half"
            node = produced[v.name]
            for i, o in enumerate(node.output):
                if o == v.name:
                    node.output[i] = inner
            for n in g.node:
                for i, x in enumerate(n.input):
                    if x == v.name:
                        n.input[i] = inner
            tail.append(helper.make_node("Cast", [inner], [v.name], to=FLOAT, name="cast_out_" + v.name))
    nodes = head + list(g.node) + tail
    del g.node[:]
    g.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))


def export_fp16(name: str, entry: dict, record: bool, force: bool) -> None:
    src = MODELS / entry["files"]["fp32"]["path"]
    out = MODELS / name / f"{name}_fp16.onnx"
    if force or not out.exists():
        log("converting", src.name, "->", out.name)
        to_fp16(src, out)
    entry["files"].setdefault("fp16", {"path": str(out.relative_to(MODELS)), "conversion": "tools/export_models.py to_fp16 (weights and activations float16, float32 inputs and outputs)"})
    finish(name, "fp16", out, entry, record)


EXPORTERS = {"resnet50": export_tv, "mobilenet_v3_large": export_tv, "bert_base": export_bert,
             "minilm_l6": export_bert, "whisper_small": export_whisper}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--fp16", action="store_true", help="derive the float16 variants from the recorded float32 files")
    ap.add_argument("--record", action="store_true", help="write hashes into models.json")
    ap.add_argument("--force", action="store_true", help="re-export even if the file exists")
    a = ap.parse_args()
    man = load_manifest()
    if a.fp16:
        for n in a.names or ["minilm_l6", "resnet50"]:
            export_fp16(n, man["models"][n], a.record, a.force)
            if a.record:
                save_manifest(man)
        return
    for n in a.names or list(EXPORTERS):
        EXPORTERS[n](n, man["models"][n], man["onnxOpset"], a.record, a.force)
        if a.record:
            save_manifest(man)
    if a.record:
        save_manifest(man)


if __name__ == "__main__":
    sys.exit(main())
