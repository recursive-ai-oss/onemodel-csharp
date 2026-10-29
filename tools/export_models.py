"""Fetch pinned weights and export the reference models to ONNX (fp32).

    export_models.py [--record] [--force] [name ...]

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


EXPORTERS = {"resnet50": export_tv, "mobilenet_v3_large": export_tv, "bert_base": export_bert,
             "minilm_l6": export_bert, "whisper_small": export_whisper}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(EXPORTERS))
    ap.add_argument("--record", action="store_true", help="write hashes into models.json")
    ap.add_argument("--force", action="store_true", help="re-export even if the file exists")
    a = ap.parse_args()
    man = load_manifest()
    for n in a.names:
        EXPORTERS[n](n, man["models"][n], man["onnxOpset"], a.record, a.force)
        if a.record:
            save_manifest(man)
    if a.record:
        save_manifest(man)


if __name__ == "__main__":
    sys.exit(main())
