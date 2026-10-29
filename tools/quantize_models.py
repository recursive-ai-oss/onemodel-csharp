"""Create int8 variants with onnxruntime.quantization (used purely as a tool).

    quantize_models.py [--record] [--force] [name ...]

* Encoders (bert_base, minilm_l6) and the whisper_small graphs: dynamic quantization, int8
  per-channel weights, activations quantized at run time.
* Conv models (resnet50, mobilenet_v3_large): static QDQ, int8 weights per channel, int8
  activations, MinMax calibration over CALIB_IMAGES seeded synthetic images (see docs/oracles.md:
  no licensed image set is fetched, so the calibration images are smooth random fields with the
  statistics of normalised photographs).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from common import MODELS, load_manifest, log, save_manifest, seeded_rng
from export_models import finish

CALIB_IMAGES = 32


def calibration_images(name: str) -> list[np.ndarray]:
    out = []
    for i in range(CALIB_IMAGES):
        rng = seeded_rng("calib", name, i)
        low = rng.standard_normal((1, 3, 7, 7), dtype=np.float32)
        img = np.repeat(np.repeat(low, 32, axis=2), 32, axis=3)  # 224x224 blocks
        img += 0.25 * rng.standard_normal(img.shape, dtype=np.float32)
        out.append(img.astype(np.float32))
    return out


def quant_dynamic(src: Path, dst: Path, name: str) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, per_channel=True)


def quant_static_qdq(src: Path, dst: Path, name: str) -> None:
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calibration_images(name))

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"input": x}

    quantize_static(str(src), str(dst), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    per_channel=True, calibrate_method=CalibrationMethod.MinMax)


# name -> [(source key, output key, mode)]
PLAN = {
    "resnet50": [("fp32", "int8_qdq", "static")],
    "mobilenet_v3_large": [("fp32", "int8_qdq", "static")],
    "bert_base": [("fp32", "int8_dyn", "dynamic")],
    "minilm_l6": [("fp32", "int8_dyn", "dynamic")],
    "whisper_small": [("encoder_fp32", "encoder_int8_dyn", "dynamic"),
                      ("decoder_fp32", "decoder_int8_dyn", "dynamic")],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(PLAN))
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    man = load_manifest()
    for n in a.names:
        entry = man["models"][n]
        for srck, dstk, mode in PLAN[n]:
            src = MODELS / entry["files"][srck]["path"]
            dst = MODELS / n / f"{n}_{dstk}.onnx"
            if a.force or not dst.exists():
                log("quantizing", src.name, "->", dst.name, mode)
                (quant_static_qdq if mode == "static" else quant_dynamic)(src, dst, n)
            slot = entry["files"].setdefault(dstk, {})
            slot["path"] = str(dst.relative_to(MODELS))
            slot["quantization"] = mode
            finish(n, dstk, dst, entry, a.record)
        if a.record:
            save_manifest(man)


if __name__ == "__main__":
    sys.exit(main())
