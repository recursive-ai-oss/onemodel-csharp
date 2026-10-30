# Oracles, models and reference data

ONNX Runtime and PyTorch are used as black boxes: they produce reference outputs, export models and quantize them. Their code is never read, copied or linked into the library. Everything below is reproduced by the scripts in `tools/`; paths come from `ONEMODEL_HOME` (default `~/onemodel`), and nothing generated is committed.

## Pins

| Package | Version | Used for |
|---|---|---|
| `onnxruntime` (Python) | 1.31.1 | reference outputs, `onnxruntime.quantization` as a tool |
| `Microsoft.ML.OnnxRuntime` (NuGet, `bench/Bar` only) | 1.31.1 | the bar runner's ONNX Runtime side |
| `onnx` | 1.23.2 | checker, model loading |
| `torch` (CPU wheels) | 2.15.0 | export of the torchvision and Hugging Face models |
| `torchvision` | 0.29.2 | ResNet-50 and MobileNetV3 definitions |
| `transformers` | 5.22.0 | BERT, MiniLM and Whisper definitions |
| `onnxscript` | 0.7.2 | required by the torch exporter package |
| `numpy` | 2.5.4 | array files |
| Python | 3.13 (via `uv`) | |

`tools/requirements-oracles.txt` holds the pins and `tools/setup-oracles.sh` creates the venv (`$ONEMODEL_HOME/venvs/oracles`) on Linux x64 and macOS arm64. It is idempotent, installs `uv` user-locally if missing, uses the CPU wheel index for torch on Linux, and writes the full `pip freeze` next to the venv.

## Models

`tools/models.json` is the manifest: for every model the source (URL or repository plus commit revision), the licence as stated there, the SHA-256 of the source weights, the SHA-256 and size of each produced file. `tools/verify_models.py` checks the cache against it.

| Model | Source | Files |
|---|---|---|
| ResNet-50 | torchvision `IMAGENET1K_V1` weights (pinned URL and hash) | fp32, int8 static QDQ |
| MobileNetV3-Large | torchvision `IMAGENET1K_V1` weights | fp32, int8 static QDQ |
| BERT-base | `google-bert/bert-base-uncased` at a pinned commit | fp32, int8 dynamic |
| MiniLM-L6 | `sentence-transformers/all-MiniLM-L6-v2` at a pinned commit | fp32, int8 dynamic |
| Whisper-small | `openai/whisper-small` at a pinned commit | encoder and decoder, each fp32 and int8 dynamic |
| CRNN recogniser `crnn-v0` 0.7.3 | fp16 `.onm` container (family-owned; fetch-only) | fp32 |
| DBNet-style detector `dbnet-lite-v1` 0.3.2 | fp16 `.onm` container (family-owned; fetch-only) | fp32 |

Export (`tools/export_models.py`): TorchScript-based `torch.onnx.export`, opset 17, weights embedded in one file, eval mode, dynamic batch (and sequence for the text models). Encoders output `last_hidden_state` only, with inputs `input_ids`, `attention_mask`, `token_type_ids` (attention is the explicit, eager form). The Whisper encoder takes `input_features` `[B,80,3000]`; the decoder takes `input_ids` and `encoder_hidden_states` and returns logits for the whole sequence without a key/value cache. These are fixed choices of this repository, not the upstream defaults.

### Reproducibility across machines

Exported on x64 Linux and on arm64 macOS with the same pins, BERT-base, MiniLM and Whisper (fp32 and int8) are **byte-identical**. The ResNet-50 and MobileNetV3 exports are **not** (same size, different bytes: the cause was not investigated; the likely one is constant folding of batch-norm into the convolution weights, which is floating-point arithmetic done on the exporting CPU), and their static int8 files follow. The manifest pins the x64 file, and other machines copy it instead of re-exporting; `export_models.py` prints a note when its output differs from the pinned hash.

### The OCR models (`tools/export_oneocr.py`)

Built from their containers with the family's training code imported read-only (the paths come from arguments or `ONEOCR_*` environment variables; the detector module's unused post-processing imports are stubbed so only torch and NumPy are needed). Opset 17, fp32, no CTC in the graph.

* Recogniser: input `[B,1,32,W]` (dynamic B and W), output `logits [B, W//4, 622]`. Ops: Conv 6, Relu 6, MaxPool 5, Transpose 5, LSTM 2 (bidirectional), MatMul 1, Add 1, plus Shape/Gather/ConstantOfShape/Slice/Squeeze/Unsqueeze/Reshape/Concat/Constant plumbing.
* Detector: input `[1,1,H,W]` (dynamic H, W, multiples of 32), outputs `prob [1,1,H/2,W/2]`, `layout [1,8,H/8,W/8]`, `orient [1,4]`. Ops: Conv 61 (depthwise ones with `group = C`), Relu 53, Add 5, Resize 5 (nearest ×2), ReduceMean 2, Concat 1, Gemm 1.

Against the fixtures' fp32 reference tensors (50 recogniser lines; 6 detector inputs; metrics defined in the script's header), maximum over all inputs:

| Output | PyTorch module: abs / rel | ONNX Runtime on the file: abs / rel | Fixtures' own fp32-vs-fp64 floor: abs / rel |
|---|---|---|---|
| recogniser logits | 3.6e-5 / 1.3e-3 | 3.8e-5 / 1.7e-3 | 9.4e-5 / 2.0e-3 |
| detector prob | 2.9e-6 / 1.1e-4 | 6.7e-6 / 2.1e-4 | 7.1e-6 / 1.3e-4 |
| detector layout | 3.8e-5 / 2.6e-5 | 5.3e-5 / 3.2e-5 | 4.2e-5 / 2.1e-5 |
| detector orient | 7.6e-6 / 4.4e-7 | 1.9e-5 / 3.8e-6 | 1.5e-5 / 3.9e-6 |

ONNX Runtime stays within 1.7× of the fixtures' floors on every output. Reference outputs are written for the recogniser at batch 1 on five fixture crops (`b1-<crop id>`) and at batch 8 (`b8`: eight crops right-padded with white to width 128), and for the detector at batch 1 on `tile0` and `wide` (`b1-tile0`, `b1-wide`), so a bar row is e.g. `crnn:fp32:8` or `dbnet:fp32:1-wide`. The exports are byte-stable on repeat runs on the same machine.

## int8 variants (`tools/quantize_models.py`)

* Encoders and Whisper: `quantize_dynamic`, int8 weights, per-channel, activations quantized at run time.
* ResNet-50 and MobileNetV3: `quantize_static`, QDQ format, int8 weights per-channel, int8 activations, MinMax calibration over 32 images. No licensed image set is fetched: the calibration images are seeded synthetic fields (a 7×7×3 normal field expanded 32× plus noise of 0.25), generated by `calibration_images()`. They exercise the quantized kernels but do not calibrate for accuracy.

## ONNX backend test data (`tools/fetch_backend_test.py`)

Only `onnx/backend/test/data/` is extracted from the archive. ONNX 1.23 stopped shipping the node cases (they are generated in memory by the installed package, with expected values from the package itself), so the pinned source is the **1.22.0** release archive, the last with frozen node data. Cases added after 1.22.0 are therefore not covered.

| Directory | Cases |
|---|---|
| `node` | 1765 |
| `pytorch-converted` | 82 |
| `pytorch-operator` | 35 |
| `simple` | 23 |
| `light` (flat model files) | 9 |
| `real` (need model-zoo downloads, not fetched) | 9 |

## Reference outputs (`tools/make_reference.py`)

For every model file and batch size (1 and the model's N: 32 text encoders, 8 conv models, 1 for Whisper) it writes, under `reference/<os>-<arch>/<model>/<variant>/b<batch>/`:

* `inputs/*.npy`, `outputs/*.npy`: NumPy format 1.0, little-endian, C order;
* `index.json`: dtypes, shapes, SHA-256 of each file, the model's SHA-256, the ONNX Runtime version and session options.

Outputs are ONNX Runtime `ORT_ENABLE_ALL`, sequential execution, 1 intra-op thread. Inputs are seeded per sample from a SHA-256 of `(model, kind, sample index)` through PCG64, so batch 1 is sample 0 of batch N on every machine. Conv inputs are standard normal `[3,224,224]`; text inputs are random token ids in `[1000,30000)` with sequence length 128, sample 0 unpadded and the others padded to a random length in `[64,128]`; Whisper uses a normal mel input scaled by 0.5, and the decoder 16 random token ids with a random hidden state.

Reference outputs are per machine class: ONNX Runtime's results differ in the last bits between x64 and arm64.

### ONNX Runtime's own noise: `ORT_ENABLE_ALL` against `ORT_DISABLE_ALL`

`noise.csv` has, per output, three numbers, with `a` = enabled and `b` = disabled: max |a−b|; that divided by max |b| ("scaled"); and max |a−b| / max(|b|, 1e-3·max|b|) ("floored relative"). Results (1 thread):

| Model / variant | Batch | x64 abs | x64 scaled | x64 floored rel | arm64 scaled | arm64 floored rel |
|---|---|---|---|---|---|---|
| ResNet-50 fp32 | 1 / 8 | 1.7e-6 / 1.9e-6 | 2.0e-7 / 2.2e-7 | 3.6e-5 / 6.9e-5 | 2.9e-7 / 2.8e-7 | 2.4e-5 / 5.8e-5 |
| MobileNetV3 fp32 | 1 / 8 | 5.1e-6 / 8.2e-6 | 9.6e-7 / 1.4e-6 | 4.2e-4 / 3.8e-4 | 8.6e-7 / 1.5e-6 | 6.0e-4 / 7.2e-4 |
| BERT-base fp32 | 1 / 32 | 7.3e-6 / 1.3e-5 | 1.1e-6 / 1.5e-6 | 3.9e-4 / 3.3e-4 | 8.5e-7 / 1.9e-6 | 1.1e-4 / 4.3e-4 |
| MiniLM-L6 fp32 | 1 / 32 | 1.3e-6 / 3.8e-6 | 2.2e-7 / 6.1e-7 | 1.8e-4 / 2.5e-4 | 2.8e-7 / 5.8e-7 | 1.2e-4 / 1.9e-4 |
| Whisper encoder fp32 | 1 | 2.1e-4 | 6.0e-6 | 7.4e-4 | 2.2e-5 | 3.2e-3 |
| Whisper decoder fp32 | 1 | 4.9e-5 | 1.9e-6 | 1.1e-3 | 2.5e-6 | 1.2e-3 |

So for fp32, **a 1e-4 relative bound is only meaningful against the scale of the output** (scaled error): ONNX Runtime itself is within ~2e-6 of its unoptimized self on that metric, but an element-wise relative error floored at 1e-3 of the maximum already reaches 1e-4 to 3e-3 between its two optimization levels, because small elements are computed as differences of large terms. The fp32 conformance metric should therefore be the scaled one (or an element-wise one with a floor at least 1e-2 of the maximum).

int8 outputs are much noisier. Between ONNX Runtime's own levels the scaled difference is 7e-3 to 2e-2 (ResNet-50), 8e-2 to 1e-1 (MobileNetV3), 6e-2 to 2e-1 (BERT-base dynamic), 3e-2 to 7e-2 (MiniLM dynamic, x64), and 0.7 for the dynamically quantized Whisper encoder (the dynamic scheme is sensitive to activation outliers). Comparing int8 against ONNX Runtime's int8 output element by element is not a stable metric, and neither is "at most ONNX Runtime's own difference": ONNX Runtime's optimization levels agree to 2e-7 on the dynamically quantized Whisper decoder only because their float kernels are identical, yet rerunning ONNX Runtime itself with 1e-7 relative noise on one input moves that output by 0.10 to 0.21 scaled, with minimum row cosines of 0.95 to 0.99 and 14 to 16 of 16 token decisions unchanged (`tools/int8_sensitivity.py`). One float32 ulp upstream flips a few 8-bit roundings, and the flips spread.

### The int8 metric

A row (model, variant, batch) passes when, for every float output compared with ONNX Runtime's `ORT_ENABLE_ALL` output:

1. **Cosine.** The 1st-percentile cosine similarity over the rows of the last axis (the minimum when there are fewer than 100 rows) is at least `min(0.99, f)`, and for outputs of rank 3 or more (encoders) so is the minimum cosine of the mean-pooled vectors (mean over the second-to-last axis). `f` is the floor defined below. A single extreme row is not gated: with thousands of rows it is an extreme-value statistic that ONNX Runtime itself does not hold from run to run.
2. **Decisions**, for models with a task head only: class per image for classifiers, token per position for the Whisper decoder and the text recognizer. The argmax over the last axis agrees on at least 95% of positions, or on at least as many as the floor agrees. A single image (one decision) must agree. Argmax over a hidden dimension is not a decision, so encoders have no part 2.

The floor `f` of each statistic (and of the decision count) is the smallest value, measured against `ORT_ENABLE_ALL`, over: ONNX Runtime's `ORT_DISABLE_ALL` output; 16 `ORT_ENABLE_ALL` runs with 1e-7 relative noise (about one float32 ulp) on the float inputs; and, for dynamic models whose first quantized input is not a graph input (token-id encoders), 16 runs with the same noise on that tensor. All of them measure ONNX Runtime's own sensitivity, and the worst one is the honest floor. See `int8.md`, "Noise floor of the model-level metric". (For the Whisper decoder the two levels agree to 2e-7, so the noise runs set the floor.)

Scaled error and ONNX Runtime's own difference are printed, not gated. Operator-level int8 (the oracle cases) must match bit for bit. `tools/check_int8.sh` runs `Bar check --dump` and `tools/check_int8.py` on any rows; `Bar check` itself prints the per-output row-cosine minimum and top-1 agreement.

## float16 files

`tools/export_models.py --fp16` writes `<model>_fp16.onnx` next to the float32 file (MiniLM-L6 and ResNet-50): float16 weights and
float16 activations, float32 graph inputs and outputs with a `Cast` at each end, finite values beyond the float16 range clamped to
±65504. `make_reference.py --variant fp16` stores ONNX Runtime's output on the same file and inputs. Both engines run it on the CPU.

Gate: scaled error of ours against ONNX Runtime on the same file ≤ 4e-3, and our error against the float32 model's output ≤ 1.5 times
ONNX Runtime's. Reason: half precision has 11 significant bits (4.9e-4 per rounding), the models chain dozens of roundings, and two
correct engines that round in slightly different places differ by that much. Measured on x64 (AVX-512), fast mode, scaled errors:

| Model (batch) | Ours vs ORT fp16 | ORT fp16 vs ORT fp32 | Ours fp16 vs ORT fp32 | ORT fp16, enabled vs disabled |
|---|---|---|---|---|
| MiniLM-L6 (1) | 9.8e-4 | 7.9e-4 | 1.05e-3 | 6.6e-4 |
| MiniLM-L6 (32) | 1.9e-3 | 1.33e-3 | 1.72e-3 | 1.26e-3 |
| ResNet-50 (1) | 3.6e-4 | 1.37e-3 | 1.02e-3 | 2.9e-7 |
| ResNet-50 (8) | 4.8e-4 | 1.49e-3 | 1.75e-3 | 2.7e-7 |

Output hashes are equal at 1 and 8 threads.
