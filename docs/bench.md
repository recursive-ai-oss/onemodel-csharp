# GPU speed against ONNX Runtime's GPU providers

`tools/bar.sh` (Linux, macOS) and `tools/bar.ps1` (Windows) time the `.Gpu` backend and a released ONNX Runtime GPU execution provider on the
same model file and the same inputs, one row per model and batch. The numbers they print are **indicative**: one machine, shared host, few
repetitions. Publishable rows come from the benchmark gate, not from these scripts.

## What is compared

| Platform | Our device | Reference provider (a black box: the released package is installed and run, never opened) |
|---|---|---|
| Linux, Windows | Vulkan | Windows: DirectML (`onnxruntime-directml`). Linux: the native WebGPU provider (`onnxruntime` + `onnxruntime-ep-webgpu`, Dawn over Vulkan) |
| macOS | Metal | The faster, per row, of CoreML (`onnxruntime`, GPU compute units only) and the WebGPU provider on Metal (`onnxruntime` + `onnxruntime-ep-webgpu`, universal2 wheel), among the configurations that make the row a gate row; when neither does, the faster one, as an information row |
| Chrome | WebGPU | onnxruntime-web's WebGPU provider in the same Chrome and adapter |

Linux: a released binary does provide the native WebGPU provider. It is a plugin library, `onnxruntime-ep-webgpu` (PyPI, Linux x86-64, Windows,
macOS arm64), registered at run time next to a core `onnxruntime` of version 1.24.4 or later; see
<https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html>. The core wheel alone does not contain it. The plugin takes one GPU
device, so the discrete GPU is chosen. CUDA (`onnxruntime-gpu`) and onemodel's CPU path are timed for information and never gate. Pins
(1.31.1 core, 0.4.0 plugin, Python 3.13) are in `tools/bar/setup_ort.sh` and `setup_ort.ps1` and recorded in every result row.

## Method

- Our time: the median of N warm runs of `Session.Run` with host arrays in and out (`GpuReport` wall time). The provider's time is measured the
  same way by `tools/bar/ort_run.py`. Both include upload and readback; GPU-only kernel time (timestamps) is reported beside ours.
- The provider is tried in several session configurations (WebGPU: default, NHWC, NCHW; CoreML: GPU compute units with ML Program and Neural Network
  formats, each with and without static shapes only). Each configuration runs in its own process, because a provider keeps its first device
  options for the life of the process. The reference time is the fastest configuration that runs on the provider and on the GPU.
- ONNX Runtime silently falls back to its CPU provider when a provider cannot be created or takes no nodes. A configuration with no node on the
  provider is never the reference. Node counts per provider are recorded (`refNodesByProvider`); most models leave 20 to 30 % of their nodes
  (shape arithmetic) to the CPU on WebGPU, and much more on CoreML.
- CoreML's `ALL` compute units may use the Neural Engine. That time is recorded (`refBestAnyComputeUnitsMs`) and not used as the reference.
- Each row records device, driver, provider versions and configuration, and the machine's load every 5 s before, during and after
  (`load-*.log`, summarised in the row). Bar: 1.5x, target 1.15x; Chrome: 2.5x, target 1.5x.
- Windows: `tools\bar.ps1` and `tools\bar\setup_ort.ps1` mirror the Linux scripts and have run on Windows 11 with an Intel Iris Xe (DirectML and the
  WebGPU provider, which picks the Iris Xe through Direct3D 12). The `onnxruntime-directml` wheel is published on its own schedule: 1.24.4 is the newest,
  so DirectML rows come from that version (`DML_ORT_VERSION`) while the other providers use 1.31.1. The runtime writes its verbose log to a pipe as
  UTF-16 on Windows; `ort_run.py` decodes it. Give `bar.ps1` an `-Out` folder: its default is under the home folder.

## Gate rows and information rows

A reference time only means something when the reference ran the model on its GPU. `ort_run.py` therefore reads, for every configuration, where the
runtime put the model (its own verbose session log, black box) and marks a row a **gate row** only when every compute node ran on the provider's
GPU device. Everything else is an **information** row: reported with its ratio, never judged against the bar.

- Provider level: the session log lists the nodes left to the CPU provider. Nodes that only move or describe data (Shape, Reshape, Gather, Concat,
  Cast, Slice, comparisons and so on, the list is `METADATA_OPS` in the script) don't count; nor does **shape and mask arithmetic**, defined as narrowly as
  possible and decided from the model's graph by each node's inputs, never by its name: a value is a *dimension* when it is a Shape output or is computed
  only from dimensions and constants (by data-movement operators or by the arithmetic below); the mask input is a graph input whose name contains
  "mask", and a value is *from the mask* when it is computed from it with all other inputs dimensions or constants; a node is shape or mask arithmetic
  when it is an Add, Sub, Mul, Div or Neg whose every non-constant input is a dimension or from the mask. That is the Add and Mul the transformer
  exports use to build gather indices into the attention mask before it is broadcast, and the same kind of index arithmetic in the speech decoder, which
  builds its causal mask from the length of its token input. Arithmetic with any other input (activations, weights, another graph input's values) is compute.
  Any other node left on the CPU provider
  (MatMulInteger and DynamicQuantizeLinear in the 8-bit models, for example) makes the row an information row. The reason is recorded
  (`refGateReason`) with the operators; shape or mask arithmetic left on the CPU is counted (`maskArithmeticOnCpu`) and named in the reason.
- Core ML: a node taken by Core ML is not necessarily on the GPU. With the provider option `ProfileComputePlan` Core ML reports the device of
  every operation of its program; the compute operations (not the data-movement ones) must all be on the GPU. The neural network format has no
  such report, so a row whose best configuration is that format is an information row even when ONNX Runtime left no compute node to the CPU.
- The reference is the fastest configuration that satisfies this; when none does, the fastest GPU-only configuration, which makes the row an
  information row. `collect.py` prints the class and the placement (`CPU 239, CoreML 7; compute nodes on CPU provider 125; Core ML compute ops ...`).

`tools/bar/reclassify.sh SRC DST --platform P --ref R` classifies an earlier run again from the placement logs without timing anything.

On the M2 (ONNX Runtime 1.31.1, WebGPU plugin 0.4.0, indicative; `bar.sh --ref coreml,webgpu`) 14 of the 20 rows are gate rows. The WebGPU provider is the
reference in twelve (MiniLM float32 at both batches and float16, BERT float32 at both batches, MobileNetV3 float32 at both batches, DBNet, CRNN,
ResNet-50 float16, Whisper encoder and decoder float32) and Core ML in the two ResNet-50 float32 rows. The 8-bit rows leave MatMulInteger, DynamicQuantizeLinear
and ConvInteger to the CPU, so they are information rows; the speech decoder's two arithmetic nodes on dimensions no longer count. Core ML leaves 125 to 245
compute nodes of every transformer to its CPU provider. The plugin wheel for macOS is universal2 (`onnxruntime-ep-webgpu` 0.4.0,
`tools/bar/setup_ort.sh webgpu`).

On the RTX 4070 (Linux, WebGPU provider on Vulkan, same pins) 14 of the 20 rows are gate rows. The five MiniLM and BERT rows (float32 and float16) and the
speech decoder are gate rows only through the shape and mask rule (an Add and a Mul left on the CPU provider); the 8-bit rows are information rows
(MatMulInteger, DynamicQuantizeLinear, QuantizeLinear or ConvInteger on the CPU).

## Running

    tools/bar/setup_ort.sh webgpu                 # or coreml; cuda for the information column
    tools/bar.sh --platform linux --ref webgpu --cuda ROW...
    tools/bar.sh --platform macos --ref coreml,webgpu ROW...          # both providers; --only webgpu --ref-only re-runs just one
    tools/bar.sh --platform linux --ref webgpu --phase cpu ROW...      # CPU columns, as a separate job
    tools/bar.sh --rows                           # the row names
    tools/gpu_browser.sh --adapters nvidia --modes fast --no-native --ort --runs 10 ROW...   # then bar/collect.py DIR --web --platform P

Results land in `DIR/bar-PLATFORM.jsonl` (one JSON row per model and batch) and `.md`.

## Results at the 1.0 gate

Speed is judged against the reference provider on the same machine and the same model file; ratio is ours divided by the reference, so lower is better.
The bar is 1.5 on a desktop (target 1.15) and 2.5 in a browser (target 1.5). `met` is a ratio above the target and within the bar, `target` is a ratio
within the target, `missed` is above the bar. Only **gate rows** are judged (the reference ran every compute node on its GPU, see above); **information
rows** are listed with their ratio and never judged. All three platforms are rows of `.Gpu` with `RecursiveAI.OneGpu`, fast mode, validation
layers off, host arrays in and out, the median of warm runs (Linux 20, Apple M2 10, Chrome 10 per page) in one benchmark-queue job per group of rows,
each group alone on its machine. Load average sampled every 5 s before, during and after is in every row of the jsonl; the one-minute load average
at the start of a group still carries the harness's own CPU reference runs of the group before it.
The library code measured here is the one released as `RecursiveAI.OneGpu` 1.0.0, which adds one Vulkan synchronization fix (staging and first-use barriers) and changes no result bit.
Reference versions: ONNX Runtime 1.31.1 with the `onnxruntime-ep-webgpu` 0.4.0 plugin (Dawn over Vulkan on Linux, over Metal on macOS), Core ML through
the same wheel on GPU compute units, onnxruntime-web 1.31.1.

### Linux, RTX 4070 (Vulkan, driver 595.99.02), Ryzen 9 7950X3D, 32 threads; reference: WebGPU provider

Queue jobs 013164, 013175, 013176 (three groups). Gate rows 14 of 20: **all met**, worst ResNet-50 float32 at batch 8 (1.40) and BERT float32 at batch 8 (1.35);
the other twelve are at or below the 1.15 target. The int8 rows are information rows: the reference leaves MatMulInteger, DynamicQuantizeLinear,
QuantizeLinear or ConvInteger to its CPU provider, so it is 17 to 35 times slower than ours. CUDA and onemodel CPU columns were not run for the gate.

| row | ours ms (GPU kernels) | reference | ref ms | ours / ref | class | verdict | other provider |
|---|---|---|---|---|---|---|---|
| minilm_fp32_b1 | 1.69 (1.17) | WebGPU EP | 4.309 | 0.39 | gate | target | - |
| minilm_fp32_b8 | 5.5 (4.83) | WebGPU EP | 6.4 | 0.86 | gate | target | - |
| minilm_fp16_b1 | 2.24 (1.42) | WebGPU EP | 3.982 | 0.56 | gate | target | - |
| minilm_int8_b1 | 2.09 (1.37) | WebGPU EP | 48.566 | 0.04 | information | target | - |
| bert_fp32_b1 | 6.49 (5.58) | WebGPU EP | 7.471 | 0.87 | gate | target | - |
| bert_fp32_b8 | 31.19 (30.35) | WebGPU EP | 23.188 | 1.35 | gate | met | - |
| bert_int8_b1 | 6.33 (5.05) | WebGPU EP | 115.786 | 0.05 | information | target | - |
| resnet50_fp32_b1 | 3.18 (2.7) | WebGPU EP | 3.73 | 0.85 | gate | target | - |
| resnet50_fp32_b8 | 16.44 (15.21) | WebGPU EP | 11.773 | 1.40 | gate | met | - |
| resnet50_fp16_b1 | 3.87 (3.07) | WebGPU EP | 4.019 | 0.96 | gate | target | - |
| resnet50_int8_b1 | 4.36 (3.39) | WebGPU EP | 104.208 | 0.04 | information | target | - |
| mobilenet_fp32_b1 | 1.45 (0.99) | WebGPU EP | 1.899 | 0.76 | gate | target | - |
| mobilenet_fp32_b8 | 3.53 (2.74) | WebGPU EP | 3.666 | 0.96 | gate | target | - |
| mobilenet_int8_b1 | 3.23 (1.96) | WebGPU EP | 112.891 | 0.03 | information | target | - |
| dbnet_fp32_b1 | 2.48 (1.8) | WebGPU EP | 3.6 | 0.69 | gate | target | - |
| crnn_fp32_b1 | 0.95 (0.52) | WebGPU EP | 19.594 | 0.05 | gate | target | - |
| whisper_enc_b1 | 57.4 (56.1) | WebGPU EP | 52.142 | 1.10 | gate | target | - |
| whisper_dec_b1 | 14.37 (12.38) | WebGPU EP | 13.55 | 1.06 | gate | target | - |
| whisper_enc_int8_b1 | 32.88 (31.46) | WebGPU EP | 650.538 | 0.05 | information | target | - |
| whisper_dec_int8_b1 | 10.45 (7.95) | WebGPU EP | 184.255 | 0.06 | information | target | - |

### Apple M2 (Metal, macOS 27.0.1); reference: the faster gate configuration of Core ML (GPU compute units) and the WebGPU provider on Metal

Three groups in the benchmark queue, 10 runs per row. Gate rows 14 of 20: twelve are met, **two are misses for 1.0: ResNet-50 float32 at batch 1 (2.16) and
batch 8 (2.21) against Core ML on the GPU** (6.07 ms and 35.27 ms; Core ML MLProgram on the GPU computes in float32 to 1e-6, so it is a fair reference).
Against the WebGPU provider on Metal the same rows are 0.82 (13.09 against 15.94 ms) and 0.81 (77.94 against 95.71 ms). The miss is documented and ships:
the levers left are per-model weight preparation for contiguous 3x3 reads and a Winograd F(2x2,3x3) kernel, estimated at about 1.8 times Core ML at batch 8, so
not enough to close it. MobileNetV3 float32 at batch 8 (1.16) is just above the target. The int8 rows are information rows (MatMulInteger, DynamicQuantizeLinear
and ConvInteger on the reference's CPU provider).

| row | ours ms (GPU kernels) | reference | ref ms | ours / ref | class | verdict | other provider |
|---|---|---|---|---|---|---|---|
| minilm_fp32_b1 | 6.67 (6.19) | WebGPU EP | 9.769 | 0.68 | gate | target | Core ML (GPU) 10.404 (information) |
| minilm_fp32_b8 | 25.91 (25.13) | WebGPU EP | 40.709 | 0.64 | gate | target | Core ML (GPU) 72.514 (information) |
| minilm_fp16_b1 | 10.61 (9.72) | WebGPU EP | 9.55 | 1.11 | gate | target | Core ML (GPU) 11.91 (information) |
| minilm_int8_b1 | 12.65 (11.61) | Core ML (GPU) | 6.917 | 1.83 | information | missed | WebGPU EP 18.781 (information) |
| bert_fp32_b1 | 24.7 (23.99) | WebGPU EP | 41.456 | 0.60 | gate | target | Core ML (GPU) 66.332 (information) |
| bert_fp32_b8 | 156.11 (155.02) | WebGPU EP | 221.278 | 0.71 | gate | target | Core ML (GPU) 512.805 (information) |
| bert_int8_b1 | 49.59 (48.7) | Core ML (GPU) | 29.128 | 1.70 | information | missed | WebGPU EP 53.508 (information) |
| resnet50_fp32_b1 | 13.09 (12.5) | Core ML (GPU) | 6.068 | 2.16 | gate | missed | WebGPU EP 15.937 (gate) |
| resnet50_fp32_b8 | 77.94 (76.77) | Core ML (GPU) | 35.268 | 2.21 | gate | missed | WebGPU EP 95.712 (gate) |
| resnet50_fp16_b1 | 13.13 (12.41) | WebGPU EP | 12.462 | 1.05 | gate | target | Core ML (GPU) 5.409 (information) |
| resnet50_int8_b1 | 21.87 (20.79) | Core ML (GPU) | 18.541 | 1.18 | information | met | WebGPU EP 69.618 (information) |
| mobilenet_fp32_b1 | 5.33 (4.83) | WebGPU EP | 5.045 | 1.06 | gate | target | Core ML (GPU) 9.886 (information) |
| mobilenet_fp32_b8 | 21.04 (19.88) | WebGPU EP | 18.072 | 1.16 | gate | met | Core ML (GPU) 28.363 (information) |
| mobilenet_int8_b1 | 11.17 (9.89) | Core ML (GPU) | 13.924 | 0.80 | information | target | WebGPU EP 50.786 (information) |
| dbnet_fp32_b1 | 17.47 (16.44) | WebGPU EP | 17.859 | 0.98 | gate | target | Core ML (GPU) 13.791 (information) |
| crnn_fp32_b1 | 2.79 (2.37) | WebGPU EP | 48.142 | 0.06 | gate | target | Core ML (GPU) 1.534 (information) |
| whisper_enc_b1 | 371.95 (370.23) | WebGPU EP | 476.461 | 0.78 | gate | target | Core ML (GPU) 1047.15 (information) |
| whisper_dec_b1 | 88.27 (85.95) | WebGPU EP | 79.586 | 1.11 | gate | target | Core ML (GPU) 132.07 (information) |
| whisper_enc_int8_b1 | 609.78 (607.89) | WebGPU EP | 563.7 | 1.08 | information | target | Core ML (GPU) 567.703 (information) |
| whisper_dec_int8_b1 | 86.91 (85.48) | Core ML (GPU) | 48.924 | 1.78 | information | missed | WebGPU EP 104.941 (information) |

### Chrome 155 on the RTX 4070, WebGPU (ahead-of-time compiled page); reference: onnxruntime-web WebGPU provider

Two groups in the benchmark queue (jobs 013179, 013183), ten warm runs per page, same adapter. The twelve float32 rows; every one is within the 2.5 bar and
eleven within the 1.5 target (ResNet-50 batch 8, 1.54, is the only one above). The float16 rows decline in this browser (no `shader-f16` on this Chrome) and the
int8 rows are information, so none is a bar row here; they run (fast and exact, two-run output identical) in the device matrix below.

| row | ours ms | onnxruntime-web ms | ours / ref | verdict (bar 2.5, target 1.5) |
|---|---|---|---|---|
| minilm_fp32_b1 | 3.8 | 7.2 | 0.53 | target |
| minilm_fp32_b8 | 10 | 9.3 | 1.08 | target |
| bert_fp32_b1 | 10.6 | 10.4 | 1.02 | target |
| bert_fp32_b8 | 38.6 | 26.8 | 1.44 | target |
| resnet50_fp32_b1 | 4.8 | 6.2 | 0.77 | target |
| resnet50_fp32_b8 | 21.6 | 14 | 1.54 | met |
| mobilenet_fp32_b1 | 3.7 | 5.8 | 0.64 | target |
| mobilenet_fp32_b8 | 7.5 | 9 | 0.83 | target |
| dbnet_fp32_b1 | 5.2 | 9.3 | 0.56 | target |
| crnn_fp32_b1 | 3.2 | 21.2 | 0.15 | target |
| whisper_enc_b1 | 68.4 | 56.6 | 1.21 | target |
| whisper_dec_b1 | 22.3 | 17.7 | 1.26 | target |

### Windows, Intel Iris Xe (Windows 11, driver 101.7085), indicative only

Not publishable: one run on a shared laptop, ONNX Runtime 1.24.4 DirectML wheel, `tools/bar.ps1`. The fp32 transformer rows (BERT, MiniLM), ResNet-50 and Whisper
**miss the 1.5 bar against DirectML (1.53 to 2.89)**: that GPU has no cooperative matrix, so the portable tiled kernels run. They ship as documented misses.
NativeAOT on Windows is untested.

| row | ours ms | DirectML ms | ours / DirectML | WebGPU EP ms | ours / WebGPU EP |
|---|---|---|---|---|---|
| bert_fp32_b1 | 116.22 | 42.543 | 2.73 | 132.137 | 0.88 |
| bert_fp32_b8 | 871.4 | 301.285 | 2.89 | 577.978 | 1.51 |
| bert_int8_b1 | 47.61 | 129.402 | 0.37 | 281.994 | 0.17 |
| crnn_fp32_b1 | 5.41 | 18.992 | 0.28 | 81.591 | 0.07 |
| dbnet_fp32_b1 | 34.32 | 27.237 | 1.26 | 41.974 | 0.82 |
| minilm_fp16_b1 | 21.06 | 22.841 | 0.92 | 16.371 | 1.29 |
| minilm_fp32_b1 | 28.64 | 10.917 | 2.62 | 19.652 | 1.46 |
| minilm_fp32_b8 | 113.63 | 45.324 | 2.51 | 92.67 | 1.23 |
| minilm_int8_b1 | 15.7 | 23.65 | 0.66 | 34.099 | 0.46 |
| mobilenet_fp32_b1 | 11.72 | 9.625 | 1.22 | 11.516 | 1.02 |
| mobilenet_fp32_b8 | 50.23 | 32.534 | 1.54 | 42.67 | 1.18 |
| mobilenet_int8_b1 | 16.78 | 48.483 | 0.35 | 481.871 | 0.03 |
| resnet50_fp16_b1 | 79.69 | 30.107 | 2.65 | 30.548 | 2.61 |
| resnet50_fp32_b1 | 51.16 | 33.364 | 1.53 | 40.245 | 1.27 |
| resnet50_fp32_b8 | 375.45 | 141.546 | 2.65 | 216.923 | 1.73 |
| resnet50_int8_b1 | 63.39 | 88.177 | 0.72 | 539.153 | 0.12 |
| whisper_dec_b1 | 315.7 | 139.074 | 2.27 | 203.836 | 1.55 |
| whisper_dec_int8_b1 | 124.29 | 312.933 | 0.4 | 1135.404 | 0.11 |
| whisper_enc_b1 | 1843.84 | 789.523 | 2.34 | 1388.869 | 1.33 |
| whisper_enc_int8_b1 | 960.42 | 1051.38 | 0.91 | 5388.266 | 0.18 |

### Correctness matrix at the gate

`OneModel.Gpu.Tests` (385 tests, 0 failed, 0 skipped): RTX 4070 and llvmpipe and RADV (Mesa 25.2.8) with the Khronos validation layer on, 0 validation messages;
Apple M2 with Metal API and shader validation fatal. All 21 rows (20 above plus the greedy decode) with `RequireGpu`, fast and exact: 100 % of nodes in one region on
every device, outputs identical between two runs in each mode (40 of 40 model-mode pairs per device), 0 validation messages. float32 rows are within a scaled error of 1.03e-5
of the CPU's exact mode on every device (Whisper-small encoder 8.0e-5), float16 within 1.2e-3, the 8-bit rows within the int8 policy (model-level error against the CPU
exact mode 3.9e-4 to 8.8e-1, reported next to the CPU's own fast-versus-exact drift). CRNN greedy ids and the Whisper decoder's greedy tokens (8 of 8) equal the CPU's on
every device. The Whisper-small encoder at batch 4 against a float64 evaluation, 16 inputs: mean error 1.14 (fast) and 1.21 (exact) times the CPU exact mode's on the RTX 4070 (max 0.90 and 0.97 times),
1.11 and 1.22 on the M2 (max 1.17 and 1.00), within 1.5 times the CPU exact mode's error. Chrome (RTX 4070): all rows fast and exact with `RequireGpu`, one region, identical runs, no page errors;
SwiftShader subset (MiniLM, DBNet, CRNN, ResNet-50, Whisper decoder) likewise. `OneModel.Tests` 999 passed, 4 skipped, golden table equal to the base, NativeAOT smoke
publish without trim or AOT warnings, run on RTX 4070 and llvmpipe with validation (all ConvInteger and model checks `ok`).
