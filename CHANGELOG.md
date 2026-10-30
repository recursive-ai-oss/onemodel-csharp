# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Faster convolution in fast mode: regular and grouped convolutions gather their input patches straight into the packed layout (no im2col matrix, no separate packing phase), their weights are packed once per session (`conv-pack` pass), tiles are balanced, and small layers use more threads (`docs/kernels-conv.md`). A single-row matrix product uses a matrix-vector kernel; large element-wise outputs and float `MaxPool` run in parallel chunks. Fast-mode results are independent of the thread count as before; exact mode is unchanged.
- A convolution followed by an `Add` of a skip connection adds it on the finished tile (graph pass `conv-residual`, fast mode); ResNet-50 at 8 threads about 10% faster.

## [0.3.0-preview.1] - 2026-10-30

Our own model file, ONNX backend-test conformance with control flow, and exact-mode results checked bit for bit across platforms. Of the 1905 `onnx/backend/test` cases, 1580 pass in fast mode and 1382 in exact mode, none fail, and the rest are listed gaps (`docs/conformance.md`). Exact-mode output hashes are equal on Windows, Linux and macOS (x64 and arm64, JIT and NativeAOT) and on browser WebAssembly (`docs/golden-hashes.md`).

### Added

- Exact-mode golden hashes: `tools/GoldenModels`, `tools/golden*.sh`, and a CI check on every platform against `testdata/graphs/golden-exact.txt`.
- The OneModel model file (`ModelFormat`, `docs/format.md`; format version 1, unstable until 1.0): a 64-byte header, a JSON manifest of the graph and 64-byte-aligned raw weights that load as memory-mapped views. `ModelFormat.Convert` turns an ONNX file into one, `ModelFormat.Save` writes an imported model, and `Model.Load` / `Session.Load` open either kind by its first eight bytes. A converted model gives bit-identical outputs. Bad files fail with `ModelFormatException` naming the field.
- `OnnxSweep conformance` and `tools/conformance.sh`: one runner over `onnx/backend/test` (node, simple, pytorch-converted, pytorch-operator) in fast and exact mode, writing per-case status and `docs/conformance.md`.
- About 70 operators for backend-test coverage (activations and trigonometry, bitwise, scatter and set operators, loss functions, LRN, Einsum, DFT/STFT and windows, GridSample, RoiAlign, NonMaxSuppression, Attention opset 23 and others; `docs/ops-coverage.md`), ConvTranspose and Conv with three spatial axes, Resize `antialias`, MelWeightMatrix and IsNaN.
- Control flow and collections: `If`, `Loop`, `Scan`, the sequence operators, `SequenceMap` and the optional operators run as ONNX defines them. Sessions carry sequences and optionals as packed `uint8` tensors (`PackedValue`, `TensorInfo.Kind`); bodies run as nested sessions (`docs/graph-runtime.md` section 8).

### Changed

- int8 packing narrows through value-preserving types, so results agree on WebAssembly, where vector narrowing saturates in compiled code and truncates in the interpreter.
- The int8 model-level metric adds a noise floor on the first quantized tensor of token-input encoders (`docs/int8.md`).
- RoiAlign without `coordinate_transformation_mode` defaults to `half_pixel` from opset 16; Squeeze and Unsqueeze accept a scalar `axes`.

## [0.2.0-preview.1] - 2026-10-30

Graph optimizations, int8 and half-precision storage. All reference models still match ONNX Runtime within 1e-4 scaled error in float32 with outputs independent of thread count; exact-mode results are unchanged by the new passes.

### Added

- Graph optimizations, chosen by `SessionOptions.OptimizationLevel` (`None`, `Basic`, `All`; `All` is the default): constant folding, matrix-product weights packed once per session with bias and activation applied per tile, convolution with a following activation, Conv+BatchNormalization folding, Gelu and attention fusion. Passes that change rounding are skipped in exact mode (`docs/graph-passes.md`).
- int8 inference: `QuantizeLinear`, `DequantizeLinear`, `DynamicQuantizeLinear`, `MatMulInteger`, `ConvInteger`, `QLinearMatMul`, `QLinearConv`; exact int32 product kernels for AVX2, AVX-512 and Arm dot product with a scalar fallback; the `qdq` and `matmul-integer-fusion` passes. See `docs/int8.md`.
- Half-precision storage: models whose weights or values are float16 or bfloat16 run (each operator widens its inputs to float32 and rounds its results back; `docs/fp16.md`). `SessionOptions.WeightStorage` keeps large MatMul, Gemm and Conv weights of float32 models as float16 or bfloat16 to halve their memory.
- `tools/export_models.py --fp16` makes float16 variants of MiniLM-L6 and ResNet-50; `Bar` takes `--weights original|fp16|bf16`.
- A direct depthwise convolution (phase planes for strides, 16/8/4/1-lane paths, four rows at a time for narrow maps) replaces the first one: 3-8x faster on MobileNetV3's layers (`docs/kernels-depthwise.md`).

### Changed

- Matrix products, convolutions and the normalisation operators use no more threads than their work can use (about 80 microseconds of work per thread), so small shapes no longer run slower on many threads; the parallel-for allocates nothing per call.

## [0.1.0-preview.1] - 2026-10-29

First preview: ONNX models load and run on the CPU through `Model.Load(...).Run(...)` or a `Session`, with no native dependencies. The reference set (ResNet-50, MobileNetV3, BERT-base, MiniLM-L6, Whisper-small encoder and decoder, and two text-recognition models) matches ONNX Runtime within 1e-4 scaled error, with outputs independent of thread count. No graph optimizations, int8 or fp16 compute yet; speed is not tuned.

### Added

- Shape and indexing operators: Shape, Size, Squeeze, Unsqueeze, Flatten, Dropout (inference), Transpose, Concat, Split, Slice, Expand, Tile, Gather, GatherElements, GatherND, ScatterND, Pad, ConstantOfShape, Range, OneHot, Trilu, NonZero and TopK. All move or compare elements only, so they are exact in both modes.
- Recurrent operators RNN, GRU and LSTM (all directions, both layouts, sequence lengths, initial state, peepholes, clipping, activation lists; fast and exact) and `Ctc`, greedy and prefix-beam CTC decoding for consumers of recognisers and speech models.
- Element-wise operators: Sub, Div, Mod, Pow, Max, Min, Sum, Mean, Exp, Log, Tanh, Sigmoid, Erf, Gelu, HardSigmoid, HardSwish, LeakyRelu, Neg, Abs, Sign, Reciprocal, Sqrt, Floor, Ceil, Round, Clip, the comparisons, And, Or, Xor, Not, Where, Cast and CastLike; all but Pow have exact kernels.
- The graph runtime: operator contract and static operator table, shape inference and constant evaluation per set of input shapes, execution plans with liveness-based buffer reuse, thread-safe sessions with fast and exact modes, and the `Model` happy path. Operators so far: Add, Mul, Relu, MatMul, Gemm, Reshape, Identity, Constant.
- `Bar check` and `tools/check_reference.sh` compare the library with stored ONNX Runtime outputs and hash outputs across thread counts; `tools/op_inventory.py` lists the operators a set of models uses. The happy path fills in `attention_mask` and `token_type_ids` for text encoders, so `m.Run(tokens)` works on them.
- Convolution and pooling operators: Conv (1-D and 2-D, groups, depthwise, dilation, `auto_pad`, bias), ConvTranspose, MaxPool (with Indices), AveragePool, GlobalAveragePool, GlobalMaxPool, BatchNormalization (inference), Resize and Upsample. `Conv` kernel jobs no longer allocate per call.
- The bar runner can time this library (`--side ours --mode fast|exact`).
- Operators: LayerNormalization, RMSNormalization, InstanceNormalization, GroupNormalization, Softmax, LogSoftmax, Hardmax, the Reduce* family, ArgMax, ArgMin and CumSum, all with exact kernels (`docs/ops-norms.md`).
- The solution skeleton: library, tests, NativeAOT and browser-wasm smoke apps, micro-benchmarks and a fuzz harness.
- Tensor layer: data types with fp16/bf16 conversion, strided views, NCHW/NHWC/blocked layouts, a pooled allocator and arena, and gradient records (public, unstable until the first preview).
- Fast and exact fp32 matrix-product, convolution, attention and softmax kernels, exact transcendental functions, and a fixed-chunk thread pool (internal for now).
- Tooling: reference-model export and pinning, reference outputs from ONNX Runtime, cross-machine output hashing, and a benchmark runner against ONNX Runtime.
- ONNX import: a protobuf reader and ONNX model reader of our own (IR versions 3 and later, any opset, external data confined to the model's directory, memory-mapped weights), and the graph representation it builds. Bad input produces `OnnxImportException`.
