# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- int8 inference: `QuantizeLinear`, `DequantizeLinear`, `DynamicQuantizeLinear`, `MatMulInteger`, `ConvInteger`, `QLinearMatMul`, `QLinearConv`; exact int32 product kernels for AVX2, AVX-512 and Arm dot product with a scalar fallback; the `qdq` and `matmul-integer-fusion` passes. See `docs/int8.md`.

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
