# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- The graph runtime: operator contract and static operator table, shape inference and constant evaluation per set of input shapes, execution plans with liveness-based buffer reuse, thread-safe sessions with fast and exact modes, and the `Model` happy path. Operators so far: Add, Mul, Relu, MatMul, Gemm, Reshape, Identity, Constant.
- `Bar check` and `tools/check_reference.sh` compare the library with stored ONNX Runtime outputs and hash outputs across thread counts; `tools/op_inventory.py` lists the operators a set of models uses. The happy path fills in `attention_mask` and `token_type_ids` for text encoders, so `m.Run(tokens)` works on them.
- The bar runner can time this library (`--side ours --mode fast|exact`).

- The solution skeleton: library, tests, NativeAOT and browser-wasm smoke apps, micro-benchmarks and a fuzz harness.
- Tensor layer: data types with fp16/bf16 conversion, strided views, NCHW/NHWC/blocked layouts, a pooled allocator and arena, and gradient records (public, unstable until the first preview).
- Fast and exact fp32 matrix-product, convolution, attention and softmax kernels, exact transcendental functions, and a fixed-chunk thread pool (internal for now).
- Tooling: reference-model export and pinning, reference outputs from ONNX Runtime, cross-machine output hashing, and a benchmark runner against ONNX Runtime.
- ONNX import: a protobuf reader and ONNX model reader of our own (IR versions 3 and later, any opset, external data confined to the model's directory, memory-mapped weights), and the graph representation it builds. Bad input produces `OnnxImportException`.
