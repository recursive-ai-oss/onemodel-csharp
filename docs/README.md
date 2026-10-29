# Documentation

Normative, language-neutral notes live here, one file per topic, so a port can follow them without the C# code.

| Topic | File |
|---|---|
| Tensors, memory, layouts, fp16/bf16 rounding, gradient records | [tensors.md](tensors.md) |
| Exact mode: definition and accumulation order | [exact-mode.md](exact-mode.md) |
| Matrix product and convolution kernels | [kernels-gemm.md](kernels-gemm.md) |
| Attention and softmax kernels | [kernels-attention.md](kernels-attention.md) |
| exp, log, tanh, erf, sigmoid, GELU: polynomials and error | [exact-transcendentals.md](exact-transcendentals.md) |
| Reference oracles, pinned versions, comparison metrics | [oracles.md](oracles.md) |
| Benchmark method | [bench-method.md](bench-method.md) |
| ONNX import: protobuf reading, tensors, graph resolution, external data, errors | [onnx-import.md](onnx-import.md) |
| Operator contract, plans, sessions, memory reuse | [graph-runtime.md](graph-runtime.md) |
