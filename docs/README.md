# Documentation

Normative, language-neutral notes live here, one file per topic, so a port can follow them without the C# code.

| Topic | File |
|---|---|
| Tensors, memory, layouts, fp16/bf16 rounding, gradient records | [tensors.md](tensors.md) |
| Reverse-mode autodiff: the engine, per-operator backward rules, what is exact | [autograd.md](autograd.md) |
| Data loading for training: deterministic batch streams, samplers, collation, prefetch | [data-loading.md](data-loading.md) |
| Exact mode: definition and accumulation order | [exact-mode.md](exact-mode.md) |
| Exact mode golden hashes: rows, results, cost of exactness | [golden-hashes.md](golden-hashes.md) |
| Matrix product and convolution kernels | [kernels-gemm.md](kernels-gemm.md) |
| Attention and softmax kernels | [kernels-attention.md](kernels-attention.md) |
| exp, log, tanh, erf, sigmoid, GELU: polynomials and error | [exact-transcendentals.md](exact-transcendentals.md) |
| Reference oracles, pinned versions, comparison metrics | [oracles.md](oracles.md) |
| Benchmark method | [bench-method.md](bench-method.md) |
| Convolution in fast mode: tiles, gathering, prepacked weights, threads | [kernels-conv.md](kernels-conv.md) |
| Depthwise convolution: phase planes, lane layout, bits | [kernels-depthwise.md](kernels-depthwise.md) |
| Half-precision storage: conversions, the widening adapter, compact weights | [fp16.md](fp16.md) |
| ONNX import: protobuf reading, tensors, graph resolution, external data, errors | [onnx-import.md](onnx-import.md) |
| Operator contract, plans, sessions, memory reuse | [graph-runtime.md](graph-runtime.md) |
| Reference models: method, operators used, results | [reference-models.md](reference-models.md) |
| Graph passes: what each rewrites, levels, bit preservation | [graph-passes.md](graph-passes.md) |
| Normalisation, softmax, reduction, arg-reduction and cumulative-sum operators | [ops-norms.md](ops-norms.md) |
| Convolution, pooling, normalisation and resize operators: supported cases, edge rules, bits | [ops-conv.md](ops-conv.md) |
| Shape and indexing operators: semantics, ONNX Runtime findings | [ops-shape-indexing.md](ops-shape-indexing.md) |
| Recurrent operators (RNN, GRU, LSTM): order of operations, open choices | [ops-recurrent.md](ops-recurrent.md) |
| CTC decoding: greedy and prefix beam search, confidences | [ctc.md](ctc.md) |
| Element-wise operators, activations, comparisons, Where, Cast | [ops-elementwise.md](ops-elementwise.md) |
| int8: operators, product kernel, requantization, QDQ rewrite | [int8.md](int8.md) |
| The OneModel model file: header, JSON manifest, aligned tensors, reading and writing rules (version 1, unstable until 1.0) | [format.md](format.md) |
| Conformance: onnx/backend/test results per operator, method, gaps | [conformance.md](conformance.md) |
| Operators added for coverage: settled semantics, exactness | [ops-coverage.md](ops-coverage.md) |
