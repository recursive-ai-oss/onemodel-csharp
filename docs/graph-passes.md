# Graph passes

Rewrites applied to the compiled graph when a session is created, selected by `SessionOptions.OptimizationLevel` (`None` = no passes, `Basic`, `All` = default) and the mode. Interface and ownership rules: [graph-runtime.md](graph-runtime.md) section 6. Every pass states whether it preserves bits; passes that do not are skipped in exact mode, and the documented exact pipeline is therefore the bit-preserving subset.

| Pass | Level | Preserves bits | What it does |
|---|---|---|---|
| `constant-fold` | Basic | yes | Evaluates every node whose inputs are all constants, once, through its own kernel in the session's mode. Skips data-dependent shapes, nodes feeding a graph output, and results larger than both the inputs and 64 KiB. A view of a constant becomes a copy. |
| `qdq` | All | no | Rewrites quantize-dequantize Conv/Gemm patterns to `QLinearConv` / `QLinearMatMul`; skipped in exact mode. See [int8.md](int8.md). |
| `matmul-integer-fusion` | All | yes | `MatMulInteger -> Cast -> Mul [-> Add]` becomes one dynamic-quantization product node. See [int8.md](int8.md). |
