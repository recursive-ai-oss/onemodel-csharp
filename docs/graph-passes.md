# Graph passes

Rewrites applied to the compiled graph when a session is created, selected by `SessionOptions.OptimizationLevel` and the mode. Interface and ownership rules: [graph-runtime.md](graph-runtime.md) section 6.

## Levels and exact mode

| Level | Passes |
|---|---|
| `None` | none; the graph runs as imported (plans still evaluate constant subgraphs) |
| `Basic` | `constant-fold` |
| `All` (default) | everything below |

Every pass states whether it **preserves bits**: results identical with and without the pass, in both modes. In `ExecutionMode.Exact` only passes that preserve bits run; the documented exact pipeline (and so the golden result) is that subset. A pass that does not preserve bits may still change fast-mode results only in the last bits (checked against the unfused graph, see the tests); fast mode stays deterministic for any thread count either way.

## Passes, in order

| Pass | Level | Preserves bits | What it does |
|---|---|---|---|
| `constant-fold` | Basic | yes | Evaluates every node whose inputs are all constants, once, through its own kernel in the session's mode. Skips data-dependent shapes, nodes feeding a graph output, and results larger than both the inputs and 64 KiB. A view of a constant becomes a copy. Shape-dependent folding (Shape, Size of a runtime input) stays in plans. |
| `conv-batchnorm-fold` | All | **no** | `Conv → BatchNormalization` (constant statistics) becomes one Conv: `W' = W·s` per output channel, `b' = (b − mean)·s + beta`, `s = scale / sqrt(var + epsilon)`, in double precision, rounded once. The scale moves before the sum, so the last bits differ. |
| `constant-fold` | Basic | yes | Again, so the folded weights and bias are plain constants. |
| `gelu` | All | **no** | `Mul(Mul(x, 1 + Erf(Div(x, √2))), 0.5)` (the opset 17 export of Gelu) becomes one Gelu node (erf form). One pass over memory instead of five; the fused erf is not the separate operators' arithmetic. |
| `attention` | All | **no** | `MatMul(Q, Kt) → [Mul or Div by a constant] → [Add mask] → Softmax(last axis) → MatMul(P, V)` becomes one `AttentionOp` running the fused kernel ([kernels-attention.md](kernels-attention.md)). Head-split transposes before Q, K, V (`[b,s,h,d]→[b,h,s,d]`; for K `(0,2,3,1)` or `(0,1,3,2)`) and the head-merge transpose after are absorbed: the kernel reads and writes the strided layouts. Any broadcastable mask is accepted; an odd head dimension takes a plain fallback. |
| `linear-pack-fuse` | All | yes | `MatMul` or `Gemm` with a constant 2-D float32 B becomes `LinearOp`: B is packed once into the kernel's panel layout (below); a following `Add` of a constant bias vector and a following activation (Relu, Clip with constant bounds, HardSwish, HardSigmoid, Sigmoid, Gelu, Tanh, LeakyRelu) run on each finished tile of C. Gemm needs alpha = 1 and no transA; `beta·C` is formed once at pass time with the multiply the kernel would do per call. |
| `conv-activation` | All | yes | `Conv → activation` (same list): the convolution's bias and the activation run in the product's tile epilogue (after the direct depthwise kernel: per plane). |

**Why the "yes" rows preserve bits.** Packing is a copy. The bias is one rounded addition per element (the stand-alone Add), and in exact mode a NaN becomes `0x7FC00000` as Add does. The activation is the operator's own kernel applied to the same values; exact-mode kernels do not depend on lane width or on how a row is cut, so the tile-row cuts change nothing. In fast mode the one possible difference is where a vector body hands over to its scalar tail inside an activation (rows are cut at tile edges), so fast-mode activations may differ from the separate operator in the last bits; fast results are still independent of the thread count because the tiles are fixed by the shape.

## Packed B

`Gemm.PackedBLength(path, n, k)`, `Gemm.PackB` and `Gemm.RunPacked` ([kernels-gemm.md](kernels-gemm.md)): B (k × n) is cut into blocks of Kc = 256 rows; within a block, strips of nr columns (nr is the path's register tile width: 32, 16, 12, 8 or 4), each strip a kc × nr panel stored contiguously, ragged strips padded with +0. Block `pc/Kc` starts at `pc · stripCount · nr`, strip `s` of a block at `s · kc · nr` within it. The layout depends only on the path. `RunPacked` skips the per-call packing and computes the same products in the same order, so its results are bit-identical to `Run` on the unpacked B.

## Tile epilogue

`IGemmEpilogue.Apply(tile, ldc, row0, col0, rows, cols, mode)` is called once per finished mc × nc tile of C (after the exact-mode NaN rewrite), from whichever thread computed the tile. It writes only the tile. Tile boundaries are fixed by the path's blocking and the shape, never by the thread count.

## Memory

Fusion removes intermediates: the fused chains above keep one output tensor instead of two to five, and the in-place reuse of the plan applies to what remains. Packed weights replace the originals (the constant is released once nothing reads it), so resident weight memory is unchanged apart from padding of ragged strips.

## Half-precision weights

* A model that contains half-precision values of its own (types, initializers, casts) is not rewritten: its graph runs as imported, because the passes' kernels are binary32.
* With `WeightStorage` set to `Binary16` or `BFloat16`, the weights that qualify are kept compact and are **not packed**: the passes pack and fold only binary32 constants, and the half-precision adapter widens a compact weight on each run. Everything around them is still rewritten (constant folding of other nodes, a Conv's batch-norm fold needs binary32 weights and is skipped for compact ones, a Conv's activation still fuses, attention and Gelu chains fuse). Packing exists to save the per-call copy of B; a compact weight is widened per call anyway. Compact and packed storage therefore never apply to the same weight. Weights below the compaction threshold (1024 elements) are binary32 and are packed.
