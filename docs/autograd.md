# Reverse-mode automatic differentiation

Normative for the engine, the per-operator backward rules and what is exact. Source: `src/OneModel/Tensors/Autograd/` and
`Ops*.cs`. The record layout (what each operator saves) is also summarised in `tensors.md` section 8. References: Goodfellow,
Bengio and Courville, *Deep Learning*, ch. 6 (back-propagation); Ioffe and Szegedy 2015 (batch normalization); Ba, Kiros and
Hinton 2016 (layer normalization); Chellapilla, Puri and Simard 2006 (im2col); Dao et al. 2022 (attention backward by
recomputation).

## 1. Model

An eager operator (`Ops.*`, or a view method of `Tensor`) computes its result immediately. If recording is on (`GradMode`) and an
input requires gradients, the result carries a `BackwardRecord`: the op, one graph edge per input that needs a gradient, the input
shapes and types, saved tensors and small integer/float attributes. A leaf is a tensor made with `WithRequiresGrad()`; its
gradient accumulates in `Tensor.Grad`. All of this is float32. Gradient tensors are contiguous and have exactly the shape of the
tensor they belong to.

## 2. The engine (`Tensor.Backward`, `Backprop`)

`y.Backward(gradient = null, retainGraph = false)`:

1. `y` must require gradients. Without `gradient`, `y` must have one element (the seed is 1); otherwise `gradient` must have
   `y`'s shape (it is copied).
2. The records reachable from `y` are ordered by an iterative depth-first search over the input edges, post-order; they are
   processed in reverse of that order, so a record runs after every record that consumes its output. The order depends only on the
   graph (inputs are visited in input order), never on threads.
3. Each record holds at most one pending gradient (all operators here have one output). A record runs its backward function with
   the pending gradient `g` and *emits* a gradient for each input that needs one. Emitting to a record adds into its pending
   gradient element by element, in the order the emissions happen; emitting to a leaf adds into `leaf.Grad` (created on the first
   emission). Sums over several consumers are therefore in the fixed order of the reverse topological traversal.
4. Backward runs with recording off. Higher-order derivatives are not supported.
5. Unless `retainGraph`, each record is released after it ran: tensors it owns are freed and its saved slots cleared. A second
   `Backward` through a released record throws `InvalidOperationException` (detected before any work).
6. A saved tensor modified in place after the forward pass (its storage version changed) throws `TensorVersionException` when its
   record runs.

Leaf gradients accumulate across calls; `Backprop.ZeroGrad(leaf)` zeroes one in place and `Backprop.ClearGrad(leaf)` frees it.

## 3. Memory: `ActivationScope`

While an `ActivationScope` is open on a thread, every tensor an `Ops` operator allocates (and every `Contiguous`/copying `Reshape`
copy) is remembered and returned to its pool when the scope is disposed (disposal skips tensors the caller already released). The
engine runs outside the scope and frees its own temporaries. Leaf gradients are never tied to a scope. A step
`using var scope = new ActivationScope(); loss = model(...); loss.Backward();` allocates no tensor memory after warm-up: the buffer
pool serves every request (`PoolStats.NativeAllocations` and `OutstandingBytes` are unchanged). It still allocates small managed
objects: one `BackwardRecord` with its arrays and `AutogradMeta` per recorded operator (about 0.3 to 1 KB each; measured 17 KB per
step for the 12-operator conv net and 58 KB for the two-layer encoder in `AutogradModelTests`).

## 4. Arithmetic: `TensorCompute`

`TensorCompute.Mode` (`ExecutionMode.Fast` default, or `Exact`) and `TensorCompute.Threads` apply to every differentiable
operator, forward and backward. The kernel path is the machine's best (`KernelPaths.Best()`).

- **Matrix products and convolutions** run on the blocked GEMM (`docs/kernels-gemm.md`) and the convolution kernels
  (`docs/ops-conv.md`). Gradient products with a transposed operand (`xᵀ·g`, `g·Wᵀ`) first copy that operand into a pooled
  scratch in row-major order, then call the same GEMM, so they follow the same canonical order: one accumulator per output,
  increasing k, from +0. Fast mode uses FMA as the forward kernels do.
- **Everything else** is scalar code in a fixed order using only IEEE `+ − × ÷ sqrt` plus the functions below.
  Sums over many elements (bias and norm-parameter gradients, reductions, broadcast-undoing sums, batch statistics, the row sums of
  softmax-type backward rules) accumulate in `double` in increasing element order and round once to float32; these are exact by
  construction. Parameter-gradient contributions of successive batch items are added in item order.
- **Transcendentals.** In fast mode `exp`, `log`, `tanh` and sigmoid use the BCL (`MathF`); in exact mode, and always for
  `erf` and the GELU forward, they use the fixed polynomials of `exact-transcendentals.md` (scalar form, which equals every vector
  form bit for bit). Softmax uses the softmax row kernel of the same mode.
- **Threads** change only speed: the engine is single-threaded and threads exist only inside the GEMM, convolution, softmax and
  attention kernels, whose chunking is a constant of the kernel and shape. Gradient hashes are equal at 1, 3 and 8 threads in both
  modes (`AutogradModelTests`).

**Exactness.** With `Mode = Exact` every operator in section 5 is bit-identical across OS, CPU, ISA path (AVX-512, AVX2, 128-bit,
scalar, NEON) and thread count: the matrix products and convolutions use the exact kernels, the rest follows the rules above. The
hashes of all gradients of the two reference models are pinned in `AutogradModelTests.ExactModeHashesArePinned`. No operator here
is flagged not-exact. `Attention` backward is exact given the exact forward kernel's flavour (it recomputes the weights with the
exact softmax row kernel). The fast and exact flavours differ in the last bits.

## 5. Operators: forward, record, backward

`g` is the incoming gradient; saved slots are `Saved(i)`; `Ints`/`Floats` are the record's attributes. "Sum-to" means sum over the
axes the forward broadcast, to the input's shape.

| Operator | Saved | Attributes | Backward |
|---|---|---|---|
| `Add`, `Sub` | none | | `g` sum-to each input (Sub negates b's) |
| `Mul` | 0 = a if b needs grad; 1 = b if a does | | `g·b`, `g·a`, sum-to |
| `Div` | 0 = a (for b); 1 = b | | `g/b`; `−(g·a)/(b·b)`; sum-to |
| `Neg` | none | | `−g` |
| `Relu` | 0 = output | | `g` where output > 0, else 0 |
| `Gelu` (erf or tanh form) | 0 = input | Ints [0 erf, 1 tanh] | erf: `g·(Φ(x) + x·φ(x))`, Φ = ½(1+erf(x/√2)), φ = e^(−x²/2)/√(2π). tanh: with u = k(x+0.044715x³), t = tanh u, k = √(2/π): `g·(½(1+t) + ½x(1−t²)k(1+3·0.044715x²))` |
| `Sigmoid` | 0 = output y | | `g·y·(1−y)` |
| `Tanh` | 0 = output y | | `g·(1−y²)` |
| `MatMul` | as Mul | | `g·bᵀ`, `aᵀ·g` (transposed views, section 4) |
| `Linear` | (composition of Reshape, MatMul with `Wᵀ`, Add, Reshape) | | |
| `Sum`, `Mean` (all or one axis) | none | [axis or −1, keepDim] | `g` broadcast to the input shape (Mean: each element `/n`) |
| `Softmax` (last axis; other axes via transposed views) | 0 = output y | | `y·(g − Σⱼ gⱼyⱼ)` per row |
| `LogSoftmax` | 0 = output y | | `g − eʸ·Σⱼ gⱼ` per row |
| `LayerNorm` (last axis) | 0 = x, 1 = γ, 2 = mean, 3 = 1/σ (owned) | Floats [ε] | x̂ = (x−μ)/σ, ĝ = g·γ: `dx = (ĝ − mean(ĝ) − x̂·mean(ĝ·x̂))/σ`; `dγ = Σ g·x̂`, `dβ = Σ g` over rows (Ba et al. 2016) |
| `BatchNorm2d` | 0 = x, 1 = γ, 2 = mean, 3 = 1/σ (owned) | Ints [training] | training: `dx = γ/σ·(g − mean(g) − x̂·mean(g·x̂))` over N,H,W per channel; eval: `dx = γ/σ·g`; `dγ = Σ g·x̂`, `dβ = Σ g` (Ioffe and Szegedy 2015) |
| `Conv2d` (groups, stride, padding, dilation) | 0 = x, 1 = W | [sH, sW, padT, padL, padB, padR, dH, dW, groups] | per image and group, with cols the patch-major unfolding [P, K] of x: `dW += g[og,P]·cols[P,K]`; `dcols[K,P] = W_gᵀ[K,og]·g[og,P]` then col2im adds each tap into `dx`; `dbias = Σ g` |
| `MaxPool2d` | 0 = Int32 flat input index of each winner | window | scatter-add `g` at the index |
| `AvgPool2d` | none | [kH, kW, sH, sW, padH, padW, countIncludePad] | each output's `g/count` added to every in-bounds element of its window |
| `GlobalAvgPool2d` | (composition: Reshape, Mean, Reshape) | | |
| `Concat` | none | [axis] | slices of `g` |
| `Embedding` | 0 = ids | [paddingIndex] | add each `g` row to its table row in id order, skipping the padding id |
| `NllLoss` | 0 = targets | [mean, ignoreIndex] | `−g/count` (or `−g`) at `[n, target_n]` of non-ignored rows |
| `CrossEntropy` | (composition: LogSoftmax, NllLoss) | | |
| `Attention` | 0 = q, 1 = k, 2 = v, 3 = additive mask (or empty) | Floats [scale] | per batch and head, P = softmax(scale·QKᵀ + mask) is recomputed; `dV = Pᵀ·g`; `dP = g·Vᵀ`; `dS = P∘(dP − rowsum(dP∘P))`; `dQ = scale·dS·K`; `dK = scale·dSᵀ·Q`. The mask gets no gradient |
| Views: `Slice`, `Select`, `Permute`/`Transpose`, `Reshape`/`Squeeze`/`Unsqueeze`/copying `Contiguous`, `Expand`, `Flip`, `Cast` | none | as `tensors.md` | scatter into zeros, inverse permutation, reshape back, sum-to, flip, cast back |

`Gelu`'s saved slot differs from the convention first written in `tensors.md` (input plus the tanh term): the slope is recomputed
in backward, which needs no extra storage.

Operators not in the table (LSTM/GRU, CTC loss, ConvTranspose, Resize, int8) have no backward yet.

## 6. Checks

- `GradientCheck.Run` (package `RecursiveAI.OneModel.Training`) compares analytic gradients with central float32 differences of
  `L = Σ W·f(x)` (W fixed pseudo-random, L summed in double). `GradCheckTests` runs it for every operator above, with
  broadcasting, strided and transposed views, groups, dilation, depthwise convolution and masks. Running the tests with `ONEMODEL_GRADCHECK_LOG=<file>` writes the table of worst errors per operator.
- `tools/autograd_oracle.py` builds a conv net and a two-layer transformer encoder in PyTorch (CPU, black box) from seeded numpy
  weights and writes weights, loss and every gradient to `testdata/autograd/`; `AutogradModelTests` runs the same models on the
  eager ops and requires each gradient within 1e-4 scaled error (max |ours − torch| / max |torch|). Gradients that are zero in theory
  (the bias of a convolution feeding a batch norm; the key bias of attention) must be below 1e-5 of the largest gradient.
