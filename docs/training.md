# Training: modules, losses, optimizers, schedules, checkpoints

Normative for package `RecursiveAI.OneModel.Training`. It is written on the tensor layer and its reverse-mode autodiff
([autograd.md](autograd.md)) and uses nothing below it. References: Kingma and Ba 2015 (Adam); Loshchilov and Hutter 2019 (AdamW);
Reddi, Kale and Kumar 2018 (AMSGrad); Sutskever et al. 2013 (Nesterov momentum); Loshchilov and Hutter 2017 (cosine annealing);
Smith and Topin 2018 (one cycle); Vaswani et al. 2017 (transformer encoder); Steele, Lea and Flood 2014 (SplitMix64). PyTorch's
public documentation is the reference for the conventions it shares (reduction, label smoothing, parameter shapes); it is also the
test oracle (section 8).

## 1. Modules

`NnModule` owns *parameters* (contiguous float32 leaves that require gradients), *buffers* (saved, not trained: batch-norm running
statistics, the dropout draw counter) and named *children*. Names contain no dot. The **state order** is fixed: a module's parameters
in registration order, then its buffers in registration order, then each child, recursively, with dotted paths (`layer0.attn.q.weight`).
`NamedParameters()` is the parameter subsequence of it, `StateDict()` the whole, and optimizers and checkpoints use these orders.
`Train(bool)` / `Eval()` set `IsTraining` on the module and every child. `LoadStateDict` copies by name into the existing tensors
(identity, gradients and optimizer references stay valid) and rejects any missing name, extra name, type or shape difference before
copying anything.

| Module | Parameters, buffers (names, shapes) | Computation |
|---|---|---|
| `Linear(in, out, bias)` | `weight` [out,in], `bias` [out] | `Ops.Linear` |
| `Conv2d(in, out, k, stride, padding, dilation, groups, bias)` | `weight` [out,in/groups,k,k], `bias` [out] | `Ops.Conv2d`, square kernel, same stride/padding/dilation on both axes |
| `BatchNorm2d(c, momentum 0.1, eps 1e-5)` | `weight` [c] = 1, `bias` [c] = 0; buffers `running_mean` = 0, `running_var` = 1 | `Ops.BatchNorm2d`; batch statistics and running update in training mode, running statistics in evaluation mode |
| `LayerNorm(n, eps 1e-5)` | `weight` = 1, `bias` = 0 | `Ops.LayerNorm` over the last axis |
| `Embedding(count, width, paddingIndex)` | `weight` [count,width] | `Ops.Embedding` |
| `Dropout(p, seed)` | buffer `draws` Int64 [1] | section 1.2 |
| `Sequential(layers...)` | children `0`, `1`, ... | applies children in order |
| `MultiHeadAttention(width, heads)` | children `q`,`k`,`v`,`o` (`Linear`, with bias) | `Ops.Attention` over heads of `width/heads`, scale `1/√(width/heads)`, optional additive mask (no gradient) |
| `TransformerEncoderLayer(width, heads, ff, dropout)` | children `attn`, `norm1`, `ff1`, `ff2`, `norm2`, and `drop1`, `drop2` when dropout > 0 | post-norm: `x1 = LN1(x + Drop1(Attn(x)))`, `y = LN2(x1 + Drop2(ff2(GELU(ff1(x1)))))`, GELU in its erf form |
| `Recurrent(gru, in, hidden, layers, bidirectional, batchFirst)` | PyTorch names `weight_ih_l0`, `weight_hh_l0`, `bias_ih_l0`, `bias_hh_l0`, `..._l0_reverse`, ... ([4H or 3H, ·]) | `Ops.Lstm` / `Ops.Gru`, zero initial state, hidden sequence out; weights from `Uniform(±1/√hidden)` in that order |
| `ConvTranspose2d(in, out, k, stride, padding, outputPadding, dilation, groups, bias)` | `weight` [in,out/groups,k,k], `bias` [out] | `Ops.ConvTranspose2d`; bound 1/√(out/groups·k·k) |
| `Lambda` | none | ReLU, GELU, sigmoid, tanh, max/avg pooling (square window, padding counted in the average), global average pooling, flatten, mean over an axis |

### 1.1 Initialisation

Weights come from `Generator`, SplitMix64: `state += 0x9E3779B97F4A7C15`, then `z = (z ^ z>>30) * 0xBF58476D1CE4E5B9`,
`z = (z ^ z>>27) * 0x94D049BB133111EB`, output `z ^ z>>31`. `NextSingle` is `(output >> 40) / 2^24`; `Uniform(a, b) = a + (b−a)·NextSingle()`.
`Linear` and `Conv2d` draw weight then bias from `Uniform(−1/√fanIn, +1/√fanIn)` in row-major order (fanIn = in, resp.
in/groups·k·k; the square root is IEEE). `Embedding` draws from `Uniform(−√3, √3)` (unit variance). Children draw in construction order
(`MultiHeadAttention`: q, k, v, o; `TransformerEncoderLayer`: attention, ff1, ff2). The streams are not PyTorch's, so a run that must
match PyTorch loads its initial weights (section 8).

### 1.2 Dropout

Training mode with p > 0: let `d` be the value of `draws` (then `draws += 1`), `stream = Mix(seed + d·0x9E3779B97F4A7C15)` (all
unsigned 64-bit, wrapping) and for element i (row-major, from 0) `bits = Mix(stream + (i+1)·0x9E3779B97F4A7C15)`,
`u = (bits >> 40) / 2^24`. The element is kept, multiplied by `1/(1−p)` (float32), when `u >= p`, else zeroed. The mask depends on
nothing else, so it is the same on every platform and thread count, and because `draws` is a saved buffer a resumed run continues the
stream. Evaluation mode, or p = 0, returns the input itself. The mask multiplies through `Ops.Mul`, so backward is the mask.

## 2. Losses (`Losses`)

All return a rank-0 float32 tensor and are compositions of differentiable ops. `mean` (default) divides by the number of counted
elements; otherwise the sum is returned.

- `CrossEntropy(logits[N,C], targets[N], labelSmoothing ε, ignoreIndex = −100)`: `NLL(LogSoftmax(logits))`; with ε > 0:
  `(1−ε)·NLL + ε·S`, `S = Σ_rows (−mean_c logp[n,c]) / counted` over rows whose target is not the ignore index. No counted row: NaN mean.
- `NllLoss`: `Ops.NllLoss`.
- `Mse`: `mean((p−t)²)`. `L1`: `mean(|p−t|)`, with `|d|` built as `relu(d) + relu(−d)` (gradient 0 at d = 0, as in PyTorch).
- `BceWithLogits(x, t)`: per element `softplus(x) − x·t`, `softplus(x) = −LogSoftmax([x, 0])[1]`, stable for any |x| (no overflow,
  no cancellation). No `pos_weight`, no class weights.

- `Ctc(logProbs[T,B,C], targets, inputLengths, targetLengths, blank, reduction, zeroInfinity)`: `Ops.CtcLoss`.

## 3. Optimizers

One parameter group per optimizer: one learning rate, one set of hyper-parameters. Parameters are the optimizer's constructor list;
those without a gradient at a step are skipped (their state does not change). A step applies, parameter by parameter and element
by element in index order, plain float32 `+ − × ÷ sqrt` with no fused multiply-add (the JIT does not contract float expressions, and
the code must not use `Fma`), so **the update is bit-identical in fast and exact mode, on every platform and thread count**. State is
float32, shaped like the parameter, zero at creation, and listed by `StateTensors` (`momentum_buffer.i`, `exp_avg.i`,
`exp_avg_sq.i`, `max_exp_avg_sq.i`, and an Int32 `initialized` flag per parameter for SGD). `StepCount` counts completed steps.

**SGD** (lr, μ momentum, d dampening, wd, nesterov), per element with gradient g: `g += wd·p`; if μ ≠ 0:
`b = first ? g : μ·b + (1−d)·g` (first = the first step in which this parameter has a gradient), `g = nesterov ? g + μ·b : b`; then
`p −= lr·g`. Nesterov needs μ > 0 and d = 0.

**Adam / AdamW** (lr, β₁, β₂, ε, wd, amsgrad), t = `StepCount + 1`:
`bc1 = 1 − β₁ᵗ`, `bc2 = 1 − β₂ᵗ` (powers by binary exponentiation in double, round to nearest at each product), `stepSize = float(lr / bc1)`,
`sqrtBc2 = float(√bc2)` (double sqrt). Per element: Adam with wd adds `wd·p` to g; AdamW first sets `p *= 1 − lr·wd`; then
`m = β₁·m + (1−β₁)·g`, `v = β₂·v + (1−β₂)·g·g` (evaluated left to right: `((1−β₂)·g)·g`),
`vd = amsgrad ? max(vmax, v) : v` (and `vmax = vd`), `denom = √vd / sqrtBc2 + ε`, `p −= stepSize·m / denom`. AdamW's default wd is 0.01.

Differences from PyTorch's implementation, all within float rounding of one step (measured, section 8): the first moment uses
`β₁m + (1−β₁)g` where PyTorch uses `lerp`; the step number is global, not per parameter; the learning-rate argument is read at each step
(a schedule sets it). Not provided: parameter groups, maximize, foreach/fused variants, `Adamax`, `RMSprop` and the rest.

## 4. Gradient clipping (`GradClip`)

`ClipNorm(params, maxNorm, infinityNorm)`: `norm` = `√Σg²` over every element of every gradient in order (squares added in double),
or the largest `|g|`; `coef = maxNorm / (norm + 1e-6)` in double, rounded to float; if `coef < 1` every gradient element is multiplied by
it. Returns `float(norm)`. A non-finite norm is returned without scaling anything. `ClipValue` clamps each element to ±c (NaN stays).

## 5. Learning-rate schedules

A schedule is a pure function `LrAt(stepIndex)` of the number of optimizer steps already taken (0 for the first step); `Apply(optimizer)`
sets the rate for `optimizer.StepCount`. There is no schedule state, so a checkpoint needs only the optimizer's step count. Arithmetic
is IEEE double (`+ − × ÷`, no `Math` function), rounded to float at the end:

- `StepLr(base, size, γ)`: `base·γ^⌊s/size⌋` (integer power by squaring).
- `CosineLr(base, T, min)`: `min + (base−min)(1 + cos(π·s/T))/2`, `s/T` clamped to 1 (the rate stays at `min` after T).
  PyTorch's recursive form turns back up after T_max; this one does not.
- `WarmupLr(after, W, f)`: for `s < W`, `r₀·(f + (1−f)·s/W)` with `r₀ = after.LrAt(0)`; for `s ≥ W`, `after.LrAt(s − W)`.
- `OneCycleLr(max, T, pct 0.3, div 25, finalDiv 1e4)`: with `initial = max/div`, `min = initial/finalDiv`, `up = pct·T − 1`, `end = T − 1`:
  half-cosine from `initial` to `max` over steps 0..up, then from `max` to `min` over up..end, `anneal(a,b,x) = b + (a−b)/2·(cos(πx)+1)`.
  Momentum is not cycled.

**`LrSchedule.Cos(t)` = cos(π t), t ∈ [0,1]**: `t ≤ 0` gives 1, `t ≥ 1` gives −1; otherwise `y = t` or `1−t` (sign −1 for `t > 0.5`),
`a = π·y` with π the double 3.14159265358979323846, `a² = a·a`, and Horner's scheme `r = 1; for k = 13 down to 1: r = 1 − a²/((2k−1)(2k))·r`;
the result is `sign·r`. This is the Maclaurin series to degree 26 (truncation below 1e-21). Measured error against `Math.Cos` over 10 001
points: 3.9e-16. It is used instead of `Math.Cos` because the C library's cosine may differ between platforms in the last bit, which
would break exact-mode checkpoints and hashes.

## 6. Checkpoints (`Checkpoint`)

File layout, little endian, the same frame as the model file ([format.md](format.md)) with a different magic:

```
offset 0                64-byte header
offset 64               manifest, UTF-8 JSON, manifest_length bytes
then zero padding to the next multiple of 64 (= data_offset)
data_offset             data region, data_length bytes; end of file = data_offset + data_length
```

Header: 8 bytes ASCII `ONEMCKPT`; u32 version = 1; u32 flags = 0; u64 manifest_offset = 64; u64 manifest_length; u64 data_offset (multiple of 64,
≥ 64 + manifest_length, ≤ file length); u64 data_length = file length − data_offset; u64 file_length (exact); u64 reserved = 0. A reader
rejects anything else, and checks every field against the real length before reading the manifest (limit 256 MiB, JSON depth 32).

Manifest: `{"format":"onemodel-checkpoint","version":1,"model":[T...],"optimizer":{"kind","step","hyper":[{"name","bits"}...],"state":[T...]},"extra":[{"key","value"}...]}`
(`optimizer` only when saved). Each tensor entry `T` is `{"name","dtype":"f32"|"i32"|"i64","shape":[...],"offset","length"}`: `offset` from the
start of the data region (a multiple of 64), `length` in bytes. Tensors appear in the state order (section 1), model first, then optimizer state in
`StateTensors` order; `hyper` lists `HyperParameters()` in order, each float as the unsigned integer of its IEEE-754 bit pattern; `extra`
holds integers sorted by key (ordinal). Tensor data is raw little-endian elements, row-major, floats by bit pattern (NaN payloads and
signed zeros survive). The writer emits compact JSON in exactly this field order, so the same state writes the same bytes.

Loading checks, before changing anything: header; manifest shape; tensor names, types and shapes equal the target's, in order; every
offset and length inside the data region and equal to the shape's size (so a hostile shape never allocates); the optimizer kind and its
fixed settings (momentum, betas, ε, amsgrad). Then it copies the tensors, restores hyper-parameters (the learning rate included) and the
step count. Errors are `CheckpointException(Kind)`: `BadHeader`, `BadManifest`, `BadTensor`, `Mismatch`. A checkpoint with optimizer state
loads into a model alone; asking for optimizer state from a model-only file is a `Mismatch`.

**Resume rule.** Model, buffers (including dropout counters), optimizer state, hyper-parameters and the step count are the whole training
state; schedules and dropout are functions of them. A run saved at step k and continued in a fresh process reproduces the uninterrupted run
bit for bit in the same arithmetic mode and thread count (and, in exact mode, on any machine). The caller supplies the data order (store
the sample counter in `extra`).

## 7. Determinism

Fast mode: forward and backward are the engine's ([autograd.md](autograd.md), thread-count independent); the optimizer, clipping,
schedules, losses' constants, dropout and the checkpoint code are scalar, fixed-order code. Exact mode: all of it is bit-identical across ISA paths,
OS and thread counts; checked by pinned hashes of the full training state after 20 steps of the conv net (AdamW + cosine + clipping) and of
the transformer classifier with dropout (`TrainingDeterminismTests`), under default, `DOTNET_EnableAVX512=0`, `DOTNET_EnableAVX2=0` and
`DOTNET_EnableHWIntrinsic=0`, at 1 and 8 threads, and by CI on every platform. Fast-mode state hashes are equal at 1, 3 and 8 threads.

## 8. What was measured against PyTorch 2.15.0 (CPU, black box)

`tools/training_oracle.py` runs seeded trainings in PyTorch and writes the initial weights, data, batch order, per-step losses and final
parameters to `testdata/training/`; the tests load the same weights and batches into our modules. Tolerance accepted: per-step loss relative
difference ≤ 1e-3 and final parameter scaled error (`max|Δ| / max|torch|`) ≤ 1e-3 — the curves overlay far inside it.

| Run | Steps | Max loss diff (abs / relative) | Final parameter scaled error | Notes |
|---|---|---|---|---|
| MLP 20-32-4, SGD momentum 0.9, wd 1e-4 | 100 | 1.9e-7 / 3.1e-6 | 9.3e-7 | |
| Conv net (conv-BN-ReLU-maxpool ×2, linear), AdamW 3e-3 wd 0.01, cosine, clip-norm 0.5 | 60 | 2.4e-7 / 2.7e-7 | 2.0e-7 | clipping active on all 60 steps; global gradient norm relative diff 2.3e-7; BN running variance 1.1e-7 |
| 2-layer transformer classifier (width 16, 2 heads), Adam 2e-3, 10-step warm-up | 80 | 1.9e-7 / 3.7e-6 | 3.2e-7 | |

Parameters whose true gradient is zero (the bias of a convolution in front of batch norm, the key bias of attention, which softmax ignores)
receive only rounding noise as gradient, and Adam turns noise into steps of the full learning rate, so these differ between any two
implementations (2e-2 and 4e-3 scaled here); they are excluded from the parameter comparison, and the batch-norm running *mean* that
absorbs the first one is not gated (the running variance is). Single-step checks (5 steps, given gradients; scaled error against PyTorch): SGD
variants ≤ 8.1e-8, Adam/AdamW variants (L2, AMSGrad, other betas) ≤ 3.4e-8; clipping ≤ 1.7e-7 with equal norms; learning-rate schedules
(100 steps) ≤ 1.2e-7 relative.

## 9. Reduced precision: measured, not built

Question: does bfloat16, float16 or loss-scaled training pay on a CPU, on the kernels we have? Method (Micikevicius et al. 2018 describe the
schemes): `bench/Micro` command `mixed [threads] [steps]` runs the float32 AdamW step of three models, reports the median step time
(steps 3 on), the peak pool bytes sampled during the step, the activation bytes held at the end of the forward pass, and the time of
one float32 → bfloat16 → float32 pass over all activations and parameters (a lower bound on what any reduced-precision scheme adds
per step). Indicative numbers, loaded hosts, not a benchmark.

| Model | Step ms: Zen 4 1t / 8t | Step ms: M2 1t / 8t | Peak MB | Activations MB | Conversion floor, % of step (Zen 4 / M2) |
|---|---|---|---|---|---|
| Transformer classifier, batch 32 × 128, width 256, 4 layers | 1173 / 692 | 1603 / 824 | 595 | 500 | 5–9 / 2–5 |
| Conv net, batch 16, 3×64×64 | 93 / 105 | 114 / 93 | 117 | 49 | 5 / 4 |
| CRNN-like (conv, 2-layer BiLSTM, linear, CTC), batch 16, 1×32×128 | 87 / 97 | 96 / 83 | 114 | 47 | 5–6 / 4 |

Findings.

- **Time does not pay.** The .NET 10 intrinsics expose no bfloat16 dot product on x86 (`Avx512Bf16` is absent; the listed classes end at
  `Avx10v2`, `AvxVnni*`) or on Arm (no `Bf16`), although the worker's CPU has AVX512_BF16 and the M2 has FEAT_BF16. Without it,
  low-precision operands are widened to float32 before the multiply, so a reduced-precision matrix product does the same arithmetic plus
  the conversion; the products are compute-bound at these sizes, so halving operand bytes buys nothing. The best case is therefore a
  step time ≥ 1.0× float32's, never the 1.2× gain required.
- **Memory pays only for the transformer.** Saved activations are 84% of its peak, so holding every activation as bfloat16 would cut
  peak memory by up to about 42% (above the 30% bar); in the conv net and CRNN they are 41–42% of the peak and gradients of the same
  size follow in backward, so the saving is about 20% (below the bar). Parameters, gradients and optimizer state are 0.8–26 MB:
  irrelevant.
- **Not built.** The saving needs bfloat16 *tensors* flowing between operators (each operator widens its inputs into scratch, narrows its
  output, and its record saves the narrow tensor and widens it again in backward). Holding a bfloat16 copy beside the float32 original
  saves nothing, because activations are released together at the end of the step and a caller may still hold the float32 tensor. That
  change touches every operator's entry point and saved-slot handling, and is a separate piece of work worth doing only if
  long-sequence transformer training on a CPU becomes a real use. Loss scaling is a float16 remedy and has nothing to scale while
  gradients stay float32, so it waits for the same work.
