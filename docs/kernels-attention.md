# Fused attention and softmax: fast and exact

Normative for the exact flavour. Source: `src/OneModel/Kernels/Attention.cs`, `Softmax.cs`, `Exact/Transcendental*.cs`.

## What is computed

For every (batch b, head h, query row i): `O[i] = softmax(scale * Q[i] K^T + mask[i]) V`, all fp32. Q is `[qLen, d]`, K and V `[kLen, d]`,
the optional additive mask `[qLen, kLen]` (strides are free; a stride of 0 broadcasts). `d` must be even.

## Work items and constants

A work item is one (b, h, block of `QB = 32` query rows). Items are independent and write disjoint output rows, so any distribution
of items over threads gives the same bits. `KB = 64` keys per block. `QB` and `KB` are part of the exact definition (they fix where
the online softmax rescales). They are constants of the kernel, never derived from the thread count or the CPU.

## The exact definition

For one query row, state `m = -inf`, `l = +0`, `acc[0..d) = +0`. For each key block (keys `kb .. kb+n`, `n <= 64`, in increasing order):

1. **Scores.** `dot[j] = (((+0 + q[0]*k[j][0]) + q[1]*k[j][1]) + ...) + q[d-1]*k[j][d-1]`: one accumulator, increasing index,
   product rounded, then sum rounded. `s[j] = mask[j] + dot[j]*scale`, with `mask = +0` when there is no mask. Lanes of a vector hold
   different `j`, so the lane width does not matter.
2. **Maximum.** `bm = max over j of s[j]`, ignoring NaN (`-inf` if none). `mNew = max(m, bm)`.
   `shift = (mNew == -inf) ? 0 : mNew`. Max is exact, so order is irrelevant.
3. **Weights.** `p[j] = expNonPositive(s[j] - shift)` (see `exact-transcendentals.md`).
4. **Block sum.** `bs` = the 16-partial-sum order: partial `t` adds `p[j]` for `j mod 16 = t` in increasing `j`, starting from `+0`
   (padding positions contribute `+0`, which changes nothing); then `a[t] = P[t] + P[t+8]` (t < 8), `b[t] = a[t] + a[t+4]` (t < 4),
   `c[t] = b[t] + b[t+2]` (t < 2), `bs = c[0] + c[1]`.
5. **Rescale.** `alpha = 0` if `m == -inf`, `1` if `m == shift`, else `expNonPositive(m - shift)`. `l = l*alpha + bs`.
   `acc[x] = acc[x]*alpha` for every x (skipped when `alpha == 1`, which changes nothing). `m = mNew`.
6. **Values.** For `j` in increasing order within the block: `acc[x] = acc[x] + p[j]*v[j][x]` (product rounded, then sum rounded).

After the last block `out[x] = acc[x] / l`. Any NaN in the output becomes `0x7FC00000`. A query row whose scores are all `-inf`
(fully masked) gives `l = 0` and therefore NaN in every output of the row; a row with a NaN score or a `+inf` score also gives
all-NaN. A fully masked *first block* followed by visible keys is fine (`alpha = 0`, `shift = 0`).

The fast flavour runs the same code with fused `acc + a*b` (FMA on x64 with FMA3, AVX-512 or Arm FMLA), `FastMath.Exp`,
`out[x] = acc[x] * (1/l)`, and is exactly as deterministic: same answer on the same machine for any thread count and any run. It differs from
the exact flavour in the last bits only.

## Softmax

`Softmax.Rows`: `m = max(x)` (NaN ignored), `e[j] = expNonPositive(x[j] - m)`, `l` by the 16-partial-sum order above over the whole row,
`y[j] = e[j] / l` (fast: `e[j] * (1/l)`); NaN becomes `0x7FC00000`.

## Layout used by the code (not part of the definition)

Keys of a block are packed transposed (`d x 64`, zero padded) so a micro-tile of 4 query rows x 2 vectors of keys streams contiguous
lanes (Goto & van de Geijn 2008, packing); `P V` uses a 4 x 2-vector tile over the head dimension. Query blocks shorter than a multiple of 4 rows
compute duplicate rows that are discarded. When `d` is not a multiple of twice the lane count the kernel narrows the lane width.
Vector loads and stores go through `IVecOps.Load/Store`; the Checked build asserts their range.

## Accuracy

Versus a double-precision naive reference, max absolute error on outputs of magnitude up to about 3 (inputs q, k ~ N(0, 1.5^2)):
fast and exact are both between 1.1e-6 and 3.4e-6, set by the fp32 accumulation of the scores (a logit error of ~1e-6 is a
relative weight error of ~1e-6). Tests use a bound of 4e-6.

## Cross-platform check

`AttentionTests.ExactOutputsMatchTheGoldenHashes` fixes SHA-256 hashes of exact attention (with and without mask, d = 32 and 64, 1500 keys) and
softmax outputs. They match at scalar, 128-, 256- and 512-bit lanes, 1 or many threads, on x64 with `DOTNET_EnableHWIntrinsic=0`, on arm64 (M2, NEON) and in a
NativeAOT build.

## Exact versus fast, measured (indicative: the shared worker carried 7 to 10 other busy threads, so these are not bar-grade)

Batch 1, fp32, medians of 3, milliseconds, AMD Ryzen 9 7950X3D (Zen 4, 32 logical cores), .NET 10.0.11. "32" is `Parallel.For` over the work items on 32 threads (bench code only).

| shape | lanes | threads | fast | exact | exact/fast |
|---|---|---|---|---|---|
| BERT-base seq 128 | AVX2 (256) | 1 / 32 | 0.964 / 0.092 | 1.410 / 0.115 | 1.46 / 1.25 |
| BERT-base seq 128 | AVX-512 | 1 / 32 | 0.741 / 0.078 | 1.435 / 0.093 | 1.93 / 1.19 |
| BERT-base seq 512 | AVX2 | 1 / 32 | 12.66 / 1.52 | 14.31 / 1.73 | 1.13 / 1.13 |
| BERT-base seq 512 | AVX-512 | 1 / 32 | 12.81 / 1.26 | 14.62 / 1.53 | 1.14 / 1.22 |
| MiniLM-L6 seq 128 (12 x 32) | AVX2 | 1 / 32 | 0.460 / 0.061 | 0.558 / 0.069 | 1.21 / 1.14 |
| MiniLM-L6 seq 128 | AVX-512 | 1 / 32 | 0.421 / 0.053 | 0.555 / 0.067 | 1.32 / 1.27 |
| Whisper-small encoder self 1500 | AVX2 | 1 / 32 | 102.8 / 10.64 | 122.1 / 12.69 | 1.19 / 1.19 |
| Whisper-small encoder self 1500 | AVX-512 | 1 / 32 | 102.3 / 9.11 | 124.0 / 11.92 | 1.21 / 1.31 |
| Whisper decoder cross, 1 step | AVX2 | 1 / 32 | 0.643 / 0.123 | 0.703 / 0.124 | 1.09 / 1.01 |
| Whisper decoder cross, 1 step | AVX-512 | 1 / 32 | 0.620 / 0.107 | 0.641 / 0.114 | 1.03 / 1.06 |
| Whisper decoder cross, 448 steps | AVX2 | 1 / 32 | 32.77 / 3.28 | 39.91 / 4.08 | 1.22 / 1.24 |
| Whisper decoder cross, 448 steps | AVX-512 | 1 / 32 | 33.39 / 2.86 | 39.44 / 3.69 | 1.18 / 1.29 |

Apple M2 (NEON, 128-bit lanes, some other load present), 1 / 8 threads: BERT-base 512 22.4 / 4.67 fast vs 27.9 / 6.16 exact (1.25 / 1.32);
MiniLM 128 0.832 / 0.190 vs 1.042 / 0.238 (1.25 / 1.25); Whisper encoder 1500 194.2 / 43.8 vs 244.4 / 55.0 (1.26 / 1.25);
decoder cross 1 step 1.032 / 0.284 vs 1.118 / 0.315 (1.08 / 1.11); 448 steps 57.8 / 11.9 vs 71.4 / 17.9 (1.24 / 1.50).
The exact outputs hash identically on both machines (e.g. BERT-base 128: `7fccb21b1fcc4d3b`).

Attention exact/fast stays between about 1.0 and 1.5 (AVX-512 BERT-128 at one thread, 1.9, is a short 0.7 ms call and noisy). Zen 4 has separate multiply and add pipes, so an unfused multiply-add costs it
little in the matrix products; the exponential's extra polynomial term, the unfused polynomial steps and the divide at the end are most of the gap. On a CPU whose
multiplier and adder share the FMA ports (Intel) the products would cost more, so this should be re-measured there.

Share of a model: counting multiply-accumulates per layer, attention (`Q K^T` and `P V`) is 2.7% of BERT-base at seq 128, 10% at seq 512,
5.3% of MiniLM-L6 at seq 128 and 25% of the Whisper-small encoder at seq 1500. If the rest of the model costs the same exact or fast, the model-level exact/fast from attention alone is
1 + share x (ratio - 1): about 1.01 (BERT 128), 1.02 (BERT 512), 1.01 (MiniLM), 1.05 (Whisper encoder). Attention is less efficient than a matrix product, so its time share is larger than its operation share;
even at three times the share the contribution stays below 1.15. The model-level ratio is then set by the GEMM and convolution exact/fast ratio and by the elementwise functions
(GELU: exact/fast 1.1 to 1.4 at AVX2; exp 2.3 to 2.6).

## Matrix kernels

The score and value products are the micro-tiles written here (no dependence on the GEMM unit). When the shared GEMM exists the packed-key
score product and the `P V` product can call it, provided its exact flavour accumulates each output in increasing reduction index from `+0`, one accumulator, unfused.
