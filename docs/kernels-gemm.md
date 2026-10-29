# Matrix product and convolution kernels

fp32, row-major, `C = A * B` with `A` m x k, `B` k x n and row strides `lda`, `ldb`, `ldc`; convolution in NCHW. Both modes
exist for every kernel; the exact mode's definition is in `exact-mode.md`.

## Structure (Goto and van de Geijn 2008, "Anatomy of high-performance matrix multiplication", section 4)

1. **Pack B** once per call, for all of `k` and `n`: for each block of `KC = 256` rows of B, strips of `NR` columns, element
   `(p, j)` of a strip at `p*NR + j`; a ragged last strip is padded with +0. Work is split into chunks of 8 strips per k-block.
2. **Tiles.** C is cut into `MC x NC` tiles (the constants below). A tile is one chunk of work. For each k-block the tile packs its
   `MC x KC` block of A into strips of `MR` rows (element `(p, r)` at `p*MR + r`, padded with +0), then for each `NR`-wide strip of
   B it runs the microkernel down the A strips, so the B strip (`KC x NR x 4` bytes, 16 KB for AVX2) stays in the first-level
   cache while the A block (about 100 KB) streams from the second level.
3. **Microkernel.** An `MR x NR` tile of accumulators in registers; each step of k loads one A column of `MR` values and one
   B row of `NR` values and updates every accumulator. A later k-block starts from the C values the earlier blocks stored; the
   first block starts from +0. Ragged edge tiles run the full microkernel on a scratch tile and copy out the valid part.

| Path | MR x NR | MC x NC | Accumulators |
|---|---|---|---|
| AVX2 + FMA | 6 x 16 | 96 x 128 | 12 of 16 ymm |
| AVX-512 | 12 x 32 | 96 x 128 | 24 of 32 zmm |
| NEON (arm64) | 8 x 12 | 96 x 120 | 24 of 32 q (multiply by element) |
| 4-lane `Vector128` (SSE2, WebAssembly SIMD) | 4 x 8 | 64 x 128 | 8 |
| Scalar | 4 x 4 | 64 x 64 | 16 |

The scalar and 4-lane paths run wherever .NET runs; the 4-lane path is what browser WebAssembly uses (the JIT lowers
`Vector128` to WebAssembly SIMD).

## Convolution

* **Regular and grouped:** im2col (Chellapilla, Puri and Simard 2006): per image and group, the input patches become the columns
  of a `(Cin/G*KH*KW) x (OutH*OutW)` matrix, rows ordered channel, kernel row, kernel column; the group's weights are A and
  the group's output channels are C. Rows of the im2col matrix are built in chunks of 16. A 1x1, stride-1, unpadded
  convolution uses the input directly as B.
* **Depthwise** (`Groups == Cin`, any channel multiplier): a direct kernel over output columns (lanes are different columns), taps
  added in row-major order. The input plane is copied into a zero-padded plane first so that no tap needs a bounds check; planes
  are processed in chunks of 4.

## Fixed chunks and thread independence

Chunk sizes (8 strips of B per packing chunk, one `MC x NC` tile per compute chunk, 16 im2col rows, 4 depthwise planes) are
constants of the kernel and the shape. They are never derived from the number of threads or processors. Every output is computed
by exactly one chunk, with the whole of `k` in one order, so there are no partial results to combine and a result cannot depend on
which thread ran a chunk or on how many threads there were. `k` is never split across threads. A fixed-chunk parallel-for
(`Threading/ParallelFor.cs`) hands out chunk indices through an atomic counter; if the pool is busy serving another caller the
chunks run on the calling thread, with the same result.

## Fast mode

Fast mode uses fused multiply-add where the hardware has it (AVX2+FMA, AVX-512, NEON) and `acc + a*b` elsewhere. The order of
additions is the same as exact mode's: one accumulator per output, increasing k, from +0, so the fused products are summed
sequentially. Consequences measured here: the output does not depend on the thread count, the k-block size, the tile shape or the
lane width; every fused path (scalar `MathF.FusedMultiplyAdd`, AVX2, AVX-512, NEON) produced the same bits on every size tested.
This is a property of the current kernels, not a promise: fast mode is only promised to be repeatable on one machine and path.
NaN payloads are whatever the hardware produced.

**Accuracy.** For a sum of `k` products accumulated sequentially in binary32 with unit roundoff `u = 2^-24`,
`|c - exact| <= k * u * sum_p |a_p * b_p|` (Higham, *Accuracy and Stability of Numerical Algorithms*, section 3.1). The tests check
fast outputs against a double-precision reference with exactly this bound plus 1e-30 for the subnormal range.

## Measuring exact against fast

`bench/Micro` command `spike table` (script: `tools/spike-x64.sh`): for each shape, each ISA path the CPU has, 1 thread and all
threads, it times fast and exact with the same inputs; a measurement is the median over three repetitions of the mean time of
calls run for at least 80 ms after two warm-up calls. Buffers are allocated once, outside the timing; packing and im2col are
inside it. "% of peak" divides by the nominal fused-multiply-add rate: `GHz x ports x lanes x 2` per core
(2 ports of 8 lanes for AVX2, 1 or 2 ports of 16 lanes for AVX-512 depending on the core, 4 pipes of 4 lanes for the Apple M2),
times the number of physical cores in use; it is a ceiling for orientation, not a measurement of the part. Model rows sum, over
the distinct GEMM and convolution shapes of one model, `count x time`, and divide the exact sum by the fast sum; they are
weighted by each shape's measured time (so by operation count and efficiency), and include only the matrix products and convolutions.
Shapes: BERT-base and MiniLM-L6 at 32 x 128 rows (and BERT at 128 rows), the Whisper-small encoder at 1500 frames, ResNet-50 at
224 x 224 batch 1, MobileNetV3-large at 224 x 224 batch 1 (squeeze-excite and classifier layers left out; attention
products are in `kernels-attention.md`).
