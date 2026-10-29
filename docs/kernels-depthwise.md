# Depthwise convolution

Applies to a `Conv` with `Groups == Cin` (one input channel per group; `Cout = Cin * multiplier`). Other convolutions are lowered to the
matrix product ([kernels-gemm.md](kernels-gemm.md)), which would be all overhead here because `k = KH * KW`.

## What each output is

For output plane `q` (image `n`, output channel `oc`; input channel `oc / multiplier`) and output position `(oy, ox)`:

```
acc = +0
for ky in 0 .. KH-1:            # row-major tap order
  for kx in 0 .. KW-1:
    acc = acc + w[oc, ky, kx] * x_padded[oy*StrideH + ky*DilH, ox*StrideW + kx*DilW]
```

`x_padded` is the input plane with the padding written as real zeros, so a padding tap is a multiplication by zero (an infinite or NaN
weight meets padding as NaN). Exact mode: the product is rounded, then the sum is rounded (no fused operation), and a NaN result is
rewritten as the canonical NaN `0x7FC00000`. Fast mode: one fused multiply-add per tap where the hardware has it (AVX2+FMA, AVX-512,
NEON), `acc + w*x` elsewhere. The bias, when present, is a separate pass: one rounded addition per output.

Every output is computed independently with this one order. The lane count of the vector code therefore never changes a result:
scalar, 4-lane, 8-lane and 16-lane paths give the same bits in exact mode, and in fast mode too wherever the fused operation is the same
(the fused scalar twin `MathF.FusedMultiplyAdd` is used when the hardware has a fused instruction, so the scalar path equals the vector
paths on that machine). Output columns that a path computes twice (see the last vector below) produce identical values both times.

## Layout of the work

* **Lanes are neighbouring output columns of one row.** A tap is one contiguous load of `W` input values at column `ox + dx / StrideW`
  (see phases) and one multiply-add with the broadcast weight.
* **Phase planes.** The zero-padded input plane is split by column phase into `StrideW` planes: plane `p` holds padded columns
  `p, p + StrideW, p + 2*StrideW, ...`. The tap at padded column `ox*StrideW + dx` (with `dx = kx * DilW`) is then at column
  `ox + dx / StrideW` of plane `dx % StrideW`, which is contiguous in `ox` for any stride. A stride-1 plane without padding is read in
  place; with padding it is copied into one zero-filled padded plane. For stride 2 the split is done with 4-lane shuffles (SSE
  `shufps`, NEON `uzp1/uzp2`) on eight source values at a time; only data moves.
* **Vectors per row.** A row is processed four vectors at a time, then one vector at a time; the last vector of a row is moved back to
  end at the last column (it recomputes up to `W-1` outputs with identical results) instead of a scalar tail. The lane type is the
  widest of 16 (AVX-512), 8 (AVX2), 4 (NEON, SSE, WebAssembly SIMD), 1 (scalar twin) that fits in the output width.
* **Narrow rows.** When the output is narrower than four vectors and at least four rows tall, four rows are computed together for one
  vector column, so that each tap's weight is shared by four multiply-adds. Remaining rows use the row-wise form.
* **Fallback.** Kernels with more than 64 taps or a column stride above 8 take a plain per-output loop in the same tap order.

## Chunks

A chunk is `clamp(4096 / (OutH * OutW), 1, 32)` output planes: a constant of the shape, never of the thread count. A chunk writes only
its own planes, so results do not depend on which thread ran it. Phase-plane scratch is per thread. The work estimate given to the pool
is `planes * (OutH*OutW*KH*KW/8 + PaddedH*PaddedW/2)` nanoseconds.

## Measured (indicative, one thread, one x64 machine with AVX2 and AVX-512, shared host)

GFLOP/s of the MobileNetV3-large depthwise layers, before (im2col-free kernel with scalar tails and strided gathers) and after:

| Layer | Before (AVX2, fast) | After (AVX2, fast) |
|---|---|---|
| 16->16 3x3/1 @112 | 13-17 | 41 |
| 64->64 3x3/2 @112 | 7.5-8 | 17-20 |
| 72->72 3x3/1 @56 | 13-16 | 34 |
| 72->72 5x5/2 @56 | 7 | 38 |
| 120->120 5x5/1 @28 | 10-12 | 50 |
| 240->240 3x3/2 @28 | 3-4 | 14 |
| 200->200 3x3/1 @14 | 4 | 27 |
| 672->672 3x3/1 @14 | 5 | 26 |
| 672->672 5x5/2 @14 | 2.8 | 9 |
| 960->960 5x5/1 @7 | 2-3 | 13 |
