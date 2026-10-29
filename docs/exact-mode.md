# Exact mode

Exact mode gives bit-identical results on every operating system, CPU, instruction set, thread count and runtime (including
WebAssembly), so a port to another language or platform can reproduce them from this text alone. Fast mode (the default) gives
up that promise for speed; its rules are in `kernels-gemm.md`.

## Rules every exact kernel follows

1. **IEEE binary32 arithmetic with round-to-nearest-even.** Only `+`, `-`, `*`, `/` and square root are used. Each operation is
   rounded to binary32 on its own: **no fused multiply-add** in any form (no `fma`, no multiply-accumulate instructions that skip
   the intermediate rounding, no estimate instructions). Flush-to-zero and denormals-are-zero modes must be off, so subnormals are
   computed, not dropped.
2. **One accumulator per output, in a stated order.** Each kernel below states the order in which the terms of an output are added.
   Vector lanes always hold different outputs, never different terms of one output, so the vector width cannot change a result.
3. **Start from +0.** An accumulator starts as positive zero. `+0 + (-0)` is `+0`, so an output is never `-0` unless it is the
   result of a later operation that yields `-0` (it cannot: a round-to-nearest sum is `-0` only when both operands are `-0`).
4. **NaN is canonical.** Hardware disagrees on the sign and payload of the NaN that `inf - inf`, `0 * inf` or a NaN input produces.
   Every NaN an exact kernel stores is replaced by the quiet NaN with bits `0x7FC00000`. Infinities, signed zeros and subnormals
   follow IEEE and need no special handling.
5. **No dependence on blocking or threads.** Cache blocking, tile shapes and chunking may differ between paths and machines; the
   order of additions inside one output may not.
6. **Transcendental functions** (exp, log, tanh, erf, sigmoid) are our own fixed polynomials, defined in
   `exact-transcendentals.md`.

Kernels that cannot meet these rules are not exact; selecting exact mode for an operator without an exact kernel is reported, not
silently approximated.

## Matrix product (`C = A * B`, row-major, A is m x k, B is k x n)

For every output `(i, j)`:

```
acc = +0
for p = 0, 1, ..., k-1 in increasing order:
    prod = round(A[i][p] * B[p][j])
    acc  = round(acc + prod)
C[i][j] = isNaN(acc) ? 0x7FC00000 : acc
```

With `k = 0` the output is `+0`. This is the whole definition: tile sizes, packing, the split of `k` into blocks (a block adds its
terms to the accumulator left by the previous block, in the same order), the assignment of tiles to threads and the vector width
do not appear in it. The product `A[i][p] * B[p][j]` is commutative, so operand order does not matter; the sum is not
associative, so the order of `p` does.

The same definition covers the score product `q . k` of attention (`exact-transcendentals.md` and `kernels-attention.md` use the
same order: one accumulator, increasing index, unfused, from +0), so attention can call the matrix product.

## Convolution (2-D, NCHW, weights `[Cout, Cin/G, KH, KW]`)

For output `(n, oc, y, x)` of group `g = oc / (Cout/G)`:

```
acc = +0
for c = 0 .. Cin/G-1, then ky = 0 .. KH-1, then kx = 0 .. KW-1 (row-major, in that nesting):
    iy = y*strideH - padTop  + ky*dilationH
    ix = x*strideW - padLeft + kx*dilationW
    v  = (0 <= iy < H and 0 <= ix < W) ? input[n][g*Cin/G + c][iy][ix] : +0
    acc = round(acc + round(weight[oc][c][ky][kx] * v))
out = isNaN(acc) ? 0x7FC00000 : acc
```

Padding taps are real products with `+0`, not skipped, so an infinite or NaN weight meeting padding gives NaN. This is the matrix
product of the weight matrix with the im2col matrix (rows ordered `c, ky, kx`), and the depthwise kernel (`Cin/G = 1`) computes
the same expression directly. Output size per axis is `floor((size + padBefore + padAfter - dilation*(K-1) - 1) / stride) + 1`.
No bias is added by these kernels.

## Checking it

* Every path (scalar, 4-lane, AVX2, AVX-512, NEON) is compared bit for bit with a naive implementation of the pseudo-code above
  for every size 0..70 in each dimension, with random strides and offsets, and with NaN, infinity, signed-zero and subnormal
  inputs (`tests/OneModel.Tests/Kernels`).
* `spike hashes` (in `bench/Micro`) hashes the outputs of sweeps over sizes 0..256, of random shapes and of the reference
  convolution and matrix shapes with SHA-256, for each path at 1 thread and at all threads, and prints one line per run; the exact
  lines must be equal on every machine. They were equal on x64 (default, AVX-512 off, AVX2 off, all hardware intrinsics off) and
  on arm64 NEON.
* Setting `DOTNET_EnableHWIntrinsic=0` selects the scalar path; `DOTNET_EnableAVX2=0` and `DOTNET_EnableAVX512=0` remove those
  paths (the .NET 10 switch for AVX-512 is `DOTNET_EnableAVX512`; `DOTNET_EnableAVX512F` no longer has an effect).
