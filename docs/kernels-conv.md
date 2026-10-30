# Convolution (fast mode)

Regular and grouped 2-D convolutions in NCHW (one-dimensional ones as height 1) run as one blocked matrix product per image and group,
with the input patches gathered straight into the packed layout of the product's B operand. Depthwise convolutions have their own
kernel ([kernels-depthwise.md](kernels-depthwise.md)). Exact mode keeps the im2col path described in [kernels-gemm.md](kernels-gemm.md);
everything below is fast mode only.

## What each output is

Same as the matrix-product path: output `(n, oc, oy, ox)` is the sum over taps `(ci, ky, kx)` in that order (channel, kernel row,
kernel column) of `w[oc, ci, ky, kx] * x_padded[n, g*Cin/G + ci, oy*StrideH + ky*DilH, ox*StrideW + kx*DilW]`, accumulated with one fused
multiply-add per tap from +0, continuing from the stored partial sum across blocks of `Kc = 256` taps. Padding taps are real
multiplications by zero. The order does not depend on the tile shape, the thread count or whether the weights were packed beforehand,
so all of those give the same bits (the tests check packed against unpacked and 1 against several threads).

## Layout of the work

* **A is the weights**, `Cout/G` rows by `K = Cin/G*KH*KW` columns, taps ordered as above. They may be packed once, when the model is
  loaded: per group, for each block of `Kc` taps, strips of `MR` rows, each strip a `kc x MR` panel (element `(p, r)` at `p*MR + r`),
  a ragged last strip padded with +0. Block `pc` of a group starts at `pc * ceil(Cout/G / MR) * MR`; strip `s` of it at `s * kc * MR`.
  Unpacked weights are packed per tile instead, as `Gemm` does for A.
* **B is the patches**, `K` rows by `OutH*OutW` columns. It is never built as a matrix: strips of `NR` columns, each `kc x NR` (element
  `(p, j)` at `p*NR + j`, a ragged last strip padded with +0), are gathered from the input planes.
* **Padding is done once.** When any padding is present the input planes are first copied into zero-bordered planes (chunks of 4
  planes), so gathering never tests for the edge. A tap `(ky, kx)` of output column `(oy, ox)` then reads padded row
  `oy*StrideH + ky*DilH`, column `ox*StrideW + kx*DilW`, which the output-size formula keeps inside the padded plane.
* **Runs.** For the columns of a tile, the columns that are consecutive in one output row form a *run* (first padded-plane position,
  place in the strip row, length). The runs are the same for every tap, so they are listed once per tile; a tap is then one copy per run
  (contiguous when `StrideW = 1`, strided otherwise). With unit strides and a one-column kernel (the 1x1 case, and `3x1`), consecutive output
  columns are consecutive in memory across rows too, and a strip is a single run.

## Tiles and chunks

A tile is `Mc` rows by `Nc` columns of one group of one image; a tile is a chunk of work. `Nc` and `Mc` are chosen from the shape only,
from the options `(Nc, Mc) = (128, 96), (64, 96), (32, 96), (32, 48), (32, 24)` (each rounded down to whole strips, at least one): the
first option that gives at least 48 chunks over the whole call, or the last option whose tile holds at least 500 000 multiply-adds. The
rows are then cut into equal row tiles, rounded up to whole strips (128 rows run as 72 + 56, not 96 + 32). Wide tiles come first because
every tile gathers its own columns.

* **One row tile** (`Cout/G <= Mc`): each tile gathers its own columns for each block of taps, and the whole product is one parallel
  phase.
* **Several row tiles:** per image and group the strips are first gathered once, in parallel (chunks of 8 strips of one block of
  taps), into one buffer laid out `[block][strip][kc][NR]`; the row tiles then share it in a second parallel phase.

The epilogue (bias, then activation) is applied to each finished tile exactly as the matrix-product path does, with rows counted from the
group's first output channel. Tile shapes are constants of the shape, so where an activation's vector body hands over to its scalar tail
does not depend on the thread count.

## Threads

The work estimate given to the pool is the multiply-add count divided by 8 per nanosecond (a core does about 40). The pool wakes at most
one thread per 80 microseconds of estimated work; with the conservative rate the small layers of MobileNetV3 stayed on one thread. Measured
on 8 threads (indicative, shared host), MobileNetV3-large's convolutions took 4.0 ms with the rate set to 40, 2.9 ms at 20, 2.1 ms at 10
and 1.9 ms at 5 or lower; ResNet-50's did not change. Depthwise convolutions use four times their estimate for the same reason.

## Memory

One packed copy of the weights replaces the original (the pass releases the original once nothing else reads it). Scratch per thread:
one panel of at most `Nc * Kc` floats, and the packed A block when the weights are not prepacked. The padded planes and the shared
strips live in the calling thread's scratch for the duration of the call.

## Max pooling and element-wise work

Two-dimensional float `MaxPool` without indices runs over planes in chunks of about 16 384 outputs, vectorised along the output row for
unit column stride; it returns the same values as the scalar rule (NaN taps ignored, first of equal taps wins, lowest float when no tap
is inside the input). Binary element-wise operators on large outputs (at least two chunks of 32 768 elements, at most two merged
dimensions) are split into chunks of 32 768 elements or whole rows; each element is computed from its own inputs, so the split changes
nothing. A single-row matrix product (`m = 1`) streams B once with a matrix-vector kernel ([kernels-gemm.md](kernels-gemm.md)).
