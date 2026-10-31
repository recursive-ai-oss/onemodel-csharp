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
* **Padding is never materialised** (fast mode). The zero border is written while gathering: for tap `(ky, kx)` and output row `oy` the
  input row is `oy*StrideH + ky*DilH - PadTop`; a row outside the plane is all zeros, and in a row the columns outside the plane (left
  and right of a run) are zeros too. These are the same products by zero that a padded copy would give, so the result is bit for bit
  the one the padded copy gave. The earlier design copied the planes into zero-bordered planes in a first parallel phase; on a 16-core
  Zen 4 desktop that phase cost 40-140 microseconds per 3x3 layer of ResNet-50 (the single thread that built the copy, and the cache
  traffic of every other core fetching it) and is gone: the convolution chain at 16/32 threads went from 12.7/11.6 to 12.3/11.1 ms.
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

## Register tile and in-place input (AVX-512)

On AVX-512 a convolution does not use the matrix product's 12x32 tile. It uses 16x16, or 8x32 when the output channels per group leave a
remainder of 1 to 8 modulo 16 (24, 40, 72, 88, ...), so the rows are padded as little as possible: 12-row strips padded a 64-channel layer to 72
rows, and 16-column strips pad small maps less (196 columns are 12.25 strips of 16 and 6.1 of 32). The choice depends on the channel count alone,
because the weights are packed when the model is loaded, before any input size is known. Every output is still summed in increasing tap order
with one fused multiply-add per tap, so the bits do not depend on the tile (the tests compare packed against unpacked weights and 1 against
several threads). Other paths keep the matrix product's tile.

With a 1x1 kernel, unit strides and no padding the patch matrix B is the input itself: row `p` is input plane `p`, `OutH*OutW` floats long. A full
strip of columns is then read where it lies (the microkernel takes the plane length as the row stride of B) instead of being copied into a packed
strip, and only a ragged last strip of columns is packed. There is no gathering phase, so the shared-strip second phase of the several-row-tile
case is not needed either: a layer is one parallel phase. On ResNet-50 (x64, 16 threads, indicative, shared host) the whole network went from
12.2 to 10.9 ms and 8 threads from 15.4 to 14.1 ms; on one thread it did not change.

Gathering with a stride of two columns (the 7x7 stem, 3x3 stride-2 layers, 1x1 downsampling) picks every other element of eight with one
shuffle on SSE and Arm64, and the padded gather finds the columns inside the plane without integer divisions for unit strides.

## Threads

The work estimate given to the pool is the multiply-add count divided by 8 per nanosecond (a core does about 40). The pool wakes at most
one thread per 20 microseconds of estimated work (it was 80; on a 16-core Zen 4 desktop, 8/16/32 threads, ResNet-50 convolution chain 18.7/14.6/14.3 ms at 80 us,
16.6/13.0/12.0 at 30 us, 16.6/12.7/11.6 at 20 us, MobileNetV3-large 3.06/2.30/2.19 ms at 80, 2.35/1.66/1.58 at 30, 2.30/1.63/1.57 at 20; indicative, shared host);
the shared gathering phase is estimated at one nanosecond per gathered element (k times n), so it is spread over threads too. With the old rate the small layers of MobileNetV3 stayed on one thread. Measured
on 8 threads (indicative, shared host), MobileNetV3-large's convolutions took 4.0 ms with the rate set to 40, 2.9 ms at 20, 2.1 ms at 10
and 1.9 ms at 5 or lower; ResNet-50's did not change. Depthwise convolutions use four times their estimate for the same reason.

## Single-column products

A convolution with a 1x1 kernel over a 1x1 input (the squeeze-and-excitation layers of MobileNet-style networks: `Cout x Cin` weights
against one vector) with prepacked weights is a matrix-vector product, and the tile microkernel would compute 32 columns to keep one.
It runs instead as follows, on every path whose strip height `MR` is a multiple of four (AVX-512, NEON, the 4-lane and scalar paths;
the AVX2 path with `MR = 6` keeps the tile kernel).

* The rows of a strip are cut into slivers of four rows. A sliver's four outputs are one 4-lane vector; each step of k broadcasts the
  input value `x[p]` and does one fused multiply-add of it with the sliver's four packed weights (`p*MR + r` in the strip). Eight slivers
  (then four, then one) run together to give the multiply-add independent chains.
* Each output is accumulated in increasing k with one fused multiply-add per step, from +0 or from the stored partial sum of the previous
  block of `Kc` taps, which is the tile microkernel's order, so the bits are the same as the tile path's (the tests compare the two).
  Where no fused multiply-add exists the multiply and add are separate, as in the 4-lane and scalar tile kernels.
* The last sliver of a ragged strip is computed on a four-float copy and only its real rows are stored.
* A chunk is eight strips of one image and group (`8 * MR` rows), a constant of the shape. The bias, activation and residual epilogue is
  applied to each chunk's rows (a `rows x 1` tile).
* The work estimate for the pool is one nanosecond per four weights, since the product is bound by streaming the weights once, so only
  layers of about 160 000 weights or more use a second thread.

Measured on MobileNetV3-large (indicative, one x64 machine with AVX-512, shared host, same job, median of three): the sixteen
squeeze-and-excitation convolutions took 0.78 ms of a 7.4 ms run on one thread and 0.87 of 3.2 ms on eight; the model runs 6.75 -> 6.24 ms
on one thread, 2.45 -> 1.41 ms on eight and 3.0 -> 2.1 ms on sixteen.

## Memory

One packed copy of the weights replaces the original (the pass releases the original once nothing else reads it). Scratch per thread:
one panel of at most `Nc * Kc` floats, and the packed A block when the weights are not prepacked. The shared
strips live in the calling thread's scratch for the duration of the call.

## Max pooling and element-wise work

Two-dimensional float `MaxPool` without indices runs over planes in chunks of about 16 384 outputs, vectorised along the output row for
unit column stride; it returns the same values as the scalar rule (NaN taps ignored, first of equal taps wins, lowest float when no tap
is inside the input). Binary element-wise operators on large outputs (at least two chunks of 32 768 elements, at most two merged
dimensions) are split into chunks of 32 768 elements or whole rows; each element is computed from its own inputs, so the split changes
nothing. A single-row matrix product (`m = 1`) streams B once with a matrix-vector kernel ([kernels-gemm.md](kernels-gemm.md)).
