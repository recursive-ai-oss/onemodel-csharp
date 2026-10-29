# Normalisation, softmax and reduction operators

Normative for the exact flavour; the fast flavour is the same unless a section says otherwise. Source: `src/OneModel/Kernels/Exact/NormKernels.cs`,
`Exact/ReduceKernels.cs`, `Kernels/Softmax.cs`, `src/OneModel/Graph/Operators/Norms/`. Operators and opsets served:

| Operator | Opsets | Notes |
|---|---|---|
| LayerNormalization | 17+ | float32; optional `Mean` and `InvStdDev` outputs |
| RMSNormalization | 23+ | float32 |
| InstanceNormalization | 6+ | float32 |
| GroupNormalization | 18-20 (scale and bias per group), 21+ (per channel) | float32 |
| Softmax, LogSoftmax, Hardmax | 1-12 (coerce to 2-D, axis default 1), 13+ (one axis, default -1) | float32 |
| ReduceSum | 1-12 (`axes` attribute), 13+ (`axes` input) | float32, float64, int32/64, uint32/64, int8, uint8 |
| ReduceMean, ReduceMax, ReduceMin, ReduceProd, ReduceL1, ReduceL2, ReduceSumSquare | 1-17 (attribute), 18+ (input) | same types; L2 float32 and float64 |
| ReduceLogSum, ReduceLogSumExp | 1-17, 18+ | float32 |
| ArgMax, ArgMin | 1+ (`select_last_index` from 12) | float32/64, int8-64, uint8-64 |
| CumSum | 11+ | float32, float64, int32/64, uint32/64 |

Not served: float16 and bfloat16 inputs, bool reductions (opset 20), `stash_type` other than float32 (the attribute is read and ignored: every
sum is accumulated in float32), and `ReduceLogSum*` on float64. The session reports these as `Unsupported` naming the operator.

## Exactness

All of these are exact kernels: only IEEE `+ - * /` and `sqrt`, no fused multiply-add, the orders below, and every NaN result rewritten to
`0x7FC00000` (also in fast mode: the kernels share one code path). The fast and exact flavours therefore give the same bits, on every
kernel path and at every thread count, except LogSoftmax and Softmax, whose fast flavour uses the fast exponential (see
`kernels-attention.md` and `exact-transcendentals.md`). The log in LogSoftmax, ReduceLogSum and ReduceLogSumExp, and the exponential in
ReduceLogSumExp, are the library's own (`exact-transcendentals.md`), evaluated one element at a time.

## The canonical sum

Every sum of `n` terms `t[0..n)` below (mean, variance, L1, L2, sum of squares, log-sum) is the sum of `RowReduce.Sum16`: 16 partial sums,
term `j` added to partial `j mod 16` in increasing `j` starting from `+0`, then `a[t] = P[t] + P[t+8]` (t < 8), `b[t] = a[t] + a[t+4]`
(t < 4), `c[t] = b[t] + b[t+2]` (t < 2), result `c[0] + c[1]`. Vector lanes hold different partials, so the lane width does not matter.

## LayerNormalization, RMSNormalization

The input is viewed as `rows x cols`: `cols` is the product of the dimensions from `axis` on (default -1; negative axes count from the end).
For each row, with `n = cols` converted to float32:

1. `mean = Sum16(x) / n` (RMSNormalization: `mean = 0`).
2. `v = Sum16((x[j] - mean)^2) / n`: the difference and the square are separate rounded operations.
3. `inv = 1 / sqrt(v + epsilon)` (RMSNormalization: `v` is the mean of squares).
4. `y[j] = (((x[j] - mean) * inv) * scale[j]) + bias[j]`, four separate operations; without `B` the last addition is skipped.

`Scale` and `B` broadcast to the normalized dimensions (right-aligned; missing leading dimensions are ones). When one has fewer elements than the
normalized extent it is expanded into a scratch array first. `Mean` and `InvStdDev` hold `mean` and `inv` per row, shaped like the input with
the normalized dimensions replaced by ones. Rows are processed in chunks of `max(1, 16384 / cols)` rows; a row is never split.

## InstanceNormalization, GroupNormalization

A row is one (sample, channel) plane, or one (sample, group) block of `C / num_groups` channels times the spatial positions. Step 1 to 4 are the
same with the scale and bias taken per row: InstanceNormalization and GroupNormalization (opsets 18-20) use one scalar scale and bias per
channel or group for the whole row; GroupNormalization at opset 21 and later has per-channel parameters, spread across the block. The
opset-18 definition indexed scale and bias by group and was corrected in 21; both are implemented as their opset defines them.

## Softmax, LogSoftmax, Hardmax

Up to opset 12 the input is flattened to two dimensions at `axis` (rows = product of the dimensions before it, columns = the rest) and each row
is normalised. From opset 13 the operator works along `axis` of the original shape (default -1). When the axis is not the last, the data is moved
to rows in a scratch copy and moved back. Rows are processed in chunks of `max(1, 16384 / cols)` rows.

- **Softmax**: `kernels-attention.md`. A row holding NaN or +inf, or consisting only of -inf, gives NaN throughout (as ONNX Runtime does).
- **LogSoftmax**: `d[j] = x[j] - max`, `l = Sum16(exp(d[j]))` (exponential for non-positive arguments), `y[j] = d[j] - log(l)`; the same NaN rows
  as Softmax; a `-inf` element in an otherwise normal row stays `-inf`.
- **Hardmax**: 1 at the first maximum of the row, 0 elsewhere; a row containing NaN is all zeros (the one probed case of ONNX Runtime).

## Reductions

**Axes.** Opsets before the change read `axes` from the attribute; later ones read the second input (int64), which may be computed at run
time (the node is then shaped at run time). Negative axes count from the end. An absent or empty `axes` reduces every axis, unless
`noop_with_empty_axes` is set: then no axis is reduced and each output is the operation applied to its single element. ONNX Runtime does it
this way (ReduceSumSquare squares, ReduceLogSum takes the log, ReduceL1 and ReduceL2 take the absolute value), so we do too. `keepdims`
(default 1) keeps reduced axes as extent 1.

**Geometry.** Dimensions of extent one are dropped and neighbouring dimensions that are both reduced or both kept are merged, leaving
alternating runs of kept and reduced extents. Two layouts follow from the shape and axes alone:

- *Column layout* (the last run is kept): the last run is `inner` contiguous outputs. Each output is accumulated sequentially over the reduced
  index tuples in row-major order, starting from the identity (`+0` for sums, `1` for products). Neighbouring outputs share a vector, so
  reducing a leading axis costs one pass over the data, not a strided gather. Work is split into tiles of 256 outputs.
- *Row layout* (the last run is reduced): each output reduces a sequence made of the reduced index tuples in row-major order, innermost run
  contiguous. Sums use the canonical sum over that sequence (element `q` of the sequence goes to partial `q mod 16`). Products multiply
  sequentially from `1`.

**Chunks.** A chunk covers a whole number of tiles (column layout) or outputs (row layout): `max(1, 32768 / work)` of them, where `work` is the
elements one unit reads. It depends on the geometry only. Every output is computed by one chunk, so no partial result is combined across chunks
and thread count cannot matter. (A reduction to a few outputs over a huge extent therefore runs on few threads; splitting it would need a
chunked combine in a fixed order, which can be added later without changing results at the sizes where it does not apply.)

**Terms and finishing.** Sum: `x`. Mean: `x`, then `sum / count` (`count` as float32; an empty reduction gives 0; integers divide, truncating
toward zero). L1: `|x|`. L2: `x*x`, then `sqrt`. SumSquare: `x*x`. LogSum: `x`, then the library's `log`. Prod: multiply. LogSumExp: `m` = the
maximum (NaN if any element is NaN), then `m + log(sum of exp(x - m))` in sequence order; `+-inf` maxima are returned as they are.

**Maximum and minimum and NaN.** The result is that of a sequential scan: the running value starts as the first element and is replaced only
by a strictly better one. NaN compares false, so it is ignored unless it is the first element, in which case the result is NaN. An empty
reduction gives `-inf` (max) or `+inf` (min). *ONNX Runtime's result for NaN inputs depends on its internal code path.* Probed with random integer-valued data with 20% NaN, 150 trials each: for reduced axes that are not innermost this rule matched every trial; for an innermost reduced axis it matched 150 of 150 trials at extents 3 and 300, 146 of 150 at extent 7, 68 of 150 at extent 50 and 0 of 150 at extent 6. So NaN in ReduceMax and ReduceMin is documented, not matched. The spec leaves NaN open.

**Other types.** Integer sums, products and sums of squares wrap around; L1 of the minimum value wraps. float64 follows the same order with
float64 arithmetic one output at a time (no vector kernel).

## ArgMax, ArgMin

Scan the axis from index 0; a later element wins when strictly greater (ArgMax) or smaller (ArgMin), or, with `select_last_index`, when greater
or equal. NaN never wins unless it is first. ONNX Runtime's behaviour with NaN depends on its code path (a 1-D input made a NaN win in the probe;
a 3x3 input did not), so NaN is documented, not matched; ties and ordinary values match. Output is int64; `keepdims` default 1. An axis of
extent 0 gives zeros. This runs on one thread.

## CumSum

Sequential along the axis in the direction of accumulation: inclusive `y[i] = y[i-1] + x[i]` with `y[0] = x[0]`; `exclusive` shifts so
`y[0] = 0` and `y[i] = y[i-1] + x[i-1]`; `reverse` runs from the last element. Floats are added in exactly that order. The axis input is read at run
time; an axis outside `[-rank, rank)` is an `InvalidInput` error. This runs on one thread.

## Conformance

Per-operator results of the `onnx/backend/test` node cases (ONNX 1.22.0 data, fast and exact mode) and of the randomized comparison with ONNX Runtime
1.31.1 (`tools/ops_norms.py`, scaled error at most 1e-4, integers exact) are recorded in the repository's conformance table when it is
published; the randomized cases cover every operator above at the opsets listed, with axis subsets, keepdims, odd widths and special values.
