# Convolution, pooling, normalisation and resize operators

Behaviour of the operators of this family beyond what the ONNX operator text says: which cases are supported, how the open edges
were settled, and which bits each operator promises. The kernels' arithmetic is in `kernels-gemm.md` (Conv) and below. Edge
behaviour was settled by running ONNX Runtime 1.31.1 as a black box (`tools/ops_conv_oracle.py` records its outputs for randomised
cases; the probe inputs are quoted below), never by reading its code.

## Operators

| Operator | Opsets | Supported | Not supported yet | Exact mode |
|---|---|---|---|---|
| Conv | 1 and later | float32; 1 or 2 spatial axes; group, dilation, strides, pads, `auto_pad`; optional bias | 3 spatial axes; other types | yes |
| ConvTranspose | 1 and later | float32; 1 or 2 spatial axes; group, dilation, strides, pads, `output_padding`, `output_shape`, `auto_pad`; optional bias | 3 spatial axes | yes |
| MaxPool | 1 and later | float32, float64, int8, uint8; 1 to 3 spatial axes; `ceil_mode`, dilations, pads, `auto_pad`; the Indices output, either `storage_order` | | yes |
| AveragePool | 1 and later | float32; 1 to 3 spatial axes; `ceil_mode`, `count_include_pad`, dilations (opset 19), pads, `auto_pad` | other types | yes |
| GlobalAveragePool, GlobalMaxPool | 1 and later | float32; any rank of at least 2 | | yes |
| BatchNormalization | 7 and later | float32, inference only | `training_mode = 1`, the running-statistics outputs, `spatial = 0` | yes |
| Resize | 10 and later | float32; any rank; nearest, linear, cubic; every coordinate mode and nearest mode; `cubic_coeff_a`, `exclude_outside`, `extrapolation_value`, `axes`, `keep_aspect_ratio_policy`; scales or sizes, constant or computed | `antialias = 1` | yes |
| Upsample | 7 to 10 | float32; nearest, linear (`bilinear`) | | yes |

"Exact" means the operator follows `exact-mode.md`: every sum is in the order written here, nothing is fused, and NaN results are
rewritten as `0x7FC00000`. Pooling, BatchNormalization and Resize use only separately rounded IEEE operations, so their fast and exact
results are the same bits; Conv and ConvTranspose follow `kernels-gemm.md`.

## Output sizes

For one spatial axis with input `n`, effective kernel `e = dilation * (kernel - 1) + 1`, stride `s`, pads `b`, `a` (before, after):

* `auto_pad` NOTSET: `floor((n + b + a - e) / s) + 1`, or with `ceil_mode` `ceil((n + b + a - e) / s) + 1`. With `ceil_mode`, if
  `(out - 1) * s >= n + b` the last window would start entirely in the end padding, and `out` is reduced by one. A window that
  does not fit (`n + b + a < e`) is an error.
* `SAME_UPPER`, `SAME_LOWER`: `out = ceil(n / s)`, total padding `max(0, (out - 1) * s + e - n)`; the odd element goes after the input
  for UPPER and before it for LOWER. `VALID`: no padding.
* **Settled against ONNX Runtime (pooling, `SAME_*` with dilation):** ONNX Runtime computes the padding of a pooling window with the
  undilated kernel and then applies the dilated size formula, so its output size differs from `ceil(n / s)` and its padding from ours.
  We follow the operator text (the formula above). No reference model uses it; `ops_conv_oracle.py` leaves those combinations out,
  and also those where ONNX Runtime's size disagrees with the text (reported as a count when it runs).
* ConvTranspose: `s * (n - 1) + output_padding + e - b - a`. With `output_shape`, the total padding is the difference to the natural
  size and is split with the extra element **before** the output (the operator text's "otherwise" branch, confirmed by ONNX Runtime's
  values); a size larger than the kernel reaches is allowed and the surplus is zero (backend case `convtranspose_output_shape`). With `SAME_*` the output is `n * s` when the kernel reaches that far, else as large as it gets (padding 0), as ONNX Runtime
  does; `SAME_UPPER` puts the extra element after, `SAME_LOWER` before.

## Conv, ConvTranspose

* One-dimensional operators run as two-dimensional ones of height 1.
* Conv's bias is added after the taps, once per output (`acc + bias`), as a separate pass over the output; in exact mode NaN results
  are canonicalised. ConvTranspose adds its bias last as well.
* ConvTranspose is the matrix product `Wg' * Xg` (rows: output channel, kernel row, kernel column; columns: input position) followed
  by col2im. An output element receives the kernel taps that land on it in increasing (kernel row, kernel column) order, starting
  from +0; the sum over input channels is inside the matrix product (`kernels-gemm.md`). A constant weight is rearranged once per plan.

## Pooling

All loops run window taps in depth, height, width order, one scalar at a time. One, two or three spatial axes are lifted to three.

* **MaxPool and NaN (settled against ONNX Runtime; partly open).** Our rule is IEEE `maxNum`: NaN taps are ignored unless every tap
  inside the input is NaN, in which case the result is NaN. The first tap inside the input that is not NaN starts the maximum, a later
  tap replaces it only if strictly greater, so the first of equal taps wins and decides the index. `-inf` data gives `-inf`.
  ONNX Runtime has no single rule. Probes with one window of the values `1..k` and a NaN at each position in turn (54 probes over
  1-D windows of 2, 3 and 5 and 2-D windows 1x2, 1x3, 1x4, 2x2, 2x3, 3x3, 4x4): the result is NaN for exactly one position per
  window shape from 2x2 up (and for 1-D 5), at a position that changes with the shape, and never otherwise; `[nan, 1, 2, 3]` with
  kernel 1x2 and stride 2 gives `[nan, 3]` while `[nan, 2]` as a single window gives `2`. The output depends on the window's size
  and position, not on a rule we could state, so exact agreement on NaN inputs is not a goal. The rule above agreed with ONNX
  Runtime in 48 of the 54 probes (propagating NaN would agree in 6), and it is the same as oneocr's. Generated comparison
  cases therefore carry infinities and `-0` but no NaN for MaxPool.
* **GlobalMaxPool:** the maximum starting from the lowest float and replaced by strictly greater values, so NaNs are ignored and a
  plane of NaNs or of `-inf` gives `-3.4028235e38`. That is what ONNX Runtime 1.31.1 does on x64; on arm64 it propagates a NaN
  instead (the same generated case gave NaN there), so the NaN behaviour of its max operators is a property of the CPU it runs on.
* **MaxPool Indices:** the flat index of the chosen element over the whole input tensor (`((n * C + c) * D + d) * H * W + ...` for
  `storage_order` 0); for `storage_order` 1 each plane is flattened with the first spatial axis fastest (for 2-D: `c * plane + w * H + h`).
  Confirmed on a 2x2x3x3 input.
* **AveragePool divisor:** the window's taps inside the input, summed in order from +0 and divided (not multiplied by a reciprocal).
  With `count_include_pad` the divisor also counts taps in the declared padding, but never taps beyond it (the overhang that
  `ceil_mode` can create). Confirmed by ONNX Runtime on 5x5 inputs with kernel 3, stride 2, pad 1, `ceil_mode` and dilation 2.
* **GlobalAveragePool:** exact mode: each plane's sum in index order from +0 divided by the plane size. Fast mode: four independent
  4-lane accumulators over 16 elements per step, combined as (a0 + a1) + (a2 + a3), then lanes 0 + 1 + 2 + 3, then the remaining elements one
  at a time, divided by the plane size (the order depends on the plane size only, and is the same on every instruction set; planes
  under 16 elements are summed in index order).
* ONNX Runtime refuses pads that are not smaller than the kernel; we do not (a window with no tap inside the input gives the type's
  lowest value for MaxPool and NaN for AveragePool).

## BatchNormalization

Per channel `a = scale / sqrt(var + epsilon)`, `b = bias - mean * a`, then `y = x * a + b`; each operation rounded on its own, no
fused multiply-add. This differs from `(x - mean) / sqrt(var + epsilon) * scale + bias` by a few rounding errors, within the 1e-4
scaled metric of `oracles.md`. The operator may run in place.

## Resize

Axes are resampled one at a time (shrinking axes first, then growing ones, each in axis order), through per-output tap tables; the
result of a separable interpolation equals the multi-dimensional one up to rounding. Coordinates are computed in float32:

* output size per axis: `floor(in * scale)` for scales, `size` for sizes; `keep_aspect_ratio_policy` `not_larger` / `not_smaller`
  takes the smaller / larger of `size / in` over the listed axes as the scale and rounds `in * scale` half away from zero.
* input coordinate `x` of output position `j` (scale `s`, `in`, `out`): `half_pixel` `(j + 0.5)/s - 0.5`; `pytorch_half_pixel` the same
  unless `out == 1` (then 0); `align_corners` `j * (in - 1) / (out - 1)` with the rounded-down output size (0 if `out == 1`). **Open:** the backend cases `downsample_scales_{linear,cubic}_align_corners` expect the unrounded size `in * s` in the divisor (scales 0.6 on 4 elements); ONNX Runtime 1.31.1 uses the output size, as here, and so do we, so those two cases fail by design; `asymmetric` `j / s`;
  `half_pixel_symmetric` `offset + (j + 0.5)/s - 0.5` with `offset = in/2 * (1 - out / (s * in))`; `tf_crop_and_resize` from `roi`.
  Opset 10 and Upsample are `asymmetric` with nearest mode `floor`.
* nearest: `floor`, `ceil`, `round_prefer_floor` (`x - floor(x) > 0.5` rounds up), `round_prefer_ceil` (`>= 0.5` rounds up); the index
  is clamped to `[0, in - 1]`. Nearest copies elements, so any value (NaN, infinities) moves unchanged.
* linear: `x` is clamped to `[0, in - 1]`, taps `floor(x)`, `floor(x) + 1` (clamped) with weights `1 - t`, `t`.
* cubic (Keys 1981): taps `floor(x) - 1 .. floor(x) + 2`; with `exclude_outside` taps outside the input get weight 0 and the others
  are divided by their sum, otherwise tap indices are clamped.
* `tf_crop_and_resize`: output positions whose coordinate lies outside `[0, in - 1]` on any axis are `extrapolation_value`.
* A tap with weight exactly zero is left out of the sum, so an exactly hit sample next to an infinity or NaN stays finite. ONNX
  Runtime's own results for interpolation near infinities or NaN vary from one case to the next (it gave `-inf` where multiplying by
  the zero weight gives NaN, and NaN where leaving the tap out gives a number), so generated linear and cubic comparison cases contain
  only finite data.
* Scales or sizes that are not constants make the node shape at run time; `roi` is read only for `tf_crop_and_resize`.
* An axis with scale exactly 1 and equal sizes is skipped (identity), except under `tf_crop_and_resize`.
