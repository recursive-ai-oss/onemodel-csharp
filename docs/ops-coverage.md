# Operators added for backend-test coverage

Semantics the ONNX text leaves open, as settled by ONNX's backend-test data or, where that is silent, by ONNX Runtime 1.31.1's output on crafted inputs. Operators are checked against the backend tests (`docs/conformance.md`) and against ONNX Runtime on random inputs (`tools/ops_coverage_oracle.py`, 340 cases, run by `CoverageOracleTests` in both modes and on every kernel path).

## Exactness

An operator is exact (results bit-identical on every machine) when it moves elements, uses only `+ - * /` and square root in a fixed order, or uses the library's own `exp`/`log` polynomials (`docs/exact-transcendentals.md`). Not exact, refused in exact mode with a message naming the operator: the trigonometric and hyperbolic functions and their inverses, LRN, LpPool, Attention (opset 23 and later), DFT, STFT, the window functions, MelWeightMatrix, GridSample, RoiAlign, SoftmaxCrossEntropyLoss, and the float64 log reductions.

## Settled semantics

- **Attention (opset 23, 24).** `softcap` is applied before the mask; `is_causal` lets query i see keys j <= i + past_sequence_length; an attention mask shorter than the key length masks the remainder.
- **MaxUnpool.** With `output_shape`, an index is a position in the shape the pooling implies and is re-based into `output_shape` (this is what the ONNX test data expects; ONNX Runtime reads it as a flat position in `output_shape` and differs, so there is no oracle case for it). An index outside either shape is an error.
- **RoiAlign.** `coordinate_transformation_mode` defaults to `half_pixel` from opset 16 and `output_half_pixel` before. `sampling_ratio` 0 uses ceil(region size / bins). In max mode each of the four weighted neighbours competes on its own and the largest wins (the backend cases expect this).
- **LpNormalization, MeanVarianceNormalization.** A zero norm gives zeros.
- **Unique.** Sorted order compares slices element by element; NaN is never equal to NaN; the three index outputs are optional.
- **Resize with `antialias` = 1** (linear and cubic). On an axis whose scale is below 1 the filter's reach grows to 1/scale times its half-width (1 for linear, 2 for cubic). Every integer position i with a non-zero weight filter((i - x) * scale) around the sample position x gets that weight; positions outside the input read the nearest edge sample, so their weights add to it (with `exclude_outside` = 1 they are dropped); weights are divided by their sum. Axes that grow, and nearest mode, are unchanged. This is ONNX Runtime's behaviour, found by reading the weight matrix off impulse inputs. Fast and exact give the same bits.
- **ConvTranspose with three spatial axes.** Direct loops in a fixed order (input element, output channel, kernel tap), no matrix product; exact.
- **MelWeightMatrix.** num_mel_bins + 2 edge points evenly spaced in mel (2595 log10(1 + f / 700)) between the lower and upper edge, converted back to hertz, then to bins as floor((dft_length + 1) f / sample_rate), in double precision; filter i rises from edge i to edge i + 1 and falls to edge i + 2, peak 1. Only float and double outputs.
- **Squeeze, Unsqueeze.** A scalar `axes` input is accepted as one axis.

## Not covered

Element types float8, float4, 4-bit and 2-bit integers, complex and string; the `ai.onnx.ml` and training domains; DeformConv, ImageDecoder; blocked quantization; Bernoulli and the Random* family; Dropout in training mode; BatchNormalization in training mode; LinearAttention, CausalConvWithState and FlexAttention (opset 27 and preview); float64 `Exp`, `Log` and `Softmax`. Models below opset 7 are outside the import range.
