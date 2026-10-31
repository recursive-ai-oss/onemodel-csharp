# Element-wise operators, activations, comparisons, Where and Cast

Normative notes for the operators in `src/OneModel/Graph/Operators/Elementwise/`. Semantics come from the ONNX operator
specification; where the specification leaves a case open, the choice below was settled by running ONNX Runtime 1.31.1 on a crafted
input (cases in `tools/ops_elementwise_oracle.py`, outputs in `testdata/ops-elementwise/cases.json.gz`).

## Operators and modes

"Exact" means the operator has an exact kernel (bit-identical on every machine, `exact-mode.md`). Every operator here except Pow is
exact: each output element is one correctly rounded IEEE operation, a comparison, a bit operation, a documented conversion, or one of
the exact transcendentals of `exact-transcendentals.md`.

| Operator | Opsets | Element types | Mode |
|---|---|---|---|
| Sub, Div | 7+ | f16, f32, f64, i8..i64, u8..u64 | exact |
| Mod | 10+ | fmod=1: f16, f32, f64, all integers. fmod=0 (default): integers only | exact |
| Pow | 7+ | base f32, f64, i32, i64; exponent those or u32, u64 | **not exact** |
| Max, Min | 6+ | all numeric incl. f16 | exact |
| Sum, Mean | 6+ | f16, f32, f64 | exact |
| Exp, Log, Tanh, Sigmoid, Erf | 6+ (Erf 9+) | f32 | exact |
| Gelu | 20+ | f32; `approximate` = `none` or `tanh` | exact |
| LeakyRelu, HardSigmoid, HardSwish | 6+ (HardSwish 14+) | f32 | exact |
| Neg | 6+ | signed numeric | exact |
| Abs, Sign | 6+ (Sign 9+) | all numeric | exact |
| Reciprocal, Sqrt, Floor, Ceil | 6+ | f16, f32, f64 | exact |
| Round | 11+ | f16, f32, f64 | exact |
| Clip | 6+ | all numeric | exact |
| Equal | 7+ | all numeric and bool | exact |
| Less, Greater | 7+ | all numeric | exact |
| LessOrEqual, GreaterOrEqual | 12+ | all numeric | exact |
| Not | 1+ | bool | exact |
| And, Or, Xor | 7+ | bool | exact |
| Where | 9+ | any stored type | exact |
| Cast, CastLike | 6+, 15+ | any stored type to any stored type | exact |

"All numeric" is f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64. Add and Mul take f16, f32, f64 and the integers. Half-precision arithmetic is done in single precision and rounded to nearest even once, which is exact for + - * /.

Not implemented (reported as `Unsupported` when the model is loaded): arithmetic on bf16;
string tensors; the float8 and 4-bit types (Cast to or from them); double-precision Exp, Log, Tanh, Sigmoid, Erf, Gelu, LeakyRelu,
HardSigmoid and HardSwish (they take f32 only); Pow with integer types narrower than 32 bits.
Operators of opsets before 6 (and the broadcast attributes of Add before 7) are not served.

## Broadcasting and in-place use

Binary operators broadcast NumPy style (`Shape.TryBroadcast`); Max, Min, Sum, Mean fold their inputs left to right, so the result
never depends on anything but the input order. Operators whose output has the type and shape of input 0 may overwrite it (the plan
decides); comparisons, Where, Cast, Pow with a different exponent shape, and the variadic operators never do.

## NaN, signed zero and special values

- **NaN results in exact mode** are rewritten to the quiet NaN with no payload and positive sign (0x7FC00000 for f32). Fast mode
  leaves what the hardware produced.
- **Max, Min:** a NaN in either operand gives NaN (the first operand's if both are). When the operands compare equal the second is
  returned, so `Max(-0, +0)` is `+0` and `Max(+0, -0)` is `-0`; ONNX Runtime agrees on the crafted cases.
- **Comparisons:** NaN compares false with everything, including itself (so `Equal(NaN, NaN)` is false and `LessOrEqual` is false).
- **Sign:** -1, 0 or +1; both zeros give `+0` (ONNX Runtime does too); NaN gives NaN.
- **Round:** half to even, keeps the sign of zero (`Round(-0.4)` is `-0`). **Floor, Ceil, Sqrt, Reciprocal:** IEEE.
- **Clip:** `min(max(x, lo), hi)` with the upper bound winning if `lo > hi`. A bound that is omitted is the type's finite extreme,
  not infinity: for floats `Clip(+inf)` with no bounds gives `+MaxValue` (ONNX Runtime does this). NaN passes through. Before opset 11
  the bounds are the attributes `min` and `max` (defaults: lowest and highest finite float); from 11 they are optional scalar inputs.
- **Gelu(-inf):** the library returns `-0` (the limit); ONNX Runtime returns NaN (it computes `-inf * 0`). Not changed: the exact
  functions are specified in `exact-transcendentals.md`.
- **Pow:** a float base uses single-precision `pow` with the exponent converted to float, a double base uses double `pow`, an integer
  base computes in double and converts back, saturating. An exponent of 2 is `x * x`. `Pow(x, 0)` is 1 for every x including NaN, as in C.
  Not exact because the platform `pow` is not bit-reproducible.

## Integers

All integer arithmetic wraps around (two's complement) for Sub, Add, Mul and Neg. `Div` truncates toward zero. Cases that would trap
in C# are defined: `x / 0` and `x % 0` give 0 (ONNX leaves them undefined), `MinValue / -1` wraps to `MinValue` and `MinValue % -1`
is 0. `Mod` with `fmod = 0` takes the sign of the divisor (floor modulus); with `fmod = 1` the sign of the dividend (C `fmod` for
floats). `Mod` with `fmod = 0` on a float type is rejected, as the specification requires `fmod = 1` there.

## Cast

Conversions go through a wide intermediate in chunks of 256: floating-point sources (f32, f64, f16, bf16) through double, which holds
all four exactly; integer and bool sources through a 64-bit integer.

| From | To | Rule |
|---|---|---|
| float | float | round to nearest even (f32 to f16 and bf16 included; overflow gives infinity) |
| float | integer | truncate toward zero, saturate at the target's range, NaN gives 0 |
| float | bool | `x != 0` (NaN is true; -0 is false) |
| integer | integer | wrap modulo 2^n (so `int32 300` to `uint8` is 44 and `-1` to `uint8` is 255) |
| integer | float | round to nearest even; u64 is read as unsigned |
| integer | bool | `x != 0` |
| bool | any | 0 or 1 |

The float-to-integer saturation and NaN rule is the library's choice where C leaves the out-of-range case undefined; the in-range
cases and the integer wrap match ONNX Runtime on all oracle cases. `CastLike` casts to the element type of its second input and never
reads that input's values. Cast to a type the library does not store is an `Unsupported` error naming the ONNX type.

## Function definitions (f32)

The transcendental functions are the ones in `exact-transcendentals.md`, run on the exact lane operations in exact mode and on the
fast lane operations (fused multiply-add where the CPU has it) in fast mode; they agree to a few ulp.

- LeakyRelu: `x < 0 ? alpha * x : x` (default alpha 0.01); NaN passes through.
- HardSigmoid: `clamp(alpha * x + beta, 0, 1)` with a rounded product and then a rounded sum (defaults 0.2 and 0.5).
- HardSwish: `x * clamp(x * 0.16666667f + 0.5f, 0, 1)`.
- Gelu: erf form `0.5 * x * (1 + erf(x / sqrt 2))` or the tanh form, both from `exact-transcendentals.md`. In fast mode the erf form uses the rational variant described there.

Lane widths: the widest group the kernel path allows (512, 256 or 128 bits) runs first and a one-lane instantiation of the same
code finishes the tail, so an element's result does not depend on its position. There is no threading in these kernels yet; they
are memory-bound and a fixed chunk size would be added with the first measurement that wants it.

## Verification

`testdata/ops-elementwise/cases.json.gz` holds 700+ cases (graph, inputs, ONNX Runtime output) over shapes of size 0 and 1, scalars,
broadcasting of rank 0 to 4, odd lengths around the vector widths, NaN, infinities, signed zeros and subnormals. The tests run every
case on every kernel path of the machine in both modes: results must match within a scaled error (largest difference over the largest
magnitude, with that magnitude floored at 1e-3) of 0 for pure IEEE operations and 1e-4 for the transcendentals, and exact-mode bits
must be equal across paths. ONNX Runtime's own Sigmoid and tanh-form Gelu are off by up to 1e-4 relative for inputs below about -10,
where the library is closer to the true value.
