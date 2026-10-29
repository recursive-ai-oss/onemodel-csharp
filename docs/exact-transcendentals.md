# Exact fp32 transcendental functions

Normative. A port that follows this text reproduces the exact results bit for bit. Source: `src/OneModel/Kernels/Exact/Transcendental.cs`
(the algorithms) and `TranscendentalOps.cs` (the lane operations). Coefficients come from `tools/transcendentals/fit.py`
(minimax fits in relative error, Lawson iteration; rounded to fp32 afterwards).

## Rules

* Only IEEE-754 binary32 `+ - * /`, comparisons, bit operations and 32-bit integer add/subtract/shift on the float's bit pattern.
  No fused multiply-add, no reciprocal or square-root estimates, no library math. Every operation rounds to nearest-even, with
  subnormals enabled.
* Every expression below is evaluated exactly as written, left to right, one rounding per operator. `acc + a*b` means a product
  rounding, then a sum rounding. Horner's rule `c_k + (...)*x` runs from the highest coefficient down.
* Special values are selected by comparisons at the end, so the lane that computed garbage on the way (for example a shift of an
  infinity) never shows. Any NaN result is the canonical quiet NaN `0x7FC00000`; NaN payloads and signs of inputs are not kept.
* "Round to nearest integer" is `((x*c) + M) - M` with `M = 12582912 (1.5*2^23)`; the integer is read back as
  `bits(x*c + M) - bits(M)`. It needs |x*c| < 2^22 and is not used outside the thresholds below.
* The same operation sequence runs for scalar, 128-, 256- and 512-bit lanes, so the width never changes a bit.

## exp(x)

1. `t = x*1.44269502 + M`; `n = t - M` (as float); `n_int = bits(t) - bits(M)`.
2. `r = (x - n*0.693145751953125) - n*1.4286068e-6`. (`0.693145751953125` has 14 significant bits, so `n*ln2Hi` is exact for |n| < 256.)
3. `q = ((((0.0013814613*r + 0.00836871)*r + 0.04166839)*r + 0.16666521)*r + 0.49999994)`, written as `c + q*r` per step;
   `p = 1 + (r + (r*r)*q)`.
4. `a = n_int >> 1` (arithmetic), `b = n_int - a`; `result = (p * float(bits((a+127)<<23))) * float(bits((b+127)<<23))`
   (two exact power-of-two scalings, so a subnormal result rounds once).
5. `x > 88.72283172607422` gives `+inf`; `x < -103.97207641601562` gives `+0`; NaN gives the canonical NaN.

`ExpNonPositive` (used by softmax and attention, argument <= 0 or NaN) runs steps 1-3, then `result = p * float(bits((n_int+127)<<23))`,
`+0` for `x < -87.33654`, canonical NaN for NaN. For x in [-87.33654, 0] it equals `exp` bit for bit (tested); below that it flushes
to zero instead of producing subnormals.

## log(x)

1. `x < 1.17549435e-38` (subnormal or zero): `xs = x * 16777216`, exponent correction `-24`; otherwise `xs = x`.
2. `ix = bits(xs) - 0x3F3504F3`; `e = ix >> 23` (arithmetic); `m = float((ix & 0x007FFFFF) + 0x3F3504F3)` in [sqrt(1/2), sqrt(2)).
   `ef = float(e) - 24*[subnormal]`, with `float(e)` formed as `float(e + 0x4B400000) - M`.
3. `f = m - 1` (exact); `s = f/(2+f)`; `z = s*s`; `w = ((((1/11)*z + 1/9)*z + 1/7)*z + 1/5)*z + 1/3` with the fp32 constants
   `0.0909090936, 0.111111112, 0.142857149, 0.200000003, 0.333333343`; `s2 = s+s`; `logm = s2 + s2*(z*w)`.
4. `result = ef*0.693145751953125 + (logm + ef*1.4286068e-6)`.
5. `x > FLT_MAX` gives `+inf`; `x < 0` gives the canonical NaN; `x == 0` (either zero) gives `-inf`; NaN gives the canonical NaN.

## sigmoid(x)

`s = exp(0 - |x|)`; `num = (x >= 0) ? 1 : s`; `result = num / (1 + s)`. NaN gives the canonical NaN.

## tanh(x)

`ax = |x|`, `z = ax*ax`. For `ax < 0.6`: `P = ((((-0.0058951583*z + 0.020795971)*z - 0.05378383)*z + 0.13331924)*z - 0.333333)`
(Horner, `c + P*z` steps), `result = ax + ax*(z*P)`. Otherwise `E = exp(ax+ax)`, `result = 1 - 2/(E+1)`. The sign bit of `x` is OR-ed
in last (so `tanh(-0) = -0`). NaN gives the canonical NaN.

## erf(x), and 1 + erf(x) for the erf form of GELU

`ax = |x|`, `z = ax*ax`.

* `ax < 0.5`: `P = (((((-0.0007692853*z + 0.0052003083)*z - 0.02686312)*z + 0.11283774)*z - 0.37612638)*z + 1.1283792)`; `erf = ax*P`;
  `1 + erf(x) = 1 + (erf OR sign)`.
* `ax >= 0.5`: `t = 1/ax`, `s = t - 1`, `G` = the degree-16 Horner polynomial in `s` with coefficients (highest first)
  `-0.00057763467, 0.0017893383, -0.00010074016, -0.004373198, 0.0034639547, 0.0015023724, -0.0009413088, -0.0018891813, -0.0002586738,
  0.0052144383, -0.0103428485, 0.015788378, -0.018338649, 0.008167262, 0.035529688, -0.1543716, 0.42758358`.
  `x*x` is formed exactly as `z + zLo`: `split = ax*4097`, `hi = split - (split - ax)`, `lo = ax - hi`,
  `zLo = (((hi*hi - z) + 2*(hi*lo)) + lo*lo)`, set to 0 when `ax >= 64`. `expNegZ = exp(0 - z) * (1 - zLo)`; `erfc = expNegZ * (t*G)`;
  `erf = 1 - erfc`; `1 + erf(x) = erfc` for `x < 0`, `2 - erfc` otherwise.

The sign bit is OR-ed onto `erf`. `erf(+-inf) = +-1`; NaN gives the canonical NaN.

## GELU

* erf form: `y = x*0.70710677`; `gelu = (0.5*x) * (1 + erf(y))` with `1 + erf` from the section above (no cancellation for negative `y`).
* tanh form: `u = 0.79788456 * (x + 0.044715*((x*x)*x))`; `gelu = x * sigmoid(u + u)`. This is `0.5x(1+tanh u)` through the exact identity
  `1 + tanh(u) = 2 sigmoid(2u)`.
* Both: `gelu(-inf) = -0`, `gelu(+inf) = +inf`, NaN gives the canonical NaN.

## Measured error

Error is `|got - reference| / spacing`, where the reference is computed in double (`Math.Exp/Log/Tanh`, erf by series and continued
fraction, accurate to about 1e-13 relative) and the spacing is the fp32 spacing at the correctly rounded reference. Sweeps use every
fp32 value in the hot range (stride 1) and every 67th value of the wide range. Produced by
`ONEMODEL_ULP_TABLE=<file> ONEMODEL_ULP_STRIDE=1 ONEMODEL_ULP_THREADS=8` with the `WriteUlpTable` test.
"Fast" is the same code with fused accumulation (and `FastMath.Exp` for `exp`); it is not bit-reproducible across CPUs.

| function | variant | sweep | values | max ulp | at x |
|---|---|---|---|---|---|
| Exp | exact | hot [-16, 16] stride 1 | 2197815297 | 0.976 | -5.8606462 |
| Exp | exact | wide [-110, 90] stride 67 | 33444880 | 0.953 | 15.626762 |
| Exp | fast | hot [-16, 16] stride 1 | 2197815297 | 2.254 | 5.2451344 |
| Exp | fast | wide [-87, 88] stride 67 | 33395972 | 2.232 | 29.654173 |
| Log | exact | hot [0.0625, 16] stride 1 | 67108865 | 1.968 | 1.0038618 |
| Log | exact | wide [0, FLT_MAX] stride 67 | 31926792 | 1.929 | 1.0077533 |
| Log | fast | same as exact (the polynomial term is too small for fusing to matter) | | 1.968 | 1.0038618 |
| Tanh | exact | hot [-10, 10] stride 1 | 2185232385 | 1.332 | -0.6310442 |
| Tanh | exact | wide [-50, 50] stride 67 | 33194474 | 1.315 | -0.6016164 |
| Tanh | fast | same as exact | | 1.332 | -0.6310442 |
| Erf | exact | hot [-6, 6] stride 1 | 2172649473 | 2.762 | -0.47693625 |
| Erf | exact | wide [-10, 10] stride 67 | 32615409 | 2.484 | -0.50197357 |
| Erf | fast | hot [-6, 6] stride 1 | 2172649473 | 2.762 | -0.47693625 |
| Sigmoid | exact | hot [-16, 16] stride 1 | 2197815297 | 2.402 | -4.157294 |
| Sigmoid | exact | wide [-110, 90] stride 67 | 33444880 | 2.264 | -2.7397301 |
| Sigmoid | fast | same as exact | | 2.402 | -4.157294 |
| GeluErf | exact | hot [-3, 10] stride 1 | 2170552321 | 11.943 | -2.8339381 |
| GeluErf | fast | hot [-3, 10] stride 1 | 2170552321 | 11.583 | -2.843721 |
| GeluTanh | exact | hot [-3, 10] stride 1 | 2170552321 | 16.996 | -2.9897327 |
| GeluTanh | fast | same as exact | | 16.996 | -2.9897327 |
| GeluErf | exact | negative tail [-6, -3] stride 1 | 8388609 | 46.145 | -5.962 |
| GeluErf | exact | negative tail [-12, -6] stride 1 | 8388609 | 179.769 | -11.988144 |
| GeluTanh | exact | negative tail [-6, -3] stride 1 | 8388609 | 61.997 | -5.8887057 |
| GeluTanh | exact | negative tail [-12, -6] stride 1 | 8388609 | 218.405 | -9.859064 |
| MathF.Exp / Log / Tanh (platform library, for comparison) | | hot, stride 8 | | 0.502 / 0.817 / 2.168 | |

The GELU errors grow in the negative tail because the relative sensitivity of GELU to a rounding of its argument is about x^2
there: even a perfect evaluation of the fp32 formula from the fp32 input cannot do better. The absolute error stays below
about 2e-7 of the input scale for x >= -3. The `ExactAccuracyIsWithinTheDocumentedBounds` test pins the bounds (exp 1.2, log 2.1, tanh 1.5,
erf 3.0, sigmoid 2.5, GELU erf 13, GELU tanh 18 ulp).

## Fast exponential

`FastMath.Exp` (fast attention and softmax): clamp x to [-87.3, 88.3]; `t = fma(x, 1.44269502, M)`, `n = t - M`,
`r = fma(n, -ln2Hi, x)` then `fma(n, -ln2Lo, r)`; `q` a degree-3 fit (`0.008312525, 0.041890115, 0.16667114, 0.4999923`),
`p = 1 + fma(r*r, q, r)`; scale by one `2^n` built in the exponent field; `+0` for `x < -87.3`. About 2.3 ulp.

## Cross-platform check

`TranscendentalTests.ExactBitsMatchTheGoldenHashes` hashes (SHA-256) the exact outputs of all seven functions over every 4099th fp32
bit pattern plus a list of special values. The same hashes were reproduced on x64 with AVX-512 / AVX2 only / SSE only /
`DOTNET_EnableHWIntrinsic=0`, on arm64 (Apple M2, NEON) and with hardware intrinsics off there, and in a NativeAOT build.
