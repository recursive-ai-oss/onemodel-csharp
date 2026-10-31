using System;
using System.Runtime.CompilerServices;

namespace RecursiveAI.OneModel.Kernels.Exact;

/// <summary>
/// fp32 exp, log, tanh, erf, sigmoid and the two GELU forms, written once over <see cref="IVecOps{TV}"/> so that scalar, 128-,
/// 256- and 512-bit instantiations run the same IEEE operations per lane and return the same bits. The algorithms are specified in
/// <c>docs/exact-transcendentals.md</c>: range reduction, coefficients, special values and measured error. With the exact lane
/// operations (<c>ExactVec*</c>) there is no fused multiply-add and no library math call anywhere on these paths. With the fast
/// lane operations the same code runs with fused accumulation and is accurate to about the same ulp but is not bit-reproducible
/// across CPUs.
/// </summary>
internal static class Transcendental
{
    // 1.5 * 2^23: adding it to a float of magnitude below 2^22 rounds to nearest-even integer, left in the low mantissa bits.
    private const float RoundMagic = 12582912f;
    private const int RoundMagicBits = 0x4B400000;

    // Canonical quiet NaN returned for any NaN input or invalid argument (positive sign, empty payload).
    private const int QuietNaNBits = 0x7FC00000;
    private const int PositiveInfinityBits = 0x7F800000;
    private const int NegativeInfinityBits = unchecked((int)0xFF800000);
    private const int AbsMaskBits = 0x7FFFFFFF;
    private const int SignMaskBits = unchecked((int)0x80000000);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TV B<TV, TO>(int bits)
        where TV : struct
        where TO : IVecOps<TV> => TO.Splat(BitConverter.Int32BitsToSingle(bits));

    /// <summary>Replaces every NaN lane by the canonical quiet NaN so that NaN bits do not depend on how the hardware propagates payloads.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static TV Canonical<TV, TO>(TV r)
        where TV : struct
        where TO : IVecOps<TV> => TO.Select(TO.Equal(r, r), r, B<TV, TO>(QuietNaNBits));

    /// <summary>e^x. Underflows to +0 below -103.97207 (after gradual underflow through the subnormals), overflows to +inf above 88.72283.</summary>
    public static TV Exp<TV, TO>(TV x)
        where TV : struct
        where TO : IVecOps<TV>
    {
        TV magic = TO.Splat(RoundMagic);

        // Range reduction (Cody & Waite): x = n*ln2 + r with n = round(x*log2(e)) and |r| <= ln2/2 (+ a rounding sliver).
        // ln2Hi has 14 significant bits so n*ln2Hi is exact for |n| < 256.
        TV t = TO.Add(TO.Mul(x, TO.Splat(1.44269502f)), magic);
        TV nf = TO.Sub(t, magic);
        TV r = TO.Sub(TO.Sub(x, TO.Mul(nf, TO.Splat(0.693145751953125f))), TO.Mul(nf, TO.Splat(1.4286068e-6f)));

        // exp(r) = 1 + (r + r*r*Q(r)), Q a degree-4 minimax fit (tools/transcendentals/fit.py).
        TV q = TO.Splat(0.0013814613f);
        q = TO.Accumulate(TO.Splat(0.00836871f), q, r);
        q = TO.Accumulate(TO.Splat(0.04166839f), q, r);
        q = TO.Accumulate(TO.Splat(0.16666521f), q, r);
        q = TO.Accumulate(TO.Splat(0.49999994f), q, r);
        TV p = TO.Add(TO.Splat(1f), TO.Accumulate(r, TO.Mul(r, r), q));

        // Scale by 2^n in two exact steps (2^a then 2^b, a = n>>1, b = n-a) so n down to -150 (subnormal results) rounds once.
        TV n = TO.SubInt(t, magic);
        TV a = TO.ShiftRightArithmetic(n, 1);
        TV b = TO.SubInt(n, a);
        TV bias = B<TV, TO>(127);
        TV pa = TO.ShiftLeft(TO.AddInt(a, bias), 23);
        TV pb = TO.ShiftLeft(TO.AddInt(b, bias), 23);
        TV res = TO.Mul(TO.Mul(p, pa), pb);

        res = TO.Select(TO.GreaterThan(x, TO.Splat(88.72283172607422f)), B<TV, TO>(PositiveInfinityBits), res);
        res = TO.Select(TO.LessThan(x, TO.Splat(-103.97207641601562f)), TO.Zero, res);
        return Canonical<TV, TO>(TO.Select(TO.Equal(x, x), res, x));
    }

    /// <summary>
    /// e^x for x &lt;= 0 (the arguments softmax produces after subtracting the row maximum). Returns exactly the bits of
    /// <see cref="Exp{TV, TO}"/> for x in [-87.33654, 0], +0 below (so no subnormal results), and the canonical NaN for NaN. For
    /// x &gt; 0 the result is not e^x; it is still the same on every lane width.
    /// </summary>
    public static TV ExpNonPositive<TV, TO>(TV x)
        where TV : struct
        where TO : IVecOps<TV>
    {
        TV magic = TO.Splat(RoundMagic);
        TV t = TO.Add(TO.Mul(x, TO.Splat(1.44269502f)), magic);
        TV nf = TO.Sub(t, magic);
        TV r = TO.Sub(TO.Sub(x, TO.Mul(nf, TO.Splat(0.693145751953125f))), TO.Mul(nf, TO.Splat(1.4286068e-6f)));
        TV q = TO.Splat(0.0013814613f);
        q = TO.Accumulate(TO.Splat(0.00836871f), q, r);
        q = TO.Accumulate(TO.Splat(0.04166839f), q, r);
        q = TO.Accumulate(TO.Splat(0.16666521f), q, r);
        q = TO.Accumulate(TO.Splat(0.49999994f), q, r);
        TV p = TO.Add(TO.Splat(1f), TO.Accumulate(r, TO.Mul(r, r), q));

        // n is in [-126, 0] here, so 2^n is a normal float built directly in the exponent field, and p * 2^n rounds once.
        TV n = TO.SubInt(t, magic);
        TV scale = TO.ShiftLeft(TO.AddInt(n, B<TV, TO>(127)), 23);
        TV res = TO.Mul(p, scale);
        res = TO.Select(TO.LessThan(x, TO.Splat(-87.33654f)), TO.Zero, res);
        return TO.Select(TO.Equal(x, x), res, B<TV, TO>(QuietNaNBits));
    }

    /// <summary>Natural logarithm. log(+-0) = -inf (as +0 and -0 compare equal), log(x &lt; 0) = NaN, log(+inf) = +inf; subnormal inputs are scaled by 2^24.</summary>
    public static TV Log<TV, TO>(TV x)
        where TV : struct
        where TO : IVecOps<TV>
    {
        TV magic = TO.Splat(RoundMagic);
        TV one = TO.Splat(1f);
        TV sqrtHalfBits = B<TV, TO>(0x3F3504F3);

        TV isSub = TO.LessThan(x, TO.Splat(1.17549435e-38f));
        TV xs = TO.Select(isSub, TO.Mul(x, TO.Splat(16777216f)), x);

        // x = m * 2^e with m in [sqrt(1/2), sqrt(2)): subtract the bits of sqrt(1/2) so the exponent field of the difference is e.
        TV ix = TO.SubInt(xs, sqrtHalfBits);
        TV ei = TO.ShiftRightArithmetic(ix, 23);
        TV m = TO.AddInt(TO.And(ix, B<TV, TO>(0x007FFFFF)), sqrtHalfBits);
        TV ef = TO.Sub(TO.AddInt(ei, B<TV, TO>(RoundMagicBits)), magic);
        ef = TO.Sub(ef, TO.Select(isSub, TO.Splat(24f), TO.Zero));

        // log(m) = 2*atanh(s), s = (m-1)/(m+1) = f/(2+f), atanh(s) = s*(1 + z/3 + z^2/5 + ...), z = s*s <= 0.0295 (Taylor series).
        TV f = TO.Sub(m, one);
        TV s = TO.Div(f, TO.Add(TO.Splat(2f), f));
        TV z = TO.Mul(s, s);
        TV w = TO.Splat(0.0909090936f);
        w = TO.Accumulate(TO.Splat(0.111111112f), w, z);
        w = TO.Accumulate(TO.Splat(0.142857149f), w, z);
        w = TO.Accumulate(TO.Splat(0.200000003f), w, z);
        w = TO.Accumulate(TO.Splat(0.333333343f), w, z);
        TV s2 = TO.Add(s, s);
        TV logm = TO.Accumulate(s2, s2, TO.Mul(z, w));

        // log(x) = e*ln2Hi + (log(m) + e*ln2Lo); e*ln2Hi is exact.
        TV res = TO.Add(TO.Mul(ef, TO.Splat(0.693145751953125f)), TO.Add(logm, TO.Mul(ef, TO.Splat(1.4286068e-6f))));

        res = TO.Select(TO.GreaterThan(x, TO.Splat(float.MaxValue)), B<TV, TO>(PositiveInfinityBits), res);
        res = TO.Select(TO.LessThan(x, TO.Zero), B<TV, TO>(QuietNaNBits), res);
        res = TO.Select(TO.Equal(x, TO.Zero), B<TV, TO>(NegativeInfinityBits), res);
        return Canonical<TV, TO>(TO.Select(TO.Equal(x, x), res, x));
    }

    /// <summary>1 / (1 + e^-x), computed from e^-|x| so it neither overflows nor loses the subnormal tail.</summary>
    public static TV Sigmoid<TV, TO>(TV x)
        where TV : struct
        where TO : IVecOps<TV>
    {
        TV ax = TO.And(x, B<TV, TO>(AbsMaskBits));
        TV s = Exp<TV, TO>(TO.Sub(TO.Zero, ax));
        TV numerator = TO.Select(TO.GreaterThanOrEqual(x, TO.Zero), TO.Splat(1f), s);
        TV res = TO.Div(numerator, TO.Add(TO.Splat(1f), s));
        return Canonical<TV, TO>(TO.Select(TO.Equal(x, x), res, x));
    }

    /// <summary>Hyperbolic tangent. Odd polynomial below |x| = 0.6, 1 - 2/(e^(2|x|)+1) above; the sign of x (including -0) is kept.</summary>
    public static TV Tanh<TV, TO>(TV x)
        where TV : struct
        where TO : IVecOps<TV>
    {
        TV sign = TO.And(x, B<TV, TO>(SignMaskBits));
        TV ax = TO.And(x, B<TV, TO>(AbsMaskBits));
        TV z = TO.Mul(ax, ax);

        // tanh(x) = x + x*z*P(z), P a degree-4 minimax fit on z in [0, 0.36].
        TV p = TO.Splat(-0.0058951583f);
        p = TO.Accumulate(TO.Splat(0.020795971f), p, z);
        p = TO.Accumulate(TO.Splat(-0.05378383f), p, z);
        p = TO.Accumulate(TO.Splat(0.13331924f), p, z);
        p = TO.Accumulate(TO.Splat(-0.333333f), p, z);
        TV small = TO.Add(ax, TO.Mul(ax, TO.Mul(z, p)));

        TV e = Exp<TV, TO>(TO.Add(ax, ax));
        TV large = TO.Sub(TO.Splat(1f), TO.Div(TO.Splat(2f), TO.Add(e, TO.Splat(1f))));

        TV res = TO.Or(TO.Select(TO.LessThan(ax, TO.Splat(0.6f)), small, large), sign);
        return Canonical<TV, TO>(TO.Select(TO.Equal(x, x), res, x));
    }

    /// <summary>
    /// Shared body of erf and the erf-form GELU: erf(x) and 1 + erf(x), the latter without cancellation for negative x.
    /// </summary>
    private static void ErfParts<TV, TO>(TV x, out TV erf, out TV onePlusErf)
        where TV : struct
        where TO : IVecOps<TV>
    {
        TV one = TO.Splat(1f);
        TV sign = TO.And(x, B<TV, TO>(SignMaskBits));
        TV ax = TO.And(x, B<TV, TO>(AbsMaskBits));
        TV z = TO.Mul(ax, ax);

        // |x| < 0.5: erf(x) = x*P(z), P a degree-5 minimax fit on z in [0, 0.25].
        TV p = TO.Splat(-0.0007692853f);
        p = TO.Accumulate(TO.Splat(0.0052003083f), p, z);
        p = TO.Accumulate(TO.Splat(-0.02686312f), p, z);
        p = TO.Accumulate(TO.Splat(0.11283774f), p, z);
        p = TO.Accumulate(TO.Splat(-0.37612638f), p, z);
        p = TO.Accumulate(TO.Splat(1.1283792f), p, z);
        TV small = TO.Mul(ax, p);

        // |x| >= 0.5: erfc(|x|) = e^(-x*x) * t * H(s), t = 1/|x| in (0, 2], s = t - 1, H a degree-16 minimax fit on s in [-1, 1]
        // (the fit variable is 2t/2 - 1; see tools/transcendentals/fit.py). x*x is formed as an exact sum zHi + zLo (Veltkamp
        // split and Dekker's product, no fused operations), because a rounded x*x would put an error of x*x*2^-24 into the
        // exponent, which is many ulp for x above 4.
        TV t = TO.Div(one, ax);
        TV s = TO.Sub(t, one);
        TV g = TO.Splat(-0.00057763467f);
        g = TO.Accumulate(TO.Splat(0.0017893383f), g, s);
        g = TO.Accumulate(TO.Splat(-0.00010074016f), g, s);
        g = TO.Accumulate(TO.Splat(-0.004373198f), g, s);
        g = TO.Accumulate(TO.Splat(0.0034639547f), g, s);
        g = TO.Accumulate(TO.Splat(0.0015023724f), g, s);
        g = TO.Accumulate(TO.Splat(-0.0009413088f), g, s);
        g = TO.Accumulate(TO.Splat(-0.0018891813f), g, s);
        g = TO.Accumulate(TO.Splat(-0.0002586738f), g, s);
        g = TO.Accumulate(TO.Splat(0.0052144383f), g, s);
        g = TO.Accumulate(TO.Splat(-0.0103428485f), g, s);
        g = TO.Accumulate(TO.Splat(0.015788378f), g, s);
        g = TO.Accumulate(TO.Splat(-0.018338649f), g, s);
        g = TO.Accumulate(TO.Splat(0.008167262f), g, s);
        g = TO.Accumulate(TO.Splat(0.035529688f), g, s);
        g = TO.Accumulate(TO.Splat(-0.1543716f), g, s);
        g = TO.Accumulate(TO.Splat(0.42758358f), g, s);
        TV split = TO.Mul(ax, TO.Splat(4097f));
        TV axHi = TO.Sub(split, TO.Sub(split, ax));
        TV axLo = TO.Sub(ax, axHi);
        TV zLo = TO.Add(TO.Add(TO.Sub(TO.Mul(axHi, axHi), z), TO.Mul(TO.Splat(2f), TO.Mul(axHi, axLo))), TO.Mul(axLo, axLo));
        zLo = TO.Select(TO.LessThan(ax, TO.Splat(64f)), zLo, TO.Zero);
        TV expNegZ = TO.Mul(Exp<TV, TO>(TO.Sub(TO.Zero, z)), TO.Sub(one, zLo));
        TV erfc = TO.Mul(expNegZ, TO.Mul(t, g));

        TV isSmall = TO.LessThan(ax, TO.Splat(0.5f));
        TV erfAbs = TO.Select(isSmall, small, TO.Sub(one, erfc));
        TV erfSigned = TO.Or(erfAbs, sign);

        TV negative = TO.LessThan(x, TO.Zero);
        TV opeLarge = TO.Select(negative, erfc, TO.Sub(TO.Splat(2f), erfc));
        TV opeSmall = TO.Add(one, TO.Or(small, sign));

        TV ok = TO.Equal(x, x);
        erf = Canonical<TV, TO>(TO.Select(ok, erfSigned, x));
        onePlusErf = Canonical<TV, TO>(TO.Select(ok, TO.Select(isSmall, opeSmall, opeLarge), x));
    }

    /// <summary>Error function. erf(+-inf) = +-1, erf(+-0) = +-0.</summary>
    public static TV Erf<TV, TO>(TV x)
        where TV : struct
        where TO : IVecOps<TV>
    {
        ErfParts<TV, TO>(x, out TV erf, out _);
        return erf;
    }

    /// <summary>GELU with the erf form: 0.5 * x * (1 + erf(x/sqrt(2))). The factor 1 + erf is formed without cancellation for negative x. gelu(-inf) = -0.</summary>
    public static TV GeluErf<TV, TO>(TV x)
        where TV : struct
        where TO : IVecOps<TV>
    {
        ErfParts<TV, TO>(TO.Mul(x, TO.Splat(0.70710677f)), out _, out TV onePlusErf);
        TV res = TO.Mul(TO.Mul(TO.Splat(0.5f), x), onePlusErf);
        res = TO.Select(TO.Equal(x, B<TV, TO>(NegativeInfinityBits)), B<TV, TO>(SignMaskBits), res);
        return Canonical<TV, TO>(TO.Select(TO.Equal(x, x), res, x));
    }

    /// <summary>
    /// GELU with the erf form for fast mode: 0.5 * x * (1 + erf(x/sqrt(2))) with erf(x/sqrt(2)) = x * P(x^2) / Q(x^2), a degree 5 over
    /// degree 5 rational minimax fit (tools/transcendentals/gelu_rational.py) on |x| &lt;= 5.6 and the limit values beyond: 1 + erf is
    /// 2 within half a unit in the last place for x above 5.6, and gelu(x) is below 1e-7 in magnitude for x below -5.6, where it
    /// returns -0. The fit's error is 2e-8 and the fp32 evaluation's 3.5e-7 (about 6 units in the last place of erf near 1),
    /// which is the accuracy of the separate Erf/Add/Mul operators of a model running in fp32; unlike <see cref="GeluErf"/> it does not
    /// keep the relative accuracy of the negative tail, where the result is below 1e-3 and its error stays below 1e-6.
    /// Costs about a sixth of <see cref="GeluErf"/>: no exponential, one division. Fast instantiations only (it uses multiply-adds).
    /// </summary>
    public static TV GeluErfFast<TV, TO>(TV x)
        where TV : struct
        where TO : IVecOps<TV>
    {
        const float Limit = 5.6f;
        TV sign = TO.And(x, B<TV, TO>(SignMaskBits));
        TV ax = TO.And(x, B<TV, TO>(AbsMaskBits));
        TV c = TO.Select(TO.GreaterThan(ax, TO.Splat(Limit)), TO.Splat(Limit), ax);
        TV z = TO.Mul(c, c);
        TV p = TO.Splat(4.4290683e-08f);
        p = TO.Accumulate(TO.Splat(1.2713091e-05f), p, z);
        p = TO.Accumulate(TO.Splat(0.0003477025f), p, z);
        p = TO.Accumulate(TO.Splat(0.009414525f), p, z);
        p = TO.Accumulate(TO.Splat(0.068752296f), p, z);
        p = TO.Accumulate(TO.Splat(0.79788446f), p, z);
        TV q = TO.Splat(1.1802138e-06f);
        q = TO.Accumulate(TO.Splat(7.510214e-05f), q, z);
        q = TO.Accumulate(TO.Splat(0.001912981f), q, z);
        q = TO.Accumulate(TO.Splat(0.028940111f), q, z);
        q = TO.Accumulate(TO.Splat(0.25283393f), q, z);
        q = TO.Accumulate(TO.Splat(1f), q, z);
        TV erf = TO.Or(TO.Mul(c, TO.Div(p, q)), sign);
        TV res = TO.Mul(TO.Mul(TO.Splat(0.5f), x), TO.Add(TO.Splat(1f), erf));
        res = TO.Select(TO.LessThan(x, TO.Splat(-Limit)), B<TV, TO>(SignMaskBits), res);
        return Canonical<TV, TO>(TO.Select(TO.Equal(x, x), res, x));
    }

    /// <summary>
    /// GELU with the tanh form: 0.5 * x * (1 + tanh(k*(x + 0.044715*x^3))), k = sqrt(2/pi). Computed through the identity
    /// 1 + tanh(u) = 2*sigmoid(2u), which is exact and has no cancellation in the negative tail, so it is x * sigmoid(2u).
    /// </summary>
    public static TV GeluTanh<TV, TO>(TV x)
        where TV : struct
        where TO : IVecOps<TV>
    {
        TV x3 = TO.Mul(TO.Mul(x, x), x);
        TV inner = TO.Add(x, TO.Mul(TO.Splat(0.044715f), x3));
        TV u = TO.Mul(TO.Splat(0.79788456f), inner);
        TV res = TO.Mul(x, Sigmoid<TV, TO>(TO.Add(u, u)));
        res = TO.Select(TO.Equal(x, B<TV, TO>(NegativeInfinityBits)), B<TV, TO>(SignMaskBits), res);
        return Canonical<TV, TO>(TO.Select(TO.Equal(x, x), res, x));
    }
}
