using System;
using System.Runtime.CompilerServices;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// The cheaper exponential used by the fast attention and softmax kernels. It trades range and accuracy for speed: a degree-5
/// fit for the fractional part, one scaling multiply, and inputs below -87.3 flushed to +0 (the result there is below the smallest
/// normal float). It is meant for softmax-like inputs (x &lt;= 0 after the row maximum is subtracted) and is accurate to about two
/// ulp over [-87.3, 88.3]. NaN inputs give NaN with an unspecified payload. See <c>docs/exact-transcendentals.md</c>.
/// </summary>
internal static class FastMath
{
    private const float RoundMagic = 12582912f;

    /// <summary>e^x for x in about [-87.3, 88.3]; +0 below that range, a large finite value (about 2.4e38) above it.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static TV Exp<TV, TO>(TV x)
        where TV : struct
        where TO : IVecOps<TV>
    {
        TV magic = TO.Splat(RoundMagic);
        TV xc = TO.Select(TO.GreaterThan(x, TO.Splat(88.3f)), TO.Splat(88.3f), x);
        xc = TO.Select(TO.LessThan(xc, TO.Splat(-87.3f)), TO.Splat(-87.3f), xc);

        // x = n*ln2 + r, |r| <= ln2/2 (Cody & Waite, as in the exact version).
        TV t = TO.Accumulate(magic, xc, TO.Splat(1.44269502f));
        TV nf = TO.Sub(t, magic);
        TV r = TO.Accumulate(xc, nf, TO.Splat(-0.693145751953125f));
        r = TO.Accumulate(r, nf, TO.Splat(-1.4286068e-6f));

        // exp(r) = 1 + (r + r*r*Q(r)), Q a degree-3 minimax fit; relative error 1.05e-7 before rounding.
        TV q = TO.Splat(0.008312525f);
        q = TO.Accumulate(TO.Splat(0.041890115f), q, r);
        q = TO.Accumulate(TO.Splat(0.16667114f), q, r);
        q = TO.Accumulate(TO.Splat(0.4999923f), q, r);
        TV p = TO.Add(TO.Splat(1f), TO.Accumulate(r, TO.Mul(r, r), q));

        // 2^n built in the exponent field; n is in [-126, 127] after the clamp.
        TV n = TO.SubInt(t, magic);
        TV scale = TO.ShiftLeft(TO.AddInt(n, TO.Splat(BitConverter.Int32BitsToSingle(127))), 23);
        TV res = TO.Mul(p, scale);
        return TO.Select(TO.LessThan(x, TO.Splat(-87.3f)), TO.Zero, res);
    }
}
