using System;
using System.Numerics;
using System.Runtime.Intrinsics;

namespace RecursiveAI.OneModel.Kernels.Exact;

// More binary element-wise operations for the shared broadcast loops (see ExactElementwise.cs for the contract). Each does one
// correctly rounded IEEE operation or a comparison and select per element, so scalar and vector lanes agree bit for bit.
// Integer rules where C# would throw: x / 0 and x % 0 give 0; MinValue / -1 wraps to MinValue and MinValue % -1 is 0.

internal readonly struct SubArith : IBinaryOp
{
    public static float Apply(float a, float b) => a - b;

    public static Vector128<float> Apply(Vector128<float> a, Vector128<float> b) => a - b;

    public static Vector256<float> Apply(Vector256<float> a, Vector256<float> b) => a - b;

    public static T Apply<T>(T a, T b) where T : unmanaged, INumber<T> => unchecked(a - b);
}

internal readonly struct DivArith : IBinaryOp
{
    public static float Apply(float a, float b) => a / b;

    public static Vector128<float> Apply(Vector128<float> a, Vector128<float> b) => a / b;

    public static Vector256<float> Apply(Vector256<float> a, Vector256<float> b) => a / b;

    public static T Apply<T>(T a, T b) where T : unmanaged, INumber<T>
    {
        if (NumKind.IsFloat<T>())
        {
            return a / b;
        }

        if (b == T.Zero)
        {
            return T.Zero;
        }

        // Signed -1 divisor: negate instead of dividing, because MinValue / -1 throws in C#.
        T minusOne = unchecked(T.Zero - T.One);
        if (T.IsNegative(minusOne) && b == minusOne)
        {
            return unchecked(T.Zero - a);
        }

        return a / b;
    }
}

/// <summary>Maximum. A NaN in either operand gives NaN (the first operand's if both are).</summary>
internal readonly struct MaxArith : IBinaryOp
{
    public static float Apply(float a, float b) => float.IsNaN(a) ? a : float.IsNaN(b) ? b : a > b ? a : b;

    public static Vector128<float> Apply(Vector128<float> a, Vector128<float> b)
    {
        Vector128<float> r = Vector128.ConditionalSelect(Vector128.GreaterThan(a, b), a, b);
        r = Vector128.ConditionalSelect(Vector128.Equals(b, b), r, b);
        return Vector128.ConditionalSelect(Vector128.Equals(a, a), r, a);
    }

    public static Vector256<float> Apply(Vector256<float> a, Vector256<float> b)
    {
        Vector256<float> r = Vector256.ConditionalSelect(Vector256.GreaterThan(a, b), a, b);
        r = Vector256.ConditionalSelect(Vector256.Equals(b, b), r, b);
        return Vector256.ConditionalSelect(Vector256.Equals(a, a), r, a);
    }

    public static T Apply<T>(T a, T b) where T : unmanaged, INumber<T> => T.IsNaN(a) ? a : T.IsNaN(b) ? b : a > b ? a : b;
}

/// <summary>Minimum. A NaN in either operand gives NaN (the first operand's if both are).</summary>
internal readonly struct MinArith : IBinaryOp
{
    public static float Apply(float a, float b) => float.IsNaN(a) ? a : float.IsNaN(b) ? b : a < b ? a : b;

    public static Vector128<float> Apply(Vector128<float> a, Vector128<float> b)
    {
        Vector128<float> r = Vector128.ConditionalSelect(Vector128.LessThan(a, b), a, b);
        r = Vector128.ConditionalSelect(Vector128.Equals(b, b), r, b);
        return Vector128.ConditionalSelect(Vector128.Equals(a, a), r, a);
    }

    public static Vector256<float> Apply(Vector256<float> a, Vector256<float> b)
    {
        Vector256<float> r = Vector256.ConditionalSelect(Vector256.LessThan(a, b), a, b);
        r = Vector256.ConditionalSelect(Vector256.Equals(b, b), r, b);
        return Vector256.ConditionalSelect(Vector256.Equals(a, a), r, a);
    }

    public static T Apply<T>(T a, T b) where T : unmanaged, INumber<T> => T.IsNaN(a) ? a : T.IsNaN(b) ? b : a < b ? a : b;
}

/// <summary>Mod with fmod = 1: the result takes the sign of the dividend (C fmod for floats, truncating remainder for integers).</summary>
internal readonly struct FmodArith : IBinaryOp
{
    public static float Apply(float a, float b) => a % b;

    public static Vector128<float> Apply(Vector128<float> a, Vector128<float> b) =>
        Vector128.Create(a[0] % b[0], a[1] % b[1], a[2] % b[2], a[3] % b[3]);

    public static Vector256<float> Apply(Vector256<float> a, Vector256<float> b) =>
        Vector256.Create(Apply(a.GetLower(), b.GetLower()), Apply(a.GetUpper(), b.GetUpper()));

    public static T Apply<T>(T a, T b) where T : unmanaged, INumber<T>
    {
        if (NumKind.IsFloat<T>())
        {
            return a % b;
        }

        T minusOne = unchecked(T.Zero - T.One);
        if (b == T.Zero || (T.IsNegative(minusOne) && b == minusOne))
        {
            return T.Zero;
        }

        return a % b;
    }
}

/// <summary>Mod with fmod = 0, integers only: the result takes the sign of the divisor (floor modulus).</summary>
internal readonly struct FloorModArith : IBinaryOp
{
    public static float Apply(float a, float b) => throw new NotSupportedException("Mod with fmod = 0 is defined for integers only.");

    public static Vector128<float> Apply(Vector128<float> a, Vector128<float> b) => throw new NotSupportedException("Mod with fmod = 0 is defined for integers only.");

    public static Vector256<float> Apply(Vector256<float> a, Vector256<float> b) => throw new NotSupportedException("Mod with fmod = 0 is defined for integers only.");

    public static T Apply<T>(T a, T b) where T : unmanaged, INumber<T>
    {
        T r = FmodArith.Apply(a, b);
        return r != T.Zero && T.IsNegative(r) != T.IsNegative(b) ? unchecked(r + b) : r;
    }
}
