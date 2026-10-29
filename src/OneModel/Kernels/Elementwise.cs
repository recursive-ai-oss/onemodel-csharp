using System;
using System.Numerics;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Kernels;

// Scalar-loop element-wise kernels for the types the vectorized float paths do not cover: comparisons, logic, select, Pow, the
// numeric unary operations, and Cast. Everything here is one IEEE operation, a comparison, a bit operation or a documented
// conversion per element, so it is the same in both modes (exact mode additionally rewrites float NaN results to the canonical NaN).

/// <summary>Type-level facts about the element types of <see cref="INumber{T}"/> kernels.</summary>
internal static class NumKind
{
    /// <summary>True for float, double and Half. (INumberBase.IsInteger tests a value, not the type.)</summary>
    public static bool IsFloat<T>() => typeof(T) == typeof(float) || typeof(T) == typeof(double) || typeof(T) == typeof(Half);

    /// <summary>The most negative finite value of a floating-point type.</summary>
    public static T LowestFinite<T>() where T : unmanaged, INumber<T> =>
        typeof(T) == typeof(float) ? T.CreateTruncating(-float.MaxValue) : typeof(T) == typeof(double) ? T.CreateTruncating(-double.MaxValue) : T.CreateTruncating(-(float)Half.MaxValue);

    /// <summary>The largest finite value of a floating-point type.</summary>
    public static T HighestFinite<T>() where T : unmanaged, INumber<T> =>
        typeof(T) == typeof(float) ? T.CreateTruncating(float.MaxValue) : typeof(T) == typeof(double) ? T.CreateTruncating(double.MaxValue) : T.CreateTruncating((float)Half.MaxValue);

    private const long QuietNaN64 = 0x7FF8000000000000L;

    /// <summary>The canonical quiet NaN of the type (0x7FC00000 for float).</summary>
    public static T CanonicalNaN<T>() where T : unmanaged, INumber<T> => T.CreateTruncating(BitConverter.Int64BitsToDouble(QuietNaN64));
}

/// <summary>Called with the concrete element type of a numeric tensor.</summary>
internal interface INumberVisitor
{
    void Visit<T>() where T : unmanaged, INumber<T>;
}

/// <summary>Called with the concrete element type of a floating-point tensor.</summary>
internal interface IFloatVisitor
{
    void Visit<T>() where T : unmanaged, IFloatingPointIeee754<T>;
}

/// <summary>Called with the concrete element type of any tensor.</summary>
internal interface ITypeVisitor
{
    void Visit<T>() where T : unmanaged;
}

/// <summary>Turns a run-time element type into a compile-time one without reflection: every case is a static instantiation.</summary>
internal static class TypeDispatch
{
    public static bool IsNumber(DType t) =>
        t is DType.F32 or DType.F64 or DType.F16 or DType.I8 or DType.I16 or DType.I32 or DType.I64 or DType.U8 or DType.U16 or DType.U32 or DType.U64;

    public static bool IsFloat(DType t) => t is DType.F32 or DType.F64 or DType.F16;

    public static bool IsSigned(DType t) => t is DType.F32 or DType.F64 or DType.F16 or DType.I8 or DType.I16 or DType.I32 or DType.I64;

    public static void Number<TV>(DType t, ref TV v) where TV : struct, INumberVisitor
    {
        switch (t)
        {
            case DType.F32: v.Visit<float>(); break;
            case DType.F64: v.Visit<double>(); break;
            case DType.F16: v.Visit<Half>(); break;
            case DType.I8: v.Visit<sbyte>(); break;
            case DType.I16: v.Visit<short>(); break;
            case DType.I32: v.Visit<int>(); break;
            case DType.I64: v.Visit<long>(); break;
            case DType.U8: v.Visit<byte>(); break;
            case DType.U16: v.Visit<ushort>(); break;
            case DType.U32: v.Visit<uint>(); break;
            case DType.U64: v.Visit<ulong>(); break;
            default: throw new InvalidOperationException($"Element type {t} is not numeric.");
        }
    }

    public static void Float<TV>(DType t, ref TV v) where TV : struct, IFloatVisitor
    {
        switch (t)
        {
            case DType.F32: v.Visit<float>(); break;
            case DType.F64: v.Visit<double>(); break;
            case DType.F16: v.Visit<Half>(); break;
            default: throw new InvalidOperationException($"Element type {t} is not floating point.");
        }
    }

    public static void Any<TV>(DType t, ref TV v) where TV : struct, ITypeVisitor
    {
        switch (t)
        {
            case DType.F32: v.Visit<float>(); break;
            case DType.F64: v.Visit<double>(); break;
            case DType.F16: v.Visit<Half>(); break;
            case DType.BF16: v.Visit<BFloat16>(); break;
            case DType.I8: v.Visit<sbyte>(); break;
            case DType.I16: v.Visit<short>(); break;
            case DType.I32: v.Visit<int>(); break;
            case DType.I64: v.Visit<long>(); break;
            case DType.U8: v.Visit<byte>(); break;
            case DType.U16: v.Visit<ushort>(); break;
            case DType.U32: v.Visit<uint>(); break;
            case DType.U64: v.Visit<ulong>(); break;
            case DType.Bool: v.Visit<bool>(); break;
            default: throw new InvalidOperationException($"Element type {t} is not stored.");
        }
    }
}

// ---- comparisons and logic -------------------------------------------------------------------------------------------------

internal interface ICompare
{
    static abstract bool Test<T>(T a, T b) where T : unmanaged, INumber<T>;
}

internal readonly struct EqualCmp : ICompare
{
    public static bool Test<T>(T a, T b) where T : unmanaged, INumber<T> => a == b;
}

internal readonly struct LessCmp : ICompare
{
    public static bool Test<T>(T a, T b) where T : unmanaged, INumber<T> => a < b;
}

internal readonly struct GreaterCmp : ICompare
{
    public static bool Test<T>(T a, T b) where T : unmanaged, INumber<T> => a > b;
}

internal readonly struct LessOrEqualCmp : ICompare
{
    public static bool Test<T>(T a, T b) where T : unmanaged, INumber<T> => a <= b;
}

internal readonly struct GreaterOrEqualCmp : ICompare
{
    public static bool Test<T>(T a, T b) where T : unmanaged, INumber<T> => a >= b;
}

/// <summary>One row of a comparison: numbers in, bool out. NaN compares false with everything (Equal, Less, ...) as in IEEE.</summary>
internal readonly struct CompareRows<T, TCmp> : IRowKernel<T, T, bool> where T : unmanaged, INumber<T> where TCmp : ICompare
{
    public static void Row(KernelPath path, bool canonicalNaN, ReadOnlySpan<T> a, int sa, ReadOnlySpan<T> b, int sb, Span<bool> c, int n)
    {
        for (int i = 0; i < n; i++)
        {
            c[i] = TCmp.Test(a[sa == 0 ? 0 : i], b[sb == 0 ? 0 : i]);
        }
    }
}

internal interface IBoolOp
{
    static abstract bool Apply(bool a, bool b);
}

internal readonly struct AndBool : IBoolOp
{
    public static bool Apply(bool a, bool b) => a & b;
}

internal readonly struct OrBool : IBoolOp
{
    public static bool Apply(bool a, bool b) => a | b;
}

internal readonly struct XorBool : IBoolOp
{
    public static bool Apply(bool a, bool b) => a ^ b;
}

internal readonly struct EqualBool : IBoolOp
{
    public static bool Apply(bool a, bool b) => a == b;
}

internal readonly struct BoolRows<TOp> : IRowKernel<bool, bool, bool> where TOp : IBoolOp
{
    public static void Row(KernelPath path, bool canonicalNaN, ReadOnlySpan<bool> a, int sa, ReadOnlySpan<bool> b, int sb, Span<bool> c, int n)
    {
        for (int i = 0; i < n; i++)
        {
            c[i] = TOp.Apply(a[sa == 0 ? 0 : i], b[sb == 0 ? 0 : i]);
        }
    }
}

// ---- Pow -------------------------------------------------------------------------------------------------------------------

/// <summary>
/// One row of Pow with base type TX and exponent type TY. A float base uses single-precision pow with the exponent converted to
/// float, a double base uses double pow, an integer base computes in double and converts back. Exponent 2 is x * x, which is what a
/// correctly rounded pow gives. Not an exact kernel: the library pow is not bit-reproducible across platforms.
/// </summary>
internal readonly struct PowRows<TX, TY> : IRowKernel<TX, TY, TX> where TX : unmanaged, INumber<TX> where TY : unmanaged, INumber<TY>
{
    public static void Row(KernelPath path, bool canonicalNaN, ReadOnlySpan<TX> a, int sa, ReadOnlySpan<TY> b, int sb, Span<TX> c, int n)
    {
        for (int i = 0; i < n; i++)
        {
            TX x = a[sa == 0 ? 0 : i];
            TY y = b[sb == 0 ? 0 : i];
            if (typeof(TX) == typeof(float))
            {
                float fx = float.CreateTruncating(x), fy = float.CreateTruncating(y);
                c[i] = TX.CreateTruncating(fy == 2f ? fx * fx : MathF.Pow(fx, fy));
            }
            else if (typeof(TX) == typeof(double))
            {
                double dx = double.CreateTruncating(x), dy = double.CreateTruncating(y);
                c[i] = TX.CreateTruncating(dy == 2.0 ? dx * dx : Math.Pow(dx, dy));
            }
            else
            {
                // Saturating, so an overflowing power does not depend on the platform's conversion of out-of-range doubles.
                c[i] = TX.CreateSaturating(Math.Pow(double.CreateTruncating(x), double.CreateTruncating(y)));
            }
        }
    }
}

// ---- unary numeric ---------------------------------------------------------------------------------------------------------

internal interface INumberUnary
{
    static abstract T Apply<T>(T x) where T : unmanaged, INumber<T>;
}

internal readonly struct NegUnary : INumberUnary
{
    public static T Apply<T>(T x) where T : unmanaged, INumber<T> => unchecked(-x);
}

internal readonly struct AbsUnary : INumberUnary
{
    public static T Apply<T>(T x) where T : unmanaged, INumber<T> => T.IsNegative(x) ? unchecked(-x) : (T.IsNaN(x) ? x : T.Abs(x));
}

internal readonly struct SignUnary : INumberUnary
{
    // -1, 0 or 1; both zeros give +0 (as ONNX Runtime does) and NaN comes back as it is. Math.Sign would throw on NaN.
    public static T Apply<T>(T x) where T : unmanaged, INumber<T> =>
        x > T.Zero ? T.One : (x < T.Zero ? unchecked(T.Zero - T.One) : (x == T.Zero ? T.Zero : x));
}

internal interface IFloatScalarUnary
{
    static abstract T Apply<T>(T x) where T : unmanaged, IFloatingPointIeee754<T>;
}

internal readonly struct ReciprocalUnary : IFloatScalarUnary
{
    public static T Apply<T>(T x) where T : unmanaged, IFloatingPointIeee754<T> => T.One / x;
}

internal readonly struct SqrtUnary : IFloatScalarUnary
{
    public static T Apply<T>(T x) where T : unmanaged, IFloatingPointIeee754<T> => T.Sqrt(x);
}

internal readonly struct FloorUnary : IFloatScalarUnary
{
    public static T Apply<T>(T x) where T : unmanaged, IFloatingPointIeee754<T> => T.Floor(x);
}

internal readonly struct CeilUnary : IFloatScalarUnary
{
    public static T Apply<T>(T x) where T : unmanaged, IFloatingPointIeee754<T> => T.Ceiling(x);
}

/// <summary>Rounds half to even, as ONNX Round defines it; -0.4 gives -0.</summary>
internal readonly struct RoundUnary : IFloatScalarUnary
{
    public static T Apply<T>(T x) where T : unmanaged, IFloatingPointIeee754<T> => T.Round(x, MidpointRounding.ToEven);
}

internal static class UnaryLoops
{
    /// <summary>y[i] = op(x[i]). y may be the same memory as x. NaN results become the canonical NaN when asked.</summary>
    public static void Number<T, TOp>(bool canonicalNaN, ReadOnlySpan<T> x, Span<T> y) where T : unmanaged, INumber<T> where TOp : INumberUnary
    {
        for (int i = 0; i < x.Length; i++)
        {
            T r = TOp.Apply(x[i]);
            y[i] = canonicalNaN && T.IsNaN(r) ? NumKind.CanonicalNaN<T>() : r;
        }
    }

    public static void Float<T, TOp>(bool canonicalNaN, ReadOnlySpan<T> x, Span<T> y) where T : unmanaged, IFloatingPointIeee754<T> where TOp : IFloatScalarUnary
    {
        for (int i = 0; i < x.Length; i++)
        {
            T r = TOp.Apply(x[i]);
            y[i] = canonicalNaN && T.IsNaN(r) ? NumKind.CanonicalNaN<T>() : r;
        }
    }

    /// <summary>
    /// y = clamp(x, lo, hi) with lo and hi each optional. A NaN input stays NaN. Callers guarantee lo &lt;= hi or accept that the
    /// upper bound wins, as the ONNX definition min(max(x, lo), hi) does.
    /// </summary>
    public static void Clip<T>(ReadOnlySpan<T> x, Span<T> y, bool hasLo, T lo, bool hasHi, T hi) where T : unmanaged, INumber<T>
    {
        for (int i = 0; i < x.Length; i++)
        {
            T v = x[i];
            if (!T.IsNaN(v))
            {
                if (hasLo && v < lo)
                {
                    v = lo;
                }

                if (hasHi && v > hi)
                {
                    v = hi;
                }
            }

            y[i] = v;
        }
    }
}

// ---- Where -----------------------------------------------------------------------------------------------------------------

internal static class SelectKernel
{
    /// <summary>
    /// o = c ? x : y with all four shapes broadcast together (o has the broadcast shape). Operands are contiguous. o must not share
    /// memory with an input.
    /// </summary>
    public static void Where<T>(Shape so, Shape sc, Shape sx, Shape sy, ReadOnlySpan<bool> c, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> o)
        where T : unmanaged
    {
        int total = so.Length;
        if (total == 0)
        {
            return;
        }

        if (c.Length < sc.Length || x.Length < sx.Length || y.Length < sy.Length || o.Length < total)
        {
            throw new ArgumentException("An operand is shorter than its shape.");
        }

        int rank = so.Rank;
        if (rank == 0)
        {
            o[0] = c[0] ? x[0] : y[0];
            return;
        }

        Span<int> size = stackalloc int[Shape.MaxRank];
        Span<int> tc = stackalloc int[Shape.MaxRank];
        Span<int> tx = stackalloc int[Shape.MaxRank];
        Span<int> ty = stackalloc int[Shape.MaxRank];
        int accC = 1, accX = 1, accY = 1;
        for (int d = rank - 1; d >= 0; d--)
        {
            size[d] = so[d];
            int dc = d - (rank - sc.Rank), dx = d - (rank - sx.Rank), dy = d - (rank - sy.Rank);
            int ec = dc >= 0 ? sc[dc] : 1, ex = dx >= 0 ? sx[dx] : 1, ey = dy >= 0 ? sy[dy] : 1;
            tc[d] = ec == 1 ? 0 : accC;
            tx[d] = ex == 1 ? 0 : accX;
            ty[d] = ey == 1 ? 0 : accY;
            accC *= ec;
            accX *= ex;
            accY *= ey;
        }

        int inner = size[rank - 1];
        int ic = tc[rank - 1], ix = tx[rank - 1], iy = ty[rank - 1];
        long outer = total / inner;
        Span<int> idx = stackalloc int[Shape.MaxRank];
        int pc = 0, px = 0, py = 0;
        for (long row = 0; row < outer; row++)
        {
            int po = (int)(row * inner);
            for (int j = 0; j < inner; j++)
            {
                o[po + j] = c[pc + j * ic] ? x[px + j * ix] : y[py + j * iy];
            }

            for (int d = rank - 2; d >= 0; d--)
            {
                idx[d]++;
                pc += tc[d];
                px += tx[d];
                py += ty[d];
                if (idx[d] < size[d])
                {
                    break;
                }

                pc -= tc[d] * size[d];
                px -= tx[d] * size[d];
                py -= ty[d] * size[d];
                idx[d] = 0;
            }
        }
    }
}

// ---- Cast ------------------------------------------------------------------------------------------------------------------

/// <summary>
/// Cast between the stored element types, in chunks through a wide intermediate. Floating-point sources (f32, f64, f16, bf16) pass
/// through double (exact for all four); integer and bool sources pass through a 64-bit integer, so no precision is lost.
/// Rules, which follow C's conversions where those are defined (docs/ops-elementwise.md):
/// float to float rounds to nearest even; float to integer truncates toward zero and saturates at the target's range, with NaN
/// giving 0; integer to integer wraps modulo 2^n; integer to float rounds to nearest even; anything to bool is "not equal to zero"
/// (NaN included); bool is 0 or 1.
/// </summary>
internal static class CastKernel
{
    private const int Chunk = 256;

    public static void Run(Tensor src, Tensor dst)
    {
        int n = src.Length;
        if (dst.Length != n)
        {
            throw new ArgumentException("Source and destination differ in length.");
        }

        if (n == 0)
        {
            return;
        }

        DType s = src.DType, d = dst.DType;
        if (s == d)
        {
            dst.CopyFrom(src);
            return;
        }

        Span<double> wide = stackalloc double[Chunk];
        Span<long> whole = stackalloc long[Chunk];
        bool fromFloat = s is DType.F32 or DType.F64 or DType.F16 or DType.BF16;
        for (int i = 0; i < n; i += Chunk)
        {
            int m = Math.Min(Chunk, n - i);
            if (fromFloat)
            {
                ReadWide(src, s, i, wide.Slice(0, m));
                WriteFromWide(dst, d, i, wide.Slice(0, m));
            }
            else
            {
                ReadWhole(src, s, i, whole.Slice(0, m));
                WriteFromWhole(dst, d, i, whole.Slice(0, m), s == DType.U64);
            }
        }
    }

    private static void ReadWide(Tensor src, DType s, int at, Span<double> w)
    {
        switch (s)
        {
            case DType.F32: { ReadOnlySpan<float> x = src.Data<float>().Slice(at, w.Length); for (int j = 0; j < w.Length; j++) { w[j] = x[j]; } break; }
            case DType.F64: src.Data<double>().Slice(at, w.Length).CopyTo(w); break;
            case DType.F16: { ReadOnlySpan<Half> x = src.Data<Half>().Slice(at, w.Length); for (int j = 0; j < w.Length; j++) { w[j] = (double)x[j]; } break; }
            case DType.BF16: { ReadOnlySpan<BFloat16> x = src.Data<BFloat16>().Slice(at, w.Length); for (int j = 0; j < w.Length; j++) { w[j] = (float)x[j]; } break; }
            default: throw new InvalidOperationException($"{s} is not a floating-point type.");
        }
    }

    private static void ReadWhole(Tensor src, DType s, int at, Span<long> w)
    {
        switch (s)
        {
            case DType.I8: Widen<sbyte>(src, at, w); break;
            case DType.I16: Widen<short>(src, at, w); break;
            case DType.I32: Widen<int>(src, at, w); break;
            case DType.I64: src.Data<long>().Slice(at, w.Length).CopyTo(w); break;
            case DType.U8: Widen<byte>(src, at, w); break;
            case DType.U16: Widen<ushort>(src, at, w); break;
            case DType.U32: Widen<uint>(src, at, w); break;
            case DType.U64: { ReadOnlySpan<ulong> x = src.Data<ulong>().Slice(at, w.Length); for (int j = 0; j < w.Length; j++) { w[j] = unchecked((long)x[j]); } break; }
            case DType.Bool: { ReadOnlySpan<bool> x = src.Data<bool>().Slice(at, w.Length); for (int j = 0; j < w.Length; j++) { w[j] = x[j] ? 1 : 0; } break; }
            default: throw new InvalidOperationException($"{s} is not an integer type.");
        }
    }

    private static void Widen<T>(Tensor src, int at, Span<long> w) where T : unmanaged, IBinaryInteger<T>
    {
        ReadOnlySpan<T> x = src.Data<T>().Slice(at, w.Length);
        for (int j = 0; j < w.Length; j++)
        {
            w[j] = long.CreateTruncating(x[j]);
        }
    }

    private static void WriteFromWide(Tensor dst, DType d, int at, ReadOnlySpan<double> w)
    {
        switch (d)
        {
            case DType.F32: { Span<float> y = dst.MutableData<float>().Slice(at, w.Length); for (int j = 0; j < w.Length; j++) { y[j] = (float)w[j]; } break; }
            case DType.F64: w.CopyTo(dst.MutableData<double>().Slice(at, w.Length)); break;
            case DType.F16: { Span<Half> y = dst.MutableData<Half>().Slice(at, w.Length); for (int j = 0; j < w.Length; j++) { y[j] = (Half)w[j]; } break; }
            case DType.BF16: { Span<BFloat16> y = dst.MutableData<BFloat16>().Slice(at, w.Length); for (int j = 0; j < w.Length; j++) { y[j] = (BFloat16)(float)w[j]; } break; }
            case DType.I8: Saturate<sbyte>(dst, at, w); break;
            case DType.I16: Saturate<short>(dst, at, w); break;
            case DType.I32: Saturate<int>(dst, at, w); break;
            case DType.I64: Saturate<long>(dst, at, w); break;
            case DType.U8: Saturate<byte>(dst, at, w); break;
            case DType.U16: Saturate<ushort>(dst, at, w); break;
            case DType.U32: Saturate<uint>(dst, at, w); break;
            case DType.U64: Saturate<ulong>(dst, at, w); break;
            case DType.Bool: { Span<bool> y = dst.MutableData<bool>().Slice(at, w.Length); for (int j = 0; j < w.Length; j++) { y[j] = w[j] != 0; } break; }
            default: throw new InvalidOperationException($"Cast to {d} is not supported.");
        }
    }

    private static void Saturate<T>(Tensor dst, int at, ReadOnlySpan<double> w) where T : unmanaged, IBinaryInteger<T>
    {
        Span<T> y = dst.MutableData<T>().Slice(at, w.Length);
        for (int j = 0; j < w.Length; j++)
        {
            y[j] = double.IsNaN(w[j]) ? T.Zero : T.CreateSaturating(w[j]);
        }
    }

    private static void WriteFromWhole(Tensor dst, DType d, int at, ReadOnlySpan<long> w, bool unsigned64)
    {
        switch (d)
        {
            case DType.F32: { Span<float> y = dst.MutableData<float>().Slice(at, w.Length); for (int j = 0; j < w.Length; j++) { y[j] = unsigned64 ? (float)unchecked((ulong)w[j]) : (float)w[j]; } break; }
            case DType.F64: { Span<double> y = dst.MutableData<double>().Slice(at, w.Length); for (int j = 0; j < w.Length; j++) { y[j] = unsigned64 ? (double)unchecked((ulong)w[j]) : w[j]; } break; }
            case DType.F16: { Span<Half> y = dst.MutableData<Half>().Slice(at, w.Length); for (int j = 0; j < w.Length; j++) { y[j] = (Half)(unsigned64 ? (double)unchecked((ulong)w[j]) : w[j]); } break; }
            case DType.BF16: { Span<BFloat16> y = dst.MutableData<BFloat16>().Slice(at, w.Length); for (int j = 0; j < w.Length; j++) { y[j] = (BFloat16)(unsigned64 ? (float)unchecked((ulong)w[j]) : (float)w[j]); } break; }
            case DType.I8: Wrap<sbyte>(dst, at, w); break;
            case DType.I16: Wrap<short>(dst, at, w); break;
            case DType.I32: Wrap<int>(dst, at, w); break;
            case DType.I64: w.CopyTo(dst.MutableData<long>().Slice(at, w.Length)); break;
            case DType.U8: Wrap<byte>(dst, at, w); break;
            case DType.U16: Wrap<ushort>(dst, at, w); break;
            case DType.U32: Wrap<uint>(dst, at, w); break;
            case DType.U64: Wrap<ulong>(dst, at, w); break;
            case DType.Bool: { Span<bool> y = dst.MutableData<bool>().Slice(at, w.Length); for (int j = 0; j < w.Length; j++) { y[j] = w[j] != 0; } break; }
            default: throw new InvalidOperationException($"Cast to {d} is not supported.");
        }
    }

    private static void Wrap<T>(Tensor dst, int at, ReadOnlySpan<long> w) where T : unmanaged, IBinaryInteger<T>
    {
        Span<T> y = dst.MutableData<T>().Slice(at, w.Length);
        for (int j = 0; j < w.Length; j++)
        {
            y[j] = T.CreateTruncating(w[j]);
        }
    }
}
