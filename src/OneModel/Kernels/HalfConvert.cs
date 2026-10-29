using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Bulk conversions between binary32 and the two 16-bit storage formats. Each gives exactly the bits of the scalar conversions in
/// <see cref="Float16"/> and <see cref="BFloat16"/> (round to nearest even, NaN forced quiet, overflow to infinity), on every
/// instruction set: the vector code is the same integer arithmetic as the scalar code, eight values at a time, so exact mode can use it.
/// docs/fp16.md has the rules.
/// </summary>
internal static unsafe class HalfConvert
{
    /// <summary>Widens <paramref name="count"/> binary16 values at <paramref name="src"/> to binary32 at <paramref name="dst"/>.</summary>
    /// <remarks>Bounds: the caller guarantees both pointers cover <paramref name="count"/> elements.</remarks>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void HalfToSingle(ushort* src, float* dst, long count)
    {
        long i = 0;
        if (Vector128.IsHardwareAccelerated)
        {
            for (; i + 8 <= count; i += 8)
            {
                (Vector128<uint> lo, Vector128<uint> hi) = Vector128.Widen(Vector128.Load(src + i));
                WidenHalf(lo).Store(dst + i);
                WidenHalf(hi).Store(dst + i + 4);
            }
        }

        for (; i < count; i++)
        {
            dst[i] = Float16.ToSingle(src[i]);
        }
    }

    // Four binary16 values, each in a 32-bit lane, to binary32. Exponent field 0 (zero and subnormals) is rescaled by a float subtraction
    // that is exact; exponent field 31 (infinity and NaN) gets the exponent 255 and the quiet bit when there is a payload.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> WidenHalf(Vector128<uint> h)
    {
        Vector128<uint> sign = Vector128.ShiftLeft(h & Vector128.Create(0x8000u), 16);
        Vector128<uint> em = Vector128.ShiftLeft(h & Vector128.Create(0x7FFFu), 13);
        Vector128<uint> exp = em & Vector128.Create(0x0F800000u);
        Vector128<uint> o = em + Vector128.Create(112u << 23);
        Vector128<uint> isSpecial = Vector128.Equals(exp, Vector128.Create(0x0F800000u));
        Vector128<uint> hasPayload = ~Vector128.Equals(em & Vector128.Create(0x007FFFFFu), Vector128<uint>.Zero);
        o = Vector128.ConditionalSelect(isSpecial, o + Vector128.Create(112u << 23), o);
        o |= isSpecial & hasPayload & Vector128.Create(0x00400000u);
        Vector128<uint> isSmall = Vector128.Equals(exp, Vector128<uint>.Zero);
        Vector128<float> f = (o + (isSmall & Vector128.Create(1u << 23))).AsSingle();
        Vector128<float> fixedUp = f - Vector128.Create(BitConverter.UInt32BitsToSingle(113u << 23));
        return (Vector128.ConditionalSelect(isSmall, fixedUp.AsUInt32(), f.AsUInt32()) | sign).AsSingle();
    }

    /// <summary>Rounds <paramref name="count"/> binary32 values to binary16 (nearest even).</summary>
    /// <remarks>Bounds: the caller guarantees both pointers cover <paramref name="count"/> elements.</remarks>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void SingleToHalf(float* src, ushort* dst, long count)
    {
        long i = 0;
        if (Vector128.IsHardwareAccelerated)
        {
            for (; i + 8 <= count; i += 8)
            {
                Vector128<uint> lo = NarrowToHalf(Vector128.Load(src + i).AsUInt32());
                Vector128<uint> hi = NarrowToHalf(Vector128.Load(src + i + 4).AsUInt32());
                Vector128.Narrow(lo, hi).Store(dst + i);
            }
        }

        for (; i < count; i++)
        {
            dst[i] = Float16.FromSingle(src[i]);
        }
    }

    // Four binary32 bit patterns to binary16 bits (in 32-bit lanes). Three regions of the magnitude: below 2^-14 the float addition of 0.5
    // leaves the half subnormal in the low bits, rounded by the hardware's own nearest-even addition; the normal range is rebiased and
    // rounded by adding 0xFFF plus the lowest kept bit; from 65520 up the result is infinity, and NaN keeps its top payload bits, quiet.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> NarrowToHalf(Vector128<uint> x)
    {
        Vector128<uint> sign = Vector128.ShiftRightLogical(x, 16) & Vector128.Create(0x8000u);
        Vector128<uint> a = x & Vector128.Create(0x7FFFFFFFu);

        Vector128<uint> m = a - Vector128.Create(112u << 23);
        Vector128<uint> normal = Vector128.ShiftRightLogical(m + Vector128.Create(0x0FFFu) + (Vector128.ShiftRightLogical(m, 13) & Vector128.Create(1u)), 13);

        Vector128<float> sub = a.AsSingle() + Vector128.Create(0.5f);
        Vector128<uint> subnormal = sub.AsUInt32() - Vector128.Create(126u << 23);

        Vector128<uint> isNaN = Vector128.GreaterThan(a, Vector128.Create(0x7F800000u));
        Vector128<uint> nan = Vector128.Create(0x7E00u) | (Vector128.ShiftRightLogical(a, 13) & Vector128.Create(0x1FFu));

        Vector128<uint> r = Vector128.ConditionalSelect(Vector128.LessThan(a, Vector128.Create(0x38800000u)), subnormal, normal);
        r = Vector128.ConditionalSelect(Vector128.GreaterThanOrEqual(a, Vector128.Create(0x477FF000u)), Vector128.Create(0x7C00u), r);
        r = Vector128.ConditionalSelect(isNaN, nan, r);
        return r | sign;
    }

    /// <summary>Widens <paramref name="count"/> bfloat16 values to binary32 (exact; NaN forced quiet).</summary>
    /// <remarks>Bounds: the caller guarantees both pointers cover <paramref name="count"/> elements.</remarks>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void BFloatToSingle(ushort* src, float* dst, long count)
    {
        long i = 0;
        if (Vector128.IsHardwareAccelerated)
        {
            for (; i + 8 <= count; i += 8)
            {
                (Vector128<uint> lo, Vector128<uint> hi) = Vector128.Widen(Vector128.Load(src + i));
                WidenBrain(lo).Store(dst + i);
                WidenBrain(hi).Store(dst + i + 4);
            }
        }

        for (; i < count; i++)
        {
            dst[i] = BFloat16.ToSingleBits(src[i]);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> WidenBrain(Vector128<uint> h)
    {
        Vector128<uint> b = Vector128.ShiftLeft(h, 16);
        Vector128<uint> isNaN = Vector128.GreaterThan(b & Vector128.Create(0x7FFFFFFFu), Vector128.Create(0x7F800000u));
        return (b | (isNaN & Vector128.Create(0x00400000u))).AsSingle();
    }

    /// <summary>Rounds <paramref name="count"/> binary32 values to bfloat16 (nearest even; NaN stays NaN, quiet).</summary>
    /// <remarks>Bounds: the caller guarantees both pointers cover <paramref name="count"/> elements.</remarks>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void SingleToBFloat(float* src, ushort* dst, long count)
    {
        long i = 0;
        if (Vector128.IsHardwareAccelerated)
        {
            for (; i + 8 <= count; i += 8)
            {
                Vector128<uint> lo = NarrowToBrain(Vector128.Load(src + i).AsUInt32());
                Vector128<uint> hi = NarrowToBrain(Vector128.Load(src + i + 4).AsUInt32());
                Vector128.Narrow(lo, hi).Store(dst + i);
            }
        }

        for (; i < count; i++)
        {
            dst[i] = BFloat16.FromSingleBits(src[i]);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> NarrowToBrain(Vector128<uint> x)
    {
        Vector128<uint> isNaN = Vector128.GreaterThan(x & Vector128.Create(0x7FFFFFFFu), Vector128.Create(0x7F800000u));
        Vector128<uint> rounded = Vector128.ShiftRightLogical(x + Vector128.Create(0x7FFFu) + (Vector128.ShiftRightLogical(x, 16) & Vector128.Create(1u)), 16);
        Vector128<uint> nan = Vector128.ShiftRightLogical(x, 16) | Vector128.Create(0x0040u);
        return Vector128.ConditionalSelect(isNaN, nan, rounded);
    }

    /// <summary>Span form of <see cref="HalfToSingle(ushort*, float*, long)"/>.</summary>
    /// <exception cref="ArgumentException"><paramref name="dst"/> is shorter than <paramref name="src"/>.</exception>
    public static void HalfToSingle(ReadOnlySpan<ushort> src, Span<float> dst)
    {
        CheckLength(src.Length, dst.Length);
        fixed (ushort* ps = src)
        fixed (float* pd = dst)
        {
            HalfToSingle(ps, pd, src.Length);
        }
    }

    /// <summary>Span form of <see cref="BFloatToSingle(ushort*, float*, long)"/>.</summary>
    /// <exception cref="ArgumentException"><paramref name="dst"/> is shorter than <paramref name="src"/>.</exception>
    public static void BFloatToSingle(ReadOnlySpan<ushort> src, Span<float> dst)
    {
        CheckLength(src.Length, dst.Length);
        fixed (ushort* ps = src)
        fixed (float* pd = dst)
        {
            BFloatToSingle(ps, pd, src.Length);
        }
    }

    /// <summary>Span form of <see cref="SingleToHalf(float*, ushort*, long)"/>.</summary>
    /// <exception cref="ArgumentException"><paramref name="dst"/> is shorter than <paramref name="src"/>.</exception>
    public static void SingleToHalf(ReadOnlySpan<float> src, Span<ushort> dst)
    {
        CheckLength(src.Length, dst.Length);
        fixed (float* ps = src)
        fixed (ushort* pd = dst)
        {
            SingleToHalf(ps, pd, src.Length);
        }
    }

    /// <summary>Span form of <see cref="SingleToBFloat(float*, ushort*, long)"/>.</summary>
    /// <exception cref="ArgumentException"><paramref name="dst"/> is shorter than <paramref name="src"/>.</exception>
    public static void SingleToBFloat(ReadOnlySpan<float> src, Span<ushort> dst)
    {
        CheckLength(src.Length, dst.Length);
        fixed (float* ps = src)
        fixed (ushort* pd = dst)
        {
            SingleToBFloat(ps, pd, src.Length);
        }
    }

    private static void CheckLength(int source, int destination)
    {
        if (destination < source)
        {
            throw new ArgumentException("The destination is shorter than the source.");
        }
    }

    /// <summary>
    /// Rounds little-endian binary32 values in <paramref name="source"/> (a model file's bytes, so possibly unaligned) into the binary16 or
    /// bfloat16 tensor <paramref name="destination"/>. Returns false, leaving the destination undefined, when a value is not finite after
    /// rounding (a binary16 overflow, an infinity or a NaN).
    /// </summary>
    /// <exception cref="ArgumentException">The byte count is not four times the element count of <paramref name="destination"/>.</exception>
    public static bool NarrowBytes(ReadOnlySpan<byte> source, Tensor destination)
    {
        Span<ushort> d = destination.RawMutableData<ushort>();
        if (!BitConverter.IsLittleEndian || source.Length != d.Length * 4L)
        {
            throw new ArgumentException("The source is not little-endian binary32 data of the destination's length.", nameof(source));
        }

        fixed (byte* ps = source)
        fixed (ushort* pd = d)
        {
            if (destination.DType == DType.F16)
            {
                SingleToHalf((float*)ps, pd, d.Length);
            }
            else
            {
                SingleToBFloat((float*)ps, pd, d.Length);
            }

            // An exponent field of all ones is infinity or NaN (binary16: 0x7C00, bfloat16: 0x7F80).
            ushort mask = destination.DType == DType.F16 ? (ushort)0x7C00 : (ushort)0x7F80;
            for (int i = 0; i < d.Length; i++)
            {
                if ((pd[i] & mask) == mask)
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>Widens a binary16 or bfloat16 tensor into a binary32 tensor of the same shape.</summary>
    public static void Widen(Tensor src, Tensor dst)
    {
        ReadOnlySpan<ushort> s = src.RawData<ushort>();
        Span<float> d = dst.MutableData<float>();
        if (d.Length < s.Length)
        {
            throw new ArgumentException("The destination is smaller than the source.", nameof(dst));
        }

        fixed (ushort* ps = s)
        fixed (float* pd = d)
        {
            if (src.DType == DType.F16)
            {
                HalfToSingle(ps, pd, s.Length);
            }
            else
            {
                BFloatToSingle(ps, pd, s.Length);
            }
        }
    }

    /// <summary>Rounds a binary32 tensor into a binary16 or bfloat16 tensor of the same shape.</summary>
    public static void Narrow(Tensor src, Tensor dst)
    {
        ReadOnlySpan<float> s = src.Data<float>();
        Span<ushort> d = dst.RawMutableData<ushort>();
        if (d.Length < s.Length)
        {
            throw new ArgumentException("The destination is smaller than the source.", nameof(dst));
        }

        fixed (float* ps = s)
        fixed (ushort* pd = d)
        {
            if (dst.DType == DType.F16)
            {
                SingleToHalf(ps, pd, s.Length);
            }
            else
            {
                SingleToBFloat(ps, pd, s.Length);
            }
        }
    }
}
