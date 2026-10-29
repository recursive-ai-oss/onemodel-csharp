using System;
using System.Collections.Generic;
using System.Threading;
using System.Runtime.CompilerServices;

namespace RecursiveAI.OneModel.Tensors;

/// <summary>
/// IEEE 754 binary16 conversions on raw bits, written with integer arithmetic only so every platform gets the same bits.
/// Rounding is round-to-nearest, ties-to-even (IEEE 754-2008 §4.3.1, roundTiesToEven); overflow rounds to infinity; values below
/// half the smallest subnormal round to a signed zero. NaN keeps its sign and its top ten payload bits and always comes out quiet
/// (quiet bit set), in both directions. Float16 to float32 is exact.
/// </summary>
public static class Float16
{
    /// <summary>Converts binary32 to binary16 bits, rounding to nearest even.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ushort FromSingle(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        uint sign = (bits >> 16) & 0x8000u;
        uint abs = bits & 0x7FFFFFFFu;

        if (abs >= 0x7F800000u)
        {
            // Infinity keeps its sign; NaN keeps the top 9 payload bits below the quiet bit, and the quiet bit is forced.
            return abs == 0x7F800000u ? (ushort)(sign | 0x7C00u) : (ushort)(sign | 0x7E00u | ((abs >> 13) & 0x1FFu));
        }

        if (abs >= 0x477FF000u)
        {
            // 65520 is the midpoint between the largest finite half (65504) and 65536; the tie goes to the even neighbour, infinity.
            return (ushort)(sign | 0x7C00u);
        }

        if (abs >= 0x38800000u)
        {
            // Normal half range (>= 2^-14): rebias the exponent (127 - 15 = 112), then round 23 mantissa bits down to 10.
            uint m = abs - (112u << 23);
            uint r = m + 0x0FFFu + ((m >> 13) & 1u);
            return (ushort)(sign | (r >> 13));
        }

        if (abs <= 0x33000000u)
        {
            // At or below 2^-25, exactly half the smallest subnormal: the tie goes to the even neighbour, zero.
            return (ushort)sign;
        }

        {
            // Subnormal half: the result is the 24-bit significand scaled so one unit is 2^-24, rounded to nearest even.
            int e = (int)(abs >> 23);
            uint mant = (abs & 0x7FFFFFu) | 0x800000u;
            int shift = 126 - e; // 14..24
            uint r = mant >> shift;
            uint rem = mant & ((1u << shift) - 1u);
            uint half = 1u << (shift - 1);
            if (rem > half || (rem == half && (r & 1u) != 0))
            {
                r++;
            }

            return (ushort)(sign | r);
        }
    }

    /// <summary>Converts binary16 bits to binary32, exactly. A signaling NaN becomes the quiet NaN with the same payload.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ToSingle(ushort half)
    {
        uint sign = ((uint)half & 0x8000u) << 16;
        uint exp = ((uint)half >> 10) & 0x1Fu;
        uint man = (uint)half & 0x3FFu;
        if (exp == 0)
        {
            // Zero or subnormal: man * 2^-24 is exact in binary32.
            float v = man * 5.9604644775390625e-8f;
            return BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(v) | sign);
        }

        if (exp == 31)
        {
            return BitConverter.UInt32BitsToSingle(man == 0 ? sign | 0x7F800000u : sign | 0x7FC00000u | (man << 13));
        }

        return BitConverter.UInt32BitsToSingle(sign | ((exp + 112u) << 23) | (man << 13));
    }

    /// <summary>Converts a run of binary32 values to binary16 bits. <paramref name="dst"/> must be at least as long as <paramref name="src"/>.</summary>
    /// <exception cref="ArgumentException"><paramref name="dst"/> is shorter than <paramref name="src"/>.</exception>
    public static void FromSingle(ReadOnlySpan<float> src, Span<ushort> dst)
    {
        if (dst.Length < src.Length)
        {
            throw new ArgumentException("The destination is shorter than the source.", nameof(dst));
        }

        for (int i = 0; i < src.Length; i++)
        {
            dst[i] = FromSingle(src[i]);
        }
    }

    /// <summary>Converts a run of binary16 bits to binary32. <paramref name="dst"/> must be at least as long as <paramref name="src"/>.</summary>
    /// <exception cref="ArgumentException"><paramref name="dst"/> is shorter than <paramref name="src"/>.</exception>
    public static void ToSingle(ReadOnlySpan<ushort> src, Span<float> dst)
    {
        if (dst.Length < src.Length)
        {
            throw new ArgumentException("The destination is shorter than the source.", nameof(dst));
        }

        for (int i = 0; i < src.Length; i++)
        {
            dst[i] = ToSingle(src[i]);
        }
    }
}

/// <summary>
/// bfloat16: the top 16 bits of an IEEE binary32 (1 sign, 8 exponent, 7 mantissa bits). Conversion from binary32 rounds to nearest,
/// ties to even, on the discarded 16 bits (overflow becomes infinity, subnormals are kept); NaN keeps its sign and top payload bits
/// and is forced quiet. Conversion to binary32 is exact.
/// </summary>
public readonly struct BFloat16 : IEquatable<BFloat16>
{
    /// <summary>The raw 16 bits.</summary>
    public ushort Bits { get; }

    /// <summary>Wraps raw bits.</summary>
    public BFloat16(ushort bits) => Bits = bits;

    /// <summary>Rounds a binary32 value to bfloat16, to nearest even.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ushort FromSingleBits(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        if ((bits & 0x7FFFFFFFu) > 0x7F800000u)
        {
            return (ushort)((bits >> 16) | 0x0040u);
        }

        return (ushort)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
    }

    /// <summary>Widens bfloat16 bits to binary32, exactly. A signaling NaN becomes the quiet NaN with the same payload.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ToSingleBits(ushort bits)
    {
        uint b = (uint)bits << 16;
        if ((b & 0x7FFFFFFFu) > 0x7F800000u)
        {
            b |= 0x00400000u;
        }

        return BitConverter.UInt32BitsToSingle(b);
    }

    /// <summary>Rounds to bfloat16 (to nearest even).</summary>
    public static explicit operator BFloat16(float value) => new(FromSingleBits(value));

    /// <summary>Widens to binary32, exactly.</summary>
    public static explicit operator float(BFloat16 value) => ToSingleBits(value.Bits);

    /// <summary>Bitwise equality (so NaNs with equal bits are equal, and +0 differs from -0).</summary>
    public bool Equals(BFloat16 other) => Bits == other.Bits;

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is BFloat16 o && Equals(o);

    /// <inheritdoc/>
    public override int GetHashCode() => Bits;

    /// <inheritdoc/>
    public override string ToString() => ((float)this).ToString(System.Globalization.CultureInfo.InvariantCulture);

    /// <summary>Bitwise equality.</summary>
    public static bool operator ==(BFloat16 a, BFloat16 b) => a.Bits == b.Bits;

    /// <summary>Bitwise inequality.</summary>
    public static bool operator !=(BFloat16 a, BFloat16 b) => a.Bits != b.Bits;
}
