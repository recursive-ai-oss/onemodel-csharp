using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Quantization and dequantization of float tensors, after Jacob et al. (2018): <c>q = saturate(round(x / scale) + zeroPoint)</c> with
/// round-half-to-even, and <c>x = (q - zeroPoint) * scale</c>. Division, rounding to nearest even and the single multiplication are
/// correctly rounded IEEE operations, and the clamp and the integer conversion happen only on values already inside the output range,
/// so the scalar and 4-lane vector paths give the same bits on every machine. A NaN input quantizes to the smallest value of the output type (as ONNX Runtime does).
/// </summary>
internal static unsafe class Int8Quantize
{
    /// <summary>Scalar reference for one element.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int QuantizeOne(float x, float scale, int zeroPoint, int lo, int hi)
    {
        float v = MathF.Round(x / scale) + zeroPoint;
        if (float.IsNaN(v))
        {
            return lo;
        }

        return v < lo ? lo : v > hi ? hi : (int)v;
    }

    /// <summary>Quantizes <paramref name="n"/> floats to uint8 (<paramref name="unsigned"/>) or int8 with one scale and zero point.</summary>
    public static void Quantize8(float* x, byte* y, long n, float scale, int zeroPoint, bool unsigned)
    {
        int lo = unsigned ? 0 : -128, hi = unsigned ? 255 : 127;
        long i = 0;
        if (Vector128.IsHardwareAccelerated)
        {
            Vector128<float> vs = Vector128.Create(scale);
            Vector128<float> vz = Vector128.Create((float)zeroPoint);
            Vector128<float> vlo = Vector128.Create((float)lo);
            Vector128<float> vhi = Vector128.Create((float)hi);
            for (; i + 16 <= n; i += 16)
            {
                Vector128<int> q0 = Quant4(x + i, vs, vz, vlo, vhi);
                Vector128<int> q1 = Quant4(x + i + 4, vs, vz, vlo, vhi);
                Vector128<int> q2 = Quant4(x + i + 8, vs, vz, vlo, vhi);
                Vector128<int> q3 = Quant4(x + i + 12, vs, vz, vlo, vhi);
                Pack16(q0, q1, q2, q3, unsigned).Store(y + i);
            }
        }

        for (; i < n; i++)
        {
            y[i] = (byte)QuantizeOne(x[i], scale, zeroPoint, lo, hi);
        }
    }

    /// <summary>
    /// Narrows sixteen already-clamped values to bytes. Every step narrows to a type that holds all the values (int to short, then short
    /// to sbyte for signed output or to byte for unsigned), so truncating and saturating narrowing agree. A runtime that implements
    /// narrowing with a saturating instruction (the browser-wasm runtime does once code is hot) then gives the same bytes as every other.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector128<byte> Pack16(Vector128<int> q0, Vector128<int> q1, Vector128<int> q2, Vector128<int> q3, bool unsigned)
    {
        Vector128<short> s0 = Vector128.Narrow(q0, q1);
        Vector128<short> s1 = Vector128.Narrow(q2, q3);
        return unsigned
            ? Vector128.Narrow(s0.AsUInt16(), s1.AsUInt16())
            : Vector128.Narrow(s0, s1).AsByte();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> Quant4(float* p, Vector128<float> scale, Vector128<float> zero, Vector128<float> lo, Vector128<float> hi)
    {
        Vector128<float> v = Vector128.Round(Vector128.Load(p) / scale) + zero;
        // NaN compares unequal to itself: send it to the smallest value, so Min and Max never see one.
        v = Vector128.ConditionalSelect(Vector128.Equals(v, v), v, lo);
        v = Vector128.Min(Vector128.Max(v, lo), hi);
        return Vector128.ConvertToInt32(v);
    }

    /// <summary>Quantizes to 16-bit integers (scalar).</summary>
    public static void Quantize16(float* x, short* y, long n, float scale, int zeroPoint, bool unsigned)
    {
        int lo = unsigned ? 0 : short.MinValue, hi = unsigned ? ushort.MaxValue : short.MaxValue;
        for (long i = 0; i < n; i++)
        {
            y[i] = (short)QuantizeOne(x[i], scale, zeroPoint, lo, hi);
        }
    }

    /// <summary>Dequantizes <paramref name="n"/> 8-bit values: (q - zeroPoint) * scale.</summary>
    public static void Dequantize8(byte* q, bool unsigned, float* y, long n, float scale, int zeroPoint)
    {
        long i = 0;
        if (Vector128.IsHardwareAccelerated)
        {
            Vector128<int> vz = Vector128.Create(zeroPoint);
            Vector128<float> vs = Vector128.Create(scale);
            for (; i + 16 <= n; i += 16)
            {
                Vector128<byte> b = Vector128.Load(q + i);
                Vector128<short> s0, s1;
                if (unsigned)
                {
                    s0 = Vector128.WidenLower(b).AsInt16();
                    s1 = Vector128.WidenUpper(b).AsInt16();
                }
                else
                {
                    Vector128<sbyte> sb = b.AsSByte();
                    s0 = Vector128.WidenLower(sb);
                    s1 = Vector128.WidenUpper(sb);
                }

                (Vector128.ConvertToSingle(Vector128.WidenLower(s0) - vz) * vs).Store(y + i);
                (Vector128.ConvertToSingle(Vector128.WidenUpper(s0) - vz) * vs).Store(y + i + 4);
                (Vector128.ConvertToSingle(Vector128.WidenLower(s1) - vz) * vs).Store(y + i + 8);
                (Vector128.ConvertToSingle(Vector128.WidenUpper(s1) - vz) * vs).Store(y + i + 12);
            }
        }

        for (; i < n; i++)
        {
            int v = unsigned ? q[i] : (sbyte)q[i];
            y[i] = (float)(v - zeroPoint) * scale;
        }
    }

    /// <summary>Dequantizes 16-bit values (scalar).</summary>
    public static void Dequantize16(short* q, bool unsigned, float* y, long n, float scale, int zeroPoint)
    {
        for (long i = 0; i < n; i++)
        {
            int v = unsigned ? (ushort)q[i] : q[i];
            y[i] = (float)(v - zeroPoint) * scale;
        }
    }

    /// <summary>
    /// The smallest and largest of <paramref name="n"/> floats, ignoring NaN (an all-NaN or empty input gives +inf and -inf). Min and
    /// max are exact whatever the order, so the lane split does not matter.
    /// </summary>
    public static void MinMax(float* x, long n, out float min, out float max)
    {
        float lo = float.PositiveInfinity, hi = float.NegativeInfinity;
        long i = 0;
        if (Vector128.IsHardwareAccelerated && n >= 8)
        {
            Vector128<float> vlo = Vector128.Create(lo), vhi = Vector128.Create(hi);
            for (; i + 4 <= n; i += 4)
            {
                Vector128<float> v = Vector128.Load(x + i);
                // Comparisons with NaN are false, so a NaN lane never replaces the running value.
                vlo = Vector128.ConditionalSelect(Vector128.LessThan(v, vlo), v, vlo);
                vhi = Vector128.ConditionalSelect(Vector128.GreaterThan(v, vhi), v, vhi);
            }

            for (int l = 0; l < 4; l++)
            {
                float a = vlo.GetElement(l), b = vhi.GetElement(l);
                if (a < lo)
                {
                    lo = a;
                }

                if (b > hi)
                {
                    hi = b;
                }
            }
        }

        for (; i < n; i++)
        {
            float v = x[i];
            if (v < lo)
            {
                lo = v;
            }

            if (v > hi)
            {
                hi = v;
            }
        }

        min = lo;
        max = hi;
    }

    /// <summary>Span form of <see cref="Quantize8(float*, byte*, long, float, int, bool)"/>; <paramref name="y"/> must be as long as <paramref name="x"/>.</summary>
    public static void Quantize8(ReadOnlySpan<float> x, Span<byte> y, float scale, int zeroPoint, bool unsigned)
    {
        if (y.Length < x.Length)
        {
            throw new ArgumentException("The output is shorter than the input.", nameof(y));
        }

        fixed (float* px = x)
        fixed (byte* py = y)
        {
            Quantize8(px, py, x.Length, scale, zeroPoint, unsigned);
        }
    }

    /// <summary>Span form of the 16-bit quantizer.</summary>
    public static void Quantize16(ReadOnlySpan<float> x, Span<short> y, float scale, int zeroPoint, bool unsigned)
    {
        if (y.Length < x.Length)
        {
            throw new ArgumentException("The output is shorter than the input.", nameof(y));
        }

        fixed (float* px = x)
        fixed (short* py = y)
        {
            Quantize16(px, py, x.Length, scale, zeroPoint, unsigned);
        }
    }

    /// <summary>One element, as <see cref="QuantizeOne"/>; for per-axis tensors whose scale changes along the contiguous axis.</summary>
    public static int Quantize1(float x, float scale, int zeroPoint, int lo, int hi) => QuantizeOne(x, scale, zeroPoint, lo, hi);

    /// <summary>Span form of <see cref="Dequantize8(byte*, bool, float*, long, float, int)"/>.</summary>
    public static void Dequantize8(ReadOnlySpan<byte> q, bool unsigned, Span<float> y, float scale, int zeroPoint)
    {
        if (y.Length < q.Length)
        {
            throw new ArgumentException("The output is shorter than the input.", nameof(y));
        }

        fixed (byte* pq = q)
        fixed (float* py = y)
        {
            Dequantize8(pq, unsigned, py, q.Length, scale, zeroPoint);
        }
    }

    /// <summary>Span form of <see cref="Dequantize16(short*, bool, float*, long, float, int)"/> (the 16-bit values as raw shorts).</summary>
    public static void Dequantize16(ReadOnlySpan<short> q, bool unsigned, Span<float> y, float scale, int zeroPoint)
    {
        if (y.Length < q.Length)
        {
            throw new ArgumentException("The output is shorter than the input.", nameof(y));
        }

        fixed (short* pq = q)
        fixed (float* py = y)
        {
            Dequantize16(pq, unsigned, py, q.Length, scale, zeroPoint);
        }
    }

    /// <summary>Span form of <see cref="MinMax(float*, long, out float, out float)"/>.</summary>
    public static void MinMax(ReadOnlySpan<float> x, out float min, out float max)
    {
        fixed (float* px = x)
        {
            MinMax(px, x.Length, out min, out max);
        }
    }

    /// <summary>Views int8 values as raw bytes (the same memory).</summary>
    public static ReadOnlySpan<byte> AsBytes(ReadOnlySpan<sbyte> v) => MemoryMarshal.AsBytes(v);

    /// <summary>Views writable int8 values as raw bytes (the same memory).</summary>
    public static Span<byte> AsBytes(Span<sbyte> v) => MemoryMarshal.AsBytes(v);

    /// <summary>Views uint16 values as raw 16-bit patterns (the same memory).</summary>
    public static ReadOnlySpan<short> AsShorts(ReadOnlySpan<ushort> v) => MemoryMarshal.Cast<ushort, short>(v);

    /// <summary>Views writable uint16 values as raw 16-bit patterns (the same memory).</summary>
    public static Span<short> AsShorts(Span<ushort> v) => MemoryMarshal.Cast<ushort, short>(v);
}
