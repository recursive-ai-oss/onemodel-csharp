using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>The parameters of a quantized addition: both inputs' types, zero points and scales, and the output's.</summary>
internal readonly record struct Int8AddParams(
    bool AUnsigned, int AZero, float AScale, bool BUnsigned, int BZero, float BScale, bool YUnsigned, int YZero, float YScale);

/// <summary>
/// Addition of two quantized tensors into a quantized tensor in one pass: <c>y = quantize(dequantize(a) + dequantize(b))</c> with the
/// same float32 operations in the same order as the three separate operators (<c>(a - za) * sa</c>, the sum, then
/// <c>round_half_even(sum / sy) + zy</c> saturated), so the bytes are the bytes of the unfused chain and the intermediate float tensors
/// are never written. Each output element depends on its own inputs only, so chunking cannot change a result.
/// </summary>
internal static unsafe class Int8Elementwise
{
    private const int ChunkElements = 16384;

    [ThreadStatic]
    private static AddJob? t_add;

    /// <summary>
    /// y[i] = quantized (a[i] + b[i]) for i in [0, n). Reads n bytes at <paramref name="a"/> and <paramref name="b"/> and writes n bytes at
    /// <paramref name="y"/> (which may be the same memory as an input: each element is read before it is written).
    /// </summary>
    public static void Add8(byte* a, byte* b, byte* y, long n, in Int8AddParams p)
    {
        int lo = p.YUnsigned ? 0 : -128, hi = p.YUnsigned ? 255 : 127;
        long i = 0;
        if (Vector128.IsHardwareAccelerated)
        {
            Vector128<int> za = Vector128.Create(p.AZero), zb = Vector128.Create(p.BZero);
            Vector128<float> sa = Vector128.Create(p.AScale), sb = Vector128.Create(p.BScale), sy = Vector128.Create(p.YScale);
            Vector128<float> zy = Vector128.Create((float)p.YZero), vlo = Vector128.Create((float)lo), vhi = Vector128.Create((float)hi);
            for (; i + 16 <= n; i += 16)
            {
                Vector128<byte> ba = Vector128.Load(a + i), bb = Vector128.Load(b + i);
                Widen(ba, p.AUnsigned, out Vector128<int> a0, out Vector128<int> a1, out Vector128<int> a2, out Vector128<int> a3);
                Widen(bb, p.BUnsigned, out Vector128<int> b0, out Vector128<int> b1, out Vector128<int> b2, out Vector128<int> b3);
                Vector128<int> q0 = Q(Vector128.ConvertToSingle(a0 - za) * sa + (Vector128.ConvertToSingle(b0 - zb) * sb), sy, zy, vlo, vhi);
                Vector128<int> q1 = Q(Vector128.ConvertToSingle(a1 - za) * sa + (Vector128.ConvertToSingle(b1 - zb) * sb), sy, zy, vlo, vhi);
                Vector128<int> q2 = Q(Vector128.ConvertToSingle(a2 - za) * sa + (Vector128.ConvertToSingle(b2 - zb) * sb), sy, zy, vlo, vhi);
                Vector128<int> q3 = Q(Vector128.ConvertToSingle(a3 - za) * sa + (Vector128.ConvertToSingle(b3 - zb) * sb), sy, zy, vlo, vhi);
                Int8Quantize.Pack16(q0, q1, q2, q3, p.YUnsigned).Store(y + i);
            }
        }

        for (; i < n; i++)
        {
            y[i] = (byte)AddOne(a[i], b[i], in p, lo, hi);
        }
    }

    /// <summary>Scalar reference for one element.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int AddOne(byte a, byte b, in Int8AddParams p, int lo, int hi)
    {
        float da = (float)((p.AUnsigned ? a : (sbyte)a) - p.AZero) * p.AScale;
        float db = (float)((p.BUnsigned ? b : (sbyte)b) - p.BZero) * p.BScale;
        return Int8Quantize.QuantizeOne(da + db, p.YScale, p.YZero, lo, hi);
    }

    // Sixteen bytes to four vectors of four ints, sign- or zero-extended.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Widen(Vector128<byte> b, bool unsigned, out Vector128<int> w0, out Vector128<int> w1, out Vector128<int> w2, out Vector128<int> w3)
    {
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

        w0 = Vector128.WidenLower(s0);
        w1 = Vector128.WidenUpper(s0);
        w2 = Vector128.WidenLower(s1);
        w3 = Vector128.WidenUpper(s1);
    }

    // Quantizes four float sums like Int8Quantize's own four-lane step (NaN goes to the smallest value).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> Q(Vector128<float> sum, Vector128<float> scale, Vector128<float> zero, Vector128<float> lo, Vector128<float> hi)
    {
        Vector128<float> v = Vector128.Round(sum / scale) + zero;
        v = Vector128.ConditionalSelect(Vector128.Equals(v, v), v, lo);
        return Vector128.ConvertToInt32(Vector128.Min(Vector128.Max(v, lo), hi));
    }

    /// <summary>
    /// As the pointer form, in chunks of 16 384 elements on up to <paramref name="threads"/> threads. Same length n for all three
    /// buffers; the chunk boundaries depend on n only.
    /// </summary>
    public static void Add8(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b, Span<byte> y, in Int8AddParams p, int threads)
    {
        long n = y.Length;
        if (a.Length < n || b.Length < n)
        {
            throw new ArgumentException("An input is shorter than the output.");
        }

        fixed (byte* pa = a, pb = b, py = y)
        {
            if (threads <= 1 || n < 2 * ChunkElements)
            {
                Add8(pa, pb, py, n, in p);
                return;
            }

            AddJob job = t_add ??= new AddJob();
            job.A = pa;
            job.B = pb;
            job.Y = py;
            job.Length = n;
            job.P = p;
            // About half a nanosecond per element; it only decides how many threads are worth waking.
            ParallelFor.Run(job, (int)((n + ChunkElements - 1) / ChunkElements), threads, n / 2);
            job.A = null;
            job.B = null;
            job.Y = null;
        }
    }

    private sealed class AddJob : ChunkJob
    {
        public byte* A;
        public byte* B;
        public byte* Y;
        public long Length;
        public Int8AddParams P;

        public override void RunChunk(int chunk)
        {
            long start = (long)chunk * ChunkElements;
            Add8(A + start, B + start, Y + start, Math.Min(ChunkElements, Length - start), in P);
        }
    }
}
