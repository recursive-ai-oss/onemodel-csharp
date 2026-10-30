using System;
using System.Runtime.Intrinsics;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Product of a quantized tensor and a quantized tensor that holds one value per row (the squeeze-and-excitation gate of a quantized
/// network), into a quantized tensor in one pass: <c>y = quantize(dequantize(a) * dequantize(b))</c> with the float32 operations of the
/// three separate operators in their order (<c>(a - za) * sa</c>, <c>(b - zb) * sb</c>, the product, then
/// <c>round_half_even(p / sy) + zy</c> saturated). The tensor is <c>outer</c> rows of <c>inner</c> elements and row <c>o</c> is
/// multiplied by <c>b[o]</c>. Each output depends on its own inputs only, so chunking cannot change a result.
/// </summary>
internal static unsafe class Int8ChannelMul
{
    private const int ChunkElements = 16384;

    [ThreadStatic]
    private static MulJob? t_job;

    /// <summary>
    /// Computes the product. Reads <c>outer * inner</c> bytes of a and <c>outer</c> bytes of b, writes <c>outer * inner</c> bytes of y
    /// (y may be a). The parameters' A side is the full tensor and B side the per-row one.
    /// </summary>
    /// <remarks>Bounds: all three spans are length-checked against outer and inner before any pointer is formed.</remarks>
    public static void Run(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b, Span<byte> y, long outer, long inner, in Int8AddParams p, int threads)
    {
        if (outer < 0 || inner < 0 || a.Length < outer * inner || y.Length < outer * inner || b.Length < outer)
        {
            throw new ArgumentException("A buffer is smaller than outer * inner (or outer) elements.");
        }

        if (outer == 0 || inner == 0)
        {
            return;
        }

        fixed (byte* pa = a, pb = b, py = y)
        {
            int perChunk = (int)Math.Max(1, Math.Min(outer, ChunkElements / inner));
            int chunks = (int)((outer + perChunk - 1) / perChunk);
            if (threads <= 1 || chunks < 2)
            {
                Rows(pa, pb, py, 0, outer, inner, in p);
                return;
            }

            MulJob job = t_job ??= new MulJob();
            job.A = pa;
            job.B = pb;
            job.Y = py;
            job.Outer = outer;
            job.Inner = inner;
            job.PerChunk = perChunk;
            job.P = p;
            // About a nanosecond per element (a division per element); it only decides how many threads are worth waking.
            ParallelFor.Run(job, chunks, threads, outer * inner);
            job.A = null;
            job.B = null;
            job.Y = null;
        }
    }

    private static void Rows(byte* a, byte* b, byte* y, long first, long end, long inner, in Int8AddParams p)
    {
        int lo = p.YUnsigned ? 0 : -128, hi = p.YUnsigned ? 255 : 127;
        for (long o = first; o < end; o++)
        {
            float fb = (float)((p.BUnsigned ? b[o] : (sbyte)b[o]) - p.BZero) * p.BScale;
            byte* ra = a + (o * inner);
            byte* ry = y + (o * inner);
            long i = 0;
            if (Vector128.IsHardwareAccelerated)
            {
                Vector128<int> za = Vector128.Create(p.AZero);
                Vector128<float> sa = Vector128.Create(p.AScale), vb = Vector128.Create(fb), sy = Vector128.Create(p.YScale);
                Vector128<float> zy = Vector128.Create((float)p.YZero), vlo = Vector128.Create((float)lo), vhi = Vector128.Create((float)hi);
                for (; i + 16 <= inner; i += 16)
                {
                    Int8Elementwise.Widen(Vector128.Load(ra + i), p.AUnsigned, out Vector128<int> a0, out Vector128<int> a1, out Vector128<int> a2, out Vector128<int> a3);
                    Vector128<int> n0 = Int8Elementwise.Q(Vector128.ConvertToSingle(a0 - za) * sa * vb, sy, zy, vlo, vhi);
                    Vector128<int> n1 = Int8Elementwise.Q(Vector128.ConvertToSingle(a1 - za) * sa * vb, sy, zy, vlo, vhi);
                    Vector128<int> n2 = Int8Elementwise.Q(Vector128.ConvertToSingle(a2 - za) * sa * vb, sy, zy, vlo, vhi);
                    Vector128<int> n3 = Int8Elementwise.Q(Vector128.ConvertToSingle(a3 - za) * sa * vb, sy, zy, vlo, vhi);
                    Int8Quantize.Pack16(n0, n1, n2, n3, p.YUnsigned).Store(ry + i);
                }
            }

            for (; i < inner; i++)
            {
                float da = (float)((p.AUnsigned ? ra[i] : (sbyte)ra[i]) - p.AZero) * p.AScale;
                ry[i] = (byte)Int8Quantize.QuantizeOne(da * fb, p.YScale, p.YZero, lo, hi);
            }
        }
    }

    private sealed class MulJob : ChunkJob
    {
        public byte* A;
        public byte* B;
        public byte* Y;
        public long Outer;
        public long Inner;
        public int PerChunk;
        public Int8AddParams P;

        public override void RunChunk(int chunk)
        {
            long first = (long)chunk * PerChunk;
            Rows(A, B, Y, first, Math.Min(Outer, first + PerChunk), Inner, in P);
        }
    }
}
