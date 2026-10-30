using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// A function of one 8-bit value applied through a 256-entry table: <c>y[i] = table[x[i]]</c> (the input byte is the index, so an int8
/// input indexes by its two's-complement byte). A dequantize, an element-wise float function and a quantize in a row (the activations
/// of a quantized network) take only 256 different inputs, so the graph pass tabulates the chain once and this kernel replaces three
/// passes over float tensors by one pass over bytes.
/// </summary>
internal static unsafe class Int8Lut
{
    private const int ChunkElements = 65536;

    [ThreadStatic]
    private static LutJob? t_job;

    /// <summary>y[i] = table[x[i]]. Reads <c>x.Length</c> bytes and the 256 table bytes, writes <c>x.Length</c> bytes (y may be x).</summary>
    /// <remarks>Bounds: y and the table are length-checked before any pointer is formed; every index is a byte, inside the table.</remarks>
    public static void Apply(ReadOnlySpan<byte> x, ReadOnlySpan<byte> table, Span<byte> y, int threads)
    {
        if (table.Length < 256)
        {
            throw new ArgumentException("The table needs 256 entries.", nameof(table));
        }

        if (y.Length < x.Length)
        {
            throw new ArgumentException("The output is shorter than the input.", nameof(y));
        }

        long n = x.Length;
        fixed (byte* px = x, pt = table, py = y)
        {
            if (threads <= 1 || n < 2 * ChunkElements)
            {
                Apply(px, pt, py, n);
                return;
            }

            LutJob job = t_job ??= new LutJob();
            job.X = px;
            job.T = pt;
            job.Y = py;
            job.Length = n;
            // About a quarter of a nanosecond per element; it only decides how many threads are worth waking.
            ParallelFor.Run(job, (int)((n + ChunkElements - 1) / ChunkElements), threads, n / 4);
            job.X = null;
            job.T = null;
            job.Y = null;
        }
    }

    private static void Apply(byte* x, byte* t, byte* y, long n)
    {
        long i = 0;
        if (AdvSimd.Arm64.IsSupported && n >= 16)
        {
            // TBL looks up 64 entries at a time and gives 0 for an index past them; TBX keeps the earlier result for one past them.
            // Subtracting 64, 128 and 192 (bytes wrap) shifts each quarter of the table into range in turn.
            (Vector128<byte>, Vector128<byte>, Vector128<byte>, Vector128<byte>) part0 = (Vector128.Load(t), Vector128.Load(t + 16), Vector128.Load(t + 32), Vector128.Load(t + 48));
            (Vector128<byte>, Vector128<byte>, Vector128<byte>, Vector128<byte>) part1 = (Vector128.Load(t + 64), Vector128.Load(t + 80), Vector128.Load(t + 96), Vector128.Load(t + 112));
            (Vector128<byte>, Vector128<byte>, Vector128<byte>, Vector128<byte>) part2 = (Vector128.Load(t + 128), Vector128.Load(t + 144), Vector128.Load(t + 160), Vector128.Load(t + 176));
            (Vector128<byte>, Vector128<byte>, Vector128<byte>, Vector128<byte>) part3 = (Vector128.Load(t + 192), Vector128.Load(t + 208), Vector128.Load(t + 224), Vector128.Load(t + 240));
            Vector128<byte> c64 = Vector128.Create((byte)64), c128 = Vector128.Create((byte)128), c192 = Vector128.Create((byte)192);
            for (; i + 16 <= n; i += 16)
            {
                Vector128<byte> idx = Vector128.Load(x + i);
                Vector128<byte> r = AdvSimd.Arm64.VectorTableLookup(part0, idx);
                r = AdvSimd.Arm64.VectorTableLookupExtension(r, part1, idx - c64);
                r = AdvSimd.Arm64.VectorTableLookupExtension(r, part2, idx - c128);
                r = AdvSimd.Arm64.VectorTableLookupExtension(r, part3, idx - c192);
                r.Store(y + i);
            }
        }

        for (; i + 8 <= n; i += 8)
        {
            // Load all eight indices before the first store, so y may alias x.
            byte a0 = x[i], a1 = x[i + 1], a2 = x[i + 2], a3 = x[i + 3], a4 = x[i + 4], a5 = x[i + 5], a6 = x[i + 6], a7 = x[i + 7];
            y[i] = t[a0];
            y[i + 1] = t[a1];
            y[i + 2] = t[a2];
            y[i + 3] = t[a3];
            y[i + 4] = t[a4];
            y[i + 5] = t[a5];
            y[i + 6] = t[a6];
            y[i + 7] = t[a7];
        }

        for (; i < n; i++)
        {
            y[i] = t[x[i]];
        }
    }

    private sealed class LutJob : ChunkJob
    {
        public byte* X;
        public byte* T;
        public byte* Y;
        public long Length;

        public override void RunChunk(int chunk)
        {
            long start = (long)chunk * ChunkElements;
            Apply(X + start, T, Y + start, Math.Min(ChunkElements, Length - start));
        }
    }
}
