using System;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Per-tensor quantization of a large float tensor in chunks of 16 384 elements, on top of <see cref="Int8Quantize"/>. Each output
/// byte depends on its own input only, so the split changes nothing; the min and max are combined in chunk-index order with the
/// same rule as one pass (a NaN never replaces a value), so they are the same for any number of threads.
/// </summary>
internal static unsafe class QuantizeParallel
{
    private const int ChunkElements = 16384;

    [ThreadStatic]
    private static QuantJob? t_quant;

    [ThreadStatic]
    private static RangeJob? t_range;

    /// <summary>Quantizes <paramref name="x"/> to 8 bits into <paramref name="y"/> (at least as long as x).</summary>
    public static void Quantize8(ReadOnlySpan<float> x, Span<byte> y, float scale, int zeroPoint, bool unsigned, int threads)
    {
        if (y.Length < x.Length)
        {
            throw new ArgumentException("The output is shorter than the input.", nameof(y));
        }

        if (threads <= 1 || x.Length < 2 * ChunkElements)
        {
            Int8Quantize.Quantize8(x, y, scale, zeroPoint, unsigned);
            return;
        }

        fixed (float* px = x)
        fixed (byte* py = y)
        {
            QuantJob job = t_quant ??= new QuantJob();
            job.X = px;
            job.Y = py;
            job.Length = x.Length;
            job.Scale = scale;
            job.ZeroPoint = zeroPoint;
            job.Unsigned = unsigned;
            // About a nanosecond per element; only decides how many threads are worth waking.
            ParallelFor.Run(job, (x.Length + ChunkElements - 1) / ChunkElements, threads, x.Length);
            job.X = null;
            job.Y = null;
        }
    }

    /// <summary>The smallest and largest non-NaN values of <paramref name="x"/> (+Inf and -Inf when there are none).</summary>
    public static void MinMax(ReadOnlySpan<float> x, out float min, out float max, int threads)
    {
        if (threads <= 1 || x.Length < 2 * ChunkElements)
        {
            Int8Quantize.MinMax(x, out min, out max);
            return;
        }

        int chunks = (x.Length + ChunkElements - 1) / ChunkElements;
        fixed (float* px = x)
        {
            RangeJob job = t_range ??= new RangeJob();
            job.Prepare(chunks);
            job.X = px;
            job.Length = x.Length;
            ParallelFor.Run(job, chunks, threads, x.Length);
            job.X = null;
            float lo = float.PositiveInfinity, hi = float.NegativeInfinity;
            for (int c = 0; c < chunks; c++)
            {
                if (job.Min[c] < lo)
                {
                    lo = job.Min[c];
                }

                if (job.Max[c] > hi)
                {
                    hi = job.Max[c];
                }
            }

            min = lo;
            max = hi;
        }
    }

    private sealed class QuantJob : ChunkJob
    {
        public float* X;
        public byte* Y;
        public int Length;
        public float Scale;
        public int ZeroPoint;
        public bool Unsigned;

        public override void RunChunk(int chunk)
        {
            int start = chunk * ChunkElements;
            int len = Math.Min(ChunkElements, Length - start);
            Int8Quantize.Quantize8(X + start, Y + start, len, Scale, ZeroPoint, Unsigned);
        }
    }

    private sealed class RangeJob : ChunkJob
    {
        public float* X;
        public int Length;
        public float[] Min = [];
        public float[] Max = [];

        // Grows once per thread; a steady-state call allocates nothing.
        public void Prepare(int chunks)
        {
            if (Min.Length < chunks)
            {
                Min = new float[chunks];
                Max = new float[chunks];
            }
        }

        public override void RunChunk(int chunk)
        {
            int start = chunk * ChunkElements;
            int len = Math.Min(ChunkElements, Length - start);
            Int8Quantize.MinMax(X + start, len, out Min[chunk], out Max[chunk]);
        }
    }
}
