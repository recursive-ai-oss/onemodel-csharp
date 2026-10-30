using System;
using RecursiveAI.OneModel.Kernels.Exact;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// A float function over a large tensor in chunks of 16 384 elements. Every output element depends on its own input element only,
/// and a chunk is a whole number of the widest lane group, so each element goes through the same lane or scalar code as in a single
/// call: the split changes nothing in the result, in either mode. Only the last chunk has a scalar tail, as the whole tensor would.
/// </summary>
internal static unsafe class UnaryParallel
{
    private const int ChunkElements = 16384;

    /// <summary>Applies <paramref name="op"/> to <paramref name="x"/> into <paramref name="y"/> (the same length; may be the same memory).</summary>
    public static void Run<TOp>(KernelPath path, bool exact, TOp op, ReadOnlySpan<float> x, Span<float> y, int threads)
        where TOp : struct, IFloatUnary
    {
        if (y.Length < x.Length)
        {
            throw new ArgumentException("The output is shorter than the input.", nameof(y));
        }

        if (threads <= 1 || x.Length < 2 * ChunkElements)
        {
            FloatUnary.Run(path, exact, op, x, y);
            return;
        }

        fixed (float* px = x, py = y)
        {
            UnaryJob<TOp> job = UnaryJob<TOp>.Get();
            job.Path = path;
            job.Exact = exact;
            job.Op = op;
            job.X = px;
            job.Y = py;
            job.Length = x.Length;
            // About a nanosecond per element is a fair low estimate (HardSwish is cheaper, erf-Gelu dearer); it only decides how many
            // threads are worth waking.
            ParallelFor.Run(job, (x.Length + ChunkElements - 1) / ChunkElements, threads, x.Length);
            job.X = null;
            job.Y = null;
        }
    }

    private sealed class UnaryJob<TOp> : ChunkJob
        where TOp : struct, IFloatUnary
    {
        // Kept per thread and refilled, so a steady-state call allocates nothing; a call is done with it before it returns.
        [ThreadStatic]
        private static UnaryJob<TOp>? t_job;

        public KernelPath Path;
        public bool Exact;
        public TOp Op;
        public float* X;
        public float* Y;
        public int Length;

        public static UnaryJob<TOp> Get() => t_job ??= new UnaryJob<TOp>();

        public override void RunChunk(int chunk)
        {
            int start = chunk * ChunkElements;
            int len = Math.Min(ChunkElements, Length - start);
            FloatUnary.Run(Path, Exact, Op, new ReadOnlySpan<float>(X + start, len), new Span<float>(Y + start, len));
        }
    }
}
