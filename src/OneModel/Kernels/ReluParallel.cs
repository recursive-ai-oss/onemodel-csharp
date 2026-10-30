using System;
using RecursiveAI.OneModel.Kernels.Exact;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Relu over a large float tensor in chunks of 32 768 elements. Every output element depends on its own input element only, so the
/// split changes nothing in the result, in either mode; the row kernel is <see cref="ExactElementwise.Relu(KernelPath, bool, ReadOnlySpan{float}, Span{float})"/>.
/// </summary>
internal static unsafe class ReluParallel
{
    private const int ChunkElements = 32768;

    [ThreadStatic]
    private static ReluJob? t_job;

    /// <summary>Applies Relu to <paramref name="x"/> into <paramref name="y"/> (the same length; may be the same memory).</summary>
    public static void Run(KernelPath path, bool canonicalNaN, ReadOnlySpan<float> x, Span<float> y, int threads)
    {
        if (y.Length < x.Length)
        {
            throw new ArgumentException("The output is shorter than the input.", nameof(y));
        }

        if (threads <= 1 || x.Length < 2 * ChunkElements)
        {
            ExactElementwise.Relu(path, canonicalNaN, x, y);
            return;
        }

        fixed (float* px = x, py = y)
        {
            ReluJob job = t_job ??= new ReluJob();
            job.Path = path;
            job.Canonical = canonicalNaN;
            job.X = px;
            job.Y = py;
            job.Length = x.Length;
            ParallelFor.Run(job, (x.Length + ChunkElements - 1) / ChunkElements, threads, x.Length);
        }
    }

    private sealed class ReluJob : ChunkJob
    {
        public KernelPath Path;
        public bool Canonical;
        public float* X;
        public float* Y;
        public int Length;

        public override void RunChunk(int chunk)
        {
            int start = chunk * ChunkElements;
            int len = Math.Min(ChunkElements, Length - start);
            ExactElementwise.Relu(Path, Canonical, new ReadOnlySpan<float>(X + start, len), new Span<float>(Y + start, len));
        }
    }
}
