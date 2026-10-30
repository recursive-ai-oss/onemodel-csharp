using System;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Graph.Operators;

// Chunked execution for operators that split their rows into chunks whose size depends only on the shapes. The work of one chunk is a
// struct method, so a run on one thread (or with a single chunk) calls it directly and allocates nothing; only a real parallel call
// wraps the struct in a job object for ParallelFor.

/// <summary>Work that is done in independent chunks, each writing only its own part of the output.</summary>
internal interface IChunkWork
{
    /// <summary>Runs chunk <paramref name="chunk"/>.</summary>
    void RunChunk(int chunk);
}

internal static class Chunked
{
    // Rough single-thread time of one chunk, in nanoseconds; only decides how many threads are worth using.
    private const long WorkPerChunkNs = 16_000;

    /// <summary>Runs chunks <c>[0, chunks)</c> of <paramref name="work"/> on up to <paramref name="threads"/> threads.</summary>
    public static void Run<TWork>(TWork work, int chunks, int threads)
        where TWork : struct, IChunkWork
    {
        if (threads <= 1 || chunks <= 1)
        {
            for (int c = 0; c < chunks; c++)
            {
                work.RunChunk(c);
            }

            return;
        }

        // A chunk is about ChunkElements elements of streaming work; a few of them are not worth waking helpers for.
        ParallelFor.Run(Job<TWork>.Get(work), chunks, threads, chunks * WorkPerChunkNs);
    }

    private sealed class Job<TWork> : ChunkJob
        where TWork : struct, IChunkWork
    {
        // Kept per thread and refilled, so a parallel call allocates nothing in steady state; the call is done with it before it returns.
        [ThreadStatic]
        private static Job<TWork>? t_job;

        private TWork _work;

        public static Job<TWork> Get(TWork work)
        {
            Job<TWork> job = t_job ??= new Job<TWork>();
            job._work = work;
            return job;
        }

        public override void RunChunk(int chunk) => _work.RunChunk(chunk);
    }
}
