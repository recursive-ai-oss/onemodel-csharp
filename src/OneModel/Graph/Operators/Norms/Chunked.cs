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

        ParallelFor.Run(new Job<TWork>(work), chunks, threads);
    }

    private sealed class Job<TWork> : ChunkJob
        where TWork : struct, IChunkWork
    {
        private readonly TWork _work;

        public Job(TWork work) => _work = work;

        public override void RunChunk(int chunk) => _work.RunChunk(chunk);
    }
}
