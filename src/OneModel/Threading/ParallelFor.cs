using System;
using System.Threading;

namespace RecursiveAI.OneModel.Threading;

/// <summary>
/// A unit of parallel work split into chunks. A chunk reads shared inputs and writes only its own part of the output (or its own
/// slot of a partial-result array that the caller combines afterwards in chunk-index order), so the result does not depend on
/// which thread ran which chunk, or on how many threads there were.
/// </summary>
internal abstract class ChunkJob
{
    /// <summary>Runs chunk <paramref name="chunk"/>, in <c>[0, chunkCount)</c>. Called once per chunk, from any thread.</summary>
    public abstract void RunChunk(int chunk);
}

/// <summary>
/// A fixed-chunk parallel-for over a small pool of dedicated worker threads. The caller decides the chunk count from the problem
/// shape alone (never from the thread count); this class only decides which thread picks up which chunk, which is not observable
/// in the results. Helpers are created on first use and live for the life of the process.
/// </summary>
internal static class ParallelFor
{
    private const int MaxThreads = 256;
    private const int SpinsBeforeSleep = 2000;

    private static readonly object s_gate = new();
    private static readonly Thread?[] s_workers = new Thread?[MaxThreads];
    private static int s_workerCount;
    private static int s_busy;

    // Job state, written by the caller before the generation is bumped.
    // The job, its chunk count and its generation travel together so a late helper never mixes the parts of two jobs.
    private sealed record Current(ChunkJob Job, int Chunks, long Generation);

    private static Current? s_current;
    // Next chunk to claim, with the job's generation in the high 32 bits. A helper that wakes late, after its job is finished and
    // maybe another has begun, fails the claim (the generation differs) and goes back to waiting, so it can never run a chunk of a
    // job it did not join.
    private static long s_next;
    private static int s_done;
    // Generation in the high 32 bits, helper count in the low 32: one atomic read tells a worker both whether there is new work and
    // whether it is wanted, so a worker can never join a job twice.
    private static long s_state;
    private static Exception? s_error;

    /// <summary>The thread count used when the caller does not choose one: the machine's logical processor count. It affects speed only.</summary>
    public static int DefaultThreads => Math.Min(Environment.ProcessorCount, MaxThreads);

    /// <summary>
    /// Runs every chunk in <c>[0, chunks)</c> of <paramref name="job"/> using at most <paramref name="threads"/> threads, the calling
    /// thread included, and returns when all have finished. If the pool is already serving another caller, the chunks run on the
    /// calling thread alone.
    /// </summary>
    /// <param name="job">The work.</param>
    /// <param name="chunks">Chunk count, from the problem shape only. Zero or negative runs nothing.</param>
    /// <param name="threads">Upper bound on threads; 1 runs everything inline.</param>
    public static void Run(ChunkJob job, int chunks, int threads)
    {
        ArgumentNullException.ThrowIfNull(job);
        if (chunks <= 0)
        {
            return;
        }

        int helpers = Math.Min(Math.Min(threads, MaxThreads) - 1, chunks - 1);
        if (helpers <= 0 || Interlocked.CompareExchange(ref s_busy, 1, 0) != 0)
        {
            for (int i = 0; i < chunks; i++)
            {
                job.RunChunk(i);
            }

            return;
        }

        try
        {
            EnsureWorkers(helpers);
            long generation = (Volatile.Read(ref s_state) >> 32) + 1;
            Volatile.Write(ref s_current, new Current(job, chunks, generation));
            s_error = null;
            Volatile.Write(ref s_done, 0);
            Volatile.Write(ref s_next, generation << 32);
            Volatile.Write(ref s_state, (generation << 32) | (uint)helpers);
            lock (s_gate)
            {
                Monitor.PulseAll(s_gate);
            }

            DrainCurrent(s_current);

            // Wait for the chunks to finish, not for the helpers to arrive: a helper that was descheduled and never joined must not
            // hold the caller up.
            var wait = new SpinWait();
            while (Volatile.Read(ref s_done) != chunks)
            {
                wait.SpinOnce();
            }

            Exception? error = s_error;
            if (error is not null)
            {
                throw new InvalidOperationException("A parallel chunk failed.", error);
            }
        }
        finally
        {
            Volatile.Write(ref s_busy, 0);
        }
    }

    private static void DrainCurrent(Current? job0)
    {
        if (job0 is null)
        {
            return;
        }

        ChunkJob job = job0.Job;
        int chunks = job0.Chunks;
        long generation = job0.Generation;
        while (true)
        {
            long current = Volatile.Read(ref s_next);
            if ((current >> 32) != generation)
            {
                return;
            }

            int i = (int)(current & 0xFFFFFFFF);
            if (i >= chunks)
            {
                return;
            }

            if (Interlocked.CompareExchange(ref s_next, current + 1, current) != current)
            {
                continue;
            }

            try
            {
                job.RunChunk(i);
            }
            catch (Exception e)
            {
                Interlocked.CompareExchange(ref s_error, e, null);
            }

            Interlocked.Increment(ref s_done);
        }
    }

    private static void EnsureWorkers(int count)
    {
        if (Volatile.Read(ref s_workerCount) >= count)
        {
            return;
        }

        lock (s_gate)
        {
            while (s_workerCount < count)
            {
                int index = s_workerCount;
                // The worker starts "having seen" the current generation, so it joins the next job and never one already finished.
                long born = Volatile.Read(ref s_state) >> 32;
                var t = new Thread(() => WorkerLoop(index, born)) { IsBackground = true, Name = "OneModel worker " + index };
                s_workers[index] = t;
                s_workerCount++;
                t.Start();
            }
        }
    }

    private static void WorkerLoop(int index, long born)
    {
        long seen = born;
        while (true)
        {
            int spins = 0;
            while (Volatile.Read(ref s_state) >> 32 == seen)
            {
                if (++spins < SpinsBeforeSleep)
                {
                    Thread.SpinWait(20);
                    continue;
                }

                lock (s_gate)
                {
                    while (Volatile.Read(ref s_state) >> 32 == seen)
                    {
                        Monitor.Wait(s_gate);
                    }
                }
            }

            long state = Volatile.Read(ref s_state);
            seen = state >> 32;
            if (index < (int)(state & 0xFFFFFFFF))
            {
                DrainCurrent(Volatile.Read(ref s_current));
            }
        }
    }
}
