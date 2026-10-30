using System;
using System.Collections.Generic;
using System.Threading;

namespace RecursiveAI.OneModel.Training.Data;

/// <summary>Changes one sample before collation (for example an augmentation that builds a new sample). Must be pure given its arguments and thread-safe.</summary>
/// <typeparam name="TSample">The sample type.</typeparam>
/// <param name="sample">The sample from the dataset.</param>
/// <param name="rng">The sample's random stream, a pure function of (seed, epoch, step, slot).</param>
/// <returns>The transformed sample.</returns>
public delegate TSample SampleTransform<TSample>(TSample sample, ref SampleRng rng);

/// <summary>Settings of a <see cref="DataLoader{TSample, TBatch}"/>.</summary>
public sealed class DataLoaderOptions
{
    /// <summary>The run seed, default 0. Together with the epoch, step and slot it fixes every random choice.</summary>
    public ulong Seed { get; init; }

    /// <summary>Background threads that build batches, default 2; 0 builds each batch on the calling thread. Where the platform cannot start threads (browser WebAssembly, WASI) the loader uses 0 whatever this says. The batches are the same for every value.</summary>
    public int Workers { get; init; } = 2;

    /// <summary>How many batches may be built ahead of the consumer, default 4, at least 1 when there are workers. The batches are the same for every value.</summary>
    public int Prefetch { get; init; } = 4;
}

/// <summary>
/// Builds the batches of a training run from a dataset, a sampler and a collator. The batch at global step <c>g</c> (epoch
/// <c>g / StepsPerEpoch</c>, step <c>g % StepsPerEpoch</c>) is a pure function of the seed and <c>g</c>: it does not depend on the
/// number of workers, the prefetch depth, timing, or which steps were built before, so a run resumed at step k sees exactly the
/// batches an uninterrupted run sees from k on (<c>docs/data-loading.md</c>).
/// </summary>
/// <typeparam name="TSample">The sample type.</typeparam>
/// <typeparam name="TBatch">The batch type.</typeparam>
public sealed class DataLoader<TSample, TBatch>
{
    private readonly IDataset<TSample> _dataset;
    private readonly IBatchSampler _sampler;
    private readonly IBatchCollator<TSample, TBatch> _collator;
    private readonly SampleTransform<TSample>? _transform;
    private readonly DataLoaderOptions _options;

    /// <summary>Creates a loader.</summary>
    /// <param name="dataset">The samples. With workers, <c>Get</c> is called concurrently.</param>
    /// <param name="sampler">Which samples form each batch.</param>
    /// <param name="collator">How samples become a batch.</param>
    /// <param name="options">Seed, workers and prefetch; defaults when null.</param>
    /// <param name="transform">Optional per-sample transform run before collation.</param>
    /// <exception cref="ArgumentOutOfRangeException">Workers is negative, or Prefetch is below 1.</exception>
    public DataLoader(IDataset<TSample> dataset, IBatchSampler sampler, IBatchCollator<TSample, TBatch> collator,
        DataLoaderOptions? options = null, SampleTransform<TSample>? transform = null)
    {
        _dataset = dataset ?? throw new ArgumentNullException(nameof(dataset));
        _sampler = sampler ?? throw new ArgumentNullException(nameof(sampler));
        _collator = collator ?? throw new ArgumentNullException(nameof(collator));
        _options = options ?? new DataLoaderOptions();
        _transform = transform;
        ArgumentOutOfRangeException.ThrowIfNegative(_options.Workers);
        ArgumentOutOfRangeException.ThrowIfLessThan(_options.Prefetch, 1);
    }

    /// <summary>Steps per epoch, from the sampler.</summary>
    public int StepsPerEpoch => _sampler.StepsPerEpoch;

    /// <summary>Builds the batch at <paramref name="globalStep"/> on the calling thread, allocating its working buffers.</summary>
    /// <param name="globalStep">The step counted from the start of the run, at least 0.</param>
    public TBatch Load(long globalStep) => Produce(globalStep, new Buffers(_sampler.BatchSize));

    /// <summary>
    /// Starts producing batches from <paramref name="firstStep"/> (to resume a run at step k, pass k). Take them with
    /// <see cref="BatchSequence{TBatch}.TryNext"/> in order; dispose the stream to stop the workers.
    /// </summary>
    /// <param name="firstStep">Global step of the first batch, default 0.</param>
    /// <param name="endStep">Global step after the last batch (exclusive), default: endless.</param>
    /// <exception cref="ArgumentOutOfRangeException">A step is negative or <paramref name="endStep"/> is below <paramref name="firstStep"/>.</exception>
    public BatchSequence<TBatch> Start(long firstStep = 0, long endStep = long.MaxValue)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(firstStep);
        ArgumentOutOfRangeException.ThrowIfLessThan(endStep, firstStep);
        return new BatchSequence<TBatch>(new Producer(this), firstStep, endStep, WorkerPolicy.Effective(_options.Workers), _options.Prefetch);
    }

    private sealed class Buffers
    {
        public readonly TSample[] Samples;
        public readonly int[] Indices;

        public Buffers(int batchSize)
        {
            Samples = new TSample[batchSize];
            Indices = new int[batchSize];
        }
    }

    private sealed class Producer(DataLoader<TSample, TBatch> owner) : IBatchProducer<TBatch>
    {
        public object NewWorkerState() => new Buffers(owner._sampler.BatchSize);

        public TBatch Produce(long step, object state) => owner.Produce(step, (Buffers)state);
    }

    private TBatch Produce(long globalStep, Buffers buf)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(globalStep);
        int spe = _sampler.StepsPerEpoch;
        long epoch = globalStep / spe;
        int step = (int)(globalStep % spe);
        int n = _sampler.Fill(epoch, step, buf.Indices);
        if ((uint)(n - 1) >= (uint)buf.Indices.Length) throw new InvalidOperationException($"The sampler returned {n} indices for a batch size of {buf.Indices.Length}.");
        int count = _dataset.Count;
        var ctx = new BatchContext(_options.Seed, epoch, step, globalStep);
        for (int i = 0; i < n; i++)
        {
            int idx = buf.Indices[i];
            if ((uint)idx >= (uint)count) throw new InvalidOperationException($"The sampler chose index {idx} but the dataset has {count} samples (step {globalStep}, slot {i}).");
            TSample s = _dataset.GetSample(idx);
            if (_transform is not null)
            {
                SampleRng rng = ctx.Rng(i, 0);
                s = _transform(s, ref rng);
            }
            buf.Samples[i] = s;
        }
        try
        {
            return _collator.Collate(buf.Samples.AsSpan(0, n), buf.Indices.AsSpan(0, n), ctx);
        }
        finally
        {
            Array.Clear(buf.Samples, 0, n); // do not keep samples alive between batches
        }
    }
}

/// <summary>Decides how many worker threads a stream really starts.</summary>
internal static class WorkerPolicy
{
    /// <summary>Test switch: pretend the platform cannot start threads (as browser WebAssembly and WASI cannot).</summary>
    internal static bool? ForceNoThreads;

    internal static bool ThreadsSupported => !(ForceNoThreads ?? (OperatingSystem.IsBrowser() || OperatingSystem.IsWasi()));

    /// <summary>The requested count, or 0 where threads cannot be started. Batches are the same either way.</summary>
    internal static int Effective(int requested) => ThreadsSupported ? requested : 0;
}

/// <summary>Builds one batch by global step. Implementations must be thread-safe given distinct worker states.</summary>
internal interface IBatchProducer<out TBatch>
{
    object NewWorkerState();

    TBatch Produce(long step, object state);
}

/// <summary>
/// An ordered, bounded, prefetching sequence of batches. Worker threads build the steps in a window ahead of the consumer, in any
/// order and at any speed; <see cref="TryNext"/> always returns them in step order, so the sequence never depends on scheduling.
/// This file is the only place in the data loader that creates threads; no result depends on which thread built it.
/// </summary>
/// <typeparam name="TBatch">The batch type.</typeparam>
public sealed class BatchSequence<TBatch> : IDisposable
{
    private readonly IBatchProducer<TBatch> _producer;
    private readonly long _end;
    private readonly int _depth;
    private readonly object _gate = new();
    private readonly Thread[] _threads;
    private readonly Slot[] _ring;
    private readonly object? _syncState;
    private long _nextClaim;   // next step a worker will build
    private long _nextTake;    // next step the consumer will return
    private bool _stopped;

    private struct Slot
    {
        public bool Ready;
        public TBatch? Batch;
        public Exception? Error;
    }

    internal BatchSequence(IBatchProducer<TBatch> producer, long first, long end, int workers, int prefetch)
    {
        _producer = producer;
        _end = end;
        _nextClaim = _nextTake = first;
        _depth = prefetch;
        _ring = new Slot[prefetch];
        if (workers == 0)
        {
            _syncState = producer.NewWorkerState();
            _threads = [];
            return;
        }
        _threads = new Thread[workers];
        for (int i = 0; i < workers; i++)
        {
            object state = producer.NewWorkerState();
            _threads[i] = new Thread(() => Work(state)) { IsBackground = true, Name = "OneModel.DataLoader" };
            _threads[i].Start();
        }
    }

    /// <summary>The global step of the batch the next <see cref="TryNext"/> returns.</summary>
    public long NextStep
    {
        get { lock (_gate) return _nextTake; }
    }

    /// <summary>Returns the next batch in step order, waiting for it if it is not ready. Call from one thread.</summary>
    /// <param name="batch">The batch; the caller owns it (and disposes it, if the batch type needs that).</param>
    /// <returns>False after the last step.</returns>
    /// <exception cref="ObjectDisposedException">The stream was disposed.</exception>
    /// <exception cref="Exception">Whatever building the batch threw (dataset, sampler, transform or collator), rethrown at that batch's turn; the stream is then stopped.</exception>
    public bool TryNext(out TBatch batch)
    {
        batch = default!;
        if (_threads.Length == 0)
        {
            long step;
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_stopped, this);
                if (_nextTake >= _end) return false;
                step = _nextTake++;
            }
            batch = _producer.Produce(step, _syncState!);
            return true;
        }
        lock (_gate)
        {
            while (true)
            {
                ObjectDisposedException.ThrowIf(_stopped, this);
                if (_nextTake >= _end) return false;
                ref Slot slot = ref _ring[(int)(_nextTake % _depth)];
                if (slot.Ready)
                {
                    Exception? error = slot.Error;
                    batch = slot.Batch!;
                    slot = default;
                    _nextTake++;
                    Monitor.PulseAll(_gate);
                    if (error is not null)
                    {
                        _stopped = true;
                        Monitor.PulseAll(_gate);
                        System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(error).Throw();
                    }
                    return true;
                }
                Monitor.Wait(_gate);
            }
        }
    }

    private void Work(object state)
    {
        while (true)
        {
            long step;
            lock (_gate)
            {
                while (!_stopped && _nextClaim < _end && _nextClaim >= _nextTake + _depth) Monitor.Wait(_gate);
                if (_stopped || _nextClaim >= _end) return;
                step = _nextClaim++;
            }
            TBatch? batch = default;
            Exception? error = null;
            try { batch = _producer.Produce(step, state); }
            catch (Exception e) { error = e; }
            lock (_gate)
            {
                // A claimed step is always inside the window, so its slot is free: the consumer cannot have moved past it.
                _ring[(int)(step % _depth)] = new Slot { Ready = true, Batch = batch, Error = error };
                Monitor.PulseAll(_gate);
            }
        }
    }

    /// <summary>Stops the workers and waits for them. Batches still waiting in the window are disposed if they are <see cref="IDisposable"/>. Safe to call twice.</summary>
    public void Dispose()
    {
        lock (_gate)
        {
            if (_stopped && _threads.Length == 0) return;
            _stopped = true;
            Monitor.PulseAll(_gate);
        }
        foreach (Thread t in _threads) t.Join();
        lock (_gate)
        {
            for (int i = 0; i < _ring.Length; i++)
            {
                if (_ring[i].Batch is IDisposable d) d.Dispose();
                _ring[i] = default;
            }
        }
    }
}
