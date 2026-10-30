using System;

namespace RecursiveAI.OneModel.Training.Data;

/// <summary>
/// Chooses which dataset indices make up each batch. Every method is a pure function of its arguments (and the sampler's constructor
/// arguments), callable from any thread in any order: that is what makes a run resumable and independent of the worker count.
/// </summary>
public interface IBatchSampler
{
    /// <summary>The largest number of indices <see cref="Fill"/> writes.</summary>
    int BatchSize { get; }

    /// <summary>The number of steps in one epoch, at least 1.</summary>
    int StepsPerEpoch { get; }

    /// <summary>Writes the dataset indices of batch (<paramref name="epoch"/>, <paramref name="stepInEpoch"/>) in batch order.</summary>
    /// <param name="epoch">Epoch number, at least 0.</param>
    /// <param name="stepInEpoch">Step within the epoch, in [0, <see cref="StepsPerEpoch"/>).</param>
    /// <param name="indices">Destination, at least <see cref="BatchSize"/> long.</param>
    /// <returns>The number of indices written, in [1, <see cref="BatchSize"/>].</returns>
    int Fill(long epoch, int stepInEpoch, Span<int> indices);
}

/// <summary>
/// Visits every sample once per epoch: in dataset order (<see cref="Sequential"/>) or in a fresh seeded order each epoch
/// (<see cref="Shuffled"/>). The order is a stateless keyed permutation, so there is no per-epoch table to build or to keep.
/// </summary>
public sealed class EpochSampler : IBatchSampler
{
    private readonly int _count;
    private readonly bool _shuffle;
    private readonly ulong _seed;
    private readonly bool _dropLast;

    private EpochSampler(int count, int batchSize, bool shuffle, ulong seed, bool dropLast)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(batchSize, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(count, 1);
        _count = count;
        BatchSize = batchSize;
        _shuffle = shuffle;
        _seed = seed;
        _dropLast = dropLast;
        if (dropLast && count < batchSize) throw new ArgumentException($"A dataset of {count} samples has no full batch of {batchSize}; use dropLast: false.", nameof(batchSize));
        StepsPerEpoch = dropLast ? count / batchSize : (int)(((long)count + batchSize - 1) / batchSize);
    }

    /// <summary>Samples in dataset order, epoch after epoch.</summary>
    /// <param name="count">Dataset size, at least 1.</param>
    /// <param name="batchSize">Indices per batch, at least 1.</param>
    /// <param name="dropLast">Skip the final short batch of each epoch (default true); otherwise it is shorter than <paramref name="batchSize"/>.</param>
    public static EpochSampler Sequential(int count, int batchSize, bool dropLast = true) => new(count, batchSize, false, 0, dropLast);

    /// <summary>Samples in a different seeded order each epoch.</summary>
    /// <param name="count">Dataset size, at least 1.</param>
    /// <param name="batchSize">Indices per batch, at least 1.</param>
    /// <param name="seed">The seed; the order of epoch e is a pure function of (seed, e).</param>
    /// <param name="dropLast">Skip the final short batch of each epoch (default true).</param>
    public static EpochSampler Shuffled(int count, int batchSize, ulong seed, bool dropLast = true) => new(count, batchSize, true, seed, dropLast);

    /// <inheritdoc/>
    public int BatchSize { get; }

    /// <inheritdoc/>
    public int StepsPerEpoch { get; }

    /// <inheritdoc/>
    public int Fill(long epoch, int stepInEpoch, Span<int> indices)
    {
        if ((uint)stepInEpoch >= (uint)StepsPerEpoch) throw new ArgumentOutOfRangeException(nameof(stepInEpoch));
        long first = (long)stepInEpoch * BatchSize;
        int n = (int)Math.Min(BatchSize, _count - first);
        if (indices.Length < n) throw new ArgumentException("The destination is shorter than the batch.", nameof(indices));
        ulong key = SampleRng.Derive(_seed, epoch, 0, 0, stream: 101);
        for (int i = 0; i < n; i++)
        {
            int p = (int)(first + i);
            indices[i] = _shuffle ? KeyedPermutation.Map(key, _count, p) : p;
        }
        return n;
    }
}

/// <summary>
/// Draws every batch slot from one of several sources with fixed probabilities (a "sources with shares per batch" recipe). Sources
/// are consecutive index ranges of one <see cref="ConcatDataset{T}"/>. Each slot is an independent draw (with replacement) keyed by
/// (seed, epoch, step, slot), so the shares hold in expectation over any run and a resumed run repeats exactly.
/// </summary>
public sealed class MixingSampler : IBatchSampler
{
    private readonly int[] _starts;
    private readonly int[] _sizes;
    private readonly double[] _cumulative;
    private readonly ulong _seed;

    /// <summary>Creates the sampler.</summary>
    /// <param name="sourceSizes">Sample count of each source, in dataset order.</param>
    /// <param name="shares">Relative weight of each source (need not sum to 1); at least one positive, none negative or NaN.</param>
    /// <param name="batchSize">Indices per batch, at least 1.</param>
    /// <param name="stepsPerEpoch">Steps that count as an epoch, at least 1; only changes which seed stream a step draws from.</param>
    /// <param name="seed">The seed.</param>
    /// <exception cref="ArgumentException">The arrays differ in length, a share is invalid, or a source with a positive share is empty.</exception>
    public MixingSampler(ReadOnlySpan<int> sourceSizes, ReadOnlySpan<double> shares, int batchSize, int stepsPerEpoch, ulong seed)
    {
        if (sourceSizes.Length == 0 || sourceSizes.Length != shares.Length) throw new ArgumentException("sourceSizes and shares must be the same non-zero length.");
        ArgumentOutOfRangeException.ThrowIfLessThan(batchSize, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(stepsPerEpoch, 1);
        _sizes = sourceSizes.ToArray();
        _starts = new int[_sizes.Length];
        _cumulative = new double[_sizes.Length];
        double total = 0;
        long offset = 0;
        for (int i = 0; i < _sizes.Length; i++)
        {
            double s = shares[i];
            if (double.IsNaN(s) || double.IsInfinity(s) || s < 0) throw new ArgumentException($"Share {i} is {s}; shares must be finite and not negative.");
            if (s > 0 && _sizes[i] < 1) throw new ArgumentException($"Source {i} has a positive share but no samples.");
            _starts[i] = checked((int)offset);
            offset += _sizes[i];
            total += s;
            _cumulative[i] = total;
        }
        if (total <= 0) throw new ArgumentException("At least one share must be positive.");
        for (int i = 0; i < _cumulative.Length; i++) _cumulative[i] /= total;
        BatchSize = batchSize;
        StepsPerEpoch = stepsPerEpoch;
        _seed = seed;
    }

    /// <inheritdoc/>
    public int BatchSize { get; }

    /// <inheritdoc/>
    public int StepsPerEpoch { get; }

    /// <inheritdoc/>
    public int Fill(long epoch, int stepInEpoch, Span<int> indices)
    {
        if ((uint)stepInEpoch >= (uint)StepsPerEpoch) throw new ArgumentOutOfRangeException(nameof(stepInEpoch));
        if (indices.Length < BatchSize) throw new ArgumentException("The destination is shorter than the batch.", nameof(indices));
        for (int slot = 0; slot < BatchSize; slot++)
        {
            var rng = new SampleRng(SampleRng.Derive(_seed, epoch, stepInEpoch, slot, stream: 102));
            double u = rng.NextDouble();
            int src = 0;
            while (src < _cumulative.Length - 1 && (u >= _cumulative[src] || _sizes[src] < 1)) src++;
            indices[slot] = _starts[src] + rng.NextInt(_sizes[src]);
        }
        return BatchSize;
    }
}

/// <summary>
/// Groups samples of similar length (for example text-line images of similar width) so batches need little padding. Each epoch the
/// samples are shuffled, cut into pools of <c>batchSize × poolBatches</c>, each pool is sorted by length, cut into batches, and the
/// batches are shuffled. The plan of an epoch is built once (O(n log n)) and cached; it is a pure function of (seed, epoch).
/// </summary>
public sealed class BucketedSampler : IBatchSampler
{
    private readonly int[] _lengths;
    private readonly int _poolBatches;
    private readonly ulong _seed;
    private readonly object _lock = new();
    private long _planEpoch = -1;
    private int[]? _plan;
    private long _prevEpoch = -1;
    private int[]? _prevPlan;

    /// <summary>Creates the sampler. The short tail of each epoch (the last <c>count mod batchSize</c> shuffled samples) is dropped.</summary>
    /// <param name="lengths">The length (for example the width) of every sample, by dataset index, non-negative.</param>
    /// <param name="batchSize">Indices per batch, at least 1.</param>
    /// <param name="seed">The seed.</param>
    /// <param name="poolBatches">Batches per sorting pool, default 50: larger pools pad less but make batches less random.</param>
    /// <exception cref="ArgumentException">There are fewer samples than one batch or a length is negative.</exception>
    public BucketedSampler(ReadOnlySpan<int> lengths, int batchSize, ulong seed, int poolBatches = 50)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(batchSize, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(poolBatches, 1);
        if (lengths.Length < batchSize) throw new ArgumentException($"{lengths.Length} samples make no full batch of {batchSize}.", nameof(lengths));
        foreach (int l in lengths) if (l < 0) throw new ArgumentException("Lengths must not be negative.", nameof(lengths));
        _lengths = lengths.ToArray();
        BatchSize = batchSize;
        StepsPerEpoch = lengths.Length / batchSize;
        _poolBatches = poolBatches;
        _seed = seed;
    }

    /// <inheritdoc/>
    public int BatchSize { get; }

    /// <inheritdoc/>
    public int StepsPerEpoch { get; }

    /// <inheritdoc/>
    public int Fill(long epoch, int stepInEpoch, Span<int> indices)
    {
        if ((uint)stepInEpoch >= (uint)StepsPerEpoch) throw new ArgumentOutOfRangeException(nameof(stepInEpoch));
        if (indices.Length < BatchSize) throw new ArgumentException("The destination is shorter than the batch.", nameof(indices));
        int[] plan = PlanFor(epoch);
        plan.AsSpan(stepInEpoch * BatchSize, BatchSize).CopyTo(indices);
        return BatchSize;
    }

    private int[] PlanFor(long epoch)
    {
        lock (_lock)
        {
            if (_planEpoch == epoch) return _plan!;
            if (_prevEpoch == epoch) return _prevPlan!;
            int[] plan = Build(epoch);
            (_prevEpoch, _prevPlan) = (_planEpoch, _plan);
            (_planEpoch, _plan) = (epoch, plan);
            return plan;
        }
    }

    private int[] Build(long epoch)
    {
        int n = _lengths.Length, b = BatchSize, steps = StepsPerEpoch;
        int used = steps * b;
        ulong key = SampleRng.Derive(_seed, epoch, 0, 0, stream: 103);
        var shuffled = new int[used];
        for (int i = 0; i < used; i++) shuffled[i] = KeyedPermutation.Map(key, n, i);
        // Sort each pool by (length, index): unique keys, so the order does not depend on the sort algorithm.
        int pool = checked(b * _poolBatches);
        var keys = new long[pool];
        for (int start = 0; start < used; start += pool)
        {
            int len = Math.Min(pool, used - start);
            for (int i = 0; i < len; i++) { int idx = shuffled[start + i]; keys[i] = ((long)_lengths[idx] << 32) | (uint)idx; }
            Array.Sort(keys, 0, len);
            for (int i = 0; i < len; i++) shuffled[start + i] = (int)(keys[i] & 0xFFFFFFFFL);
        }
        ulong orderKey = SampleRng.Derive(_seed, epoch, 1, 0, stream: 104);
        var plan = new int[used];
        for (int s = 0; s < steps; s++)
            Array.Copy(shuffled, KeyedPermutation.Map(orderKey, steps, s) * b, plan, s * b, b);
        return plan;
    }
}
