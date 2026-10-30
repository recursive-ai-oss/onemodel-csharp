using System;
using System.Collections.Generic;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Training.Data;

/// <summary>Where a batch sits in the run, and the per-sample random streams derived from it.</summary>
public readonly struct BatchContext
{
    private readonly ulong _seed;

    /// <summary>Describes one place in a run; the loader makes these, and custom collators' tests may too.</summary>
    /// <param name="seed">The run seed.</param>
    /// <param name="epoch">The epoch.</param>
    /// <param name="step">The step within the epoch.</param>
    /// <param name="globalStep">The step from the start of the run.</param>
    public BatchContext(ulong seed, long epoch, int step, long globalStep)
    {
        _seed = seed;
        Epoch = epoch;
        Step = step;
        GlobalStep = globalStep;
    }

    /// <summary>The epoch, at least 0.</summary>
    public long Epoch { get; }

    /// <summary>The step within the epoch.</summary>
    public int Step { get; }

    /// <summary>The step counted from the start of the run: <c>Epoch × StepsPerEpoch + Step</c>.</summary>
    public long GlobalStep { get; }

    /// <summary>The random generator of one sample slot. Two calls with the same arguments return generators that produce the same numbers.</summary>
    /// <param name="slot">Position of the sample in the batch.</param>
    /// <param name="stream">Independent stream number: 0 is the loader's sample transform, 1 the collator's augmentation, 2 and up are free for custom collators.</param>
    public SampleRng Rng(int slot, int stream = 0) => new(SampleRng.Derive(_seed, Epoch, Step, slot, stream));
}

/// <summary>Turns the samples of one batch into the batch object the training loop consumes.</summary>
/// <typeparam name="TSample">The sample type.</typeparam>
/// <typeparam name="TBatch">The batch type.</typeparam>
public interface IBatchCollator<TSample, out TBatch>
{
    /// <summary>Builds a batch. Called from several threads at once when the loader has workers, so it must not share unsynchronised state.</summary>
    /// <param name="samples">The (transformed) samples in batch order.</param>
    /// <param name="indices">The dataset index of each sample.</param>
    /// <param name="context">Position in the run and the per-slot random streams.</param>
    TBatch Collate(ReadOnlySpan<TSample> samples, ReadOnlySpan<int> indices, in BatchContext context);
}

/// <summary>
/// One image with an optional label sequence: planar float32 pixels (channel, row, column), plus the label sequence for CTC-style
/// training. Samples of one batch share channels and height and may differ in width.
/// </summary>
/// <param name="Pixels">At least <c>Channels × Height × Width</c> floats, row-major per channel. Not copied or modified by the loader.</param>
/// <param name="Channels">1 for grayscale, 3 for RGB.</param>
/// <param name="Height">Rows.</param>
/// <param name="Width">Columns of real content.</param>
/// <param name="Target">The label sequence (for CTC: the characters, no blank); may be empty.</param>
/// <param name="Label">A single class label, default 0.</param>
public readonly record struct ImageExample(ReadOnlyMemory<float> Pixels, int Channels, int Height, int Width, ReadOnlyMemory<int> Target = default, int Label = 0);

/// <summary>An in-place augmentation of one image in a contiguous scratch copy.</summary>
/// <param name="pixels">Planar <c>[channels, height, width]</c> floats of this sample only (a copy: the dataset's data is never modified).</param>
/// <param name="channels">Channels.</param>
/// <param name="height">Rows.</param>
/// <param name="width">Columns; the augmentation may not change it.</param>
/// <param name="rng">The sample's random stream, a pure function of (seed, epoch, step, slot).</param>
public delegate void ImageAugment(Span<float> pixels, int channels, int height, int width, ref SampleRng rng);

/// <summary>
/// A collated image batch. Dispose it after the training step: its tensors go back to the pool and the object is reused, which is what
/// keeps steady-state loading allocation-free.
/// </summary>
public sealed class ImageBatch : IDisposable
{
    private readonly ImageBatchCollator? _owner;

    internal ImageBatch(ImageBatchCollator owner) => _owner = owner;

    /// <summary>Float32 <c>[B, C, H, Wp]</c>, each image at the left, the rest filled with the pad value; <c>Wp</c> is the widest width rounded up to the collator's multiple.</summary>
    public Tensor Images { get; internal set; }

    /// <summary>Int32 <c>[B]</c>: the real width of each image. For a network that downsamples width by a factor f, the CTC input length is <c>ceil(width / f)</c>.</summary>
    public Tensor Widths { get; internal set; }

    /// <summary>Int32 <c>[sum of target lengths]</c>: every sample's target, concatenated in batch order (the CTC loss layout).</summary>
    public Tensor Targets { get; internal set; }

    /// <summary>Int32 <c>[B]</c>: the length of each sample's target.</summary>
    public Tensor TargetLengths { get; internal set; }

    /// <summary>Int32 <c>[B]</c>: each sample's class label.</summary>
    public Tensor Labels { get; internal set; }

    /// <summary>Int32 <c>[B]</c>: each sample's dataset index.</summary>
    public Tensor Indices { get; internal set; }

    /// <summary>The epoch this batch belongs to.</summary>
    public long Epoch { get; internal set; }

    /// <summary>The step within the epoch.</summary>
    public int Step { get; internal set; }

    /// <summary>The number of images.</summary>
    public int Count => Widths.IsEmpty ? 0 : Widths.Length;

    /// <summary>Returns the tensors to the pool. Safe to call twice.</summary>
    public void Dispose()
    {
        if (Images.IsEmpty) return;
        Images.Dispose();
        Widths.Dispose();
        Targets.Dispose();
        TargetLengths.Dispose();
        Labels.Dispose();
        Indices.Dispose();
        Images = Widths = Targets = TargetLengths = Labels = Indices = default;
        _owner?.Recycle(this);
    }
}

/// <summary>
/// Stacks <see cref="ImageExample"/>s into an <see cref="ImageBatch"/>: variable widths are padded to the widest image rounded up to
/// <see cref="WidthMultiple"/>, widths and CTC-style concatenated targets are recorded, and an optional in-place augmentation runs on
/// each sample with that sample's own seeded random stream. Thread-safe.
/// </summary>
public sealed class ImageBatchCollator : IBatchCollator<ImageExample, ImageBatch>
{
    private readonly BufferPool _pool;
    private readonly Stack<ImageBatch> _free = new();
    private readonly Stack<float[]> _scratch = new();

    /// <summary>Creates a collator.</summary>
    /// <param name="widthMultiple">The padded width is rounded up to a multiple of this (for example the network's total width stride); default 1.</param>
    /// <param name="padValue">The value of padding pixels, default 0.</param>
    /// <param name="augment">Optional in-place augmentation per sample; null for none.</param>
    /// <param name="pool">Pool for the batch tensors; default <see cref="BufferPool.Shared"/>.</param>
    public ImageBatchCollator(int widthMultiple = 1, float padValue = 0f, ImageAugment? augment = null, BufferPool? pool = null)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(widthMultiple, 1);
        WidthMultiple = widthMultiple;
        PadValue = padValue;
        Augment = augment;
        _pool = pool ?? BufferPool.Shared;
    }

    /// <summary>The padded width is a multiple of this.</summary>
    public int WidthMultiple { get; }

    /// <summary>The value of padding pixels.</summary>
    public float PadValue { get; }

    /// <summary>The augmentation, or null.</summary>
    public ImageAugment? Augment { get; }

    internal void Recycle(ImageBatch b)
    {
        lock (_free) _free.Push(b);
    }

    /// <inheritdoc/>
    /// <exception cref="ArgumentException">The batch is empty, samples differ in channels or height, a size is not positive, or pixels or targets are too short.</exception>
    public ImageBatch Collate(ReadOnlySpan<ImageExample> samples, ReadOnlySpan<int> indices, in BatchContext context)
    {
        int b = samples.Length;
        if (b == 0) throw new ArgumentException("A batch needs at least one sample.", nameof(samples));
        if (indices.Length != b) throw new ArgumentException("indices and samples differ in length.", nameof(indices));
        int c = samples[0].Channels, h = samples[0].Height, maxW = 0;
        long totalTarget = 0;
        for (int i = 0; i < b; i++)
        {
            ref readonly ImageExample s = ref samples[i];
            if (s.Channels != c || s.Height != h) throw new ArgumentException($"Sample {i} is {s.Channels}x{s.Height} but sample 0 is {c}x{h}; samples of a batch share channels and height.", nameof(samples));
            if (s.Channels < 1 || s.Height < 1 || s.Width < 1) throw new ArgumentException($"Sample {i} has a non-positive size.", nameof(samples));
            if ((long)s.Channels * s.Height * s.Width > s.Pixels.Length) throw new ArgumentException($"Sample {i} has {s.Pixels.Length} pixels, fewer than {s.Channels}x{s.Height}x{s.Width}.", nameof(samples));
            maxW = Math.Max(maxW, s.Width);
            totalTarget += s.Target.Length;
        }
        long padded = ((long)maxW + WidthMultiple - 1) / WidthMultiple * WidthMultiple;
        if (padded * c * h * b > int.MaxValue / 4) throw new ArgumentException("The padded batch would be larger than 2^31 bytes.", nameof(samples));
        int wp = (int)padded;

        ImageBatch batch;
        lock (_free) batch = _free.Count > 0 ? _free.Pop() : new ImageBatch(this);
        Tensor images = default, widths = default, targets = default, targetLengths = default, labels = default, idx = default;
        float[]? scratch = null;
        try
        {
            images = Tensor.Create(DType.F32, Shape.Create(b, c, h, wp), _pool);
            widths = Tensor.Create(DType.I32, Shape.Create(b), _pool);
            targets = Tensor.Create(DType.I32, Shape.Create(checked((int)totalTarget)), _pool);
            targetLengths = Tensor.Create(DType.I32, Shape.Create(b), _pool);
            labels = Tensor.Create(DType.I32, Shape.Create(b), _pool);
            idx = Tensor.Create(DType.I32, Shape.Create(b), _pool);
            Span<float> img = images.MutableData<float>();
            Span<int> wd = widths.MutableData<int>(), tg = targets.MutableData<int>(), tl = targetLengths.MutableData<int>(),
                lb = labels.MutableData<int>(), ix = idx.MutableData<int>();
            if (Augment is not null) scratch = RentScratch(samples);
            int tpos = 0;
            for (int i = 0; i < b; i++)
            {
                ref readonly ImageExample s = ref samples[i];
                ReadOnlySpan<float> src = s.Pixels.Span;
                if (Augment is not null)
                {
                    Span<float> copy = scratch!.AsSpan(0, s.Channels * s.Height * s.Width);
                    src.Slice(0, copy.Length).CopyTo(copy);
                    SampleRng rng = context.Rng(i, 1);
                    Augment(copy, s.Channels, s.Height, s.Width, ref rng);
                    src = copy;
                }
                int w = s.Width;
                for (int ch = 0; ch < c; ch++)
                    for (int y = 0; y < h; y++)
                    {
                        Span<float> row = img.Slice(((i * c + ch) * h + y) * wp, wp);
                        src.Slice((ch * h + y) * w, w).CopyTo(row);
                        row.Slice(w).Fill(PadValue);
                    }
                wd[i] = w;
                tl[i] = s.Target.Length;
                s.Target.Span.CopyTo(tg.Slice(tpos));
                tpos += s.Target.Length;
                lb[i] = s.Label;
                ix[i] = indices[i];
            }
        }
        catch
        {
            images.Dispose(); widths.Dispose(); targets.Dispose(); targetLengths.Dispose(); labels.Dispose(); idx.Dispose();
            Recycle(batch);
            throw;
        }
        finally
        {
            if (scratch is not null) lock (_scratch) _scratch.Push(scratch);
        }
        batch.Images = images; batch.Widths = widths; batch.Targets = targets;
        batch.TargetLengths = targetLengths; batch.Labels = labels; batch.Indices = idx;
        batch.Epoch = context.Epoch; batch.Step = context.Step;
        return batch;
    }

    private float[] RentScratch(ReadOnlySpan<ImageExample> samples)
    {
        int need = 0;
        foreach (ref readonly ImageExample s in samples) need = Math.Max(need, s.Channels * s.Height * s.Width);
        float[]? a = null;
        lock (_scratch) if (_scratch.Count > 0) a = _scratch.Pop();
        if (a is null || a.Length < need) a = new float[Math.Max(need, a is null ? 0 : a.Length * 2)];
        return a;
    }
}
