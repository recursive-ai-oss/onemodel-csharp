using System.Security.Cryptography;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Training.Data;

namespace RecursiveAI.OneModel.Tests.Training;

public class DataLoadingTests
{
    private static List<ImageExample> MakeImages(int n, int height = 8, int minW = 5, int maxW = 40, int channels = 1)
    {
        var rng = new SampleRng(12345);
        var list = new List<ImageExample>(n);
        for (int i = 0; i < n; i++)
        {
            int w = minW + rng.NextInt(maxW - minW + 1);
            var px = new float[channels * height * w];
            for (int k = 0; k < px.Length; k++) px[k] = (i * 31 + k) % 17 / 17f;
            var target = new int[rng.NextInt(5)];
            for (int k = 0; k < target.Length; k++) target[k] = 1 + rng.NextInt(9);
            list.Add(new ImageExample(px, channels, height, w, target, i % 3));
        }
        return list;
    }

    private static ImageAugment Jitter => (Span<float> px, int c, int h, int w, ref SampleRng rng) =>
    {
        float gain = rng.NextSingle(0.5f, 1.5f), noise = rng.NextNormal() * 0.01f;
        for (int i = 0; i < px.Length; i++) px[i] = px[i] * gain + noise;
    };

    private static SampleTransform<ImageExample> Flip => (ImageExample s, ref SampleRng rng) =>
    {
        if (rng.NextInt(2) == 0) return s;
        // Reverse the targets: allocates, as a user transform may.
        int[] t = s.Target.ToArray();
        Array.Reverse(t);
        return s with { Target = t };
    };

    private static void HashBatch(IncrementalHash h, ImageBatch b)
    {
        h.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(b.Images.Data<float>()));
        foreach (Tensor t in new[] { b.Widths, b.Targets, b.TargetLengths, b.Labels, b.Indices })
            h.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(t.Data<int>()));
        h.AppendData(BitConverter.GetBytes(b.Epoch));
        h.AppendData(BitConverter.GetBytes(b.Step));
    }

    private static string StreamHash(IBatchSampler sampler, IDataset<ImageExample> ds, int workers, int prefetch, long first, long end, int multiple = 4)
    {
        var loader = new DataLoader<ImageExample, ImageBatch>(ds, sampler, new ImageBatchCollator(multiple, 0.5f, Jitter),
            new DataLoaderOptions { Seed = 77, Workers = workers, Prefetch = prefetch }, Flip);
        using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        using BatchSequence<ImageBatch> s = loader.Start(first, end);
        while (s.TryNext(out ImageBatch? b)) { HashBatch(h, b); b.Dispose(); }
        return Convert.ToHexString(h.GetHashAndReset());
    }

    [Theory]
    [InlineData(1)] [InlineData(2)] [InlineData(3)] [InlineData(7)] [InlineData(100)] [InlineData(1000)] [InlineData(4097)]
    public void KeyedPermutationIsBijective(int n)
    {
        var seen = new bool[n];
        var idx = new int[n];
        Assert.Equal(n, EpochSampler.Shuffled(n, n, 99).Fill(0, 0, idx));
        foreach (int p in idx)
        {
            Assert.InRange(p, 0, n - 1);
            Assert.False(seen[p]);
            seen[p] = true;
        }
    }

    [Fact]
    public void BatchSequenceIsIdenticalAcrossWorkersPrefetchAndResume()
    {
        var ds = new ListDataset<ImageExample>(MakeImages(103));
        var sampler = EpochSampler.Shuffled(103, 8, seed: 5);   // 12 steps per epoch: crosses epochs
        const long total = 40;
        string reference = StreamHash(sampler, ds, 0, 1, 0, total);
        foreach ((int w, int p) in new[] { (1, 1), (1, 4), (4, 1), (4, 3), (4, 16), (2, 8) })
            Assert.Equal(reference, StreamHash(sampler, ds, w, p, 0, total));

        // Resume: the tail from step k equals the tail of the uninterrupted run, at any worker count.
        const long k = 17;
        string tail = StreamHash(sampler, ds, 0, 1, k, total);
        Assert.Equal(tail, StreamHash(sampler, ds, 3, 5, k, total));
        // And it is the tail: hashing the first k then the rest in one hasher reproduces the whole.
        var loader = new DataLoader<ImageExample, ImageBatch>(ds, sampler, new ImageBatchCollator(4, 0.5f, Jitter), new DataLoaderOptions { Seed = 77, Workers = 0 }, Flip);
        using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        for (long g = 0; g < total; g++) { using ImageBatch b = loader.Load(g); HashBatch(h, b); }
        Assert.Equal(reference, Convert.ToHexString(h.GetHashAndReset()));
    }

    [Fact]
    public void MixingAndBucketedStreamsAreScheduleIndependent()
    {
        var parts = new[] { new ListDataset<ImageExample>(MakeImages(30)), new ListDataset<ImageExample>(MakeImages(50)), new ListDataset<ImageExample>(MakeImages(20)) };
        var ds = new ConcatDataset<ImageExample>(parts);
        var mix = new MixingSampler([30, 50, 20], [3, 1, 1], 8, 10, seed: 9);
        Assert.Equal(StreamHash(mix, ds, 0, 1, 0, 25), StreamHash(mix, ds, 4, 6, 0, 25));
        var lengths = Enumerable.Range(0, 100).Select(i => ds.GetSample(i).Width).ToArray();
        var bucket = new BucketedSampler(lengths, 8, seed: 3, poolBatches: 4);
        Assert.Equal(StreamHash(bucket, ds, 0, 1, 0, 30), StreamHash(bucket, ds, 4, 3, 0, 30));
    }

    [Fact]
    public void MixingSamplerHitsItsShares()
    {
        int[] sizes = [400, 300, 300, 50];
        double[] shares = [0.5, 0.3, 0.2, 0];
        var s = new MixingSampler(sizes, shares, 100, 100, seed: 1);
        var counts = new int[4];
        var idx = new int[100];
        int[] starts = [0, 400, 700, 1000];
        for (int step = 0; step < 100; step++)   // 10,000 draws
        {
            Assert.Equal(100, s.Fill(0, step, idx));
            foreach (int i in idx) { int src = i >= 1000 ? 3 : i >= 700 ? 2 : i >= 400 ? 1 : 0; counts[src]++; Assert.InRange(i, starts[src], starts[src] + sizes[src] - 1); }
        }
        Assert.Equal(0, counts[3]);
        for (int j = 0; j < 3; j++)
        {
            double p = shares[j], sd = Math.Sqrt(10000 * p * (1 - p));
            Assert.InRange(counts[j], 10000 * p - 4 * sd, 10000 * p + 4 * sd);
        }
        // Pure: the same step gives the same indices.
        var again = new int[100]; s.Fill(0, 7, again); var first = new int[100]; s.Fill(0, 7, first);
        Assert.Equal(first, again);
    }

    [Fact]
    public void ShuffledSamplerVisitsEverySampleOncePerEpochAndDiffersBetweenEpochs()
    {
        var s = EpochSampler.Shuffled(50, 8, 4, dropLast: false);
        Assert.Equal(7, s.StepsPerEpoch);
        var order = new List<int>[2];
        for (int e = 0; e < 2; e++)
        {
            order[e] = [];
            var buf = new int[8];
            for (int st = 0; st < 7; st++) { int n = s.Fill(e, st, buf); Assert.Equal(st == 6 ? 2 : 8, n); order[e].AddRange(buf.Take(n)); }
            Assert.Equal(Enumerable.Range(0, 50), order[e].OrderBy(x => x));
        }
        Assert.NotEqual(order[0], order[1]);
        var seq = new int[8]; EpochSampler.Sequential(16, 8).Fill(3, 1, seq);
        Assert.Equal(Enumerable.Range(8, 8), seq);
    }

    [Fact]
    public void BucketedSamplerCutsPadding()
    {
        var images = MakeImages(400, minW: 5, maxW: 200);
        int[] lengths = images.Select(i => i.Width).ToArray();
        long Padding(IBatchSampler s)
        {
            long waste = 0; var buf = new int[16];
            for (int st = 0; st < s.StepsPerEpoch; st++)
            {
                s.Fill(0, st, buf);
                int max = buf.Max(i => lengths[i]);
                waste += buf.Sum(i => max - lengths[i]);
            }
            return waste;
        }
        var bucket = new BucketedSampler(lengths, 16, 1, poolBatches: 10);
        long a = Padding(bucket), b = Padding(EpochSampler.Shuffled(400, 16, 1));
        Assert.True(a * 3 < b, $"bucketed waste {a} vs shuffled {b}");
        // Every kept sample appears once per epoch.
        var seen = new HashSet<int>(); var buf2 = new int[16];
        for (int st = 0; st < bucket.StepsPerEpoch; st++) { bucket.Fill(2, st, buf2); foreach (int i in buf2) Assert.True(seen.Add(i)); }
        Assert.Equal(400, seen.Count);
    }

    [Fact]
    public void CollatorPadsAndConcatenatesTargets()
    {
        ImageExample Make(int w, float v, int[] t) { var p = new float[2 * 3 * w]; Array.Fill(p, v); return new ImageExample(p, 2, 3, w, t, (int)v); }
        ImageExample[] s = [Make(5, 1, [3, 4]), Make(9, 2, []), Make(2, 3, [7])];
        using ImageBatch b = new ImageBatchCollator(widthMultiple: 8, padValue: -1f).Collate(s, [10, 11, 12], new BatchContext(0, 0, 0, 0));
        Assert.Equal(Shape.Create(3, 2, 3, 16), b.Images.Shape);
        Assert.True(b.Widths.Data<int>().SequenceEqual([5, 9, 2]));
        Assert.True(b.Targets.Data<int>().SequenceEqual([3, 4, 7]));
        Assert.True(b.TargetLengths.Data<int>().SequenceEqual([2, 0, 1]));
        Assert.True(b.Labels.Data<int>().SequenceEqual([1, 2, 3]));
        Assert.True(b.Indices.Data<int>().SequenceEqual([10, 11, 12]));
        ReadOnlySpan<float> img = b.Images.Data<float>();
        for (int i = 0; i < 3; i++)
            for (int row = 0; row < 6; row++)
                for (int x = 0; x < 16; x++)
                    Assert.Equal(x < s[i].Width ? i + 1 : -1f, img[(i * 6 + row) * 16 + x]);
        Assert.Throws<ArgumentException>(() => new ImageBatchCollator().Collate([Make(5, 1, []), new ImageExample(new float[3 * 2 * 4], 3, 2, 4)], [0, 1], new BatchContext(0, 0, 0, 0)));
    }

    [Fact]
    public void PerSampleSeedsDependOnPositionOnly()
    {
        var ctx = new BatchContext(7, 2, 3, 40);
        SampleRng a = ctx.Rng(5), b = ctx.Rng(5), c = ctx.Rng(6), d = ctx.Rng(5, 1);
        ulong x = a.NextUInt64();
        Assert.Equal(x, b.NextUInt64());
        Assert.NotEqual(x, c.NextUInt64());
        Assert.NotEqual(x, d.NextUInt64());
        var r = new SampleRng(1);
        for (int i = 0; i < 1000; i++) { Assert.InRange(r.NextInt(7), 0, 6); float f = r.NextSingle(); Assert.True(f >= 0 && f < 1); }
    }

    [Fact]
    public void DatasetErrorSurfacesAtItsTurn()
    {
        var ds = new MapDataset<int, ImageExample>(new ListDataset<int>(Enumerable.Range(0, 64).ToList()), i =>
            i >= 24 && i < 32 ? throw new InvalidOperationException("bad sample") : new ImageExample(new float[16], 1, 4, 4));
        var loader = new DataLoader<ImageExample, ImageBatch>(ds, EpochSampler.Sequential(64, 8), new ImageBatchCollator(), new DataLoaderOptions { Workers = 3, Prefetch = 6 });
        using BatchSequence<ImageBatch> s = loader.Start();
        for (int i = 0; i < 3; i++) { Assert.True(s.TryNext(out ImageBatch? b)); b.Dispose(); }
        Assert.Throws<InvalidOperationException>(() => s.TryNext(out _));
        Assert.Throws<ObjectDisposedException>(() => s.TryNext(out _));
    }

    [Fact]
    public void RecordFileDatasetReadsRecords()
    {
        string path = Path.Combine(Path.GetTempPath(), "onemodel-records-" + Guid.NewGuid().ToString("N"));
        try
        {
            var bytes = new byte[4 + 10 * 6];
            for (int i = 0; i < bytes.Length; i++) bytes[i] = (byte)i;
            File.WriteAllBytes(path, bytes);
            // Closed before the file is rewritten: Windows refuses to write a file another handle has open.
            using (var ds = new RecordFileDataset<int>(path, 6, (r, i) => r[0] * 1000 + i, headerBytes: 4))
            {
                Assert.Equal(10, ds.Count);
                Assert.Equal((4 + 6 * 3) * 1000 + 3, ds.GetSample(3));
                Assert.Throws<ArgumentOutOfRangeException>(() => ds.GetSample(10));
            }
            File.WriteAllBytes(path, new byte[11]);
            Assert.Throws<InvalidDataException>(() => new RecordFileDataset<int>(path, 6, (r, i) => 0));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void ImageHelpersConvertPixels()
    {
        byte[] rgb = [255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30];  // 2x2
        var dst = new float[12];
        ImageTensors.Rgb8ToPlanar(rgb, 2, 2, dst, [0.5f, 0.5f, 0.5f], [0.5f, 0.5f, 0.5f]);
        Assert.Equal(1f, dst[0]); Assert.Equal(-1f, dst[1]); Assert.Equal(-1f, dst[2]);      // red plane
        Assert.Equal(-1f, dst[4]); Assert.Equal(1f, dst[5]);                                  // green plane
        byte[] gray = [0, 255, 99, 99, 51, 102];   // 2x2 with stride 4 (two bytes of row padding)
        using Tensor t = ImageTensors.Gray8ToTensor(gray, 2, 2);
        Assert.Equal(Shape.Create(1, 2, 2), t.Shape);
        Assert.Equal(0f, t.Data<float>()[0]); Assert.Equal(1f, t.Data<float>()[1]); Assert.Equal(99 / 255f, t.Data<float>()[2], 1e-6f); Assert.Equal(99 / 255f, t.Data<float>()[3], 1e-6f);
        var strided = new float[4];
        ImageTensors.Gray8ToPlanar(new byte[] { 0, 255, 7, 7, 51, 102 }, 2, 2, strided, stride: 4);
        Assert.Equal(0f, strided[0]); Assert.Equal(1f, strided[1]); Assert.Equal(0.2f, strided[2], 1e-6f); Assert.Equal(0.4f, strided[3], 1e-6f);
        Assert.Throws<ArgumentException>(() => ImageTensors.Gray8ToPlanar(new byte[3], 2, 2, new float[4]));
    }

    [Fact]
    public void SteadyStateLoadingAllocatesNothing()
    {
        var ds = new ListDataset<ImageExample>(MakeImages(64));
        var loader = new DataLoader<ImageExample, ImageBatch>(ds, EpochSampler.Shuffled(64, 8, 1), new ImageBatchCollator(4, 0f, Jitter),
            new DataLoaderOptions { Workers = 0 });
        using BatchSequence<ImageBatch> s = loader.Start();
        for (int i = 0; i < 400; i++) { s.TryNext(out ImageBatch? w); w.Dispose(); }
        // The minimum over three windows: a late tier-up of a method by the JIT can allocate once on this thread while other
        // tests load the machine, which is not the loader allocating per batch.
        long per = long.MaxValue;
        for (int window = 0; window < 3 && per != 0; window++)
        {
            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 100; i++) { s.TryNext(out ImageBatch? b); b.Dispose(); }
            per = Math.Min(per, (GC.GetAllocatedBytesForCurrentThread() - before) / 100);
        }
        Assert.Equal(0, per);
    }

    [Fact]
    public void WithoutThreadSupportTheLoaderRunsInlineWithTheSameBatches()
    {
        var ds = new ListDataset<ImageExample>(MakeImages(50));
        var sampler = EpochSampler.Shuffled(50, 8, 2);
        string threaded = StreamHash(sampler, ds, 3, 4, 0, 14);
        Assert.Equal(0, WorkerPolicy.Effective(0));
        Assert.Equal(3, WorkerPolicy.Effective(3));
        try
        {
            WorkerPolicy.ForceNoThreads = true;
            Assert.Equal(0, WorkerPolicy.Effective(3));
            var before = System.Diagnostics.Process.GetCurrentProcess().Threads.Count;
            Assert.Equal(threaded, StreamHash(sampler, ds, 3, 4, 0, 14));
            Assert.True(System.Diagnostics.Process.GetCurrentProcess().Threads.Count <= before + 1);
        }
        finally { WorkerPolicy.ForceNoThreads = null; }
    }
}
