using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Diagnostics;
using RecursiveAI.OneModel.Kernels;

namespace RecursiveAI.OneModel.Benchmarks;

/// <summary>
/// Times each distinct convolution of ResNet-50 and MobileNetV3 (batch 1, fp32, fast mode) at several thread counts, then the
/// whole chain back to back as a model run would issue it. Usage: <c>convscale [--threads 1,8,16,32] [--net resnet|mobilenet] [--ms 300]</c>.
/// </summary>
internal static class ConvScale
{
    private sealed record Layer(string Name, ConvShape Shape, int Count);

    private static ConvShape C(int cin, int h, int cout, int k, int stride, int groups = 1) =>
        new(1, cin, h, h, cout, k, k, stride, stride, k / 2, k / 2, k / 2, k / 2, 1, 1, groups);

    private static Layer[] ResNet() =>
    [
        new("conv1 7x7s2 3>64 @224", new ConvShape(1, 3, 224, 224, 64, 7, 7, 2, 2, 3, 3, 3, 3, 1, 1, 1), 1),
        new("1x1 64>64 @56", C(64, 56, 64, 1, 1), 1),
        new("3x3 64>64 @56", C(64, 56, 64, 3, 1), 3),
        new("1x1 64>256 @56", C(64, 56, 256, 1, 1), 4),
        new("1x1 256>64 @56", C(256, 56, 64, 1, 1), 2),
        new("1x1 256>128 @56", C(256, 56, 128, 1, 1), 1),
        new("3x3 128>128 s2 @56", C(128, 56, 128, 3, 2), 1),
        new("1x1 256>512 s2 @56", C(256, 56, 512, 1, 2), 1),
        new("1x1 128>512 @28", C(128, 28, 512, 1, 1), 4),
        new("1x1 512>128 @28", C(512, 28, 128, 1, 1), 3),
        new("3x3 128>128 @28", C(128, 28, 128, 3, 1), 3),
        new("1x1 512>256 @28", C(512, 28, 256, 1, 1), 1),
        new("3x3 256>256 s2 @28", C(256, 28, 256, 3, 2), 1),
        new("1x1 512>1024 s2 @28", C(512, 28, 1024, 1, 2), 1),
        new("1x1 256>1024 @14", C(256, 14, 1024, 1, 1), 6),
        new("1x1 1024>256 @14", C(1024, 14, 256, 1, 1), 5),
        new("3x3 256>256 @14", C(256, 14, 256, 3, 1), 5),
        new("1x1 1024>512 @14", C(1024, 14, 512, 1, 1), 1),
        new("3x3 512>512 s2 @14", C(512, 14, 512, 3, 2), 1),
        new("1x1 1024>2048 s2 @14", C(1024, 14, 2048, 1, 2), 1),
        new("1x1 512>2048 @7", C(512, 7, 2048, 1, 1), 3),
        new("1x1 2048>512 @7", C(2048, 7, 512, 1, 1), 2),
        new("3x3 512>512 @7", C(512, 7, 512, 3, 1), 2),
    ];

    private static Layer[] MobileNet() =>
    [
        new("1x1 16>64 @112", C(16, 112, 64, 1, 1), 1),
        new("dw3x3 64 s2 @112", C(64, 112, 64, 3, 2, 64), 1),
        new("1x1 24>72 @56", C(24, 56, 72, 1, 1), 2),
        new("dw5x5 72 s2 @56", C(72, 56, 72, 5, 2, 72), 1),
        new("1x1 72>24 @28", C(72, 28, 24, 1, 1), 1),
        new("1x1 40>120 @28", C(40, 28, 120, 1, 1), 2),
        new("1x1 120>40 @28", C(120, 28, 40, 1, 1), 2),
        new("1x1 40>240 @28", C(40, 28, 240, 1, 1), 1),
        new("dw3x3 240 s2 @28", C(240, 28, 240, 3, 2, 240), 1),
        new("1x1 80>200 @14", C(80, 14, 200, 1, 1), 2),
        new("dw3x3 200 @14", C(200, 14, 200, 3, 1, 200), 2),
        new("1x1 200>80 @14", C(200, 14, 80, 1, 1), 2),
        new("1x1 80>480 @14", C(80, 14, 480, 1, 1), 1),
        new("dw3x3 480 @14", C(480, 14, 480, 3, 1, 480), 2),
        new("1x1 480>112 @14", C(480, 14, 112, 1, 1), 1),
        new("1x1 112>672 @14", C(112, 14, 672, 1, 1), 3),
        new("dw5x5 672 s2 @14", C(672, 14, 672, 5, 2, 672), 1),
        new("dw5x5 672 @7", C(672, 7, 672, 5, 1, 672), 1),
        new("1x1 672>160 @7", C(672, 7, 160, 1, 1), 1),
        new("1x1 160>960 @7", C(160, 7, 960, 1, 1), 2),
        new("1x1 960>160 @7", C(960, 7, 160, 1, 1), 2),
        new("dw5x5 960 @7", C(960, 7, 960, 5, 1, 960), 2),
        new("1x1 960>1280 @1", new ConvShape(1, 960, 1, 1, 1280, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1), 1),
    ];

    private sealed class Spin : RecursiveAI.OneModel.Threading.ChunkJob
    {
        public int Ns;
        public override void RunChunk(int chunk)
        {
            long t0 = Stopwatch.GetTimestamp();
            long ticks = Ns * Stopwatch.Frequency / 1_000_000_000L;
            while (Stopwatch.GetTimestamp() - t0 < ticks) { }
        }
    }

    /// <summary>Cost of one parallel call: empty chunks and chunks of a few microseconds, at several thread counts.</summary>
    public static void PoolCost(int[] threads)
    {
        var job = new Spin();
        foreach (int chunkNs in new[] { 0, 2000, 10000 })
        {
            job.Ns = chunkNs;
            foreach (int t in threads)
            {
                int chunks = 48;
                for (int i = 0; i < 2000; i++) RecursiveAI.OneModel.Threading.ParallelFor.Run(job, chunks, t);
                double best = double.MaxValue;
                for (int b = 0; b < 20; b++)
                {
                    long t0 = Stopwatch.GetTimestamp();
                    for (int i = 0; i < 500; i++) RecursiveAI.OneModel.Threading.ParallelFor.Run(job, chunks, t);
                    best = Math.Min(best, (Stopwatch.GetTimestamp() - t0) * 1e6 / Stopwatch.Frequency / 500);
                }

                double ideal = chunkNs / 1000.0 * Math.Ceiling(chunks / (double)t);
                Console.WriteLine($"poolcost chunk={chunkNs}ns chunks=48 threads={t}: {best:F2} us per call (ideal {ideal:F2})");
            }
        }
    }

    public static int Run(string[] args)
    {
        if (args.Length > 0 && args[0] == "poolcost")
        {
            PoolCost(args.Length > 1 ? args[1].Split(',').Select(int.Parse).ToArray() : [1, 2, 4, 8, 16, 32]);
            return 0;
        }

        int[] threads = [1, 8, 16, 32];
        string net = "resnet";
        int targetMs = 300;
        for (int i = 0; i + 1 < args.Length; i += 2)
        {
            switch (args[i])
            {
                case "--threads": threads = args[i + 1].Split(',').Select(int.Parse).ToArray(); break;
                case "--net": net = args[i + 1]; break;
                case "--ms": targetMs = int.Parse(args[i + 1]); break;
            }
        }

        Layer[] layers = net == "mobilenet" ? MobileNet() : ResNet();
        KernelPath path = KernelPaths.Best();
        Console.WriteLine($"# path {path}, net {net}, load {File.ReadAllText("/proc/loadavg").Trim()}");
        Console.WriteLine("layer\tcount\tGFLOP\t" + string.Join("\t", threads.Select(t => $"ms@{t}")) + "\t" + string.Join("\t", threads.Skip(1).Select(t => $"x@{t}")));

        var prepared = new List<(Layer L, float[] In, float[] W, float[] Packed, float[] Out, float[] Bias)>();
        var rng = new Random(7);
        foreach (Layer l in layers)
        {
            ConvShape s = l.Shape;
            var input = new float[(long)s.N * s.Cin * s.H * s.W];
            var w = new float[(long)s.Cout * (s.Cin / s.Groups) * s.KH * s.KW];
            for (int i = 0; i < input.Length; i++) input[i] = (float)rng.NextDouble() - 0.5f;
            for (int i = 0; i < w.Length; i++) w[i] = ((float)rng.NextDouble() - 0.5f) * 0.1f;
            float[] packed = [];
            if (!s.IsDepthwise)
            {
                packed = new float[ConvPacked.PackedWeightsLength(path, s)];
                ConvPacked.PackWeights(path, s, w, packed);
            }

            prepared.Add((l, input, w, packed, new float[(long)s.N * s.Cout * s.OutH * s.OutW], new float[s.Cout]));
        }

        var totals = new double[threads.Length];
        var totalsRelu = new double[threads.Length];
        foreach (var p in prepared)
        {
            var times = new double[threads.Length];
            for (int ti = 0; ti < threads.Length; ti++)
            {
                times[ti] = Time(p, path, threads[ti], targetMs);
                totals[ti] += times[ti] * p.L.Count;
            }

            Console.WriteLine($"{p.L.Name}\t{p.L.Count}\t{p.L.Shape.Flops / 1e9:F3}\t" + string.Join("\t", times.Select(t => (t * 1e3).ToString("F3"))) +
                "\t" + string.Join("\t", times.Skip(1).Select(t => (times[0] / t).ToString("F2"))));
        }

        Console.WriteLine("TOTAL(weighted by count)\t\t\t" + string.Join("\t", totals.Select(t => (t * 1e3).ToString("F2"))) + "\t" + string.Join("\t", totals.Skip(1).Select(t => (totals[0] / t).ToString("F2"))));

        // The whole chain back to back, as a model run issues it.
        var chain = new double[threads.Length];
        for (int ti = 0; ti < threads.Length; ti++)
        {
            double best = double.MaxValue;
            for (int rep = 0; rep < 12; rep++)
            {
                long t0 = Stopwatch.GetTimestamp();
                foreach (var p in prepared)
                {
                    for (int c = 0; c < p.L.Count; c++) One(p, path, threads[ti]);
                }

                best = Math.Min(best, (Stopwatch.GetTimestamp() - t0) / (double)Stopwatch.Frequency);
            }

            chain[ti] = best;
        }

        Console.WriteLine("CHAIN ms\t\t\t" + string.Join("\t", chain.Select(t => (t * 1e3).ToString("F2"))) + "\t" + string.Join("\t", chain.Skip(1).Select(t => (chain[0] / t).ToString("F2"))));
        Console.WriteLine("# load after " + File.ReadAllText("/proc/loadavg").Trim());
        return 0;
    }

    private static void One((Layer L, float[] In, float[] W, float[] Packed, float[] Out, float[] Bias) p, KernelPath path, int threads)
    {
        if (p.L.Shape.IsDepthwise)
        {
            Conv.Run(KernelMode.Fast, path, p.L.Shape, p.In, p.W, p.Out, threads, p.Bias);
        }
        else
        {
            Conv.RunPacked(path, p.L.Shape, p.In, p.Packed, p.Out, threads, p.Bias);
        }
    }

    // Seconds for one call: the minimum over batches of calls that each take about a tenth of the target.
    private static double Time((Layer L, float[] In, float[] W, float[] Packed, float[] Out, float[] Bias) p, KernelPath path, int threads, int targetMs)
    {
        for (int i = 0; i < 5; i++) One(p, path, threads);
        long t0 = Stopwatch.GetTimestamp();
        One(p, path, threads);
        double one = Math.Max(1e-6, (Stopwatch.GetTimestamp() - t0) / (double)Stopwatch.Frequency);
        int per = Math.Max(1, (int)(targetMs / 1000.0 / 10 / one));
        double best = double.MaxValue;
        for (int b = 0; b < 10; b++)
        {
            t0 = Stopwatch.GetTimestamp();
            for (int i = 0; i < per; i++) One(p, path, threads);
            best = Math.Min(best, (Stopwatch.GetTimestamp() - t0) / (double)Stopwatch.Frequency / per);
        }

        return best;
    }
}
