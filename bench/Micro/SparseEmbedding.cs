using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Training;

namespace RecursiveAI.OneModel.Benchmarks;

/// <summary>
/// Memory and step time of fine-tuning with a large token embedding (250 002 x 768) in three ways: a dense gradient with AdamW,
/// a row-sparse gradient with SparseAdam on the table (AdamW on the rest), and a frozen table. One encoder layer of width 768 on
/// 16 x 128 token ids per step. Usage: <c>sparseembed dense|sparse|frozen [threads] [steps]</c>; run each mode in its own process
/// so the peak working set is its own. Indicative timings only.
/// </summary>
internal static class SparseEmbedding
{
    private sealed class Net : NnModule
    {
        public Embedding Emb { get; }
        private readonly TransformerEncoderLayer _layer;
        private readonly Linear _cls;

        public Net(Generator g, int vocab, int width, bool sparse)
        {
            Emb = RegisterModule("emb", new Embedding(vocab, width, g, 1, sparse));
            _layer = RegisterModule("layer", new TransformerEncoderLayer(width, 12, 4 * width, g));
            _cls = RegisterModule("cls", new Linear(width, 2, g));
        }

        public override Tensor Forward(Tensor ids) => _cls.Forward(Ops.Mean(_layer.Forward(Emb.Forward(ids)), 1));
    }

    public static int Run(string[] args)
    {
        string mode = args.Length > 0 ? args[0] : "sparse";
        int threads = args.Length > 1 ? int.Parse(args[1], CultureInfo.InvariantCulture) : 8;
        int steps = args.Length > 2 ? int.Parse(args[2], CultureInfo.InvariantCulture) : 5;
        TensorCompute.Threads = threads;
        const int vocab = 250_002, width = 768, batch = 16, seq = 128;
        using var net = new Net(new Generator(1), vocab, width, sparse: mode == "sparse");
        if (mode == "frozen")
        {
            net.Freeze("emb");
        }

        var restParams = new List<Tensor>();
        foreach ((string n, Tensor t) in net.NamedParameters())
        {
            if (!n.StartsWith("emb.", StringComparison.Ordinal))
            {
                restParams.Add(t);
            }
        }

        using var rest = new AdamW(restParams, 1e-4f);
        using Optimizer? table = mode switch
        {
            "dense" => new AdamW([net.Emb.Weight], 1e-4f),
            "sparse" => new SparseAdam([net.Emb.Weight], 1e-4f),
            _ => null,
        };
        var rng = new Random(7);
        var times = new List<double>();
        long gradBytes = 0;
        for (int s = 0; s < steps; s++)
        {
            var ids = new int[batch * seq];
            for (int i = 0; i < ids.Length; i++)
            {
                ids[i] = rng.Next(0, vocab);
            }

            var labels = new long[batch];
            for (int i = 0; i < batch; i++)
            {
                labels[i] = rng.Next(0, 2);
            }

            var sw = Stopwatch.StartNew();
            using (var scope = new ActivationScope())
            {
                Tensor idt = Tensor.FromSpan<int>(ids, Shape.Create(batch, seq));
                Tensor yt = Tensor.FromSpan<long>(labels, Shape.Create(batch));
                ActivationScope.Track(idt);
                ActivationScope.Track(yt);
                rest.ZeroGrad();
                table?.ZeroGrad();
                Losses.CrossEntropy(net.Forward(idt), yt).Backward();
                Tensor w = net.Emb.Weight;
                gradBytes = w.SparseGrad is { } sg ? sg.ByteCount : (long)w.Grad.Length * sizeof(float);
                table?.Step();
                rest.Step();
            }

            sw.Stop();
            times.Add(sw.Elapsed.TotalMilliseconds);
        }

        long stateBytes = 0;
        foreach (Optimizer? o in new Optimizer?[] { rest, table })
        {
            if (o is null)
            {
                continue;
            }

            foreach ((_, Tensor t) in o.StateTensors)
            {
                stateBytes += (long)t.Length * t.ElementSize;
            }
        }

        times.Sort();
        double median = times[times.Count / 2];
        using Process p = Process.GetCurrentProcess();
        Console.WriteLine($"{mode}: threads {threads}, embedding gradient {gradBytes / 1048576.0:F1} MiB, optimizer state {stateBytes / 1048576.0:F0} MiB, step median {median:F0} ms (min {times[0]:F0}, max {times[^1]:F0}), peak working set {p.PeakWorkingSet64 / 1048576.0:F0} MiB");
        return 0;
    }
}
