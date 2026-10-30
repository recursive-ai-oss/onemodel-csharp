using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Training;

namespace RecursiveAI.OneModel.Benchmarks;

/// <summary>
/// Measures where reduced-precision training could pay on a CPU: step time and peak pool memory of three training steps in float32,
/// the share of that memory held by activations (what bf16 storage could halve), and the cost of the float32 to bfloat16 and
/// back conversion that every reduced-precision scheme adds. Usage: <c>mixed [threads] [steps]</c>. Indicative timings only.
/// </summary>
internal static class MixedPrecision
{
    private sealed class Encoder : NnModule
    {
        private readonly Embedding _emb;
        private readonly TransformerEncoderLayer[] _layers;
        private readonly Linear _cls;

        public Encoder(Generator g, int vocab, int width, int heads, int ff, int layers, int classes)
        {
            _emb = RegisterModule("emb", new Embedding(vocab, width, g));
            _layers = new TransformerEncoderLayer[layers];
            for (int i = 0; i < layers; i++)
            {
                _layers[i] = RegisterModule("layer" + i.ToString(CultureInfo.InvariantCulture), new TransformerEncoderLayer(width, heads, ff, g));
            }

            _cls = RegisterModule("cls", new Linear(width, classes, g));
        }

        public override Tensor Forward(Tensor ids)
        {
            Tensor x = _emb.Forward(ids);
            foreach (TransformerEncoderLayer l in _layers)
            {
                x = l.Forward(x);
            }

            return _cls.Forward(Ops.Mean(x, 1));
        }
    }

    private sealed class Crnn : NnModule
    {
        private readonly Sequential _conv;
        private readonly Recurrent _rnn;
        private readonly Linear _out;

        public Crnn(Generator g, int classes)
        {
            _conv = RegisterModule("conv", new Sequential(
                new Conv2d(1, 32, 3, g, padding: 1), new BatchNorm2d(32), Lambda.Relu(), Lambda.MaxPool2d(2),
                new Conv2d(32, 64, 3, g, padding: 1), new BatchNorm2d(64), Lambda.Relu(), Lambda.MaxPool2d(2)));
            _rnn = RegisterModule("rnn", new Recurrent(false, 64, 128, g, bidirectional: true));
            _out = RegisterModule("out", new Linear(256, classes, g));
        }

        public override Tensor Forward(Tensor x)
        {
            Tensor f = Ops.Mean(_conv.Forward(x), 2); // [B, 64, T]
            Tensor seq = f.Permute(2, 0, 1);          // [T, B, 64]
            Tensor h = _rnn.Forward(seq);             // [T, B, 256]
            return Ops.LogSoftmax(_out.Forward(h));   // [T, B, C]
        }
    }

    private sealed class Case(string name, NnModule model, Func<Tensor> loss)
    {
        public string Name { get; } = name;
        public NnModule Model { get; } = model;
        public Func<Tensor> Loss { get; } = loss;
    }

    private static Tensor Rand(Generator g, params int[] dims)
    {
        Tensor t = Tensor.Create(DType.F32, Shape.Create(dims));
        foreach (ref float v in t.MutableData<float>())
        {
            v = g.Uniform(-1f, 1f);
        }

        ActivationScope.Track(t);
        return t;
    }

    private static Case Transformer()
    {
        var g = new Generator(1);
        var m = new Encoder(g, 1000, 256, 4, 1024, 4, 10);
        const int B = 32, S = 128;
        var ids = new int[B * S];
        var lab = new int[B];
        for (int i = 0; i < ids.Length; i++) ids[i] = (int)((uint)(i * 2654435761u) % 1000);
        for (int i = 0; i < B; i++) lab[i] = i % 10;
        return new Case("transformer b32 s128 w256 x4", m, () =>
        {
            Tensor x = Tensor.FromSpan<int>(ids, Shape.Create(B, S));
            ActivationScope.Track(x);
            Tensor y = Tensor.FromSpan<int>(lab, Shape.Create(B));
            ActivationScope.Track(y);
            return Losses.CrossEntropy(m.Forward(x), y);
        });
    }

    private static Case Conv()
    {
        var g = new Generator(2);
        var m = new Sequential(
            new Conv2d(3, 32, 3, g, padding: 1), new BatchNorm2d(32), Lambda.Relu(), Lambda.MaxPool2d(2),
            new Conv2d(32, 64, 3, g, padding: 1), new BatchNorm2d(64), Lambda.Relu(), Lambda.MaxPool2d(2),
            new Conv2d(64, 128, 3, g, padding: 1), new BatchNorm2d(128), Lambda.Relu(), Lambda.GlobalAvgPool2d(),
            Lambda.Flatten(), new Linear(128, 10, g));
        var lab = new int[16];
        for (int i = 0; i < lab.Length; i++) lab[i] = i % 10;
        return new Case("conv b16 3x64x64", m, () =>
        {
            Tensor y = Tensor.FromSpan<int>(lab, Shape.Create(16));
            ActivationScope.Track(y);
            return Losses.CrossEntropy(m.Forward(Rand(g, 16, 3, 64, 64)), y);
        });
    }

    private static Case CrnnCase()
    {
        var g = new Generator(3);
        const int B = 16, C = 80, T = 32;
        var m = new Crnn(g, C);
        var targets = new int[B * 8];
        for (int i = 0; i < targets.Length; i++) targets[i] = 1 + (i * 7 % (C - 1));
        var inLen = new int[B];
        var tgLen = new int[B];
        Array.Fill(inLen, T);
        Array.Fill(tgLen, 8);
        return new Case("crnn b16 1x32x128 lstm128x2", m, () =>
            Losses.Ctc(m.Forward(Rand(g, B, 1, 32, 128)), targets, inLen, tgLen, zeroInfinity: true));
    }

    public static int Run(string[] args)
    {
        int threads = args.Length > 0 ? int.Parse(args[0], CultureInfo.InvariantCulture) : 1;
        int steps = args.Length > 1 ? int.Parse(args[1], CultureInfo.InvariantCulture) : 7;
        TensorCompute.Threads = threads;
        Console.WriteLine("# threads " + threads + ", cores " + Environment.ProcessorCount + ", " + Environment.OSVersion);
        // The BCL's hardware intrinsics are the only route to a native bfloat16 dot product (no native code allowed).
        var isa = typeof(object).Assembly.GetTypes()
            .Where(t => t.Namespace is "System.Runtime.Intrinsics.X86" or "System.Runtime.Intrinsics.Arm")
            .Select(t => t.Namespace![^3..] + "." + t.Name).Where(n => !n.Contains('+', StringComparison.Ordinal)).Distinct().Order().ToList();
        Console.WriteLine("# intrinsic classes: " + string.Join(" ", isa));
        Console.WriteLine("# any bf16 class: " + isa.Any(n => n.Contains("bf", StringComparison.OrdinalIgnoreCase)));

        Console.WriteLine("model | fp32 step ms | param+grad+opt MB | peak MB | activations MB (after forward) | bf16 convert floor ms | floor/step");
        foreach (Case c in new[] { Transformer(), Conv(), CrnnCase() })
        {
            var opt = new AdamW(c.Model.Parameters(), 1e-3f);
            var times = new List<double>();
            long peak = 0, act = 0;
            long baseline = BufferPool.Shared.GetStats().OutstandingBytes;
            for (int s = 0; s < steps; s++)
            {
                using var scope = new ActivationScope();
                long before = BufferPool.Shared.GetStats().OutstandingBytes;
                long stepPeak = 0;
                using var stop = new CancellationTokenSource();
                var sampler = new Thread(() =>
                {
                    while (!stop.IsCancellationRequested)
                    {
                        stepPeak = Math.Max(stepPeak, BufferPool.Shared.GetStats().OutstandingBytes);
                        Thread.Sleep(0);
                    }
                });
                sampler.Start();
                long t0 = Stopwatch.GetTimestamp();
                Tensor loss = c.Loss();
                long afterFwd = BufferPool.Shared.GetStats().OutstandingBytes;
                opt.ZeroGrad();
                loss.Backward();
                opt.Step();
                double ms = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
                stop.Cancel();
                sampler.Join();
                if (s >= 2)
                {
                    times.Add(ms);
                }

                peak = Math.Max(peak, stepPeak);
                act = Math.Max(act, afterFwd - before);
            }

            times.Sort();
            double step = times[times.Count / 2];
            long paramBytes = c.Model.Parameters().Sum(p => (long)p.Length * 4);
            long fixedBytes = BufferPool.Shared.GetStats().OutstandingBytes - baseline + paramBytes;
            // Conversion floor: round every activation and parameter float to bfloat16 and back once.
            long floats = (act + paramBytes) / 4;
            var src = new float[floats];
            var half = new ushort[floats];
            double conv = double.MaxValue;
            for (int r = 0; r < 5; r++)
            {
                long t0 = Stopwatch.GetTimestamp();
                HalfConvert.SingleToBFloat(src, half);
                HalfConvert.BFloatToSingle(half, src);
                conv = Math.Min(conv, Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
            }

            Console.WriteLine(string.Join(" | ", c.Name, F(step), F(fixedBytes / 1048576.0), F(peak / 1048576.0), F(act / 1048576.0), F(conv), F(conv / step)));
        }

        return 0;
    }

    private static string F(double v) => v.ToString("0.00", CultureInfo.InvariantCulture);
}
