using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Tests.Training;
using RecursiveAI.OneModel.Training;
using RecursiveAI.OneModel.Tracing;
using static RecursiveAI.OneModel.Tests.Tracing.TraceHelpers;

namespace RecursiveAI.OneModel.Tests.Tracing;

// Tracing through the training modules (evaluation mode).
[Collection("autograd")]
public class ModuleTraceTests
{
    private static void Randomize(NnModule m, int seed)
    {
        var rng = new Random(seed);
        foreach ((string name, Tensor t) in m.NamedBuffers())
        {
            if (t.DType != DType.F32)
            {
                continue;
            }

            Span<float> d = t.MutableData<float>();
            for (int i = 0; i < d.Length; i++)
            {
                d[i] = name.EndsWith("running_var", StringComparison.Ordinal) ? 0.5f + (float)rng.NextDouble() : (float)(rng.NextDouble() - 0.5);
            }
        }
    }

    private static Sequential ConvNet()
    {
        var g = new Generator(5);
        var net = new Sequential(
            new Conv2d(3, 4, 3, g, padding: 1), new BatchNorm2d(4), Lambda.Relu(), Lambda.MaxPool2d(2),
            new Conv2d(4, 8, 3, g, padding: 1), new BatchNorm2d(8), Lambda.Relu(), Lambda.AvgPool2d(2),
            Lambda.Flatten(), new Linear(72, 10, g));
        Randomize(net, 6);
        return net;
    }

    private sealed class CrnnModule : NnModule
    {
        private readonly Conv2d _conv;
        private readonly Recurrent _rnn;
        private readonly Linear _head;

        public CrnnModule(Generator g)
        {
            _conv = RegisterModule("conv", new Conv2d(1, 4, 3, g, padding: 1));
            _rnn = RegisterModule("rnn", new Recurrent(false, 32, 6, g, layers: 2, bidirectional: true));
            _head = RegisterModule("head", new Linear(12, 7, g));
        }

        public override Tensor Forward(Tensor x)
        {
            int b = x.Shape[0];
            Tensor h = Lambda.MaxPool2d(2).Forward(Ops.Relu(_conv.Forward(x)));
            Tensor seq = h.Permute(3, 0, 1, 2).Reshape(h.Shape[3], b, 32);
            return Ops.LogSoftmax(_head.Forward(_rnn.Forward(seq)));
        }
    }

    [Fact]
    public void ConvNetModuleTracesAndFollowsTheBatch()
    {
        Sequential net = ConvNet();
        net.Train(true);
        using TracedModel tm = net.Trace(TraceModels.Cnn.Input(4), "n", null, null, null);
        Assert.True(net.IsTraining); // restored
        net.Eval();
        using Session s = tm.CreateSession();
        foreach (int n in new[] { 4, 1, 7 })
        {
            Tensor x = TraceModels.Cnn.Input(n);
            Assert.True(Scaled(Run(s, ["input"], [x], "output"), net.Forward(x)) <= 1e-5f, $"batch {n}");
        }

        // Weights keep their state-dict names (a matrix-product weight through its transposed copy).
        Assert.Contains(tm.Ir.Graph.Initializers, t => t.Name.StartsWith("9.weight", StringComparison.Ordinal));
        Assert.Contains(tm.Ir.Graph.Initializers, t => t.Name == "1.running_mean");
    }

    [Fact]
    public void EncoderModuleTracesAndFollowsBatchAndLength()
    {
        var net = new EncoderClassifier(new Generator(3), 50, 16, 2, 32, 2, 5);
        net.Eval();
        using TracedModel tm = net.Trace(TraceModels.Ids(1, 50, 3, 5), "b", "n");
        using Session s = tm.CreateSession();
        foreach ((int b, int n) in new[] { (3, 5), (1, 9), (4, 2) })
        {
            Tensor ids = TraceModels.Ids(b + n, 50, b, n);
            Assert.True(Scaled(Run(s, ["input"], [ids], "output"), net.Forward(ids)) <= 1e-5f, $"{b}x{n}");
        }

        // Steady state: no allocation per run at a fixed shape.
        Tensor fixedIds = TraceModels.Ids(2, 50, 3, 5);
        Tensor[] ins = [fixedIds];
        Tensor[] outs = new Tensor[1];
        for (int i = 0; i < 4; i++)
        {
            s.Run(["input"], ins, ["output"], outs);
        }

        long best = long.MaxValue;
        for (int attempt = 0; attempt < 3 && best != 0; attempt++)
        {
            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 5; i++)
            {
                s.Run(["input"], ins, ["output"], outs);
            }

            best = Math.Min(best, GC.GetAllocatedBytesForCurrentThread() - before);
        }

        Assert.Equal(0, best);
    }

    [Fact]
    public void CrnnModuleTracesAndFollowsBatchAndWidth()
    {
        var net = new CrnnModule(new Generator(8));
        net.Eval();
        using TracedModel tm = net.Trace(TraceModels.Crnn.Input(3, 20), "b", null, null, "w");
        using Session s = tm.CreateSession();
        foreach ((int b, int w) in new[] { (3, 20), (1, 12), (5, 32) })
        {
            Tensor x = TraceModels.Crnn.Input(b, w);
            Assert.True(Scaled(Run(s, ["input"], [x], "output"), net.Forward(x)) <= 1e-5f, $"{b}x{w}");
        }
    }

    [Fact]
    public void TrainingModeDropoutIsRefusedAndEvalIsPassedThrough()
    {
        var g = new Generator(1);
        var net = new Sequential(new Linear(3, 3, g), new Dropout(0.5f, 1));
        using (Tracer t = Tracer.Begin())
        {
            net.Train(true);
            Tensor x = t.Input("x", TraceModels.Rand(1, 1f, 2, 3));
            Assert.Equal(TraceErrorKind.UnsupportedOperation, Assert.Throws<TraceException>(() => net.Forward(x)).Kind);
        }

        using TracedModel tm = net.Trace(TraceModels.Rand(1, 1f, 2, 3));
        Assert.Equal(2, tm.NodeCount); // MatMul and Add; the evaluation-mode dropout is the identity
        Assert.True(net.IsTraining);
    }

    [Fact]
    public void ModuleSaveLoadRunIsBitIdentical()
    {
        Sequential net = ConvNet();
        using TracedModel tm = net.Trace(TraceModels.Cnn.Input(4), "n", null, null, null);
        string path = Path.Combine(Path.GetTempPath(), "trace-" + Guid.NewGuid().ToString("N") + ".onemodel");
        try
        {
            tm.Save(path);
            using Session mem = tm.CreateSession();
            using Session loaded = Session.Load(path);
            Tensor x = TraceModels.Cnn.Input(3);
            Assert.True(BitEqual(Run(mem, ["input"], [x], "output"), Run(loaded, ["input"], [x], "output")));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
