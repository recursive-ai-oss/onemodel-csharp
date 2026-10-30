using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Tracing;
using static RecursiveAI.OneModel.Tests.Tracing.TraceHelpers;

namespace RecursiveAI.OneModel.Tests.Tracing;

[Collection("autograd")]
public class TraceModelTests
{
    private static TracedModel TraceCnn(TraceModels.Cnn m, int n, out Tensor eager)
    {
        using Tracer t = Tracer.Begin();
        Tensor x = t.Input("x", TraceModels.Cnn.Input(n), "n", null, null, null);
        eager = m.Forward(x);
        return t.Finish("y", eager);
    }

    private static TracedModel TraceEncoder(TraceModels.Encoder m, int b, int n, out Tensor eager)
    {
        using Tracer t = Tracer.Begin();
        Tensor ids = t.Input("ids", TraceModels.Ids(7, 50, b, n), "b", "n");
        Tensor mask = t.Input("mask", TraceModels.Encoder.Mask(b, n, 8), "b", "n", "n");
        eager = m.Forward(ids, mask);
        return t.Finish("y", eager);
    }

    private static TracedModel TraceCrnn(TraceModels.Crnn m, int b, int w, out Tensor eager)
    {
        using Tracer t = Tracer.Begin();
        Tensor x = t.Input("x", TraceModels.Crnn.Input(b, w), "b", null, null, "w");
        eager = m.Forward(x);
        return t.Finish("y", eager);
    }

    [Fact]
    public void ConvNetMatchesEagerAndFollowsTheBatchSize()
    {
        var m = new TraceModels.Cnn();
        using TracedModel tm = TraceCnn(m, 4, out Tensor traced);
        using Session s = tm.CreateSession();
        Tensor x4 = TraceModels.Cnn.Input(4);
        Assert.True(Scaled(Run(s, ["x"], [x4]), traced) <= 1e-5f);

        // The batch axis is symbolic: other batch sizes give what the eager model gives on them.
        foreach (int n in new[] { 1, 7 })
        {
            Tensor x = TraceModels.Cnn.Input(n);
            Assert.True(Scaled(Run(s, ["x"], [x]), m.Forward(x)) <= 1e-5f, $"batch {n}");
        }
    }

    [Fact]
    public void EncoderMatchesEagerAndFollowsBatchAndLength()
    {
        var m = new TraceModels.Encoder();
        using TracedModel tm = TraceEncoder(m, 3, 5, out Tensor traced);
        using Session s = tm.CreateSession();
        Tensor ids = TraceModels.Ids(7, 50, 3, 5), mask = TraceModels.Encoder.Mask(3, 5, 8);
        Assert.True(Scaled(Run(s, ["ids", "mask"], [ids, mask]), traced) <= 1e-5f);
        foreach ((int b, int n) in new[] { (1, 4), (4, 9), (2, 13) })
        {
            Tensor i2 = TraceModels.Ids(9, 50, b, n), m2 = TraceModels.Encoder.Mask(b, n, 3);
            Assert.True(Scaled(Run(s, ["ids", "mask"], [i2, m2]), m.Forward(i2, m2)) <= 1e-5f, $"{b}x{n}");
        }
    }

    [Fact]
    public void CrnnLikeModelMatchesEagerAndFollowsBatchAndWidth()
    {
        var m = new TraceModels.Crnn();
        using TracedModel tm = TraceCrnn(m, 3, 20, out Tensor traced);
        using Session s = tm.CreateSession();
        Assert.True(Scaled(Run(s, ["x"], [TraceModels.Crnn.Input(3, 20)]), traced) <= 1e-5f);
        foreach ((int b, int w) in new[] { (1, 12), (5, 32) })
        {
            Tensor x = TraceModels.Crnn.Input(b, w);
            Assert.True(Scaled(Run(s, ["x"], [x]), m.Forward(x)) <= 1e-5f, $"{b}x{w}");
        }
    }

    [Fact]
    public void SaveLoadRunIsBitIdentical()
    {
        var m = new TraceModels.Encoder();
        using TracedModel tm = TraceEncoder(m, 3, 5, out _);
        string path = Path.Combine(Path.GetTempPath(), "trace-" + Guid.NewGuid().ToString("N") + ".onemodel");
        try
        {
            tm.Save(path);
            using Session mem = tm.CreateSession();
            using Session loaded = Session.Load(path);
            Tensor ids = TraceModels.Ids(11, 50, 2, 6), mask = TraceModels.Encoder.Mask(2, 6, 4);
            Assert.True(BitEqual(Run(mem, ["ids", "mask"], [ids, mask]), Run(loaded, ["ids", "mask"], [ids, mask])));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ExportedOnnxImportsAndAgrees()
    {
        var m = new TraceModels.Crnn();
        using TracedModel tm = TraceCrnn(m, 3, 20, out _);
        using Session a = tm.CreateSession();
        using Session b = Session.Load(new MemoryStream(tm.ToOnnx()));
        Tensor x = TraceModels.Crnn.Input(2, 16);
        Assert.True(BitEqual(Run(a, ["x"], [x]), Run(b, ["x"], [x])));
    }

    [Fact]
    public void ExactModeSessionMatchesExactEagerWhereTheOperationsAreShared()
    {
        var m = new TraceModels.Cnn();
        ExecutionMode before = TensorCompute.Mode;
        try
        {
            TensorCompute.Mode = ExecutionMode.Exact;
            using TracedModel tm = TraceCnn(m, 4, out Tensor traced);
            using Session s = tm.CreateSession(new SessionOptions { Mode = ExecutionMode.Exact });
            Tensor got = Run(s, ["x"], [TraceModels.Cnn.Input(4)]);
            // Conv, pooling, relu and the matrix product are the same canonical kernels in both layers; batch norm, log-softmax and the
            // bias adds are written differently, so equality is to the last bits, not guaranteed identical.
            Assert.True(Scaled(got, traced) <= 1e-6f);
        }
        finally
        {
            TensorCompute.Mode = before;
        }
    }
}

[Collection("autograd")]
public class TraceFailureTests
{
    private static Tensor Example() => TraceModels.Rand(1, 1f, 2, 3);

    [Fact]
    public void ReadingATracedValueFailsNamingTheRead()
    {
        using Tracer t = Tracer.Begin();
        Tensor x = t.Input("x", Example());
        Tensor y = Ops.Relu(x);
        TraceException e = Assert.Throws<TraceException>(() => y.ToArray<float>());
        Assert.Equal(TraceErrorKind.DataDependent, e.Kind);
        Assert.Contains("ToArray", e.Message);
        Assert.Contains("Relu", e.Message);
        Assert.Equal(TraceErrorKind.DataDependent, Assert.Throws<TraceException>(() => x.Data<float>().Length).Kind);
        Assert.Equal(TraceErrorKind.DataDependent, Assert.Throws<TraceException>(() => y.Get<float>(0, 0)).Kind);
        Assert.Equal(TraceErrorKind.DataDependent, Assert.Throws<TraceException>(() => Tensor.Create(DType.F32, y.Shape).CopyFrom(y)).Kind);
    }

    [Fact]
    public void DataDependentBranchFails()
    {
        using Tracer t = Tracer.Begin();
        Tensor x = t.Input("x", Example());
        Tensor y = Ops.Sum(x);
        TraceException e = Assert.Throws<TraceException>(() => y.Data<float>()[0] > 0 ? Ops.Relu(x) : Ops.Neg(x));
        Assert.Equal(TraceErrorKind.DataDependent, e.Kind);
    }

    [Fact]
    public void InPlaceMutationOfATracedTensorFails()
    {
        using Tracer t = Tracer.Begin();
        Tensor x = t.Input("x", Example());
        Tensor y = Ops.Relu(x);
        Assert.Equal(TraceErrorKind.InPlaceMutation, Assert.Throws<TraceException>(() => y.Fill(0f)).Kind);
        Assert.Equal(TraceErrorKind.InPlaceMutation, Assert.Throws<TraceException>(() => x.MutableData<float>()).Kind);
        Assert.Equal(TraceErrorKind.InPlaceMutation, Assert.Throws<TraceException>(() => y.Set(1f, 0, 0)).Kind);
    }

    [Fact]
    public void MutatingACapturedConstantDuringTheTraceFails()
    {
        Tensor w = TraceModels.Rand(2, 1f, 2, 3);
        using Tracer t = Tracer.Begin();
        Tensor x = t.Input("x", Example());
        Tensor y = Ops.Add(x, w);
        w.Fill(0f);
        TraceException e = Assert.Throws<TraceException>(() => t.Finish("y", y));
        Assert.Equal(TraceErrorKind.InPlaceMutation, e.Kind);
    }

    [Fact]
    public void NoGradInsideATraceFails()
    {
        using Tracer t = Tracer.Begin();
        _ = t.Input("x", Example());
        Assert.Equal(TraceErrorKind.NoGradInTrace, Assert.Throws<TraceException>(() => GradMode.NoGrad()).Kind);
    }

    [Fact]
    public void OperationsWithoutAGraphEquivalentFail()
    {
        using Tracer t = Tracer.Begin();
        Tensor lp = t.Input("lp", TraceModels.Rand(3, 1f, 5, 2, 4));
        Tensor lsm = Ops.LogSoftmax(lp);
        TraceException e = Assert.Throws<TraceException>(() =>
            Ops.CtcLoss(lsm, new[] { 1, 2 }, new[] { 5, 5 }, new[] { 1, 1 }));
        Assert.Equal(TraceErrorKind.UnsupportedOperation, e.Kind);
        Assert.Contains("CtcLoss", e.Message);
    }

    [Fact]
    public void TrainingModeBatchNormFails()
    {
        using Tracer t = Tracer.Begin();
        Tensor x = t.Input("x", TraceModels.Rand(3, 1f, 2, 3, 4, 4));
        TraceException e = Assert.Throws<TraceException>(() => Ops.BatchNorm2d(x, TraceModels.Const(1, 3), TraceModels.Const(0, 3),
            TraceModels.Const(0, 3), TraceModels.Const(1, 3), training: true));
        Assert.Equal(TraceErrorKind.UnsupportedOperation, e.Kind);
    }

    [Fact]
    public void AnOutputThatDoesNotDependOnAnInputFails()
    {
        using Tracer t = Tracer.Begin();
        _ = t.Input("x", Example());
        Tensor c = TraceModels.Rand(4, 1f, 2, 3);
        TraceException e = Assert.Throws<TraceException>(() => t.Finish("y", Ops.Relu(c)));
        Assert.Equal(TraceErrorKind.InvalidUsage, e.Kind);
    }

    [Fact]
    public void TracersDoNotNestAndStopOnDispose()
    {
        using (Tracer t = Tracer.Begin())
        {
            Assert.Equal(TraceErrorKind.InvalidUsage, Assert.Throws<TraceException>(() => Tracer.Begin()).Kind);
            Assert.True(t.IsRecording);
        }

        using Tracer again = Tracer.Begin();
        Assert.True(again.IsRecording);
        // After a trace the values are ordinary again.
        Tensor x = again.Input("x", Example());
        Tensor y = Ops.Relu(x);
        again.Finish("y", y).Dispose();
        Assert.Equal(6, y.ToArray<float>().Length);
    }

    [Fact]
    public void SymbolicAxesNeedDistinctSizesAtLeastTwo()
    {
        using Tracer t = Tracer.Begin();
        Assert.Throws<TraceException>(() => t.Input("a", TraceModels.Rand(1, 1f, 1, 3), "n", null));
        _ = t.Input("b", TraceModels.Rand(1, 1f, 4, 3), "n", null);
        Assert.Throws<TraceException>(() => t.Input("c", TraceModels.Rand(1, 1f, 4, 3), "m", null));
    }

    [Fact]
    public void SlicingASymbolicAxisFails()
    {
        using Tracer t = Tracer.Begin();
        Tensor x = t.Input("x", TraceModels.Rand(1, 1f, 4, 3), "n", null);
        TraceException e = Assert.Throws<TraceException>(() => x.Slice(0, 0, 2));
        Assert.Equal(TraceErrorKind.SymbolicShape, e.Kind);
    }
}
