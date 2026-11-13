using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tests.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu.Tests;

/// <summary>
/// The recurrent layers on the device: LSTM, GRU and RNN with their options, against the CPU session in the same mode (float32 scaled
/// error 1e-5), run twice alike, in one region with nothing left on the CPU. Sequence lengths are validated on the host.
/// </summary>
public class RecurrentTests
{
    private static Tensor Make(float[] v, params int[] shape) => Tensor.FromSpan<float>(v, Shape.Create(shape));

    private static byte[] Ints32(int[] v)
    {
        byte[] raw = new byte[v.Length * 4];
        for (int i = 0; i < v.Length; i++)
        {
            System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(raw.AsSpan(i * 4), v[i]);
        }

        return raw;
    }

    public sealed record Layer(string Op, int Seq, int Batch, int Input, int Hidden, string Direction = "forward", int LayoutAttr = 0,
        bool Bias = true, bool InitialH = false, bool InitialC = false, bool Peepholes = false, string Lengths = "none", int[]? LengthValues = null,
        (string, object)[]? Extra = null);

    private static int Gates(string op) => op switch { "LSTM" => 4, "GRU" => 3, _ => 1 };

    /// <summary>X, W, R, B, sequence_lens, initial_h, initial_c, P as initializers or inputs; outputs y, y_h and (LSTM) y_c.</summary>
    private static ModelBuilder Build(Layer l, string x = "x", string y = "y", string prefix = "", ModelBuilder? b = null)
    {
        int dirs = l.Direction == "bidirectional" ? 2 : 1, g = Gates(l.Op), gh = g * l.Hidden;
        int seed = 100 + prefix.Length * 7;
        b ??= new ModelBuilder(14);
        var inputs = new List<string> { x, prefix + "w", prefix + "r", l.Bias ? prefix + "b" : "" };
        b.Initializer(prefix + "w", GpuTestHost.Random(dirs * gh * l.Input, seed + 1, 0.6f), dirs, gh, l.Input);
        b.Initializer(prefix + "r", GpuTestHost.Random(dirs * gh * l.Hidden, seed + 2, 0.6f), dirs, gh, l.Hidden);
        if (l.Bias)
        {
            b.Initializer(prefix + "b", GpuTestHost.Random(dirs * 2 * gh, seed + 3, 0.5f), dirs, 2 * gh);
        }

        switch (l.Lengths)
        {
            case "const":
                b.RawInitializer(prefix + "lens", 6, [l.Batch], Ints32(l.LengthValues!));
                inputs.Add(prefix + "lens");
                break;
            case "input":
                inputs.Add("lens");
                break;
            default:
                inputs.Add("");
                break;
        }

        long[] stateDims = l.LayoutAttr == 1 ? [l.Batch, dirs, l.Hidden] : [dirs, l.Batch, l.Hidden];
        int stateLen = dirs * l.Batch * l.Hidden;
        if (l.InitialH)
        {
            b.Initializer(prefix + "h0", GpuTestHost.Random(stateLen, seed + 4, 0.8f), stateDims);
            inputs.Add(prefix + "h0");
        }
        else
        {
            inputs.Add("");
        }

        if (l.Op == "LSTM")
        {
            if (l.InitialC)
            {
                b.Initializer(prefix + "c0", GpuTestHost.Random(stateLen, seed + 5, 0.8f), stateDims);
                inputs.Add(prefix + "c0");
            }
            else
            {
                inputs.Add("");
            }

            if (l.Peepholes)
            {
                b.Initializer(prefix + "p", GpuTestHost.Random(dirs * 3 * l.Hidden, seed + 6, 0.7f), dirs, 3 * l.Hidden);
                inputs.Add(prefix + "p");
            }
        }

        while (inputs.Count > 0 && inputs[^1].Length == 0)
        {
            inputs.RemoveAt(inputs.Count - 1);
        }

        var attrs = new List<(string, object)> { ("hidden_size", (long)l.Hidden), ("direction", l.Direction), ("layout", (long)l.LayoutAttr) };
        if (l.Extra is not null)
        {
            attrs.AddRange(l.Extra);
        }

        string[] outs = l.Op == "LSTM" ? [y, prefix + "yh", prefix + "yc"] : [y, prefix + "yh"];
        b.Node(l.Op, [.. inputs], outs, [.. attrs]);
        return b;
    }

    private static Dictionary<string, Tensor> Feed(Layer l, int seed = 3)
    {
        var feed = new Dictionary<string, Tensor>
        {
            ["x"] = l.LayoutAttr == 1 ? Make(GpuTestHost.Random(l.Batch * l.Seq * l.Input, seed, 2f), l.Batch, l.Seq, l.Input)
                : Make(GpuTestHost.Random(l.Seq * l.Batch * l.Input, seed, 2f), l.Seq, l.Batch, l.Input),
        };
        if (l.Lengths == "input")
        {
            feed["lens"] = Tensor.FromSpan<int>(l.LengthValues!, Shape.Create(l.Batch));
        }

        return feed;
    }

    private static ModelBuilder WithIo(ModelBuilder b, Layer l)
    {
        b.Input("x", 1, l.LayoutAttr == 1 ? [(object)l.Batch, l.Seq, l.Input] : [l.Seq, l.Batch, l.Input]);
        if (l.Lengths == "input")
        {
            b.Input("lens", 6, l.Batch);
        }

        return b;
    }

    /// <summary>Runs the model on the CPU and the device in <paramref name="mode"/>; returns the worst scaled error over <paramref name="outputs"/>.</summary>
    private static double Compare(byte[] model, Dictionary<string, Tensor> feed, ArithmeticMode mode, string[] outputs, out SessionDiagnostics diag)
    {
        GpuTestHost.Require();
        var expected = new Dictionary<string, float[]>();
        using (Session cpu = GpuTestHost.Load(model, GpuTestHost.Cpu(mode)))
        using (SessionOutputs o = cpu.Run(feed))
        {
            foreach (string n in outputs)
            {
                expected[n] = o[n].Data<float>().ToArray();
            }
        }

        using Session gpu = GpuTestHost.Load(model, GpuTestHost.Gpu(mode, require: true));
        var first = new Dictionary<string, float[]>();
        using (SessionOutputs o = gpu.Run(feed))
        {
            foreach (string n in outputs)
            {
                first[n] = o[n].Data<float>().ToArray();
            }
        }

        using (SessionOutputs o = gpu.Run(feed))
        {
            foreach (string n in outputs)
            {
                Assert.Equal(GpuTestHost.Hash(first[n]), GpuTestHost.Hash(o[n].Data<float>()));
            }
        }

        diag = gpu.GetDiagnostics(feed);
        Assert.True(diag.Nodes.All(n => n.FallbackReason is null), string.Join("; ", diag.Nodes.Where(n => n.FallbackReason is not null).Select(n => n.OpType + ": " + n.FallbackReason)));
        Assert.Equal(1, diag.DeviceRegions);
        GpuTestHost.AssertValidationClean(gpu);
        return outputs.Max(n => GpuTestHost.ScaledError(first[n], expected[n]));
    }

    private static string[] OutputsOf(Layer l, string y = "y", string prefix = "") => l.Op == "LSTM" ? [y, prefix + "yh", prefix + "yc"] : [y, prefix + "yh"];

    private static byte[] Finish(ModelBuilder b, string[] outputs)
    {
        foreach (string o in outputs)
        {
            b.Output(o, 1);
        }

        return b.Build();
    }

    private static string[] Strs(params string[] v) => v;

    private static float[] Flts(params float[] v) => v;

    public static TheoryData<Layer> Layers => new()
    {
        new Layer("LSTM", 7, 3, 5, 8, "bidirectional"),
        new Layer("LSTM", 6, 4, 9, 8, "forward", 1, InitialH: true, InitialC: true, Peepholes: true, Extra: [("clip", 1.5f), ("input_forget", 1L)]),
        new Layer("LSTM", 5, 2, 6, 7, "reverse", Bias: false),
        new Layer("LSTM", 11, 9, 19, 37, "bidirectional", InitialH: true),
        new Layer("GRU", 6, 3, 5, 8, "forward"),
        new Layer("GRU", 7, 4, 9, 11, "bidirectional", 1, InitialH: true, Extra: [("linear_before_reset", 1L)]),
        new Layer("GRU", 5, 2, 4, 33, "reverse", Extra: [("clip", 0.9f)]),
        new Layer("RNN", 8, 3, 5, 8, "forward"),
        new Layer("RNN", 6, 3, 5, 9, "bidirectional", 1, InitialH: true, Extra: [("activations", Strs("Relu", "LeakyRelu")), ("activation_alpha", Flts(0.2f))]),
        new Layer("LSTM", 6, 3, 5, 8, "bidirectional", Extra: [("activations", Strs("HardSigmoid", "Tanh", "Softsign", "Sigmoid", "ScaledTanh", "Elu")), ("activation_alpha", Flts(0.3f, 0.9f, 1.2f)), ("activation_beta", Flts(0.4f, 0.8f))]),
        new Layer("GRU", 6, 3, 5, 8, "forward", Extra: [("activations", Strs("Affine", "Softplus")), ("activation_alpha", Flts(0.5f)), ("activation_beta", Flts(0.1f))]),
        new Layer("LSTM", 1, 1, 1, 1),
    };

    [Theory]
    [MemberData(nameof(Layers))]
    public void LayerMatchesTheCpu(Layer l)
    {
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            byte[] model = Finish(WithIo(Build(l), l), OutputsOf(l));
            using var feed = new FeedScope(Feed(l));
            double err = Compare(model, feed.Values, mode, OutputsOf(l), out SessionDiagnostics d);
            Assert.Equal(1, d.NodesOnDevice);
            Assert.True(err <= 1e-5, $"{l.Op} {l.Direction} layout {l.LayoutAttr} {mode}: scaled error {err}");
        }
    }

    [Theory]
    [InlineData("const")]
    [InlineData("input")]
    public void SequenceLengthsMatchTheCpu(string kind)
    {
        // Rows of different lengths, one of length zero; Y is zero past each length and the states stop at it.
        foreach (string op in Strs("LSTM", "GRU"))
        {
            var l = new Layer(op, 7, 4, 5, 8, "bidirectional", Lengths: kind, LengthValues: [7, 3, 0, 5], InitialH: true);
            byte[] model = Finish(WithIo(Build(l), l), OutputsOf(l));
            using var feed = new FeedScope(Feed(l));
            Assert.True(Compare(model, feed.Values, ArithmeticMode.Fast, OutputsOf(l), out _) <= 1e-5);
        }
    }

    [Fact]
    public void ALengthOutsideTheSequenceIsTheCpusInputError()
    {
        GpuTestHost.Require();
        var l = new Layer("LSTM", 7, 3, 5, 8, Lengths: "input", LengthValues: [7, 3, 2]);
        byte[] model = Finish(WithIo(Build(l), l), OutputsOf(l));
        using var feed = new FeedScope(Feed(l));
        using Session gpu = GpuTestHost.Load(model, GpuTestHost.Gpu(ArithmeticMode.Fast, require: true));
        using (gpu.Run(feed.Values))
        {
        }

        using var bad = new FeedScope(new Dictionary<string, Tensor> { ["x"] = feed.Values["x"], ["lens"] = Tensor.FromSpan<int>([7, 9, 2], Shape.Create(3)) }, owns: false);
        SessionException onGpu = Assert.Throws<SessionException>(() => gpu.Run(bad.Values));
        using Session cpu = GpuTestHost.Load(model, GpuTestHost.Cpu(ArithmeticMode.Fast));
        SessionException onCpu = Assert.Throws<SessionException>(() => cpu.Run(bad.Values));
        Assert.Equal(SessionErrorKind.InvalidInput, onGpu.Kind);
        Assert.Equal(onCpu.Message, onGpu.Message);
        // The session still works afterwards.
        using (gpu.Run(feed.Values))
        {
        }
    }

    [Fact]
    public void TwoStackedLayersShareTheScratchAndRunInOneRegion()
    {
        // The text-line recognizer's shape: a bidirectional layer, its output rearranged to [seq, batch, 2 * hidden], a second one.
        var a = new Layer("LSTM", 20, 3, 12, 16, "bidirectional");
        var c = new Layer("LSTM", 20, 3, 32, 16, "bidirectional");
        ModelBuilder b = WithIo(Build(a, "x", "y1"), a);
        b.Node("Transpose", ["y1"], ["t"], ("perm", new long[] { 0, 2, 1, 3 }));
        b.Initializer("shape", [20L, 3L, 32L], 3);
        b.Node("Reshape", ["t", "shape"], ["rs"]);
        Build(c, "rs", "y", "s", b);
        string[] outs = ["y", "syh", "syc"];
        byte[] model = Finish(b, outs);
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            using var feed = new FeedScope(Feed(a));
            double err = Compare(model, feed.Values, mode, outs, out SessionDiagnostics d);
            Assert.Equal(d.Nodes.Length, d.NodesOnDevice);
            Assert.True(err <= 1e-5, $"{mode}: {err}");
        }
    }

    [Fact]
    public void ARunTimeWeightStaysOnTheCpuOrRuns()
    {
        // W is computed by a node rather than stored: the layer must either run correctly on the device or say why it did not.
        var l = new Layer("GRU", 5, 2, 4, 6);
        int gh = 3 * 6;
        ModelBuilder b = new ModelBuilder(14).Input("x", 1, 5, 2, 4).Input("w0", 1, 1, gh, 4);
        b.Initializer("two", [2f], 1);
        b.Node("Mul", ["w0", "two"], ["w"]);
        b.Initializer("r", GpuTestHost.Random(gh * 6, 5, 0.6f), 1, gh, 6);
        b.Node("GRU", ["x", "w", "r"], ["y", "yh"], ("hidden_size", 6L));
        byte[] model = Finish(b, ["y", "yh"]);
        using var feed = new FeedScope(new Dictionary<string, Tensor>
        {
            ["x"] = Make(GpuTestHost.Random(5 * 2 * 4, 1, 2f), 5, 2, 4),
            ["w0"] = Make(GpuTestHost.Random(gh * 4, 2, 0.6f), 1, gh, 4),
        });
        GpuTestHost.Require();
        float[] expected;
        using (Session cpu = GpuTestHost.Load(model, GpuTestHost.Cpu(ArithmeticMode.Fast)))
        using (SessionOutputs o = cpu.Run(feed.Values))
        {
            expected = o["y"].Data<float>().ToArray();
        }

        using Session gpu = GpuTestHost.Load(model, GpuTestHost.Gpu(ArithmeticMode.Fast));
        using SessionOutputs g = gpu.Run(feed.Values);
        Assert.True(GpuTestHost.ScaledError(g["y"].Data<float>(), expected) <= 1e-5);
        _ = l;
    }

    private sealed class FeedScope : IDisposable
    {
        private readonly bool _owns;

        public FeedScope(Dictionary<string, Tensor> values, bool owns = true)
        {
            Values = values;
            _owns = owns;
        }

        public Dictionary<string, Tensor> Values { get; }

        public void Dispose()
        {
            if (_owns)
            {
                foreach (Tensor t in Values.Values)
                {
                    t.Dispose();
                }
            }
        }
    }
}
