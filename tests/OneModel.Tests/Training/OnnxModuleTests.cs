using System.Runtime.CompilerServices;
using RecursiveAI.OneModel.Format;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Tests.Tensors;
using RecursiveAI.OneModel.Training;

namespace RecursiveAI.OneModel.Tests.Training;

// Graphs written by tools/finetune_oracle.py with PyTorch's gradients of the same function: a ModernBERT-style encoder (rotary
// embeddings in the graph, local-window mask, GeGLU, bias-free layer norm), a BERT-style encoder, and a graph using every operator
// the trainable path supports.
[Collection("autograd")]
public class OnnxModuleTests
{
    private static string Dir([CallerFilePath] string file = "") => OracleFile.DataDir("finetune", file);

    private static string Path(string name) => System.IO.Path.Combine(Dir(), name + ".onnx");

    private static double Scaled(ReadOnlySpan<float> a, ReadOnlySpan<float> reference)
    {
        double worst = 0, scale = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs((double)a[i] - reference[i]));
            scale = Math.Max(scale, Math.Abs(reference[i]));
        }

        return worst / Math.Max(scale, 1e-12);
    }

    private static Dictionary<string, Tensor> Inputs(OracleFile f, OnnxModule m) => m.InputNames.ToDictionary(n => n, n => f.MakeInts("in." + n, int64: true));

    private static Tensor Loss(OracleFile f, Tensor[] outputs)
    {
        Tensor total = default;
        for (int k = 0; k < outputs.Length; k++)
        {
            Tensor w = f.Make($"w.{k}");
            ActivationScope.Track(w);
            Tensor term = Ops.Sum(Ops.Mul(outputs[k], w));
            total = total.IsEmpty ? term : Ops.Add(total, term);
        }

        return total;
    }

    [Theory]
    [InlineData("mini_modern")]
    [InlineData("mini_bert")]
    [InlineData("ops_zoo")]
    public void ForwardMatchesSessionAndGradientsMatchPyTorch(string name)
    {
        OracleFile f = OracleFile.Load(name, "finetune");
        using OnnxModule m = OnnxModule.Load(Path(name));
        using Session session = Session.Load(Path(name));
        Dictionary<string, Tensor> inputs = Inputs(f, m);
        if (name == "ops_zoo")
        {
            inputs = new() { ["x"] = f.Make("in.x"), ["ids"] = f.MakeInts("in.ids", int64: true) };
        }

        using var scope = new ActivationScope();
        Tensor[] outs = m.Forward(inputs);
        SessionOutputs reference = session.Run(inputs);
        for (int k = 0; k < outs.Length; k++)
        {
            Tensor expect = reference[m.OutputNames[k]];
            Assert.Equal(expect.Shape, outs[k].Shape);
            double e = Scaled(outs[k].Detach().Contiguous().Data<float>(), expect.Data<float>());
            Assert.True(e <= 1e-5, $"{name} output {m.OutputNames[k]}: scaled error against the session {e:E2}");
            double p = Scaled(outs[k].Detach().Contiguous().Data<float>(), f.Values($"out.{k}"));
            Assert.True(p <= 1e-5, $"{name} output {k}: scaled error against PyTorch {p:E2}");
        }

        Loss(f, outs).Backward();
        int compared = 0;
        // A gradient that is zero in exact arithmetic (a key bias under softmax) is rounding noise on both sides: it is only required to
        // stay at noise level, far below the largest gradient in the model.
        double gmax = f.Names.Where(n => n.StartsWith("grad.", StringComparison.Ordinal)).Max(n => f.Values(n).Max(Math.Abs));
        foreach ((string pname, Tensor p) in m.NamedParameters())
        {
            if (!f.Names.Contains("grad." + pname))
            {
                continue;
            }

            Assert.False(p.Grad.IsEmpty, $"{name}: no gradient for {pname}");
            ReadOnlySpan<float> want = f.Values("grad." + pname);
            if (want.ToArray().Max(Math.Abs) < 1e-4 * gmax)
            {
                Assert.True(p.Grad.Data<float>().ToArray().Max(Math.Abs) < 1e-6 * gmax, $"{name}: gradient of {pname} should be at noise level");
            }
            else
            {
                double g = Scaled(p.Grad.Data<float>(), want);
                Assert.True(g <= 1e-5, $"{name}: gradient of {pname}: scaled error against PyTorch {g:E2}");
            }

            compared++;
        }

        string[] missing = f.Names.Where(n => n.StartsWith("grad.", StringComparison.Ordinal) && !m.NamedParameters().Any(p => "grad." + p.Name == n)).ToArray();
        Assert.True(missing.Length == 0, $"{name}: reference gradients with no parameter: {string.Join(", ", missing)}; parameters: {string.Join(", ", m.NamedParameters().Select(p => p.Name))}");
        Assert.Equal(f.Names.Count(n => n.StartsWith("grad.", StringComparison.Ordinal)), compared);
    }

    [Fact]
    public void ParametersAreInitializersInFileOrder()
    {
        using IrModel ir = OnnxImporter.Load(Path("mini_modern"));
        string[] expected = ir.Graph.Initializers
            .Where(t => t.DataType == 1 && t.ElementCount > 1)
            .Select(t => t.Name)
            .ToArray();
        using OnnxModule m = OnnxModule.Load(Path("mini_modern"));
        string[] frozen = m.NamedBuffers().Select(b => b.Name).ToArray();
        string[] trained = m.NamedParameters().Select(p => p.Name).ToArray();
        // The rotary frequency tables feed Sin/Cos, so they are buffers; everything else trains, in file order.
        Assert.True(frozen.Count(n => n.StartsWith("onnx::Mul", StringComparison.Ordinal)) >= 2);
        Assert.Equal(expected.Where(n => !frozen.Contains(n)), trained);
        Assert.Contains(trained, n => n.Contains('.', StringComparison.Ordinal));
        Assert.Equal(trained.Length + frozen.Length, m.StateDict().Count);
    }

    [Fact]
    public void FastGradientsAreEqualAtOneAndEightThreads()
    {
        foreach (string name in new[] { "mini_modern", "ops_zoo" })
        {
            OracleFile f = OracleFile.Load(name, "finetune");
            byte[][] hashes = new byte[2][];
            int slot = 0;
            foreach (int threads in new[] { 1, 8 })
            {
                using OnnxModule m = OnnxModule.Load(Path(name));
                using var scope = new ActivationScope();
                using var compute = TensorCompute.Use(ArithmeticMode.Fast, threads);
                Dictionary<string, Tensor> inputs = name == "ops_zoo"
                    ? new() { ["x"] = f.Make("in.x"), ["ids"] = f.MakeInts("in.ids", int64: true) }
                    : Inputs(f, m);
                Loss(f, m.Forward(inputs)).Backward();
                var bytes = new List<byte>();
                foreach ((_, Tensor p) in m.NamedParameters())
                {
                    bytes.AddRange(System.Runtime.InteropServices.MemoryMarshal.AsBytes(p.Grad.Data<float>()).ToArray());
                }

                hashes[slot++] = bytes.ToArray();
            }

            Assert.True(hashes[0].AsSpan().SequenceEqual(hashes[1]), $"{name}: gradients differ between 1 and 8 threads");
        }
    }

    [Fact]
    public void TrainedWeightsRunInTheSessionAndRoundTripThroughTheFormat()
    {
        OracleFile f = OracleFile.Load("mini_bert", "finetune");
        using OnnxModule m = OnnxModule.Load(Path("mini_bert"));
        var opt = new AdamW(m.Parameters(), 1e-2f);
        float first = 0, last = 0;
        Dictionary<string, Tensor> inputs = Inputs(f, m);
        for (int step = 0; step < 5; step++)
        {
            using var scope = new ActivationScope();
            opt.ZeroGrad();
            Tensor loss = Loss(f, m.Forward(inputs));
            if (step == 0)
            {
                first = loss.Data<float>()[0];
            }

            last = loss.Data<float>()[0];
            loss.Backward();
            opt.Step();
        }

        Assert.True(last < first, $"loss did not fall: {first} -> {last}");

        string file = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"onemodel-finetune-{Environment.ProcessId}.onemodel");
        try
        {
            using var scope = new ActivationScope();
            Tensor[] moduleOut = m.Forward(inputs);
            using Session fromModule = m.CreateSession();
            m.Save(file);
            Assert.True(ModelFormat.IsModelFormatFile(file));
            using Session fromFile = Session.Load(file);
            using OnnxModule reloaded = OnnxModule.Load(file);
            for (int k = 0; k < moduleOut.Length; k++)
            {
                ReadOnlySpan<float> want = moduleOut[k].Detach().Contiguous().Data<float>();
                Assert.True(Scaled(fromModule.Run(inputs)[m.OutputNames[k]].Data<float>(), want) <= 1e-4);
                Assert.True(Scaled(fromFile.Run(inputs)[m.OutputNames[k]].Data<float>(), want) <= 1e-4);
                Assert.True(Scaled(reloaded.Forward(inputs)[k].Detach().Contiguous().Data<float>(), want) <= 1e-6);
            }
        }
        finally
        {
            File.Delete(file);
        }
    }

    [Fact]
    public void UnsupportedOperatorsAreAllNamedAtLoad()
    {
        SessionException e = Assert.Throws<SessionException>(() => OnnxModule.Load(Path("unsupported")));
        Assert.Equal(SessionErrorKind.UnsupportedOperator, e.Kind);
        Assert.Equal(["Atan", "Clip", "Sinh"], e.Operators.ToArray());
    }

    [Fact]
    public void FrozenInitializersGetNoGradient()
    {
        OracleFile f = OracleFile.Load("mini_bert", "finetune");
        using OnnxModule m = OnnxModule.Load(Path("mini_bert"), new OnnxModuleOptions { IsTrainable = n => !n.StartsWith("word", StringComparison.Ordinal) });
        Assert.DoesNotContain(m.NamedParameters(), p => p.Name.StartsWith("word", StringComparison.Ordinal));
        Assert.Contains(m.NamedBuffers(), p => p.Name.StartsWith("word", StringComparison.Ordinal));
        using var scope = new ActivationScope();
        Loss(f, m.Forward(Inputs(f, m))).Backward();
        Assert.All(m.NamedParameters(), p => Assert.False(p.Tensor.Grad.IsEmpty));
    }
}
