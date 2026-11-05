using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Training;

namespace RecursiveAI.OneModel.Tests.Training;

// Fine-tuning the exported multilingual encoders (tools/export_finetune_models.py, tools/finetune_reference.py). The models are
// fetch-only, so these tests do nothing unless ONEMODEL_CACHE points at a cache that holds the exports and the PyTorch reference.
public class OnnxModuleRealModelTests
{
    private sealed class RefFile
    {
        private readonly string _bin;
        private readonly Dictionary<string, (Shape Shape, long Offset)> _items = new();

        public RefFile(string dir)
        {
            _bin = Path.Combine(dir, "ref.bin");
            using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "ref.json")));
            foreach (JsonProperty p in doc.RootElement.GetProperty("tensors").EnumerateObject())
            {
                _items[p.Name] = (Shape.Create(p.Value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray()),
                    p.Value.GetProperty("offset").GetInt64());
            }
        }

        public IEnumerable<string> Names => _items.Keys;

        public Shape ShapeOf(string name) => _items[name].Shape;

        public float[] Read(string name)
        {
            (Shape shape, long offset) = _items[name];
            var values = new float[shape.Length];
            using var fs = new FileStream(_bin, FileMode.Open, FileAccess.Read, FileShare.Read);
            fs.Position = offset * sizeof(float);
            var bytes = new byte[Math.Min(values.Length, 1 << 24) * sizeof(float)];
            int done = 0;
            while (done < values.Length)
            {
                int n = Math.Min(values.Length - done, bytes.Length / sizeof(float));
                fs.ReadExactly(bytes, 0, n * sizeof(float));
                Buffer.BlockCopy(bytes, 0, values, done * sizeof(float), n * sizeof(float));
                done += n;
            }

            return values;
        }

        public Tensor Make(string name) => Tensor.FromSpan<float>(Read(name), _items[name].Shape);

        public Tensor MakeIds(string name) => Tensor.FromSpan<long>(Read(name).Select(v => (long)v).ToArray(), _items[name].Shape);
    }

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

    [Theory]
    [InlineData("xlm_roberta_base")]
    [InlineData("mmbert_base")]
    public void FineTuneMatchesPyTorch(string name)
    {
        string? cache = Environment.GetEnvironmentVariable("ONEMODEL_CACHE");
        string model = cache is null ? "" : Path.Combine(cache, "models", name, name + ".onnx");
        string refDir = cache is null ? "" : Path.Combine(cache, "finetune", name);
        Assert.SkipUnless(File.Exists(model) && File.Exists(Path.Combine(refDir, "ref.bin")), $"{name} export or reference not found under ONEMODEL_CACHE");
        int threads = int.TryParse(Environment.GetEnvironmentVariable("ONEMODEL_FINETUNE_THREADS"), out int t) ? t : 8;
        TensorCompute.Threads = threads;
        var log = new List<string>();
        void Say(string line)
        {
            log.Add(line);
            Console.WriteLine(line);
        }

        var rf = new RefFile(refDir);
        var sw = Stopwatch.StartNew();
        using OnnxModule m = OnnxModule.Load(model);
        Say($"{name}: loaded in {sw.Elapsed.TotalSeconds:F1} s, {m.NamedParameters().Count} parameters, {m.NamedBuffers().Count} buffers, inputs {string.Join(",", m.InputNames)}");
        Tensor ids = rf.MakeIds("in.input_ids"), mask = rf.MakeIds("in.attention_mask");
        var inputs = new Dictionary<string, Tensor> { ["input_ids"] = ids, ["attention_mask"] = mask };

        // Forward against the graph session and against PyTorch.
        using (Session session = Session.Load(model, new ModelOptions { IntraOpThreads = threads }))
        using (var scope = new ActivationScope())
        {
            Tensor hidden = m.Forward(inputs)[0];
            using SessionOutputs r = session.Run(inputs);
            double vsSession = Scaled(hidden.Detach().Contiguous().Data<float>(), r[m.OutputNames[0]].Data<float>());
            double vsTorch = Scaled(hidden.Detach().Contiguous().Data<float>(), rf.Read("out.hidden"));
            Say($"forward: scaled error vs Session {vsSession:E2}, vs PyTorch {vsTorch:E2}");
            Assert.True(vsSession <= 1e-4, $"forward vs session {vsSession:E2}");
            Assert.True(vsTorch <= 1e-4, $"forward vs PyTorch {vsTorch:E2}");
        }

        Tensor proj = rf.Make("in.P"), target = rf.Make("in.target");
        float[] refLosses = rf.Read("out.losses");
        var opt = new AdamW(m.Parameters(), float.Parse(JsonDocument.Parse(File.ReadAllText(Path.Combine(refDir, "ref.json"))).RootElement.GetProperty("lr").GetRawText(), CultureInfo.InvariantCulture));
        var stepTimes = new List<double>();
        var parameters = m.NamedParameters().ToDictionary(p => p.Name, p => p.Tensor);
        for (int step = 0; step < refLosses.Length; step++)
        {
            sw.Restart();
            using var scope = new ActivationScope();
            opt.ZeroGrad();
            Tensor pooled = Ops.Mean(m.Forward(inputs)[0], 1);
            Tensor loss = Losses.Mse(Ops.MatMul(pooled, proj), target);
            float value = loss.Data<float>()[0];
            loss.Backward();
            if (step == 0)
            {
                foreach (string g in rf.Names.Where(n => n.StartsWith("grad.", StringComparison.Ordinal)))
                {
                    Tensor grad = parameters[g["grad.".Length..]].Grad;
                    Assert.False(grad.IsEmpty, $"no gradient for {g}");
                    double e = Scaled(grad.Data<float>(), rf.Read(g));
                    Say($"gradient {g[5..]}: scaled error vs PyTorch {e:E2}");
                    Assert.True(e <= 1e-4, $"{g}: {e:E2}");
                }
            }

            opt.Step();
            stepTimes.Add(sw.Elapsed.TotalSeconds);
            double rel = Math.Abs(value - refLosses[step]) / Math.Abs(refLosses[step]);
            Say($"step {step}: loss {value:G9} (PyTorch {refLosses[step]:G9}), relative difference {rel:E2}, {stepTimes[^1]:F2} s");
            Assert.True(rel <= 1e-4, $"step {step}: loss {value} vs {refLosses[step]}");
        }

        // The trained weights run in the graph session and match the module.
        using (var scope = new ActivationScope())
        {
            Tensor hidden = m.Forward(inputs)[0];
            using Session trained = m.CreateSession(new ModelOptions { IntraOpThreads = threads });
            using SessionOutputs r = trained.Run(inputs);
            double e = Scaled(r[m.OutputNames[0]].Data<float>(), hidden.Detach().Contiguous().Data<float>());
            Say($"after {refLosses.Length} steps: written-back model in Session vs module forward, scaled error {e:E2}");
            Assert.True(e <= 1e-4, $"round trip {e:E2}");
        }

        using Process self = Process.GetCurrentProcess();
        Say($"threads {threads}; step time mean {stepTimes.Skip(1).DefaultIfEmpty(stepTimes[0]).Average():F2} s (first {stepTimes[0]:F2} s); peak RSS {self.PeakWorkingSet64 / (1024.0 * 1024 * 1024):F2} GiB");
        File.WriteAllLines(Path.Combine(refDir, "result.txt"), log);
    }
}
