using System.Text.Json;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Graph.Operators.Norms;

/// <summary>
/// Compares the operators with ONNX Runtime's outputs on the randomized cases written by tools/ops_norms.py. Set ONEMODEL_OPS_ORACLE to
/// the directory the script wrote; without it the test does nothing. Floats must agree within a scaled error of 1e-4 (NaN and
/// infinities in the same places), integers exactly. Fast mode must give the same bits on one thread and many; exact mode must give
/// the same bits on every kernel path.
/// </summary>
public class OracleNormsTests
{
    private sealed record Spec(string Name, string DType, int[] Shape);

    private sealed record Meta(List<Spec> Inputs, List<Spec> Outputs);

    private static Meta ReadMeta(string dir)
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "meta.json")));
        static List<Spec> Specs(JsonElement e) =>
        [
            .. e.EnumerateArray().Select(s => new Spec(
                s.TryGetProperty("name", out JsonElement n) ? n.GetString()! : "",
                s.GetProperty("dtype").GetString()!,
                [.. s.GetProperty("shape").EnumerateArray().Select(d => d.GetInt32())])),
        ];
        return new Meta(Specs(doc.RootElement.GetProperty("inputs")), Specs(doc.RootElement.GetProperty("outputs")));
    }

    private static Tensor[] RunCase(string dir, Meta meta, ExecutionMode mode, KernelPath? path, int threads)
    {
        using IrModel ir = OnnxImporter.Load(Path.Combine(dir, "model.onnx"));
        using var s = Session.Create(ir, new SessionOptions { Mode = mode, IntraOpThreads = threads }, OpRegistry.Default, path, default);
        var inputs = new Tensor[meta.Inputs.Count];
        var names = new string[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            Spec sp = meta.Inputs[i];
            byte[] raw = File.ReadAllBytes(Path.Combine(dir, $"input_{i}.bin"));
            inputs[i] = sp.DType == "float32"
                ? Tensor.FromSpan<float>(System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(raw), Shape.Create(sp.Shape))
                : Tensor.FromSpan<long>(System.Runtime.InteropServices.MemoryMarshal.Cast<byte, long>(raw), Shape.Create(sp.Shape));
            names[i] = s.Inputs[i].Name;
        }

        var outNames = new string[meta.Outputs.Count];
        for (int i = 0; i < outNames.Length; i++)
        {
            outNames[i] = s.Outputs[i].Name;
        }

        var outs = new Tensor[outNames.Length];
        s.Run(names, inputs, outNames, outs);
        foreach (Tensor t in inputs)
        {
            t.Dispose();
        }

        return outs;
    }

    private static string? Compare(Tensor got, Spec want, byte[] raw)
    {
        if (got.Shape != Shape.Create(want.Shape))
        {
            return $"shape {got.Shape} vs [{string.Join(',', want.Shape)}]";
        }

        if (want.DType == "int64")
        {
            long[] w = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, long>(raw).ToArray();
            long[] g = got.ToArray<long>();
            for (int i = 0; i < w.Length; i++)
            {
                if (g[i] != w[i])
                {
                    return $"element {i}: {g[i]} vs {w[i]}";
                }
            }

            return null;
        }

        float[] wf = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(raw).ToArray();
        float[] gf = got.ToArray<float>();
        double max = 0, scale = 1e-30;
        for (int i = 0; i < wf.Length; i++)
        {
            if (float.IsNaN(wf[i]) || float.IsInfinity(wf[i]) || float.IsNaN(gf[i]) || float.IsInfinity(gf[i]))
            {
                if (!(float.IsNaN(wf[i]) && float.IsNaN(gf[i])) && wf[i] != gf[i])
                {
                    return $"element {i}: {gf[i]} vs {wf[i]}";
                }

                continue;
            }

            max = Math.Max(max, Math.Abs((double)gf[i] - wf[i]));
            scale = Math.Max(scale, Math.Abs(wf[i]));
        }

        return max / scale <= 1e-4 ? null : $"scaled error {max / scale:g3}";
    }

    private static uint[] Bits(Tensor t) => t.DType == DType.F32 ? t.ToArray<float>().Select(BitConverter.SingleToUInt32Bits).ToArray() : t.ToArray<long>().Select(v => (uint)v).ToArray();

    [Fact]
    public void MatchesOnnxRuntimeOnRandomizedCases()
    {
        string? root = Environment.GetEnvironmentVariable("ONEMODEL_OPS_ORACLE");
        if (string.IsNullOrEmpty(root))
        {
            return;
        }

        var failures = new List<string>();
        int cases = 0;
        foreach (string dir in Directory.GetDirectories(root).Order(StringComparer.Ordinal))
        {
            string name = Path.GetFileName(dir);
            cases++;
            try
            {
                Meta meta = ReadMeta(dir);
                Tensor[] fast = RunCase(dir, meta, ExecutionMode.Fast, null, 1);
                Tensor[] many = RunCase(dir, meta, ExecutionMode.Fast, null, 7);
                Tensor[] exact = RunCase(dir, meta, ExecutionMode.Exact, KernelPath.Scalar, 1);
                for (int i = 0; i < fast.Length; i++)
                {
                    byte[] raw = File.ReadAllBytes(Path.Combine(dir, $"output_{i}.bin"));
                    string? e = Compare(fast[i], meta.Outputs[i], raw) ?? Compare(exact[i], meta.Outputs[i], raw);
                    if (e is not null)
                    {
                        failures.Add($"{name} output {i}: {e}");
                    }

                    if (!Bits(fast[i]).SequenceEqual(Bits(many[i])))
                    {
                        failures.Add($"{name} output {i}: fast mode differs between 1 and 7 threads");
                    }
                }

                foreach (KernelPath path in KernelPaths.All.Where(KernelPaths.IsSupported).Where(p => p != KernelPath.Scalar))
                {
                    Tensor[] other = RunCase(dir, meta, ExecutionMode.Exact, path, 3);
                    for (int i = 0; i < exact.Length; i++)
                    {
                        if (!Bits(exact[i]).SequenceEqual(Bits(other[i])))
                        {
                            failures.Add($"{name} output {i}: exact mode differs between Scalar and {path}");
                        }

                        other[i].Dispose();
                    }
                }

                foreach (Tensor t in fast.Concat(many).Concat(exact))
                {
                    t.Dispose();
                }
            }
            catch (Exception ex)
            {
                failures.Add($"{name}: {ex.GetType().Name}: {ex.Message}");
            }
        }

        Assert.True(cases > 0, "no cases under " + root);
        Assert.True(failures.Count == 0, $"{failures.Count} of {cases} cases failed:{Environment.NewLine}{string.Join(Environment.NewLine, failures.Take(40))}");
    }
}
