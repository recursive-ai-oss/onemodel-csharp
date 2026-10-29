using System.Runtime.CompilerServices;
using System.Text;
using System.Text.RegularExpressions;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Graph;

/// <summary>
/// Runs the operator cases from onnx/backend/test/data/node for the operators implemented so far. The data is fetched by
/// tools/fetch_backend_test.py, never committed; set ONEMODEL_BACKEND_TEST to its node directory (or its parent). Without it the
/// test does nothing. ONEMODEL_BACKEND_REPORT names a file that receives the per-case result list.
/// </summary>
public partial class BackendNodeTests
{
    // ONNX's own defaults for node tests (onnx/backend/test/runner): rtol 1e-3, atol 1e-7.
    private const double Rtol = 1e-3;
    private const double Atol = 1e-7;

    [GeneratedRegex("^test_(add|mul|relu|matmul|gemm|reshape|identity|constant)(_|$)")]
    private static partial Regex Wanted();

    [Theory]
    [InlineData(ExecutionMode.Fast)]
    [InlineData(ExecutionMode.Exact)]
    public void NodeCasesForTheFirstOperatorSet(ExecutionMode mode)
    {
        string? root = Environment.GetEnvironmentVariable("ONEMODEL_BACKEND_TEST");
        if (string.IsNullOrEmpty(root))
        {
            return;
        }

        string node = Directory.Exists(Path.Combine(root, "node")) ? Path.Combine(root, "node") : root;
        var report = new StringBuilder();
        var failures = new List<string>();
        var perOpset = new SortedDictionary<long, (int Pass, int Fail, int Unsupported)>();
        foreach (string dir in Directory.GetDirectories(node).Order(StringComparer.Ordinal))
        {
            string name = Path.GetFileName(dir);
            if (!Wanted().IsMatch(name) || name.Contains("constantofshape", StringComparison.Ordinal) ||
                name.Contains("constant_pad", StringComparison.Ordinal) || name.Contains("matmulinteger", StringComparison.Ordinal))
            {
                continue;
            }

            long opset = 0;
            string outcome;
            try
            {
                outcome = RunCase(dir, mode, out opset);
            }
            catch (SessionException e) when (e.Kind is SessionErrorKind.Unsupported or SessionErrorKind.UnsupportedOperator)
            {
                outcome = "UNSUPPORTED " + e.Message;
            }
            catch (OnnxImportException e) when (e.Kind == OnnxErrorKind.Unsupported)
            {
                outcome = "UNSUPPORTED " + e.Message;
            }
            catch (Exception e)
            {
                outcome = "FAIL " + e.GetType().Name + ": " + e.Message;
            }

            (int p, int f, int u) = perOpset.GetValueOrDefault(opset);
            if (outcome == "PASS")
            {
                p++;
            }
            else if (outcome.StartsWith("UNSUPPORTED", StringComparison.Ordinal))
            {
                u++;
            }
            else
            {
                f++;
                failures.Add($"{name}: {outcome}");
            }

            perOpset[opset] = (p, f, u);
            report.Append(name).Append(" (opset ").Append(opset).Append("): ").AppendLine(outcome);
        }

        report.AppendLine($"mode {mode}");
        foreach ((long opset, (int p, int f, int u)) in perOpset)
        {
            report.AppendLine($"opset {opset}: pass {p}, fail {f}, unsupported {u}");
        }

        string? file = Environment.GetEnvironmentVariable("ONEMODEL_BACKEND_REPORT");
        if (!string.IsNullOrEmpty(file))
        {
            File.AppendAllText(file, report.ToString());
        }

        Assert.True(perOpset.Count > 0, "no node cases found under " + node);
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    internal static string RunCase(string dir, ExecutionMode mode, out long opset)
    {
        using IrModel model = OnnxImporter.Load(Path.Combine(dir, "model.onnx"));
        opset = model.DefaultOpset;
        using var session = Session.Create(model, new SessionOptions { Mode = mode });
        int sets = 0;
        foreach (string setDir in Directory.GetDirectories(dir, "test_data_set_*").Order(StringComparer.Ordinal))
        {
            sets++;
            var inputs = new Tensor[session.Inputs.Length];
            var names = new string[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                names[i] = session.Inputs[i].Name;
                inputs[i] = OnnxImporter.LoadTensor(File.ReadAllBytes(Path.Combine(setDir, $"input_{i}.pb"))).ToTensor();
            }

            var outNames = new string[session.Outputs.Length];
            var outs = new Tensor[outNames.Length];
            for (int i = 0; i < outNames.Length; i++)
            {
                outNames[i] = session.Outputs[i].Name;
            }

            try
            {
                session.Run(names, inputs, outNames, outs);
                for (int i = 0; i < outs.Length; i++)
                {
                    using Tensor expected = OnnxImporter.LoadTensor(File.ReadAllBytes(Path.Combine(setDir, $"output_{i}.pb"))).ToTensor();
                    if (!Same(outs[i], expected, out string why))
                    {
                        return $"FAIL output {i} ({outNames[i]}): {why}";
                    }
                }
            }
            finally
            {
                foreach (Tensor t in inputs)
                {
                    t.Dispose();
                }

                foreach (Tensor t in outs)
                {
                    if (!t.IsEmpty)
                    {
                        t.Dispose();
                    }
                }
            }
        }

        return sets == 0 ? "FAIL no test data sets" : "PASS";
    }

    private static bool Same(Tensor got, Tensor want, out string why)
    {
        why = "";
        if (got.DType != want.DType)
        {
            why = $"type {got.DType} vs {want.DType}";
            return false;
        }

        if (got.Shape != want.Shape)
        {
            why = $"shape {got.Shape} vs {want.Shape}";
            return false;
        }

        switch (got.DType)
        {
            case DType.F32:
                return Close(got.ToArray<float>(), want.ToArray<float>(), out why);
            case DType.F64:
                return Close(got.ToArray<double>(), want.ToArray<double>(), out why);
            case DType.F16:
                return Close(got.ToArray<Half>().Select(v => (double)v).ToArray(), want.ToArray<Half>().Select(v => (double)v).ToArray(), out why);
            case DType.BF16:
                return Close(got.ToArray<BFloat16>().Select(v => (double)(float)v).ToArray(), want.ToArray<BFloat16>().Select(v => (double)(float)v).ToArray(), out why);
            case DType.I8: return Exactly(got.ToArray<sbyte>(), want.ToArray<sbyte>(), out why);
            case DType.U8: return Exactly(got.ToArray<byte>(), want.ToArray<byte>(), out why);
            case DType.I16: return Exactly(got.ToArray<short>(), want.ToArray<short>(), out why);
            case DType.U16: return Exactly(got.ToArray<ushort>(), want.ToArray<ushort>(), out why);
            case DType.I32: return Exactly(got.ToArray<int>(), want.ToArray<int>(), out why);
            case DType.U32: return Exactly(got.ToArray<uint>(), want.ToArray<uint>(), out why);
            case DType.I64: return Exactly(got.ToArray<long>(), want.ToArray<long>(), out why);
            case DType.U64: return Exactly(got.ToArray<ulong>(), want.ToArray<ulong>(), out why);
            case DType.Bool: return Exactly(got.ToArray<bool>(), want.ToArray<bool>(), out why);
            default:
                why = $"comparison of {got.DType} is not implemented";
                return false;
        }
    }

    private static bool Close(float[] got, float[] want, out string why) => Close(got.Select(v => (double)v).ToArray(), want.Select(v => (double)v).ToArray(), out why);

    private static bool Close(double[] got, double[] want, out string why)
    {
        why = "";
        for (int i = 0; i < got.Length; i++)
        {
            double g = got[i], w = want[i];
            if (double.IsNaN(w) ? double.IsNaN(g) : (g == w || Math.Abs(g - w) <= Atol + (Rtol * Math.Abs(w))))
            {
                continue;
            }

            why = $"element {i}: got {g:R}, want {w:R}";
            return false;
        }

        return true;
    }

    private static bool Exactly<T>(T[] got, T[] want, out string why) where T : unmanaged
    {
        why = "";
        for (int i = 0; i < got.Length; i++)
        {
            if (!EqualityComparer<T>.Default.Equals(got[i], want[i]))
            {
                why = $"element {i}: got {got[i]}, want {want[i]}";
                return false;
            }
        }

        return true;
    }
}
