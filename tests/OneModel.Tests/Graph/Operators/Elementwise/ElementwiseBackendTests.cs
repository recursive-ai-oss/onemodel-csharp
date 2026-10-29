using System.Text;
using System.Text.RegularExpressions;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Graph.Operators.Elementwise;

/// <summary>
/// Runs the onnx/backend/test node cases for the element-wise operators in both modes. The data is fetched by
/// tools/fetch_backend_test.py, never committed; set ONEMODEL_BACKEND_TEST to its node directory (or its parent). Without it the
/// test does nothing. ONEMODEL_BACKEND_REPORT names a file that receives a per-case list and a per-operator table.
/// </summary>
public partial class ElementwiseBackendTests
{
    // ONNX's own defaults for node tests: rtol 1e-3, atol 1e-7.
    private const double Rtol = 1e-3;
    private const double Atol = 1e-7;

    [GeneratedRegex("^test_(sub|div|pow|sqrt|reciprocal|neg|abs|exp|log|erf|tanh|sigmoid|hardsigmoid|hardswish|gelu|leakyrelu|clip|min|max|sum|mean|floor|ceil|round|sign|mod|equal|less|greater|less_equal|greater_equal|not|where|cast|castlike|add|mul)(_|$)|^test_(and|or|xor)(\\d|_|$)")]
    private static partial Regex Wanted();

    private static string OpOf(string name)
    {
        Match m = Regex.Match(name, "^test_([a-z]+)");
        return m.Success ? m.Groups[1].Value : name;
    }

    [Theory]
    [InlineData(ExecutionMode.Fast)]
    [InlineData(ExecutionMode.Exact)]
    public void NodeCasesForTheElementwiseOperators(ExecutionMode mode)
    {
        string? root = Environment.GetEnvironmentVariable("ONEMODEL_BACKEND_TEST");
        if (string.IsNullOrEmpty(root))
        {
            return;
        }

        string node = Directory.Exists(Path.Combine(root, "node")) ? Path.Combine(root, "node") : root;
        var report = new StringBuilder();
        var failures = new List<string>();
        var perOp = new SortedDictionary<string, (int Pass, int Fail, int Unsupported)>(StringComparer.Ordinal);
        foreach (string dir in Directory.GetDirectories(node).Order(StringComparer.Ordinal))
        {
            string name = Path.GetFileName(dir);
            if (!Wanted().IsMatch(name))
            {
                continue;
            }

            string outcome;
            try
            {
                outcome = RunCase(dir, mode, out long opset);
                report.Append(name).Append(" (opset ").Append(opset).Append("): ").AppendLine(outcome);
            }
            catch (SessionException e) when (e.Kind is SessionErrorKind.Unsupported or SessionErrorKind.UnsupportedOperator ||
                                             e.Message.Contains("Exact mode is not available", StringComparison.Ordinal))
            {
                // A model with an operator that has no exact kernel (Pow) cannot be created in exact mode: counted with the unsupported.
                outcome = "UNSUPPORTED " + e.Message;
                report.Append(name).Append(": ").AppendLine(outcome);
            }
            catch (Exception e) when (e is OnnxImportException { Kind: OnnxErrorKind.Unsupported } or NotSupportedException)
            {
                outcome = "UNSUPPORTED " + e.Message;
                report.Append(name).Append(": ").AppendLine(outcome);
            }
            catch (Exception e)
            {
                outcome = "FAIL " + e.GetType().Name + ": " + e.Message;
                report.Append(name).Append(": ").AppendLine(outcome);
            }

            string op = OpOf(name);
            (int p, int f, int u) = perOp.GetValueOrDefault(op);
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

            perOp[op] = (p, f, u);
        }

        report.AppendLine($"mode {mode}");
        foreach ((string op, (int p, int f, int u)) in perOp)
        {
            report.AppendLine($"op {op}: pass {p}, fail {f}, unsupported {u}");
        }

        string? file = Environment.GetEnvironmentVariable("ONEMODEL_BACKEND_REPORT");
        if (!string.IsNullOrEmpty(file))
        {
            File.AppendAllText(file, report.ToString());
        }

        Assert.True(perOp.Count > 0, "no node cases found under " + node);
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    private static string RunCase(string dir, ExecutionMode mode, out long opset)
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
                    string? why = Compare(outs[i], expected);
                    if (why is not null)
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

    private static string? Compare(Tensor got, Tensor want)
    {
        if (got.DType != want.DType)
        {
            return $"type {got.DType} vs {want.DType}";
        }

        if (got.Shape != want.Shape)
        {
            return $"shape {got.Shape} vs {want.Shape}";
        }

        byte[] gb = TensorTestHelpers.BytesOf(got), wb = TensorTestHelpers.BytesOf(want);
        if (!TensorTestHelpers.IsFloat(got.DType))
        {
            return gb.AsSpan().SequenceEqual(wb) ? null : "values differ";
        }

        double[] g = TensorTestHelpers.Doubles(got.DType, gb), w = TensorTestHelpers.Doubles(want.DType, wb);
        for (int i = 0; i < g.Length; i++)
        {
            if (double.IsNaN(w[i]) ? double.IsNaN(g[i]) : (g[i] == w[i] || Math.Abs(g[i] - w[i]) <= Atol + (Rtol * Math.Abs(w[i]))))
            {
                continue;
            }

            return $"element {i}: got {g[i]:R}, want {w[i]:R}";
        }

        return null;
    }
}
