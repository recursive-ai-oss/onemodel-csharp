using System.Text;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Graph.Operators.ShapeIndexing;

/// <summary>
/// Compares the shape and indexing operators with ONNX Runtime on the cases tools/ops_shape_indexing.py generates (one node per
/// model, edge cases included). Set ONEMODEL_OPS_ORACLE to the directory the script wrote; without it the test does nothing.
/// A case ONNX Runtime refuses (an `expect_error` file) must fail here too, unless it is one of the listed lenient cases where we
/// deliberately accept more: those are counted, not failed.
/// </summary>
public class ShapeIndexingOracleTests
{
    // Inputs ONNX Runtime rejects and we define instead, or whose result the spec leaves open (docs/ops-shape-indexing.md): by case tag,
    // or by the start of ONNX Runtime's message (an element-type combination it has no kernel for).
    private static readonly string[] Lenient = ["_reflect", "_negative", "_num", "_scalar_indices"];

    [Theory]
    [InlineData(ExecutionMode.Fast)]
    [InlineData(ExecutionMode.Exact)]
    public void MatchesOnnxRuntime(ExecutionMode mode)
    {
        string? root = Environment.GetEnvironmentVariable("ONEMODEL_OPS_ORACLE");
        if (string.IsNullOrEmpty(root))
        {
            return;
        }

        var report = new StringBuilder();
        var failures = new List<string>();
        var table = new SortedDictionary<string, (int Pass, int Fail, int Errors, int Lenient)>(StringComparer.Ordinal);
        foreach (string dir in Directory.GetDirectories(root).Order(StringComparer.Ordinal))
        {
            string name = Path.GetFileName(dir);
            string op = name.Split('_')[1];
            (int p, int f, int e, int l) = table.GetValueOrDefault(op);
            string outcome;
            if (name.EndsWith("_nan", StringComparison.Ordinal))
            {
                // NaN order in TopK is not specified, and ONNX Runtime's differs between its selection paths.
                outcome = "LENIENT (NaN ordering is unspecified; not compared)";
            }
            else if (File.Exists(Path.Combine(dir, "expect_error")))
            {
                // The input data of an error case exists, the expected outputs do not: run the model and expect a refusal.
                outcome = RunExpectingError(dir, mode);
            }
            else
            {
                try
                {
                    outcome = BackendNodeTests.RunCase(dir, mode, out _);
                }
                catch (Exception ex)
                {
                    outcome = "FAIL " + ex.GetType().Name + ": " + ex.Message;
                }
            }

            if (outcome == "PASS")
            {
                p++;
            }
            else if (outcome == "REFUSED")
            {
                e++;
            }
            else if (outcome.StartsWith("LENIENT", StringComparison.Ordinal))
            {
                l++;
            }
            else
            {
                f++;
                failures.Add($"{name}: {outcome}");
            }

            table[op] = (p, f, e, l);
            report.Append(name).Append(": ").AppendLine(outcome);
        }

        report.AppendLine($"mode {mode}");
        foreach ((string op, (int p, int f, int e, int l)) in table)
        {
            report.AppendLine($"| {op} | {p} | {f} | {e} | {l} |");
        }

        string? file = Environment.GetEnvironmentVariable("ONEMODEL_OPS_REPORT");
        if (!string.IsNullOrEmpty(file))
        {
            File.AppendAllText(file, report.ToString());
        }

        Assert.True(table.Count > 0, "no cases found under " + root);
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures.Take(40)));
    }

    private static string RunExpectingError(string dir, ExecutionMode mode)
    {
        string name = Path.GetFileName(dir);
        try
        {
            using IrModel model = OnnxImporter.Load(Path.Combine(dir, "model.onnx"));
            using var session = Session.Create(model, new SessionOptions { Mode = mode });
            var inputs = new Tensor[session.Inputs.Length];
            var names = new string[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                names[i] = session.Inputs[i].Name;
                inputs[i] = OnnxImporter.LoadTensor(File.ReadAllBytes(Path.Combine(dir, "test_data_set_0", $"input_{i}.pb"))).ToTensor();
            }

            var outNames = session.Outputs.Select(o => o.Name).ToArray();
            var outs = new Tensor[outNames.Length];
            try
            {
                session.Run(names, inputs, outNames, outs);
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
        catch (SessionException)
        {
            return "REFUSED";
        }
        catch (Exception ex)
        {
            return "FAIL " + ex.GetType().Name + " instead of a SessionException: " + ex.Message;
        }

        string why = File.ReadAllText(Path.Combine(dir, "expect_error"));
        return Lenient.Any(name.Contains) || why.Contains("NOT_IMPLEMENTED", StringComparison.Ordinal) ? "LENIENT (ONNX Runtime refuses, we compute a result)" : "FAIL ONNX Runtime refuses this case but we ran it";
    }
}
