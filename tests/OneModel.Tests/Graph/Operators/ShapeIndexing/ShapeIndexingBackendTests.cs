using System.Text;
using System.Text.RegularExpressions;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;

namespace RecursiveAI.OneModel.Tests.Graph.Operators.ShapeIndexing;

/// <summary>
/// Runs the onnx/backend/test node cases of the shape and indexing operators in both modes and tallies pass, fail and gap per
/// operator. The data is fetched by tools/fetch_backend_test.py; set ONEMODEL_BACKEND_TEST to its node directory (or its parent).
/// Without it the test does nothing. ONEMODEL_BACKEND_REPORT names a file that receives the per-case list and the table.
/// </summary>
public class ShapeIndexingBackendTests
{
    // Order matters: the first pattern that matches a case names its operator.
    private static readonly (string Op, Regex Pattern)[] Ops =
    [
        ("GatherElements", new Regex("^test_gather_elements_")),
        ("GatherND", new Regex("^test_gathernd_")),
        ("Gather", new Regex("^test_gather_")),
        ("ScatterND", new Regex("^test_scatternd")),
        ("Transpose", new Regex("^test_transpose_")),
        ("Concat", new Regex("^test_concat_")),
        ("Split", new Regex("^test_split_(equal|variable|zero|1d|2d)")),
        ("Slice", new Regex("^test_slice")),
        ("Squeeze", new Regex("^test_squeeze")),
        ("Unsqueeze", new Regex("^test_unsqueeze")),
        ("Flatten", new Regex("^test_flatten_")),
        ("Expand", new Regex("^test_expand_")),
        ("Tile", new Regex("^test_tile")),
        ("ConstantOfShape", new Regex("^test_constantofshape_")),
        ("Range", new Regex("^test_range_")),
        ("Pad", new Regex("^test_(constant_pad|reflect_pad|edge_pad|wrap_pad)")),
        ("Dropout", new Regex("^test_(training_)?dropout_")),
        ("NonZero", new Regex("^test_nonzero_")),
        ("TopK", new Regex("^test_top_k")),
        ("OneHot", new Regex("^test_onehot_")),
        ("Trilu", new Regex("^test_(tril|triu)")),
        ("Shape", new Regex("^test_shape")),
        ("Size", new Regex("^test_size")),
    ];

    [Theory]
    [InlineData(ExecutionMode.Fast)]
    [InlineData(ExecutionMode.Exact)]
    public void NodeCases(ExecutionMode mode)
    {
        string? root = Environment.GetEnvironmentVariable("ONEMODEL_BACKEND_TEST");
        if (string.IsNullOrEmpty(root))
        {
            return;
        }

        string node = Directory.Exists(Path.Combine(root, "node")) ? Path.Combine(root, "node") : root;
        var report = new StringBuilder();
        var failures = new List<string>();
        var table = new SortedDictionary<string, (int Pass, int Fail, int Gap)>(StringComparer.Ordinal);
        foreach (string dir in Directory.GetDirectories(node).Order(StringComparer.Ordinal))
        {
            string name = Path.GetFileName(dir);
            string? op = null;
            foreach ((string o, Regex rx) in Ops)
            {
                if (rx.IsMatch(name))
                {
                    op = o;
                    break;
                }
            }

            if (op is null)
            {
                continue;
            }

            long opset = 0;
            string outcome;
            try
            {
                outcome = BackendNodeTests.RunCase(dir, mode, out opset);
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

            (int p, int f, int g) = table.GetValueOrDefault(op);
            if (outcome == "PASS")
            {
                p++;
            }
            else if (outcome.StartsWith("UNSUPPORTED", StringComparison.Ordinal))
            {
                g++;
            }
            else
            {
                f++;
                failures.Add($"{name}: {outcome}");
            }

            table[op] = (p, f, g);
            report.Append(name).Append(" (opset ").Append(opset).Append("): ").AppendLine(outcome);
        }

        report.AppendLine($"mode {mode}");
        foreach ((string op, (int p, int f, int g)) in table)
        {
            report.AppendLine($"| {op} | {p} | {f} | {g} |");
        }

        string? file = Environment.GetEnvironmentVariable("ONEMODEL_BACKEND_REPORT");
        if (!string.IsNullOrEmpty(file))
        {
            File.AppendAllText(file, report.ToString());
        }

        Assert.True(table.Count > 0, "no node cases found under " + node);
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }
}
