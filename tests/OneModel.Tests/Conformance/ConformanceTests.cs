using System.Text;
using RecursiveAI.OneModel.Graph;

namespace RecursiveAI.OneModel.Tests.Conformance;

/// <summary>
/// Runs every onnx/backend/test case (node, simple, pytorch-converted, pytorch-operator) in both modes and requires that none
/// fails: a case either passes or is a listed gap (see docs/conformance.md). The data is fetched by tools/fetch_backend_test.py,
/// never committed; set ONEMODEL_BACKEND_TEST to its directory. Without it the test does nothing. ONEMODEL_BACKEND_REPORT names a
/// file that receives the per-case result lines.
/// </summary>
public class ConformanceTests
{
    [Theory]
    [InlineData(ArithmeticMode.Fast)]
    [InlineData(ArithmeticMode.Exact)]
    public void NoBackendCaseFails(ArithmeticMode mode)
    {
        string? root = Environment.GetEnvironmentVariable("ONEMODEL_BACKEND_TEST");
        if (string.IsNullOrEmpty(root))
        {
            return;
        }

        var cases = RecursiveAI.OneModel.Conformance.ConformanceRunner.FindCases(root);
        Assert.True(cases.Count > 0, "no cases found under " + root);
        var results = cases.Select(c => RecursiveAI.OneModel.Conformance.ConformanceRunner.RunCase(c.Group, c.Dir, mode)).ToList();
        string? file = Environment.GetEnvironmentVariable("ONEMODEL_BACKEND_REPORT");
        if (!string.IsNullOrEmpty(file))
        {
            File.AppendAllText(file, RecursiveAI.OneModel.Conformance.ConformanceRunner.ToTsv(results));
        }

        var failed = results.Where(r => r.Status == RecursiveAI.OneModel.Conformance.CaseStatus.Fail).ToList();
        var sb = new StringBuilder();
        foreach (var r in failed)
        {
            sb.Append(r.Group).Append('/').Append(r.Name).Append(": ").AppendLine(r.Reason);
        }

        Assert.True(failed.Count == 0, $"{failed.Count} of {results.Count} cases fail in {mode} mode:{Environment.NewLine}{sb}");
    }
}
