using System.IO.Compression;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Graph.Operators.Elementwise;

/// <summary>
/// Runs the element-wise operator cases in testdata/ops-elementwise (small graphs and the outputs ONNX Runtime gave for them,
/// made by tools/ops_elementwise_oracle.py) through the whole stack on every kernel path this machine has, in both modes.
/// Fast and exact results must match the oracle within the case's scaled tolerance; exact results must also be bit-identical
/// across paths.
/// </summary>
public class ElementwiseOracleTests
{
    // Operators without an exact kernel: a session in exact mode refuses them.
    private static readonly HashSet<string> NotExact = ["Pow"];

    private sealed record Tens(DType Type, int[] Shape, byte[] Data);

    private sealed record Case(string Name, string Op, int Opset, double Tol, byte[] Model, Tens?[] Inputs, Tens Output);

    private static string TestDataFile([CallerFilePath] string file = "")
    {
        string? dir = Path.GetDirectoryName(file);
        while (dir is not null && !File.Exists(Path.Combine(dir, "global.json")))
        {
            dir = Path.GetDirectoryName(dir);
        }

        return Path.Combine(dir ?? throw new InvalidOperationException("global.json not found"), "testdata", "ops-elementwise", "cases.json.gz");
    }

    private static DType ParseType(string s) => s switch
    {
        "f32" => DType.F32, "f64" => DType.F64, "f16" => DType.F16, "i8" => DType.I8, "i16" => DType.I16, "i32" => DType.I32,
        "i64" => DType.I64, "u8" => DType.U8, "u16" => DType.U16, "u32" => DType.U32, "u64" => DType.U64, "bool" => DType.Bool,
        _ => throw new InvalidOperationException("type " + s),
    };

    private static Tens ReadTensor(JsonElement e) => new(
        ParseType(e.GetProperty("dtype").GetString()!),
        e.GetProperty("shape").EnumerateArray().Select(d => d.GetInt32()).ToArray(),
        Convert.FromBase64String(e.GetProperty("data").GetString()!));

    private static List<Case> Load()
    {
        using var gz = new GZipStream(File.OpenRead(TestDataFile()), CompressionMode.Decompress);
        using JsonDocument doc = JsonDocument.Parse(gz);
        var list = new List<Case>();
        foreach (JsonElement c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            list.Add(new Case(
                c.GetProperty("name").GetString()!, c.GetProperty("op").GetString()!, c.GetProperty("opset").GetInt32(),
                c.GetProperty("tol").GetDouble(), Convert.FromBase64String(c.GetProperty("model").GetString()!),
                c.GetProperty("inputs").EnumerateArray().Select(i => i.ValueKind == JsonValueKind.Null ? null : ReadTensor(i)).ToArray(),
                ReadTensor(c.GetProperty("output"))));
        }

        return list;
    }

    /// <summary>Null when equal within tolerance, else why not. The scale has a floor of 1e-3 so outputs that are all tiny are judged by absolute error.</summary>
    private static string? Compare(Tens want, Tensor got, double tol)
    {
        if (got.DType != want.Type)
        {
            return $"type {got.DType}, oracle {want.Type}";
        }

        if (!got.Shape.Equals(Shape.Create(want.Shape)))
        {
            return $"shape {got.Shape}, oracle [{string.Join(',', want.Shape)}]";
        }

        byte[] gb = TensorTestHelpers.BytesOf(got);
        if (!TensorTestHelpers.IsFloat(want.Type))
        {
            return gb.AsSpan().SequenceEqual(want.Data) ? null : "integer or bool values differ";
        }

        double[] g = TensorTestHelpers.Doubles(want.Type, gb), w = TensorTestHelpers.Doubles(want.Type, want.Data);
        double scale = 0;
        for (int i = 0; i < w.Length; i++)
        {
            if (double.IsFinite(w[i]))
            {
                scale = Math.Max(scale, Math.Abs(w[i]));
            }
        }

        for (int i = 0; i < w.Length; i++)
        {
            if (double.IsNaN(w[i]) || double.IsNaN(g[i]))
            {
                if (double.IsNaN(w[i]) != double.IsNaN(g[i]))
                {
                    return $"[{i}] NaN mismatch: got {g[i]}, oracle {w[i]}";
                }
            }
            else if (double.IsInfinity(w[i]) || double.IsInfinity(g[i]))
            {
                if (g[i] != w[i])
                {
                    return $"[{i}] infinity mismatch: got {g[i]}, oracle {w[i]}";
                }
            }
            else if (tol == 0)
            {
                if (BitConverter.DoubleToInt64Bits(g[i]) != BitConverter.DoubleToInt64Bits(w[i]))
                {
                    return $"[{i}] got {g[i]:R}, oracle {w[i]:R}";
                }
            }
            else if (Math.Abs(g[i] - w[i]) > tol * Math.Max(scale, 1e-3))
            {
                return $"[{i}] got {g[i]:R}, oracle {w[i]:R} (scale {scale:R})";
            }
        }

        return null;
    }

    private static Tensor RunOne(Case c, ArithmeticMode mode, KernelPath path, int threads)
    {
        using IrModel ir = OnnxImporter.Load(c.Model);
        using var s = Session.Create(ir, new ModelOptions { Mode = mode, IntraOpThreads = threads }, OpRegistry.Default, path, default);
        var present = c.Inputs.Where(i => i is not null).ToArray();
        var tensors = new Tensor[present.Length];
        var names = new string[present.Length];
        for (int i = 0; i < present.Length; i++)
        {
            tensors[i] = TensorTestHelpers.Make(present[i]!.Type, present[i]!.Shape, present[i]!.Data);
            names[i] = s.Inputs[i].Name;
        }

        Span<Tensor> outs = [default];
        try
        {
            s.Run(names, tensors, [s.Outputs[0].Name], outs);
            return outs[0];
        }
        finally
        {
            foreach (Tensor t in tensors)
            {
                t.Dispose();
            }
        }
    }

    [Fact]
    public void MatchesTheOracleOnEveryPathInBothModes()
    {
        List<Case> cases = Load();
        Assert.True(cases.Count > 500);
        var failures = new List<string>();
        var seen = new HashSet<string>();
        void Fail(Case c, ArithmeticMode mode, string message)
        {
            if (seen.Add(c.Name + mode))
            {
                failures.Add($"{c.Name} {mode} {message}");
            }
        }

        foreach (Case c in cases)
        {
            foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
            {
                if (mode == ArithmeticMode.Exact && NotExact.Contains(c.Op))
                {
                    continue;
                }

                byte[]? first = null;
                foreach (KernelPath path in KernelPaths.All.Where(KernelPaths.IsSupported))
                {
                    try
                    {
                        using Tensor got = RunOne(c, mode, path, 1);
                        string? why = Compare(c.Output, got, c.Tol);
                        if (why is not null)
                        {
                            Fail(c, mode, $"{path}: {why}");
                        }

                        if (mode == ArithmeticMode.Exact && TensorTestHelpers.IsFloat(got.DType))
                        {
                            byte[] bits = TensorTestHelpers.BytesOf(got);
                            if (first is null)
                            {
                                first = bits;
                            }
                            else if (!first.AsSpan().SequenceEqual(bits))
                            {
                                Fail(c, mode, $"{path}: bits differ from the first path");
                            }
                        }
                    }
                    catch (Exception e)
                    {
                        Fail(c, mode, $"{path}: {e.GetType().Name}: {e.Message}");
                    }
                }
            }
        }

        Assert.True(failures.Count == 0, $"{failures.Count} failures:\n" + string.Join("\n", failures.Take(80)));
    }

    [Fact]
    public void ExactModeRefusesPow()
    {
        Case c = Load().First(x => x.Op == "Pow");
        Assert.ThrowsAny<Exception>(() => RunOne(c, ArithmeticMode.Exact, KernelPath.Scalar, 1));
    }
}
