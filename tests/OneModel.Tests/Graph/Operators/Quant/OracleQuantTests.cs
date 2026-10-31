using System.Runtime.InteropServices;
using System.Text.Json;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Graph.Operators.Quant;

/// <summary>
/// Compares the 8-bit quantization operators with ONNX Runtime's outputs on the cases written by tools/ops_quant_oracle.py. Set
/// ONEMODEL_QUANT_ORACLE to the directory the script wrote; without it the test does nothing. Integer outputs must be equal
/// (a case may allow one step in the last place where ONNX Runtime requantizes with its own float rounding), float outputs equal
/// to a scaled error of 1e-6 unless the case states more (whole quantized networks, where one flipped rounding is amplified). The result must be identical on every kernel path and on one thread and many, in both modes.
/// </summary>
public class OracleQuantTests
{
    private sealed record Spec(string Name, string DType, int[] Shape);

    private static (List<Spec> In, List<Spec> Out, int Tol, double FloatTol) ReadMeta(string dir)
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "meta.json")));
        static List<Spec> Specs(JsonElement e) =>
        [
            .. e.EnumerateArray().Select(s => new Spec(
                s.TryGetProperty("name", out JsonElement n) ? n.GetString()! : "",
                s.GetProperty("dtype").GetString()!,
                [.. s.GetProperty("shape").EnumerateArray().Select(d => d.GetInt32())])),
        ];
        return (Specs(doc.RootElement.GetProperty("inputs")), Specs(doc.RootElement.GetProperty("outputs")), doc.RootElement.GetProperty("tolerance").GetInt32(),
            doc.RootElement.TryGetProperty("float_tol", out JsonElement ft) ? ft.GetDouble() : 1e-6);
    }

    private static Tensor FromRaw(Spec sp, byte[] raw)
    {
        Shape shape = Shape.Create(sp.Shape);
        return sp.DType switch
        {
            "float32" => Tensor.FromSpan(MemoryMarshal.Cast<byte, float>(raw), shape),
            "uint8" => Tensor.FromSpan<byte>(raw, shape),
            "int8" => Tensor.FromSpan(MemoryMarshal.Cast<byte, sbyte>(raw), shape),
            "int32" => Tensor.FromSpan(MemoryMarshal.Cast<byte, int>(raw), shape),
            "uint16" => Tensor.FromSpan(MemoryMarshal.Cast<byte, ushort>(raw), shape),
            "int16" => Tensor.FromSpan(MemoryMarshal.Cast<byte, short>(raw), shape),
            _ => throw new NotSupportedException(sp.DType),
        };
    }

    private static Tensor[] RunCase(string dir, List<Spec> ins, int outputs, ArithmeticMode mode, KernelPath? path, int threads)
    {
        using IrModel ir = OnnxImporter.Load(Path.Combine(dir, "model.onnx"));
        using var s = Session.Create(ir, new ModelOptions { Mode = mode, IntraOpThreads = threads }, OpRegistry.Default, path, default);
        var inputs = new Tensor[ins.Count];
        var names = new string[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            inputs[i] = FromRaw(ins[i], File.ReadAllBytes(Path.Combine(dir, $"input_{i}.bin")));
            names[i] = s.Inputs[i].Name;
        }

        var outNames = new string[outputs];
        for (int i = 0; i < outNames.Length; i++)
        {
            outNames[i] = s.Outputs[i].Name;
        }

        var outs = new Tensor[outputs];
        s.Run(names, inputs, outNames, outs);
        foreach (Tensor t in inputs)
        {
            t.Dispose();
        }

        return outs;
    }

    private static byte[] Bytes(Tensor t) => t.DType switch
    {
        DType.F32 => MemoryMarshal.AsBytes(t.ToArray<float>().AsSpan()).ToArray(),
        DType.U8 => t.ToArray<byte>(),
        DType.I8 => MemoryMarshal.AsBytes(t.ToArray<sbyte>().AsSpan()).ToArray(),
        DType.I32 => MemoryMarshal.AsBytes(t.ToArray<int>().AsSpan()).ToArray(),
        DType.U16 => MemoryMarshal.AsBytes(t.ToArray<ushort>().AsSpan()).ToArray(),
        DType.I16 => MemoryMarshal.AsBytes(t.ToArray<short>().AsSpan()).ToArray(),
        _ => throw new NotSupportedException(t.DType.ToString()),
    };

    private static long At(byte[] raw, string dtype, int i) => dtype switch
    {
        "uint8" => raw[i],
        "int8" => (sbyte)raw[i],
        "int32" => BitConverter.ToInt32(raw, i * 4),
        "uint16" => BitConverter.ToUInt16(raw, i * 2),
        "int16" => BitConverter.ToInt16(raw, i * 2),
        _ => throw new NotSupportedException(dtype),
    };

    /// <summary>Returns an error text, or null; <paramref name="offByOne"/> counts elements that differ by one step.</summary>
    private static string? Compare(Tensor got, Spec want, byte[] raw, int tol, double floatTol, ref int offByOne)
    {
        if (got.Shape != Shape.Create(want.Shape))
        {
            return $"shape {got.Shape} vs [{string.Join(',', want.Shape)}]";
        }

        byte[] g = Bytes(got);
        if (want.DType == "float32")
        {
            float[] wf = MemoryMarshal.Cast<byte, float>(raw).ToArray();
            float[] gf = MemoryMarshal.Cast<byte, float>(g).ToArray();
            double max = 0, scale = 1e-30;
            for (int i = 0; i < wf.Length; i++)
            {
                scale = Math.Max(scale, Math.Abs((double)wf[i]));
                max = Math.Max(max, Math.Abs((double)gf[i] - wf[i]));
            }

            return max / scale <= floatTol ? null : $"scaled error {max / scale:g3}";
        }

        int n = want.Shape.Aggregate(1, (a, b) => a * b);
        for (int i = 0; i < n; i++)
        {
            long d = Math.Abs(At(g, want.DType, i) - At(raw, want.DType, i));
            if (d == 0)
            {
                continue;
            }

            if (d > tol)
            {
                return $"element {i}: {At(g, want.DType, i)} vs {At(raw, want.DType, i)}";
            }

            offByOne++;
        }

        return null;
    }

    [Fact]
    public void MatchesOnnxRuntime()
    {
        string? root = Environment.GetEnvironmentVariable("ONEMODEL_QUANT_ORACLE");
        if (string.IsNullOrEmpty(root))
        {
            return;
        }

        var failures = new List<string>();
        int cases = 0, offByOne = 0, elements = 0;
        foreach (string dir in Directory.GetDirectories(root).Order(StringComparer.Ordinal))
        {
            string name = Path.GetFileName(dir);
            cases++;
            try
            {
                (List<Spec> ins, List<Spec> outs, int tol, double floatTol) = ReadMeta(dir);
                Tensor[] fast = RunCase(dir, ins, outs.Count, ArithmeticMode.Fast, null, 1);
                Tensor[] many = RunCase(dir, ins, outs.Count, ArithmeticMode.Fast, null, 7);
                Tensor[] exact = RunCase(dir, ins, outs.Count, ArithmeticMode.Exact, null, 3);
                for (int i = 0; i < outs.Count; i++)
                {
                    byte[] raw = File.ReadAllBytes(Path.Combine(dir, $"output_{i}.bin"));
                    string? e = Compare(fast[i], outs[i], raw, tol, floatTol, ref offByOne);
                    elements += outs[i].Shape.Aggregate(1, (a, b) => a * b);
                    if (e is not null)
                    {
                        failures.Add($"{name} output {i}: {e}");
                    }

                    if (!Bytes(fast[i]).SequenceEqual(Bytes(many[i])) || !Bytes(fast[i]).SequenceEqual(Bytes(exact[i])))
                    {
                        failures.Add($"{name} output {i}: results differ between 1 thread, 7 threads and exact mode");
                    }
                }

                foreach (KernelPath path in KernelPaths.All.Where(KernelPaths.IsSupported).Where(p => p != KernelPath.Scalar))
                {
                    Tensor[] other = RunCase(dir, ins, outs.Count, ArithmeticMode.Fast, path, 3);
                    for (int i = 0; i < other.Length; i++)
                    {
                        if (!Bytes(fast[i]).SequenceEqual(Bytes(other[i])))
                        {
                            failures.Add($"{name} output {i}: differs between the default path and {path}");
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
        Assert.True(failures.Count == 0,
            $"{failures.Count} of {cases} cases failed ({offByOne} of {elements} elements off by one step in allowed cases):{Environment.NewLine}{string.Join(Environment.NewLine, failures.Take(60))}");
        Console.WriteLine($"quant oracle: {cases} cases, {offByOne} of {elements} elements off by one step");
    }
}
