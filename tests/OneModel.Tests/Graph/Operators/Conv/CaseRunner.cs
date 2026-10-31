using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Graph.Operators.ConvFamily;

/// <summary>Runs one case directory (model.onnx, test_data_set_*/input_*.pb, output_*.pb) and compares with the recorded outputs.</summary>
internal static class CaseRunner
{
    /// <param name="dir">The case directory.</param>
    /// <param name="mode">Session mode.</param>
    /// <param name="scaled">True: max error scaled by the largest expected magnitude must be at most 1e-4 (generated cases). False: ONNX's backend-test tolerance, rtol 1e-3 and atol 1e-7 per element.</param>
    /// <param name="opset">The model's default opset.</param>
    /// <param name="hash">If given, receives the raw bytes of every output, to compare runs bit for bit.</param>
    /// <param name="threads">Intra-op threads.</param>
    public static string Run(string dir, ArithmeticMode mode, bool scaled, out long opset, System.Security.Cryptography.IncrementalHash? hash = null, int threads = 2)
    {
        opset = 0;
        try
        {
            using IrModel model = OnnxImporter.Load(Path.Combine(dir, "model.onnx"));
            opset = model.DefaultOpset;
            using var session = Session.Create(model, new ModelOptions { Mode = mode, IntraOpThreads = threads });
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
                        if (hash is not null)
                        {
                            Append(hash, outs[i]);
                        }

                        using Tensor expected = OnnxImporter.LoadTensor(File.ReadAllBytes(Path.Combine(setDir, $"output_{i}.pb"))).ToTensor();
                        if (!Same(outs[i], expected, scaled, out string why))
                        {
                            return $"FAIL output {i}: {why}";
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
        catch (SessionException e) when (e.Kind is SessionErrorKind.Unsupported or SessionErrorKind.UnsupportedOperator)
        {
            return "UNSUPPORTED " + e.Message;
        }
        catch (OnnxImportException e) when (e.Kind == OnnxErrorKind.Unsupported)
        {
            return "UNSUPPORTED " + e.Message;
        }
        catch (Exception e)
        {
            return "FAIL " + e.GetType().Name + ": " + e.Message;
        }
    }

    private static void Append(System.Security.Cryptography.IncrementalHash hash, Tensor t)
    {
        switch (t.DType)
        {
            case DType.F32: hash.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(t.Data<float>())); break;
            case DType.I64: hash.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(t.Data<long>())); break;
            case DType.F64: hash.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(t.Data<double>())); break;
            case DType.I8: hash.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(t.Data<sbyte>())); break;
            case DType.U8: hash.AppendData(t.Data<byte>()); break;
            default: throw new NotSupportedException(t.DType.ToString());
        }
    }

    private static bool Same(Tensor got, Tensor want, bool scaled, out string why)
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
                return Close(got.ToArray<float>().Select(v => (double)v).ToArray(), want.ToArray<float>().Select(v => (double)v).ToArray(), scaled, out why);
            case DType.F64:
                return Close(got.ToArray<double>(), want.ToArray<double>(), scaled, out why);
            case DType.I64:
                return Exactly(got.ToArray<long>(), want.ToArray<long>(), out why);
            case DType.I8:
                return Exactly(got.ToArray<sbyte>(), want.ToArray<sbyte>(), out why);
            case DType.U8:
                return Exactly(got.ToArray<byte>(), want.ToArray<byte>(), out why);
            default:
                why = $"comparison of {got.DType} is not implemented";
                return false;
        }
    }

    private static bool Exactly<T>(T[] got, T[] want, out string why) where T : IEquatable<T>
    {
        why = "";
        for (int i = 0; i < got.Length; i++)
        {
            if (!got[i].Equals(want[i]))
            {
                why = $"element {i}: {got[i]} vs {want[i]}";
                return false;
            }
        }

        return true;
    }

    private static bool Close(double[] got, double[] want, bool scaled, out string why)
    {
        why = "";
        double max = 0;
        foreach (double v in want)
        {
            if (double.IsFinite(v))
            {
                max = Math.Max(max, Math.Abs(v));
            }
        }

        for (int i = 0; i < got.Length; i++)
        {
            double a = got[i], b = want[i];
            if (double.IsNaN(a) || double.IsNaN(b))
            {
                if (double.IsNaN(a) != double.IsNaN(b))
                {
                    why = $"element {i}: {a} vs {b}";
                    return false;
                }

                continue;
            }

            if (double.IsInfinity(a) || double.IsInfinity(b))
            {
                if (a != b)
                {
                    why = $"element {i}: {a} vs {b}";
                    return false;
                }

                continue;
            }

            double err = Math.Abs(a - b);
            bool ok = scaled ? err <= 1e-4 * Math.Max(max, 1e-6) : err <= 1e-7 + (1e-3 * Math.Abs(b));
            if (!ok)
            {
                why = $"element {i}: {a} vs {b} (max |expected| {max})";
                return false;
            }
        }

        return true;
    }
}
