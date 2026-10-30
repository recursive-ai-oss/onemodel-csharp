using System;
using System.IO;
using RecursiveAI.OneModel.Format;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;
using SharpFuzz;

namespace RecursiveAI.OneModel.Fuzz;

/// <summary>
/// Entry point of the fuzz harness. SharpFuzz instruments the library and libFuzzer drives
/// <see cref="Fuzzer.LibFuzzer.Run(ReadOnlySpanAction)"/>. The first argument picks the target:
/// <c>onnx</c> (a whole model file; the default), <c>tensor</c> (one serialized TensorProto) or <c>format</c> (a file in the
/// OneModel format; a file that loads is saved and loaded again).
/// A reader may reject input only with <see cref="OnnxImportException"/> (<see cref="ModelFormatException"/> for the format);
/// any other exception is a finding.
/// If <c>ONEMODEL_FUZZ_EXTDIR</c> names a directory, models may refer to external data inside it.
/// </summary>
internal static class Program
{
    private static readonly string? ExternalDirectory = Environment.GetEnvironmentVariable("ONEMODEL_FUZZ_EXTDIR");

    private static void Main(string[] args)
    {
        string target = args.Length > 0 ? args[0] : "onnx";
        switch (target)
        {
            case "onnx":
                Fuzzer.LibFuzzer.Run(static input => FuzzModel(input));
                break;
            case "tensor":
                Fuzzer.LibFuzzer.Run(static input => FuzzTensor(input));
                break;
            case "format":
                Fuzzer.LibFuzzer.Run(static input => FuzzFormat(input));
                break;
            default:
                Console.Error.WriteLine("unknown target '" + target + "'; use onnx, tensor or format");
                Environment.ExitCode = 2;
                break;
        }
    }

    private static void FuzzModel(ReadOnlySpan<byte> input)
    {
        var options = new OnnxLoadOptions { MemoryMap = false, ExternalDataDirectory = ExternalDirectory };
        try
        {
            using IrModel model = OnnxImporter.Load(input.ToArray(), options);
            // Touch everything an importer promises: every initializer's bytes must be readable.
            foreach (IrValue v in model.Graph.Values)
            {
                if (v.Initializer is { } t)
                {
                    _ = t.RawData.Length;
                }
            }
        }
        catch (OnnxImportException)
        {
            // Rejecting bad input is the expected outcome.
        }
    }

    private static void FuzzFormat(ReadOnlySpan<byte> input)
    {
        try
        {
            using IrModel model = ModelFormat.Load(input.ToArray());
            foreach (IrValue v in model.Graph.Values)
            {
                if (v.Initializer is { } t)
                {
                    _ = t.RawData.Length;
                }
            }

            // Whatever loads must save, and what it saves must load.
            using var ms = new System.IO.MemoryStream();
            ModelFormat.Save(model, ms);
            using IrModel again = ModelFormat.Load(ms.ToArray());
        }
        catch (ModelFormatException)
        {
            // Rejecting bad input is the expected outcome.
        }
    }

    private static void FuzzTensor(ReadOnlySpan<byte> input)
    {
        try
        {
            TensorData t = OnnxImporter.LoadTensor(input.ToArray());
            if (t.TryGetDType(out _) && t.ElementCount < (1 << 20))
            {
                try
                {
                    using Tensor x = t.ToTensor();
                }
                catch (TensorShapeException)
                {
                    // More than 8 dimensions is a limit of the tensor layer, not a bug.
                }
            }
        }
        catch (OnnxImportException)
        {
        }
    }
}
