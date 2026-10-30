using System;
using System.Collections.Generic;
using System.IO;
using RecursiveAI.OneModel.Format;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Training;
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
            case "checkpoint":
                Fuzzer.LibFuzzer.Run(static input => FuzzCheckpoint(input));
                break;
            case "seed-checkpoint":
                // Writes a valid checkpoint of the fuzz target's model to args[1], as a seed for the corpus.
                using (Sequential m = CheckpointModel())
                using (var o = new Adam(m.Parameters()))
                {
                    Checkpoint.Save(args[1], m, o, new Dictionary<string, long> { ["step"] = 3 });
                }

                break;
            default:
                Console.Error.WriteLine("unknown target '" + target + "'; use onnx, tensor, format or checkpoint");
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

    private static Sequential CheckpointModel()
    {
        var g = new Generator(1);
        return new Sequential(new Linear(3, 2, g), new BatchNorm2d(2), new Dropout(0.1f, 5));
    }

    private static void FuzzCheckpoint(ReadOnlySpan<byte> input)
    {
        using Sequential model = CheckpointModel();
        using var opt = new Adam(model.Parameters());
        try
        {
            Checkpoint.Load(new System.IO.MemoryStream(input.ToArray()), model, opt);
            // Whatever loads must save, and what it saves must load into a fresh pair.
            using var ms = new System.IO.MemoryStream();
            Checkpoint.Save(ms, model, opt);
            using Sequential again = CheckpointModel();
            using var opt2 = new Adam(again.Parameters());
            Checkpoint.Load(new System.IO.MemoryStream(ms.ToArray()), again, opt2);
        }
        catch (CheckpointException)
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
