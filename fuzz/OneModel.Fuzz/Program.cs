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
                // Writes valid checkpoints of every fuzz configuration into the directory args[1], as corpus seeds.
                Directory.CreateDirectory(args[1]);
                for (int c = 0; c < CheckpointConfigs; c++)
                {
                    (NnModule m, Optimizer? o) = CheckpointConfig(c);
                    using (m)
                    using (o)
                    {
                        Checkpoint.Save(Path.Combine(args[1], $"seed-{c}.ckpt"), m, o, new Dictionary<string, long> { ["step"] = 3, ["epoch"] = 1 });
                        if (o is not null)
                        {
                            Checkpoint.Save(Path.Combine(args[1], $"seed-{c}-model-only.ckpt"), m);
                        }
                    }
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
        var options = new ModelLoadOptions { MemoryMap = false, ExternalDataDirectory = ExternalDirectory };
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

        // The stream route reads with a size limit: a non-seekable stream, once whole and once cut short by the limit.
        foreach (long limit in new long[] { int.MaxValue, Math.Max(1, input.Length / 2) })
        {
            try
            {
                using IrModel viaStream = OnnxImporter.Load(new OneWayStream(input.ToArray()), new ModelLoadOptions { MemoryMap = false, MaxModelBytes = limit });
            }
            catch (OnnxImportException)
            {
                // Expected for bad or oversize input.
            }
        }
    }

    private sealed class OneWayStream(byte[] data) : Stream
    {
        private int _at;
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override int Read(byte[] buffer, int offset, int count)
        {
            int n = Math.Min(count, Math.Min(data.Length - _at, 4099));
            Array.Copy(data, _at, buffer, offset, n);
            _at += n;
            return n;
        }

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    private const int CheckpointConfigs = 4;

    // The model and optimizer pairs a checkpoint input is tried against: batch norm, dropout and Adam; an embedding with SGD
    // momentum; a bare linear layer with no optimizer; an LSTM with AdamW and AMSGrad.
    private static (NnModule Model, Optimizer? Optimizer) CheckpointConfig(int index)
    {
        var g = new Generator(1);
        switch (index)
        {
            case 0:
                var a = new Sequential(new Linear(3, 2, g), new BatchNorm2d(2), new Dropout(0.1f, 5));
                return (a, new Adam(a.Parameters()));
            case 1:
                var b = new Sequential(new Embedding(4, 3, g), new Linear(3, 2, g));
                return (b, new Sgd(b.Parameters(), 0.1f, momentum: 0.9f));
            case 2:
                return (new Linear(2, 2, g), null);
            default:
                var d = new Recurrent(gru: false, 2, 2, g);
                return (d, new AdamW(d.Parameters(), amsgrad: true));
        }
    }

    // The reader rejects a file whose header lengths disagree with its size, which almost every mutation breaks. So each input is
    // also tried with the header's length fields repaired, which lets mutations of the manifest and the data reach the code behind.
    private static byte[] RepairHeader(byte[] input)
    {
        if (input.Length < 64 || !input.AsSpan(0, 8).SequenceEqual("ONEMCKPT"u8))
        {
            return input;
        }

        byte[] f = (byte[])input.Clone();
        ulong manifestLength = System.Buffers.Binary.BinaryPrimitives.ReadUInt64LittleEndian(f.AsSpan(24));
        if (manifestLength > (ulong)(f.Length - 64))
        {
            manifestLength = (ulong)(f.Length - 64);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(f.AsSpan(24), manifestLength);
        }

        ulong dataOffset = (64 + manifestLength + 63) / 64 * 64;
        if (dataOffset > (ulong)f.Length)
        {
            return f;
        }

        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(f.AsSpan(8), 1);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(f.AsSpan(12), 0);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(f.AsSpan(16), 64);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(f.AsSpan(32), dataOffset);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(f.AsSpan(40), (ulong)f.Length - dataOffset);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(f.AsSpan(48), (ulong)f.Length);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(f.AsSpan(56), 0);
        return f;
    }

    private static void FuzzCheckpoint(ReadOnlySpan<byte> input)
    {
        byte[] raw = input.ToArray();
        byte[] repaired = RepairHeader(raw);
        for (int c = 0; c < CheckpointConfigs; c++)
        {
            TryCheckpoint(raw, c);
            if (!ReferenceEquals(repaired, raw))
            {
                TryCheckpoint(repaired, c);
            }
        }
    }

    private static void TryCheckpoint(byte[] bytes, int config)
    {
        (NnModule model, Optimizer? opt) = CheckpointConfig(config);
        using (model)
        using (opt)
        {
            try
            {
                Checkpoint.Load(new MemoryStream(bytes), model, opt);
                // Whatever loads must save, and what it saves must load into a fresh pair.
                using var ms = new MemoryStream();
                Checkpoint.Save(ms, model, opt);
                (NnModule again, Optimizer? opt2) = CheckpointConfig(config);
                using (again)
                using (opt2)
                {
                    Checkpoint.Load(new MemoryStream(ms.ToArray()), again, opt2);
                }
            }
            catch (CheckpointException)
            {
                // Rejecting bad input is the expected outcome.
            }
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
