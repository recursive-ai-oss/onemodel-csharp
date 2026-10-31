using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Training;

/// <summary>What went wrong reading a checkpoint.</summary>
public enum CheckpointErrorKind
{
    /// <summary>The header is missing, malformed, or inconsistent with the file length.</summary>
    BadHeader,
    /// <summary>The manifest is not the expected JSON, or a field is missing or out of range.</summary>
    BadManifest,
    /// <summary>A tensor entry points outside the data region, or its size disagrees with its shape.</summary>
    BadTensor,
    /// <summary>The checkpoint is well-formed but does not fit the model or optimizer it is loaded into.</summary>
    Mismatch,
}

/// <summary>A checkpoint could not be read or applied. <see cref="Kind"/> says which check failed; the message names the field.</summary>
public sealed class CheckpointException : OneModelException
{
    /// <summary>Creates the exception.</summary>
    /// <param name="kind">Which check failed.</param>
    /// <param name="message">What was wrong and where.</param>
    public CheckpointException(CheckpointErrorKind kind, string message) : base(OneModelErrorCategory.Checkpoint, message) => Kind = kind;

    /// <summary>Which check failed.</summary>
    public CheckpointErrorKind Kind { get; }
}

/// <summary>Values read back from a checkpoint besides the model and optimizer state.</summary>
/// <param name="Extra">The integer values saved with the checkpoint (for example the epoch), by key.</param>
/// <param name="HadOptimizer">Whether the file holds optimizer state.</param>
public sealed record CheckpointInfo(IReadOnlyDictionary<string, long> Extra, bool HadOptimizer);

/// <summary>
/// Saves and loads a model's state (parameters and buffers) and an optimizer's state. The file is a header, a JSON manifest and
/// 64-byte-aligned little-endian tensors (layout in <c>docs/training.md</c>), the same shape as the model format with its own magic.
/// Saving then loading restores every bit; the same state always writes the same bytes.
/// </summary>
public static class Checkpoint
{
    private const string Magic = "ONEMCKPT";
    private const int HeaderSize = 64;
    private const long MaxManifest = 256L << 20;

    /// <summary>Writes a checkpoint file (to a temporary name first, then renamed over <paramref name="path"/>).</summary>
    /// <param name="path">Destination file.</param>
    /// <param name="model">The module whose <see cref="NnModule.StateDict"/> is saved.</param>
    /// <param name="optimizer">The optimizer whose hyper-parameters, step count and state tensors are saved, or null for none.</param>
    /// <param name="extra">Integer values to keep alongside (epoch, sample counter, ...), written in key order; or null.</param>
    public static void Save(string path, NnModule model, Optimizer? optimizer = null, IReadOnlyDictionary<string, long>? extra = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        string temp = path + ".tmp";
        try
        {
            using (FileStream f = File.Create(temp))
            {
                Save(f, model, optimizer, extra);
            }

            File.Move(temp, path, overwrite: true);
        }
        catch
        {
            File.Delete(temp);
            throw;
        }
    }

    /// <summary>Writes a checkpoint to a stream.</summary>
    /// <param name="stream">A writable stream; it is left open.</param>
    /// <param name="model">The module whose <see cref="NnModule.StateDict"/> is saved.</param>
    /// <param name="optimizer">The optimizer to save, or null.</param>
    /// <param name="extra">Integer values to keep alongside, or null.</param>
    public static void Save(Stream stream, NnModule model, Optimizer? optimizer = null, IReadOnlyDictionary<string, long>? extra = null)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(model);
        List<(string Name, Tensor Tensor)> modelState = model.StateDict();
        IReadOnlyList<(string Name, Tensor Tensor)> optState = optimizer?.StateTensors ?? [];

        // Lay out the data region: model tensors, then optimizer tensors, each at a multiple of 64.
        var offsets = new long[modelState.Count + optState.Count];
        long cursor = 0;
        for (int i = 0; i < offsets.Length; i++)
        {
            Tensor t = i < modelState.Count ? modelState[i].Tensor : optState[i - modelState.Count].Tensor;
            offsets[i] = cursor;
            cursor = Align64(cursor + ByteLength(t));
        }

        byte[] manifest = WriteManifest(modelState, optState, offsets, optimizer, extra);
        long dataOffset = Align64(HeaderSize + manifest.Length);
        long fileLength = dataOffset + cursor;

        Span<byte> header = stackalloc byte[HeaderSize];
        header.Clear();
        Encoding.ASCII.GetBytes(Magic, header);
        BinaryPrimitives.WriteUInt32LittleEndian(header[8..], 1);
        BinaryPrimitives.WriteUInt64LittleEndian(header[16..], HeaderSize);
        BinaryPrimitives.WriteUInt64LittleEndian(header[24..], (ulong)manifest.Length);
        BinaryPrimitives.WriteUInt64LittleEndian(header[32..], (ulong)dataOffset);
        BinaryPrimitives.WriteUInt64LittleEndian(header[40..], (ulong)cursor);
        BinaryPrimitives.WriteUInt64LittleEndian(header[48..], (ulong)fileLength);
        stream.Write(header);
        stream.Write(manifest);
        WriteZeros(stream, dataOffset - HeaderSize - manifest.Length);

        long written = 0;
        for (int i = 0; i < offsets.Length; i++)
        {
            Tensor t = i < modelState.Count ? modelState[i].Tensor : optState[i - modelState.Count].Tensor;
            WriteZeros(stream, offsets[i] - written);
            written = offsets[i];
            byte[] bytes = new byte[ByteLength(t)];
            Encode(t, bytes);
            stream.Write(bytes);
            written += bytes.Length;
        }

        WriteZeros(stream, cursor - written);
    }

    /// <summary>Reads a checkpoint file into a module and optionally an optimizer.</summary>
    /// <param name="path">The file.</param>
    /// <param name="model">The module to fill; its names, types and shapes must equal the saved ones.</param>
    /// <param name="optimizer">The optimizer to fill (same kind, parameter shapes and fixed settings as when saved), or null to skip optimizer state.</param>
    /// <returns>The extra values and whether optimizer state was present.</returns>
    /// <exception cref="CheckpointException">The file is malformed or does not fit the model or optimizer. Nothing is modified in that case.</exception>
    public static CheckpointInfo Load(string path, NnModule model, Optimizer? optimizer = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        using FileStream f = File.OpenRead(path);
        return Load(f, model, optimizer);
    }

    /// <summary>Reads a checkpoint from a stream (read to its end) into a module and optionally an optimizer.</summary>
    /// <param name="stream">A readable stream positioned at the start of the checkpoint; it is left open.</param>
    /// <param name="model">The module to fill.</param>
    /// <param name="optimizer">The optimizer to fill, or null.</param>
    /// <returns>The extra values and whether optimizer state was present.</returns>
    /// <exception cref="CheckpointException">The data is malformed or does not fit. Nothing is modified in that case.</exception>
    public static CheckpointInfo Load(Stream stream, NnModule model, Optimizer? optimizer = null)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(model);
        using var ms = new MemoryStream();
        stream.CopyTo(ms);
        byte[] file = ms.ToArray();
        return Read(file, model, optimizer);
    }

    // ---------------------------------------------------------------------------------------------------------------- reading

    private static CheckpointInfo Read(byte[] file, NnModule model, Optimizer? optimizer)
    {
        if (file.Length < HeaderSize)
        {
            throw new CheckpointException(CheckpointErrorKind.BadHeader, $"The file has {file.Length} bytes, less than the {HeaderSize}-byte header.");
        }

        if (Encoding.ASCII.GetString(file, 0, 8) != Magic)
        {
            throw new CheckpointException(CheckpointErrorKind.BadHeader, "The magic is not ONEMCKPT.");
        }

        ReadOnlySpan<byte> h = file;
        uint version = BinaryPrimitives.ReadUInt32LittleEndian(h[8..]);
        uint flags = BinaryPrimitives.ReadUInt32LittleEndian(h[12..]);
        ulong manifestOffset = BinaryPrimitives.ReadUInt64LittleEndian(h[16..]);
        ulong manifestLength = BinaryPrimitives.ReadUInt64LittleEndian(h[24..]);
        ulong dataOffset = BinaryPrimitives.ReadUInt64LittleEndian(h[32..]);
        ulong dataLength = BinaryPrimitives.ReadUInt64LittleEndian(h[40..]);
        ulong fileLength = BinaryPrimitives.ReadUInt64LittleEndian(h[48..]);
        ulong reserved = BinaryPrimitives.ReadUInt64LittleEndian(h[56..]);
        if (version != 1 || flags != 0 || reserved != 0)
        {
            throw new CheckpointException(CheckpointErrorKind.BadHeader, $"Unsupported version {version}, flags {flags} or reserved field.");
        }

        if (fileLength != (ulong)file.Length)
        {
            throw new CheckpointException(CheckpointErrorKind.BadHeader, $"The header says {fileLength} bytes, the file has {file.Length} (truncated or extended).");
        }

        if (manifestOffset != HeaderSize || manifestLength > MaxManifest || manifestLength > fileLength - HeaderSize ||
            dataOffset % 64 != 0 || dataOffset < HeaderSize + manifestLength || dataOffset > fileLength || dataLength != fileLength - dataOffset)
        {
            throw new CheckpointException(CheckpointErrorKind.BadHeader, "Manifest or data region offsets are inconsistent with the file length.");
        }

        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(file.AsMemory(HeaderSize, (int)manifestLength), new JsonDocumentOptions { MaxDepth = 32 });
        }
        catch (JsonException e)
        {
            throw new CheckpointException(CheckpointErrorKind.BadManifest, "The manifest is not valid JSON: " + e.Message);
        }

        using (doc)
        {
            JsonElement root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object || Str(root, "format") != "onemodel-checkpoint" || Int(root, "version") != 1)
            {
                throw new CheckpointException(CheckpointErrorKind.BadManifest, "The manifest is not a version 1 onemodel-checkpoint.");
            }

            long dataStart = (long)dataOffset, dataLen = (long)dataLength;
            List<(string Name, Tensor Tensor)> modelState = model.StateDict();
            List<Entry> modelEntries = Entries(Arr(root, "model"));
            CheckFit("model", modelEntries, modelState, dataLen);

            bool hasOptimizer = root.TryGetProperty("optimizer", out JsonElement opt) && opt.ValueKind == JsonValueKind.Object;
            List<Entry>? optEntries = null;
            Dictionary<string, float>? hyper = null;
            long step = 0;
            if (hasOptimizer && optimizer is not null)
            {
                if (Str(opt, "kind") != optimizer.Kind)
                {
                    throw new CheckpointException(CheckpointErrorKind.Mismatch, $"The checkpoint holds a '{Str(opt, "kind")}' optimizer, not '{optimizer.Kind}'.");
                }

                step = Int(opt, "step");
                if (step < 0)
                {
                    throw new CheckpointException(CheckpointErrorKind.BadManifest, "The optimizer step count is negative.");
                }

                hyper = new Dictionary<string, float>(StringComparer.Ordinal);
                foreach (JsonElement hp in Arr(opt, "hyper").EnumerateArray())
                {
                    long bits = Int(hp, "bits");
                    if (bits < 0 || bits > uint.MaxValue)
                    {
                        throw new CheckpointException(CheckpointErrorKind.BadManifest, "A hyper-parameter bit pattern is out of range.");
                    }

                    hyper[Str(hp, "name")] = BitConverter.Int32BitsToSingle(unchecked((int)(uint)bits));
                }

                foreach ((string name, _) in optimizer.HyperParameters())
                {
                    if (!hyper.ContainsKey(name))
                    {
                        throw new CheckpointException(CheckpointErrorKind.Mismatch, $"The checkpoint lacks hyper-parameter '{name}'.");
                    }
                }

                optEntries = Entries(Arr(opt, "state"));
                var state = new List<(string, Tensor)>();
                foreach ((string n, Tensor t) in optimizer.StateTensors)
                {
                    state.Add((n, t));
                }

                CheckFit("optimizer state", optEntries, state, dataLen);
            }
            else if (optimizer is not null)
            {
                throw new CheckpointException(CheckpointErrorKind.Mismatch, "The checkpoint holds no optimizer state.");
            }

            var extra = new Dictionary<string, long>(StringComparer.Ordinal);
            if (root.TryGetProperty("extra", out JsonElement ex) && ex.ValueKind == JsonValueKind.Array)
            {
                foreach (JsonElement e in ex.EnumerateArray())
                {
                    extra[Str(e, "key")] = Int(e, "value");
                }
            }

            // Everything checked except the optimizer's own settings, which it validates before it changes anything.
            if (hyper is not null && optimizer is not null)
            {
                try
                {
                    optimizer.SetHyperParameters(hyper);
                }
                catch (ArgumentException e)
                {
                    throw new CheckpointException(CheckpointErrorKind.Mismatch, e.Message);
                }
            }

            Fill(file, dataStart, modelEntries, modelState);
            if (optEntries is not null && optimizer is not null && hyper is not null)
            {
                var state = new List<(string, Tensor)>();
                foreach ((string n, Tensor t) in optimizer.StateTensors)
                {
                    state.Add((n, t));
                }

                Fill(file, dataStart, optEntries, state);
                optimizer.RestoreStepCount(step);
            }

            return new CheckpointInfo(extra, hasOptimizer);
        }
    }

    private readonly record struct Entry(string Name, DType DType, int[] Dims, long Offset, long Length);

    private static List<Entry> Entries(JsonElement array)
    {
        var list = new List<Entry>();
        foreach (JsonElement e in array.EnumerateArray())
        {
            string name = Str(e, "name");
            DType dt = Str(e, "dtype") switch
            {
                "f32" => DType.F32,
                "i32" => DType.I32,
                "i64" => DType.I64,
                var other => throw new CheckpointException(CheckpointErrorKind.BadManifest, $"Tensor '{name}' has unsupported dtype '{other}'."),
            };
            var dims = new List<int>();
            foreach (JsonElement d in Arr(e, "shape").EnumerateArray())
            {
                if (!d.TryGetInt32(out int v) || v < 0)
                {
                    throw new CheckpointException(CheckpointErrorKind.BadManifest, $"Tensor '{name}' has a bad dimension.");
                }

                dims.Add(v);
            }

            list.Add(new Entry(name, dt, dims.ToArray(), Int(e, "offset"), Int(e, "length")));
        }

        return list;
    }

    // Names, types and shapes must equal the destination's, in order; offsets and lengths must lie inside the data region.
    private static void CheckFit(string what, List<Entry> entries, IReadOnlyList<(string Name, Tensor Tensor)> dest, long dataLength)
    {
        if (entries.Count != dest.Count)
        {
            throw new CheckpointException(CheckpointErrorKind.Mismatch, $"The checkpoint has {entries.Count} {what} tensors, the target has {dest.Count}.");
        }

        for (int i = 0; i < entries.Count; i++)
        {
            Entry e = entries[i];
            (string name, Tensor t) = dest[i];
            if (e.Name != name || e.DType != t.DType || !e.Dims.AsSpan().SequenceEqual(t.Shape.Dimensions))
            {
                throw new CheckpointException(CheckpointErrorKind.Mismatch, $"{what} entry {i} is '{e.Name}' {e.DType}[{string.Join(',', e.Dims)}], the target expects '{name}' {t.DType}{t.Shape}.");
            }

            long expected = ByteLength(t);
            if (e.Length != expected || e.Offset < 0 || e.Offset > dataLength || e.Length > dataLength - e.Offset)
            {
                throw new CheckpointException(CheckpointErrorKind.BadTensor, $"{what} tensor '{name}' has offset {e.Offset}, length {e.Length} (expected {expected}) in a data region of {dataLength} bytes.");
            }
        }
    }

    private static void Fill(byte[] file, long dataStart, List<Entry> entries, IReadOnlyList<(string Name, Tensor Tensor)> dest)
    {
        for (int i = 0; i < entries.Count; i++)
        {
            Decode(file.AsSpan((int)(dataStart + entries[i].Offset), (int)entries[i].Length), dest[i].Tensor);
        }
    }

    private static string Str(JsonElement o, string name) =>
        o.ValueKind == JsonValueKind.Object && o.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.String
            ? GetText(v, name)
            : throw new CheckpointException(CheckpointErrorKind.BadManifest, $"Manifest field '{name}' is missing or not a string.");

    private static string GetText(JsonElement v, string name)
    {
        try
        {
            return v.GetString()!;
        }
        catch (InvalidOperationException)
        {
            throw new CheckpointException(CheckpointErrorKind.BadManifest, $"Manifest field '{name}' is not valid UTF-8 text.");
        }
    }

    private static long Int(JsonElement o, string name) =>
        o.ValueKind == JsonValueKind.Object && o.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number && v.TryGetInt64(out long x)
            ? x
            : throw new CheckpointException(CheckpointErrorKind.BadManifest, $"Manifest field '{name}' is missing or not an integer.");

    private static JsonElement Arr(JsonElement o, string name) =>
        o.ValueKind == JsonValueKind.Object && o.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Array
            ? v
            : throw new CheckpointException(CheckpointErrorKind.BadManifest, $"Manifest field '{name}' is missing or not an array.");

    // ---------------------------------------------------------------------------------------------------------------- writing

    private static byte[] WriteManifest(List<(string Name, Tensor Tensor)> model, IReadOnlyList<(string Name, Tensor Tensor)> optState, long[] offsets,
        Optimizer? optimizer, IReadOnlyDictionary<string, long>? extra)
    {
        using var ms = new MemoryStream();
        using (var w = new Utf8JsonWriter(ms))
        {
            w.WriteStartObject();
            w.WriteString("format", "onemodel-checkpoint");
            w.WriteNumber("version", 1);
            w.WritePropertyName("model");
            WriteEntries(w, model, offsets, 0);
            if (optimizer is not null)
            {
                w.WritePropertyName("optimizer");
                w.WriteStartObject();
                w.WriteString("kind", optimizer.Kind);
                w.WriteNumber("step", optimizer.StepCount);
                w.WritePropertyName("hyper");
                w.WriteStartArray();
                foreach ((string name, float value) in optimizer.HyperParameters())
                {
                    w.WriteStartObject();
                    w.WriteString("name", name);
                    w.WriteNumber("bits", (uint)BitConverter.SingleToInt32Bits(value));
                    w.WriteEndObject();
                }

                w.WriteEndArray();
                w.WritePropertyName("state");
                WriteEntries(w, optState, offsets, model.Count);
                w.WriteEndObject();
            }

            w.WritePropertyName("extra");
            w.WriteStartArray();
            if (extra is not null)
            {
                var keys = new List<string>(extra.Keys);
                keys.Sort(StringComparer.Ordinal);
                foreach (string k in keys)
                {
                    w.WriteStartObject();
                    w.WriteString("key", k);
                    w.WriteNumber("value", extra[k]);
                    w.WriteEndObject();
                }
            }

            w.WriteEndArray();
            w.WriteEndObject();
        }

        return ms.ToArray();
    }

    private static void WriteEntries(Utf8JsonWriter w, IReadOnlyList<(string Name, Tensor Tensor)> tensors, long[] offsets, int offsetBase)
    {
        w.WriteStartArray();
        for (int i = 0; i < tensors.Count; i++)
        {
            Tensor t = tensors[i].Tensor;
            w.WriteStartObject();
            w.WriteString("name", tensors[i].Name);
            w.WriteString("dtype", t.DType switch
            {
                DType.F32 => "f32",
                DType.I32 => "i32",
                DType.I64 => "i64",
                _ => throw new NotSupportedException($"Checkpoints store float32, Int32 and Int64 tensors, not {t.DType}."),
            });
            w.WritePropertyName("shape");
            w.WriteStartArray();
            foreach (int d in t.Shape.Dimensions)
            {
                w.WriteNumberValue(d);
            }

            w.WriteEndArray();
            w.WriteNumber("offset", offsets[offsetBase + i]);
            w.WriteNumber("length", ByteLength(t));
            w.WriteEndObject();
        }

        w.WriteEndArray();
    }

    private static long Align64(long v) => (v + 63) & ~63L;

    private static long ByteLength(in Tensor t) => (long)t.Length * t.ElementSize;

    private static void WriteZeros(Stream s, long count)
    {
        Span<byte> zeros = stackalloc byte[64];
        zeros.Clear();
        while (count > 0)
        {
            int n = (int)Math.Min(count, 64);
            s.Write(zeros[..n]);
            count -= n;
        }
    }

    private static void Encode(in Tensor t, Span<byte> dst)
    {
        switch (t.DType)
        {
            case DType.F32:
                ReadOnlySpan<float> f = t.Data<float>();
                for (int i = 0; i < f.Length; i++)
                {
                    BinaryPrimitives.WriteInt32LittleEndian(dst[(4 * i)..], BitConverter.SingleToInt32Bits(f[i]));
                }

                break;
            case DType.I32:
                ReadOnlySpan<int> n = t.Data<int>();
                for (int i = 0; i < n.Length; i++)
                {
                    BinaryPrimitives.WriteInt32LittleEndian(dst[(4 * i)..], n[i]);
                }

                break;
            case DType.I64:
                ReadOnlySpan<long> l = t.Data<long>();
                for (int i = 0; i < l.Length; i++)
                {
                    BinaryPrimitives.WriteInt64LittleEndian(dst[(8 * i)..], l[i]);
                }

                break;
            default:
                throw new NotSupportedException($"Checkpoints store float32, Int32 and Int64 tensors, not {t.DType}.");
        }
    }

    private static void Decode(ReadOnlySpan<byte> src, in Tensor t)
    {
        switch (t.DType)
        {
            case DType.F32:
                Span<float> f = t.MutableData<float>();
                for (int i = 0; i < f.Length; i++)
                {
                    f[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(src[(4 * i)..]));
                }

                break;
            case DType.I32:
                Span<int> n = t.MutableData<int>();
                for (int i = 0; i < n.Length; i++)
                {
                    n[i] = BinaryPrimitives.ReadInt32LittleEndian(src[(4 * i)..]);
                }

                break;
            case DType.I64:
                Span<long> l = t.MutableData<long>();
                for (int i = 0; i < l.Length; i++)
                {
                    l[i] = BinaryPrimitives.ReadInt64LittleEndian(src[(8 * i)..]);
                }

                break;
        }
    }
}
