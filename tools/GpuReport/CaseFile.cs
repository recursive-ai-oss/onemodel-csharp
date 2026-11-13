using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using RecursiveAI.OneModel.Tensors;

namespace GpuReport;

/// <summary>
/// One case for a host that cannot compute its own reference: a 4-byte little-endian length, that many bytes of JSON listing the
/// entries, then the raw little-endian blobs (each 8-byte aligned, offsets relative to the first blob). Inputs are float32 or int64;
/// expected outputs are float32 (a float16 output is widened).
/// </summary>
internal static class CaseFile
{
    public static void Write(string file, string model, int batch, int seq, Dictionary<string, Tensor> inputs, Dictionary<string, float[]> outputs, Dictionary<string, int[]> outputDims)
    {
        using var blobs = new MemoryStream();
        var json = new MemoryStream();
        using (var w = new Utf8JsonWriter(json))
        {
            w.WriteStartObject();
            w.WriteString("model", model);
            w.WriteNumber("batch", batch);
            w.WriteNumber("seq", seq);
            w.WriteStartArray("entries");
            foreach ((string name, Tensor t) in inputs)
            {
                byte[] raw = t.DType switch
                {
                    DType.F32 => System.Runtime.InteropServices.MemoryMarshal.AsBytes(t.Data<float>()).ToArray(),
                    DType.I64 => System.Runtime.InteropServices.MemoryMarshal.AsBytes(t.Data<long>()).ToArray(),
                    _ => throw new NotSupportedException($"input {name} is {t.DType}"),
                };
                Entry(w, blobs, "input", name, t.DType == DType.F32 ? "f32" : "i64", [.. Dims(t)], raw);
            }

            foreach ((string name, float[] v) in outputs)
            {
                Entry(w, blobs, "output", name, "f32", outputDims[name], System.Runtime.InteropServices.MemoryMarshal.AsBytes<float>(v).ToArray());
            }

            w.WriteEndArray();
            w.WriteEndObject();
        }

        using var f = File.Create(file);
        f.Write(BitConverter.GetBytes((uint)json.Length));
        f.Write(json.ToArray());
        blobs.Position = 0;
        blobs.CopyTo(f);
    }

    private static IEnumerable<int> Dims(Tensor t)
    {
        for (int i = 0; i < t.Shape.Rank; i++)
        {
            yield return t.Shape[i];
        }
    }

    private static void Entry(Utf8JsonWriter w, MemoryStream blobs, string role, string name, string dtype, int[] dims, byte[] raw)
    {
        while (blobs.Position % 8 != 0)
        {
            blobs.WriteByte(0);
        }

        w.WriteStartObject();
        w.WriteString("role", role);
        w.WriteString("name", name);
        w.WriteString("dtype", dtype);
        w.WriteStartArray("dims");
        foreach (int d in dims)
        {
            w.WriteNumberValue(d);
        }

        w.WriteEndArray();
        w.WriteNumber("offset", blobs.Position);
        w.WriteNumber("bytes", raw.Length);
        w.WriteEndObject();
        blobs.Write(raw);
    }
}
