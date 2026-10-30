using System.Buffers.Binary;
using System.Text;
using System.Text.Json;

namespace CrnnFinetune;

/// <summary>
/// Writes a recognizer in the line-recognizer model container (versions 0 and 1) from the container specification: a magic, a
/// version, a JSON manifest, then 64-byte-aligned float16 tensors. Batch normalization is folded into the convolutions in double
/// precision (scale = gamma / sqrt(var + eps)) and weights are rounded to binary16 to nearest even.
/// </summary>
internal static class Container
{
    private const double BnEps = 1e-5;

    public static void Export(string weightsDir, string templatePath, string outPath, int formatVersion, string provenance)
    {
        var (config, _, tensors) = WeightsFile.Read(weightsDir);
        var byName = tensors.ToDictionary(t => t.Name, t => t.Data);
        var table = new List<(string Name, int[] Shape, float[] Data)>();
        var shapes = tensors.ToDictionary(t => t.Name, t => t.Shape);
        for (int i = 0; i < config.Out.Length; i++)
        {
            float[] w = byName[$"conv.{i}.0.weight"], gamma = byName[$"conv.{i}.1.weight"], beta = byName[$"conv.{i}.1.bias"];
            float[] mean = byName[$"conv.{i}.1.running_mean"], var = byName[$"conv.{i}.1.running_var"];
            int[] ws = shapes[$"conv.{i}.0.weight"];
            int per = w.Length / ws[0];
            var fw = new float[w.Length];
            var fb = new float[ws[0]];
            for (int o = 0; o < ws[0]; o++)
            {
                double scale = (double)gamma[o] / Math.Sqrt((double)var[o] + BnEps);   // IEEE sqrt: the same on every machine
                for (int k = 0; k < per; k++) fw[(o * per) + k] = (float)(w[(o * per) + k] * scale);
                fb[o] = (float)(beta[o] - (mean[o] * scale));
            }

            table.Add(($"conv.{i}.weight", ws, fw));
            table.Add(($"conv.{i}.bias", [ws[0]], fb));
        }

        foreach (var t in tensors.Where(t => t.Name.StartsWith("lstm.", StringComparison.Ordinal) || t.Name.StartsWith("fc.", StringComparison.Ordinal)))
            table.Add((t.Name, t.Shape, t.Data));

        byte[] tpl = File.ReadAllBytes(templatePath);
        if (!tpl.AsSpan(0, 8).SequenceEqual("ONEOCRM\0"u8)) throw new InvalidDataException("template is not a model container");
        int mlen = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(tpl.AsSpan(12, 4)));
        using var tdoc = JsonDocument.Parse(tpl.AsMemory(16, mlen));

        // tensor payload
        var offsets = new List<(long Offset, long Length)>();
        long pos = 0;
        foreach (var t in table)
        {
            long len = t.Data.Length * 2L;
            offsets.Add((pos, len));
            pos = (pos + len + 63) / 64 * 64;
        }

        var manifest = new MemoryStream();
        using (var w = new Utf8JsonWriter(manifest))
        {
            w.WriteStartObject();
            foreach (JsonProperty p in tdoc.RootElement.EnumerateObject())
            {
                if (p.Name is "tensors" or "provenance") continue;
                if (p.Name == "calibration" && formatVersion == 0) continue;
                w.WritePropertyName(p.Name);
                p.Value.WriteTo(w);
            }

            w.WriteStartArray("tensors");
            for (int i = 0; i < table.Count; i++)
            {
                w.WriteStartObject();
                w.WriteString("name", table[i].Name);
                w.WriteString("dtype", "f16");
                w.WriteStartArray("shape");
                foreach (int d in table[i].Shape) w.WriteNumberValue(d);
                w.WriteEndArray();
                w.WriteNumber("offset", offsets[i].Offset);
                w.WriteNumber("length", offsets[i].Length);
                w.WriteEndObject();
            }

            w.WriteEndArray();
            w.WriteStartObject("provenance");
            w.WriteString("note", provenance);
            w.WriteEndObject();
            w.WriteEndObject();
        }

        // The version field is rewritten: the template may be version 0 or 1.
        using var fs = File.Create(outPath);
        Span<byte> head = stackalloc byte[16];
        "ONEOCRM\0"u8.CopyTo(head);
        BinaryPrimitives.WriteUInt32LittleEndian(head[8..], (uint)formatVersion);
        BinaryPrimitives.WriteUInt32LittleEndian(head[12..], (uint)manifest.Length);
        fs.Write(head);
        fs.Write(manifest.GetBuffer(), 0, (int)manifest.Length);
        long dataStart = (16 + manifest.Length + 63) / 64 * 64;
        fs.Write(new byte[dataStart - 16 - manifest.Length]);
        var data = new byte[offsets[^1].Offset + offsets[^1].Length];   // zero padding between tensors
        for (int i = 0; i < table.Count; i++)
            for (int k = 0; k < table[i].Data.Length; k++)
                BinaryPrimitives.WriteHalfLittleEndian(data.AsSpan((int)offsets[i].Offset + (k * 2), 2), (Half)table[i].Data[k]);   // nearest even
        fs.Write(data);
    }
}
