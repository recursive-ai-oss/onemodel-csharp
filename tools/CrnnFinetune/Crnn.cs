using System.Text.Json;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Training;

namespace CrnnFinetune;

/// <summary>Layer sizes of the recognizer, read from the weights manifest.</summary>
internal sealed record CrnnConfig(int[] Out, int[] Kernel, int[] Padding, int[][] Pool, int Hidden, int LstmLayers, int Classes)
{
    private static readonly int[] NoPool = [1, 1];

    public static CrnnConfig From(JsonElement c)
    {
        var conv = c.GetProperty("conv").EnumerateArray().ToArray();
        int[] Pair(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();
        return new CrnnConfig(
            conv.Select(e => e.GetProperty("out").GetInt32()).ToArray(),
            conv.Select(e => Pair(e.GetProperty("kernel"))[0]).ToArray(),
            conv.Select(e => Pair(e.GetProperty("padding"))[0]).ToArray(),
            conv.Select(e => e.GetProperty("pool").ValueKind == JsonValueKind.Null ? NoPool : Pair(e.GetProperty("pool"))).ToArray(),
            c.GetProperty("lstm").GetProperty("hidden").GetInt32(),
            c.GetProperty("lstm").GetProperty("layers").GetInt32(),
            c.GetProperty("classes").GetInt32());
    }

    /// <summary>Sequence length for an input width: each pool drops a trailing partial window.</summary>
    public int OutLength(int width)
    {
        foreach (int[] p in Pool) width /= p[1];
        return width;
    }
}

/// <summary>conv (no bias) -> batch norm -> ReLU -> optional max pool; children "0" and "1" as in the reference's state dict.</summary>
internal sealed class ConvBlock : NnModule
{
    private readonly Conv2d _conv;
    private readonly BatchNorm2d _bn;
    private readonly Pool2dParams? _pool;

    public ConvBlock(int cin, int cout, int kernel, int padding, int[] pool, Generator g)
    {
        _conv = RegisterModule("0", new Conv2d(cin, cout, kernel, g, padding: padding, bias: false));
        _bn = RegisterModule("1", new BatchNorm2d(cout));
        _pool = pool[0] * pool[1] > 1 ? new Pool2dParams(pool[0], pool[1], pool[0], pool[1], 0, 0) : null;
    }

    public override Tensor Forward(Tensor x)
    {
        x = Ops.Relu(_bn.Forward(_conv.Forward(x)));
        return _pool is { } p ? Ops.MaxPool2d(x, p) : x;
    }
}

/// <summary>Convolutional stack -> BiLSTM -> linear, CTC-ready logits [B, T, classes] (Shi, Bai and Yao 2015).</summary>
internal sealed class Crnn : NnModule
{
    public CrnnConfig Config { get; }
    private readonly Sequential _conv;
    private readonly Recurrent _lstm;
    private readonly Linear _fc;

    public Crnn(CrnnConfig c)
    {
        Config = c;
        var g = new Generator(1);
        var blocks = new NnModule[c.Out.Length];
        int cin = 1;
        for (int i = 0; i < blocks.Length; i++)
        {
            blocks[i] = new ConvBlock(cin, c.Out[i], c.Kernel[i], c.Padding[i], c.Pool[i], g);
            cin = c.Out[i];
        }

        _conv = RegisterModule("conv", new Sequential(blocks));
        _lstm = RegisterModule("lstm", new Recurrent(false, cin, c.Hidden, g, c.LstmLayers, bidirectional: true, batchFirst: true));
        _fc = RegisterModule("fc", new Linear(2 * c.Hidden, c.Classes, g));
    }

    public override Tensor Forward(Tensor x) => Forward(x, default);

    /// <param name="x">Normalized [B, 1, 32, W].</param>
    /// <param name="widths">True widths per row when the batch is padded.</param>
    public Tensor Forward(Tensor x, ReadOnlySpan<int> widths)
    {
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        Tensor f = _conv.Forward(x);
        ConvMs = System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        t0 = System.Diagnostics.Stopwatch.GetTimestamp();                        // [B, C, 1, T]
        int b = f.Shape[0], ch = f.Shape[1], t = f.Shape[3];
        Tensor seq = f.Reshape(b, ch, t).Transpose(1, 2);   // [B, T, C]
        Tensor o;
        if (widths.IsEmpty)
        {
            o = _lstm.Forward(seq);
        }
        else
        {
            var len = new int[b];
            for (int i = 0; i < b; i++) len[i] = Config.OutLength(widths[i]);
            o = _lstm.Forward(seq, len);
        }

        LstmMs = System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        return _fc.Forward(o);
    }

    /// <summary>Forward time of the conv stack and of the recurrent part in the last call.</summary>
    public double ConvMs, LstmMs;
}

/// <summary>Weights in the dump format: weights.json (config and tensor table) and weights.bin (float32 little endian).</summary>
internal static class WeightsFile
{
    public static (CrnnConfig Config, string ConfigJson, List<(string Name, int[] Shape, float[] Data)> Tensors) Read(string dir)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, "weights.json")));
        byte[] bin = File.ReadAllBytes(Path.Combine(dir, "weights.bin"));
        var list = new List<(string, int[], float[])>();
        foreach (JsonElement t in doc.RootElement.GetProperty("tensors").EnumerateArray())
        {
            int[] shape = t.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            long off = t.GetProperty("offset").GetInt64();
            int n = shape.Aggregate(1, (a, d) => checked(a * d));
            var data = new float[n];
            Buffer.BlockCopy(bin, checked((int)(off * 4)), data, 0, n * 4);
            list.Add((t.GetProperty("name").GetString()!, shape, data));
        }

        return (CrnnConfig.From(doc.RootElement.GetProperty("config")), doc.RootElement.GetProperty("config").GetRawText(), list);
    }

    public static void Write(string dir, string configJson, IEnumerable<(string Name, Tensor T)> state)
    {
        Directory.CreateDirectory(dir);
        using var bin = File.Create(Path.Combine(dir, "weights.bin"));
        var table = new List<object>();
        long off = 0;
        foreach ((string name, Tensor t) in state)
        {
            ReadOnlySpan<float> d = t.Data<float>();
            bin.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(d));
            table.Add(new { name, shape = Enumerable.Range(0, t.Rank).Select(i => t.Shape[i]).ToArray(), offset = off });
            off += d.Length;
        }

        File.WriteAllText(Path.Combine(dir, "weights.json"),
            "{\"config\":" + configJson + ",\"tensors\":" + JsonSerializer.Serialize(table) + "}");
    }
}
