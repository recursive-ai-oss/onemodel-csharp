using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text.Json;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Tensors;

// The two small models of tools/autograd_oracle.py, built on the eager ops, and a reader for its weight/gradient files.
internal sealed class OracleFile
{
    private readonly Dictionary<string, (Shape Shape, float[] Data)> _items = new();

    public static string DataDir([CallerFilePath] string file = "")
    {
        string? dir = Path.GetDirectoryName(file);
        while (dir is not null && !File.Exists(Path.Combine(dir, "global.json")))
        {
            dir = Path.GetDirectoryName(dir);
        }

        return Path.Combine(dir ?? throw new InvalidOperationException("global.json not found"), "testdata", "autograd");
    }

    public static OracleFile Load(string model)
    {
        string basePath = Path.Combine(DataDir(), model);
        byte[] bin = File.ReadAllBytes(basePath + ".bin");
        float[] all = new float[bin.Length / 4];
        Buffer.BlockCopy(bin, 0, all, 0, bin.Length);
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(basePath + ".json"));
        var f = new OracleFile();
        foreach (JsonProperty p in doc.RootElement.GetProperty("tensors").EnumerateObject())
        {
            int[] dims = p.Value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            Shape shape = Shape.Create(dims);
            int off = p.Value.GetProperty("offset").GetInt32();
            f._items[p.Name] = (shape, all.AsSpan(off, shape.Length).ToArray());
        }

        return f;
    }

    public IEnumerable<string> Names => _items.Keys;

    public Shape ShapeOf(string name) => _items[name].Shape;

    public float[] Values(string name) => _items[name].Data;

    public Tensor Make(string name, bool requiresGrad = false)
    {
        (Shape shape, float[] data) = _items[name];
        Tensor t = Tensor.FromSpan<float>(data, shape);
        return requiresGrad ? t.WithRequiresGrad() : t;
    }

    public Tensor MakeInts(string name, bool int64)
    {
        (Shape shape, float[] data) = _items[name];
        return int64
            ? Tensor.FromSpan<long>(data.Select(v => (long)v).ToArray(), shape)
            : Tensor.FromSpan<int>(data.Select(v => (int)v).ToArray(), shape);
    }
}

internal static class GradHash
{
    // SHA-256 over the gradients in name order: name, shape, then the raw float bits.
    public static string Of(IEnumerable<KeyValuePair<string, Tensor>> grads, float loss)
    {
        using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        h.AppendData(BitConverter.GetBytes(loss));
        foreach ((string name, Tensor g) in grads.OrderBy(k => k.Key, StringComparer.Ordinal))
        {
            h.AppendData(System.Text.Encoding.UTF8.GetBytes(name));
            foreach (int d in g.Shape.Dimensions)
            {
                h.AppendData(BitConverter.GetBytes(d));
            }

            h.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(g.Data<float>()));
        }

        return Convert.ToHexString(h.GetHashAndReset());
    }
}

internal sealed class CnnModel
{
    public readonly Dictionary<string, Tensor> Params = new();
    public readonly Tensor X, Targets, Rm1, Rv1, Rm2, Rv2;

    public CnnModel(OracleFile o)
    {
        foreach (string n in new[] { "c1w", "c1b", "bn1g", "bn1b", "c2w", "c2b", "bn2g", "bn2b", "fcw", "fcb" })
        {
            Params[n] = o.Make("in." + n, requiresGrad: true);
        }

        X = o.Make("in.x", requiresGrad: true);
        Params["x"] = X;
        Targets = o.MakeInts("in.targets", int64: true);
        Rm1 = Tensor.Zeros(DType.F32, Shape.Create(4));
        Rv1 = Tensor.Create(DType.F32, Shape.Create(4));
        Rv1.Fill(1f);
        Rm2 = Tensor.Zeros(DType.F32, Shape.Create(8));
        Rv2 = Tensor.Create(DType.F32, Shape.Create(8));
        Rv2.Fill(1f);
    }

    public Tensor Forward(out Tensor logits)
    {
        Tensor h = Ops.Conv2d(X, Params["c1w"], Params["c1b"], new Conv2dParams(PadH: 1, PadW: 1));
        h = Ops.BatchNorm2d(h, Params["bn1g"], Params["bn1b"], Rm1, Rv1, training: true);
        h = Ops.MaxPool2d(Ops.Relu(h), new Pool2dParams(2, 2, 2, 2, 0, 0));
        h = Ops.Conv2d(h, Params["c2w"], Params["c2b"], new Conv2dParams(PadH: 1, PadW: 1));
        h = Ops.BatchNorm2d(h, Params["bn2g"], Params["bn2b"], Rm2, Rv2, training: true);
        h = Ops.AvgPool2d(Ops.Relu(h), new Pool2dParams(2, 2, 2, 2, 0, 0));
        logits = Ops.Linear(h.Reshape(4, 72), Params["fcw"], Params["fcb"]);
        return Ops.NllLoss(Ops.LogSoftmax(logits), Targets);
    }

    /// <summary>One training step's worth of work: zero gradients, forward, backward. Returns the loss.</summary>
    public float Step()
    {
        foreach (Tensor p in Params.Values)
        {
            Backprop.ZeroGrad(p);
        }

        using var scope = new ActivationScope();
        Tensor loss = Forward(out _);
        float value = loss.Data<float>()[0];
        loss.Backward();
        return value;
    }

    public Dictionary<string, Tensor> Grads() => Params.ToDictionary(k => k.Key, k => k.Value.Grad);
}

internal sealed class EncoderModel
{
    private const int D = 16, Heads = 2, Layers = 2, B = 2, N = 5;
    public readonly Dictionary<string, Tensor> Params = new();
    public readonly Tensor Ids, Mask, Proj;

    public EncoderModel(OracleFile o)
    {
        foreach (string n in o.Names)
        {
            if (n.StartsWith("in.", StringComparison.Ordinal) && n is not ("in.ids" or "in.mask" or "in.proj"))
            {
                Params[n[3..]] = o.Make(n, requiresGrad: true);
            }
        }

        Ids = o.MakeInts("in.ids", int64: false);
        Mask = o.Make("in.mask");
        Proj = o.Make("in.proj");
    }

    private static Tensor Split(Tensor z) => z.Reshape(B, N, Heads, D / Heads).Permute(0, 2, 1, 3);

    public Tensor Forward(out Tensor y)
    {
        Tensor x = Ops.Embedding(Params["emb"], Ids);
        for (int i = 0; i < Layers; i++)
        {
            string p = $"l{i}.";
            Tensor q = Split(Ops.Linear(x, Params[p + "wq"], Params[p + "bq"]));
            Tensor k = Split(Ops.Linear(x, Params[p + "wk"], Params[p + "bk"]));
            Tensor v = Split(Ops.Linear(x, Params[p + "wv"], Params[p + "bv"]));
            Tensor a = Ops.Attention(q, k, v, Mask).Permute(0, 2, 1, 3).Reshape(B, N, D);
            Tensor x1 = Ops.LayerNorm(Ops.Add(x, Ops.Linear(a, Params[p + "wo"], Params[p + "bo"])), Params[p + "ln1g"], Params[p + "ln1b"]);
            Tensor f = Ops.Linear(Ops.Gelu(Ops.Linear(x1, Params[p + "w1"], Params[p + "b1"])), Params[p + "w2"], Params[p + "b2"]);
            x = Ops.LayerNorm(Ops.Add(x1, f), Params[p + "ln2g"], Params[p + "ln2b"]);
        }

        y = x;
        return Ops.Sum(Ops.Mul(x, Proj));
    }

    public float Step()
    {
        foreach (Tensor p in Params.Values)
        {
            Backprop.ZeroGrad(p);
        }

        using var scope = new ActivationScope();
        Tensor loss = Forward(out _);
        float value = loss.Data<float>()[0];
        loss.Backward();
        return value;
    }

    public Dictionary<string, Tensor> Grads() => Params.ToDictionary(k => k.Key, k => k.Value.Grad);
}
