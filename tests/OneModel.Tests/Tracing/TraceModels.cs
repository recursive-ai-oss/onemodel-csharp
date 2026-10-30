using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Tracing;

namespace RecursiveAI.OneModel.Tests.Tracing;

// The three models of the tracing checks, written on the eager ops with seeded weights: the conv net and the 2-layer encoder of the
// autograd checks (evaluation mode) and a CRNN-like conv -> BiLSTM -> linear model with CTC-style log-probabilities.
internal static class TraceModels
{
    public static Tensor Rand(int seed, float scale, params int[] dims)
    {
        var rng = new Random(seed);
        Shape shape = Shape.Create(dims);
        var data = new float[shape.Length];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        }

        return Tensor.FromSpan<float>(data, shape);
    }

    public static Tensor Const(float value, params int[] dims)
    {
        Tensor t = Tensor.Create(DType.F32, Shape.Create(dims));
        t.Fill(value);
        return t;
    }

    public static Tensor Ids(int seed, int vocab, params int[] dims)
    {
        var rng = new Random(seed);
        Shape shape = Shape.Create(dims);
        var data = new long[shape.Length];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = rng.Next(vocab);
        }

        return Tensor.FromSpan<long>(data, shape);
    }

    public sealed class Cnn
    {
        public readonly Dictionary<string, Tensor> P = [];

        public Cnn()
        {
            P["c1w"] = Rand(1, 0.3f, 4, 3, 3, 3);
            P["c1b"] = Rand(2, 0.1f, 4);
            P["bn1g"] = Rand(3, 0.2f, 4).Also(t => Ops.Add(t, Const(1, 1)));
            P["bn1b"] = Rand(4, 0.1f, 4);
            P["rm1"] = Rand(5, 0.2f, 4);
            P["rv1"] = Ops.Add(Rand(6, 0.2f, 4), Const(1f, 1));
            P["c2w"] = Rand(7, 0.2f, 8, 4, 3, 3);
            P["c2b"] = Rand(8, 0.1f, 8);
            P["bn2g"] = Ops.Add(Rand(9, 0.2f, 8), Const(1, 1));
            P["bn2b"] = Rand(10, 0.1f, 8);
            P["rm2"] = Rand(11, 0.2f, 8);
            P["rv2"] = Ops.Add(Rand(12, 0.2f, 8), Const(1f, 1));
            P["fcw"] = Rand(13, 0.2f, 10, 72);
            P["fcb"] = Rand(14, 0.1f, 10);
        }

        public Tensor Forward(Tensor x)
        {
            int n = x.Shape[0];
            Tensor h = Ops.Conv2d(x, P["c1w"], P["c1b"], new Conv2dParams(PadH: 1, PadW: 1));
            h = Ops.BatchNorm2d(h, P["bn1g"], P["bn1b"], P["rm1"], P["rv1"], training: false);
            h = Ops.MaxPool2d(Ops.Relu(h), new Pool2dParams(2, 2, 2, 2, 0, 0));
            h = Ops.Conv2d(h, P["c2w"], P["c2b"], new Conv2dParams(PadH: 1, PadW: 1));
            h = Ops.BatchNorm2d(h, P["bn2g"], P["bn2b"], P["rm2"], P["rv2"], training: false);
            h = Ops.AvgPool2d(Ops.Relu(h), new Pool2dParams(2, 2, 2, 2, 0, 0));
            return Ops.LogSoftmax(Ops.Linear(h.Reshape(n, 72), P["fcw"], P["fcb"]));
        }

        public static Tensor Input(int n) => Rand(100 + n, 1f, n, 3, 12, 12);
    }

    public sealed class Encoder
    {
        private const int D = 16, Heads = 2, Layers = 2, Vocab = 50;
        public readonly Dictionary<string, Tensor> P = [];

        public Encoder()
        {
            int seed = 200;
            P["emb"] = Rand(seed++, 0.5f, Vocab, D);
            for (int i = 0; i < Layers; i++)
            {
                string p = $"l{i}.";
                foreach (string w in new[] { "wq", "wk", "wv", "wo" })
                {
                    P[p + w] = Rand(seed++, 0.25f, D, D);
                    P[p + "b" + w[1]] = Rand(seed++, 0.05f, D);
                }

                P[p + "ln1g"] = Ops.Add(Rand(seed++, 0.1f, D), Const(1, 1));
                P[p + "ln1b"] = Rand(seed++, 0.05f, D);
                P[p + "w1"] = Rand(seed++, 0.25f, 32, D);
                P[p + "b1"] = Rand(seed++, 0.05f, 32);
                P[p + "w2"] = Rand(seed++, 0.2f, D, 32);
                P[p + "b2"] = Rand(seed++, 0.05f, D);
                P[p + "ln2g"] = Ops.Add(Rand(seed++, 0.1f, D), Const(1, 1));
                P[p + "ln2b"] = Rand(seed++, 0.05f, D);
            }
        }

        public Tensor Forward(Tensor ids, Tensor mask)
        {
            int b = ids.Shape[0], n = ids.Shape[1];
            Tensor Split(Tensor z) => z.Reshape(b, n, Heads, D / Heads).Permute(0, 2, 1, 3);
            Tensor x = Ops.Embedding(P["emb"], ids);
            for (int i = 0; i < Layers; i++)
            {
                string p = $"l{i}.";
                Tensor q = Split(Ops.Linear(x, P[p + "wq"], P[p + "bq"]));
                Tensor k = Split(Ops.Linear(x, P[p + "wk"], P[p + "bk"]));
                Tensor v = Split(Ops.Linear(x, P[p + "wv"], P[p + "bv"]));
                Tensor a = Ops.Attention(q, k, v, mask).Permute(0, 2, 1, 3).Reshape(b, n, D);
                Tensor x1 = Ops.LayerNorm(Ops.Add(x, Ops.Linear(a, P[p + "wo"], P[p + "bo"])), P[p + "ln1g"], P[p + "ln1b"]);
                Tensor f = Ops.Linear(Ops.Gelu(Ops.Linear(x1, P[p + "w1"], P[p + "b1"])), P[p + "w2"], P[p + "b2"]);
                x = Ops.LayerNorm(Ops.Add(x1, f), P[p + "ln2g"], P[p + "ln2b"]);
            }

            return x;
        }

        public static Tensor Mask(int b, int n, int seed)
        {
            var rng = new Random(seed);
            var d = new float[b * n * n];
            for (int bi = 0; bi < b; bi++)
            {
                int valid = n - rng.Next(0, 2);
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        d[(((bi * n) + i) * n) + j] = j < valid ? 0f : -1e4f;
                    }
                }
            }

            return Tensor.FromSpan<float>(d, Shape.Create(b, n, n));
        }
    }

    public sealed class Crnn
    {
        private const int H = 6, Classes = 7;
        public readonly Dictionary<string, Tensor> P = [];
        public readonly RnnLayer[] Layers;

        public Crnn()
        {
            P["cw"] = Rand(300, 0.4f, 4, 1, 3, 3);
            P["cb"] = Rand(301, 0.1f, 4);
            int inSize = 4 * 8;
            var layers = new List<RnnLayer>();
            for (int l = 0; l < 2; l++)
            {
                int isz = l == 0 ? inSize : 2 * H;
                for (int d = 0; d < 2; d++)
                {
                    int s = 310 + (l * 10) + (d * 5);
                    layers.Add(new RnnLayer(Rand(s, 0.3f, 4 * H, isz), Rand(s + 1, 0.3f, 4 * H, H), Rand(s + 2, 0.1f, 4 * H), Rand(s + 3, 0.1f, 4 * H)));
                }
            }

            Layers = [.. layers];
            P["fw"] = Rand(340, 0.3f, Classes, 2 * H);
            P["fb"] = Rand(341, 0.1f, Classes);
        }

        // x: [B, 1, 16, W] -> log-probabilities [W/2, B, classes]
        public Tensor Forward(Tensor x)
        {
            int b = x.Shape[0];
            Tensor h = Ops.Relu(Ops.Conv2d(x, P["cw"], P["cb"], new Conv2dParams(PadH: 1, PadW: 1)));
            h = Ops.MaxPool2d(h, new Pool2dParams(2, 2, 2, 2, 0, 0)); // [B, 4, 8, W/2]
            int w = h.Shape[3];
            Tensor seq = h.Permute(3, 0, 1, 2).Reshape(w, b, 4 * 8);
            Tensor y = Ops.Lstm(seq, Layers, bidirectional: true);
            return Ops.LogSoftmax(Ops.Linear(y, P["fw"], P["fb"]));
        }

        public static Tensor Input(int b, int w) => Rand(400 + b + w, 1f, b, 1, 16, w);
    }
}

internal static class TraceExtensions
{
    public static Tensor Also(this Tensor t, Func<Tensor, Tensor> f) => f(t);
}

internal static class TraceHelpers
{
    public static float Scaled(Tensor a, Tensor b)
    {
        float[] x = a.ToArray<float>(), y = b.ToArray<float>();
        Assert.Equal(x.Length, y.Length);
        double max = 0, diff = 0;
        for (int i = 0; i < x.Length; i++)
        {
            max = Math.Max(max, Math.Abs(y[i]));
            diff = Math.Max(diff, Math.Abs(x[i] - y[i]));
        }

        return (float)(diff / Math.Max(max, 1e-30));
    }

    public static Tensor Run(Session s, string[] names, Tensor[] inputs, string output = "y")
    {
        Span<Tensor> outs = new Tensor[1];
        s.Run(names, inputs, [output], outs);
        return outs[0];
    }

    public static bool BitEqual(Tensor a, Tensor b)
    {
        if (a.Shape != b.Shape)
        {
            return false;
        }

        return a.ToArray<float>().AsSpan().SequenceEqual(b.ToArray<float>());
    }
}
