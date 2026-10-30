using System.Buffers.Binary;
using RecursiveAI.OneModel.Tensors;

namespace CrnnFinetune;

/// <summary>One training batch: padded 8-bit images [B][32][W], true widths and concatenated CTC targets.</summary>
internal sealed record Batch(int Rows, int Width, int[] Widths, int[] TargetLengths, int[] Targets, byte[] Pixels)
{
    /// <summary>Reads a batch file: int32 B, W; widths[B]; target lengths[B]; targets[sum]; B*32*W bytes.</summary>
    public static Batch Read(string path)
    {
        byte[] f = File.ReadAllBytes(path);
        int p = 0;
        int Next() { int v = BinaryPrimitives.ReadInt32LittleEndian(f.AsSpan(p, 4)); p += 4; return v; }
        int b = Next(), w = Next();
        if (b < 1 || w < 1) throw new InvalidDataException($"{path}: bad batch header");
        var widths = new int[b];
        var tl = new int[b];
        for (int i = 0; i < b; i++) widths[i] = Next();
        for (int i = 0; i < b; i++) tl[i] = Next();
        var targets = new int[tl.Sum()];
        for (int i = 0; i < targets.Length; i++) targets[i] = Next();
        long n = (long)b * 32 * w;
        if (f.Length - p != n) throw new InvalidDataException($"{path}: expected {n} pixel bytes, found {f.Length - p}");
        return new Batch(b, w, widths, tl, targets, f.AsSpan(p).ToArray());
    }

    /// <summary>x = p/255, then (x − 0.5)/0.5, as float32 [B, 1, 32, W].</summary>
    public Tensor ToInput() => Normalize(Pixels, [Rows, 1, 32, Width]);

    public static Tensor Normalize(ReadOnlySpan<byte> px, int[] dims)
    {
        var x = new float[px.Length];
        for (int i = 0; i < x.Length; i++) x[i] = ((px[i] / 255f) - 0.5f) / 0.5f;
        Tensor t = Tensor.FromSpan<float>(x, Shape.Create(dims));
        ActivationScope.Track(t);
        return t;
    }
}

/// <summary>A validation line set: labels as class ids and 32-px-high gray images of varying width.</summary>
internal sealed class LineSet
{
    public int[][] Labels = [];
    public int[] Widths = [];
    public byte[][] Images = [];

    public static LineSet Read(string path)
    {
        byte[] f = File.ReadAllBytes(path);
        int p = 0;
        int Next() { int v = BinaryPrimitives.ReadInt32LittleEndian(f.AsSpan(p, 4)); p += 4; return v; }
        int n = Next();
        var s = new LineSet { Labels = new int[n][], Widths = new int[n], Images = new byte[n][] };
        for (int i = 0; i < n; i++)
        {
            int w = Next(), len = Next();
            if (w < 1 || len < 0 || p + (4L * len) + (32L * w) > f.Length) throw new InvalidDataException($"{path}: line {i} out of range");
            s.Widths[i] = w;
            s.Labels[i] = new int[len];
            for (int k = 0; k < len; k++) s.Labels[i][k] = Next();
            s.Images[i] = f.AsSpan(p, 32 * w).ToArray();
            p += 32 * w;
        }

        return s;
    }

    /// <summary>Greedy-decoded character error rate, batches of 128 in width order padded with each row's last column.</summary>
    public (double Cer, double LineAccuracy, int Lines) Evaluate(Crnn model, int batchSize = 128)
    {
        model.Eval();
        int n = Widths.Length;
        int[] order = Enumerable.Range(0, n).OrderBy(i => Widths[i]).ToArray();   // stable
        long errors = 0, chars = 0, exact = 0;
        using var noGrad = GradMode.NoGrad();
        for (int s = 0; s < n; s += batchSize)
        {
            int[] idx = order.Skip(s).Take(batchSize).ToArray();
            int wmax = idx.Max(i => Widths[i]);
            var px = new byte[idx.Length * 32 * wmax];
            for (int k = 0; k < idx.Length; k++)
            {
                int w = Widths[idx[k]];
                for (int r = 0; r < 32; r++)
                {
                    byte[] im = Images[idx[k]];
                    int row = ((k * 32) + r) * wmax;
                    Array.Copy(im, r * w, px, row, w);
                    Array.Fill(px, im[(r * w) + w - 1], row + w, wmax - w);
                }
            }

            using var scope = new ActivationScope();
            Tensor x = Batch.Normalize(px, [idx.Length, 1, 32, wmax]);
            Tensor logits = model.Forward(x, idx.Select(i => Widths[i]).ToArray());
            int t = logits.Shape[1], c = logits.Shape[2];
            ReadOnlySpan<float> lg = logits.Data<float>();
            for (int k = 0; k < idx.Length; k++)
            {
                int len = model.Config.OutLength(Widths[idx[k]]);
                var hyp = new List<int>();
                int prev = 0;
                for (int ti = 0; ti < len; ti++)
                {
                    ReadOnlySpan<float> row = lg.Slice(((k * t) + ti) * c, c);
                    int best = 0;
                    for (int j = 1; j < c; j++) if (row[j] > row[best]) best = j;   // lowest index on ties
                    if (best != prev && best != 0) hyp.Add(best);
                    prev = best;
                }

                int[] rf = Labels[idx[k]];
                errors += EditDistance(hyp, rf);
                chars += rf.Length;
                if (hyp.SequenceEqual(rf)) exact++;
            }
        }

        return ((double)errors / chars, (double)exact / n, n);
    }

    private static int EditDistance(List<int> a, int[] b)
    {
        var prev = new int[b.Length + 1];
        for (int j = 0; j <= b.Length; j++) prev[j] = j;
        for (int i = 1; i <= a.Count; i++)
        {
            var cur = new int[b.Length + 1];
            cur[0] = i;
            for (int j = 1; j <= b.Length; j++)
                cur[j] = Math.Min(Math.Min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1));
            prev = cur;
        }

        return prev[b.Length];
    }
}
