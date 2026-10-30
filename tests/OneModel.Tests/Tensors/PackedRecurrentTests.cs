using RecursiveAI.OneModel.Tensors;
using static RecursiveAI.OneModel.Tests.Tensors.GradCheck;

namespace RecursiveAI.OneModel.Tests.Tensors;

[Collection("autograd")]
public class PackedRecurrentTests
{
    private static Shape S(params int[] d) => Shape.Create(d);

    private static Tensor Rand(Random r, params int[] dims)
    {
        var s = S(dims);
        var a = new float[s.Length];
        for (int i = 0; i < a.Length; i++) a[i] = (float)((r.NextDouble() * 1.2) - 0.6);
        return Tensor.FromSpan<float>(a, s);
    }

    private static RnnLayer[] Weights(Random r, int inSize, int h, int gates, int layers, int dirs)
    {
        var list = new List<RnnLayer>();
        for (int l = 0; l < layers; l++)
        {
            for (int d = 0; d < dirs; d++)
            {
                int i = l == 0 ? inSize : h * dirs;
                list.Add(new RnnLayer(Rand(r, gates * h, i), Rand(r, gates * h, h), Rand(r, gates * h), Rand(r, gates * h)));
            }
        }

        return list.ToArray();
    }

    // A padded batch with lengths must give, for every row, what that row gives alone at its own length, and zeros after it.
    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void PaddedRowsEqualRowsRunAlone(bool bidirectional, bool gru)
    {
        var r = new Random(7);
        int steps = 6, batch = 4, inSize = 3, h = 5, dirs = bidirectional ? 2 : 1;
        int[] lengths = [6, 3, 1, 4];
        RnnLayer[] w = Weights(r, inSize, h, gru ? 3 : 4, 2, dirs);
        Tensor x = Rand(r, steps, batch, inSize);
        Tensor packed = gru ? Ops.Gru(x, w, bidirectional, false, null, lengths) : Ops.Lstm(x, w, bidirectional, false, null, lengths);
        float[] got = packed.ToArray<float>();
        float[] xs = x.ToArray<float>();
        for (int b = 0; b < batch; b++)
        {
            int len = lengths[b];
            var row = new float[len * inSize];
            for (int t = 0; t < len; t++)
                for (int k = 0; k < inSize; k++) row[(t * inSize) + k] = xs[(((t * batch) + b) * inSize) + k];
            Tensor xr = Tensor.FromSpan<float>(row, S(len, 1, inSize));
            float[] want = (gru ? Ops.Gru(xr, w, bidirectional) : Ops.Lstm(xr, w, bidirectional)).ToArray<float>();
            for (int t = 0; t < steps; t++)
                for (int k = 0; k < h * dirs; k++)
                {
                    float g = got[(((t * batch) + b) * h * dirs) + k];
                    if (t < len) Assert.True(Math.Abs(g - want[(t * h * dirs) + k]) < 1e-5f, $"row {b} t {t} k {k}: {g} vs {want[(t * h * dirs) + k]}");
                    else Assert.Equal(0f, g);
                }
        }
    }

    [Fact]
    public void GradientsWithLengths()
    {
        Func<Random, float> small = Uniform(-0.6f, 0.6f);
        Run("Lstm bidir lengths", t => Ops.Lstm(t[0], [new RnnLayer(t[1], t[2], t[3], t[4]), new RnnLayer(t[5], t[6], t[7], t[8])], true, true, null, new[] { 4, 2, 3 }),
            [S(3, 4, 2), S(8, 2), S(8, 2), S(8), S(8), S(8, 2), S(8, 2), S(8), S(8)], Enumerable.Repeat(small, 9).ToArray(), h: 2e-2, tolerance: 5e-3);
    }
}
