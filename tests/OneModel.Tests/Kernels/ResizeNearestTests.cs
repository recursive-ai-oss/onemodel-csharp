using RecursiveAI.OneModel.Kernels;

namespace RecursiveAI.OneModel.Tests.Kernels;

public class ResizeNearestTests
{
    // Nearest with asymmetric coordinates and floor: output j reads input floor(j / scale), clamped. Checked against a direct
    // element-by-element loop, with one thread and with many (the single pass splits its rows into chunks).
    [Theory]
    [InlineData(new[] { 1, 32, 32, 40 }, new[] { 1, 32, 64, 80 })]
    [InlineData(new[] { 2, 3, 5, 7 }, new[] { 2, 3, 10, 7 })]
    [InlineData(new[] { 2, 3, 5, 7 }, new[] { 2, 6, 5, 21 })]
    [InlineData(new[] { 1, 1, 9, 11 }, new[] { 1, 1, 4, 5 })]
    [InlineData(new[] { 1, 1, 9, 11 }, new[] { 1, 1, 13, 17 })]
    [InlineData(new[] { 6 }, new[] { 15 })]
    [InlineData(new[] { 3, 4, 5, 6, 7 }, new[] { 3, 8, 5, 12, 7 })]
    [InlineData(new[] { 2, 700, 3 }, new[] { 2, 1400, 3 })]
    public void MatchesADirectLoop(int[] inDims, int[] outDims)
    {
        int rank = inDims.Length;
        long inLen = 1, outLen = 1;
        for (int a = 0; a < rank; a++)
        {
            inLen *= inDims[a];
            outLen *= outDims[a];
        }

        var rng = new Random(11);
        float[] x = new float[inLen];
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = (float)((rng.NextDouble() * 2) - 1);
        }

        float[] scales = new float[rank];
        for (int a = 0; a < rank; a++)
        {
            scales[a] = (float)outDims[a] / inDims[a];
        }

        float[] want = new float[outLen];
        for (long o = 0; o < outLen; o++)
        {
            long rem = o, src = 0, stride = 1;
            for (int a = rank - 1; a >= 0; a--)
            {
                int j = (int)(rem % outDims[a]);
                rem /= outDims[a];
                int i = Math.Clamp((int)MathF.Floor(j / scales[a]), 0, inDims[a] - 1);
                src += i * stride;
                stride *= inDims[a];
            }

            want[o] = x[src];
        }

        var options = new ResizeOptions(ResizeMode.Nearest, ResizeCoord.Asymmetric, ResizeNearest.Floor, -0.75f, false, 0f);
        foreach (int threads in new[] { 1, 8 })
        {
            float[] got = new float[outLen];
            Resize.Run(options, inDims, outDims, scales, default, x, got, new float[1], new float[1], new int[outDims.Sum()], new float[outDims.Sum()], false, threads);
            Assert.True(want.AsSpan().SequenceEqual(got), $"threads {threads}");
        }
    }
}
