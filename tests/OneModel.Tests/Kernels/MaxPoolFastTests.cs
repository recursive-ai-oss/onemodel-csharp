using RecursiveAI.OneModel.Kernels;

namespace RecursiveAI.OneModel.Tests.Kernels;

public class MaxPoolFastTests
{
    private static PoolGeom Geometry(int h, int w, int kh, int kw, int sh, int sw, int padH, int padW, int outH, int outW) =>
        new(new Dim3(1, h, w), new Dim3(1, kh, kw), new Dim3(1, sh, sw), new Dim3(1, 1, 1), new Dim3(0, padH, padW), new Dim3(0, padH, padW), new Dim3(1, outH, outW));

    [Fact]
    public void MatchesTheGenericLoopBitForBit()
    {
        // Output sizes beyond the natural ones give windows that lie partly or wholly outside the input (ceil mode, large padding).
        (int H, int W, int KH, int KW, int SH, int SW, int PH, int PW, int OH, int OW)[] cases =
        [
            (112, 112, 3, 3, 2, 2, 1, 1, 56, 56),
            (13, 17, 3, 3, 1, 1, 1, 1, 13, 17),
            (9, 31, 2, 2, 2, 2, 0, 0, 4, 15),
            (9, 31, 2, 2, 2, 2, 0, 0, 5, 16),
            (7, 7, 5, 5, 1, 1, 2, 2, 7, 7),
            (10, 33, 3, 1, 1, 3, 1, 0, 10, 11),
            (5, 5, 3, 3, 3, 3, 2, 2, 3, 3),
            (1, 40, 1, 7, 1, 2, 0, 3, 1, 20),
            (6, 6, 4, 4, 1, 1, 3, 3, 9, 9),
            (3, 3, 3, 3, 1, 1, 0, 0, 1, 1),
        ];
        var rng = new KernelTestData.Rng(77);
        foreach (bool special in new[] { false, true })
        {
            foreach (var c in cases)
            {
                PoolGeom g = Geometry(c.H, c.W, c.KH, c.KW, c.SH, c.SW, c.PH, c.PW, c.OH, c.OW);
                const int planes = 5;
                var x = new float[planes * c.H * c.W];
                KernelTestData.Fill(ref rng, x, special);
                if (special)
                {
                    // Whole rows of NaN and of -infinity, so some windows have no usable tap.
                    for (int i = 0; i < c.W && i < x.Length; i++)
                    {
                        x[i] = float.NaN;
                    }

                    for (int i = c.W; i < 2 * c.W && i < x.Length; i++)
                    {
                        x[i] = float.NegativeInfinity;
                    }
                }

                var want = new float[planes * c.OH * c.OW];
                var got = new float[want.Length];
                Pooling.Max<float>(g, planes, x, want, default, false);
                foreach (int threads in new[] { 1, 4 })
                {
                    Array.Clear(got);
                    MaxPoolFast.Run(g, planes, x, got, threads);
                    for (int i = 0; i < want.Length; i++)
                    {
                        bool same = float.IsNaN(want[i]) ? float.IsNaN(got[i]) : BitConverter.SingleToUInt32Bits(want[i]) == BitConverter.SingleToUInt32Bits(got[i]);
                        Assert.True(same, $"{c} special={special} at {i}: {want[i]} vs {got[i]}");
                    }
                }
            }
        }
    }
}
