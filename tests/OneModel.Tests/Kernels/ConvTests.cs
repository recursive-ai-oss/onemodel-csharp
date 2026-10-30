using RecursiveAI.OneModel.Kernels;

namespace RecursiveAI.OneModel.Tests.Kernels;

public class ConvTests
{
    private static KernelPath[] Paths() => KernelPaths.All.Where(KernelPaths.IsSupported).ToArray();

    /// <summary>Convolution shapes that cover regular, grouped, depthwise (with multiplier), pointwise, strided, dilated and padded cases.</summary>
    internal static List<ConvShape> Shapes()
    {
        var list = new List<ConvShape>
        {
            new(1, 3, 17, 19, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            new(2, 4, 9, 11, 6, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 2),
            new(1, 5, 12, 12, 7, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1),
            new(1, 5, 12, 12, 7, 1, 1, 2, 2, 0, 0, 0, 0, 1, 1, 1),
            new(1, 3, 23, 23, 5, 7, 7, 2, 2, 3, 3, 3, 3, 1, 1, 1),
            new(1, 6, 15, 14, 6, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 6),
            new(2, 6, 15, 14, 12, 3, 3, 2, 2, 0, 1, 2, 1, 1, 1, 6),
            new(1, 4, 20, 21, 4, 5, 5, 1, 1, 2, 2, 2, 2, 1, 1, 4),
            new(1, 4, 20, 21, 4, 5, 5, 2, 2, 2, 2, 2, 2, 1, 1, 4),
            new(1, 3, 20, 21, 3, 3, 3, 1, 2, 2, 1, 0, 3, 2, 3, 3),
            new(1, 2, 5, 5, 3, 3, 3, 1, 1, 0, 0, 0, 0, 1, 1, 1),
            new(1, 1, 3, 3, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            // Depthwise: widths below, at and above the vector widths, strides 1 and 2, 3x3 and 5x5, multiplier, dilation, unpadded,
            // a column stride of 3 (three phase planes), and the two shapes that take the plain fallback (81 taps; stride above 8).
            new(1, 5, 7, 7, 5, 5, 5, 1, 1, 2, 2, 2, 2, 1, 1, 5),
            new(2, 6, 14, 14, 6, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 6),
            new(1, 9, 17, 17, 9, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 9),
            new(1, 4, 33, 35, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 4),
            new(1, 3, 28, 28, 3, 5, 5, 2, 2, 2, 2, 2, 2, 1, 1, 3),
            new(1, 3, 40, 40, 3, 3, 3, 1, 1, 0, 0, 0, 0, 1, 1, 3),
            new(1, 3, 20, 40, 3, 3, 3, 1, 3, 1, 1, 1, 1, 1, 1, 3),
            new(1, 2, 13, 40, 2, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 2),
            new(1, 2, 30, 30, 2, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2, 2),
            new(1, 2, 12, 12, 2, 9, 9, 1, 1, 4, 4, 4, 4, 1, 1, 2),
            new(1, 2, 10, 64, 2, 3, 3, 1, 10, 1, 1, 1, 1, 1, 1, 2),
            new(1, 2, 3, 5, 2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2),
        };
        return list;
    }

    private static (float[] In, float[] W) Data(ConvShape s, bool special)
    {
        var rng = new KernelTestData.Rng(1234 + (ulong)(s.Cin * 31 + s.H));
        var input = new float[s.N * s.Cin * s.H * s.W];
        var weights = new float[s.Cout * (s.Cin / s.Groups) * s.KH * s.KW];
        KernelTestData.Fill(ref rng, input, special);
        KernelTestData.Fill(ref rng, weights, special);
        return (input, weights);
    }

    /// <summary>Naive convolution in the canonical order (channel, kernel row, kernel column), padding as real zeros.</summary>
    internal static void Reference(ConvShape s, float[] input, float[] weights, float[] output, bool exact, double[]? dbl, double[]? scale)
    {
        int cinG = s.Cin / s.Groups;
        int coutG = s.Cout / s.Groups;
        int oh = s.OutH;
        int ow = s.OutW;
        for (int n = 0; n < s.N; n++)
        {
            for (int oc = 0; oc < s.Cout; oc++)
            {
                int g = oc / coutG;
                for (int y = 0; y < oh; y++)
                {
                    for (int x = 0; x < ow; x++)
                    {
                        float acc = 0f;
                        double d = 0;
                        double abs = 0;
                        for (int ci = 0; ci < cinG; ci++)
                        {
                            for (int ky = 0; ky < s.KH; ky++)
                            {
                                for (int kx = 0; kx < s.KW; kx++)
                                {
                                    int iy = (y * s.StrideH) - s.PadTop + (ky * s.DilH);
                                    int ix = (x * s.StrideW) - s.PadLeft + (kx * s.DilW);
                                    float v = iy >= 0 && iy < s.H && ix >= 0 && ix < s.W
                                        ? input[((((n * s.Cin) + (g * cinG) + ci) * s.H) + iy) * s.W + ix]
                                        : 0f;
                                    float wv = weights[((((oc * cinG) + ci) * s.KH) + ky) * s.KW + kx];
                                    float prod = wv * v;
                                    acc += prod;
                                    d += (double)wv * v;
                                    abs += Math.Abs((double)wv * v);
                                }
                            }
                        }

                        int o = ((((n * s.Cout) + oc) * oh) + y) * ow + x;
                        output[o] = float.IsNaN(acc) ? BitConverter.UInt32BitsToSingle(KernelTestData.CanonicalNaN) : acc;
                        if (dbl is not null)
                        {
                            dbl[o] = d;
                            scale![o] = abs;
                        }
                    }
                }
            }
        }
    }

    private static float[] Run(ConvShape s, float[] input, float[] weights, KernelMode mode, KernelPath path, int threads)
    {
        var o = new float[s.N * s.Cout * s.OutH * s.OutW];
        Conv.Run(mode, path, s, input, weights, o, threads);
        return o;
    }

    [Fact]
    public void ExactMatchesCanonicalOrderOnEveryPathAndThreadCount()
    {
        foreach (bool special in new[] { false, true })
        {
            foreach (ConvShape s in Shapes())
            {
                (float[] input, float[] weights) = Data(s, special);
                var expected = new float[s.N * s.Cout * s.OutH * s.OutW];
                Reference(s, input, weights, expected, true, null, null);
                foreach (KernelPath path in Paths())
                {
                    foreach (int threads in new[] { 1, 5 })
                    {
                        float[] got = Run(s, input, weights, KernelMode.Exact, path, threads);
                        Assert.True(KernelTestData.SameBits(expected, got), $"{path} x{threads} differs on {s} special={special}");
                    }
                }
            }
        }
    }

    [Fact]
    public void FastIsWithinToleranceAndIndependentOfThreads()
    {
        foreach (ConvShape s in Shapes())
        {
            (float[] input, float[] weights) = Data(s, false);
            int len = s.N * s.Cout * s.OutH * s.OutW;
            var dbl = new double[len];
            var scale = new double[len];
            Reference(s, input, weights, new float[len], true, dbl, scale);
            int k = (s.Cin / s.Groups) * s.KH * s.KW;
            foreach (KernelPath path in Paths())
            {
                float[] one = Run(s, input, weights, KernelMode.Fast, path, 1);
                Assert.True(KernelTestData.SameBits(one, Run(s, input, weights, KernelMode.Fast, path, 7)), $"{path} thread dependence on {s}");
                for (int i = 0; i < len; i++)
                {
                    double tol = (k * Math.Pow(2, -24) * scale[i]) + 1e-30;
                    Assert.True(Math.Abs(one[i] - dbl[i]) <= tol, $"{path} {s} at {i}: {one[i]} vs {dbl[i]}");
                }
            }
        }
    }
}
