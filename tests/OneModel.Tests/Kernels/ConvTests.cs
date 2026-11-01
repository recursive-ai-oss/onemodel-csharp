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
            // Wide rows with leftover vectors (the tails are taken four rows at a time, then singly), and a channel multiplier of four.
            new(1, 3, 9, 100, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3),
            new(1, 2, 7, 70, 2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2),
            new(1, 2, 12, 150, 2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2),
            new(1, 1, 20, 24, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1),
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
    [Fact]
    public void FastPackedWeightsGiveTheSameBitsAsUnpackedAndStayWithinTolerance()
    {
        // More than Kc taps, more than one column tile, more than one row tile, groups, strides, padding and a ragged last strip.
        ConvShape[] shapes =
        [
            new(1, 40, 30, 31, 70, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            new(2, 33, 17, 19, 50, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1),
            new(1, 300, 9, 9, 130, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1),
            new(1, 64, 28, 28, 64, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1),
            // Pointwise layers read their input in place: groups, two images, a ragged last strip of columns and of rows.
            new(2, 48, 10, 13, 40, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 2),
            new(1, 130, 7, 7, 33, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1),
            new(1, 24, 21, 23, 36, 5, 5, 1, 1, 2, 2, 2, 2, 1, 1, 3),
            new(1, 20, 25, 25, 20, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2, 1),
            new(1, 8, 1, 300, 12, 1, 7, 1, 2, 0, 3, 0, 3, 1, 1, 1),
            new(1, 512, 7, 7, 96, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            // A 1x1 input and kernel (a squeeze-and-excitation layer) runs as a matrix-vector product: ragged rows and slivers, more than
            // Kc taps, groups, two images.
            new(1, 672, 1, 1, 168, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1),
            new(2, 700, 1, 1, 97, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1),
            new(1, 24, 1, 1, 36, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 3),
            new(1, 5, 1, 1, 7, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1),
            new(1, 520, 1, 1, 1000, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1),
        ];
        foreach (ConvShape s in shapes)
        {
            (float[] input, float[] weights) = Data(s, false);
            int len = s.N * s.Cout * s.OutH * s.OutW;
            var dbl = new double[len];
            var scale = new double[len];
            Reference(s, input, weights, new float[len], true, dbl, scale);
            int k = (s.Cin / s.Groups) * s.KH * s.KW;
            foreach (KernelPath path in Paths())
            {
                float[] plain = Run(s, input, weights, KernelMode.Fast, path, 1);
                var packed = new float[ConvPacked.PackedWeightsLength(path, s)];
                ConvPacked.PackWeights(path, s, weights, packed);
                // 3x3 stride-1 layers with enough channels use Winograd's transform: not the direct sums' bits, but the same on every thread count.
                bool winograd = ConvWinograd.Eligible(s);
                var first = new float[len];
                foreach (int threads in new[] { 1, 6 })
                {
                    var o = new float[len];
                    Conv.RunPacked(path, s, input, packed, o, threads);
                    if (winograd)
                    {
                        if (threads == 1)
                        {
                            o.CopyTo(first, 0);
                        }

                        Assert.True(KernelTestData.SameBits(first, o), $"{path} x{threads} Winograd thread dependence on {s}");
                    }
                    else
                    {
                        Assert.True(KernelTestData.SameBits(plain, o), $"{path} x{threads} packed differs on {s}");
                    }
                }

                float[] got = winograd ? first : plain;
                double factor = winograd ? 16 : 1;
                for (int i = 0; i < len; i++)
                {
                    double tol = (factor * k * Math.Pow(2, -24) * scale[i]) + 1e-30;
                    Assert.True(Math.Abs(got[i] - dbl[i]) <= tol, $"{path} {s} at {i}: {got[i]} vs {dbl[i]}");
                }
            }
        }
    }

    [Fact]
    public void FastBatchesInWavesGiveTheBitsOfOneImageAtATime()
    {
        // Several row tiles with a small tile count per image: the images are gathered and multiplied in waves, which must not change a bit.
        ConvShape[] shapes =
        [
            new(5, 64, 14, 14, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1),
            new(7, 48, 9, 9, 200, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 2),
            new(3, 128, 7, 7, 300, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1),
        ];
        foreach (ConvShape s in shapes)
        {
            (float[] input, float[] weights) = Data(s, false);
            int inPer = s.Cin * s.H * s.W;
            int outPer = s.Cout * s.OutH * s.OutW;
            ConvShape one = s with { N = 1 };
            foreach (KernelPath path in Paths())
            {
                var packed = new float[ConvPacked.PackedWeightsLength(path, s)];
                ConvPacked.PackWeights(path, s, weights, packed);
                var expected = new float[s.N * outPer];
                for (int img = 0; img < s.N; img++)
                {
                    var o1 = new float[outPer];
                    Conv.RunPacked(path, one, input.AsSpan(img * inPer, inPer).ToArray(), packed, o1, 1);
                    o1.CopyTo(expected, img * outPer);
                }

                foreach (int threads in new[] { 1, 6 })
                {
                    var o = new float[s.N * outPer];
                    Conv.RunPacked(path, s, input, packed, o, threads);
                    Assert.True(KernelTestData.SameBits(expected, o), $"{path} x{threads} batch differs on {s}");
                }
            }
        }
    }

    [Fact]
    public void WinogradMatchesTheReferenceAndIsIndependentOfThreads()
    {
        // Odd and even output sizes, every padding, several images, more than Kc channels, channel counts that leave ragged row strips,
        // widths that are and are not multiples of eight, maps smaller than a tile row group.
        ConvShape[] shapes =
        [
            new(1, 32, 8, 32, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            new(2, 32, 9, 11, 40, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            new(1, 36, 14, 14, 33, 3, 3, 1, 1, 0, 0, 0, 0, 1, 1, 1),
            new(3, 48, 7, 7, 40, 3, 3, 1, 1, 2, 2, 2, 2, 1, 1, 1),
            new(1, 300, 6, 10, 40, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            new(1, 32, 3, 3, 32, 3, 3, 1, 1, 1, 0, 1, 0, 1, 1, 1),
            new(2, 40, 29, 37, 72, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            new(1, 32, 5, 100, 130, 3, 3, 1, 1, 0, 1, 2, 1, 1, 1, 1),
        ];
        Assert.False(ConvWinograd.Eligible(new(1, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1)));
        Assert.False(ConvWinograd.Eligible(new(1, 64, 14, 14, 64, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1)));
        foreach (ConvShape s in shapes)
        {
            Assert.True(ConvWinograd.Eligible(s));
            (float[] input, float[] weights) = Data(s, false);
            int len = s.N * s.Cout * s.OutH * s.OutW;
            var dbl = new double[len];
            var scale = new double[len];
            Reference(s, input, weights, new float[len], true, dbl, scale);
            int k = s.Cin * 9;
            var bias = new float[s.Cout];
            for (int i = 0; i < bias.Length; i++)
            {
                bias[i] = (i % 7) - 3f;
            }

            foreach (KernelPath path in Paths())
            {
                var packed = new float[ConvPacked.PackedWeightsLength(path, s)];
                ConvPacked.PackWeights(path, s, weights, packed);
                var one = new float[len];
                Conv.RunPacked(path, s, input, packed, one, 1);
                var many = new float[len];
                Conv.RunPacked(path, s, input, packed, many, 7);
                Assert.True(KernelTestData.SameBits(one, many), $"{path} thread dependence on {s}");
                for (int i = 0; i < len; i++)
                {
                    double tol = (16 * k * Math.Pow(2, -24) * scale[i]) + 1e-30;
                    Assert.True(Math.Abs(one[i] - dbl[i]) <= tol, $"{path} {s} at {i}: {one[i]} vs {dbl[i]}");
                }

                // Bias and Relu go through the epilogue on each chunk's rows.
                var act = new float[len];
                Conv.RunPacked(path, s, input, packed, act, 3, bias, new ReluActivation());
                int plane = s.OutH * s.OutW;
                for (int i = 0; i < len; i++)
                {
                    float expect = one[i] + bias[(i / plane) % s.Cout];
                    expect = expect <= 0f ? 0f : expect;
                    Assert.Equal(expect, act[i]);
                }

                // The skip connection is added after the bias and before the activation.
                var res = new float[len];
                for (int i = 0; i < len; i++)
                {
                    res[i] = ((i * 37) % 11) - 5f;
                }

                var fused = new float[len];
                Conv.RunPacked(path, s, input, packed, fused, 3, bias, new ReluActivation(), res);
                for (int i = 0; i < len; i++)
                {
                    float expect = (one[i] + bias[(i / plane) % s.Cout]) + res[i];
                    expect = expect <= 0f ? 0f : expect;
                    Assert.Equal(expect, fused[i]);
                }
            }
        }
    }
}
