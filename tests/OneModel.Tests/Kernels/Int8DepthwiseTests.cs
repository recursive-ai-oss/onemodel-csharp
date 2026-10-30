using RecursiveAI.OneModel.Kernels;

namespace RecursiveAI.OneModel.Tests.Kernels;

/// <summary>The direct int8 depthwise convolution against a plain loop, over strides, dilations, paddings and both element types.</summary>
public class Int8DepthwiseTests
{
    [Fact]
    public void MatchesPlainLoop()
    {
        var rng = new Random(777);
        int[] sizes = [1, 2, 3, 7, 8, 15, 16, 17, 31, 40];
        for (int t = 0; t < 300; t++)
        {
            int c = rng.Next(1, 6), n = rng.Next(1, 3);
            int kh = rng.Next(1, 6), kw = rng.Next(1, 6);
            int sy = rng.Next(1, 3), sx = rng.Next(1, 4);
            int dy = rng.Next(1, 3), dx = rng.Next(1, 3);
            int pt = rng.Next(0, 4), pl = rng.Next(0, 4), pb = rng.Next(0, 4), pr = rng.Next(0, 4);
            int h = sizes[rng.Next(sizes.Length)], w = sizes[rng.Next(sizes.Length)];
            var s = new ConvShape(n, c, h, w, c, kh, kw, sy, sx, pt, pl, pb, pr, dy, dx, c);
            if (h + pt + pb - (dy * (kh - 1)) - 1 < 0 || w + pl + pr - (dx * (kw - 1)) - 1 < 0)
            {
                continue;
            }

            int oh = s.OutH, ow = s.OutW;
            bool xU8 = rng.Next(2) == 0, wU8 = rng.Next(2) == 0, toInt = rng.Next(2) == 0, qU = rng.Next(2) == 0, perCh = rng.Next(2) == 0;
            byte[] x = new byte[n * c * h * w], wt = new byte[c * kh * kw];
            rng.NextBytes(x);
            rng.NextBytes(wt);
            for (int i = 0; i < x.Length; i += 5)
            {
                x[i] = (byte)(rng.Next(2) == 0 ? 0 : 255);
            }

            int xz = rng.Next(-128, 128), wzs = rng.Next(-128, 128);
            int[] wzv = new int[c].Select(_ => rng.Next(-128, 128)).ToArray();
            int[] bias = rng.Next(2) == 0 ? new int[c].Select(_ => rng.Next(-2000, 2000)).ToArray() : [];
            float mult = 0.003f;
            int yz = rng.Next(0, 90);
            int[] gotI = new int[toInt ? n * c * oh * ow : 0];
            byte[] gotQ = new byte[toInt ? 0 : n * c * oh * ow];
            foreach (KernelPath path in KernelPaths.All.Where(KernelPaths.IsSupported))
            {
                var call = new Int8ConvCall
                {
                    Format = Int8Format.For(path), S = s, X = x, XU8 = xU8, XZeroRaw = xU8 ? xz + 128 : xz, XZeroShifted = xz,
                    WRaw = wt, WU8 = wU8, WZeroScalar = wzs, WZeroVec = perCh ? wzv : default, OutInt32 = gotI, OutQuant = gotQ,
                    QuantUnsigned = qU, QuantZero = yz, Bias = bias, Mult = mult, Threads = rng.Next(2) == 0 ? 1 : 3,
                };
                Int8Conv.Run(ref call);
                for (int p = 0; p < n * c; p++)
                {
                    int ch = p % c;
                    int za = perCh ? wzv[ch] : wzs;
                    for (int oy = 0; oy < oh; oy++)
                    {
                        for (int ox = 0; ox < ow; ox++)
                        {
                            long acc = bias.Length > 0 ? bias[ch] : 0;
                            for (int ky = 0; ky < kh; ky++)
                            {
                                for (int kx = 0; kx < kw; kx++)
                                {
                                    int iy = (oy * sy) - pt + (ky * dy), ix = (ox * sx) - pl + (kx * dx);
                                    if (iy < 0 || iy >= h || ix < 0 || ix >= w)
                                    {
                                        continue;
                                    }

                                    byte xb = x[(((p * h) + iy) * w) + ix];
                                    byte wb = wt[(((ch * kh) + ky) * kw) + kx];
                                    acc += (long)((xU8 ? xb - 128 : (sbyte)xb) - xz) * ((wU8 ? wb - 128 : (sbyte)wb) - za);
                                }
                            }

                            int o = (((p * oh) + oy) * ow) + ox;
                            if (toInt)
                            {
                                Assert.Equal((int)acc, gotI[o]);
                            }
                            else
                            {
                                int want = Int8Gemm.Requant((int)acc, mult, yz, qU ? 0 : -128, qU ? 255 : 127);
                                Assert.Equal((byte)want, gotQ[o]);
                            }
                        }
                    }
                }
            }
        }
    }
}
