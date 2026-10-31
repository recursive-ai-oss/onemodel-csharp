using RecursiveAI.OneModel.Kernels;

namespace RecursiveAI.OneModel.Tests.Kernels;

/// <summary>The integer convolution (patch gather, pack, product) against a plain loop, over geometries, groups, zero points and types.</summary>
public class Int8ConvTests
{
    [Fact]
    public void MatchesPlainLoop()
    {
        var rng = new Random(4242);
        int[] sizes = [1, 2, 3, 4, 5, 7, 8, 14, 15, 16, 17, 31, 33];
        int cases = 0;
        for (int t = 0; t < 400; t++)
        {
            // The first few cases are large enough to be split over several threads and several row tiles.
            bool big = t < 4;
            int groups = !big && rng.Next(3) == 0 ? rng.Next(1, 4) : 1;
            int cinG = big ? 16 : rng.Next(1, 6), coutG = big ? 70 + (40 * t) : rng.Next(1, 20);
            int cin = cinG * groups, cout = coutG * groups, n = rng.Next(1, 3);
            int kh = big ? 3 : rng.Next(1, 5), kw = big ? 3 : rng.Next(1, 5);
            int sy = big ? 1 + (t & 1) : rng.Next(1, 3), sx = big ? 1 + (t & 1) : rng.Next(5) == 0 ? 2 : 1;
            int dy = big ? 1 : rng.Next(1, 3), dx = big ? 1 : rng.Next(1, 3);
            int pt = big ? 1 : rng.Next(0, 3), pl = big ? 1 : rng.Next(0, 3), pb = big ? 1 : rng.Next(0, 3), pr = big ? 1 : rng.Next(0, 3);
            int h = big ? 24 : sizes[rng.Next(sizes.Length)], w = big ? 25 : sizes[rng.Next(sizes.Length)];
            var s = new ConvShape(n, cin, h, w, cout, kh, kw, sy, sx, pt, pl, pb, pr, dy, dx, groups);
            if (h + pt + pb - (dy * (kh - 1)) - 1 < 0 || w + pl + pr - (dx * (kw - 1)) - 1 < 0)
            {
                continue;
            }

            int oh = s.OutH, ow = s.OutW;
            bool xU8 = rng.Next(2) == 0, wU8 = rng.Next(2) == 0, toInt = rng.Next(2) == 0, qU = rng.Next(2) == 0;
            int zeroMode = rng.Next(3); // 0: weight zero points all zero, 1: one value, 2: one per output channel
            byte[] x = new byte[n * cin * h * w], wt = new byte[cout * cinG * kh * kw];
            rng.NextBytes(x);
            rng.NextBytes(wt);
            for (int i = 0; i < x.Length; i += 5)
            {
                x[i] = (byte)(rng.Next(2) == 0 ? 0 : 255);
            }

            int xz = rng.Next(-128, 128);
            int wzs = zeroMode == 1 ? rng.Next(-128, 128) : 0;
            int[] wzv = new int[cout].Select(_ => rng.Next(-128, 128)).ToArray();
            int[] bias = rng.Next(2) == 0 ? new int[cout].Select(_ => rng.Next(-2000, 2000)).ToArray() : [];
            float mult = 0.0005f;
            int yz = rng.Next(0, 90);
            int[] gotI = new int[toInt ? n * cout * oh * ow : 0];
            byte[] gotQ = new byte[toInt ? 0 : n * cout * oh * ow];
            foreach (KernelPath path in KernelPaths.All.Where(KernelPaths.IsSupported))
            {
                var call = new Int8ConvCall
                {
                    Format = Int8Format.For(path), S = s, X = x, XU8 = xU8, XZeroRaw = xU8 ? xz + 128 : xz, XZeroShifted = xz,
                    WRaw = wt, WU8 = wU8, WZeroScalar = wzs, WZeroVec = zeroMode == 2 ? wzv : default, OutInt32 = gotI, OutQuant = gotQ,
                    QuantUnsigned = qU, QuantZero = yz, Bias = bias, Mult = mult, Threads = big ? 4 : rng.Next(2) == 0 ? 1 : 3,
                };
                Int8Conv.Run(ref call);
                for (int img = 0; img < n; img++)
                {
                    for (int co = 0; co < cout; co++)
                    {
                        int g = co / coutG;
                        int za = zeroMode == 2 ? wzv[co] : wzs;
                        for (int oy = 0; oy < oh; oy++)
                        {
                            for (int ox = 0; ox < ow; ox++)
                            {
                                long acc = bias.Length > 0 ? bias[co] : 0;
                                for (int ci = 0; ci < cinG; ci++)
                                {
                                    for (int ky = 0; ky < kh; ky++)
                                    {
                                        for (int kx = 0; kx < kw; kx++)
                                        {
                                            int iy = (oy * sy) - pt + (ky * dy), ix = (ox * sx) - pl + (kx * dx);
                                            int xv = xz;
                                            if (iy >= 0 && iy < h && ix >= 0 && ix < w)
                                            {
                                                byte xb = x[((((img * cin) + (g * cinG) + ci) * h + iy) * w) + ix];
                                                xv = xU8 ? xb - 128 : (sbyte)xb;
                                            }

                                            byte wb = wt[((((co * cinG) + ci) * kh + ky) * kw) + kx];
                                            acc += (long)(xv - xz) * ((wU8 ? wb - 128 : (sbyte)wb) - za);
                                        }
                                    }
                                }

                                int o = (((img * cout) + co) * oh + oy) * ow + ox;
                                if (toInt)
                                {
                                    Assert.True((int)acc == gotI[o], $"{path} {s} [{img},{co},{oy},{ox}]: {gotI[o]} vs {acc}");
                                }
                                else
                                {
                                    int want = Int8Gemm.Requant((int)acc, mult, yz, qU ? 0 : -128, qU ? 255 : 127);
                                    Assert.True((byte)want == gotQ[o], $"{path} {s} [{img},{co},{oy},{ox}]: {gotQ[o]} vs {want}");
                                }
                            }
                        }
                    }
                }

                cases++;
            }
        }

        Assert.True(cases > 100);
    }
}
