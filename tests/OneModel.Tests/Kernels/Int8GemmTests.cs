using RecursiveAI.OneModel.Kernels;

namespace RecursiveAI.OneModel.Tests.Kernels;

/// <summary>The integer matrix product against a naive 64-bit reference, on every kernel path this machine has.</summary>
public class Int8GemmTests
{
    private static IEnumerable<KernelPath> Paths => KernelPaths.All.Where(KernelPaths.IsSupported);

    [Fact]
    public void MatchesNaiveReferenceOnEveryPath()
    {
        var rng = new Random(12345);
        int[] ms = [1, 2, 5, 6, 7, 12, 13, 47, 48, 49, 97];
        int[] ns = [1, 3, 4, 15, 16, 17, 31, 32, 33, 63, 64, 65, 130];
        int[] ks = [0, 1, 2, 3, 4, 5, 15, 16, 17, 255, 513, 1030];
        int cases = 0;
        foreach (KernelPath path in Paths)
        {
            Int8Format f = Int8Format.For(path);
            for (int t = 0; t < 120; t++)
            {
                int m = ms[rng.Next(ms.Length)], n = ns[rng.Next(ns.Length)], k = ks[rng.Next(ks.Length)];
                bool aU8 = rng.Next(2) == 0, bU8 = rng.Next(2) == 0, packA = rng.Next(2) == 0;
                bool perRow = rng.Next(2) == 0, perCol = rng.Next(2) == 0, toInt = rng.Next(3) == 0, toFloat = !toInt && rng.Next(3) == 0;
                int zaS = rng.Next(3) == 0 ? 0 : 7;
                int threads = rng.Next(2) == 0 ? 1 : 5;
                byte[] a = new byte[m * k], b = new byte[k * n];
                // Extremes are the likely overflow and sign-extension traps: mix them in.
                for (int i = 0; i < a.Length; i++)
                {
                    a[i] = rng.Next(6) == 0 ? (byte)(rng.Next(2) == 0 ? 0 : 255) : (byte)rng.Next(256);
                }

                for (int i = 0; i < b.Length; i++)
                {
                    b[i] = rng.Next(6) == 0 ? (byte)(rng.Next(2) == 0 ? 0x80 : 0x7F) : (byte)rng.Next(256);
                }

                int[] za = new int[m], zb = new int[n];
                for (int i = 0; i < m; i++)
                {
                    za[i] = perRow ? (rng.Next(4) == 0 ? 0 : rng.Next(-128, 128)) : zaS;
                }

                for (int j = 0; j < n; j++)
                {
                    zb[j] = perCol ? rng.Next(-128, 128) : -3;
                }

                int[] bias = rng.Next(2) == 0 ? new int[m].Select(_ => rng.Next(-1000, 1000)).ToArray() : [];
                float[] mult = rng.Next(2) == 0 ? new float[n].Select(_ => rng.NextSingle() * 0.01f).ToArray() : [];
                float mscalar = 0.0007f;
                bool unsignedOut = rng.Next(2) == 0;
                int yz = rng.Next(0, 100);

                byte[] packed = new byte[f.PackedBBytes(k, n)];
                int[] sums = new int[n];
                Int8Matmul.PackB(in f, b, n, bU8, k, n, packed, sums, threads);
                Int8Packed? pa = packA ? Int8Packed.ForA(in f, a, k, aU8, m, k) : null;
                int[] gotI = new int[m * n];
                byte[] gotQ = new byte[m * n];
                float[] gotF = new float[m * n];
                var call = new Int8Call
                {
                    Format = f, M = m, N = n, K = k, ARaw = a, Lda = k, AU8 = aU8, APacked = pa, BPacked = packed, BSums = sums,
                    ZaScalar = zaS, ZaVec = perRow ? za : default, ZbScalar = -3, ZbVec = perCol ? zb : default, Ldc = n, Bias = bias,
                    Mult = mscalar, MultVec = mult, MultPerRow = false, QuantUnsigned = unsignedOut, QuantZero = yz, Threads = threads,
                };
                if (toInt)
                {
                    call.OutInt32 = gotI;
                }
                else if (toFloat)
                {
                    call.OutFloat = gotF;
                }
                else
                {
                    call.OutQuant = gotQ;
                }

                Int8Matmul.Run(ref call);

                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        long acc = 0;
                        for (int p = 0; p < k; p++)
                        {
                            long av = (aU8 ? a[i * k + p] : (sbyte)a[i * k + p]) - (long)(za[i] + (aU8 ? 128 : 0));
                            long bv = (bU8 ? b[p * n + j] : (sbyte)b[p * n + j]) - (long)(zb[j] + (bU8 ? 128 : 0));
                            acc += av * bv;
                        }

                        acc += bias.Length > 0 ? bias[i] : 0;
                        if (toInt)
                        {
                            Assert.True((int)acc == gotI[i * n + j], $"{path} m{m} n{n} k{k} [{i},{j}]: {gotI[i * n + j]} vs {acc}");
                        }
                        else if (toFloat)
                        {
                            float mu = mult.Length > 0 ? mult[j] : mscalar;
                            float wantF = (float)(int)acc * mu;
                            Assert.True(BitConverter.SingleToInt32Bits(wantF) == BitConverter.SingleToInt32Bits(gotF[i * n + j]), $"{path} float m{m} n{n} k{k} [{i},{j}]: {gotF[i * n + j]} vs {wantF}");
                        }
                        else
                        {
                            float mu = mult.Length > 0 ? mult[j] : mscalar;
                            float fv = MathF.Round((float)(int)acc * mu) + yz;
                            int lo = unsignedOut ? 0 : -128, hi = unsignedOut ? 255 : 127;
                            int want = fv < lo ? lo : fv > hi ? hi : (int)fv;
                            Assert.True(gotQ[i * n + j] == (byte)want, $"{path} requant m{m} n{n} k{k} [{i},{j}]: {gotQ[i * n + j]} vs {(byte)want}");
                        }
                    }
                }

                cases++;
            }
        }

        Assert.True(cases > 0);
    }

    [Fact]
    public void RequantizationMatchesTheFormulaForExtremeScales()
    {
        // Large, tiny, negative, infinite and NaN multipliers: the vector paths clamp before rounding when the scale is finite and
        // must still give clamp(round_half_even(sum * mult) + zero) for every sum, and the smallest value for a NaN scale.
        float[] mults = [0.37f, 1f, 123456.7f, 1e-7f, -0.5f, -3.3e5f, 0f, 3e38f, float.PositiveInfinity, float.NegativeInfinity, float.NaN, 0.5f, 0.25f];
        var rng = new Random(99);
        foreach (KernelPath path in Paths)
        {
            Int8Format f = Int8Format.For(path);
            foreach (float mu in mults)
            {
                foreach (bool unsignedOut in new[] { false, true })
                {
                    const int m = 7, n = 70, k = 33;
                    byte[] a = new byte[m * k], b = new byte[k * n];
                    rng.NextBytes(a);
                    rng.NextBytes(b);
                    byte[] packed = new byte[f.PackedBBytes(k, n)];
                    int[] sums = new int[n];
                    Int8Matmul.PackB(in f, b, n, true, k, n, packed, sums, 1);
                    byte[] got = new byte[m * n];
                    int yz = unsignedOut ? 17 : -9;
                    var call = new Int8Call
                    {
                        Format = f, M = m, N = n, K = k, ARaw = a, Lda = k, AU8 = false, BPacked = packed, BSums = sums,
                        ZaScalar = 0, ZbScalar = -3, Ldc = n, Mult = mu, QuantUnsigned = unsignedOut, QuantZero = yz, Threads = 1, OutQuant = got,
                    };
                    Int8Matmul.Run(ref call);
                    int lo = unsignedOut ? 0 : -128, hi = unsignedOut ? 255 : 127;
                    for (int i = 0; i < m; i++)
                    {
                        for (int j = 0; j < n; j++)
                        {
                            long acc = 0;
                            for (int p = 0; p < k; p++)
                            {
                                acc += (long)(sbyte)a[i * k + p] * ((b[p * n + j] - 128) + 3);
                            }

                            float fv = MathF.Round((float)(int)acc * mu) + yz;
                            int want = float.IsNaN(fv) ? lo : fv < lo ? lo : fv > hi ? hi : (int)fv;
                            Assert.True(got[i * n + j] == (byte)want, $"{path} mult {mu} unsigned={unsignedOut} [{i},{j}] sum {acc}: {got[i * n + j]} vs {(byte)want}");
                        }
                    }
                }
            }
        }
    }

    [Fact]
    public void ChunkSizeDependsOnThreadsButTheProductDoesNot()
    {
        var rng = new Random(77);
        foreach (KernelPath path in Paths)
        {
            Int8Format f = Int8Format.For(path);
            // Large enough that several threads are worth using, so the chunk size is the model's choice rather than the default.
            int m = 130, n = 301, k = 700;
            byte[] a = new byte[m * k], b = new byte[k * n];
            rng.NextBytes(a);
            rng.NextBytes(b);
            byte[] packed = new byte[f.PackedBBytes(k, n)];
            int[] sums = new int[n];
            Int8Matmul.PackB(in f, b, n, true, k, n, packed, sums, 1);
            int[]? first = null;
            foreach (int threads in new[] { 1, 3, 8, 32 })
            {
                int[] got = new int[m * n];
                var call = new Int8Call
                {
                    Format = f, M = m, N = n, K = k, ARaw = a, Lda = k, AU8 = true, BPacked = packed, BSums = sums,
                    ZaScalar = 5, ZbScalar = 3, Ldc = n, Mult = 1f, OutInt32 = got, Threads = threads,
                };
                Int8Matmul.Run(ref call);
                first ??= got;
                Assert.True(first.AsSpan().SequenceEqual(got), $"{path}: {threads} threads differ from 1");
            }

            // 128 x 768 x 768 on 32 threads: the default would be 36 chunks, two rounds with the second nearly empty.
            Int8Format t = Int8Gemm.ChooseTile(in f, 128, 768, 768, 32);
            Assert.True(t.Mc % f.Mr == 0 && t.Nc % f.Nr == 0 && t.Mc <= f.Mc && t.Nc <= f.Nc && t.Mc >= f.Mr && t.Nc >= f.Nr);
            long chunks = (long)((128 + t.Mc - 1) / t.Mc) * ((768 + t.Nc - 1) / t.Nc);
            Assert.True(chunks > 36, $"{path}: {chunks} chunks of {t.Mc} x {t.Nc}");
            // One thread keeps the default: the product only consults ChooseTile when more than one thread is usable, so the
            // model's answer for one thread (which may differ per flavour) is never used.
        }
    }
}
