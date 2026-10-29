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
                bool perRow = rng.Next(2) == 0, perCol = rng.Next(2) == 0, toInt = rng.Next(2) == 0;
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
                    za[i] = perRow ? rng.Next(-128, 128) : 7;
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
                var call = new Int8Call
                {
                    Format = f, M = m, N = n, K = k, ARaw = a, Lda = k, AU8 = aU8, APacked = pa, BPacked = packed, BSums = sums,
                    ZaScalar = 7, ZaVec = perRow ? za : default, ZbScalar = -3, ZbVec = perCol ? zb : default, Ldc = n, Bias = bias,
                    Mult = mscalar, MultVec = mult, MultPerRow = false, QuantUnsigned = unsignedOut, QuantZero = yz, Threads = threads,
                };
                if (toInt)
                {
                    call.OutInt32 = gotI;
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
}
