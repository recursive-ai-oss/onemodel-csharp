using RecursiveAI.OneModel.Kernels;

namespace RecursiveAI.OneModel.Tests.Kernels;

public class GemmTests
{
    private static KernelPath[] Paths() => KernelPaths.All.Where(KernelPaths.IsSupported).ToArray();

    [Fact]
    public void ExactMatchesCanonicalOrderOnEveryPath()
    {
        foreach (bool special in new[] { false, true })
        {
            foreach (KernelTestData.GemmCase g in KernelTestData.SweepCases(70, 150, special))
            {
                (float[] a, float[] b, float[] expected) = KernelTestData.Materialize(g);
                KernelTestData.ReferenceExact(g.M, g.N, g.K, a.AsSpan(g.OffA), g.Lda, b.AsSpan(g.OffB), g.Ldb, expected.AsSpan(g.OffC), g.Ldc);
                foreach (KernelPath path in Paths())
                {
                    float[] got = KernelTestData.RunKernel(g, KernelMode.Exact, path, 1);
                    Assert.True(KernelTestData.SameBits(expected, got), $"{path} differs on {g}");
                }
            }
        }
    }

    [Fact]
    public void ExactIsIndependentOfThreadCount()
    {
        foreach (KernelPath path in Paths())
        {
            string one = KernelTestData.SweepHash(KernelMode.Exact, path, 1, 40, 40, true);
            Assert.Equal(one, KernelTestData.SweepHash(KernelMode.Exact, path, 8, 40, 40, true));
        }
    }

    [Fact]
    public void FastIsIndependentOfThreadCountAndRepeatable()
    {
        foreach (KernelPath path in Paths())
        {
            string one = KernelTestData.SweepHash(KernelMode.Fast, path, 1, 40, 40, true);
            Assert.Equal(one, KernelTestData.SweepHash(KernelMode.Fast, path, 8, 40, 40, true));
            Assert.Equal(one, KernelTestData.SweepHash(KernelMode.Fast, path, 3, 40, 40, true));
        }
    }

    [Fact]
    public void FastIsWithinTheDocumentedToleranceOfADoubleReference()
    {
        // Bound: |c - exact| <= k * 2^-24 * sum|a_p b_p| (the standard bound for a sequentially summed dot product with k terms, unit
        // roundoff 2^-24), plus a small absolute floor for the subnormal range.
        foreach (KernelTestData.GemmCase g in KernelTestData.SweepCases(70, 100, false))
        {
            (float[] a, float[] b, _) = KernelTestData.Materialize(g);
            var reference = new double[g.M * g.N];
            var scale = new double[g.M * g.N];
            KernelTestData.ReferenceDouble(g.M, g.N, g.K, a.AsSpan(g.OffA), g.Lda, b.AsSpan(g.OffB), g.Ldb, reference, scale);
            foreach (KernelPath path in Paths())
            {
                float[] got = KernelTestData.RunKernel(g, KernelMode.Fast, path, 2);
                for (int i = 0; i < g.M; i++)
                {
                    for (int j = 0; j < g.N; j++)
                    {
                        double tol = (g.K * Math.Pow(2, -24) * scale[(i * g.N) + j]) + 1e-30;
                        double err = Math.Abs(got[g.OffC + (i * g.Ldc) + j] - reference[(i * g.N) + j]);
                        Assert.True(err <= tol, $"{path} {g} at ({i},{j}): error {err} > {tol}");
                    }
                }
            }
        }
    }

    [Fact]
    public void OutputOutsideTheMatrixIsUntouched()
    {
        foreach (KernelPath path in Paths())
        {
            var g = new KernelTestData.GemmCase(13, 21, 9, 12, 25, 27, 3, 5, 7, 99, false);
            float[] c = KernelTestData.RunKernel(g, KernelMode.Exact, path, 1);
            for (int idx = 0; idx < c.Length; idx++)
            {
                int rel = idx - g.OffC;
                bool inside = rel >= 0 && rel / g.Ldc < g.M && rel % g.Ldc < g.N;
                if (!inside)
                {
                    Assert.Equal(0xDEADBEEFu, BitConverter.SingleToUInt32Bits(c[idx]));
                }
            }
        }
    }

    [Fact]
    public void ZeroKGivesPositiveZero()
    {
        var g = new KernelTestData.GemmCase(3, 4, 0, 1, 4, 4, 0, 0, 0, 1, false);
        foreach (KernelMode mode in new[] { KernelMode.Fast, KernelMode.Exact })
        {
            float[] c = KernelTestData.RunKernel(g, mode, KernelPath.Scalar, 1);
            for (int i = 0; i < 12; i++)
            {
                Assert.Equal(0u, BitConverter.SingleToUInt32Bits(c[i]));
            }
        }
    }
    [Fact]
    public void SingleRowProductGivesTheTileProductsBits()
    {
        // One row takes the matrix-vector kernels; two identical rows take the tile kernels. Ragged n, more than one k block, and
        // packed and unpacked B must all agree, at any thread count.
        var rng = new KernelTestData.Rng(321);
        foreach (KernelPath path in Paths())
        {
            foreach ((int n, int k) in new[] { (1, 5), (7, 300), (33, 257), (128, 64), (129, 513), (1000, 600), (31, 1) })
            {
                var a = new float[2 * k];
                var b = new float[k * n];
                KernelTestData.Fill(ref rng, a.AsSpan(0, k), false);
                a.AsSpan(0, k).CopyTo(a.AsSpan(k, k));
                KernelTestData.Fill(ref rng, b, false);
                var two = new float[2 * n];
                Gemm.Run(KernelMode.Fast, path, 2, n, k, a, k, b, n, two, n, 1);
                var packed = new float[Gemm.PackedBLength(path, n, k)];
                Gemm.PackB(path, b, n, n, k, packed);
                foreach (int threads in new[] { 1, 6 })
                {
                    var one = new float[n];
                    Gemm.Run(KernelMode.Fast, path, 1, n, k, a, k, b, n, one, n, threads);
                    Assert.True(KernelTestData.SameBits(two.AsSpan(0, n).ToArray(), one), $"{path} unpacked n={n} k={k} threads={threads}");
                    var onePacked = new float[n];
                    Gemm.RunPacked(KernelMode.Fast, path, 1, n, k, a, k, packed, onePacked, n, threads);
                    Assert.True(KernelTestData.SameBits(two.AsSpan(0, n).ToArray(), onePacked), $"{path} packed n={n} k={k} threads={threads}");
                }
            }
        }
    }

    [Fact]
    public void SequenceSizedProductsAreExactAndIndependentOfThreads()
    {
        // A few hundred rows by several column chunks takes the shared packing of A and the finer column chunks.
        foreach (KernelPath path in Paths())
        {
            foreach ((int m, int n, int k) in new[] { (100, 300, 300), (128, 768, 600), (130, 1500, 257), (7, 700, 520) })
            {
                var g = new KernelTestData.GemmCase(m, n, k, k, n, n, 0, 0, 0, 5, false);
                (float[] a, float[] b, float[] expected) = KernelTestData.Materialize(g);
                KernelTestData.ReferenceExact(m, n, k, a, k, b, n, expected, n);
                foreach (int threads in new[] { 1, 8 })
                {
                    Assert.True(KernelTestData.SameBits(expected, KernelTestData.RunKernel(g, KernelMode.Exact, path, threads)), $"{path} exact {m}x{n}x{k} threads={threads}");
                }

                Assert.True(KernelTestData.SameBits(KernelTestData.RunKernel(g, KernelMode.Fast, path, 1), KernelTestData.RunKernel(g, KernelMode.Fast, path, 8)), $"{path} fast {m}x{n}x{k}");
            }
        }
    }
}
