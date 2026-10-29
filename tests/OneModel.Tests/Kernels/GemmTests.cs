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
}
