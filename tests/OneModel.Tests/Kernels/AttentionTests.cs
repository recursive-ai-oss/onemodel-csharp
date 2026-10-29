using System;
using System.Collections.Generic;
using RecursiveAI.OneModel.Kernels;

namespace RecursiveAI.OneModel.Tests.Kernels;

public class AttentionTests
{
    private static readonly VecIsa[] AllIsas = [VecIsa.Scalar, VecIsa.V128, VecIsa.V256, VecIsa.V512];

    public sealed record AttnCase(int Batch, int Heads, int QLen, int KLen, int Dim, bool Mask);

    public static IEnumerable<TheoryDataRow<AttnCase>> Cases() =>
    [
        new(new AttnCase(1, 1, 3, 10, 2, false)),
        new(new AttnCase(1, 2, 5, 7, 32, false)),
        new(new AttnCase(1, 1, 1, 1, 32, false)),
        new(new AttnCase(1, 1, 1, 130, 64, true)),
        new(new AttnCase(2, 3, 37, 150, 32, true)),
        new(new AttnCase(1, 2, 64, 64, 64, false)),
        new(new AttnCase(1, 1, 33, 65, 16, true)),
        new(new AttnCase(1, 1, 128, 128, 64, true)),
        new(new AttnCase(1, 2, 50, 200, 40, false)),
    ];

    private static float[] RandomData(int n, int seed, double scale)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++)
        {
            // Box-Muller; deterministic from the seed.
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            a[i] = (float)(scale * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        return a;
    }

    private sealed class Data
    {
        public AttentionShape Shape;
        public float[] Q = [];
        public float[] K = [];
        public float[] V = [];
        public float[] Mask = [];
    }

    private static Data Make(AttnCase c, int seed)
    {
        AttentionShape s = AttentionShape.Contiguous(c.Batch, c.Heads, c.QLen, c.KLen, c.Dim, 1f / MathF.Sqrt(c.Dim));
        var d = new Data
        {
            Shape = s,
            Q = RandomData(c.Batch * c.Heads * c.QLen * c.Dim, seed, 1.5),
            K = RandomData(c.Batch * c.Heads * c.KLen * c.Dim, seed + 1, 1.5),
            V = RandomData(c.Batch * c.Heads * c.KLen * c.Dim, seed + 2, 1.0),
        };
        if (c.Mask)
        {
            d.Mask = new float[c.Batch * c.QLen * c.KLen];
            var rng = new Random(seed + 3);
            for (int i = 0; i < d.Mask.Length; i++)
            {
                // About a third of the positions are masked out; the first key of every row stays visible.
                int col = i % c.KLen;
                d.Mask[i] = col != 0 && rng.Next(3) == 0 ? float.NegativeInfinity : (float)((rng.NextDouble() * 2) - 1);
            }
        }

        return d;
    }

    private static float[] Reference(Data d)
    {
        AttentionShape s = d.Shape;
        var o = new float[s.Batch * s.Heads * s.QLen * s.HeadDim];
        var p = new double[s.KLen];
        for (int b = 0; b < s.Batch; b++)
        {
            for (int h = 0; h < s.Heads; h++)
            {
                for (int i = 0; i < s.QLen; i++)
                {
                    double max = double.NegativeInfinity;
                    for (int j = 0; j < s.KLen; j++)
                    {
                        double dot = 0;
                        for (int x = 0; x < s.HeadDim; x++)
                        {
                            dot += (double)d.Q[(int)((b * s.QBatchStride) + (h * s.QHeadStride) + (i * s.QRowStride) + x)]
                                 * d.K[(int)((b * s.KBatchStride) + (h * s.KHeadStride) + (j * s.KRowStride) + x)];
                        }

                        p[j] = (dot * s.Scale) + (d.Mask.Length > 0 ? d.Mask[(int)((b * s.MaskBatchStride) + (i * s.MaskRowStride) + j)] : 0);
                        max = Math.Max(max, p[j]);
                    }

                    double sum = 0;
                    for (int j = 0; j < s.KLen; j++)
                    {
                        p[j] = Math.Exp(p[j] - max);
                        sum += p[j];
                    }

                    for (int x = 0; x < s.HeadDim; x++)
                    {
                        double acc = 0;
                        for (int j = 0; j < s.KLen; j++)
                        {
                            acc += p[j] * d.V[(int)((b * s.VBatchStride) + (h * s.VHeadStride) + (j * s.VRowStride) + x)];
                        }

                        o[(int)((b * s.OBatchStride) + (h * s.OHeadStride) + (i * s.ORowStride) + x)] = (float)(acc / sum);
                    }
                }
            }
        }

        return o;
    }

    /// <summary>Runs all work items split into chunks of <paramref name="itemsPerTask"/> on up to <paramref name="threads"/> threads.</summary>
    private static float[] Run(Data d, bool exact, VecIsa isa, int threads, int itemsPerTask)
    {
        AttentionShape s = d.Shape;
        var o = new float[s.Batch * s.Heads * s.QLen * s.HeadDim];
        int items = Attention.ItemCount(s);
        itemsPerTask = Math.Max(1, Math.Min(itemsPerTask, items));
        int tasks = (items + itemsPerTask - 1) / itemsPerTask;
        var po = new ParallelOptions { MaxDegreeOfParallelism = threads };
        Parallel.For(0, tasks, po, () => new float[Attention.ScratchLength(s.HeadDim)], (t, _, scratch) =>
        {
            int a = t * itemsPerTask;
            int b = Math.Min(items, a + itemsPerTask);
            Attention.Compute(exact, isa, s, d.Q, d.K, d.V, d.Mask, o, a, b, scratch);
            return scratch;
        }, _ => { });
        return o;
    }

    private static double MaxErr(float[] got, float[] want)
    {
        double max = 0;
        for (int i = 0; i < got.Length; i++)
        {
            max = Math.Max(max, Math.Abs(got[i] - (double)want[i]));
        }

        return max;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void FastAndExactMatchTheDoubleReference(AttnCase c)
    {
        Data d = Make(c, 17);
        float[] want = Reference(d);
        foreach (VecIsa isa in AllIsas)
        {
            Assert.True(MaxErr(Run(d, false, isa, 1, 1), want) < 4e-6, $"fast {isa}");
            Assert.True(MaxErr(Run(d, true, isa, 1, 1), want) < 4e-6, $"exact {isa}");
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void ExactBitsAreEqualAcrossLaneWidthsAndPartitions(AttnCase c)
    {
        Data d = Make(c, 23);
        float[] want = Run(d, true, VecIsa.Scalar, 1, int.MaxValue);
        foreach (VecIsa isa in AllIsas)
        {
            AssertSameBits(want, Run(d, true, isa, 1, int.MaxValue), $"{isa} 1 thread");
            AssertSameBits(want, Run(d, true, isa, 8, 1), $"{isa} 8 threads, 1 item per task");
            AssertSameBits(want, Run(d, true, isa, 3, 2), $"{isa} 3 threads, 2 items per task");
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void FastBitsDoNotDependOnPartitionOrRepeat(AttnCase c)
    {
        Data d = Make(c, 29);
        foreach (VecIsa isa in AllIsas)
        {
            float[] want = Run(d, false, isa, 1, int.MaxValue);
            AssertSameBits(want, Run(d, false, isa, 1, int.MaxValue), $"{isa} repeat");
            AssertSameBits(want, Run(d, false, isa, 8, 1), $"{isa} 8 threads");
        }
    }

    private static void AssertSameBits(float[] want, float[] got, string what)
    {
        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < want.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(want[i]) != BitConverter.SingleToInt32Bits(got[i]))
            {
                Assert.Fail($"{what}: element {i}: want {want[i]:R} got {got[i]:R}");
            }
        }
    }

    [Fact]
    public void FullyMaskedAndNonFiniteRowsGiveTheCanonicalNaN()
    {
        var c = new AttnCase(1, 1, 4, 70, 32, true);
        Data d = Make(c, 5);
        for (int j = 0; j < 70; j++)
        {
            d.Mask[(1 * 70) + j] = float.NegativeInfinity; // row 1 fully masked
        }

        d.Q[(2 * 32) + 3] = float.NaN; // row 2 has a NaN query
        foreach (bool exact in new[] { true, false })
        {
            float[] want = Run(d, exact, VecIsa.Scalar, 1, 1);
            foreach (int r in new[] { 1, 2 })
            {
                Assert.All(want.AsSpan(r * 32, 32).ToArray(), x => Assert.Equal(0x7FC00000u, BitConverter.SingleToUInt32Bits(x)));
            }

            Assert.All(want.AsSpan(0, 32).ToArray(), x => Assert.True(float.IsFinite(x)));
            Assert.All(want.AsSpan(96, 32).ToArray(), x => Assert.True(float.IsFinite(x)));

            if (exact)
            {
                foreach (VecIsa isa in AllIsas)
                {
                    AssertSameBits(want, Run(d, true, isa, 1, 1), $"{isa}");
                }
            }
        }

        // A partly masked first block (all -inf there) followed by visible keys must not produce NaN.
        var c2 = new AttnCase(1, 1, 1, 130, 32, true);
        Data d2 = Make(c2, 6);
        for (int j = 0; j < 64; j++)
        {
            d2.Mask[j] = float.NegativeInfinity;
        }

        Assert.True(MaxErr(Run(d2, true, VecIsa.V256, 1, 1), Reference(d2)) < 4e-6);
        Assert.True(MaxErr(Run(d2, false, VecIsa.V256, 1, 1), Reference(d2)) < 4e-6);
    }

    [Fact]
    public void SizesAreValidated()
    {
        AttentionShape s = AttentionShape.Contiguous(1, 1, 4, 4, 32, 1f);
        var scratch = new float[Attention.ScratchLength(32)];
        Assert.Throws<ArgumentException>(() => Attention.Compute(true, VecIsa.Auto, s, new float[10], new float[128], new float[128], default, new float[128], 0, 1, scratch));
        Assert.Throws<ArgumentException>(() => Attention.Compute(true, VecIsa.Auto, s, new float[128], new float[128], new float[128], default, new float[128], 0, 1, new float[3]));
        AttentionShape odd = AttentionShape.Contiguous(1, 1, 4, 4, 31, 1f);
        Assert.Throws<ArgumentException>(() => Attention.Compute(true, VecIsa.Auto, odd, new float[124], new float[124], new float[124], default, new float[124], 0, 1, scratch));
        Assert.Throws<ArgumentOutOfRangeException>(() => Attention.Compute(true, VecIsa.Auto, s, new float[128], new float[128], new float[128], default, new float[128], 0, 2, scratch));
    }

    [Fact]
    public void SoftmaxMatchesTheReferenceAndIsBitEqualAcrossWidths()
    {
        foreach (int cols in new[] { 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 128, 250, 512, 1500 })
        {
            const int rows = 5;
            int stride = cols + 3;
            float[] x = RandomData(rows * stride, 99 + cols, 3.0);
            var want = new double[rows * cols];
            for (int r = 0; r < rows; r++)
            {
                double max = double.NegativeInfinity;
                for (int j = 0; j < cols; j++)
                {
                    max = Math.Max(max, x[(r * stride) + j]);
                }

                double sum = 0;
                for (int j = 0; j < cols; j++)
                {
                    sum += Math.Exp(x[(r * stride) + j] - max);
                }

                for (int j = 0; j < cols; j++)
                {
                    want[(r * cols) + j] = Math.Exp(x[(r * stride) + j] - max) / sum;
                }
            }

            float[]? first = null;
            foreach (VecIsa isa in AllIsas)
            {
                foreach (bool exact in new[] { true, false })
                {
                    var y = new float[rows * cols];
                    Softmax.Rows(exact, isa, x, y, rows, cols, stride, cols);
                    for (int i = 0; i < y.Length; i++)
                    {
                        Assert.True(Math.Abs(y[i] - want[i]) < 2e-6 * want[i] + 1e-9, $"cols {cols} {isa} exact={exact} i={i}: {y[i]} vs {want[i]}");
                    }

                    if (exact)
                    {
                        first ??= y;
                        AssertSameBits(first, y, $"softmax cols {cols} {isa}");
                    }
                }
            }
        }
    }

    [Fact]
    public void SoftmaxSpecialRows()
    {
        float inf = float.PositiveInfinity;
        float[][] rows =
        [
            [float.NegativeInfinity, 1f, 2f, float.NegativeInfinity, 3f],
            [float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity],
            [1f, float.NaN, 2f],
            [1f, inf, 2f],
            [-0f, 0f, 0f, -0f, 1e30f, -1e30f],
        ];
        foreach (float[] row in rows)
        {
            float[]? first = null;
            foreach (VecIsa isa in AllIsas)
            {
                var y = new float[row.Length];
                Softmax.Rows(true, isa, row, y, 1, row.Length, row.Length, row.Length);
                first ??= y;
                AssertSameBits(first, y, $"{isa}");
            }
        }

        var all = new float[3];
        Softmax.Rows(true, VecIsa.V256, rows[1], all, 1, 3, 3, 3);
        Assert.All(all, x => Assert.Equal(0x7FC00000u, BitConverter.SingleToUInt32Bits(x)));
        var one = new float[5];
        Softmax.Rows(true, VecIsa.V256, rows[0], one, 1, 5, 5, 5);
        Assert.Equal(0f, one[0]);
        Assert.Equal(0f, one[3]);
        Assert.InRange(one[4], 0.66f, 0.67f);
    }
}
