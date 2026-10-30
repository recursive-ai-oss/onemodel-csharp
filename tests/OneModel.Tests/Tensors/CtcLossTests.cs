using RecursiveAI.OneModel.Tensors;
using static RecursiveAI.OneModel.Tests.Tensors.GradCheck;

namespace RecursiveAI.OneModel.Tests.Tensors;

[Collection("autograd")]
public class CtcLossTests
{
    private static Shape S(params int[] d) => Shape.Create(d);

    [Fact]
    public void SingleFrameIsNegativeLogProbability()
    {
        using Tensor lp = Tensor.FromSpan<float>([MathF.Log(0.25f), MathF.Log(0.75f)], S(1, 1, 2));
        using Tensor loss = Ops.CtcLoss(lp, [1], [1], [1], reduction: CtcReduction.Sum);
        Assert.Equal(-MathF.Log(0.75f), loss.Data<float>()[0], 6);
    }

    [Fact]
    public void TwoFrameAlignmentsAreSummed()
    {
        // target [1], T=2, C=2: alignments (1,1) (0,1) (1,0): p = y1y1 + y0y1 + y1y0.
        float[] p0 = [0.4f, 0.7f], p1 = [0.6f, 0.3f];
        float[] lpv = [MathF.Log(p0[0]), MathF.Log(p1[0]), MathF.Log(p0[1]), MathF.Log(p1[1])];
        using Tensor lp = Tensor.FromSpan<float>(lpv, S(2, 1, 2));
        using Tensor loss = Ops.CtcLoss(lp, [1], [2], [1], reduction: CtcReduction.None);
        double p = (0.6 * 0.3) + (0.4 * 0.3) + (0.6 * 0.7);
        Assert.Equal(-Math.Log(p), loss.Data<float>()[0], 5);
    }

    [Fact]
    public void ImpossibleAlignmentIsInfiniteOrZero()
    {
        using Tensor lp = Ops.LogSoftmax(Tensor.FromSpan<float>(new float[2 * 3], S(2, 1, 3)));
        using Tensor inf = Ops.CtcLoss(lp, [1, 1], [2], [2], reduction: CtcReduction.None);
        Assert.True(float.IsPositiveInfinity(inf.Data<float>()[0]));
        using Tensor zero = Ops.CtcLoss(lp, [1, 1], [2], [2], reduction: CtcReduction.None, zeroInfinity: true);
        Assert.Equal(0f, zero.Data<float>()[0]);
    }

    [Theory]
    [InlineData(CtcReduction.None)]
    [InlineData(CtcReduction.Mean)]
    [InlineData(CtcReduction.Sum)]
    public void GradientMatchesFiniteDifferences(CtcReduction reduction)
    {
        // Three sequences: ordinary, repeated labels needing a blank between, and a shorter valid input with an empty-ish target.
        int[] targets = [1, 2, 2, 3, 1, 1];
        int[] inLen = [7, 6, 4];
        int[] tgtLen = [2, 3, 1];
        Run($"CtcLoss {reduction}", t =>
        {
            Tensor lp = Ops.LogSoftmax(t[0]);
            return Ops.CtcLoss(lp, targets, inLen, tgtLen, 0, reduction);
        }, [S(7, 3, 4)], [Uniform(-1.5f, 1.5f)], h: 1e-2, tolerance: 5e-3);
    }

    [Fact]
    public void GradientWithEmptyTargetsBlankIndexAndZeroInfinity()
    {
        Run("CtcLoss empty target + impossible (zero-infinity), blank=3", t =>
            Ops.CtcLoss(Ops.LogSoftmax(t[0]), [0, 1, 2, 2], [5, 5, 2, 1], [0, 2, 2, 0], 3, CtcReduction.Mean, zeroInfinity: true),
            [S(5, 4, 4)], [Uniform(-1.5f, 1.5f)], h: 1e-2, tolerance: 5e-3);
    }

    [Fact]
    public void BadArgumentsThrow()
    {
        using Tensor lp = Tensor.Zeros(DType.F32, S(3, 2, 4));
        Assert.Throws<ArgumentException>(() => Ops.CtcLoss(lp, [1], [3, 3], [2, 0]));
        Assert.Throws<ArgumentException>(() => Ops.CtcLoss(lp, [0], [3, 3], [1, 0]));
        Assert.Throws<ArgumentException>(() => Ops.CtcLoss(lp, [9], [3, 3], [1, 0]));
        Assert.Throws<ArgumentException>(() => Ops.CtcLoss(lp, [1], [4, 3], [1, 0]));
        Assert.Throws<ArgumentException>(() => Ops.CtcLoss(lp, [1], [3, 3], [1, 0], blank: 7));
    }
}
