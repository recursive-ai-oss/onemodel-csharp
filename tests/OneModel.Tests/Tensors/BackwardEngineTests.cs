using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Tensors;

[Collection("autograd")]
public class BackwardEngineTests
{
    private static Shape S(params int[] d) => Shape.Create(d);

    private static Tensor Leaf(params float[] v) => Tensor.FromSpan<float>(v, S(v.Length)).WithRequiresGrad();

    [Fact]
    public void DiamondGraphSumsBothPaths()
    {
        // y = sum(x*x + 3x) -> dy/dx = 2x + 3
        Tensor x = Leaf(1f, 2f, -3f);
        using (new ActivationScope())
        {
            Tensor three = Tensor.Scalar(3f);
            Ops.Sum(Ops.Add(Ops.Mul(x, x), Ops.Mul(x, three))).Backward();
        }

        Assert.Equal([5f, 7f, -3f], x.Grad.ToArray<float>());
    }

    [Fact]
    public void BackwardTwiceNeedsRetainGraph()
    {
        Tensor x = Leaf(1f, 2f);
        using var scope = new ActivationScope();
        Tensor y = Ops.Sum(Ops.Mul(x, x));
        y.Backward(retainGraph: true);
        y.Backward();
        Assert.Equal([4f, 8f], x.Grad.ToArray<float>());
        Assert.Throws<InvalidOperationException>(() => y.Backward());
    }

    [Fact]
    public void NonScalarNeedsAGradientAndLeafNeedsGradFlag()
    {
        Tensor x = Leaf(1f, 2f);
        using var scope = new ActivationScope();
        Tensor y = Ops.Mul(x, x);
        Assert.Throws<TensorShapeException>(() => y.Backward());
        Assert.Throws<TensorShapeException>(() => y.Backward(Tensor.Zeros(DType.F32, S(3))));
        Tensor plain = Tensor.FromSpan<float>([1f], S(1));
        Assert.Throws<InvalidOperationException>(() => plain.Backward());
        y.Backward(Tensor.FromSpan<float>([1f, 10f], S(2)));
        Assert.Equal([2f, 40f], x.Grad.ToArray<float>());
    }

    [Fact]
    public void NoGradRecordsNothing()
    {
        Tensor x = Leaf(1f, 2f);
        using var scope = new ActivationScope();
        using (GradMode.NoGrad())
        {
            Tensor y = Ops.Sum(Ops.Mul(x, x));
            Assert.False(y.RequiresGrad);
        }

        Assert.True(Ops.Sum(x).RequiresGrad);
    }

    [Fact]
    public void InPlaceChangeOfASavedTensorIsCaught()
    {
        Tensor x = Leaf(1f, 2f);
        using var scope = new ActivationScope();
        Tensor w = Tensor.FromSpan<float>([3f, 4f], S(2));
        Tensor y = Ops.Sum(Ops.Mul(x, w));
        w.MutableData<float>()[0] = 9f;
        Assert.Throws<TensorVersionException>(() => y.Backward());
    }

    [Fact]
    public void DetachedAndUnusedInputsGetNoGradient()
    {
        Tensor a = Leaf(1f, 2f), b = Leaf(3f, 4f);
        using (new ActivationScope())
        {
            Ops.Sum(Ops.Mul(a, b.Detach())).Backward();
        }

        Assert.False(a.Grad.IsEmpty);
        Assert.True(b.Grad.IsEmpty);
        Assert.Equal([3f, 4f], a.Grad.ToArray<float>());
    }

    [Fact]
    public void ScopeFreesIntermediatesAndSurvivesManualDispose()
    {
        PoolStats before = BufferPool.Shared.GetStats();
        Tensor x = Leaf(1f, 2f);
        using (new ActivationScope())
        {
            Tensor y = Ops.Mul(x, x);
            Tensor z = Ops.Sum(y);
            y.Dispose(); // the scope tolerates tensors released early
            z.Backward();
        }

        // Only x, its gradient and (parallel tests aside) nothing else stays out: far less than a pooled block's worth of leaks.
        Assert.True(BufferPool.Shared.GetStats().OutstandingBytes - before.OutstandingBytes < 4096);
    }
}
