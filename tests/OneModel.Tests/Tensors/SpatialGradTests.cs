using RecursiveAI.OneModel.Tensors;
using static RecursiveAI.OneModel.Tests.Tensors.GradCheck;

namespace RecursiveAI.OneModel.Tests.Tensors;

[Collection("autograd")]
public class SpatialGradTests
{
    private static Shape S(params int[] d) => Shape.Create(d);

    [Fact]
    public void ConvTransposeGradients()
    {
        Run("ConvT 3x3 s1", t => Ops.ConvTranspose2d(t[0], t[1], t[2]), [S(2, 3, 4, 5), S(3, 2, 3, 3), S(2)]);
        Run("ConvT 2x2 s2 (decoder upsample)", t => Ops.ConvTranspose2d(t[0], t[1], t[2], new ConvTranspose2dParams(StrideH: 2, StrideW: 2)),
            [S(1, 4, 3, 3), S(4, 3, 2, 2), S(3)]);
        Run("ConvT 3x3 s2 pad1 outpad1", t => Ops.ConvTranspose2d(t[0], t[1], default, new ConvTranspose2dParams(2, 2, 1, 1, 1, 1)),
            [S(2, 2, 4, 3), S(2, 3, 3, 3)]);
        Run("ConvT groups=2 dil=2 asym stride", t => Ops.ConvTranspose2d(t[0], t[1], t[2], new ConvTranspose2dParams(1, 2, 1, 0, 0, 0, 2, 1, 2)),
            [S(2, 4, 4, 3), S(4, 2, 3, 2), S(4)]);
    }

    [Fact]
    public void ConvTransposeMatchesDefinition()
    {
        // One input pixel with value 2 and a 2x2 kernel at stride 2: the output is the kernel times 2.
        using Tensor x = Tensor.FromSpan<float>([2f], S(1, 1, 1, 1));
        using Tensor w = Tensor.FromSpan<float>([1f, 2f, 3f, 4f], S(1, 1, 2, 2));
        using Tensor y = Ops.ConvTranspose2d(x, w, default, new ConvTranspose2dParams(StrideH: 2, StrideW: 2));
        Assert.Equal(S(1, 1, 2, 2), y.Shape);
        Assert.True(y.Data<float>().SequenceEqual([2f, 4f, 6f, 8f]));
    }

    [Theory]
    [InlineData(InterpolationMode.Nearest, false)]
    [InlineData(InterpolationMode.Bilinear, false)]
    [InlineData(InterpolationMode.Bilinear, true)]
    public void ResizeGradients(InterpolationMode mode, bool align)
    {
        Run($"Resize up {mode} align={align}", t => Ops.Resize2d(t[0], 7, 5, mode, align), [S(2, 2, 3, 4)]);
        Run($"Resize down {mode} align={align}", t => Ops.Resize2d(t[0], 2, 3, mode, align), [S(1, 2, 5, 7)]);
        Run($"Resize 1x1 target {mode} align={align}", t => Ops.Resize2d(t[0], 1, 1, mode, align), [S(1, 1, 3, 3)]);
    }

    [Fact]
    public void ResizeNearestRepeatsPixels()
    {
        using Tensor x = Tensor.FromSpan<float>([1f, 2f, 3f, 4f], S(1, 1, 2, 2));
        using Tensor y = Ops.Resize2d(x, 4, 4);
        Assert.True(y.Data<float>().SequenceEqual([1f, 1f, 2f, 2f, 1f, 1f, 2f, 2f, 3f, 3f, 4f, 4f, 3f, 3f, 4f, 4f]));
    }
}
