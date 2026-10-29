using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Tensors;

public class LayoutTests
{
    [Theory]
    [InlineData(1, 1, 1, 1, 8)]
    [InlineData(2, 3, 4, 5, 8)]
    [InlineData(1, 16, 3, 3, 8)]
    [InlineData(2, 13, 2, 7, 16)]
    [InlineData(3, 5, 1, 6, 4)]
    [InlineData(2, 0, 3, 3, 8)]
    public void ConversionsRoundTripAndKeepLogicalValues(int n, int c, int h, int w, int block)
    {
        int len = n * c * h * w;
        using Tensor nchw = Tensor.FromSpan<int>(Enumerable.Range(1, len).ToArray(), Shape.Create(n, c, h, w));
        Assert.Equal(TensorLayout.Nchw, nchw.Layout);

        using Tensor nhwc = nchw.ToLayout(TensorLayout.Nhwc);
        Assert.Equal(nchw.Shape, nhwc.Shape);
        if (c > 1 && h * w > 1)
        {
            Assert.Equal(TensorLayout.Nhwc, nhwc.Layout);
        }

        Assert.Equal(nchw.ToArray<int>(), nhwc.ToArray<int>());
        if (len > 0)
        {
            Assert.Equal(nchw.Get<int>(0, c - 1, h - 1, 0), nhwc.Get<int>(0, c - 1, h - 1, 0));
        }

        using Tensor blocked = nchw.ToLayout(TensorLayout.NchwC(block));
        Assert.True(blocked.IsBlocked);
        Assert.Equal(TensorLayout.NchwC(block), blocked.Layout);
        Assert.Equal(nchw.Shape, blocked.Shape);
        int cb = (c + block - 1) / block;
        Assert.Equal(n * cb * h * w * block, blocked.Data<int>().Length);
        Assert.Throws<NotSupportedException>(() => blocked.Permute(1, 0, 2, 3));
        Assert.Throws<InvalidOperationException>(() => blocked.Strides);

        // Spot-check the documented order [n][c/B][h][w][c%B], and zero padding.
        ReadOnlySpan<int> phys = blocked.Data<int>();
        for (int i = 0; i < n; i++)
        {
            for (int ch = 0; ch < cb * block; ch++)
            {
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        int p = phys[(((((i * cb) + (ch / block)) * h + y) * w) + x) * block + (ch % block)];
                        Assert.Equal(ch < c ? nchw.Get<int>(i, ch, y, x) : 0, p);
                    }
                }
            }
        }

        using Tensor back = blocked.ToLayout(TensorLayout.Nchw);
        Assert.Equal(nchw.ToArray<int>(), back.ToArray<int>());
        using Tensor viaNhwc = nhwc.ToLayout(TensorLayout.NchwC(block));
        Assert.Equal(blocked.Data<int>().ToArray(), viaNhwc.Data<int>().ToArray());
        using Tensor blockedToNhwc = blocked.ToLayout(TensorLayout.Nhwc);
        Assert.Equal(nchw.ToArray<int>(), blockedToNhwc.ToArray<int>());
    }

    [Fact]
    public void SlicedAndFlippedInputsConvert()
    {
        using Tensor big = Tensor.FromSpan<ushort>(Enumerable.Range(0, 2 * 6 * 4 * 4).Select(i => (ushort)i).ToArray(), Shape.Create(2, 6, 4, 4));
        Tensor view = big.Slice(1, 1, 6, 2).Flip(3).Slice(2, 1, 4);
        using Tensor blocked = view.ToLayout(TensorLayout.NchwC(4));
        using Tensor back = blocked.ToLayout(TensorLayout.Nchw);
        Assert.Equal(view.ToArray<ushort>(), back.ToArray<ushort>());
        Assert.Throws<TensorShapeException>(() => big.Select(0, 0).ToLayout(TensorLayout.Nhwc));
        Assert.Throws<ArgumentOutOfRangeException>(() => TensorLayout.NchwC(0));
    }
}
