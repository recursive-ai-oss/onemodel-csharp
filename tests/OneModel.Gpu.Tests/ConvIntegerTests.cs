using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tests.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu.Tests;

/// <summary>ONNX ConvInteger on the device against the CPU session, bit for bit in both modes and in one region.</summary>
public class ConvIntegerTests
{
    private const int U8 = 2, I32 = 6;

    [Theory]
    // n, c, h, w, m, kh, kw, group, stride, pad, dilation, per-channel weight zero point, x zero point present
    [InlineData(1, 3, 9, 11, 4, 3, 3, 1, 1, 1, 1, false, true)]
    [InlineData(2, 4, 8, 8, 6, 3, 3, 2, 2, 0, 1, true, true)]
    [InlineData(1, 6, 7, 5, 6, 3, 3, 6, 1, 1, 1, true, false)]
    [InlineData(3, 5, 10, 10, 7, 1, 1, 1, 1, 0, 1, false, true)]
    [InlineData(1, 2, 13, 9, 3, 3, 3, 1, 1, 2, 2, false, true)]
    public void TwoDimensionalMatchesTheCpuBitForBit(int n, int c, int h, int w, int m, int kh, int kw, int group, int stride, int pad, int dilation, bool perChannel, bool xZero)
    {
        var b = new ModelBuilder().Input("x", U8, n, c, h, w).Output("y", I32)
            .RawInitializer("w", U8, [m, c / group, kh, kw], MaskTests.Bytes(m * (c / group) * kh * kw, 31, false));
        string[] inputs = ["x", "w"];
        if (xZero || perChannel)
        {
            b = xZero ? b.RawInitializer("xz", U8, [1], [117]) : b.RawInitializer("xz", U8, [1], [0]);
            b = perChannel ? b.RawInitializer("wz", U8, [m], MaskTests.Bytes(m, 32, false)) : b.RawInitializer("wz", U8, [1], [9]);
            inputs = ["x", "w", "xz", "wz"];
        }

        byte[] model = b.Node("ConvInteger", inputs, ["y"], ("group", (long)group), ("strides", new long[] { stride, stride }),
            ("pads", new long[] { pad, pad, pad, pad }), ("dilations", new long[] { dilation, dilation })).Build();
        using Tensor x = Tensor.FromSpan<byte>(MaskTests.Bytes(n * c * h * w, 33, false), Shape.Create(n, c, h, w));
        MaskTests.Same(model, new() { ["x"] = x }, ["y"]);
    }

    [Theory]
    [InlineData(1, 80, 3000, 96, 3, 1, 1, false)]
    [InlineData(2, 96, 301, 96, 3, 2, 1, true)]
    public void OneDimensionalMatchesTheCpuBitForBit(int n, int c, int len, int m, int k, int stride, int pad, bool perChannel)
    {
        var b = new ModelBuilder().Input("x", U8, n, c, len).Output("y", I32)
            .RawInitializer("w", U8, [m, c, k], MaskTests.Bytes(m * c * k, 41, false)).RawInitializer("xz", U8, [1], [128]);
        b = perChannel ? b.RawInitializer("wz", U8, [m], MaskTests.Bytes(m, 42, false)) : b.RawInitializer("wz", U8, [1], [127]);
        byte[] model = b.Node("ConvInteger", ["x", "w", "xz", "wz"], ["y"], ("strides", new long[] { stride }), ("pads", new long[] { pad, pad })).Build();
        using Tensor x = Tensor.FromSpan<byte>(MaskTests.Bytes(n * c * len, 43, false), Shape.Create(n, c, len));
        MaskTests.Same(model, new() { ["x"] = x }, ["y"]);
    }

    [Theory]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void SignedOperandsInAnyMixMatchTheCpuBitForBit(bool xSigned, bool wSigned)
    {
        int xt = xSigned ? 3 : U8, wt = wSigned ? 3 : U8;
        const int n = 2, c = 6, h = 9, w = 7, m = 8, group = 2;
        byte[] model = new ModelBuilder().Input("x", xt, n, c, h, w).Output("y", I32)
            .RawInitializer("w", wt, [m, c / group, 3, 3], MaskTests.Bytes(m * (c / group) * 9, 51, wSigned))
            .RawInitializer("xz", xt, [1], [xSigned ? unchecked((byte)(sbyte)-7) : (byte)120])
            .RawInitializer("wz", wt, [m], MaskTests.Bytes(m, 52, wSigned))
            .Node("ConvInteger", ["x", "w", "xz", "wz"], ["y"], ("group", (long)group), ("pads", new long[] { 1, 1, 1, 1 })).Build();
        byte[] data = MaskTests.Bytes(n * c * h * w, 53, xSigned);
        using Tensor x = xSigned
            ? Tensor.FromSpan<sbyte>(System.Runtime.InteropServices.MemoryMarshal.Cast<byte, sbyte>(data).ToArray(), Shape.Create(n, c, h, w))
            : Tensor.FromSpan<byte>(data, Shape.Create(n, c, h, w));
        MaskTests.Same(model, new() { ["x"] = x }, ["y"]);
    }
}
