using System;
using System.IO;
using RecursiveAI.OneModel;
using Xunit;

namespace RecursiveAI.OneModel.Tests.Graph;

public class HappyPathTests
{
    /// <summary>A text encoder takes ids, a mask and segment ids; the one-liner fills the last two in.</summary>
    [Fact]
    public void TokenIdsAloneRunAModelWithMaskAndSegmentInputs()
    {
        byte[] bytes = new ModelBuilder()
            .Input("input_ids", 7, "batch", "seq")
            .Input("attention_mask", 7, "batch", "seq")
            .Input("token_type_ids", 7, "batch", "seq")
            .Initializer("ten", [10L], 1)
            .Node("Add", ["input_ids", "attention_mask"], ["a"])
            .Node("Mul", ["token_type_ids", "ten"], ["b"])
            .Node("Add", ["a", "b"], ["y"])
            .Output("y", 7, "batch", "seq")
            .Build();
        using var m = Model.Load(new MemoryStream(bytes));

        // ids + mask (ones) + 10 * segment ids (zeros)
        long[] y = m.Run<long, long>(new long[] { 5, 6, 7 });
        Assert.Equal([6L, 7L, 8L], y);
        Assert.Equal([6L, 7L, 8L, 9L], m.Run<long, long>(new long[] { 5, 6, 7, 8 }, [2, 2]));
    }

    /// <summary>The README's one-liner on a real encoder, when its file is supplied (ONEMODEL_MINILM=path to the fp32 export).</summary>
    [Fact]
    public void ReadmeSnippetRunsMiniLm()
    {
        string? path = Environment.GetEnvironmentVariable("ONEMODEL_MINILM");
        Assert.SkipUnless(!string.IsNullOrEmpty(path) && File.Exists(path), "ONEMODEL_MINILM not set");
        long[] tokens = [101, 7592, 2088, 102];
        using var m = Model.Load(path!);
        float[] v = m.Run(tokens);
        Assert.Equal(tokens.Length * 384, v.Length);
        Assert.All(v, x => Assert.True(float.IsFinite(x)));
    }
}
