using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tests.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu.Tests;

/// <summary>A warm run builds no bind groups, however many distinct weights the model has, and growing the arena drops the old ones.</summary>
public class BindingCacheTests
{
    // The kernel families count the bind groups they have built.
    private static int BindGroups(GpuBackend backend) =>
        backend.Elementwise.BindGroupsCreated + backend.Norm.BindGroupsCreated + backend.Softmax.BindGroupsCreated + backend.Reduce.BindGroupsCreated
        + backend.Transpose.BindGroupsCreated + backend.Integer.BindGroupsCreated + backend.GetGemm(DeviceElement.F32).BindGroupsCreated + backend.Gather.BindGroupsCreated
        + backend.Attention.BindGroupsCreated + backend.Int8.BindGroupsCreated + backend.DynamicQuantize.BindGroupsCreated + backend.Dequantize.BindGroupsCreated
        + backend.Recurrent.BindGroupsCreated + backend.Conv.BindGroupsCreated + backend.Pool.BindGroupsCreated + backend.Resize.BindGroupsCreated + backend.Layout.BindGroupsCreated;

    private static byte[] Chain(int products)
    {
        var m = new ModelBuilder().Input("x", 1, "B", 16).Output("y", 1);
        string prev = "x";
        for (int i = 0; i < products; i++)
        {
            string o = i == products - 1 ? "y" : $"t{i}", t = $"a{i}";
            m.Initializer($"w{i}", GpuTestHost.Random(256, 100 + i, 0.4f), 16, 16).Initializer($"b{i}", GpuTestHost.Random(16, 300 + i), 16)
                .Node("MatMul", [prev, $"w{i}"], [t]).Node("Add", [t, $"b{i}"], [o]);
            prev = o;
        }

        return m.Build();
    }

    [Fact]
    public void AWarmRunBuildsNoBindGroupsAndAGrownArenaRebuildsThemOnce()
    {
        GpuTestHost.Require();
        using Session s = GpuTestHost.Load(Chain(150), GpuTestHost.Gpu());
        var backend = (GpuBackend)s.Backend!;
        using Tensor small = Tensor.FromSpan<float>(GpuTestHost.Random(3 * 16, 1), Shape.Create(3, 16));
        using Tensor big = Tensor.FromSpan<float>(GpuTestHost.Random(40000 * 16, 2), Shape.Create(40000, 16));

        s.Run(new Dictionary<string, Tensor> { ["x"] = small }).Dispose();
        int afterFirst = BindGroups(backend);
        // Each product with its bias is one dispatch: 150 sets of buffers (without folding it would be 300).
        Assert.True(afterFirst >= 150, $"150 products with their biases touched {afterFirst} sets");
        s.Run(new Dictionary<string, Tensor> { ["x"] = small }).Dispose();
        Assert.Equal(afterFirst, BindGroups(backend));

        // A shape that needs a bigger slab replaces the buffers: the sets are built again once, then cached again.
        s.Run(new Dictionary<string, Tensor> { ["x"] = big }).Dispose();
        int afterGrowth = BindGroups(backend);
        Assert.True(afterGrowth > afterFirst);
        s.Run(new Dictionary<string, Tensor> { ["x"] = big }).Dispose();
        Assert.Equal(afterGrowth, BindGroups(backend));

        // A few-row shape can pick another kernel variant than the big one, whose sets the growth dropped: it builds them once, then both shapes are cached.
        s.Run(new Dictionary<string, Tensor> { ["x"] = small }).Dispose();
        int afterBoth = BindGroups(backend);
        s.Run(new Dictionary<string, Tensor> { ["x"] = big }).Dispose();
        s.Run(new Dictionary<string, Tensor> { ["x"] = small }).Dispose();
        Assert.Equal(afterBoth, BindGroups(backend));
        GpuTestHost.AssertValidationClean(s);
    }
}
