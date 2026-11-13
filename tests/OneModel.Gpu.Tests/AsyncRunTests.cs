using System.Security.Cryptography;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tests.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu.Tests;

/// <summary>
/// The awaited run a browser needs, and the arena layout a browser needs (an operator's inputs, outputs and scratch in different buffers).
/// Both are checked here on whatever device the machine has: they must change no bits.
/// </summary>
public class AsyncRunTests
{
    private const int F = 64;

    // Device, host, device: the awaited run crosses a host operator between two regions.
    private static byte[] Mixed() => new ModelBuilder()
        .Input("x", 1, "N", F).Output("y", 1, "N", F)
        .Initializer("w1", GpuTestHost.Random(F * F, 1), F, F).Initializer("b", GpuTestHost.Random(F, 2), F)
        .Initializer("w2", GpuTestHost.Random(F * F, 3), F, F).Initializer("w3", GpuTestHost.Random(F * F, 4), F, F)
        .Node("MatMul", ["x", "w1"], ["m1"]).Node("Add", ["m1", "b"], ["a1"]).Node("Floor", ["a1"], ["r1"])
        .Node("MatMul", ["r1", "w2"], ["m2"]).Node("Relu", ["m2"], ["m3"]).Node("MatMul", ["m3", "w3"], ["y"]).Build();

    private static string Hash(SessionOutputs o) => Convert.ToHexString(SHA256.HashData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(o["y"].Data<float>())));

    [Theory]
    [InlineData(ArithmeticMode.Fast)]
    [InlineData(ArithmeticMode.Exact)]
    public async Task AwaitedRunGivesTheBlockingRunsBits(ArithmeticMode mode)
    {
        GpuTestHost.Require();
        using Session s = GpuTestHost.Load(Mixed(), GpuTestHost.Gpu(mode));
        using Tensor x = Tensor.FromSpan<float>(GpuTestHost.Random(7 * F, 9), Shape.Create(7, F));
        var feed = new Dictionary<string, Tensor> { ["x"] = x };
        using SessionOutputs blocking = s.Run(feed);
        using SessionOutputs awaited = await OneModelGpu.RunAsync(s, feed);
        Assert.Equal(Hash(blocking), Hash(awaited));
        using SessionOutputs again = await OneModelGpu.RunAsync(s, feed);
        Assert.Equal(Hash(blocking), Hash(again));
        GpuTestHost.AssertValidationClean(s);
    }

    [Fact]
    public async Task ACanceledTokenStopsTheRunBeforeItStarts()
    {
        GpuTestHost.Require();
        using Session s = GpuTestHost.Load(Mixed(), GpuTestHost.Gpu());
        using Tensor x = Tensor.FromSpan<float>(GpuTestHost.Random(7 * F, 9), Shape.Create(7, F));
        var feed = new Dictionary<string, Tensor> { ["x"] = x };
        using var canceled = new CancellationTokenSource();
        await canceled.CancelAsync();
        await Assert.ThrowsAsync<OperationCanceledException>(async () => (await OneModelGpu.RunAsync(s, feed, null, canceled.Token)).Dispose());
        using SessionOutputs ok = await OneModelGpu.RunAsync(s, feed);
    }

    [Fact]
    public async Task AwaitedRunsOfOneSessionTakeTurns()
    {
        GpuTestHost.Require();
        using Session s = GpuTestHost.Load(Mixed(), GpuTestHost.Gpu());
        using Tensor x = Tensor.FromSpan<float>(GpuTestHost.Random(5 * F, 9), Shape.Create(5, F));
        var feed = new Dictionary<string, Tensor> { ["x"] = x };
        using SessionOutputs reference = s.Run(feed);
        Task<SessionOutputs>[] pending = [OneModelGpu.RunAsync(s, feed).AsTask(), OneModelGpu.RunAsync(s, feed).AsTask(), OneModelGpu.RunAsync(s, feed).AsTask()];
        foreach (Task<SessionOutputs> p in pending)
        {
            using SessionOutputs o = await p;
            Assert.Equal(Hash(reference), Hash(o));
        }

        GpuTestHost.AssertValidationClean(s);
    }

    [Fact]
    public void TheSeparateBuffersLayoutChangesNoBits()
    {
        GpuTestHost.Require();
        using Session plain = GpuTestHost.Load(Mixed(), GpuTestHost.Gpu());
        using Session apart = GpuTestHost.Load(Mixed(), GpuTestHost.Gpu());
        ((GpuBackend)apart.Backend!).SeparateReadWriteSlabs = true;
        using Tensor x = Tensor.FromSpan<float>(GpuTestHost.Random(9 * F, 9), Shape.Create(9, F));
        var feed = new Dictionary<string, Tensor> { ["x"] = x };
        using SessionOutputs a = plain.Run(feed);
        using SessionOutputs b = apart.Run(feed);
        Assert.Equal(Hash(a), Hash(b));
        GpuTestHost.AssertValidationClean(apart);
    }

    [Fact]
    public void ThePlannerKeepsConflictingValuesInDifferentSlabsEvenWhenTheirLifetimesAreDisjoint()
    {
        GpuTestHost.Require();
        using Session s = GpuTestHost.Load(Mixed(), GpuTestHost.Gpu());
        DeviceArena arena = ((GpuBackend)s.Backend!).Arena;

        // A chain a -> b -> c -> d: each is read by the next one's producer, and lifetimes touch only neighbours. Without constraints one slab holds all four.
        List<LiveValue> chain = [new(0, DeviceElement.F32, 100, 0, 1), new(1, DeviceElement.F32, 100, 1, 2), new(2, DeviceElement.F32, 100, 2, 3), new(3, DeviceElement.F32, 100, 3, 4)];
        Assert.All(arena.Plan([.. chain]).Slots, slot => Assert.Equal(0, slot.Slab));
        DeviceArena.Layout apart = arena.Plan([.. chain], [(0, 1), (1, 2), (2, 3)]);
        for (int i = 0; i < 3; i++)
        {
            Assert.NotEqual(apart.Slots[i].Slab, apart.Slots[i + 1].Slab);
        }

        // Values of different element types are different buffers and never need to be kept apart.
        DeviceArena.Layout mixed = arena.Plan([new(0, DeviceElement.I64, 10, 0, 1), new(1, DeviceElement.F32, 10, 1, 2)], [(0, 1)]);
        Assert.Equal(0, mixed.Slots[0].Slab);
        Assert.Equal(0, mixed.Slots[1].Slab);
    }
}
