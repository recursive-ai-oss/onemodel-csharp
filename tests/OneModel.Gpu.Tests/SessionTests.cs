using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tests.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu.Tests;

/// <summary>Partitioning, transfers, the arena, fallback reasons, RequireGpu and options, on a small model.</summary>
public class SessionTests
{
    private const int F = 64;

    // x -> MatMul -> Add(bias) -> Floor -> MatMul -> MatMul -> y: Floor has no GPU kernel, so two device regions surround one host operator.
    private static byte[] Mixed() => new ModelBuilder()
        .Input("x", 1, "N", F).Output("y", 1, "N", F)
        .Initializer("w1", GpuTestHost.Random(F * F, 1), F, F).Initializer("b", GpuTestHost.Random(F, 2), F)
        .Initializer("w2", GpuTestHost.Random(F * F, 3), F, F).Initializer("w3", GpuTestHost.Random(F * F, 4), F, F)
        .Node("MatMul", ["x", "w1"], ["m1"]).Node("Add", ["m1", "b"], ["a1"]).Node("Floor", ["a1"], ["r1"])
        .Node("MatMul", ["r1", "w2"], ["m2"]).Node("MatMul", ["m2", "w3"], ["y"]).Build();

    // Everything runs on the device: two products in a row.
    private static byte[] AllDevice() => new ModelBuilder()
        .Input("x", 1, "N", F).Output("y", 1, "N", F)
        .Initializer("w1", GpuTestHost.Random(F * F, 1), F, F).Initializer("w2", GpuTestHost.Random(F * F, 3), F, F)
        .Node("MatMul", ["x", "w1"], ["m1"]).Node("MatMul", ["m1", "w2"], ["y"]).Build();

    private static Dictionary<string, Tensor> Feed(int rows, out Tensor owner)
    {
        owner = Tensor.FromSpan<float>(GpuTestHost.Random(rows * F, 9), Shape.Create(rows, F));
        return new Dictionary<string, Tensor> { ["x"] = owner };
    }

    [Fact]
    public void ADeviceOnlyModelMovesTensorsOnlyAtItsEnds()
    {
        GpuTestHost.Require();
        using Session s = GpuTestHost.Load(AllDevice(), GpuTestHost.Gpu());
        var feed = Feed(10, out Tensor x);
        using (x)
        {
            SessionDiagnostics d = s.GetDiagnostics(feed);
            Assert.Equal(1, d.DeviceRegions);
            Assert.Equal(2, d.NodesOnDevice);
            Assert.Equal(0, d.NodesOnHost);
            Assert.Equal(10 * F * 4, d.BytesToDevicePerRun);
            Assert.Equal(10 * F * 4, d.BytesFromDevicePerRun);
            Assert.All(d.Nodes, n => Assert.Null(n.FallbackReason));
            using SessionOutputs o = s.Run(feed);
            GpuTestHost.AssertValidationClean(s);
        }
    }

    [Fact]
    public void UnsupportedOperatorsStayOnTheCpuWithTheirReasons()
    {
        GpuTestHost.Require();
        using Session s = GpuTestHost.Load(Mixed(), GpuTestHost.Gpu());
        var feed = Feed(5, out Tensor x);
        using (x)
        {
            SessionDiagnostics d = s.GetDiagnostics(feed);
            Assert.Equal(["MatMul", "Add", "Floor", "MatMul", "MatMul"], d.Nodes.Select(n => n.OpType));
            Assert.Equal(["gpu", "gpu", "cpu", "gpu", "gpu"], d.Nodes.Select(n => n.Placement));
            Assert.Equal([0, 0, -1, 1, 1], d.Nodes.Select(n => n.Region));
            Assert.Equal("no GPU kernel for Floor", d.Nodes[2].FallbackReason);
            Assert.Equal(2, d.DeviceRegions);
            // Region 0 takes x and returns a1; region 1 takes r1 and returns y.
            Assert.Equal(2 * 5 * F * 4, d.BytesToDevicePerRun);
            Assert.Equal(2 * 5 * F * 4, d.BytesFromDevicePerRun);

            // The result is the CPU's within the fast tolerance.
            using Session cpu = GpuTestHost.Load(Mixed(), GpuTestHost.Cpu());
            using SessionOutputs a = s.Run(feed);
            using SessionOutputs b = cpu.Run(feed);
            Assert.True(GpuTestHost.ScaledError(a["y"].Data<float>(), b["y"].Data<float>()) <= 1e-5);
            GpuTestHost.AssertValidationClean(s);
        }
    }

    [Fact]
    public void RequireGpuNamesEveryUnsupportedOperator()
    {
        GpuTestHost.Require();
        SessionException e = Assert.Throws<SessionException>(() => GpuTestHost.Load(Mixed(), GpuTestHost.Gpu(require: true)));
        Assert.Equal(SessionErrorKind.DeviceUnsupported, e.Kind);
        Assert.Equal(["Floor"], e.Operators);
        Assert.Contains("no GPU kernel for Floor", e.Message);

        // A model the device can run completely is accepted.
        using Session ok = GpuTestHost.Load(AllDevice(), GpuTestHost.Gpu(require: true));
        var feed = Feed(3, out Tensor x);
        using (x)
        {
            using SessionOutputs o = ok.Run(feed);
            Assert.Equal(3 * F, o["y"].Length);
        }
    }

    [Fact]
    public void RequireGpuWithoutTheGpuDeviceIsRejected()
    {
        Assert.Throws<ArgumentException>(() => GpuTestHost.Load(AllDevice(), new ModelOptions { RequireGpu = true }));
    }

    [Fact]
    public void ASecondRunOfTheSameShapeAllocatesNothingOnTheDevice()
    {
        GpuTestHost.Require();
        using Session s = GpuTestHost.Load(Mixed(), GpuTestHost.Gpu());
        var feed = Feed(32, out Tensor x);
        using (x)
        {
            Assert.Equal(0, OneModelGpu.GetInfo(s).DeviceAllocations);
            using (s.Run(feed))
            {
            }

            GpuSessionInfo first = OneModelGpu.GetInfo(s);
            int stats1 = ((GpuBackend)s.Backend!).Device.MemoryStats.Allocations;
            using (s.Run(feed))
            {
            }

            GpuSessionInfo second = OneModelGpu.GetInfo(s);
            int stats2 = ((GpuBackend)s.Backend!).Device.MemoryStats.Allocations;
            Assert.True(first.DeviceAllocations > 0);
            Assert.True(first.ArenaBytes > 0);
            Assert.Equal(first.DeviceAllocations, second.DeviceAllocations);
            Assert.Equal(first.ArenaBytes, second.ArenaBytes);
            Assert.Equal(stats1, stats2);

            // A smaller batch fits in what exists; a larger one grows the arena once and then stays put.
            var small = Feed(4, out Tensor xs);
            using (xs)
            {
                using (s.Run(small))
                {
                }
            }

            Assert.Equal(first.DeviceAllocations, OneModelGpu.GetInfo(s).DeviceAllocations);
            var big = Feed(20000, out Tensor xb);
            using (xb)
            {
                using (s.Run(big))
                {
                }

                long grown = OneModelGpu.GetInfo(s).DeviceAllocations;
                Assert.True(grown > first.DeviceAllocations);
                using (s.Run(big))
                {
                }

                Assert.Equal(grown, OneModelGpu.GetInfo(s).DeviceAllocations);
            }

            GpuTestHost.AssertValidationClean(s);
        }
    }

    [Fact]
    public void ExactModeOnTheGpuGivesTheSameBitsOnEveryRunAndConcurrently()
    {
        GpuTestHost.Require();
        using Session s = GpuTestHost.Load(Mixed(), GpuTestHost.Gpu(ArithmeticMode.Exact));
        var feed = Feed(33, out Tensor x);
        using (x)
        {
            string Once()
            {
                using SessionOutputs o = s.Run(feed);
                return GpuTestHost.Hash(o["y"].Data<float>());
            }

            string h = Once();
            Assert.Equal(h, Once());
            var results = new string[4];
            Parallel.For(0, results.Length, new ParallelOptions { MaxDegreeOfParallelism = 4 }, i =>
            {
                string last = "";
                for (int k = 0; k < 5; k++)
                {
                    last = Once();
                    Assert.Equal(h, last);
                }

                results[i] = last;
            });
            Assert.All(results, r => Assert.Equal(h, r));
            Assert.All(OneModelGpu.GetInfo(s).KernelVariants, v => Assert.Contains("deterministic", v));
            GpuTestHost.AssertValidationClean(s);
        }
    }

    [Fact]
    public void TimingsAddUpWhenCollected()
    {
        GpuTestHost.Require();
        using Session s = GpuTestHost.Load(AllDevice(), GpuTestHost.Gpu());
        var feed = Feed(256, out Tensor x);
        using (x)
        {
            using (s.Run(feed))
            {
            }

            OneModelGpu.ResetTimings(s);
            for (int i = 0; i < 3; i++)
            {
                using (s.Run(feed))
                {
                }
            }

            GpuTimings t = OneModelGpu.GetInfo(s).Timings;
            Assert.Equal(3, t.Regions);
            Assert.True(t.TotalMilliseconds > 0);
            Assert.True(t.TotalMilliseconds + 1e-6 >= t.UploadMilliseconds + t.SubmitAndReadbackMilliseconds - 1e-6);
            TestContext.Current.SendDiagnosticMessage($"timings over 3 runs: {t}");
        }
    }

    [Fact]
    public void InfoNamesTheAdapterAndTheKernelVariant()
    {
        GpuTestHost.Require();
        using Session s = GpuTestHost.Load(AllDevice(), GpuTestHost.Gpu());
        var feed = Feed(2, out Tensor x);
        using (x)
        {
            using (s.Run(feed))
            {
            }

            GpuSessionInfo info = OneModelGpu.GetInfo(s, feed);
            Assert.False(string.IsNullOrEmpty(info.Adapter));
            Assert.NotNull(info.Plan);
            Assert.Equal(2.0 * 2 * 2 * F * F, info.Plan!.Flops);
            Assert.Single(info.KernelVariants);
            TestContext.Current.SendDiagnosticMessage($"adapter: {info.Adapter} ({info.Api}); {string.Join(", ", info.KernelVariants)}");
        }
    }

    [Fact]
    public void ReleasingASessionLeavesNothingBehindOnTheDevice()
    {
        GpuTestHost.Require();
        GpuBackend backend;
        using (Session s = GpuTestHost.Load(Mixed(), GpuTestHost.Gpu()))
        {
            backend = (GpuBackend)s.Backend!;
            var feed = Feed(6, out Tensor x);
            using (x)
            {
                using (s.Run(feed))
                {
                }
            }

            Assert.False(backend.IsDisposed);
        }

        Assert.True(backend.IsDisposed);
        Assert.Empty(backend.MessagesAfterDispose);
    }

    [Fact]
    public void DisposingASessionTwiceIsHarmless()
    {
        GpuTestHost.Require();
        Session s = GpuTestHost.Load(Mixed(), GpuTestHost.Gpu());
        s.Dispose();
        s.Dispose();
        ((GpuBackend)s.Backend!).Dispose();
        Assert.True(((GpuBackend)s.Backend!).IsDisposed);
    }

    // Floor has no GPU kernel. On a constant it is evaluated once while the plan is made and never runs, so RequireGpu does not name it;
    // on the data of an input it does run, and creation fails naming it.
    [Fact]
    public void RequireGpuDoesNotNameAnOperatorThatFoldsAway()
    {
        GpuTestHost.Require();
        byte[] folded = new ModelBuilder()
            .Input("x", 1, "N", F).Output("y", 1, "N", F)
            .Initializer("w1", GpuTestHost.Random(F * F, 1), F, F).Initializer("c", GpuTestHost.Random(F, 2, 8f), F)
            .Node("Floor", ["c"], ["cf"]).Node("MatMul", ["x", "w1"], ["m"]).Node("Add", ["m", "cf"], ["y"]).Build();
        using Session s = GpuTestHost.Load(folded, GpuTestHost.Gpu(require: true));
        var feed = Feed(6, out Tensor x);
        using (x)
        {
            SessionDiagnostics d = s.GetDiagnostics(feed);
            Assert.Equal(0, d.NodesOnHost);
            Assert.Equal(1, d.DeviceRegions);
            using SessionOutputs o = s.Run(feed);
        }

        SessionException e = Assert.Throws<SessionException>(() => GpuTestHost.Load(Mixed(), GpuTestHost.Gpu(require: true)));
        Assert.Contains("Floor", e.Operators);
    }
}
