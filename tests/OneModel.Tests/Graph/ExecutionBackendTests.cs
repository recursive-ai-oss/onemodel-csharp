using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Graph;

/// <summary>
/// The execution-backend seam, exercised with a backend that claims a few operators and runs them through the built-in kernels:
/// results must be bit for bit those of a session without a backend, and the placement records must say what happened.
/// </summary>
public class ExecutionBackendTests
{
    // x -> MatMul -> Add(bias) -> Relu -> MatMul -> Relu -> y, so MatMul/Relu are claimed, Add is not, and the two claimed runs differ in size.
    private static byte[] BuildModel(int features)
    {
        var rng = new Random(11);
        float[] Weights(int n) => [.. Enumerable.Range(0, n).Select(_ => (float)rng.NextDouble() - 0.5f)];
        return new ModelBuilder()
            .Input("x", 1, "N", features).Output("y", 1, "N", features)
            .Initializer("w1", Weights(features * features), features, features)
            .Initializer("b", Weights(features), features)
            .Initializer("w2", Weights(features * features), features, features)
            .Node("MatMul", ["x", "w1"], ["m1"])
            .Node("Add", ["m1", "b"], ["a1"])
            .Node("Relu", ["a1"], ["r1"])
            .Node("MatMul", ["r1", "w2"], ["m2"])
            .Node("Relu", ["m2"], ["y"])
            .Build();
    }

    private static float[] Input(int rows, int features)
    {
        var rng = new Random(5);
        return [.. Enumerable.Range(0, rows * features).Select(_ => (float)rng.NextDouble() - 0.5f)];
    }

    private sealed class FakeBackend : IExecutionBackend
    {
        private readonly HashSet<string> _ops;
        public int Disposed;
        public int Compiled;
        public int RegionRuns;
        public BackendRegion[] LastRegions { get; private set; } = [];

        public FakeBackend(params string[] ops) => _ops = [.. ops];

        public string Name => "fake";

        public string? CheckNode(CompiledNode node, ArithmeticMode mode) =>
            _ops.Contains(node.Info.OpType) ? null : $"{node.Info.OpType} is not in the fake's list";

        public string? ClaimStep(ExecutionPlan plan, PlanStep step) =>
            _ops.Contains(step.Node.Info.OpType) ? null : "not claimed by the fake";

        public IBackendProgram Compile(ExecutionPlan plan, IReadOnlyList<BackendRegion> regions)
        {
            Compiled++;
            LastRegions = [.. regions];
            return new Program(this, plan, regions);
        }

        public void Dispose() => Interlocked.Increment(ref Disposed);

        private sealed class Program(FakeBackend owner, ExecutionPlan plan, IReadOnlyList<BackendRegion> regions) : IBackendProgram
        {
            public long BytesToBackendPerRun => regions.Sum(r => r.Inputs.Sum(id => (long)plan.Shapes[id].Length * 4));

            public long BytesFromBackendPerRun => regions.Sum(r => r.Outputs.Sum(id => (long)plan.Shapes[id].Length * 4));

            public void Run(int region, Tensor[] values)
            {
                Interlocked.Increment(ref owner.RegionRuns);
                using var arena = new Arena(plan.Env.Pool);
                foreach (PlanStep step in regions[region].Steps)
                {
                    CompiledNode n = step.Node;
                    var ins = new Tensor[n.Inputs.Length];
                    for (int i = 0; i < ins.Length; i++)
                    {
                        ins[i] = n.Inputs[i] >= 0 ? values[n.Inputs[i]] : default;
                    }

                    var outs = new Tensor[n.Outputs.Length];
                    for (int o = 0; o < outs.Length; o++)
                    {
                        outs[o] = n.Outputs[o] >= 0 ? values[n.Outputs[o]] : default;
                    }

                    n.Kernel.Run(new OpRun(ins, outs, plan.Env.Mode, plan.Env.Path, plan.Env.Threads, arena, step.State));
                }
            }

            public void Dispose()
            {
            }
        }
    }

    private static Session CreateSession(byte[] model, ModelOptions options, IExecutionBackend? backend)
    {
        using IrModel ir = OnnxImporter.Load(new MemoryStream(model));
        return Session.Create(ir, options, OpRegistry.Default, null, backend, default);
    }

    [Theory]
    [InlineData(ArithmeticMode.Fast, OptimizationLevel.None)]
    [InlineData(ArithmeticMode.Fast, OptimizationLevel.All)]
    [InlineData(ArithmeticMode.Exact, OptimizationLevel.None)]
    [InlineData(ArithmeticMode.Exact, OptimizationLevel.All)]
    public void ARunThroughABackendIsBitIdenticalToTheBuiltInPath(ArithmeticMode mode, OptimizationLevel level)
    {
        byte[] model = BuildModel(64);
        var options = new ModelOptions { Mode = mode, OptimizationLevel = level, IntraOpThreads = 2 };
        float[] x = Input(7, 64);
        float[] expected = new float[7 * 64], actual = new float[7 * 64];

        using (Session plain = CreateSession(model, options, null))
        {
            plain.Run("x", x.AsSpan(), [7, 64], "y", expected.AsSpan());
        }

        var fake = new FakeBackend("MatMul", "Relu");
        using (Session s = CreateSession(model, options, fake))
        {
            s.Run("x", x.AsSpan(), [7, 64], "y", actual.AsSpan());
            Array.Clear(actual);
            s.Run("x", x.AsSpan(), [7, 64], "y", actual.AsSpan());
        }

        Assert.Equal(SessionTests.Hash(expected), SessionTests.Hash(actual));
        Assert.True(fake.RegionRuns > 0);
        Assert.Equal(1, fake.Disposed);
    }

    [Fact]
    public void RegionsAreMaximalRunsAndPlacementsNameTheReasons()
    {
        byte[] model = BuildModel(32);
        var fake = new FakeBackend("MatMul", "Relu");
        using Session s = CreateSession(model, new ModelOptions { OptimizationLevel = OptimizationLevel.None }, fake);

        // Session creation: Add is the one node the backend can never run.
        NodeDecline d = Assert.Single(s.BackendDeclines);
        Assert.Equal("Add", d.OpType);
        Assert.Contains("fake", d.Reason);

        var (nodes, regions, toBackend, fromBackend) = s.PlacementOf([DType.F32], [Shape.Create([3, 32])]);
        Assert.Equal(["MatMul", "Add", "Relu", "MatMul", "Relu"], nodes.Select(n => n.OpType));
        Assert.Equal(["fake", "cpu", "fake", "fake", "fake"], nodes.Select(n => n.Backend));
        Assert.Equal(2, regions);
        Assert.Equal([0, -1, 1, 1, 1], nodes.Select(n => n.Region));
        Assert.Equal("not claimed by the fake", nodes[1].Reason);
        Assert.All(nodes.Where(n => n.Backend == "fake"), n => Assert.Null(n.Reason));

        // Region 0 reads the graph input and returns m1. Region 1 reads a1, keeps r1 and m2 to itself and returns y.
        BackendRegion r0 = fake.LastRegions[0], r1 = fake.LastRegions[1];
        Assert.Single(r0.Inputs);
        Assert.Single(r0.Outputs);
        Assert.Single(r1.Inputs);
        Assert.Single(r1.Outputs);
        Assert.Single(r0.Constants);
        Assert.Single(r1.Constants);
        Assert.Equal(2 * 3 * 32 * 4, toBackend);
        Assert.Equal(2 * 3 * 32 * 4, fromBackend);

        // A second shape signature gets its own plan and its own compile.
        s.PlacementOf([DType.F32], [Shape.Create([5, 32])]);
        Assert.Equal(2, fake.Compiled);
        Assert.Equal(2, s.CachedPlanCount);
    }

    [Fact]
    public void WithoutABackendEveryNodeIsOnTheHost()
    {
        using Session s = CreateSession(BuildModel(16), new ModelOptions(), null);
        var (nodes, regions, toBackend, fromBackend) = s.PlacementOf([DType.F32], [Shape.Create([2, 16])]);
        Assert.NotEmpty(nodes);
        Assert.All(nodes, n => Assert.Equal("cpu", n.Backend));
        Assert.All(nodes, n => Assert.Null(n.Reason));
        Assert.Equal(0, regions);
        Assert.Equal(0, toBackend + fromBackend);
        Assert.Empty(s.BackendDeclines);
    }

    [Fact]
    public void ABackendThatClaimsNothingLeavesNoRegionsAndStillOwnsItsLifetime()
    {
        var fake = new FakeBackend();
        using (Session s = CreateSession(BuildModel(16), new ModelOptions { OptimizationLevel = OptimizationLevel.None }, fake))
        {
            var (nodes, regions, _, _) = s.PlacementOf([DType.F32], [Shape.Create([2, 16])]);
            Assert.Equal(0, regions);
            Assert.All(nodes, n => Assert.Equal("not claimed by the fake", n.Reason));
            Assert.Equal(0, fake.Compiled);
        }

        Assert.Equal(1, fake.Disposed);
    }

    [Fact]
    public void CreationFailureDisposesTheBackend()
    {
        byte[] model = new ModelBuilder().Input("x", 1, 2).Output("y", 1, 2).Node("NoSuchOperator", ["x"], ["y"]).Build();
        var fake = new FakeBackend();
        Assert.Throws<SessionException>(() => CreateSession(model, new ModelOptions(), fake));
        Assert.Equal(1, fake.Disposed);
    }
}
