using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Onnx;

namespace RecursiveAI.OneModel.Tests.Graph.Operators.Norms;

public class NormAllocationTests
{
    private static long AllocatedBytes(Action loop)
    {
        // Best of three, as in the pool tests: runtime recompilation on the test thread is not the code under test.
        long best = long.MaxValue;
        for (int attempt = 0; attempt < 3 && best != 0; attempt++)
        {
            long before = GC.GetAllocatedBytesForCurrentThread();
            loop();
            best = Math.Min(best, GC.GetAllocatedBytesForCurrentThread() - before);
        }

        return best;
    }

    public static IEnumerable<object[]> Cases()
    {
        var rng = new Random(9);
        float[] Rand(int n) => Enumerable.Range(0, n).Select(_ => (float)rng.NextDouble()).ToArray();
        yield return ["LayerNormalization", new ModelBuilder(17).Input("x", 1, 4, 10, 48).Initializer("s", Rand(48), 48).Initializer("b", Rand(48), 48)
            .Output("y", 1, 4, 10, 48).Node("LayerNormalization", ["x", "s", "b"], ["y"]).Build(), new[] { 4, 10, 48 }, 1920];
        yield return ["Softmax last", new ModelBuilder(13).Input("x", 1, 4, 10, 48).Output("y", 1, 4, 10, 48).Node("Softmax", ["x"], ["y"]).Build(), new[] { 4, 10, 48 }, 1920];
        yield return ["Softmax middle", new ModelBuilder(13).Input("x", 1, 4, 10, 48).Output("y", 1, 4, 10, 48).Node("Softmax", ["x"], ["y"], ("axis", 1L)).Build(), new[] { 4, 10, 48 }, 1920];
        yield return ["LogSoftmax", new ModelBuilder(13).Input("x", 1, 4, 10, 48).Output("y", 1, 4, 10, 48).Node("LogSoftmax", ["x"], ["y"]).Build(), new[] { 4, 10, 48 }, 1920];
        yield return ["ReduceMean last", new ModelBuilder(17).Input("x", 1, 4, 10, 48).Output("y", 1, 4, 10, 1).Node("ReduceMean", ["x"], ["y"], ("axes", new long[] { -1 })).Build(), new[] { 4, 10, 48 }, 40];
        yield return ["ReduceMax first", new ModelBuilder(17).Input("x", 1, 4, 10, 48).Output("y", 1, 1, 10, 48).Node("ReduceMax", ["x"], ["y"], ("axes", new long[] { 0 })).Build(), new[] { 4, 10, 48 }, 480];
        yield return ["ReduceSum two runs", new ModelBuilder(13).Input("x", 1, 4, 10, 48).Output("y", 1, 10).Node("ReduceSum", ["x"], ["y"], ("axes", new long[] { 0, 2 }), ("keepdims", 0L)).Build(), new[] { 4, 10, 48 }, 10];
        yield return ["GroupNormalization", new ModelBuilder(21).Input("x", 1, 2, 8, 30).Initializer("s", Rand(8), 8).Initializer("b", Rand(8), 8)
            .Output("y", 1, 2, 8, 30).Node("GroupNormalization", ["x", "s", "b"], ["y"], ("num_groups", 2L)).Build(), new[] { 2, 8, 30 }, 480];
        yield return ["InstanceNormalization", new ModelBuilder(6).Input("x", 1, 2, 8, 30).Initializer("s", Rand(8), 8).Initializer("b", Rand(8), 8)
            .Output("y", 1, 2, 8, 30).Node("InstanceNormalization", ["x", "s", "b"], ["y"]).Build(), new[] { 2, 8, 30 }, 480];
        yield return ["CumSum", new ModelBuilder(14).Input("x", 1, 4, 10, 48).Initializer("a", new long[] { 1 }).Output("y", 1, 4, 10, 48).Node("CumSum", ["x", "a"], ["y"]).Build(), new[] { 4, 10, 48 }, 1920];
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void SteadyStateRunAllocatesNothing(string name, byte[] model, int[] shape, int outLength)
    {
        _ = name;
        float[] input = new float[shape.Aggregate(1, (a, b) => a * b)];
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = (i * 37 % 101) / 50f;
        }

        float[] y = new float[outLength];
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            using IrModel ir = OnnxImporter.Load(model);
            using var s = Session.Create(ir, new ModelOptions { Mode = mode, IntraOpThreads = 1 }, OpRegistry.Default, null, default);
            for (int i = 0; i < 20; i++)
            {
                s.Run("x", input.AsSpan(), shape, "y", y.AsSpan());
            }

            Assert.Equal(0, AllocatedBytes(() =>
            {
                for (int i = 0; i < 200; i++)
                {
                    s.Run("x", input.AsSpan(), shape, "y", y.AsSpan());
                }
            }));
        }
    }
}
