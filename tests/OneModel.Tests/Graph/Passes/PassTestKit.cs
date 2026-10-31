using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Onnx;

namespace RecursiveAI.OneModel.Tests.Graph.Passes;

internal static class PassTestKit
{
    public static float[] RandomFloats(int n, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var v = new float[n];
        for (int i = 0; i < n; i++)
        {
            v[i] = ((float)rng.NextDouble() * 2f - 1f) * scale;
        }

        return v;
    }

    /// <summary>Runs a one-input, one-output float model at a level and returns its output.</summary>
    public static float[] Run(byte[] model, float[] input, int[] shape, OptimizationLevel level, ArithmeticMode mode, int threads = 1)
    {
        using var m = Model.Load(new MemoryStream(model), new ModelOptions { OptimizationLevel = level, Mode = mode, IntraOpThreads = threads });
        return m.Run(input, shape);
    }

    /// <summary>The kernels of the nodes that remain after the passes of a level (in node order).</summary>
    public static string[] Kernels(byte[] model, OptimizationLevel level, ArithmeticMode mode)
    {
        using IrModel ir = OnnxImporter.Load(new MemoryStream(model), null, default);
        using CompiledGraph g = CompiledGraph.Create(ir.Graph, OpRegistry.Default, mode, WeightStorage.Original, level, KernelPaths.Best(), 1, default);
        return [.. g.Nodes.Select(n => n.Kernel.GetType().Name)];
    }

    public static bool BitEqual(float[] a, float[] b) =>
        a.Length == b.Length && a.AsSpan().SequenceEqual(b) && Enumerable.Range(0, a.Length).All(i => BitConverter.SingleToInt32Bits(a[i]) == BitConverter.SingleToInt32Bits(b[i]));
}
