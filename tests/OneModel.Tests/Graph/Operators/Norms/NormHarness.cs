using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Graph.Operators.Norms;

/// <summary>Runs one-node models through the whole stack and returns the outputs.</summary>
internal static class NormHarness
{
    public sealed record Out(Shape Shape, float[]? F, long[]? L);

    public static Out[] Run(byte[] model, ArithmeticMode mode, KernelPath? path, int threads, int outputs, params (float[] Data, int[] Shape)[] inputs)
    {
        using IrModel ir = OnnxImporter.Load(model);
        using var s = Session.Create(ir, new ModelOptions { Mode = mode, IntraOpThreads = threads }, OpRegistry.Default, path, default);
        var tensors = new Tensor[inputs.Length];
        var names = new string[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            tensors[i] = Tensor.FromSpan<float>(inputs[i].Data, Shape.Create(inputs[i].Shape));
            names[i] = s.Inputs[i].Name;
        }

        var outNames = new string[outputs];
        for (int i = 0; i < outputs; i++)
        {
            outNames[i] = s.Outputs[i].Name;
        }

        var outs = new Tensor[outputs];
        s.Run(names, tensors, outNames, outs);
        var result = new Out[outputs];
        for (int i = 0; i < outputs; i++)
        {
            result[i] = outs[i].DType == DType.I64
                ? new Out(outs[i].Shape, null, outs[i].ToArray<long>())
                : new Out(outs[i].Shape, outs[i].ToArray<float>(), null);
            outs[i].Dispose();
        }

        foreach (Tensor t in tensors)
        {
            t.Dispose();
        }

        return result;
    }

    public static IEnumerable<KernelPath> Paths => KernelPaths.All.Where(KernelPaths.IsSupported);

    public static float[] Random(Random rng, int n, double lo = -2, double hi = 2)
    {
        float[] r = new float[n];
        for (int i = 0; i < n; i++)
        {
            r[i] = (float)(lo + (rng.NextDouble() * (hi - lo)));
        }

        return r;
    }

    public static int Count(int[] s) => s.Aggregate(1, (a, b) => a * b);

    public static uint Bits(float f) => BitConverter.SingleToUInt32Bits(f);

    /// <summary>max |got - want| / max |want| (the metric of the oracle notes).</summary>
    public static double ScaledError(float[] got, double[] want)
    {
        Assert.Equal(want.Length, got.Length);
        double max = 0, scale = 1e-30;
        for (int i = 0; i < want.Length; i++)
        {
            if (double.IsNaN(want[i]) || double.IsInfinity(want[i]))
            {
                Assert.True(float.IsNaN(got[i]) == double.IsNaN(want[i]) && (double.IsNaN(want[i]) || got[i] == (float)want[i]), $"element {i}: {got[i]} vs {want[i]}");
                continue;
            }

            max = Math.Max(max, Math.Abs(got[i] - want[i]));
            scale = Math.Max(scale, Math.Abs(want[i]));
        }

        return max / scale;
    }

    public static void AssertSameBits(float[] a, float[] b, string what)
    {
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
        {
            Assert.True(Bits(a[i]) == Bits(b[i]), $"{what}: element {i}: {Bits(a[i]):x8} vs {Bits(b[i]):x8}");
        }
    }
}
