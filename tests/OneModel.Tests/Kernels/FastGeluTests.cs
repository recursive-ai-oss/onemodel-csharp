using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Kernels.Exact;

namespace RecursiveAI.OneModel.Tests.Kernels;

/// <summary>The fast-mode erf Gelu (a rational fit, no exponential) against the exact-mode kernel, which is within a few ulp of the true value.</summary>
public class FastGeluTests
{
    private static float[] Inputs()
    {
        var xs = new List<float>();
        for (double v = -16; v <= 16; v += 0.00037)
        {
            xs.Add((float)v);
        }

        xs.AddRange([0f, -0f, 1e-30f, -1e-30f, 1e-45f, 1e30f, -1e30f, 5.6f, -5.6f, 5.6001f, -5.6001f, float.MaxValue, float.MinValue]);
        return [.. xs];
    }

    [Fact]
    public void AgreesWithTheExactKernelOnEveryPath()
    {
        float[] x = Inputs();
        var fast = new float[x.Length];
        var exact = new float[x.Length];
        foreach (KernelPath path in KernelPaths.All)
        {
            if (!KernelPaths.IsSupported(path))
            {
                continue;
            }

            FloatUnary.Run(path, false, new GeluUnary(false), x, fast);
            FloatUnary.Run(path, true, new GeluUnary(false), x, exact);
            for (int i = 0; i < x.Length; i++)
            {
                // Absolute error relative to the larger of 1 and |x|: 2.4e-7 measured; the limit leaves room for other machines.
                double err = Math.Abs((double)fast[i] - exact[i]) / Math.Max(1.0, Math.Abs(x[i]));
                Assert.True(err <= 1e-6, $"{path}: gelu({x[i]:R}) fast {fast[i]:R} exact {exact[i]:R}");
            }
        }
    }

    [Fact]
    public void SpecialValuesFollowTheExactKernel()
    {
        float inf = float.PositiveInfinity;
        float[] x = [inf, float.NegativeInfinity, float.NaN, 0f, -0f];
        var fast = new float[x.Length];
        FloatUnary.Run(KernelPaths.Best(), false, new GeluUnary(false), x, fast);
        Assert.Equal(inf, fast[0]);
        Assert.Equal(BitConverter.SingleToInt32Bits(-0f), BitConverter.SingleToInt32Bits(fast[1]));
        Assert.True(float.IsNaN(fast[2]));
        Assert.Equal(0f, fast[3]);
        Assert.Equal(0f, fast[4]);
    }

    [Fact]
    public void ExactModeStillUsesTheAccurateKernel()
    {
        // The exact kernel keeps the relative accuracy of the negative tail: gelu(-4) = -4 * 0.5 * erfc(4/sqrt 2) = -1.2668e-4.
        var y = new float[1];
        FloatUnary.Run(KernelPaths.Best(), true, new GeluUnary(false), [-4f], y);
        Assert.InRange(y[0], -1.26696e-4f, -1.26680e-4f);
    }
}
