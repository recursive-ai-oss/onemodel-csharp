using RecursiveAI.OneModel.Kernels;

namespace RecursiveAI.OneModel.Tests.Kernels;

/// <summary>The vector quantization loops (128 and 256 bit, any length and tail) against the scalar definition.</summary>
public class Int8QuantizeVectorTests
{
    private static float[] Data(int n, int seed)
    {
        var rng = new Random(seed);
        var x = new float[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = (float)((rng.NextDouble() * 2 - 1) * 40);
            switch (rng.Next(12))
            {
                case 0: x[i] = (float)Math.Round(x[i] * 2) / 2; break; // exact ties after division by 0.5
                case 1: x[i] = float.NaN; break;
                case 2: x[i] = rng.Next(2) == 0 ? float.PositiveInfinity : float.NegativeInfinity; break;
                case 3: x[i] = 0f; break;
            }
        }

        return x;
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void QuantizeMatchesTheScalarDefinitionAtEveryLength(bool isUnsigned)
    {
        int lo = isUnsigned ? 0 : -128, hi = isUnsigned ? 255 : 127;
        foreach (float scale in new[] { 0.5f, 0.1234f, 3f })
        {
            for (int n = 0; n <= 150; n++)
            {
                float[] x = Data(n, n + 1);
                var y = new byte[n];
                Int8Quantize.Quantize8(x, y, scale, isUnsigned ? 7 : 0, isUnsigned);
                for (int i = 0; i < n; i++)
                {
                    byte want = (byte)Int8Quantize.QuantizeOne(x[i], scale, isUnsigned ? 7 : 0, lo, hi);
                    Assert.True(want == y[i], $"n={n} i={i} x={x[i]:R} scale={scale} want {want} got {y[i]}");
                }
            }
        }
    }

    [Fact]
    public void MinMaxIgnoresNaNAndMatchesTheScalarDefinitionAtEveryLength()
    {
        for (int n = 0; n <= 300; n++)
        {
            float[] x = Data(n, 1000 + n);
            float wantLo = float.PositiveInfinity, wantHi = float.NegativeInfinity;
            foreach (float v in x)
            {
                if (v < wantLo)
                {
                    wantLo = v;
                }

                if (v > wantHi)
                {
                    wantHi = v;
                }
            }

            Int8Quantize.MinMax(x, out float lo, out float hi);
            Assert.True(lo == wantLo && hi == wantHi, $"n={n}: min {lo} max {hi}, want {wantLo} {wantHi}");
        }
    }
}
