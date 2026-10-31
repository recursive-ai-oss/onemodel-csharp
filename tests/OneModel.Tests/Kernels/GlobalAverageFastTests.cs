using RecursiveAI.OneModel.Kernels;

namespace RecursiveAI.OneModel.Tests.Kernels;

public class GlobalAverageFastTests
{
    [Fact]
    public void AgreesWithADoubleSumAndIsRepeatable()
    {
        var rng = new Random(5);
        foreach (int plane in new[] { 1, 2, 7, 15, 16, 17, 31, 49, 64, 100, 196, 784, 3136 })
        {
            const int planes = 5;
            var x = new float[planes * plane];
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = ((float)rng.NextDouble() * 4f) - 1f;
            }

            var y = new float[planes];
            var again = new float[planes];
            Pooling.GlobalAverageFast(x, y, plane);
            Pooling.GlobalAverageFast(x, again, plane);
            Assert.Equal(y, again);
            for (int p = 0; p < planes; p++)
            {
                double sum = 0;
                for (int i = 0; i < plane; i++)
                {
                    sum += x[(p * plane) + i];
                }

                Assert.Equal(sum / plane, y[p], 1e-5);
            }
        }
    }

    [Fact]
    public void PlanesUnderSixteenElementsSumInIndexOrder()
    {
        var x = new float[] { 1e8f, 1f, -1e8f, 1f, 3f };
        var fast = new float[1];
        var exact = new float[1];
        Pooling.GlobalAverageFast(x, fast, 5);
        Pooling.GlobalAverage(x, exact, 5);
        Assert.Equal(BitConverter.SingleToUInt32Bits(exact[0]), BitConverter.SingleToUInt32Bits(fast[0]));
    }
}
