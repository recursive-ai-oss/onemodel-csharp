using RecursiveAI.OneModel.Kernels;

namespace RecursiveAI.OneModel.Tests.Kernels;

/// <summary>
/// The vector addition and product of quantized tensors give the scalar chain's bytes for every pair of input bytes, including the
/// sums that fall exactly on, or within an ulp of, a rounding boundary (where the reciprocal shortcut must give way to the division).
/// </summary>
public class Int8ElementwiseTests
{
    public static IEnumerable<object[]> Params()
    {
        // (a scale, a zero, b scale, b zero, y scale, y zero): halves and quarters land sums exactly on rounding ties.
        yield return [0.5f, 3, 0.5f, -7, 1f, 0];
        yield return [0.25f, 0, 0.5f, 9, 1f, 5];
        yield return [0.125f, -5, 0.375f, 2, 0.25f, -3];
        yield return [0.1f, 11, 0.07f, 0, 0.1f, 4];
        yield return [0.031f, 5, 0.0173f, -9, 0.047f, 3];
        yield return [1f, 0, 1f, 0, 3f, 0];
        yield return [3f, 1, 5f, 2, 7f, 0];
        yield return [0.5f, 0, 0.5f, 0, 1e-39f, 0];
    }

    [Fact]
    public void EveryLengthEqualsTheScalarChain()
    {
        // Lengths around the vector widths (8, 16, 32 lanes) exercise every tail.
        var rng = new Random(3);
        foreach (bool aU in new[] { false, true })
        {
            var p = new Int8AddParams(aU, 5, 0.031f, !aU, -9, 0.0173f, aU, 3, 0.047f);
            for (int n = 0; n <= 100; n++)
            {
                var a = new byte[n];
                var b = new byte[n];
                rng.NextBytes(a);
                rng.NextBytes(b);
                var y = new byte[n];
                Int8Elementwise.Add8(a, b, y, in p, 1);
                int lo = aU ? 0 : -128, hi = aU ? 255 : 127;
                for (int i = 0; i < n; i++)
                {
                    Assert.True((byte)Int8Elementwise.AddOne(a[i], b[i], in p, lo, hi) == y[i], $"n={n} i={i}");
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(Params))]
    public void AdditionEqualsTheScalarChain(float sa, int za, float sb, int zb, float sy, int zy)
    {
        foreach (bool aU in new[] { false, true })
        {
            foreach (bool yU in new[] { false, true })
            {
                var p = new Int8AddParams(aU, za, sa, !aU, zb, sb, yU, zy, sy);
                var a = new byte[65536];
                var b = new byte[65536];
                for (int i = 0; i < 65536; i++)
                {
                    a[i] = (byte)(i >> 8);
                    b[i] = (byte)i;
                }

                var y = new byte[65536];
                Int8Elementwise.Add8(a, b, y, in p, 1);
                int lo = yU ? 0 : -128, hi = yU ? 255 : 127;
                for (int i = 0; i < 65536; i++)
                {
                    Assert.True((byte)Int8Elementwise.AddOne(a[i], b[i], in p, lo, hi) == y[i], $"a={a[i]} b={b[i]} aU={aU} yU={yU}");
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(Params))]
    public void RowProductEqualsTheScalarChain(float sa, int za, float sb, int zb, float sy, int zy)
    {
        foreach (bool aU in new[] { false, true })
        {
            var p = new Int8AddParams(aU, za, sa, !aU, zb, sb, !aU, zy, sy);
            const int rows = 256, inner = 37;
            var a = new byte[rows * inner];
            var b = new byte[rows];
            for (int i = 0; i < a.Length; i++)
            {
                a[i] = (byte)(i * 7);
            }

            for (int i = 0; i < rows; i++)
            {
                b[i] = (byte)i;
            }

            var y = new byte[a.Length];
            Int8ChannelMul.Run(a, b, y, rows, inner, in p, 1);
            int lo = p.YUnsigned ? 0 : -128, hi = p.YUnsigned ? 255 : 127;
            for (int i = 0; i < a.Length; i++)
            {
                Assert.True((byte)Int8Elementwise.MulOne(a[i], b[i / inner], in p, lo, hi) == y[i], $"i={i} aU={aU}");
            }
        }
    }
}
