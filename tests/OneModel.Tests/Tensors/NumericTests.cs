using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Tensors;

public class NumericTests
{
    [Fact]
    public void OnnxCodesMapAndUnsupportedOnesAreNamed()
    {
        Assert.Equal(DType.F32, DTypes.FromOnnx(1));
        Assert.Equal(DType.BF16, DTypes.FromOnnx(16));
        Assert.Equal(DType.Bool, DTypes.FromOnnx(9));
        foreach (DType t in Enum.GetValues<DType>())
        {
            if (t != DType.Undefined)
            {
                Assert.Equal(t, DTypes.FromOnnx(DTypes.ToOnnx(t)));
            }
        }

        NotSupportedException e = Assert.Throws<NotSupportedException>(() => DTypes.FromOnnx(8));
        Assert.Contains("STRING", e.Message, StringComparison.Ordinal);
        e = Assert.Throws<NotSupportedException>(() => DTypes.FromOnnx(17));
        Assert.Contains("FLOAT8E4M3FN", e.Message, StringComparison.Ordinal);
        Assert.Equal("UNKNOWN(99)", DTypes.OnnxName(99));
        Assert.Equal(2, DTypes.SizeOf(DType.BF16));
        Assert.Equal(DType.F16, DTypes.Of<Half>());
        Assert.Equal(DType.BF16, DTypes.Of<BFloat16>());
    }

    [Fact]
    public void Float16ToSingleIsExactForAllBitPatterns()
    {
        for (int i = 0; i <= ushort.MaxValue; i++)
        {
            ushort h = (ushort)i;
            float ours = Float16.ToSingle(h);
            float bcl = (float)BitConverter.UInt16BitsToHalf(h);
            if (float.IsNaN(bcl))
            {
                Assert.True(float.IsNaN(ours));
                uint expected = ((uint)(h & 0x8000) << 16) | 0x7FC00000u | ((uint)(h & 0x3FF) << 13);
                Assert.Equal(expected, BitConverter.SingleToUInt32Bits(ours));
            }
            else
            {
                Assert.Equal(BitConverter.SingleToUInt32Bits(bcl), BitConverter.SingleToUInt32Bits(ours));
            }
        }
    }

    [Fact]
    public void Float16RoundTripsAllPatterns()
    {
        for (int i = 0; i <= ushort.MaxValue; i++)
        {
            ushort h = (ushort)i;
            ushort back = Float16.FromSingle(Float16.ToSingle(h));
            bool nan = (h & 0x7C00) == 0x7C00 && (h & 0x3FF) != 0;
            Assert.Equal(nan ? (ushort)(h | 0x200) : h, back);
        }
    }

    // Independent reference: the sorted list of non-negative finite halves as doubles; pick the nearer neighbour exactly,
    // ties to the even bit pattern, overflow at the midpoint 65520 to infinity.
    private static readonly double[] s_halves = Enumerable.Range(0, 0x7C00).Select(b => (double)Float16.ToSingle((ushort)b)).ToArray();

    private static ushort ReferenceHalf(float x)
    {
        if (float.IsNaN(x))
        {
            return (ushort)(((BitConverter.SingleToUInt32Bits(x) >> 16) & 0x8000) | 0x7E00 | ((BitConverter.SingleToUInt32Bits(x) >> 13) & 0x1FF));
        }

        ushort sign = (ushort)(BitConverter.SingleToUInt32Bits(x) >> 16 & 0x8000);
        double a = Math.Abs((double)x);
        if (double.IsInfinity(a))
        {
            return (ushort)(sign | 0x7C00);
        }

        int idx = Array.BinarySearch(s_halves, a);
        if (idx >= 0)
        {
            return (ushort)(sign | idx);
        }

        int hi = ~idx; // first element greater than a
        if (hi >= s_halves.Length)
        {
            // Above the largest finite half: 65520 is the tie with 65536 (even mantissa side is infinity).
            return (ushort)(sign | (a >= 65520.0 ? 0x7C00 : 0x7BFF));
        }

        int lo = hi - 1;
        double dl = a - s_halves[lo], dh = s_halves[hi] - a;
        int pick = dl < dh ? lo : dh < dl ? hi : ((lo & 1) == 0 ? lo : hi);
        return (ushort)(sign | pick);
    }

    [Fact]
    public void Float16FromSingleRoundsToNearestEvenOnSweeps()
    {
        var inputs = new List<float>();
        for (int b = 0; b < 0x7BFF; b++)
        {
            double mid = (s_halves[b] + s_halves[b + 1]) / 2; // exact in float: 12 significant bits at most
            float m = (float)mid;
            inputs.Add(m);
            inputs.Add(MathF.BitIncrement(m));
            inputs.Add(MathF.BitDecrement(m));
            inputs.Add((float)s_halves[b]);
        }

        inputs.AddRange([0f, -0f, 65504f, 65519.996f, 65520f, 65521f, 1e9f, float.MaxValue, float.PositiveInfinity, float.NegativeInfinity,
            5.960464477539063e-8f, 2.9802322387695312e-8f, 2.98023246e-8f, 1e-10f, float.Epsilon, float.NaN]);
        var rng = new Random(12345);
        for (int i = 0; i < 2_000_000; i++)
        {
            inputs.Add(BitConverter.UInt32BitsToSingle((uint)rng.NextInt64(0, 1L << 32)));
        }

        foreach (float x in inputs)
        {
            float v = x;
            Assert.True(ReferenceHalf(v) == Float16.FromSingle(v), $"x bits {BitConverter.SingleToUInt32Bits(v):X8}");
            if (!float.IsNaN(v))
            {
                Assert.Equal(BitConverter.HalfToUInt16Bits((Half)v), Float16.FromSingle(v)); // agrees with the BCL too
            }
        }

        Assert.Equal(0x8000, Float16.FromSingle(-0f));
        Assert.Equal(0x7E01 | 0x0000, Float16.FromSingle(BitConverter.UInt32BitsToSingle(0x7F802000u)));
    }

    [Fact]
    public void BFloat16RoundsToNearestEvenAndKeepsSpecials()
    {
        for (int i = 0; i <= ushort.MaxValue; i++)
        {
            ushort b = (ushort)i;
            float f = BFloat16.ToSingleBits(b);
            bool nan = (b & 0x7F80) == 0x7F80 && (b & 0x7F) != 0;
            Assert.Equal(nan ? (ushort)(b | 0x40) : b, BFloat16.FromSingleBits(f));
        }

        var rng = new Random(777);
        for (int i = 0; i < 2_000_000; i++)
        {
            uint bits = (uint)rng.NextInt64(0, 1L << 32);
            if (i % 4 == 0)
            {
                bits = (bits & 0xFFFF0000u) | 0x8000u; // exact ties
            }

            float x = BitConverter.UInt32BitsToSingle(bits);
            ushort got = BFloat16.FromSingleBits(x);
            if (float.IsNaN(x))
            {
                Assert.True(float.IsNaN(BFloat16.ToSingleBits(got)));
                Assert.Equal(bits >> 16 | 0x40, (uint)got);
                continue;
            }

            // Reference: the two truncation neighbours, nearer wins exactly, ties to the even pattern.
            ushort lo = (ushort)(bits >> 16);
            ushort hi = (ushort)(lo + 1);
            double dl = Math.Abs((double)x - BFloat16.ToSingleBits(lo));
            double dh = Math.Abs((double)BFloat16.ToSingleBits(hi) - x);
            bool hiInfinite = float.IsInfinity(BFloat16.ToSingleBits(hi)) && (bits & 0x7FFFFFFF) >= 0x7F7F8000;
            ushort expected;
            if (hiInfinite)
            {
                // Between the largest finite value and infinity the halfway point is the tie; the tie goes to the even side (infinity).
                expected = (bits & 0xFFFF) >= 0x8000 ? hi : lo;
            }
            else
            {
                expected = dl < dh ? lo : dh < dl ? hi : ((lo & 1) == 0 ? lo : hi);
            }

            Assert.Equal(expected, got);
        }

        Assert.Equal(0x7F80, BFloat16.FromSingleBits(float.PositiveInfinity));
        Assert.Equal(0x7F80, BFloat16.FromSingleBits(float.MaxValue));
        Assert.Equal(0x8000, BFloat16.FromSingleBits(-0f));
        Assert.Equal((ushort)0x0000, BFloat16.FromSingleBits(BitConverter.UInt32BitsToSingle(0x00008000u))); // subnormal tie to even (0)
        Assert.Equal((ushort)0x0002, BFloat16.FromSingleBits(BitConverter.UInt32BitsToSingle(0x00018000u))); // tie to even (2)
    }

    [Fact]
    public void CastFollowsDocumentedRules()
    {
        using Tensor f = Tensor.FromSpan<float>([1.9f, -1.9f, float.NaN, 300f, -300f, 0.5f, float.PositiveInfinity], Shape.Create(7));
        using Tensor i8 = f.Cast(DType.I8);
        Assert.Equal(new sbyte[] { 1, -1, 0, 127, -128, 0, 127 }, i8.ToArray<sbyte>());
        using Tensor u8 = f.Cast(DType.U8);
        Assert.Equal(new byte[] { 1, 0, 0, 255, 0, 0, 255 }, u8.ToArray<byte>());
        using Tensor b = f.Cast(DType.Bool);
        Assert.All(b.ToArray<bool>(), Assert.True);
        using Tensor wide = Tensor.FromSpan<int>([-1, 300, int.MinValue], Shape.Create(3));
        using Tensor wrapped = wide.Cast(DType.U8);
        Assert.Equal(new byte[] { 255, 44, 0 }, wrapped.ToArray<byte>());
        using Tensor h = f.Cast(DType.F16);
        using Tensor back = h.Cast(DType.F32);
        Assert.Equal(1.900390625f, back.ToArray<float>()[0]);
        using Tensor bf = f.Cast(DType.BF16);
        Assert.Equal(DType.BF16, bf.DType);
    }
}
