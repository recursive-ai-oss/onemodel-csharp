using System.Buffers.Binary;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Graph;

public class HalfPrecisionTests
{
    private static void Check(int seeds)
    {
        // Every binary16 pattern widens to the scalar answer.
        var all = new ushort[65536];
        for (int i = 0; i < all.Length; i++)
        {
            all[i] = (ushort)i;
        }

        var wide = new float[all.Length];
        HalfConvert.HalfToSingle(all, wide);

        for (int i = 0; i < all.Length; i++)
        {
            Assert.Equal(BitConverter.SingleToUInt32Bits(Float16.ToSingle(all[i])), BitConverter.SingleToUInt32Bits(wide[i]));
        }

        HalfConvert.BFloatToSingle(all, wide);

        for (int i = 0; i < all.Length; i++)
        {
            Assert.Equal(BitConverter.SingleToUInt32Bits(BFloat16.ToSingleBits(all[i])), BitConverter.SingleToUInt32Bits(wide[i]));
        }

        // Narrowing: edge patterns around every rounding boundary plus random bit patterns, at odd lengths so tails run.
        var rng = new Random(seeds);
        var bits = new List<uint>();
        foreach (uint e in new uint[] { 0, 0x33000000, 0x38000000, 0x38800000, 0x477FE000, 0x477FF000, 0x47800000, 0x7F7FFFFF, 0x7F800000, 0x7FC00000, 0x7F800001 })
        {
            for (uint d = 0; d < 40; d++)
            {
                bits.Add(e + d);
                bits.Add(e - d);
                bits.Add(0x80000000u | (e + d));
            }
        }

        for (int i = 0; i < 4_000_000; i++)
        {
            bits.Add((uint)rng.NextInt64(0, 1L << 32));
        }

        // Values that land exactly on half and bfloat16 tie points: a 16-bit pattern with the next bit set and nothing below.
        for (int i = 0; i < 200_000; i++)
        {
            bits.Add(((uint)rng.Next(0, 65536) << 16) | 0x8000u);
            bits.Add(((uint)rng.Next(0, 65536) << 16) | 0x7FFFu);
            bits.Add(((uint)rng.Next(0, 1 << 13) << 13) | 0x1000u | ((uint)rng.Next(0, 0x7F) << 24));
        }

        var src = new float[bits.Count + 3];
        for (int i = 0; i < bits.Count; i++)
        {
            src[i] = BitConverter.UInt32BitsToSingle(bits[i]);
        }

        int n = bits.Count + 3;
        var narrow = new ushort[n];
        HalfConvert.SingleToHalf(src, narrow);

        for (int i = 0; i < n; i++)
        {
            Assert.True(Float16.FromSingle(src[i]) == narrow[i], $"binary16 of {BitConverter.SingleToUInt32Bits(src[i]):X8}");
        }

        HalfConvert.SingleToBFloat(src, narrow);

        for (int i = 0; i < n; i++)
        {
            Assert.True(BFloat16.FromSingleBits(src[i]) == narrow[i], $"bfloat16 of {BitConverter.SingleToUInt32Bits(src[i]):X8}");
        }
    }

    [Fact]
    public void BulkConversionsGiveTheScalarBits() => Check(11);

    private static float[] Weights(int n, int seed, float scale)
    {
        var rng = new Random(seed);
        var w = new float[n];
        for (int i = 0; i < n; i++)
        {
            w[i] = (float)((rng.NextDouble() - 0.5) * scale);
        }

        return w;
    }

    private static byte[] HalfBytes(float[] values, bool brain)
    {
        var raw = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteUInt16LittleEndian(raw.AsSpan(i * 2), brain ? BFloat16.FromSingleBits(values[i]) : Float16.FromSingle(values[i]));
        }

        return raw;
    }

    private static float[] Round(float[] values, bool brain)
    {
        var r = new float[values.Length];
        for (int i = 0; i < r.Length; i++)
        {
            r[i] = brain ? BFloat16.ToSingleBits(BFloat16.FromSingleBits(values[i])) : Float16.ToSingle(Float16.FromSingle(values[i]));
        }

        return r;
    }

    private const int In = 64, Mid = 2048, Out = 8;

    // x -> MatMul(w1) -> Add(b) -> Relu -> MatMul(w2) -> y, in float32 with the given (already rounded) weights.
    private static byte[] FloatModel(float[] w1, float[] b, float[] w2) =>
        new ModelBuilder().Input("x", 1, "N", In).Output("y", 1, "N", Out)
            .Initializer("w1", w1, In, Mid).Initializer("b", b, Mid).Initializer("w2", w2, Mid, Out)
            .Node("MatMul", ["x", "w1"], ["h"]).Node("Add", ["h", "b"], ["hb"]).Node("Relu", ["hb"], ["r"]).Node("MatMul", ["r", "w2"], ["y"])
            .Build();

    // The same network stored in half precision: weights in the file as binary16 or bfloat16, activations cast to that type after the input.
    private static byte[] HalfModel(float[] w1, float[] b, float[] w2, bool brain)
    {
        int t = brain ? 16 : 10;
        return new ModelBuilder().Input("x", 1, "N", In).Output("y", 1, "N", Out)
            .RawInitializer("w1", t, [In, Mid], HalfBytes(w1, brain)).RawInitializer("b", t, [Mid], HalfBytes(b, brain))
            .RawInitializer("w2", t, [Mid, Out], HalfBytes(w2, brain))
            .Node("Cast", ["x"], ["xh"], ("to", (long)t))
            .Node("MatMul", ["xh", "w1"], ["h"]).Node("Add", ["h", "b"], ["hb"]).Node("Relu", ["hb"], ["r"]).Node("MatMul", ["r", "w2"], ["yh"])
            .Node("Cast", ["yh"], ["y"], ("to", 1L))
            .Build();
    }

    private static float[] Run(byte[] model, float[] x, ModelOptions options)
    {
        using var m = Model.Load(new MemoryStream(model), options);
        return m.Run(x, [x.Length / In, In]);
    }

    [Theory]
    [InlineData(false, ArithmeticMode.Fast)]
    [InlineData(true, ArithmeticMode.Fast)]
    [InlineData(false, ArithmeticMode.Exact)]
    public void HalfPrecisionModelMatchesFloatModelWithRoundedWeights(bool brain, ArithmeticMode mode)
    {
        float[] w1 = Weights(In * Mid, 1, 0.2f), b = Weights(Mid, 2, 0.1f), w2 = Weights(Mid * Out, 3, 0.1f);
        float[] x = Weights(5 * In, 4, 2f);
        float[] reference = Run(FloatModel(Round(w1, brain), Round(b, brain), Round(w2, brain)), x, new ModelOptions { Mode = mode });
        byte[] half = HalfModel(w1, b, w2, brain);
        float[] got = Run(half, x, new ModelOptions { Mode = mode, IntraOpThreads = 1 });

        // The only difference is rounding each activation to half precision: 2^-11 (binary16) or 2^-8 (bfloat16) relative per operator.
        double max = reference.Max(Math.Abs), diff = 0;
        for (int i = 0; i < got.Length; i++)
        {
            diff = Math.Max(diff, Math.Abs((double)got[i] - reference[i]));
        }

        Assert.True(diff / max < (brain ? 3e-2 : 3e-3), $"scaled error {diff / max}");
        Assert.Equal(got, Run(half, x, new ModelOptions { Mode = mode, IntraOpThreads = 4 }));
    }

    [Theory]
    [InlineData(WeightStorage.Binary16)]
    [InlineData(WeightStorage.BFloat16)]
    public void CompactWeightsGiveTheResultOfRoundedWeights(WeightStorage storage)
    {
        bool brain = storage == WeightStorage.BFloat16;
        float[] w1 = Weights(In * Mid, 1, 0.2f), b = Weights(Mid, 2, 0.1f), w2 = Weights(Mid * Out, 3, 0.1f);
        float[] x = Weights(5 * In, 4, 2f);

        // Activations stay float32 and widening is exact, so the result equals the float32 model on weights rounded in advance, bit for bit.
        // (The bias is below the size limit and stays float32, so it is not rounded.)
        float[] reference = Run(FloatModel(Round(w1, brain), b, Round(w2, brain)), x, new ModelOptions { IntraOpThreads = 1 });
        float[] got = Run(FloatModel(w1, b, w2), x, new ModelOptions { IntraOpThreads = 1, WeightStorage = storage });
        Assert.Equal(reference, got);
        Assert.Equal(reference, Run(FloatModel(w1, b, w2), x, new ModelOptions { IntraOpThreads = 4, WeightStorage = storage }));
    }
}
