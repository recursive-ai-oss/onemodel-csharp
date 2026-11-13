using System.Runtime.InteropServices;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tests.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu.Tests;

/// <summary>
/// The integer and boolean operators around an attention mask, each against the CPU session bit for bit: integer arithmetic,
/// comparisons, logic, Where, Expand, Cast between every pair of element types, Gather of booleans, and the generators. The whole model
/// must run on the device, in one region.
/// </summary>
public class MaskTests
{
    private const int F32 = 1, I32 = 6, I64 = 7, Bool = 9, F16 = 10;

    private static object[] Dims(int[] s) => [.. s.Select(d => (object)d)];

    private static int Count(int[] s) => s.Aggregate(1, (a, b) => a * b);

    private static byte[] Raw(in Tensor t) => t.DType switch
    {
        DType.F32 => MemoryMarshal.AsBytes(t.Data<float>()).ToArray(),
        DType.F16 => MemoryMarshal.AsBytes(t.Data<Half>()).ToArray(),
        DType.I32 => MemoryMarshal.AsBytes(t.Data<int>()).ToArray(),
        DType.I64 => MemoryMarshal.AsBytes(t.Data<long>()).ToArray(),
        DType.Bool => MemoryMarshal.AsBytes(t.Data<bool>()).ToArray(),
        DType.U8 => t.Data<byte>().ToArray(),
        _ => throw new NotSupportedException(t.DType.ToString()),
    };

    private static byte[] Hash(byte[] b) => System.Security.Cryptography.SHA256.HashData(b);

    /// <summary>Runs the model on the CPU and on the device in both modes; every output must be bit-identical and the model must run on the device in one region.</summary>
    internal static void Same(byte[] model, Dictionary<string, Tensor> feed, string[] outputs, int regions = 1)
    {
        GpuTestHost.Require();
        foreach (ArithmeticMode mode in new[] { ArithmeticMode.Fast, ArithmeticMode.Exact })
        {
            var expected = new Dictionary<string, byte[]>();
            using (Session cpu = GpuTestHost.Load(model, GpuTestHost.Cpu(mode)))
            using (SessionOutputs o = cpu.Run(feed))
            {
                foreach (string name in outputs)
                {
                    expected[name] = Raw(o[name]);
                }
            }

            using Session gpu = GpuTestHost.Load(model, GpuTestHost.Gpu(mode, require: true));
            for (int run = 0; run < 2; run++)
            {
                using SessionOutputs o = gpu.Run(feed);
                foreach (string name in outputs)
                {
                    Assert.True(expected[name].AsSpan().SequenceEqual(Raw(o[name])), $"{mode} run {run}: output {name} differs from the CPU's");
                }
            }

            SessionDiagnostics d = gpu.GetDiagnostics(feed);
            Assert.Equal(regions, d.DeviceRegions);
            Assert.All(d.Nodes, n => Assert.Null(n.FallbackReason));
            GpuTestHost.AssertValidationClean(gpu);
        }
    }

    private static Tensor Ints(long[] v, int[] shape, bool wide) =>
        wide ? Tensor.FromSpan<long>(v, Shape.Create(shape)) : Tensor.FromSpan<int>([.. v.Select(x => (int)x)], Shape.Create(shape));

    // Values that hit the edges of each width, plus a spread of small signed ones and zeros.
    private static long[] Edge(int n, int seed, bool wide)
    {
        long[] special = wide
            ? [long.MinValue, long.MaxValue, -1, 0, 1, 2, -2, 7, -7, 1L << 40, -(1L << 40), int.MaxValue, int.MinValue, 4294967296L, -4294967297L]
            : [int.MinValue, int.MaxValue, -1, 0, 1, 2, -2, 7, -7, 65536, -65537, 46341];
        var rng = new Random(seed);
        return [.. Enumerable.Range(0, n).Select(_ => rng.Next(3) == 0 ? special[rng.Next(special.Length)] : rng.Next(-20, 21))];
    }

    // ---- arithmetic -------------------------------------------------------------------------------------------------------

    public static IEnumerable<object[]> ArithmeticCases()
    {
        foreach (string op in new[] { "Add", "Sub", "Mul", "Div", "Min", "Max", "Mod", "ModC" })
        {
            foreach (bool wide in new[] { false, true })
            {
                foreach ((int[] a, int[] b) in new (int[], int[])[] { ([3, 1, 5], [4, 1]), ([7], [7]), ([1], [2, 3]), ([2, 3, 4, 5], [5]), ([33], [1]) })
                {
                    yield return [op, wide, a, b];
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(ArithmeticCases))]
    public void IntegerArithmeticMatchesTheCpuBitForBit(string op, bool wide, int[] a, int[] b)
    {
        int type = wide ? I64 : I32;
        string name = op == "ModC" ? "Mod" : op;
        (string, object)[] attrs = op switch { "Mod" => [("fmod", 0L)], "ModC" => [("fmod", 1L)], _ => [] };
        byte[] model = new ModelBuilder().Input("a", type, Dims(a)).Input("b", type, Dims(b)).Output("y", type).Node(name, ["a", "b"], ["y"], attrs).Build();
        using Tensor ta = Ints(Edge(Count(a), 1, wide), a, wide), tb = Ints(Edge(Count(b), 2, wide), b, wide);
        Same(model, new() { ["a"] = ta, ["b"] = tb }, ["y"]);
    }

    [Theory]
    [InlineData("Neg", false)]
    [InlineData("Neg", true)]
    [InlineData("Abs", false)]
    [InlineData("Abs", true)]
    public void IntegerNegAndAbsMatchTheCpu(string op, bool wide)
    {
        int type = wide ? I64 : I32;
        byte[] model = new ModelBuilder().Input("x", type, 5, 9).Output("y", type).Node(op, ["x"], ["y"]).Build();
        using Tensor x = Ints(Edge(45, 3, wide), [5, 9], wide);
        Same(model, new() { ["x"] = x }, ["y"]);
    }

    // ---- comparisons, logic, Where ---------------------------------------------------------------------------------------

    private static Tensor Make(int type, int n, int seed, int[] shape)
    {
        switch (type)
        {
            case F32:
                {
                    var rng = new Random(seed);
                    float[] special = [0f, -0f, 1f, -1f, float.NaN, float.PositiveInfinity, float.NegativeInfinity, 0.5f, 1e-30f];
                    return Tensor.FromSpan<float>([.. Enumerable.Range(0, n).Select(_ => rng.Next(3) == 0 ? special[rng.Next(special.Length)] : (float)Math.Round(rng.NextDouble() * 4 - 2))], Shape.Create(shape));
                }

            case F16:
                {
                    var rng = new Random(seed);
                    Half[] special = [(Half)0f, (Half)1f, (Half)(-1f), Half.NaN, Half.PositiveInfinity, (Half)0.5f];
                    return Tensor.FromSpan<Half>([.. Enumerable.Range(0, n).Select(_ => rng.Next(3) == 0 ? special[rng.Next(special.Length)] : (Half)Math.Round(rng.NextDouble() * 4 - 2))], Shape.Create(shape));
                }

            case Bool:
                {
                    var rng = new Random(seed);
                    return Tensor.FromSpan<bool>([.. Enumerable.Range(0, n).Select(_ => rng.Next(2) == 0)], Shape.Create(shape));
                }

            default:
                return Ints(Edge(n, seed, type == I64), shape, type == I64);
        }
    }

    [Theory]
    [InlineData("Equal", F32)]
    [InlineData("Less", F32)]
    [InlineData("Greater", F32)]
    [InlineData("LessOrEqual", F32)]
    [InlineData("GreaterOrEqual", F32)]
    [InlineData("Equal", F16)]
    [InlineData("Less", F16)]
    [InlineData("Equal", I32)]
    [InlineData("Less", I32)]
    [InlineData("GreaterOrEqual", I32)]
    [InlineData("Equal", I64)]
    [InlineData("Less", I64)]
    [InlineData("Greater", I64)]
    [InlineData("LessOrEqual", I64)]
    [InlineData("Equal", Bool)]
    public void ComparisonsMatchTheCpu(string op, int type)
    {
        int[] a = [3, 1, 6], b = [5, 1];
        byte[] model = new ModelBuilder().Input("a", type, Dims(a)).Input("b", type, Dims(b)).Output("y", Bool).Node(op, ["a", "b"], ["y"]).Build();
        using Tensor ta = Make(type, Count(a), 4, a), tb = Make(type, Count(b), 5, b);
        Same(model, new() { ["a"] = ta, ["b"] = tb }, ["y"]);
    }

    [Theory]
    [InlineData("And")]
    [InlineData("Or")]
    [InlineData("Xor")]
    public void LogicMatchesTheCpu(string op)
    {
        int[] a = [4, 1, 7], b = [3, 1];
        byte[] model = new ModelBuilder().Input("a", Bool, Dims(a)).Input("b", Bool, Dims(b)).Output("y", Bool).Node(op, ["a", "b"], ["y"]).Build();
        using Tensor ta = Make(Bool, Count(a), 6, a), tb = Make(Bool, Count(b), 7, b);
        Same(model, new() { ["a"] = ta, ["b"] = tb }, ["y"]);
    }

    [Fact]
    public void NotMatchesTheCpu()
    {
        byte[] model = new ModelBuilder().Input("x", Bool, 3, 11).Output("y", Bool).Node("Not", ["x"], ["y"]).Build();
        using Tensor x = Make(Bool, 33, 8, [3, 11]);
        Same(model, new() { ["x"] = x }, ["y"]);
    }

    [Theory]
    [InlineData(F32)]
    [InlineData(F16)]
    [InlineData(I32)]
    [InlineData(I64)]
    public void WhereMatchesTheCpu(int type)
    {
        int[] c = [3, 1, 6], a = [1, 5, 1], b = [6];
        byte[] model = new ModelBuilder().Input("c", Bool, Dims(c)).Input("a", type, Dims(a)).Input("b", type, Dims(b)).Output("y", type)
            .Node("Where", ["c", "a", "b"], ["y"]).Build();
        using Tensor tc = Make(Bool, Count(c), 9, c), ta = Make(type, Count(a), 10, a), tb = Make(type, Count(b), 11, b);
        Same(model, new() { ["c"] = tc, ["a"] = ta, ["b"] = tb }, ["y"]);
    }

    // ---- Cast -------------------------------------------------------------------------------------------------------------

    [Theory]
    [InlineData(F32)]
    [InlineData(F16)]
    [InlineData(I32)]
    [InlineData(I64)]
    [InlineData(Bool)]
    public void CastBetweenEveryPairOfTypesMatchesTheCpu(int from)
    {
        int[] shape = [3, 17];
        foreach (int to in new[] { F32, F16, I32, I64, Bool })
        {
            // Float values past the integer range saturate, NaN becomes 0: the cases that differ between conversions.
            using Tensor x = from == F32
                ? Tensor.FromSpan<float>([.. Enumerable.Range(0, 51).Select(i => i switch { 0 => float.NaN, 1 => 1e10f, 2 => -1e10f, 3 => -0f, 4 => 2.5f, 5 => -2.5f, 6 => float.PositiveInfinity, 7 => 3e9f, 8 => 70000f, _ => (i - 25) * 0.75f })], Shape.Create(shape))
                : Make(from, 51, 12, shape);
            byte[] model = new ModelBuilder().Input("x", from, 3, 17).Output("y", to).Node("Cast", ["x"], ["y"], ("to", (long)to)).Build();
            Same(model, new() { ["x"] = x }, ["y"]);
        }
    }

    // ---- Expand and Gather of booleans ------------------------------------------------------------------------------------

    [Theory]
    [InlineData(F32)]
    [InlineData(F16)]
    [InlineData(I32)]
    [InlineData(I64)]
    [InlineData(Bool)]
    public void ExpandMatchesTheCpu(int type)
    {
        byte[] model = new ModelBuilder().Input("x", type, 2, 1, 5).Output("y", type).Initializer("shape", [3L, 1, 4, 5], 4).Node("Expand", ["x", "shape"], ["y"]).Build();
        using Tensor x = Make(type, 10, 13, [2, 1, 5]);
        Same(model, new() { ["x"] = x }, ["y"]);
    }

    [Theory]
    [InlineData(Bool, false)]
    [InlineData(Bool, true)]
    [InlineData(I32, false)]
    [InlineData(I64, true)]
    [InlineData(F32, false)]
    public void GatherOfEveryDeviceTypeMatchesTheCpu(int type, bool wideIndices)
    {
        // Bool data comes through the int32 gather: rows of a matrix picked by indices that include negatives and repeats.
        long[] ix = [3, 0, -1, 2, 2, -4];
        var b = new ModelBuilder().Input("x", type, 4, 6).Output("y", type);
        b = wideIndices ? b.Initializer("ix", ix, 6) : b.RawInitializer("ix", I32, [6], [.. ix.SelectMany(v => BitConverter.GetBytes((int)v))]);
        byte[] model = b.Node("Gather", ["x", "ix"], ["y"], ("axis", 0L)).Build();
        using Tensor x = Make(type, 24, 14, [4, 6]);
        Same(model, new() { ["x"] = x }, ["y"]);
    }

    // ---- generators (above the size the plan folds on its own) ------------------------------------------------------------

    // Hand-built TensorProto for a one-element attribute value of a type the builder has no helper for.
    private static ModelBuilder.TensorValue Scalar(int type, byte[] raw)
    {
        var s = new List<byte> { 0x10, (byte)type, 0x42, 1, (byte)'v', 0x4A, (byte)raw.Length };
        s.AddRange(raw);
        return new ModelBuilder.TensorValue([.. s]);
    }

    [Fact]
    public void RangeAndConstantOfShapeMatchTheCpu()
    {
        byte[] model = new ModelBuilder()
            .Input("seed", F32, 1).Output("a", I64).Output("b", F32).Output("c", I32).Output("d", Bool).Output("e", F32)
            .Initializer("s0", [5L], 1).Initializer("l0", [60005L], 1).Initializer("d0", [3L], 1)
            .Initializer("s1", [0.5f], 1).Initializer("l1", [30000.5f], 1).Initializer("d1", [0.25f], 1)
            .Initializer("shape", [300L, 300], 2)
            .Node("Range", ["s0", "l0", "d0"], ["a"])
            .Node("Range", ["s1", "l1", "d1"], ["b"])
            .Node("ConstantOfShape", ["shape"], ["c"], ("value", Scalar(I32, BitConverter.GetBytes(-7))))
            .Node("ConstantOfShape", ["shape"], ["d"], ("value", Scalar(Bool, [1])))
            .Node("ConstantOfShape", ["shape"], ["e0"])
            .Node("Add", ["e0", "seed"], ["e"])
            .Build();
        using Tensor seed = Tensor.FromSpan<float>([1.5f], Shape.Create(1));
        Same(model, new() { ["seed"] = seed }, ["a", "b", "c", "d", "e"]);
    }

    // ---- the attention-mask chain as exporters write it -------------------------------------------------------------------

    // attention_mask [B, S] int64 -> bool -> [B, 1, 1, S] -> expanded over the query axis, and-ed with a causal triangle made from two
    // ranges and a comparison, negated, and turned into an additive float mask by Where. At this size nothing folds.
    private static byte[] MaskChain(int batch, int seq) => new ModelBuilder()
        .Input("attention_mask", I64, batch, seq).Output("mask", F32).Output("allowed", Bool)
        .Initializer("axes", [1L, 2], 2).Initializer("target", [batch, 1L, seq, seq], 4)
        .Initializer("r0", [0L], 1).Initializer("rs", [(long)seq], 1).Initializer("one", [1L], 1)
        .Initializer("rowAxes", [1L], 1).Initializer("colAxes", [0L], 1)
        .Initializer("neg", [float.NegativeInfinity], 1).Initializer("zero", [0f], 1)
        .Node("Cast", ["attention_mask"], ["m"], ("to", (long)Bool))
        .Node("Unsqueeze", ["m", "axes"], ["m4"])
        .Node("Expand", ["m4", "target"], ["me"])
        .Node("Range", ["r0", "rs", "one"], ["q"])
        .Node("Range", ["r0", "rs", "one"], ["k"])
        .Node("Unsqueeze", ["q", "rowAxes"], ["qc"])
        .Node("Unsqueeze", ["k", "colAxes"], ["kr"])
        .Node("GreaterOrEqual", ["qc", "kr"], ["tri"])
        .Node("And", ["me", "tri"], ["allowed"])
        .Node("Not", ["allowed"], ["blocked"])
        .Node("Where", ["blocked", "neg", "zero"], ["mask"])
        .Build();

    [Fact]
    public void TheMaskChainRunsOnTheDeviceInOneRegion()
    {
        int batch = 2, seq = 300;
        long[] m = new long[batch * seq];
        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < seq; t++)
            {
                m[b * seq + t] = t < seq - 13 * (b + 1) ? 1 : 0;
            }
        }

        using Tensor mask = Tensor.FromSpan<long>(m, Shape.Create(batch, seq));
        Same(MaskChain(batch, seq), new() { ["attention_mask"] = mask }, ["mask", "allowed"]);
    }

    // ---- QLinearMatMul ----------------------------------------------------------------------------------------------------

    internal static byte[] Bytes(int n, int seed, bool signedValues)
    {
        var rng = new Random(seed);
        return [.. Enumerable.Range(0, n).Select(_ => rng.Next(6) switch { 0 => (byte)0, 1 => (byte)255, 2 => (byte)128, _ => (byte)rng.Next(256) })
            .Select(v => signedValues ? unchecked((byte)(sbyte)(v - 128)) : v)];
    }

    [Theory]
    [InlineData(5, 17, 9, false, false, false, false)]
    [InlineData(1, 64, 33, true, true, false, true)]
    [InlineData(70, 129, 37, false, true, true, true)]
    [InlineData(8, 384, 20, true, false, true, false)]
    [InlineData(3, 7, 1, false, false, false, false)]
    public void QLinearMatMulMatchesTheCpuBitForBit(int m, int k, int n, bool bSigned, bool perColumn, bool perRowZero, bool bias)
    {
        int bType = bSigned ? 3 : 2;
        var b = new ModelBuilder().Input("a", 2, m, k).Output("y", 2)
            .RawInitializer("w", bType, [k, n], Bytes(k * n, 21, bSigned))
            .Initializer("sa", [0.021f], 1).Initializer("sy", [0.37f], 1)
            .Initializer("sb", perColumn ? [.. Enumerable.Range(0, n).Select(i => 0.004f + 0.0007f * i)] : [0.0061f], perColumn ? n : 1)
            .RawInitializer("zy", 2, [1], [131]);
        b = perRowZero ? b.RawInitializer("za", 2, [m], Bytes(m, 22, false)) : b.RawInitializer("za", 2, [1], [117]);
        b = perColumn ? b.RawInitializer("zb", bType, [n], Bytes(n, 23, bSigned)) : b.RawInitializer("zb", bType, [1], [bSigned ? unchecked((byte)(sbyte)-3) : (byte)9]);
        string[] inputs = ["a", "sa", "za", "w", "sb", "zb", "sy", "zy"];
        if (bias)
        {
            b = b.RawInitializer("bias", I32, [n], [.. Enumerable.Range(0, n).SelectMany(i => BitConverter.GetBytes((i - n / 2) * 977))]);
            inputs = [.. inputs, "bias"];
        }

        byte[] model = b.Node("QLinearMatMul", inputs, ["y"]).Build();
        using Tensor a = Tensor.FromSpan<byte>(Bytes(m * k, 24, false), Shape.Create(m, k));
        Same(model, new() { ["a"] = a }, ["y"]);
    }
}
