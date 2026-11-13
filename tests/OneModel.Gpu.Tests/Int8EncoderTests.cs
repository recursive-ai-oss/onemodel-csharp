using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tests.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu.Tests;

/// <summary>
/// The operators at the head of a dynamically quantized encoder: <c>DequantizeLinear</c> of uint8, int8 and int32 values, per tensor and
/// per axis, and <c>Gather</c> on a uint8 table. Their results are integers converted and scaled by the CPU's own arithmetic, so every
/// comparison is bit for bit, in both modes.
/// </summary>
public class Int8EncoderTests
{
    private static byte[] Bytes(int n, int seed)
    {
        var rng = new Random(seed);
        byte[] b = new byte[n];
        rng.NextBytes(b);
        return b;
    }

    private static byte[] Floats(float[] v)
    {
        byte[] raw = new byte[v.Length * 4];
        for (int i = 0; i < v.Length; i++)
        {
            System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(raw.AsSpan(i * 4), v[i]);
        }

        return raw;
    }

    private static float[] Scales(int n, int seed) => [.. GpuTestHost.Random(n, seed, 0.2f).Select(v => Math.Abs(v) + 0.003f)];

    private static string Bits(ReadOnlySpan<float> v) => GpuTestHost.Hash(v);

    // Runs the model on the CPU and the device and requires identical bits and one region with nothing left on the CPU.
    private static void BitExact(byte[] model, Dictionary<string, Tensor> feed, ArithmeticMode mode)
    {
        GpuTestHost.Require();
        float[] expected;
        using (Session cpu = GpuTestHost.Load(model, GpuTestHost.Cpu(mode)))
        using (SessionOutputs o = cpu.Run(feed))
        {
            expected = o["y"].Data<float>().ToArray();
        }

        using Session gpu = GpuTestHost.Load(model, GpuTestHost.Gpu(mode, require: true));
        SessionDiagnostics d = gpu.GetDiagnostics(feed);
        Assert.Equal(1, d.DeviceRegions);
        Assert.True(d.Nodes.All(n => n.FallbackReason is null));
        for (int run = 0; run < 2; run++)
        {
            using SessionOutputs o = gpu.Run(feed);
            Assert.Equal(Bits(expected), Bits(o["y"].Data<float>()));
        }

        GpuTestHost.AssertValidationClean(gpu);
    }

    [Theory]
    [InlineData(ArithmeticMode.Fast, false, -1, true)]
    [InlineData(ArithmeticMode.Exact, false, -1, true)]
    [InlineData(ArithmeticMode.Fast, false, -1, false)]
    [InlineData(ArithmeticMode.Fast, true, 1, true)]
    [InlineData(ArithmeticMode.Exact, true, 2, true)]
    [InlineData(ArithmeticMode.Fast, true, 0, false)]
    [InlineData(ArithmeticMode.Fast, true, -1, true)]
    public void DequantizeLinearOfUint8MatchesTheCpuBitForBit(ArithmeticMode mode, bool perAxis, int axis, bool zeroPoint)
    {
        int[] shape = [3, 5, 7];
        int count = shape.Aggregate(1, (a, b) => a * b);
        int ax = axis < 0 ? axis + 3 : axis;
        int sn = perAxis ? shape[ax] : 1;
        long[] sdims = perAxis ? [sn] : [];
        var b = new ModelBuilder(13).Input("x", 2, 3, 5, 7).Output("y", 1, 3, 5, 7).RawInitializer("s", 1, sdims, Floats(Scales(sn, 5)));
        if (zeroPoint)
        {
            b.RawInitializer("z", 2, sdims, Bytes(sn, 9));
        }

        byte[] model = b.Node("DequantizeLinear", zeroPoint ? ["x", "s", "z"] : ["x", "s"], ["y"], ("axis", (long)axis)).Build();
        using Tensor x = Tensor.FromSpan<byte>(Bytes(count, 1), Shape.Create(shape));
        BitExact(model, new() { ["x"] = x }, mode);
    }

    [Theory]
    [InlineData(ArithmeticMode.Fast, -1, true)]
    [InlineData(ArithmeticMode.Exact, 0, false)]
    public void DequantizeLinearOfInt32MatchesTheCpuBitForBit(ArithmeticMode mode, int axis, bool perAxis)
    {
        int[] shape = [4, 6];
        int ax = axis < 0 ? axis + 2 : axis;
        int sn = perAxis ? shape[ax] : 1;
        long[] sdims = perAxis ? [sn] : [];
        int[] values = [.. new Random(3).GetItems([int.MinValue, int.MaxValue, 0, -1, 1, 123456789, -987654321, 5], 24)];
        byte[] raw = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++)
        {
            System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(raw.AsSpan(i * 4), values[i]);
        }

        byte[] zero = new byte[sn * 4];
        for (int i = 0; i < sn; i++)
        {
            System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(zero.AsSpan(i * 4), i * 1000 - 17);
        }

        byte[] model = new ModelBuilder(13).Input("x", 6, 4, 6).Output("y", 1, 4, 6)
            .RawInitializer("s", 1, sdims, Floats(Scales(sn, 5))).RawInitializer("z", 6, sdims, zero)
            .Node("DequantizeLinear", ["x", "s", "z"], ["y"], ("axis", (long)axis)).Build();
        using Tensor x = Tensor.FromSpan<int>(values, Shape.Create(shape));
        BitExact(model, new() { ["x"] = x }, mode);
    }

    private static byte[] Build(int rows, int width, bool zeroPoint)
    {
        var b = new ModelBuilder(13).Input("ids", 7, "B", "S").Output("y", 1, "B", "S", width)
            .RawInitializer("table", 2, [rows, width], Bytes(rows * width, 4))
            .RawInitializer("s", 1, [], Floats([0.0173f]));
        string[] deq = ["g", "s"];
        if (zeroPoint)
        {
            b.RawInitializer("z", 2, [], [131]);
            deq = ["g", "s", "z"];
        }

        return b.Node("Gather", ["table", "ids"], ["g"]).Node("DequantizeLinear", deq, ["y"]).Build();
    }

    [Theory]
    [InlineData(ArithmeticMode.Fast, 24, true)]
    [InlineData(ArithmeticMode.Exact, 24, true)]
    [InlineData(ArithmeticMode.Fast, 7, false)]
    [InlineData(ArithmeticMode.Fast, 1, true)]
    [InlineData(ArithmeticMode.Exact, 384, true)]
    public void AnEightBitEmbeddingLookupAndItsDequantizeRunOnTheDeviceAndRejectABadId(ArithmeticMode mode, int width, bool zeroPoint)
    {
        GpuTestHost.Require();
        const int rows = 50;
        byte[] model = Build(rows, width, zeroPoint);
        using Tensor ids = Tensor.FromSpan<long>([0, 49, 7, -1, 12, 12], Shape.Create(2, 3));
        var feed = new Dictionary<string, Tensor> { ["ids"] = ids };
        BitExact(model, feed, mode);

        using Session g = GpuTestHost.Load(model, GpuTestHost.Gpu(mode, require: true));
        using Tensor bad = Tensor.FromSpan<long>([1, 2, 50, 3, 4, 5], Shape.Create(2, 3));
        SessionException e = Assert.Throws<SessionException>(() => g.Run(new Dictionary<string, Tensor> { ["ids"] = bad }));
        Assert.Equal(SessionErrorKind.InvalidInput, e.Kind);
        using Session c = GpuTestHost.Load(model, GpuTestHost.Cpu(mode));
        SessionException ce = Assert.Throws<SessionException>(() => c.Run(new Dictionary<string, Tensor> { ["ids"] = bad }));
        Assert.Equal(ce.Message, e.Message);
        using (SessionOutputs o = g.Run(feed))
        {
            Assert.Equal(Bits(o["y"].Data<float>()), Bits(GpuReference(model, feed, mode)));
        }

        GpuTestHost.AssertValidationClean(g);
    }

    private static float[] GpuReference(byte[] model, Dictionary<string, Tensor> feed, ArithmeticMode mode)
    {
        using Session c = GpuTestHost.Load(model, GpuTestHost.Cpu(mode));
        using SessionOutputs o = c.Run(feed);
        return o["y"].Data<float>().ToArray();
    }

    // ids -> uint8 table -> DequantizeLinear -> DynamicQuantizeLinear -> MatMulInteger -> Cast -> Mul -> Add: the head of a quantized encoder.
    private static byte[] Encoder(int rows, int width, int n) => new ModelBuilder(13)
        .Input("ids", 7, "B", "S").Output("y", 1, "B", "S", n)
        .RawInitializer("table", 2, [rows, width], Bytes(rows * width, 4)).RawInitializer("ts", 1, [], Floats([0.0173f])).RawInitializer("tz", 2, [], [131])
        .RawInitializer("w", 3, [width, n], Bytes(width * n, 6)).RawInitializer("wzp", 3, [], [3])
        .RawInitializer("ws", 1, [], Floats([0.011f])).Initializer("bias", GpuTestHost.Random(n, 5), n)
        .Node("Gather", ["table", "ids"], ["g"]).Node("DequantizeLinear", ["g", "ts", "tz"], ["e"])
        .Node("DynamicQuantizeLinear", ["e"], ["eq", "es", "ez"]).Node("MatMulInteger", ["eq", "w", "ez", "wzp"], ["acc"])
        .Node("Mul", ["es", "ws"], ["sc"]).Node("Cast", ["acc"], ["accf"], ("to", 1L)).Node("Mul", ["accf", "sc"], ["scaled"]).Node("Add", ["scaled", "bias"], ["y"]).Build();

    private static int BindGroups(GpuBackend backend) =>
        backend.Elementwise.BindGroupsCreated + backend.Gather.BindGroupsCreated + backend.Int8.BindGroupsCreated + backend.DynamicQuantize.BindGroupsCreated + backend.Dequantize.BindGroupsCreated;

    [Theory]
    [InlineData(ArithmeticMode.Fast)]
    [InlineData(ArithmeticMode.Exact)]
    public void AQuantizedEncoderHeadIsOneRegionBitExactAndAWarmRunBuildsNoBindGroups(ArithmeticMode mode)
    {
        GpuTestHost.Require();
        byte[] model = Encoder(60, 32, 48);
        using Tensor small = Tensor.FromSpan<long>([0, 59, 7, 1, 12, 12], Shape.Create(2, 3));
        long[] many = [.. Enumerable.Range(0, 20000).Select(i => (long)(i * 7 % 60))];
        using Tensor big = Tensor.FromSpan<long>(many, Shape.Create(100, 200));
        BitExact(model, new() { ["ids"] = small }, mode);

        using Session g = GpuTestHost.Load(model, GpuTestHost.Gpu(mode, require: true));
        var backend = (GpuBackend)g.Backend!;
        g.Run(new Dictionary<string, Tensor> { ["ids"] = small }).Dispose();
        int first = BindGroups(backend);
        Assert.True(first > 0);
        g.Run(new Dictionary<string, Tensor> { ["ids"] = small }).Dispose();
        Assert.Equal(first, BindGroups(backend));

        // A bigger shape replaces the arena's buffers, which drops the cached groups; they are built once for the new buffers.
        var bigFeed = new Dictionary<string, Tensor> { ["ids"] = big };
        float[] cpuBig;
        using (Session c = GpuTestHost.Load(model, GpuTestHost.Cpu(mode)))
        using (SessionOutputs o = c.Run(bigFeed))
        {
            cpuBig = o["y"].Data<float>().ToArray();
        }

        using (SessionOutputs o = g.Run(bigFeed))
        {
            Assert.Equal(Bits(cpuBig), Bits(o["y"].Data<float>()));
        }

        // Going back to the small shape names the bigger sums buffer the big plan made, which is a set not seen yet; after that, nothing is built.
        g.Run(bigFeed).Dispose();
        g.Run(new Dictionary<string, Tensor> { ["ids"] = small }).Dispose();
        int settled = BindGroups(backend);
        g.Run(bigFeed).Dispose();
        g.Run(new Dictionary<string, Tensor> { ["ids"] = small }).Dispose();
        Assert.Equal(settled, BindGroups(backend));
        GpuTestHost.AssertValidationClean(g);
    }

    // Three quantized products whose outputs grow (16, 64, 256 columns), so the 8-bit kernel's sums buffer would have to grow during the
    // first run if it were not sized beforehand.
    private static byte[] GrowingProducts()
    {
        var b = new ModelBuilder(13).Input("x", 1, "M", 8).Output("y", 1, "M", 256);
        string prev = "x";
        int k = 8;
        int[] widths = [16, 64, 256];
        for (int i = 0; i < widths.Length; i++)
        {
            int n = widths[i];
            string w = $"w{i}", ws = $"ws{i}", bias = $"bias{i}", o = i == widths.Length - 1 ? "y" : $"o{i}";
            b.RawInitializer(w, 3, [k, n], Bytes(k * n, 20 + i)).RawInitializer($"z{i}", 3, [], [3]).RawInitializer(ws, 1, [], Floats([0.011f + i * 0.002f]))
                .Initializer(bias, GpuTestHost.Random(n, 30 + i), n)
                .Node("DynamicQuantizeLinear", [prev], [$"q{i}", $"s{i}", $"p{i}"]).Node("MatMulInteger", [$"q{i}", w, $"p{i}", $"z{i}"], [$"acc{i}"])
                .Node("Mul", [$"s{i}", ws], [$"sc{i}"]).Node("Cast", [$"acc{i}"], [$"f{i}"], ("to", 1L)).Node("Mul", [$"f{i}", $"sc{i}"], [$"m{i}"]).Node("Add", [$"m{i}", bias], [o]);
            prev = o;
            k = n;
        }

        return b.Build();
    }

    [Theory]
    [InlineData(ArithmeticMode.Fast)]
    [InlineData(ArithmeticMode.Exact)]
    public void AWarmRunBuildsNoBindGroupsEvenWhenTheProductsGrow(ArithmeticMode mode)
    {
        GpuTestHost.Require();
        byte[] model = GrowingProducts();
        using Tensor x = Tensor.FromSpan<float>(GpuTestHost.Random(300 * 8, 3, 5f), Shape.Create(300, 8));
        var feed = new Dictionary<string, Tensor> { ["x"] = x };
        using Session g = GpuTestHost.Load(model, GpuTestHost.Gpu(mode, require: true));
        var backend = (GpuBackend)g.Backend!;
        float[] first;
        using (SessionOutputs o = g.Run(feed))
        {
            first = o["y"].Data<float>().ToArray();
        }

        int afterFirst = BindGroups(backend);
        using (SessionOutputs o = g.Run(feed))
        {
            Assert.Equal(Bits(first), Bits(o["y"].Data<float>()));
        }

        Assert.Equal(afterFirst, BindGroups(backend));
        using Session c = GpuTestHost.Load(model, GpuTestHost.Cpu(mode));
        using SessionOutputs oc = c.Run(feed);
        Assert.Equal(Bits(oc["y"].Data<float>()), Bits(first));
        GpuTestHost.AssertValidationClean(g);
    }

    // ---- QuantizeLinear ----------------------------------------------------------------------------------------------------

    // Values that stress the rounding: exact halves (ties to even), beyond both saturation ends, signed zero, infinities, NaN.
    private static float[] QuantInputs(int n, int seed, float scale)
    {
        float[] v = [.. GpuTestHost.Random(n, seed, 600f * scale)];
        float[] special = [0.5f * scale, 1.5f * scale, 2.5f * scale, -0.5f * scale, -1.5f * scale, 1000f * scale, -1000f * scale, -0f, 0f, float.PositiveInfinity, float.NegativeInfinity, float.NaN, 254.5f * scale, 255.5f * scale];
        for (int i = 0; i < special.Length && i < n; i++)
        {
            v[(i * 7) % n] = special[i];
        }

        return v;
    }

    [Theory]
    [InlineData(ArithmeticMode.Fast, false, -1, true)]
    [InlineData(ArithmeticMode.Exact, false, -1, true)]
    [InlineData(ArithmeticMode.Fast, false, -1, false)]
    [InlineData(ArithmeticMode.Fast, true, 1, true)]
    [InlineData(ArithmeticMode.Exact, true, 2, true)]
    [InlineData(ArithmeticMode.Fast, true, 0, false)]
    [InlineData(ArithmeticMode.Fast, true, -1, true)]
    public void QuantizeLinearMatchesTheCpuBitForBit(ArithmeticMode mode, bool perAxis, int axis, bool zeroPoint)
    {
        GpuTestHost.Require();
        int[] shape = [3, 5, 7];
        int count = shape.Aggregate(1, (a, b) => a * b);
        int ax = axis < 0 ? axis + 3 : axis;
        int sn = perAxis ? shape[ax] : 1;
        long[] sdims = perAxis ? [sn] : [];
        float[] scales = Scales(sn, 5);
        var b = new ModelBuilder(13).Input("x", 1, 3, 5, 7).Output("y", 2).RawInitializer("s", 1, sdims, Floats(scales));
        if (zeroPoint)
        {
            b.RawInitializer("z", 2, sdims, Bytes(sn, 9));
        }

        byte[] model = b.Node("QuantizeLinear", zeroPoint ? ["x", "s", "z"] : ["x", "s"], ["y"], ("axis", (long)axis)).Build();
        using Tensor x = Tensor.FromSpan<float>(QuantInputs(count, 2, scales[0]), Shape.Create(shape));
        var feed = new Dictionary<string, Tensor> { ["x"] = x };
        byte[] expected;
        using (Session cpu = GpuTestHost.Load(model, GpuTestHost.Cpu(mode)))
        using (SessionOutputs o = cpu.Run(feed))
        {
            expected = o["y"].Data<byte>().ToArray();
        }

        using Session gpu = GpuTestHost.Load(model, GpuTestHost.Gpu(mode, require: true));
        Assert.Equal(1, gpu.GetDiagnostics(feed).DeviceRegions);
        for (int run = 0; run < 2; run++)
        {
            using SessionOutputs o = gpu.Run(feed);
            Assert.Equal(expected, o["y"].Data<byte>().ToArray());
        }

        GpuTestHost.AssertValidationClean(gpu);
    }

    [Fact]
    public void QuantizeThenDequantizeRunInOneRegionWithTheCpusBits()
    {
        float[] scale = [0.037f];
        byte[] model = new ModelBuilder(13).Input("x", 1, 2, 3, 11).Output("y", 1)
            .RawInitializer("s", 1, [], Floats(scale)).RawInitializer("z", 2, [], [131])
            .Node("QuantizeLinear", ["x", "s", "z"], ["q"]).Node("DequantizeLinear", ["q", "s", "z"], ["y"]).Build();
        using Tensor x = Tensor.FromSpan<float>(QuantInputs(66, 4, scale[0]), Shape.Create(2, 3, 11));
        BitExact(model, new() { ["x"] = x }, ArithmeticMode.Fast);
    }

    [Theory]
    [InlineData(ArithmeticMode.Fast, false, -1, true)]
    [InlineData(ArithmeticMode.Exact, false, -1, false)]
    [InlineData(ArithmeticMode.Fast, true, 1, true)]
    [InlineData(ArithmeticMode.Exact, true, 0, true)]
    public void QuantizeLinearToInt8MatchesTheCpuBitForBit(ArithmeticMode mode, bool perAxis, int axis, bool zeroPoint)
    {
        GpuTestHost.Require();
        int[] shape = [3, 5, 7];
        int count = shape.Aggregate(1, (a, b) => a * b);
        int sn = perAxis ? shape[axis] : 1;
        long[] sdims = perAxis ? [sn] : [];
        float[] scales = Scales(sn, 5);
        var b = new ModelBuilder(13).Input("x", 1, 3, 5, 7).Output("y", 3).RawInitializer("s", 1, sdims, Floats(scales));
        if (zeroPoint)
        {
            b.RawInitializer("z", 3, sdims, Bytes(sn, 9));
        }

        byte[] model = b.Node("QuantizeLinear", zeroPoint ? ["x", "s", "z"] : ["x", "s"], ["y"], ("axis", (long)axis), ("output_dtype", 3L)).Build();
        using Tensor x = Tensor.FromSpan<float>(QuantInputs(count, 2, scales[0]), Shape.Create(shape));
        var feed = new Dictionary<string, Tensor> { ["x"] = x };
        sbyte[] expected;
        using (Session cpu = GpuTestHost.Load(model, GpuTestHost.Cpu(mode)))
        using (SessionOutputs o = cpu.Run(feed))
        {
            expected = o["y"].Data<sbyte>().ToArray();
        }

        using Session gpu = GpuTestHost.Load(model, GpuTestHost.Gpu(mode, require: true));
        for (int run = 0; run < 2; run++)
        {
            using SessionOutputs o = gpu.Run(feed);
            Assert.Equal(expected, o["y"].Data<sbyte>().ToArray());
        }

        GpuTestHost.AssertValidationClean(gpu);
    }

    [Fact]
    public void Int8QuantizeThenDequantizeIsOneRegionWithTheCpusBits()
    {
        // The quantize-dequantize pairs of a QDQ model: int8 values made and consumed on the device.
        byte[] model = new ModelBuilder(13).Input("x", 1, 2, 3, 11).Output("y", 1)
            .RawInitializer("s", 1, [], Floats([0.05f])).RawInitializer("z", 3, [], [unchecked((byte)-7)])
            .Node("QuantizeLinear", ["x", "s", "z"], ["q"]).Node("DequantizeLinear", ["q", "s", "z"], ["y"]).Build();
        using Tensor x = Tensor.FromSpan<float>(QuantInputs(66, 4, 0.05f), Shape.Create(2, 3, 11));
        BitExact(model, new() { ["x"] = x }, ArithmeticMode.Fast);
        BitExact(model, new() { ["x"] = x }, ArithmeticMode.Exact);
    }

    [Fact]
    public void SixteenBitQuantizeStaysOnTheCpuWithItsReason()
    {
        GpuTestHost.Require();
        byte[] model = new ModelBuilder(13).Input("x", 1, 4, 5).Output("y", 4).RawInitializer("s", 1, [], Floats([0.1f]))
            .Node("QuantizeLinear", ["x", "s"], ["y"], ("output_dtype", 4L)).Build();
        using Tensor x = Tensor.FromSpan<float>(QuantInputs(20, 4, 0.1f), Shape.Create(4, 5));
        var feed = new Dictionary<string, Tensor> { ["x"] = x };
        using Session gpu = GpuTestHost.Load(model, GpuTestHost.Gpu(ArithmeticMode.Fast));
        SessionDiagnostics d = gpu.GetDiagnostics(feed);
        Assert.Equal(0, d.NodesOnDevice);
        Assert.Contains("16-bit", d.Nodes[0].FallbackReason);
    }
}
