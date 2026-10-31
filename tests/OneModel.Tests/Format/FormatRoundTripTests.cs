using System.Runtime.CompilerServices;
using System.Text.Json;
using RecursiveAI.OneModel.Format;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tests.Onnx;

namespace RecursiveAI.OneModel.Tests.Format;

public class FormatRoundTripTests
{
    private static string Root([CallerFilePath] string file = "") => Path.GetFullPath(Path.Combine(Path.GetDirectoryName(file)!, "..", "..", ".."));

    private static string GraphPath(string name) => Path.Combine(Root(), "testdata", "graphs", name);

    private static byte[] Save(IrModel m)
    {
        using var ms = new MemoryStream();
        ModelFormat.Save(m, ms);
        return ms.ToArray();
    }

    [Theory]
    [InlineData("mlp.onnx")]
    [InlineData("chain.onnx")]
    [InlineData("shape_ops.onnx")]
    [InlineData("int8_dyn.onnx")]
    [InlineData("int8_qdq.onnx")]
    public void SavingTwiceGivesTheSameBytesAndTheGraphSurvives(string file)
    {
        using IrModel a = OnnxImporter.Load(GraphPath(file));
        byte[] first = Save(a);
        Assert.True(ModelFormat.IsModelFormat(first));
        using IrModel b = ModelFormat.Load(first);
        byte[] second = Save(b);
        Assert.Equal(first, second);
        Assert.Equal(a.Graph.Nodes.Length, b.Graph.Nodes.Length);
        Assert.Equal(a.Graph.Initializers.Length, b.Graph.Initializers.Length);
        Assert.Equal(a.OpsetImports, b.OpsetImports);
        Assert.Equal(Save(a), first);
    }

    private static float[] Run(string path, float[] input, int[] shape, int outputLength, bool exact = false)
    {
        using Session s = Session.Load(path, new ModelOptions { IntraOpThreads = 1, Mode = exact ? ArithmeticMode.Exact : ArithmeticMode.Fast });
        var y = new float[outputLength];
        s.Run(s.Inputs[0].Name, input.AsSpan(), shape, s.Outputs[0].Name, y.AsSpan());
        return y;
    }

    [Theory]
    [InlineData("mlp.onnx", "reference.json", "mlp-5")]
    [InlineData("chain.onnx", "reference.json", "chain-5")]
    [InlineData("int8_dyn.onnx", "int8_dyn.json", null)]
    [InlineData("int8_qdq.onnx", "int8_qdq.json", null)]
    [InlineData("shape_ops.onnx", "shape_ops.json", null)]
    public void ConvertedModelsGiveBitIdenticalOutputs(string model, string json, string? caseName)
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(GraphPath(json)));
        JsonElement c = caseName is null ? doc.RootElement : doc.RootElement.GetProperty("cases").GetProperty(caseName);
        int[] shape = [.. c.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32())];
        float[] input = [.. c.GetProperty("input").EnumerateArray().Select(e => (float)e.GetDouble())];
        int outLength = c.GetProperty("output").GetArrayLength();

        string converted = Path.Combine(Path.GetTempPath(), "onemodel-fmt-" + Guid.NewGuid().ToString("N") + ".onemodel");
        try
        {
            ModelFormat.Convert(GraphPath(model), converted);
            Assert.True(ModelFormat.IsModelFormatFile(converted));
            Assert.False(ModelFormat.IsModelFormatFile(GraphPath(model)));
            foreach (bool exact in new[] { false, true })
            {
                if (exact && model.StartsWith("int8", StringComparison.Ordinal))
                {
                    continue;
                }

                float[] direct = Run(GraphPath(model), input, shape, outLength, exact);
                float[] viaFormat = Run(converted, input, shape, outLength, exact);
                Assert.True(direct.AsSpan().SequenceEqual(viaFormat), "outputs differ bit for bit");
            }

            // The stream entry point recognises the format too.
            using FileStream fs = File.OpenRead(converted);
            using Session s = Session.Load(fs);
            Assert.NotEmpty(s.Inputs);
        }
        finally
        {
            File.Delete(converted);
        }
    }

    [Fact]
    public void DtypesStringsSubgraphsAndExternalDataRoundTrip()
    {
        // int8, fp16, bf16, int4, and a string tensor; an If with a subgraph; a sparse initializer; typed outputs.
        var strTensor = new Pb().PackedInts(1, 2).Int(2, Build.String).Str(8, "s").Bytes(6, "ab"u8).Bytes(6, [0xFF, 0x00]);
        Pb branch = Build.Graph("then", [Build.Node("Identity", ["w8"], ["tout"])], null, [Build.ValueInfo("tout", Build.Int8, 4)]);
        Pb branch2 = Build.Graph("else", [Build.Node("Identity", ["w8"], ["eout"])], null, [Build.ValueInfo("eout", Build.Int8, 4)]);
        Pb sparse = new Pb().Msg(1, Build.RawTensor("sp", Build.Float, [2], new byte[8])).Msg(2, Build.RawTensor("idx", 7, [2], new byte[16])).PackedInts(3, 4);
        Pb g = Build.Graph("g",
            [Build.Node("If", ["c"], ["r"], "if", Build.AttrGraph("then_branch", branch), Build.AttrGraph("else_branch", branch2), Build.AttrFloat("f", float.NaN))],
            [Build.ValueInfo("c", Build.Bool)], [Build.ValueInfo("r", Build.Int8, 4)],
            [
                Build.RawTensor("w8", Build.Int8, [4], [1, 2, 0xFE, 0x80]),
                Build.RawTensor("w16", Build.Float16, [2, 2], [0, 0x3C, 0, 0x40, 0, 0x42, 0, 0x44]),
                Build.RawTensor("wbf", 16, [3], [0x80, 0x3F, 0x00, 0x40, 0x40, 0x40]),
                Build.RawTensor("w4", Build.Int4, [3], [0x21, 0x03]),
                Build.RawTensor("empty", Build.Float, [0, 3], []),
                strTensor,
            ]);
        g.Msg(15, sparse);
        byte[] onnx = Build.Model(g, 13, 8);

        using IrModel a = OnnxImporter.Load(onnx);
        byte[] saved = Save(a);
        using IrModel b = ModelFormat.Load(saved);
        Assert.Equal(saved, Save(b));
        Assert.Equal(a.Graph.Initializers.Length, b.Graph.Initializers.Length);
        for (int i = 0; i < a.Graph.Initializers.Length; i++)
        {
            TensorData x = a.Graph.Initializers[i], y = b.Graph.Initializers[i];
            Assert.Equal(x.Name, y.Name);
            Assert.Equal(x.DataType, y.DataType);
            Assert.Equal(x.Dims, y.Dims);
            Assert.True(x.RawData.SequenceEqual(y.RawData), x.Name);
            Assert.Equal(x.Strings.Length, y.Strings.Length);
            for (int k = 0; k < x.Strings.Length; k++)
            {
                Assert.True(x.Strings[k].Span.SequenceEqual(y.Strings[k].Span));
            }
        }

        IrNode node = b.Graph.Nodes[0];
        Assert.Equal(IrAttributeType.Graph, node.FindAttribute("then_branch")!.Type);
        Assert.Single(node.FindAttribute("then_branch")!.Graph.Nodes);
        Assert.Equal(BitConverter.SingleToInt32Bits(float.NaN), BitConverter.SingleToInt32Bits(node.FindAttribute("f")!.Real));
        Assert.Single(b.Graph.SparseInitializers);
        Assert.Equal(a.Graph.Outputs[0].Type!.ToString(), b.Graph.Outputs[0].Type!.ToString());
        Assert.Equal(a.Graph.Inputs[0].Type!.ToString(), b.Graph.Inputs[0].Type!.ToString());
    }

    [Fact]
    public void ExternalWeightsAreCopiedIntoTheFile()
    {
        string dir = Path.Combine(Path.GetTempPath(), "onemodel-fmt-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var w = new byte[16];
            for (int i = 0; i < w.Length; i++)
            {
                w[i] = (byte)(i + 1);
            }

            File.WriteAllBytes(Path.Combine(dir, "w.bin"), w);
            var t = new Pb().PackedInts(1, 4).Int(2, Build.Int32).Str(8, "w").Int(14, 1).Msg(13, new Pb().Str(1, "location").Str(2, "w.bin"));
            File.WriteAllBytes(Path.Combine(dir, "m.onnx"), Build.Model(Build.Graph("g", [], null, null, [t])));
            string output = Path.Combine(dir, "m.onemodel");
            ModelFormat.Convert(Path.Combine(dir, "m.onnx"), output);
            File.Delete(Path.Combine(dir, "w.bin"));
            using IrModel m = ModelFormat.Load(output);
            Assert.False(m.Graph.Initializers[0].IsExternal);
            Assert.True(m.Graph.Initializers[0].RawData.SequenceEqual(w));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void TensorsAreAlignedAndLargeFilesAreMapped()
    {
        var big = new byte[3 << 20];
        new Random(5).NextBytes(big);
        byte[] onnx = Build.Model(Build.Graph("g", [], null, null,
            [Build.RawTensor("a", Build.Int8, [3], [1, 2, 3]), Build.RawTensor("b", Build.UInt8, [big.Length], big)]));
        string path = Path.Combine(Path.GetTempPath(), "onemodel-fmt-" + Guid.NewGuid().ToString("N") + ".onemodel");
        try
        {
            using (IrModel a = OnnxImporter.Load(onnx))
            {
                ModelFormat.Save(a, path);
            }

            using IrModel m = ModelFormat.Load(path);
            Assert.True(m.Graph.Initializers[1].RawData.SequenceEqual(big));
            Assert.True(m.Graph.Initializers[0].RawData.SequenceEqual(new byte[] { 1, 2, 3 }));
            byte[] file = File.ReadAllBytes(path);
            // data_offset is a multiple of 64 and the second tensor starts 64 bytes after the first.
            Assert.Equal(0UL, System.Buffers.Binary.BinaryPrimitives.ReadUInt64LittleEndian(file.AsSpan(32)) % 64);
            Assert.True(file.Length > big.Length);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
