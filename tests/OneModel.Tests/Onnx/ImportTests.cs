using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Onnx;

public class ImportTests
{
    [Fact]
    public void SimpleGraphIsResolved()
    {
        using IrModel m = OnnxImporter.Load(Build.AddModel());
        Assert.Equal(8, m.IrVersion);
        Assert.Equal(13, m.DefaultOpset);
        IrGraph g = m.Graph;
        IrNode add = Assert.Single(g.Nodes);
        Assert.Equal("Add", add.OpType);
        Assert.Equal(13, add.OpsetVersion);
        Assert.Equal(["x", "w"], add.Inputs.Select(v => v!.Name));
        Assert.Same(add, add.Outputs[0]!.Producer);
        Assert.Equal("y", Assert.Single(g.Outputs).Name);
        IrValue w = g.FindValue("w")!;
        Assert.NotNull(w.Initializer);
        Assert.False(w.IsGraphInput);
        Assert.Equal([add], w.Consumers);
        IrValue x = g.Inputs[0];
        Assert.Equal(IrTypeKind.Tensor, x.Type!.Kind);
        Assert.Equal(1, x.Type.ElemType);
        Assert.Equal([2L, 2L], x.Type.Dims.Select(d => d.Value));
        using Tensor t = w.Initializer!.ToTensor();
        Assert.Equal([1f, 2f, 3f, 4f], t.Data<float>().ToArray());
    }

    [Fact]
    public void OmittedOptionalInputsAreNull()
    {
        byte[] bytes = Build.Model(Build.Graph("g",
            [Build.Node("Clip", ["x", "", "hi"], ["y"])],
            [Build.ValueInfo("x", Build.Float), Build.ValueInfo("hi", Build.Float)], [Build.ValueInfo("y", Build.Float)]));
        using IrModel m = OnnxImporter.Load(bytes);
        IrNode n = m.Graph.Nodes[0];
        Assert.Null(n.Inputs[1]);
        Assert.Equal("", n.InputNames[1]);
        Assert.Equal("hi", n.Inputs[2]!.Name);
    }

    [Fact]
    public void TypedFieldsBecomeLittleEndianBytes()
    {
        TensorData Load(int dtype, Action<Pb> add)
        {
            Pb tp = new Pb().PackedInts(1, 3).Int(2, dtype).Str(8, "t");
            add(tp);
            return OnnxImporter.LoadTensor(tp.ToArray());
        }

        Assert.Equal([0, 0, 192, 63, 0, 0, 0, 192, 0, 0, 0, 0], Load(Build.Float, t => t.PackedFloats(4, 1.5f, -2f, 0f)).RawData.ToArray());
        Assert.Equal([255, 255, 255, 255, 2, 0, 0, 0, 3, 0, 0, 0], Load(Build.Int32, t => t.PackedInts(5, -1, 2, 3)).RawData.ToArray());
        Assert.Equal([1, 2, 255], Load(Build.UInt8, t => t.PackedInts(5, 1, 2, 255)).RawData.ToArray());
        Assert.Equal([1, 0, 1], Load(Build.Bool, t => t.PackedInts(5, 1, 0, 7)).RawData.ToArray());
        Assert.Equal([0, 0x3C, 0, 0x40, 0xFF, 0xFF], Load(Build.Float16, t => t.PackedInts(5, 0x3C00, 0x4000, 0xFFFF)).RawData.ToArray());
        Assert.Equal(24, Load(Build.Int64, t => t.PackedInts(7, -1, 0, long.MaxValue)).RawLength);
        Assert.Equal([1, 0, 0, 0, 2, 0, 0, 0, 255, 255, 255, 255], Load(Build.UInt32, t => t.PackedInts(11, 1, 2, uint.MaxValue)).RawData.ToArray());
        Assert.Equal(24, Load(Build.UInt64, t => t.PackedInts(11, 1, 2, -1)).RawLength);
        // Unpacked repeated entries are legal too.
        TensorData unpacked = Load(Build.Int32, t => t.Int(5, 7).Int(5, 8).Int(5, 9));
        Assert.Equal(3, unpacked.ElementCount);
        Assert.Equal(12, unpacked.RawLength);
    }

    [Fact]
    public void DataThatDoesNotMatchTheShapeIsRejected()
    {
        Pb T(Action<Pb> add) { var t = new Pb().PackedInts(1, 2, 2).Int(2, Build.Float); add(t); return t; }
        var e = Assert.Throws<OnnxImportException>(() => OnnxImporter.LoadTensor(T(t => t.Bytes(9, new byte[15])).ToArray()));
        Assert.Equal(OnnxErrorKind.InvalidModel, e.Kind);
        Assert.Contains("15", e.Message, StringComparison.Ordinal);
        Assert.Throws<OnnxImportException>(() => OnnxImporter.LoadTensor(T(t => t.PackedFloats(4, 1, 2, 3)).ToArray()));
        Assert.Throws<OnnxImportException>(() => OnnxImporter.LoadTensor(T(_ => { }).ToArray())); // 4 elements, no data
        Assert.Throws<OnnxImportException>(() => OnnxImporter.LoadTensor(T(t => t.PackedInts(5, 1, 2, 3, 4)).ToArray())); // int32_data on a float tensor
    }

    [Fact]
    public void SubByteAndStringTensors()
    {
        TensorData i4 = OnnxImporter.LoadTensor(new Pb().PackedInts(1, 5).Int(2, Build.Int4).Bytes(9, new byte[3]).ToArray());
        Assert.Equal(3, i4.RawLength);
        Assert.False(i4.TryGetDType(out _));
        Assert.Throws<NotSupportedException>(() => i4.ToTensor());
        Assert.Throws<OnnxImportException>(() => OnnxImporter.LoadTensor(new Pb().PackedInts(1, 5).Int(2, Build.Int4).Bytes(9, new byte[4]).ToArray()));

        TensorData s = OnnxImporter.LoadTensor(new Pb().PackedInts(1, 2).Int(2, Build.String).Bytes(6, "ab"u8).Bytes(6, ""u8).ToArray());
        Assert.Equal(2, s.Strings.Length);
        Assert.Equal("ab"u8.ToArray(), s.Strings[0].ToArray());
        Assert.Throws<OnnxImportException>(() => OnnxImporter.LoadTensor(new Pb().PackedInts(1, 3).Int(2, Build.String).Bytes(6, "ab"u8).ToArray()));
    }

    [Fact]
    public void SegmentsAreUnsupported()
    {
        Pb t = new Pb().PackedInts(1, 1).Int(2, Build.Float).Msg(3, new Pb().Int(1, 0).Int(2, 1)).Bytes(9, new byte[4]);
        var e = Assert.Throws<OnnxImportException>(() => OnnxImporter.LoadTensor(t.ToArray()));
        Assert.Equal(OnnxErrorKind.Unsupported, e.Kind);
    }

    [Fact]
    public void SubgraphsSeeOuterValuesAndAreCaptured()
    {
        Pb Branch(string name, string result) => Build.Graph(name,
            [Build.Node("Add", ["x", "w"], [result])], null, [Build.ValueInfo(result, Build.Float)]);
        byte[] bytes = Build.Model(Build.Graph("main",
            [
                Build.Node("Identity", ["a"], ["w"]),
                Build.Node("If", ["cond"], ["out"], "if", Build.AttrGraph("then_branch", Branch("t", "r1")), Build.AttrGraph("else_branch", Branch("e", "r2"))),
            ],
            [Build.ValueInfo("a", Build.Float, 1), Build.ValueInfo("x", Build.Float, 1), Build.ValueInfo("cond", Build.Bool)],
            [Build.ValueInfo("out", Build.Float, 1)]));
        using IrModel m = OnnxImporter.Load(bytes);
        IrNode iff = m.Graph.Nodes.Single(n => n.OpType == "If");
        IrGraph then = iff.FindAttribute("then_branch")!.Graph;
        Assert.Same(m.Graph, then.Parent);
        Assert.Equal(["x", "w"], then.Captures.Select(v => v.Name));
        Assert.Equal(["x", "w"], then.Nodes[0].Inputs.Select(v => v!.Name));
        Assert.Same(m.Graph.FindValue("w"), then.Nodes[0].Inputs[1]);
        // The If node counts as a consumer of what its branches read.
        Assert.Contains(iff, m.Graph.FindValue("w")!.Consumers);
        // And the producer of w must come first.
        Assert.Equal(["Identity", "If"], m.Graph.Nodes.Select(n => n.OpType));
    }

    [Fact]
    public void NodesOutOfOrderAreSorted()
    {
        byte[] bytes = Build.Model(Build.Graph("g",
            [Build.Node("Relu", ["t"], ["y"], "second"), Build.Node("Neg", ["x"], ["t"], "first")],
            [Build.ValueInfo("x", Build.Float)], [Build.ValueInfo("y", Build.Float)]));
        using IrModel m = OnnxImporter.Load(bytes);
        Assert.True(m.Graph.NodesWereReordered);
        Assert.Equal(["first", "second"], m.Graph.Nodes.Select(n => n.Name));
        using IrModel ok = OnnxImporter.Load(Build.AddModel());
        Assert.False(ok.Graph.NodesWereReordered);
    }

    [Theory]
    [InlineData("dangling")]
    [InlineData("duplicate-output")]
    [InlineData("duplicate-input")]
    [InlineData("cycle")]
    [InlineData("missing-graph-output")]
    [InlineData("unknown-domain")]
    public void BrokenGraphsAreRejectedWithTypedErrors(string what)
    {
        Pb g = what switch
        {
            "dangling" => Build.Graph("g", [Build.Node("Neg", ["nope"], ["y"])], null, [Build.ValueInfo("y", Build.Float)]),
            "duplicate-output" => Build.Graph("g", [Build.Node("Neg", ["x"], ["y"]), Build.Node("Abs", ["x"], ["y"])], [Build.ValueInfo("x", Build.Float)], [Build.ValueInfo("y", Build.Float)]),
            "duplicate-input" => Build.Graph("g", [], [Build.ValueInfo("x", Build.Float), Build.ValueInfo("x", Build.Float)], null),
            "cycle" => Build.Graph("g", [Build.Node("Neg", ["b"], ["a"]), Build.Node("Abs", ["a"], ["b"])], null, [Build.ValueInfo("a", Build.Float)]),
            "missing-graph-output" => Build.Graph("g", [], [Build.ValueInfo("x", Build.Float)], [Build.ValueInfo("zzz", Build.Float)]),
            _ => Build.Graph("g", [new Pb().Str(1, "x").Str(2, "y").Str(4, "Foo").Str(7, "vendor")], [Build.ValueInfo("x", Build.Float)], [Build.ValueInfo("y", Build.Float)]),
        };
        var e = Assert.Throws<OnnxImportException>(() => OnnxImporter.Load(Build.Model(g)));
        Assert.Equal(OnnxErrorKind.InvalidModel, e.Kind);
        Assert.NotNull(e.FieldPath);
    }

    [Fact]
    public void AttributesOfEveryKind()
    {
        Pb attrs = new Pb().Str(1, "r").Float(2, 0.5f).Int(20, 1);
        byte[] bytes = Build.Model(Build.Graph("g",
            [Build.Node("Foo", ["x"], ["y"], "n", attrs, Build.AttrInt("i", -7),
                new Pb().Str(1, "ints").PackedInts(8, 1, -2, 3).Int(20, 7),
                new Pb().Str(1, "floats").PackedFloats(7, 1f, 2f).Int(20, 6),
                new Pb().Str(1, "s").Bytes(4, "hi"u8).Int(20, 3),
                new Pb().Str(1, "untyped").Int(3, 9),
                new Pb().Str(1, "t").Msg(5, Build.FloatTensor("", [1], 3f)).Int(20, 4))],
            [Build.ValueInfo("x", Build.Float)], [Build.ValueInfo("y", Build.Float)]));
        using IrModel m = OnnxImporter.Load(bytes);
        IrNode n = m.Graph.Nodes[0];
        Assert.Equal(0.5f, n.FindAttribute("r")!.Real);
        Assert.Equal(-7, n.FindAttribute("i")!.Whole);
        Assert.Equal([1L, -2L, 3L], n.FindAttribute("ints")!.Wholes.ToArray());
        Assert.Equal([1f, 2f], n.FindAttribute("floats")!.Reals.ToArray());
        Assert.Equal("hi", n.FindAttribute("s")!.GetText());
        Assert.Equal(IrAttributeType.Whole, n.FindAttribute("untyped")!.Type); // type inferred from the field that is set
        Assert.Equal(3f, n.FindAttribute("t")!.Tensor.ToTensor().Data<float>()[0]);
        Assert.Throws<InvalidOperationException>(() => n.FindAttribute("r")!.Whole);
    }

    [Fact]
    public void TypesOfEveryKind()
    {
        Pb Seq(Pb inner) => new Pb().Msg(4, new Pb().Msg(1, inner));
        Pb tensor = new Pb().Msg(1, new Pb().Int(1, 7).Msg(2, new Pb().Msg(1, new Pb().Str(2, "N")).Msg(1, new Pb().Int(1, 4)).Msg(1, new Pb())));
        byte[] bytes = Build.Model(Build.Graph("g", [],
            [new Pb().Str(1, "a").Msg(2, tensor), new Pb().Str(1, "b").Msg(2, Seq(tensor)), new Pb().Str(1, "c").Msg(2, new Pb().Msg(9, new Pb().Msg(1, tensor))),
             new Pb().Str(1, "d").Msg(2, new Pb().Msg(5, new Pb().Int(1, 8).Msg(2, tensor))), new Pb().Str(1, "e").Msg(2, new Pb())], null));
        using IrModel m = OnnxImporter.Load(bytes);
        Assert.Equal("tensor(7)[N,4,?]", m.Graph.Inputs[0].Type!.ToString());
        Assert.Equal("seq<tensor(7)[N,4,?]>", m.Graph.Inputs[1].Type!.ToString());
        Assert.Equal("opt<tensor(7)[N,4,?]>", m.Graph.Inputs[2].Type!.ToString());
        Assert.Equal("map<8,tensor(7)[N,4,?]>", m.Graph.Inputs[3].Type!.ToString());
        Assert.Null(m.Graph.Inputs[4].Type);
        Assert.False(m.Graph.Inputs[0].Type!.Dims[0].IsFixed);
        Assert.True(m.Graph.Inputs[0].Type!.Dims[1].IsFixed);
    }

    [Fact]
    public void ModelFieldsAndFunctions()
    {
        var fn = new Pb().Str(1, "F").Str(10, "local").Str(4, "a").Str(5, "b").Str(6, "alpha")
            .Msg(7, Build.Node("Neg", ["a"], ["b"])).Msg(9, new Pb().Str(1, "").Int(2, 14));
        byte[] bytes = new Pb().Int(1, 10).Str(2, "me").Str(3, "1.2").Int(5, 4).Msg(7, Build.Graph("g", [], [Build.ValueInfo("x", Build.Float)], null))
            .Msg(8, new Pb().Str(1, "").Int(2, 21)).Msg(8, new Pb().Str(1, "ai.onnx.ml").Int(2, 3))
            .Msg(14, new Pb().Str(1, "k").Str(2, "v")).Msg(25, fn).ToArray();
        using IrModel m = OnnxImporter.Load(bytes);
        Assert.Equal("me", m.ProducerName);
        Assert.Equal(21, m.DefaultOpset);
        Assert.Equal(4, m.ModelVersion);
        Assert.Equal("v", m.Metadata[0].Value);
        IrFunction f = Assert.Single(m.Functions);
        Assert.Equal(("local", "F"), (f.Domain, f.Name));
        Assert.Equal(["alpha"], f.AttributeNames);
        Assert.Equal("Neg", f.Nodes[0].OpType);
    }

    [Fact]
    public void OldIrVersionsAndMissingPartsAreRejected()
    {
        Assert.Equal(OnnxErrorKind.InvalidModel, Assert.Throws<OnnxImportException>(() => OnnxImporter.Load(Build.Model(Build.Graph("g", []), ir: 2))).Kind);
        Assert.Equal(OnnxErrorKind.InvalidModel, Assert.Throws<OnnxImportException>(() => OnnxImporter.Load(new Pb().Int(1, 8).ToArray())).Kind);
        Assert.Equal(OnnxErrorKind.InvalidModel, Assert.Throws<OnnxImportException>(() => OnnxImporter.Load([])).Kind);
        using IrModel newer = OnnxImporter.Load(Build.Model(Build.Graph("g", []), ir: 99));
        Assert.Equal(99, newer.IrVersion);
    }

    [Fact]
    public void LoadFromFileStreamAndDisposal()
    {
        string path = Path.Combine(Path.GetTempPath(), "onemodel-" + Guid.NewGuid().ToString("N") + ".onnx");
        File.WriteAllBytes(path, Build.AddModel());
        try
        {
            TensorData kept;
            using (IrModel m = OnnxImporter.Load(path))
            {
                kept = m.Graph.FindValue("w")!.Initializer!;
                Assert.Equal(16, kept.RawData.Length);
            }

            Assert.Throws<ObjectDisposedException>(() => kept.RawData.Length);
            using IrModel s = OnnxImporter.Load(new MemoryStream(File.ReadAllBytes(path)));
            Assert.Single(s.Graph.Nodes);
            Assert.Throws<FileNotFoundException>(() => OnnxImporter.Load(path + ".missing"));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void LargeFilesAreMappedAndStillCorrect()
    {
        // Above the mapping threshold the file is memory-mapped; the data must read back the same.
        string path = Path.Combine(Path.GetTempPath(), "onemodel-" + Guid.NewGuid().ToString("N") + ".onnx");
        var big = new byte[3 << 20];
        new Random(5).NextBytes(big);
        byte[] bytes = Build.Model(Build.Graph("g", [], null, null, [Build.RawTensor("big", Build.UInt8, [big.Length], big)]));
        File.WriteAllBytes(path, bytes);
        try
        {
            using IrModel m = OnnxImporter.Load(path);
            Assert.True(m.Graph.FindValue("big")!.Initializer!.RawData.SequenceEqual(big));
            using IrModel unmapped = OnnxImporter.Load(path, new ModelLoadOptions { MemoryMap = false });
            Assert.True(unmapped.Graph.FindValue("big")!.Initializer!.RawData.SequenceEqual(big));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void CancellationIsHonoured()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.ThrowsAny<OperationCanceledException>(() => OnnxImporter.Load(Build.AddModel(), null, cts.Token));
    }
}
