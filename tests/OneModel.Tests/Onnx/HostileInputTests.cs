using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;

namespace RecursiveAI.OneModel.Tests.Onnx;

/// <summary>
/// Hostile files. Each case is built here, written to testdata/onnx/hostile/&lt;kind&gt;-&lt;name&gt;.onnx (set ONEMODEL_WRITE_TESTDATA=1 to
/// regenerate) and must be rejected with an <see cref="OnnxImportException"/> of that kind, from memory and from the file.
/// </summary>
public class HostileInputTests
{
    private static string RepoRoot([System.Runtime.CompilerServices.CallerFilePath] string path = "")
    {
        string? dir = Path.GetDirectoryName(path);
        while (dir is not null && !File.Exists(Path.Combine(dir, "global.json")))
        {
            dir = Path.GetDirectoryName(dir);
        }

        return dir ?? throw new InvalidOperationException("global.json not found above " + path);
    }

    private static string HostileDir => Path.Combine(RepoRoot(), "testdata", "onnx", "hostile");

    /// <summary>A graph nested <paramref name="depth"/> levels deep through If branches.</summary>
    private static byte[] Nested(int depth)
    {
        Pb g = Build.Graph("leaf", []);
        for (int i = 0; i < depth; i++)
        {
            g = Build.Graph("g" + i, [Build.Node("If", ["c"], ["o" + i], "", Build.AttrGraph("then_branch", g))], i == depth - 1 ? [Build.ValueInfo("c", Build.Bool)] : null);
        }

        return Build.Model(g);
    }

    private static byte[] ModelWithTensor(Pb tensor) => Build.Model(Build.Graph("g", [], null, null, [tensor]));

    private static byte[] ModelWithBytes(Func<Pb, Pb> graphBody) => new Pb().Int(1, 8).Msg(7, graphBody(new Pb())).Msg(8, new Pb().Str(1, "").Int(2, 13)).ToArray();

    public static TheoryData<string, OnnxErrorKind, byte[]> Cases()
    {
        var data = new TheoryData<string, OnnxErrorKind, byte[]>();
        foreach ((string name, OnnxErrorKind kind, byte[] bytes) in CaseList())
        {
            data.Add(name, kind, bytes);
        }

        return data;
    }

    private static List<(string Name, OnnxErrorKind Kind, byte[] Bytes)> CaseList()
    {
        byte[] valid = Build.AddModel();
        var data = new List<(string, OnnxErrorKind, byte[])>();
        void Add(string name, OnnxErrorKind kind, byte[] bytes) => data.Add((name, kind, bytes));

        Add("truncated-half", OnnxErrorKind.Malformed, valid[..(valid.Length / 2)]);
        Add("truncated-in-varint", OnnxErrorKind.Malformed, [0x08, 0x80]);
        Add("varint-eleven-bytes", OnnxErrorKind.Malformed, [0x08, .. Enumerable.Repeat((byte)0x80, 10), 0x00]);
        Add("varint-overflow-64", OnnxErrorKind.Malformed, [0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x02]);
        Add("length-past-end", OnnxErrorKind.Malformed, [0x3A, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F, 0x01]);
        Add("length-max-uint64", OnnxErrorKind.Malformed, [0x3A, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01]);
        Add("group-start", OnnxErrorKind.Malformed, [0x0B, 0x0C]);
        Add("field-zero", OnnxErrorKind.Malformed, [0x00, 0x01]);
        Add("wire-type-six", OnnxErrorKind.Malformed, [0x0E, 0x00]);
        Add("fixed32-cut", OnnxErrorKind.Malformed, [0x0D, 0x01, 0x02]);
        Add("fixed64-cut", OnnxErrorKind.Malformed, [0x09, 0x01, 0x02, 0x03]);
        Add("bad-utf8-name", OnnxErrorKind.Malformed, ModelWithBytes(g => g.Bytes(2, [0xC3, 0x28])));
        Add("nesting-300", OnnxErrorKind.Malformed, Nested(300));
        Add("packed-floats-odd", OnnxErrorKind.Malformed, ModelWithTensor(new Pb().PackedInts(1, 1).Int(2, 1).Bytes(4, new byte[7])));
        Add("empty-file", OnnxErrorKind.InvalidModel, []);
        Add("no-graph", OnnxErrorKind.InvalidModel, new Pb().Int(1, 8).ToArray());
        Add("dims-product-overflow", OnnxErrorKind.InvalidModel, ModelWithTensor(new Pb().PackedInts(1, 1L << 40, 1L << 40, 1L << 40).Int(2, 1).Str(8, "t")));
        Add("dims-2-pow-40-no-data", OnnxErrorKind.Unsupported, ModelWithTensor(new Pb().PackedInts(1, 1L << 40).Int(2, 1).Str(8, "t")));
        Add("dims-negative", OnnxErrorKind.InvalidModel, ModelWithTensor(new Pb().PackedInts(1, -1).Int(2, 1).Str(8, "t")));
        Add("dims-zero-then-huge", OnnxErrorKind.InvalidModel, ModelWithTensor(new Pb().PackedInts(1, 0, long.MaxValue, long.MaxValue).Int(2, 1).Bytes(9, new byte[4]).Str(8, "t")));
        Add("raw-size-mismatch", OnnxErrorKind.InvalidModel, ModelWithTensor(new Pb().PackedInts(1, 4).Int(2, 1).Bytes(9, new byte[3]).Str(8, "t")));
        Add("unknown-dtype", OnnxErrorKind.Unsupported, ModelWithTensor(new Pb().PackedInts(1, 1).Int(2, 99).Bytes(9, new byte[4]).Str(8, "t")));
        Add("many-strings-few-dims", OnnxErrorKind.InvalidModel, ModelWithTensor(new Pb().PackedInts(1, 1).Int(2, 8).Bytes(6, ""u8).Bytes(6, ""u8).Str(8, "t")));
        Add("external-no-directory", OnnxErrorKind.ExternalData, ModelWithTensor(External("w.bin", null, null)));
        Add("sparse-missing-indices", OnnxErrorKind.InvalidModel, ModelWithBytes(g => g.Msg(15, new Pb().Msg(1, Build.FloatTensor("s", [1], 1f)))));
        return data;
    }

    private static Pb External(string location, string? offset, string? length)
    {
        var t = new Pb().PackedInts(1, 4).Int(2, 1).Str(8, "t").Int(14, 1).Msg(13, new Pb().Str(1, "location").Str(2, location));
        if (offset is not null)
        {
            t.Msg(13, new Pb().Str(1, "offset").Str(2, offset));
        }

        if (length is not null)
        {
            t.Msg(13, new Pb().Str(1, "length").Str(2, length));
        }

        return t;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void RejectedWithTheRightKind(string name, OnnxErrorKind kind, byte[] bytes)
    {
        var e = Assert.Throws<OnnxImportException>(() => OnnxImporter.Load(bytes));
        Assert.Equal(kind, e.Kind);
        Assert.False(string.IsNullOrEmpty(e.Message));
        _ = name;
    }

    [Fact]
    public void CommittedReprosMatchTheGeneratorAndAreRejected()
    {
        string dir = HostileDir;
        bool write = Environment.GetEnvironmentVariable("ONEMODEL_WRITE_TESTDATA") == "1";
        if (write)
        {
            Directory.CreateDirectory(dir);
        }

        var expectedFiles = new HashSet<string>();
        foreach ((string name, OnnxErrorKind kind, byte[] bytes) in CaseList())
        {
            string file = Path.Combine(dir, $"{kind}-{name}.onnx");
            expectedFiles.Add(file);
            if (write)
            {
                File.WriteAllBytes(file, bytes);
            }

            Assert.True(File.Exists(file), $"missing {file}; run with ONEMODEL_WRITE_TESTDATA=1");
            Assert.Equal(bytes, File.ReadAllBytes(file));
            Assert.True(new FileInfo(file).Length < 1 << 20);
            var e = Assert.Throws<OnnxImportException>(() => OnnxImporter.Load(file));
            Assert.Equal(kind, e.Kind);
            Assert.Equal(file, e.InputName);
        }

        Assert.All(Directory.EnumerateFiles(dir, "*.onnx"), f => Assert.Contains(f, expectedFiles));
    }

    [Fact]
    public void EveryPrefixOfAValidModelFailsCleanlyOrLoads()
    {
        // Whatever the cut point, the only acceptable outcomes are a model or an OnnxImportException.
        byte[] valid = Build.Model(Build.Graph("g",
            [Build.Node("Add", ["x", "w"], ["y"], "add", Build.AttrInt("a", 1))],
            [Build.ValueInfo("x", Build.Float, 2, 2)], [Build.ValueInfo("y", Build.Float, 2, 2)],
            [Build.FloatTensor("w", [2, 2], 1, 2, 3, 4)]));
        int failed = 0;
        for (int n = 0; n < valid.Length; n++)
        {
            try
            {
                using IrModel m = OnnxImporter.Load(valid[..n]);
            }
            catch (OnnxImportException)
            {
                failed++;
            }
        }

        Assert.True(failed > valid.Length / 2);
    }

    [Fact]
    public void EverySingleByteFlipFailsCleanlyOrLoads()
    {
        byte[] valid = Build.AddModel();
        foreach (byte pattern in new byte[] { 0x00, 0xFF, 0x80, 0x7F })
        {
            for (int i = 0; i < valid.Length; i++)
            {
                byte[] copy = (byte[])valid.Clone();
                copy[i] = pattern;
                try
                {
                    using IrModel m = OnnxImporter.Load(copy);
                    foreach (IrValue v in m.Graph.Values)
                    {
                        _ = v.Initializer?.RawData.Length;
                    }
                }
                catch (OnnxImportException)
                {
                }
            }
        }
    }

    [Fact]
    public void NestingWithinTheLimitIsFineAndTheLimitIsConfigurable()
    {
        using IrModel ok = OnnxImporter.Load(Nested(20), new OnnxLoadOptions { MaxNesting = 128 });
        Assert.NotNull(ok.Graph);
        var e = Assert.Throws<OnnxImportException>(() => OnnxImporter.Load(Nested(20), new OnnxLoadOptions { MaxNesting = 30 }));
        Assert.Equal(OnnxErrorKind.Malformed, e.Kind);
        Assert.Contains("nested", e.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ErrorsNameTheOffsetAndFieldPath()
    {
        byte[] bytes = Build.Model(Build.Graph("g", [Build.Node("Add", ["x"], ["y"], "a"), new Pb().Bytes(4, [0xC3, 0x28])]));
        var e = Assert.Throws<OnnxImportException>(() => OnnxImporter.Load(bytes));
        Assert.Contains("node[1]", e.FieldPath, StringComparison.Ordinal);
        Assert.True(e.Offset > 0);
        Assert.Contains("ModelProto.graph.node[1]", e.Message, StringComparison.Ordinal);
    }
}
