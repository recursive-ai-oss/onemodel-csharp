using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Text;
using RecursiveAI.OneModel.Format;
using RecursiveAI.OneModel.Graph;

namespace RecursiveAI.OneModel.Tests.Format;

/// <summary>
/// Truncated and corrupt files. Each case is built here from a hand-assembled valid file (the header and manifest are written
/// by this test straight from docs/format.md, independently of the library's writer), written to
/// testdata/format/hostile/&lt;kind&gt;-&lt;name&gt;.onemodel (set ONEMODEL_WRITE_TESTDATA=1 to regenerate) and must be rejected with a
/// <see cref="ModelFormatException"/> of that kind, from memory and from the file.
/// </summary>
public class FormatHostileTests
{
    private const string Base =
        """
        {"format":"onemodel","version":1,"ir_version":8,"opsets":[{"domain":"","version":13}],
         "graph":{"name":"g","inputs":[{"name":"x","type":{"kind":"tensor","elem":1,"shape":[2,2]}}],"outputs":[{"name":"y"}],
          "initializers":[{"name":"w","dtype":1,"dims":[2,2],"data":{"offset":0,"length":16}}],
          "nodes":[{"name":"add","op":"Add","inputs":["x","w"],"outputs":["y"]}]}}
        """;

    private static readonly byte[] Data = [.. BitConverter.GetBytes(1f), .. BitConverter.GetBytes(2f), .. BitConverter.GetBytes(3f), .. BitConverter.GetBytes(4f)];

    private static string HostileDir([CallerFilePath] string file = "") =>
        Path.GetFullPath(Path.Combine(Path.GetDirectoryName(file)!, "..", "..", "..", "testdata", "format", "hostile"));

    /// <summary>A file assembled by hand: header, manifest, zero padding to a multiple of 64, data.</summary>
    private static byte[] File(string manifest, byte[]? data = null)
    {
        data ??= Data;
        byte[] m = Encoding.UTF8.GetBytes(manifest);
        long dataOffset = (64 + m.Length + 63) / 64 * 64;
        var file = new byte[dataOffset + data.Length];
        "ONEMODEL"u8.CopyTo(file);
        BinaryPrimitives.WriteUInt32LittleEndian(file.AsSpan(8), 1);
        BinaryPrimitives.WriteUInt64LittleEndian(file.AsSpan(16), 64);
        BinaryPrimitives.WriteUInt64LittleEndian(file.AsSpan(24), (ulong)m.Length);
        BinaryPrimitives.WriteUInt64LittleEndian(file.AsSpan(32), (ulong)dataOffset);
        BinaryPrimitives.WriteUInt64LittleEndian(file.AsSpan(40), (ulong)data.Length);
        BinaryPrimitives.WriteUInt64LittleEndian(file.AsSpan(48), (ulong)file.Length);
        m.CopyTo(file, 64);
        data.CopyTo(file, dataOffset);
        return file;
    }

    private static byte[] Patch(Action<byte[]> edit)
    {
        byte[] f = File(Base);
        edit(f);
        return f;
    }

    private static byte[] Edit(string from, string to)
    {
        Assert.Contains(from, Base, StringComparison.Ordinal);
        return File(Base.Replace(from, to, StringComparison.Ordinal));
    }

    private static string Nest(int depth)
    {
        // A chain of If nodes, each holding the next graph in its then_branch.
        string g = "{\"name\":\"leaf\"}";
        for (int i = 0; i < depth; i++)
        {
            g = "{\"name\":\"g" + i + "\",\"nodes\":[{\"op\":\"If\",\"inputs\":[],\"outputs\":[\"o" + i + "\"],\"attributes\":[{\"name\":\"then_branch\",\"type\":5,\"value\":" + g + "}]}]}";
        }

        return "{\"format\":\"onemodel\",\"version\":1,\"ir_version\":8,\"opsets\":[{\"domain\":\"\",\"version\":13}],\"graph\":" + g + "}";
    }

    private static List<(string Name, ModelFormatErrorKind Kind, byte[] Bytes)> CaseList()
    {
        var data = new List<(string, ModelFormatErrorKind, byte[])>();
        void Add(string name, ModelFormatErrorKind kind, byte[] bytes) => data.Add((name, kind, bytes));
        const ModelFormatErrorKind Bad = ModelFormatErrorKind.Malformed, Invalid = ModelFormatErrorKind.InvalidModel, Unsup = ModelFormatErrorKind.Unsupported;
        byte[] valid = File(Base);

        // Header.
        Add("empty-file", Bad, []);
        Add("short-header", Bad, valid[..40]);
        Add("bad-magic", Bad, Patch(f => f[0] = (byte)'X'));
        Add("onnx-bytes", Bad, [0x08, 0x08, 0x12, 0x04, .. new byte[100]]);
        Add("version-2", Unsup, Patch(f => f[8] = 2));
        Add("flags-set", Unsup, Patch(f => f[12] = 1));
        Add("reserved-set", Bad, Patch(f => f[56] = 1));
        Add("truncated-in-data", Bad, valid[..^3]);
        Add("truncated-in-manifest", Bad, valid[..100]);
        Add("trailing-byte", Bad, [.. valid, 0]);
        Add("manifest-offset-0", Bad, Patch(f => f[16] = 0));
        Add("manifest-length-past-end", Bad, Patch(f => BinaryPrimitives.WriteUInt64LittleEndian(f.AsSpan(24), (ulong)f.Length)));
        Add("manifest-length-2-pow-63", Bad, Patch(f => BinaryPrimitives.WriteUInt64LittleEndian(f.AsSpan(24), 1UL << 63)));
        Add("manifest-length-max", Bad, Patch(f => BinaryPrimitives.WriteUInt64LittleEndian(f.AsSpan(24), ulong.MaxValue)));
        Add("data-offset-unaligned", Bad, Patch(f => f[32] += 1));
        Add("data-offset-inside-manifest", Bad, Patch(f => BinaryPrimitives.WriteUInt64LittleEndian(f.AsSpan(32), 64)));
        Add("data-offset-past-end", Bad, Patch(f => BinaryPrimitives.WriteUInt64LittleEndian(f.AsSpan(32), (ulong)(f.Length + 64))));
        Add("data-length-short", Bad, Patch(f => f[40] -= 1));
        Add("file-length-lies", Bad, Patch(f => f[48] += 1));

        // Manifest syntax.
        Add("json-garbage", Bad, File("not json at all"));
        Add("json-trailing-comma", Bad, File(Base.Replace("\"version\":1,", "\"version\":1,,", StringComparison.Ordinal)));
        Add("json-cut", Bad, File(Base[..(Base.Length / 2)]));
        Add("json-depth-5000", Bad, File(new string('[', 5000) + new string(']', 5000)));
        Add("json-bad-utf8", Bad, Patch(f => f[64 + Base.IndexOf("\"g\"", StringComparison.Ordinal) + 1] = 0xFF));
        Add("manifest-is-array", Bad, File("[]"));
        Add("manifest-empty-object", Bad, File("{}"));

        // Manifest meaning.
        Add("wrong-format-name", Bad, Edit("\"onemodel\"", "\"other\""));
        Add("manifest-version-mismatch", Unsup, Edit("\"version\":1,", "\"version\":2,"));
        Add("version-is-float", Bad, Edit("\"version\":1,", "\"version\":1.5,"));
        Add("ir-version-old", Invalid, Edit("\"ir_version\":8", "\"ir_version\":2"));
        Add("ir-version-string", Bad, Edit("\"ir_version\":8", "\"ir_version\":\"8\""));
        Add("no-graph", Bad, File("{\"format\":\"onemodel\",\"version\":1,\"ir_version\":8}"));
        Add("graph-not-object", Bad, File("{\"format\":\"onemodel\",\"version\":1,\"ir_version\":8,\"graph\":[]}"));
        Add("node-op-number", Bad, Edit("\"op\":\"Add\"", "\"op\":7"));
        Add("node-without-op", Bad, Edit("\"op\":\"Add\",", ""));
        Add("inputs-not-array", Bad, Edit("\"inputs\":[\"x\",\"w\"]", "\"inputs\":\"x\""));
        Add("input-name-number", Bad, Edit("\"inputs\":[\"x\",\"w\"]", "\"inputs\":[\"x\",3]"));
        Add("tensor-dtype-unknown", Unsup, Edit("\"dtype\":1", "\"dtype\":99"));
        Add("tensor-dtype-zero", Unsup, Edit("\"dtype\":1", "\"dtype\":0"));
        Add("tensor-length-mismatch", Invalid, Edit("\"length\":16", "\"length\":12"));
        Add("tensor-offset-past-data", Bad, Edit("\"offset\":0", "\"offset\":64"));
        Add("tensor-offset-unaligned", Bad, Edit("\"offset\":0", "\"offset\":4"));
        Add("tensor-offset-negative", Bad, Edit("\"offset\":0", "\"offset\":-64"));
        Add("tensor-offset-int64-max", Bad, Edit("\"offset\":0", "\"offset\":9223372036854775807"));
        Add("tensor-offset-beyond-int64", Bad, Edit("\"offset\":0", "\"offset\":99999999999999999999"));
        Add("tensor-without-data", Bad, Edit(",\"data\":{\"offset\":0,\"length\":16}", ""));
        Add("tensor-dim-negative", Invalid, Edit("\"dims\":[2,2]", "\"dims\":[2,-2]"));
        Add("tensor-dims-overflow", Invalid, Edit("\"dims\":[2,2]", "\"dims\":[4294967296,4294967296,4294967296]"));
        Add("tensor-over-2-gb", Unsup, Edit("\"dims\":[2,2]", "\"dims\":[1099511627776]"));
        Add("tensor-dim-float", Bad, Edit("\"dims\":[2,2]", "\"dims\":[2,2.5]"));
        Add("string-tensor-count", Invalid, Edit("\"dtype\":1,\"dims\":[2,2],\"data\":{\"offset\":0,\"length\":16}", "\"dtype\":8,\"dims\":[2,2],\"strings\":[\"YQ==\"]"));
        Add("string-tensor-bad-base64", Bad, Edit("\"dtype\":1,\"dims\":[2,2],\"data\":{\"offset\":0,\"length\":16}", "\"dtype\":8,\"dims\":[1],\"strings\":[\"@@@\"]"));
        Add("attribute-type-99", Unsup, Edit("\"outputs\":[\"y\"]}]", "\"outputs\":[\"y\"],\"attributes\":[{\"name\":\"a\",\"type\":99,\"value\":1}]}]"));
        Add("attribute-no-value", Bad, Edit("\"outputs\":[\"y\"]}]", "\"outputs\":[\"y\"],\"attributes\":[{\"name\":\"a\",\"type\":2}]}]"));
        Add("attribute-int-is-string", Bad, Edit("\"outputs\":[\"y\"]}]", "\"outputs\":[\"y\"],\"attributes\":[{\"name\":\"a\",\"type\":2,\"value\":\"1\"}]}]"));
        Add("attribute-float-bits-negative", Bad, Edit("\"outputs\":[\"y\"]}]", "\"outputs\":[\"y\"],\"attributes\":[{\"name\":\"a\",\"type\":1,\"value\":-1}]}]"));
        Add("attribute-undefined-no-ref", Invalid, Edit("\"outputs\":[\"y\"]}]", "\"outputs\":[\"y\"],\"attributes\":[{\"name\":\"a\",\"type\":0}]}]"));
        Add("type-kind-unknown", Unsup, Edit("\"kind\":\"tensor\"", "\"kind\":\"blob\""));
        Add("type-shape-negative", Bad, Edit("\"shape\":[2,2]", "\"shape\":[2,-5]"));
        Add("duplicate-name", Invalid, Edit("\"outputs\":[\"y\"]}]", "\"outputs\":[\"x\"]}]"));
        Add("undefined-input", Invalid, Edit("\"inputs\":[\"x\",\"w\"]", "\"inputs\":[\"x\",\"nowhere\"]"));
        Add("graph-output-undefined", Invalid, Edit("\"outputs\":[{\"name\":\"y\"}]", "\"outputs\":[{\"name\":\"zz\"}]"));
        Add("domain-not-imported", Invalid, Edit("\"op\":\"Add\"", "\"op\":\"Add\",\"domain\":\"com.example\""));
        Add("subgraphs-nested-100", Unsup, File(Nest(100)));
        return data;
    }

    public static TheoryData<string, ModelFormatErrorKind, byte[]> Cases()
    {
        var data = new TheoryData<string, ModelFormatErrorKind, byte[]>();
        foreach ((string name, ModelFormatErrorKind kind, byte[] bytes) in CaseList())
        {
            data.Add(name, kind, bytes);
        }

        return data;
    }

    [Fact]
    public void TheHandWrittenFileLoads()
    {
        using IrModel m = ModelFormat.Load(File(Base));
        Assert.Single(m.Graph.Nodes);
        Assert.Equal(Data, m.Graph.Initializers[0].RawData.ToArray());
        Assert.Equal("tensor(1)[2,2]", m.Graph.Inputs[0].Type!.ToString());
    }

    [Fact]
    public void NestingJustInsideTheLimitLoads()
    {
        using IrModel m = ModelFormat.Load(File(Nest(30)));
        Assert.NotNull(m.Graph);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void RejectedWithTheRightKind(string name, ModelFormatErrorKind kind, byte[] bytes)
    {
        var e = Assert.Throws<ModelFormatException>(() => ModelFormat.Load(bytes));
        Assert.Equal(kind, e.Kind);
        Assert.False(string.IsNullOrEmpty(e.Message));
        _ = name;
    }

    [Fact]
    public void ErrorsNameTheFieldAndWhatWasExpected()
    {
        var e = Assert.Throws<ModelFormatException>(() => ModelFormat.Load(Edit("\"length\":16", "\"length\":12")));
        Assert.Contains("graph.initializers[0].data.length", e.FieldPath, StringComparison.Ordinal);
        Assert.Contains("12", e.Message, StringComparison.Ordinal);
        Assert.Contains("16", e.Message, StringComparison.Ordinal);
        e = Assert.Throws<ModelFormatException>(() => ModelFormat.Load(Edit("\"inputs\":[\"x\",\"w\"]", "\"inputs\":[\"x\",3]")));
        Assert.Equal("manifest.graph.nodes[0].inputs[1]", e.FieldPath);
        Assert.Contains("expected a string", e.Message, StringComparison.Ordinal);
        e = Assert.Throws<ModelFormatException>(() => ModelFormat.Load(Patch(f => f[8] = 2)));
        Assert.Equal("version", e.FieldPath);
        Assert.Equal(8, e.Offset);
    }

    [Fact]
    public void CommittedReprosMatchTheGeneratorAndAreRejected()
    {
        string dir = HostileDir();
        bool write = Environment.GetEnvironmentVariable("ONEMODEL_WRITE_TESTDATA") == "1";
        if (write)
        {
            Directory.CreateDirectory(dir);
        }

        var expectedFiles = new HashSet<string>();
        foreach ((string name, ModelFormatErrorKind kind, byte[] bytes) in CaseList())
        {
            string file = Path.Combine(dir, $"{kind}-{name}.onemodel");
            expectedFiles.Add(file);
            if (write)
            {
                System.IO.File.WriteAllBytes(file, bytes);
            }

            Assert.True(System.IO.File.Exists(file), $"missing {file}; run with ONEMODEL_WRITE_TESTDATA=1");
            Assert.Equal(bytes, System.IO.File.ReadAllBytes(file));
            Assert.True(new FileInfo(file).Length < 1 << 20);
            var e = Assert.Throws<ModelFormatException>(() => ModelFormat.Load(file));
            Assert.Equal(kind, e.Kind);
            Assert.Equal(file, e.InputName);
        }

        Assert.All(Directory.EnumerateFiles(dir, "*.onemodel"), f => Assert.Contains(f, expectedFiles));
    }

    [Fact]
    public void EveryPrefixAndEverySingleByteFlipFailsCleanlyOrLoads()
    {
        byte[] valid = File(Base);
        int failed = 0;
        for (int n = 0; n < valid.Length; n++)
        {
            try
            {
                using IrModel m = ModelFormat.Load(valid[..n]);
            }
            catch (ModelFormatException)
            {
                failed++;
            }
        }

        Assert.Equal(valid.Length, failed);

        foreach (byte pattern in new byte[] { 0x00, 0xFF, 0x80, 0x22 })
        {
            for (int i = 0; i < valid.Length; i++)
            {
                byte[] copy = (byte[])valid.Clone();
                copy[i] = pattern;
                try
                {
                    using IrModel m = ModelFormat.Load(copy);
                    foreach (IrValue v in m.Graph.Values)
                    {
                        _ = v.Initializer?.RawData.Length;
                    }
                }
                catch (ModelFormatException)
                {
                }
            }
        }
    }

    [Fact]
    public void NonexistentFileIsAnIoError()
    {
        var e = Assert.Throws<ModelFormatException>(() => ModelFormat.Load(Path.Combine(Path.GetTempPath(), "onemodel-none-" + Guid.NewGuid().ToString("N"))));
        Assert.Equal(ModelFormatErrorKind.Io, e.Kind);
    }
}
