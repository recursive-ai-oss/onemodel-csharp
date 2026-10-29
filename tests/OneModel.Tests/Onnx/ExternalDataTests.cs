using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;

namespace RecursiveAI.OneModel.Tests.Onnx;

public sealed class ExternalDataTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "onemodel-ext-" + Guid.NewGuid().ToString("N"));
    private readonly string _dir;

    public ExternalDataTests()
    {
        _dir = Path.Combine(_root, "model");
        Directory.CreateDirectory(Path.Combine(_dir, "sub"));
        var data = new byte[8192];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (byte)(i * 7);
        }

        File.WriteAllBytes(Path.Combine(_dir, "w.bin"), data);
        File.WriteAllBytes(Path.Combine(_dir, "sub", "v.bin"), data[..64]);
        File.WriteAllBytes(Path.Combine(_root, "secret.bin"), new byte[64]);
    }

    public void Dispose() => Directory.Delete(_root, recursive: true);

    private static Pb Tensor(string location, string? offset = null, string? length = null, long count = 4)
    {
        var t = new Pb().PackedInts(1, count).Int(2, Build.Float).Str(8, "w").Int(14, 1).Msg(13, new Pb().Str(1, "location").Str(2, location));
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

    private string Write(Pb tensor)
    {
        string path = Path.Combine(_dir, "m.onnx");
        File.WriteAllBytes(path, Build.Model(Build.Graph("g", [], null, null, [tensor])));
        return path;
    }

    private OnnxImportException Fails(Pb tensor, bool map = true) =>
        Assert.Throws<OnnxImportException>(() => OnnxImporter.Load(Write(tensor), new OnnxLoadOptions { MemoryMap = map }));

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void OffsetAndLengthSelectTheRange(bool map)
    {
        using IrModel m = OnnxImporter.Load(Write(Tensor("w.bin", "4096", "16")), new OnnxLoadOptions { MemoryMap = map });
        TensorData t = m.Graph.Initializers[0];
        Assert.True(t.IsExternal);
        byte[] expected = new byte[16];
        for (int i = 0; i < 16; i++)
        {
            expected[i] = (byte)((4096 + i) * 7);
        }

        Assert.Equal(expected, t.RawData.ToArray());
    }

    [Fact]
    public void LengthMayBeOmittedAndSubdirectoriesWork()
    {
        using IrModel m = OnnxImporter.Load(Write(Tensor("sub/v.bin", null, null)));
        Assert.Equal(16, m.Graph.Initializers[0].RawLength);
        using IrModel m2 = OnnxImporter.Load(Write(Tensor("./sub/v.bin", "16")));
        Assert.Equal(16, m2.Graph.Initializers[0].RawLength);
    }

    [Theory]
    [InlineData("../secret.bin")]
    [InlineData("sub/../../secret.bin")]
    [InlineData("sub\\..\\..\\secret.bin")]
    [InlineData("/etc/passwd")]
    [InlineData("\\windows\\win.ini")]
    [InlineData("C:\\windows\\win.ini")]
    [InlineData("..")]
    [InlineData("w.bin\0.txt")]
    [InlineData("")]
    public void PathsOutsideTheDirectoryAreRefused(string location)
    {
        OnnxImportException e = Fails(Tensor(location));
        Assert.Equal(OnnxErrorKind.ExternalData, e.Kind);
    }

    [Fact]
    public void AbsolutePathToAnExistingFileIsRefused()
    {
        OnnxImportException e = Fails(Tensor(Path.Combine(_root, "secret.bin")));
        Assert.Equal(OnnxErrorKind.ExternalData, e.Kind);
        Assert.Contains("absolute", e.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void SymbolicLinksOutOfTheDirectoryAreRefused()
    {
        try
        {
            File.CreateSymbolicLink(Path.Combine(_dir, "link.bin"), Path.Combine(_root, "secret.bin"));
            Directory.CreateSymbolicLink(Path.Combine(_dir, "linkdir"), _root);
        }
        catch (Exception e) when (e is UnauthorizedAccessException or IOException or PlatformNotSupportedException)
        {
            return; // symbolic links are not available here
        }

        Assert.Equal(OnnxErrorKind.ExternalData, Fails(Tensor("link.bin")).Kind);
        Assert.Equal(OnnxErrorKind.ExternalData, Fails(Tensor("linkdir/secret.bin")).Kind);
        // A link that stays inside the directory is fine.
        File.CreateSymbolicLink(Path.Combine(_dir, "inside.bin"), Path.Combine(_dir, "w.bin"));
        using IrModel m = OnnxImporter.Load(Write(Tensor("inside.bin")));
        Assert.Equal(16, m.Graph.Initializers[0].RawLength);
    }

    [Theory]
    [InlineData("w.bin", "9000", null)]          // offset past the end
    [InlineData("w.bin", "8190", null)]          // offset + length past the end
    [InlineData("w.bin", "-1", null)]            // negative offset
    [InlineData("w.bin", "abc", null)]           // not a number
    [InlineData("w.bin", "99999999999999999999", null)]
    [InlineData("w.bin", "0", "17")]             // length disagrees with the shape
    [InlineData("w.bin", "0", "-16")]
    [InlineData("missing.bin", null, null)]
    [InlineData("sub", null, null)]              // a directory
    public void BadRangesAndMissingFilesAreTypedErrors(string location, string? offset, string? length)
    {
        Assert.Equal(OnnxErrorKind.ExternalData, Fails(Tensor(location, offset, length)).Kind);
        Assert.Equal(OnnxErrorKind.ExternalData, Fails(Tensor(location, offset, length), map: false).Kind);
    }

    [Fact]
    public void HugeTensorFromExternalDataIsRefusedBeforeAnyAllocation()
    {
        Assert.Equal(OnnxErrorKind.Unsupported, Fails(Tensor("w.bin", null, null, count: 1L << 40)).Kind);
    }

    [Fact]
    public void StreamsNeedAnExplicitDirectory()
    {
        byte[] bytes = File.ReadAllBytes(Write(Tensor("w.bin", "0", "16")));
        Assert.Equal(OnnxErrorKind.ExternalData, Assert.Throws<OnnxImportException>(() => OnnxImporter.Load(new MemoryStream(bytes))).Kind);
        using IrModel m = OnnxImporter.Load(new MemoryStream(bytes), new OnnxLoadOptions { ExternalDataDirectory = _dir });
        Assert.Equal(16, m.Graph.Initializers[0].RawLength);
    }

    [Fact]
    public void ModelsReadFromADifferentDirectoryDoNotEscapeThroughTheOptionEither()
    {
        byte[] bytes = Build.Model(Build.Graph("g", [], null, null, [Tensor("../secret.bin")]));
        var e = Assert.Throws<OnnxImportException>(() => OnnxImporter.Load(bytes, new OnnxLoadOptions { ExternalDataDirectory = _dir }));
        Assert.Equal(OnnxErrorKind.ExternalData, e.Kind);
    }

    private static string RepoRoot([System.Runtime.CompilerServices.CallerFilePath] string path = "")
    {
        string? dir = Path.GetDirectoryName(path);
        while (dir is not null && !File.Exists(Path.Combine(dir, "global.json")))
        {
            dir = Path.GetDirectoryName(dir);
        }

        return dir ?? throw new InvalidOperationException("global.json not found above " + path);
    }

    [Fact]
    public void CommittedSampleLoadsAndMatchesItsGenerator()
    {
        // testdata/onnx/external holds a small model whose two initializers live in weights.bin. It doubles as a fuzzing seed.
        string dir = Path.Combine(RepoRoot(), "testdata", "onnx", "external");
        var weights = new byte[8192];
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = (byte)(i * 7);
        }

        Pb Ext(string name, long offset) => new Pb().PackedInts(1, 4).Int(2, Build.Float).Str(8, name).Int(14, 1)
            .Msg(13, new Pb().Str(1, "location").Str(2, "weights.bin"))
            .Msg(13, new Pb().Str(1, "offset").Str(2, offset.ToString(System.Globalization.CultureInfo.InvariantCulture)))
            .Msg(13, new Pb().Str(1, "length").Str(2, "16"));
        byte[] model = Build.Model(Build.Graph("g", [Build.Node("Add", ["x", "a"], ["t"]), Build.Node("Add", ["t", "b"], ["y"])],
            [Build.ValueInfo("x", Build.Float, 4)], [Build.ValueInfo("y", Build.Float, 4)], [Ext("a", 0), Ext("b", 4096)]));
        if (Environment.GetEnvironmentVariable("ONEMODEL_WRITE_TESTDATA") == "1")
        {
            Directory.CreateDirectory(dir);
            File.WriteAllBytes(Path.Combine(dir, "model.onnx"), model);
            File.WriteAllBytes(Path.Combine(dir, "weights.bin"), weights);
        }

        Assert.Equal(model, File.ReadAllBytes(Path.Combine(dir, "model.onnx")));
        Assert.Equal(weights, File.ReadAllBytes(Path.Combine(dir, "weights.bin")));
        using IrModel m = OnnxImporter.Load(Path.Combine(dir, "model.onnx"));
        Assert.Equal(weights[4096..4112], m.Graph.FindValue("b")!.Initializer!.RawData.ToArray());
    }
}
