using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Text;
using RecursiveAI.OneModel.Training;

namespace RecursiveAI.OneModel.Tests.Training;

/// <summary>
/// Malformed checkpoints. Each case is assembled here by hand from docs/training.md section 6 (independently of the library's writer),
/// written to testdata/checkpoint/hostile/&lt;kind&gt;-&lt;name&gt;.ckpt (set ONEMODEL_WRITE_TESTDATA=1 to regenerate) and must be rejected
/// with a <see cref="CheckpointException"/> of that kind, from memory and from the file, leaving the model untouched.
/// </summary>
public class CheckpointHostileTests
{
    // A Linear(2, 2) with bias: state weight [2,2] then bias [2].
    private const string Base =
        """{"format":"onemodel-checkpoint","version":1,"model":[{"name":"weight","dtype":"f32","shape":[2,2],"offset":0,"length":16},{"name":"bias","dtype":"f32","shape":[2],"offset":64,"length":8}],"extra":[]}""";

    private static string HostileDir([CallerFilePath] string file = "") =>
        Path.GetFullPath(Path.Combine(Path.GetDirectoryName(file)!, "..", "..", "..", "testdata", "checkpoint", "hostile"));

    private static byte[] Data()
    {
        var d = new byte[72];
        for (int i = 0; i < 4; i++)
        {
            BitConverter.GetBytes(i + 1f).CopyTo(d, 4 * i);
        }

        BitConverter.GetBytes(5f).CopyTo(d, 64);
        BitConverter.GetBytes(6f).CopyTo(d, 68);
        return d;
    }

    private static byte[] File(string manifest)
    {
        byte[] m = Encoding.UTF8.GetBytes(manifest);
        byte[] data = Data();
        long dataOffset = (64 + m.Length + 63) / 64 * 64;
        var file = new byte[dataOffset + data.Length];
        "ONEMCKPT"u8.CopyTo(file);
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

    private static byte[] Edit(string from, string to)
    {
        Assert.Contains(from, Base, StringComparison.Ordinal);
        return File(Base.Replace(from, to, StringComparison.Ordinal));
    }

    private static byte[] Patch(Action<byte[]> edit)
    {
        byte[] f = File(Base);
        edit(f);
        return f;
    }

    private static void U64(byte[] f, int at, ulong v) => BinaryPrimitives.WriteUInt64LittleEndian(f.AsSpan(at), v);

    private static IEnumerable<(CheckpointErrorKind Kind, string Name, byte[] Bytes)> Cases()
    {
        byte[] good = File(Base);
        yield return (CheckpointErrorKind.BadHeader, "empty", []);
        yield return (CheckpointErrorKind.BadHeader, "short-header", good[..40]);
        yield return (CheckpointErrorKind.BadHeader, "magic", Patch(f => f[0] = (byte)'X'));
        yield return (CheckpointErrorKind.BadHeader, "version-2", Patch(f => f[8] = 2));
        yield return (CheckpointErrorKind.BadHeader, "flags", Patch(f => f[12] = 1));
        yield return (CheckpointErrorKind.BadHeader, "reserved", Patch(f => f[56] = 1));
        yield return (CheckpointErrorKind.BadHeader, "truncated", good[..^1]);
        yield return (CheckpointErrorKind.BadHeader, "extended", [.. good, 0]);
        yield return (CheckpointErrorKind.BadHeader, "manifest-offset", Patch(f => U64(f, 16, 65)));
        yield return (CheckpointErrorKind.BadHeader, "manifest-length-max", Patch(f => U64(f, 24, ulong.MaxValue)));
        yield return (CheckpointErrorKind.BadHeader, "manifest-length-past-end", Patch(f => U64(f, 24, (ulong)good.Length)));
        yield return (CheckpointErrorKind.BadHeader, "data-offset-unaligned", Patch(f => U64(f, 32, 65)));
        yield return (CheckpointErrorKind.BadHeader, "data-offset-inside-manifest", Patch(f => U64(f, 32, 64)));
        yield return (CheckpointErrorKind.BadHeader, "data-offset-past-end", Patch(f => U64(f, 32, (ulong)(good.Length + 64))));
        yield return (CheckpointErrorKind.BadHeader, "data-length", Patch(f => U64(f, 40, 1)));
        yield return (CheckpointErrorKind.BadHeader, "file-length-max", Patch(f => U64(f, 48, ulong.MaxValue)));
        yield return (CheckpointErrorKind.BadManifest, "not-json", File("not json at all"));
        yield return (CheckpointErrorKind.BadManifest, "json-array", File("[]"));
        yield return (CheckpointErrorKind.BadManifest, "json-too-deep", File(new string('[', 100) + new string(']', 100)));
        yield return (CheckpointErrorKind.BadManifest, "format-name", Edit("onemodel-checkpoint", "onemodel"));
        yield return (CheckpointErrorKind.BadManifest, "manifest-version", Edit("\"version\":1", "\"version\":2"));
        yield return (CheckpointErrorKind.BadManifest, "model-missing", File("""{"format":"onemodel-checkpoint","version":1}"""));
        yield return (CheckpointErrorKind.BadManifest, "model-not-array", Edit("\"model\":[", "\"model\":{\"a\":["));
        yield return (CheckpointErrorKind.BadManifest, "dtype-unsupported", Edit("\"f32\",\"shape\":[2,2]", "\"f16\",\"shape\":[2,2]"));
        yield return (CheckpointErrorKind.BadManifest, "dim-negative", Edit("\"shape\":[2,2]", "\"shape\":[2,-2]"));
        yield return (CheckpointErrorKind.BadManifest, "dim-overflow", Edit("\"shape\":[2,2]", "\"shape\":[2,99999999999]"));
        yield return (CheckpointErrorKind.BadManifest, "name-not-string", Edit("\"name\":\"weight\"", "\"name\":7"));
        yield return (CheckpointErrorKind.BadManifest, "offset-fraction", Edit("\"offset\":0", "\"offset\":0.5"));
        yield return (CheckpointErrorKind.BadManifest, "offset-missing", Edit("\"offset\":0,", ""));
        yield return (CheckpointErrorKind.BadManifest, "invalid-utf8", Patch(f => f[Array.IndexOf(f, (byte)'w', 64)] = 0xFF));
        yield return (CheckpointErrorKind.BadTensor, "offset-past-data", Edit("\"offset\":0", "\"offset\":4096"));
        yield return (CheckpointErrorKind.BadTensor, "offset-negative", Edit("\"offset\":0", "\"offset\":-64"));
        yield return (CheckpointErrorKind.BadTensor, "offset-max", Edit("\"offset\":0", "\"offset\":9223372036854775807"));
        yield return (CheckpointErrorKind.BadTensor, "length-short", Edit("\"length\":16", "\"length\":12"));
        yield return (CheckpointErrorKind.BadTensor, "length-huge", Edit("\"length\":16", "\"length\":9223372036854775807"));
        yield return (CheckpointErrorKind.BadTensor, "runs-off-end", Edit("\"offset\":64,\"length\":8", "\"offset\":68,\"length\":8"));
        yield return (CheckpointErrorKind.Mismatch, "name", Edit("\"weight\"", "\"kernel\""));
        yield return (CheckpointErrorKind.Mismatch, "shape", Edit("\"shape\":[2,2]", "\"shape\":[4,1]"));
        yield return (CheckpointErrorKind.Mismatch, "shape-huge", Edit("\"shape\":[2,2]", "\"shape\":[2147483647,2147483647]"));
        yield return (CheckpointErrorKind.Mismatch, "dtype", Edit("\"f32\",\"shape\":[2,2]", "\"i32\",\"shape\":[2,2]"));
        yield return (CheckpointErrorKind.Mismatch, "count", File("""{"format":"onemodel-checkpoint","version":1,"model":[],"extra":[]}"""));
        yield return (CheckpointErrorKind.Mismatch, "order", Edit("\"weight\"", "\"zz\"").Length > 0
            ? File(Base.Replace("\"weight\"", "\"tmp\"").Replace("\"bias\"", "\"weight\"").Replace("\"tmp\"", "\"bias\""))
            : []);
    }

    public static IEnumerable<object[]> Names() => Cases().Select(c => new object[] { $"{c.Kind}-{c.Name}" });

    private static (CheckpointErrorKind Kind, byte[] Bytes) Find(string key) =>
        Cases().Where(c => $"{c.Kind}-{c.Name}" == key).Select(c => (c.Kind, c.Bytes)).Single();

    private static Linear Fresh() => new(2, 2, new Generator(1));

    [Fact]
    public void BaseCaseLoads()
    {
        using Linear m = Fresh();
        Assert.False(Checkpoint.Load(new MemoryStream(File(Base)), m).HadOptimizer);
        Assert.Equal([1f, 2f, 3f, 4f], m.Weight.ToArray<float>());
        Assert.Equal([5f, 6f], m.Bias.ToArray<float>());
    }

    [Fact]
    public void WriteTestData()
    {
        if (Environment.GetEnvironmentVariable("ONEMODEL_WRITE_TESTDATA") != "1")
        {
            return;
        }

        Directory.CreateDirectory(HostileDir());
        foreach ((CheckpointErrorKind kind, string name, byte[] bytes) in Cases())
        {
            System.IO.File.WriteAllBytes(Path.Combine(HostileDir(), $"{kind}-{name}.ckpt"), bytes);
        }
    }

    [Theory]
    [MemberData(nameof(Names))]
    public void HostileFileIsRejectedFromMemoryAndFile(string key)
    {
        (CheckpointErrorKind kind, byte[] bytes) = Find(key);
        using Linear m = Fresh();
        string before = TrainCase.StateHash(m, null);
        Assert.Equal(kind, Assert.Throws<CheckpointException>(() => Checkpoint.Load(new MemoryStream(bytes), m)).Kind);

        string path = Path.Combine(HostileDir(), key + ".ckpt");
        Assert.True(System.IO.File.Exists(path), $"{path} is missing; run with ONEMODEL_WRITE_TESTDATA=1");
        Assert.Equal(bytes, System.IO.File.ReadAllBytes(path));
        Assert.Equal(kind, Assert.Throws<CheckpointException>(() => Checkpoint.Load(path, m)).Kind);
        Assert.Equal(before, TrainCase.StateHash(m, null));
    }
}
