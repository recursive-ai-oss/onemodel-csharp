using RecursiveAI.OneModel.Onnx;

namespace RecursiveAI.OneModel.Tests.Onnx;

public class ProtoReaderTests
{
    private static ParseContext Ctx() => new("test", 128, null, CancellationToken.None);

    private static ulong Varint(params byte[] bytes)
    {
        var r = new ProtoReader(bytes, 0, Ctx(), 0);
        ulong v = r.ReadVarint();
        Assert.True(r.AtEnd);
        return v;
    }

    private static void Rejects(Action a) => Assert.Equal(OnnxErrorKind.Malformed, Assert.Throws<OnnxImportException>(a).Kind);

    [Fact]
    public void VarintsDecodeToTheirValues()
    {
        Assert.Equal(0UL, Varint(0));
        Assert.Equal(127UL, Varint(0x7F));
        Assert.Equal(128UL, Varint(0x80, 0x01));
        Assert.Equal(300UL, Varint(0xAC, 0x02));
        Assert.Equal(ulong.MaxValue, Varint(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01));
        Assert.Equal(1UL, Varint(0x81, 0x80, 0x00)); // overlong but legal
    }

    [Fact]
    public void BadVarintsAreRejected()
    {
        Rejects(() => Varint());
        Rejects(() => Varint(0x80));
        Rejects(() => Varint(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x02));
        Rejects(() => Varint(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01));
    }

    [Fact]
    public void TagsAndLengthsAreChecked()
    {
        Rejects(() => { var r = new ProtoReader([0x00], 0, Ctx(), 0); r.TryReadTag(out _, out _); });
        Rejects(() => { var r = new ProtoReader([0x0B], 0, Ctx(), 0); r.TryReadTag(out _, out _); });
        Rejects(() => { var r = new ProtoReader([0x0C], 0, Ctx(), 0); r.TryReadTag(out _, out _); });
        Rejects(() => { var r = new ProtoReader([0x0F], 0, Ctx(), 0); r.TryReadTag(out _, out _); });
        Rejects(() => { var r = new ProtoReader([0x02, 0x05, 0x01], 0, Ctx(), 0); r.TryReadTag(out _, out _); _ = r.ReadBytes(); });

        var ok = new ProtoReader([0x12, 0x02, 0xAA, 0xBB, 0x18, 0x96, 0x01], 0, Ctx(), 0);
        Assert.True(ok.TryReadTag(out int f, out WireType w));
        Assert.Equal((2, WireType.Length), (f, w));
        Assert.Equal([0xAA, 0xBB], ok.ReadBytes().ToArray());
        Assert.True(ok.TryReadTag(out f, out w));
        Assert.Equal((3, WireType.Varint), (f, w));
        Assert.Equal(150L, ok.ReadInt64());
        Assert.False(ok.TryReadTag(out _, out _));
    }

    [Fact]
    public void NestingDepthIsLimited()
    {
        ParseContext ctx = new("test", 8, null, CancellationToken.None);
        Assert.Equal(8, ctx.MaxNesting);
        var r = new ProtoReader([0x0A, 0x00], 0, ctx, 0);
        Assert.True(r.TryReadTag(out _, out _));
        Rejects(() => _ = new ProtoReader([], 0, ctx, 9));
    }

    [Fact]
    public void FixedWidthValuesAreLittleEndian()
    {
        var r = new ProtoReader([0x00, 0x00, 0x80, 0x3F, 1, 0, 0, 0, 0, 0, 0, 0x80], 0, Ctx(), 0);
        Assert.Equal(1f, BitConverter.UInt32BitsToSingle(r.ReadFixed32()));
        Assert.Equal(0x8000000000000001UL, r.ReadFixed64());
        Rejects(() => { var s = new ProtoReader([1, 2, 3], 0, Ctx(), 0); s.ReadFixed32(); });
    }
}
