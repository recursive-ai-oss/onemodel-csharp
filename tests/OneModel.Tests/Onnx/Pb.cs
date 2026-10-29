using System.Buffers.Binary;
using System.Text;

namespace RecursiveAI.OneModel.Tests.Onnx;

/// <summary>Writes protobuf wire format by hand, so tests can build valid and deliberately broken ONNX files without any protobuf library.</summary>
internal sealed class Pb
{
    private readonly List<byte> _bytes = [];

    public byte[] ToArray() => [.. _bytes];

    public Pb Varint(ulong v)
    {
        while (v >= 0x80)
        {
            _bytes.Add((byte)(v | 0x80));
            v >>= 7;
        }

        _bytes.Add((byte)v);
        return this;
    }

    public Pb Tag(int field, int wire) => Varint(((ulong)field << 3) | (uint)wire);

    public Pb Int(int field, long v) => Tag(field, 0).Varint(unchecked((ulong)v));

    public Pb Bytes(int field, ReadOnlySpan<byte> b)
    {
        Tag(field, 2).Varint((ulong)b.Length);
        foreach (byte x in b)
        {
            _bytes.Add(x);
        }

        return this;
    }

    public Pb Str(int field, string s) => Bytes(field, Encoding.UTF8.GetBytes(s));

    public Pb Msg(int field, Pb m) => Bytes(field, m.ToArray());

    public Pb Float(int field, float f)
    {
        Tag(field, 5);
        Span<byte> s = stackalloc byte[4];
        BinaryPrimitives.WriteSingleLittleEndian(s, f);
        foreach (byte x in s)
        {
            _bytes.Add(x);
        }

        return this;
    }

    public Pb PackedInts(int field, params long[] values)
    {
        var inner = new Pb();
        foreach (long v in values)
        {
            inner.Varint(unchecked((ulong)v));
        }

        return Bytes(field, inner.ToArray());
    }

    public Pb PackedFloats(int field, params float[] values)
    {
        var b = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(b.AsSpan(i * 4), values[i]);
        }

        return Bytes(field, b);
    }

    public Pb Raw(params byte[] b)
    {
        _bytes.AddRange(b);
        return this;
    }
}

/// <summary>Builders for the ONNX messages the tests need (field numbers from onnx.proto).</summary>
internal static class Build
{
    public const int Float = 1, UInt8 = 2, Int8 = 3, Int32 = 6, Int64 = 7, String = 8, Bool = 9, Float16 = 10, Double = 11, UInt32 = 12, UInt64 = 13, Int4 = 22;

    public static Pb RawTensor(string name, int dtype, long[] dims, byte[] raw) =>
        new Pb().PackedInts(1, dims).Int(2, dtype).Str(8, name).Bytes(9, raw);

    public static Pb FloatTensor(string name, long[] dims, params float[] values)
    {
        var raw = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(raw.AsSpan(i * 4), values[i]);
        }

        return RawTensor(name, Float, dims, raw);
    }

    public static Pb ValueInfo(string name, int elemType, params long[] dims)
    {
        var shape = new Pb();
        foreach (long d in dims)
        {
            shape.Msg(1, d >= 0 ? new Pb().Int(1, d) : new Pb().Str(2, "N"));
        }

        var tensorType = new Pb().Int(1, elemType).Msg(2, shape);
        return new Pb().Str(1, name).Msg(2, new Pb().Msg(1, tensorType));
    }

    public static Pb Node(string op, string[] inputs, string[] outputs, string name = "", params Pb[] attrs)
    {
        var n = new Pb();
        foreach (string i in inputs)
        {
            n.Str(1, i);
        }

        foreach (string o in outputs)
        {
            n.Str(2, o);
        }

        n.Str(3, name).Str(4, op);
        foreach (Pb a in attrs)
        {
            n.Msg(5, a);
        }

        return n;
    }

    public static Pb AttrInt(string name, long v) => new Pb().Str(1, name).Int(3, v).Int(20, 2);

    public static Pb AttrFloat(string name, float v) => new Pb().Str(1, name).Float(2, v).Int(20, 1);

    public static Pb AttrGraph(string name, Pb graph) => new Pb().Str(1, name).Msg(6, graph).Int(20, 5);

    public static Pb Graph(string name, Pb[] nodes, Pb[]? inputs = null, Pb[]? outputs = null, Pb[]? initializers = null)
    {
        var g = new Pb();
        foreach (Pb n in nodes)
        {
            g.Msg(1, n);
        }

        g.Str(2, name);
        foreach (Pb t in initializers ?? [])
        {
            g.Msg(5, t);
        }

        foreach (Pb i in inputs ?? [])
        {
            g.Msg(11, i);
        }

        foreach (Pb o in outputs ?? [])
        {
            g.Msg(12, o);
        }

        return g;
    }

    public static byte[] Model(Pb graph, long opset = 13, long ir = 8)
    {
        return new Pb().Int(1, ir).Str(2, "test").Msg(7, graph).Msg(8, new Pb().Str(1, "").Int(2, opset)).ToArray();
    }

    /// <summary>x + w, with w an initializer.</summary>
    public static byte[] AddModel() => Model(Graph("g",
        [Node("Add", ["x", "w"], ["y"], "add")],
        [ValueInfo("x", Float, 2, 2)], [ValueInfo("y", Float, 2, 2)],
        [FloatTensor("w", [2, 2], 1, 2, 3, 4)]));
}
