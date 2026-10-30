using System;
using System.Collections.Generic;
using System.Text;

namespace RecursiveAI.OneModel.Onnx;

/// <summary>An attribute of an <see cref="OnnxNodeSpec"/>: onnx.proto <c>AttributeProto</c> with the kinds the tracer needs.</summary>
internal readonly record struct OnnxAttrSpec(string Name, int Type, long Int, float Float, string? Text, long[]? Ints)
{
    // AttributeProto.AttributeType: FLOAT = 1, INT = 2, STRING = 3, INTS = 7.
    public static OnnxAttrSpec OfInt(string name, long v) => new(name, 2, v, 0, null, null);

    public static OnnxAttrSpec OfFloat(string name, float v) => new(name, 1, 0, v, null, null);

    public static OnnxAttrSpec OfText(string name, string v) => new(name, 3, 0, 0, v, null);

    public static OnnxAttrSpec OfInts(string name, params long[] v) => new(name, 7, 0, 0, null, v);
}

internal sealed class OnnxNodeSpec
{
    public required string OpType { get; init; }

    public required string Name { get; init; }

    public required string[] Inputs { get; init; }

    public required string[] Outputs { get; init; }

    public List<OnnxAttrSpec> Attributes { get; } = [];
}

internal sealed class OnnxInitializerSpec
{
    public required string Name { get; init; }

    public required int DataType { get; init; }

    public required long[] Dims { get; init; }

    public required byte[] Raw { get; init; }
}

/// <summary>One dimension of a declared input or output: a fixed size, a named symbol, or unknown (both null).</summary>
internal readonly record struct OnnxDimSpec(long? Value, string? Param);

internal sealed class OnnxValueSpec
{
    public required string Name { get; init; }

    public required int DataType { get; init; }

    public required OnnxDimSpec[] Dims { get; init; }
}

/// <summary>
/// Writes a ModelProto from nodes, initializers and declared inputs and outputs. The field numbers and wire types come from
/// onnx.proto; the encoding (varints, length-delimited fields, packed repeated int64, little-endian fixed32) from the protobuf
/// encoding spec. Everything is written in field-number order, so the same graph gives the same bytes.
/// </summary>
internal static class OnnxWriter
{
    public const long IrVersion = 9;

    public static byte[] Write(int opset, string graphName, IReadOnlyList<OnnxNodeSpec> nodes, IReadOnlyList<OnnxInitializerSpec> initializers,
        IReadOnlyList<OnnxValueSpec> inputs, IReadOnlyList<OnnxValueSpec> outputs)
    {
        var graph = new Msg();
        foreach (OnnxNodeSpec n in nodes)
        {
            graph.Message(1, WriteNode(n));
        }

        graph.Text(2, graphName);
        foreach (OnnxInitializerSpec t in initializers)
        {
            graph.Message(5, WriteTensor(t));
        }

        foreach (OnnxValueSpec v in inputs)
        {
            graph.Message(11, WriteValueInfo(v));
        }

        foreach (OnnxValueSpec v in outputs)
        {
            graph.Message(12, WriteValueInfo(v));
        }

        var model = new Msg();
        model.Varint(1, IrVersion);
        model.Text(2, "onemodel-trace");
        model.Text(3, "1");
        model.Message(7, graph);
        var opsetMsg = new Msg();
        opsetMsg.Text(1, "");
        opsetMsg.Varint(2, opset);
        model.Message(8, opsetMsg);
        return model.ToArray();
    }

    private static Msg WriteNode(OnnxNodeSpec n)
    {
        var m = new Msg();
        foreach (string i in n.Inputs)
        {
            m.Text(1, i);
        }

        foreach (string o in n.Outputs)
        {
            m.Text(2, o);
        }

        m.Text(3, n.Name);
        m.Text(4, n.OpType);
        foreach (OnnxAttrSpec a in n.Attributes)
        {
            var am = new Msg();
            am.Text(1, a.Name);
            switch (a.Type)
            {
                case 1: am.Fixed32(2, BitConverter.SingleToInt32Bits(a.Float)); break;
                case 2: am.Varint(3, a.Int); break;
                case 3: am.Text(4, a.Text!); break;
                default:
                    foreach (long v in a.Ints!)
                    {
                        am.Varint(8, v);
                    }

                    break;
            }

            am.Varint(20, a.Type);
            m.Message(5, am);
        }

        return m;
    }

    private static Msg WriteTensor(OnnxInitializerSpec t)
    {
        var m = new Msg();
        foreach (long d in t.Dims)
        {
            m.Varint(1, d);
        }

        m.Varint(2, t.DataType);
        m.Text(8, t.Name);
        m.Bytes(9, t.Raw);
        return m;
    }

    private static Msg WriteValueInfo(OnnxValueSpec v)
    {
        var shape = new Msg();
        foreach (OnnxDimSpec d in v.Dims)
        {
            var dm = new Msg();
            if (d.Value is { } n)
            {
                dm.Varint(1, n);
            }
            else if (d.Param is { } p)
            {
                dm.Text(2, p);
            }

            shape.Message(1, dm);
        }

        var tensorType = new Msg();
        tensorType.Varint(1, v.DataType);
        tensorType.Message(2, shape);
        var type = new Msg();
        type.Message(1, tensorType);
        var info = new Msg();
        info.Text(1, v.Name);
        info.Message(2, type);
        return info;
    }

    // A message under construction: fields are appended in the order the caller writes them (callers write in field order).
    private sealed class Msg
    {
        private readonly List<byte> _s = [];

        public byte[] ToArray() => [.. _s];

        public void Varint(int field, long value)
        {
            Tag(field, 0);
            WriteVarint((ulong)value);
        }

        public void Fixed32(int field, int value)
        {
            Tag(field, 5);
            Span<byte> b = stackalloc byte[4];
            System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(b, value);
            _s.AddRange(b.ToArray());
        }

        public void Text(int field, string value) => Bytes(field, Encoding.UTF8.GetBytes(value));

        public void Bytes(int field, byte[] value)
        {
            Tag(field, 2);
            WriteVarint((ulong)value.Length);
            _s.AddRange(value);
        }

        public void Message(int field, Msg inner) => Bytes(field, inner.ToArray());

        private void Tag(int field, int wire) => WriteVarint((ulong)((field << 3) | wire));

        private void WriteVarint(ulong v)
        {
            while (v >= 0x80)
            {
                _s.Add((byte)(v | 0x80));
                v >>= 7;
            }

            _s.Add((byte)v);
        }
    }
}
