using System.Buffers.Binary;
using System.Text;

namespace RecursiveAI.OneModel.Tests.Graph;

/// <summary>Writes small ONNX models as protobuf bytes (field numbers from onnx.proto), so tests can build exactly the graph they need.</summary>
internal sealed class ModelBuilder
{
    private readonly List<byte[]> _nodes = [];
    private readonly List<byte[]> _inits = [];
    private readonly List<byte[]> _inputs = [];
    private readonly List<byte[]> _outputs = [];
    private readonly int _opset;

    public ModelBuilder(int opset = 17)
    {
        _opset = opset;
    }

    public ModelBuilder Input(string name, int elemType, params object[] dims)
    {
        _inputs.Add(ValueInfo(name, elemType, dims));
        return this;
    }

    public ModelBuilder Output(string name, int elemType, params object[] dims)
    {
        _outputs.Add(ValueInfo(name, elemType, dims));
        return this;
    }

    public ModelBuilder Initializer(string name, float[] values, params long[] dims)
    {
        byte[] raw = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(raw.AsSpan(i * 4), values[i]);
        }

        _inits.Add(Tensor(name, 1, dims, raw));
        return this;
    }

    public ModelBuilder Initializer(string name, long[] values, params long[] dims)
    {
        byte[] raw = new byte[values.Length * 8];
        for (int i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteInt64LittleEndian(raw.AsSpan(i * 8), values[i]);
        }

        _inits.Add(Tensor(name, 7, dims, raw));
        return this;
    }

    /// <summary>Adds a node. Attributes are (name, value) pairs where value is long, float, long[] or a tensor message from <see cref="TensorValue"/>.</summary>
    public ModelBuilder Node(string opType, string[] inputs, string[] outputs, params (string Name, object Value)[] attributes)
    {
        var w = new Msg();
        foreach (string i in inputs)
        {
            w.String(1, i);
        }

        foreach (string o in outputs)
        {
            w.String(2, o);
        }

        w.String(3, opType + "_" + _nodes.Count);
        w.String(4, opType);
        foreach ((string name, object value) in attributes)
        {
            var a = new Msg();
            a.String(1, name);
            switch (value)
            {
                case long l:
                    a.Varint(3, (ulong)l);
                    a.Varint(20, 2);
                    break;
                case int i32:
                    a.Varint(3, (ulong)i32);
                    a.Varint(20, 2);
                    break;
                case float f:
                    a.Fixed32(2, BitConverter.SingleToUInt32Bits(f));
                    a.Varint(20, 1);
                    break;
                case long[] ls:
                    foreach (long l in ls)
                    {
                        a.Varint(8, (ulong)l);
                    }

                    a.Varint(20, 7);
                    break;
                case TensorValue t:
                    a.Bytes(5, t.Bytes);
                    a.Varint(20, 4);
                    break;
                default:
                    throw new ArgumentException("unsupported attribute value");
            }

            w.Bytes(5, a.ToArray());
        }

        _nodes.Add(w.ToArray());
        return this;
    }

    public sealed record TensorValue(byte[] Bytes);

    public static TensorValue FloatTensor(string name, float[] values, params long[] dims)
    {
        byte[] raw = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(raw.AsSpan(i * 4), values[i]);
        }

        return new TensorValue(Tensor(name, 1, dims, raw));
    }

    public static TensorValue Int64Tensor(string name, long[] values, params long[] dims)
    {
        byte[] raw = new byte[values.Length * 8];
        for (int i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteInt64LittleEndian(raw.AsSpan(i * 8), values[i]);
        }

        return new TensorValue(Tensor(name, 7, dims, raw));
    }

    public byte[] Build()
    {
        var g = new Msg();
        foreach (byte[] n in _nodes)
        {
            g.Bytes(1, n);
        }

        g.String(2, "test");
        foreach (byte[] t in _inits)
        {
            g.Bytes(5, t);
        }

        foreach (byte[] i in _inputs)
        {
            g.Bytes(11, i);
        }

        foreach (byte[] o in _outputs)
        {
            g.Bytes(12, o);
        }

        var opset = new Msg();
        opset.String(1, "");
        opset.Varint(2, (ulong)_opset);
        var m = new Msg();
        m.Varint(1, 8);
        m.Bytes(8, opset.ToArray());
        m.Bytes(7, g.ToArray());
        return m.ToArray();
    }

    private static byte[] Tensor(string name, int dataType, long[] dims, byte[] raw)
    {
        var t = new Msg();
        foreach (long d in dims)
        {
            t.Varint(1, (ulong)d);
        }

        t.Varint(2, (ulong)dataType);
        t.String(8, name);
        t.Bytes(9, raw);
        return t.ToArray();
    }

    private static byte[] ValueInfo(string name, int elemType, object[] dims)
    {
        var shape = new Msg();
        foreach (object d in dims)
        {
            var dim = new Msg();
            if (d is string s)
            {
                dim.String(2, s);
            }
            else
            {
                dim.Varint(1, (ulong)Convert.ToInt64(d));
            }

            shape.Bytes(1, dim.ToArray());
        }

        var tensor = new Msg();
        tensor.Varint(1, (ulong)elemType);
        tensor.Bytes(2, shape.ToArray());
        var type = new Msg();
        type.Bytes(1, tensor.ToArray());
        var vi = new Msg();
        vi.String(1, name);
        vi.Bytes(2, type.ToArray());
        return vi.ToArray();
    }

    private sealed class Msg
    {
        private readonly List<byte> _s = [];

        public void Varint(int field, ulong value)
        {
            Raw((ulong)(field << 3));
            Raw(value);
        }

        public void Fixed32(int field, uint value)
        {
            Raw((ulong)((field << 3) | 5));
            Span<byte> b = stackalloc byte[4];
            BinaryPrimitives.WriteUInt32LittleEndian(b, value);
            _s.AddRange(b.ToArray());
        }

        public void Bytes(int field, byte[] data)
        {
            Raw((ulong)((field << 3) | 2));
            Raw((ulong)data.Length);
            _s.AddRange(data);
        }

        public void String(int field, string text) => Bytes(field, Encoding.UTF8.GetBytes(text));

        public byte[] ToArray() => [.. _s];

        private void Raw(ulong v)
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
