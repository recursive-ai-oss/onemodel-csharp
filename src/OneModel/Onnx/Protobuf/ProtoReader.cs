using System;
using System.Buffers.Binary;
using System.Globalization;
using System.Text;

namespace RecursiveAI.OneModel.Onnx;

/// <summary>Protobuf wire types (protobuf encoding spec). Groups (3, 4) are rejected, so they have no name here.</summary>
internal enum WireType
{
    Varint = 0,
    Fixed64 = 1,
    Length = 2,
    Fixed32 = 5,
}

/// <summary>
/// State shared by every reader of one import: the input's name, the field path for error messages, the nesting limit, the
/// cancellation token and the external-data resolver.
/// </summary>
internal sealed class ParseContext
{
    private readonly string[] _names = new string[260];
    private readonly int[] _indices = new int[260];
    private int _depth;

    public ParseContext(string? inputName, int maxNesting, ExternalDataResolver? external, System.Threading.CancellationToken cancel)
    {
        InputName = inputName;
        MaxNesting = Math.Clamp(maxNesting, 8, 250);
        Cancel = cancel;
        External = external;
    }

    public string? InputName { get; }

    /// <summary>The bytes being parsed. Tensor data points into it by offset.</summary>
    public Graph.ByteSource Main { get; set; } = null!;

    public int MaxNesting { get; }

    /// <summary>The name of the outermost message, for error paths.</summary>
    public string Root { get; init; } = "ModelProto";
    public System.Threading.CancellationToken Cancel { get; }
    public ExternalDataResolver? External { get; }

    /// <summary>Enters a named field (index -1 when it is not repeated).</summary>
    public void Push(string name, int index = -1)
    {
        if (_depth < _names.Length)
        {
            _names[_depth] = name;
            _indices[_depth] = index;
        }

        _depth++;
    }

    public void Pop() => _depth--;

    public string Path()
    {
        var sb = new StringBuilder(Root);
        for (int i = 0; i < Math.Min(_depth, _names.Length); i++)
        {
            sb.Append('.').Append(_names[i]);
            if (_indices[i] >= 0)
            {
                sb.Append('[').Append(_indices[i].ToString(CultureInfo.InvariantCulture)).Append(']');
            }
        }

        return sb.ToString();
    }

    public OnnxImportException Fail(OnnxErrorKind kind, string message, long offset = -1) =>
        new(kind, message, InputName, offset, Path());
}

/// <summary>
/// Reads protobuf wire format (https://protobuf.dev/programming-guides/encoding/) from a span. It checks every varint's width and
/// overflow, every length against the bytes that remain, and the nesting depth, and it throws <see cref="OnnxImportException"/>
/// for anything else; it never reads outside its span. Nothing is copied: strings and byte fields come back as slices.
/// </summary>
internal ref struct ProtoReader
{
    private static readonly UTF8Encoding StrictUtf8 = new(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);

    private readonly ReadOnlySpan<byte> _data;
    private readonly long _baseOffset;
    private readonly int _depth;
    private int _pos;

    public ProtoReader(ReadOnlySpan<byte> data, long baseOffset, ParseContext context, int depth)
    {
        if (depth > context.MaxNesting)
        {
            throw context.Fail(OnnxErrorKind.Malformed, $"Messages are nested more than {context.MaxNesting} deep.", baseOffset);
        }

        _data = data;
        _baseOffset = baseOffset;
        Context = context;
        _depth = depth;
        _pos = 0;
    }

    public ParseContext Context { get; }

    public readonly bool AtEnd => _pos >= _data.Length;

    /// <summary>The offset of the next unread byte from the start of the input.</summary>
    public readonly long Offset => _baseOffset + _pos;

    public readonly int Remaining => _data.Length - _pos;

    public readonly OnnxImportException Fail(string message) => Context.Fail(OnnxErrorKind.Malformed, message, Offset);

    /// <summary>Reads the next field key. Returns false at the end of the data.</summary>
    public bool TryReadTag(out int field, out WireType wireType)
    {
        if (_pos >= _data.Length)
        {
            field = 0;
            wireType = default;
            return false;
        }

        long start = Offset;
        ulong tag = ReadVarint();
        if (tag > uint.MaxValue)
        {
            throw Context.Fail(OnnxErrorKind.Malformed, "Field key does not fit in 32 bits.", start);
        }

        field = (int)(tag >> 3);
        int wt = (int)(tag & 7);
        if (field == 0)
        {
            throw Context.Fail(OnnxErrorKind.Malformed, "Field number 0 is not allowed.", start);
        }

        if (wt is 3 or 4)
        {
            throw Context.Fail(OnnxErrorKind.Malformed, "Group wire types are not supported.", start);
        }

        if (wt is not (0 or 1 or 2 or 5))
        {
            throw Context.Fail(OnnxErrorKind.Malformed, $"Invalid wire type {wt}.", start);
        }

        wireType = (WireType)wt;
        return true;
    }

    /// <summary>Reads a varint of up to 10 bytes; the tenth byte may carry only the top bit of a 64-bit value.</summary>
    public ulong ReadVarint()
    {
        long start = Offset;
        ulong result = 0;
        for (int i = 0; i < 10; i++)
        {
            if (_pos >= _data.Length)
            {
                throw Context.Fail(OnnxErrorKind.Malformed, "Varint is cut off by the end of the data.", start);
            }

            byte b = _data[_pos++];
            if (i == 9 && b > 1)
            {
                throw Context.Fail(OnnxErrorKind.Malformed, "Varint is longer than 64 bits.", start);
            }

            result |= (ulong)(b & 0x7F) << (7 * i);
            if ((b & 0x80) == 0)
            {
                return result;
            }
        }

        throw Context.Fail(OnnxErrorKind.Malformed, "Varint is longer than 10 bytes.", start);
    }

    /// <summary>Reads an int32/enum varint; like protobuf, keeps the low 32 bits of a wider value.</summary>
    public int ReadInt32() => unchecked((int)ReadVarint());

    public long ReadInt64() => unchecked((long)ReadVarint());

    public uint ReadFixed32()
    {
        Need(4, "fixed 32-bit value");
        uint v = BinaryPrimitives.ReadUInt32LittleEndian(_data.Slice(_pos, 4));
        _pos += 4;
        return v;
    }

    public ulong ReadFixed64()
    {
        Need(8, "fixed 64-bit value");
        ulong v = BinaryPrimitives.ReadUInt64LittleEndian(_data.Slice(_pos, 8));
        _pos += 8;
        return v;
    }

    /// <summary>Reads a length prefix and returns the payload, checking the length against the bytes that remain.</summary>
    public ReadOnlySpan<byte> ReadBytes(out long payloadOffset)
    {
        long start = Offset;
        ulong len = ReadVarint();
        if (len > (ulong)(_data.Length - _pos))
        {
            throw Context.Fail(OnnxErrorKind.Malformed,
                $"Length {len} runs past the end of the data ({_data.Length - _pos} bytes remain).", start);
        }

        payloadOffset = Offset;
        ReadOnlySpan<byte> payload = _data.Slice(_pos, (int)len);
        _pos += (int)len;
        return payload;
    }

    public ReadOnlySpan<byte> ReadBytes() => ReadBytes(out _);

    /// <summary>Reads a UTF-8 <c>string</c> field; invalid UTF-8 is an error, as protobuf requires.</summary>
    public string ReadString()
    {
        ReadOnlySpan<byte> raw = ReadBytes(out long at);
        if (raw.IsEmpty)
        {
            return string.Empty;
        }

        try
        {
            return StrictUtf8.GetString(raw);
        }
        catch (ArgumentException)
        {
            throw Context.Fail(OnnxErrorKind.Malformed, "String field is not valid UTF-8.", at);
        }
    }

    /// <summary>Reads a length-delimited field as a nested message, one level deeper.</summary>
    public ProtoReader ReadMessage()
    {
        ReadOnlySpan<byte> payload = ReadBytes(out long at);
        return new ProtoReader(payload, at, Context, _depth + 1);
    }

    /// <summary>Skips a field of the given wire type.</summary>
    public void Skip(WireType wireType)
    {
        switch (wireType)
        {
            case WireType.Varint: _ = ReadVarint(); break;
            case WireType.Fixed64: Need(8, "fixed 64-bit value"); _pos += 8; break;
            case WireType.Fixed32: Need(4, "fixed 32-bit value"); _pos += 4; break;
            default: _ = ReadBytes(); break;
        }
    }

    /// <summary>Throws unless the field has the wire type its schema declares.</summary>
    public readonly void Expect(WireType actual, WireType expected, string field)
    {
        if (actual != expected)
        {
            throw Context.Fail(OnnxErrorKind.Malformed, $"Field {field} has wire type {actual}; expected {expected}.", Offset);
        }
    }

    private readonly void Need(int count, string what)
    {
        if (_data.Length - _pos < count)
        {
            throw Context.Fail(OnnxErrorKind.Malformed, $"The {what} is cut off by the end of the data.", Offset);
        }
    }
}
