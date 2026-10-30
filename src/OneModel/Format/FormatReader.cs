using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Text;
using System.Text.Json;
using System.Text.Unicode;
using System.Threading;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Format;

/// <summary>
/// Reads the OneModel format (docs/format.md). The header is checked against the file's real length, the manifest is parsed with
/// the BCL's non-recursive JSON parser, and every tensor range is checked against the data region before a view is made.
/// Tensors stay lazy views over the file's bytes (a memory map for large files). Graph and type nesting is limited to
/// <see cref="MaxNesting"/>. Names, node order and consumer lists are then resolved by the same code the ONNX importer uses.
/// </summary>
internal sealed class FormatReader
{
    public const int MaxNesting = 64;
    private const long MaxManifestBytes = 1L << 29;
    private const int JsonMaxDepth = 1024;

    private readonly string _source;
    private readonly ByteSource _bytes;
    private readonly CancellationToken _ct;
    private readonly List<(string? Name, int Index)> _path = [];
    private long _dataOffset;
    private long _dataLength;
    private int _depth;

    private FormatReader(ByteSource bytes, string source, CancellationToken ct)
    {
        _bytes = bytes;
        _source = source;
        _ct = ct;
    }

    /// <summary>Reads a model from <paramref name="bytes"/>. On success the model owns the source; on failure the caller disposes it.</summary>
    public static IrModel Read(ByteSource bytes, string source, CancellationToken ct)
    {
        var r = new FormatReader(bytes, source, ct);
        IrModel model = r.ReadAll();
        model.SetOwned([bytes]);
        return model;
    }

    private IrModel ReadAll()
    {
        byte[] manifest = ReadHeaderAndManifest();
        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(manifest, new JsonDocumentOptions { MaxDepth = JsonMaxDepth });
        }
        catch (JsonException e)
        {
            throw new ModelFormatException(ModelFormatErrorKind.Malformed, "The manifest is not valid JSON: " + e.Message, _source,
                FormatWriter.HeaderSize + (e.BytePositionInLine ?? 0), "manifest", e);
        }

        using (doc)
        {
            JsonElement root = doc.RootElement;
            Object(root);
            string format = ReqStr(root, "format");
            if (format != ModelFormat.Name)
            {
                throw Fail(ModelFormatErrorKind.Malformed, $"expected the string '{ModelFormat.Name}', found '{Clip(format)}'", "format");
            }

            if (ReqInt(root, "version") != ModelFormat.Version)
            {
                throw Fail(ModelFormatErrorKind.Unsupported, "the manifest version differs from the header's", "version");
            }

            var model = new IrModel
            {
                IrVersion = ReqInt(root, "ir_version"),
                ProducerName = OptStr(root, "producer_name"),
                ProducerVersion = OptStr(root, "producer_version"),
                Domain = OptStr(root, "domain"),
                ModelVersion = OptInt(root, "model_version"),
                DocString = OptStr(root, "doc"),
            };
            if (model.IrVersion < 3)
            {
                throw Fail(ModelFormatErrorKind.InvalidModel, $"ir_version {model.IrVersion} is older than 3, the oldest this reader accepts", "ir_version");
            }

            model.OpsetImports = List(root, "opsets", ReadOpset);
            model.Metadata = List(root, "metadata", ReadEntry);
            model.Graph = Child(root, "graph", ReadGraph);
            model.Functions = List(root, "functions", ReadFunction);

            try
            {
                GraphResolver.ResolveMain(model, new ParseContext(null, 128, null, _ct) { Main = _bytes, Root = "manifest" });
            }
            catch (OnnxImportException e)
            {
                throw new ModelFormatException(ModelFormatErrorKind.InvalidModel, e.Message, _source, -1, null, e);
            }

            return model;
        }
    }

    // ---- header -----------------------------------------------------------------------------------------------------------

    private byte[] ReadHeaderAndManifest()
    {
        long length = _bytes.Length;
        if (length < FormatWriter.HeaderSize)
        {
            throw new ModelFormatException(ModelFormatErrorKind.Malformed,
                $"the file is {length} bytes; the header alone needs {FormatWriter.HeaderSize}", _source, 0, "header");
        }

        ReadOnlySpan<byte> h = _bytes.GetSpan(0, FormatWriter.HeaderSize);
        if (!h[..8].SequenceEqual(FormatWriter.Magic))
        {
            throw new ModelFormatException(ModelFormatErrorKind.Malformed, "the file does not start with the magic bytes 'ONEMODEL'", _source, 0, "magic");
        }

        uint version = BinaryPrimitives.ReadUInt32LittleEndian(h[8..]);
        if (version != ModelFormat.Version)
        {
            throw new ModelFormatException(ModelFormatErrorKind.Unsupported,
                $"format version {version}; this reader reads version {ModelFormat.Version}", _source, 8, "version");
        }

        uint flags = BinaryPrimitives.ReadUInt32LittleEndian(h[12..]);
        if (flags != 0)
        {
            throw new ModelFormatException(ModelFormatErrorKind.Unsupported, $"flags 0x{flags:x} are set; version 1 defines none", _source, 12, "flags");
        }

        long manifestOffset = Field(h, 16, "manifest_offset");
        long manifestLength = Field(h, 24, "manifest_length");
        _dataOffset = Field(h, 32, "data_offset");
        _dataLength = Field(h, 40, "data_length");
        long fileLength = Field(h, 48, "file_length");
        if (BinaryPrimitives.ReadUInt64LittleEndian(h[56..]) != 0)
        {
            throw new ModelFormatException(ModelFormatErrorKind.Malformed, "the reserved header bytes are not zero", _source, 56, "reserved");
        }

        if (fileLength != length)
        {
            throw new ModelFormatException(ModelFormatErrorKind.Malformed,
                $"the header says the file is {fileLength} bytes; it is {length} (truncated or extended)", _source, 48, "file_length");
        }

        if (manifestOffset != FormatWriter.HeaderSize)
        {
            throw new ModelFormatException(ModelFormatErrorKind.Malformed, $"expected {FormatWriter.HeaderSize}, found {manifestOffset}", _source, 16, "manifest_offset");
        }

        if (manifestLength > MaxManifestBytes || manifestLength > length - manifestOffset)
        {
            throw new ModelFormatException(ModelFormatErrorKind.Malformed,
                $"the manifest is {manifestLength} bytes; at most {Math.Min(MaxManifestBytes, length - manifestOffset)} fit", _source, 24, "manifest_length");
        }

        if (_dataOffset % FormatWriter.Alignment != 0 || _dataOffset < manifestOffset + manifestLength || _dataOffset > length)
        {
            throw new ModelFormatException(ModelFormatErrorKind.Malformed,
                $"the data region must start on a 64-byte boundary after the manifest and inside the file; found {_dataOffset}", _source, 32, "data_offset");
        }

        if (_dataLength != length - _dataOffset)
        {
            throw new ModelFormatException(ModelFormatErrorKind.Malformed,
                $"the data region must run to the end of the file ({length - _dataOffset} bytes); the header says {_dataLength}", _source, 40, "data_length");
        }

        ReadOnlySpan<byte> manifest = _bytes.GetSpan(manifestOffset, (int)manifestLength);
        if (!Utf8.IsValid(manifest))
        {
            throw new ModelFormatException(ModelFormatErrorKind.Malformed, "the manifest is not valid UTF-8", _source, manifestOffset, "manifest");
        }

        return manifest.ToArray();
    }

    private long Field(ReadOnlySpan<byte> h, int at, string name)
    {
        ulong v = BinaryPrimitives.ReadUInt64LittleEndian(h[at..]);
        if (v > long.MaxValue)
        {
            throw new ModelFormatException(ModelFormatErrorKind.Malformed, $"{v} is larger than 2^63-1", _source, at, name);
        }

        return (long)v;
    }

    // ---- model pieces -----------------------------------------------------------------------------------------------------

    private OpsetImport ReadOpset(JsonElement e)
    {
        Object(e);
        return new OpsetImport(OptStr(e, "domain"), ReqInt(e, "version"));
    }

    private KeyValuePair<string, string> ReadEntry(JsonElement e)
    {
        Object(e);
        return new KeyValuePair<string, string>(OptStr(e, "key"), OptStr(e, "value"));
    }

    private IrFunction ReadFunction(JsonElement e)
    {
        Object(e);
        return new IrFunction
        {
            Name = OptStr(e, "name"),
            Domain = OptStr(e, "domain"),
            Overload = OptStr(e, "overload"),
            DocString = OptStr(e, "doc"),
            Inputs = List(e, "inputs", Str),
            Outputs = List(e, "outputs", Str),
            AttributeNames = List(e, "attribute_names", Str),
            AttributeDefaults = List(e, "attribute_defaults", ReadAttribute),
            Nodes = List(e, "nodes", ReadNode),
            OpsetImports = List(e, "opsets", ReadOpset),
        };
    }

    private IrGraph ReadGraph(JsonElement e)
    {
        _ct.ThrowIfCancellationRequested();
        Descend();
        Object(e);
        var raw = new RawGraph();
        var g = new IrGraph { Name = OptStr(e, "name"), DocString = OptStr(e, "doc"), Raw = raw };
        raw.Inputs.AddRange(List(e, "inputs", ReadTypedName));
        raw.Outputs.AddRange(List(e, "outputs", ReadTypedName));
        raw.ValueInfos.AddRange(List(e, "value_info", ReadTypedName));
        raw.Initializers.AddRange(List(e, "initializers", ReadTensor));
        raw.SparseInitializers.AddRange(List(e, "sparse_initializers", ReadSparse));
        raw.Nodes.AddRange(List(e, "nodes", ReadNode));
        _depth--;
        return g;
    }

    private (string Name, IrType? Type) ReadTypedName(JsonElement e)
    {
        Object(e);
        string name = ReqStr(e, "name");
        IrType? type = e.TryGetProperty("type", out _) ? Child(e, "type", ReadType) : null;
        return (name, type);
    }

    private IrNode ReadNode(JsonElement e)
    {
        Object(e);
        return new IrNode
        {
            Name = OptStr(e, "name"),
            OpType = ReqStr(e, "op"),
            Domain = OptStr(e, "domain"),
            Overload = OptStr(e, "overload"),
            DocString = OptStr(e, "doc"),
            InputNames = List(e, "inputs", Str),
            OutputNames = List(e, "outputs", Str),
            Attributes = List(e, "attributes", ReadAttribute),
        };
    }

    private IrAttr ReadAttribute(JsonElement e)
    {
        Object(e);
        string name = ReqStr(e, "name");
        long code = ReqInt(e, "type");
        string doc = OptStr(e, "doc");
        string reference = OptStr(e, "ref");
        if (code is < 0 or > 14)
        {
            throw Fail(ModelFormatErrorKind.Unsupported, $"attribute type {code} is not one of 0..14", "type");
        }

        var type = (IrAttributeType)code;
        if (reference.Length > 0)
        {
            return new IrAttr(name, type, reference, doc, null, 0);
        }

        if (type == IrAttributeType.Undefined)
        {
            throw Fail(ModelFormatErrorKind.InvalidModel, $"attribute '{name}' has no type and no ref", "type");
        }

        if (!e.TryGetProperty("value", out _))
        {
            throw Fail(ModelFormatErrorKind.Malformed, $"attribute '{name}' needs a 'value'", "value");
        }

        return type switch
        {
            IrAttributeType.Real => new IrAttr(name, type, "", doc, null, unchecked((int)Child(e, "value", Bits))),
            IrAttributeType.Whole => new IrAttr(name, type, "", doc, null, Child(e, "value", Int)),
            IrAttributeType.Text => new IrAttr(name, type, "", doc, Child(e, "value", Bytes), 0),
            IrAttributeType.Tensor => new IrAttr(name, type, "", doc, Child(e, "value", ReadTensor), 0),
            IrAttributeType.Graph => new IrAttr(name, type, "", doc, Child(e, "value", ReadGraph), 0),
            IrAttributeType.SparseTensor => new IrAttr(name, type, "", doc, Child(e, "value", ReadSparse), 0),
            IrAttributeType.TypeProto => new IrAttr(name, type, "", doc, Child(e, "value", ReadType), 0),
            IrAttributeType.Reals => new IrAttr(name, type, "", doc, List(e, "value", static (r, x) => BitConverter.UInt32BitsToSingle(r.Bits(x))), 0),
            IrAttributeType.Wholes => new IrAttr(name, type, "", doc, List(e, "value", Int), 0),
            IrAttributeType.Texts => new IrAttr(name, type, "", doc, List(e, "value", static (r, x) => (ReadOnlyMemory<byte>)r.Bytes(x)), 0),
            IrAttributeType.Tensors => new IrAttr(name, type, "", doc, List(e, "value", ReadTensor), 0),
            IrAttributeType.Graphs => new IrAttr(name, type, "", doc, List(e, "value", ReadGraph), 0),
            IrAttributeType.SparseTensors => new IrAttr(name, type, "", doc, List(e, "value", ReadSparse), 0),
            _ => new IrAttr(name, type, "", doc, List(e, "value", ReadType), 0),
        };
    }

    private SparseTensorData ReadSparse(JsonElement e)
    {
        Object(e);
        TensorData values = Child(e, "values", ReadTensor);
        TensorData indices = Child(e, "indices", ReadTensor);
        ImmutableArray<long> dims = List(e, "dims", Int);
        foreach (long d in dims)
        {
            if (d < 0)
            {
                throw Fail(ModelFormatErrorKind.InvalidModel, $"expected non-negative dimensions, found {d}", "dims");
            }
        }

        return new SparseTensorData(values, indices, dims);
    }

    private TensorData ReadTensor(JsonElement e)
    {
        Object(e);
        string name = OptStr(e, "name");
        long dtypeCode = ReqInt(e, "dtype");
        if (dtypeCode <= 0 || dtypeCode > int.MaxValue || (dtypeCode != 8 && TensorData.BitsPerElement((int)dtypeCode) < 0))
        {
            throw Fail(ModelFormatErrorKind.Unsupported, $"unknown tensor dtype {dtypeCode}", "dtype");
        }

        int dtype = (int)dtypeCode;
        ImmutableArray<long> dims = List(e, "dims", Int);
        long count = 1;
        foreach (long d in dims)
        {
            if (d < 0)
            {
                throw Fail(ModelFormatErrorKind.InvalidModel, $"tensor '{name}': expected non-negative dimensions, found {d}", "dims");
            }

            if (d != 0 && count > (1L << 60) / d)
            {
                throw Fail(ModelFormatErrorKind.InvalidModel, $"tensor '{name}' has more than 2^60 elements", "dims");
            }

            count = count == 0 || d == 0 ? 0 : count * d;
        }

        string doc = OptStr(e, "doc");
        if (dtype == 8)
        {
            ImmutableArray<ReadOnlyMemory<byte>> strings = List(e, "strings", static (r, x) => (ReadOnlyMemory<byte>)r.Bytes(x));
            if (strings.Length != count)
            {
                throw Fail(ModelFormatErrorKind.InvalidModel, $"string tensor '{name}' has {strings.Length} elements; its shape needs {count}", "strings");
            }

            return new TensorData(name, dtype, dims, count, null, 0, 0, strings, false, doc);
        }

        if (!e.TryGetProperty("data", out JsonElement data) || data.ValueKind != JsonValueKind.Object)
        {
            throw Fail(ModelFormatErrorKind.Malformed, $"tensor '{name}' needs a 'data' object with offset and length", "data");
        }

        Enter("data");
        long offset = ReqInt(data, "offset");
        long length = ReqInt(data, "length");
        Leave();
        long expected = TensorData.RawByteLength(dtype, count);
        if (expected > int.MaxValue)
        {
            throw Fail(ModelFormatErrorKind.Unsupported, $"tensor '{name}' needs {expected} bytes; at most 2^31-1 are supported", "data");
        }

        if (length != expected)
        {
            throw Fail(ModelFormatErrorKind.InvalidModel,
                $"{DTypes.OnnxName(dtype)} tensor '{name}' has {length} bytes; its shape needs {expected}", "data.length");
        }

        if (offset < 0 || offset > _dataLength - length)
        {
            throw Fail(ModelFormatErrorKind.Malformed,
                $"tensor '{name}' at {offset}+{length} lies outside the {_dataLength}-byte data region", "data.offset");
        }

        if (length > 0 && offset % FormatWriter.Alignment != 0)
        {
            throw Fail(ModelFormatErrorKind.Malformed, $"tensor '{name}' starts at {offset}, which is not a multiple of {FormatWriter.Alignment}", "data.offset");
        }

        return new TensorData(name, dtype, dims, count, _bytes, _dataOffset + offset, (int)length, [], false, doc);
    }

    private IrType ReadType(JsonElement e)
    {
        Descend();
        Object(e);
        string kind = ReqStr(e, "kind");
        IrType t;
        switch (kind)
        {
            case "tensor":
            case "sparse_tensor":
                t = new IrType(kind == "tensor" ? IrTypeKind.Tensor : IrTypeKind.SparseTensor) { ElemType = ReqInt32(e, "elem"), Dims = [] };
                if (e.TryGetProperty("shape", out _))
                {
                    t.HasShape = true;
                    t.Dims = List(e, "shape", ReadDim);
                }

                break;
            case "sequence":
            case "optional":
                t = new IrType(kind == "sequence" ? IrTypeKind.Sequence : IrTypeKind.Optional);
                t.Inner = e.TryGetProperty("inner", out _) ? Child(e, "inner", ReadType) : null;
                break;
            case "map":
                t = new IrType(IrTypeKind.Map) { ElemType = ReqInt32(e, "key") };
                t.Inner = e.TryGetProperty("inner", out _) ? Child(e, "inner", ReadType) : null;
                break;
            case "opaque":
                t = new IrType(IrTypeKind.Opaque) { OpaqueDomain = OptStr(e, "domain"), OpaqueName = OptStr(e, "name") };
                break;
            default:
                throw Fail(ModelFormatErrorKind.Unsupported, $"unknown type kind '{Clip(kind)}'", "kind");
        }

        _depth--;
        return t;
    }

    private IrDim ReadDim(JsonElement e)
    {
        switch (e.ValueKind)
        {
            case JsonValueKind.Null: return IrDim.Unknown;
            case JsonValueKind.String: return new IrDim(-1, Str(e));
            default:
                long v = Int(e);
                if (v < 0)
                {
                    throw Fail(ModelFormatErrorKind.Malformed, $"expected a dimension of 0 or more, a name or null, found {v}");
                }

                return new IrDim(v, null);
        }
    }

    // ---- JSON access, with a path for errors ------------------------------------------------------------------------------

    private void Descend()
    {
        if (++_depth > MaxNesting)
        {
            throw Fail(ModelFormatErrorKind.Unsupported, $"graphs and types are nested deeper than {MaxNesting} levels");
        }
    }

    private void Enter(string name) => _path.Add((name, -1));

    private void Leave() => _path.RemoveAt(_path.Count - 1);

    private ModelFormatException Fail(ModelFormatErrorKind kind, string message, string? field = null)
    {
        var sb = new StringBuilder("manifest");
        foreach ((string? name, int index) in _path)
        {
            if (name is not null)
            {
                sb.Append('.').Append(name);
            }
            else
            {
                sb.Append('[').Append(index).Append(']');
            }
        }

        if (field is not null)
        {
            sb.Append('.').Append(field);
        }

        return new ModelFormatException(kind, message, _source, -1, sb.ToString());
    }

    private static string Clip(string s) => s.Length <= 40 ? s : s[..40] + "...";

    // Names the kind of a value and, for a number, its text. Text of other kinds is left out: it may not be valid Unicode.
    private static string Show(JsonElement e) => e.ValueKind == JsonValueKind.Number ? Clip(e.GetRawText()) : Kind(e);

    private static string Kind(JsonElement e) => e.ValueKind.ToString().ToLowerInvariant();

    private void Object(JsonElement e)
    {
        if (e.ValueKind != JsonValueKind.Object)
        {
            throw Fail(ModelFormatErrorKind.Malformed, $"expected an object, found {Kind(e)}");
        }
    }

    private long Int(JsonElement e)
    {
        if (e.ValueKind != JsonValueKind.Number || !e.TryGetInt64(out long v))
        {
            throw Fail(ModelFormatErrorKind.Malformed, $"expected a 64-bit integer, found {Show(e)}");
        }

        return v;
    }

    private uint Bits(JsonElement e)
    {
        if (e.ValueKind != JsonValueKind.Number || !e.TryGetUInt32(out uint v))
        {
            throw Fail(ModelFormatErrorKind.Malformed, $"expected an unsigned 32-bit integer (float bits), found {Show(e)}");
        }

        return v;
    }

    private string Str(JsonElement e)
    {
        if (e.ValueKind != JsonValueKind.String)
        {
            throw Fail(ModelFormatErrorKind.Malformed, $"expected a string, found {Kind(e)}");
        }

        try
        {
            return e.GetString() ?? "";
        }
        catch (InvalidOperationException)
        {
            throw Fail(ModelFormatErrorKind.Malformed, "the string is not valid Unicode text");
        }
    }

    private byte[] Bytes(JsonElement e)
    {
        if (e.ValueKind != JsonValueKind.String || !e.TryGetBytesFromBase64(out byte[]? bytes))
        {
            throw Fail(ModelFormatErrorKind.Malformed, "expected a base64 string");
        }

        return bytes;
    }

    private string ReqStr(JsonElement o, string name)
    {
        if (!o.TryGetProperty(name, out JsonElement v))
        {
            throw Fail(ModelFormatErrorKind.Malformed, "required field is missing", name);
        }

        return Child(o, name, Str, v);
    }

    private string OptStr(JsonElement o, string name) => o.TryGetProperty(name, out JsonElement v) ? Child(o, name, Str, v) : "";

    private long ReqInt(JsonElement o, string name)
    {
        if (!o.TryGetProperty(name, out JsonElement v))
        {
            throw Fail(ModelFormatErrorKind.Malformed, "required field is missing", name);
        }

        return Child(o, name, Int, v);
    }

    private int ReqInt32(JsonElement o, string name)
    {
        long v = ReqInt(o, name);
        if (v is < int.MinValue or > int.MaxValue)
        {
            throw Fail(ModelFormatErrorKind.Malformed, $"expected a 32-bit integer, found {v}", name);
        }

        return (int)v;
    }

    private long OptInt(JsonElement o, string name) => o.TryGetProperty(name, out JsonElement v) ? Child(o, name, Int, v) : 0;

    private T Child<T>(JsonElement o, string name, Func<JsonElement, T> read, JsonElement? value = null)
    {
        JsonElement v = value.GetValueOrDefault();
        if (value is null && !o.TryGetProperty(name, out v))
        {
            throw Fail(ModelFormatErrorKind.Malformed, "required field is missing", name);
        }

        Enter(name);
        T result = read(v);
        Leave();
        return result;
    }

    private ImmutableArray<T> List<T>(JsonElement o, string name, Func<JsonElement, T> read) =>
        List(o, name, (_, x) => read(x));

    private ImmutableArray<T> List<T>(JsonElement o, string name, Func<FormatReader, JsonElement, T> read)
    {
        if (!o.TryGetProperty(name, out JsonElement arr))
        {
            return [];
        }

        Enter(name);
        if (arr.ValueKind != JsonValueKind.Array)
        {
            throw Fail(ModelFormatErrorKind.Malformed, $"expected an array, found {Kind(arr)}");
        }

        var b = ImmutableArray.CreateBuilder<T>(arr.GetArrayLength());
        int i = 0;
        foreach (JsonElement item in arr.EnumerateArray())
        {
            _path.Add((null, i++));
            b.Add(read(this, item));
            Leave();
        }

        Leave();
        return b.MoveToImmutable();
    }
}
