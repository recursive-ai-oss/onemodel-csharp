using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Collections.Immutable;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Onnx;

/// <summary>The unresolved contents of a <c>GraphProto</c>, kept until <see cref="GraphResolver"/> turns it into an <see cref="IrGraph"/>.</summary>
internal sealed class RawGraph
{
    public readonly List<IrNode> Nodes = [];
    public readonly List<(string Name, IrType? Type)> Inputs = [];
    public readonly List<(string Name, IrType? Type)> Outputs = [];
    public readonly List<(string Name, IrType? Type)> ValueInfos = [];
    public readonly List<TensorData> Initializers = [];
    public readonly List<SparseTensorData> SparseInitializers = [];
}

/// <summary>
/// Reads the ONNX messages (onnx.proto: ModelProto, GraphProto, NodeProto, AttributeProto, TensorProto, SparseTensorProto,
/// TypeProto, ValueInfoProto, FunctionProto) by field number. A field with an unexpected wire type is treated as unknown, as
/// protobuf does. Field numbers and the packed layouts come from onnx.proto and the protobuf encoding spec.
/// </summary>
internal static class OnnxParser
{
    public static IrModel ParseModel(ref ProtoReader r)
    {
        ParseContext ctx = r.Context;
        var model = new IrModel();
        var opsets = ImmutableArray.CreateBuilder<OpsetImport>();
        var functions = ImmutableArray.CreateBuilder<IrFunction>();
        var meta = ImmutableArray.CreateBuilder<KeyValuePair<string, string>>();
        bool irSet = false;
        bool hasGraph = false;
        while (r.TryReadTag(out int field, out WireType wt))
        {
            ctx.Cancel.ThrowIfCancellationRequested();
            switch (field)
            {
                case 1 when wt == WireType.Varint: model.IrVersion = r.ReadInt64(); irSet = true; break;
                case 2 when wt == WireType.Length: model.ProducerName = r.ReadString(); break;
                case 3 when wt == WireType.Length: model.ProducerVersion = r.ReadString(); break;
                case 4 when wt == WireType.Length: model.Domain = r.ReadString(); break;
                case 5 when wt == WireType.Varint: model.ModelVersion = r.ReadInt64(); break;
                case 6 when wt == WireType.Length: model.DocString = r.ReadString(); break;
                case 7 when wt == WireType.Length:
                    ctx.Push("graph");
                    {
                        ProtoReader g = r.ReadMessage();
                        model.Graph = ParseGraph(ref g);
                    }

                    ctx.Pop();
                    hasGraph = true;
                    break;
                case 8 when wt == WireType.Length:
                    ctx.Push("opset_import", opsets.Count);
                    {
                        ProtoReader o = r.ReadMessage();
                        opsets.Add(ParseOpset(ref o));
                    }

                    ctx.Pop();
                    break;
                case 14 when wt == WireType.Length:
                    ctx.Push("metadata_props", meta.Count);
                    {
                        ProtoReader m = r.ReadMessage();
                        meta.Add(ParseEntry(ref m));
                    }

                    ctx.Pop();
                    break;
                case 25 when wt == WireType.Length:
                    ctx.Push("functions", functions.Count);
                    {
                        ProtoReader f = r.ReadMessage();
                        functions.Add(ParseFunction(ref f));
                    }

                    ctx.Pop();
                    break;
                default: r.Skip(wt); break;
            }
        }

        if (!irSet || model.IrVersion < 3)
        {
            throw ctx.Fail(OnnxErrorKind.InvalidModel,
                irSet ? $"ir_version {model.IrVersion} is older than 3, the oldest this importer reads." : "The model has no ir_version.");
        }

        if (!hasGraph)
        {
            throw ctx.Fail(OnnxErrorKind.InvalidModel, "The model has no graph.");
        }

        model.OpsetImports = opsets.ToImmutable();
        model.Functions = functions.ToImmutable();
        model.Metadata = meta.ToImmutable();
        return model;
    }

    private static OpsetImport ParseOpset(ref ProtoReader r)
    {
        string domain = "";
        long version = 0;
        while (r.TryReadTag(out int field, out WireType wt))
        {
            switch (field)
            {
                case 1 when wt == WireType.Length: domain = r.ReadString(); break;
                case 2 when wt == WireType.Varint: version = r.ReadInt64(); break;
                default: r.Skip(wt); break;
            }
        }

        return new OpsetImport(domain, version);
    }

    private static KeyValuePair<string, string> ParseEntry(ref ProtoReader r)
    {
        string key = "", value = "";
        while (r.TryReadTag(out int field, out WireType wt))
        {
            switch (field)
            {
                case 1 when wt == WireType.Length: key = r.ReadString(); break;
                case 2 when wt == WireType.Length: value = r.ReadString(); break;
                default: r.Skip(wt); break;
            }
        }

        return new KeyValuePair<string, string>(key, value);
    }

    // ---- graph --------------------------------------------------------------------------------------------------------------

    public static IrGraph ParseGraph(ref ProtoReader r)
    {
        ParseContext ctx = r.Context;
        var g = new IrGraph();
        var raw = new RawGraph();
        g.Raw = raw;
        while (r.TryReadTag(out int field, out WireType wt))
        {
            switch (field)
            {
                case 1 when wt == WireType.Length:
                    ctx.Cancel.ThrowIfCancellationRequested();
                    ctx.Push("node", raw.Nodes.Count);
                    {
                        ProtoReader n = r.ReadMessage();
                        raw.Nodes.Add(ParseNode(ref n));
                    }

                    ctx.Pop();
                    break;
                case 2 when wt == WireType.Length: g.Name = r.ReadString(); break;
                case 5 when wt == WireType.Length:
                    ctx.Push("initializer", raw.Initializers.Count);
                    {
                        ProtoReader t = r.ReadMessage();
                        raw.Initializers.Add(ParseTensor(ref t));
                    }

                    ctx.Pop();
                    break;
                case 10 when wt == WireType.Length: g.DocString = r.ReadString(); break;
                case 11 when wt == WireType.Length: ParseValueInfoInto(ref r, raw.Inputs, "input"); break;
                case 12 when wt == WireType.Length: ParseValueInfoInto(ref r, raw.Outputs, "output"); break;
                case 13 when wt == WireType.Length: ParseValueInfoInto(ref r, raw.ValueInfos, "value_info"); break;
                case 15 when wt == WireType.Length:
                    ctx.Push("sparse_initializer", raw.SparseInitializers.Count);
                    {
                        ProtoReader t = r.ReadMessage();
                        raw.SparseInitializers.Add(ParseSparse(ref t));
                    }

                    ctx.Pop();
                    break;
                default: r.Skip(wt); break;
            }
        }

        return g;
    }

    private static void ParseValueInfoInto(ref ProtoReader r, List<(string, IrType?)> list, string fieldName)
    {
        ParseContext ctx = r.Context;
        ctx.Push(fieldName, list.Count);
        ProtoReader v = r.ReadMessage();
        string name = "";
        IrType? type = null;
        while (v.TryReadTag(out int field, out WireType wt))
        {
            switch (field)
            {
                case 1 when wt == WireType.Length: name = v.ReadString(); break;
                case 2 when wt == WireType.Length:
                    ctx.Push("type");
                    {
                        ProtoReader t = v.ReadMessage();
                        type = ParseType(ref t);
                    }

                    ctx.Pop();
                    break;
                default: v.Skip(wt); break;
            }
        }

        list.Add((name, type));
        ctx.Pop();
    }

    // ---- node and attribute -------------------------------------------------------------------------------------------------

    public static IrNode ParseNode(ref ProtoReader r)
    {
        ParseContext ctx = r.Context;
        var node = new IrNode();
        var inputs = ImmutableArray.CreateBuilder<string>();
        var outputs = ImmutableArray.CreateBuilder<string>();
        var attrs = ImmutableArray.CreateBuilder<IrAttr>();
        while (r.TryReadTag(out int field, out WireType wt))
        {
            switch (field)
            {
                case 1 when wt == WireType.Length: inputs.Add(r.ReadString()); break;
                case 2 when wt == WireType.Length: outputs.Add(r.ReadString()); break;
                case 3 when wt == WireType.Length: node.Name = r.ReadString(); break;
                case 4 when wt == WireType.Length: node.OpType = r.ReadString(); break;
                case 6 when wt == WireType.Length: node.DocString = r.ReadString(); break;
                case 7 when wt == WireType.Length: node.Domain = r.ReadString(); break;
                case 8 when wt == WireType.Length: node.Overload = r.ReadString(); break;
                case 5 when wt == WireType.Length:
                    ctx.Push("attribute", attrs.Count);
                    {
                        ProtoReader a = r.ReadMessage();
                        attrs.Add(ParseAttribute(ref a));
                    }

                    ctx.Pop();
                    break;
                default: r.Skip(wt); break;
            }
        }

        node.InputNames = inputs.ToImmutable();
        node.OutputNames = outputs.ToImmutable();
        node.Attributes = attrs.ToImmutable();
        return node;
    }

    public static IrAttr ParseAttribute(ref ProtoReader r)
    {
        ParseContext ctx = r.Context;
        string name = "", refName = "", doc = "";
        int declared = 0;
        bool hasF = false, hasI = false, hasS = false, hasT = false, hasG = false, hasSparse = false, hasTp = false;
        long scalar, scalarF = 0, scalarI = 0;
        byte[]? s = null;
        TensorData? t = null;
        IrGraph? g = null;
        SparseTensorData? sparse = null;
        IrType? tp = null;
        var floats = ImmutableArray.CreateBuilder<float>();
        var ints = ImmutableArray.CreateBuilder<long>();
        var strings = ImmutableArray.CreateBuilder<ReadOnlyMemory<byte>>();
        var tensors = ImmutableArray.CreateBuilder<TensorData>();
        var graphs = ImmutableArray.CreateBuilder<IrGraph>();
        var sparses = ImmutableArray.CreateBuilder<SparseTensorData>();
        var types = ImmutableArray.CreateBuilder<IrType>();
        bool hasFloats = false, hasInts = false, hasStrings = false, hasTensors = false, hasGraphs = false, hasSparses = false, hasTypes = false;

        while (r.TryReadTag(out int field, out WireType wt))
        {
            switch (field)
            {
                case 1 when wt == WireType.Length: name = r.ReadString(); break;
                case 13 when wt == WireType.Length: doc = r.ReadString(); break;
                case 21 when wt == WireType.Length: refName = r.ReadString(); break;
                case 20 when wt == WireType.Varint: declared = r.ReadInt32(); break;
                case 2 when wt == WireType.Fixed32: scalarF = unchecked((int)r.ReadFixed32()); hasF = true; break;
                case 3 when wt == WireType.Varint: scalarI = r.ReadInt64(); hasI = true; break;
                case 4 when wt == WireType.Length: s = r.ReadBytes().ToArray(); hasS = true; break;
                case 5 when wt == WireType.Length:
                    ctx.Push("t");
                    {
                        ProtoReader m = r.ReadMessage();
                        t = ParseTensor(ref m);
                    }

                    ctx.Pop();
                    hasT = true;
                    break;
                case 6 when wt == WireType.Length:
                    ctx.Push("g");
                    {
                        ProtoReader m = r.ReadMessage();
                        g = ParseGraph(ref m);
                    }

                    ctx.Pop();
                    hasG = true;
                    break;
                case 22 when wt == WireType.Length:
                    ctx.Push("sparse_tensor");
                    {
                        ProtoReader m = r.ReadMessage();
                        sparse = ParseSparse(ref m);
                    }

                    ctx.Pop();
                    hasSparse = true;
                    break;
                case 14 when wt == WireType.Length:
                    ctx.Push("tp");
                    {
                        ProtoReader m = r.ReadMessage();
                        tp = ParseType(ref m);
                    }

                    ctx.Pop();
                    hasTp = true;
                    break;
                case 7 when wt is WireType.Length or WireType.Fixed32:
                    hasFloats = true;
                    if (wt == WireType.Fixed32)
                    {
                        floats.Add(BitConverter.UInt32BitsToSingle(r.ReadFixed32()));
                    }
                    else
                    {
                        ReadOnlySpan<byte> p = r.ReadBytes(out long at);
                        if (p.Length % 4 != 0)
                        {
                            throw ctx.Fail(OnnxErrorKind.Malformed, "Packed floats are not a multiple of 4 bytes.", at);
                        }

                        for (int i = 0; i < p.Length; i += 4)
                        {
                            floats.Add(BitConverter.UInt32BitsToSingle(BinaryPrimitives.ReadUInt32LittleEndian(p.Slice(i, 4))));
                        }
                    }

                    break;
                case 8 when wt is WireType.Length or WireType.Varint:
                    hasInts = true;
                    if (wt == WireType.Varint)
                    {
                        ints.Add(r.ReadInt64());
                    }
                    else
                    {
                        ProtoReader p = r.ReadMessage();
                        while (!p.AtEnd)
                        {
                            ints.Add(p.ReadInt64());
                        }
                    }

                    break;
                case 9 when wt == WireType.Length: strings.Add(r.ReadBytes().ToArray()); hasStrings = true; break;
                case 10 when wt == WireType.Length:
                    ctx.Push("tensors", tensors.Count);
                    {
                        ProtoReader m = r.ReadMessage();
                        tensors.Add(ParseTensor(ref m));
                    }

                    ctx.Pop();
                    hasTensors = true;
                    break;
                case 11 when wt == WireType.Length:
                    ctx.Push("graphs", graphs.Count);
                    {
                        ProtoReader m = r.ReadMessage();
                        graphs.Add(ParseGraph(ref m));
                    }

                    ctx.Pop();
                    hasGraphs = true;
                    break;
                case 23 when wt == WireType.Length:
                    ctx.Push("sparse_tensors", sparses.Count);
                    {
                        ProtoReader m = r.ReadMessage();
                        sparses.Add(ParseSparse(ref m));
                    }

                    ctx.Pop();
                    hasSparses = true;
                    break;
                case 15 when wt == WireType.Length:
                    ctx.Push("type_protos", types.Count);
                    {
                        ProtoReader m = r.ReadMessage();
                        types.Add(ParseType(ref m) ?? throw ctx.Fail(OnnxErrorKind.InvalidModel, "A type_protos entry has no type set.", m.Offset));
                    }

                    ctx.Pop();
                    hasTypes = true;
                    break;
                default: r.Skip(wt); break;
            }
        }

        if (declared is < 0 or > 14)
        {
            throw ctx.Fail(OnnxErrorKind.Unsupported, $"Attribute '{name}' has unknown type {declared}.");
        }

        var type = (IrAttributeType)declared;
        if (type == IrAttributeType.Undefined)
        {
            // Models from before the type field was required: infer from the field that is set.
            type = hasF ? IrAttributeType.Real : hasI ? IrAttributeType.Whole : hasS ? IrAttributeType.Text
                : hasT ? IrAttributeType.Tensor : hasG ? IrAttributeType.Graph : hasSparse ? IrAttributeType.SparseTensor
                : hasTp ? IrAttributeType.TypeProto : hasFloats ? IrAttributeType.Reals : hasInts ? IrAttributeType.Wholes
                : hasStrings ? IrAttributeType.Texts : hasTensors ? IrAttributeType.Tensors : hasGraphs ? IrAttributeType.Graphs
                : hasSparses ? IrAttributeType.SparseTensors : hasTypes ? IrAttributeType.TypeProtos : IrAttributeType.Undefined;
            if (type == IrAttributeType.Undefined && refName.Length == 0)
            {
                throw ctx.Fail(OnnxErrorKind.InvalidModel, $"Attribute '{name}' has no type and no value to infer one from.");
            }
        }

        if (refName.Length > 0)
        {
            return new IrAttr(name, type, refName, doc, null, 0);
        }

        object? value;
        switch (type)
        {
            case IrAttributeType.Real:
            case IrAttributeType.Whole:
                value = null;
                break;
            case IrAttributeType.Text: value = s ?? []; break;
            case IrAttributeType.Tensor: value = t ?? throw Missing(ctx, name, "t"); break;
            case IrAttributeType.Graph: value = g ?? throw Missing(ctx, name, "g"); break;
            case IrAttributeType.SparseTensor: value = sparse ?? throw Missing(ctx, name, "sparse_tensor"); break;
            case IrAttributeType.TypeProto: value = tp ?? throw Missing(ctx, name, "tp"); break;
            case IrAttributeType.Reals: value = floats.ToImmutable(); break;
            case IrAttributeType.Wholes: value = ints.ToImmutable(); break;
            case IrAttributeType.Texts: value = strings.ToImmutable(); break;
            case IrAttributeType.Tensors: value = tensors.ToImmutable(); break;
            case IrAttributeType.Graphs: value = graphs.ToImmutable(); break;
            case IrAttributeType.SparseTensors: value = sparses.ToImmutable(); break;
            default: value = types.ToImmutable(); break;
        }

        // A scalar attribute whose field is absent takes the protobuf default, zero.
        scalar = type == IrAttributeType.Real ? (hasF ? scalarF : 0) : type == IrAttributeType.Whole ? (hasI ? scalarI : 0) : 0;

        return new IrAttr(name, type, "", doc, value, scalar);
    }

    private static OnnxImportException Missing(ParseContext ctx, string name, string field) =>
        ctx.Fail(OnnxErrorKind.InvalidModel, $"Attribute '{name}' is declared to hold a value in '{field}' but that field is absent.");

    // ---- types ---------------------------------------------------------------------------------------------------------------

    /// <summary>Reads a <c>TypeProto</c>; returns null when no type is set in it.</summary>
    public static IrType? ParseType(ref ProtoReader r)
    {
        ParseContext ctx = r.Context;
        IrType? result = null;
        while (r.TryReadTag(out int field, out WireType wt))
        {
            switch (field)
            {
                case 1 when wt == WireType.Length: result = ParseTensorType(ref r, IrTypeKind.Tensor); break;
                case 8 when wt == WireType.Length: result = ParseTensorType(ref r, IrTypeKind.SparseTensor); break;
                case 4 when wt == WireType.Length:
                case 9 when wt == WireType.Length:
                    {
                        var t = new IrType(field == 4 ? IrTypeKind.Sequence : IrTypeKind.Optional);
                        ProtoReader m = r.ReadMessage();
                        while (m.TryReadTag(out int f2, out WireType w2))
                        {
                            if (f2 == 1 && w2 == WireType.Length)
                            {
                                ctx.Push("elem_type");
                                ProtoReader inner = m.ReadMessage();
                                t.Inner = ParseType(ref inner);
                                ctx.Pop();
                            }
                            else
                            {
                                m.Skip(w2);
                            }
                        }

                        result = t;
                        break;
                    }

                case 5 when wt == WireType.Length:
                    {
                        var t = new IrType(IrTypeKind.Map);
                        ProtoReader m = r.ReadMessage();
                        while (m.TryReadTag(out int f2, out WireType w2))
                        {
                            if (f2 == 1 && w2 == WireType.Varint)
                            {
                                t.ElemType = m.ReadInt32();
                            }
                            else if (f2 == 2 && w2 == WireType.Length)
                            {
                                ctx.Push("value_type");
                                ProtoReader inner = m.ReadMessage();
                                t.Inner = ParseType(ref inner);
                                ctx.Pop();
                            }
                            else
                            {
                                m.Skip(w2);
                            }
                        }

                        result = t;
                        break;
                    }

                case 7 when wt == WireType.Length:
                    {
                        var t = new IrType(IrTypeKind.Opaque) { OpaqueDomain = "", OpaqueName = "" };
                        ProtoReader m = r.ReadMessage();
                        while (m.TryReadTag(out int f2, out WireType w2))
                        {
                            if (f2 == 1 && w2 == WireType.Length)
                            {
                                t.OpaqueDomain = m.ReadString();
                            }
                            else if (f2 == 2 && w2 == WireType.Length)
                            {
                                t.OpaqueName = m.ReadString();
                            }
                            else
                            {
                                m.Skip(w2);
                            }
                        }

                        result = t;
                        break;
                    }

                default: r.Skip(wt); break;
            }
        }

        return result;
    }

    private static IrType ParseTensorType(ref ProtoReader r, IrTypeKind kind)
    {
        ParseContext ctx = r.Context;
        var t = new IrType(kind);
        ProtoReader m = r.ReadMessage();
        while (m.TryReadTag(out int field, out WireType wt))
        {
            if (field == 1 && wt == WireType.Varint)
            {
                t.ElemType = m.ReadInt32();
            }
            else if (field == 2 && wt == WireType.Length)
            {
                t.HasShape = true;
                ctx.Push("shape");
                ProtoReader sh = m.ReadMessage();
                var dims = ImmutableArray.CreateBuilder<IrDim>();
                while (sh.TryReadTag(out int sf, out WireType sw))
                {
                    if (sf == 1 && sw == WireType.Length)
                    {
                        ProtoReader d = sh.ReadMessage();
                        long value = -1;
                        string? param = null;
                        while (d.TryReadTag(out int df, out WireType dw))
                        {
                            if (df == 1 && dw == WireType.Varint)
                            {
                                value = d.ReadInt64();
                                param = null;
                            }
                            else if (df == 2 && dw == WireType.Length)
                            {
                                param = d.ReadString();
                                value = -1;
                            }
                            else
                            {
                                d.Skip(dw);
                            }
                        }

                        dims.Add(new IrDim(value, param));
                    }
                    else
                    {
                        sh.Skip(sw);
                    }
                }

                t.Dims = dims.ToImmutable();
                ctx.Pop();
            }
            else
            {
                m.Skip(wt);
            }
        }

        if (!t.HasShape)
        {
            t.Dims = [];
        }

        return t;
    }

    // ---- tensors --------------------------------------------------------------------------------------------------------------

    public static SparseTensorData ParseSparse(ref ProtoReader r)
    {
        ParseContext ctx = r.Context;
        TensorData? values = null, indices = null;
        var dims = ImmutableArray.CreateBuilder<long>();
        while (r.TryReadTag(out int field, out WireType wt))
        {
            switch (field)
            {
                case 1 when wt == WireType.Length:
                    ctx.Push("values");
                    {
                        ProtoReader m = r.ReadMessage();
                        values = ParseTensor(ref m);
                    }

                    ctx.Pop();
                    break;
                case 2 when wt == WireType.Length:
                    ctx.Push("indices");
                    {
                        ProtoReader m = r.ReadMessage();
                        indices = ParseTensor(ref m);
                    }

                    ctx.Pop();
                    break;
                case 3 when wt == WireType.Varint: dims.Add(r.ReadInt64()); break;
                case 3 when wt == WireType.Length:
                    {
                        ProtoReader p = r.ReadMessage();
                        while (!p.AtEnd)
                        {
                            dims.Add(p.ReadInt64());
                        }

                        break;
                    }

                default: r.Skip(wt); break;
            }
        }

        if (values is null || indices is null)
        {
            throw ctx.Fail(OnnxErrorKind.InvalidModel, "A sparse tensor needs both values and indices.");
        }

        foreach (long d in dims)
        {
            if (d < 0)
            {
                throw ctx.Fail(OnnxErrorKind.InvalidModel, $"A sparse tensor has the negative dimension {d}.");
            }
        }

        return new SparseTensorData(values, indices, dims.ToImmutable());
    }

    public static TensorData ParseTensor(ref ProtoReader r)
    {
        ParseContext ctx = r.Context;
        ctx.Cancel.ThrowIfCancellationRequested();
        var dims = ImmutableArray.CreateBuilder<long>();
        int dataType = 0;
        string name = "", doc = "";
        bool hasSegment = false, external = false;
        bool hasRaw = false;
        long rawOffset = 0;
        int rawLength = 0;
        ByteList floatData = default, int32Data = default, int64Data = default, doubleData = default, uint64Data = default;
        bool hasFloat = false, hasInt32 = false, hasInt64 = false, hasDouble = false, hasUInt64 = false;
        var strings = ImmutableArray.CreateBuilder<ReadOnlyMemory<byte>>();
        string? extLocation = null, extOffset = null, extLength = null;

        while (r.TryReadTag(out int field, out WireType wt))
        {
            switch (field)
            {
                case 1 when wt == WireType.Varint: dims.Add(r.ReadInt64()); break;
                case 1 when wt == WireType.Length:
                    {
                        ProtoReader p = r.ReadMessage();
                        while (!p.AtEnd)
                        {
                            dims.Add(p.ReadInt64());
                        }

                        break;
                    }

                case 2 when wt == WireType.Varint: dataType = r.ReadInt32(); break;
                case 3 when wt == WireType.Length: r.Skip(wt); hasSegment = true; break;
                case 4 when wt == WireType.Fixed32: hasFloat = true; floatData.AppendUInt32(r.ReadFixed32(), ctx); break;
                case 4 when wt == WireType.Length:
                    {
                        hasFloat = true;
                        ReadOnlySpan<byte> p = r.ReadBytes(out long at);
                        AppendPacked(ref floatData, p, 4, at, ctx);
                        break;
                    }

                case 5 when wt == WireType.Varint: hasInt32 = true; int32Data.AppendUInt32(unchecked((uint)r.ReadVarint()), ctx); break;
                case 5 when wt == WireType.Length:
                    {
                        hasInt32 = true;
                        ProtoReader p = r.ReadMessage();
                        while (!p.AtEnd)
                        {
                            int32Data.AppendUInt32(unchecked((uint)p.ReadVarint()), ctx);
                        }

                        break;
                    }

                case 6 when wt == WireType.Length: strings.Add(r.ReadBytes().ToArray()); break;
                case 7 when wt == WireType.Varint: hasInt64 = true; int64Data.AppendUInt64(r.ReadVarint(), ctx); break;
                case 7 when wt == WireType.Length:
                    {
                        hasInt64 = true;
                        ProtoReader p = r.ReadMessage();
                        while (!p.AtEnd)
                        {
                            int64Data.AppendUInt64(p.ReadVarint(), ctx);
                        }

                        break;
                    }

                case 8 when wt == WireType.Length: name = r.ReadString(); break;
                case 9 when wt == WireType.Length:
                    {
                        ReadOnlySpan<byte> p = r.ReadBytes(out long at);
                        hasRaw = true;
                        rawOffset = at;
                        rawLength = p.Length;
                        break;
                    }

                case 10 when wt == WireType.Fixed64: hasDouble = true; doubleData.AppendUInt64(r.ReadFixed64(), ctx); break;
                case 10 when wt == WireType.Length:
                    {
                        hasDouble = true;
                        ReadOnlySpan<byte> p = r.ReadBytes(out long at);
                        AppendPacked(ref doubleData, p, 8, at, ctx);
                        break;
                    }

                case 11 when wt == WireType.Varint: hasUInt64 = true; uint64Data.AppendUInt64(r.ReadVarint(), ctx); break;
                case 11 when wt == WireType.Length:
                    {
                        hasUInt64 = true;
                        ProtoReader p = r.ReadMessage();
                        while (!p.AtEnd)
                        {
                            uint64Data.AppendUInt64(p.ReadVarint(), ctx);
                        }

                        break;
                    }

                case 12 when wt == WireType.Length: doc = r.ReadString(); break;
                case 13 when wt == WireType.Length:
                    {
                        ProtoReader e = r.ReadMessage();
                        KeyValuePair<string, string> kv = ParseEntry(ref e);
                        switch (kv.Key)
                        {
                            case "location": extLocation = kv.Value; break;
                            case "offset": extOffset = kv.Value; break;
                            case "length": extLength = kv.Value; break;
                            default: break; // "checksum" and unknown keys are not used.
                        }

                        break;
                    }

                case 14 when wt == WireType.Varint:
                    {
                        int loc = r.ReadInt32();
                        if (loc is not (0 or 1))
                        {
                            throw ctx.Fail(OnnxErrorKind.InvalidModel, $"data_location {loc} is not DEFAULT or EXTERNAL.");
                        }

                        external = loc == 1;
                        break;
                    }

                default: r.Skip(wt); break;
            }
        }

        if (hasSegment)
        {
            throw ctx.Fail(OnnxErrorKind.Unsupported, "Tensors stored as segments are not supported.");
        }

        if (dataType <= 0 || (dataType != 8 && DTypeInfoBits(dataType) < 0))
        {
            throw ctx.Fail(OnnxErrorKind.Unsupported, dataType == 0 ? "The tensor has no data_type." : $"Unknown tensor data_type {dataType}.");
        }

        // Element count: the product of the dimensions, checked.
        long count = 1;
        foreach (long d in dims)
        {
            if (d < 0)
            {
                throw ctx.Fail(OnnxErrorKind.InvalidModel, $"Tensor '{name}' has the negative dimension {d}.");
            }

            if (d != 0 && count > (1L << 60) / d)
            {
                throw ctx.Fail(OnnxErrorKind.InvalidModel, $"Tensor '{name}' has more than 2^60 elements.");
            }

            count = count == 0 || d == 0 ? 0 : count * d;
        }

        ImmutableArray<long> dimsArray = dims.ToImmutable();
        string typeName = DTypes.OnnxName(dataType);

        if (dataType == 8)
        {
            if (hasRaw || external || hasFloat || hasInt32 || hasInt64 || hasDouble || hasUInt64)
            {
                throw ctx.Fail(OnnxErrorKind.InvalidModel, $"String tensor '{name}' must keep its elements in string_data.");
            }

            if (strings.Count != count)
            {
                throw ctx.Fail(OnnxErrorKind.InvalidModel, $"String tensor '{name}' has {strings.Count} elements in string_data; its shape needs {count}.");
            }

            return new TensorData(name, dataType, dimsArray, count, null, 0, 0, strings.ToImmutable(), false, doc);
        }

        if (strings.Count > 0)
        {
            throw ctx.Fail(OnnxErrorKind.InvalidModel, $"{typeName} tensor '{name}' has string_data.");
        }

        long expected = TensorData.RawByteLength(dataType, count);
        if (expected > int.MaxValue)
        {
            throw ctx.Fail(OnnxErrorKind.Unsupported, $"Tensor '{name}' needs {expected} bytes; at most 2^31-1 are supported.");
        }

        if (external)
        {
            if (hasRaw || hasFloat || hasInt32 || hasInt64 || hasDouble || hasUInt64)
            {
                throw ctx.Fail(OnnxErrorKind.InvalidModel, $"Tensor '{name}' is external but also holds inline data.");
            }

            if (ctx.External is null)
            {
                throw ctx.Fail(OnnxErrorKind.ExternalData,
                    $"Tensor '{name}' uses external data, but no directory is set for it (load from a file path, or set ExternalDataDirectory).");
            }

            ByteSource src = ctx.External.Resolve(extLocation, extOffset, extLength, expected, ctx, out long off, out int len);
            return new TensorData(name, dataType, dimsArray, count, src, off, len, [], true, doc);
        }

        if (hasRaw)
        {
            if (hasFloat || hasInt32 || hasInt64 || hasDouble || hasUInt64)
            {
                throw ctx.Fail(OnnxErrorKind.InvalidModel, $"Tensor '{name}' has both raw_data and a typed data field.");
            }

            if (rawLength != expected)
            {
                throw ctx.Fail(OnnxErrorKind.InvalidModel,
                    $"{typeName} tensor '{name}' has {rawLength} bytes of raw_data; its shape needs {expected}.");
            }

            return new TensorData(name, dataType, dimsArray, count, ctx.Main, rawOffset, rawLength, [], false, doc);
        }

        byte[] bytes = FromTypedFields(dataType, name, count, expected, floatData, int32Data, int64Data, doubleData, uint64Data,
            hasFloat, hasInt32, hasInt64, hasDouble, hasUInt64, ctx);
        return new TensorData(name, dataType, dimsArray, count, ByteSource.FromArray(bytes), 0, bytes.Length, [], false, doc);
    }

    private static int DTypeInfoBits(int dataType) => TensorData.BitsPerElement(dataType);

    private static void AppendPacked(ref ByteList list, ReadOnlySpan<byte> payload, int width, long at, ParseContext ctx)
    {
        if (payload.Length % width != 0)
        {
            throw ctx.Fail(OnnxErrorKind.Malformed, $"Packed values are not a multiple of {width} bytes.", at);
        }

        Span<byte> dst = list.Grow(payload.Length, ctx);
        if (BitConverter.IsLittleEndian)
        {
            payload.CopyTo(dst);
        }
        else
        {
            for (int i = 0; i < payload.Length; i += width)
            {
                payload.Slice(i, width).CopyTo(dst.Slice(i, width));
                dst.Slice(i, width).Reverse();
            }
        }
    }

    /// <summary>
    /// Builds a tensor's little-endian bytes from the typed field its element type uses (onnx.proto, TensorProto): float_data for
    /// float and complex64, double_data for double and complex128, int64_data, uint64_data for uint32 and uint64, and int32_data
    /// for every other numeric type, one element per entry (sub-byte types: one packed byte per entry, six-bit types one element).
    /// </summary>
    private static byte[] FromTypedFields(int dataType, string name, long count, long expected, ByteList floatData, ByteList int32Data,
        ByteList int64Data, ByteList doubleData, ByteList uint64Data, bool hasFloat, bool hasInt32, bool hasInt64, bool hasDouble,
        bool hasUInt64, ParseContext ctx)
    {
        string typeName = DTypes.OnnxName(dataType);
        bool wantFloat = dataType is 1 or 14, wantDouble = dataType is 11 or 15, wantInt64 = dataType == 7, wantUInt64 = dataType is 12 or 13;
        bool wantInt32 = !(wantFloat || wantDouble || wantInt64 || wantUInt64);
        if ((hasFloat && !wantFloat) || (hasDouble && !wantDouble) || (hasInt64 && !wantInt64) || (hasUInt64 && !wantUInt64) || (hasInt32 && !wantInt32))
        {
            throw ctx.Fail(OnnxErrorKind.InvalidModel, $"{typeName} tensor '{name}' uses a typed data field that does not belong to its type.");
        }

        byte[] result;
        if (wantFloat)
        {
            result = floatData.ToArray();
        }
        else if (wantDouble)
        {
            result = doubleData.ToArray();
        }
        else if (wantInt64)
        {
            result = int64Data.ToArray();
        }
        else if (dataType == 13)
        {
            result = uint64Data.ToArray();
        }
        else if (dataType == 12)
        {
            ReadOnlySpan<byte> src = uint64Data.Span;
            result = new byte[src.Length / 2];
            for (int i = 0; i < src.Length / 8; i++)
            {
                BinaryPrimitives.WriteUInt32LittleEndian(result.AsSpan(i * 4), unchecked((uint)BinaryPrimitives.ReadUInt64LittleEndian(src.Slice(i * 8, 8))));
            }
        }
        else
        {
            ReadOnlySpan<byte> src = int32Data.Span;
            int n = src.Length / 4;
            int bits = TensorData.BitsPerElement(dataType);
            if (dataType == 6)
            {
                result = src.ToArray();
            }
            else if (bits == 16)
            {
                result = new byte[n * 2];
                for (int i = 0; i < n; i++)
                {
                    BinaryPrimitives.WriteUInt16LittleEndian(result.AsSpan(i * 2), unchecked((ushort)BinaryPrimitives.ReadUInt32LittleEndian(src.Slice(i * 4, 4))));
                }
            }
            else if (bits == 6)
            {
                // Four six-bit elements per three bytes, as one 24-bit little-endian bitstream (onnx.proto, TensorProto.raw_data).
                result = new byte[(n * 6L + 7) / 8];
                for (int i = 0; i < n; i++)
                {
                    uint e = BinaryPrimitives.ReadUInt32LittleEndian(src.Slice(i * 4, 4)) & 0x3F;
                    int bit = i * 6;
                    result[bit >> 3] |= (byte)(e << (bit & 7));
                    if ((bit & 7) > 2)
                    {
                        result[(bit >> 3) + 1] |= (byte)(e >> (8 - (bit & 7)));
                    }
                }
            }
            else
            {
                // 8-bit types and the packed 4- and 2-bit types: one byte per entry. Bool is normalized to 0 or 1.
                result = new byte[n];
                for (int i = 0; i < n; i++)
                {
                    uint e = BinaryPrimitives.ReadUInt32LittleEndian(src.Slice(i * 4, 4));
                    result[i] = dataType == 9 ? (byte)(e != 0 ? 1 : 0) : unchecked((byte)e);
                }
            }
        }

        if (result.Length != expected)
        {
            throw ctx.Fail(OnnxErrorKind.InvalidModel,
                count == 0 ? $"{typeName} tensor '{name}' has no elements but carries data."
                           : $"{typeName} tensor '{name}' has {result.Length} bytes of data; its shape ({count} elements) needs {expected}.");
        }

        return result;
    }

    // ---- functions ------------------------------------------------------------------------------------------------------------

    public static IrFunction ParseFunction(ref ProtoReader r)
    {
        ParseContext ctx = r.Context;
        var f = new IrFunction();
        var inputs = ImmutableArray.CreateBuilder<string>();
        var outputs = ImmutableArray.CreateBuilder<string>();
        var attrNames = ImmutableArray.CreateBuilder<string>();
        var attrDefaults = ImmutableArray.CreateBuilder<IrAttr>();
        var nodes = ImmutableArray.CreateBuilder<IrNode>();
        var opsets = ImmutableArray.CreateBuilder<OpsetImport>();
        while (r.TryReadTag(out int field, out WireType wt))
        {
            switch (field)
            {
                case 1 when wt == WireType.Length: f.Name = r.ReadString(); break;
                case 4 when wt == WireType.Length: inputs.Add(r.ReadString()); break;
                case 5 when wt == WireType.Length: outputs.Add(r.ReadString()); break;
                case 6 when wt == WireType.Length: attrNames.Add(r.ReadString()); break;
                case 8 when wt == WireType.Length: f.DocString = r.ReadString(); break;
                case 10 when wt == WireType.Length: f.Domain = r.ReadString(); break;
                case 13 when wt == WireType.Length: f.Overload = r.ReadString(); break;
                case 7 when wt == WireType.Length:
                    ctx.Push("node", nodes.Count);
                    {
                        ProtoReader m = r.ReadMessage();
                        nodes.Add(ParseNode(ref m));
                    }

                    ctx.Pop();
                    break;
                case 9 when wt == WireType.Length:
                    {
                        ProtoReader m = r.ReadMessage();
                        opsets.Add(ParseOpset(ref m));
                        break;
                    }

                case 11 when wt == WireType.Length:
                    ctx.Push("attribute_proto", attrDefaults.Count);
                    {
                        ProtoReader m = r.ReadMessage();
                        attrDefaults.Add(ParseAttribute(ref m));
                    }

                    ctx.Pop();
                    break;
                default: r.Skip(wt); break;
            }
        }

        f.Inputs = inputs.ToImmutable();
        f.Outputs = outputs.ToImmutable();
        f.AttributeNames = attrNames.ToImmutable();
        f.AttributeDefaults = attrDefaults.ToImmutable();
        f.Nodes = nodes.ToImmutable();
        f.OpsetImports = opsets.ToImmutable();
        return f;
    }
}
