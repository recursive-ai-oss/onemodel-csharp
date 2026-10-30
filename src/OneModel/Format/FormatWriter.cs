using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading;
using RecursiveAI.OneModel.Graph;

namespace RecursiveAI.OneModel.Format;

/// <summary>
/// Writes an <see cref="IrModel"/> in the OneModel format (docs/format.md): header, JSON manifest, 64-byte-aligned tensor bytes.
/// Two passes: the first walks the model and assigns every tensor's offset in the data region, so the manifest can be written
/// before the data; the second streams the bytes. The same model always produces the same bytes.
/// </summary>
internal static class FormatWriter
{
    public const int HeaderSize = 64;
    public const int Alignment = 64;
    public static readonly byte[] Magic = "ONEMODEL"u8.ToArray();
    private static readonly byte[] Zeros = new byte[Alignment];

    public static void Write(IrModel model, Stream output, CancellationToken ct)
    {
        // Pass 1: where does each tensor go?
        var tensors = new List<TensorData>();
        var offsets = new Dictionary<TensorData, long>(ReferenceEqualityComparer.Instance);
        long next = 0;
        void Place(TensorData t)
        {
            if (t.DataType == 8 || t.RawLength == 0 || offsets.ContainsKey(t))
            {
                return;
            }

            offsets[t] = next;
            tensors.Add(t);
            next += (t.RawLength + Alignment - 1) / Alignment * Alignment;
        }

        CollectGraph(model.Graph, Place);
        foreach (IrFunction f in model.Functions)
        {
            foreach (IrNode n in f.Nodes)
            {
                CollectAttributes(n.Attributes, Place);
            }

            CollectAttributes(f.AttributeDefaults, Place);
        }

        // The last tensor is not padded, so the file ends with its last byte.
        long dataLength = tensors.Count == 0 ? 0 : offsets[tensors[^1]] + tensors[^1].RawLength;

        var manifest = new ArrayBufferWriter<byte>(1 << 16);
        using (var w = new Utf8JsonWriter(manifest, new JsonWriterOptions { Indented = false }))
        {
            new ManifestWriter(w, offsets, ct).WriteModel(model);
        }

        long manifestLength = manifest.WrittenCount;
        long dataOffset = (HeaderSize + manifestLength + Alignment - 1) / Alignment * Alignment;

        byte[] header = new byte[HeaderSize];
        Magic.CopyTo(header, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(8), ModelFormat.Version);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(12), 0);
        BinaryPrimitives.WriteUInt64LittleEndian(header.AsSpan(16), HeaderSize);
        BinaryPrimitives.WriteUInt64LittleEndian(header.AsSpan(24), (ulong)manifestLength);
        BinaryPrimitives.WriteUInt64LittleEndian(header.AsSpan(32), (ulong)dataOffset);
        BinaryPrimitives.WriteUInt64LittleEndian(header.AsSpan(40), (ulong)dataLength);
        BinaryPrimitives.WriteUInt64LittleEndian(header.AsSpan(48), (ulong)(dataOffset + dataLength));
        output.Write(header, 0, HeaderSize);
        output.Write(manifest.WrittenSpan);
        WritePadding(output, dataOffset - HeaderSize - manifestLength);

        // Pass 2: the bytes, each tensor starting on a 64-byte boundary.
        long written = 0;
        foreach (TensorData t in tensors)
        {
            ct.ThrowIfCancellationRequested();
            WritePadding(output, offsets[t] - written);
            output.Write(t.RawData);
            written = offsets[t] + t.RawLength;
        }
    }

    private static void WritePadding(Stream output, long count)
    {
        while (count > 0)
        {
            int n = (int)Math.Min(count, Zeros.Length);
            output.Write(Zeros, 0, n);
            count -= n;
        }
    }

    private static void CollectGraph(IrGraph g, Action<TensorData> place)
    {
        foreach (TensorData t in g.Initializers)
        {
            place(t);
        }

        foreach (SparseTensorData s in g.SparseInitializers)
        {
            place(s.Values);
            place(s.Indices);
        }

        foreach (IrNode n in g.Nodes)
        {
            CollectAttributes(n.Attributes, place);
        }
    }

    private static void CollectAttributes(IEnumerable<IrAttr> attrs, Action<TensorData> place)
    {
        foreach (IrAttr a in attrs)
        {
            if (a.RefAttrName.Length > 0)
            {
                continue;
            }

            switch (a.Type)
            {
                case IrAttributeType.Tensor: place(a.Tensor); break;
                case IrAttributeType.Tensors:
                    foreach (TensorData t in a.Tensors)
                    {
                        place(t);
                    }

                    break;
                case IrAttributeType.SparseTensor:
                    place(a.SparseTensor.Values);
                    place(a.SparseTensor.Indices);
                    break;
                case IrAttributeType.SparseTensors:
                    foreach (SparseTensorData s in a.SparseTensors)
                    {
                        place(s.Values);
                        place(s.Indices);
                    }

                    break;
                case IrAttributeType.Graph: CollectGraph(a.Graph, place); break;
                case IrAttributeType.Graphs:
                    foreach (IrGraph g in a.Graphs)
                    {
                        CollectGraph(g, place);
                    }

                    break;
            }
        }
    }

    private sealed class ManifestWriter(Utf8JsonWriter w, Dictionary<TensorData, long> offsets, CancellationToken ct)
    {
        public void WriteModel(IrModel m)
        {
            w.WriteStartObject();
            w.WriteString("format", ModelFormat.Name);
            w.WriteNumber("version", ModelFormat.Version);
            w.WriteNumber("ir_version", m.IrVersion);
            w.WriteString("producer_name", m.ProducerName);
            w.WriteString("producer_version", m.ProducerVersion);
            w.WriteString("domain", m.Domain);
            w.WriteNumber("model_version", m.ModelVersion);
            Doc("doc", m.DocString);
            w.WriteStartArray("opsets");
            foreach (OpsetImport o in m.OpsetImports)
            {
                WriteOpset(o);
            }

            w.WriteEndArray();
            w.WriteStartArray("metadata");
            foreach (KeyValuePair<string, string> kv in m.Metadata)
            {
                w.WriteStartObject();
                w.WriteString("key", kv.Key);
                w.WriteString("value", kv.Value);
                w.WriteEndObject();
            }

            w.WriteEndArray();
            w.WritePropertyName("graph");
            WriteGraph(m.Graph);
            w.WriteStartArray("functions");
            foreach (IrFunction f in m.Functions)
            {
                WriteFunction(f);
            }

            w.WriteEndArray();
            w.WriteEndObject();
        }

        private void Doc(string name, string value)
        {
            if (value.Length > 0)
            {
                w.WriteString(name, value);
            }
        }

        private void WriteOpset(OpsetImport o)
        {
            w.WriteStartObject();
            w.WriteString("domain", o.Domain);
            w.WriteNumber("version", o.Version);
            w.WriteEndObject();
        }

        private void WriteFunction(IrFunction f)
        {
            w.WriteStartObject();
            w.WriteString("name", f.Name);
            w.WriteString("domain", f.Domain);
            w.WriteString("overload", f.Overload);
            Doc("doc", f.DocString);
            Strings("inputs", f.Inputs);
            Strings("outputs", f.Outputs);
            Strings("attribute_names", f.AttributeNames);
            w.WriteStartArray("attribute_defaults");
            foreach (IrAttr a in f.AttributeDefaults)
            {
                WriteAttribute(a);
            }

            w.WriteEndArray();
            w.WriteStartArray("nodes");
            foreach (IrNode n in f.Nodes)
            {
                WriteNode(n);
            }

            w.WriteEndArray();
            w.WriteStartArray("opsets");
            foreach (OpsetImport o in f.OpsetImports)
            {
                WriteOpset(o);
            }

            w.WriteEndArray();
            w.WriteEndObject();
        }

        private void Strings(string name, IEnumerable<string> values)
        {
            w.WriteStartArray(name);
            foreach (string s in values)
            {
                w.WriteStringValue(s);
            }

            w.WriteEndArray();
        }

        private void WriteGraph(IrGraph g)
        {
            ct.ThrowIfCancellationRequested();
            w.WriteStartObject();
            w.WriteString("name", g.Name);
            Doc("doc", g.DocString);
            WriteValueList("inputs", g.Inputs);
            WriteValueList("outputs", g.Outputs);

            // Types of every other value that has one; inputs and outputs carry theirs in their own lists.
            w.WriteStartArray("value_info");
            foreach (IrValue v in g.Values)
            {
                if (v.Type is not null && !v.IsGraphInput && !g.Outputs.Contains(v))
                {
                    WriteTypedName(v);
                }
            }

            w.WriteEndArray();
            w.WriteStartArray("initializers");
            foreach (TensorData t in g.Initializers)
            {
                WriteTensor(t);
            }

            w.WriteEndArray();
            w.WriteStartArray("sparse_initializers");
            foreach (SparseTensorData s in g.SparseInitializers)
            {
                WriteSparse(s);
            }

            w.WriteEndArray();
            w.WriteStartArray("nodes");
            foreach (IrNode n in g.Nodes)
            {
                WriteNode(n);
            }

            w.WriteEndArray();
            w.WriteEndObject();
        }

        private void WriteValueList(string name, IEnumerable<IrValue> values)
        {
            w.WriteStartArray(name);
            foreach (IrValue v in values)
            {
                WriteTypedName(v);
            }

            w.WriteEndArray();
        }

        private void WriteTypedName(IrValue v)
        {
            w.WriteStartObject();
            w.WriteString("name", v.Name);
            if (v.Type is not null)
            {
                w.WritePropertyName("type");
                WriteType(v.Type);
            }

            w.WriteEndObject();
        }

        private void WriteNode(IrNode n)
        {
            w.WriteStartObject();
            w.WriteString("name", n.Name);
            w.WriteString("op", n.OpType);
            w.WriteString("domain", n.Domain);
            w.WriteString("overload", n.Overload);
            Doc("doc", n.DocString);
            Strings("inputs", n.InputNames);
            Strings("outputs", n.OutputNames);
            w.WriteStartArray("attributes");
            foreach (IrAttr a in n.Attributes)
            {
                WriteAttribute(a);
            }

            w.WriteEndArray();
            w.WriteEndObject();
        }

        private void WriteAttribute(IrAttr a)
        {
            w.WriteStartObject();
            w.WriteString("name", a.Name);
            w.WriteNumber("type", (int)a.Type);
            Doc("doc", a.DocString);
            if (a.RefAttrName.Length > 0)
            {
                w.WriteString("ref", a.RefAttrName);
                w.WriteEndObject();
                return;
            }

            w.WritePropertyName("value");
            switch (a.Type)
            {
                case IrAttributeType.Real: w.WriteNumberValue(FloatBits(a.Real)); break;
                case IrAttributeType.Whole: w.WriteNumberValue(a.Whole); break;
                case IrAttributeType.Text: w.WriteBase64StringValue(a.Text.Span); break;
                case IrAttributeType.Tensor: WriteTensor(a.Tensor); break;
                case IrAttributeType.Graph: WriteGraph(a.Graph); break;
                case IrAttributeType.SparseTensor: WriteSparse(a.SparseTensor); break;
                case IrAttributeType.TypeProto: WriteType(a.TypeValue); break;
                case IrAttributeType.Reals:
                    w.WriteStartArray();
                    foreach (float f in a.Reals)
                    {
                        w.WriteNumberValue(FloatBits(f));
                    }

                    w.WriteEndArray();
                    break;
                case IrAttributeType.Wholes:
                    w.WriteStartArray();
                    foreach (long v in a.Wholes)
                    {
                        w.WriteNumberValue(v);
                    }

                    w.WriteEndArray();
                    break;
                case IrAttributeType.Texts:
                    w.WriteStartArray();
                    foreach (ReadOnlyMemory<byte> s in a.Texts)
                    {
                        w.WriteBase64StringValue(s.Span);
                    }

                    w.WriteEndArray();
                    break;
                case IrAttributeType.Tensors:
                    w.WriteStartArray();
                    foreach (TensorData t in a.Tensors)
                    {
                        WriteTensor(t);
                    }

                    w.WriteEndArray();
                    break;
                case IrAttributeType.Graphs:
                    w.WriteStartArray();
                    foreach (IrGraph g in a.Graphs)
                    {
                        WriteGraph(g);
                    }

                    w.WriteEndArray();
                    break;
                case IrAttributeType.SparseTensors:
                    w.WriteStartArray();
                    foreach (SparseTensorData s in a.SparseTensors)
                    {
                        WriteSparse(s);
                    }

                    w.WriteEndArray();
                    break;
                case IrAttributeType.TypeProtos:
                    w.WriteStartArray();
                    foreach (IrType t in a.TypeValues)
                    {
                        WriteType(t);
                    }

                    w.WriteEndArray();
                    break;
                default:
                    throw new ModelFormatException(ModelFormatErrorKind.InvalidModel, $"Attribute '{a.Name}' has no type, so it cannot be saved.");
            }

            w.WriteEndObject();
        }

        private static uint FloatBits(float f) => unchecked((uint)BitConverter.SingleToInt32Bits(f));

        private void WriteSparse(SparseTensorData s)
        {
            w.WriteStartObject();
            w.WritePropertyName("values");
            WriteTensor(s.Values);
            w.WritePropertyName("indices");
            WriteTensor(s.Indices);
            w.WriteStartArray("dims");
            foreach (long d in s.Dims)
            {
                w.WriteNumberValue(d);
            }

            w.WriteEndArray();
            w.WriteEndObject();
        }

        private void WriteTensor(TensorData t)
        {
            w.WriteStartObject();
            w.WriteString("name", t.Name);
            w.WriteNumber("dtype", t.DataType);
            w.WriteStartArray("dims");
            foreach (long d in t.Dims)
            {
                w.WriteNumberValue(d);
            }

            w.WriteEndArray();
            Doc("doc", t.DocString);
            if (t.DataType == 8)
            {
                w.WriteStartArray("strings");
                foreach (ReadOnlyMemory<byte> s in t.Strings)
                {
                    w.WriteBase64StringValue(s.Span);
                }

                w.WriteEndArray();
            }
            else
            {
                w.WriteStartObject("data");
                w.WriteNumber("offset", t.RawLength == 0 ? 0 : offsets[t]);
                w.WriteNumber("length", t.RawLength);
                w.WriteEndObject();
            }

            w.WriteEndObject();
        }

        private void WriteType(IrType t)
        {
            w.WriteStartObject();
            switch (t.Kind)
            {
                case IrTypeKind.Tensor:
                case IrTypeKind.SparseTensor:
                    w.WriteString("kind", t.Kind == IrTypeKind.Tensor ? "tensor" : "sparse_tensor");
                    w.WriteNumber("elem", t.ElemType);
                    if (t.HasShape)
                    {
                        w.WriteStartArray("shape");
                        foreach (IrDim d in t.Dims)
                        {
                            if (d.Param is not null)
                            {
                                w.WriteStringValue(d.Param);
                            }
                            else if (d.Value >= 0)
                            {
                                w.WriteNumberValue(d.Value);
                            }
                            else
                            {
                                w.WriteNullValue();
                            }
                        }

                        w.WriteEndArray();
                    }

                    break;
                case IrTypeKind.Sequence:
                case IrTypeKind.Optional:
                    w.WriteString("kind", t.Kind == IrTypeKind.Sequence ? "sequence" : "optional");
                    InnerType(t);
                    break;
                case IrTypeKind.Map:
                    w.WriteString("kind", "map");
                    w.WriteNumber("key", t.ElemType);
                    InnerType(t);
                    break;
                default:
                    w.WriteString("kind", "opaque");
                    w.WriteString("domain", t.OpaqueDomain ?? "");
                    w.WriteString("name", t.OpaqueName ?? "");
                    break;
            }

            w.WriteEndObject();
        }

        private void InnerType(IrType t)
        {
            if (t.Inner is not null)
            {
                w.WritePropertyName("inner");
                WriteType(t.Inner);
            }
        }
    }
}
