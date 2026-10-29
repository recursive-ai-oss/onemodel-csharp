using System.Buffers.Binary;
using System.Collections.Immutable;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using RecursiveAI.OneModel.Graph;

namespace RecursiveAI.OneModel.OnnxSweep;

/// <summary>
/// Writes a canonical text description of an imported model. tools/onnx_dump.py writes the same text from the onnx Python package;
/// the two files are compared with diff. Floats are written as their bit patterns so no formatting can differ.
/// </summary>
internal static class Dump
{
    public static string Model(IrModel m)
    {
        var sb = new StringBuilder();
        sb.Append(CultureInfo.InvariantCulture, $"model ir={m.IrVersion} producer={m.ProducerName}|{m.ProducerVersion} domain={m.Domain} version={m.ModelVersion}\n");
        foreach (OpsetImport o in m.OpsetImports)
        {
            sb.Append(CultureInfo.InvariantCulture, $"opset {o.Domain}={o.Version}\n");
        }

        foreach (KeyValuePair<string, string> kv in m.Metadata)
        {
            sb.Append(CultureInfo.InvariantCulture, $"meta {kv.Key}={kv.Value}\n");
        }

        Graph(sb, m.Graph, 0);
        foreach (IrFunction f in m.Functions)
        {
            sb.Append(CultureInfo.InvariantCulture, $"function {f.Domain}|{f.Name}|{f.Overload} in=[{string.Join(',', f.Inputs)}] out=[{string.Join(',', f.Outputs)}] attrs=[{string.Join(',', f.AttributeNames)}] nodes={f.Nodes.Length}\n");
            foreach (OpsetImport o in f.OpsetImports)
            {
                sb.Append(CultureInfo.InvariantCulture, $"  fopset {o.Domain}={o.Version}\n");
            }

            foreach (IrAttr a in f.AttributeDefaults)
            {
                Attr(sb, a, 2);
            }

            foreach (IrNode n in f.Nodes)
            {
                Node(sb, n, 2);
            }
        }

        return sb.ToString();
    }

    private static void Graph(StringBuilder sb, IrGraph g, int ind)
    {
        string p = new(' ', ind);
        sb.Append(CultureInfo.InvariantCulture, $"{p}graph name={g.Name}\n");
        foreach (IrValue v in g.Inputs)
        {
            sb.Append(CultureInfo.InvariantCulture, $"{p} input {v.Name} {Type(v.Type)}\n");
        }

        foreach (IrValue v in g.Outputs)
        {
            sb.Append(CultureInfo.InvariantCulture, $"{p} output {v.Name} {Type(v.Type)}\n");
        }

        var outputNames = new HashSet<string>(g.Outputs.Select(v => v.Name), StringComparer.Ordinal);
        foreach (IrValue v in g.Values.Where(v => v.Type is not null && !v.IsGraphInput && !outputNames.Contains(v.Name)).OrderBy(v => v.Name, StringComparer.Ordinal))
        {
            sb.Append(CultureInfo.InvariantCulture, $"{p} valueinfo {v.Name} {Type(v.Type)}\n");
        }

        foreach (TensorData t in g.Initializers)
        {
            sb.Append(CultureInfo.InvariantCulture, $"{p} init {Tensor(t)}\n");
        }

        foreach (SparseTensorData s in g.SparseInitializers)
        {
            sb.Append(CultureInfo.InvariantCulture, $"{p} sparse {Sparse(s)}\n");
        }

        foreach (IrNode n in g.Nodes)
        {
            Node(sb, n, ind + 1);
        }
    }

    private static void Node(StringBuilder sb, IrNode n, int ind)
    {
        string p = new(' ', ind);
        sb.Append(CultureInfo.InvariantCulture, $"{p}node op={n.OpType} domain={n.Domain} name={n.Name} overload={n.Overload} in=[{string.Join(',', n.InputNames)}] out=[{string.Join(',', n.OutputNames)}]\n");
        foreach (IrAttr a in n.Attributes)
        {
            Attr(sb, a, ind + 1);
        }
    }

    private static void Attr(StringBuilder sb, IrAttr a, int ind)
    {
        string p = new(' ', ind);
        string head = $"{p}attr {a.Name} type={(int)a.Type} ref={a.RefAttrName}";
        if (a.RefAttrName.Length > 0)
        {
            sb.Append(head).Append('\n');
            return;
        }

        switch (a.Type)
        {
            case IrAttributeType.Real: sb.Append(CultureInfo.InvariantCulture, $"{head} {F(a.Real)}\n"); break;
            case IrAttributeType.Whole: sb.Append(CultureInfo.InvariantCulture, $"{head} {a.Whole}\n"); break;
            case IrAttributeType.Text: sb.Append(CultureInfo.InvariantCulture, $"{head} {Convert.ToHexString(a.Text.Span)}\n"); break;
            case IrAttributeType.Tensor: sb.Append(CultureInfo.InvariantCulture, $"{head} {Tensor(a.Tensor)}\n"); break;
            case IrAttributeType.SparseTensor: sb.Append(CultureInfo.InvariantCulture, $"{head} {Sparse(a.SparseTensor)}\n"); break;
            case IrAttributeType.TypeProto: sb.Append(CultureInfo.InvariantCulture, $"{head} {Type(a.TypeValue)}\n"); break;
            case IrAttributeType.Reals: sb.Append(CultureInfo.InvariantCulture, $"{head} [{string.Join(',', a.Reals.Select(F))}]\n"); break;
            case IrAttributeType.Wholes: sb.Append(CultureInfo.InvariantCulture, $"{head} [{string.Join(',', a.Wholes)}]\n"); break;
            case IrAttributeType.Texts: sb.Append(CultureInfo.InvariantCulture, $"{head} [{string.Join(',', a.Texts.Select(s => Convert.ToHexString(s.Span)))}]\n"); break;
            case IrAttributeType.Tensors:
                sb.Append(head).Append('\n');
                foreach (TensorData t in a.Tensors)
                {
                    sb.Append(CultureInfo.InvariantCulture, $"{p} {Tensor(t)}\n");
                }

                break;
            case IrAttributeType.SparseTensors:
                sb.Append(head).Append('\n');
                foreach (SparseTensorData t in a.SparseTensors)
                {
                    sb.Append(CultureInfo.InvariantCulture, $"{p} {Sparse(t)}\n");
                }

                break;
            case IrAttributeType.TypeProtos: sb.Append(CultureInfo.InvariantCulture, $"{head} [{string.Join(';', a.TypeValues.Select(Type))}]\n"); break;
            case IrAttributeType.Graph:
                sb.Append(head).Append('\n');
                Graph(sb, a.Graph, ind + 1);
                break;
            case IrAttributeType.Graphs:
                sb.Append(head).Append('\n');
                foreach (IrGraph g in a.Graphs)
                {
                    Graph(sb, g, ind + 1);
                }

                break;
            default: sb.Append(head).Append('\n'); break;
        }
    }

    private static string F(float f) => BitConverter.SingleToUInt32Bits(f).ToString("x8", CultureInfo.InvariantCulture);

    private static string Type(IrType? t) => t is null ? "none" : t.ToString();

    private static string Sparse(SparseTensorData s) =>
        $"dims=[{string.Join(',', s.Dims)}] values={Tensor(s.Values)} indices={Tensor(s.Indices)}";

    /// <summary>Name, element type, dimensions, and the SHA-256 of the little-endian bytes (strings: each element length-prefixed with 4 bytes).</summary>
    public static string Tensor(TensorData t)
    {
        string hash;
        if (t.DataType == 8)
        {
            using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            Span<byte> len = stackalloc byte[4];
            foreach (ReadOnlyMemory<byte> s in t.Strings)
            {
                BinaryPrimitives.WriteUInt32LittleEndian(len, (uint)s.Length);
                h.AppendData(len);
                h.AppendData(s.Span);
            }

            hash = Convert.ToHexStringLower(h.GetHashAndReset());
        }
        else
        {
            hash = Convert.ToHexStringLower(SHA256.HashData(t.RawData));
        }

        return $"{t.Name} dtype={t.DataType} dims=[{string.Join(',', t.Dims)}] sha256={hash}";
    }
}
