using System.Globalization;
using System.Text;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Conformance;

/// <summary>The outcome class of one conformance case.</summary>
public enum CaseStatus
{
    /// <summary>Every output matched at ONNX's tolerances.</summary>
    Pass,

    /// <summary>The case ran (or tried to) and the result or an exception is wrong. This is a defect.</summary>
    Fail,

    /// <summary>An operator, or an opset version or attribute combination of one, has no kernel.</summary>
    UnsupportedOp,

    /// <summary>An element type or value kind (sequence, optional, map, string, 4-bit, 8-bit float, complex) is not implemented.</summary>
    UnsupportedType,

    /// <summary>Exact mode only: the case uses an operator without an exact kernel.</summary>
    NotExact,

    /// <summary>A failing case whose difference is understood and documented (see <see cref="ConformanceRunner.KnownDifferences"/>). Not a defect.</summary>
    KnownDiff,
}

/// <summary>One case's result in one mode.</summary>
/// <param name="Group">The data directory the case came from: node, simple, pytorch-converted or pytorch-operator.</param>
/// <param name="Name">The case directory name.</param>
/// <param name="Opset">The model's default-domain opset, or 0 when the model could not be read.</param>
/// <param name="Ops">The distinct operator types the model's graph and subgraphs use (outside the default domain written <c>domain.Op</c>), separated by commas.</param>
/// <param name="Mode">The session mode.</param>
/// <param name="Status">The outcome.</param>
/// <param name="Reason">Empty for a pass; otherwise one line saying what was wrong.</param>
public readonly record struct CaseResult(string Group, string Name, long Opset, string Ops, ExecutionMode Mode, CaseStatus Status, string Reason);

/// <summary>
/// Runs the frozen onnx/backend/test cases (node, simple, pytorch-converted, pytorch-operator) through a <see cref="Session"/> and
/// classifies each one. The data is fetched by tools/fetch_backend_test.py and never committed. Tolerances are ONNX's backend-test
/// defaults for floating point (rtol 1e-3, atol 1e-7, NaN equal to NaN); integers, booleans and strings must match exactly.
/// </summary>
public static class ConformanceRunner
{
    /// <summary>The data directories that hold cases, in report order.</summary>
    public static readonly string[] Groups = ["node", "simple", "pytorch-converted", "pytorch-operator"];

    private const double Rtol = 1e-3;
    private const double Atol = 1e-7;

    /// <summary>Lists the case directories under a backend-test data root, sorted by group then name.</summary>
    /// <param name="root">The directory that holds node/, simple/, ... or a single group directory.</param>
    public static List<(string Group, string Dir)> FindCases(string root)
    {
        var list = new List<(string, string)>();
        foreach (string g in Groups)
        {
            string gd = Path.Combine(root, g);
            if (!Directory.Exists(gd))
            {
                continue;
            }

            foreach (string d in Directory.GetDirectories(gd).Order(StringComparer.Ordinal))
            {
                if (File.Exists(Path.Combine(d, "model.onnx")))
                {
                    list.Add((g, d));
                }
            }
        }

        return list;
    }

    /// <summary>Runs one case in one mode. Never throws for a case's own faults.</summary>
    public static CaseResult RunCase(string group, string dir, ExecutionMode mode, int threads = 1)
    {
        CaseResult r = RunCaseCore(group, dir, mode, threads);
        if (r.Status == CaseStatus.Fail && KnownDifferences.TryGetValue(r.Name, out string? why))
        {
            return r with { Status = CaseStatus.KnownDiff, Reason = why };
        }

        return r;
    }

    /// <summary>
    /// Cases that fail on purpose, each with the reason. Everything else that fails is a defect.
    /// </summary>
    public static readonly IReadOnlyDictionary<string, string> KnownDifferences = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["test_resize_downsample_scales_linear_align_corners"] = "Resize align_corners follows ONNX Runtime where the two backend cases disagree (docs/ops-conv.md)",
        ["test_resize_downsample_scales_cubic_align_corners"] = "Resize align_corners follows ONNX Runtime where the two backend cases disagree (docs/ops-conv.md)",
        ["test_dft_inverse"] = "expected data carries 1.9e-7 float32 FFT noise, above the case's 1e-7 absolute tolerance",
        ["test_dft_inverse_opset19"] = "expected data carries 1.9e-7 float32 FFT noise, above the case's 1e-7 absolute tolerance",
    };

    private static CaseResult RunCaseCore(string group, string dir, ExecutionMode mode, int threads)
    {
        string name = Path.GetFileName(dir);
        long opset = 0;
        string ops = "";
        IrModel? model = null;
        try
        {
            try
            {
                model = OnnxImporter.Load(Path.Combine(dir, "model.onnx"));
            }
            catch (OnnxImportException e) when (e.Kind == OnnxErrorKind.Unsupported)
            {
                return new(group, name, 0, "", mode, CaseStatus.UnsupportedType, OneLine(e.Message));
            }

            opset = model.DefaultOpset;
            ops = string.Join(',', OpsOf(model.Graph).Order(StringComparer.Ordinal));
            string? typeGap = TypeGap(model.Graph);
            Session session;
            try
            {
                session = Session.Create(model, new SessionOptions { Mode = mode, IntraOpThreads = threads });
            }
            catch (SessionException e) when (Classify(e, typeGap) is { } status)
            {
                return new(group, name, opset, ops, mode, status, typeGap ?? OneLine(e.Message));
            }

            using (session)
            {
                if (typeGap is not null)
                {
                    return new(group, name, opset, ops, mode, CaseStatus.UnsupportedType, typeGap);
                }

                try
                {
                    (CaseStatus status, string reason) = RunSets(session, dir);
                    return new(group, name, opset, ops, mode, status, reason);
                }
                catch (SessionException e) when (Classify(e, typeGap) is { } status)
                {
                    // Plans are made per input shape, so some operator limits only show on the first run.
                    return new(group, name, opset, ops, mode, status, typeGap ?? OneLine(e.Message));
                }
            }
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            return new(group, name, opset, ops, mode, CaseStatus.Fail, OneLine(e.GetType().Name + ": " + e.Message));
        }
        finally
        {
            model?.Dispose();
        }
    }

    /// <summary>Writes results as tab-separated lines: group, case, opset, ops, mode, status, reason.</summary>
    public static string ToTsv(IEnumerable<CaseResult> results)
    {
        var sb = new StringBuilder();
        foreach (CaseResult r in results)
        {
            sb.Append(r.Group).Append('\t').Append(r.Name).Append('\t').Append(r.Opset.ToString(CultureInfo.InvariantCulture)).Append('\t')
              .Append(r.Ops).Append('\t').Append(r.Mode == ExecutionMode.Exact ? "exact" : "fast").Append('\t')
              .Append(StatusName(r.Status)).Append('\t').Append(r.Reason).Append('\n');
        }

        return sb.ToString();
    }

    /// <summary>The status as written in reports.</summary>
    public static string StatusName(CaseStatus s) => s switch
    {
        CaseStatus.Pass => "pass",
        CaseStatus.Fail => "fail",
        CaseStatus.UnsupportedOp => "unsupported-op",
        CaseStatus.UnsupportedType => "unsupported-type",
        CaseStatus.KnownDiff => "known-diff",
        _ => "not-exact",
    };

    // Operators say "not supported" when they reject an element type, an attribute value or a rank they do not handle yet; those are
    // gaps. Any other session error on a valid case is a defect and stays a failure.
    private static CaseStatus? Classify(SessionException e, string? typeGap)
    {
        if (e.Kind == SessionErrorKind.UnsupportedOperator)
        {
            return CaseStatus.UnsupportedOp;
        }

        if (e.Kind == SessionErrorKind.ExactModeUnavailable)
        {
            return CaseStatus.NotExact;
        }

        if (typeGap is not null)
        {
            return CaseStatus.UnsupportedType;
        }

        bool says = e.Message.Contains("not supported", StringComparison.Ordinal);
        if (e.Kind == SessionErrorKind.Unsupported || (e.Kind == SessionErrorKind.InvalidModel && says))
        {
            return e.Message.Contains("element type", StringComparison.Ordinal) ? CaseStatus.UnsupportedType : CaseStatus.UnsupportedOp;
        }

        return null;
    }

    private static string OneLine(string s)
    {
        s = s.ReplaceLineEndings(" ").Replace('\t', ' ');
        return s.Length > 300 ? s[..300] : s;
    }

    private static HashSet<string> OpsOf(IrGraph g)
    {
        var set = new HashSet<string>(StringComparer.Ordinal);
        Collect(g, set);
        return set;

        static void Collect(IrGraph g, HashSet<string> set)
        {
            foreach (IrNode n in g.Nodes)
            {
                set.Add(n.IsDefaultDomain ? n.OpType : n.Domain + "." + n.OpType);

                foreach (IrAttr a in n.Attributes)
                {
                    if (a.Type == IrAttributeType.Graph)
                    {
                        Collect(a.Graph, set);
                    }
                    else if (a.Type == IrAttributeType.Graphs)
                    {
                        foreach (IrGraph sg in a.Graphs)
                        {
                            Collect(sg, set);
                        }
                    }
                }
            }
        }
    }

    // Element types the tensor layer has: everything else is a gap of type, however the failure would otherwise surface.
    private static readonly HashSet<int> KnownElem = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16];

    private static string? TypeGap(IrGraph g)
    {
        foreach (IrValue v in g.Inputs.Concat(g.Outputs))
        {
            if (v.Type is not { } t)
            {
                continue;
            }

            if (!TypeKnown(t, out string what))
            {
                return $"value '{v.Name}' {what}";
            }
        }

        return null;
    }

    // Tensors of known element types, and sequences and optionals of them.
    private static bool TypeKnown(IrType t, out string what)
    {
        what = "";
        switch (t.Kind)
        {
            case IrTypeKind.Tensor:
                if (!KnownElem.Contains(t.ElemType))
                {
                    what = $"has element type {ElemName(t.ElemType)}";
                    return false;
                }

                return true;
            case IrTypeKind.Sequence or IrTypeKind.Optional when t.Inner is { } inner && inner.Kind != IrTypeKind.Optional:
                return TypeKnown(inner, out what);
            default:
                what = $"is a {t.Kind.ToString().ToLowerInvariant()}";
                return false;
        }
    }

    private static string ElemName(int code) => code switch
    {
        8 => "string",
        14 => "complex64",
        15 => "complex128",
        17 => "float8e4m3fn",
        18 => "float8e4m3fnuz",
        19 => "float8e5m2",
        20 => "float8e5m2fnuz",
        21 => "uint4",
        22 => "int4",
        23 => "float4e2m1",
        24 => "float8e8m0",
        25 => "uint2",
        26 => "int2",
        _ => code.ToString(CultureInfo.InvariantCulture),
    };

    private static (CaseStatus, string) RunSets(Session session, string dir)
    {
        int sets = 0;
        foreach (string setDir in Directory.GetDirectories(dir, "test_data_set_*").Order(StringComparer.Ordinal))
        {
            sets++;
            var inputs = new Tensor[session.Inputs.Length];
            var names = new string[inputs.Length];
            var outNames = new string[session.Outputs.Length];
            var outs = new Tensor[outNames.Length];
            try
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    names[i] = session.Inputs[i].Name;
                    inputs[i] = LoadValue(File.ReadAllBytes(Path.Combine(setDir, $"input_{i}.pb")), session.Inputs[i].Kind);
                }

                for (int i = 0; i < outNames.Length; i++)
                {
                    outNames[i] = session.Outputs[i].Name;
                }

                session.Run(names, inputs, outNames, outs);
                for (int i = 0; i < outs.Length; i++)
                {
                    using Tensor expected = LoadValue(File.ReadAllBytes(Path.Combine(setDir, $"output_{i}.pb")), session.Outputs[i].Kind);
                    if (!SameValue(outs[i], expected, out string why))
                    {
                        return (CaseStatus.Fail, OneLine($"output {i} ({outNames[i]}): {why}"));
                    }
                }
            }
            finally
            {
                foreach (Tensor t in inputs)
                {
                    if (!t.IsEmpty)
                    {
                        t.Dispose();
                    }
                }

                foreach (Tensor t in outs)
                {
                    if (!t.IsEmpty)
                    {
                        t.Dispose();
                    }
                }
            }
        }

        return sets == 0 ? (CaseStatus.Fail, "no test data sets") : (CaseStatus.Pass, "");
    }

    // A sequence or optional is compared by kind, then element by element.
    private static bool SameValue(Tensor got, Tensor want, out string why)
    {
        why = "";
        bool gp = PackedValue.IsPacked(got), wp = PackedValue.IsPacked(want);
        if (!gp && !wp)
        {
            return Same(got, want, out why);
        }

        if (gp != wp || PackedValue.Kind(got) != PackedValue.Kind(want))
        {
            why = "a sequence or optional of another kind";
            return false;
        }

        Tensor[] a = PackedValue.Items(got), b = PackedValue.Items(want);
        try
        {
            if (a.Length != b.Length)
            {
                why = $"sequence length {a.Length} vs {b.Length}";
                return false;
            }

            for (int i = 0; i < a.Length; i++)
            {
                if (!Same(a[i], b[i], out string w))
                {
                    why = $"element {i}: {w}";
                    return false;
                }
            }

            return true;
        }
        finally
        {
            foreach (Tensor t in a.Concat(b))
            {
                t.Dispose();
            }
        }
    }

    // ---- sequence and optional values in the test data (SequenceProto, OptionalProto) ----
    // Read here with a few lines of wire parsing, so the library's own reader stays tensor-only.

    private static Tensor LoadValue(byte[] data, TensorInfoKind kind)
    {
        switch (kind)
        {
            case TensorInfoKind.Sequence:
            {
                Tensor[] items = LoadSequence(data);
                try
                {
                    return PackedValue.Sequence(items);
                }
                finally
                {
                    foreach (Tensor t in items)
                    {
                        t.Dispose();
                    }
                }
            }

            case TensorInfoKind.Optional:
                return LoadOptional(data);
            default:
                return OnnxImporter.LoadTensor(data).ToTensor();
        }
    }

    private static Tensor[] LoadSequence(byte[] data)
    {
        var items = new List<Tensor>();
        try
        {
            foreach ((int field, ulong v, byte[]? bytes) in Fields(data))
            {
                if (field == 3 && bytes is not null)
                {
                    items.Add(OnnxImporter.LoadTensor(bytes).ToTensor());
                }
                else if (field == 2 && v != 1)
                {
                    throw new SessionException(SessionErrorKind.Unsupported, "a sequence of a non-tensor element type is not supported");
                }
                else if (field is 4 or 5 or 6 or 7)
                {
                    throw new SessionException(SessionErrorKind.Unsupported, "a sequence of a non-tensor element type is not supported");
                }
            }

            return [.. items];
        }
        catch
        {
            foreach (Tensor t in items)
            {
                t.Dispose();
            }

            throw;
        }
    }

    private static Tensor LoadOptional(byte[] data)
    {
        foreach ((int field, _, byte[]? bytes) in Fields(data))
        {
            if (field == 3 && bytes is not null)
            {
                using Tensor t = OnnxImporter.LoadTensor(bytes).ToTensor();
                return PackedValue.Optional(t);
            }

            if (field == 5 && bytes is not null)
            {
                Tensor[] items = LoadSequence(bytes);
                try
                {
                    return PackedValue.OptionalSequence(items);
                }
                finally
                {
                    foreach (Tensor t in items)
                    {
                        t.Dispose();
                    }
                }
            }

            if (field is 4 or 6 or 7)
            {
                throw new SessionException(SessionErrorKind.Unsupported, "an optional of this element type is not supported");
            }
        }

        return PackedValue.EmptyOptional();
    }

    private static IEnumerable<(int Field, ulong Varint, byte[]? Bytes)> Fields(byte[] data)
    {
        int at = 0;
        while (at < data.Length)
        {
            ulong tag = Varint(data, ref at);
            int field = (int)(tag >> 3);
            switch ((int)(tag & 7))
            {
                case 0:
                    yield return (field, Varint(data, ref at), null);
                    break;
                case 2:
                    int len = checked((int)Varint(data, ref at));
                    yield return (field, 0, data.AsSpan(at, len).ToArray());
                    at += len;
                    break;
                case 1:
                    at += 8;
                    break;
                case 5:
                    at += 4;
                    break;
                default:
                    throw new FormatException("unsupported wire type in a test value");
            }
        }
    }

    private static ulong Varint(byte[] d, ref int at)
    {
        ulong v = 0;
        for (int shift = 0; ; shift += 7)
        {
            byte b = d[at++];
            v |= (ulong)(b & 0x7f) << shift;
            if ((b & 0x80) == 0)
            {
                return v;
            }
        }
    }

    private static bool Same(Tensor got, Tensor want, out string why)
    {
        why = "";
        if (got.DType != want.DType)
        {
            why = $"type {got.DType} vs {want.DType}";
            return false;
        }

        if (got.Shape != want.Shape)
        {
            why = $"shape {got.Shape} vs {want.Shape}";
            return false;
        }

        switch (got.DType)
        {
            case DType.F32: return Close(got.ToArray<float>().Select(v => (double)v), want.ToArray<float>().Select(v => (double)v), out why);
            case DType.F64: return Close(got.ToArray<double>(), want.ToArray<double>(), out why);
            case DType.F16: return Close(got.ToArray<Half>().Select(v => (double)v), want.ToArray<Half>().Select(v => (double)v), out why);
            case DType.BF16: return Close(got.ToArray<BFloat16>().Select(v => (double)(float)v), want.ToArray<BFloat16>().Select(v => (double)(float)v), out why);
            case DType.I8: return Exactly(got.ToArray<sbyte>(), want.ToArray<sbyte>(), out why);
            case DType.U8: return Exactly(got.ToArray<byte>(), want.ToArray<byte>(), out why);
            case DType.I16: return Exactly(got.ToArray<short>(), want.ToArray<short>(), out why);
            case DType.U16: return Exactly(got.ToArray<ushort>(), want.ToArray<ushort>(), out why);
            case DType.I32: return Exactly(got.ToArray<int>(), want.ToArray<int>(), out why);
            case DType.U32: return Exactly(got.ToArray<uint>(), want.ToArray<uint>(), out why);
            case DType.I64: return Exactly(got.ToArray<long>(), want.ToArray<long>(), out why);
            case DType.U64: return Exactly(got.ToArray<ulong>(), want.ToArray<ulong>(), out why);
            case DType.Bool: return Exactly(got.ToArray<bool>(), want.ToArray<bool>(), out why);
            default:
                why = $"comparison of {got.DType} is not implemented";
                return false;
        }
    }

    private static bool Close(IEnumerable<double> gotSeq, IEnumerable<double> wantSeq, out string why)
    {
        why = "";
        double[] got = gotSeq.ToArray(), want = wantSeq.ToArray();
        for (int i = 0; i < got.Length; i++)
        {
            double g = got[i], w = want[i];
            if (double.IsNaN(w) ? double.IsNaN(g) : (g == w || Math.Abs(g - w) <= Atol + (Rtol * Math.Abs(w))))
            {
                continue;
            }

            why = $"element {i}: got {g:R}, want {w:R}";
            return false;
        }

        return true;
    }

    private static bool Exactly<T>(T[] got, T[] want, out string why) where T : unmanaged
    {
        why = "";
        for (int i = 0; i < got.Length; i++)
        {
            if (!EqualityComparer<T>.Default.Equals(got[i], want[i]))
            {
                why = $"element {i}: got {got[i]}, want {want[i]}";
                return false;
            }
        }

        return true;
    }
}
