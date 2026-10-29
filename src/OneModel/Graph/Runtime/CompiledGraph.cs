using System;
using System.Collections.Generic;
using System.Threading;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Runtime;

/// <summary>One node after operator lookup: its kernel and the value ids it reads and writes (-1 for an omitted optional slot).</summary>
internal sealed class CompiledNode
{
    public CompiledNode(NodeInfo info, OpKernel kernel, int[] inputs, int[] outputs)
    {
        Info = info;
        Kernel = kernel;
        Inputs = inputs;
        Outputs = outputs;
    }

    public NodeInfo Info { get; }

    public OpKernel Kernel { get; }

    public int[] Inputs { get; }

    public int[] Outputs { get; }
}

/// <summary>
/// A graph with its operators resolved and its weights loaded: everything about a model that does not depend on input shapes.
/// Immutable after the <c>Create</c> factory; an <see cref="ExecutionPlan"/> adds the shape-dependent part.
/// </summary>
internal sealed class CompiledGraph : IDisposable
{
    private CompiledGraph()
    {
    }

    public int ValueCount { get; private set; }

    public string[] ValueNames { get; private set; } = [];

    /// <summary>Per value id: its weights when it is an initializer, else an empty tensor.</summary>
    public Tensor[] Constants { get; private set; } = [];

    /// <summary>Value ids of the inputs a caller must supply, in graph order (initializer-backed inputs excluded).</summary>
    public int[] InputIds { get; private set; } = [];

    public int[] OutputIds { get; private set; } = [];

    public TensorInfo[] Inputs { get; private set; } = [];

    public TensorInfo[] Outputs { get; private set; } = [];

    /// <summary>The nodes that contribute to an output, in an order where every node follows the producers of its inputs.</summary>
    public CompiledNode[] Nodes { get; private set; } = [];

    public Dictionary<string, int> InputIndex { get; } = [];

    public Dictionary<string, int> OutputIndex { get; } = [];

    /// <summary>
    /// Resolves operators, checks the mode, and loads initializers.
    /// </summary>
    /// <exception cref="SessionException">An operator is missing, or exact mode lacks a kernel, or the model is malformed.</exception>
    public static CompiledGraph Create(IrGraph graph, OpRegistry registry, ExecutionMode mode, CancellationToken ct) =>
        Create(graph, registry, mode, WeightStorage.Original, ct);

    /// <summary>As above, also storing large float32 weights as <paramref name="storage"/> says.</summary>
    public static CompiledGraph Create(IrGraph graph, OpRegistry registry, ExecutionMode mode, WeightStorage storage, CancellationToken ct)
    {
        var g = new CompiledGraph();
        try
        {
            g.Build(graph, registry, mode, storage, ct);
            return g;
        }
        catch
        {
            g.Dispose();
            throw;
        }
    }

    private readonly List<OpKernel> _kernels = [];

    private void Build(IrGraph graph, OpRegistry registry, ExecutionMode mode, WeightStorage storage, CancellationToken ct)
    {
        var ids = new Dictionary<IrValue, int>(ReferenceEqualityComparer.Instance);
        ValueNames = new string[graph.Values.Length];
        for (int i = 0; i < graph.Values.Length; i++)
        {
            ids[graph.Values[i]] = i;
            ValueNames[i] = graph.Values[i].Name;
        }

        ValueCount = graph.Values.Length;
        Constants = new Tensor[ValueCount];

        // Which nodes matter: those from which a graph output can be reached.
        var needed = new bool[graph.Nodes.Length];
        var nodeIndex = new Dictionary<IrNode, int>(ReferenceEqualityComparer.Instance);
        for (int i = 0; i < graph.Nodes.Length; i++)
        {
            nodeIndex[graph.Nodes[i]] = i;
        }

        var stack = new Stack<IrValue>();
        foreach (IrValue o in graph.Outputs)
        {
            stack.Push(o);
        }

        var seen = new HashSet<IrValue>(ReferenceEqualityComparer.Instance);
        while (stack.Count > 0)
        {
            IrValue v = stack.Pop();
            if (!seen.Add(v) || v.Producer is not { } p || !nodeIndex.TryGetValue(p, out int ni))
            {
                continue;
            }

            needed[ni] = true;
            foreach (IrValue? inp in p.Inputs)
            {
                if (inp is not null)
                {
                    stack.Push(inp);
                }
            }
        }

        // Operators: collect every problem, then report them together.
        var missing = new List<string>();
        var notExact = new List<string>();
        var nodes = new List<CompiledNode>();
        for (int i = 0; i < graph.Nodes.Length; i++)
        {
            ct.ThrowIfCancellationRequested();
            if (!needed[i])
            {
                continue;
            }

            IrNode n = graph.Nodes[i];
            if (!registry.TryFind(n.Domain, n.OpType, n.OpsetVersion, out OpDef? def))
            {
                string label = $"{(n.IsDefaultDomain ? "" : n.Domain + ".")}{n.OpType} (opset {n.OpsetVersion})";
                if (!missing.Contains(label))
                {
                    missing.Add(label);
                }

                continue;
            }

            var info = new NodeInfo(n, n.OpsetVersion);
            OpKernel kernel;
            try
            {
                kernel = def.Create(info);
            }
            catch (Exception e) when (e is InvalidOperationException or ArgumentException or NotSupportedException or OverflowException)
            {
                // An attribute of the wrong kind or an impossible value in the model, not a bug in the kernel.
                throw new SessionException(SessionErrorKind.InvalidModel, $"{info.Describe()}: {e.Message}", n.Name, default, e);
            }

            _kernels.Add(kernel);
            if (mode == ExecutionMode.Exact && !kernel.SupportsExact && !notExact.Contains(n.OpType))
            {
                notExact.Add(n.OpType);
            }

            nodes.Add(new CompiledNode(info, kernel, IdsOf(n.Inputs, ids), IdsOf(n.Outputs, ids)));
        }

        if (missing.Count > 0)
        {
            throw new SessionException(SessionErrorKind.UnsupportedOperator,
                $"The model uses operators this library does not implement: {string.Join(", ", missing)}.", null, [.. missing]);
        }

        if (notExact.Count > 0)
        {
            throw new SessionException(SessionErrorKind.ExactModeUnavailable,
                $"Exact mode is not available: no exact kernel exists for {string.Join(", ", notExact)}. Use ExecutionMode.Fast or remove these operators.",
                null, [.. notExact]);
        }

        Nodes = [.. nodes];

        // Inputs and outputs.
        var inputIds = new List<int>();
        var inputInfos = new List<TensorInfo>();
        foreach (IrValue v in graph.Inputs)
        {
            if (v.Initializer is not null)
            {
                continue;
            }

            inputIds.Add(ids[v]);
            inputInfos.Add(DescribeValue(v, "input"));
            InputIndex[v.Name] = inputIds.Count - 1;
        }

        InputIds = [.. inputIds];
        Inputs = [.. inputInfos];
        var outputIds = new List<int>();
        var outputInfos = new List<TensorInfo>();
        foreach (IrValue v in graph.Outputs)
        {
            outputIds.Add(ids[v]);
            outputInfos.Add(DescribeValue(v, "output"));
            OutputIndex[v.Name] = outputIds.Count - 1;
        }

        OutputIds = [.. outputIds];
        Outputs = [.. outputInfos];

        // Weights: only those a kept node or an output reads.
        var used = new HashSet<int>();
        foreach (CompiledNode n in Nodes)
        {
            foreach (int id in n.Inputs)
            {
                if (id >= 0)
                {
                    used.Add(id);
                }
            }
        }

        foreach (int id in OutputIds)
        {
            used.Add(id);
        }

        foreach (int id in used)
        {
            ct.ThrowIfCancellationRequested();
            IrValue v = graph.Values[id];
            if (v.SparseInitializer is not null)
            {
                throw new SessionException(SessionErrorKind.Unsupported, $"'{v.Name}' is a sparse initializer; sparse weights are not supported yet.");
            }

            if (v.Initializer is { } init)
            {
                if (!init.TryGetDType(out _))
                {
                    throw new SessionException(SessionErrorKind.Unsupported,
                        $"Initializer '{v.Name}' has element type {DTypes.OnnxName(init.DataType)}, which this library cannot store.");
                }

                Constants[id] = init.ToTensor();
            }
        }

        UseHalfPrecision(graph, storage, ct);
    }

    // Half-precision support: weights that the session was asked to keep compact are rounded now, and when the model has half-precision
    // values anywhere (or weights were compacted) every kernel that can be is wrapped in the adapter that widens its half-precision inputs.
    private void UseHalfPrecision(IrGraph graph, WeightStorage storage, CancellationToken ct)
    {
        var keep = new bool[Nodes.Length][];
        bool anyCompact = false;
        if (storage != WeightStorage.Original)
        {
            DType target = storage == WeightStorage.Binary16 ? DType.F16 : DType.BF16;
            var outputs = new HashSet<int>(OutputIds);
            for (int id = 0; id < ValueCount; id++)
            {
                ct.ThrowIfCancellationRequested();
                Tensor w = Constants[id];
                if (w.IsEmpty || w.DType != DType.F32 || w.Shape.Length < CompactMinElements || outputs.Contains(id))
                {
                    continue;
                }

                // Only when every reader is a matrix product or convolution reading it as its weight (input 1).
                bool ok = true;
                bool read = false;
                foreach (CompiledNode n in Nodes)
                {
                    for (int i = 0; i < n.Inputs.Length; i++)
                    {
                        if (n.Inputs[i] == id)
                        {
                            read = true;
                            ok &= i == 1 && n.Info.OpType is "MatMul" or "Gemm" or "Conv" && n.Info.Ir.IsDefaultDomain;
                        }
                    }
                }

                if (!ok || !read)
                {
                    continue;
                }

                // A weight that float16 cannot hold stays as it is.
                if (target == DType.F16 && !FitsBinary16(w.Data<float>()))
                {
                    continue;
                }

                Tensor compact = Tensor.Create(target, w.Shape);
                HalfConvert.Narrow(w, compact);
                w.Dispose();
                Constants[id] = compact;
                for (int ni = 0; ni < Nodes.Length; ni++)
                {
                    if (Nodes[ni].Inputs.Length > 1 && Nodes[ni].Inputs[1] == id)
                    {
                        (keep[ni] ??= new bool[Nodes[ni].Inputs.Length])[1] = true;
                        anyCompact = true;
                    }
                }
            }
        }

        if (!anyCompact && !HasHalfPrecision(graph))
        {
            return;
        }

        for (int ni = 0; ni < Nodes.Length; ni++)
        {
            CompiledNode n = Nodes[ni];
            if (HalfAdapterKernel.CanWrap(n.Info.OpType, n.Kernel))
            {
                Nodes[ni] = new CompiledNode(n.Info, HalfAdapterKernel.Wrap(n.Info.OpType, n.Kernel, keep[ni] ?? []), n.Inputs, n.Outputs);
            }
        }
    }

    private const int CompactMinElements = 1024;

    private static bool FitsBinary16(ReadOnlySpan<float> values)
    {
        foreach (float v in values)
        {
            if (!(MathF.Abs(v) < 65504f))
            {
                return false;
            }
        }

        return true;
    }

    private static bool HasHalfPrecision(IrGraph graph)
    {
        static bool Half(int onnxType) => onnxType is (int)DType.F16 or (int)DType.BF16;
        foreach (IrValue v in graph.Values)
        {
            if ((v.Type is { } t && Half(t.ElemType)) || (v.Initializer is { } init && Half((int)init.DataType)))
            {
                return true;
            }
        }

        foreach (IrNode n in graph.Nodes)
        {
            if (n.OpType is "Cast" or "CastLike" && n.FindAttribute("to") is { Type: IrAttributeType.Whole } to && Half((int)to.Whole))
            {
                return true;
            }

            if (n.OpType == "Constant" && n.FindAttribute("value") is { Type: IrAttributeType.Tensor } v && Half((int)v.Tensor.DataType))
            {
                return true;
            }
        }

        return false;
    }

    private static int[] IdsOf(System.Collections.Immutable.ImmutableArray<IrValue?> values, Dictionary<IrValue, int> ids)
    {
        var r = new int[values.Length];
        for (int i = 0; i < r.Length; i++)
        {
            r[i] = values[i] is { } v ? ids[v] : -1;
        }

        return r;
    }

    private static TensorInfo DescribeValue(IrValue v, string what)
    {
        IrType? t = v.Type;
        if (t is null)
        {
            throw new SessionException(SessionErrorKind.InvalidModel, $"Graph {what} '{v.Name}' declares no type.");
        }

        if (t.Kind != IrTypeKind.Tensor)
        {
            throw new SessionException(SessionErrorKind.Unsupported, $"Graph {what} '{v.Name}' is a {t.Kind}; only tensors are supported as inputs and outputs.");
        }

        if (!DTypes.TryFromOnnx(t.ElemType, out DType dt))
        {
            throw new SessionException(SessionErrorKind.Unsupported,
                $"Graph {what} '{v.Name}' has element type {DTypes.OnnxName(t.ElemType)}, which this library cannot store.");
        }

        return new TensorInfo(v.Name, dt, t.HasShape ? t.Dims : [], t.HasShape);
    }

    private int _refs = 1;

    /// <summary>Another owner (a plan) keeps this graph alive until it calls <see cref="Dispose"/>.</summary>
    public void AddRef() => Interlocked.Increment(ref _refs);

    /// <summary>Drops one owner; the weights and kernels are released when the last one is gone.</summary>
    public void Dispose()
    {
        if (Interlocked.Decrement(ref _refs) == 0)
        {
            DisposeCore();
        }
    }

    private void DisposeCore()
    {
        foreach (OpKernel k in _kernels)
        {
            k.Dispose();
        }

        _kernels.Clear();
        foreach (Tensor c in Constants)
        {
            if (!c.IsEmpty)
            {
                c.Dispose();
            }
        }

        Constants = [];
    }
}
