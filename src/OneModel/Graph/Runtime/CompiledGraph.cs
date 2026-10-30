using System;
using System.Collections.Generic;
using System.Threading;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Graph.Passes;
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
    public static CompiledGraph Create(IrGraph graph, OpRegistry registry, ExecutionMode mode, WeightStorage storage, CancellationToken ct) =>
        Create(graph, registry, mode, storage, OptimizationLevel.None, KernelPath.Scalar, 1, ct);

    /// <summary>As above, then runs the graph passes the level and mode select (Graph/Passes) for a session on <paramref name="path"/>.</summary>
    public static CompiledGraph Create(IrGraph graph, OpRegistry registry, ExecutionMode mode, WeightStorage storage, OptimizationLevel level,
        KernelPath path, int threads, CancellationToken ct)
    {
        var g = new CompiledGraph();
        try
        {
            g.Build(graph, registry, mode, storage, level, path, threads, ct);
            return g;
        }
        catch
        {
            g.Dispose();
            throw;
        }
    }

    private readonly List<OpKernel> _kernels = [];

    private void Build(IrGraph graph, OpRegistry registry, ExecutionMode mode, WeightStorage storage, OptimizationLevel level, KernelPath path,
        int threads, CancellationToken ct)
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

        var compact = new bool[Nodes.Length][];
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

                Constants[id] = CompactWeight(id, init, storage, compact) ?? init.ToTensor();
            }
        }

        // Graph passes work on binary32 graphs. A model with half-precision values of its own keeps its graph as written. Weights kept
        // compact stay compact: the passes only pack and fold binary32 constants, and the compact weights are found again by value id
        // afterwards, because passes move and replace nodes.
        if (level != OptimizationLevel.None && !HasHalfPrecision(graph))
        {
            var compactIds = new HashSet<int>();
            for (int ni = 0; ni < Nodes.Length; ni++)
            {
                if (compact[ni] is not null)
                {
                    compactIds.Add(Nodes[ni].Inputs[1]);
                }
            }

            RunPasses(nodes, level, mode, path, threads, ct);
            compact = new bool[Nodes.Length][];
            for (int ni = 0; ni < Nodes.Length; ni++)
            {
                if (Nodes[ni].Inputs.Length > 1 && compactIds.Contains(Nodes[ni].Inputs[1]))
                {
                    compact[ni] = new bool[Nodes[ni].Inputs.Length];
                    compact[ni][1] = true;
                }
            }
        }

        UseHalfPrecision(graph, compact);
    }

    // Graph passes (Graph/Passes): they edit the node list, add values and constants, and may add kernels. Whatever happens, the
    // arrays this graph disposes from are rebuilt from what the pass graph holds, so nothing leaks if a pass throws.
    private void RunPasses(List<CompiledNode> nodes, OptimizationLevel level, ExecutionMode mode, KernelPath path, int threads, CancellationToken ct)
    {
        var pg = new PassGraph(nodes, [.. ValueNames], [.. Constants], InputIds, OutputIds, _kernels, mode, path, threads);
        try
        {
            GraphPasses.Run(pg, level, ct);
        }
        finally
        {
            ValueNames = [.. pg.Names];
            Constants = [.. pg.Constants];
            ValueCount = ValueNames.Length;
            Nodes = [.. nodes];
        }

        // Constants that no kept node or output reads any more (folded inputs, replaced weights) are released now.
        var used = new bool[ValueCount];
        foreach (CompiledNode n in Nodes)
        {
            foreach (int id in n.Inputs)
            {
                if (id >= 0)
                {
                    used[id] = true;
                }
            }
        }

        foreach (int id in OutputIds)
        {
            used[id] = true;
        }

        for (int i = 0; i < Constants.Length; i++)
        {
            if (!used[i] && !Constants[i].IsEmpty)
            {
                Constants[i].Dispose();
                Constants[i] = default;
            }
        }
    }

    // A float32 weight that the session was asked to keep compact is rounded straight from the model's bytes, so the binary32 copy never
    // exists. Returns null when the weight does not qualify: it must be float32 with enough values, read by nothing but MatMul, Gemm or
    // Conv as their weight (input 1), not be a graph output, and survive the rounding as finite values.
    private Tensor? CompactWeight(int id, TensorData init, WeightStorage storage, bool[][] compact)
    {
        if (storage == WeightStorage.Original || init.DataType != (int)DType.F32 || init.ElementCount < CompactMinElements ||
            init.RawData.Length != init.ElementCount * 4 || Array.IndexOf(OutputIds, id) >= 0)
        {
            return null;
        }

        bool read = false;
        foreach (CompiledNode n in Nodes)
        {
            for (int i = 0; i < n.Inputs.Length; i++)
            {
                if (n.Inputs[i] == id)
                {
                    read = true;
                    if (i != 1 || n.Info.OpType is not ("MatMul" or "Gemm" or "Conv") || !n.Info.Ir.IsDefaultDomain)
                    {
                        return null;
                    }
                }
            }
        }

        if (!read)
        {
            return null;
        }

        Tensor t = Tensor.Create(storage == WeightStorage.Binary16 ? DType.F16 : DType.BF16, Shape.Create(init.Dims.AsSpan()));
        if (!HalfConvert.NarrowBytes(init.RawData, t))
        {
            t.Dispose();
            return null;
        }

        for (int ni = 0; ni < Nodes.Length; ni++)
        {
            if (Nodes[ni].Inputs.Length > 1 && Nodes[ni].Inputs[1] == id)
            {
                (compact[ni] ??= new bool[Nodes[ni].Inputs.Length])[1] = true;
            }
        }

        return t;
    }

    // Half-precision support: when the model has half-precision values anywhere, or weights were kept compact, every kernel that can be is
    // wrapped in the adapter that widens its half-precision inputs.
    private void UseHalfPrecision(IrGraph graph, bool[][] compact)
    {
        bool anyCompact = Array.Exists(compact, c => c is not null);
        if (!anyCompact && !HasHalfPrecision(graph))
        {
            return;
        }

        for (int ni = 0; ni < Nodes.Length; ni++)
        {
            CompiledNode n = Nodes[ni];
            if (HalfAdapterKernel.CanWrap(n.Info.OpType, n.Kernel))
            {
                Nodes[ni] = new CompiledNode(n.Info, HalfAdapterKernel.Wrap(n.Info.OpType, n.Kernel, compact[ni] ?? []), n.Inputs, n.Outputs);
            }
        }
    }

    private const int CompactMinElements = 1024;

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
