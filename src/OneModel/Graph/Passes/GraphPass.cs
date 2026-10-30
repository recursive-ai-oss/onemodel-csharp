using System;
using System.Collections.Generic;
using System.Threading;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Passes;

/// <summary>
/// One rewrite of the compiled graph, run once when a session is created, before any plan exists. A pass sees nodes whose
/// operators are already resolved and weights that are already loaded, and may replace nodes, add constants and add kernels. It
/// never sees input shapes: whatever it does must hold for every shape (shape-dependent work stays in plans).
/// </summary>
internal interface IGraphPass
{
    /// <summary>A short name for diagnostics and docs/graph-passes.md.</summary>
    string Name { get; }

    /// <summary>The lowest <see cref="OptimizationLevel"/> that runs the pass.</summary>
    OptimizationLevel MinLevel { get; }

    /// <summary>
    /// True when results are bit-identical with and without the pass in both modes. A pass that returns false is skipped in
    /// <see cref="ExecutionMode.Exact"/> (it may still change fast-mode results in the last bits).
    /// </summary>
    bool PreservesBits { get; }

    /// <summary>Rewrites <paramref name="graph"/> in place.</summary>
    void Run(PassGraph graph, CancellationToken ct);
}

/// <summary>
/// The mutable graph a pass edits: nodes in an order where every node follows the producers of its inputs, value ids (indexes into
/// the names and constants), and the ids of the graph's outputs. Ownership: kernels added with <see cref="Own{T}"/> and constants
/// set with <see cref="SetConstant"/> or <see cref="AddValue"/> belong to the graph and are released with it.
/// </summary>
internal sealed class PassGraph
{
    private readonly List<string> _names;
    private readonly List<Tensor> _constants;
    private readonly List<OpKernel> _ownedKernels;

    internal PassGraph(List<CompiledNode> nodes, List<string> names, List<Tensor> constants, int[] inputIds, int[] outputIds,
        List<OpKernel> kernels, ExecutionMode mode, KernelPath path, int threads)
    {
        Nodes = nodes;
        _names = names;
        _constants = constants;
        InputIds = inputIds;
        OutputIds = outputIds;
        _ownedKernels = kernels;
        Mode = mode;
        Path = path;
        Threads = threads;
    }

    /// <summary>The nodes in dependency order; a pass edits the list directly and keeps that order.</summary>
    public List<CompiledNode> Nodes { get; }

    public int[] InputIds { get; }

    public int[] OutputIds { get; }

    public ExecutionMode Mode { get; }

    /// <summary>The instruction set the session runs on; passes that lay data out for a kernel (packing) need it.</summary>
    public KernelPath Path { get; }

    public int Threads { get; }

    public int ValueCount => _names.Count;

    public string NameOf(int id) => _names[id];

    /// <summary>The weights of value <paramref name="id"/>, or an empty tensor when it is not a constant.</summary>
    public Tensor Constant(int id) => id >= 0 && id < _constants.Count ? _constants[id] : default;

    public bool IsConstant(int id) => !Constant(id).IsEmpty;

    /// <summary>Adds a value (with a constant when given; the graph then owns the tensor) and returns its id.</summary>
    public int AddValue(string name, Tensor constant = default)
    {
        _names.Add(name);
        _constants.Add(constant);
        return _names.Count - 1;
    }

    /// <summary>Makes value <paramref name="id"/> a constant; the graph owns <paramref name="value"/>.</summary>
    public void SetConstant(int id, Tensor value) => _constants[id] = value;

    /// <summary>
    /// Releases the weights of constant <paramref name="id"/> now (a pass that has packed them elsewhere), so the unpacked and the
    /// packed copy are never both resident for long. Only for a value no remaining node reads.
    /// </summary>
    public void ReleaseConstant(int id)
    {
        Tensor c = _constants[id];
        if (!c.IsEmpty)
        {
            c.Dispose();
            _constants[id] = default;
        }
    }

    /// <summary>Hands a kernel a pass created to the graph, which disposes it when the session ends.</summary>
    public T Own<T>(T kernel) where T : OpKernel
    {
        _ownedKernels.Add(kernel);
        return kernel;
    }

    /// <summary>How many node inputs read each value, plus one per graph output.</summary>
    public int[] UseCounts()
    {
        var uses = new int[ValueCount];
        foreach (CompiledNode n in Nodes)
        {
            foreach (int id in n.Inputs)
            {
                if (id >= 0)
                {
                    uses[id]++;
                }
            }
        }

        foreach (int id in OutputIds)
        {
            uses[id]++;
        }

        return uses;
    }

    /// <summary>For each value id, the index in <see cref="Nodes"/> of the node producing it, or -1.</summary>
    public int[] Producers()
    {
        var p = new int[ValueCount];
        Array.Fill(p, -1);
        for (int i = 0; i < Nodes.Count; i++)
        {
            foreach (int id in Nodes[i].Outputs)
            {
                if (id >= 0)
                {
                    p[id] = i;
                }
            }
        }

        return p;
    }

    /// <summary>Removes the nodes none of whose outputs is read or is a graph output, repeatedly.</summary>
    public void RemoveDeadNodes()
    {
        while (true)
        {
            int[] uses = UseCounts();
            int removed = Nodes.RemoveAll(n =>
            {
                foreach (int id in n.Outputs)
                {
                    if (id >= 0 && uses[id] > 0)
                    {
                        return false;
                    }
                }

                return true;
            });
            if (removed == 0)
            {
                return;
            }
        }
    }

    internal List<Tensor> Constants => _constants;

    internal List<string> Names => _names;
}

/// <summary>The passes in the order they run, and the rule that picks them. A static table, like the operator table.</summary>
internal static class GraphPasses
{
    private static readonly IGraphPass[] s_all =
    [
        // The QDQ rewrite reads the DequantizeLinear nodes of the weights, so it runs before constant folding turns them into floats.
        new Operators.Quant.QdqPass(),
        new Operators.Quant.MatMulIntegerFusionPass(),
        new ConstantFoldPass(),
        new ConvBatchNormFoldPass(),
        new ConstantFoldPass(),
        new GeluFusionPass(),
        new AttentionFusionPass(),
        new LinearFusionPass(),
        new ConvActivationFusionPass(),
        new ConvPackPass(),
    ];

    /// <summary>The names of the passes that run for a level and mode, in order.</summary>
    public static IEnumerable<string> Selected(OptimizationLevel level, ExecutionMode mode)
    {
        foreach (IGraphPass p in s_all)
        {
            if (Applies(p, level, mode))
            {
                yield return p.Name;
            }
        }
    }

    private static bool Applies(IGraphPass p, OptimizationLevel level, ExecutionMode mode) =>
        level >= p.MinLevel && (mode != ExecutionMode.Exact || p.PreservesBits);

    /// <summary>Runs the passes that apply, then drops nodes nothing reads.</summary>
    public static void Run(PassGraph graph, OptimizationLevel level, CancellationToken ct)
    {
        if (level == OptimizationLevel.None)
        {
            return;
        }

        foreach (IGraphPass p in s_all)
        {
            ct.ThrowIfCancellationRequested();
            if (Applies(p, level, graph.Mode))
            {
                p.Run(graph, ct);
                graph.RemoveDeadNodes();
            }
        }
    }
}
