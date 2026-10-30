using System;
using System.Collections.Generic;
using System.Threading;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Runtime;

/// <summary>
/// An operator with graph-valued attributes (If, Loop, Scan, SequenceMap). When the compiled graph creates the kernel it hands over a
/// <see cref="SubgraphBinder"/>, from which the kernel compiles each body into a nested <see cref="Session"/>.
/// </summary>
internal interface ISubgraphKernel
{
    /// <summary>Called once, right after the kernel is created. May throw <see cref="SessionException"/> for a body that cannot be compiled.</summary>
    void Bind(SubgraphBinder binder);
}

/// <summary>What a kernel needs to compile its bodies in the same mode and on the same kernel path as the session around it.</summary>
internal sealed class SubgraphBinder
{
    private readonly OpRegistry _registry;
    private readonly ExecutionMode _mode;
    private readonly KernelPath _path;
    private readonly int _threads;
    private readonly CancellationToken _ct;

    public SubgraphBinder(IrNode node, IReadOnlyList<IrValue> captures, OpRegistry registry, ExecutionMode mode, KernelPath path, int threads, CancellationToken ct)
    {
        Node = node;
        Captures = captures;
        _registry = registry;
        _mode = mode;
        _path = path;
        _threads = threads;
        _ct = ct;
    }

    /// <summary>The node whose bodies are compiled.</summary>
    public IrNode Node { get; }

    /// <summary>The outer values the node's bodies read, in the order they follow the node's declared inputs at run time.</summary>
    public IReadOnlyList<IrValue> Captures { get; }

    /// <summary>Compiles a body. Every captured value becomes an extra input of the nested session, after the body's own inputs.</summary>
    public NestedGraph Compile(IrGraph body)
    {
        Session s = Session.CreateNested(body, Captures, _registry, _mode, _path, _threads, _ct);
        var names = new List<string>();
        var slots = new List<int>();
        for (int i = 0; i < body.Inputs.Length; i++)
        {
            if (body.Inputs[i].Initializer is null)
            {
                names.Add(body.Inputs[i].Name);
                slots.Add(i);
            }
        }

        int own = names.Count;
        for (int k = 0; k < Captures.Count; k++)
        {
            if (Captures[k].Initializer is null)
            {
                names.Add(Captures[k].Name);
                slots.Add(body.Inputs.Length + k);
            }
        }

        var outs = new string[body.Outputs.Length];
        for (int i = 0; i < outs.Length; i++)
        {
            outs[i] = body.Outputs[i].Name;
        }

        return new NestedGraph(s, [.. names], [.. slots], body.Inputs.Length, outs);
    }
}

/// <summary>A compiled body: a session plus the bookkeeping to feed it declared inputs and captured values.</summary>
internal sealed class NestedGraph : IDisposable
{
    private readonly Session _session;
    private readonly string[] _names;
    private readonly int[] _slots;
    private readonly int _declared;

    public NestedGraph(Session session, string[] names, int[] slots, int declared, string[] outputs)
    {
        _session = session;
        _names = names;
        _slots = slots;
        _declared = declared;
        OutputNames = outputs;
    }

    public string[] OutputNames { get; }

    public int OutputCount => OutputNames.Length;

    /// <summary>The number of inputs the body declares.</summary>
    public int DeclaredInputs => _declared;

    /// <summary>
    /// Runs the body. <paramref name="declared"/> holds one tensor per declared body input (the ones the body gives an initializer are
    /// ignored); <paramref name="captured"/> the node's implicit inputs. Outputs are new tensors the caller disposes.
    /// </summary>
    public Tensor[] Run(ReadOnlySpan<Tensor> declared, ReadOnlySpan<Tensor> captured)
    {
        if (declared.Length != _declared)
        {
            throw new SessionException(SessionErrorKind.InvalidModel, $"A subgraph takes {_declared} inputs but {declared.Length} were given.");
        }

        var inputs = new Tensor[_names.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            int slot = _slots[i];
            inputs[i] = slot < _declared ? declared[slot] : captured[slot - _declared];
        }

        var outputs = new Tensor[OutputNames.Length];
        _session.Run(_names, inputs, OutputNames, outputs);
        return outputs;
    }

    public void Dispose() => _session.Dispose();
}
