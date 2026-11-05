using System;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Runtime;

/// <summary>
/// One graph node run on its own with concrete tensors, outside a session: the training package uses it for the parts of an imported
/// graph that carry no gradient (shape arithmetic, masks, tables), so they behave exactly as the graph runtime does.
/// </summary>
internal sealed class EagerNode : IDisposable
{
    private readonly OpKernel _kernel;
    private readonly NodeInfo _info;
    private readonly InferContext _infer = new();

    private EagerNode(OpKernel kernel, NodeInfo info)
    {
        _kernel = kernel;
        _info = info;
    }

    /// <summary>Looks the node's operator up. Returns null when the registry has no kernel for it.</summary>
    public static EagerNode? TryCreate(IrNode node)
    {
        if (!OpRegistry.Default.TryFind(node.Domain, node.OpType, node.OpsetVersion, out OpDef def))
        {
            return null;
        }

        var info = new NodeInfo(node, node.OpsetVersion);
        try
        {
            return new EagerNode(def.Create(info), info);
        }
        catch (Exception e) when (e is InvalidOperationException or ArgumentException or NotSupportedException or OverflowException)
        {
            throw new SessionException(SessionErrorKind.InvalidModel, $"{info.Describe()}: {e.Message}", node.Name, default, e);
        }
    }

    public bool SupportsExact => _kernel.SupportsExact;

    /// <summary>The node's output count.</summary>
    public int OutputCount => _info.OutputCount;

    /// <summary>
    /// Runs the node. <paramref name="inputs"/> are in the node's input order (an omitted optional input is an empty tensor) and are
    /// made contiguous here when needed. Outputs are new tensors from <paramref name="pool"/> (tracked by the open activation scope),
    /// views of an input for view-like operators, or the kernel's constant.
    /// </summary>
    public Tensor[] Run(ReadOnlySpan<Tensor> inputs, ArithmeticMode mode, KernelPath path, int threads, Arena scratch, BufferPool? pool)
    {
        int nIn = inputs.Length, nOut = _info.OutputCount;
        if (_kernel.OutputsDataDependent)
        {
            throw _info.Unsupported("its output size depends on the data, which imported training graphs do not support");
        }

        var ins = new Tensor[nIn];
        for (int i = 0; i < nIn; i++)
        {
            Tensor t = inputs[i];
            if (t.IsEmpty || t.IsContiguous)
            {
                ins[i] = t;
            }
            else
            {
                ins[i] = t.Detach().Contiguous(pool);
                ActivationScope.Track(ins[i]);
            }
        }

        _infer.Begin(_info, nIn, nOut);
        for (int i = 0; i < nIn; i++)
        {
            if (!ins[i].IsEmpty)
            {
                _infer.SetInput(i, ins[i].DType, ins[i].Shape, ins[i]);
            }
        }

        try
        {
            _kernel.Infer(_infer);
        }
        catch (Exception e) when (e is ArgumentException or InvalidOperationException)
        {
            throw new SessionException(SessionErrorKind.InvalidModel, $"{_info.Describe()}: {e.Message}", _info.Name, default, e);
        }

        var outs = new Tensor[nOut];
        bool anyConst = false;
        for (int o = 0; o < nOut; o++)
        {
            anyConst |= _infer.OutputIsSet(o) && !_infer.OutputConstant(o).IsEmpty;
        }

        if (anyConst)
        {
            for (int o = 0; o < nOut; o++)
            {
                outs[o] = _infer.OutputConstant(o);
                if (_infer.OutputConstantOwned(o))
                {
                    ActivationScope.Track(outs[o]);
                }
            }

            return outs;
        }

        if (_kernel.ViewOfInput >= 0)
        {
            Tensor src = ins[_kernel.ViewOfInput];
            outs[0] = src.TryReshape(_infer.OutputShape(0), out Tensor view)
                ? view
                : throw new SessionException(SessionErrorKind.InvalidModel, $"{_info.Describe()}: cannot view {src.Shape} as {_infer.OutputShape(0)}.", _info.Name);
            return outs;
        }

        for (int o = 0; o < nOut; o++)
        {
            if (_infer.OutputIsSet(o))
            {
                outs[o] = Tensor.Create(_infer.OutputType(o), _infer.OutputShape(o), pool);
                ActivationScope.Track(outs[o]);
            }
        }

        try
        {
            _kernel.Run(new OpRun(ins, outs, mode, path, threads, scratch));
        }
        finally
        {
            scratch.Reset();
        }

        return outs;
    }

    public void Dispose() => _kernel.Dispose();
}
