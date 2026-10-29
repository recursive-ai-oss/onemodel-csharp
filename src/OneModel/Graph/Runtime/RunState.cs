using System;
using System.Collections.Generic;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Runtime;

/// <summary>
/// The mutable part of one run: the value table, the slab the planned values live in, and an arena for scratch and run-time-shaped
/// values. Belongs to one run at a time; the plan keeps idle ones for reuse so a steady-state run allocates nothing.
/// </summary>
internal sealed class RunState : IDisposable
{
    private readonly ExecutionPlan _plan;
    private readonly TensorBuffer? _slab;
    private readonly Arena _arena;
    private readonly Tensor[] _inBuf;
    private readonly Tensor[] _outBuf;
    private readonly List<Tensor> _runOwned = [];
    private InferContext? _ictx;

    public RunState(ExecutionPlan plan)
    {
        _plan = plan;
        _arena = new Arena(plan.Env.Pool);
        _inBuf = new Tensor[Math.Max(1, plan.MaxNodeInputs)];
        _outBuf = new Tensor[Math.Max(1, plan.MaxNodeOutputs)];
        Values = new Tensor[plan.Graph.ValueCount];
        Staging = new Tensor[plan.InTypes.Length];
        if (plan.SlabBytes > 0)
        {
            _slab = plan.Env.Pool.Rent(plan.SlabBytes);
        }

        for (int v = 0; v < Values.Length; v++)
        {
            switch (plan.Kinds[v])
            {
                case ValueKind.Const:
                    Values[v] = plan.Consts[v];
                    break;
                case ValueKind.Static when _slab is not null && !IsViewOutput(v):
                    Values[v] = Tensor.Over(_slab, plan.Offsets[v], plan.Types[v], plan.Shapes[v]);
                    break;
            }
        }

        for (int i = 0; i < Staging.Length; i++)
        {
            if (_slab is not null)
            {
                Staging[i] = Tensor.Over(_slab, plan.StagingOffsets[i], plan.InTypes[i], plan.InShapes[i]);
            }
        }
    }

    // A static view output has no storage of its own; it is built from its source on every run.
    private bool IsViewOutput(int v)
    {
        _viewOut ??= ComputeViewOutputs();
        return _viewOut[v];
    }

    private bool[]? _viewOut;

    private bool[] ComputeViewOutputs()
    {
        var r = new bool[_plan.Graph.ValueCount];
        foreach (PlanStep st in _plan.Steps)
        {
            if (st.IsView && st.Node.Outputs[0] >= 0)
            {
                r[st.Node.Outputs[0]] = true;
            }
        }

        return r;
    }

    /// <summary>The value table, indexed by value id.</summary>
    public Tensor[] Values { get; }

    /// <summary>For each input, a tensor in the slab that a span input is copied into.</summary>
    public Tensor[] Staging { get; }

    /// <summary>Runs every step in order. Inputs must already be in <see cref="Values"/>.</summary>
    public void Execute()
    {
        _arena.Reset();
        try
        {
            foreach (PlanStep step in _plan.Steps)
            {
                if (step.Dynamic)
                {
                    ExecuteDynamic(step);
                }
                else if (step.IsView)
                {
                    CompiledNode n = step.Node;
                    Tensor src = Values[n.Inputs[n.Kernel.ViewOfInput]];
                    if (n.Outputs[0] >= 0)
                    {
                        Values[n.Outputs[0]] = MakeView(src, step.OutShapes[0], n);
                    }
                }
                else
                {
                    ExecuteStatic(step);
                }

#if CHECKED
                PoisonFreed(step);
#endif
            }
        }
        finally
        {
            foreach (Tensor t in _runOwned)
            {
                t.Dispose();
            }

            _runOwned.Clear();
        }
    }

    /// <summary>Forgets the caller's tensors and the values shaped at run time, so a pooled state keeps nothing alive.</summary>
    public void Clear()
    {
        foreach (int id in _plan.Graph.InputIds)
        {
            Values[id] = default;
        }

        for (int v = 0; v < Values.Length; v++)
        {
            if (_plan.Kinds[v] == ValueKind.Dynamic)
            {
                Values[v] = default;
            }
        }
    }

    private static Tensor MakeView(in Tensor src, Shape shape, CompiledNode n)
    {
        if (!src.TryReshape(shape, out Tensor view))
        {
            throw new SessionException(SessionErrorKind.InvalidModel, $"{n.Info.Describe()}: cannot view {src.Shape} as {shape}.", n.Info.Name);
        }

        return view;
    }

    private void ExecuteStatic(PlanStep step)
    {
        CompiledNode n = step.Node;
        int nIn = n.Inputs.Length, nOut = n.Outputs.Length;
        for (int i = 0; i < nIn; i++)
        {
            _inBuf[i] = n.Inputs[i] >= 0 ? Values[n.Inputs[i]] : default;
        }

        for (int o = 0; o < nOut; o++)
        {
            int id = n.Outputs[o];
            if (id >= 0)
            {
                _outBuf[o] = Values[id];
            }
            else if (step.OutTypes[o] != DType.Undefined)
            {
                _outBuf[o] = Tensor.Create(step.OutTypes[o], step.OutShapes[o], _arena);
            }
            else
            {
                _outBuf[o] = default;
            }
        }

        n.Kernel.Run(new OpRun(_inBuf.AsSpan(0, nIn), _outBuf.AsSpan(0, nOut), _plan.Env.Mode, _plan.Env.Path, _plan.Env.Threads, _arena, step.State));
    }

    private void ExecuteDynamic(PlanStep step)
    {
        CompiledNode n = step.Node;
        OpKernel k = n.Kernel;
        int nIn = n.Inputs.Length, nOut = n.Outputs.Length;
        for (int i = 0; i < nIn; i++)
        {
            _inBuf[i] = n.Inputs[i] >= 0 ? Values[n.Inputs[i]] : default;
        }

        InferContext ictx = _ictx ??= new InferContext();
        if (!k.OutputsDataDependent)
        {
            ictx.Begin(n.Info, nIn, nOut);
            for (int i = 0; i < nIn; i++)
            {
                if (!_inBuf[i].IsEmpty)
                {
                    ictx.SetInput(i, _inBuf[i].DType, _inBuf[i].Shape, _inBuf[i]);
                }
            }

            try
            {
                k.Infer(ictx);
            }
            catch (Exception e) when (e is ArgumentException or InvalidOperationException)
            {
                throw new SessionException(SessionErrorKind.InvalidModel, $"{n.Info.Describe()}: {e.Message}", n.Info.Name, default, e);
            }

            if (ictx.NeedsRunTimeValues)
            {
                throw new SessionException(SessionErrorKind.InvalidModel, $"{n.Info.Describe()}: shape inference needs values that are not available.", n.Info.Name);
            }
        }

        if (!k.OutputsDataDependent)
        {
            bool anyConst = false;
            for (int o = 0; o < nOut; o++)
            {
                anyConst |= ictx.OutputIsSet(o) && !ictx.OutputConstant(o).IsEmpty;
            }

            if (anyConst)
            {
                for (int o = 0; o < nOut; o++)
                {
                    Tensor c = ictx.OutputConstant(o);
                    if (n.Outputs[o] >= 0)
                    {
                        Values[n.Outputs[o]] = c;
                    }

                    if (ictx.OutputConstantOwned(o) && !c.IsEmpty)
                    {
                        _runOwned.Add(c);
                    }
                }

                return;
            }

            if (step.IsView)
            {
                if (n.Outputs[0] >= 0)
                {
                    Values[n.Outputs[0]] = MakeView(_inBuf[k.ViewOfInput], ictx.OutputShape(0), n);
                }

                return;
            }

            for (int o = 0; o < nOut; o++)
            {
                _outBuf[o] = ictx.OutputIsSet(o) ? Tensor.Create(ictx.OutputType(o), ictx.OutputShape(o), _arena) : default;
            }
        }
        else
        {
            for (int o = 0; o < nOut; o++)
            {
                _outBuf[o] = default;
            }
        }

        k.Run(new OpRun(_inBuf.AsSpan(0, nIn), _outBuf.AsSpan(0, nOut), _plan.Env.Mode, _plan.Env.Path, _plan.Env.Threads, _arena));
        for (int o = 0; o < nOut; o++)
        {
            if (n.Outputs[o] < 0)
            {
                continue;
            }

            if (_outBuf[o].IsEmpty)
            {
                throw new SessionException(SessionErrorKind.InvalidModel, $"{n.Info.Describe()}: output {o} was not produced.", n.Info.Name);
            }

            Values[n.Outputs[o]] = _outBuf[o];
        }
    }

#if CHECKED
    // Freed slab ranges are overwritten so a read of a dead value shows up as garbage in the tests.
    private void PoisonFreed(PlanStep step)
    {
        if (_slab is null)
        {
            return;
        }

        foreach ((long offset, long size) in step.Freed)
        {
            _slab.PoisonRange(offset, size);
        }
    }
#endif

    public void Dispose()
    {
        _arena.Dispose();
        if (_slab is not null)
        {
            _slab.Release(_slab.Generation);
        }
    }
}
