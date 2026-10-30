using System;
using System.Collections.Generic;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Flow;

/// <summary>
/// If: runs <c>then_branch</c> or <c>else_branch</c> by the boolean condition. Each branch is a nested session over the outer values
/// it reads; the outputs are the branch's outputs.
/// </summary>
internal sealed class IfOp : OpKernel, ISubgraphKernel
{
    public static readonly OpDef Def = new("If", 1, int.MaxValue, static n => new IfOp(n));

    private readonly NodeInfo _info;
    private NestedGraph? _then, _else;

    private IfOp(NodeInfo n) => _info = n;

    public override bool SupportsExact => true;

    public override bool OutputsDataDependent => true;

    public void Bind(SubgraphBinder b)
    {
        _then = b.Compile(Body(_info, "then_branch"));
        _else = b.Compile(Body(_info, "else_branch"));
    }

    internal static IrGraph Body(NodeInfo n, string attribute) =>
        n.Find(attribute) is { Type: IrAttributeType.Graph } a ? a.Graph : throw n.Error($"attribute '{attribute}' (a graph) is missing");

    public override void Infer(InferContext ctx) => throw new InvalidOperationException("If is shaped when it runs.");

    public override void Run(in OpRun run)
    {
        Tensor cond = run.Inputs[0];
        if (cond.DType != DType.Bool || cond.Length != 1)
        {
            throw new SessionException(SessionErrorKind.InvalidInput, "If: the condition must be a boolean with one element");
        }

        NestedGraph body = cond.Data<bool>()[0] ? _then! : _else!;
        Tensor[] outs = body.Run([], run.Inputs.Slice(_info.InputCount));
        try
        {
            for (int i = 0; i < outs.Length; i++)
            {
                FlowUtil.Emit(run, i, outs[i]);
            }
        }
        finally
        {
            FlowUtil.DisposeAll(outs);
        }
    }

    public override void Dispose()
    {
        _then?.Dispose();
        _else?.Dispose();
        base.Dispose();
    }
}

/// <summary>
/// Loop: runs <c>body</c> while the trip count and the condition allow it. Body inputs are the iteration number, the condition and the
/// loop-carried values; body outputs the next condition, the carried values and the scan outputs, which are stacked on a new first axis.
/// </summary>
internal sealed class LoopOp : OpKernel, ISubgraphKernel
{
    public static readonly OpDef Def = new("Loop", 1, int.MaxValue, static n => new LoopOp(n));

    private readonly NodeInfo _info;
    private NestedGraph? _body;
    private IrGraph? _ir;

    private LoopOp(NodeInfo n) => _info = n;

    public override bool SupportsExact => true;

    public override bool OutputsDataDependent => true;

    public void Bind(SubgraphBinder b)
    {
        _ir = IfOp.Body(_info, "body");
        if (_info.InputCount < 2 || _ir.Inputs.Length != _info.InputCount || _ir.Outputs.Length < _info.InputCount - 1)
        {
            throw _info.Error("the body must take iteration, condition and one value per carried input, and return the condition and the carried values");
        }

        _body = b.Compile(_ir);
    }

    public override void Infer(InferContext ctx) => throw new InvalidOperationException("Loop is shaped when it runs.");

    public override void Run(in OpRun run)
    {
        int carriedCount = _info.InputCount - 2, scanCount = _body!.OutputCount - 1 - carriedCount;
        ReadOnlySpan<Tensor> inputs = run.Inputs;
        long trips = long.MaxValue;
        if (!inputs[0].IsEmpty)
        {
            trips = FlowUtil.ScalarInt(inputs[0]);
        }

        bool cond = true;
        if (!inputs[1].IsEmpty)
        {
            if (inputs[1].DType != DType.Bool || inputs[1].Length != 1)
            {
                throw new SessionException(SessionErrorKind.InvalidInput, "Loop: the condition must be a boolean with one element");
            }

            cond = inputs[1].Data<bool>()[0];
        }

        ReadOnlySpan<Tensor> captured = inputs.Slice(_info.InputCount);
        var carried = new Tensor[carriedCount];
        var owned = new bool[carriedCount];
        for (int i = 0; i < carriedCount; i++)
        {
            carried[i] = inputs[2 + i];
        }

        var scans = new List<Tensor>[scanCount];
        for (int k = 0; k < scanCount; k++)
        {
            scans[k] = [];
        }

        try
        {
            var declared = new Tensor[2 + carriedCount];
            for (long it = 0; it < trips && cond; it++)
            {
                Tensor iter = Tensor.FromSpan<long>([it], FlowUtil.ScalarShape);
                Tensor c = Tensor.FromSpan<bool>([true], FlowUtil.ScalarShape);
                Tensor[] outs;
                try
                {
                    declared[0] = iter;
                    declared[1] = c;
                    carried.CopyTo(declared, 2);
                    outs = _body.Run(declared, captured);
                }
                finally
                {
                    iter.Dispose();
                    c.Dispose();
                }

                if (outs[0].DType != DType.Bool || outs[0].Length != 1)
                {
                    FlowUtil.DisposeAll(outs);
                    throw new SessionException(SessionErrorKind.InvalidModel, "Loop: the body's first output must be a boolean with one element");
                }

                cond = outs[0].Data<bool>()[0];
                outs[0].Dispose();
                for (int i = 0; i < carriedCount; i++)
                {
                    if (owned[i])
                    {
                        carried[i].Dispose();
                    }

                    carried[i] = outs[1 + i];
                    owned[i] = true;
                }

                for (int k = 0; k < scanCount; k++)
                {
                    scans[k].Add(outs[1 + carriedCount + k]);
                }
            }

            for (int i = 0; i < carriedCount; i++)
            {
                FlowUtil.Emit(run, i, carried[i]);
            }

            for (int k = 0; k < scanCount; k++)
            {
                List<Tensor> list = scans[k];
                if (carriedCount + k >= run.Outputs.Length)
                {
                    continue;
                }

                if (list.Count == 0)
                {
                    IrType? t = _ir!.Outputs[1 + carriedCount + k].Type;
                    DType type = t is not null && DTypes.TryFromOnnx(t.ElemType, out DType dt) ? dt : DType.F32;
                    run.AllocateOutput(carriedCount + k, type, Shape.Create(0));
                    continue;
                }

                Tensor stacked = FlowUtil.AllocateStacked(run, carriedCount + k, list[0].DType, list[0].Shape.Dimensions, 0, list.Count);
                for (int j = 0; j < list.Count; j++)
                {
                    FlowUtil.Place(stacked, 0, j, list[j]);
                }
            }
        }
        finally
        {
            for (int i = 0; i < carriedCount; i++)
            {
                if (owned[i])
                {
                    carried[i].Dispose();
                }
            }

            foreach (List<Tensor> l in scans)
            {
                FlowUtil.DisposeAll(l);
            }
        }
    }

    public override void Dispose()
    {
        _body?.Dispose();
        base.Dispose();
    }
}

/// <summary>
/// Scan (opset 9 and later): runs <c>body</c> once per element along the scan axis of the scan inputs, threading the state through. Scan
/// outputs are stacked along their scan output axis; reverse directions read and write back to front. Opset 8 adds a batch axis and
/// optional per-example sequence lengths.
/// </summary>
internal sealed class ScanOp : OpKernel, ISubgraphKernel
{
    public static readonly OpDef Def = new("Scan", 8, int.MaxValue, static n => new ScanOp(n));

    private readonly NodeInfo _info;
    private readonly bool _v8;
    private readonly int _scanInputs;
    private NestedGraph? _body;

    private ScanOp(NodeInfo n)
    {
        _info = n;
        _v8 = n.Opset < 9;
        _scanInputs = (int)n.GetInt("num_scan_inputs", -1);
        if (_scanInputs < 1)
        {
            throw n.Error("num_scan_inputs must be at least 1");
        }
    }

    public override bool SupportsExact => true;

    public override bool OutputsDataDependent => true;

    public void Bind(SubgraphBinder b)
    {
        IrGraph ir = IfOp.Body(_info, "body");
        int states = _info.InputCount - _scanInputs - (_v8 ? 1 : 0);
        if (states < 0 || ir.Inputs.Length != states + _scanInputs || ir.Outputs.Length < states)
        {
            throw _info.Error("the body must take the states then one slice per scan input, and return the states then the scan outputs");
        }

        _body = b.Compile(ir);
    }

    public override void Infer(InferContext ctx) => throw new InvalidOperationException("Scan is shaped when it runs.");

    public override void Run(in OpRun run)
    {
        if (_v8)
        {
            RunV8(run);
            return;
        }

        ReadOnlySpan<Tensor> inputs = run.Inputs;
        int m = _scanInputs, n = _info.InputCount - m, k = _body!.OutputCount - n;
        System.Collections.Immutable.ImmutableArray<long> inAxes = _info.GetInts("scan_input_axes"), inDirs = _info.GetInts("scan_input_directions");
        System.Collections.Immutable.ImmutableArray<long> outAxes = _info.GetInts("scan_output_axes"), outDirs = _info.GetInts("scan_output_directions");
        var axes = new int[m];
        for (int j = 0; j < m; j++)
        {
            Tensor t = inputs[n + j];
            axes[j] = FlowUtil.Axis(inAxes.IsDefaultOrEmpty ? 0 : inAxes[j], t.Rank, "Scan scan_input_axes");
        }

        int len = inputs[n].Shape[axes[0]];
        for (int j = 1; j < m; j++)
        {
            if (inputs[n + j].Shape[axes[j]] != len)
            {
                throw new SessionException(SessionErrorKind.InvalidInput, "Scan: the scan inputs have different lengths along their scan axes");
            }
        }

        ReadOnlySpan<Tensor> captured = inputs.Slice(_info.InputCount);
        var state = new Tensor[n];
        var owned = new bool[n];
        for (int i = 0; i < n; i++)
        {
            state[i] = inputs[i];
        }

        var stacked = new Tensor[k];
        var declared = new Tensor[n + m];
        var slices = new Tensor[m];
        try
        {
            for (int t = 0; t < len; t++)
            {
                for (int j = 0; j < m; j++)
                {
                    int at = !inDirs.IsDefaultOrEmpty && inDirs[j] == 1 ? len - 1 - t : t;
                    slices[j] = FlowUtil.Narrow(inputs[n + j], axes[j], at, 1, drop: true);
                }

                Tensor[] outs;
                try
                {
                    state.CopyTo(declared, 0);
                    slices.CopyTo(declared, n);
                    outs = _body.Run(declared, captured);
                }
                finally
                {
                    FlowUtil.DisposeAll(slices);
                    Array.Clear(slices);
                }

                try
                {
                    for (int q = 0; q < k; q++)
                    {
                        Tensor o = outs[n + q];
                        if (stacked[q].IsEmpty)
                        {
                            int ax = FlowUtil.Axis(outAxes.IsDefaultOrEmpty ? 0 : outAxes[q], o.Rank + 1, "Scan scan_output_axes");
                            stacked[q] = FlowUtil.AllocateStacked(run, n + q, o.DType, o.Shape.Dimensions, ax, len);
                        }

                        int pos = !outDirs.IsDefaultOrEmpty && outDirs[q] == 1 ? len - 1 - t : t;
                        int axis = FlowUtil.Axis(outAxes.IsDefaultOrEmpty ? 0 : outAxes[q], o.Rank + 1, "Scan scan_output_axes");
                        FlowUtil.Place(stacked[q], axis, pos, o);
                    }

                    for (int i = 0; i < n; i++)
                    {
                        if (owned[i])
                        {
                            state[i].Dispose();
                        }

                        state[i] = outs[i];
                        owned[i] = true;
                        outs[i] = default;
                    }
                }
                finally
                {
                    FlowUtil.DisposeAll(outs);
                }
            }

            for (int i = 0; i < n; i++)
            {
                FlowUtil.Emit(run, i, state[i]);
            }

            for (int q = 0; q < k; q++)
            {
                if (stacked[q].IsEmpty && n + q < run.Outputs.Length)
                {
                    run.AllocateOutput(n + q, DType.F32, Shape.Create(0));
                }
            }
        }
        finally
        {
            for (int i = 0; i < n; i++)
            {
                if (owned[i])
                {
                    state[i].Dispose();
                }
            }
        }
    }

    // Opset 8: inputs are [sequence_lens], the initial states (batch first) and the scan inputs (batch, time, ...).
    private void RunV8(in OpRun run)
    {
        ReadOnlySpan<Tensor> inputs = run.Inputs;
        int m = _scanInputs, n = _info.InputCount - 1 - m, k = _body!.OutputCount - n;
        System.Collections.Immutable.ImmutableArray<long> dirs = _info.GetInts("directions");
        Tensor first = inputs[1 + n];
        int batch = first.Shape[0], maxLen = first.Shape[1];
        ReadOnlySpan<Tensor> captured = inputs.Slice(_info.InputCount);
        var finalState = new Tensor[n];
        for (int i = 0; i < n; i++)
        {
            finalState[i] = run.AllocateOutput(i, inputs[1 + i].DType, inputs[1 + i].Shape);
        }

        var outs3 = new Tensor[k];
        var declared = new Tensor[n + m];
        Span<int> sd = stackalloc int[Shape.MaxRank];
        for (int b = 0; b < batch; b++)
        {
            int len = inputs[0].IsEmpty ? maxLen : (int)Math.Clamp(inputs[0].DType == DType.I64 ? inputs[0].Data<long>()[b] : inputs[0].Data<int>()[b], 0, maxLen);
            var state = new Tensor[n];
            for (int i = 0; i < n; i++)
            {
                state[i] = FlowUtil.Narrow(inputs[1 + i], 0, b, 1, drop: true);
            }

            try
            {
                for (int t = 0; t < len; t++)
                {
                    var slices = new Tensor[m];
                    try
                    {
                        for (int j = 0; j < m; j++)
                        {
                            Tensor src = inputs[1 + n + j];
                            using Tensor row = FlowUtil.Narrow(src, 0, b, 1, drop: true);
                            int at = !dirs.IsDefaultOrEmpty && dirs[j] == 1 ? len - 1 - t : t;
                            slices[j] = FlowUtil.Narrow(row, 0, at, 1, drop: true);
                        }

                        state.CopyTo(declared, 0);
                        slices.CopyTo(declared, n);
                        Tensor[] outs = _body.Run(declared, captured);
                        try
                        {
                            for (int q = 0; q < k; q++)
                            {
                                Tensor o = outs[n + q];
                                if (outs3[q].IsEmpty)
                                {
                                    sd[0] = batch;
                                    sd[1] = maxLen;
                                    o.Shape.Dimensions.CopyTo(sd.Slice(2));
                                    outs3[q] = run.AllocateOutput(n + q, o.DType, Shape.Create(sd.Slice(0, 2 + o.Rank)));
                                    PackedValue.MutableBytes(outs3[q]).Clear();
                                }

                                int es = o.ElementSize, block = o.Length * es;
                                int pos = !dirs.IsDefaultOrEmpty && q < dirs.Length && dirs[q] == 1 ? len - 1 - t : t;
                                if (block > 0)
                                {
                                    PackedValue.Bytes(o).CopyTo(PackedValue.MutableBytes(outs3[q]).Slice(((b * maxLen) + pos) * block, block));
                                }
                            }

                            for (int i = 0; i < n; i++)
                            {
                                state[i].Dispose();
                                state[i] = outs[i];
                                outs[i] = default;
                            }
                        }
                        finally
                        {
                            FlowUtil.DisposeAll(outs);
                        }
                    }
                    finally
                    {
                        FlowUtil.DisposeAll(slices);
                    }
                }

                for (int i = 0; i < n; i++)
                {
                    int block = state[i].Length * state[i].ElementSize;
                    if (block > 0)
                    {
                        PackedValue.Bytes(state[i]).CopyTo(PackedValue.MutableBytes(finalState[i]).Slice(b * block, block));
                    }
                }
            }
            finally
            {
                FlowUtil.DisposeAll(state);
            }
        }

        for (int q = 0; q < k; q++)
        {
            if (outs3[q].IsEmpty && n + q < run.Outputs.Length)
            {
                run.AllocateOutput(n + q, DType.F32, Shape.Create(batch, maxLen));
            }
        }
    }

    public override void Dispose()
    {
        _body?.Dispose();
        base.Dispose();
    }
}
