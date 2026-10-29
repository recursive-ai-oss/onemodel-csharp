using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

// Gathers and scatters: operators whose output positions are chosen by an index tensor. They copy elements of any type; an index
// that is out of range is an InvalidInput error at run time (negative indices count from the end).

/// <summary>Gather: out = data[..., indices, ...] along an axis; the output shape is data[:axis] + indices.shape + data[axis+1:].</summary>
internal sealed class GatherOp : OpKernel
{
    public static readonly OpDef Def = new("Gather", 1, int.MaxValue, static n => new GatherOp(n));

    private readonly long _axis;

    private GatherOp(NodeInfo node)
    {
        _axis = node.GetInt("axis", 0);
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects data and indices");
        }

        Shape d = ctx.InputShape(0), ix = ctx.InputShape(1);
        if (ctx.InputType(1) is not (DType.I32 or DType.I64))
        {
            throw ctx.Fail($"indices must be int32 or int64, not {ctx.InputType(1)}");
        }

        if (d.Rank == 0)
        {
            throw ctx.Fail("cannot gather from a scalar");
        }

        int axis = OpUtil.Axis(ctx, _axis, d.Rank);
        int rank = d.Rank - 1 + ix.Rank;
        if (rank > Shape.MaxRank)
        {
            throw ctx.Fail($"the output rank {rank} is above {Shape.MaxRank}");
        }

        Span<long> dims = stackalloc long[Shape.MaxRank];
        int n = 0;
        for (int i = 0; i < axis; i++)
        {
            dims[n++] = d[i];
        }

        for (int i = 0; i < ix.Rank; i++)
        {
            dims[n++] = ix[i];
        }

        for (int i = axis + 1; i < d.Rank; i++)
        {
            dims[n++] = d[i];
        }

        ctx.SetOutput(0, ctx.InputType(0), OpUtil.MakeShape(ctx, dims.Slice(0, n)));
    }

    public override void Run(in OpRun run)
    {
        Tensor data = run.Inputs[0];
        int axis = (int)(_axis < 0 ? _axis + data.Rank : _axis);
        ReadOnlySpan<int> dims = data.Shape.Dimensions;
        if (run.Outputs[0].Length == 0)
        {
            return;
        }

        Movement.Gather(data, run.Inputs[1], run.Outputs[0], OpUtil.Product(dims, 0, axis), dims[axis], OpUtil.Product(dims, axis + 1, dims.Length));
    }
}

/// <summary>GatherElements: out[p] = data[p with its axis coordinate replaced by indices[p]]; the output has the indices' shape.</summary>
internal sealed class GatherElementsOp : OpKernel
{
    public static readonly OpDef Def = new("GatherElements", 11, int.MaxValue, static n => new GatherElementsOp(n));

    private readonly long _axis;

    private GatherElementsOp(NodeInfo node)
    {
        _axis = node.GetInt("axis", 0);
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects data and indices");
        }

        Shape d = ctx.InputShape(0), ix = ctx.InputShape(1);
        if (ctx.InputType(1) is not (DType.I32 or DType.I64))
        {
            throw ctx.Fail($"indices must be int32 or int64, not {ctx.InputType(1)}");
        }

        if (d.Rank == 0 || d.Rank != ix.Rank)
        {
            throw ctx.Fail($"data {d} and indices {ix} must have the same rank, at least 1");
        }

        int axis = OpUtil.Axis(ctx, _axis, d.Rank);
        for (int i = 0; i < d.Rank; i++)
        {
            if (i != axis && ix[i] > d[i])
            {
                throw ctx.Fail($"indices extent {ix[i]} exceeds data extent {d[i]} in axis {i}");
            }
        }

        ctx.SetOutput(0, ctx.InputType(0), ix);
    }

    public override void Run(in OpRun run)
    {
        Tensor data = run.Inputs[0];
        int axis = (int)(_axis < 0 ? _axis + data.Rank : _axis);
        Movement.GatherElements(data, run.Inputs[1], run.Outputs[0], axis);
    }
}

/// <summary>GatherND: gathers slices of data selected by the k-tuples in the last axis of indices, after batch_dims leading batch axes.</summary>
internal sealed class GatherNDOp : OpKernel
{
    public static readonly OpDef Def = new("GatherND", 11, int.MaxValue, static n => new GatherNDOp(n));

    private readonly int _batchDims;

    private GatherNDOp(NodeInfo node)
    {
        long b = node.GetInt("batch_dims", 0);
        _batchDims = b is < 0 or > Shape.MaxRank ? throw node.Error($"batch_dims {b} is out of range") : (int)b;
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects data and indices");
        }

        Shape d = ctx.InputShape(0), ix = ctx.InputShape(1);
        if (ctx.InputType(1) != DType.I64)
        {
            throw ctx.Fail($"indices must be int64, not {ctx.InputType(1)}");
        }

        if (ix.Rank < 1 || ix.Rank <= _batchDims || d.Rank <= _batchDims)
        {
            throw ctx.Fail($"data {d} and indices {ix} do not have batch_dims={_batchDims} leading axes plus more");
        }

        for (int i = 0; i < _batchDims; i++)
        {
            if (d[i] != ix[i])
            {
                throw ctx.Fail($"batch axis {i} differs: data {d[i]}, indices {ix[i]}");
            }
        }

        int k = ix[ix.Rank - 1];
        if (k < 1 || k > d.Rank - _batchDims)
        {
            throw ctx.Fail($"the index tuples have {k} entries but data {d} has {d.Rank - _batchDims} axes after batch_dims");
        }

        int rank = ix.Rank - 1 + d.Rank - _batchDims - k;
        if (rank > Shape.MaxRank)
        {
            throw ctx.Fail($"the output rank {rank} is above {Shape.MaxRank}");
        }

        Span<long> dims = stackalloc long[Shape.MaxRank];
        int n = 0;
        for (int i = 0; i < ix.Rank - 1; i++)
        {
            dims[n++] = ix[i];
        }

        for (int i = _batchDims + k; i < d.Rank; i++)
        {
            dims[n++] = d[i];
        }

        ctx.SetOutput(0, ctx.InputType(0), OpUtil.MakeShape(ctx, dims.Slice(0, n)));
    }

    public override void Run(in OpRun run) => Movement.GatherND(run.Inputs[0], run.Inputs[1], run.Outputs[0], _batchDims);
}

/// <summary>ScatterND: a copy of data in which the slices selected by indices are replaced by (or, with reduction, combined with) updates.</summary>
internal sealed class ScatterNDOp : OpKernel
{
    public static readonly OpDef Def = new("ScatterND", 11, int.MaxValue, static n => new ScatterNDOp(n));

    private readonly int _reduction;

    private ScatterNDOp(NodeInfo node)
    {
        _reduction = node.GetString("reduction", "none") switch
        {
            "none" => 0,
            "add" => 1,
            "mul" => 2,
            "max" => 3,
            "min" => 4,
            var other => throw node.Error($"unknown reduction '{other}'"),
        };
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 3 || !ctx.HasInput(0) || !ctx.HasInput(1) || !ctx.HasInput(2))
        {
            throw ctx.Fail("expects data, indices and updates");
        }

        Shape d = ctx.InputShape(0), ix = ctx.InputShape(1), u = ctx.InputShape(2);
        if (ctx.InputType(1) != DType.I64)
        {
            throw ctx.Fail($"indices must be int64, not {ctx.InputType(1)}");
        }

        if (ctx.InputType(2) != ctx.InputType(0))
        {
            throw ctx.Fail($"updates are {ctx.InputType(2)} but data is {ctx.InputType(0)}");
        }

        if (_reduction != 0 && ctx.InputType(0) is not (DType.F32 or DType.F64 or DType.I32 or DType.I64))
        {
            throw ctx.Unsupported($"reduction on element type {ctx.InputType(0)}");
        }

        if (ix.Rank < 1 || d.Rank < 1)
        {
            throw ctx.Fail($"data {d} and indices {ix} must have rank at least 1");
        }

        int k = ix[ix.Rank - 1];
        if (k < 1 || k > d.Rank)
        {
            throw ctx.Fail($"the index tuples have {k} entries for data of rank {d.Rank}");
        }

        // updates.shape must be indices.shape[:-1] + data.shape[k:]
        int expectedRank = ix.Rank - 1 + d.Rank - k;
        bool ok = u.Rank == expectedRank;
        for (int i = 0; ok && i < ix.Rank - 1; i++)
        {
            ok = u[i] == ix[i];
        }

        for (int i = k; ok && i < d.Rank; i++)
        {
            ok = u[ix.Rank - 1 + i - k] == d[i];
        }

        if (!ok)
        {
            throw ctx.Fail($"updates {u} do not match indices {ix} and data {d}");
        }

        ctx.SetOutput(0, ctx.InputType(0), d);
    }

    public override void Run(in OpRun run) => Movement.ScatterND(run.Inputs[0], run.Inputs[1], run.Inputs[2], run.Outputs[0], _reduction);
}
