using System;
using System.Numerics;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Scatter;

/// <summary>
/// ScatterElements (opset 11 on) and the older Scatter (opset 9 to 10): the output is a copy of data in which, for every position p of
/// indices, the element whose coordinate along the axis is indices[p] (negative values count from the end) and whose other
/// coordinates are p's takes updates[p], combined with the old value by the reduction (none, add, mul, max or min). Positions
/// run in increasing order, so with duplicate indices the last update wins under none and the combination is applied in that order.
/// </summary>
internal sealed class ScatterElementsOp : OpKernel
{
    public static readonly OpDef Def = new("ScatterElements", 11, int.MaxValue, static n => new ScatterElementsOp(n));
    public static readonly OpDef Legacy = new("Scatter", 9, 10, static n => new ScatterElementsOp(n));

    private enum Reduction { None, Add, Mul, Max, Min }

    private readonly long _axis;
    private readonly Reduction _reduction;

    private ScatterElementsOp(NodeInfo n)
    {
        _axis = n.GetInt("axis", 0);
        _reduction = n.GetString("reduction", "none") switch
        {
            "none" => Reduction.None,
            "add" => Reduction.Add,
            "mul" => Reduction.Mul,
            "max" => Reduction.Max,
            "min" => Reduction.Min,
            var other => throw n.Error($"attribute 'reduction' is '{other}', expected none, add, mul, max or min"),
        };
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 3 || !ctx.HasInput(0) || !ctx.HasInput(1) || !ctx.HasInput(2))
        {
            throw ctx.Fail("expects data, indices and updates");
        }

        Shape d = ctx.InputShape(0), ix = ctx.InputShape(1);
        if (ctx.InputType(1) is not (DType.I32 or DType.I64))
        {
            throw ctx.Fail($"indices must be int32 or int64, not {ctx.InputType(1)}");
        }

        if (ctx.InputType(2) != ctx.InputType(0))
        {
            throw ctx.Fail($"updates are {ctx.InputType(2)} but data is {ctx.InputType(0)}");
        }

        if (d.Rank == 0 || d.Rank != ix.Rank || ix != ctx.InputShape(2))
        {
            throw ctx.Fail($"data {d}, indices {ix} and updates {ctx.InputShape(2)} must have the same rank (at least 1) and indices must match updates");
        }

        int axis = OpUtil.Axis(ctx, _axis, d.Rank);
        for (int i = 0; i < d.Rank; i++)
        {
            if (i != axis && ix[i] > d[i])
            {
                throw ctx.Fail($"indices extent {ix[i]} exceeds data extent {d[i]} in axis {i}");
            }
        }

        if (_reduction != Reduction.None && !TypeDispatch.IsNumber(ctx.InputType(0)))
        {
            throw ctx.Unsupported($"reduction on element type {ctx.InputType(0)} is not supported");
        }

        if (!TypeDispatch.IsNumber(ctx.InputType(0)) && ctx.InputType(0) is not (DType.Bool or DType.BF16))
        {
            throw ctx.Unsupported($"element type {ctx.InputType(0)} is not supported");
        }

        ctx.SetOutput(0, ctx.InputType(0), d);
    }

    public override void Run(in OpRun run)
    {
        Tensor data = run.Inputs[0], y = run.Outputs[0];
        Copy(data, y);
        if (run.Inputs[1].Length == 0)
        {
            return;
        }

        int axis = (int)(_axis < 0 ? _axis + data.Rank : _axis);
        var job = new Job { Data = y, Indices = run.Inputs[1], Updates = run.Inputs[2], Axis = axis, Reduction = _reduction };
        if (TypeDispatch.IsNumber(data.DType))
        {
            TypeDispatch.Number(data.DType, ref job);
        }
        else
        {
            // bool and bfloat16 only move: the update replaces the old value.
            job.Move();
        }
    }

    private static void Copy(Tensor from, Tensor to)
    {
        switch (from.ElementSize)
        {
            case 1: from.RawData<byte>().CopyTo(to.RawMutableData<byte>()); break;
            case 2: from.RawData<ushort>().CopyTo(to.RawMutableData<ushort>()); break;
            case 4: from.RawData<uint>().CopyTo(to.RawMutableData<uint>()); break;
            default: from.RawData<ulong>().CopyTo(to.RawMutableData<ulong>()); break;
        }
    }

    private struct Job : INumberVisitor
    {
        public Tensor Data, Indices, Updates;
        public int Axis;
        public Reduction Reduction;

        public void Visit<T>() where T : unmanaged, INumber<T>
        {
            Span<T> y = Data.MutableData<T>();
            ReadOnlySpan<T> u = Updates.Data<T>();
            for (int p = 0; p < u.Length; p++)
            {
                int target = Target(p);
                T old = y[target], v = u[p];
                y[target] = Reduction switch
                {
                    Reduction.Add => unchecked(old + v),
                    Reduction.Mul => unchecked(old * v),
                    Reduction.Max => v > old || T.IsNaN(v) ? v : old,
                    Reduction.Min => v < old || T.IsNaN(v) ? v : old,
                    _ => v,
                };
            }
        }

        public void Move()
        {
            if (Data.ElementSize == 1)
            {
                Span<byte> y = Data.RawMutableData<byte>();
                ReadOnlySpan<byte> u = Updates.RawData<byte>();
                for (int p = 0; p < u.Length; p++)
                {
                    y[Target(p)] = u[p];
                }
            }
            else
            {
                Span<ushort> y = Data.RawMutableData<ushort>();
                ReadOnlySpan<ushort> u = Updates.RawData<ushort>();
                for (int p = 0; p < u.Length; p++)
                {
                    y[Target(p)] = u[p];
                }
            }
        }

        // The element offset in data that position p of indices points at: p's coordinates with the axis coordinate replaced.
        private readonly int Target(int p)
        {
            ReadOnlySpan<int> dd = Data.Shape.Dimensions, id = Indices.Shape.Dimensions;
            int axisExtent = dd[Axis];
            long k = IntVec.Of(Indices)[p];
            if (k < -axisExtent || k >= axisExtent)
            {
                throw new SessionException(SessionErrorKind.InvalidInput, $"ScatterElements: index {k} at position {p} is outside [{-axisExtent}, {axisExtent})");
            }

            if (k < 0)
            {
                k += axisExtent;
            }

            int target = 0, stride = 1, rest = p;
            for (int d = dd.Length - 1; d >= 0; d--)
            {
                int coord = d == Axis ? (int)k : rest % id[d];
                rest /= id[d];
                target += coord * stride;
                stride *= dd[d];
            }

            return target;
        }
    }
}

/// <summary>
/// Compress: the slices of the input along an axis (the flattened input when there is no axis attribute) at which the 1-D bool
/// condition is true; a condition shorter than the axis keeps nothing beyond its end.
/// </summary>
internal sealed class CompressOp : OpKernel
{
    public static readonly OpDef Def = new("Compress", 9, int.MaxValue, static n => new CompressOp(n));

    private readonly bool _hasAxis;
    private readonly long _axis;

    private CompressOp(NodeInfo n)
    {
        _hasAxis = n.Has("axis");
        _axis = n.GetInt("axis", 0);
    }

    public override bool SupportsExact => true;

    public override bool OutputsDataDependent => true;

    public override void Infer(InferContext ctx) => throw new InvalidOperationException("Compress is shaped when it runs.");

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], cond = run.Inputs[1];
        if (cond.DType != DType.Bool || cond.Rank != 1)
        {
            throw new SessionException(SessionErrorKind.InvalidInput, "Compress: the condition must be a 1-D bool tensor");
        }

        ReadOnlySpan<bool> c = cond.Data<bool>();
        int outer, axisLen, inner;
        Shape outShapeBase = x.Shape;
        int axis = 0;
        if (_hasAxis)
        {
            if (_axis < -x.Rank || _axis >= x.Rank)
            {
                throw new SessionException(SessionErrorKind.InvalidInput, $"Compress: axis {_axis} is outside the {x.Rank} dimensions of the input");
            }

            axis = (int)(_axis < 0 ? _axis + x.Rank : _axis);
            outer = OpUtil.Product(x.Shape.Dimensions, 0, axis);
            axisLen = x.Shape[axis];
            inner = OpUtil.Product(x.Shape.Dimensions, axis + 1, x.Rank);
        }
        else
        {
            outer = 1;
            axisLen = x.Length;
            inner = 1;
        }

        int limit = Math.Min(axisLen, c.Length);
        int kept = 0;
        for (int i = 0; i < limit; i++)
        {
            if (c[i])
            {
                kept++;
            }
        }

        Shape os;
        if (_hasAxis)
        {
            Span<int> dims = stackalloc int[Shape.MaxRank];
            x.Shape.Dimensions.CopyTo(dims);
            dims[axis] = kept;
            os = Shape.Create(dims.Slice(0, x.Rank));
        }
        else
        {
            os = Shape.Create(kept);
        }

        Tensor y = run.AllocateOutput(0, x.DType, os);
        switch (x.ElementSize)
        {
            case 1: Pack(x.RawData<byte>(), y.RawMutableData<byte>(), c, outer, axisLen, inner, limit); break;
            case 2: Pack(x.RawData<ushort>(), y.RawMutableData<ushort>(), c, outer, axisLen, inner, limit); break;
            case 4: Pack(x.RawData<uint>(), y.RawMutableData<uint>(), c, outer, axisLen, inner, limit); break;
            default: Pack(x.RawData<ulong>(), y.RawMutableData<ulong>(), c, outer, axisLen, inner, limit); break;
        }
    }

    private static void Pack<T>(ReadOnlySpan<T> x, Span<T> y, ReadOnlySpan<bool> c, int outer, int axisLen, int inner, int limit) where T : unmanaged
    {
        int o = 0;
        for (int a = 0; a < outer; a++)
        {
            for (int i = 0; i < limit; i++)
            {
                if (c[i])
                {
                    x.Slice(((a * axisLen) + i) * inner, inner).CopyTo(y.Slice(o, inner));
                    o += inner;
                }
            }
        }
    }
}
