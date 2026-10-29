using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

// Operators that make tensors from shapes or indices: ConstantOfShape, Range, OneHot, Trilu, NonZero, TopK.

/// <summary>ConstantOfShape: a tensor of the shape given by the input, filled with the one element of the value attribute (default float 0).</summary>
internal sealed class ConstantOfShapeOp : OpKernel
{
    public static readonly OpDef Def = new("ConstantOfShape", 9, int.MaxValue, static n => new ConstantOfShapeOp(n));

    private readonly Tensor _value;

    private ConstantOfShapeOp(NodeInfo node)
    {
        IrAttr? a = node.Find("value");
        if (a is null)
        {
            _value = Tensor.Scalar(0f);
            return;
        }

        if (a.Type != IrAttributeType.Tensor)
        {
            throw node.Error($"attribute 'value' is {a.Type}, expected a tensor");
        }

        if (!a.Tensor.TryGetDType(out _))
        {
            throw node.Unsupported($"element type {DTypes.OnnxName(a.Tensor.DataType)} is not supported");
        }

        _value = a.Tensor.ToTensor();
        if (_value.Length != 1)
        {
            _value.Dispose();
            throw node.Error("the value attribute must hold exactly one element");
        }
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects a shape input");
        }

        if (ctx.InputType(0) != DType.I64 || ctx.InputShape(0).Rank != 1)
        {
            throw ctx.Fail($"the shape input must be a 1-D int64 tensor, not {ctx.InputType(0)}{ctx.InputShape(0)}");
        }

        if (!ctx.TryGetInputValue(0, out Tensor v))
        {
            ctx.RequireRunTimeValues();
            return;
        }

        ReadOnlySpan<long> dims = v.Data<long>();
        for (int i = 0; i < dims.Length; i++)
        {
            if (dims[i] < 0)
            {
                throw ctx.Fail($"extent {dims[i]} is negative");
            }
        }

        ctx.SetOutput(0, _value.DType, OpUtil.MakeShape(ctx, dims));
    }

    public override void Run(in OpRun run) => Movement.FillWith(run.Outputs[0], _value);

    public override void Dispose()
    {
        if (!_value.IsEmpty)
        {
            _value.Dispose();
        }

        base.Dispose();
    }
}

/// <summary>Range: start, start + delta, ... up to (not including) limit; ceil((limit - start) / delta) elements.</summary>
internal sealed class RangeOp : OpKernel
{
    public static readonly OpDef Def = new("Range", 11, int.MaxValue, static _ => new RangeOp());

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 3 || !ctx.HasInput(0) || !ctx.HasInput(1) || !ctx.HasInput(2))
        {
            throw ctx.Fail("expects start, limit and delta");
        }

        DType t = ctx.InputType(0);
        if (t is not (DType.F32 or DType.F64 or DType.F16 or DType.BF16 or DType.I16 or DType.I32 or DType.I64))
        {
            throw ctx.Unsupported($"element type {t} is not supported");
        }

        for (int i = 1; i < 3; i++)
        {
            if (ctx.InputType(i) != t)
            {
                throw ctx.Fail($"input {i} is {ctx.InputType(i)} but start is {t}");
            }
        }

        for (int i = 0; i < 3; i++)
        {
            if (ctx.InputShape(i).Length != 1 || ctx.InputShape(i).Rank > 1)
            {
                throw ctx.Fail($"input {i} must be a scalar, not {ctx.InputShape(i)}");
            }
        }

        if (!(ctx.TryGetInputValue(0, out Tensor a) & ctx.TryGetInputValue(1, out Tensor b) & ctx.TryGetInputValue(2, out Tensor c)))
        {
            ctx.RequireRunTimeValues();
            return;
        }

        long n = Count(a, b, c, out string? error);
        if (error is not null)
        {
            throw ctx.Fail(error);
        }

        ctx.SetOutput(0, t, OpUtil.MakeShape(ctx, [n]));
    }

    private static long Count(in Tensor start, in Tensor limit, in Tensor delta, out string? error)
    {
        error = null;
        switch (start.DType)
        {
            case DType.F32:
            {
                float s = start.Data<float>()[0], l = limit.Data<float>()[0], d = delta.Data<float>()[0];
                return CountFloat(((double)l - s) / d, d == 0 || float.IsNaN(l - s) || double.IsInfinity(((double)l - s) / d), out error);
            }

            case DType.F16 or DType.BF16:
            {
                // Half-precision ranges are computed in float and rounded on output.
                float s = HalfToSingle(start), l = HalfToSingle(limit), d = HalfToSingle(delta);
                return CountFloat(((double)l - s) / d, d == 0 || float.IsNaN(l - s) || double.IsInfinity(((double)l - s) / d), out error);
            }

            case DType.F64:
            {
                double s = start.Data<double>()[0], l = limit.Data<double>()[0], d = delta.Data<double>()[0];
                return CountFloat((l - s) / d, d == 0 || double.IsNaN(l - s) || double.IsInfinity((l - s) / d), out error);
            }

            case DType.I16:
                return CountInt(start.Data<short>()[0], limit.Data<short>()[0], delta.Data<short>()[0], out error);
            case DType.I32:
                return CountInt(start.Data<int>()[0], limit.Data<int>()[0], delta.Data<int>()[0], out error);
            default:
                return CountInt(start.Data<long>()[0], limit.Data<long>()[0], delta.Data<long>()[0], out error);
        }
    }

    private static float HalfToSingle(in Tensor t) =>
        t.DType == DType.F16 ? Float16.ToSingle(t.RawData<ushort>()[0]) : BFloat16.ToSingleBits(t.RawData<ushort>()[0]);

    private static long CountFloat(double quotient, bool invalid, out string? error)
    {
        error = invalid ? "delta is 0 or the range is not finite" : null;
        return invalid ? 0 : (long)Math.Max(Math.Ceiling(quotient), 0);
    }

    private static long CountInt(long s, long l, long d, out string? error)
    {
        error = d == 0 ? "delta is 0" : null;
        if (d == 0)
        {
            return 0;
        }

        // ceil(span / |d|) for a span in d's direction, in 128 bits so int64 extremes cannot overflow.
        if (d > 0)
        {
            return l > s ? (long)((((Int128)l - s) + d - 1) / d) : 0;
        }

        return s > l ? (long)((((Int128)s - l) - d - 1) / -(Int128)d) : 0;
    }

    public override void Run(in OpRun run)
    {
        Tensor y = run.Outputs[0];
        switch (y.DType)
        {
            case DType.F32: Movement.Range(y.MutableData<float>(), run.Inputs[0].Data<float>()[0], run.Inputs[2].Data<float>()[0]); break;
            case DType.F64: Movement.Range(y.MutableData<double>(), run.Inputs[0].Data<double>()[0], run.Inputs[2].Data<double>()[0]); break;
            case DType.F16 or DType.BF16:
            {
                float start = HalfToSingle(run.Inputs[0]), delta = HalfToSingle(run.Inputs[2]);
                Span<ushort> o = y.RawMutableData<ushort>();
                for (int i = 0; i < o.Length; i++)
                {
                    float v = start + (i * delta);
                    o[i] = y.DType == DType.F16 ? Float16.FromSingle(v) : BFloat16.FromSingleBits(v);
                }

                break;
            }

            case DType.I16: Movement.Range(y.MutableData<short>(), run.Inputs[0].Data<short>()[0], run.Inputs[2].Data<short>()[0]); break;
            case DType.I32: Movement.Range(y.MutableData<int>(), run.Inputs[0].Data<int>()[0], run.Inputs[2].Data<int>()[0]); break;
            default: Movement.Range(y.MutableData<long>(), run.Inputs[0].Data<long>()[0], run.Inputs[2].Data<long>()[0]); break;
        }
    }
}

/// <summary>OneHot: the values input's "on" value where an index equals the position along a new depth axis, its "off" value elsewhere.</summary>
internal sealed class OneHotOp : OpKernel
{
    public static readonly OpDef Def = new("OneHot", 9, int.MaxValue, static n => new OneHotOp(n));

    private readonly long _axis;

    private OneHotOp(NodeInfo node)
    {
        _axis = node.GetInt("axis", -1);
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 3 || !ctx.HasInput(0) || !ctx.HasInput(1) || !ctx.HasInput(2))
        {
            throw ctx.Fail("expects indices, depth and values");
        }

        if (ctx.InputType(0) is not (DType.I64 or DType.I32 or DType.F32 or DType.F64 or DType.I16 or DType.I8 or DType.U8))
        {
            throw ctx.Unsupported($"indices of element type {ctx.InputType(0)}");
        }

        if (ctx.InputShape(1).Length != 1 || ctx.InputShape(1).Rank > 1)
        {
            throw ctx.Fail($"depth must hold one value, not {ctx.InputShape(1)}");
        }

        if (ctx.InputShape(2).Rank != 1 || ctx.InputShape(2)[0] != 2)
        {
            throw ctx.Fail($"values must be [off, on], not {ctx.InputShape(2)}");
        }

        Shape ix = ctx.InputShape(0);
        int outRank = ix.Rank + 1;
        if (outRank > Shape.MaxRank)
        {
            throw ctx.Fail($"the output rank {outRank} is above {Shape.MaxRank}");
        }

        int axis = OpUtil.Axis(ctx, _axis, outRank);
        if (!ctx.TryGetInputValue(1, out Tensor depthTensor))
        {
            ctx.RequireRunTimeValues();
            return;
        }

        long depth = OpUtil.ScalarAsLong(depthTensor);
        if (depth < 0)
        {
            throw ctx.Fail($"depth {depth} is negative");
        }

        Span<long> dims = stackalloc long[Shape.MaxRank];
        for (int o = 0, i = 0; o < outRank; o++)
        {
            dims[o] = o == axis ? depth : ix[i++];
        }

        ctx.SetOutput(0, ctx.InputType(2), OpUtil.MakeShape(ctx, dims.Slice(0, outRank)));
    }

    public override void Run(in OpRun run)
    {
        Tensor ix = run.Inputs[0], y = run.Outputs[0];
        int axis = (int)(_axis < 0 ? _axis + y.Rank : _axis);
        Movement.OneHot(ix, run.Inputs[2], y, OpUtil.Product(ix.Shape.Dimensions, 0, axis), y.Shape[axis], OpUtil.Product(ix.Shape.Dimensions, axis, ix.Rank));
    }
}

/// <summary>Trilu: the upper (default) or lower triangle of the last two axes relative to diagonal k, zeros elsewhere.</summary>
internal sealed class TriluOp : OpKernel
{
    public static readonly OpDef Def = new("Trilu", 14, int.MaxValue, static n => new TriluOp(n));

    private readonly bool _upper;

    private TriluOp(NodeInfo node)
    {
        _upper = node.GetInt("upper", 1) != 0;
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        if (ctx.InputShape(0).Rank < 2)
        {
            throw ctx.Fail($"the input {ctx.InputShape(0)} must have at least 2 axes");
        }

        if (ctx.HasInput(1))
        {
            if (ctx.InputType(1) != DType.I64 || ctx.InputShape(1).Length != 1)
            {
                throw ctx.Fail("k must be one int64 value");
            }
        }

        ctx.SetOutput(0, ctx.InputType(0), ctx.InputShape(0));
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0];
        long k = run.Inputs.Length > 1 && !run.Inputs[1].IsEmpty ? run.Inputs[1].Data<long>()[0] : 0;
        int rows = x.Shape[x.Rank - 2], cols = x.Shape[x.Rank - 1];
        if (x.Length == 0)
        {
            return;
        }

        Movement.Trilu(x, run.Outputs[0], x.Length / (rows * cols), rows, cols, k, _upper);
    }
}

/// <summary>NonZero: the coordinates of the non-zero elements, row-major, as an int64 [rank, count] tensor.</summary>
internal sealed class NonZeroOp : OpKernel
{
    public static readonly OpDef Def = new("NonZero", 9, int.MaxValue, static _ => new NonZeroOp());

    public override bool SupportsExact => true;

    public override bool OutputsDataDependent => true;

    public override void Infer(InferContext ctx) => throw new InvalidOperationException("NonZero is shaped when it runs.");

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0];
        int count = Movement.CountNonZero(x);
        Tensor y = run.AllocateOutput(0, DType.I64, Shape.Create(Math.Max(x.Rank, 1), count));
        Movement.NonZero(x, y);
    }
}

/// <summary>
/// TopK: the k largest (or smallest) elements along an axis with their indices, best first; ties keep input order. k is an
/// attribute before opset 10 and an input from then on.
/// </summary>
internal sealed class TopKOp : OpKernel
{
    public static readonly OpDef Def = new("TopK", 1, int.MaxValue, static n => new TopKOp(n));

    private readonly bool _kIsInput;
    private readonly long _kAttribute;
    private readonly long _axis;
    private readonly bool _largest;

    private TopKOp(NodeInfo node)
    {
        _kIsInput = node.Opset >= 10;
        _kAttribute = _kIsInput ? 0 : node.GetInt("k", -1);
        _axis = node.GetInt("axis", -1);
        _largest = node.GetInt("largest", 1) != 0;
        if (!_kIsInput && !node.Has("k"))
        {
            throw node.Error("has no k attribute");
        }
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects data");
        }

        DType t = ctx.InputType(0);
        if (t is DType.F16 or DType.BF16 or DType.Bool or DType.Undefined)
        {
            throw ctx.Unsupported($"element type {t}");
        }

        Shape s = ctx.InputShape(0);
        if (s.Rank == 0)
        {
            throw ctx.Fail("cannot select from a scalar");
        }

        int axis = OpUtil.Axis(ctx, _axis, s.Rank);
        long k = _kAttribute;
        if (_kIsInput)
        {
            OpUtil.RequireIntegers(ctx, 1, "k");
            if (ctx.InputShape(1).Length != 1)
            {
                throw ctx.Fail("k must hold one value");
            }

            if (!ctx.TryGetInputValue(1, out Tensor kt))
            {
                ctx.RequireRunTimeValues();
                return;
            }

            k = OpUtil.First(kt);
        }

        if (k < 0 || k > s[axis])
        {
            throw ctx.Fail($"k = {k} is outside [0, {s[axis]}] for axis {axis}");
        }

        Shape o = s.With(axis, (int)k);
        ctx.SetOutput(0, t, o);
        if (ctx.OutputCount > 1)
        {
            ctx.SetOutput(1, DType.I64, o);
        }
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], values = run.Outputs[0];
        int axis = (int)(_axis < 0 ? _axis + x.Rank : _axis);
        ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int n = dims[axis];
        int k = values.Shape[axis];
        if (k == 0 || x.Length == 0)
        {
            return;
        }

        // The indices output is part of the node even when nobody reads it.
        Tensor indices = run.Outputs.Length > 1 && !run.Outputs[1].IsEmpty ? run.Outputs[1] : run.Scratch(DType.I64, values.Shape);
        Tensor scratch = run.Scratch(DType.I32, Shape.Create(2 * n));
        Movement.TopK(x, values, indices, OpUtil.Product(dims, 0, axis), n, OpUtil.Product(dims, axis + 1, dims.Length), k, _largest,
            scratch.MutableData<int>());
    }
}
