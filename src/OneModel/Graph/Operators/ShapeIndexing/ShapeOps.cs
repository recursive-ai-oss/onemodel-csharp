using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

// Operators that read or reinterpret shapes only. Shape and Size are constants of the plan; Squeeze, Unsqueeze, Flatten and
// (without a mask output) Dropout are views, so none of them moves a byte.

/// <summary>Shape: the input's extents as an int64 vector, optionally the slice [start, end) of them (opset 15).</summary>
internal sealed class ShapeOp : OpKernel
{
    public static readonly OpDef Def = new("Shape", 1, int.MaxValue, static n => new ShapeOp(n));

    private readonly long _start;
    private readonly long? _end;

    private ShapeOp(NodeInfo node)
    {
        _start = node.GetInt("start", 0);
        _end = node.Has("end") ? node.GetInt("end", 0) : null;
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        Shape s = ctx.InputShape(0);
        int rank = s.Rank;
        long start = Clamp(_start, rank);
        long end = Clamp(_end ?? rank, rank);
        int n = (int)Math.Max(end - start, 0);
        long[] dims = new long[n];
        for (int i = 0; i < n; i++)
        {
            dims[i] = s[(int)start + i];
        }

        ctx.SetConstantOutput(0, Tensor.FromSpan<long>(dims, Shape.Create(n)), planOwns: true);
    }

    // Negative values count from the end; both ends clamp to [0, rank].
    private static long Clamp(long v, int rank) => Math.Clamp(v < 0 ? v + rank : v, 0, rank);

    public override void Run(in OpRun run) => throw new InvalidOperationException("Shape is evaluated when the plan is made; it never runs.");
}

/// <summary>Size: the number of elements of the input, as an int64 scalar.</summary>
internal sealed class SizeOp : OpKernel
{
    public static readonly OpDef Def = new("Size", 1, int.MaxValue, static _ => new SizeOp());

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        ctx.SetConstantOutput(0, Tensor.Scalar<long>(ctx.InputShape(0).Length), planOwns: true);
    }

    public override void Run(in OpRun run) => throw new InvalidOperationException("Size is evaluated when the plan is made; it never runs.");
}

/// <summary>Squeeze: removes extent-1 dimensions, the listed axes (attribute before opset 13, input from 13) or all of them.</summary>
internal sealed class SqueezeOp : OpKernel
{
    public static readonly OpDef Def = new("Squeeze", 1, int.MaxValue, static n => new SqueezeOp(n));

    private readonly bool _axesAreInput;
    private readonly long[]? _axesAttribute;

    private SqueezeOp(NodeInfo node)
    {
        _axesAreInput = node.Opset >= 13;
        _axesAttribute = !_axesAreInput && node.Has("axes") ? [.. node.GetInts("axes")] : null;
    }

    public override bool SupportsExact => true;

    public override int ViewOfInput => 0;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects data");
        }

        Shape s = ctx.InputShape(0);
        int rank = s.Rank;
        Span<bool> drop = stackalloc bool[Shape.MaxRank];
        if (_axesAreInput && ctx.HasInput(1))
        {
            OpUtil.RequireIntegers(ctx, 1, "axes");
            if (!ctx.TryGetInputValue(1, out Tensor v))
            {
                ctx.RequireRunTimeValues();
                return;
            }

            IntVec axes = IntVec.Of(v);
            for (int i = 0; i < axes.Length; i++)
            {
                Mark(ctx, s, drop, axes[i]);
            }
        }
        else if (_axesAttribute is not null)
        {
            foreach (long a in _axesAttribute)
            {
                Mark(ctx, s, drop, a);
            }
        }
        else
        {
            for (int i = 0; i < rank; i++)
            {
                drop[i] = s[i] == 1;
            }
        }

        Span<long> dims = stackalloc long[Shape.MaxRank];
        int n = 0;
        for (int i = 0; i < rank; i++)
        {
            if (!drop[i])
            {
                dims[n++] = s[i];
            }
        }

        ctx.SetOutput(0, ctx.InputType(0), OpUtil.MakeShape(ctx, dims.Slice(0, n)));
    }

    private static void Mark(InferContext ctx, Shape s, Span<bool> drop, long axis)
    {
        int a = OpUtil.Axis(ctx, axis, s.Rank);
        if (s[a] != 1)
        {
            throw ctx.Fail($"cannot squeeze axis {a}: its extent is {s[a]}, not 1");
        }

        if (drop[a])
        {
            throw ctx.Fail($"axis {a} is listed twice");
        }

        drop[a] = true;
    }

    public override void Run(in OpRun run) => throw new InvalidOperationException("Squeeze is a view; the session builds it.");
}

/// <summary>Unsqueeze: inserts extent-1 dimensions at the listed axes of the output (attribute before opset 13, input from 13).</summary>
internal sealed class UnsqueezeOp : OpKernel
{
    public static readonly OpDef Def = new("Unsqueeze", 1, int.MaxValue, static n => new UnsqueezeOp(n));

    private readonly bool _axesAreInput;
    private readonly long[] _axesAttribute;

    private UnsqueezeOp(NodeInfo node)
    {
        _axesAreInput = node.Opset >= 13;
        _axesAttribute = [.. node.GetInts("axes")];
        if (!_axesAreInput && !node.Has("axes"))
        {
            throw node.Error("has no axes attribute");
        }
    }

    public override bool SupportsExact => true;

    public override int ViewOfInput => 0;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects data");
        }

        Shape s = ctx.InputShape(0);
        Span<long> axes = stackalloc long[Shape.MaxRank + 1];
        int count;
        if (_axesAreInput)
        {
            OpUtil.RequireIntegers(ctx, 1, "axes");
            if (!ctx.TryGetInputValue(1, out Tensor v))
            {
                ctx.RequireRunTimeValues();
                return;
            }

            IntVec iv = IntVec.Of(v);
            count = iv.Length;
            if (count > Shape.MaxRank)
            {
                throw ctx.Fail($"{count} axes would give a rank above {Shape.MaxRank}");
            }

            for (int i = 0; i < count; i++)
            {
                axes[i] = iv[i];
            }
        }
        else
        {
            count = _axesAttribute.Length;
            if (count > Shape.MaxRank)
            {
                throw ctx.Fail($"{count} axes would give a rank above {Shape.MaxRank}");
            }

            _axesAttribute.CopyTo(axes);
        }

        int outRank = s.Rank + count;
        if (outRank > Shape.MaxRank)
        {
            throw ctx.Fail($"the output rank {outRank} is above {Shape.MaxRank}");
        }

        Span<bool> inserted = stackalloc bool[Shape.MaxRank];
        for (int i = 0; i < count; i++)
        {
            int a = OpUtil.Axis(ctx, axes[i], outRank);
            if (inserted[a])
            {
                throw ctx.Fail($"axis {a} is listed twice");
            }

            inserted[a] = true;
        }

        Span<long> dims = stackalloc long[Shape.MaxRank];
        for (int o = 0, src = 0; o < outRank; o++)
        {
            dims[o] = inserted[o] ? 1 : s[src++];
        }

        ctx.SetOutput(0, ctx.InputType(0), OpUtil.MakeShape(ctx, dims.Slice(0, outRank)));
    }

    public override void Run(in OpRun run) => throw new InvalidOperationException("Unsqueeze is a view; the session builds it.");
}

/// <summary>Flatten: a 2-D view [product of dims before axis, product of the rest]. Axis 0 gives [1, size].</summary>
internal sealed class FlattenOp : OpKernel
{
    public static readonly OpDef Def = new("Flatten", 1, int.MaxValue, static n => new FlattenOp(n));

    private readonly long _axis;

    private FlattenOp(NodeInfo node)
    {
        _axis = node.GetInt("axis", 1);
    }

    public override bool SupportsExact => true;

    public override int ViewOfInput => 0;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        Shape s = ctx.InputShape(0);
        int rank = s.Rank;
        if (_axis < -rank || _axis > rank)
        {
            throw ctx.Fail($"axis {_axis} is outside [{-rank}, {rank}] for a rank-{rank} input");
        }

        int axis = (int)(_axis < 0 ? _axis + rank : _axis);
        int outer = OpUtil.Product(s.Dimensions, 0, axis);
        int inner = OpUtil.Product(s.Dimensions, axis, rank);
        ctx.SetOutput(0, ctx.InputType(0), Shape.Create(outer, inner));
    }

    public override void Run(in OpRun run) => throw new InvalidOperationException("Flatten is a view; the session builds it.");
}

/// <summary>
/// Dropout in inference: the output is the input (a view), and the optional mask output is all true. Training mode (a true
/// <c>training_mode</c> input) is not supported: it needs random numbers.
/// </summary>
internal sealed class DropoutOp : OpKernel
{
    public static readonly OpDef Def = new("Dropout", 1, int.MaxValue, static n => new DropoutOp(n));

    private readonly bool _hasMask;

    private DropoutOp(NodeInfo node)
    {
        _hasMask = node.Ir.OutputNames.Length > 1 && node.Ir.OutputNames[1].Length > 0;
    }

    public override bool SupportsExact => true;

    // With a mask output the node must run to fill it; without one the output is the input.
    public override int ViewOfInput => _hasMask ? -1 : 0;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects data");
        }

        if (ctx.HasInput(2))
        {
            // A training_mode fed at run time is read when the node is shaped for that run, so a true value is refused, not ignored.
            if (!ctx.TryGetInputValue(2, out Tensor mode))
            {
                ctx.RequireRunTimeValues();
                return;
            }

            if (mode.RawData<byte>()[0] != 0)
            {
                throw ctx.Unsupported("training mode is not supported");
            }
        }

        ctx.SetOutput(0, ctx.InputType(0), ctx.InputShape(0));
        if (_hasMask)
        {
            ctx.SetOutput(1, DType.Bool, ctx.InputShape(0));
        }
    }

    public override void Run(in OpRun run)
    {
        run.Outputs[0].CopyFrom(run.Inputs[0]);
        run.Outputs[1].RawMutableBytes().Fill(1);
    }
}
