using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

// Operators that copy elements to new positions with an affine map (strides): Transpose, Concat, Split, Slice, Expand, Tile. Each one
// states its strides and calls Movement.Move, which merges axes, uses memmove for contiguous runs and tiles real transposes.

/// <summary>Transpose: permutes the axes (default: reverses them) and copies into row-major order.</summary>
internal sealed class TransposeOp : OpKernel
{
    public static readonly OpDef Def = new("Transpose", 1, int.MaxValue, static n => new TransposeOp(n));

    private readonly long[]? _perm;

    private TransposeOp(NodeInfo node)
    {
        _perm = node.Has("perm") ? [.. node.GetInts("perm")] : null;
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
        if (_perm is not null && _perm.Length != rank)
        {
            throw ctx.Fail($"perm has {_perm.Length} entries for a rank-{rank} input");
        }

        Span<long> dims = stackalloc long[Shape.MaxRank];
        Span<bool> seen = stackalloc bool[Shape.MaxRank];
        for (int k = 0; k < rank; k++)
        {
            long p = _perm is null ? rank - 1 - k : _perm[k];
            if (p < 0 || p >= rank || seen[(int)p])
            {
                throw ctx.Fail("perm is not a permutation of the input's axes");
            }

            seen[(int)p] = true;
            dims[k] = s[(int)p];
        }

        ctx.SetOutput(0, ctx.InputType(0), Shape.Create((ReadOnlySpan<long>)dims.Slice(0, rank)));
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], y = run.Outputs[0];
        int rank = x.Rank;
        Span<int> inStr = stackalloc int[Shape.MaxRank];
        Span<int> srcStr = stackalloc int[Shape.MaxRank];
        Span<int> dstStr = stackalloc int[Shape.MaxRank];
        OpUtil.RowMajor(x.Shape.Dimensions, inStr);
        OpUtil.RowMajor(y.Shape.Dimensions, dstStr);
        for (int k = 0; k < rank; k++)
        {
            srcStr[k] = inStr[_perm is null ? rank - 1 - k : (int)_perm[k]];
        }

        Movement.Move(x, 0, srcStr.Slice(0, rank), y, 0, dstStr.Slice(0, rank), y.Shape.Dimensions);
    }
}

/// <summary>Concat: joins the inputs along an axis.</summary>
internal sealed class ConcatOp : OpKernel
{
    public static readonly OpDef Def = new("Concat", 1, int.MaxValue, static n => new ConcatOp(n));

    private readonly long _axis;

    private ConcatOp(NodeInfo node)
    {
        // The axis became required in opset 4; before that it defaulted to 1.
        _axis = node.Opset < 4 ? node.GetInt("axis", 1) : node.Has("axis") ? node.GetInt("axis", 0) : throw node.Error("has no axis attribute");
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount == 0 || !ctx.HasInput(0))
        {
            throw ctx.Fail("expects at least one input");
        }

        Shape first = ctx.InputShape(0);
        DType t = ctx.InputType(0);
        int rank = first.Rank;
        if (rank == 0)
        {
            throw ctx.Fail("cannot join scalars");
        }

        int axis = OpUtil.Axis(ctx, _axis, rank);
        long total = 0;
        for (int i = 0; i < ctx.InputCount; i++)
        {
            if (!ctx.HasInput(i))
            {
                throw ctx.Fail($"input {i} is missing");
            }

            Shape s = ctx.InputShape(i);
            if (ctx.InputType(i) != t)
            {
                throw ctx.Fail($"input {i} is {ctx.InputType(i)} but input 0 is {t}");
            }

            if (s.Rank != rank)
            {
                throw ctx.Fail($"input {i} has rank {s.Rank} but input 0 has rank {rank}");
            }

            for (int d = 0; d < rank; d++)
            {
                if (d != axis && s[d] != first[d])
                {
                    throw ctx.Fail($"input {i} {s} differs from input 0 {first} in axis {d}");
                }
            }

            total += s[axis];
        }

        Span<long> dims = stackalloc long[Shape.MaxRank];
        for (int d = 0; d < rank; d++)
        {
            dims[d] = d == axis ? total : first[d];
        }

        ctx.SetOutput(0, t, OpUtil.MakeShape(ctx, dims.Slice(0, rank)));
    }

    public override void Run(in OpRun run)
    {
        Tensor y = run.Outputs[0];
        int axis = (int)(_axis < 0 ? _axis + y.Rank : _axis);
        int outer = OpUtil.Product(y.Shape.Dimensions, 0, axis);
        int inner = OpUtil.Product(y.Shape.Dimensions, axis + 1, y.Rank);
        int rowOut = y.Shape[axis] * inner;
        int at = 0;
        Span<int> dims = [outer, 0];
        Span<int> srcStr = [0, 1];
        Span<int> dstStr = [rowOut, 1];
        for (int i = 0; i < run.Inputs.Length; i++)
        {
            Tensor x = run.Inputs[i];
            int row = x.Shape[axis] * inner;
            dims[1] = row;
            srcStr[0] = row;
            Movement.Move(x, 0, srcStr, y, at * inner, dstStr, dims);
            at += x.Shape[axis];
        }
    }
}

/// <summary>Split: cuts the input along an axis into consecutive parts (sizes from an attribute or input, or equal).</summary>
internal sealed class SplitOp : OpKernel
{
    public static readonly OpDef Def = new("Split", 1, int.MaxValue, static n => new SplitOp(n));

    private readonly long _axis;
    private readonly long[]? _splitAttribute;
    private readonly bool _splitIsInput;
    private readonly long _numOutputs;
    private readonly bool _hasNumOutputs;
    private readonly bool _uneven; // opset 18: the last part may be smaller

    private SplitOp(NodeInfo node)
    {
        _axis = node.GetInt("axis", 0);
        _splitIsInput = node.Opset >= 13;
        _splitAttribute = !_splitIsInput && node.Has("split") ? [.. node.GetInts("split")] : null;
        _hasNumOutputs = node.Has("num_outputs");
        _numOutputs = _hasNumOutputs ? node.GetInt("num_outputs", 0) : node.OutputCount;
        _uneven = node.Opset >= 18;
        if (_numOutputs < 1)
        {
            throw node.Error("num_outputs must be at least 1");
        }
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects data");
        }

        Shape s = ctx.InputShape(0);
        int rank = s.Rank;
        if (rank == 0)
        {
            throw ctx.Fail("cannot split a scalar");
        }

        int axis = OpUtil.Axis(ctx, _axis, rank);
        int dim = s[axis];
        int parts = ctx.OutputCount;
        long[] sizes = new long[parts];

        if (_splitIsInput && ctx.HasInput(1))
        {
            OpUtil.RequireIntegers(ctx, 1, "split");
            if (!ctx.TryGetInputValue(1, out Tensor v))
            {
                ctx.RequireRunTimeValues();
                return;
            }

            IntVec iv = IntVec.Of(v);
            if (iv.Length != parts)
            {
                throw ctx.Fail($"split lists {iv.Length} sizes for {parts} outputs");
            }

            for (int i = 0; i < parts; i++)
            {
                sizes[i] = iv[i];
            }
        }
        else if (_splitAttribute is not null)
        {
            if (_splitAttribute.Length != parts)
            {
                throw ctx.Fail($"split lists {_splitAttribute.Length} sizes for {parts} outputs");
            }

            _splitAttribute.CopyTo(sizes, 0);
        }
        else if (_uneven)
        {
            if (_numOutputs != parts)
            {
                throw ctx.Fail($"num_outputs is {_numOutputs} but the node has {parts} outputs");
            }

            // Opset 18: parts of ceil(dim / n), the last one takes what is left.
            long chunk = (dim + parts - 1) / parts;
            for (int i = 0; i < parts; i++)
            {
                sizes[i] = Math.Clamp(dim - (i * chunk), 0, chunk);
            }
        }
        else
        {
            if (dim % parts != 0)
            {
                throw ctx.Fail($"extent {dim} of axis {axis} does not divide into {parts} equal parts");
            }

            Array.Fill(sizes, dim / parts);
        }

        long sum = 0;
        Span<long> dims = stackalloc long[Shape.MaxRank];
        for (int i = 0; i < rank; i++)
        {
            dims[i] = s[i];
        }

        for (int i = 0; i < parts; i++)
        {
            if (sizes[i] < 0)
            {
                throw ctx.Fail($"split size {sizes[i]} is negative");
            }

            sum += sizes[i];
            dims[axis] = sizes[i];
            ctx.SetOutput(i, ctx.InputType(0), OpUtil.MakeShape(ctx, dims.Slice(0, rank)));
        }

        if (sum != dim)
        {
            throw ctx.Fail($"the split sizes add up to {sum} but axis {axis} has extent {dim}");
        }
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0];
        int axis = (int)(_axis < 0 ? _axis + x.Rank : _axis);
        int outer = OpUtil.Product(x.Shape.Dimensions, 0, axis);
        int inner = OpUtil.Product(x.Shape.Dimensions, axis + 1, x.Rank);
        int rowIn = x.Shape[axis] * inner;
        int at = 0;
        Span<int> dims = [outer, 0];
        Span<int> srcStr = [rowIn, 1];
        Span<int> dstStr = [0, 1];
        for (int i = 0; i < run.Outputs.Length; i++)
        {
            Tensor y = run.Outputs[i];
            int row = y.Shape[axis] * inner;
            dims[1] = row;
            dstStr[0] = row;
            Movement.Move(x, at * inner, srcStr, y, 0, dstStr, dims);
            at += y.Shape[axis];
        }
    }
}

/// <summary>Slice: a strided sub-range along chosen axes, with negative, out-of-range and negative-step bounds clamped as ONNX defines.</summary>
internal sealed class SliceOp : OpKernel
{
    public static readonly OpDef Def = new("Slice", 1, int.MaxValue, static n => new SliceOp(n));

    private readonly bool _fromInputs; // opset 10 and later
    private readonly long[] _starts;
    private readonly long[] _ends;
    private readonly long[]? _axes;

    private SliceOp(NodeInfo node)
    {
        _fromInputs = node.Opset >= 10;
        _starts = [];
        _ends = [];
        if (!_fromInputs)
        {
            if (!node.Has("starts") || !node.Has("ends"))
            {
                throw node.Error("needs starts and ends attributes");
            }

            _starts = [.. node.GetInts("starts")];
            _ends = [.. node.GetInts("ends")];
            _axes = node.Has("axes") ? [.. node.GetInts("axes")] : null;
        }
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects data");
        }

        if (_fromInputs)
        {
            if (ctx.InputCount < 3)
            {
                throw ctx.Fail("expects data, starts and ends");
            }

            OpUtil.RequireIntegers(ctx, 1, "starts");
            OpUtil.RequireIntegers(ctx, 2, "ends");
            bool known = ctx.TryGetInputValue(1, out Tensor st) & ctx.TryGetInputValue(2, out Tensor en);
            Tensor ax = default, sp = default;
            if (ctx.HasInput(3))
            {
                OpUtil.RequireIntegers(ctx, 3, "axes");
                known &= ctx.TryGetInputValue(3, out ax);
            }

            if (ctx.HasInput(4))
            {
                OpUtil.RequireIntegers(ctx, 4, "steps");
                known &= ctx.TryGetInputValue(4, out sp);
            }

            if (!known)
            {
                ctx.RequireRunTimeValues();
                return;
            }

            Finish(ctx, IntVec.Of(st), IntVec.Of(en), ctx.HasInput(3) ? IntVec.Of(ax) : default, ctx.HasInput(3),
                ctx.HasInput(4) ? IntVec.Of(sp) : default, ctx.HasInput(4));
        }
        else
        {
            Finish(ctx, new IntVec(_starts), new IntVec(_ends), _axes is null ? default : new IntVec(_axes), _axes is not null, default, false);
        }
    }

    private static void Finish(InferContext ctx, IntVec starts, IntVec ends, IntVec axes, bool hasAxes, IntVec steps, bool hasSteps)
    {
        Shape s = ctx.InputShape(0);
        Span<int> start = stackalloc int[Shape.MaxRank];
        Span<int> step = stackalloc int[Shape.MaxRank];
        Span<int> count = stackalloc int[Shape.MaxRank];
        string? error = Resolve(s, starts, ends, axes, hasAxes, steps, hasSteps, start, step, count);
        if (error is not null)
        {
            throw ctx.Fail(error);
        }

        ctx.SetOutput(0, ctx.InputType(0), Shape.Create(count.Slice(0, s.Rank)));
    }

    /// <summary>
    /// Turns starts, ends, axes and steps into a start, a step and an output extent per axis of <paramref name="s"/>. A negative
    /// start or end counts from the end; both clamp into the axis (to [0, n] for positive steps, [0, n-1] and [-1, n-1] for negative
    /// ones), so INT_MAX and INT64_MAX mean "to the end". Returns an error message, or null.
    /// </summary>
    private static string? Resolve(Shape s, IntVec starts, IntVec ends, IntVec axes, bool hasAxes, IntVec steps, bool hasSteps,
        Span<int> start, Span<int> step, Span<int> count)
    {
        int rank = s.Rank;
        for (int a = 0; a < rank; a++)
        {
            start[a] = 0;
            step[a] = 1;
            count[a] = s[a];
        }

        int n = starts.Length;
        if (ends.Length != n || (hasAxes && axes.Length != n) || (hasSteps && steps.Length != n))
        {
            return "starts, ends, axes and steps must have the same length";
        }

        if (n > rank)
        {
            return $"{n} slice axes for a rank-{rank} input";
        }

        Span<bool> done = stackalloc bool[Shape.MaxRank];
        for (int i = 0; i < n; i++)
        {
            long axis = hasAxes ? axes[i] : i;
            if (axis < -rank || axis >= rank)
            {
                return $"axis {axis} is outside [{-rank}, {rank - 1}]";
            }

            int a = (int)(axis < 0 ? axis + rank : axis);
            if (done[a])
            {
                return $"axis {a} is sliced twice";
            }

            done[a] = true;
            long st = hasSteps ? steps[i] : 1;
            if (st == 0)
            {
                return "a step is 0";
            }

            // A step beyond the axis length selects at most one element, so bounding it keeps the arithmetic in range.
            st = Math.Clamp(st, -int.MaxValue, int.MaxValue);
            long dim = s[a];
            long b = starts[i], e = ends[i];
            if (st < 0 && (e == int.MaxValue || e == long.MaxValue))
            {
                // Black-box finding: ONNX Runtime reads an end of INT32_MAX or INT64_MAX with a negative step as "all the way to the
                // beginning" (like INT_MIN), where the spec's clamping alone would give an empty slice. Models in the wild use it.
                e = long.MinValue;
            }
            if (b < 0)
            {
                b += dim;
            }

            if (e < 0)
            {
                e += dim;
            }

            long c;
            if (dim == 0)
            {
                b = 0;
                c = 0;
            }
            else if (st > 0)
            {
                b = Math.Clamp(b, 0, dim);
                e = Math.Clamp(e, 0, dim);
                c = e > b ? ((e - b - 1) / st) + 1 : 0;
            }
            else
            {
                b = Math.Clamp(b, 0, dim - 1);
                e = Math.Clamp(e, -1, dim - 1);
                c = b > e ? ((b - e - 1) / -st) + 1 : 0;
            }

            start[a] = (int)Math.Max(b, 0);
            step[a] = (int)st;
            count[a] = (int)c;
        }

        return null;
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], y = run.Outputs[0];
        Shape s = x.Shape;
        int rank = s.Rank;
        Span<int> start = stackalloc int[Shape.MaxRank];
        Span<int> step = stackalloc int[Shape.MaxRank];
        Span<int> count = stackalloc int[Shape.MaxRank];
        string? error;
        if (_fromInputs)
        {
            bool hasAxes = run.Inputs.Length > 3 && !run.Inputs[3].IsEmpty;
            bool hasSteps = run.Inputs.Length > 4 && !run.Inputs[4].IsEmpty;
            error = Resolve(s, IntVec.Of(run.Inputs[1]), IntVec.Of(run.Inputs[2]), hasAxes ? IntVec.Of(run.Inputs[3]) : default, hasAxes,
                hasSteps ? IntVec.Of(run.Inputs[4]) : default, hasSteps, start, step, count);
        }
        else
        {
            error = Resolve(s, new IntVec(_starts), new IntVec(_ends), _axes is null ? default : new IntVec(_axes), _axes is not null, default,
                false, start, step, count);
        }

        if (error is not null)
        {
            throw new SessionException(SessionErrorKind.InvalidInput, "Slice: " + error);
        }

        Span<int> inStr = stackalloc int[Shape.MaxRank];
        Span<int> srcStr = stackalloc int[Shape.MaxRank];
        Span<int> dstStr = stackalloc int[Shape.MaxRank];
        OpUtil.RowMajor(s.Dimensions, inStr);
        OpUtil.RowMajor(y.Shape.Dimensions, dstStr);
        long offset = 0;
        for (int a = 0; a < rank; a++)
        {
            srcStr[a] = inStr[a] * step[a];
            offset += (long)start[a] * inStr[a];
        }

        Movement.Move(x, (int)offset, srcStr.Slice(0, rank), y, 0, dstStr.Slice(0, rank), y.Shape.Dimensions);
    }
}

/// <summary>Expand: broadcasts the input to the (bidirectionally broadcast) shape given by the second input.</summary>
internal sealed class ExpandOp : OpKernel
{
    public static readonly OpDef Def = new("Expand", 8, int.MaxValue, static _ => new ExpandOp());

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects data and shape inputs");
        }

        if (ctx.InputType(1) != DType.I64)
        {
            throw ctx.Fail($"the shape input must be int64, not {ctx.InputType(1)}");
        }

        OpUtil.RequireIntegers(ctx, 1, "shape");
        if (!ctx.TryGetInputValue(1, out Tensor target))
        {
            ctx.RequireRunTimeValues();
            return;
        }

        ReadOnlySpan<long> spec = target.Data<long>();
        Shape given = OpUtil.MakeShape(ctx, spec);
        if (!Shape.TryBroadcast(ctx.InputShape(0), given, out Shape result))
        {
            throw ctx.Fail($"{ctx.InputShape(0)} cannot be expanded to {given}");
        }

        ctx.SetOutput(0, ctx.InputType(0), result);
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], y = run.Outputs[0];
        int rank = y.Rank, shift = rank - x.Rank;
        Span<int> inStr = stackalloc int[Shape.MaxRank];
        Span<int> srcStr = stackalloc int[Shape.MaxRank];
        Span<int> dstStr = stackalloc int[Shape.MaxRank];
        OpUtil.RowMajor(x.Shape.Dimensions, inStr);
        OpUtil.RowMajor(y.Shape.Dimensions, dstStr);
        for (int d = 0; d < rank; d++)
        {
            int a = d - shift;
            srcStr[d] = a < 0 || x.Shape[a] == 1 ? 0 : inStr[a];
        }

        Movement.Move(x, 0, srcStr.Slice(0, rank), y, 0, dstStr.Slice(0, rank), y.Shape.Dimensions);
    }
}

/// <summary>Tile: repeats the input along each axis the number of times in the second input.</summary>
internal sealed class TileOp : OpKernel
{
    public static readonly OpDef Def = new("Tile", 6, int.MaxValue, static _ => new TileOp());

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects data and repeats inputs");
        }

        OpUtil.RequireIntegers(ctx, 1, "repeats");
        Shape s = ctx.InputShape(0);
        if (ctx.InputShape(1).Rank != 1 || ctx.InputShape(1)[0] != s.Rank)
        {
            throw ctx.Fail($"repeats {ctx.InputShape(1)} must hold one count per axis of the rank-{s.Rank} input");
        }

        if (!ctx.TryGetInputValue(1, out Tensor v))
        {
            ctx.RequireRunTimeValues();
            return;
        }

        IntVec reps = IntVec.Of(v);
        Span<long> dims = stackalloc long[Shape.MaxRank];
        for (int a = 0; a < s.Rank; a++)
        {
            if (reps[a] < 0)
            {
                throw ctx.Fail($"repeat count {reps[a]} is negative");
            }

            dims[a] = reps[a] > int.MaxValue ? long.MaxValue : s[a] * reps[a];
        }

        ctx.SetOutput(0, ctx.InputType(0), OpUtil.MakeShape(ctx, dims.Slice(0, s.Rank)));
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], y = run.Outputs[0];
        int rank = x.Rank;
        if (rank == 0)
        {
            y.CopyFrom(x);
            return;
        }

        // Each axis becomes (repeat, extent): the repeat axis does not move in the source, the extent axis does.
        Span<int> inStr = stackalloc int[Shape.MaxRank];
        Span<int> outStr = stackalloc int[Shape.MaxRank];
        Span<int> dims = stackalloc int[Shape.MaxRank * 2];
        Span<int> srcStr = stackalloc int[Shape.MaxRank * 2];
        Span<int> dstStr = stackalloc int[Shape.MaxRank * 2];
        OpUtil.RowMajor(x.Shape.Dimensions, inStr);
        OpUtil.RowMajor(y.Shape.Dimensions, outStr);
        for (int a = 0; a < rank; a++)
        {
            int n = x.Shape[a];
            dims[2 * a] = n == 0 ? 0 : y.Shape[a] / n;
            dims[(2 * a) + 1] = n;
            srcStr[2 * a] = 0;
            srcStr[(2 * a) + 1] = inStr[a];
            dstStr[2 * a] = n * outStr[a];
            dstStr[(2 * a) + 1] = outStr[a];
        }

        Movement.Move(x, 0, srcStr.Slice(0, 2 * rank), y, 0, dstStr.Slice(0, 2 * rank), dims.Slice(0, 2 * rank));
    }
}
