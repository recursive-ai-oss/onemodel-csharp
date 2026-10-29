using System;
using System.Collections.Immutable;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Kernels.Exact;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

// The Reduce* operators and ArgMax/ArgMin. The arithmetic and the accumulation orders are in Kernels/Exact/ReduceKernels.cs and
// docs/ops-norms.md. Opsets before the change that moved `axes` from an attribute to an input (13 for ReduceSum, 18 for the rest) read
// the attribute; later ones read the second input, which may be computed at run time.

/// <summary>Float32 reduction work: the vector kernels, or the scalar log-sum-exp.</summary>
internal readonly struct ReduceFloatWork : IChunkWork
{
    public required ReduceOp Op { get; init; }

    public required ReduceGeometry Geometry { get; init; }

    public required Tensor X { get; init; }

    public required Tensor Y { get; init; }

    public required VecIsa Isa { get; init; }

    public int Chunks => ReduceKernels.IsVectorOp(Op) ? Geometry.Chunks : Geometry.GenericChunks;

    public void RunChunk(int chunk)
    {
        ReduceGeometry g = Geometry;
        if (ReduceKernels.IsVectorOp(Op))
        {
            long u0 = (long)chunk * g.UnitsPerChunk;
            ReduceKernels.FloatRange(Op, Isa, g, X.Data<float>(), Y.MutableData<float>(), u0, Math.Min(g.Units, u0 + g.UnitsPerChunk));
        }
        else
        {
            long o0 = (long)chunk * g.OutputsPerChunk;
            ReduceKernels.LogSumExpRange(g, X.Data<float>(), Y.MutableData<float>(), o0, Math.Min(g.Outputs, o0 + g.OutputsPerChunk));
        }
    }
}

/// <summary>Reduction work for the other element types.</summary>
internal readonly struct ReduceGenericWork<T> : IChunkWork
    where T : unmanaged, System.Numerics.INumber<T>, System.Numerics.IMinMaxValue<T>
{
    public required ReduceOp Op { get; init; }

    public required ReduceGeometry Geometry { get; init; }

    public required Tensor X { get; init; }

    public required Tensor Y { get; init; }

    public int Chunks => Geometry.GenericChunks;

    public void RunChunk(int chunk)
    {
        ReduceGeometry g = Geometry;
        long o0 = (long)chunk * g.OutputsPerChunk;
        ReduceKernels.GenericRange(Op, g, X.Data<T>(), Y.MutableData<T>(), o0, Math.Min(g.Outputs, o0 + g.OutputsPerChunk));
    }
}

/// <summary>ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd, ReduceL1, ReduceL2, ReduceSumSquare, ReduceLogSum, ReduceLogSumExp.</summary>
internal sealed class ReduceKernel : OpKernel
{
    private readonly ReduceOp _op;
    private readonly bool _axesInput;
    private readonly ImmutableArray<long> _axes;
    private readonly bool _keepDims;
    private readonly bool _noopWithEmptyAxes;

    internal ReduceKernel(NodeInfo node, ReduceOp op, bool axesInput)
    {
        _op = op;
        _axesInput = axesInput;
        _keepDims = node.GetInt("keepdims", 1) != 0;
        _noopWithEmptyAxes = node.GetInt("noop_with_empty_axes", 0) != 0;
        _axes = axesInput ? [] : node.GetInts("axes");
    }

    public static readonly OpDef[] Defs =
    [
        new("ReduceSum", 1, 12, static n => new ReduceKernel(n, ReduceOp.Sum, false)),
        new("ReduceSum", 13, int.MaxValue, static n => new ReduceKernel(n, ReduceOp.Sum, true)),
        new("ReduceMean", 1, 17, static n => new ReduceKernel(n, ReduceOp.Mean, false)),
        new("ReduceMean", 18, int.MaxValue, static n => new ReduceKernel(n, ReduceOp.Mean, true)),
        new("ReduceMax", 1, 17, static n => new ReduceKernel(n, ReduceOp.Max, false)),
        new("ReduceMax", 18, int.MaxValue, static n => new ReduceKernel(n, ReduceOp.Max, true)),
        new("ReduceMin", 1, 17, static n => new ReduceKernel(n, ReduceOp.Min, false)),
        new("ReduceMin", 18, int.MaxValue, static n => new ReduceKernel(n, ReduceOp.Min, true)),
        new("ReduceProd", 1, 17, static n => new ReduceKernel(n, ReduceOp.Prod, false)),
        new("ReduceProd", 18, int.MaxValue, static n => new ReduceKernel(n, ReduceOp.Prod, true)),
        new("ReduceL1", 1, 17, static n => new ReduceKernel(n, ReduceOp.L1, false)),
        new("ReduceL1", 18, int.MaxValue, static n => new ReduceKernel(n, ReduceOp.L1, true)),
        new("ReduceL2", 1, 17, static n => new ReduceKernel(n, ReduceOp.L2, false)),
        new("ReduceL2", 18, int.MaxValue, static n => new ReduceKernel(n, ReduceOp.L2, true)),
        new("ReduceSumSquare", 1, 17, static n => new ReduceKernel(n, ReduceOp.SumSquare, false)),
        new("ReduceSumSquare", 18, int.MaxValue, static n => new ReduceKernel(n, ReduceOp.SumSquare, true)),
        new("ReduceLogSum", 1, 17, static n => new ReduceKernel(n, ReduceOp.LogSum, false)),
        new("ReduceLogSum", 18, int.MaxValue, static n => new ReduceKernel(n, ReduceOp.LogSum, true)),
        new("ReduceLogSumExp", 1, 17, static n => new ReduceKernel(n, ReduceOp.LogSumExp, false)),
        new("ReduceLogSumExp", 18, int.MaxValue, static n => new ReduceKernel(n, ReduceOp.LogSumExp, true)),
    ];

    // Every float32 reduction is exact. The other element types are integers (exact by construction) and float64, whose plain
    // IEEE operations in a fixed order are exact too; float64 L2 and the log forms are not offered.
    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects the data input");
        }

        DType t = ctx.InputType(0);
        bool floatOnly = _op is ReduceOp.L2 or ReduceOp.LogSum or ReduceOp.LogSumExp;
        bool ok = t switch
        {
            DType.F32 => true,
            DType.F64 => _op is not (ReduceOp.LogSum or ReduceOp.LogSumExp),
            DType.I32 or DType.I64 or DType.U32 or DType.U64 or DType.I8 or DType.U8 => !floatOnly,
            _ => false,
        };
        if (!ok)
        {
            throw ctx.Unsupported($"element type {t} is not supported yet");
        }

        ReadOnlySpan<long> axes = _axes.AsSpan();
        if (_axesInput && ctx.HasInput(1))
        {
            if (ctx.InputType(1) != DType.I64)
            {
                throw ctx.Fail("the axes input must be int64");
            }

            if (!ctx.TryGetInputValue(1, out Tensor v))
            {
                ctx.RequireRunTimeValues();
                return;
            }

            axes = v.Data<long>();
        }

        Shape x = ctx.InputShape(0);
        Span<bool> mask = stackalloc bool[Shape.MaxRank];
        string? error = Resolve(x.Dimensions, axes, mask);
        if (error is not null)
        {
            throw ctx.Fail(error);
        }

        Span<int> d = stackalloc int[Shape.MaxRank];
        int r = 0;
        for (int i = 0; i < x.Rank; i++)
        {
            if (!mask[i])
            {
                d[r++] = x[i];
            }
            else if (_keepDims)
            {
                d[r++] = 1;
            }
        }

        ctx.SetOutput(0, t, Shape.Create(d.Slice(0, r)));
    }

    // Marks the reduced axes; returns an error message or null. An empty axes list reduces everything, unless noop_with_empty_axes is
    // set: then nothing is reduced and each output is the operation applied to a single element (so ReduceSumSquare squares and
    // ReduceLogSum takes the log, as ONNX Runtime does, rather than returning the input untouched).
    private string? Resolve(ReadOnlySpan<int> dims, ReadOnlySpan<long> axes, Span<bool> mask)
    {
        mask.Slice(0, dims.Length).Clear();
        if (axes.IsEmpty)
        {
            mask.Slice(0, dims.Length).Fill(!_noopWithEmptyAxes);
            return null;
        }

        foreach (long a in axes)
        {
            if (a < -dims.Length || a >= dims.Length)
            {
                return $"axis {a} is outside [-{dims.Length}, {dims.Length}) for a {dims.Length}-D input";
            }

            mask[(int)(a < 0 ? a + dims.Length : a)] = true;
        }

        return null;
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], y = run.Outputs[0];
        ReadOnlySpan<long> axes = _axes.AsSpan();
        if (_axesInput && run.Inputs.Length > 1 && !run.Inputs[1].IsEmpty)
        {
            axes = run.Inputs[1].Data<long>();
        }

        Span<bool> mask = stackalloc bool[Shape.MaxRank];
        string? error = Resolve(x.Shape.Dimensions, axes, mask);
        if (error is not null)
        {
            throw new SessionException(SessionErrorKind.InvalidInput, "ReduceKernel: " + error);
        }

        if (y.Length == 0)
        {
            return;
        }

        ReduceGeometry g = ReduceGeometry.Create(x.Shape.Dimensions, mask.Slice(0, x.Rank));
        switch (x.DType)
        {
            case DType.F32:
                var fw = new ReduceFloatWork { Op = _op, Geometry = g, X = x, Y = y, Isa = RowIsa.Of(run.Path) };
                Chunked.Run(fw, fw.Chunks, run.Threads);
                break;
            case DType.F64: Generic<double>(run, g, x, y); break;
            case DType.I32: Generic<int>(run, g, x, y); break;
            case DType.I64: Generic<long>(run, g, x, y); break;
            case DType.U32: Generic<uint>(run, g, x, y); break;
            case DType.U64: Generic<ulong>(run, g, x, y); break;
            case DType.I8: Generic<sbyte>(run, g, x, y); break;
            case DType.U8: Generic<byte>(run, g, x, y); break;
            default:
                throw new InvalidOperationException($"Element type {x.DType} reached the kernel although inference rejects it.");
        }
    }

    private void Generic<T>(in OpRun run, in ReduceGeometry g, Tensor x, Tensor y)
        where T : unmanaged, System.Numerics.INumber<T>, System.Numerics.IMinMaxValue<T>
    {
        var w = new ReduceGenericWork<T> { Op = _op, Geometry = g, X = x, Y = y };
        Chunked.Run(w, w.Chunks, run.Threads);
    }
}

/// <summary>ArgMax and ArgMin: the index of the first (or last) extreme element along one axis, as int64.</summary>
internal sealed class ArgExtremeKernel : OpKernel
{
    public static readonly OpDef ArgMaxDef = new("ArgMax", 1, int.MaxValue, static n => new ArgExtremeKernel(n, true));

    public static readonly OpDef ArgMinDef = new("ArgMin", 1, int.MaxValue, static n => new ArgExtremeKernel(n, false));

    private readonly bool _max;
    private readonly long _axis;
    private readonly bool _keepDims;
    private readonly bool _selectLast;

    private ArgExtremeKernel(NodeInfo node, bool max)
    {
        _max = max;
        _axis = node.GetInt("axis", 0);
        _keepDims = node.GetInt("keepdims", 1) != 0;
        _selectLast = node.GetInt("select_last_index", 0) != 0;
    }

    // Comparisons and index bookkeeping only: exact for every supported type.
    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        DType t = ctx.InputType(0);
        if (t is not (DType.F32 or DType.F64 or DType.I8 or DType.U8 or DType.I16 or DType.U16 or DType.I32 or DType.I64 or DType.U32 or DType.U64))
        {
            throw ctx.Unsupported($"element type {t} is not supported yet");
        }

        Shape x = ctx.InputShape(0);
        if (x.Rank == 0)
        {
            throw ctx.Fail("the input must have at least one dimension");
        }

        int axis = NormCommon.Axis(ctx, _axis, x.Rank);
        Span<int> d = stackalloc int[Shape.MaxRank];
        int r = 0;
        for (int i = 0; i < x.Rank; i++)
        {
            if (i != axis)
            {
                d[r++] = x[i];
            }
            else if (_keepDims)
            {
                d[r++] = 1;
            }
        }

        ctx.SetOutput(0, DType.I64, Shape.Create(d.Slice(0, r)));
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], y = run.Outputs[0];
        int axis = (int)(_axis < 0 ? _axis + x.Rank : _axis);
        int outer = 1, inner = 1, length = x.Shape[axis];
        for (int i = 0; i < x.Rank; i++)
        {
            if (i < axis)
            {
                outer *= x.Shape[i];
            }
            else if (i > axis)
            {
                inner *= x.Shape[i];
            }
        }

        if (y.Length == 0)
        {
            return;
        }

        switch (x.DType)
        {
            case DType.F32: Arg<float>(run, x, y, outer, length, inner); break;
            case DType.F64: Arg<double>(run, x, y, outer, length, inner); break;
            case DType.I8: Arg<sbyte>(run, x, y, outer, length, inner); break;
            case DType.U8: Arg<byte>(run, x, y, outer, length, inner); break;
            case DType.I16: Arg<short>(run, x, y, outer, length, inner); break;
            case DType.U16: Arg<ushort>(run, x, y, outer, length, inner); break;
            case DType.I32: Arg<int>(run, x, y, outer, length, inner); break;
            case DType.I64: Arg<long>(run, x, y, outer, length, inner); break;
            case DType.U32: Arg<uint>(run, x, y, outer, length, inner); break;
            case DType.U64: Arg<ulong>(run, x, y, outer, length, inner); break;
            default:
                throw new InvalidOperationException($"Element type {x.DType} reached the kernel although inference rejects it.");
        }
    }

    private void Arg<T>(in OpRun run, Tensor x, Tensor y, int outer, int length, int inner)
        where T : unmanaged, System.Numerics.INumber<T>
    {
        Tensor best = run.Scratch(x.DType, Shape.Create(inner));
        ReduceKernels.ArgExtreme(_max, _selectLast, x.Data<T>(), y.MutableData<long>(), best.MutableData<T>(), outer, length, inner);
    }
}
