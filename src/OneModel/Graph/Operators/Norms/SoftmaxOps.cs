using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Kernels.Exact;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

// Softmax, LogSoftmax and Hardmax. Two axis rules exist. Up to opset 12 the input is coerced to a 2-D matrix at `axis` (default 1): the
// dimensions before the axis form the rows and everything from the axis on forms one row to normalise. From opset 13 the operator works
// along the single axis `axis` (default -1) of the original shape. When that axis is not the last the data is moved into rows first.

internal enum SoftmaxKind
{
    Softmax,
    LogSoftmax,
    Hardmax,
}

/// <summary>Softmax-family work over rows of a contiguous matrix.</summary>
internal readonly struct SoftmaxRowsWork : IChunkWork
{
    public required Tensor Src { get; init; }

    public required Tensor Dst { get; init; }

    public required int Rows { get; init; }

    public required int Cols { get; init; }

    public required SoftmaxKind Kind { get; init; }

    public required bool Exact { get; init; }

    public required VecIsa Isa { get; init; }

    public int Chunks => (Rows + NormCommon.RowsPerChunk(Cols) - 1) / NormCommon.RowsPerChunk(Cols);

    public void RunChunk(int chunk)
    {
        int per = NormCommon.RowsPerChunk(Cols);
        int r0 = chunk * per, r1 = Math.Min(Rows, r0 + per);
        ReadOnlySpan<float> s = Src.Data<float>();
        Span<float> d = Dst.MutableData<float>();
        switch (Kind)
        {
            case SoftmaxKind.Softmax:
                Softmax.Rows(Exact, Isa, s.Slice(r0 * Cols, (r1 - r0) * Cols), d.Slice(r0 * Cols, (r1 - r0) * Cols), r1 - r0, Cols, Cols, Cols);
                break;
            case SoftmaxKind.LogSoftmax:
                NormKernels.LogSoftmaxRows(Exact, Isa, s, d, r0, r1, Cols);
                break;
            default:
                for (int r = r0; r < r1; r++)
                {
                    HardmaxRow(s.Slice(r * Cols, Cols), d.Slice(r * Cols, Cols));
                }

                break;
        }
    }

    // The first maximum gets 1 and the rest 0; a row holding NaN is all zeros (the behaviour of ONNX Runtime on the probed input).
    private static void HardmaxRow(ReadOnlySpan<float> x, Span<float> y)
    {
        int best = 0;
        float b = x[0];
        bool nan = false;
        for (int j = 0; j < x.Length; j++)
        {
            float v = x[j];
            nan |= float.IsNaN(v);
            if (v > b)
            {
                b = v;
                best = j;
            }
        }

        y.Clear();
        if (!nan)
        {
            y[best] = 1f;
        }
    }
}

/// <summary>Moves between <c>[outer, length, inner]</c> and <c>[outer, inner, length]</c> so a middle axis becomes the row.</summary>
internal static class AxisMove
{
    public static void ToRows(ReadOnlySpan<float> x, Span<float> t, int outer, int length, int inner)
    {
        for (int o = 0; o < outer; o++)
        {
            ReadOnlySpan<float> src = x.Slice(o * length * inner, length * inner);
            Span<float> dst = t.Slice(o * length * inner, length * inner);
            for (int r = 0; r < length; r++)
            {
                for (int i = 0; i < inner; i++)
                {
                    dst[(i * length) + r] = src[(r * inner) + i];
                }
            }
        }
    }

    public static void FromRows(ReadOnlySpan<float> t, Span<float> y, int outer, int length, int inner)
    {
        for (int o = 0; o < outer; o++)
        {
            ReadOnlySpan<float> src = t.Slice(o * length * inner, length * inner);
            Span<float> dst = y.Slice(o * length * inner, length * inner);
            for (int r = 0; r < length; r++)
            {
                for (int i = 0; i < inner; i++)
                {
                    dst[(r * inner) + i] = src[(i * length) + r];
                }
            }
        }
    }
}

/// <summary>The shared implementation of Softmax, LogSoftmax and Hardmax.</summary>
internal abstract class SoftmaxFamilyOp : OpKernel
{
    private readonly long _axis;
    private readonly bool _coerce;
    private readonly SoftmaxKind _kind;

    protected SoftmaxFamilyOp(NodeInfo node, SoftmaxKind kind, bool coerce)
    {
        _kind = kind;
        _coerce = coerce;
        _axis = node.GetInt("axis", coerce ? 1 : -1);
    }

    public override bool SupportsExact => true;

    public override int InPlaceInput => 0;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        NormCommon.RequireFloat(ctx, 0);
        Shape x = ctx.InputShape(0);
        if (x.Rank == 0)
        {
            throw ctx.Fail("the input must have at least one dimension");
        }

        _ = NormCommon.Axis(ctx, _axis, x.Rank);
        ctx.SetOutput(0, DType.F32, x);
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], y = run.Outputs[0];
        if (x.Length == 0)
        {
            return;
        }

        int axis = (int)(_axis < 0 ? _axis + x.Rank : _axis);
        int outer = 1, length, inner = 1;
        if (_coerce)
        {
            length = 1;
            for (int i = 0; i < x.Rank; i++)
            {
                if (i < axis)
                {
                    outer *= x.Shape[i];
                }
                else
                {
                    length *= x.Shape[i];
                }
            }
        }
        else
        {
            length = x.Shape[axis];
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
        }

        VecIsa isa = RowIsa.Of(run.Path);
        if (inner == 1)
        {
            var work = new SoftmaxRowsWork { Src = x, Dst = y, Rows = outer, Cols = length, Kind = _kind, Exact = run.Exact, Isa = isa };
            Chunked.Run(work, work.Chunks, run.Threads);
            return;
        }

        // The axis is in the middle: make it the row, work in place in the scratch copy, and move the result back.
        Tensor t = run.Scratch(DType.F32, x.Shape);
        AxisMove.ToRows(x.Data<float>(), t.MutableData<float>(), outer, length, inner);
        var rows = new SoftmaxRowsWork { Src = t, Dst = t, Rows = outer * inner, Cols = length, Kind = _kind, Exact = run.Exact, Isa = isa };
        Chunked.Run(rows, rows.Chunks, run.Threads);
        AxisMove.FromRows(t.Data<float>(), y.MutableData<float>(), outer, length, inner);
    }
}

/// <summary>Softmax: <c>exp(x - max) / sum</c> along the axis. Opset 1 to 12 coerce the input to 2-D at the axis.</summary>
internal sealed class SoftmaxOp : SoftmaxFamilyOp
{
    public static readonly OpDef DefCoerce = new("Softmax", 1, 12, static n => new SoftmaxOp(n, true));

    public static readonly OpDef DefAxis = new("Softmax", 13, int.MaxValue, static n => new SoftmaxOp(n, false));

    private SoftmaxOp(NodeInfo node, bool coerce)
        : base(node, SoftmaxKind.Softmax, coerce)
    {
    }
}

/// <summary>LogSoftmax: <c>x - max - log(sum(exp(x - max)))</c> along the axis. Opset 1 to 12 coerce the input to 2-D at the axis.</summary>
internal sealed class LogSoftmaxOp : SoftmaxFamilyOp
{
    public static readonly OpDef DefCoerce = new("LogSoftmax", 1, 12, static n => new LogSoftmaxOp(n, true));

    public static readonly OpDef DefAxis = new("LogSoftmax", 13, int.MaxValue, static n => new LogSoftmaxOp(n, false));

    private LogSoftmaxOp(NodeInfo node, bool coerce)
        : base(node, SoftmaxKind.LogSoftmax, coerce)
    {
    }
}

/// <summary>Hardmax: one-hot of the first maximum along the axis. Opset 1 to 12 coerce the input to 2-D at the axis.</summary>
internal sealed class HardmaxOp : SoftmaxFamilyOp
{
    public static readonly OpDef DefCoerce = new("Hardmax", 1, 12, static n => new HardmaxOp(n, true));

    public static readonly OpDef DefAxis = new("Hardmax", 13, int.MaxValue, static n => new HardmaxOp(n, false));

    private HardmaxOp(NodeInfo node, bool coerce)
        : base(node, SoftmaxKind.Hardmax, coerce)
    {
    }
}
