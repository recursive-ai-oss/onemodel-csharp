using System;
using System.Numerics;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

/// <summary>
/// CumSum (opset 11 and 14): running sum along the axis given by the second input (a scalar, int32 or int64, read at run time).
/// <c>exclusive</c> shifts the sums by one so element i holds the sum of the elements before it; <c>reverse</c> accumulates from the
/// end. Sums are sequential in the direction of accumulation, which is exact for integers and a single fixed order for floats. The
/// loops run on one thread: the work is a streaming pass and the order is inherently serial along the axis.
/// </summary>
internal sealed class CumSumOp : OpKernel
{
    public static readonly OpDef Def = new("CumSum", 11, int.MaxValue, static n => new CumSumOp(n));

    private readonly bool _exclusive;
    private readonly bool _reverse;

    private CumSumOp(NodeInfo node)
    {
        _exclusive = node.GetInt("exclusive", 0) != 0;
        _reverse = node.GetInt("reverse", 0) != 0;
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects x and axis");
        }

        DType t = ctx.InputType(0);
        if (t is not (DType.F32 or DType.F64 or DType.I32 or DType.I64 or DType.U32 or DType.U64))
        {
            throw ctx.Unsupported($"element type {t} is not supported yet");
        }

        if (ctx.InputType(1) is not (DType.I32 or DType.I64) || ctx.InputShape(1).Length != 1)
        {
            throw ctx.Fail("axis must be a scalar int32 or int64");
        }

        if (ctx.InputShape(0).Rank == 0)
        {
            throw ctx.Fail("x must have at least one dimension");
        }

        ctx.SetOutput(0, t, ctx.InputShape(0));
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], y = run.Outputs[0];
        long axis = run.Inputs[1].DType == DType.I32 ? run.Inputs[1].Data<int>()[0] : run.Inputs[1].Data<long>()[0];
        if (axis < -x.Rank || axis >= x.Rank)
        {
            throw new SessionException(SessionErrorKind.InvalidInput, $"CumSum: axis {axis} is outside [-{x.Rank}, {x.Rank})");
        }

        int a = (int)(axis < 0 ? axis + x.Rank : axis);
        int outer = 1, inner = 1, length = x.Shape[a];
        for (int i = 0; i < x.Rank; i++)
        {
            if (i < a)
            {
                outer *= x.Shape[i];
            }
            else if (i > a)
            {
                inner *= x.Shape[i];
            }
        }

        if (x.Length == 0)
        {
            return;
        }

        switch (x.DType)
        {
            case DType.F32:
                Span<float> yf = y.MutableData<float>();
                Scan(x.Data<float>(), yf, outer, length, inner);
                if (run.Exact)
                {
                    // Exact mode rewrites every NaN to the canonical quiet NaN, whatever the hardware produced.
                    for (int i = 0; i < yf.Length; i++)
                    {
                        if (float.IsNaN(yf[i]))
                        {
                            yf[i] = Kernels.Exact.ExactElementwise.CanonicalNaN;
                        }
                    }
                }

                break;
            case DType.F64: Scan(x.Data<double>(), y.MutableData<double>(), outer, length, inner); break;
            case DType.I32: Scan(x.Data<int>(), y.MutableData<int>(), outer, length, inner); break;
            case DType.I64: Scan(x.Data<long>(), y.MutableData<long>(), outer, length, inner); break;
            case DType.U32: Scan(x.Data<uint>(), y.MutableData<uint>(), outer, length, inner); break;
            case DType.U64: Scan(x.Data<ulong>(), y.MutableData<ulong>(), outer, length, inner); break;
            default:
                throw new InvalidOperationException($"Element type {x.DType} reached the kernel although inference rejects it.");
        }
    }

    private void Scan<T>(ReadOnlySpan<T> x, Span<T> y, int outer, int length, int inner)
        where T : unmanaged, INumber<T>
    {
        for (int o = 0; o < outer; o++)
        {
            ReadOnlySpan<T> src = x.Slice(o * length * inner, length * inner);
            Span<T> dst = y.Slice(o * length * inner, length * inner);
            for (int step = 0; step < length; step++)
            {
                int r = _reverse ? length - 1 - step : step;
                Span<T> row = dst.Slice(r * inner, inner);
                if (step == 0)
                {
                    if (_exclusive)
                    {
                        row.Clear();
                    }
                    else
                    {
                        src.Slice(r * inner, inner).CopyTo(row);
                    }

                    continue;
                }

                int prev = _reverse ? r + 1 : r - 1;
                ReadOnlySpan<T> before = dst.Slice(prev * inner, inner);
                // Exclusive: add the element just passed; inclusive: add this one.
                ReadOnlySpan<T> add = src.Slice((_exclusive ? prev : r) * inner, inner);
                for (int i = 0; i < inner; i++)
                {
                    row[i] = unchecked(before[i] + add[i]);
                }
            }
        }
    }
}
