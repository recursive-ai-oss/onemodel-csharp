using System;
using System.Numerics;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

// Smaller numeric operators of the default domain. Sums and products run in a fixed sequential order and use only IEEE +, -, *, / and
// square root, so the ones marked exact give the same bits everywhere; the ones that need pow, exp or determinants' pivoting in double
// say so.

/// <summary>CumProd (opset 26): running product along the axis in the second input; exclusive and reverse as for CumSum.</summary>
internal sealed class CumProdOp : OpKernel
{
    public static readonly OpDef Def = new("CumProd", 26, int.MaxValue, static n => new CumProdOp(n));

    private readonly bool _exclusive, _reverse;

    private CumProdOp(NodeInfo n)
    {
        _exclusive = n.GetInt("exclusive", 0) != 0;
        _reverse = n.GetInt("reverse", 0) != 0;
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects x and axis");
        }

        DType t = ctx.InputType(0);
        if (!TypeDispatch.IsNumber(t))
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
        Tensor x = run.Inputs[0];
        long axis = OpUtil.First(run.Inputs[1]);
        if (axis < -x.Rank || axis >= x.Rank)
        {
            throw new SessionException(SessionErrorKind.InvalidInput, $"CumProd: axis {axis} is outside the {x.Rank} dimensions of x");
        }

        int a = (int)(axis < 0 ? axis + x.Rank : axis);
        var v = new Visitor
        {
            X = x, Y = run.Outputs[0], Exclusive = _exclusive, Reverse = _reverse,
            Outer = OpUtil.Product(x.Shape.Dimensions, 0, a), Len = x.Shape[a], Inner = OpUtil.Product(x.Shape.Dimensions, a + 1, x.Rank),
        };
        TypeDispatch.Number(x.DType, ref v);
    }

    private struct Visitor : INumberVisitor
    {
        public Tensor X, Y;
        public bool Exclusive, Reverse;
        public int Outer, Len, Inner;

        public void Visit<T>() where T : unmanaged, INumber<T>
        {
            ReadOnlySpan<T> x = X.Data<T>();
            Span<T> y = Y.MutableData<T>();
            for (int o = 0; o < Outer; o++)
            {
                for (int i = 0; i < Inner; i++)
                {
                    T acc = T.One;
                    for (int s = 0; s < Len; s++)
                    {
                        int k = Reverse ? Len - 1 - s : s;
                        int at = (((o * Len) + k) * Inner) + i;
                        if (Exclusive)
                        {
                            y[at] = acc;
                            acc = unchecked(acc * x[at]);
                        }
                        else
                        {
                            acc = unchecked(acc * x[at]);
                            y[at] = acc;
                        }
                    }
                }
            }
        }
    }
}

/// <summary>LpNormalization: x divided by its L1 or L2 norm along an axis (attributes axis = -1, p = 1 or 2).</summary>
internal sealed class LpNormalizationOp : OpKernel
{
    public static readonly OpDef Def = new("LpNormalization", 1, int.MaxValue, static n => new LpNormalizationOp(n));

    private readonly long _axis, _p;

    private LpNormalizationOp(NodeInfo n)
    {
        _axis = n.GetInt("axis", -1);
        _p = n.GetInt("p", 2);
        if (_p is not (1 or 2))
        {
            throw n.Error("attribute 'p' must be 1 or 2");
        }
    }

    public override bool SupportsExact => true;

    public override int InPlaceInput => -1;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        if (ctx.InputType(0) is not (DType.F32 or DType.F64))
        {
            throw ctx.Unsupported($"element type {ctx.InputType(0)} is not supported yet");
        }

        _ = OpUtil.Axis(ctx, _axis, ctx.InputShape(0).Rank);
        ctx.SetOutput(0, ctx.InputType(0), ctx.InputShape(0));
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0];
        int a = (int)(_axis < 0 ? _axis + x.Rank : _axis);
        var v = new Visitor { X = x, Y = run.Outputs[0], P = (int)_p, Outer = OpUtil.Product(x.Shape.Dimensions, 0, a), Len = x.Shape[a], Inner = OpUtil.Product(x.Shape.Dimensions, a + 1, x.Rank) };
        if (x.DType == DType.F32)
        {
            v.Visit<float>();
        }
        else
        {
            v.Visit<double>();
        }
    }

    private struct Visitor
    {
        public Tensor X, Y;
        public int P, Outer, Len, Inner;

        public void Visit<T>() where T : unmanaged, IFloatingPointIeee754<T>
        {
            ReadOnlySpan<T> x = X.Data<T>();
            Span<T> y = Y.MutableData<T>();
            for (int o = 0; o < Outer; o++)
            {
                for (int i = 0; i < Inner; i++)
                {
                    T sum = T.Zero;
                    for (int k = 0; k < Len; k++)
                    {
                        T e = x[(((o * Len) + k) * Inner) + i];
                        sum += P == 1 ? T.Abs(e) : e * e;
                    }

                    T norm = P == 1 ? sum : T.Sqrt(sum);
                    for (int k = 0; k < Len; k++)
                    {
                        int at = (((o * Len) + k) * Inner) + i;
                        // A zero norm gives 0, as the backend case l2normalization_axis_0 requires.
                        y[at] = norm == T.Zero ? T.Zero : x[at] / norm;
                    }
                }
            }
        }
    }
}

/// <summary>MeanVarianceNormalization: (x - mean) / sqrt(variance + 1e-9) over the attribute axes (default 0, 2, 3), the variance being the mean squared deviation.</summary>
internal sealed class MeanVarianceNormalizationOp : OpKernel
{
    public static readonly OpDef Def = new("MeanVarianceNormalization", 9, int.MaxValue, static n => new MeanVarianceNormalizationOp(n));

    private readonly long[] _axes;

    private MeanVarianceNormalizationOp(NodeInfo n)
    {
        var a = n.GetInts("axes");
        _axes = a.IsDefaultOrEmpty ? [0, 2, 3] : [.. a];
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        if (ctx.InputType(0) is not (DType.F32 or DType.F64))
        {
            throw ctx.Unsupported($"element type {ctx.InputType(0)} is not supported yet");
        }

        foreach (long a in _axes)
        {
            _ = OpUtil.Axis(ctx, a, ctx.InputShape(0).Rank);
        }

        ctx.SetOutput(0, ctx.InputType(0), ctx.InputShape(0));
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0];
        Span<bool> mask = stackalloc bool[Shape.MaxRank];
        mask.Clear();
        foreach (long a in _axes)
        {
            mask[(int)(a < 0 ? a + x.Rank : a)] = true;
        }

        if (x.DType == DType.F32)
        {
            Normalize<float>(x, run.Outputs[0], mask.Slice(0, x.Rank));
        }
        else
        {
            Normalize<double>(x, run.Outputs[0], mask.Slice(0, x.Rank));
        }
    }

    private static void Normalize<T>(Tensor xt, Tensor yt, ReadOnlySpan<bool> mask) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ReadOnlySpan<T> x = xt.Data<T>();
        Span<T> y = yt.MutableData<T>();
        Span<int> buf = stackalloc int[4 * Shape.MaxRank];
        ReduceExtra.Walk(xt.Shape.Dimensions, mask, buf, out int outputs, out int count, out Span<int> kd, out Span<int> ks, out Span<int> rd, out Span<int> rs);
        T n = T.CreateTruncating(count);
        T eps = T.CreateTruncating(1e-9);
        for (int o = 0; o < outputs; o++)
        {
            int b = ReduceExtra.Base(o, kd, ks);
            T sum = T.Zero;
            for (int r = 0; r < count; r++)
            {
                sum += x[b + ReduceExtra.Offset(r, rd, rs)];
            }

            T mean = sum / n;
            T sq = T.Zero;
            for (int r = 0; r < count; r++)
            {
                T d = x[b + ReduceExtra.Offset(r, rd, rs)] - mean;
                sq += d * d;
            }

            T inv = T.Sqrt((sq / n) + eps);
            for (int r = 0; r < count; r++)
            {
                int at = b + ReduceExtra.Offset(r, rd, rs);
                y[at] = (x[at] - mean) / inv;
            }
        }
    }
}

/// <summary>
/// LRN: local response normalisation across channels, y = x / (bias + alpha / size * sum of squares over a window of 'size' channels)^beta,
/// the window being channels c - floor((size - 1) / 2) to c + ceil((size - 1) / 2) clipped to the tensor. Uses pow, so it is not exact.
/// </summary>
internal sealed class LrnOp : OpKernel
{
    public static readonly OpDef Def = new("LRN", 1, int.MaxValue, static n => new LrnOp(n));

    private readonly float _alpha, _beta, _bias;
    private readonly int _size;

    private LrnOp(NodeInfo n)
    {
        _alpha = n.GetFloat("alpha", 1e-4f);
        _beta = n.GetFloat("beta", 0.75f);
        _bias = n.GetFloat("bias", 1f);
        long size = n.GetInt("size", 0);
        if (size < 1 || size > 1 << 20)
        {
            throw n.Error("attribute 'size' is required and must be at least 1");
        }

        _size = (int)size;
    }

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        if (ctx.InputType(0) != DType.F32 || ctx.InputShape(0).Rank < 2)
        {
            throw ctx.Unsupported("only float32 inputs of rank 2 or more are supported");
        }

        ctx.SetOutput(0, DType.F32, ctx.InputShape(0));
    }

    public override void Run(in OpRun run)
    {
        Tensor xt = run.Inputs[0];
        ReadOnlySpan<float> x = xt.Data<float>();
        Span<float> y = run.Outputs[0].MutableData<float>();
        int n = xt.Shape[0], c = xt.Shape[1], inner = OpUtil.Product(xt.Shape.Dimensions, 2, xt.Rank);
        int before = (_size - 1) / 2, after = _size / 2;
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < c; k++)
            {
                int lo = Math.Max(0, k - before), hi = Math.Min(c - 1, k + after);
                for (int s = 0; s < inner; s++)
                {
                    float sum = 0;
                    for (int j = lo; j <= hi; j++)
                    {
                        float e = x[(((i * c) + j) * inner) + s];
                        sum += e * e;
                    }

                    int at = (((i * c) + k) * inner) + s;
                    y[at] = x[at] / MathF.Pow(_bias + (_alpha / _size * sum), _beta);
                }
            }
        }
    }
}

/// <summary>Det: the determinant of each trailing square matrix, by LU decomposition with partial pivoting in binary64 (the product of the pivots, signed by the row swaps).</summary>
internal sealed class DetOp : OpKernel
{
    public static readonly OpDef Def = new("Det", 11, int.MaxValue, static _ => new DetOp());

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        Shape x = ctx.InputShape(0);
        if (ctx.InputType(0) is not (DType.F32 or DType.F64))
        {
            throw ctx.Unsupported($"element type {ctx.InputType(0)} is not supported yet");
        }

        if (x.Rank < 2 || x[x.Rank - 1] != x[x.Rank - 2])
        {
            throw ctx.Fail($"the input {x} must end in two equal dimensions");
        }

        ctx.SetOutput(0, ctx.InputType(0), Shape.Create(x.Dimensions[..(x.Rank - 2)]));
    }

    public override void Run(in OpRun run)
    {
        Tensor xt = run.Inputs[0];
        int n = xt.Shape[xt.Rank - 1];
        int batch = n == 0 ? Math.Max(xt.Length, 1) : xt.Length / (n * n);
        if (xt.DType == DType.F32)
        {
            ReadOnlySpan<float> x = xt.Data<float>();
            Span<float> y = run.Outputs[0].MutableData<float>();
            for (int b = 0; b < y.Length; b++)
            {
                y[b] = (float)Determinant(x.Slice(b * n * n, n * n), n);
            }
        }
        else
        {
            ReadOnlySpan<double> x = xt.Data<double>();
            Span<double> y = run.Outputs[0].MutableData<double>();
            for (int b = 0; b < y.Length; b++)
            {
                y[b] = Determinant(x.Slice(b * n * n, n * n), n);
            }
        }

        _ = batch;
    }

    private static double Determinant<T>(ReadOnlySpan<T> m, int n) where T : unmanaged, IFloatingPointIeee754<T>
    {
        // Small matrices only: the working copy is on the stack up to 16 x 16 and on the heap beyond, which no model needs.
        Span<double> a = n <= 16 ? stackalloc double[n * n] : new double[n * n];
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = double.CreateTruncating(m[i]);
        }

        double det = 1;
        for (int col = 0; col < n; col++)
        {
            int pivot = col;
            for (int r = col + 1; r < n; r++)
            {
                if (Math.Abs(a[(r * n) + col]) > Math.Abs(a[(pivot * n) + col]))
                {
                    pivot = r;
                }
            }

            if (a[(pivot * n) + col] == 0)
            {
                return 0;
            }

            if (pivot != col)
            {
                for (int k = 0; k < n; k++)
                {
                    (a[(pivot * n) + k], a[(col * n) + k]) = (a[(col * n) + k], a[(pivot * n) + k]);
                }

                det = -det;
            }

            det *= a[(col * n) + col];
            for (int r = col + 1; r < n; r++)
            {
                double f = a[(r * n) + col] / a[(col * n) + col];
                for (int k = col + 1; k < n; k++)
                {
                    a[(r * n) + k] -= f * a[(col * n) + k];
                }
            }
        }

        return det;
    }
}
