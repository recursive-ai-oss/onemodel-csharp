using System;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Kernels.Exact;
namespace RecursiveAI.OneModel.Tensors;

/// <summary>Geometry of a 2-D convolution: stride, symmetric zero padding, dilation (each per axis) and channel groups.</summary>
/// <param name="StrideH">Vertical step of the window, at least 1.</param>
/// <param name="StrideW">Horizontal step, at least 1.</param>
/// <param name="PadH">Rows of zeros added at the top and at the bottom, at least 0.</param>
/// <param name="PadW">Columns of zeros added left and right, at least 0.</param>
/// <param name="DilH">Spacing between kernel rows, at least 1.</param>
/// <param name="DilW">Spacing between kernel columns, at least 1.</param>
/// <param name="Groups">Channel groups; both channel counts must be divisible by it. Groups equal to the input channels is a depthwise convolution.</param>
public readonly record struct Conv2dParams(int StrideH = 1, int StrideW = 1, int PadH = 0, int PadW = 0, int DilH = 1, int DilW = 1, int Groups = 1);

public static partial class Ops
{
    private static bool Recording(params ReadOnlySpan<Tensor> inputs)
    {
        if (!GradMode.IsEnabled)
        {
            return false;
        }

        foreach (Tensor t in inputs)
        {
            if (t.RequiresGrad)
            {
                return true;
            }
        }

        return false;
    }

    // The tensor itself if contiguous, else a contiguous copy owned by the caller (freed by the record or the caller).
    private static Tensor Contig(in Tensor t, out bool copied)
    {
        RequireFloat32(t);
        copied = !t.IsContiguous;
        return copied ? t.Clone() : t;
    }

    private static void Free(in Tensor t, bool copied)
    {
        if (copied)
        {
            t.Dispose();
        }
    }

    // Saves `t` (a contiguous version of an input) in the record; a copy made for the purpose belongs to the record.
    private static void SaveContig(BackwardRecord rec, int slot, in Tensor t, bool copied) => rec.Save(slot, t, owned: copied);

    // ---------------------------------------------------------------------------------------------------------------- activations

    /// <summary>
    /// Gaussian error linear unit, <c>x·Φ(x)</c> (erf form) or the tanh approximation. Uses our fixed polynomials in both modes.
    /// Record: slot 0 = the input (the slope is recomputed), Ints = [0 erf form, 1 tanh form].
    /// </summary>
    /// <param name="x">Float32 input of any shape and strides.</param>
    /// <param name="approximateTanh">True for the tanh form <c>0.5x(1+tanh(√(2/π)(x+0.044715x³)))</c>; default false (erf).</param>
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    public static Tensor Gelu(in Tensor x, bool approximateTanh = false, BufferPool? pool = null)
    {
        Tensor xc = Contig(x, out bool copied);
        Tensor r = NewOut(x.Shape, pool);
        ReadOnlySpan<float> s = xc.Data<float>();
        Span<float> d = r.MutableData<float>();
        // The polynomial forms, in both modes: the lane loop and its scalar tail give the bits of Fn.GeluErf / Fn.GeluTanh.
        if (approximateTanh)
        {
            VectorMaps.Map1(new GeluUnary(true), s, d, exact: true);
        }
        else
        {
            VectorMaps.Map1(new GeluUnary(false), s, d, exact: true);
        }

        if (!Recording(x))
        {
            Free(xc, copied);
            return r;
        }

        BackwardRecord rec = BackwardRecord.Create(OpKind.Gelu, 1, x);
        SaveContig(rec, 0, xc, copied);
        rec.SetInts(approximateTanh ? 1 : 0);
        return rec.Attach(r);
    }

    /// <summary>Logistic sigmoid <c>1/(1+e^-x)</c>. Record: slot 0 = the output.</summary>
    public static Tensor Sigmoid(in Tensor x, BufferPool? pool = null) => Activation(OpKind.Sigmoid, x, pool);

    /// <summary>Hyperbolic tangent. Record: slot 0 = the output.</summary>
    public static Tensor Tanh(in Tensor x, BufferPool? pool = null) => Activation(OpKind.Tanh, x, pool);

    private static Tensor Activation(OpKind op, in Tensor x, BufferPool? pool)
    {
        bool exact = TensorCompute.Exact;
        Tensor xc = Contig(x, out bool copied);
        Tensor r = NewOut(x.Shape, pool);
        ReadOnlySpan<float> s = xc.Data<float>();
        Span<float> d = r.MutableData<float>();
        if (op == OpKind.Sigmoid)
        {
            VectorMaps.Map1(new SigmoidUnary(), s, d);
        }
        else
        {
            VectorMaps.Map1(new TanhUnary(), s, d);
        }

        Free(xc, copied);
        if (!Recording(x))
        {
            return r;
        }

        BackwardRecord rec = BackwardRecord.Create(op, 1, x);
        rec.Save(0, r);
        return rec.Attach(r);
    }

    // ---------------------------------------------------------------------------------------------------------------- reductions

    /// <summary>Sum of all elements, as a rank-0 tensor. Accumulates in double in element order. Record: Ints = [-1].</summary>
    public static Tensor Sum(in Tensor x, BufferPool? pool = null) => Reduce(OpKind.Sum, x, 0, false, pool, all: true);

    /// <summary>Sum over one axis (negative counts from the end). Record: Ints = [axis, keepDim].</summary>
    /// <param name="x">Float32 input.</param>
    /// <param name="axis">Axis to reduce; negative counts from the end.</param>
    /// <param name="keepDim">Keep the reduced axis with extent 1 (default false removes it).</param>
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    public static Tensor Sum(in Tensor x, int axis, bool keepDim = false, BufferPool? pool = null) => Reduce(OpKind.Sum, x, axis, keepDim, pool);

    /// <summary>Mean of all elements, as a rank-0 tensor (the double sum divided by the count). Record: Ints = [-1].</summary>
    public static Tensor Mean(in Tensor x, BufferPool? pool = null) => Reduce(OpKind.Mean, x, 0, false, pool, all: true);

    /// <summary>Mean over one axis. Record: Ints = [axis, keepDim].</summary>
    /// <param name="x">Float32 input.</param>
    /// <param name="axis">Axis to reduce; negative counts from the end.</param>
    /// <param name="keepDim">Keep the reduced axis with extent 1 (default false removes it).</param>
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    public static Tensor Mean(in Tensor x, int axis, bool keepDim = false, BufferPool? pool = null) => Reduce(OpKind.Mean, x, axis, keepDim, pool);

    private static Tensor Reduce(OpKind op, in Tensor x, int axis, bool keepDim, BufferPool? pool, bool all = false)
    {
        Tensor xc = Contig(x, out bool copied);
        ReadOnlySpan<float> s = xc.Data<float>();
        int outer = 1, n = s.Length, inner = 1, ax = -1;
        Shape os = default;
        if (!all)
        {
            ax = Shape.NormalizeAxis(axis, x.Rank);
            n = x.Shape[ax];
            for (int i = 0; i < ax; i++)
            {
                outer *= x.Shape[i];
            }

            for (int i = ax + 1; i < x.Rank; i++)
            {
                inner *= x.Shape[i];
            }

            Span<int> od = stackalloc int[Shape.MaxRank];
            int c = 0;
            for (int i = 0; i < x.Rank; i++)
            {
                if (i != ax)
                {
                    od[c++] = x.Shape[i];
                }
                else if (keepDim)
                {
                    od[c++] = 1;
                }
            }

            os = Shape.Create(od.Slice(0, c));
        }

        Tensor r = NewOut(os, pool);
        Span<float> d = r.MutableData<float>();
        for (int o = 0; o < outer; o++)
        {
            for (int i = 0; i < inner; i++)
            {
                double acc = 0;
                for (int k = 0; k < n; k++)
                {
                    acc += s[(((o * n) + k) * inner) + i];
                }

                d[(o * inner) + i] = op == OpKind.Mean ? (float)(acc / n) : (float)acc;
            }
        }

        Free(xc, copied);
        if (!Recording(x))
        {
            return r;
        }

        BackwardRecord rec = BackwardRecord.Create(op, 0, x);
        rec.SetInts(all ? -1 : ax, keepDim ? 1 : 0);
        return rec.Attach(r);
    }
}
