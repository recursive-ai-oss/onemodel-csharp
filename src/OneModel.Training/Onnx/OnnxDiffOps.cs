using System;
using System.Collections.Generic;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Training;

/// <summary>An imported node's forward on the eager tensor operators, so it records a backward pass.</summary>
/// <param name="node">The node, for its attributes and opset.</param>
/// <param name="x">The input tensors in node order; an omitted optional input is an empty tensor.</param>
internal delegate Tensor[] DiffHandler(IrNode node, ReadOnlySpan<Tensor> x);

// The operators an imported graph can run with gradients. A node outside this table runs on the graph runtime's own kernel, which is
// allowed only while none of its inputs depends on a trainable parameter.
internal static class OnnxDiffOps
{
    // Operators whose outputs never carry a gradient (shapes, comparisons, indices), whatever their inputs.
    public static readonly HashSet<string> NoGradient = new(StringComparer.Ordinal)
    {
        "Shape", "Size", "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "Not", "And", "Or", "Xor", "IsNaN", "IsInf",
        "ArgMax", "ArgMin", "NonZero", "ConstantOfShape", "Floor", "Ceil", "Round", "Sign", "Range", "EyeLike", "RandomNormalLike", "RandomUniformLike",
    };

    public static readonly Dictionary<string, DiffHandler> Table = new(StringComparer.Ordinal)
    {
        ["Add"] = static (_, x) => [Ops.Add(x[0], x[1])],
        ["Sub"] = static (_, x) => [Ops.Sub(x[0], x[1])],
        ["Mul"] = static (_, x) => [Ops.Mul(x[0], x[1])],
        ["Div"] = static (_, x) => [Ops.Div(x[0], x[1])],
        ["Neg"] = static (_, x) => [Ops.Neg(x[0])],
        ["Relu"] = static (_, x) => [Ops.Relu(x[0])],
        ["Sigmoid"] = static (_, x) => [Ops.Sigmoid(x[0])],
        ["Tanh"] = static (_, x) => [Ops.Tanh(x[0])],
        ["Erf"] = static (_, x) => [Ops.Erf(x[0])],
        ["Exp"] = static (_, x) => [Ops.Exp(x[0])],
        ["Sqrt"] = static (_, x) => [Ops.Sqrt(x[0])],
        ["Reciprocal"] = static (_, x) => [Ops.Div(Scalar(1f), x[0])],
        ["Pow"] = static (n, x) => [Ops.Pow(x[0], ScalarValue(n, x[1]))],
        ["Gelu"] = static (n, x) => [Ops.Gelu(x[0], Text(n, "approximate", "none") == "tanh")],
        ["Identity"] = static (_, x) => [x[0]],
        ["Dropout"] = static (_, x) => [x[0]],
        ["MatMul"] = static (_, x) => [MatMul(x[0], x[1])],
        ["Gemm"] = Gemm,
        ["Transpose"] = Transpose,
        ["Reshape"] = Reshape,
        ["Flatten"] = Flatten,
        ["Squeeze"] = Squeeze,
        ["Unsqueeze"] = Unsqueeze,
        ["Slice"] = Slice,
        ["Split"] = Split,
        ["Concat"] = static (n, x) => [Ops.Concat(x, (int)Int(n, "axis", 0))],
        ["Gather"] = static (n, x) => [Ops.Gather(x[0], x[1], (int)Int(n, "axis", 0))],
        ["Softmax"] = static (n, x) => [Softmax(n, x[0], log: false)],
        ["LogSoftmax"] = static (n, x) => [Softmax(n, x[0], log: true)],
        ["LayerNormalization"] = LayerNormalization,
        ["ReduceMean"] = static (n, x) => [Reduce(n, x, mean: true)],
        ["ReduceSum"] = static (n, x) => [Reduce(n, x, mean: false)],
        ["Cast"] = Cast,
        ["Expand"] = Expand,
        ["Where"] = static (_, x) => [Ops.Where(x[0], x[1], x[2])],
    };

    // ------------------------------------------------------------------------------------------------------------ attributes, values

    private static long Int(IrNode n, string name, long fallback) => n.FindAttribute(name) is { } a ? a.Whole : fallback;

    private static float Float(IrNode n, string name, float fallback) => n.FindAttribute(name) is { } a ? a.Real : fallback;

    private static string Text(IrNode n, string name, string fallback) => n.FindAttribute(name) is { } a ? a.GetText() : fallback;

    private static long[]? Ints(IrNode n, string name) => n.FindAttribute(name) is { } a ? [.. a.Wholes] : null;

    private static Tensor Scalar(float v)
    {
        Tensor t = Tensor.Scalar(v);
        ActivationScope.Track(t);
        return t;
    }

    private static float ScalarValue(IrNode n, in Tensor t)
    {
        if (t.IsEmpty || t.Length != 1)
        {
            throw new NotSupportedException($"{n.OpType} '{n.Name}': only a single-element exponent is supported for a trainable input.");
        }

        Tensor c = t.IsContiguous ? t : t.Detach().Contiguous();
        return c.DType switch
        {
            DType.F32 => c.Data<float>()[0],
            DType.I64 => c.Data<long>()[0],
            DType.I32 => c.Data<int>()[0],
            _ => throw new NotSupportedException($"{n.OpType} '{n.Name}': exponent type {c.DType} is not supported."),
        };
    }

    internal static long[] Longs(in Tensor t)
    {
        Tensor c = t.IsContiguous ? t : t.Detach().Contiguous();
        if (c.DType == DType.I64)
        {
            return c.Data<long>().ToArray();
        }

        if (c.DType == DType.I32)
        {
            ReadOnlySpan<int> s = c.Data<int>();
            var r = new long[s.Length];
            for (int i = 0; i < r.Length; i++)
            {
                r[i] = s[i];
            }

            return r;
        }

        throw new NotSupportedException($"Expected an integer tensor, found {c.DType}.");
    }

    // The axes of Squeeze/Unsqueeze/Reduce: an input from opset 13 (18 for ReduceMean), an attribute before.
    private static long[]? AxesOf(IrNode n, ReadOnlySpan<Tensor> x, int inputIndex) =>
        x.Length > inputIndex && !x[inputIndex].IsEmpty ? Longs(x[inputIndex]) : Ints(n, "axes");

    private static Shape ShapeOf(ReadOnlySpan<long> dims)
    {
        Span<int> d = stackalloc int[Shape.MaxRank];
        if (dims.Length > Shape.MaxRank)
        {
            throw new TensorShapeException($"Rank {dims.Length} exceeds the maximum of {Shape.MaxRank}.");
        }

        for (int i = 0; i < dims.Length; i++)
        {
            d[i] = checked((int)dims[i]);
        }

        return Shape.Create(d.Slice(0, dims.Length));
    }

    // ------------------------------------------------------------------------------------------------------------ operators

    private static Tensor MatMul(in Tensor a0, in Tensor b0)
    {
        Tensor a = a0, b = b0;
        bool aVec = a.Rank == 1, bVec = b.Rank == 1;
        if (aVec)
        {
            a = a.Unsqueeze(0);
        }

        if (bVec)
        {
            b = b.Unsqueeze(1);
        }

        Tensor r = Ops.MatMulBatched(a, b);
        if (aVec)
        {
            r = r.Squeeze(r.Rank - 2);
        }

        if (bVec)
        {
            r = r.Squeeze(r.Rank - 1);
        }

        return r;
    }

    private static Tensor[] Gemm(IrNode n, ReadOnlySpan<Tensor> x)
    {
        Tensor a = Int(n, "transA", 0) != 0 ? x[0].T() : x[0];
        Tensor b = Int(n, "transB", 0) != 0 ? x[1].T() : x[1];
        float alpha = Float(n, "alpha", 1f), beta = Float(n, "beta", 1f);
        Tensor y = Ops.MatMul(a, b);
        if (alpha != 1f)
        {
            y = Ops.Mul(y, Scalar(alpha));
        }

        if (x.Length > 2 && !x[2].IsEmpty)
        {
            Tensor c = beta != 1f ? Ops.Mul(x[2], Scalar(beta)) : x[2];
            y = Ops.Add(y, c);
        }

        return [y];
    }

    private static Tensor[] Transpose(IrNode n, ReadOnlySpan<Tensor> x)
    {
        long[]? perm = Ints(n, "perm");
        Span<int> p = stackalloc int[Shape.MaxRank];
        for (int i = 0; i < x[0].Rank; i++)
        {
            p[i] = perm is null ? x[0].Rank - 1 - i : (int)perm[i];
        }

        return [x[0].Permute(p.Slice(0, x[0].Rank))];
    }

    private static Tensor[] Reshape(IrNode n, ReadOnlySpan<Tensor> x)
    {
        long[] spec = Longs(x[1]);
        bool allowZero = Int(n, "allowzero", 0) != 0;
        Span<long> dims = stackalloc long[Shape.MaxRank];
        if (spec.Length > Shape.MaxRank)
        {
            throw new TensorShapeException($"Reshape to rank {spec.Length} exceeds the maximum of {Shape.MaxRank}.");
        }

        long known = 1;
        int unknown = -1;
        for (int i = 0; i < spec.Length; i++)
        {
            long d = spec[i];
            if (d == 0 && !allowZero)
            {
                d = i < x[0].Rank ? x[0].Shape[i] : throw new TensorShapeException($"Reshape: 0 at position {i} has no matching input axis.");
            }

            if (d == -1)
            {
                unknown = i;
            }
            else
            {
                known *= d;
            }

            dims[i] = d;
        }

        if (unknown >= 0)
        {
            dims[unknown] = known == 0 ? 0 : x[0].Length / known;
        }

        return [x[0].Reshape(ShapeOf(dims.Slice(0, spec.Length)))];
    }

    private static Tensor[] Flatten(IrNode n, ReadOnlySpan<Tensor> x)
    {
        int axis = (int)Int(n, "axis", 1);
        if (axis < 0)
        {
            axis += x[0].Rank;
        }

        int outer = 1;
        for (int i = 0; i < axis; i++)
        {
            outer *= x[0].Shape[i];
        }

        int inner = outer == 0 ? 0 : x[0].Length / outer;
        return [x[0].Reshape(Shape.Create(outer, inner))];
    }

    private static Tensor[] Squeeze(IrNode n, ReadOnlySpan<Tensor> x)
    {
        long[]? axes = AxesOf(n, x, 1);
        if (axes is null)
        {
            return [x[0].Squeeze()];
        }

        int rank = x[0].Rank;
        var sorted = new int[axes.Length];
        for (int i = 0; i < sorted.Length; i++)
        {
            sorted[i] = Shape.NormalizeAxis((int)axes[i], rank);
        }

        Array.Sort(sorted);
        Tensor t = x[0];
        for (int i = sorted.Length - 1; i >= 0; i--)
        {
            t = t.Squeeze(sorted[i]);
        }

        return [t];
    }

    private static Tensor[] Unsqueeze(IrNode n, ReadOnlySpan<Tensor> x)
    {
        long[] axes = AxesOf(n, x, 1) ?? throw new NotSupportedException($"Unsqueeze '{n.Name}' has no axes.");
        int outRank = x[0].Rank + axes.Length;
        var sorted = new int[axes.Length];
        for (int i = 0; i < sorted.Length; i++)
        {
            sorted[i] = Shape.NormalizeAxis((int)axes[i], outRank);
        }

        Array.Sort(sorted);
        Tensor t = x[0];
        foreach (int a in sorted)
        {
            t = t.Unsqueeze(a);
        }

        return [t];
    }

    private static Tensor[] Slice(IrNode n, ReadOnlySpan<Tensor> x)
    {
        long[] starts, ends;
        long[]? axes, steps;
        if (x.Length > 1)
        {
            starts = Longs(x[1]);
            ends = Longs(x[2]);
            axes = x.Length > 3 && !x[3].IsEmpty ? Longs(x[3]) : null;
            steps = x.Length > 4 && !x[4].IsEmpty ? Longs(x[4]) : null;
        }
        else
        {
            starts = Ints(n, "starts") ?? [];
            ends = Ints(n, "ends") ?? [];
            axes = Ints(n, "axes");
            steps = null;
        }

        Tensor t = x[0];
        for (int i = 0; i < starts.Length; i++)
        {
            int axis = Shape.NormalizeAxis((int)(axes is null ? i : axes[i]), t.Rank);
            long step = steps is null ? 1 : steps[i];
            if (step < 1)
            {
                throw new NotSupportedException($"Slice '{n.Name}': a non-positive step ({step}) on a trainable value is not supported.");
            }

            long dim = t.Shape[axis];
            long s = starts[i] < 0 ? starts[i] + dim : starts[i];
            long e = ends[i] < 0 ? ends[i] + dim : ends[i];
            s = Math.Clamp(s, 0, dim);
            e = Math.Clamp(e, 0, dim);
            if (e < s)
            {
                e = s;
            }

            t = t.Slice(axis, (int)s, (int)e, (int)step);
        }

        return [t];
    }

    private static Tensor[] Split(IrNode n, ReadOnlySpan<Tensor> x)
    {
        int axis = Shape.NormalizeAxis((int)Int(n, "axis", 0), x[0].Rank);
        int dim = x[0].Shape[axis];
        int outputs = n.OutputNames.Length;
        long[]? sizes = x.Length > 1 && !x[1].IsEmpty ? Longs(x[1]) : Ints(n, "split");
        if (sizes is null)
        {
            long count = n.FindAttribute("num_outputs") is { } a ? a.Whole : outputs;
            long chunk = (dim + count - 1) / count;
            sizes = new long[count];
            long left = dim;
            for (int i = 0; i < count; i++)
            {
                sizes[i] = Math.Min(chunk, left);
                left -= sizes[i];
            }
        }

        var result = new Tensor[sizes.Length];
        int start = 0;
        for (int i = 0; i < sizes.Length; i++)
        {
            result[i] = x[0].Slice(axis, start, start + (int)sizes[i]);
            start += (int)sizes[i];
        }

        return result;
    }

    private static Tensor Softmax(IrNode n, in Tensor x, bool log)
    {
        int axis = (int)Int(n, "axis", n.OpsetVersion >= 13 ? -1 : 1);
        if (n.OpsetVersion >= 13 || axis == -1 || axis == x.Rank - 1)
        {
            return log ? Ops.LogSoftmax(x, axis) : Ops.Softmax(x, axis);
        }

        // Before opset 13 the input is flattened to 2-D at the axis and the softmax runs over the second dimension.
        int a = Shape.NormalizeAxis(axis, x.Rank);
        int outer = 1;
        for (int i = 0; i < a; i++)
        {
            outer *= x.Shape[i];
        }

        Tensor flat = x.Reshape(Shape.Create(outer, outer == 0 ? 0 : x.Length / outer));
        Tensor y = log ? Ops.LogSoftmax(flat, -1) : Ops.Softmax(flat, -1);
        return y.Reshape(x.Shape);
    }

    private static Tensor[] LayerNormalization(IrNode n, ReadOnlySpan<Tensor> x)
    {
        Tensor data = x[0];
        int axis = Shape.NormalizeAxis((int)Int(n, "axis", -1), data.Rank);
        float eps = Float(n, "epsilon", 1e-5f);
        int features = 1;
        for (int i = axis; i < data.Rank; i++)
        {
            features *= data.Shape[i];
        }

        // Normalizing over several trailing axes is normalizing over their flattened extent.
        Tensor d = axis == data.Rank - 1 ? data : data.Reshape(Shape.Create(features == 0 ? 0 : data.Length / features, features));
        Tensor scale = x[1].Rank == 1 ? x[1] : x[1].Reshape(Shape.Create(features));
        Tensor bias;
        if (x.Length > 2 && !x[2].IsEmpty)
        {
            bias = x[2].Rank == 1 ? x[2] : x[2].Reshape(Shape.Create(features));
        }
        else
        {
            bias = Tensor.Zeros(DType.F32, Shape.Create(features));
            ActivationScope.Track(bias);
        }

        Tensor y = Ops.LayerNorm(d, scale, bias, eps);
        return [axis == data.Rank - 1 ? y : y.Reshape(data.Shape)];
    }

    private static Tensor Reduce(IrNode n, ReadOnlySpan<Tensor> x, bool mean)
    {
        bool keep = Int(n, "keepdims", 1) != 0;
        long[]? axes = AxesOf(n, x, 1);
        Tensor t = x[0];
        if (axes is null || axes.Length == 0)
        {
            if (Int(n, "noop_with_empty_axes", 0) != 0)
            {
                return t;
            }

            axes = new long[t.Rank];
            for (int i = 0; i < axes.Length; i++)
            {
                axes[i] = i;
            }
        }

        var sorted = new int[axes.Length];
        for (int i = 0; i < sorted.Length; i++)
        {
            sorted[i] = Shape.NormalizeAxis((int)axes[i], x[0].Rank);
        }

        Array.Sort(sorted);
        // A mean over several axes is the mean of the means over equal-sized groups, so reducing one axis at a time is exact in value.
        for (int i = sorted.Length - 1; i >= 0; i--)
        {
            t = mean ? Ops.Mean(t, sorted[i], keep) : Ops.Sum(t, sorted[i], keep);
        }

        return t;
    }

    private static Tensor[] Cast(IrNode n, ReadOnlySpan<Tensor> x) =>
        [x[0].DType == DType.F32 ? x[0] : x[0].Cast(DType.F32)];

    private static Tensor[] Expand(IrNode n, ReadOnlySpan<Tensor> x)
    {
        long[] dims = Longs(x[1]);
        Shape target = Shape.Broadcast(x[0].Shape, ShapeOf(dims));
        return [x[0].Expand(target)];
    }
}
