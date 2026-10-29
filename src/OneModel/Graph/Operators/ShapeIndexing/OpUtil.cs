using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

/// <summary>A read-only list of int32 or int64 values (an axes, starts or pads input), read without copying or converting.</summary>
internal readonly ref struct IntVec
{
    private readonly ReadOnlySpan<long> _wide;
    private readonly ReadOnlySpan<int> _narrow;

    public IntVec(ReadOnlySpan<long> values)
    {
        _wide = values;
        _narrow = default;
        Length = values.Length;
    }

    public IntVec(ReadOnlySpan<int> values)
    {
        _wide = default;
        _narrow = values;
        Length = values.Length;
    }

    public int Length { get; }

    public long this[int i] => _wide.Length > 0 ? _wide[i] : _narrow[i];

    /// <summary>The values of a contiguous int32 or int64 tensor (callers have checked the type when shaping the node).</summary>
    public static IntVec Of(in Tensor t) => t.DType switch
    {
        DType.I64 => new IntVec(t.Data<long>()),
        DType.I32 => new IntVec(t.Data<int>()),
        _ => throw new InvalidOperationException($"Expected int32 or int64 values, not {t.DType}."),
    };
}

/// <summary>Small helpers shared by the shape and indexing operators.</summary>
internal static class OpUtil
{
    /// <summary>Builds a shape from 64-bit extents, reporting an extent that is negative or too large as a model error.</summary>
    public static Shape MakeShape(InferContext ctx, ReadOnlySpan<long> dims)
    {
        try
        {
            return Shape.Create(dims);
        }
        catch (TensorShapeException e)
        {
            throw ctx.Fail(e.Message);
        }
    }

    /// <summary>
    /// Builds a shape for an output that can be far larger than any input (ConstantOfShape, Expand, Tile, Pad, Range, OneHot) and
    /// refuses one whose bytes exceed what a tensor can hold (2^31-1), as a model error naming the node.
    /// </summary>
    public static Shape MakeGrownShape(InferContext ctx, ReadOnlySpan<long> dims, DType type)
    {
        Shape s = MakeShape(ctx, dims);
        if ((long)s.Length * DTypes.SizeOf(type) > int.MaxValue)
        {
            throw ctx.Fail($"the output {s} of {type} would be larger than 2 GiB");
        }

        return s;
    }

    /// <summary>Normalizes a possibly negative axis against <paramref name="rank"/> (valid range [-rank, rank)).</summary>
    public static int Axis(InferContext ctx, long axis, int rank, string what = "axis")
    {
        if (axis < -rank || axis >= rank)
        {
            throw ctx.Fail($"{what} {axis} is outside [{-rank}, {rank - 1}] for a rank-{rank} input");
        }

        return (int)(axis < 0 ? axis + rank : axis);
    }

    /// <summary>Requires input <paramref name="i"/> to be present, 1-D (or a scalar when <paramref name="allowScalar"/>) and int32 or int64.</summary>
    public static void RequireIntegers(InferContext ctx, int i, string what, bool allowScalar = false)
    {
        if (!ctx.HasInput(i))
        {
            throw ctx.Fail($"{what} input is missing");
        }

        if (ctx.InputType(i) is not (DType.I32 or DType.I64))
        {
            throw ctx.Fail($"{what} must be int32 or int64, not {ctx.InputType(i)}");
        }

        int rank = ctx.InputShape(i).Rank;
        if (rank > 1 || (rank == 0 && !allowScalar))
        {
            throw ctx.Fail($"{what} must be 1-D, not {ctx.InputShape(i)}");
        }
    }

    /// <summary>The first element of an int32 or int64 tensor.</summary>
    public static long First(in Tensor t) => IntVec.Of(t)[0];

    /// <summary>The first element of a numeric tensor as an int64 (floats truncate toward zero).</summary>
    public static long ScalarAsLong(in Tensor t) => t.DType switch
    {
        DType.I64 => t.Data<long>()[0],
        DType.I32 => t.Data<int>()[0],
        DType.I16 => t.Data<short>()[0],
        DType.I8 => t.Data<sbyte>()[0],
        DType.U8 => t.Data<byte>()[0],
        DType.F32 => (long)t.Data<float>()[0],
        DType.F64 => (long)t.Data<double>()[0],
        _ => throw new InvalidOperationException($"Cannot read a {t.DType} as an integer."),
    };

    /// <summary>Row-major strides of <paramref name="dims"/> in elements (extents of 0 count as 1).</summary>
    public static void RowMajor(ReadOnlySpan<int> dims, Span<int> strides)
    {
        long acc = 1;
        for (int i = dims.Length - 1; i >= 0; i--)
        {
            strides[i] = (int)acc;
            acc *= Math.Max(dims[i], 1);
        }
    }

    /// <summary>Product of dims[from..to).</summary>
    public static int Product(ReadOnlySpan<int> dims, int from, int to)
    {
        long p = 1;
        for (int i = from; i < to; i++)
        {
            p *= dims[i];
        }

        return (int)p;
    }
}
