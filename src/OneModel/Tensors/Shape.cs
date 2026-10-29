using System;
using System.Collections.Generic;
using System.Threading;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Text;

namespace RecursiveAI.OneModel.Tensors;

/// <summary>A tensor's shape, strides or a requested size is invalid, or its arithmetic overflows.</summary>
public sealed class TensorShapeException : ArgumentException
{
    /// <summary>Creates the exception with a message naming what was wrong.</summary>
    public TensorShapeException(string message) : base(message) { }

    /// <summary>Creates the exception with a message and the offending parameter.</summary>
    public TensorShapeException(string message, string? paramName) : base(message, paramName) { }
}

[InlineArray(Shape.MaxRank)]
internal struct DimArray
{
    private int _e0;
}

/// <summary>
/// The extents of a tensor: up to <see cref="MaxRank"/> dimensions, each at least 0, whose product (the element count) fits in an
/// <see cref="int"/>. Rank 0 is a scalar with one element. A value type with no allocation; the default value is the scalar shape.
/// </summary>
public readonly struct Shape : IEquatable<Shape>
{
    /// <summary>The largest supported rank.</summary>
    public const int MaxRank = 8;

    private readonly DimArray _d;
    private readonly int _rank;

    private Shape(in DimArray d, int rank)
    {
        _d = d;
        _rank = rank;
    }

    /// <summary>Number of dimensions.</summary>
    public int Rank => _rank;

    /// <summary>The extent of dimension <paramref name="axis"/> (0-based, no negative indexing; see <see cref="NormalizeAxis"/>).</summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="axis"/> is outside [0, Rank).</exception>
    public int this[int axis]
    {
        get
        {
            if ((uint)axis >= (uint)_rank)
            {
                throw new ArgumentOutOfRangeException(nameof(axis), axis, "The axis is outside the shape's rank.");
            }

            return _d[axis];
        }
    }

    /// <summary>The element count: the product of the extents (1 for a scalar, 0 if any extent is 0).</summary>
    public int Length
    {
        get
        {
            // The product was checked to fit in an int when the shape was built.
            int n = 1;
            for (int i = 0; i < _rank; i++)
            {
                n *= _d[i];
            }

            return n;
        }
    }

    /// <summary>The extents as a span of length <see cref="Rank"/>.</summary>
    [UnscopedRef]
    public ReadOnlySpan<int> Dimensions => ((ReadOnlySpan<int>)_d).Slice(0, _rank);

    /// <summary>Builds a shape from extents.</summary>
    /// <exception cref="TensorShapeException">The rank exceeds <see cref="MaxRank"/>, an extent is negative, or the element count exceeds <see cref="int.MaxValue"/>.</exception>
    public static Shape Create(params ReadOnlySpan<int> dims)
    {
        if (dims.Length > MaxRank)
        {
            throw new TensorShapeException($"Rank {dims.Length} exceeds the maximum of {MaxRank}.", nameof(dims));
        }

        DimArray d = default;
        long n = 1;
        for (int i = 0; i < dims.Length; i++)
        {
            if (dims[i] < 0)
            {
                throw new TensorShapeException($"Dimension {i} is negative ({dims[i]}).", nameof(dims));
            }

            d[i] = dims[i];
            n *= dims[i];
            if (n > int.MaxValue)
            {
                // The product is bounded by int.MaxValue before each multiply, so a long never overflows here; zero extents reset it.
                throw new TensorShapeException($"The element count overflows 2^31-1 at dimension {i}.", nameof(dims));
            }
        }

        // A zero anywhere makes the count 0, but the check above ran left to right: 100000 x 100000 x 0 is rejected too, which
        // keeps the rule simple (every prefix product must fit).
        return new Shape(d, dims.Length);
    }

    /// <summary>Builds a shape from 64-bit extents (as ONNX stores them).</summary>
    /// <exception cref="TensorShapeException">As <see cref="Create(ReadOnlySpan{int})"/>, or an extent does not fit in an int.</exception>
    public static Shape Create(ReadOnlySpan<long> dims)
    {
        if (dims.Length > MaxRank)
        {
            throw new TensorShapeException($"Rank {dims.Length} exceeds the maximum of {MaxRank}.", nameof(dims));
        }

        Span<int> tmp = stackalloc int[MaxRank];
        for (int i = 0; i < dims.Length; i++)
        {
            if (dims[i] < 0 || dims[i] > int.MaxValue)
            {
                throw new TensorShapeException($"Dimension {i} ({dims[i]}) is negative or larger than 2^31-1.", nameof(dims));
            }

            tmp[i] = (int)dims[i];
        }

        return Create(tmp.Slice(0, dims.Length));
    }

    /// <summary>Normalizes a possibly negative axis (-1 is the last) against <paramref name="rank"/>.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The axis is outside [-rank, rank).</exception>
    public static int NormalizeAxis(int axis, int rank)
    {
        int a = axis < 0 ? axis + rank : axis;
        if ((uint)a >= (uint)rank)
        {
            throw new ArgumentOutOfRangeException(nameof(axis), axis, $"The axis is outside [-{rank}, {rank}).");
        }

        return a;
    }

    /// <summary>
    /// Multidirectional (numpy/ONNX) broadcasting: shapes are right-aligned and each pair of extents must be equal or one of them 1.
    /// </summary>
    public static bool TryBroadcast(Shape a, Shape b, out Shape result)
    {
        int rank = Math.Max(a._rank, b._rank);
        DimArray d = default;
        for (int i = 0; i < rank; i++)
        {
            int ia = i - (rank - a._rank), ib = i - (rank - b._rank);
            int x = ia >= 0 ? a._d[ia] : 1, y = ib >= 0 ? b._d[ib] : 1;
            if (x == y || y == 1)
            {
                d[i] = x;
            }
            else if (x == 1)
            {
                d[i] = y;
            }
            else
            {
                result = default;
                return false;
            }
        }

        // The result's count is the larger of the operands' counts per axis, so it can still overflow (e.g. [65536,1] with [1,65536]).
        long n = 1;
        for (int i = 0; i < rank; i++)
        {
            n *= d[i];
            if (n > int.MaxValue)
            {
                result = default;
                return false;
            }
        }

        result = new Shape(d, rank);
        return true;
    }

    /// <summary>Broadcasts two shapes.</summary>
    /// <exception cref="TensorShapeException">The shapes are not broadcast-compatible, or the result is too large.</exception>
    public static Shape Broadcast(Shape a, Shape b) =>
        TryBroadcast(a, b, out Shape r) ? r : throw new TensorShapeException($"Shapes {a} and {b} cannot be broadcast together.");

    /// <summary>Row-major contiguous strides for this shape (extents of 0 are treated as 1 so strides stay meaningful).</summary>
    public Strides ContiguousStrides()
    {
        DimArray s = default;
        int acc = 1;
        for (int i = _rank - 1; i >= 0; i--)
        {
            s[i] = acc;
            acc *= Math.Max(_d[i], 1);
        }

        return new Strides(s, _rank);
    }

    internal int Unchecked(int i) => _d[i];

    internal Shape With(int axis, int value)
    {
        DimArray d = _d;
        d[axis] = value;
        return new Shape(d, _rank);
    }

    internal Shape Remove(int axis)
    {
        DimArray d = default;
        for (int i = 0, j = 0; i < _rank; i++)
        {
            if (i != axis)
            {
                d[j++] = _d[i];
            }
        }

        return new Shape(d, _rank - 1);
    }

    internal Shape Insert(int axis, int value)
    {
        DimArray d = default;
        for (int i = 0, j = 0; i <= _rank; i++)
        {
            d[i] = i == axis ? value : _d[j++];
        }

        return new Shape(d, _rank + 1);
    }

    /// <summary>Extent-wise equality.</summary>
    public bool Equals(Shape other)
    {
        if (_rank != other._rank)
        {
            return false;
        }

        for (int i = 0; i < _rank; i++)
        {
            if (_d[i] != other._d[i])
            {
                return false;
            }
        }

        return true;
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is Shape s && Equals(s);

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        HashCode h = default;
        for (int i = 0; i < _rank; i++)
        {
            h.Add(_d[i]);
        }

        return h.ToHashCode();
    }

    /// <summary>Extent-wise equality.</summary>
    public static bool operator ==(Shape a, Shape b) => a.Equals(b);

    /// <summary>Extent-wise inequality.</summary>
    public static bool operator !=(Shape a, Shape b) => !a.Equals(b);

    /// <summary>Formats as <c>[2,3,4]</c>.</summary>
    public override string ToString()
    {
        var sb = new StringBuilder("[");
        for (int i = 0; i < _rank; i++)
        {
            if (i > 0)
            {
                sb.Append(',');
            }

            sb.Append(_d[i]);
        }

        return sb.Append(']').ToString();
    }
}

/// <summary>
/// Per-dimension steps, in elements, from one index to the next. May be zero (a broadcast dimension) or negative (a flipped
/// dimension). Row-major contiguous strides are <see cref="Shape.ContiguousStrides"/>.
/// </summary>
public readonly struct Strides : IEquatable<Strides>
{
    private readonly DimArray _d;
    private readonly int _rank;

    internal Strides(in DimArray d, int rank)
    {
        _d = d;
        _rank = rank;
    }

    /// <summary>Number of dimensions (equals the shape's rank).</summary>
    public int Rank => _rank;

    /// <summary>The stride of dimension <paramref name="axis"/> in elements.</summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="axis"/> is outside [0, Rank).</exception>
    public int this[int axis]
    {
        get
        {
            if ((uint)axis >= (uint)_rank)
            {
                throw new ArgumentOutOfRangeException(nameof(axis), axis, "The axis is outside the strides' rank.");
            }

            return _d[axis];
        }
    }

    /// <summary>The strides as a span of length <see cref="Rank"/>.</summary>
    [UnscopedRef]
    public ReadOnlySpan<int> Values => ((ReadOnlySpan<int>)_d).Slice(0, _rank);

    /// <summary>Builds strides from values.</summary>
    /// <exception cref="TensorShapeException">The rank exceeds <see cref="Shape.MaxRank"/>.</exception>
    public static Strides Create(params ReadOnlySpan<int> values)
    {
        if (values.Length > Shape.MaxRank)
        {
            throw new TensorShapeException($"Rank {values.Length} exceeds the maximum of {Shape.MaxRank}.", nameof(values));
        }

        DimArray d = default;
        values.CopyTo(d);
        return new Strides(d, values.Length);
    }

    internal int Unchecked(int i) => _d[i];

    internal Strides With(int axis, int value)
    {
        DimArray d = _d;
        d[axis] = value;
        return new Strides(d, _rank);
    }

    internal Strides Remove(int axis)
    {
        DimArray d = default;
        for (int i = 0, j = 0; i < _rank; i++)
        {
            if (i != axis)
            {
                d[j++] = _d[i];
            }
        }

        return new Strides(d, _rank - 1);
    }

    internal Strides Insert(int axis, int value)
    {
        DimArray d = default;
        for (int i = 0, j = 0; i <= _rank; i++)
        {
            d[i] = i == axis ? value : _d[j++];
        }

        return new Strides(d, _rank + 1);
    }

    /// <summary>True when these are the row-major strides of <paramref name="shape"/>; dimensions of extent 1 may have any stride.</summary>
    public bool IsContiguousFor(Shape shape)
    {
        if (shape.Rank != _rank)
        {
            return false;
        }

        if (shape.Length == 0)
        {
            return true;
        }

        int expected = 1;
        for (int i = _rank - 1; i >= 0; i--)
        {
            int extent = shape.Unchecked(i);
            if (extent != 1 && _d[i] != expected)
            {
                return false;
            }

            expected *= extent;
        }

        return true;
    }

    /// <summary>
    /// Computes the lowest and highest element offset (relative to the start offset) a tensor of <paramref name="shape"/> with these
    /// strides reaches. Both are 0 for an empty tensor.
    /// </summary>
    /// <exception cref="TensorShapeException">The reach does not fit in an int.</exception>
    public void GetExtent(Shape shape, out long lowest, out long highest)
    {
        lowest = 0;
        highest = 0;
        if (shape.Length == 0)
        {
            return;
        }

        for (int i = 0; i < _rank; i++)
        {
            long reach = (long)(shape.Unchecked(i) - 1) * _d[i];
            if (reach < 0)
            {
                lowest += reach;
            }
            else
            {
                highest += reach;
            }
        }

        if (lowest < int.MinValue || highest > int.MaxValue)
        {
            throw new TensorShapeException("The strides reach beyond 2^31 elements.");
        }
    }

    /// <summary>Equality of all strides.</summary>
    public bool Equals(Strides other)
    {
        if (_rank != other._rank)
        {
            return false;
        }

        for (int i = 0; i < _rank; i++)
        {
            if (_d[i] != other._d[i])
            {
                return false;
            }
        }

        return true;
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is Strides s && Equals(s);

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        HashCode h = default;
        for (int i = 0; i < _rank; i++)
        {
            h.Add(_d[i]);
        }

        return h.ToHashCode();
    }

    /// <summary>Equality.</summary>
    public static bool operator ==(Strides a, Strides b) => a.Equals(b);

    /// <summary>Inequality.</summary>
    public static bool operator !=(Strides a, Strides b) => !a.Equals(b);

    /// <summary>Formats as <c>(12,4,1)</c>.</summary>
    public override string ToString()
    {
        var sb = new StringBuilder("(");
        for (int i = 0; i < _rank; i++)
        {
            if (i > 0)
            {
                sb.Append(',');
            }

            sb.Append(_d[i]);
        }

        return sb.Append(')').ToString();
    }
}
