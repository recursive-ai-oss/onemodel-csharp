using System;
using System.Collections.Generic;
using System.Threading;
namespace RecursiveAI.OneModel.Tensors;

// Views: operations that change only shape, strides and offset over the same storage. Each records a BackwardRecord when gradients
// are being tracked; the record holds the input shape and the integer parameters, which is all the inverse view needs.
public readonly partial struct Tensor
{
    private Tensor Derive(OpKind op, Shape shape, Strides strides, long offset, params ReadOnlySpan<long> ints)
    {
        Tensor view = Make(shape, strides, offset);
        if (GradMode.ShouldRecord(this))
        {
            BackwardRecord r = BackwardRecord.Create(op, 0, this);
            r.SetInts(ints);
            return r.Attach(view);
        }

        return view;
    }

    /// <summary>Reorders dimensions: output dimension i is input dimension <c>perm[i]</c>. No data moves.</summary>
    /// <param name="perm">A permutation of 0..rank-1; negative entries count from the end.</param>
    /// <exception cref="ArgumentException">Not a permutation of the dimensions.</exception>
    public Tensor Permute(params ReadOnlySpan<int> perm)
    {
        ThrowIfEmptyOrBlocked();
        int rank = Rank;
        if (perm.Length != rank)
        {
            throw new TensorShapeException($"A permutation of {perm.Length} entries cannot reorder rank {rank}.", nameof(perm));
        }

        Span<int> dims = stackalloc int[Shape.MaxRank];
        Span<int> str = stackalloc int[Shape.MaxRank];
        Span<long> norm = stackalloc long[Shape.MaxRank];
        uint seen = 0;
        for (int i = 0; i < rank; i++)
        {
            int a = Shape.NormalizeAxis(perm[i], rank);
            if ((seen & (1u << a)) != 0)
            {
                throw new TensorShapeException("The permutation repeats an axis.", nameof(perm));
            }

            seen |= 1u << a;
            dims[i] = _shape.Unchecked(a);
            str[i] = _strides.Unchecked(a);
            norm[i] = a;
        }

        return Derive(OpKind.Permute, Shape.Create(dims.Slice(0, rank)), Strides.Create(str.Slice(0, rank)), _offset, norm.Slice(0, rank));
    }

    /// <summary>Swaps two dimensions (negative axes count from the end). No data moves.</summary>
    public Tensor Transpose(int axis0, int axis1)
    {
        ThrowIfEmptyOrBlocked();
        int a = Shape.NormalizeAxis(axis0, Rank), b = Shape.NormalizeAxis(axis1, Rank);
        Span<int> p = stackalloc int[Shape.MaxRank];
        for (int i = 0; i < Rank; i++)
        {
            p[i] = i;
        }

        (p[a], p[b]) = (p[b], p[a]);
        return Permute(p.Slice(0, Rank));
    }

    /// <summary>The matrix transpose of a rank-2 tensor (swaps its two dimensions).</summary>
    public Tensor T() => Rank == 2 ? Transpose(0, 1) : throw new TensorShapeException("T() needs a rank-2 tensor.");

    /// <summary>
    /// Elements <c>start, start+step, ...</c> below <paramref name="end"/> along one axis. No data moves.
    /// </summary>
    /// <param name="axis">Axis, negative counts from the end.</param>
    /// <param name="start">First index, 0 to the axis size.</param>
    /// <param name="end">One past the last index, start to the axis size.</param>
    /// <param name="step">Step, at least 1. Use <see cref="Flip"/> for reversal.</param>
    /// <exception cref="ArgumentException">Out-of-range bounds or a step below 1.</exception>
    public Tensor Slice(int axis, int start, int end, int step = 1)
    {
        ThrowIfEmptyOrBlocked();
        int a = Shape.NormalizeAxis(axis, Rank);
        int size = _shape.Unchecked(a);
        if (step < 1 || start < 0 || end < start || end > size)
        {
            throw new TensorShapeException($"Slice [{start}:{end}:{step}] is invalid for an axis of size {size}.");
        }

        int count = (end - start + step - 1) / step;
        long stride = (long)_strides.Unchecked(a) * step;
        if (stride > int.MaxValue)
        {
            throw new TensorShapeException("The slice stride overflows.");
        }

        long offset = _offset + (count == 0 ? 0 : (long)start * _strides.Unchecked(a));
        return Derive(OpKind.Slice, _shape.With(a, count), _strides.With(a, (int)stride), offset, a, start, end, step);
    }

    /// <summary>Picks one index along an axis and removes the axis. No data moves.</summary>
    public Tensor Select(int axis, int index)
    {
        ThrowIfEmptyOrBlocked();
        int a = Shape.NormalizeAxis(axis, Rank);
        if ((uint)index >= (uint)_shape.Unchecked(a))
        {
            throw new TensorShapeException($"Index {index} is outside axis {a} of {_shape}.");
        }

        return Derive(OpKind.Select, _shape.Remove(a), _strides.Remove(a), _offset + ((long)index * _strides.Unchecked(a)), a, index);
    }

    /// <summary>Reverses one axis using a negative stride. No data moves.</summary>
    public Tensor Flip(int axis)
    {
        ThrowIfEmptyOrBlocked();
        int a = Shape.NormalizeAxis(axis, Rank);
        int size = _shape.Unchecked(a);
        long offset = _offset + (size == 0 ? 0 : (long)(size - 1) * _strides.Unchecked(a));
        return Derive(OpKind.Flip, _shape, _strides.With(a, -_strides.Unchecked(a)), offset, a);
    }

    /// <summary>Inserts a dimension of extent 1 at <paramref name="axis"/> (0 to rank; negative counts from rank+1).</summary>
    public Tensor Unsqueeze(int axis)
    {
        ThrowIfEmptyOrBlocked();
        if (Rank == Shape.MaxRank)
        {
            throw new TensorShapeException($"Rank {Shape.MaxRank + 1} exceeds the maximum.");
        }

        int a = Shape.NormalizeAxis(axis, Rank + 1);
        int stride = a < Rank ? _shape.Unchecked(a) * _strides.Unchecked(a) : 1;
        return Derive(OpKind.Reshape, _shape.Insert(a, 1), _strides.Insert(a, stride), _offset);
    }

    /// <summary>Removes one dimension, which must have extent 1.</summary>
    /// <exception cref="TensorShapeException">The extent is not 1.</exception>
    public Tensor Squeeze(int axis)
    {
        ThrowIfEmptyOrBlocked();
        int a = Shape.NormalizeAxis(axis, Rank);
        if (_shape.Unchecked(a) != 1)
        {
            throw new TensorShapeException($"Cannot squeeze axis {a} of {_shape}: its extent is not 1.");
        }

        return Derive(OpKind.Reshape, _shape.Remove(a), _strides.Remove(a), _offset);
    }

    /// <summary>Removes every dimension of extent 1.</summary>
    public Tensor Squeeze()
    {
        ThrowIfEmptyOrBlocked();
        Span<int> d = stackalloc int[Shape.MaxRank];
        Span<int> s = stackalloc int[Shape.MaxRank];
        int n = 0;
        for (int i = 0; i < Rank; i++)
        {
            if (_shape.Unchecked(i) != 1)
            {
                d[n] = _shape.Unchecked(i);
                s[n++] = _strides.Unchecked(i);
            }
        }

        return Derive(OpKind.Reshape, Shape.Create(d.Slice(0, n)), Strides.Create(s.Slice(0, n)), _offset);
    }

    /// <summary>
    /// Broadcasts to <paramref name="target"/>: shapes are right-aligned, an extent of 1 (or a missing leading dimension) becomes the
    /// target's extent with stride 0. No data moves; writing through the result is the caller's concern (elements alias).
    /// </summary>
    /// <exception cref="TensorShapeException">An extent is neither 1 nor equal to the target's.</exception>
    public Tensor Expand(Shape target)
    {
        ThrowIfEmptyOrBlocked();
        int lead = target.Rank - Rank;
        if (lead < 0)
        {
            throw new TensorShapeException($"Cannot expand {_shape} to the lower rank {target}.");
        }

        Span<int> s = stackalloc int[Shape.MaxRank];
        for (int i = 0; i < target.Rank; i++)
        {
            int src = i - lead;
            if (src < 0)
            {
                s[i] = 0;
            }
            else if (_shape.Unchecked(src) == target.Unchecked(i))
            {
                s[i] = _strides.Unchecked(src);
            }
            else if (_shape.Unchecked(src) == 1)
            {
                s[i] = 0;
            }
            else
            {
                throw new TensorShapeException($"Cannot expand {_shape} to {target}.");
            }
        }

        return Derive(OpKind.Expand, target, Strides.Create(s.Slice(0, target.Rank)), _offset);
    }

    /// <summary>
    /// Reinterprets the shape without copying when the strides allow it (always for a contiguous tensor).
    /// </summary>
    /// <param name="shape">The new shape; same element count.</param>
    /// <param name="view">The view, when this returns true.</param>
    public bool TryReshape(Shape shape, out Tensor view)
    {
        ThrowIfEmptyOrBlocked();
        if (shape.Length != Length)
        {
            throw new TensorShapeException($"Cannot reshape {_shape} ({Length} elements) to {shape} ({shape.Length} elements).");
        }

        if (!TryViewStrides(shape, out Strides st))
        {
            view = default;
            return false;
        }

        view = Derive(OpKind.Reshape, shape, st, _offset);
        return true;
    }

    /// <summary>A reshaped view; throws if the strides do not allow one (see <see cref="Reshape(Shape)"/> for the copying form).</summary>
    /// <exception cref="InvalidOperationException">The reshape needs a copy.</exception>
    public Tensor View(Shape shape) =>
        TryReshape(shape, out Tensor v) ? v : throw new InvalidOperationException($"Cannot view {_shape} with strides {_strides} as {shape} without a copy.");

    /// <summary>
    /// Reshapes to <paramref name="shape"/>: a view when the strides allow it, otherwise a contiguous copy (new storage, owned by the
    /// caller) reshaped. Row-major element order is preserved either way.
    /// </summary>
    public Tensor Reshape(Shape shape)
    {
        if (TryReshape(shape, out Tensor v))
        {
            return v;
        }

        // Backward of "copy then view" is a plain reshape of the gradient to the input shape, which is what a Reshape record says.
        bool track = GradMode.ShouldRecord(this);
        Tensor copy;
        using (GradMode.NoGrad())
        {
            copy = Contiguous();
        }

        Tensor view = copy.Make(shape, shape.ContiguousStrides(), 0);
        return track ? Rerecord(view) : view;
    }

    /// <summary>
    /// Reshapes with one extent allowed to be -1, inferred from the element count.
    /// </summary>
    /// <exception cref="TensorShapeException">More than one -1, a non-divisible count, or a negative extent other than -1.</exception>
    public Tensor Reshape(params ReadOnlySpan<int> dims)
    {
        ThrowIfEmptyOrBlocked();
        Span<int> d = stackalloc int[Shape.MaxRank];
        if (dims.Length > Shape.MaxRank)
        {
            throw new TensorShapeException($"Rank {dims.Length} exceeds the maximum.");
        }

        int unknown = -1;
        long known = 1;
        for (int i = 0; i < dims.Length; i++)
        {
            d[i] = dims[i];
            if (dims[i] == -1)
            {
                if (unknown >= 0)
                {
                    throw new TensorShapeException("Only one extent may be -1.");
                }

                unknown = i;
            }
            else if (dims[i] < 0)
            {
                throw new TensorShapeException($"Extent {dims[i]} is negative.");
            }
            else
            {
                known *= dims[i];
            }
        }

        if (unknown >= 0)
        {
            if (known == 0 || Length % known != 0)
            {
                throw new TensorShapeException($"Cannot infer an extent: {Length} elements into {known}-element rows.");
            }

            d[unknown] = (int)(Length / known);
        }

        return Reshape(Shape.Create(d.Slice(0, dims.Length)));
    }

    // Attaches a Reshape record (input = this) to a tensor that was made without one.
    private Tensor Rerecord(in Tensor result)
    {
        BackwardRecord r = BackwardRecord.Create(OpKind.Reshape, 0, this);
        return r.Attach(result);
    }

    // Strides for viewing this tensor's elements, in row-major order, as `shape`; false if no stride assignment exists.
    // Method: drop extent-1 dimensions (their stride never matters); merge neighbouring old dimensions whose strides chain
    // (stride[i] == stride[i+1] * size[i+1]) into runs that behave like one long dimension; then each run must be re-split into
    // a group of new dimensions whose extents multiply to the run's length, with strides laid out contiguously from the run's
    // innermost stride.
    private bool TryViewStrides(Shape shape, out Strides result)
    {
        Span<int> st = stackalloc int[Shape.MaxRank];
        int newRank = shape.Rank;
        if (Length == 0)
        {
            result = shape.ContiguousStrides();
            return true;
        }

        Span<int> runLen = stackalloc int[Shape.MaxRank];
        Span<int> runStride = stackalloc int[Shape.MaxRank];
        int runs = 0;
        for (int i = 0; i < Rank; i++)
        {
            int size = _shape.Unchecked(i);
            if (size == 1)
            {
                continue;
            }

            int stride = _strides.Unchecked(i);
            if (runs > 0 && (long)runStride[runs - 1] == (long)stride * size)
            {
                runLen[runs - 1] *= size;
                runStride[runs - 1] = stride;
            }
            else
            {
                runLen[runs] = size;
                runStride[runs] = stride;
                runs++;
            }
        }

        int j = 0; // next new dimension to assign
        for (int r = 0; r < runs; r++)
        {
            int first = j;
            long acc = 1;
            while (acc < runLen[r])
            {
                if (j >= newRank)
                {
                    result = default;
                    return false;
                }

                acc *= shape.Unchecked(j++);
            }

            if (acc != runLen[r])
            {
                result = default;
                return false;
            }

            long cur = runStride[r];
            for (int k = j - 1; k >= first; k--)
            {
                st[k] = (int)cur;
                cur *= shape.Unchecked(k);
            }
        }

        // Remaining new dimensions all have extent 1 (the counts matched); their stride is irrelevant.
        for (int k = j; k < newRank; k++)
        {
            if (shape.Unchecked(k) != 1)
            {
                result = default;
                return false;
            }

            st[k] = 1;
        }

        // New extent-1 dimensions swallowed at the start of a run (extent 1 multiplies acc by 1) were assigned a stride above.
        result = Strides.Create(st.Slice(0, newRank));
        return true;
    }
}
