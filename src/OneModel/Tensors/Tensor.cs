using System;
using System.Collections.Generic;
using System.Threading;
using System.Runtime.CompilerServices;

namespace RecursiveAI.OneModel.Tensors;

/// <summary>
/// A strided view over a <see cref="TensorBuffer"/>: element type, logical shape, strides in elements, and an offset. A value type,
/// so views, slices and broadcasts cost no allocation; all views of a buffer share its storage and its version counter. The logical
/// shape is what operations see; strides say where elements live, so transposes, slices, flips and broadcasts are free and
/// <see cref="Contiguous"/> materializes only when asked.
/// </summary>
/// <remarks>
/// <para>A default <see cref="Tensor"/> is empty (<see cref="IsEmpty"/>). Rank is at most <see cref="Shape.MaxRank"/>, the element
/// count at most 2^31-1 and the storage at most 2^31-1 bytes.</para>
/// <para>Ownership: a tensor made by <c>Create</c> owns its storage until <see cref="Dispose"/> (called on any copy or view; it
/// releases the storage for all of them), or until the arena it came from is reset. Using a tensor after that throws
/// <see cref="StaleTensorException"/>. A tensor that is never disposed leaks only until the GC finalizes its buffer.</para>
/// <para>Gradients: a tensor <see cref="RequiresGrad"/> when created with <see cref="WithRequiresGrad"/> or produced by an
/// operation on such a tensor while <see cref="GradMode.IsEnabled"/>; those operations also attach a <see cref="BackwardRecord"/>.</para>
/// </remarks>
public readonly partial struct Tensor : IDisposable
{
    private readonly TensorBuffer? _buffer;
    private readonly int _generation;
    private readonly int _offset;
    private readonly Shape _shape;
    private readonly Strides _strides;
    private readonly DType _dtype;
    private readonly int _block;
    private readonly AutogradMeta? _meta;

    private Tensor(TensorBuffer buffer, int generation, int offset, Shape shape, Strides strides, DType dtype, int block, AutogradMeta? meta)
    {
        _buffer = buffer;
        _generation = generation;
        _offset = offset;
        _shape = shape;
        _strides = strides;
        _dtype = dtype;
        _block = block;
        _meta = meta;
    }

    // ---- creation ---------------------------------------------------------------------------------------------------------

    /// <summary>Allocates a contiguous tensor from a pool.</summary>
    /// <param name="dtype">The element type.</param>
    /// <param name="shape">The shape.</param>
    /// <param name="pool">The pool; <see cref="BufferPool.Shared"/> when null.</param>
    /// <param name="zero">Zero the contents; otherwise they are unspecified.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="dtype"/> is not a stored type.</exception>
    /// <exception cref="TensorShapeException">The storage would exceed 2^31-1 bytes.</exception>
    public static Tensor Create(DType dtype, Shape shape, BufferPool? pool = null, bool zero = false)
    {
        long bytes = ByteCount(dtype, shape.Length);
        TensorBuffer b = (pool ?? BufferPool.Shared).Rent(bytes, zero);
        return new Tensor(b, b.Generation, 0, shape, shape.ContiguousStrides(), dtype, 0, null);
    }

    /// <summary>Allocates a contiguous tensor from an arena; it becomes stale at the arena's next reset.</summary>
    /// <inheritdoc cref="Create(DType, Shape, BufferPool?, bool)"/>
    public static Tensor Create(DType dtype, Shape shape, Arena arena, bool zero = false)
    {
        ArgumentNullException.ThrowIfNull(arena);
        long bytes = ByteCount(dtype, shape.Length);
        TensorBuffer b = arena.Allocate(bytes, zero);
        return new Tensor(b, b.Generation, 0, shape, shape.ContiguousStrides(), dtype, 0, null);
    }

    /// <summary>Allocates a zero-filled contiguous tensor.</summary>
    public static Tensor Zeros(DType dtype, Shape shape, BufferPool? pool = null) => Create(dtype, shape, pool, zero: true);

    /// <summary>Allocates a tensor holding a copy of <paramref name="data"/> (row-major).</summary>
    /// <exception cref="ArgumentException"><paramref name="data"/> does not have exactly <paramref name="shape"/>'s element count.</exception>
    public static Tensor FromSpan<T>(ReadOnlySpan<T> data, Shape shape, BufferPool? pool = null) where T : unmanaged
    {
        if (data.Length != shape.Length)
        {
            throw new ArgumentException($"{data.Length} values do not fill shape {shape} ({shape.Length} elements).", nameof(data));
        }

        Tensor t = Create(DTypes.Of<T>(), shape, pool);
        data.CopyTo(t.MutableData<T>());
        return t;
    }

    /// <summary>Allocates a scalar (rank 0) tensor.</summary>
    public static Tensor Scalar<T>(T value, BufferPool? pool = null) where T : unmanaged => FromSpan([value], default, pool);

    private static long ByteCount(DType dtype, int count)
    {
        DTypes.ThrowIfNotStorable(dtype);
        long bytes = (long)count * DTypes.SizeOf(dtype);
        if (bytes > int.MaxValue)
        {
            throw new TensorShapeException($"{count} elements of {dtype} need {bytes} bytes; the limit is 2^31-1.");
        }

        return bytes;
    }

    // Builds a view and checks that every reachable element lies inside the buffer.
    private Tensor Make(Shape shape, Strides strides, long offset, int block = 0)
    {
        if (offset < int.MinValue || offset > int.MaxValue)
        {
            throw new TensorShapeException("The view's offset overflows.");
        }

        if (block == 0 && shape.Length > 0)
        {
            strides.GetExtent(shape, out long lo, out long hi);
            long elems = _buffer!.Length / DTypes.SizeOf(_dtype);
            if (offset + lo < 0 || offset + hi >= elems)
            {
                throw new TensorShapeException($"A view of shape {shape}, strides {strides}, offset {offset} reaches outside its {elems}-element storage.");
            }
        }

        return new Tensor(_buffer!, _generation, (int)offset, shape, strides, _dtype, block, null);
    }

    /// <summary>
    /// A view with explicit strides and offset over this tensor's storage (offset counted in elements from the storage start). Not
    /// differentiable; for kernels and importers.
    /// </summary>
    /// <exception cref="TensorShapeException">The view would reach outside the storage.</exception>
    public Tensor AsStrided(Shape shape, Strides strides, int offset)
    {
        ThrowIfEmptyOrBlocked();
        if (shape.Rank != strides.Rank)
        {
            throw new TensorShapeException("Shape and strides have different ranks.");
        }

        return Make(shape, strides, offset);
    }

    // ---- properties -------------------------------------------------------------------------------------------------------

    /// <summary>True for a default (unallocated) tensor.</summary>
    public bool IsEmpty => _buffer is null;

    /// <summary>The element type.</summary>
    public DType DType => _dtype;

    /// <summary>The logical shape. For a blocked layout this is the logical (N, C, H, W), not the physical one.</summary>
    public Shape Shape => _shape;

    /// <summary>The strides in elements.</summary>
    /// <exception cref="InvalidOperationException">The tensor uses a blocked layout, which strides cannot describe.</exception>
    public Strides Strides => _block > 0 ? throw new InvalidOperationException("A blocked-layout tensor has no strides.") : _strides;

    /// <summary>Number of dimensions.</summary>
    public int Rank => _shape.Rank;

    /// <summary>Number of logical elements.</summary>
    public int Length => _shape.Length;

    /// <summary>Size of one element in bytes.</summary>
    public int ElementSize => DTypes.SizeOf(_dtype);

    /// <summary>Offset of the first element from the start of the storage, in elements.</summary>
    public int Offset => _offset;

    /// <summary>The storage's version counter (see <see cref="TensorBuffer.Version"/>); 0 for an empty tensor.</summary>
    public long Version => _buffer?.Version ?? 0;

    /// <summary>The storage, for alignment checks and sharing tests. Null for an empty tensor.</summary>
    public TensorBuffer? Buffer => _buffer;

    /// <summary>True for the blocked <see cref="LayoutKind.NchwC"/> layout.</summary>
    public bool IsBlocked => _block > 0;

    /// <summary>True when elements are in row-major order without gaps (size-1 dimensions may have any stride). A blocked tensor is contiguous in its physical order.</summary>
    public bool IsContiguous => _block > 0 || _strides.IsContiguousFor(_shape);

    /// <summary>
    /// The named layout of this tensor. Rank-4 tensors are classified from their strides: contiguous is NCHW, the channels-last
    /// pattern is NHWC (when both hold, because C or H·W is 1, NCHW wins); everything else is Strided.
    /// </summary>
    public TensorLayout Layout
    {
        get
        {
            if (_block > 0)
            {
                return TensorLayout.NchwC(_block);
            }

            if (_shape.Rank != 4)
            {
                return TensorLayout.Strided;
            }

            if (IsContiguous)
            {
                return TensorLayout.Nchw;
            }

            int n = _shape.Unchecked(0), c = _shape.Unchecked(1), h = _shape.Unchecked(2), w = _shape.Unchecked(3);
            long sC = 1, sW = c, sH = (long)w * c, sN = (long)h * w * c;
            Strides st = _strides;
            return Matches(st, 0, sN, n) && Matches(st, 1, sC, c) && Matches(st, 2, sH, h) && Matches(st, 3, sW, w)
                ? TensorLayout.Nhwc
                : TensorLayout.Strided;

            static bool Matches(Strides s, int axis, long expected, int extent) => extent == 1 || s.Unchecked(axis) == expected;
        }
    }

    /// <summary>Number of elements in the physical storage of a blocked tensor ([N][ceil(C/B)][H][W][B]); equals <see cref="Length"/> otherwise.</summary>
    public int PhysicalLength
    {
        get
        {
            if (_block == 0)
            {
                return Length;
            }

            int cb = (_shape.Unchecked(1) + _block - 1) / _block;
            return checked(_shape.Unchecked(0) * cb * _shape.Unchecked(2) * _shape.Unchecked(3) * _block);
        }
    }

    // ---- data access ------------------------------------------------------------------------------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void CheckType<T>() where T : unmanaged
    {
        if (_buffer is null)
        {
            throw new InvalidOperationException("The tensor is empty.");
        }

        if (DTypes.Of<T>() != _dtype)
        {
            throw new ArgumentException($"The tensor holds {_dtype}, not {typeof(T).Name}.");
        }
    }

    /// <summary>The elements as a read-only span, in row-major order. For a blocked tensor, the physical storage.</summary>
    /// <exception cref="ArgumentException"><typeparamref name="T"/> does not match <see cref="DType"/>.</exception>
    /// <exception cref="InvalidOperationException">The tensor is empty or not contiguous.</exception>
    /// <exception cref="StaleTensorException">The storage was released.</exception>
    public ReadOnlySpan<T> Data<T>() where T : unmanaged
    {
        CheckType<T>();
        if (!IsContiguous)
        {
            throw new InvalidOperationException("The tensor is not contiguous; call Contiguous() first or use Storage<T>() with the strides.");
        }

        ReadOnlySpan<T> all = _buffer!.GetSpan<T>(_generation);
        int n = PhysicalLength;
        return n == 0 ? default : all.Slice(_offset, n);
    }

    /// <summary>
    /// The elements as a writable span. Counts as a modification: it increases the storage's <see cref="Version"/> so a tensor saved
    /// for backward notices. Writes through a span obtained earlier are not tracked.
    /// </summary>
    /// <inheritdoc cref="Data{T}"/>
    public Span<T> MutableData<T>() where T : unmanaged
    {
        CheckType<T>();
        if (!IsContiguous)
        {
            throw new InvalidOperationException("The tensor is not contiguous; use MutableStorage<T>() with the strides.");
        }

        Span<T> all = _buffer!.GetSpan<T>(_generation);
        _buffer.Touch();
        int n = PhysicalLength;
        return n == 0 ? default : all.Slice(_offset, n);
    }

    /// <summary>The whole storage as a read-only span; with <see cref="Offset"/> and <see cref="Strides"/> this is what a strided kernel indexes.</summary>
    /// <inheritdoc cref="Data{T}"/>
    public ReadOnlySpan<T> Storage<T>() where T : unmanaged
    {
        CheckType<T>();
        return _buffer!.GetSpan<T>(_generation);
    }

    /// <summary>The whole storage as a writable span; bumps the version like <see cref="MutableData{T}"/>.</summary>
    /// <inheritdoc cref="Data{T}"/>
    public Span<T> MutableStorage<T>() where T : unmanaged
    {
        CheckType<T>();
        Span<T> s = _buffer!.GetSpan<T>(_generation);
        _buffer.Touch();
        return s;
    }

    /// <summary>Reads the element at a multi-index (one index per dimension).</summary>
    /// <exception cref="ArgumentException">Wrong index count, or an index outside its dimension.</exception>
    public T Get<T>(params ReadOnlySpan<int> index) where T : unmanaged
    {
        CheckType<T>();
        ThrowIfBlocked();
        return _buffer!.GetSpan<T>(_generation)[ElementOffset(index)];
    }

    /// <summary>Writes the element at a multi-index; bumps the version.</summary>
    /// <inheritdoc cref="Get{T}"/>
    public void Set<T>(T value, params ReadOnlySpan<int> index) where T : unmanaged
    {
        CheckType<T>();
        ThrowIfBlocked();
        _buffer!.GetSpan<T>(_generation)[ElementOffset(index)] = value;
        _buffer.Touch();
    }

    private int ElementOffset(ReadOnlySpan<int> index)
    {
        if (index.Length != _shape.Rank)
        {
            throw new ArgumentException($"Expected {_shape.Rank} indices, got {index.Length}.", nameof(index));
        }

        long off = _offset;
        for (int i = 0; i < index.Length; i++)
        {
            if ((uint)index[i] >= (uint)_shape.Unchecked(i))
            {
                throw new ArgumentException($"Index {index[i]} is outside dimension {i} of {_shape}.", nameof(index));
            }

            off += (long)index[i] * _strides.Unchecked(i);
        }

        return (int)off;
    }

    // ---- lifetime ---------------------------------------------------------------------------------------------------------

    /// <summary>
    /// Releases the storage to its pool for this tensor and every view or copy of the handle. Does nothing for arena storage (the
    /// arena reclaims it at <see cref="Arena.Reset"/>) or an empty tensor.
    /// </summary>
    /// <exception cref="StaleTensorException">The storage was already released.</exception>
    public void Dispose() => _buffer?.Release(_generation);

    // ---- autograd ---------------------------------------------------------------------------------------------------------

    /// <summary>True if gradients flow to this tensor.</summary>
    public bool RequiresGrad => _meta is { RequiresGrad: true };

    /// <summary>The autograd node, or null if the tensor does not take part in differentiation.</summary>
    public AutogradMeta? Autograd => _meta;

    /// <summary>The accumulated gradient (empty if none).</summary>
    public Tensor Grad => _meta?.Grad ?? default;

    /// <summary>
    /// Returns a handle to the same data that is a gradient leaf (<paramref name="requiresGrad"/> true) or is not tracked (false).
    /// Because a tensor is a value, the flag lives on the returned handle; keep and use that handle.
    /// </summary>
    /// <exception cref="InvalidOperationException">Gradients were requested for a non-floating-point tensor.</exception>
    public Tensor WithRequiresGrad(bool requiresGrad = true)
    {
        if (requiresGrad && !DTypes.IsFloatingPoint(_dtype))
        {
            throw new InvalidOperationException($"Only floating-point tensors can require gradients, not {_dtype}.");
        }

        ThrowIfEmpty();
        return WithAutograd(requiresGrad ? new AutogradMeta(true, null, 0) : null);
    }

    /// <summary>A handle to the same data with no autograd state.</summary>
    public Tensor Detach() => WithAutograd(null);

    internal Tensor WithAutograd(AutogradMeta? meta) =>
        new(_buffer!, _generation, _offset, _shape, _strides, _dtype, _block, meta);

    // ---- guards -----------------------------------------------------------------------------------------------------------

    private void ThrowIfEmpty()
    {
        if (_buffer is null)
        {
            throw new InvalidOperationException("The tensor is empty.");
        }
    }

    private void ThrowIfBlocked()
    {
        if (_block > 0)
        {
            throw new NotSupportedException("A blocked-layout tensor supports only ToLayout and its physical span; convert it to NCHW first.");
        }
    }

    private void ThrowIfEmptyOrBlocked()
    {
        ThrowIfEmpty();
        ThrowIfBlocked();
    }

    /// <inheritdoc/>
    public override string ToString() => _buffer is null ? "Tensor(empty)" : $"Tensor({_dtype}{_shape}, {Layout})";
}
