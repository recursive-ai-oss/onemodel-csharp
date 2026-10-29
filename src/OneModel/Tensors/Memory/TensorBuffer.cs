using System;
using System.Collections.Generic;
using System.Threading;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace RecursiveAI.OneModel.Tensors;

/// <summary>A tensor (or a view of it) was used after its storage went back to the pool or its arena was reset, or released twice.</summary>
public sealed class StaleTensorException : InvalidOperationException
{
    /// <summary>Creates the exception.</summary>
    public StaleTensorException(string message) : base(message) { }
}

/// <summary>
/// A 64-byte-aligned block of native memory that tensors view. Obtain one from <see cref="BufferPool.Rent"/> or
/// <see cref="Arena.Allocate"/>, normally through <c>Tensor.Create</c>. All tensors viewing the same buffer share its
/// <see cref="Version"/> counter. A buffer is identified by a generation number: tensors remember the generation they were created
/// in, and every access compares it, so a tensor that outlives its storage fails with <see cref="StaleTensorException"/> instead of
/// reading memory that now belongs to someone else.
/// </summary>
/// <remarks>
/// Pointer use is confined to this file, <c>BufferPool.cs</c> and <c>Arena.cs</c>. Bounds: a span handed out by
/// <see cref="GetSpan{T}"/> covers exactly [0, <see cref="Length"/>) bytes of an allocation that is at least that long (the pool
/// rounds capacity up, never down; the arena reserves the aligned length). Under the <c>CHECKED</c> build symbol, 64 canary bytes
/// follow every buffer's length and are verified at release, and released memory is overwritten with 0xFF (a NaN for floats).
/// </remarks>
public sealed unsafe class TensorBuffer
{
    internal const int Alignment = 64;
#if CHECKED
    internal const int Guard = 64;
#else
    internal const int Guard = 0;
#endif
    private const byte PoisonByte = 0xFF;
    private const byte CanaryByte = 0xA5;

    private byte* _ptr;
    private long _capacity;
    private long _length;
    private int _generation;
    private long _version;
    private int _live;
    private readonly BufferPool? _pool;
    private readonly Arena? _arena;

    // Native block owned by a pool; the pool recycles it.
    internal TensorBuffer(BufferPool pool, long capacity)
    {
        _pool = pool;
        _capacity = capacity;
        _ptr = (byte*)NativeMemory.AlignedAlloc((nuint)(capacity + Guard), Alignment);
#if CHECKED
        Unsafe.InitBlockUnaligned(_ptr, PoisonByte, (uint)(capacity + Guard));
#endif
    }

    // A slice of an arena chunk; owns no memory.
    internal TensorBuffer(Arena arena)
    {
        _arena = arena;
        GC.SuppressFinalize(this);
    }

    /// <summary>Frees a native block that was rented and never returned (a leak), so leaking costs memory only until the GC runs.</summary>
    ~TensorBuffer() => FreeNative();

    /// <summary>Usable size in bytes.</summary>
    public long Length => _length;

    /// <summary>
    /// The version counter shared by every tensor viewing this buffer. It starts at 0 and increases by one each time an in-place
    /// operation (or a call to a tensor's mutable accessors) may have changed the contents.
    /// </summary>
    public long Version => Volatile.Read(ref _version);

    /// <summary>True from rent (or allocate) until release.</summary>
    public bool IsLive => Volatile.Read(ref _live) == 1;

    /// <summary>The alignment of the first byte (64).</summary>
    public static int AlignmentBytes => Alignment;

    internal int Generation => Volatile.Read(ref _generation);

    internal long Capacity => _capacity;

    internal byte* Pointer => _ptr;

    internal bool IsArena => _arena is not null;

    internal void Touch() => Interlocked.Increment(ref _version);

    /// <summary>Address of the first byte, for alignment checks in tests and kernels.</summary>
    public nuint Address => (nuint)_ptr;

    /// <summary>The whole buffer as <typeparamref name="T"/> elements (any trailing partial element is excluded).</summary>
    /// <param name="generation">The generation the caller's tensor was created in.</param>
    /// <exception cref="StaleTensorException">The generation no longer matches: the storage was released.</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal Span<T> GetSpan<T>(int generation) where T : unmanaged
    {
        if (generation != Volatile.Read(ref _generation))
        {
            ThrowStale();
        }

        return new Span<T>(_ptr, (int)(_length / sizeof(T)));
    }

    // Called by the owner when the buffer is handed out. length <= capacity (pool) or fits the arena slice.
    internal void Prepare(long length, bool zero)
    {
        _length = length;
        Volatile.Write(ref _version, 0);
        if (zero)
        {
            new Span<byte>(_ptr, (int)length).Clear();
        }
#if CHECKED
        else
        {
            new Span<byte>(_ptr, (int)length).Fill(PoisonByte);
        }

        new Span<byte>(_ptr + length, Guard).Fill(CanaryByte);
#endif
        Volatile.Write(ref _live, 1);
    }

#if CHECKED
    // The execution plan reuses ranges of one buffer for many values; a range whose value is dead is overwritten so a stale read
    // shows up as garbage. The caller passes a range inside this buffer.
    internal void PoisonRange(long offset, long size)
    {
        if (offset < 0 || size < 0 || offset + size > _length)
        {
            throw new ArgumentOutOfRangeException(nameof(offset), "The range lies outside the buffer.");
        }

        Unsafe.InitBlockUnaligned(_ptr + offset, PoisonByte, (uint)size);
    }
#endif

    internal void Rebind(byte* ptr, long length)
    {
        _ptr = ptr;
        _capacity = length;
    }

    /// <summary>
    /// Ends this generation. Returns true when the caller must now recycle the block; throws <see cref="StaleTensorException"/> if
    /// the generation was already ended (a double release).
    /// </summary>
    internal bool EndGeneration(int generation, out bool overrun)
    {
        overrun = false;
        if (Interlocked.CompareExchange(ref _generation, unchecked(generation + 1), generation) != generation)
        {
            throw new StaleTensorException("The tensor's storage was already released (double release or a stale handle).");
        }

        Volatile.Write(ref _live, 0);
#if CHECKED
        ReadOnlySpan<byte> canary = new(_ptr + _length, Guard);
        foreach (byte b in canary)
        {
            if (b != CanaryByte)
            {
                overrun = true;
                break;
            }
        }

        Unsafe.InitBlockUnaligned(_ptr, PoisonByte, (uint)(_length + Guard));
#endif
        return true;
    }

    /// <summary>Releases the storage this handle's tensors view back to its pool, or does nothing for arena storage.</summary>
    /// <exception cref="StaleTensorException">Already released.</exception>
    internal void Release(int generation)
    {
        bool overrun;
        if (_arena is not null)
        {
            // Arena memory is reclaimed in bulk by Arena.Reset; releasing one tensor only validates the handle.
            if (generation != Volatile.Read(ref _generation))
            {
                throw new StaleTensorException("The arena allocation was already reset.");
            }

            return;
        }

        EndGeneration(generation, out overrun);
        _pool!.Return(this);
        if (overrun)
        {
            throw new InvalidOperationException("A kernel wrote past the end of a buffer (canary bytes were overwritten).");
        }
    }

    internal void FreeNative()
    {
        byte* p = _ptr;
        _ptr = null;
        if (p != null && _arena is null)
        {
            NativeMemory.AlignedFree(p);
        }
    }

    private static void ThrowStale() =>
        throw new StaleTensorException("The tensor's storage was returned to its pool (or its arena was reset) after the tensor was created.");
}
