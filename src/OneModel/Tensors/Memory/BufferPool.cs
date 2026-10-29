using System;
using System.Collections.Generic;
using System.Threading;
using System.Numerics;

namespace RecursiveAI.OneModel.Tensors;

/// <summary>Counters of a <see cref="BufferPool"/>.</summary>
/// <param name="Rents">Total calls to <see cref="BufferPool.Rent"/>.</param>
/// <param name="NativeAllocations">Rents that had to allocate native memory because no idle block of the size class existed. Constant in a steady-state loop.</param>
/// <param name="IdleBytes">Bytes held by idle blocks, ready for reuse.</param>
/// <param name="OutstandingBytes">Bytes of capacity currently rented.</param>
public readonly record struct PoolStats(long Rents, long NativeAllocations, long IdleBytes, long OutstandingBytes);

/// <summary>
/// A thread-safe pool of 64-byte-aligned native blocks in size classes. A request is rounded up to its class, and a released
/// block waits in its class's free list, so a loop that rents and releases the same sizes allocates nothing after its first
/// iteration. Class sizes are a function of the requested size alone: four classes per doubling above 256 bytes
/// (base·{1, 1.25, 1.5, 1.75}, base a power of two), 64-byte steps below, so no class wastes more than 25% and nothing depends on thread
/// count or timing.
/// </summary>
/// <remarks>
/// Rented memory is uninitialized unless <c>zero</c> is requested (under <c>CHECKED</c> it is filled with 0xFF, a NaN for floats, so
/// a read of memory nobody wrote is visible). A single request is limited to <see cref="int.MaxValue"/> bytes.
/// </remarks>
public sealed class BufferPool : IDisposable
{
    private readonly object _gate = new();
    private readonly Dictionary<long, Stack<TensorBuffer>> _free = [];
    private long _rents, _nativeAllocs, _idleBytes, _outstandingBytes;
    private bool _disposed;

    /// <summary>The process-wide pool used when a caller passes none.</summary>
    public static BufferPool Shared { get; } = new();

    /// <summary>
    /// Idle bytes the pool keeps; a released block that would exceed it is freed instead. Default 2 GiB. Affects speed and memory
    /// only, never results.
    /// </summary>
    public long MaxIdleBytes { get; set; } = 2L << 30;

    /// <summary>The capacity the pool allocates for a request of <paramref name="bytes"/> bytes (at least 64).</summary>
    /// <exception cref="TensorShapeException"><paramref name="bytes"/> is negative or above <see cref="int.MaxValue"/>.</exception>
    public static long ClassSize(long bytes)
    {
        if ((ulong)bytes > int.MaxValue)
        {
            throw new TensorShapeException($"A buffer of {bytes} bytes is outside [0, 2^31-1].", nameof(bytes));
        }

        if (bytes <= 256)
        {
            return Math.Max(64, (bytes + 63) & ~63L);
        }

        int msb = BitOperations.Log2((ulong)(bytes - 1));
        long b = 1L << msb;
        long step = b >> 2;
        return b + ((bytes - b + step - 1) / step * step);
    }

    /// <summary>Rents a block of at least <paramref name="bytes"/> bytes; <see cref="TensorBuffer.Length"/> is exactly <paramref name="bytes"/>.</summary>
    /// <param name="bytes">Size in bytes, 0 to 2^31-1.</param>
    /// <param name="zero">Clear the contents; otherwise they are unspecified.</param>
    /// <exception cref="TensorShapeException"><paramref name="bytes"/> is out of range.</exception>
    /// <exception cref="ObjectDisposedException">The pool was disposed.</exception>
    public TensorBuffer Rent(long bytes, bool zero = false)
    {
        long cap = ClassSize(bytes);
        TensorBuffer? b = null;
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            _rents++;
            if (_free.TryGetValue(cap, out Stack<TensorBuffer>? st) && st.Count > 0)
            {
                b = st.Pop();
                _idleBytes -= cap;
            }
            else
            {
                _nativeAllocs++;
            }

            _outstandingBytes += cap;
        }

        b ??= new TensorBuffer(this, cap);
        b.Prepare(bytes, zero);
        return b;
    }

    internal void Return(TensorBuffer b)
    {
        long cap = b.Capacity;
        bool keep;
        lock (_gate)
        {
            _outstandingBytes -= cap;
            keep = !_disposed && _idleBytes + cap <= MaxIdleBytes;
            if (keep)
            {
                if (!_free.TryGetValue(cap, out Stack<TensorBuffer>? st))
                {
                    _free[cap] = st = new Stack<TensorBuffer>();
                }

                st.Push(b);
                _idleBytes += cap;
            }
        }

        if (!keep)
        {
            b.FreeNative();
        }
    }

    /// <summary>A snapshot of the counters.</summary>
    public PoolStats GetStats()
    {
        lock (_gate)
        {
            return new PoolStats(_rents, _nativeAllocs, _idleBytes, _outstandingBytes);
        }
    }

    /// <summary>Frees every idle block. Rented blocks are unaffected.</summary>
    public void Trim()
    {
        List<TensorBuffer> doomed = [];
        lock (_gate)
        {
            foreach (Stack<TensorBuffer> st in _free.Values)
            {
                while (st.Count > 0)
                {
                    doomed.Add(st.Pop());
                }
            }

            _idleBytes = 0;
        }

        foreach (TensorBuffer b in doomed)
        {
            b.FreeNative();
        }
    }

    /// <summary>Frees idle blocks and refuses further rents. Blocks still rented are freed when returned or collected.</summary>
    public void Dispose()
    {
        lock (_gate)
        {
            _disposed = true;
        }

        Trim();
    }
}
