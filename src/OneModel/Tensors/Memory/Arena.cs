using System;
using System.Collections.Generic;
using System.Threading;
namespace RecursiveAI.OneModel.Tensors;

/// <summary>
/// Bump allocation for per-run state. An arena takes a few large chunks from a <see cref="BufferPool"/> and hands out 64-byte-aligned
/// buffers from them; <see cref="Reset"/> reclaims everything at once and invalidates every tensor allocated since (they throw
/// <see cref="StaleTensorException"/>). After one warm-up run with the same allocation pattern, further runs allocate nothing.
/// An arena is for one run at a time and is not thread-safe; concurrent runs each use their own.
/// </summary>
/// <remarks>
/// Bounds: every returned buffer lies inside one chunk and covers [offset, offset + length) with offset a multiple of 64; the
/// reserved span is the length rounded up to 64 (plus 64 canary bytes under <c>CHECKED</c>), so neighbours never overlap.
/// </remarks>
public sealed class Arena : IDisposable
{
    private readonly BufferPool _pool;
    private readonly long _chunkBytes;
    private readonly List<TensorBuffer> _chunks = [];
    private readonly List<TensorBuffer> _slots = [];
    private int _slotCount;
    private int _chunkIndex;
    private long _cursor;
    private long _used;
    private bool _disposed;

    /// <summary>Creates an arena.</summary>
    /// <param name="pool">Where chunks come from; <see cref="BufferPool.Shared"/> when null.</param>
    /// <param name="chunkBytes">Preferred chunk size, default 1 MiB. A larger request gets a chunk of its own size.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="chunkBytes"/> is not positive.</exception>
    public Arena(BufferPool? pool = null, long chunkBytes = 1 << 20)
    {
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(chunkBytes, 0L);
        _pool = pool ?? BufferPool.Shared;
        _chunkBytes = chunkBytes;
    }

    /// <summary>Bytes handed out since the last <see cref="Reset"/> (aligned lengths, including guard bytes under <c>CHECKED</c>).</summary>
    public long UsedBytes => _used;

    /// <summary>Number of chunks currently held.</summary>
    public int ChunkCount => _chunks.Count;

    /// <summary>Allocates <paramref name="bytes"/> bytes, 64-byte aligned.</summary>
    /// <param name="bytes">Size, 0 to 2^31-1.</param>
    /// <param name="zero">Clear the contents; otherwise they are unspecified.</param>
    /// <exception cref="TensorShapeException"><paramref name="bytes"/> is out of range.</exception>
    /// <exception cref="ObjectDisposedException">The arena was disposed.</exception>
    public unsafe TensorBuffer Allocate(long bytes, bool zero = false)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((ulong)bytes > int.MaxValue)
        {
            throw new TensorShapeException($"A buffer of {bytes} bytes is outside [0, 2^31-1].", nameof(bytes));
        }

        long need = ((bytes + 63) & ~63L) + TensorBuffer.Guard;
        if (need == 0)
        {
            need = 64;
        }

        while (true)
        {
            if (_chunkIndex < _chunks.Count)
            {
                if (_chunks[_chunkIndex].Length - _cursor >= need)
                {
                    break;
                }

                _chunkIndex++;
                _cursor = 0;
            }
            else
            {
                _chunks.Add(_pool.Rent(Math.Min(Math.Max(_chunkBytes, need), int.MaxValue)));
                _cursor = 0;
            }
        }

        TensorBuffer chunk = _chunks[_chunkIndex];
        TensorBuffer slot;
        if (_slotCount < _slots.Count)
        {
            slot = _slots[_slotCount];
        }
        else
        {
            slot = new TensorBuffer(this);
            _slots.Add(slot);
        }

        _slotCount++;
        slot.Rebind(chunk.Pointer + _cursor, need);
        slot.Prepare(bytes, zero);
        _cursor += need;
        _used += need;
        return slot;
    }

    /// <summary>Reclaims all allocations. Every tensor allocated from this arena since the last reset becomes stale.</summary>
    public void Reset()
    {
        for (int i = 0; i < _slotCount; i++)
        {
            TensorBuffer s = _slots[i];
            s.EndGeneration(s.Generation, out bool overrun);
            if (overrun)
            {
                throw new InvalidOperationException("A kernel wrote past the end of an arena buffer (canary bytes were overwritten).");
            }
        }

        _slotCount = 0;
        _chunkIndex = 0;
        _cursor = 0;
        _used = 0;
    }

    /// <summary>Resets the arena and returns its chunks to the pool.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        Reset();
        _disposed = true;
        foreach (TensorBuffer c in _chunks)
        {
            c.Release(c.Generation);
        }

        _chunks.Clear();
    }
}
