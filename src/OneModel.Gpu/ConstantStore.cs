using System;
using System.Collections.Generic;
using RecursiveAI.OneGpu;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu;

/// <summary>A constant tensor on the device, in the element type it has on the host.</summary>
internal sealed class DeviceConstant : IDisposable
{
    private IDisposable? _buffer;

    public DeviceConstant(GpuBuffer<float> f32)
    {
        _buffer = f32;
        F32 = f32;
        Bytes = f32.Length * 4L;
        Length = f32.Length;
    }

    public DeviceConstant(GpuBuffer<Half> f16)
    {
        _buffer = f16;
        F16 = f16;
        Bytes = f16.Length * 2L;
        Length = f16.Length;
    }

    public DeviceConstant(IDisposable buffer, long bytes, int length)
    {
        _buffer = buffer;
        Bytes = bytes;
        Length = length;
    }

    public GpuBuffer<float>? F32 { get; private set; }

    public GpuBuffer<Half>? F16 { get; private set; }

    public long Bytes { get; }

    public int Length { get; }

    /// <summary>The buffer as the host representation of its element type, or null when the constant has another.</summary>
    public GpuBuffer<T>? As<T>() where T : unmanaged => _buffer as GpuBuffer<T>;

    public void Dispose()
    {
        _buffer?.Dispose();
        _buffer = null;
        F32 = null;
        F16 = null;
    }
}

/// <summary>
/// Weights on the device. A weight of the model is uploaded once for the whole session and shared by every plan; a constant a plan made
/// for itself (a folded value) belongs to that plan's program. Not thread-safe; the backend's gate guards it.
/// </summary>
internal sealed class ConstantStore : IDisposable
{
    private readonly GpuDevice _device;
    private readonly Dictionary<int, DeviceConstant> _graph = [];
    private readonly Dictionary<(int, bool), ConstantChunks> _split = [];

    public ConstantStore(GpuDevice device) => _device = device;

    public long Bytes { get; private set; }

    public int Buffers { get; private set; }

    /// <summary>
    /// The device copy of constant value <paramref name="id"/> of <paramref name="plan"/>, uploading it if it is new. Set
    /// <paramref name="planLocal"/> when the plan made the tensor itself, so the caller owns the returned copy and must add it to the
    /// program's owned list.
    /// </summary>
    public DeviceConstant Get(ExecutionPlan plan, int id, out bool planLocal)
    {
        Tensor t = plan.Consts[id];
        Tensor shared = plan.Graph.Constants[id];
        planLocal = shared.IsEmpty || !ReferenceEquals(shared.Buffer, t.Buffer) || shared.Offset != t.Offset;
        if (!planLocal && _graph.TryGetValue(id, out DeviceConstant? have))
        {
            return have;
        }

        DeviceConstant made = Upload(t);
        Bytes += made.Bytes;
        Buffers++;
        if (!planLocal)
        {
            _graph[id] = made;
        }

        return made;
    }

    /// <summary>
    /// The device copy of constant value <paramref name="id"/> as several buffers, for a constant larger than one buffer binding: a float32 or
    /// float16 matrix of <paramref name="rows"/> by <paramref name="columns"/> elements cut into pieces of whole rows
    /// (<paramref name="byColumns"/> false) or whole columns (true, each piece repacked dense), none above <paramref name="maxBytes"/>.
    /// Pieces are even in size except the last. Set <paramref name="planLocal"/> when the plan made the tensor itself (the caller then owns the pieces).
    /// </summary>
    public ConstantChunks GetSplit(ExecutionPlan plan, int id, int rows, int columns, bool byColumns, long maxBytes, out bool planLocal)
    {
        Tensor t = plan.Consts[id];
        Tensor shared = plan.Graph.Constants[id];
        planLocal = shared.IsEmpty || !ReferenceEquals(shared.Buffer, t.Buffer) || shared.Offset != t.Offset;
        if (!planLocal && _split.TryGetValue((id, byColumns), out ConstantChunks? have))
        {
            return have;
        }

        Tensor src = t.IsContiguous ? t : t.Contiguous();
        ConstantChunks made;
        try
        {
            made = src.DType switch
            {
                DType.F32 => Split<float>(src.Data<float>(), rows, columns, byColumns, maxBytes),
                DType.F16 => Split<Half>(src.Data<Half>(), rows, columns, byColumns, maxBytes),
                _ => throw new InvalidOperationException($"A {src.DType} constant cannot be split on the device."),
            };
        }
        finally
        {
            if (!ReferenceEquals(src.Buffer, t.Buffer))
            {
                src.Dispose();
            }
        }

        Bytes += made.Bytes;
        Buffers += made.Pieces.Length;
        if (!planLocal)
        {
            _split[(id, byColumns)] = made;
        }

        return made;
    }

    private ConstantChunks Split<T>(ReadOnlySpan<T> data, int rows, int columns, bool byColumns, long maxBytes) where T : unmanaged
    {
        int size = System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
        int extent = byColumns ? columns : rows, other = byColumns ? rows : columns;
        long maxExtent = Math.Max(1, maxBytes / ((long)other * size));
        int pieces = (int)((extent + maxExtent - 1) / maxExtent);
        int per = (extent + pieces - 1) / pieces;
        if (size == 2 && !byColumns && (columns & 1) != 0)
        {
            throw new InvalidOperationException("Half-precision rows of an odd length cannot be split.");
        }

        var list = new List<DeviceConstant>();
        long bytes = 0;
        try
        {
            for (int start = 0; start < extent; start += per)
            {
                int count = Math.Min(per, extent - start);
                T[] piece;
                if (byColumns)
                {
                    piece = new T[checked(rows * count)];
                    for (int r = 0; r < rows; r++)
                    {
                        data.Slice(checked(r * columns + start), count).CopyTo(piece.AsSpan(r * count, count));
                    }
                }
                else
                {
                    piece = data.Slice(checked(start * columns), checked(count * columns)).ToArray();
                }

                if (size == 2 && (piece.Length & 1) != 0)
                {
                    Array.Resize(ref piece, piece.Length + 1);
                }

                DeviceConstant c = size == 4
                    ? new DeviceConstant(_device.CreateBuffer<float>(System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(piece), GpuDevice.DefaultComputeUsage, "weights-f32-piece"))
                    : new DeviceConstant(_device.CreateBuffer<Half>(System.Runtime.InteropServices.MemoryMarshal.Cast<T, Half>(piece), GpuDevice.DefaultComputeUsage, "weights-f16-piece"));
                list.Add(c);
                bytes += c.Bytes;
            }
        }
        catch
        {
            foreach (DeviceConstant c in list)
            {
                c.Dispose();
            }

            throw;
        }

        return new ConstantChunks([.. list], per, bytes);
    }

    /// <summary>Uploads a host tensor (copying it first when it is not contiguous).</summary>
    public DeviceConstant Upload(in Tensor t)
    {
        Tensor src = t.IsContiguous ? t : t.Contiguous();
        try
        {
            return src.DType switch
            {
                DType.F32 => new DeviceConstant(_device.CreateBuffer<float>(src.Data<float>(), GpuDevice.DefaultComputeUsage, "weights-f32")),
                DType.F16 => new DeviceConstant(UploadHalf(src.Data<Half>())),
                DType.I64 => Make(_device.CreateBuffer<long>(src.Data<long>(), GpuDevice.DefaultComputeUsage, "weights-i64"), 8),
                DType.I32 => Make(_device.CreateBuffer<int>(src.Data<int>(), GpuDevice.DefaultComputeUsage, "weights-i32"), 4),
                DType.U8 => MakeBytes(src.Data<byte>(), static (d, data, l) => d.CreateBuffer<byte>(data, GpuDevice.DefaultComputeUsage, l)),
                DType.Bool => MakeBytes(System.Runtime.InteropServices.MemoryMarshal.AsBytes(src.Data<bool>()), static (d, data, l) => d.CreateBuffer<byte>(data, GpuDevice.DefaultComputeUsage, l)),
                DType.I8 => MakeBytes(src.Data<sbyte>(), static (d, data, l) => d.CreateBuffer<sbyte>(data, GpuDevice.DefaultComputeUsage, l)),
                _ => throw new InvalidOperationException($"A {src.DType} constant cannot be placed on the device."),
            };
        }
        finally
        {
            if (!ReferenceEquals(src.Buffer, t.Buffer))
            {
                src.Dispose();
            }
        }
    }

    private static DeviceConstant Make<T>(GpuBuffer<T> buffer, int elementBytes) where T : unmanaged => new(buffer, (long)buffer.Length * elementBytes, buffer.Length);

    // 8-bit data is padded to a whole number of 32-bit words, which the kernels read.
    private DeviceConstant MakeBytes<T>(ReadOnlySpan<T> data, Func<GpuDevice, T[], string, GpuBuffer<T>> create) where T : unmanaged
    {
        T[] padded = new T[(data.Length + 3) / 4 * 4];
        data.CopyTo(padded);
        return Make(create(_device, padded, "weights-8"), 1);
    }

    // The kernels read half-precision buffers as 32-bit words, so the allocation holds a whole number of them.
    private GpuBuffer<Half> UploadHalf(ReadOnlySpan<Half> data)
    {
        if (data.Length % 2 == 0)
        {
            return _device.CreateBuffer<Half>(data, GpuDevice.DefaultComputeUsage, "weights-f16");
        }

        Half[] padded = new Half[data.Length + 1];
        data.CopyTo(padded);
        return _device.CreateBuffer<Half>(padded, GpuDevice.DefaultComputeUsage, "weights-f16");
    }

    public void Dispose()
    {
        foreach (DeviceConstant c in _graph.Values)
        {
            c.Dispose();
        }

        _graph.Clear();
        foreach (ConstantChunks c in _split.Values)
        {
            c.Dispose();
        }

        _split.Clear();
    }
}

/// <summary>A constant held as several device buffers, each no larger than one binding: pieces of <see cref="Per"/> rows or columns (the last may be shorter).</summary>
internal sealed class ConstantChunks : IDisposable
{
    public ConstantChunks(DeviceConstant[] pieces, int per, long bytes)
    {
        Pieces = pieces;
        Per = per;
        Bytes = bytes;
    }

    public DeviceConstant[] Pieces { get; }

    /// <summary>Rows (or columns) in every piece but possibly the last.</summary>
    public int Per { get; }

    public long Bytes { get; }

    public void Dispose()
    {
        foreach (DeviceConstant c in Pieces)
        {
            c.Dispose();
        }
    }
}
