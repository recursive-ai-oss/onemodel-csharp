using System;
using System.Collections.Generic;
using RecursiveAI.OneGpu;

namespace RecursiveAI.OneModel.Gpu;

/// <summary>The element type a value has on the device.</summary>
internal enum DeviceElement : byte
{
    /// <summary>binary32</summary>
    F32,

    /// <summary>binary16</summary>
    F16,

    /// <summary>Signed 32-bit integer.</summary>
    I32,

    /// <summary>Signed 64-bit integer.</summary>
    I64,

    /// <summary>Unsigned 8-bit integer (a quantized activation).</summary>
    U8,

    /// <summary>Boolean, one byte (0 or 1) per element, four to a 32-bit word, as the CPU holds it.</summary>
    Bool,

    /// <summary>Signed 8-bit integer (a quantized activation made at run time).</summary>
    I8,
}

/// <summary>Where one value lives in the arena: a slab, and an element offset into it.</summary>
internal readonly record struct ValueSlot(DeviceElement Element, int Slab, long Offset, long Length);

/// <summary>A value that needs device memory from step <see cref="Start"/> to step <see cref="End"/>, both inclusive, on one timeline. <see cref="Key"/> numbers the values of one planning call from 0.</summary>
internal readonly record struct LiveValue(int Key, DeviceElement Element, long Length, int Start, int End);

/// <summary>
/// Device memory for the intermediate tensors of every plan of a session: a few big buffers ("slabs") per element type that plans carve
/// up by lifetime. A slab only grows, and only when a plan needs more than any earlier one, so a shape that has run before allocates nothing.
/// Not thread-safe; the backend's gate guards it.
/// </summary>
internal sealed class DeviceArena : IDisposable
{
    // Offsets are kept 64 bytes apart so copies and wide loads start aligned.
    private const long AlignBytes = 64;

    // Growth is rounded up so a run of slightly larger shapes does not reallocate every time.
    private const long GrowthQuantumBytes = 1 << 20;

    private readonly GpuDevice _device;
    private readonly Action _buffersReplaced;
    private const int ElementKinds = 7;
    private readonly List<Slab>[] _slabs = [[], [], [], [], [], [], []];

    public DeviceArena(GpuDevice device, Action buffersReplaced)
    {
        _device = device;
        _buffersReplaced = buffersReplaced;
        // One slab holds at most what a single storage-buffer binding can cover, because a kernel binds the whole buffer.
        ulong binding = device.Capabilities.Compute.MaxStorageBufferBindingBytes;
        ulong buffer = device.Capabilities.Compute.MaxBufferBytes;
        ulong limit = Math.Min(binding == 0 ? ulong.MaxValue : binding, buffer == 0 ? ulong.MaxValue : buffer);
        SlabLimitBytes = (long)Math.Min(limit, 1UL << 30);
    }

    /// <summary>Largest slab, in bytes.</summary>
    public long SlabLimitBytes { get; }

    /// <summary>Bytes held by the slabs now.</summary>
    public long Bytes { get; private set; }

    /// <summary>Device buffers created so far, including the ones that replaced a smaller slab.</summary>
    public int Allocations { get; private set; }

    public static int ElementBytes(DeviceElement e) => e switch
    {
        DeviceElement.F32 or DeviceElement.I32 => 4,
        DeviceElement.F16 => 2,
        DeviceElement.I64 => 8,
        _ => 1,
    };

    private static long AlignElements(DeviceElement e) => AlignBytes / ElementBytes(e);

    private List<Slab> SlabsOf(DeviceElement e) => _slabs[(int)e];

    public GpuBuffer<float> F32(int slab) => Get<float>(DeviceElement.F32, slab);

    public GpuBuffer<Half> F16(int slab) => Get<Half>(DeviceElement.F16, slab);

    /// <summary>The slab buffer of an element type; <typeparamref name="T"/> must be the type's host representation.</summary>
    public GpuBuffer<T> Get<T>(DeviceElement e, int slab) where T : unmanaged =>
        (GpuBuffer<T>?)_slabs[(int)e][slab].Buffer ?? throw new InvalidOperationException("The arena slab was released.");

    /// <summary>Slab capacities (elements) a layout needs, one list per element type (indexed by <see cref="DeviceElement"/>).</summary>
    public sealed record Layout(ValueSlot[] Slots, long[][] SlabElements)
    {
        public long Bytes
        {
            get
            {
                long b = 0;
                for (int t = 0; t < SlabElements.Length; t++)
                {
                    foreach (long e in SlabElements[t])
                    {
                        b += e * ElementBytes((DeviceElement)t);
                    }
                }

                return b;
            }
        }
    }

    /// <summary>
    /// Assigns every value a slab and an offset so that values alive at the same time never overlap, reusing space as values die. Greedy
    /// first fit in order of first use, which is what graph schedules want: most values die soon after they are made.
    /// </summary>
    /// <param name="values">The values to place; lengths are element counts and must be positive.</param>
    /// <param name="apart">Pairs of keys that must not share a slab, even when their lifetimes never overlap: a device that refuses one buffer
    /// bound for reading and for writing in a single dispatch needs an operator's inputs, outputs and scratch in different buffers.</param>
    public Layout Plan(List<LiveValue> values, List<(int A, int B)>? apart = null)
    {
        Dictionary<int, List<int>>? neighbours = null;
        if (apart is { Count: > 0 })
        {
            neighbours = [];
            foreach ((int a, int b) in apart)
            {
                if (a == b)
                {
                    continue;
                }

                if (!neighbours.TryGetValue(a, out List<int>? la)) { neighbours[a] = la = []; }
                if (!neighbours.TryGetValue(b, out List<int>? lb)) { neighbours[b] = lb = []; }
                la.Add(b);
                lb.Add(a);
            }
        }

        var slabOfKey = new Dictionary<int, (int Element, int Slab)>();
        var slots = new ValueSlot[values.Count];
        var sizes = new List<long>[ElementKinds];
        var placed = new List<(int Slab, long Offset, long Length, int Start, int End)>[ElementKinds];
        for (int i = 0; i < ElementKinds; i++)
        {
            sizes[i] = [];
            placed[i] = [];
        }

        values.Sort(static (a, b) => a.Start != b.Start ? a.Start.CompareTo(b.Start) : b.Length.CompareTo(a.Length));
        foreach (LiveValue v in values)
        {
            int t = (int)v.Element;
            long limit = SlabLimitBytes / ElementBytes(v.Element);
            long align = AlignElements(v.Element);
            long length = checked((v.Length + align - 1) / align * align);
            if (length > limit)
            {
                throw new GpuUnsupportedException($"a tensor of {v.Length} elements does not fit one buffer binding on this device",
                    nameof(GpuComputeCapabilities.MaxStorageBufferBindingBytes), $"at most {limit} elements", $"{v.Length} elements");
            }

            List<(int Slab, long Offset, long Length, int Start, int End)> list = placed[t];
            bool done = false;
            for (int slab = 0; slab <= sizes[t].Count && !done; slab++)
            {
                if (neighbours is not null && slab < sizes[t].Count && neighbours.TryGetValue(v.Key, out List<int>? close) && NeighbourIn(close, slabOfKey, t, slab))
                {
                    continue;
                }

                if (slab == sizes[t].Count)
                {
                    sizes[t].Add(0);
                }

                // The first gap in this slab, between values whose lifetimes overlap this one, that holds the value.
                var busy = new List<(long Offset, long End)>();
                foreach (var p in list)
                {
                    if (p.Slab == slab && p.Start <= v.End && v.Start <= p.End)
                    {
                        busy.Add((p.Offset, p.Offset + p.Length));
                    }
                }

                busy.Sort(static (a, b) => a.Offset.CompareTo(b.Offset));
                long at = 0;
                foreach ((long offset, long end) in busy)
                {
                    if (offset - at >= length)
                    {
                        break;
                    }

                    at = Math.Max(at, end);
                }

                if (at + length <= limit)
                {
                    list.Add((slab, at, length, v.Start, v.End));
                    slots[v.Key] = new ValueSlot(v.Element, slab, at, v.Length);
                    slabOfKey[v.Key] = (t, slab);
                    sizes[t][slab] = Math.Max(sizes[t][slab], at + length);
                    done = true;
                }
            }
        }

        var slabElements = new long[ElementKinds][];
        for (int i = 0; i < ElementKinds; i++)
        {
            slabElements[i] = [.. sizes[i]];
        }

        return new Layout(slots, slabElements);
    }

    private static bool NeighbourIn(List<int> keys, Dictionary<int, (int Element, int Slab)> slabOfKey, int element, int slab)
    {
        foreach (int k in keys)
        {
            if (slabOfKey.TryGetValue(k, out (int Element, int Slab) at) && at.Element == element && at.Slab == slab)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>Makes every slab the layout uses at least as big as it needs; creates or replaces buffers only where it must.</summary>
    public void Ensure(Layout layout)
    {
        bool replaced = false;
        for (int t = 0; t < ElementKinds; t++)
        {
            replaced |= Ensure((DeviceElement)t, layout.SlabElements[t]);
        }

        if (replaced)
        {
            // Kernels cache bind groups by buffer; the old buffers must not stay reachable from them.
            _buffersReplaced();
        }
    }

    private bool Ensure(DeviceElement e, long[] elements)
    {
        List<Slab> slabs = SlabsOf(e);
        bool replaced = false;
        int bytes = ElementBytes(e);
        for (int i = 0; i < elements.Length; i++)
        {
            if (i == slabs.Count)
            {
                slabs.Add(new Slab());
            }

            Slab s = slabs[i];
            if (elements[i] <= s.Elements)
            {
                continue;
            }

            long quantum = GrowthQuantumBytes / bytes;
            long want = Math.Min((elements[i] + quantum - 1) / quantum * quantum, SlabLimitBytes / bytes);
            want = Math.Max(want, elements[i]);
            if (bytes == 1)
            {
                // Byte kernels read and write whole 32-bit words.
                want = (want + 3) / 4 * 4;
            }

            if (want > int.MaxValue)
            {
                throw new GpuUnsupportedException("the arena slab has more elements than a kernel can index", "ElementIndex32", "at most 2147483647 elements", $"{want} elements");
            }

            s.Buffer?.Dispose();
            s.Buffer = e switch
            {
                DeviceElement.F32 => _device.CreateBuffer<float>((int)want, label: "arena-f32"),
                DeviceElement.F16 => _device.CreateBuffer<Half>((int)want, label: "arena-f16"),
                DeviceElement.I32 => _device.CreateBuffer<int>((int)want, label: "arena-i32"),
                DeviceElement.I64 => _device.CreateBuffer<long>((int)want, label: "arena-i64"),
                DeviceElement.Bool => _device.CreateBuffer<byte>((int)want, label: "arena-bool"),
                DeviceElement.I8 => _device.CreateBuffer<sbyte>((int)want, label: "arena-i8"),
                _ => _device.CreateBuffer<byte>((int)want, label: "arena-u8"),
            };

            Bytes += (want - s.Elements) * bytes;
            s.Elements = want;
            Allocations++;
            replaced = true;
        }

        return replaced;
    }

    public void Dispose()
    {
        foreach (List<Slab> list in _slabs)
        {
            foreach (Slab s in list)
            {
                s.Buffer?.Dispose();
            }

            list.Clear();
        }

        Bytes = 0;
    }

    private sealed class Slab
    {
        public IDisposable? Buffer;
        public long Elements;
    }
}
