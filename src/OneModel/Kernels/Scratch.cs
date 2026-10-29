using System;
using System.Runtime.InteropServices;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Per-thread scratch memory for packing and im2col, 64-byte aligned so every vector load is aligned to a cache line. Each slot
/// grows to the largest size asked for and is kept for the life of the thread, so repeated calls allocate nothing. Slots are
/// separate so a kernel can hold two scratch areas at once.
/// </summary>
internal static unsafe class Scratch
{
    /// <summary>Packed A block (used by whichever thread runs a tile).</summary>
    public const int PackA = 0;

    /// <summary>The whole packed B (used by the thread that called the GEMM).</summary>
    public const int PackB = 1;

    /// <summary>The im2col matrix or padded plane (used by the thread that called the convolution or runs the plane).</summary>
    public const int Im2Col = 2;

    /// <summary>Padded depthwise input plane.</summary>
    public const int Plane = 3;

    private const int Slots = 4;

    [ThreadStatic]
    private static float*[]? t_pointers;

    [ThreadStatic]
    private static long[]? t_lengths;

    /// <summary>Returns at least <paramref name="floats"/> floats of 64-byte-aligned memory for <paramref name="slot"/>; contents are undefined.</summary>
    public static float* Get(int slot, long floats)
    {
        t_pointers ??= new float*[Slots];
        t_lengths ??= new long[Slots];
        if (floats < 16)
        {
            floats = 16;
        }

        if (t_lengths[slot] < floats)
        {
            if (t_pointers[slot] != null)
            {
                NativeMemory.AlignedFree(t_pointers[slot]);
            }

            // Round up so slow growth does not reallocate on every call.
            long rounded = (floats + 4095) & ~4095L;
            t_pointers[slot] = (float*)NativeMemory.AlignedAlloc((nuint)(rounded * sizeof(float)), 64);
            t_lengths[slot] = rounded;
        }

        return t_pointers[slot];
    }
}
