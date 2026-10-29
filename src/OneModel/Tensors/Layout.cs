using System;
using System.Collections.Generic;
using System.Threading;
namespace RecursiveAI.OneModel.Tensors;

/// <summary>The physical arrangement of a rank-4 image tensor whose logical shape is always (N, C, H, W).</summary>
public enum LayoutKind : byte
{
    /// <summary>No named arrangement: any strides (a transposed, sliced or broadcast view, or a tensor of another rank).</summary>
    Strided = 0,
    /// <summary>Row-major (N, C, H, W): strides (C·H·W, H·W, W, 1).</summary>
    Nchw = 1,
    /// <summary>Channels last: storage order (N, H, W, C), so strides (H·W·C, 1, W·C, C) over the logical (N, C, H, W) shape.</summary>
    Nhwc = 2,
    /// <summary>
    /// Blocked channels: storage is [N][ceil(C/B)][H][W][B] with block size B (the channel blocks of 8 or 16 that a SIMD microkernel
    /// wants in one vector). Channels past C in the last block are zero. Not describable by strides, so it is an opaque layout:
    /// only conversion and the physical span are available.
    /// </summary>
    NchwC = 3,
}

/// <summary>A layout kind together with its block size (0 unless the kind is <see cref="LayoutKind.NchwC"/>).</summary>
/// <param name="Kind">The arrangement.</param>
/// <param name="BlockSize">The channel block size B for <see cref="LayoutKind.NchwC"/>, otherwise 0.</param>
public readonly record struct TensorLayout(LayoutKind Kind, int BlockSize)
{
    /// <summary>Plain (no named arrangement).</summary>
    public static TensorLayout Strided => new(LayoutKind.Strided, 0);

    /// <summary>Row-major NCHW.</summary>
    public static TensorLayout Nchw => new(LayoutKind.Nchw, 0);

    /// <summary>Channels-last NHWC.</summary>
    public static TensorLayout Nhwc => new(LayoutKind.Nhwc, 0);

    /// <summary>Blocked NCHWc with block size <paramref name="blockSize"/>.</summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="blockSize"/> is outside [1, 256].</exception>
    public static TensorLayout NchwC(int blockSize)
    {
        if (blockSize is < 1 or > 256)
        {
            throw new ArgumentOutOfRangeException(nameof(blockSize), blockSize, "The block size must be in [1, 256].");
        }

        return new TensorLayout(LayoutKind.NchwC, blockSize);
    }

    /// <summary>Formats as <c>NCHW</c>, <c>NHWC</c>, <c>NCHW8c</c> or <c>Strided</c>.</summary>
    public override string ToString() => Kind switch
    {
        LayoutKind.Nchw => "NCHW",
        LayoutKind.Nhwc => "NHWC",
        LayoutKind.NchwC => $"NCHW{BlockSize}c",
        _ => "Strided",
    };
}
