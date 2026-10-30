using System;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Training.Data;

/// <summary>Builds planar float32 image data from raw 8-bit pixel buffers. Decoding image files is out of scope: the .NET libraries have no codecs.</summary>
public static class ImageTensors
{
    /// <summary>Converts 8-bit grayscale pixels to floats: <c>(p/255 − mean) / std</c>.</summary>
    /// <param name="pixels">Row-major, <paramref name="height"/> rows of <paramref name="width"/> bytes, rows <paramref name="stride"/> bytes apart.</param>
    /// <param name="width">Columns, at least 1.</param>
    /// <param name="height">Rows, at least 1.</param>
    /// <param name="destination">At least <c>width × height</c> floats; receives <c>[1, height, width]</c>.</param>
    /// <param name="mean">Subtracted after scaling to [0, 1], default 0.</param>
    /// <param name="std">Divisor, default 1; must not be 0.</param>
    /// <param name="stride">Bytes between row starts; 0 (default) means <paramref name="width"/>.</param>
    /// <exception cref="ArgumentException">A size is not positive, or a buffer is too short.</exception>
    public static void Gray8ToPlanar(ReadOnlySpan<byte> pixels, int width, int height, Span<float> destination, float mean = 0f, float std = 1f, int stride = 0)
    {
        stride = CheckGeometry(pixels.Length, width, height, 1, stride, destination.Length, std);
        float scale = 1f / (255f * std), offset = mean / std;
        for (int y = 0; y < height; y++)
        {
            ReadOnlySpan<byte> src = pixels.Slice(y * stride, width);
            Span<float> dst = destination.Slice(y * width, width);
            for (int x = 0; x < width; x++) dst[x] = src[x] * scale - offset;
        }
    }

    /// <summary>Converts interleaved 8-bit RGB (R,G,B,R,G,B,…) to planar floats <c>[3, height, width]</c>: <c>(p/255 − mean[c]) / std[c]</c>.</summary>
    /// <param name="pixels">Row-major, each row <paramref name="width"/>×3 bytes, rows <paramref name="stride"/> bytes apart.</param>
    /// <param name="width">Columns, at least 1.</param>
    /// <param name="height">Rows, at least 1.</param>
    /// <param name="destination">At least <c>3 × width × height</c> floats.</param>
    /// <param name="mean">Three per-channel means, or empty for 0.</param>
    /// <param name="std">Three per-channel divisors, or empty for 1; none may be 0.</param>
    /// <param name="stride">Bytes between row starts; 0 (default) means <c>3 × width</c>.</param>
    /// <exception cref="ArgumentException">A size is not positive, a buffer is too short, or mean or std does not have three entries.</exception>
    public static void Rgb8ToPlanar(ReadOnlySpan<byte> pixels, int width, int height, Span<float> destination,
        ReadOnlySpan<float> mean = default, ReadOnlySpan<float> std = default, int stride = 0)
    {
        if (mean.Length != 0 && mean.Length != 3) throw new ArgumentException("mean needs three values.", nameof(mean));
        if (std.Length != 0 && std.Length != 3) throw new ArgumentException("std needs three values.", nameof(std));
        stride = CheckGeometry(pixels.Length, width, height, 3, stride, destination.Length, 1f);
        for (int c = 0; c < 3; c++)
        {
            float m = mean.Length == 0 ? 0f : mean[c], s = std.Length == 0 ? 1f : std[c];
            if (s == 0f) throw new ArgumentException("std must not contain 0.", nameof(std));
            float scale = 1f / (255f * s), offset = m / s;
            for (int y = 0; y < height; y++)
            {
                ReadOnlySpan<byte> src = pixels.Slice(y * stride, width * 3);
                Span<float> dst = destination.Slice((c * height + y) * width, width);
                for (int x = 0; x < width; x++) dst[x] = src[x * 3 + c] * scale - offset;
            }
        }
    }

    /// <summary>Makes a float32 tensor <c>[1, height, width]</c> from 8-bit grayscale pixels (see <see cref="Gray8ToPlanar"/>).</summary>
    /// <param name="pixels">The pixels.</param>
    /// <param name="width">Columns.</param>
    /// <param name="height">Rows.</param>
    /// <param name="mean">Mean, default 0.</param>
    /// <param name="std">Divisor, default 1.</param>
    /// <param name="pool">Pool for the tensor; default <see cref="BufferPool.Shared"/>.</param>
    /// <returns>A new tensor the caller disposes.</returns>
    public static Tensor Gray8ToTensor(ReadOnlySpan<byte> pixels, int width, int height, float mean = 0f, float std = 1f, BufferPool? pool = null)
    {
        CheckGeometry(pixels.Length, width, height, 1, 0, int.MaxValue, std);
        Tensor t = Tensor.Create(DType.F32, Shape.Create(1, height, width), pool);
        Gray8ToPlanar(pixels, width, height, t.MutableData<float>(), mean, std);
        return t;
    }

    /// <summary>Makes a float32 tensor <c>[3, height, width]</c> from interleaved 8-bit RGB pixels (see <see cref="Rgb8ToPlanar"/>).</summary>
    /// <param name="pixels">The pixels.</param>
    /// <param name="width">Columns.</param>
    /// <param name="height">Rows.</param>
    /// <param name="mean">Three means, or empty.</param>
    /// <param name="std">Three divisors, or empty.</param>
    /// <param name="pool">Pool for the tensor; default <see cref="BufferPool.Shared"/>.</param>
    /// <returns>A new tensor the caller disposes.</returns>
    public static Tensor Rgb8ToTensor(ReadOnlySpan<byte> pixels, int width, int height, ReadOnlySpan<float> mean = default, ReadOnlySpan<float> std = default, BufferPool? pool = null)
    {
        CheckGeometry(pixels.Length, width, height, 3, 0, int.MaxValue, 1f);
        Tensor t = Tensor.Create(DType.F32, Shape.Create(3, height, width), pool);
        Rgb8ToPlanar(pixels, width, height, t.MutableData<float>(), mean, std);
        return t;
    }

    private static int CheckGeometry(int pixelBytes, int width, int height, int channels, int stride, int destLength, float std)
    {
        if (width < 1 || height < 1) throw new ArgumentException($"The image size {width}x{height} must be positive.");
        if (std == 0f || float.IsNaN(std)) throw new ArgumentException("std must not be 0 or NaN.", nameof(std));
        long row = (long)width * channels;
        long s = stride == 0 ? row : stride;
        if (s < row) throw new ArgumentException($"stride {s} is smaller than a row of {row} bytes.", nameof(stride));
        if ((height - 1) * s + row > pixelBytes) throw new ArgumentException($"{pixelBytes} bytes are too few for a {width}x{height}x{channels} image.");
        if (row * height > destLength) throw new ArgumentException("The destination is too short.");
        return (int)s;
    }
}
