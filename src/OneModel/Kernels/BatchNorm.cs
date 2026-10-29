using System;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Inference batch normalisation: per channel c, <c>y = x * a[c] + b[c]</c> with <c>a = scale / sqrt(var + epsilon)</c> and
/// <c>b = bias - mean * a</c>. Every step is a separately rounded IEEE multiply, add, divide or square root (never fused), so the
/// result is the same on every path and in both modes; exact mode only rewrites NaN results as the canonical NaN.
/// </summary>
internal static class BatchNorm
{
    /// <summary>
    /// Normalises <paramref name="x"/> (N x C x plane) into <paramref name="y"/> (same shape; may be the same memory).
    /// </summary>
    /// <exception cref="ArgumentException">A buffer is too small.</exception>
    public static void Inference(
        ReadOnlySpan<float> x, Span<float> y, int n, int channels, int plane,
        ReadOnlySpan<float> scale, ReadOnlySpan<float> bias, ReadOnlySpan<float> mean, ReadOnlySpan<float> variance, float epsilon,
        bool canonicalNaN)
    {
        long total = (long)n * channels * plane;
        if (x.Length < total || y.Length < total || scale.Length < channels || bias.Length < channels || mean.Length < channels ||
            variance.Length < channels)
        {
            throw new ArgumentException("A buffer is smaller than the batch-normalisation shape needs.");
        }

        for (int img = 0; img < n; img++)
        {
            for (int c = 0; c < channels; c++)
            {
                float a = scale[c] / MathF.Sqrt(variance[c] + epsilon);
                float b = bias[c] - (mean[c] * a);
                int offset = ((img * channels) + c) * plane;
                ReadOnlySpan<float> src = x.Slice(offset, plane);
                Span<float> dst = y.Slice(offset, plane);
                for (int i = 0; i < dst.Length; i++)
                {
                    float v = (src[i] * a) + b;
                    dst[i] = canonicalNaN && float.IsNaN(v) ? BitConverter.UInt32BitsToSingle(0x7FC00000u) : v;
                }
            }
        }
    }
}
