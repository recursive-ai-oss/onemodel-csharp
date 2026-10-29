using System;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>Interpolation of Resize.</summary>
internal enum ResizeMode
{
    /// <summary>The nearest source element (chosen by <see cref="ResizeNearest"/>).</summary>
    Nearest,

    /// <summary>Linear, bilinear, trilinear...: two taps per axis.</summary>
    Linear,

    /// <summary>Cubic convolution (Keys 1981): four taps per axis.</summary>
    Cubic,
}

/// <summary>How an output coordinate maps to an input coordinate.</summary>
internal enum ResizeCoord
{
    HalfPixel,
    HalfPixelSymmetric,
    PytorchHalfPixel,
    AlignCorners,
    Asymmetric,
    TfCropAndResize,
}

/// <summary>How a fractional input coordinate picks a source element in nearest mode.</summary>
internal enum ResizeNearest
{
    RoundPreferFloor,
    RoundPreferCeil,
    Floor,
    Ceil,
}

/// <summary>Resize attributes.</summary>
internal readonly record struct ResizeOptions(
    ResizeMode Mode, ResizeCoord Coord, ResizeNearest Nearest, float CubicA, bool ExcludeOutside, float ExtrapolationValue);

/// <summary>
/// N-dimensional Resize done one axis at a time: each axis is resampled from per-output-position tap tables (source index and
/// weight), axes that shrink first so the intermediate never exceeds the larger of input and output. Nearest copies elements, so
/// it moves bits unchanged; linear and cubic sum the taps in increasing source order, each product and sum rounded separately.
/// Coordinates are computed in float32. Nothing is fused, so fast and exact give the same bits (exact also canonicalises NaN).
/// </summary>
internal static class Resize
{
    /// <summary>Taps per axis for a mode.</summary>
    public static int TapsPer(ResizeMode mode) => mode switch { ResizeMode.Nearest => 1, ResizeMode.Linear => 2, _ => 4 };

    /// <summary>
    /// Resamples <paramref name="x"/> (extents <paramref name="inDims"/>) into <paramref name="y"/> (extents <paramref name="outDims"/>).
    /// </summary>
    /// <param name="o">Attributes.</param>
    /// <param name="inDims">Input extents.</param>
    /// <param name="outDims">Output extents, positive.</param>
    /// <param name="scales">Per-axis scale used by the coordinate mapping.</param>
    /// <param name="roi">Region of interest, 2 * rank values (starts then ends, as fractions), used by tf_crop_and_resize; may be empty.</param>
    /// <param name="x">Input.</param>
    /// <param name="y">Output.</param>
    /// <param name="bufA">Scratch of at least max(input, output) elements.</param>
    /// <param name="bufB">Scratch of at least max(input, output) elements.</param>
    /// <param name="tapIdx">Scratch of at least TapsPer(mode) * max output extent.</param>
    /// <param name="tapW">Scratch of the same size.</param>
    /// <param name="canonicalNaN">Rewrite NaN results as the canonical NaN (exact mode).</param>
    public static void Run(
        in ResizeOptions o, ReadOnlySpan<int> inDims, ReadOnlySpan<int> outDims, ReadOnlySpan<float> scales, ReadOnlySpan<float> roi,
        ReadOnlySpan<float> x, Span<float> y, Span<float> bufA, Span<float> bufB, Span<int> tapIdx, Span<float> tapW, bool canonicalNaN)
    {
        int rank = inDims.Length;
        long inLen = 1, outLen = 1;
        for (int i = 0; i < rank; i++)
        {
            inLen *= inDims[i];
            outLen *= outDims[i];
        }

        if (outLen == 0)
        {
            return;
        }

        if (x.Length < inLen || y.Length < outLen || bufA.Length < Math.Max(inLen, outLen) || bufB.Length < Math.Max(inLen, outLen))
        {
            throw new ArgumentException("A buffer is smaller than the resize shape needs.");
        }

        // Axes that need a pass: shrinking ones first, then growing ones, each group in axis order.
        Span<int> order = stackalloc int[Shape8];
        int passes = 0;
        for (int pass = 0; pass < 2; pass++)
        {
            for (int a = 0; a < rank; a++)
            {
                bool grow = outDims[a] > inDims[a];
                if (grow == (pass == 1) && !IsIdentity(o, inDims[a], outDims[a], scales[a]))
                {
                    order[passes++] = a;
                }
            }
        }

        Span<int> cur = stackalloc int[Shape8];
        inDims.CopyTo(cur);
        ReadOnlySpan<float> src = x;
        int taps = TapsPer(o.Mode);
        for (int p = 0; p < passes; p++)
        {
            int a = order[p];
            Span<float> dst = p == passes - 1 ? y : (p % 2 == 0 ? bufA : bufB);
            long outer = 1, inner = 1;
            for (int i = 0; i < a; i++)
            {
                outer *= cur[i];
            }

            for (int i = a + 1; i < rank; i++)
            {
                inner *= cur[i];
            }

            BuildTaps(o, inDims[a], outDims[a], scales[a], roi.IsEmpty ? 0f : roi[a], roi.IsEmpty ? 1f : roi[a + rank], tapIdx, tapW);
            ApplyAxis(src, dst, (int)outer, inDims[a], outDims[a], (int)inner, taps, tapIdx, tapW, o.Mode == ResizeMode.Nearest);
            cur[a] = outDims[a];
            src = dst;
        }

        if (passes == 0)
        {
            x.Slice(0, (int)outLen).CopyTo(y);
        }

        if (o.Coord == ResizeCoord.TfCropAndResize)
        {
            for (int a = 0; a < rank; a++)
            {
                long outer = 1, inner = 1;
                for (int i = 0; i < a; i++)
                {
                    outer *= outDims[i];
                }

                for (int i = a + 1; i < rank; i++)
                {
                    inner *= outDims[i];
                }

                for (int j = 0; j < outDims[a]; j++)
                {
                    float xo = Coordinate(o, j, inDims[a], outDims[a], scales[a], roi.IsEmpty ? 0f : roi[a], roi.IsEmpty ? 1f : roi[a + rank]);
                    if (xo < 0 || xo > inDims[a] - 1)
                    {
                        for (long q = 0; q < outer; q++)
                        {
                            y.Slice((int)(((q * outDims[a]) + j) * inner), (int)inner).Fill(o.ExtrapolationValue);
                        }
                    }
                }
            }
        }

        if (canonicalNaN)
        {
            for (int i = 0; i < outLen; i++)
            {
                if (float.IsNaN(y[i]))
                {
                    y[i] = BitConverter.UInt32BitsToSingle(0x7FC00000u);
                }
            }
        }
    }

    private const int Shape8 = 8;

    // An axis whose every output element is the same-index input element: nothing to do.
    private static bool IsIdentity(in ResizeOptions o, int inN, int outN, float scale) =>
        inN == outN && scale == 1f && o.Coord != ResizeCoord.TfCropAndResize;

    // The input coordinate of output position j.
    private static float Coordinate(in ResizeOptions o, int j, int inN, int outN, float scale, float roiStart, float roiEnd)
    {
        switch (o.Coord)
        {
            case ResizeCoord.HalfPixel:
                return (((float)j + 0.5f) / scale) - 0.5f;
            case ResizeCoord.PytorchHalfPixel:
                return outN > 1 ? (((float)j + 0.5f) / scale) - 0.5f : 0f;
            case ResizeCoord.AlignCorners:
                return outN == 1 ? 0f : (float)j * (inN - 1) / (outN - 1);
            case ResizeCoord.Asymmetric:
                return (float)j / scale;
            case ResizeCoord.HalfPixelSymmetric:
            {
                float outputWidth = scale * inN;
                float adjustment = outN / outputWidth;
                float center = inN / 2f;
                float offset = center * (1f - adjustment);
                return offset + ((((float)j + 0.5f) / scale) - 0.5f);
            }

            default: // TfCropAndResize
                return outN > 1
                    ? (roiStart * (inN - 1)) + ((float)j * (roiEnd - roiStart) * (inN - 1) / (outN - 1))
                    : 0.5f * (roiStart + roiEnd) * (inN - 1);
        }
    }

    private static void BuildTaps(in ResizeOptions o, int inN, int outN, float scale, float roiStart, float roiEnd, Span<int> idx, Span<float> w)
    {
        int taps = TapsPer(o.Mode);
        for (int j = 0; j < outN; j++)
        {
            float xo = Coordinate(o, j, inN, outN, scale, roiStart, roiEnd);
            int b = j * taps;
            switch (o.Mode)
            {
                case ResizeMode.Nearest:
                {
                    float f = MathF.Floor(xo);
                    int pick = o.Nearest switch
                    {
                        ResizeNearest.Floor => (int)f,
                        ResizeNearest.Ceil => (int)MathF.Ceiling(xo),
                        ResizeNearest.RoundPreferCeil => xo - f >= 0.5f ? (int)f + 1 : (int)f,
                        _ => xo - f > 0.5f ? (int)f + 1 : (int)f,
                    };
                    idx[b] = Math.Clamp(pick, 0, inN - 1);
                    w[b] = 1f;
                    break;
                }

                case ResizeMode.Linear:
                {
                    float c = Math.Clamp(xo, 0f, inN - 1);
                    float f = MathF.Floor(c);
                    float t = c - f;
                    int i0 = (int)f;
                    idx[b] = i0;
                    idx[b + 1] = Math.Min(i0 + 1, inN - 1);
                    w[b] = 1f - t;
                    w[b + 1] = t;
                    break;
                }

                default:
                {
                    float f = MathF.Floor(xo);
                    float t = xo - f;
                    float a = o.CubicA;
                    float u = t + 1f, v = 1f - t, z = 2f - t;
                    Span<float> cw = stackalloc float[4];
                    cw[0] = ((((a * u) - (5f * a)) * u) + (8f * a)) * u - (4f * a);
                    cw[1] = ((((a + 2f) * t) - (a + 3f)) * t * t) + 1f;
                    cw[2] = ((((a + 2f) * v) - (a + 3f)) * v * v) + 1f;
                    cw[3] = ((((a * z) - (5f * a)) * z) + (8f * a)) * z - (4f * a);
                    int i0 = (int)f - 1;
                    float sum = 0f;
                    for (int k = 0; k < 4; k++)
                    {
                        int at = i0 + k;
                        if (o.ExcludeOutside && (at < 0 || at >= inN))
                        {
                            cw[k] = 0f;
                        }

                        sum += cw[k];
                    }

                    for (int k = 0; k < 4; k++)
                    {
                        int at = i0 + k;
                        idx[b + k] = Math.Clamp(at, 0, inN - 1);
                        w[b + k] = o.ExcludeOutside ? cw[k] / sum : cw[k];
                    }

                    break;
                }
            }
        }
    }

    // dst[q, j, i] = sum over taps t of w[j, t] * src[q, idx[j, t], i], taps in increasing order.
    private static void ApplyAxis(ReadOnlySpan<float> src, Span<float> dst, int outer, int inN, int outN, int inner, int taps,
        ReadOnlySpan<int> idx, ReadOnlySpan<float> w, bool copy)
    {
        for (int q = 0; q < outer; q++)
        {
            for (int j = 0; j < outN; j++)
            {
                Span<float> d = dst.Slice(((q * outN) + j) * inner, inner);
                int b = j * taps;
                ReadOnlySpan<float> s0 = src.Slice(((q * inN) + idx[b]) * inner, inner);
                if (copy)
                {
                    s0.CopyTo(d);
                    continue;
                }

                // Taps with weight zero are left out, so an infinity or NaN next to an exactly-hit sample cannot turn it into NaN
                // (0 * inf); ONNX Runtime's results show the same (docs/ops-conv.md).
                int first = 0;
                while (first < taps - 1 && w[b + first] == 0f)
                {
                    first++;
                }

                float w0 = w[b + first];
                ReadOnlySpan<float> sf = src.Slice(((q * inN) + idx[b + first]) * inner, inner);
                for (int i = 0; i < d.Length; i++)
                {
                    d[i] = w0 * sf[i];
                }

                for (int t = first + 1; t < taps; t++)
                {
                    float wt = w[b + t];
                    if (wt == 0f)
                    {
                        continue;
                    }

                    ReadOnlySpan<float> st = src.Slice(((q * inN) + idx[b + t]) * inner, inner);
                    for (int i = 0; i < d.Length; i++)
                    {
                        d[i] += wt * st[i];
                    }
                }
            }
        }
    }
}
