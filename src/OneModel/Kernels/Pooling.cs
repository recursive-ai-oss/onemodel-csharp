using System;
using System.Numerics;
using System.Runtime.Intrinsics;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>Three extents (depth, height, width). Pooling over one or two spatial axes lifts the missing leading axes to extent 1.</summary>
internal readonly record struct Dim3(int D, int H, int W)
{
    /// <summary>D * H * W.</summary>
    public long Volume => (long)D * H * W;
}

/// <summary>
/// The geometry of a pooling window over <c>planes</c> independent volumes (batch x channel). Padding is the amount added before
/// (<c>Pad</c>) the volume on each axis; the window taps at <c>out * Stride - Pad + k * Dilation</c>.
/// </summary>
internal readonly record struct PoolGeom(Dim3 In, Dim3 Kernel, Dim3 Stride, Dim3 Dilation, Dim3 Pad, Dim3 PadEnd, Dim3 Out);

/// <summary>
/// Max and average pooling by direct loops over the window taps in depth, height, width order, one scalar at a time. There is no
/// reassociation anywhere, so the results are the same on every path and in both modes (exact mode only rewrites NaN results as the
/// canonical NaN).
/// </summary>
internal static class Pooling
{
    /// <summary>
    /// Max pooling with IEEE maxNum semantics: NaN taps are ignored unless every tap inside the input is NaN, and then the first of
    /// them is the result. The result is the first tap inside the input that is not NaN; a later tap replaces it only if strictly
    /// greater, so on a tie the first tap wins (it decides the index). ONNX Runtime's own result for windows containing NaN depends on
    /// where the window sits and how wide it is; this is the rule that agrees with it most often (docs/ops-conv.md). A window with no
    /// tap inside the input yields the type's lowest value.
    /// </summary>
    /// <param name="g">Geometry.</param>
    /// <param name="planes">Number of volumes.</param>
    /// <param name="x">Input, <c>planes * In.Volume</c> elements.</param>
    /// <param name="y">Output, <c>planes * Out.Volume</c> elements.</param>
    /// <param name="indices">Receives, for each output, the flat index of the chosen input element over the whole input tensor; empty to skip.</param>
    /// <param name="columnMajor">Flatten each volume column-major (width slowest) for <paramref name="indices"/>.</param>
    public static void Max<T>(in PoolGeom g, long planes, ReadOnlySpan<T> x, Span<T> y, Span<long> indices, bool columnMajor)
        where T : unmanaged, INumber<T>, IMinMaxValue<T>
    {
        long inVol = g.In.Volume, outVol = g.Out.Volume;
        if (x.Length < planes * inVol || y.Length < planes * outVol || (!indices.IsEmpty && indices.Length < planes * outVol))
        {
            throw new ArgumentException("A buffer is smaller than the pooling geometry needs.");
        }

        bool wantIndex = !indices.IsEmpty;
        for (long p = 0; p < planes; p++)
        {
            ReadOnlySpan<T> src = x.Slice((int)(p * inVol), (int)inVol);
            long o = p * outVol;
            for (int od = 0; od < g.Out.D; od++)
            {
                for (int oh = 0; oh < g.Out.H; oh++)
                {
                    for (int ow = 0; ow < g.Out.W; ow++, o++)
                    {
                        T best = T.MinValue;
                        int bestAt = 0;
                        bool any = false, sawNaN = false;
                        int nanAt = 0;
                        T nan = default;
                        for (int kd = 0; kd < g.Kernel.D; kd++)
                        {
                            int d = (od * g.Stride.D) - g.Pad.D + (kd * g.Dilation.D);
                            if ((uint)d >= (uint)g.In.D)
                            {
                                continue;
                            }

                            for (int kh = 0; kh < g.Kernel.H; kh++)
                            {
                                int h = (oh * g.Stride.H) - g.Pad.H + (kh * g.Dilation.H);
                                if ((uint)h >= (uint)g.In.H)
                                {
                                    continue;
                                }

                                for (int kw = 0; kw < g.Kernel.W; kw++)
                                {
                                    int w = (ow * g.Stride.W) - g.Pad.W + (kw * g.Dilation.W);
                                    if ((uint)w >= (uint)g.In.W)
                                    {
                                        continue;
                                    }

                                    int at = (((d * g.In.H) + h) * g.In.W) + w;
                                    T v = src[at];
                                    if (T.IsNaN(v))
                                    {
                                        if (!sawNaN)
                                        {
                                            sawNaN = true;
                                            nan = v;
                                            nanAt = at;
                                        }

                                        continue;
                                    }

                                    if (!any || v > best)
                                    {
                                        best = v;
                                        bestAt = at;
                                        any = true;
                                    }
                                }
                            }
                        }

                        if (!any && sawNaN)
                        {
                            // Every tap was NaN.
                            best = nan;
                            bestAt = nanAt;
                        }

                        y[(int)o] = best;
                        if (wantIndex)
                        {
                            long flat = bestAt;
                            if (columnMajor)
                            {
                                int w = bestAt % g.In.W, h = (bestAt / g.In.W) % g.In.H, d = bestAt / (g.In.W * g.In.H);
                                flat = (((long)w * g.In.H) + h) * g.In.D + d;
                            }

                            indices[(int)o] = (p * inVol) + flat;
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Average pooling: the sum of the window's taps inside the input, in window order from +0, divided by the tap count. With
    /// <paramref name="countIncludePad"/> the count includes taps that fall in the padding (but not beyond it, which only
    /// happens in ceil mode); otherwise only taps inside the input are counted.
    /// </summary>
    public static void Average(in PoolGeom g, long planes, ReadOnlySpan<float> x, Span<float> y, bool countIncludePad)
    {
        long inVol = g.In.Volume, outVol = g.Out.Volume;
        if (x.Length < planes * inVol || y.Length < planes * outVol)
        {
            throw new ArgumentException("A buffer is smaller than the pooling geometry needs.");
        }

        for (long p = 0; p < planes; p++)
        {
            ReadOnlySpan<float> src = x.Slice((int)(p * inVol), (int)inVol);
            long o = p * outVol;
            for (int od = 0; od < g.Out.D; od++)
            {
                for (int oh = 0; oh < g.Out.H; oh++)
                {
                    for (int ow = 0; ow < g.Out.W; ow++, o++)
                    {
                        float sum = 0f;
                        int count = 0;
                        for (int kd = 0; kd < g.Kernel.D; kd++)
                        {
                            int d = (od * g.Stride.D) - g.Pad.D + (kd * g.Dilation.D);
                            bool dIn = (uint)d < (uint)g.In.D;
                            if (!dIn && !(countIncludePad && InPadding(d, g.Pad.D, g.In.D, g.PadEnd.D)))
                            {
                                continue;
                            }

                            for (int kh = 0; kh < g.Kernel.H; kh++)
                            {
                                int h = (oh * g.Stride.H) - g.Pad.H + (kh * g.Dilation.H);
                                bool hIn = (uint)h < (uint)g.In.H;
                                if (!hIn && !(countIncludePad && InPadding(h, g.Pad.H, g.In.H, g.PadEnd.H)))
                                {
                                    continue;
                                }

                                for (int kw = 0; kw < g.Kernel.W; kw++)
                                {
                                    int w = (ow * g.Stride.W) - g.Pad.W + (kw * g.Dilation.W);
                                    if (dIn && hIn && (uint)w < (uint)g.In.W)
                                    {
                                        sum += src[(((d * g.In.H) + h) * g.In.W) + w];
                                        count++;
                                    }
                                    else if (countIncludePad && InPadding(w, g.Pad.W, g.In.W, g.PadEnd.W))
                                    {
                                        count++;
                                    }
                                }
                            }
                        }

                        y[(int)o] = count == 0 ? float.NaN : sum / count;
                    }
                }
            }
        }
    }

    // Whether a tap position lies in the input's padding (before it, or after it up to the padded end), not beyond.
    private static bool InPadding(int pos, int padBegin, int extent, int padEnd) => pos >= -padBegin && pos < extent + padEnd;

    /// <summary>Global average: each plane's sum in index order from +0, divided by the plane size.</summary>
    public static void GlobalAverage(ReadOnlySpan<float> x, Span<float> y, int plane)
    {
        for (int p = 0; p < y.Length; p++)
        {
            float sum = 0f;
            foreach (float v in x.Slice(p * plane, plane))
            {
                sum += v;
            }

            y[p] = sum / plane;
        }
    }

    /// <summary>
    /// Fast-mode global average: each plane is summed in four independent 4-lane accumulators (16 elements per step), which are
    /// combined as (a0 + a1) + (a2 + a3), then lane 0 + 1 + 2 + 3, then the remaining elements one at a time, and divided by the plane
    /// size. The order is fixed (it depends on the plane size only, and is the same on every instruction set), but it is not the
    /// sequential order of <see cref="GlobalAverage"/>, which exact mode keeps.
    /// </summary>
    public static void GlobalAverageFast(ReadOnlySpan<float> x, Span<float> y, int plane)
    {
        for (int p = 0; p < y.Length; p++)
        {
            ReadOnlySpan<float> v = x.Slice(p * plane, plane);
            int i = 0;
            float sum;
            if (plane >= 16)
            {
                Vector128<float> a0 = Vector128<float>.Zero, a1 = a0, a2 = a0, a3 = a0;
                for (; i + 16 <= plane; i += 16)
                {
                    a0 += Vector128.Create(v.Slice(i, 4));
                    a1 += Vector128.Create(v.Slice(i + 4, 4));
                    a2 += Vector128.Create(v.Slice(i + 8, 4));
                    a3 += Vector128.Create(v.Slice(i + 12, 4));
                }

                Vector128<float> a = (a0 + a1) + (a2 + a3);
                sum = ((a.GetElement(0) + a.GetElement(1)) + a.GetElement(2)) + a.GetElement(3);
            }
            else
            {
                sum = 0f;
            }

            for (; i < plane; i++)
            {
                sum += v[i];
            }

            y[p] = sum / plane;
        }
    }

    /// <summary>
    /// Global max: each plane's maximum, starting from the lowest float and replaced by strictly greater values, so NaNs are ignored
    /// and a plane of NaNs or -infinity gives the lowest float (as ONNX Runtime does; see docs/ops-conv.md).
    /// </summary>
    public static void GlobalMax(ReadOnlySpan<float> x, Span<float> y, int plane)
    {
        for (int p = 0; p < y.Length; p++)
        {
            float best = float.MinValue;
            foreach (float v in x.Slice(p * plane, plane))
            {
                if (v > best)
                {
                    best = v;
                }
            }

            y[p] = best;
        }
    }

    /// <summary>Rewrites every NaN as the canonical quiet NaN 0x7FC00000 (exact mode).</summary>
    public static void CanonicalizeNaN(Span<float> y)
    {
        for (int i = 0; i < y.Length; i++)
        {
            if (float.IsNaN(y[i]))
            {
                y[i] = BitConverter.UInt32BitsToSingle(0x7FC00000u);
            }
        }
    }
}
