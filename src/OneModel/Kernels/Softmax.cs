using System;
using System.Runtime.Intrinsics;
using RecursiveAI.OneModel.Kernels.Exact;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Row-wise softmax over the last axis, fast and exact. For a row x of length n: m = max(x); e[j] = exp(x[j] - m);
/// l = sum of e in the canonical order of <see cref="RowReduce.Sum16"/>; y[j] = e[j] / l. The exact flavour is bit-identical
/// at every lane width; the fast flavour uses the fast exponential and multiplies by 1/l. Rows containing NaN or +inf, and rows
/// that are entirely -inf, produce NaN (the canonical quiet NaN) in every output of the row.
/// </summary>
internal static class Softmax
{
    internal const uint QuietNaNBits = 0x7FC00000u;

    /// <summary>Softmax of each of <paramref name="rows"/> rows of <paramref name="cols"/> elements.</summary>
    /// <param name="exact">Selects the exact (bit-reproducible) flavour.</param>
    /// <param name="isa">The lane width; <see cref="VecIsa.Auto"/> picks the widest hardware width.</param>
    /// <param name="src">Input; row r starts at <c>r * srcStride</c>.</param>
    /// <param name="dst">Output; row r starts at <c>r * dstStride</c>. May alias <paramref name="src"/> exactly (same offsets and strides).</param>
    /// <param name="rows">Number of rows (0 or more).</param>
    /// <param name="cols">Elements per row (0 or more).</param>
    /// <param name="srcStride">Elements between row starts of <paramref name="src"/>, at least <paramref name="cols"/>.</param>
    /// <param name="dstStride">Elements between row starts of <paramref name="dst"/>, at least <paramref name="cols"/>.</param>
    /// <exception cref="ArgumentException">A span is too short for the stated shape, or a stride is smaller than <paramref name="cols"/>.</exception>
    public static void Rows(bool exact, VecIsa isa, ReadOnlySpan<float> src, Span<float> dst, int rows, int cols, int srcStride, int dstStride)
    {
        if (rows < 0 || cols < 0 || (rows > 1 && (srcStride < cols || dstStride < cols)))
        {
            throw new ArgumentException("Negative size or a row stride smaller than the row length.");
        }

        if (rows == 0 || cols == 0)
        {
            return;
        }

        long srcNeed = ((long)(rows - 1) * srcStride) + cols;
        long dstNeed = ((long)(rows - 1) * dstStride) + cols;
        if (src.Length < srcNeed || dst.Length < dstNeed)
        {
            throw new ArgumentException("A span is shorter than rows, cols and the strides require.");
        }

        isa = VecIsaInfo.Resolve(isa);
        if (exact)
        {
            switch (isa)
            {
                case VecIsa.V512: Run<Vector512<float>, ExactVec512>(src, dst, rows, cols, srcStride, dstStride); break;
                case VecIsa.V256: Run<Vector256<float>, ExactVec256>(src, dst, rows, cols, srcStride, dstStride); break;
                case VecIsa.V128: Run<Vector128<float>, ExactVec128>(src, dst, rows, cols, srcStride, dstStride); break;
                default: Run<float, ExactVecScalar>(src, dst, rows, cols, srcStride, dstStride); break;
            }
        }
        else
        {
            switch (isa)
            {
                case VecIsa.V512: Run<Vector512<float>, FastVec512>(src, dst, rows, cols, srcStride, dstStride); break;
                case VecIsa.V256: Run<Vector256<float>, FastVec256>(src, dst, rows, cols, srcStride, dstStride); break;
                case VecIsa.V128: Run<Vector128<float>, FastVec128>(src, dst, rows, cols, srcStride, dstStride); break;
                default: Run<float, FastVecScalar>(src, dst, rows, cols, srcStride, dstStride); break;
            }
        }
    }

    // Bounds: Rows() has checked that src and dst hold (rows-1)*stride + cols elements; each vector access below is inside a row.
    private static void Run<TV, TO>(ReadOnlySpan<float> src, Span<float> dst, int rows, int cols, int srcStride, int dstStride)
        where TV : struct
        where TO : IVecOps<TV>
    {
        int lanes = TO.Lanes;
        Span<float> tail = stackalloc float[16];
        int body = cols - (cols % lanes);
        float nan = BitConverter.UInt32BitsToSingle(QuietNaNBits);

        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<float> x = src.Slice(r * srcStride, cols);
            Span<float> y = dst.Slice(r * dstStride, cols);

            float max = RowReduce.Max<TV, TO>(x);
            TV vmax = TO.Splat(max);

            // e[j] = exp(x[j] - max); the tail (cols % lanes elements) goes through a padded chunk and runs the same vector code.
            for (int j = 0; j < body; j += lanes)
            {
                TO.Store(y, j, TO.SoftmaxExp(TO.Sub(TO.Load(x, j), vmax)));
            }

            if (body < cols)
            {
                tail.Clear();
                x.Slice(body).CopyTo(tail);
                TO.Store(tail, 0, TO.SoftmaxExp(TO.Sub(TO.Load(tail, 0), vmax)));
                tail.Slice(0, cols - body).CopyTo(y.Slice(body));
            }

            float sum = RowReduce.Sum16<TV, TO>(y);

            if (TO.Fused)
            {
                float inv = 1f / sum;
                TV vinv = TO.Splat(inv);
                for (int j = 0; j < body; j += lanes)
                {
                    TO.Store(y, j, Transcendental.Canonical<TV, TO>(TO.Mul(TO.Load(y, j), vinv)));
                }

                for (int j = body; j < cols; j++)
                {
                    float v = y[j] * inv;
                    y[j] = float.IsNaN(v) ? nan : v;
                }
            }
            else
            {
                TV vsum = TO.Splat(sum);
                for (int j = 0; j < body; j += lanes)
                {
                    TO.Store(y, j, Transcendental.Canonical<TV, TO>(TO.Div(TO.Load(y, j), vsum)));
                }

                for (int j = body; j < cols; j++)
                {
                    float v = y[j] / sum;
                    y[j] = float.IsNaN(v) ? nan : v;
                }
            }
        }
    }
}

/// <summary>Row reductions with a fixed, lane-width-independent order, shared by softmax and attention.</summary>
internal static class RowReduce
{
    /// <summary>
    /// Sum in the canonical order: 16 partial sums, element j added to partial j mod 16 in increasing j, then combined by halving
    /// (p[t] + p[t+8], then t+4, t+2, t+1). Every lane width reproduces it, because lanes are independent partials.
    /// </summary>
    public static float Sum16<TV, TO>(ReadOnlySpan<float> x)
        where TV : struct
        where TO : IVecOps<TV>
    {
        Span<float> part = stackalloc float[16];
        part.Clear();
        int lanes = TO.Lanes;
        int full = x.Length & ~15;
        for (int c = 0; c < full; c += 16)
        {
            for (int t = 0; t < 16; t += lanes)
            {
                TO.Store(part, t, TO.Add(TO.Load(part, t), TO.Load(x, c + t)));
            }
        }

        for (int j = full; j < x.Length; j++)
        {
            part[j & 15] += x[j];
        }

        float a0 = part[0] + part[8];
        float a1 = part[1] + part[9];
        float a2 = part[2] + part[10];
        float a3 = part[3] + part[11];
        float a4 = part[4] + part[12];
        float a5 = part[5] + part[13];
        float a6 = part[6] + part[14];
        float a7 = part[7] + part[15];
        float b0 = a0 + a4;
        float b1 = a1 + a5;
        float b2 = a2 + a6;
        float b3 = a3 + a7;
        float c0 = b0 + b2;
        float c1 = b1 + b3;
        return c0 + c1;
    }

    /// <summary>
    /// Maximum of the non-NaN elements (-inf if none). The result does not depend on the order or lane width, since max is exact.
    /// </summary>
    public static float Max<TV, TO>(ReadOnlySpan<float> x)
        where TV : struct
        where TO : IVecOps<TV>
    {
        int lanes = TO.Lanes;
        int body = x.Length - (x.Length % lanes);
        TV m = TO.Splat(float.NegativeInfinity);
        for (int j = 0; j < body; j += lanes)
        {
            TV v = TO.Load(x, j);
            m = TO.Select(TO.GreaterThan(v, m), v, m);
        }

        Span<float> buf = stackalloc float[16];
        TO.Store(buf, 0, m);
        float max = float.NegativeInfinity;
        for (int t = 0; t < lanes; t++)
        {
            max = buf[t] > max ? buf[t] : max;
        }

        for (int j = body; j < x.Length; j++)
        {
            float v = x[j];
            max = v > max ? v : max;
        }

        return max;
    }
}
