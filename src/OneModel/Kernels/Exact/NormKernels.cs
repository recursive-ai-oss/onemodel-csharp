using System;
using System.Runtime.Intrinsics;

namespace RecursiveAI.OneModel.Kernels.Exact;

// Row-wise normalisation kernels (LayerNormalization, RMSNormalization, InstanceNormalization, GroupNormalization) and the log-softmax
// row kernel. docs/ops-norms.md is the normative text.
//
// Every operation is a single IEEE add, subtract, multiply, divide or square root, and sums use the canonical 16-partial-sum order of
// RowReduce.Sum16, so the fast and the exact flavour give the same bits here, at every lane width, with and without hardware
// intrinsics. (Fast LogSoftmax uses the fast exponential, so it differs from exact LogSoftmax by the exponential's error.)

/// <summary>Maps the session's kernel path to the lane width of the row kernels.</summary>
internal static class RowIsa
{
    public static VecIsa Of(KernelPath path) => path switch
    {
        KernelPath.Avx512 => VecIsa.V512,
        KernelPath.Avx2 => VecIsa.V256,
        KernelPath.Neon or KernelPath.Portable128 => VecIsa.V128,
        _ => VecIsa.Scalar,
    };
}

/// <summary>Row kernels that normalise each row of a matrix to zero mean and unit variance (or unit root mean square).</summary>
internal static class NormKernels
{
    private const uint QuietNaNBits = 0x7FC00000u;

    /// <summary>
    /// Normalises rows <c>[row0, row1)</c> of <paramref name="x"/> (row length <paramref name="cols"/>) into <paramref name="y"/>:
    /// <c>mean = Sum16(x) / n</c> (zero when <paramref name="rms"/>), <c>v = Sum16((x - mean)^2) / n</c>,
    /// <c>inv = 1 / sqrt(v + eps)</c>, <c>y = (((x - mean) * inv) * scale) + bias</c> as separate operations. The scale and bias are
    /// either per element (<paramref name="scale"/>, <paramref name="bias"/>, length <paramref name="cols"/>) or, when those are
    /// empty, the scalars <paramref name="scale0"/> and <paramref name="bias0"/> (the bias is added only when
    /// <paramref name="hasBias"/>). <paramref name="x"/> and <paramref name="y"/> may be the same memory. A non-empty
    /// <paramref name="rowMean"/> receives each row's mean at index <c>row</c>, and <paramref name="rowInv"/> its factor <c>inv</c>.
    /// </summary>
    public static void Rows(VecIsa isa, ReadOnlySpan<float> x, Span<float> y, int row0, int row1, int cols, bool rms,
        float eps, ReadOnlySpan<float> scale, float scale0, ReadOnlySpan<float> bias, bool hasBias, float bias0, Span<float> rowMean,
        Span<float> rowInv)
    {
        if (cols <= 0 || row1 <= row0)
        {
            return;
        }

        if (row0 < 0 || x.Length < (long)row1 * cols || y.Length < (long)row1 * cols ||
            (!scale.IsEmpty && scale.Length < cols) || (!bias.IsEmpty && bias.Length < cols) ||
            (!rowMean.IsEmpty && rowMean.Length < row1) || (!rowInv.IsEmpty && rowInv.Length < row1))
        {
            throw new ArgumentException("A span is shorter than the rows and columns require.");
        }

        isa = VecIsaInfo.Resolve(isa);
        switch (isa)
        {
            case VecIsa.V512: Run<Vector512<float>, ExactVec512>(x, y, row0, row1, cols, rms, eps, scale, scale0, bias, hasBias, bias0, rowMean, rowInv); break;
            case VecIsa.V256: Run<Vector256<float>, ExactVec256>(x, y, row0, row1, cols, rms, eps, scale, scale0, bias, hasBias, bias0, rowMean, rowInv); break;
            case VecIsa.V128: Run<Vector128<float>, ExactVec128>(x, y, row0, row1, cols, rms, eps, scale, scale0, bias, hasBias, bias0, rowMean, rowInv); break;
            default: Run<float, ExactVecScalar>(x, y, row0, row1, cols, rms, eps, scale, scale0, bias, hasBias, bias0, rowMean, rowInv); break;
        }
    }

    // Bounds: Rows() checked that x and y hold row1 * cols elements and the per-element spans hold cols; every access is inside them.
    private static void Run<TV, TO>(ReadOnlySpan<float> x, Span<float> y, int row0, int row1, int cols, bool rms, float eps,
        ReadOnlySpan<float> scale, float scale0, ReadOnlySpan<float> bias, bool hasBias, float bias0, Span<float> rowMean,
        Span<float> rowInv)
        where TV : struct
        where TO : IVecOps<TV>
    {
        int lanes = TO.Lanes;
        int body = cols - (cols % lanes);
        float n = cols;
        float nan = BitConverter.UInt32BitsToSingle(QuietNaNBits);
        bool perElementScale = !scale.IsEmpty, perElementBias = !bias.IsEmpty;
        TV vs0 = TO.Splat(scale0), vb0 = TO.Splat(bias0);

        for (int r = row0; r < row1; r++)
        {
            ReadOnlySpan<float> xr = x.Slice(r * cols, cols);
            Span<float> yr = y.Slice(r * cols, cols);

            float mean = rms ? 0f : RowReduce.Sum16<TV, TO>(xr) / n;
            float ssq = SumSquaredDeviations<TV, TO>(xr, mean);
            float inv = 1f / float.Sqrt((ssq / n) + eps);
            if (!rowMean.IsEmpty)
            {
                rowMean[r] = mean;
            }

            if (!rowInv.IsEmpty)
            {
                rowInv[r] = inv;
            }

            TV vm = TO.Splat(mean), vi = TO.Splat(inv);
            for (int j = 0; j < body; j += lanes)
            {
                TV t = TO.Mul(TO.Mul(TO.Sub(TO.Load(xr, j), vm), vi), perElementScale ? TO.Load(scale, j) : vs0);
                if (perElementBias)
                {
                    t = TO.Add(t, TO.Load(bias, j));
                }
                else if (hasBias)
                {
                    t = TO.Add(t, vb0);
                }

                TO.Store(yr, j, Transcendental.Canonical<TV, TO>(t));
            }

            for (int j = body; j < cols; j++)
            {
                float t = ((xr[j] - mean) * inv) * (perElementScale ? scale[j] : scale0);
                if (perElementBias)
                {
                    t += bias[j];
                }
                else if (hasBias)
                {
                    t += bias0;
                }

                yr[j] = float.IsNaN(t) ? nan : t;
            }
        }
    }

    /// <summary>Sum of (x[j] - mean)^2 in the canonical 16-partial-sum order of <see cref="RowReduce.Sum16"/>.</summary>
    private static float SumSquaredDeviations<TV, TO>(ReadOnlySpan<float> x, float mean)
        where TV : struct
        where TO : IVecOps<TV>
    {
        Span<float> part = stackalloc float[16];
        part.Clear();
        int lanes = TO.Lanes;
        int full = x.Length & ~15;
        TV vm = TO.Splat(mean);
        for (int c = 0; c < full; c += 16)
        {
            for (int t = 0; t < 16; t += lanes)
            {
                TV d = TO.Sub(TO.Load(x, c + t), vm);
                TO.Store(part, t, TO.Add(TO.Load(part, t), TO.Mul(d, d)));
            }
        }

        for (int j = full; j < x.Length; j++)
        {
            float d = x[j] - mean;
            part[j & 15] += d * d;
        }

        return Combine16(part);
    }

    /// <summary>The halving combine of <see cref="RowReduce.Sum16"/>: p[t] + p[t+8], then t+4, t+2, t+1.</summary>
    internal static float Combine16(ReadOnlySpan<float> part)
    {
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
        return (b0 + b2) + (b1 + b3);
    }

    /// <summary>
    /// Log-softmax of rows <c>[row0, row1)</c>: <c>d = x - max</c>, <c>l = Sum16(exp(d))</c>, <c>y = d - log(l)</c>. Rows holding NaN,
    /// +inf, or only -inf give NaN in every output of the row. <paramref name="x"/> and <paramref name="y"/> may be the same memory.
    /// </summary>
    public static void LogSoftmaxRows(bool exact, VecIsa isa, ReadOnlySpan<float> x, Span<float> y, int row0, int row1, int cols)
    {
        if (cols <= 0 || row1 <= row0)
        {
            return;
        }

        if (row0 < 0 || x.Length < (long)row1 * cols || y.Length < (long)row1 * cols)
        {
            throw new ArgumentException("A span is shorter than the rows and columns require.");
        }

        isa = VecIsaInfo.Resolve(isa);
        if (exact)
        {
            switch (isa)
            {
                case VecIsa.V512: LogSoftmax<Vector512<float>, ExactVec512, ExactVecScalar>(x, y, row0, row1, cols); break;
                case VecIsa.V256: LogSoftmax<Vector256<float>, ExactVec256, ExactVecScalar>(x, y, row0, row1, cols); break;
                case VecIsa.V128: LogSoftmax<Vector128<float>, ExactVec128, ExactVecScalar>(x, y, row0, row1, cols); break;
                default: LogSoftmax<float, ExactVecScalar, ExactVecScalar>(x, y, row0, row1, cols); break;
            }
        }
        else
        {
            switch (isa)
            {
                case VecIsa.V512: LogSoftmax<Vector512<float>, FastVec512, FastVecScalar>(x, y, row0, row1, cols); break;
                case VecIsa.V256: LogSoftmax<Vector256<float>, FastVec256, FastVecScalar>(x, y, row0, row1, cols); break;
                case VecIsa.V128: LogSoftmax<Vector128<float>, FastVec128, FastVecScalar>(x, y, row0, row1, cols); break;
                default: LogSoftmax<float, FastVecScalar, FastVecScalar>(x, y, row0, row1, cols); break;
            }
        }
    }

    // TS is the one-lane flavour used for the scalar log of the row sum, so a vector log is not run for one value.
    private static void LogSoftmax<TV, TO, TS>(ReadOnlySpan<float> x, Span<float> y, int row0, int row1, int cols)
        where TV : struct
        where TO : IVecOps<TV>
        where TS : IVecOps<float>
    {
        int lanes = TO.Lanes;
        int body = cols - (cols % lanes);
        Span<float> part = stackalloc float[16];
        Span<float> pad = stackalloc float[16];
        float nan = BitConverter.UInt32BitsToSingle(QuietNaNBits);

        for (int r = row0; r < row1; r++)
        {
            ReadOnlySpan<float> xr = x.Slice(r * cols, cols);
            Span<float> yr = y.Slice(r * cols, cols);
            float max = RowReduce.Max<TV, TO>(xr);
            TV vmax = TO.Splat(max);

            // d = x - max goes to y first (in place is safe: each element is read once, then written). The exponentials are summed
            // on the fly in the canonical order, 16 elements at a time.
            for (int j = 0; j < body; j += lanes)
            {
                TO.Store(yr, j, TO.Sub(TO.Load(xr, j), vmax));
            }

            for (int j = body; j < cols; j++)
            {
                yr[j] = xr[j] - max;
            }

            part.Clear();
            int full = cols & ~15;
            for (int c = 0; c < full; c += 16)
            {
                for (int t = 0; t < 16; t += lanes)
                {
                    TO.Store(part, t, TO.Add(TO.Load(part, t), TO.SoftmaxExp(TO.Load(yr, c + t))));
                }
            }

            if (full < cols)
            {
                // The last partial block runs through a zero-padded copy so it uses the same vector exponential; only its real
                // elements are added, each to the partial it belongs to.
                int rest = cols - full;
                pad.Clear();
                yr.Slice(full).CopyTo(pad);
                for (int t = 0; t < 16; t += lanes)
                {
                    TO.Store(pad, t, TO.SoftmaxExp(TO.Load(pad, t)));
                }

                for (int j = 0; j < rest; j++)
                {
                    part[j] += pad[j];
                }
            }

            float sum = Combine16(part);
            float logSum = Transcendental.Log<float, TS>(sum);
            if (float.IsNaN(logSum) || float.IsNaN(sum))
            {
                yr.Fill(nan);
                continue;
            }

            TV vl = TO.Splat(logSum);
            for (int j = 0; j < body; j += lanes)
            {
                TO.Store(yr, j, Transcendental.Canonical<TV, TO>(TO.Sub(TO.Load(yr, j), vl)));
            }

            for (int j = body; j < cols; j++)
            {
                float v = yr[j] - logSum;
                yr[j] = float.IsNaN(v) ? nan : v;
            }
        }
    }
}
