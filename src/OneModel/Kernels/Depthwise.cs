using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using RecursiveAI.OneModel.Kernels.Exact;
using RecursiveAI.OneModel.Kernels.Fast;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Direct depthwise convolution (one input channel per group, optionally several output channels per input channel).
/// </summary>
/// <remarks>
/// <para>
/// The matrix-product lowering is all overhead here (k = KH*KW), so each output plane is computed straight from its input plane.
/// Lanes are neighbouring output columns of one row, so a tap is one contiguous load and one fused or unfused multiply-add per
/// lane. Each output is the sum over taps in row-major order of weight * input, accumulated one tap at a time from +0; every lane is
/// an independent output, so the lane count never changes a result and every path, the scalar twin included, gives the same bits
/// (exact mode: unfused multiply then add; fast mode: one fused operation per tap).
/// </para>
/// <para>
/// To make every load contiguous whatever the stride, the zero-padded input plane is split by column phase into StrideW planes
/// (plane p holds padded columns p, p + StrideW, ...): tap column ox * StrideW + dx then sits at column ox + dx / StrideW of phase
/// plane dx % StrideW. A stride-1 unpadded plane is read in place. The last vector of a row is moved back to end at the last column,
/// so it recomputes a few outputs with identical results instead of needing a scalar tail. Rows narrower than four vectors are
/// processed four rows at a time so that a tap's weight is still shared by four multiply-adds.
/// </para>
/// <para>docs/kernels-depthwise.md is the normative text.</para>
/// </remarks>
internal static unsafe class Depthwise
{
    private const int MaxTaps = 64;
    private const int MaxPhases = 8;

    // Floats of staging for Flat: its vectors may run up to one width past the last output.
    private const int FlatStage = 512;

    // Output elements per chunk of planes; the planes per chunk follow from the plane size alone (never the thread count).
    private const int ElementsPerChunk = 4096;
    private const int MaxPlanesPerChunk = 32;

    /// <summary>
    /// Computes the depthwise convolution described by <paramref name="s"/> (Groups == Cin) from <paramref name="input"/>
    /// (N x Cin x H x W) and <paramref name="weights"/> (Cout x 1 x KH x KW) into <paramref name="output"/> (N x Cout x OutH x OutW).
    /// </summary>
    /// <remarks>
    /// Bounds: the caller (<see cref="Conv.Run"/>) has validated the shape and that the three buffers cover the extents above. Every
    /// load is at most the last element of the padded plane, as worked out in the comments of <c>Job.RunChunk</c>.
    /// </remarks>
    public static void Run(
        KernelMode mode, KernelPath path, in ConvShape s, float* input, float* weights, float* output, int threads, float* bias = null,
        FusedActivation? activation = null)
    {
        int planes = s.N * s.Cout;
        int outPlane = s.OutH * s.OutW;
        if (mode == KernelMode.Exact)
        {
            // Exact mode keeps its separate bias and activation passes (the caller's), so its bits are those of the stand-alone operators.
            bias = null;
            activation = null;
        }

        int perChunk = Math.Clamp(ElementsPerChunk / Math.Max(1, outPlane), 1, MaxPlanesPerChunk);
        int multiplier = s.Cout / s.Cin;
        if (multiplier > 1)
        {
            // Output channels of one input channel read the same padded copy, which a chunk builds once. Up to four per chunk keeps
            // enough chunks for the pool; the choice depends on the shape alone.
            perChunk = Math.Max(perChunk, multiplier % 4 == 0 ? 4 : multiplier % 2 == 0 ? 2 : 1);
        }

        int chunks = (planes + perChunk - 1) / perChunk;
        bool exact = mode == KernelMode.Exact;
        ChunkJob job = (exact, path) switch
        {
            (true, KernelPath.Avx512) => Job<ExactLanes16, ExactLanes8, ExactLanes4, ExactLanes1>.Get(s, input, weights, output, true, perChunk, bias, activation, path),
            (true, KernelPath.Avx2) => Job<ExactLanes8, ExactLanes4, ExactLanes4, ExactLanes1>.Get(s, input, weights, output, true, perChunk, bias, activation, path),
            (true, KernelPath.Neon or KernelPath.Portable128) => Job<ExactLanes4, ExactLanes4, ExactLanes4, ExactLanes1>.Get(s, input, weights, output, true, perChunk, bias, activation, path),
            (true, _) => Job<ExactLanes1, ExactLanes1, ExactLanes1, ExactLanes1>.Get(s, input, weights, output, true, perChunk, bias, activation, path),
            (false, KernelPath.Avx512) => Job<FastLanes16, FastLanes8, FastLanes4, FastLanes1>.Get(s, input, weights, output, false, perChunk, bias, activation, path),
            (false, KernelPath.Avx2) => Job<FastLanes8, FastLanes4, FastLanes4, FastLanes1>.Get(s, input, weights, output, false, perChunk, bias, activation, path),
            (false, KernelPath.Neon or KernelPath.Portable128) => Job<FastLanes4, FastLanes4, FastLanes4, FastLanes1>.Get(s, input, weights, output, false, perChunk, bias, activation, path),
            _ => Job<FastLanes1, FastLanes1, FastLanes1, FastLanes1>.Get(s, input, weights, output, false, perChunk, bias, activation, path),
        };

        // Nanoseconds on one thread, roughly: the taps at several outputs per nanosecond, plus building the padded plane. Four times
        // that is what the pool is told: the layers are small, and a spinning pool wins them back with fewer than 80 microseconds per
        // thread (MobileNetV3-large's depthwise layers: 1.9 ms to 1.5 ms on 8 threads).
        long work = (long)planes * (((long)outPlane * s.KH * s.KW / 8) + (((long)(s.H + s.PadTop + s.PadBottom) * (s.W + s.PadLeft + s.PadRight)) / 2));
        ParallelFor.Run(job, chunks, threads, work * 4);
    }

    /// <summary>
    /// <typeparamref name="T0"/>, <typeparamref name="T1"/> and <typeparamref name="T2"/> are lane types of decreasing width (equal
    /// widths are fine); a plane uses the widest one that fits its row. <typeparamref name="TS"/> is the one-lane twin.
    /// </summary>
    private sealed class Job<T0, T1, T2, TS> : ChunkJob
        where T0 : struct, ILanes<T0>
        where T1 : struct, ILanes<T1>
        where T2 : struct, ILanes<T2>
        where TS : struct, ILanes<TS>
    {
        [ThreadStatic]
        private static Job<T0, T1, T2, TS>? t_job;

        private ConvShape _s;
        private float* _input;
        private float* _weights;
        private float* _output;
        private bool _canonicalNaN;
        private int _planesPerChunk;
        private float* _bias;
        private FusedActivation? _activation;
        private KernelPath _path;

        /// <summary>The calling thread's job object, refilled: a steady-state call allocates nothing.</summary>
        public static Job<T0, T1, T2, TS> Get(
            in ConvShape s, float* input, float* weights, float* output, bool canonicalNaN, int planesPerChunk, float* bias,
            FusedActivation? activation, KernelPath path)
        {
            Job<T0, T1, T2, TS> job = t_job ??= new Job<T0, T1, T2, TS>();
            job._s = s;
            job._input = input;
            job._weights = weights;
            job._output = output;
            job._canonicalNaN = canonicalNaN;
            job._planesPerChunk = planesPerChunk;
            job._bias = bias;
            job._activation = activation;
            job._path = path;
            return job;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public override void RunChunk(int chunk)
        {
            ConvShape s = _s;
            int mult = s.Cout / s.Cin;
            int outH = s.OutH;
            int outW = s.OutW;
            int planes = s.N * s.Cout;
            int taps = s.KH * s.KW;
            int hp = s.H + s.PadTop + s.PadBottom;
            int wp = s.W + s.PadLeft + s.PadRight;
            int phases = s.StrideW;
            bool simple = taps <= MaxTaps && phases <= MaxPhases;
            bool padded = s.PadTop != 0 || s.PadLeft != 0 || s.PadBottom != 0 || s.PadRight != 0;
            bool copy = padded || phases > 1;

            // Phase plane geometry: pitch is the row length of one phase plane, phaseSize its element count.
            int pitch = copy ? (wp + phases - 1) / phases : s.W;
            long phaseSize = (long)hp * pitch;
            float* scratch = simple && copy ? Scratch.Get(Scratch.Plane, (phaseSize * phases) + 64) : null;

            // Tap table: offset of each tap from the first element of the plane (its phase plane included), in row-major tap order.
            int* tapOff = stackalloc int[MaxTaps];
            if (simple)
            {
                for (int ky = 0; ky < s.KH; ky++)
                {
                    for (int kx = 0; kx < s.KW; kx++)
                    {
                        int dx = kx * s.DilW;
                        tapOff[(ky * s.KW) + kx] = (int)(((dx % phases) * phaseSize) + ((long)ky * s.DilH * pitch) + (dx / phases));
                    }
                }
            }

            // Narrow planes, unit column stride, padded copy in scratch: see Flat. The result is staged in tmp (at most 512 floats).
            bool flat = simple && copy && phases == 1 && outW <= T0.Width && ((long)outH * pitch) + T0.Width <= FlatStage;
            float* tmp = stackalloc float[FlatStage];

            int firstPlane = chunk * _planesPerChunk;
            int endPlane = Math.Min(planes, firstPlane + _planesPerChunk);
            // Image and output channel of the plane, counted up rather than divided out for every plane; the padded copy is rebuilt only
            // when the input channel changes (output channels of one input channel share it).
            int img = firstPlane / s.Cout;
            int oc = firstPlane % s.Cout;
            long builtFor = -1;
            for (int q = firstPlane; q < endPlane; q++)
            {
                int ic = mult == 1 ? oc : oc / mult;
                float* plane = _input + ((((long)img * s.Cin) + ic) * s.H * s.W);
                float* w = _weights + ((long)oc * taps);
                float* dst = _output + ((long)q * outH * outW);
                int ocNow = oc, imgNow = img;
                if (++oc == s.Cout)
                {
                    oc = 0;
                    img++;
                }

                if (!simple)
                {
                    Reference(plane, w, dst, in s);
                }
                else
                {
                    float* src = plane;
                    if (copy)
                    {
                        long key = ((long)imgNow * s.Cin) + ic;
                        if (key != builtFor)
                        {
                            BuildPhases(plane, scratch, in s, phases, hp, pitch, phaseSize);
                            builtFor = key;
                        }


                        src = scratch;
                    }

                    long rowStride = (long)s.StrideH * pitch;
                    if (flat)
                    {
                        Flat<T0>(src, tapOff, w, taps, dst, tmp, outH, outW, pitch);
                    }
                    else if (T0.Width <= outW)
                    {
                        Compute<T0>(src, rowStride, tapOff, w, taps, dst, outH, outW);
                    }
                    else if (T1.Width <= outW)
                    {
                        Compute<T1>(src, rowStride, tapOff, w, taps, dst, outH, outW);
                    }
                    else if (T2.Width <= outW)
                    {
                        Compute<T2>(src, rowStride, tapOff, w, taps, dst, outH, outW);
                    }
                    else
                    {
                        Compute<TS>(src, rowStride, tapOff, w, taps, dst, outH, outW);
                    }
                }

                // Bias and activation while the plane is still in this core's cache (fast mode only): a separate pass over planes
                // written by other cores costs more than the convolution itself on small layers. One rounded add, then the function,
                // exactly what the stand-alone passes do.
                if (_bias != null || _activation != null)
                {
                    int count = outH * outW;
                    if (_bias != null && _activation is null or ReluActivation)
                    {
                        // The usual case in one pass: one rounded add, then Relu (see RowEpilogue).
                        if (_activation is null)
                        {
                            RowEpilogue.Add(dst, count, _bias[ocNow]);
                        }
                        else
                        {
                            RowEpilogue.BiasRelu(dst, count, _bias[ocNow]);
                        }
                    }
                    else
                    {
                        var row = new Span<float>(dst, count);
                        if (_bias != null)
                        {
                            BiasActivationEpilogue.AddScalar(row, _bias[ocNow], false);
                        }

                        _activation?.Apply(_path, false, row);
                    }
                }

                if (_canonicalNaN)
                {
                    uint* bits = (uint*)dst;
                    for (int i = 0; i < outH * outW; i++)
                    {
                        if ((bits[i] & 0x7FFFFFFFu) > 0x7F800000u)
                        {
                            bits[i] = 0x7FC00000u;
                        }
                    }
                }
            }
        }

        // Rows of a few dozen floats: a vector loop is cheaper than the library calls for the same work, which add up over hundreds of
        // small planes. Bounds: the callers pass ranges inside the scratch plane, the staging buffer or the output plane.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CopyFloats(float* src, float* dst, int count)
        {
            int i = 0;
            if (Vector256.IsHardwareAccelerated)
            {
                for (; i + 8 <= count; i += 8)
                {
                    Vector256.Store(Vector256.Load(src + i), dst + i);
                }
            }

            for (; i + 4 <= count; i += 4)
            {
                Vector128.Store(Vector128.Load(src + i), dst + i);
            }

            for (; i < count; i++)
            {
                dst[i] = src[i];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ZeroFill(float* dst, int count)
        {
            if (count >= 512)
            {
                new Span<float>(dst, count).Clear();
                return;
            }

            int i = 0;
            if (Vector256.IsHardwareAccelerated)
            {
                for (; i + 8 <= count; i += 8)
                {
                    Vector256.Store(Vector256<float>.Zero, dst + i);
                }
            }

            for (; i + 4 <= count; i += 4)
            {
                Vector128.Store(Vector128<float>.Zero, dst + i);
            }

            for (; i < count; i++)
            {
                dst[i] = 0f;
            }
        }

        /// <summary>
        /// Zero-fills the phase planes and scatters the input plane into them: padded column c goes to phase c % phases, column
        /// c / phases. With one phase this is a row-by-row copy into the padded plane.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void BuildPhases(float* plane, float* scratch, in ConvShape s, int phases, int hp, int pitch, long phaseSize)
        {
            if (phases == 1)
            {
                // Only the border needs clearing: the rows below overwrite the interior.
                long top = (long)s.PadTop * pitch;
                ZeroFill(scratch, checked((int)top));
                int right = pitch - s.PadLeft - s.W;
                for (int y = 0; y < s.H; y++)
                {
                    float* rowBase = scratch + ((long)(y + s.PadTop) * pitch);
                    ZeroFill(rowBase, s.PadLeft);
                    CopyFloats(plane + ((long)y * s.W), rowBase + s.PadLeft, s.W);
                    ZeroFill(rowBase + s.PadLeft + s.W, right);
                }

                float* bottom = scratch + ((long)(s.PadTop + s.H) * pitch);
                ZeroFill(bottom, checked((int)(phaseSize - ((long)(s.PadTop + s.H) * pitch))));
                return;
            }

            ZeroFill(scratch, checked((int)(phaseSize * phases)));
            for (int y = 0; y < s.H; y++)
            {
                float* srcRow = plane + ((long)y * s.W);
                if (phases == 1)
                {
                    CopyFloats(srcRow, scratch + ((long)(y + s.PadTop) * pitch) + s.PadLeft, s.W);
                    continue;
                }

                float* rowBase = scratch + ((long)(y + s.PadTop) * pitch);
                if (phases == 2)
                {
                    SplitEvenOdd(srcRow, s.W, rowBase, phaseSize, s.PadLeft);
                    continue;
                }

                int phase = s.PadLeft % phases;
                int col = s.PadLeft / phases;
                for (int x = 0; x < s.W; x++)
                {
                    rowBase[(phase * phaseSize) + col] = srcRow[x];
                    if (++phase == phases)
                    {
                        phase = 0;
                        col++;
                    }
                }
            }
        }

        // Two phases: the even columns of the source row go to one phase plane and the odd ones to the other, which of the two depends
        // on the parity of the left padding. Four-lane shuffles pick the even and the odd elements of eight at a time; the scalar tail
        // and the machines without the shuffles do the same one element at a time. Only data moves, so the result is the same either way.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void SplitEvenOdd(float* srcRow, int width, float* rowBase, long phaseSize, int padLeft)
        {
            float* even = rowBase + ((padLeft & 1) * phaseSize) + (padLeft >> 1);
            float* odd = rowBase + (((padLeft + 1) & 1) * phaseSize) + ((padLeft + 1) >> 1);
            int x = 0;
            if (Sse.IsSupported)
            {
                for (; x + 8 <= width; x += 8)
                {
                    Vector128<float> a = Vector128.Load(srcRow + x);
                    Vector128<float> b = Vector128.Load(srcRow + x + 4);
                    Sse.Shuffle(a, b, 0x88).Store(even + (x >> 1));
                    Sse.Shuffle(a, b, 0xDD).Store(odd + (x >> 1));
                }
            }
            else if (AdvSimd.Arm64.IsSupported)
            {
                for (; x + 8 <= width; x += 8)
                {
                    Vector128<float> a = Vector128.Load(srcRow + x);
                    Vector128<float> b = Vector128.Load(srcRow + x + 4);
                    AdvSimd.Arm64.UnzipEven(a, b).Store(even + (x >> 1));
                    AdvSimd.Arm64.UnzipOdd(a, b).Store(odd + (x >> 1));
                }
            }

            for (; x < width; x++)
            {
                if ((x & 1) == 0)
                {
                    even[x >> 1] = srcRow[x];
                }
                else
                {
                    odd[x >> 1] = srcRow[x];
                }
            }
        }

        /// <summary>
        /// Planes narrower than one vector (the 7x7 and 14x14 maps at the end of MobileNet-style networks): the lanes run along the
        /// padded plane as one flat row, so a vector holds the end of one output row and the start of the next, and every tap is one
        /// full-width load instead of one per row. The positions that fall in the padding columns are computed too and dropped when
        /// the rows are copied out; each lane is still an independent output summed over the taps in the usual order, so the bits are those
        /// of the row-wise form. The loads stay inside the scratch plane for the valid lanes; the others read the slack after it.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void Flat<T>(float* src, int* tapOff, float* w, int taps, float* dst, float* tmp, int outH, int outW, int pitch)
            where T : struct, ILanes<T>
        {
            int width = T.Width;
            int total = ((outH - 1) * pitch) + outW;
            int v = 0;
            for (; v + (4 * width) <= total; v += 4 * width)
            {
                T a0 = T.Zero, a1 = T.Zero, a2 = T.Zero, a3 = T.Zero;
                for (int t = 0; t < taps; t++)
                {
                    T wv = T.Broadcast(w[t]);
                    float* p = src + tapOff[t] + v;
                    a0 = T.MulAdd(wv, T.Load(p), a0);
                    a1 = T.MulAdd(wv, T.Load(p + width), a1);
                    a2 = T.MulAdd(wv, T.Load(p + (2 * width)), a2);
                    a3 = T.MulAdd(wv, T.Load(p + (3 * width)), a3);
                }

                a0.Store(tmp + v);
                a1.Store(tmp + v + width);
                a2.Store(tmp + v + (2 * width));
                a3.Store(tmp + v + (3 * width));
            }

            for (; v < total; v += width)
            {
                T a = T.Zero;
                for (int t = 0; t < taps; t++)
                {
                    a = T.MulAdd(T.Broadcast(w[t]), T.Load(src + tapOff[t] + v), a);
                }

                a.Store(tmp + v);
            }

            for (int oy = 0; oy < outH; oy++)
            {
                CopyFloats(tmp + ((long)oy * pitch), dst + ((long)oy * outW), outW);
            }
        }

        /// <summary>Picks the row-wise or the four-rows-at-a-time form for lane type <typeparamref name="T"/> (outW >= its width).</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void Compute<T>(float* src, long rowStride, int* tapOff, float* w, int taps, float* dst, int outH, int outW)
            where T : struct, ILanes<T>
        {
            int rows = 0;
            if (outW < 4 * T.Width && outH >= 4)
            {
                rows = outH & ~3;
                FourRows<T>(src, rowStride, tapOff, w, taps, dst, rows, outW);
            }
            else if (outW > 4 * T.Width && outH >= 4)
            {
                // Wide rows: blocks of four vectors per row, then the leftover vectors of four rows together. A lone leftover vector is
                // one chain of dependent multiply-adds and would wait on the latency of every tap; four rows keep the unit busy.
                int main = outW / (4 * T.Width) * (4 * T.Width);
                OneRowMain<T>(src, rowStride, tapOff, w, taps, dst, 0, outH, outW, main);
                if (main < outW)
                {
                    TailRows<T>(src, rowStride, tapOff, w, taps, dst, outH, outW, main);
                }

                return;
            }

            OneRow<T>(src, rowStride, tapOff, w, taps, dst, rows, outH, outW);
        }

        // Rows [y0, y1), the columns [0, main) in blocks of four vectors (main is a multiple of four vectors).
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void OneRowMain<T>(float* src, long rowStride, int* tapOff, float* w, int taps, float* dst, int y0, int y1, int outW, int main)
            where T : struct, ILanes<T>
        {
            int width = T.Width;
            for (int oy = y0; oy < y1; oy++)
            {
                float* rowIn = src + (oy * rowStride);
                float* o = dst + ((long)oy * outW);
                for (int ox = 0; ox < main; ox += 4 * width)
                {
                    T a0 = T.Zero, a1 = T.Zero, a2 = T.Zero, a3 = T.Zero;
                    for (int t = 0; t < taps; t++)
                    {
                        T wv = T.Broadcast(w[t]);
                        float* p = rowIn + tapOff[t] + ox;
                        a0 = T.MulAdd(wv, T.Load(p), a0);
                        a1 = T.MulAdd(wv, T.Load(p + width), a1);
                        a2 = T.MulAdd(wv, T.Load(p + (2 * width)), a2);
                        a3 = T.MulAdd(wv, T.Load(p + (3 * width)), a3);
                    }

                    a0.Store(o + ox);
                    a1.Store(o + ox + width);
                    a2.Store(o + ox + (2 * width));
                    a3.Store(o + ox + (3 * width));
                }
            }
        }

        // Columns [main, outW) one vector at a time (the last moved back to end at the last column), four rows at a time, then single rows.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void TailRows<T>(float* src, long rowStride, int* tapOff, float* w, int taps, float* dst, int outH, int outW, int main)
            where T : struct, ILanes<T>
        {
            int width = T.Width;
            int oy = 0;
            for (; oy + 4 <= outH; oy += 4)
            {
                float* r0 = src + (oy * rowStride);
                float* r1 = r0 + rowStride;
                float* r2 = r1 + rowStride;
                float* r3 = r2 + rowStride;
                float* o = dst + ((long)oy * outW);
                for (int ox = main; ox < outW; ox += width)
                {
                    int at = ox + width <= outW ? ox : outW - width;
                    T a0 = T.Zero, a1 = T.Zero, a2 = T.Zero, a3 = T.Zero;
                    for (int t = 0; t < taps; t++)
                    {
                        T wv = T.Broadcast(w[t]);
                        int off = tapOff[t] + at;
                        a0 = T.MulAdd(wv, T.Load(r0 + off), a0);
                        a1 = T.MulAdd(wv, T.Load(r1 + off), a1);
                        a2 = T.MulAdd(wv, T.Load(r2 + off), a2);
                        a3 = T.MulAdd(wv, T.Load(r3 + off), a3);
                    }

                    a0.Store(o + at);
                    a1.Store(o + outW + at);
                    a2.Store(o + (2L * outW) + at);
                    a3.Store(o + (3L * outW) + at);
                }
            }

            for (; oy < outH; oy++)
            {
                float* rowIn = src + (oy * rowStride);
                float* o = dst + ((long)oy * outW);
                for (int ox = main; ox < outW; ox += width)
                {
                    int at = ox + width <= outW ? ox : outW - width;
                    T a = T.Zero;
                    for (int t = 0; t < taps; t++)
                    {
                        a = T.MulAdd(T.Broadcast(w[t]), T.Load(rowIn + tapOff[t] + at), a);
                    }

                    a.Store(o + at);
                }
            }
        }

        // Rows [y0, y1), each in steps of four vectors, then one, then the last vector moved back to end at the last column.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void OneRow<T>(float* src, long rowStride, int* tapOff, float* w, int taps, float* dst, int y0, int y1, int outW)
            where T : struct, ILanes<T>
        {
            int width = T.Width;
            for (int oy = y0; oy < y1; oy++)
            {
                float* rowIn = src + (oy * rowStride);
                float* o = dst + ((long)oy * outW);
                int ox = 0;
                for (; ox + (4 * width) <= outW; ox += 4 * width)
                {
                    T a0 = T.Zero, a1 = T.Zero, a2 = T.Zero, a3 = T.Zero;
                    for (int t = 0; t < taps; t++)
                    {
                        T wv = T.Broadcast(w[t]);
                        float* p = rowIn + tapOff[t] + ox;
                        a0 = T.MulAdd(wv, T.Load(p), a0);
                        a1 = T.MulAdd(wv, T.Load(p + width), a1);
                        a2 = T.MulAdd(wv, T.Load(p + (2 * width)), a2);
                        a3 = T.MulAdd(wv, T.Load(p + (3 * width)), a3);
                    }

                    a0.Store(o + ox);
                    a1.Store(o + ox + width);
                    a2.Store(o + ox + (2 * width));
                    a3.Store(o + ox + (3 * width));
                }

                for (; ox < outW; ox += width)
                {
                    int at = ox + width <= outW ? ox : outW - width;
                    T a = T.Zero;
                    for (int t = 0; t < taps; t++)
                    {
                        a = T.MulAdd(T.Broadcast(w[t]), T.Load(rowIn + tapOff[t] + at), a);
                    }

                    a.Store(o + at);
                }
            }
        }

        // Rows [0, rows) four at a time (rows is a multiple of four), one vector wide, the last vector moved back as above.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void FourRows<T>(float* src, long rowStride, int* tapOff, float* w, int taps, float* dst, int rows, int outW)
            where T : struct, ILanes<T>
        {
            int width = T.Width;
            for (int oy = 0; oy < rows; oy += 4)
            {
                float* r0 = src + (oy * rowStride);
                float* r1 = r0 + rowStride;
                float* r2 = r1 + rowStride;
                float* r3 = r2 + rowStride;
                float* o = dst + ((long)oy * outW);
                for (int ox = 0; ox < outW; ox += width)
                {
                    int at = ox + width <= outW ? ox : outW - width;
                    T a0 = T.Zero, a1 = T.Zero, a2 = T.Zero, a3 = T.Zero;
                    for (int t = 0; t < taps; t++)
                    {
                        T wv = T.Broadcast(w[t]);
                        int off = tapOff[t] + at;
                        a0 = T.MulAdd(wv, T.Load(r0 + off), a0);
                        a1 = T.MulAdd(wv, T.Load(r1 + off), a1);
                        a2 = T.MulAdd(wv, T.Load(r2 + off), a2);
                        a3 = T.MulAdd(wv, T.Load(r3 + off), a3);
                    }

                    a0.Store(o + at);
                    a1.Store(o + outW + at);
                    a2.Store(o + (2L * outW) + at);
                    a3.Store(o + (3L * outW) + at);
                }
            }
        }

        // Kernels with more than MaxTaps taps or a column stride above MaxPhases: the same sum, one output at a time with bounds tests.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void Reference(float* plane, float* w, float* dst, in ConvShape s)
        {
            for (int oy = 0; oy < s.OutH; oy++)
            {
                for (int ox = 0; ox < s.OutW; ox++)
                {
                    TS acc = TS.Zero;
                    for (int ky = 0; ky < s.KH; ky++)
                    {
                        int iy = (oy * s.StrideH) - s.PadTop + (ky * s.DilH);
                        for (int kx = 0; kx < s.KW; kx++)
                        {
                            int ix = (ox * s.StrideW) - s.PadLeft + (kx * s.DilW);
                            float v = iy >= 0 && iy < s.H && ix >= 0 && ix < s.W ? plane[((long)iy * s.W) + ix] : 0f;
                            acc = TS.MulAdd(TS.Broadcast(w[(ky * s.KW) + kx]), TS.Broadcast(v), acc);
                        }
                    }

                    acc.Store(dst + ((long)oy * s.OutW) + ox);
                }
            }
        }
    }
}
