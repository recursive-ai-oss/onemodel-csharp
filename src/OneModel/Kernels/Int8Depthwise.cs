using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Direct int8 depthwise convolution of one input plane. The plane is first widened to 16 bits with its zero point subtracted and
/// laid out zero-padded, one copy per column residue modulo the horizontal stride, so that every kernel tap of a whole block of
/// output columns is one contiguous vector load, with no bounds tests in the inner loop. Sums are exact 32-bit integers, so the result
/// equals the scalar loop's on every path.
/// Bounds the caller guarantees: <c>x</c> points at H*W bytes, <c>wv</c> at KH*KW ints (weights with their zero point already
/// subtracted), and the output pointer at OutH*OutW elements; the scratch is sized here.
/// </summary>
internal static unsafe class Int8Depthwise
{
    // 0, 1, ..., 15: compared with a count it gives the mask of the first count lanes.
    private static Vector128<byte> Lanes => Vector128.Create((byte)0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

    /// <summary>Whether <see cref="Plane"/> handles this geometry (horizontal stride 1 or 2, no negative padding, vectors available).</summary>
    public static bool Supports(in ConvShape s) =>
        Vector128.IsHardwareAccelerated && s.StrideW is 1 or 2 && s.PadLeft >= 0 && s.PadTop >= 0 && s.DilW >= 1 && s.DilH >= 1 &&
        s.StrideH >= 1;

    // The padded-plane layout, a function of the geometry alone: rows of the padded copy, columns of an output row rounded up to a whole
    // number of vectors, and the row pitch of the padded copy in shorts.
    private static void Layout(in ConvShape s, int outH, int outW, out int hp, out int block, out int owPad, out int stride)
    {
        hp = ((outH - 1) * s.StrideH) + ((s.KH - 1) * s.DilH) + 1;
        int maxOff = ((s.KW - 1) * s.DilW) / s.StrideW;
        block = Vector256.IsHardwareAccelerated ? 16 : 8;
        owPad = (outW + block - 1) / block * block;
        stride = owPad + maxOff + 16;
    }

    /// <summary>
    /// Fills <paramref name="offsets"/> (KH*KW ints) with, for each kernel tap, where its first input element sits in the padded copy
    /// relative to the start of an output row's base. They depend on the geometry only, so a caller computes them once for all planes.
    /// </summary>
    public static void TapOffsets(in ConvShape s, int* offsets)
    {
        Layout(in s, s.OutH, s.OutW, out int hp, out _, out _, out int stride);
        for (int ky = 0; ky < s.KH; ky++)
        {
            for (int kx = 0; kx < s.KW; kx++)
            {
                int cx = kx * s.DilW;
                offsets[(ky * s.KW) + kx] = ((((cx % s.StrideW) * hp) + (ky * s.DilH)) * stride) + (cx / s.StrideW);
            }
        }
    }

    public static void Plane(in ConvShape s, byte* x, bool xU8, int xZero, int* wv, int* offsets, int bias, float mult,
        int* outInt, byte* outQuant, bool quantUnsigned, int quantZero)
    {
        int sx = s.StrideW, sy = s.StrideH;
        int outH = s.OutH, outW = s.OutW, kh = s.KH, kw = s.KW;
        Layout(in s, outH, outW, out int hp, out int block, out int owPad, out int stride);
        int wNeed = ((outW - 1) * sx) + ((kw - 1) * s.DilW) + 1;
        int maxOff = ((kw - 1) * s.DilW) / sx;

        long shorts = (long)sx * hp * stride;
        long ints = (4L * owPad) + 16;
        // Scratch is counted in 4-byte units: the padded planes, then one row of int32 sums.
        float* scratch = Scratch.Get(Scratch.Im2Col, ((shorts + 1) / 2) + ints);
        short* sub = (short*)scratch;
        int* acc = (int*)(sub + (((shorts + 1) / 2) * 2));
        new Span<short>(sub, (int)shorts).Clear();

        // Widen the plane: value = shifted input - zero point, as 16-bit; padding stays zero (the input's zero point).
        int bias8 = xU8 ? 128 + xZero : xZero;
        byte* tmp = stackalloc byte[16];
        for (int iy = 0; iy < s.H; iy++)
        {
            int pr = iy + s.PadTop;
            if (pr >= hp)
            {
                break;
            }

            byte* src = x + ((long)iy * s.W);
            int cols = Math.Min(s.W, wNeed - s.PadLeft);
            if (cols <= 0)
            {
                continue;
            }

            if (sx == 1)
            {
                short* dst = sub + ((long)pr * stride) + s.PadLeft;
                int i = 0;
                Vector128<short> vb = Vector128.Create((short)bias8);
                for (; i + 16 <= cols; i += 16)
                {
                    Vector128<byte> v = Vector128.Load(src + i);
                    Vector128<short> lo, hi;
                    if (xU8)
                    {
                        lo = Vector128.WidenLower(v).AsInt16() - vb;
                        hi = Vector128.WidenUpper(v).AsInt16() - vb;
                    }
                    else
                    {
                        Vector128<sbyte> sv = v.AsSByte();
                        lo = Vector128.WidenLower(sv) - vb;
                        hi = Vector128.WidenUpper(sv) - vb;
                    }

                    lo.Store(dst + i);
                    hi.Store(dst + i + 8);
                }

                if (i < cols)
                {
                    // The tail is one more vector. Inside the plane it is read in place and the lanes past the row are cleared; at the
                    // plane's very end it comes through a 16-byte stack copy padded with the zero point (which widens to the zero padding).
                    // Either way the extra lanes land in the row's slack, which is zero padding.
                    Vector128<byte> v;
                    if (src + i + 16 <= x + ((long)s.H * s.W))
                    {
                        v = Vector128.Load(src + i);
                    }
                    else
                    {
                        for (int t = 0; t < 16; t++)
                        {
                            tmp[t] = i + t < cols ? src[i + t] : (byte)bias8;
                        }

                        v = Vector128.Load(tmp);
                    }

                    Widen(v, xU8, vb, out Vector128<short> t0, out Vector128<short> t1);
                    Vector128<sbyte> keep = Vector128.LessThan(Lanes, Vector128.Create((byte)(cols - i))).AsSByte();
                    (t0 & Vector128.WidenLower(keep)).Store(dst + i);
                    (t1 & Vector128.WidenUpper(keep)).Store(dst + i + 8);
                }
            }
            else
            {
                // Column c of the padded plane goes to copy c % 2, position c / 2.
                short* d0 = sub + ((long)pr * stride);
                short* d1 = sub + (((long)hp + pr) * stride);
                int i = 0;
                if ((s.PadLeft & 1) != 0 && cols > 0)
                {
                    // Start on an even padded column so the pairs of a vector load split into copy 0 (even) and copy 1 (odd).
                    int pc0 = s.PadLeft;
                    d1[pc0 >> 1] = (short)((xU8 ? src[0] : (sbyte)src[0]) - bias8);
                    i = 1;
                }

                if (Sse2.IsSupported || AdvSimd.Arm64.IsSupported)
                {
                    Vector128<short> vb = Vector128.Create((short)bias8);
                    for (; i + 32 <= cols; i += 32)
                    {
                        int h = (i + s.PadLeft) >> 1;
                        Vector128<byte> a = Vector128.Load(src + i), b = Vector128.Load(src + i + 16);
                        SplitEvenOdd(a, b, out Vector128<byte> ev, out Vector128<byte> od);
                        Widen(ev, xU8, vb, out Vector128<short> e0, out Vector128<short> e1);
                        Widen(od, xU8, vb, out Vector128<short> o0, out Vector128<short> o1);
                        e0.Store(d0 + h);
                        e1.Store(d0 + h + 8);
                        o0.Store(d1 + h);
                        o1.Store(d1 + h + 8);
                    }
                }

                if (i < cols && (Sse2.IsSupported || AdvSimd.Arm64.IsSupported) && src + i + 32 <= x + ((long)s.H * s.W) &&
                    ((i + s.PadLeft) >> 1) + 16 <= stride)
                {
                    // The last, partial block of the row: both halves read in place, the lanes past the row cleared.
                    int h = (i + s.PadLeft) >> 1, rem = cols - i;
                    Vector128<short> vb = Vector128.Create((short)bias8);
                    SplitEvenOdd(Vector128.Load(src + i), Vector128.Load(src + i + 16), out Vector128<byte> ev, out Vector128<byte> od);
                    Widen(ev, xU8, vb, out Vector128<short> e0, out Vector128<short> e1);
                    Widen(od, xU8, vb, out Vector128<short> o0, out Vector128<short> o1);
                    Vector128<sbyte> keepE = Vector128.LessThan(Lanes, Vector128.Create((byte)((rem + 1) >> 1))).AsSByte();
                    Vector128<sbyte> keepO = Vector128.LessThan(Lanes, Vector128.Create((byte)(rem >> 1))).AsSByte();
                    (e0 & Vector128.WidenLower(keepE)).Store(d0 + h);
                    (e1 & Vector128.WidenUpper(keepE)).Store(d0 + h + 8);
                    (o0 & Vector128.WidenLower(keepO)).Store(d1 + h);
                    (o1 & Vector128.WidenUpper(keepO)).Store(d1 + h + 8);
                    i = cols;
                }

                for (; i < cols; i++)
                {
                    int pc = i + s.PadLeft;
                    short v = (short)((xU8 ? src[i] : (sbyte)src[i]) - bias8);
                    if ((pc & 1) == 0)
                    {
                        d0[pc >> 1] = v;
                    }
                    else
                    {
                        d1[pc >> 1] = v;
                    }
                }
            }
        }

        // The non-zero taps: weight and the offset of its first input element from the start of the output row's base. Consecutive taps
        // are paired (an odd one out pairs with a zero weight) so the 16-bit multiply-add can take two taps per instruction.
        int* tw = stackalloc int[(kh * kw) + 1];
        int* to = stackalloc int[(kh * kw) + 1];
        int taps = 0;
        for (int t = 0; t < kh * kw; t++)
        {
            if (wv[t] == 0)
            {
                continue;
            }

            tw[taps] = wv[t];
            to[taps] = offsets[t];
            taps++;
        }

        bool pairs = block == 16 && Avx2.IsSupported;
        if (pairs && (taps & 1) != 0)
        {
            tw[taps] = 0;
            to[taps] = to[taps - 1];
            taps++;
        }

        if (!pairs && AdvSimd.IsSupported && block == 8 && owPad <= 16)
        {
            // Narrow planes: one row has only two or four independent accumulators, and a multiply-accumulate waits for the previous one
            // on the same register, so a row alone runs at the instruction's latency. Four rows of eight columns (or two of sixteen) at
            // once give eight independent chains.
            int per = owPad == 8 ? 4 : 2;
            for (int oy0 = 0; oy0 < outH; oy0 += per)
            {
                int rows = Math.Min(per, outH - oy0);
                if (owPad == 8)
                {
                    NarrowRows8(sub, oy0, rows, sy, stride, tw, to, taps, bias, acc);
                }
                else
                {
                    NarrowRows16(sub, oy0, rows, sy, stride, tw, to, taps, bias, acc);
                }

                for (int r = 0; r < rows; r++)
                {
                    EmitRow(acc + (r * owPad), oy0 + r, outW, outH, owPad, outInt, outQuant, mult, quantZero, quantUnsigned);
                }
            }

            return;
        }

        for (int oy = 0; oy < outH; oy++)
        {
            if (pairs)
            {
                RowPairs256(sub + ((long)oy * sy * stride), tw, to, taps, bias, owPad, acc);
            }
            else if (block == 16)
            {
                RowV256(sub + ((long)oy * sy * stride), tw, to, taps, bias, owPad, acc);
            }
            else
            {
                RowV128(sub + ((long)oy * sy * stride), tw, to, taps, bias, owPad, acc);
            }

            EmitRow(acc, oy, outW, outH, owPad, outInt, outQuant, mult, quantZero, quantUnsigned);
        }
    }

    // Writes the sums of output row oy: as they are, or requantized to bytes. Rows are written in increasing order, so a requantized
    // row may store a few bytes past its end (the next row overwrites them) as long as they stay inside the plane.
    private static void EmitRow(int* acc, int oy, int outW, int outH, int owPad, int* outInt, byte* outQuant, float mult, int quantZero, bool quantUnsigned)
    {
        long o = (long)oy * outW;
        if (outInt != null)
        {
            for (int i = 0; i < outW; i++)
            {
                outInt[o + i] = acc[i];
            }
        }
        else if (o + owPad <= (long)outH * outW)
        {
            Int8Gemm.RequantRowPadded(acc, outQuant + o, owPad, mult, quantZero, quantUnsigned);
        }
        else
        {
            Int8Gemm.RequantRow(acc, outQuant + o, outW, mult, quantZero, quantUnsigned);
        }
    }

    // Four output rows of eight columns: rows beyond `rows` repeat the first row's inputs and their sums are not used.
    private static void NarrowRows8(short* sub, int oy0, int rows, int sy, int stride, int* tw, int* to, int taps, int bias, int* acc)
    {
        short* b0 = sub + ((long)oy0 * sy * stride);
        short* b1 = rows > 1 ? sub + ((long)(oy0 + 1) * sy * stride) : b0;
        short* b2 = rows > 2 ? sub + ((long)(oy0 + 2) * sy * stride) : b0;
        short* b3 = rows > 3 ? sub + ((long)(oy0 + 3) * sy * stride) : b0;
        Vector128<int> vbias = Vector128.Create(bias);
        Vector128<int> a0 = vbias, a1 = vbias, a2 = vbias, a3 = vbias, a4 = vbias, a5 = vbias, a6 = vbias, a7 = vbias;
        for (int t = 0; t < taps; t++)
        {
            int off = to[t];
            Vector128<short> w = Vector128.Create((short)tw[t]);
            Vector64<short> wl = w.GetLower();
            Vector128<short> v0 = Vector128.Load(b0 + off), v1 = Vector128.Load(b1 + off), v2 = Vector128.Load(b2 + off), v3 = Vector128.Load(b3 + off);
            a0 = AdvSimd.MultiplyWideningLowerAndAdd(a0, v0.GetLower(), wl);
            a1 = AdvSimd.MultiplyWideningUpperAndAdd(a1, v0, w);
            a2 = AdvSimd.MultiplyWideningLowerAndAdd(a2, v1.GetLower(), wl);
            a3 = AdvSimd.MultiplyWideningUpperAndAdd(a3, v1, w);
            a4 = AdvSimd.MultiplyWideningLowerAndAdd(a4, v2.GetLower(), wl);
            a5 = AdvSimd.MultiplyWideningUpperAndAdd(a5, v2, w);
            a6 = AdvSimd.MultiplyWideningLowerAndAdd(a6, v3.GetLower(), wl);
            a7 = AdvSimd.MultiplyWideningUpperAndAdd(a7, v3, w);
        }

        a0.Store(acc);
        a1.Store(acc + 4);
        a2.Store(acc + 8);
        a3.Store(acc + 12);
        a4.Store(acc + 16);
        a5.Store(acc + 20);
        a6.Store(acc + 24);
        a7.Store(acc + 28);
    }

    // Two output rows of sixteen columns.
    private static void NarrowRows16(short* sub, int oy0, int rows, int sy, int stride, int* tw, int* to, int taps, int bias, int* acc)
    {
        short* b0 = sub + ((long)oy0 * sy * stride);
        short* b1 = rows > 1 ? sub + ((long)(oy0 + 1) * sy * stride) : b0;
        Vector128<int> vbias = Vector128.Create(bias);
        Vector128<int> a0 = vbias, a1 = vbias, a2 = vbias, a3 = vbias, a4 = vbias, a5 = vbias, a6 = vbias, a7 = vbias;
        for (int t = 0; t < taps; t++)
        {
            int off = to[t];
            Vector128<short> w = Vector128.Create((short)tw[t]);
            Vector64<short> wl = w.GetLower();
            Vector128<short> v0 = Vector128.Load(b0 + off), v1 = Vector128.Load(b0 + off + 8), v2 = Vector128.Load(b1 + off), v3 = Vector128.Load(b1 + off + 8);
            a0 = AdvSimd.MultiplyWideningLowerAndAdd(a0, v0.GetLower(), wl);
            a1 = AdvSimd.MultiplyWideningUpperAndAdd(a1, v0, w);
            a2 = AdvSimd.MultiplyWideningLowerAndAdd(a2, v1.GetLower(), wl);
            a3 = AdvSimd.MultiplyWideningUpperAndAdd(a3, v1, w);
            a4 = AdvSimd.MultiplyWideningLowerAndAdd(a4, v2.GetLower(), wl);
            a5 = AdvSimd.MultiplyWideningUpperAndAdd(a5, v2, w);
            a6 = AdvSimd.MultiplyWideningLowerAndAdd(a6, v3.GetLower(), wl);
            a7 = AdvSimd.MultiplyWideningUpperAndAdd(a7, v3, w);
        }

        a0.Store(acc);
        a1.Store(acc + 4);
        a2.Store(acc + 8);
        a3.Store(acc + 12);
        a4.Store(acc + 16);
        a5.Store(acc + 20);
        a6.Store(acc + 24);
        a7.Store(acc + 28);
    }

    // Sixteen bytes to sixteen 16-bit values minus the bias, as two vectors.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Widen(Vector128<byte> v, bool u8, Vector128<short> bias, out Vector128<short> lo, out Vector128<short> hi)
    {
        if (u8)
        {
            lo = Vector128.WidenLower(v).AsInt16() - bias;
            hi = Vector128.WidenUpper(v).AsInt16() - bias;
        }
        else
        {
            Vector128<sbyte> sv = v.AsSByte();
            lo = Vector128.WidenLower(sv) - bias;
            hi = Vector128.WidenUpper(sv) - bias;
        }
    }

    // The even-indexed and the odd-indexed bytes of the 32 bytes a, b.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SplitEvenOdd(Vector128<byte> a, Vector128<byte> b, out Vector128<byte> even, out Vector128<byte> odd)
    {
        if (Sse2.IsSupported)
        {
            Vector128<short> m = Vector128.Create((short)0x00FF);
            even = Sse2.PackUnsignedSaturate(a.AsInt16() & m, b.AsInt16() & m);
            odd = Sse2.PackUnsignedSaturate(Sse2.ShiftRightLogical(a.AsUInt16(), 8).AsInt16(), Sse2.ShiftRightLogical(b.AsUInt16(), 8).AsInt16());
            return;
        }

        even = AdvSimd.Arm64.UnzipEven(a, b);
        odd = AdvSimd.Arm64.UnzipOdd(a, b);
    }

    // One output row on AVX2, 16 columns per step, two taps per multiply-add (vpmaddwd): the two taps' columns are interleaved
    // into 16-bit pairs and multiplied by the pair of weights, so a pair of taps costs two loads, two unpacks, two multiply-adds and
    // two adds. Sums are exact 32-bit integers (an input is at most 255 in magnitude and a weight at most 255).
    private static void RowPairs256(short* rowBase, int* tw, int* to, int taps, int bias, int owPad, int* acc)
    {
        Vector256<int> vbias = Vector256.Create(bias);
        for (int ox = 0; ox < owPad; ox += 16)
        {
            // a0 holds columns 0-3 and 8-11, a1 columns 4-7 and 12-15 (the unpacks work inside 128-bit halves).
            Vector256<int> a0 = vbias, a1 = vbias;
            for (int t = 0; t < taps; t += 2)
            {
                Vector256<short> v0 = Avx.LoadVector256(rowBase + to[t] + ox);
                Vector256<short> v1 = Avx.LoadVector256(rowBase + to[t + 1] + ox);
                Vector256<short> w = Vector256.Create((tw[t] & 0xFFFF) | (tw[t + 1] << 16)).AsInt16();
                a0 = Avx2.Add(a0, Avx2.MultiplyAddAdjacent(Avx2.UnpackLow(v0, v1), w));
                a1 = Avx2.Add(a1, Avx2.MultiplyAddAdjacent(Avx2.UnpackHigh(v0, v1), w));
            }

            a0.GetLower().Store(acc + ox);
            a1.GetLower().Store(acc + ox + 4);
            a0.GetUpper().Store(acc + ox + 8);
            a1.GetUpper().Store(acc + ox + 12);
        }
    }

    // One output row, 16 columns per step: 16 sums in two 256-bit registers across every tap.
    private static void RowV256(short* rowBase, int* tw, int* to, int taps, int bias, int owPad, int* acc)
    {
        Vector256<int> vbias = Vector256.Create(bias);
        for (int ox = 0; ox < owPad; ox += 16)
        {
            Vector256<int> a0 = vbias, a1 = vbias;
            for (int t = 0; t < taps; t++)
            {
                Vector256<short> v = Vector256.Load(rowBase + to[t] + ox);
                Vector256<int> vw = Vector256.Create(tw[t]);
                a0 += Vector256.WidenLower(v) * vw;
                a1 += Vector256.WidenUpper(v) * vw;
            }

            a0.Store(acc + ox);
            a1.Store(acc + ox + 8);
        }
    }

    // The same with 128-bit vectors: 8 columns per step. On Arm the widening multiply-accumulate (smlal, smlal2) does the multiply and
    // the add of each half in one instruction.
    private static void RowV128(short* rowBase, int* tw, int* to, int taps, int bias, int owPad, int* acc)
    {
        Vector128<int> vbias = Vector128.Create(bias);
        int ox = 0;
        if (AdvSimd.IsSupported)
        {
            // Four blocks of eight columns at a time: eight independent accumulators hide the latency of the multiply-accumulate and
            // each tap's weight is loaded once for all of them.
            for (; ox + 32 <= owPad; ox += 32)
            {
                Vector128<int> a0 = vbias, a1 = vbias, a2 = vbias, a3 = vbias, a4 = vbias, a5 = vbias, a6 = vbias, a7 = vbias;
                for (int t = 0; t < taps; t++)
                {
                    short* p = rowBase + to[t] + ox;
                    Vector128<short> w = Vector128.Create((short)tw[t]);
                    Vector64<short> wl = w.GetLower();
                    Vector128<short> v0 = Vector128.Load(p), v1 = Vector128.Load(p + 8), v2 = Vector128.Load(p + 16), v3 = Vector128.Load(p + 24);
                    a0 = AdvSimd.MultiplyWideningLowerAndAdd(a0, v0.GetLower(), wl);
                    a1 = AdvSimd.MultiplyWideningUpperAndAdd(a1, v0, w);
                    a2 = AdvSimd.MultiplyWideningLowerAndAdd(a2, v1.GetLower(), wl);
                    a3 = AdvSimd.MultiplyWideningUpperAndAdd(a3, v1, w);
                    a4 = AdvSimd.MultiplyWideningLowerAndAdd(a4, v2.GetLower(), wl);
                    a5 = AdvSimd.MultiplyWideningUpperAndAdd(a5, v2, w);
                    a6 = AdvSimd.MultiplyWideningLowerAndAdd(a6, v3.GetLower(), wl);
                    a7 = AdvSimd.MultiplyWideningUpperAndAdd(a7, v3, w);
                }

                a0.Store(acc + ox);
                a1.Store(acc + ox + 4);
                a2.Store(acc + ox + 8);
                a3.Store(acc + ox + 12);
                a4.Store(acc + ox + 16);
                a5.Store(acc + ox + 20);
                a6.Store(acc + ox + 24);
                a7.Store(acc + ox + 28);
            }
        }

        for (; ox < owPad; ox += 8)
        {
            Vector128<int> a0 = vbias, a1 = vbias;
            if (AdvSimd.IsSupported)
            {
                for (int t = 0; t < taps; t++)
                {
                    Vector128<short> v = Vector128.Load(rowBase + to[t] + ox);
                    Vector128<short> w = Vector128.Create((short)tw[t]);
                    a0 = AdvSimd.MultiplyWideningLowerAndAdd(a0, v.GetLower(), w.GetLower());
                    a1 = AdvSimd.MultiplyWideningUpperAndAdd(a1, v, w);
                }
            }
            else
            {
                for (int t = 0; t < taps; t++)
                {
                    Vector128<short> v = Vector128.Load(rowBase + to[t] + ox);
                    Vector128<int> vw = Vector128.Create(tw[t]);
                    a0 += Vector128.WidenLower(v) * vw;
                    a1 += Vector128.WidenUpper(v) * vw;
                }
            }

            a0.Store(acc + ox);
            a1.Store(acc + ox + 4);
        }
    }
}
