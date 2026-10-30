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
    /// <summary>Whether <see cref="Plane"/> handles this geometry (horizontal stride 1 or 2, no negative padding, vectors available).</summary>
    public static bool Supports(in ConvShape s) =>
        Vector128.IsHardwareAccelerated && s.StrideW is 1 or 2 && s.PadLeft >= 0 && s.PadTop >= 0 && s.DilW >= 1 && s.DilH >= 1 &&
        s.StrideH >= 1;

    public static void Plane(in ConvShape s, byte* x, bool xU8, int xZero, int* wv, int bias, float mult,
        int* outInt, byte* outQuant, bool quantUnsigned, int quantZero)
    {
        int sx = s.StrideW, sy = s.StrideH;
        int outH = s.OutH, outW = s.OutW, kh = s.KH, kw = s.KW;
        int hp = ((outH - 1) * sy) + ((kh - 1) * s.DilH) + 1;
        int wNeed = ((outW - 1) * sx) + ((kw - 1) * s.DilW) + 1;
        int maxOff = ((kw - 1) * s.DilW) / sx;
        int block = Vector256.IsHardwareAccelerated ? 16 : 8;
        int owPad = (outW + block - 1) / block * block;
        int stride = owPad + maxOff + 16;

        long shorts = (long)sx * hp * stride;
        long ints = (long)owPad + 16;
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
                    // The tail through a 16-byte stack copy padded with the zero point (which widens to the zero padding): the extra
                    // lanes land in the row's slack, which is zero padding anyway.
                    for (int t = 0; t < 16; t++)
                    {
                        tmp[t] = i + t < cols ? src[i + t] : (byte)bias8;
                    }

                    Widen(Vector128.Load(tmp), xU8, vb, out Vector128<short> t0, out Vector128<short> t1);
                    t0.Store(dst + i);
                    t1.Store(dst + i + 8);
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

            int ky = t / kw, kx = t - (ky * kw);
            int cx = kx * s.DilW;
            tw[taps] = wv[t];
            to[taps] = (((cx % sx) * hp) + (ky * s.DilH)) * stride + (cx / sx);
            taps++;
        }

        bool pairs = block == 16 && Avx2.IsSupported;
        if (pairs && (taps & 1) != 0)
        {
            tw[taps] = 0;
            to[taps] = to[taps - 1];
            taps++;
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

            long o = (long)oy * outW;
            if (outInt != null)
            {
                for (int i = 0; i < outW; i++)
                {
                    outInt[o + i] = acc[i];
                }
            }
            else
            {
                Int8Gemm.RequantRow(acc, outQuant + o, outW, mult, quantZero, quantUnsigned);
            }
        }
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
