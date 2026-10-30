using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

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

                for (; i < cols; i++)
                {
                    dst[i] = (short)((xU8 ? src[i] : (sbyte)src[i]) - bias8);
                }
            }
            else
            {
                // Column c of the padded plane goes to copy c % 2, position c / 2.
                short* d0 = sub + ((long)pr * stride);
                short* d1 = sub + (((long)hp + pr) * stride);
                for (int i = 0; i < cols; i++)
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

        // The non-zero taps: weight and the offset of its first input element from the start of the output row's base.
        int* tw = stackalloc int[kh * kw];
        int* to = stackalloc int[kh * kw];
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

        for (int oy = 0; oy < outH; oy++)
        {
            if (block == 16)
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

    // The same with 128-bit vectors: 8 columns per step.
    private static void RowV128(short* rowBase, int* tw, int* to, int taps, int bias, int owPad, int* acc)
    {
        Vector128<int> vbias = Vector128.Create(bias);
        for (int ox = 0; ox < owPad; ox += 8)
        {
            Vector128<int> a0 = vbias, a1 = vbias;
            for (int t = 0; t < taps; t++)
            {
                Vector128<short> v = Vector128.Load(rowBase + to[t] + ox);
                Vector128<int> vw = Vector128.Create(tw[t]);
                a0 += Vector128.WidenLower(v) * vw;
                a1 += Vector128.WidenUpper(v) * vw;
            }

            a0.Store(acc + ox);
            a1.Store(acc + ox + 4);
        }
    }
}
