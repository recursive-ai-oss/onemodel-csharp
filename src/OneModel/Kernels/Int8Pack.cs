using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Vector packing of the B operand of the int8 product (Goto and van de Geijn 2008, section 4: pack B into strips the
/// microkernel streams): sixteen columns at a time, each group of <see cref="Int8Format.G"/> rows is interleaved column by column
/// with one unpack (x86) or zip (Arm) per register, so a strip costs a load, a few shuffles and a store per sixteen elements
/// instead of a scalar loop. The bytes written are exactly those of the scalar packer in <see cref="Int8Gemm"/>; integer data, so
/// there is nothing to round and every path agrees. Used for flavours with 16-column register blocks (every non-scalar flavour).
/// </summary>
/// <remarks>
/// Bounds each entry point establishes (the caller guarantees the extents; see the entry points): reads stay inside the k x n source
/// (a block of sixteen columns is loaded whole only when all sixteen exist, otherwise through a 16-byte stack copy), and writes
/// stay inside the strips [s0, s1) of the packed buffer (Nr * KPad bytes each) and the column sums of their valid columns.
/// </remarks>
internal static unsafe class Int8Pack
{
    /// <summary>True when the vector packers can serve the flavour: a non-scalar flavour on a machine with the shuffle instructions.</summary>
    public static bool Supported(in Int8Format f) =>
        f.Isa != Int8Isa.Scalar && f.Nr % 16 == 0 && ((f.G == 2 && (Sse2.IsSupported || AdvSimd.Arm64.IsSupported)) || (f.G == 4 && AdvSimd.Arm64.IsSupported));

    // Writes the interleave of g rows of sixteen columns: for each column its g bytes (one k group), 16 * g bytes at d.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Interleave(int g, Vector128<sbyte> r0, Vector128<sbyte> r1, Vector128<sbyte> r2, Vector128<sbyte> r3, sbyte* d)
    {
        if (g == 2)
        {
            if (Sse2.IsSupported)
            {
                Sse2.UnpackLow(r0, r1).Store(d);
                Sse2.UnpackHigh(r0, r1).Store(d + 16);
            }
            else
            {
                AdvSimd.Arm64.ZipLow(r0, r1).Store(d);
                AdvSimd.Arm64.ZipHigh(r0, r1).Store(d + 16);
            }

            return;
        }

        // g == 4 (Arm): bytes of two rows zipped, then those pairs zipped as 16-bit units.
        Vector128<short> a = AdvSimd.Arm64.ZipLow(r0, r1).AsInt16();
        Vector128<short> b = AdvSimd.Arm64.ZipHigh(r0, r1).AsInt16();
        Vector128<short> c = AdvSimd.Arm64.ZipLow(r2, r3).AsInt16();
        Vector128<short> e = AdvSimd.Arm64.ZipHigh(r2, r3).AsInt16();
        AdvSimd.Arm64.ZipLow(a, c).AsSByte().Store(d);
        AdvSimd.Arm64.ZipHigh(a, c).AsSByte().Store(d + 16);
        AdvSimd.Arm64.ZipLow(b, e).AsSByte().Store(d + 32);
        AdvSimd.Arm64.ZipHigh(b, e).AsSByte().Store(d + 48);
    }

    // The sixteen even-indexed bytes of the 32 bytes at p: lane l of a stride-2 run reads p[2 * l].
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector128<sbyte> Even(sbyte* p)
    {
        Vector128<sbyte> a = Vector128.Load(p), b = Vector128.Load(p + 16);
        if (Sse2.IsSupported)
        {
            Vector128<short> m = Vector128.Create((short)0x00FF);
            return Sse2.PackUnsignedSaturate(a.AsInt16() & m, b.AsInt16() & m).AsSByte();
        }

        return AdvSimd.Arm64.UnzipEven(a, b);
    }

    // Column sums of sixteen columns kept in 16-bit lanes (a value is at most 128 in magnitude, so 255 rows cannot overflow) and
    // flushed into 32-bit lanes every 128 rows.
    private struct Sums16
    {
        public Vector128<short> Lo, Hi;
        public Vector128<int> I0, I1, I2, I3;
        public int Rows;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Add(Vector128<sbyte> v)
        {
            Lo += Vector128.WidenLower(v);
            Hi += Vector128.WidenUpper(v);
            if (++Rows == 128)
            {
                Flush();
            }
        }

        public void Flush()
        {
            I0 += Vector128.WidenLower(Lo);
            I1 += Vector128.WidenUpper(Lo);
            I2 += Vector128.WidenLower(Hi);
            I3 += Vector128.WidenUpper(Hi);
            Lo = default;
            Hi = default;
            Rows = 0;
        }

        // Stores the first n sums at dst.
        public void Store(int* dst, int n)
        {
            Flush();
            int* tmp = stackalloc int[16];
            I0.Store(tmp);
            I1.Store(tmp + 4);
            I2.Store(tmp + 8);
            I3.Store(tmp + 12);
            for (int c = 0; c < n; c++)
            {
                dst[c] = tmp[c];
            }
        }
    }

    // Loads sixteen columns of one source row into the shifted signed domain; fewer than sixteen are read through a stack copy and the
    // rest read as zero.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<sbyte> LoadRow(byte* p, int cols, bool u8, byte* tmp)
    {
        if (cols == 16)
        {
            Vector128<byte> v = Vector128.Load(p);
            return (u8 ? v ^ Vector128.Create((byte)0x80) : v).AsSByte();
        }

        for (int c = 0; c < 16; c++)
        {
            tmp[c] = c < cols ? p[c] : (byte)(u8 ? 0x80 : 0);
        }

        Vector128<byte> w = Vector128.Load(tmp);
        return (u8 ? w ^ Vector128.Create((byte)0x80) : w).AsSByte();
    }

    /// <summary>
    /// Packs strips [s0, s1) of B (k rows by n columns, row stride <paramref name="ldb"/>) as <see cref="Int8Gemm.PackB"/> does and
    /// writes the sums of their valid columns to <paramref name="colSums"/> when <paramref name="wantSums"/>. Reads the rows of the
    /// strips' columns; writes Nr * KPad(k) bytes per strip at <paramref name="dst"/> and the valid columns' sums.
    /// </summary>
    public static void PackBStrips(in Int8Format f, byte* src, long ldb, bool u8, int k, int n, byte* dst, int* colSums, bool wantSums, int s0, int s1)
    {
        int nr = f.Nr, g = f.G, kp = f.KPad(k), groups = kp / g, blocks = nr / 16;
        byte* tmp = stackalloc byte[16];
        for (int s = s0; s < s1; s++)
        {
            sbyte* strip = (sbyte*)dst + ((long)s * nr * kp);
            int j0 = s * nr;
            int cols = Math.Min(nr, n - j0);
            for (int blk = 0; blk < blocks; blk++)
            {
                int c0 = blk * 16;
                int cb = Math.Max(0, Math.Min(16, cols - c0));
                sbyte* d = strip + (c0 * g);
                long groupStride = (long)nr * g;
                Sums16 sums = default;
                if (cb == 0)
                {
                    for (int q = 0; q < groups; q++)
                    {
                        Interleave(g, default, default, default, default, d + (q * groupStride));
                    }

                    continue;
                }

                byte* col = src + j0 + c0;
                for (int q = 0; q < groups; q++)
                {
                    int p = q * g;
                    Vector128<sbyte> r0 = default, r1 = default, r2 = default, r3 = default;
                    if (p < k)
                    {
                        r0 = LoadRow(col + (p * ldb), cb, u8, tmp);
                        if (wantSums)
                        {
                            sums.Add(r0);
                        }
                    }

                    if (p + 1 < k)
                    {
                        r1 = LoadRow(col + ((p + 1) * ldb), cb, u8, tmp);
                        if (wantSums)
                        {
                            sums.Add(r1);
                        }
                    }

                    if (g == 4)
                    {
                        if (p + 2 < k)
                        {
                            r2 = LoadRow(col + ((p + 2) * ldb), cb, u8, tmp);
                            if (wantSums)
                            {
                                sums.Add(r2);
                            }
                        }

                        if (p + 3 < k)
                        {
                            r3 = LoadRow(col + ((p + 3) * ldb), cb, u8, tmp);
                            if (wantSums)
                            {
                                sums.Add(r3);
                            }
                        }
                    }

                    Interleave(g, r0, r1, r2, r3, d + (q * groupStride));
                }

                if (wantSums)
                {
                    sums.Store(colSums + j0 + c0, cb);
                }
            }
        }
    }

    /// <summary>
    /// Copies planes [p0, p1) of H x W bytes into zero-point-bordered planes of Hp x Wp signed bytes, already shifted
    /// into the signed domain (uint8 minus 128), so the patch gather needs no bounds checks and no shifting. Reads those planes at
    /// <paramref name="src"/>; writes the same planes, Hp x Wp each, at <paramref name="dst"/>.
    /// </summary>
    public static void PadPlanes(byte* src, bool u8, sbyte pad, int h, int w, int padTop, int padLeft, int padBottom, int padRight, sbyte* dst, int p0, int p1)
    {
        int wp = w + padLeft + padRight, hp = h + padTop + padBottom;
        for (int p = p0; p < p1; p++)
        {
            byte* sp = src + ((long)p * h * w);
            sbyte* dp = dst + ((long)p * hp * wp);
            new Span<sbyte>(dp, padTop * wp).Fill(pad);
            for (int y = 0; y < h; y++)
            {
                sbyte* row = dp + ((padTop + y) * wp);
                new Span<sbyte>(row, padLeft).Fill(pad);
                byte* r = sp + ((long)y * w);
                sbyte* o = row + padLeft;
                int x = 0;
                if (Vector128.IsHardwareAccelerated)
                {
                    Vector128<byte> flip = u8 ? Vector128.Create((byte)0x80) : default;
                    for (; x + 16 <= w; x += 16)
                    {
                        (Vector128.Load(r + x) ^ flip).AsSByte().Store(o + x);
                    }
                }

                for (; x < w; x++)
                {
                    o[x] = (sbyte)Int8Gemm.Shift(r[x], u8);
                }

                new Span<sbyte>(row + padLeft + w, padRight).Fill(pad);
            }

            new Span<sbyte>(dp + ((padTop + h) * wp), padBottom * wp).Fill(pad);
        }
    }

    /// <summary>
    /// Gathers the patches of a convolution with horizontal stride 1 or 2 into packed strips of B straight from the bordered planes made by
    /// <see cref="PadPlanes"/> (Chellapilla et al. 2006, im2col, fused with packing). A block of sixteen consecutive output
    /// positions is a few runs, each inside one output row, and a run of one tap is contiguous in the padded plane: one 16-byte load
    /// per run (two and an even-byte pick for stride 2), masked into place. Needs OutW >= 4 (at most five runs per block) and 32 bytes of slack before and after the
    /// planes (a run's load may start up to 30 bytes before its first lane). Reads inside the padded planes and that slack; writes
    /// Nr * KPad(k) bytes per strip in [s0, s1) and the sums of the valid columns.
    /// </summary>
    public static void Im2ColStrips(in Int8Format f, in ConvShape s, sbyte* planes, int planeSize, int cinG, int k, sbyte* dst, int* colSums, bool wantSums, int s0, int s1)
    {
        int nr = f.Nr, g = f.G, kp = f.KPad(k), groups = kp / g, blocks = nr / 16;
        int outW = s.OutW, nCols = s.OutH * outW;
        int wp = s.W + s.PadLeft + s.PadRight;
        int tapRow = s.DilH * wp, tapCol = s.DilW, sw = s.StrideW;
        int* segOff = stackalloc int[8];
        Vector128<sbyte>* masks = stackalloc Vector128<sbyte>[8];
        Vector128<sbyte> idx = Vector128.Create((sbyte)0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        for (int st = s0; st < s1; st++)
        {
            sbyte* strip = dst + ((long)st * nr * kp);
            int j0 = st * nr;
            for (int blk = 0; blk < blocks; blk++)
            {
                int n0 = j0 + (blk * 16);
                int valid = Math.Min(16, nCols - n0);
                sbyte* d = strip + (blk * 16 * g);
                long groupStride = (long)nr * g;
                if (valid <= 0)
                {
                    for (int q = 0; q < groups; q++)
                    {
                        Interleave(g, default, default, default, default, d + (q * groupStride));
                    }

                    continue;
                }

                int nseg = 0;
                for (int lane = 0; lane < valid;)
                {
                    int n = n0 + lane;
                    int oy = n / outW, ox = n - (oy * outW);
                    int len = Math.Min(outW - ox, valid - lane);
                    segOff[nseg] = (oy * s.StrideH * wp) + ((ox - lane) * sw);
                    masks[nseg] = Vector128.GreaterThanOrEqual(idx, Vector128.Create((sbyte)lane)) & Vector128.LessThan(idx, Vector128.Create((sbyte)(lane + len)));
                    nseg++;
                    lane += len;
                }

                bool whole = nseg == 1 && valid == 16;
                Sums16 sums = default;
                int ci = 0, ky = 0, kx = 0;
                for (int q = 0; q < groups; q++)
                {
                    Vector128<sbyte> r0 = default, r1 = default, r2 = default, r3 = default;
                    for (int j = 0; j < g; j++)
                    {
                        if ((q * g) + j >= k)
                        {
                            break;
                        }

                        sbyte* tap = planes + ((long)ci * planeSize) + (ky * tapRow) + (kx * tapCol);
                        Vector128<sbyte> v;
                        if (whole)
                        {
                            v = sw == 1 ? Vector128.Load(tap + segOff[0]) : Even(tap + segOff[0]);
                        }
                        else
                        {
                            v = default;
                            for (int sg = 0; sg < nseg; sg++)
                            {
                                v = Vector128.ConditionalSelect(masks[sg], sw == 1 ? Vector128.Load(tap + segOff[sg]) : Even(tap + segOff[sg]), v);
                            }
                        }

                        if (wantSums)
                        {
                            sums.Add(v);
                        }

                        if (j == 0)
                        {
                            r0 = v;
                        }
                        else if (j == 1)
                        {
                            r1 = v;
                        }
                        else if (j == 2)
                        {
                            r2 = v;
                        }
                        else
                        {
                            r3 = v;
                        }

                        if (++kx == s.KW)
                        {
                            kx = 0;
                            if (++ky == s.KH)
                            {
                                ky = 0;
                                ci++;
                            }
                        }
                    }

                    Interleave(g, r0, r1, r2, r3, d + (q * groupStride));
                }

                if (wantSums)
                {
                    sums.Store(colSums + n0, valid);
                }
            }
        }
    }
}
