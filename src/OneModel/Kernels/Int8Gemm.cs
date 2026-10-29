using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>The int8 microkernel flavours. Each computes the same exact 32-bit integer sums; they differ in how they get them.</summary>
internal enum Int8Isa
{
    /// <summary>Plain scalar code. Runs everywhere.</summary>
    Scalar = 0,

    /// <summary>AVX2: sign-extend to 16 bits, <c>vpmaddwd</c>.</summary>
    Avx2 = 1,

    /// <summary>AVX-512BW: the same at 512 bits.</summary>
    Avx512 = 2,

    /// <summary>Arm dot product (SDOT).</summary>
    Dot = 3,
}

/// <summary>
/// The register tile, cache blocks and packed layout of an int8 flavour. A and B are packed in groups of <see cref="G"/> consecutive
/// k values (2 for the 16-bit pair multiply-add of x86, 4 for the Arm dot product): inside a k group the packed A holds, for each
/// of the Mr rows, its G values (as 16-bit integers when <see cref="ABytes"/> is 2, else as bytes), and the packed B holds, for each
/// of the Nr columns, its G bytes. Mc and Nc are multiples of Mr and Nr and set the chunk size, so they are constants of the flavour
/// and never depend on the thread count.
/// </summary>
internal readonly struct Int8Format
{
    public Int8Format(Int8Isa isa, int mr, int nr, int mc, int nc, int g, int aBytes)
    {
        Isa = isa;
        Mr = mr;
        Nr = nr;
        Mc = mc;
        Nc = nc;
        G = g;
        ABytes = aBytes;
    }

    public Int8Isa Isa { get; }

    public int Mr { get; }

    public int Nr { get; }

    public int Mc { get; }

    public int Nc { get; }

    public int G { get; }

    public int ABytes { get; }

    /// <summary>k rounded up to a whole number of k groups.</summary>
    public int KPad(int k) => (k + G - 1) / G * G;

    /// <summary>Bytes of A packed for <paramref name="m"/> rows and <paramref name="k"/> columns.</summary>
    public long PackedABytes(int m, int k) => (long)((m + Mr - 1) / Mr) * Mr * KPad(k) * ABytes;

    /// <summary>Bytes of B packed for <paramref name="k"/> rows and <paramref name="n"/> columns.</summary>
    public long PackedBBytes(int k, int n) => (long)((n + Nr - 1) / Nr) * Nr * KPad(k);

    /// <summary>The flavour for a kernel path on this machine: the widest the path allows and the hardware supports.</summary>
    public static Int8Format For(KernelPath path)
    {
        if (path == KernelPath.Avx512 && Avx512BW.IsSupported)
        {
            return new(Int8Isa.Avx512, Int8GemmAvx512.Mr, Int8GemmAvx512.Nr, 48, 64, 2, 2);
        }

        if (path is KernelPath.Avx512 or KernelPath.Avx2 && Avx2.IsSupported)
        {
            return new(Int8Isa.Avx2, Int8GemmAvx2.Mr, Int8GemmAvx2.Nr, 48, 64, 2, 2);
        }

        if (path == KernelPath.Neon && Dp.IsSupported)
        {
            return new(Int8Isa.Dot, Int8GemmDot.Mr, Int8GemmDot.Nr, 48, 64, 4, 1);
        }

        return new(Int8Isa.Scalar, 4, 4, 32, 32, 2, 1);
    }
}

/// <summary>A zero point that is one value for every row (or column) or one value each.</summary>
internal readonly unsafe struct Int8Zero
{
    private readonly int _scalar;
    private readonly int* _vector;

    public Int8Zero(int scalar)
    {
        _scalar = scalar;
        _vector = null;
    }

    public Int8Zero(int* vector)
    {
        _scalar = 0;
        _vector = vector;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int Get(int i) => _vector == null ? _scalar : _vector[i];
}

/// <summary>
/// The A operand: raw 8-bit rows to be packed by each chunk (<see cref="Raw"/>), or rows packed ahead of time with their sums
/// (<see cref="Packed"/>, <see cref="RowSums"/>).
/// </summary>
internal unsafe struct Int8A
{
    public byte* Raw;
    public long Ld;
    public bool U8;
    public byte* Packed;
    public int* RowSums;
}

/// <summary>What happens to each int32 result: stored as is, or scaled, rounded and saturated to 8 bits.</summary>
internal unsafe struct Int8Epilogue
{
    /// <summary>Destination for the int32 results; null selects the 8-bit output.</summary>
    public int* OutInt32;

    /// <summary>Destination for 8-bit results.</summary>
    public byte* OutQuant;

    /// <summary>True: the 8-bit output is uint8, else int8.</summary>
    public bool QuantUnsigned;

    /// <summary>The output zero point, added after rounding.</summary>
    public int QuantZero;

    /// <summary>Row stride of the output, in elements.</summary>
    public long Ldc;

    /// <summary>int32 added to each sum (a bias): one per row, or one per column when <see cref="BiasPerCol"/>; null for none.</summary>
    public int* Bias;

    public bool BiasPerCol;

    /// <summary>The multiplier applied to each sum before rounding, when <see cref="MultVec"/> is null.</summary>
    public float Mult;

    /// <summary>One multiplier per row (<see cref="MultPerRow"/>) or per column, or null.</summary>
    public float* MultVec;

    public bool MultPerRow;
}

/// <summary>
/// Integer matrix product Y = (A - zA) (B - zB) of 8-bit matrices into 32-bit sums, blocked after Goto and van de Geijn (2008) like the
/// float product, and the integer-arithmetic-only inference scheme of Jacob et al. (2018): the zero points are not subtracted in the
/// inner loop. With A' = A - zA' and B' = B - zB' (8-bit signed values, uint8 shifted by 128 so both are int8; zA', zB' the zero
/// points shifted the same way), the inner loop computes S = sum_k A'.B' on the packed int8 data, and the zero points are folded in
/// once per output:
/// <code>Y[i,j] = S[i,j] - zA'[i] * colSum[j] - zB'[j] * (rowSum[i] - K * zA'[i])</code>
/// (all 32-bit two's complement wrapping arithmetic, as exact as the integers themselves).
/// Every microkernel accumulates the exact integer sums, so results are identical on every path, tile shape and thread count.
/// Each output is computed by exactly one chunk. docs/int8.md has the normative text.
/// </summary>
internal static unsafe class Int8Gemm
{
    // k block: a kc x nr strip of packed B (8 KB at kc 512, nr 16) stays in the first-level cache while A strips stream by.
    private const int Kc = 512;
    private const int PackStripsPerChunk = 8;

    [ThreadStatic]
    private static PackBJob? t_packJob;

    [ThreadStatic]
    private static TileJob? t_tileJob;

    /// <summary>
    /// Scales a 32-bit sum into an 8-bit range: round_half_even(acc * mult) + zeroPoint, saturated to [lo, hi]. The float multiply and
    /// the rounding are single correctly rounded operations, so the result is the same on every machine; a NaN (a NaN scale) gives the
    /// zero point.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int Requant(int acc, float mult, int zeroPoint, int lo, int hi)
    {
        float v = MathF.Round((float)acc * mult) + zeroPoint;
        return v < lo ? lo : v > hi ? hi : !float.IsNaN(v) ? (int)v : Math.Clamp(zeroPoint, lo, hi);
    }

    /// <summary>The value of a stored 8-bit element in the shifted signed domain: uint8 minus 128, int8 as is.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int Shift(byte v, bool u8) => u8 ? v - 128 : (sbyte)v;

    /// <summary>
    /// Packs <paramref name="rows"/> rows of A (row-major, row stride <paramref name="lda"/>, <paramref name="k"/> columns, uint8 or
    /// int8) into strips of Mr rows, ragged edges and the k tail padded with zero, and writes each row's sum of shifted values to
    /// <paramref name="rowSums"/>. Reads rows * k bytes at <paramref name="src"/>; writes <see cref="Int8Format.PackedABytes"/>
    /// bytes at <paramref name="dst"/> and <paramref name="rows"/> ints at <paramref name="rowSums"/>.
    /// </summary>
    public static void PackA(in Int8Format f, byte* src, long lda, bool u8, int rows, int k, byte* dst, int* rowSums)
    {
        int kp = f.KPad(k);
        int mr = f.Mr;
        int g = f.G;
        int strips = (rows + mr - 1) / mr;
        for (int s = 0; s < strips; s++)
        {
            byte* strip = dst + ((long)s * mr * kp * f.ABytes);
            for (int r = 0; r < mr; r++)
            {
                int row = (s * mr) + r;
                byte* line = row < rows ? src + (row * lda) : null;
                int sum = 0;
                if (f.ABytes == 2)
                {
                    short* d = (short*)strip;
                    for (int p = 0; p < kp; p++)
                    {
                        int v = line != null && p < k ? Shift(line[p], u8) : 0;
                        sum += v;
                        d[((((p / g) * mr) + r) * g) + (p % g)] = (short)v;
                    }
                }
                else
                {
                    sbyte* d = (sbyte*)strip;
                    for (int p = 0; p < kp; p++)
                    {
                        int v = line != null && p < k ? Shift(line[p], u8) : 0;
                        sum += v;
                        d[((((p / g) * mr) + r) * g) + (p % g)] = (sbyte)v;
                    }
                }

                if (row < rows)
                {
                    rowSums[row] = sum;
                }
            }
        }
    }

    /// <summary>
    /// Packs B (k rows by n columns, row-major, row stride <paramref name="ldb"/>, uint8 or int8) into strips of Nr columns, padded
    /// with zero, and writes each column's sum of shifted values to <paramref name="colSums"/>. Reads the k x n matrix; writes
    /// <see cref="Int8Format.PackedBBytes"/> bytes at <paramref name="dst"/> and n ints at <paramref name="colSums"/>.
    /// </summary>
    public static void PackB(in Int8Format f, byte* src, long ldb, bool u8, int k, int n, byte* dst, int* colSums, int threads)
    {
        int strips = (n + f.Nr - 1) / f.Nr;
        PackBJob job = t_packJob ??= new PackBJob();
        job.F = f;
        job.Src = src;
        job.Ldb = ldb;
        job.U8 = u8;
        job.K = k;
        job.N = n;
        job.Dst = dst;
        job.ColSums = colSums;
        job.Strips = strips;
        ParallelFor.Run(job, (strips + PackStripsPerChunk - 1) / PackStripsPerChunk, threads);
    }

    /// <summary>
    /// Computes the m x n int32 sums for A (m x k) and packed B and applies the zero points and the epilogue. The caller has made
    /// every pointer cover what the formats say; m, n are positive.
    /// </summary>
    /// <param name="f">The flavour; A and B must be packed in it.</param>
    /// <param name="m">Rows of A and of the output.</param>
    /// <param name="n">Columns of B and of the output.</param>
    /// <param name="k">Columns of A and rows of B.</param>
    /// <param name="a">A, raw or packed.</param>
    /// <param name="packedB">B packed by <see cref="PackB"/>.</param>
    /// <param name="colSums">B's column sums from <see cref="PackB"/>.</param>
    /// <param name="zeroA">A's zero point per row, in the shifted domain (zero point minus 128 for uint8).</param>
    /// <param name="zeroB">B's zero point per column, in the shifted domain.</param>
    /// <param name="ep">The output.</param>
    /// <param name="threads">Upper bound on threads; changes speed only.</param>
    public static void Run(in Int8Format f, int m, int n, int k, in Int8A a, byte* packedB, int* colSums, in Int8Zero zeroA,
        in Int8Zero zeroB, in Int8Epilogue ep, int threads)
    {
        TileJob job = t_tileJob ??= new TileJob();
        job.F = f;
        job.M = m;
        job.N = n;
        job.K = k;
        job.A = a;
        job.PackedB = packedB;
        job.ColSums = colSums;
        job.ZeroA = zeroA;
        job.ZeroB = zeroB;
        job.Ep = ep;
        job.TilesN = (n + f.Nc - 1) / f.Nc;
        job.Micro = f.Isa switch
        {
            Int8Isa.Avx512 => &Int8GemmAvx512.Kernel,
            Int8Isa.Avx2 => &Int8GemmAvx2.Kernel,
            Int8Isa.Dot => &Int8GemmDot.Kernel,
            _ => &ScalarKernel,
        };
        int tilesM = (m + f.Mc - 1) / f.Mc;
        ParallelFor.Run(job, tilesM * job.TilesN, threads);
    }

    // The reference microkernel: a 4x4 tile over k pairs, A as bytes (k pair per row), B as bytes (k pair per column).
    private static void ScalarKernel(int kg, byte* a, byte* b, int* c, nint ldc, int accumulate)
    {
        int* acc = stackalloc int[16];
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                acc[(i * 4) + j] = accumulate != 0 ? c[(i * ldc) + j] : 0;
            }
        }

        for (int p = 0; p < kg; p++)
        {
            for (int i = 0; i < 4; i++)
            {
                int a0 = (sbyte)a[(i * 2)];
                int a1 = (sbyte)a[(i * 2) + 1];
                for (int j = 0; j < 4; j++)
                {
                    acc[(i * 4) + j] += (a0 * (sbyte)b[j * 2]) + (a1 * (sbyte)b[(j * 2) + 1]);
                }
            }

            a += 8;
            b += 8;
        }

        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                c[(i * ldc) + j] = acc[(i * 4) + j];
            }
        }
    }

    // Per-thread scratch: packed A slab, its row sums, and the int32 tile.
    private static class Slab
    {
        [ThreadStatic]
        private static void*[]? t_ptr;

        [ThreadStatic]
        private static long[]? t_len;

        public static byte* Get(int slot, long bytes)
        {
            t_ptr ??= new void*[3];
            t_len ??= new long[3];
            if (bytes < 64)
            {
                bytes = 64;
            }

            if (t_len[slot] < bytes)
            {
                if (t_ptr[slot] != null)
                {
                    NativeMemory.AlignedFree(t_ptr[slot]);
                }

                long rounded = (bytes + 4095) & ~4095L;
                t_ptr[slot] = NativeMemory.AlignedAlloc((nuint)rounded, 64);
                t_len[slot] = rounded;
            }

            return (byte*)t_ptr[slot];
        }
    }

    private sealed class PackBJob : ChunkJob
    {
        public Int8Format F;
        public byte* Src;
        public long Ldb;
        public bool U8;
        public int K;
        public int N;
        public byte* Dst;
        public int* ColSums;
        public int Strips;

        public override void RunChunk(int chunk)
        {
            int s0 = chunk * PackStripsPerChunk;
            int s1 = Math.Min(Strips, s0 + PackStripsPerChunk);
            int nr = F.Nr;
            int g = F.G;
            int kp = F.KPad(K);
            int* sums = stackalloc int[64];
            for (int s = s0; s < s1; s++)
            {
                byte* strip = Dst + ((long)s * nr * kp);
                int j0 = s * nr;
                int cols = Math.Min(nr, N - j0);
                for (int c = 0; c < nr; c++)
                {
                    sums[c] = 0;
                }

                for (int p = 0; p < kp; p++)
                {
                    sbyte* d = (sbyte*)strip + ((p / g) * nr * g) + (p % g);
                    if (p < K)
                    {
                        byte* line = Src + (p * Ldb) + j0;
                        for (int c = 0; c < cols; c++)
                        {
                            int v = Shift(line[c], U8);
                            sums[c] += v;
                            d[c * g] = (sbyte)v;
                        }

                        for (int c = cols; c < nr; c++)
                        {
                            d[c * g] = 0;
                        }
                    }
                    else
                    {
                        for (int c = 0; c < nr; c++)
                        {
                            d[c * g] = 0;
                        }
                    }
                }

                for (int c = 0; c < cols; c++)
                {
                    ColSums[j0 + c] = sums[c];
                }
            }
        }
    }

    private sealed class TileJob : ChunkJob
    {
        public Int8Format F;
        public int M;
        public int N;
        public int K;
        public Int8A A;
        public byte* PackedB;
        public int* ColSums;
        public Int8Zero ZeroA;
        public Int8Zero ZeroB;
        public Int8Epilogue Ep;
        public int TilesN;
        public delegate*<int, byte*, byte*, int*, nint, int, void> Micro;

        public override void RunChunk(int chunk)
        {
            Int8Format f = F;
            int i0 = (chunk / TilesN) * f.Mc;
            int j0 = (chunk % TilesN) * f.Nc;
            int mc = Math.Min(f.Mc, M - i0);
            int nc = Math.Min(f.Nc, N - j0);
            int mStrips = (mc + f.Mr - 1) / f.Mr;
            int nStrips = (nc + f.Nr - 1) / f.Nr;
            int kp = f.KPad(K);
            long stripA = (long)f.Mr * kp * f.ABytes;
            long stripB = (long)f.Nr * kp;
            int* tile = (int*)Slab.Get(2, (long)f.Mc * f.Nc * sizeof(int));

            byte* aBase;
            int* rowSums;
            if (A.Packed != null)
            {
                aBase = A.Packed + ((i0 / f.Mr) * stripA);
                rowSums = A.RowSums + i0;
            }
            else
            {
                aBase = Slab.Get(0, mStrips * stripA);
                rowSums = (int*)Slab.Get(1, f.Mc * sizeof(int));
                PackA(in f, A.Raw + (i0 * A.Ld), A.Ld, A.U8, mc, K, aBase, rowSums);
            }

            byte* bBase = PackedB + ((j0 / f.Nr) * stripB);
            if (kp == 0)
            {
                new Span<int>(tile, f.Mc * f.Nc).Clear();
            }

            for (int pc = 0; pc < kp; pc += Kc)
            {
                int kgCount = Math.Min(Kc, kp - pc) / f.G;
                long aOff = (long)(pc / f.G) * f.Mr * f.G * f.ABytes;
                long bOff = (long)(pc / f.G) * f.Nr * f.G;
                int accumulate = pc > 0 ? 1 : 0;
                for (int js = 0; js < nStrips; js++)
                {
                    byte* bp = bBase + (js * stripB) + bOff;
                    for (int isr = 0; isr < mStrips; isr++)
                    {
                        byte* ap = aBase + (isr * stripA) + aOff;
                        Micro(kgCount, ap, bp, tile + (isr * f.Mr * f.Nc) + (js * f.Nr), f.Nc, accumulate);
                    }
                }
            }

            Epilogue(tile, f.Nc, i0, j0, mc, nc, rowSums);
        }

        private void Epilogue(int* tile, int ldt, int i0, int j0, int mc, int nc, int* rowSums)
        {
            ref readonly Int8Epilogue ep = ref Ep;
            for (int i = 0; i < mc; i++)
            {
                int gi = i0 + i;
                int za = ZeroA.Get(gi);
                int t = unchecked(rowSums[i] - (K * za));
                int biasRow = ep.Bias != null && !ep.BiasPerCol ? ep.Bias[gi] : 0;
                int* row = tile + (i * ldt);
                if (ep.OutInt32 != null)
                {
                    int* dst = ep.OutInt32 + (gi * ep.Ldc) + j0;
                    for (int j = 0; j < nc; j++)
                    {
                        int gj = j0 + j;
                        int bias = ep.BiasPerCol ? ep.Bias[gj] : biasRow;
                        dst[j] = unchecked(row[j] - (za * ColSums[gj]) - (ZeroB.Get(gj) * t) + bias);
                    }

                    continue;
                }

                byte* q = ep.OutQuant + (gi * ep.Ldc) + j0;
                float mRow = ep.MultVec != null && ep.MultPerRow ? ep.MultVec[gi] : ep.Mult;
                int lo = ep.QuantUnsigned ? 0 : -128;
                int hi = ep.QuantUnsigned ? 255 : 127;
                for (int j = 0; j < nc; j++)
                {
                    int gj = j0 + j;
                    int bias = ep.BiasPerCol ? ep.Bias[gj] : biasRow;
                    int acc = unchecked(row[j] - (za * ColSums[gj]) - (ZeroB.Get(gj) * t) + bias);
                    float mult = ep.MultVec != null && !ep.MultPerRow ? ep.MultVec[gj] : mRow;
                    q[j] = (byte)Requant(acc, mult, ep.QuantZero, lo, hi);
                }
            }
        }
    }
}
