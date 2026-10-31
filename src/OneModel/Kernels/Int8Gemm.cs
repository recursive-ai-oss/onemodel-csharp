using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
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

/// <summary>
/// Produces strips of packed B on demand, on the thread that multiplies them (so a convolution gathers its patches straight into the
/// cache of the core that uses them, and the gather runs in parallel with the product instead of before it).
/// </summary>
internal abstract unsafe class Int8BPanelSource
{
    /// <summary>
    /// Packs strips [<paramref name="firstStrip"/>, <paramref name="firstStrip"/> + <paramref name="strips"/>) of B, in the layout of the
    /// format, to <paramref name="dst"/> (strip after strip, Nr * KPad bytes each) and the sums of their valid columns to
    /// <paramref name="sums"/> (Nr per strip, in strip order) when the caller needs them.
    /// </summary>
    public abstract void Pack(int firstStrip, int strips, byte* dst, int* sums);
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

    public bool IsScalar => _vector == null;

    public int Scalar => _scalar;

    public int* Vector => _vector;

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

    /// <summary>Destination for float32 results (sum * scale, plus <see cref="BiasFloat"/>); used when <see cref="OutInt32"/> is null.</summary>
    public float* OutFloat;

    /// <summary>A float32 bias per column added after the scaling of a float32 output, or null.</summary>
    public float* BiasFloat;

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
    private const int PackAStripsPerChunk = 4;

    [ThreadStatic]
    private static PackBJob? t_packJob;

    [ThreadStatic]
    private static TileJob? t_tileJob;

    [ThreadStatic]
    private static PackAJob? t_packAJob;

    /// <summary>
    /// Scales a 32-bit sum into an 8-bit range: round_half_even(acc * mult) + zeroPoint, saturated to [lo, hi]. The float multiply and
    /// the rounding are single correctly rounded operations, so the result is the same on every machine; a NaN (a NaN scale) gives the
    /// smallest value, as the vector path does.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int Requant(int acc, float mult, int zeroPoint, int lo, int hi)
    {
        float v = MathF.Round((float)acc * mult) + zeroPoint;
        return v < lo ? lo : v > hi ? hi : !float.IsNaN(v) ? (int)v : lo;
    }

    /// <summary>
    /// As <see cref="RequantRow"/> for <paramref name="n"/> a multiple of 8 when the caller can spare the bytes: writes exactly n bytes
    /// and reads exactly n ints, with no scalar tail. A caller that writes the rows of a plane in order uses it with n a little past the
    /// row's width; the next row overwrites the extra bytes.
    /// </summary>
    internal static void RequantRowPadded(int* row, byte* q, int n, float mult, int zeroPoint, bool unsigned)
    {
        int lo = unsigned ? 0 : -128;
        int hi = unsigned ? 255 : 127;
        Vector128<float> vm = Vector128.Create(mult), vz = Vector128.Create((float)zeroPoint);
        Vector128<float> vlo = Vector128.Create((float)lo), vhi = Vector128.Create((float)hi);
        int j = 0;
        for (; j + 16 <= n; j += 16)
        {
            Vector128<int> q0 = Quant4(row + j, vm, vz, vlo, vhi);
            Vector128<int> q1 = Quant4(row + j + 4, vm, vz, vlo, vhi);
            Vector128<int> q2 = Quant4(row + j + 8, vm, vz, vlo, vhi);
            Vector128<int> q3 = Quant4(row + j + 12, vm, vz, vlo, vhi);
            Int8Quantize.Pack16(q0, q1, q2, q3, unsigned).Store(q + j);
        }

        if (j < n)
        {
            Vector128<int> q0 = Quant4(row + j, vm, vz, vlo, vhi);
            Vector128<int> q1 = Quant4(row + j + 4, vm, vz, vlo, vhi);
            Int8Quantize.Pack16(q0, q1, q0, q1, unsigned).GetLower().Store(q + j);
        }
    }

    /// <summary>
    /// Requantizes <paramref name="n"/> sums with one multiplier: <see cref="Requant"/> on every element, four lanes at a time where
    /// the vectors are hardware accelerated. Writes n bytes at <paramref name="q"/>; reads n ints at <paramref name="row"/> rounded up to a
    /// multiple of 16 (the values past n are converted and dropped, so the memory must be readable but may hold anything).
    /// </summary>
    internal static void RequantRow(int* row, byte* q, int n, float mult, int zeroPoint, bool unsigned)
    {
        int lo = unsigned ? 0 : -128;
        int hi = unsigned ? 255 : 127;
        int j = 0;
        if (Vector128.IsHardwareAccelerated)
        {
            Vector128<float> vm = Vector128.Create(mult), vz = Vector128.Create((float)zeroPoint);
            Vector128<float> vlo = Vector128.Create((float)lo), vhi = Vector128.Create((float)hi);
            for (; j + 16 <= n; j += 16)
            {
                Vector128<int> q0 = Quant4(row + j, vm, vz, vlo, vhi);
                Vector128<int> q1 = Quant4(row + j + 4, vm, vz, vlo, vhi);
                Vector128<int> q2 = Quant4(row + j + 8, vm, vz, vlo, vhi);
                Vector128<int> q3 = Quant4(row + j + 12, vm, vz, vlo, vhi);
                Int8Quantize.Pack16(q0, q1, q2, q3, unsigned).Store(q + j);
            }

            if (j < n)
            {
                // The tail through one more vector step and a 16-byte stack copy: sixteen ints are read from the row (the caller
                // guarantees that many past the last block, see the remarks), only n - j bytes are written.
                Vector128<int> q0 = Quant4(row + j, vm, vz, vlo, vhi);
                Vector128<int> q1 = Quant4(row + j + 4, vm, vz, vlo, vhi);
                Vector128<int> q2 = Quant4(row + j + 8, vm, vz, vlo, vhi);
                Vector128<int> q3 = Quant4(row + j + 12, vm, vz, vlo, vhi);
                byte* tmp = stackalloc byte[16];
                Int8Quantize.Pack16(q0, q1, q2, q3, unsigned).Store(tmp);
                for (int i = 0; j + i < n; i++)
                {
                    q[j + i] = tmp[i];
                }

                return;
            }
        }

        for (; j < n; j++)
        {
            q[j] = (byte)Requant(row[j], mult, zeroPoint, lo, hi);
        }
    }

    private static Vector128<int> QuantSum(Vector128<int> sum, Vector128<float> mult, Vector128<float> zero, Vector128<float> lo, Vector128<float> hi)
    {
        Vector128<float> v = Vector128.Round(Vector128.ConvertToSingle(sum) * mult) + zero;
        v = Vector128.ConditionalSelect(Vector128.Equals(v, v), v, lo);
        return Vector128.ConvertToInt32(Vector128.Min(Vector128.Max(v, lo), hi));
    }

    private static Vector128<int> Quant4(int* p, Vector128<float> mult, Vector128<float> zero, Vector128<float> lo, Vector128<float> hi)
    {
        Vector128<float> v = Vector128.Round(Vector128.ConvertToSingle(Vector128.Load(p)) * mult) + zero;
        v = Vector128.ConditionalSelect(Vector128.Equals(v, v), v, lo);
        return Vector128.ConvertToInt32(Vector128.Min(Vector128.Max(v, lo), hi));
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
        int strips = (rows + f.Mr - 1) / f.Mr;
        PackAStrips(in f, src, lda, u8, rows, k, dst, rowSums, 0, strips);
    }

    // Packs strips [s0, s1) of A. The loops run k group by k group so every write is sequential and nothing divides in the inner loop.
    private static void PackAStrips(in Int8Format f, byte* src, long lda, bool u8, int rows, int k, byte* dst, int* rowSums, int s0, int s1)
    {
        int kp = f.KPad(k);
        int mr = f.Mr;
        int g = f.G;
        int groups = kp / g;
        int* sums = stackalloc int[16];
        for (int s = s0; s < s1; s++)
        {
            byte* strip = dst + ((long)s * mr * kp * f.ABytes);
            for (int r = 0; r < mr; r++)
            {
                sums[r] = 0;
            }

            for (int r = 0; r < mr; r++)
            {
                int row = (s * mr) + r;
                byte* line = row < rows ? src + (row * lda) : null;
                int sum = 0;
                if (f.ABytes == 2)
                {
                    short* d = (short*)strip + (r * g);
                    for (int q = 0; q < groups; q++)
                    {
                        int p = q * g;
                        for (int j = 0; j < g; j++)
                        {
                            int v = line != null && p + j < k ? Shift(line[p + j], u8) : 0;
                            sum += v;
                            d[j] = (short)v;
                        }

                        d += mr * g;
                    }
                }
                else
                {
                    sbyte* d = (sbyte*)strip + (r * g);
                    for (int q = 0; q < groups; q++)
                    {
                        int p = q * g;
                        for (int j = 0; j < g; j++)
                        {
                            int v = line != null && p + j < k ? Shift(line[p + j], u8) : 0;
                            sum += v;
                            d[j] = (sbyte)v;
                        }

                        d += mr * g;
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
    public static void PackB(in Int8Format f, byte* src, long ldb, bool u8, int k, int n, byte* dst, int* colSums, int threads, bool wantSums = true)
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
        job.WantSums = wantSums;
        job.Vector = Int8Pack.Supported(in f);
        // Work estimates are nanoseconds on one thread and only decide how many threads are worth waking: about 1 ns to pack an element
        // of B with the scalar packer, a tenth of that with the vector one.
        ParallelFor.Run(job, (strips + PackStripsPerChunk - 1) / PackStripsPerChunk, threads, job.Vector ? (long)k * n / 8 : (long)k * n);
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
    /// <param name="source">When not null, B is packed on demand by this source and <paramref name="packedB"/> and <paramref name="colSums"/> are not read.</param>
    public static void Run(in Int8Format f, int m, int n, int k, in Int8A a, byte* packedB, int* colSums, in Int8Zero zeroA,
        in Int8Zero zeroB, in Int8Epilogue ep, int threads, Int8BPanelSource? source = null)
    {
        Int8A a2 = a;
        if (a.Packed == null)
        {
            // Pack all of A once, in parallel, so the tile chunks (one per column block) do not each repack their rows.
            int aStrips = (m + f.Mr - 1) / f.Mr;
            byte* pa = Slab.Get(3, f.PackedABytes(m, k));
            int* rs = (int*)Slab.Get(4, (long)m * sizeof(int));
            PackAJob pj = t_packAJob ??= new PackAJob();
            pj.F = f;
            pj.Src = a.Raw;
            pj.Lda = a.Ld;
            pj.U8 = a.U8;
            pj.Rows = m;
            pj.K = k;
            pj.Dst = pa;
            pj.RowSums = rs;
            pj.Strips = aStrips;
            ParallelFor.Run(pj, (aStrips + PackAStripsPerChunk - 1) / PackAStripsPerChunk, threads, (long)m * k / 2);
            a2 = default;
            a2.Packed = pa;
            a2.RowSums = rs;
        }

        TileJob job = t_tileJob ??= new TileJob();
        job.F = f;
        job.M = m;
        job.N = n;
        job.K = k;
        job.A = a2;
        job.PackedB = packedB;
        job.ColSums = colSums;
        job.ZeroA = zeroA;
        job.ZeroB = zeroB;
        job.Ep = ep;
        job.TilesN = (n + f.Nc - 1) / f.Nc;
        job.Source = source;
        job.Micro = f.Isa switch
        {
            Int8Isa.Avx512 => &Int8GemmAvx512.Kernel,
            Int8Isa.Avx2 => &Int8GemmAvx2.Kernel,
            Int8Isa.Dot => &Int8GemmDot.Kernel,
            _ => &ScalarKernel,
        };
        int tilesM = (m + f.Mc - 1) / f.Mc;
        job.TilesM = tilesM;
        // With an on-demand B a chunk packs its panel of columns once and multiplies it with a group of row tiles. Packing a panel costs a
        // good fraction of one tile's product, so the groups are as large as can be while there are still two chunks per usable thread.
        // The choice moves work between chunks and never changes a result (every output is one exact integer computed by one chunk).
        // Plus about half a nanosecond per output for the epilogue, which is most of the time when k is small.
        long work = ((long)m * n * k / 50) + ((long)m * n / 2);
        int usable = (int)Math.Clamp(work / ParallelFor.MinWorkPerThread, 1, Math.Max(1, threads));
        int group = tilesM;
        while (group > 1 && ((tilesM + group - 1) / group * job.TilesN) < 2 * usable)
        {
            group--;
        }

        job.GroupTiles = source == null ? 1 : group;
        int groups = (tilesM + job.GroupTiles - 1) / job.GroupTiles;
        // About 50 multiply-adds per nanosecond per thread at the measured 100 giga-operations per second.
        ParallelFor.Run(job, groups * job.TilesN, threads, work);
        job.Source = null;
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
            t_ptr ??= new void*[7];
            t_len ??= new long[7];
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
        public bool WantSums;
        public bool Vector;

        public override void RunChunk(int chunk)
        {
            int s0 = chunk * PackStripsPerChunk;
            int s1 = Math.Min(Strips, s0 + PackStripsPerChunk);
            if (Vector)
            {
                Int8Pack.PackBStrips(in F, Src, Ldb, U8, K, N, Dst, ColSums, WantSums, s0, s1);
                return;
            }

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

    private sealed class PackAJob : ChunkJob
    {
        public Int8Format F;
        public byte* Src;
        public long Lda;
        public bool U8;
        public int Rows;
        public int K;
        public byte* Dst;
        public int* RowSums;
        public int Strips;

        public override void RunChunk(int chunk)
        {
            int s0 = chunk * PackAStripsPerChunk;
            PackAStrips(in F, Src, Lda, U8, Rows, K, Dst, RowSums, s0, Math.Min(Strips, s0 + PackAStripsPerChunk));
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
        public int TilesM;
        public int GroupTiles = 1;
        public Int8BPanelSource? Source;
        public delegate*<int, byte*, byte*, int*, nint, int, void> Micro;

        public override void RunChunk(int chunk)
        {
            Int8Format f = F;
            int j0 = (chunk % TilesN) * f.Nc;
            int nc = Math.Min(f.Nc, N - j0);
            int kp = f.KPad(K);
            long stripB = (long)f.Nr * kp;
            byte* bBase;
            int* colSums;
            if (Source != null)
            {
                int nStr = (nc + f.Nr - 1) / f.Nr;
                bBase = Slab.Get(5, nStr * stripB);
                colSums = (int*)Slab.Get(6, (long)nStr * f.Nr * sizeof(int));
                Source.Pack(j0 / f.Nr, nStr, bBase, colSums);
            }
            else
            {
                bBase = PackedB + ((j0 / f.Nr) * stripB);
                colSums = ColSums + j0;
            }

            int t0 = (chunk / TilesN) * GroupTiles;
            int t1 = Math.Min(TilesM, t0 + GroupTiles);
            for (int t = t0; t < t1; t++)
            {
                Tile(t * f.Mc, j0, nc, bBase, colSums);
            }
        }

        // One tile of rows i0.. by the columns j0.. of the panel at bBase (whose first strip holds column j0); colSums[0] is column j0's sum.
        private void Tile(int i0, int j0, int nc, byte* bBase, int* colSums)
        {
            Int8Format f = F;
            int mc = Math.Min(f.Mc, M - i0);
            int mStrips = (mc + f.Mr - 1) / f.Mr;
            int nStrips = (nc + f.Nr - 1) / f.Nr;
            int kp = f.KPad(K);
            long stripA = (long)f.Mr * kp * f.ABytes;
            long stripB = (long)f.Nr * kp;
            int* tile = (int*)Slab.Get(2, (long)f.Mc * f.Nc * sizeof(int));

            byte* aBase = A.Packed + ((i0 / f.Mr) * stripA);
            int* rowSums = A.RowSums + i0;

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
                if (nc == 1 && f.Isa == Int8Isa.Dot)
                {
                    // A matrix-vector product: the sixteen-column kernel would spend fifteen sixteenths of its work on padding.
                    Int8GemmDot.Gemv(kgCount, aBase + aOff, stripA, mStrips, bBase + bOff, tile, f.Nc, accumulate);
                    continue;
                }

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

            Epilogue(tile, f.Nc, i0, j0, mc, nc, rowSums, colSums);
        }

        // Applies the zero points and bias to the finished sums of a tile (in place, four lanes at a time), then writes them as int32,
        // float32 (sum * scale [+ bias]) or scaled, rounded and saturated 8-bit values.
        private void Epilogue(int* tile, int ldt, int i0, int j0, int mc, int nc, int* rowSums, int* colSums)
        {
            ref readonly Int8Epilogue ep = ref Ep;
            for (int i = 0; i < mc; i++)
            {
                int gi = i0 + i;
                int za = ZeroA.Get(gi);
                int t = unchecked(rowSums[i] - (K * za));
                int biasRow = ep.Bias != null && !ep.BiasPerCol ? ep.Bias[gi] : 0;
                int* row = tile + (i * ldt);
                if (za == 0 && ZeroB.IsScalar && !ep.BiasPerCol && Vector128.IsHardwareAccelerated)
                {
                    // With a zero A zero point (symmetric weights) the correction is one constant per row, folded into the bias: one pass
                    // from the sums to the output instead of two.
                    FusedRow(row, nc, gi, j0, unchecked(biasRow - (ZeroB.Scalar * t)));
                    continue;
                }

                int* cs = colSums;
                int j = 0;
                if (Vector128.IsHardwareAccelerated)
                {
                    Vector128<int> vza = Vector128.Create(za), vt = Vector128.Create(t), vbr = Vector128.Create(biasRow);
                    Vector128<int> vzbs = Vector128.Create(ZeroB.Scalar);
                    for (; j + 4 <= nc; j += 4)
                    {
                        Vector128<int> zb = ZeroB.IsScalar ? vzbs : Vector128.Load(ZeroB.Vector + j0 + j);
                        Vector128<int> bias = ep.BiasPerCol ? Vector128.Load(ep.Bias + j0 + j) : vbr;
                        Vector128<int> acc = Vector128.Load(row + j) - (vza * Vector128.Load(cs + j)) - (zb * vt) + bias;
                        acc.Store(row + j);
                    }
                }

                for (; j < nc; j++)
                {
                    int bias = ep.BiasPerCol ? ep.Bias[j0 + j] : biasRow;
                    row[j] = unchecked(row[j] - (za * cs[j]) - (ZeroB.Get(j0 + j) * t) + bias);
                }

                if (ep.OutInt32 != null)
                {
                    new ReadOnlySpan<int>(row, nc).CopyTo(new Span<int>(ep.OutInt32 + (gi * ep.Ldc) + j0, nc));
                }
                else if (ep.OutFloat != null)
                {
                    StoreFloat(row, ep.OutFloat + (gi * ep.Ldc) + j0, nc, j0, gi);
                }
                else
                {
                    StoreQuant(row, ep.OutQuant + (gi * ep.Ldc) + j0, nc, j0, gi);
                }
            }
        }

        // One row of a tile whose only correction is the constant c: out = f(sum + c), with the same operations as the two-pass route.
        private void FusedRow(int* row, int nc, int gi, int j0, int c)
        {
            ref readonly Int8Epilogue ep = ref Ep;
            Vector128<int> vc = Vector128.Create(c);
            int j = 0;
            if (ep.OutInt32 != null)
            {
                int* dst = ep.OutInt32 + (gi * ep.Ldc) + j0;
                for (; j + 4 <= nc; j += 4)
                {
                    (Vector128.Load(row + j) + vc).Store(dst + j);
                }

                for (; j < nc; j++)
                {
                    dst[j] = unchecked(row[j] + c);
                }

                return;
            }

            float mRow = ep.MultVec != null && ep.MultPerRow ? ep.MultVec[gi] : ep.Mult;
            bool perCol = ep.MultVec != null && !ep.MultPerRow;
            if (ep.OutFloat != null)
            {
                float* dst = ep.OutFloat + (gi * ep.Ldc) + j0;
                Vector128<float> vm = Vector128.Create(mRow);
                for (; j + 4 <= nc; j += 4)
                {
                    Vector128<float> m = perCol ? Vector128.Load(ep.MultVec + j0 + j) : vm;
                    Vector128<float> v = Vector128.ConvertToSingle(Vector128.Load(row + j) + vc) * m;
                    if (ep.BiasFloat != null)
                    {
                        v += Vector128.Load(ep.BiasFloat + j0 + j);
                    }

                    v.Store(dst + j);
                }

                for (; j < nc; j++)
                {
                    float v = (float)unchecked(row[j] + c) * (perCol ? ep.MultVec[j0 + j] : mRow);
                    dst[j] = ep.BiasFloat != null ? v + ep.BiasFloat[j0 + j] : v;
                }

                return;
            }

            byte* q = ep.OutQuant + (gi * ep.Ldc) + j0;
            int lo = ep.QuantUnsigned ? 0 : -128;
            int hi = ep.QuantUnsigned ? 255 : 127;
            Vector128<float> vms = Vector128.Create(mRow), vz = Vector128.Create((float)ep.QuantZero);
            Vector128<float> vlo = Vector128.Create((float)lo), vhi = Vector128.Create((float)hi);
            for (; j + 16 <= nc; j += 16)
            {
                Vector128<int> q0 = QuantSum(Vector128.Load(row + j) + vc, perCol ? Vector128.Load(ep.MultVec + j0 + j) : vms, vz, vlo, vhi);
                Vector128<int> q1 = QuantSum(Vector128.Load(row + j + 4) + vc, perCol ? Vector128.Load(ep.MultVec + j0 + j + 4) : vms, vz, vlo, vhi);
                Vector128<int> q2 = QuantSum(Vector128.Load(row + j + 8) + vc, perCol ? Vector128.Load(ep.MultVec + j0 + j + 8) : vms, vz, vlo, vhi);
                Vector128<int> q3 = QuantSum(Vector128.Load(row + j + 12) + vc, perCol ? Vector128.Load(ep.MultVec + j0 + j + 12) : vms, vz, vlo, vhi);
                Int8Quantize.Pack16(q0, q1, q2, q3, ep.QuantUnsigned).Store(q + j);
            }

            for (; j < nc; j++)
            {
                q[j] = (byte)Requant(unchecked(row[j] + c), perCol ? ep.MultVec[j0 + j] : mRow, ep.QuantZero, lo, hi);
            }
        }

        private void StoreFloat(int* row, float* dst, int nc, int j0, int gi)
        {
            ref readonly Int8Epilogue ep = ref Ep;
            // The same two correctly rounded operations an unfused Cast, Mul (and Add) would do, so the bits are the same.
            float mRow = ep.MultVec != null && ep.MultPerRow ? ep.MultVec[gi] : ep.Mult;
            bool perCol = ep.MultVec != null && !ep.MultPerRow;
            int j = 0;
            if (Vector128.IsHardwareAccelerated)
            {
                Vector128<float> vm = Vector128.Create(mRow);
                for (; j + 4 <= nc; j += 4)
                {
                    Vector128<float> m = perCol ? Vector128.Load(ep.MultVec + j0 + j) : vm;
                    Vector128<float> v = Vector128.ConvertToSingle(Vector128.Load(row + j)) * m;
                    if (ep.BiasFloat != null)
                    {
                        v += Vector128.Load(ep.BiasFloat + j0 + j);
                    }

                    v.Store(dst + j);
                }
            }

            for (; j < nc; j++)
            {
                float v = (float)row[j] * (perCol ? ep.MultVec[j0 + j] : mRow);
                dst[j] = ep.BiasFloat != null ? v + ep.BiasFloat[j0 + j] : v;
            }
        }

        private void StoreQuant(int* row, byte* q, int nc, int j0, int gi)
        {
            ref readonly Int8Epilogue ep = ref Ep;
            float mRow = ep.MultVec != null && ep.MultPerRow ? ep.MultVec[gi] : ep.Mult;
            bool perCol = ep.MultVec != null && !ep.MultPerRow;
            int lo = ep.QuantUnsigned ? 0 : -128;
            int hi = ep.QuantUnsigned ? 255 : 127;
            int j = 0;
            if (Vector128.IsHardwareAccelerated)
            {
                Vector128<float> vm = Vector128.Create(mRow), vz = Vector128.Create((float)ep.QuantZero);
                Vector128<float> vlo = Vector128.Create((float)lo), vhi = Vector128.Create((float)hi);
                for (; j + 16 <= nc; j += 16)
                {
                    Vector128<int> q0 = Quant4(row + j, perCol ? Vector128.Load(ep.MultVec + j0 + j) : vm, vz, vlo, vhi);
                    Vector128<int> q1 = Quant4(row + j + 4, perCol ? Vector128.Load(ep.MultVec + j0 + j + 4) : vm, vz, vlo, vhi);
                    Vector128<int> q2 = Quant4(row + j + 8, perCol ? Vector128.Load(ep.MultVec + j0 + j + 8) : vm, vz, vlo, vhi);
                    Vector128<int> q3 = Quant4(row + j + 12, perCol ? Vector128.Load(ep.MultVec + j0 + j + 12) : vm, vz, vlo, vhi);
                    Int8Quantize.Pack16(q0, q1, q2, q3, ep.QuantUnsigned).Store(q + j);
                }
            }

            for (; j < nc; j++)
            {
                q[j] = (byte)Requant(row[j], perCol ? ep.MultVec[j0 + j] : mRow, ep.QuantZero, lo, hi);
            }
        }
    }
}
