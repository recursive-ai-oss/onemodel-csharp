using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using RecursiveAI.OneModel.Kernels.Exact;
using RecursiveAI.OneModel.Kernels.Fast;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Single-precision matrix product C = A * B on row-major matrices, blocked after Goto and van de Geijn (2008): B is packed into
/// kc x nr strips, A into mc x kc blocks of mr-row strips, and a register-tiled microkernel computes mr x nr tiles of C from them.
/// Work is split into mc x nc tiles of C; a tile is one chunk of a <see cref="ParallelFor"/>, and each output is computed by exactly
/// one chunk, in one fixed order over k, so the result never depends on the thread count.
/// </summary>
/// <remarks>
/// The k loop is never split across threads. Within an output the products are accumulated in increasing k, continuing from the
/// stored partial sum across k blocks, so even the k block size does not change a bit. docs/kernels-gemm.md and docs/exact-mode.md
/// give the canonical order.
/// </remarks>
internal static unsafe class Gemm
{
    // Cache blocking (Goto and van de Geijn 2008, section 4.2): a kc x nr strip of packed B (16 KB at kc 256, nr 16) stays in the
    // first-level cache while the microkernel sweeps the mc x kc block of A (about 100 KB) from the second-level cache.
    private const int Kc = 256;

    // Strips of B packed per chunk of the packing phase.
    private const int PackStripsPerChunk = 8;

    /// <summary>
    /// Computes C[m x n] = A[m x k] * B[k x n]. C is overwritten; with k = 0 it is filled with +0. In exact mode every NaN in C is
    /// replaced by the quiet NaN 0x7FC00000, because hardware disagrees about NaN payloads and signs.
    /// </summary>
    /// <param name="mode">Fast or exact.</param>
    /// <param name="path">The instruction set; must be supported.</param>
    /// <param name="m">Rows of A and C, at least 0.</param>
    /// <param name="n">Columns of B and C, at least 0.</param>
    /// <param name="k">Columns of A and rows of B, at least 0.</param>
    /// <param name="a">A, row-major, row stride <paramref name="lda"/> (at least k).</param>
    /// <param name="lda">Row stride of A.</param>
    /// <param name="b">B, row-major, row stride <paramref name="ldb"/> (at least n).</param>
    /// <param name="ldb">Row stride of B.</param>
    /// <param name="c">C, row-major, row stride <paramref name="ldc"/> (at least n). Must not overlap A or B.</param>
    /// <param name="ldc">Row stride of C.</param>
    /// <param name="threads">Upper bound on threads, at least 1. Changes speed only.</param>
    public static void Run(
        KernelMode mode, KernelPath path, int m, int n, int k,
        ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc, int threads)
    {
        if (m < 0 || n < 0 || k < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(m), "Matrix dimensions must not be negative.");
        }

        if (!KernelPaths.IsSupported(path))
        {
            throw new PlatformNotSupportedException($"Kernel path {path} is not supported on this machine.");
        }

        if (m == 0 || n == 0)
        {
            return;
        }

        if (ldc < n || (long)(m - 1) * ldc + n > c.Length)
        {
            throw new ArgumentException("C is smaller than m x n with its row stride.", nameof(c));
        }

        if (k > 0)
        {
            if (lda < k || (long)(m - 1) * lda + k > a.Length)
            {
                throw new ArgumentException("A is smaller than m x k with its row stride.", nameof(a));
            }

            if (ldb < n || (long)(k - 1) * ldb + n > b.Length)
            {
                throw new ArgumentException("B is smaller than k x n with its row stride.", nameof(b));
            }
        }

        fixed (float* pa = a, pb = b, pc = c)
        {
            RunCore(mode, path, m, n, k, pa, lda, pb, ldb, pc, ldc, threads);
        }
    }

    /// <summary>
    /// The pointer form of <see cref="Run"/>; the caller has validated that A, B and C cover the extents described there, and that
    /// m and n are positive.
    /// </summary>
    internal static void RunCore(
        KernelMode mode, KernelPath path, int m, int n, int k,
        float* a, nint lda, float* b, nint ldb, float* c, nint ldc, int threads)
    {
        if (k == 0)
        {
            for (int i = 0; i < m; i++)
            {
                new Span<float>(c + (i * ldc), n).Clear();
            }

            return;
        }

        Blocking blk = GetBlocking(path);
        int nStrips = (n + blk.Nr - 1) / blk.Nr;
        int kBlocks = (k + Kc - 1) / Kc;
        float* packedB = Scratch.Get(Scratch.PackB, (long)nStrips * blk.Nr * k);

        // Phase 1: pack all of B. Chunks are (k block, group of strips), fixed by the shape.
        int stripGroups = (nStrips + PackStripsPerChunk - 1) / PackStripsPerChunk;
        // The job objects are kept per thread and refilled, so a steady-state call allocates nothing; a call finishes with its jobs
        // before it returns, so the next call on the thread can reuse them.
        PackBJob packJob = t_packJob ??= new PackBJob();
        packJob.B = b;
        packJob.Ldb = ldb;
        packJob.N = n;
        packJob.K = k;
        packJob.Nr = blk.Nr;
        packJob.NStrips = nStrips;
        packJob.Groups = stripGroups;
        packJob.Dst = packedB;
        ParallelFor.Run(packJob, kBlocks * stripGroups, threads);

        // Phase 2: compute tiles of C.
        int tilesM = (m + blk.Mc - 1) / blk.Mc;
        int tilesN = (n + blk.Nc - 1) / blk.Nc;
        TileJob job = t_tileJob ??= new TileJob();
        job.Mode = mode;
        job.Path = path;
        job.Blk = blk;
        job.M = m;
        job.N = n;
        job.K = k;
        job.A = a;
        job.Lda = lda;
        job.PackedB = packedB;
        job.NStrips = nStrips;
        job.C = c;
        job.Ldc = ldc;
        job.TilesN = tilesN;
        job.Micro = SelectMicro(mode, path);
        ParallelFor.Run(job, tilesM * tilesN, threads);
    }

    /// <summary>The register tile and cache blocks of a path.</summary>
    internal readonly record struct Blocking(int Mr, int Nr, int Mc, int Nc);

    internal static Blocking GetBlocking(KernelPath path) => path switch
    {
        // Mc and Nc are multiples of Mr and Nr. They set the chunk size, so they are constants of the path, not of the machine.
        KernelPath.Avx512 => new(12, 32, 96, 128),
        KernelPath.Avx2 => new(6, 16, 96, 128),
        KernelPath.Neon => new(8, 12, 96, 120),
        KernelPath.Portable128 => new(4, 8, 64, 128),
        _ => new(4, 4, 64, 64),
    };

    private static delegate*<int, float*, float*, float*, nint, int, void> SelectMicro(KernelMode mode, KernelPath path)
    {
        if (mode == KernelMode.Exact)
        {
            return path switch
            {
                KernelPath.Avx512 => &ExactGemmAvx512.Kernel,
                KernelPath.Avx2 => &ExactGemmAvx2.Kernel,
                KernelPath.Neon => &ExactGemmNeon.Kernel,
                KernelPath.Portable128 => &ExactGemmPortable.Kernel,
                _ => &ExactGemmScalar.Kernel,
            };
        }

        return path switch
        {
            KernelPath.Avx512 => &FastGemmAvx512.Kernel,
            KernelPath.Avx2 => &FastGemmAvx2.Kernel,
            KernelPath.Neon => &FastGemmNeon.Kernel,
            KernelPath.Portable128 => &FastGemmPortable.Kernel,
            _ => &FastGemmScalar.Kernel,
        };
    }

    [ThreadStatic]
    private static PackBJob? t_packJob;

    [ThreadStatic]
    private static TileJob? t_tileJob;

    private sealed class PackBJob : ChunkJob
    {
        public float* B;
        public nint Ldb;
        public int N;
        public int K;
        public int Nr;
        public int NStrips;
        public int Groups;
        public float* Dst;

        public override void RunChunk(int chunk)
        {
            int kb = chunk / Groups;
            int g = chunk % Groups;
            int pc = kb * Kc;
            int kc = Math.Min(Kc, K - pc);
            int s0 = g * PackStripsPerChunk;
            int s1 = Math.Min(NStrips, s0 + PackStripsPerChunk);
            // Packed layout: k block at pc * NStrips * Nr, strip s of that block at s * kc * Nr within it.
            float* dst = Dst + ((long)pc * NStrips * Nr) + ((long)s0 * kc * Nr);
            GemmPack.PackBStrips(B + (pc * Ldb), Ldb, kc, N, Nr, s0, s1, dst);
        }
    }

    private sealed class TileJob : ChunkJob
    {
        public KernelMode Mode;
        public KernelPath Path;
        public Blocking Blk;
        public int M;
        public int N;
        public int K;
        public float* A;
        public nint Lda;
        public float* PackedB;
        public int NStrips;
        public float* C;
        public nint Ldc;
        public int TilesN;
        public delegate*<int, float*, float*, float*, nint, int, void> Micro;

        public override void RunChunk(int chunk)
        {
            int mr = Blk.Mr;
            int nr = Blk.Nr;
            int i0 = (chunk / TilesN) * Blk.Mc;
            int j0 = (chunk % TilesN) * Blk.Nc;
            int mc = Math.Min(Blk.Mc, M - i0);
            int nc = Math.Min(Blk.Nc, N - j0);
            float* packedA = Scratch.Get(Scratch.PackA, (long)Blk.Mc * Kc);
            float* edge = stackalloc float[Blk.Mr * Blk.Nr];
            int firstStrip = j0 / nr;

            for (int pc = 0; pc < K; pc += Kc)
            {
                int kc = Math.Min(Kc, K - pc);
                GemmPack.PackA(A + (i0 * Lda) + pc, Lda, mc, kc, mr, packedA);
                float* bBlock = PackedB + ((long)pc * NStrips * nr);
                int accumulate = pc > 0 ? 1 : 0;

                for (int jr = 0; jr < nc; jr += nr)
                {
                    int nb = Math.Min(nr, nc - jr);
                    float* bp = bBlock + ((long)(firstStrip + (jr / nr)) * kc * nr);
                    for (int ir = 0; ir < mc; ir += mr)
                    {
                        int mb = Math.Min(mr, mc - ir);
                        float* ap = packedA + ((long)(ir / mr) * kc * mr);
                        float* cp = C + ((i0 + ir) * Ldc) + j0 + jr;
                        if (mb == mr && nb == nr)
                        {
                            Micro(kc, ap, bp, cp, Ldc, accumulate);
                        }
                        else
                        {
                            // Ragged edge: run the full tile on a scratch copy and move only the valid part.
                            if (accumulate != 0)
                            {
                                for (int r = 0; r < mb; r++)
                                {
                                    for (int j = 0; j < nb; j++)
                                    {
                                        edge[(r * nr) + j] = cp[(r * Ldc) + j];
                                    }
                                }
                            }

                            Micro(kc, ap, bp, edge, nr, accumulate);
                            for (int r = 0; r < mb; r++)
                            {
                                for (int j = 0; j < nb; j++)
                                {
                                    cp[(r * Ldc) + j] = edge[(r * nr) + j];
                                }
                            }
                        }
                    }
                }
            }

            if (Mode == KernelMode.Exact)
            {
                CanonicalizeNaN(C + (i0 * Ldc) + j0, Ldc, mc, nc);
            }
        }
    }

    private static void CanonicalizeNaN(float* c, nint ldc, int rows, int cols)
    {
        uint* bits = (uint*)c;
        for (int r = 0; r < rows; r++)
        {
            uint* row = bits + (r * ldc);
            for (int j = 0; j < cols; j++)
            {
                // NaN: exponent all ones and a non-zero mantissa.
                if ((row[j] & 0x7FFFFFFFu) > 0x7F800000u)
                {
                    row[j] = 0x7FC00000u;
                }
            }
        }
    }
}
