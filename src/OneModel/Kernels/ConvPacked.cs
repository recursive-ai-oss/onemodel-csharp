using System;
using System.Runtime.Intrinsics;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Fast-mode convolution as one blocked matrix product per (image, group) with no im2col matrix (Goto and van de Geijn 2008,
/// section 4; im2col from Chellapilla, Puri and Simard 2006). The weights are A (Cout/G rows by Cin/G*KH*KW columns) and the input
/// patches are B. The patches are gathered straight from the (zero-padded) input planes into the kc x nr strips the microkernel
/// reads, so the unfolded matrix is never written. When one tile covers all the rows, each tile gathers its own columns and the
/// whole product is a single parallel phase; when the rows need several tiles, the strips are gathered once, in parallel, and the
/// tiles share them. The weights can be packed once when the model is loaded (<see cref="PackWeights"/>).
/// </summary>
/// <remarks>
/// Every output is the sum of its taps in increasing tap order, continuing from the stored partial sum across Kc blocks, whatever
/// the tile shape, so the result is the same for every tile shape and thread count. Tile sizes are chosen from the shape alone.
/// </remarks>
internal static unsafe class ConvPacked
{
    // Chunks the tile choice aims for: enough to keep a few dozen threads busy at batch 1, few enough that each is worth its wake-up.
    private const int TargetChunks = 48;

    // Multiply-adds per nanosecond assumed when telling the pool how much work a call holds. A core sustains about 40, but the pool's
    // rule (one thread per 80 microseconds of work) is conservative for a spinning pool: measured on MobileNetV3's small layers, the
    // convolutions ran 2x faster on 8 threads with the estimate set to 8 than with 40, and ResNet-50's large layers did not change.
    private const long MacsPerNs = 8;

    // Work below this (multiply-adds) is not worth splitting further: about 15 microseconds at the rate a core sustains.
    private const long MinChunkMacs = 500_000;

    // Strips gathered per chunk of the shared gathering phase.
    private const int StripsPerPackChunk = 8;

    // Planes padded per chunk.
    private const int PlanesPerPadChunk = 4;

    [ThreadStatic]
    private static Job? t_job;

    [ThreadStatic]
    private static PackJob? t_packJob;

    [ThreadStatic]
    private static PadJob? t_padJob;

    /// <summary>Floats <see cref="PackWeights"/> writes for weights of <paramref name="s"/> packed for <paramref name="path"/>.</summary>
    public static long PackedWeightsLength(KernelPath path, in ConvShape s)
    {
        int mr = Gemm.GetBlocking(path).Mr;
        long coutG = s.Cout / s.Groups;
        long k = (long)(s.Cin / s.Groups) * s.KH * s.KW;
        return s.Groups * ((coutG + mr - 1) / mr) * mr * k;
    }

    /// <summary>
    /// Packs the weights (Cout x Cin/G x KH x KW, contiguous) once, per group, into mr-row strips of Kc-column panels
    /// (<see cref="GemmPack.PackAPanels"/>). Copies only.
    /// </summary>
    /// <param name="path">The instruction set the packed copy is for; <see cref="Run"/> must be given the same one.</param>
    /// <param name="s">The convolution; only the channel counts, groups and kernel size are read.</param>
    /// <param name="weights">At least Cout * Cin/G * KH * KW floats.</param>
    /// <param name="dst">At least <see cref="PackedWeightsLength"/> floats.</param>
    public static void PackWeights(KernelPath path, in ConvShape s, ReadOnlySpan<float> weights, Span<float> dst)
    {
        int cinG = s.Cin / s.Groups;
        int coutG = s.Cout / s.Groups;
        int k = cinG * s.KH * s.KW;
        if (weights.Length < (long)s.Cout * k)
        {
            throw new ArgumentException("The weights are smaller than Cout x Cin/Groups x KH x KW.", nameof(weights));
        }

        long perGroup = PackedWeightsLength(path, s) / s.Groups;
        if (dst.Length < perGroup * s.Groups)
        {
            throw new ArgumentException("The destination is smaller than PackedWeightsLength.", nameof(dst));
        }

        int mr = Gemm.GetBlocking(path).Mr;
        fixed (float* pw = weights, pd = dst)
        {
            for (int g = 0; g < s.Groups; g++)
            {
                GemmPack.PackAPanels(pw + ((long)g * coutG * k), k, coutG, k, mr, Gemm.Kc, pd + (g * perGroup));
            }
        }
    }

    /// <summary>
    /// Runs the convolution. Exactly one of <paramref name="weights"/> and <paramref name="packedWeights"/> is non-null.
    /// </summary>
    /// <remarks>
    /// Bounds the caller (<see cref="Conv.Run"/>, <see cref="Conv.RunPacked"/>) has checked: <c>input</c> covers N*Cin*H*W floats,
    /// <c>output</c> N*Cout*OutH*OutW, <c>weights</c> Cout*Cin/G*KH*KW, <c>packedWeights</c> <see cref="PackedWeightsLength"/>, and
    /// the shape is valid with a non-empty output. A tap of an output column reads row oy*StrideH + ky*DilH and column
    /// ox*StrideW + kx*DilW of the padded plane, which the output size formula keeps inside it; a tile writes rows below Cout/G of
    /// its group by columns below OutH*OutW.
    /// </remarks>
    public static void Run(
        KernelPath path, in ConvShape s, float* input, float* weights, float* packedWeights, float* output, int threads,
        IGemmEpilogue? epilogue)
    {
        Gemm.Blocking blk = Gemm.GetBlocking(path);
        int cinG = s.Cin / s.Groups;
        int coutG = s.Cout / s.Groups;
        int k = cinG * s.KH * s.KW;
        int n = s.OutH * s.OutW;
        ChooseTiles(blk, coutG, n, k, (long)s.N * s.Groups, out int mc, out int nc);
        int tilesM = (coutG + mc - 1) / mc;
        int tilesN = (n + nc - 1) / nc;
        long macs = (long)s.N * s.Cout * n * k;

        // Zero padding is done once, into a copy of the planes, so that gathering never tests for the edge.
        int hp = s.H + s.PadTop + s.PadBottom;
        int wp = s.W + s.PadLeft + s.PadRight;
        float* planes = input;
        if (hp != s.H || wp != s.W)
        {
            long total = (long)s.N * s.Cin * hp * wp;
            planes = Scratch.Get(Scratch.Im2Col, total);
            PadJob pad = t_padJob ??= new PadJob();
            pad.Src = input;
            pad.Dst = planes;
            pad.H = s.H;
            pad.W = s.W;
            pad.Hp = hp;
            pad.Wp = wp;
            pad.PadTop = s.PadTop;
            pad.PadLeft = s.PadLeft;
            pad.Planes = s.N * s.Cin;
            ParallelFor.Run(pad, (pad.Planes + PlanesPerPadChunk - 1) / PlanesPerPadChunk, threads, (long)s.N * s.Cin * hp * wp / 8);
        }

        Job job = t_job ??= new Job();
        job.Setup(path, blk, s, hp, wp, mc, nc, tilesM, tilesN, planes, weights, packedWeights, output, epilogue);
        try
        {
            long tileWork = (long)tilesM * tilesN;
            if (tilesM == 1)
            {
                // One tile covers all the rows: each tile gathers its own columns.
                long chunks = tileWork * s.Groups * s.N;
                ParallelFor.Run(job, checked((int)chunks), threads, macs / MacsPerNs);
            }
            else
            {
                // Several row tiles share the gathered columns: gather once per (image, group), then run the tiles.
                int nStrips = (n + blk.Nr - 1) / blk.Nr;
                float* shared = Scratch.Get(Scratch.PackB, (long)nStrips * blk.Nr * k);
                int stripGroups = (nStrips + StripsPerPackChunk - 1) / StripsPerPackChunk;
                int kBlocks = (k + Gemm.Kc - 1) / Gemm.Kc;
                PackJob pack = t_packJob ??= new PackJob();
                pack.Owner = job;
                pack.Dst = shared;
                pack.NStrips = nStrips;
                pack.StripGroups = stripGroups;
                job.SharedPanels = shared;
                job.SharedStrips = nStrips;
                long groupMacs = macs / ((long)s.N * s.Groups);
                for (int img = 0; img < s.N; img++)
                {
                    for (int g = 0; g < s.Groups; g++)
                    {
                        job.Img = img;
                        job.Group = g;
                        ParallelFor.Run(pack, kBlocks * stripGroups, threads, (long)k * n / 8);
                        ParallelFor.Run(job, checked((int)tileWork), threads, groupMacs / MacsPerNs);
                    }
                }

                pack.Owner = null;
                job.SharedPanels = null;
            }
        }
        finally
        {
            job.Release();
        }
    }

    // Tile sizes from the shape alone: wide tiles first, shorter ones only when the shape would otherwise give too few chunks. The
    // row tiles of a shape are made equal (rounded up to whole strips), so 128 rows run as 72 + 56, not 96 + 32.
    private static void ChooseTiles(Gemm.Blocking blk, int m, int n, int k, long batches, out int mc, out int nc)
    {
        ReadOnlySpan<(int Nc, int Mc)> options = [(128, 96), (64, 96), (32, 96), (32, 48), (32, 24)];
        mc = 0;
        nc = 0;
        for (int i = 0; i < options.Length; i++)
        {
            int ncI = Math.Max(1, options[i].Nc / blk.Nr) * blk.Nr;
            int mcI = Math.Max(1, options[i].Mc / blk.Mr) * blk.Mr;
            if (i > 0 && (long)mcI * ncI * k < MinChunkMacs)
            {
                break;
            }

            int tilesM = (m + mcI - 1) / mcI;
            mcI = (((m + tilesM - 1) / tilesM) + blk.Mr - 1) / blk.Mr * blk.Mr;
            mc = mcI;
            nc = ncI;
            long chunks = (long)tilesM * ((n + nc - 1) / nc) * batches;
            if (chunks >= TargetChunks)
            {
                break;
            }
        }
    }

    // Copies the planes into zero-bordered planes.
    private sealed class PadJob : ChunkJob
    {
        public float* Src;
        public float* Dst;
        public int H;
        public int W;
        public int Hp;
        public int Wp;
        public int PadTop;
        public int PadLeft;
        public int Planes;

        public override void RunChunk(int chunk)
        {
            int end = Math.Min(Planes, (chunk + 1) * PlanesPerPadChunk);
            for (int q = chunk * PlanesPerPadChunk; q < end; q++)
            {
                float* src = Src + ((long)q * H * W);
                float* dst = Dst + ((long)q * Hp * Wp);
                new Span<float>(dst, PadTop * Wp).Clear();
                for (int y = 0; y < H; y++)
                {
                    float* row = dst + ((long)(PadTop + y) * Wp);
                    new Span<float>(row, PadLeft).Clear();
                    new Span<float>(src + ((long)y * W), W).CopyTo(new Span<float>(row + PadLeft, W));
                    new Span<float>(row + PadLeft + W, Wp - PadLeft - W).Clear();
                }

                int bottom = Hp - PadTop - H;
                new Span<float>(dst + ((long)(PadTop + H) * Wp), bottom * Wp).Clear();
            }
        }
    }

    // The shared gathering phase: chunk = (k block, group of strips) of the current (image, group).
    private sealed class PackJob : ChunkJob
    {
        public Job? Owner;
        public float* Dst;
        public int NStrips;
        public int StripGroups;

        public override void RunChunk(int chunk)
        {
            Job job = Owner!;
            int kb = chunk / StripGroups;
            int sg = chunk % StripGroups;
            int pc = kb * Gemm.Kc;
            int kc = Math.Min(Gemm.Kc, job.K - pc);
            int s0 = sg * StripsPerPackChunk;
            int s1 = Math.Min(NStrips, s0 + StripsPerPackChunk);
            int col0 = s0 * job.Nr;
            int cols = Math.Min(job.N, s1 * job.Nr) - col0;
            // Layout: k block at pc * NStrips * nr, strip s of it at s * kc * nr.
            job.PackPanel(Dst + ((long)pc * NStrips * job.Nr) + ((long)s0 * kc * job.Nr), job.PlanesOf(job.Img, job.Group), pc, kc, col0, cols);
        }
    }

    private sealed class Job : ChunkJob
    {
        private int _mr;
        private int _mc;
        private int _nc;
        private int _tilesM;
        private int _tilesN;
        private int _groups;
        private int _cin;
        private int _cinG;
        private int _cout;
        private int _coutG;
        private int _kh;
        private int _kw;
        private int _strideH;
        private int _strideW;
        private int _dilH;
        private int _dilW;
        private int _hp;
        private int _wp;
        private int _outW;
        private bool _contiguous;
        private long _mPad;
        private float* _input;
        private float* _weights;
        private float* _packedWeights;
        private float* _output;
        private IGemmEpilogue? _epilogue;
        private delegate*<int, float*, float*, float*, nint, int, void> _micro;

        // The shared-gathering mode: the strips and the (image, group) being computed.
        public float* SharedPanels;
        public int SharedStrips;
        public int Img;
        public int Group;

        public int Nr { get; private set; }

        public int K { get; private set; }

        public int N { get; private set; }

        public float* PlanesOf(int img, int g) => _input + (((long)img * _cin) + ((long)g * _cinG)) * _hp * _wp;

        public void Setup(
            KernelPath path, Gemm.Blocking blk, in ConvShape s, int hp, int wp, int mc, int nc, int tilesM, int tilesN,
            float* input, float* weights, float* packedWeights, float* output, IGemmEpilogue? epilogue)
        {
            _mr = blk.Mr;
            Nr = blk.Nr;
            _mc = mc;
            _nc = nc;
            _tilesM = tilesM;
            _tilesN = tilesN;
            _groups = s.Groups;
            _cin = s.Cin;
            _cinG = s.Cin / s.Groups;
            _cout = s.Cout;
            _coutG = s.Cout / s.Groups;
            _kh = s.KH;
            _kw = s.KW;
            K = _cinG * s.KH * s.KW;
            _hp = hp;
            _wp = wp;
            _strideH = s.StrideH;
            _strideW = s.StrideW;
            _dilH = s.DilH;
            _dilW = s.DilW;
            _outW = s.OutW;
            N = s.OutH * _outW;
            // When consecutive output columns are consecutive in the padded plane (unit strides and a one-column kernel), any run of
            // columns is one contiguous copy, rows included.
            _contiguous = s.StrideH == 1 && s.StrideW == 1 && wp == _outW;
            _mPad = (long)((_coutG + _mr - 1) / _mr) * _mr;
            _input = input;
            _weights = weights;
            _packedWeights = packedWeights;
            _output = output;
            _epilogue = epilogue;
            _micro = Gemm.SelectMicro(KernelMode.Fast, path);
            SharedPanels = null;
        }

        public void Release() => _epilogue = null;

        public override void RunChunk(int chunk)
        {
            int perImage = _tilesM * _tilesN;
            if (SharedPanels is null)
            {
                int tile = chunk % perImage;
                int ig = chunk / perImage;
                RunTile(ig / _groups, ig % _groups, tile, null);
            }
            else
            {
                RunTile(Img, Group, chunk, SharedPanels);
            }
        }

        private void RunTile(int img, int g, int tile, float* shared)
        {
            int i0 = (tile / _tilesN) * _mc;
            int j0 = (tile % _tilesN) * _nc;
            int mc = Math.Min(_mc, _coutG - i0);
            int nc = Math.Min(_nc, N - j0);
            int mr = _mr;
            int nr = Nr;
            int kTotal = K;

            float* planes = PlanesOf(img, g);
            float* c = _output + ((((long)img * _cout) + ((long)g * _coutG) + i0) * N) + j0;
            nint ldc = N;
            int nStrips = (nc + nr - 1) / nr;
            float* panel = shared is null ? Scratch.Get(Scratch.PackPanel, (long)nStrips * nr * Gemm.Kc) : null;
            float* packedA = _packedWeights is null ? Scratch.Get(Scratch.PackA, (long)_mc * Gemm.Kc) : null;
            float* edge = stackalloc float[mr * nr];

            for (int pc = 0; pc < kTotal; pc += Gemm.Kc)
            {
                int kc = Math.Min(Gemm.Kc, kTotal - pc);
                float* bBlock;
                if (shared is null)
                {
                    PackPanel(panel, planes, pc, kc, j0, nc);
                    bBlock = panel;
                }
                else
                {
                    bBlock = shared + ((long)pc * SharedStrips * nr) + ((long)(j0 / nr) * kc * nr);
                }

                float* aBlock;
                if (_packedWeights is not null)
                {
                    // Packed layout (GemmPack.PackAPanels): block pc at pc * mPad, strip s of it at s * kc * mr.
                    aBlock = _packedWeights + ((long)g * _mPad * kTotal) + (pc * _mPad) + ((long)(i0 / mr) * kc * mr);
                }
                else
                {
                    GemmPack.PackA(_weights + ((((long)g * _coutG) + i0) * kTotal) + pc, kTotal, mc, kc, mr, packedA);
                    aBlock = packedA;
                }

                int accumulate = pc > 0 ? 1 : 0;
                for (int jr = 0; jr < nc; jr += nr)
                {
                    int nb = Math.Min(nr, nc - jr);
                    float* bp = bBlock + ((long)(jr / nr) * kc * nr);
                    for (int ir = 0; ir < mc; ir += mr)
                    {
                        int mb = Math.Min(mr, mc - ir);
                        float* ap = aBlock + ((long)(ir / mr) * kc * mr);
                        float* cp = c + (ir * ldc) + jr;
                        if (mb == mr && nb == nr)
                        {
                            _micro(kc, ap, bp, cp, ldc, accumulate);
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
                                        edge[(r * nr) + j] = cp[(r * ldc) + j];
                                    }
                                }
                            }

                            _micro(kc, ap, bp, edge, nr, accumulate);
                            for (int r = 0; r < mb; r++)
                            {
                                for (int j = 0; j < nb; j++)
                                {
                                    cp[(r * ldc) + j] = edge[(r * nr) + j];
                                }
                            }
                        }
                    }
                }
            }

            _epilogue?.Apply(c, ldc, (g * _coutG) + i0, j0, mc, nc, KernelMode.Fast);
        }

        /// <summary>
        /// Gathers taps pc..pc+kc-1 of the patches of output columns col0..col0+cols-1 into strips of nr columns: strip s at s*kc*nr,
        /// element (p, j) at p*nr + j, the last strip padded with +0. <paramref name="planes"/> is the first padded input plane of the
        /// group. Padding taps are zeros already in the padded plane, so they are real multiplications by zero, as in im2col.
        /// </summary>
        public void PackPanel(float* dst, float* planes, int pc, int kc, int col0, int cols)
        {
            int nr = Nr;
            int nStrips = (cols + nr - 1) / nr;
            long stripStride = (long)kc * nr;

            // The same runs serve every tap: a run is a stretch of consecutive output columns in one output row (or, for the
            // contiguous case, in one strip), given by its first input position, its place in the strip row and its length.
            // At most one run per column, and a call covers at most StripsPerPackChunk strips or one tile of at most 128 columns.
            int maxRuns = cols;
            int* runSrc = stackalloc int[Math.Min(maxRuns, 512)];
            int* runDst = stackalloc int[Math.Min(maxRuns, 512)];
            int* runLen = stackalloc int[Math.Min(maxRuns, 512)];
            int nRuns = 0;
            for (int s = 0; s < nStrips; s++)
            {
                int colStart = col0 + (s * nr);
                int sc = Math.Min(nr, cols - (s * nr));
                if (_contiguous)
                {
                    runSrc[nRuns] = colStart;
                    runDst[nRuns] = (int)(s * stripStride);
                    runLen[nRuns++] = sc;
                    continue;
                }

                int oy = colStart / _outW;
                int ox = colStart - (oy * _outW);
                int j = 0;
                while (j < sc)
                {
                    int seg = Math.Min(sc - j, _outW - ox);
                    runSrc[nRuns] = (oy * _strideH * _wp) + (ox * _strideW);
                    runDst[nRuns] = (int)(s * stripStride) + j;
                    runLen[nRuns++] = seg;
                    j += seg;
                    oy++;
                    ox = 0;
                }
            }

            int lastCols = cols - ((nStrips - 1) * nr);
            int kx = pc % _kw;
            int t = pc / _kw;
            int ky = t % _kh;
            int ci = t / _kh;
            long planeSize = (long)_hp * _wp;
            for (int p = 0; p < kc; p++)
            {
                float* tap = planes + (ci * planeSize) + ((long)ky * _dilH * _wp) + ((long)kx * _dilW);
                float* row = dst + ((long)p * nr);
                if (_strideW == 1)
                {
                    for (int r = 0; r < nRuns; r++)
                    {
                        CopyRun(tap + runSrc[r], row + runDst[r], runLen[r]);
                    }
                }
                else
                {
                    for (int r = 0; r < nRuns; r++)
                    {
                        float* sp = tap + runSrc[r];
                        float* dp = row + runDst[r];
                        int len = runLen[r];
                        for (int j = 0; j < len; j++)
                        {
                            dp[j] = sp[j * _strideW];
                        }
                    }
                }

                if (lastCols < nr)
                {
                    new Span<float>(row + ((nStrips - 1) * stripStride) + lastCols, nr - lastCols).Clear();
                }

                if (++kx == _kw)
                {
                    kx = 0;
                    if (++ky == _kh)
                    {
                        ky = 0;
                        ci++;
                    }
                }
            }
        }

        private static void CopyRun(float* src, float* dst, int len)
        {
            int j = 0;
            if (Vector256.IsHardwareAccelerated)
            {
                for (; j + 8 <= len; j += 8)
                {
                    Vector256.Store(Vector256.Load(src + j), dst + j);
                }
            }

            for (; j + 4 <= len; j += 4)
            {
                Vector128.Store(Vector128.Load(src + j), dst + j);
            }

            for (; j < len; j++)
            {
                dst[j] = src[j];
            }
        }
    }
}
