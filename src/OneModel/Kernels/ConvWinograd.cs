using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Fast-mode 3x3 stride-1 convolution by Winograd's minimal filtering F(2x2, 3x3) (Lavin and Gray 2016, section 3): each 2x2 block of
/// outputs ("tile") needs 16 multiplications per channel pair instead of 36, in exchange for transforms of the input patch
/// (4x4 -> 4x4), of the weights (3x3 -> 4x4, done once when the model is loaded) and of the products (4x4 -> 2x2).
/// For the 16 transform positions the products over the input channels are 16 independent matrix products
/// (Cout x Cin times Cin x tiles); they run on the same register-tile microkernels as the other convolutions.
/// </summary>
/// <remarks>
/// The sums differ from the direct convolution's in the last bits (the weights and the input are transformed first), so this path
/// is for fast mode only and is chosen by the shape of the weights alone (<see cref="Eligible"/>), never by the thread count. Every
/// output is computed by one chunk (a block of tile rows of one image) with a fixed order of operations, so the result does not
/// depend on the thread count. The chunk's transformed input (16 positions x Cin x tiles) and products (16 x Cout x tiles) live in
/// per-thread scratch and are sized to stay near the core's cache.
/// </remarks>
internal static unsafe class ConvWinograd
{
    // Channel counts below this do not repay the transforms (measured: 16 -> 32 channels at 32x40 is no faster than the direct path).
    private const int MinChannels = 32;

    // The transformed weights (16 x Cin x Cout floats) are read by every chunk, so they should stay in the caches the chunks share:
    // beyond Cin * Cout = 40 960 (a 2.5 MB copy; 192 x 192 channels fit) the direct path is faster (ResNet-50's 256 x 256 layers at 14 x 14
    // ran 20% slower, its 512 x 512 layers at 7 x 7 almost three times slower, one thread).
    private const long MaxChannelProduct = 40_960;

    // Floats of transformed input plus products one chunk may hold (1 MiB): about the size of a core's second-level cache.
    private const long BudgetFloats = 256 * 1024;

    // Chunks the tile-row split aims for, as in ConvPacked.
    private const int TargetChunks = 48;

    // Work below this (multiply-adds) is not worth a chunk of its own.
    private const long MinChunkMacs = 500_000;

    private const int Positions = 16;

    [ThreadStatic]
    private static Job? t_job;

    /// <summary>Whether a convolution of this kind uses the Winograd layout of packed weights: 3x3, unit strides and dilations, one group, enough channels.</summary>
    public static bool Eligible(in ConvShape s) =>
        s.Groups == 1 && s.KH == 3 && s.KW == 3 && s.StrideH == 1 && s.StrideW == 1 && s.DilH == 1 && s.DilW == 1 &&
        s.Cin >= MinChannels && s.Cout >= MinChannels && (long)s.Cin * s.Cout <= MaxChannelProduct;

    /// <summary>Floats of the packed weights: for each of the 16 positions, a Cout x Cin matrix in the panel layout of <see cref="GemmPack.PackAPanels"/>.</summary>
    public static long PackedLength(KernelPath path, in ConvShape s)
    {
        int mr = ConvPacked.GetBlocking(path, s.Cout).Mr;
        return Positions * (long)((s.Cout + mr - 1) / mr) * mr * s.Cin;
    }

    /// <summary>
    /// Transforms the weights (Cout x Cin x 3 x 3, contiguous) to U = G g G^T per channel pair, computed in double precision and rounded
    /// once, and packs position p's Cout x Cin matrix as A panels. Bounds: reads Cout*Cin*9 floats at <paramref name="w"/>, writes
    /// <see cref="PackedLength"/> floats at <paramref name="dst"/>; the caller has checked both.
    /// </summary>
    public static void PackWeights(KernelPath path, in ConvShape s, float* w, float* dst)
    {
        int cin = s.Cin, cout = s.Cout;
        int mr = ConvPacked.GetBlocking(path, cout).Mr;
        long perPos = (long)((cout + mr - 1) / mr) * mr * cin;
        var u = new float[Positions * (long)cout * cin];
        Span<double> t = stackalloc double[12];
        for (int co = 0; co < cout; co++)
        {
            for (int ci = 0; ci < cin; ci++)
            {
                float* g = w + (((long)co * cin) + ci) * 9;
                for (int c = 0; c < 3; c++)
                {
                    double g0 = g[c], g1 = g[3 + c], g2 = g[6 + c];
                    t[c] = g0;
                    t[3 + c] = 0.5 * (g0 + g1 + g2);
                    t[6 + c] = 0.5 * (g0 - g1 + g2);
                    t[9 + c] = g2;
                }

                for (int i = 0; i < 4; i++)
                {
                    double a = t[i * 3], b = t[(i * 3) + 1], c2 = t[(i * 3) + 2];
                    long at = ((long)co * cin) + ci;
                    u[((i * 4) * (long)cout * cin) + at] = (float)a;
                    u[(((i * 4) + 1) * (long)cout * cin) + at] = (float)(0.5 * (a + b + c2));
                    u[(((i * 4) + 2) * (long)cout * cin) + at] = (float)(0.5 * (a - b + c2));
                    u[(((i * 4) + 3) * (long)cout * cin) + at] = (float)c2;
                }
            }
        }

        fixed (float* pu = u)
        {
            for (int p = 0; p < Positions; p++)
            {
                GemmPack.PackAPanels(pu + (p * (long)cout * cin), cin, cout, cin, mr, Gemm.Kc, dst + (p * perPos));
            }
        }
    }

    /// <summary>
    /// Runs the convolution with weights packed by <see cref="PackWeights"/>. Bounds (checked by <see cref="Conv.RunPacked"/>): the input
    /// holds N*Cin*H*W floats, the output N*Cout*OutH*OutW, the packed weights <see cref="PackedLength"/>; a patch read is clipped to
    /// the plane, a store is clipped to the output.
    /// </summary>
    public static void Run(KernelPath path, in ConvShape s, float* input, float* packed, float* output, int threads, IGemmEpilogue? epilogue)
    {
        Gemm.Blocking blk = ConvPacked.GetBlocking(path, s.Cout);
        int outH = s.OutH, outW = s.OutW;
        int tilesY = (outH + 1) / 2;
        int tilesX = (outW + 1) / 2;
        int txPad = (tilesX + 3) & ~3;
        int coutPad = (s.Cout + blk.Mr - 1) / blk.Mr * blk.Mr;

        // Tile rows per block: as many as the cache budget allows, fewer while that leaves too few blocks to share between cores.
        long perTile = Positions * (long)(s.Cin + coutPad);
        int rows = (int)Math.Clamp(BudgetFloats / (perTile * txPad), 1, tilesY);
        long macsPerRow = (long)txPad * Positions * s.Cin * s.Cout;
        while (rows > 1 && (long)s.N * ((tilesY + rows - 1) / rows) < TargetChunks && (rows / 2) * macsPerRow >= MinChunkMacs)
        {
            rows /= 2;
        }

        int blocksPerImage = (tilesY + rows - 1) / rows;
        int stripsFull = ((rows * txPad) + blk.Nr - 1) / blk.Nr;

        // The output channels of a block split into groups of whole row strips when the blocks alone leave too few chunks for the cores.
        int rowStrips = coutPad / blk.Mr;
        long blockMacs = rows * macsPerRow;
        int split = 1;
        while (split < rowStrips && (long)s.N * blocksPerImage * split < TargetChunks && blockMacs / (split + 1) >= MinChunkMacs)
        {
            split++;
        }

        int stripsPerGroup = (rowStrips + split - 1) / split;
        split = (rowStrips + stripsPerGroup - 1) / stripsPerGroup;

        int blocks = checked(s.N * blocksPerImage);
        // One cache line is added to the distance between positions: a distance that is a multiple of 4 KiB would send the 16 stores of one
        // tile group (and later the 16 loads of the output transform) to the same few cache sets.
        long posStrideV = ((long)stripsFull * blk.Nr * s.Cin) + 16;
        float* v = Scratch.Get(Scratch.WinogradV, blocks * Positions * posStrideV);
        Job job = t_job ??= new Job();
        job.Setup(path, blk, s, input, packed, output, epilogue, rows, blocksPerImage, txPad, coutPad, stripsFull, split, stripsPerGroup, v, posStrideV);
        try
        {
            long tileChannels = (long)s.N * tilesY * txPad * s.Cin;
            ParallelFor.Run(job.Transform, checked(blocks * job.ChannelGroups), threads, tileChannels * 5);
            long macs = (long)s.N * Positions * s.Cin * s.Cout * tilesY * txPad;
            ParallelFor.Run(job.Product, checked(blocks * split), threads, macs / 8);
        }
        finally
        {
            job.Release();
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> Even(Vector128<float> a, Vector128<float> b)
    {
        if (Sse.IsSupported)
        {
            return Sse.Shuffle(a, b, 0x88);
        }

        if (AdvSimd.Arm64.IsSupported)
        {
            return AdvSimd.Arm64.UnzipEven(a, b);
        }

        return Vector128.Create(a.GetElement(0), a.GetElement(2), b.GetElement(0), b.GetElement(2));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> Odd(Vector128<float> a, Vector128<float> b)
    {
        if (Sse.IsSupported)
        {
            return Sse.Shuffle(a, b, 0xDD);
        }

        if (AdvSimd.Arm64.IsSupported)
        {
            return AdvSimd.Arm64.UnzipOdd(a, b);
        }

        return Vector128.Create(a.GetElement(1), a.GetElement(3), b.GetElement(1), b.GetElement(3));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> ZipLow(Vector128<float> a, Vector128<float> b)
    {
        if (Sse.IsSupported)
        {
            return Sse.UnpackLow(a, b);
        }

        if (AdvSimd.Arm64.IsSupported)
        {
            return AdvSimd.Arm64.ZipLow(a, b);
        }

        return Vector128.Create(a.GetElement(0), b.GetElement(0), a.GetElement(1), b.GetElement(1));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> ZipHigh(Vector128<float> a, Vector128<float> b)
    {
        if (Sse.IsSupported)
        {
            return Sse.UnpackHigh(a, b);
        }

        if (AdvSimd.Arm64.IsSupported)
        {
            return AdvSimd.Arm64.ZipHigh(a, b);
        }

        return Vector128.Create(a.GetElement(2), b.GetElement(2), a.GetElement(3), b.GetElement(3));
    }

    // Two parallel phases per call: the input transform (chunks of a few channels of one block of tile rows of one image, into one shared
    // buffer), then the products and the output transform (chunks of a group of row strips of one block). A block is a few whole tile
    // rows of one image; every output is produced by exactly one product chunk.
    private sealed class Job
    {
        private const int ChannelsPerChunk = 16;
        private int _mr, _nr, _n, _cin, _cout, _h, _w, _padTop, _padLeft, _outH, _outW;
        private int _tilesX, _tilesY, _rows, _blocksPerImage, _txPad, _coutPad, _stripsFull, _split, _stripsPerGroup;
        private long _posStrideV;
        private float* _v;
        private float* _input;
        private float* _packed;
        private float* _output;
        private IGemmEpilogue? _epilogue;
        private delegate*<int, float*, float*, float*, nint, int, void> _micro;

        public Job()
        {
            Transform = new TransformPhase(this);
            Product = new ProductPhase(this);
        }

        public ChunkJob Transform { get; }

        public ChunkJob Product { get; }

        public int ChannelGroups => (_cin + ChannelsPerChunk - 1) / ChannelsPerChunk;

        public void Setup(
            KernelPath path, Gemm.Blocking blk, in ConvShape s, float* input, float* packed, float* output, IGemmEpilogue? epilogue,
            int rows, int blocksPerImage, int txPad, int coutPad, int stripsFull, int split, int stripsPerGroup, float* v, long posStrideV)
        {
            _mr = blk.Mr;
            _nr = blk.Nr;
            _n = s.N;
            _cin = s.Cin;
            _cout = s.Cout;
            _h = s.H;
            _w = s.W;
            _padTop = s.PadTop;
            _padLeft = s.PadLeft;
            _outH = s.OutH;
            _outW = s.OutW;
            _tilesX = (_outW + 1) / 2;
            _tilesY = (_outH + 1) / 2;
            _rows = rows;
            _blocksPerImage = blocksPerImage;
            _txPad = txPad;
            _coutPad = coutPad;
            _stripsFull = stripsFull;
            _split = split;
            _stripsPerGroup = stripsPerGroup;
            _v = v;
            _posStrideV = posStrideV;
            _input = input;
            _packed = packed;
            _output = output;
            _epilogue = epilogue;
            _micro = ConvPacked.SelectMicro(path, blk);
        }

        public void Release() => _epilogue = null;

        private sealed class TransformPhase(Job owner) : ChunkJob
        {
            public override void RunChunk(int chunk) => owner.RunTransform(chunk);
        }

        private sealed class ProductPhase(Job owner) : ChunkJob
        {
            public override void RunChunk(int chunk) => owner.RunProduct(chunk);
        }

        private void RunTransform(int chunk)
        {
            int groups = ChannelGroups;
            int block = chunk / groups;
            int c0 = (chunk % groups) * ChannelsPerChunk;
            int img = block / _blocksPerImage;
            int ty0 = (block % _blocksPerImage) * _rows;
            int nRows = Math.Min(_tilesY, ty0 + _rows) - ty0;
            int strips = (nRows * _txPad + _nr - 1) / _nr;
            long ldm = (long)_stripsFull * _nr;
            float* v = _v + (block * Positions * _posStrideV);
            TransformInput(img, ty0, nRows, strips, c0, Math.Min(_cin, c0 + ChannelsPerChunk), v, _posStrideV, ldm);
        }

        private void RunProduct(int chunk)
        {
            int block = chunk / _split;
            int group = chunk % _split;
            int img = block / _blocksPerImage;
            int ty0 = (block % _blocksPerImage) * _rows;
            int nRows = Math.Min(_tilesY, ty0 + _rows) - ty0;
            int strips = (nRows * _txPad + _nr - 1) / _nr;
            long ldm = (long)_stripsFull * _nr;
            int rowStrip0 = group * _stripsPerGroup;
            int rowStrip1 = Math.Min(_coutPad / _mr, rowStrip0 + _stripsPerGroup);
            long posStrideM = ((long)(rowStrip1 - rowStrip0) * _mr * ldm) + 16;
            float* v = _v + (block * Positions * _posStrideV);
            float* m = Scratch.Get(Scratch.WinogradM, Positions * posStrideM);
            Multiply(strips, rowStrip0, rowStrip1, v, _posStrideV, m, posStrideM, ldm);
            TransformOutput(img, ty0, nRows, rowStrip0 * _mr, Math.Min(_cout, rowStrip1 * _mr), m, posStrideM, ldm);
        }

        // V = B^T d B for every channel and tile of the chunk, four tiles (consecutive columns of one tile row) per vector, written in
        // the packed-B layout of each position: block pc of Kc channels at pc*ldm, strip s of it at s*kc*nr, channel row p at p*nr.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private void TransformInput(int img, int ty0, int nRows, int strips, int c0, int c1, float* v, long posStride, long ldm)
        {
            int nr = _nr;
            int groups = strips * nr / 4;
            float* window = stackalloc float[16];
            for (int c = c0; c < c1; c++)
            {
                float* plane = _input + (((long)img * _cin) + c) * _h * _w;
                int pc = c / Gemm.Kc * Gemm.Kc;
                int kc = Math.Min(Gemm.Kc, _cin - pc);
                float* chan = v + (pc * ldm) + ((long)(c - pc) * nr);
                for (int g = 0; g < groups; g++)
                {
                    int tau = g * 4;
                    int r = tau / _txPad;
                    float* dst = chan + ((long)(tau / nr) * kc * nr) + (tau % nr);
                    if (r >= nRows)
                    {
                        for (int p = 0; p < Positions; p++)
                        {
                            Vector128.Store(Vector128<float>.Zero, dst + (p * posStride));
                        }

                        continue;
                    }

                    int ty = ty0 + r;
                    int bx = (2 * (tau % _txPad)) - _padLeft;
                    bool inside = bx >= 0 && bx + 10 <= _w;
                    Vector128<float> d00, d01, d02, d03, d10, d11, d12, d13, d20, d21, d22, d23, d30, d31, d32, d33;
                    LoadRow(plane, (2 * ty) - _padTop, bx, inside, window, out d00, out d01, out d02, out d03);
                    LoadRow(plane, (2 * ty) - _padTop + 1, bx, inside, window, out d10, out d11, out d12, out d13);
                    LoadRow(plane, (2 * ty) - _padTop + 2, bx, inside, window, out d20, out d21, out d22, out d23);
                    LoadRow(plane, (2 * ty) - _padTop + 3, bx, inside, window, out d30, out d31, out d32, out d33);

                    // B^T d: for each column j, t0 = d0 - d2, t1 = d1 + d2, t2 = d2 - d1, t3 = d1 - d3.
                    Vector128<float> t00 = d00 - d20, t10 = d10 + d20, t20 = d20 - d10, t30 = d10 - d30;
                    Vector128<float> t01 = d01 - d21, t11 = d11 + d21, t21 = d21 - d11, t31 = d11 - d31;
                    Vector128<float> t02 = d02 - d22, t12 = d12 + d22, t22 = d22 - d12, t32 = d12 - d32;
                    Vector128<float> t03 = d03 - d23, t13 = d13 + d23, t23 = d23 - d13, t33 = d13 - d33;

                    // (B^T d) B: for each row i, v0 = t0 - t2, v1 = t1 + t2, v2 = t2 - t1, v3 = t1 - t3.
                    Vector128.Store(t00 - t02, dst);
                    Vector128.Store(t01 + t02, dst + posStride);
                    Vector128.Store(t02 - t01, dst + (2 * posStride));
                    Vector128.Store(t01 - t03, dst + (3 * posStride));
                    Vector128.Store(t10 - t12, dst + (4 * posStride));
                    Vector128.Store(t11 + t12, dst + (5 * posStride));
                    Vector128.Store(t12 - t11, dst + (6 * posStride));
                    Vector128.Store(t11 - t13, dst + (7 * posStride));
                    Vector128.Store(t20 - t22, dst + (8 * posStride));
                    Vector128.Store(t21 + t22, dst + (9 * posStride));
                    Vector128.Store(t22 - t21, dst + (10 * posStride));
                    Vector128.Store(t21 - t23, dst + (11 * posStride));
                    Vector128.Store(t30 - t32, dst + (12 * posStride));
                    Vector128.Store(t31 + t32, dst + (13 * posStride));
                    Vector128.Store(t32 - t31, dst + (14 * posStride));
                    Vector128.Store(t31 - t33, dst + (15 * posStride));
                }
            }
        }

        // The four input columns j = 0..3 of a 4x4 patch, for four tiles side by side: lane l of column j is x[iy, bx + 2l + j]. The ten
        // floats bx..bx+9 are loaded (from the plane, or from a zero-padded copy at the plane's edges) and split into even and odd
        // positions with shuffles. A row outside the plane is zeros.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void LoadRow(float* plane, int iy, int bx, bool inside, float* window, out Vector128<float> c0, out Vector128<float> c1, out Vector128<float> c2, out Vector128<float> c3)
        {
            if ((uint)iy >= (uint)_h)
            {
                c0 = c1 = c2 = c3 = Vector128<float>.Zero;
                return;
            }

            float* p;
            if (inside)
            {
                p = plane + ((long)iy * _w) + bx;
            }
            else
            {
                float* row = plane + ((long)iy * _w);
                for (int k = 0; k < 10; k++)
                {
                    int x = bx + k;
                    window[k] = (uint)x < (uint)_w ? row[x] : 0f;
                }

                p = window;
            }

            Vector128<float> w0 = Vector128.Load(p), w4 = Vector128.Load(p + 4), w2 = Vector128.Load(p + 2), w6 = Vector128.Load(p + 6);
            c0 = Even(w0, w4);
            c1 = Odd(w0, w4);
            c2 = Even(w2, w6);
            c3 = Odd(w2, w6);
        }

        // For each position, M_p = U_p x V_p over the input channels in blocks of Kc, on the register-tile microkernel. M_p has coutPad
        // rows (the packed weights are padded with zero rows) and ldm columns, so every microkernel call is a full tile.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private void Multiply(int strips, int rowStrip0, int rowStrip1, float* v, long posStrideV, float* m, long posStrideM, long ldm)
        {
            int mr = _mr, nr = _nr;
            long perPos = (long)_coutPad * _cin;
            for (int p = 0; p < Positions; p++)
            {
                float* vp = v + (p * posStrideV);
                float* mp = m + (p * posStrideM);
                float* up = _packed + (p * perPos);
                for (int pc = 0; pc < _cin; pc += Gemm.Kc)
                {
                    int kc = Math.Min(Gemm.Kc, _cin - pc);
                    int accumulate = pc > 0 ? 1 : 0;
                    for (int s = 0; s < strips; s++)
                    {
                        float* bp = vp + (pc * ldm) + ((long)s * kc * nr);
                        for (int i = rowStrip0; i < rowStrip1; i++)
                        {
                            float* ap = up + ((long)pc * _coutPad) + ((long)i * kc * mr);
                            _micro(kc, ap, bp, mp + ((i - rowStrip0) * mr * ldm) + (s * nr), (nint)ldm, accumulate);
                        }
                    }
                }
            }
        }

        // Y = A^T M A for every output channel and tile, four tiles per vector; the 2x2 results are stored (interleaved along the row)
        // into the output, clipped at its edges, and the epilogue is applied to the chunk's rows of that channel.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private void TransformOutput(int img, int ty0, int nRows, int co0, int co1, float* m, long posStride, long ldm)
        {
            float* spill = stackalloc float[16];
            int outRows = Math.Min(_outH, 2 * (ty0 + nRows)) - (2 * ty0);
            long plane = (long)_outH * _outW;
            for (int co = co0; co < co1; co++)
            {
                float* chanOut = _output + ((((long)img * _cout) + co) * plane);
                float* src = m + ((co - co0) * ldm);
                for (int r = 0; r < nRows; r++)
                {
                    int oy = 2 * (ty0 + r);
                    bool second = oy + 1 < _outH;
                    float* row0 = chanOut + ((long)oy * _outW);
                    float* row1 = row0 + _outW;
                    for (int tx0 = 0; tx0 < _tilesX; tx0 += 4)
                    {
                        float* q = src + ((long)r * _txPad) + tx0;
                        Vector128<float> m00 = Vector128.Load(q), m01 = Vector128.Load(q + posStride), m02 = Vector128.Load(q + (2 * posStride)), m03 = Vector128.Load(q + (3 * posStride));
                        Vector128<float> m10 = Vector128.Load(q + (4 * posStride)), m11 = Vector128.Load(q + (5 * posStride)), m12 = Vector128.Load(q + (6 * posStride)), m13 = Vector128.Load(q + (7 * posStride));
                        Vector128<float> m20 = Vector128.Load(q + (8 * posStride)), m21 = Vector128.Load(q + (9 * posStride)), m22 = Vector128.Load(q + (10 * posStride)), m23 = Vector128.Load(q + (11 * posStride));
                        Vector128<float> m30 = Vector128.Load(q + (12 * posStride)), m31 = Vector128.Load(q + (13 * posStride)), m32 = Vector128.Load(q + (14 * posStride)), m33 = Vector128.Load(q + (15 * posStride));

                        // A^T M: for each column j, a0 = m0 + m1 + m2, a1 = m1 - m2 - m3.
                        Vector128<float> a00 = m00 + m10 + m20, a10 = m10 - m20 - m30;
                        Vector128<float> a01 = m01 + m11 + m21, a11 = m11 - m21 - m31;
                        Vector128<float> a02 = m02 + m12 + m22, a12 = m12 - m22 - m32;
                        Vector128<float> a03 = m03 + m13 + m23, a13 = m13 - m23 - m33;

                        // (A^T M) A: for each row i, y0 = a0 + a1 + a2, y1 = a1 - a2 - a3.
                        Vector128<float> y00 = a00 + a01 + a02, y01 = a01 - a02 - a03;
                        Vector128<float> y10 = a10 + a11 + a12, y11 = a11 - a12 - a13;

                        int ox = 2 * tx0;
                        if (ox + 8 <= _outW)
                        {
                            Vector128.Store(ZipLow(y00, y01), row0 + ox);
                            Vector128.Store(ZipHigh(y00, y01), row0 + ox + 4);
                            if (second)
                            {
                                Vector128.Store(ZipLow(y10, y11), row1 + ox);
                                Vector128.Store(ZipHigh(y10, y11), row1 + ox + 4);
                            }

                            continue;
                        }

                        Vector128.Store(y00, spill);
                        Vector128.Store(y01, spill + 4);
                        Vector128.Store(y10, spill + 8);
                        Vector128.Store(y11, spill + 12);
                        for (int l = 0; l < 4 && ox + (2 * l) < _outW; l++)
                        {
                            int x = ox + (2 * l);
                            row0[x] = spill[l];
                            if (x + 1 < _outW)
                            {
                                row0[x + 1] = spill[4 + l];
                            }

                            if (second)
                            {
                                row1[x] = spill[8 + l];
                                if (x + 1 < _outW)
                                {
                                    row1[x + 1] = spill[12 + l];
                                }
                            }
                        }
                    }
                }

                if (_epilogue is not null)
                {
                    // The chunk's rows of this channel are one contiguous run of the plane.
                    long start = (long)(2 * ty0) * _outW;
                    _epilogue.Apply(chanOut + start, 1, co, (int)start, 1, outRows * _outW, KernelMode.Fast);
                }
            }
        }
    }
}
