using System;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Runs the time steps of one recurrent layer (one or two directions) as a sequence of <see cref="ParallelFor"/> calls, one per
/// step (two for a GRU with <c>linear_before_reset = 0</c>). A chunk is a slice of <see cref="RecurrentKernel.ChunkWidth"/> hidden
/// units of one direction; the chunk list depends on the hidden size and the number of directions only. The hidden state ping-pongs
/// between two buffers, so a chunk reads the whole previous state while other chunks write the new one.
/// </summary>
/// <remarks>
/// A <see cref="ChunkJob"/> runs on other threads, where the spans of the caller's stack are not reachable, so the job keeps raw
/// pointers to the layer's buffers. Every pointer comes from a span pinned by a <c>fixed</c> statement that encloses the whole run, and
/// is turned back into a span with the span's own length before use, so all accesses in the step code are bounds-checked as before.
/// </remarks>
internal static unsafe class RecurrentSteps
{
    [ThreadStatic]
    private static StepJob? t_job;

    /// <summary>
    /// Runs <paramref name="maxLen"/> steps of the given directions. Their hidden state starts in each direction's <c>Hbuf</c> (cell state in
    /// <c>Cbuf</c>). Returns whether the final hidden state is in <c>Hnext</c> (true when <paramref name="maxLen"/> is odd).
    /// </summary>
    public static bool Run(ref RecurrentDirection d0, ref RecurrentDirection d1, int dirs, int maxLen)
    {
        if (maxLen <= 0 || d0.Batch <= 0)
        {
            return false;
        }

        if (!UsesThreads(ref d0, dirs))
        {
            RunInline(ref d0, ref d1, dirs, maxLen);
            return (maxLen & 1) == 1;
        }

        StepJob job = t_job ??= new StepJob();
        // Pin every buffer of both directions for the whole run; the job copies the pointers and lengths.
        fixed (float* xw0 = d0.Xw, rec0 = d0.Rec, hb0 = d0.Hbuf, cb0 = d0.Cbuf, tmp0 = d0.Tmp, tmp20 = d0.Tmp2, tmp30 = d0.Tmp3, hn0 = d0.Hnext,
            rt0 = d0.RT, wb0 = d0.Wb, rb0 = d0.Rb, pe0 = d0.Peep, y0 = d0.Y)
        fixed (int* lens0 = d0.Lens)
        {
            job.D0.Set(ref d0, xw0, rec0, hb0, cb0, tmp0, tmp20, tmp30, hn0, rt0, wb0, rb0, pe0, y0, lens0);
            if (dirs == 2)
            {
                fixed (float* xw1 = d1.Xw, rec1 = d1.Rec, hb1 = d1.Hbuf, cb1 = d1.Cbuf, tmp1 = d1.Tmp, tmp21 = d1.Tmp2, tmp31 = d1.Tmp3, hn1 = d1.Hnext,
                    rt1 = d1.RT, wb1 = d1.Wb, rb1 = d1.Rb, pe1 = d1.Peep, y1 = d1.Y)
                fixed (int* lens1 = d1.Lens)
                {
                    job.D1.Set(ref d1, xw1, rec1, hb1, cb1, tmp1, tmp21, tmp31, hn1, rt1, wb1, rb1, pe1, y1, lens1);
                    Loop(job, ref d0, dirs, maxLen);
                }
            }
            else
            {
                Loop(job, ref d0, 1, maxLen);
            }
        }

        job.D0.Clear();
        job.D1.Clear();
        return (maxLen & 1) == 1;
    }

    // Single-thread time of one step in nanoseconds, about 16 multiply-adds per nanosecond for the streaming product: decides only how
    // many threads are worth waking (ParallelFor), never a chunk boundary.
    private static long StepWork(ref RecurrentDirection d, int dirs) =>
        (long)dirs * d.Batch * RecurrentKernel.Gates(d.Cell) * d.Hidden * d.Hidden / 16;

    private static bool UsesThreads(ref RecurrentDirection d, int dirs) =>
        Math.Min(d.Threads, StepWork(ref d, dirs) / ParallelFor.MinWorkPerThread) > 1;

    // The steps on the calling thread, chunk by chunk in order: the same arithmetic as the job, without the per-chunk cost of rebuilding
    // the layer's views, which matters when a step is only a few microseconds (batch 1).
    private static void RunInline(ref RecurrentDirection d0, ref RecurrentDirection d1, int dirs, int maxLen)
    {
        int hid = d0.Hidden;
        bool gruSplit = d0.Cell == RecurrentCell.Gru && !d0.LinearBeforeReset;
        Span<float> a0 = d0.Hbuf, b0 = d0.Hnext, a1 = default, b1 = default;
        if (dirs == 2)
        {
            a1 = d1.Hbuf;
            b1 = d1.Hnext;
        }

        for (int s = 0; s < maxLen; s++)
        {
            bool even = (s & 1) == 0;
            d0.Hbuf = even ? a0 : b0;
            d0.Hnext = even ? b0 : a0;
            if (dirs == 2)
            {
                d1.Hbuf = even ? a1 : b1;
                d1.Hnext = even ? b1 : a1;
            }

            for (int phase = 0; phase < (gruSplit ? 2 : 1); phase++)
            {
                for (int j0 = 0; j0 < hid; j0 += RecurrentKernel.ChunkWidth)
                {
                    int n = Math.Min(RecurrentKernel.ChunkWidth, hid - j0);
                    RecurrentKernel.StepChunk(ref d0, phase, s, j0, n, gruSplit);
                    if (dirs == 2)
                    {
                        RecurrentKernel.StepChunk(ref d1, phase, s, j0, n, gruSplit);
                    }
                }
            }
        }

        d0.Hbuf = a0;
        d0.Hnext = b0;
        if (dirs == 2)
        {
            d1.Hbuf = a1;
            d1.Hnext = b1;
        }
    }

    private static void Loop(StepJob job, ref RecurrentDirection d0, int dirs, int maxLen)
    {
        int hid = d0.Hidden;
        bool gruSplit = d0.Cell == RecurrentCell.Gru && !d0.LinearBeforeReset;
        int perDir = (hid + RecurrentKernel.ChunkWidth - 1) / RecurrentKernel.ChunkWidth;
        job.ChunksPerDir = perDir;
        job.GruSplit = gruSplit;
        int chunks = dirs * perDir;

        long work = StepWork(ref d0, dirs);
        for (int s = 0; s < maxLen; s++)
        {
            job.Step = s;
            job.Phase = 0;
            ParallelFor.Run(job, chunks, d0.Threads, gruSplit ? work * 2 / 3 : work);
            if (gruSplit)
            {
                job.Phase = 1;
                ParallelFor.Run(job, chunks, d0.Threads, work / 3);
            }
        }
    }

    // One direction as pointers and lengths: what a chunk needs to rebuild a RecurrentDirection on its own thread.
    private struct DirPtrs
    {
        public RecurrentCell Cell;
        public bool Exact;
        public KernelPath Path;
        public int Seq, Batch, Hidden;
        public bool Reverse;
        public int XTimeStep, XBatchStep;
        public int YOffset, YTimeStep, YBatchStep;
        public RecurrentActivation F, G, Hact;
        public float Clip;
        public bool InputForget, LinearBeforeReset;

        public float* Xw, Rec, Hbuf, Cbuf, Tmp, Tmp2, Tmp3, Hnext, RT, Wb, Rb, Peep, Y;
        public int* Lens;
        public int XwLen, RecLen, HbufLen, CbufLen, TmpLen, Tmp2Len, Tmp3Len, HnextLen, RTLen, WbLen, RbLen, PeepLen, YLen, LensLen;

        public void Set(ref RecurrentDirection d, float* xw, float* rec, float* hbuf, float* cbuf, float* tmp, float* tmp2, float* tmp3, float* hnext,
            float* rt, float* wb, float* rb, float* peep, float* y, int* lens)
        {
            Cell = d.Cell;
            Exact = d.Exact;
            Path = d.Path;
            Seq = d.Seq;
            Batch = d.Batch;
            Hidden = d.Hidden;
            Reverse = d.Reverse;
            XTimeStep = d.XTimeStep;
            XBatchStep = d.XBatchStep;
            YOffset = d.YOffset;
            YTimeStep = d.YTimeStep;
            YBatchStep = d.YBatchStep;
            F = d.F;
            G = d.G;
            Hact = d.Hact;
            Clip = d.Clip;
            InputForget = d.InputForget;
            LinearBeforeReset = d.LinearBeforeReset;
            Xw = xw; XwLen = d.Xw.Length;
            Rec = rec; RecLen = d.Rec.Length;
            Hbuf = hbuf; HbufLen = d.Hbuf.Length;
            Cbuf = cbuf; CbufLen = d.Cbuf.Length;
            Tmp = tmp; TmpLen = d.Tmp.Length;
            Tmp2 = tmp2; Tmp2Len = d.Tmp2.Length;
            Tmp3 = tmp3; Tmp3Len = d.Tmp3.Length;
            Hnext = hnext; HnextLen = d.Hnext.Length;
            RT = rt; RTLen = d.RT.Length;
            Wb = wb; WbLen = d.Wb.Length;
            Rb = rb; RbLen = d.Rb.Length;
            Peep = peep; PeepLen = d.Peep.Length;
            Y = y; YLen = d.Y.Length;
            Lens = lens; LensLen = d.Lens.Length;
        }

        public void Clear() => this = default;

        // Spans over the pinned buffers, each with the length of the span it came from. Hbuf and Hnext swap on odd steps.
        public readonly RecurrentDirection ToDirection(int step)
        {
            var d = new RecurrentDirection
            {
                Cell = Cell,
                Exact = Exact,
                Path = Path,
                Threads = 1,
                Seq = Seq,
                Batch = Batch,
                Hidden = Hidden,
                Reverse = Reverse,
                XTimeStep = XTimeStep,
                XBatchStep = XBatchStep,
                YOffset = YOffset,
                YTimeStep = YTimeStep,
                YBatchStep = YBatchStep,
                F = F,
                G = G,
                Hact = Hact,
                Clip = Clip,
                InputForget = InputForget,
                LinearBeforeReset = LinearBeforeReset,
                Xw = new Span<float>(Xw, XwLen),
                Rec = new Span<float>(Rec, RecLen),
                Cbuf = new Span<float>(Cbuf, CbufLen),
                Tmp = new Span<float>(Tmp, TmpLen),
                Tmp2 = new Span<float>(Tmp2, Tmp2Len),
                Tmp3 = new Span<float>(Tmp3, Tmp3Len),
                RT = new ReadOnlySpan<float>(RT, RTLen),
                Wb = new ReadOnlySpan<float>(Wb, WbLen),
                Rb = new ReadOnlySpan<float>(Rb, RbLen),
                Peep = new ReadOnlySpan<float>(Peep, PeepLen),
                Y = new Span<float>(Y, YLen),
                Lens = new ReadOnlySpan<int>(Lens, LensLen),
            };
            var a = new Span<float>(Hbuf, HbufLen);
            var b = new Span<float>(Hnext, HnextLen);
            d.Hbuf = (step & 1) == 0 ? a : b;
            d.Hnext = (step & 1) == 0 ? b : a;
            return d;
        }
    }

    private sealed class StepJob : ChunkJob
    {
        public DirPtrs D0, D1;
        public int ChunksPerDir, Step, Phase;
        public bool GruSplit;

        public override void RunChunk(int chunk)
        {
            int dir = chunk / ChunksPerDir;
            int j0 = (chunk - (dir * ChunksPerDir)) * RecurrentKernel.ChunkWidth;
            ref DirPtrs p = ref (dir == 0 ? ref D0 : ref D1);
            RecurrentDirection d = p.ToDirection(Step);
            RecurrentKernel.StepChunk(ref d, Phase, Step, j0, Math.Min(RecurrentKernel.ChunkWidth, d.Hidden - j0), GruSplit);
        }
    }
}
