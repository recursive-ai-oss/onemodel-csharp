using System;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Runs <see cref="Attention.Compute"/> over all its work items in parallel. A work item is a (batch, head, block of 32 queries)
/// triple that writes its own output rows, so which thread runs which item is invisible in the result. Each item is one chunk, a
/// constant of the problem shape.
/// </summary>
internal static unsafe class AttentionDriver
{
    [ThreadStatic]
    private static ItemJob? t_job;

    /// <summary>
    /// Computes O = softmax(scale * Q K^T + mask) V for every (batch, head) described by <paramref name="s"/>. Spans must cover the
    /// extents the strides imply (checked by <see cref="Attention.Compute"/> in every item, before any pointer is formed here).
    /// </summary>
    /// <param name="exact">The exact flavour of docs/kernels-attention.md, or the fast one.</param>
    /// <param name="path">The session's instruction set, from which the lane width is taken.</param>
    /// <param name="s">Geometry; the head dimension must be even.</param>
    /// <param name="q">Queries.</param>
    /// <param name="k">Keys.</param>
    /// <param name="v">Values.</param>
    /// <param name="mask">Additive mask, or empty.</param>
    /// <param name="o">Output.</param>
    /// <param name="threads">Upper bound on threads; changes speed only.</param>
    public static void Run(bool exact, KernelPath path, in AttentionShape s, ReadOnlySpan<float> q, ReadOnlySpan<float> k,
        ReadOnlySpan<float> v, ReadOnlySpan<float> mask, Span<float> o, int threads)
    {
        int items = Attention.ItemCount(s);
        if (items == 0)
        {
            return;
        }

        // Fail on bad extents here, on the calling thread, rather than inside a chunk.
        int scratchLen = Attention.ScratchLength(s.HeadDim);
        Attention.Compute(exact, IsaOf(path), s, q, k, v, mask, o, 0, 0, new Span<float>(Scratch.Get(Scratch.Attention, scratchLen), scratchLen));

        fixed (float* pq = q, pk = k, pv = v, pmask = mask, po = o)
        {
            ItemJob job = t_job ??= new ItemJob();
            job.Exact = exact;
            job.Isa = IsaOf(path);
            job.Shape = s;
            job.Q = pq;
            job.QLen = q.Length;
            job.K = pk;
            job.KLen = k.Length;
            job.V = pv;
            job.VLen = v.Length;
            job.Mask = pmask;
            job.MaskLen = mask.Length;
            job.O = po;
            job.OLen = o.Length;
            ParallelFor.Run(job, items, threads);
        }
    }

    private static VecIsa IsaOf(KernelPath path) => path switch
    {
        KernelPath.Avx512 => VecIsa.V512,
        KernelPath.Avx2 => VecIsa.V256,
        KernelPath.Neon or KernelPath.Portable128 => VecIsa.V128,
        _ => VecIsa.Scalar,
    };

    private sealed class ItemJob : ChunkJob
    {
        public bool Exact;
        public VecIsa Isa;
        public AttentionShape Shape;
        public float* Q, K, V, Mask, O;
        public int QLen, KLen, VLen, MaskLen, OLen;

        // Bounds: the spans these pointers came from were validated against the strides by Attention.Compute before the job was
        // queued, and each is kept alive (fixed) until ParallelFor.Run returns.
        public override void RunChunk(int chunk)
        {
            var scratch = new Span<float>(Scratch.Get(Scratch.Attention, Attention.ScratchLength(Shape.HeadDim)), Attention.ScratchLength(Shape.HeadDim));
            Attention.Compute(Exact, Isa, Shape, new ReadOnlySpan<float>(Q, QLen), new ReadOnlySpan<float>(K, KLen),
                new ReadOnlySpan<float>(V, VLen), new ReadOnlySpan<float>(Mask, MaskLen), new Span<float>(O, OLen), chunk, chunk + 1, scratch);
        }
    }
}
