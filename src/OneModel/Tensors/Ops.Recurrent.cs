using System;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Kernels.Exact;
namespace RecursiveAI.OneModel.Tensors;

/// <summary>
/// The weights of one direction of one recurrent layer, in the gate layout PyTorch documents (<c>torch.nn.LSTM</c>, <c>torch.nn.GRU</c>):
/// LSTM gates are stacked input, forget, cell, output (<c>4H</c> rows); GRU gates reset, update, new (<c>3H</c> rows).
/// </summary>
/// <param name="WIh">Input weights <c>[G·H, inputSize]</c>.</param>
/// <param name="WHh">Recurrent weights <c>[G·H, H]</c>.</param>
/// <param name="BIh">Input bias <c>[G·H]</c>, or an empty tensor for none.</param>
/// <param name="BHh">Recurrent bias <c>[G·H]</c>, or an empty tensor for none.</param>
/// <param name="H0">Initial hidden state <c>[batch, H]</c>, or an empty tensor for zeros.</param>
/// <param name="C0">Initial cell state <c>[batch, H]</c> (LSTM only), or an empty tensor for zeros.</param>
public readonly record struct RnnLayer(Tensor WIh, Tensor WHh, Tensor BIh = default, Tensor BHh = default, Tensor H0 = default, Tensor C0 = default);

public static partial class Ops
{
    // ---------------------------------------------------------------------------------------------------------------- recurrent layers

    /// <summary>
    /// One direction of one LSTM layer over a whole sequence: for each step <c>i,f,g,o = σ,σ,tanh,σ (W_ih·x + b_ih + W_hh·h + b_hh)</c>,
    /// <c>c = f·c' + i·g</c>, <c>h = o·tanh(c)</c>. Returns every step's hidden state <c>[T, B, H]</c> (the final state of a forward pass is
    /// step <c>T-1</c>, of a reverse pass step 0). Matrix products run on the blocked GEMM, activations on the lane loops, and
    /// backward is backpropagation through time.
    /// Record: inputs x, wIh, wHh, bIh, bHh, h0, c0; slots 0 x, 1 wIh, 2 wHh, 3 activated gates <c>[T,B,4H]</c>, 4 cell states
    /// <c>[T,B,H]</c>, 5 tanh of the cell states, 6 the output, 7 h0 and 8 c0 (empty when not given); Ints = [reverse, T, B, I, H].
    /// </summary>
    /// <param name="x">Input <c>[T, B, I]</c>, time first.</param>
    /// <param name="wIh">Input weights <c>[4H, I]</c>.</param>
    /// <param name="wHh">Recurrent weights <c>[4H, H]</c>.</param>
    /// <param name="bIh">Optional input bias <c>[4H]</c>.</param>
    /// <param name="bHh">Optional recurrent bias <c>[4H]</c>.</param>
    /// <param name="h0">Optional initial hidden state <c>[B, H]</c>; zeros when empty.</param>
    /// <param name="c0">Optional initial cell state <c>[B, H]</c>; zeros when empty.</param>
    /// <param name="reverse">Run from the last step to the first; the output is still indexed by time.</param>
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    /// <exception cref="TensorShapeException">A shape does not fit, or a dimension is zero.</exception>
    public static Tensor LstmLayer(in Tensor x, in Tensor wIh, in Tensor wHh, in Tensor bIh = default, in Tensor bHh = default,
        in Tensor h0 = default, in Tensor c0 = default, bool reverse = false, BufferPool? pool = null)
    {
        int h = CheckRnn("LstmLayer", 4, x, wIh, wHh, bIh, bHh, h0, c0, out int steps, out int batch, out int inSize);
        bool exact = TensorCompute.Exact;
        bool recording = Recording(x, wIh, wHh, bIh, bHh, h0, c0);
        Tensor xc = Contig(x, out bool xCopied), wi = Contig(wIh, out bool wiCopied), wh = Contig(wHh, out bool whCopied);
        Tensor h0c = h0.IsEmpty ? default : Contig(h0, out _), c0c = c0.IsEmpty ? default : Contig(c0, out _);
        int h4 = 4 * h, bh = batch * h;

        Tensor y = NewOut(Shape.Create(steps, batch, h), pool);
        Tensor gates = NewOut(Shape.Create(steps, batch, h4), null);
        Tensor cseq = NewOut(Shape.Create(steps, batch, h), null);
        Tensor tanhC = NewOut(Shape.Create(steps, batch, h), null);
        Tensor whT = Tensor.Create(DType.F32, Shape.Create(h, h4));
        Tensor rec = Tensor.Create(DType.F32, Shape.Create(batch, h4));
        Tensor zeros = Tensor.Zeros(DType.F32, Shape.Create(batch, h));
        Tensor bias = Tensor.Create(DType.F32, Shape.Create(h4));
        try
        {
            Span<float> gv = gates.MutableData<float>(), cv = cseq.MutableData<float>(), tv = tanhC.MutableData<float>(), yv = y.MutableData<float>();
            Span<float> rv = rec.MutableData<float>(), bv = bias.MutableData<float>();
            Dense.Transpose(wh.InternalData<float>(), h, h4, h, whT.MutableData<float>());
            Dense.Gemm(false, true, steps * batch, h4, inSize, xc.InternalData<float>(), inSize, wi.InternalData<float>(), inSize, gv, h4);
            SumBias(bIh, bHh, bv);
            bool hasBias = !bIh.IsEmpty || !bHh.IsEmpty;
            ReadOnlySpan<float> zv = zeros.InternalData<float>();
            for (int f = 0; f < steps; f++)
            {
                int t = reverse ? steps - 1 - f : f;
                int pt = reverse ? t + 1 : t - 1;
                bool first = f == 0;
                ReadOnlySpan<float> hp = first ? (h0c.IsEmpty ? zv : h0c.InternalData<float>()) : yv.Slice(pt * bh, bh);
                ReadOnlySpan<float> cp = first ? (c0c.IsEmpty ? zv : c0c.InternalData<float>()) : cv.Slice(pt * bh, bh);
                Dense.Gemm(false, false, batch, h4, h, hp, h, whT.InternalData<float>(), h4, rv, h4);
                for (int b = 0; b < batch; b++)
                {
                    Span<float> g = gv.Slice(((t * batch) + b) * h4, h4);
                    ReadOnlySpan<float> rr = rv.Slice(b * h4, h4);
                    TrainElementwise.Binary(TensorCompute.Path, exact, new AddBin(), g, rr, g, h4);
                    if (hasBias)
                    {
                        TrainElementwise.Binary(TensorCompute.Path, exact, new AddBin(), g, bv, g, h4);
                    }

                    VectorMaps.Map1(new SigmoidUnary(), g.Slice(0, 2 * h), g.Slice(0, 2 * h));
                    VectorMaps.Map1(new TanhUnary(), g.Slice(2 * h, h), g.Slice(2 * h, h));
                    VectorMaps.Map1(new SigmoidUnary(), g.Slice(3 * h, h), g.Slice(3 * h, h));
                    Span<float> c = cv.Slice(((t * batch) + b) * h, h);
                    ReadOnlySpan<float> cprev = cp.Slice(b * h, h);
                    for (int j = 0; j < h; j++)
                    {
                        float fc = g[h + j] * cprev[j];
                        float ig = g[j] * g[(2 * h) + j];
                        c[j] = fc + ig;
                    }

                    Span<float> tc = tv.Slice(((t * batch) + b) * h, h);
                    VectorMaps.Map1(new TanhUnary(), c, tc);
                    Span<float> hout = yv.Slice(((t * batch) + b) * h, h);
                    for (int j = 0; j < h; j++)
                    {
                        hout[j] = g[(3 * h) + j] * tc[j];
                    }
                }
            }
        }
        finally
        {
            whT.Dispose();
            rec.Dispose();
            zeros.Dispose();
            bias.Dispose();
        }

        if (!recording)
        {
            gates.Dispose();
            cseq.Dispose();
            tanhC.Dispose();
            Free(xc, xCopied);
            Free(wi, wiCopied);
            Free(wh, whCopied);
            return y;
        }

        BackwardRecord br = BackwardRecord.Create(OpKind.Lstm, 9, x, wIh, wHh, bIh, bHh, h0, c0);
        SaveContig(br, 0, xc, xCopied);
        SaveContig(br, 1, wi, wiCopied);
        SaveContig(br, 2, wh, whCopied);
        br.Save(3, gates, owned: true);
        br.Save(4, cseq, owned: true);
        br.Save(5, tanhC, owned: true);
        br.Save(6, y);
        if (!h0c.IsEmpty)
        {
            br.Save(7, h0c);
        }

        if (!c0c.IsEmpty)
        {
            br.Save(8, c0c);
        }

        br.SetInts(reverse ? 1 : 0, steps, batch, inSize, h);
        return br.Attach(y);
    }

    /// <summary>
    /// One direction of one GRU layer over a whole sequence, with PyTorch's formulation: <c>r = σ(W_ir x + b_ir + W_hr h' + b_hr)</c>,
    /// <c>z = σ(W_iz x + b_iz + W_hz h' + b_hz)</c>, <c>n = tanh(W_in x + b_in + r·(W_hn h' + b_hn))</c>, <c>h = (1−z)·n + z·h'</c>.
    /// Returns every step's hidden state <c>[T, B, H]</c>.
    /// Record: inputs x, wIh, wHh, bIh, bHh, h0; slots 0 x, 1 wIh, 2 wHh, 3 activated r,z,n <c>[T,B,3H]</c>, 4 the hidden-side
    /// <c>W_hn h' + b_hn</c> <c>[T,B,H]</c>, 5 the output, 6 h0 (empty when not given); Ints = [reverse, T, B, I, H].
    /// </summary>
    /// <param name="x">Input <c>[T, B, I]</c>.</param>
    /// <param name="wIh">Input weights <c>[3H, I]</c>.</param>
    /// <param name="wHh">Recurrent weights <c>[3H, H]</c>.</param>
    /// <param name="bIh">Optional input bias <c>[3H]</c>.</param>
    /// <param name="bHh">Optional recurrent bias <c>[3H]</c>.</param>
    /// <param name="h0">Optional initial state <c>[B, H]</c>; zeros when empty.</param>
    /// <param name="reverse">Run from the last step to the first.</param>
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    /// <exception cref="TensorShapeException">A shape does not fit, or a dimension is zero.</exception>
    public static Tensor GruLayer(in Tensor x, in Tensor wIh, in Tensor wHh, in Tensor bIh = default, in Tensor bHh = default,
        in Tensor h0 = default, bool reverse = false, BufferPool? pool = null)
    {
        int h = CheckRnn("GruLayer", 3, x, wIh, wHh, bIh, bHh, h0, default, out int steps, out int batch, out int inSize);
        bool exact = TensorCompute.Exact;
        bool recording = Recording(x, wIh, wHh, bIh, bHh, h0);
        Tensor xc = Contig(x, out bool xCopied), wi = Contig(wIh, out bool wiCopied), wh = Contig(wHh, out bool whCopied);
        Tensor h0c = h0.IsEmpty ? default : Contig(h0, out _);
        int h3 = 3 * h, bh = batch * h;

        Tensor y = NewOut(Shape.Create(steps, batch, h), pool);
        Tensor gates = NewOut(Shape.Create(steps, batch, h3), null);
        Tensor hn = NewOut(Shape.Create(steps, batch, h), null);
        Tensor whT = Tensor.Create(DType.F32, Shape.Create(h, h3));
        Tensor rec = Tensor.Create(DType.F32, Shape.Create(batch, h3));
        Tensor zeros = Tensor.Zeros(DType.F32, Shape.Create(batch, h));
        bool biCopied = false, bhCopied = false;
        try
        {
            Span<float> gv = gates.MutableData<float>(), nv = hn.MutableData<float>(), yv = y.MutableData<float>(), rv = rec.MutableData<float>();
            Dense.Transpose(wh.InternalData<float>(), h, h3, h, whT.MutableData<float>());
            Dense.Gemm(false, true, steps * batch, h3, inSize, xc.InternalData<float>(), inSize, wi.InternalData<float>(), inSize, gv, h3);
            ReadOnlySpan<float> zv = zeros.InternalData<float>();
            Tensor biT = bIh.IsEmpty ? default : Contig(bIh, out biCopied);
            Tensor bhT = bHh.IsEmpty ? default : Contig(bHh, out bhCopied);
            ReadOnlySpan<float> bi = biT.IsEmpty ? default : biT.InternalData<float>();
            ReadOnlySpan<float> bhv = bhT.IsEmpty ? default : bhT.InternalData<float>();
            for (int f = 0; f < steps; f++)
            {
                int t = reverse ? steps - 1 - f : f;
                int pt = reverse ? t + 1 : t - 1;
                ReadOnlySpan<float> hp = f == 0 ? (h0c.IsEmpty ? zv : h0c.InternalData<float>()) : yv.Slice(pt * bh, bh);
                Dense.Gemm(false, false, batch, h3, h, hp, h, whT.InternalData<float>(), h3, rv, h3);
                for (int b = 0; b < batch; b++)
                {
                    Span<float> g = gv.Slice(((t * batch) + b) * h3, h3);
                    Span<float> rr = rv.Slice(b * h3, h3);
                    if (!bi.IsEmpty)
                    {
                        TrainElementwise.Binary(TensorCompute.Path, exact, new AddBin(), g, bi, g, h3);
                    }

                    if (!bhv.IsEmpty)
                    {
                        TrainElementwise.Binary(TensorCompute.Path, exact, new AddBin(), rr, bhv, rr, h3);
                    }

                    // r and z see the sum of both sides; n sees the input side plus r times the hidden side.
                    TrainElementwise.Binary(TensorCompute.Path, exact, new AddBin(), g, rr, g, 2 * h);
                    VectorMaps.Map1(new SigmoidUnary(), g.Slice(0, 2 * h), g.Slice(0, 2 * h));
                    Span<float> hside = nv.Slice(((t * batch) + b) * h, h);
                    rr.Slice(2 * h, h).CopyTo(hside);
                    for (int j = 0; j < h; j++)
                    {
                        g[(2 * h) + j] += g[j] * hside[j];
                    }

                    VectorMaps.Map1(new TanhUnary(), g.Slice(2 * h, h), g.Slice(2 * h, h));
                    Span<float> hout = yv.Slice(((t * batch) + b) * h, h);
                    ReadOnlySpan<float> hprev = hp.Slice(b * h, h);
                    for (int j = 0; j < h; j++)
                    {
                        float z = g[h + j];
                        float keep = (1f - z) * g[(2 * h) + j];
                        hout[j] = keep + (z * hprev[j]);
                    }
                }
            }

            Free(biT, biCopied);
            Free(bhT, bhCopied);
        }
        finally
        {
            whT.Dispose();
            rec.Dispose();
            zeros.Dispose();
        }

        if (!recording)
        {
            gates.Dispose();
            hn.Dispose();
            Free(xc, xCopied);
            Free(wi, wiCopied);
            Free(wh, whCopied);
            return y;
        }

        BackwardRecord br = BackwardRecord.Create(OpKind.Gru, 7, x, wIh, wHh, bIh, bHh, h0);
        SaveContig(br, 0, xc, xCopied);
        SaveContig(br, 1, wi, wiCopied);
        SaveContig(br, 2, wh, whCopied);
        br.Save(3, gates, owned: true);
        br.Save(4, hn, owned: true);
        br.Save(5, y);
        if (!h0c.IsEmpty)
        {
            br.Save(6, h0c);
        }

        br.SetInts(reverse ? 1 : 0, steps, batch, inSize, h);
        return br.Attach(y);
    }

    /// <summary>
    /// A multi-layer, optionally bidirectional LSTM (<c>torch.nn.LSTM</c> semantics, no dropout). <paramref name="layers"/> holds one entry
    /// per layer and direction in PyTorch's order: layer 0 forward, layer 0 reverse, layer 1 forward, … The input of layer l+1 is the
    /// concatenation of layer l's forward and reverse outputs. Built from <see cref="LstmLayer"/>, <see cref="Concat"/> and views, so
    /// gradients flow through every layer by the ordinary engine.
    /// </summary>
    /// <param name="x">Input <c>[T, B, I]</c>, or <c>[B, T, I]</c> with <paramref name="batchFirst"/>.</param>
    /// <param name="layers">The weights; the length must be a multiple of the direction count (1 or 2).</param>
    /// <param name="bidirectional">Run each layer in both directions.</param>
    /// <param name="batchFirst">Input and output are batch-major.</param>
    /// <param name="pool">Pool for the results; default <see cref="BufferPool.Shared"/>.</param>
    /// <returns>The last layer's output <c>[T, B, dirs·H]</c> (or batch-major).</returns>
    public static Tensor Lstm(in Tensor x, ReadOnlySpan<RnnLayer> layers, bool bidirectional = false, bool batchFirst = false, BufferPool? pool = null) =>
        Stack(x, layers, bidirectional, batchFirst, pool, gru: false);

    /// <summary>A multi-layer, optionally bidirectional GRU; see <see cref="Lstm"/> for the layer order and shapes.</summary>
    /// <param name="x">Input <c>[T, B, I]</c>, or <c>[B, T, I]</c> with <paramref name="batchFirst"/>.</param>
    /// <param name="layers">The weights, one entry per layer and direction (<c>C0</c> is ignored).</param>
    /// <param name="bidirectional">Run each layer in both directions.</param>
    /// <param name="batchFirst">Input and output are batch-major.</param>
    /// <param name="pool">Pool for the results; default <see cref="BufferPool.Shared"/>.</param>
    public static Tensor Gru(in Tensor x, ReadOnlySpan<RnnLayer> layers, bool bidirectional = false, bool batchFirst = false, BufferPool? pool = null) =>
        Stack(x, layers, bidirectional, batchFirst, pool, gru: true);

    private static Tensor Stack(in Tensor x, ReadOnlySpan<RnnLayer> layers, bool bidirectional, bool batchFirst, BufferPool? pool, bool gru)
    {
        int dirs = bidirectional ? 2 : 1;
        if (layers.Length == 0 || layers.Length % dirs != 0)
        {
            throw new ArgumentException($"Expected a multiple of {dirs} layer entries, got {layers.Length}.", nameof(layers));
        }

        if (x.Rank != 3)
        {
            throw new TensorShapeException($"A recurrent network needs a rank-3 input, not {x.Shape}.");
        }

        Tensor cur = batchFirst ? x.Transpose(0, 1) : x;
        Tensor prev = default;
        for (int l = 0; l < layers.Length / dirs; l++)
        {
            Tensor outF = RunLayer(cur, layers[l * dirs], false, pool, gru);
            Tensor next = outF;
            if (bidirectional)
            {
                Tensor outR = RunLayer(cur, layers[(l * dirs) + 1], true, pool, gru);
                next = Concat([outF, outR], 2, pool);
            }

            prev = next;
            cur = next;
        }

        return batchFirst ? prev.Transpose(0, 1) : prev;
    }

    private static Tensor RunLayer(in Tensor x, in RnnLayer w, bool reverse, BufferPool? pool, bool gru) =>
        gru ? GruLayer(x, w.WIh, w.WHh, w.BIh, w.BHh, w.H0, reverse, pool)
            : LstmLayer(x, w.WIh, w.WHh, w.BIh, w.BHh, w.H0, w.C0, reverse, pool);

    private static void SumBias(in Tensor a, in Tensor b, Span<float> dst)
    {
        dst.Clear();
        if (!a.IsEmpty)
        {
            Tensor c = Contig(a, out bool copied);
            c.InternalData<float>().CopyTo(dst);
            Free(c, copied);
        }

        if (!b.IsEmpty)
        {
            Tensor c = Contig(b, out bool copied);
            ReadOnlySpan<float> s = c.InternalData<float>();
            for (int i = 0; i < dst.Length; i++)
            {
                dst[i] += s[i];
            }

            Free(c, copied);
        }
    }

    private static int CheckRnn(string name, int gateCount, in Tensor x, in Tensor wIh, in Tensor wHh, in Tensor bIh, in Tensor bHh,
        in Tensor h0, in Tensor c0, out int steps, out int batch, out int inSize)
    {
        RequireFloat32(x);
        RequireFloat32(wIh);
        RequireFloat32(wHh);
        if (x.Rank != 3 || wIh.Rank != 2 || wHh.Rank != 2)
        {
            throw new TensorShapeException($"{name} needs x [T,B,I], wIh [{gateCount}H,I] and wHh [{gateCount}H,H], not {x.Shape}, {wIh.Shape}, {wHh.Shape}.");
        }

        steps = x.Shape[0];
        batch = x.Shape[1];
        inSize = x.Shape[2];
        int rows = wIh.Shape[0], h = rows / gateCount;
        if (h < 1 || rows != gateCount * h || wIh.Shape[1] != inSize || wHh.Shape[0] != rows || wHh.Shape[1] != h)
        {
            throw new TensorShapeException($"{name} weights {wIh.Shape} and {wHh.Shape} do not fit input {x.Shape}.");
        }

        if (steps < 1 || batch < 1 || inSize < 1)
        {
            throw new TensorShapeException($"{name} needs at least one step, one batch row and one input feature, got {x.Shape}.");
        }

        foreach (Tensor b in new[] { bIh, bHh })
        {
            if (!b.IsEmpty)
            {
                RequireFloat32(b);
                if (b.Rank != 1 || b.Shape[0] != rows)
                {
                    throw new TensorShapeException($"{name} biases must have shape [{rows}], not {b.Shape}.");
                }
            }
        }

        foreach (Tensor s in new[] { h0, c0 })
        {
            if (!s.IsEmpty)
            {
                RequireFloat32(s);
                if (s.Rank != 2 || s.Shape[0] != batch || s.Shape[1] != h)
                {
                    throw new TensorShapeException($"{name} initial states must have shape [{batch},{h}], not {s.Shape}.");
                }
            }
        }

        return h;
    }
}
