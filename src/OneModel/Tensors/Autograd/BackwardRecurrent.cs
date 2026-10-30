using System;
using RecursiveAI.OneModel.Kernels;
namespace RecursiveAI.OneModel.Tensors;

// Backpropagation through time for LSTM and GRU layers. Per step the gate gradients are plain IEEE products and sums in a fixed
// order; every matrix product is the blocked GEMM, and parameter gradients are single products over all steps (a sum over t and b
// in the GEMM's canonical order). Bias gradients sum in double in increasing (t, b) order and round once.
internal static partial class BackwardOps
{
    private static void LstmBackward(BackwardRecord r, in Tensor gy)
    {
        Tensor x = r.Saved(0).Unpack(), wIh = r.Saved(1).Unpack(), wHh = r.Saved(2).Unpack();
        Tensor gates = r.Saved(3).Unpack(), cseq = r.Saved(4).Unpack(), tanhC = r.Saved(5).Unpack(), y = r.Saved(6).Unpack();
        Tensor h0 = r.Saved(7).IsEmpty ? default : r.Saved(7).Unpack(), c0 = r.Saved(8).IsEmpty ? default : r.Saved(8).Unpack();
        bool reverse = r.Ints[0] != 0;
        int steps = (int)r.Ints[1], batch = (int)r.Ints[2], inSize = (int)r.Ints[3], h = (int)r.Ints[4];
        int h4 = 4 * h, bh = batch * h, rows = steps * batch;

        Tensor dG = Tensor.Create(DType.F32, Shape.Create(rows, h4));
        Tensor dhn = Tensor.Zeros(DType.F32, Shape.Create(batch, h));
        Tensor dcn = Tensor.Zeros(DType.F32, Shape.Create(batch, h));
        Tensor zeros = Tensor.Zeros(DType.F32, Shape.Create(batch, h));
        Tensor hPrev = Tensor.Create(DType.F32, Shape.Create(rows, h));
        try
        {
            ReadOnlySpan<float> gv = gates.InternalData<float>(), cv = cseq.InternalData<float>(), tv = tanhC.InternalData<float>(), yv = y.InternalData<float>(), gyv = gy.InternalData<float>();
            ReadOnlySpan<float> zv = zeros.InternalData<float>();
            Span<float> dgv = dG.MutableData<float>(), dh = dhn.MutableData<float>(), dc = dcn.MutableData<float>(), hp = hPrev.MutableData<float>();
            for (int s = 0; s < steps; s++)
            {
                int t = reverse ? s : steps - 1 - s;
                int pt = reverse ? t + 1 : t - 1;
                bool first = reverse ? t == steps - 1 : t == 0;
                ReadOnlySpan<float> cp = first ? (c0.IsEmpty ? zv : c0.InternalData<float>()) : cv.Slice(pt * bh, bh);
                ReadOnlySpan<float> hpv = first ? (h0.IsEmpty ? zv : h0.InternalData<float>()) : yv.Slice(pt * bh, bh);
                hpv.CopyTo(hp.Slice(t * bh, bh));
                for (int b = 0; b < batch; b++)
                {
                    int rowT = (t * batch) + b;
                    ReadOnlySpan<float> g = gv.Slice(rowT * h4, h4);
                    Span<float> d = dgv.Slice(rowT * h4, h4);
                    for (int j = 0; j < h; j++)
                    {
                        int k = (b * h) + j;
                        float gi = g[j], gf = g[h + j], gg = g[(2 * h) + j], go = g[(3 * h) + j];
                        float tc = tv[(rowT * h) + j];
                        float dH = gyv[(rowT * h) + j] + dh[k];
                        float dO = dH * tc;
                        float dC = dc[k] + ((dH * go) * (1f - (tc * tc)));
                        float dI = dC * gg, dF = dC * cp[k], dGg = dC * gi;
                        dc[k] = dC * gf;
                        d[j] = (dI * gi) * (1f - gi);
                        d[h + j] = (dF * gf) * (1f - gf);
                        d[(2 * h) + j] = dGg * (1f - (gg * gg));
                        d[(3 * h) + j] = (dO * go) * (1f - go);
                    }
                }

                // dh' = dG_t · W_hh  ([B,4H] x [4H,H])
                if (s < steps - 1 || r.NeedsInputGrad(5))
                {
                    Dense.Gemm(false, false, batch, h, h4, dgv.Slice(t * batch * h4, batch * h4), h4, wHh.InternalData<float>(), h, dh, h);
                }
            }

            if (r.NeedsInputGrad(0))
            {
                Tensor dx = Tensor.Create(DType.F32, x.Shape);
                Dense.Gemm(false, false, rows, inSize, h4, dgv, h4, wIh.InternalData<float>(), inSize, dx.MutableData<float>(), inSize);
                Engine.Emit(r, 0, dx, owned: true);
            }

            if (r.NeedsInputGrad(1))
            {
                Tensor dw = Tensor.Create(DType.F32, wIh.Shape);
                Dense.Gemm(true, false, h4, inSize, rows, dgv, h4, x.InternalData<float>(), inSize, dw.MutableData<float>(), inSize);
                Engine.Emit(r, 1, dw, owned: true);
            }

            if (r.NeedsInputGrad(2))
            {
                Tensor dw = Tensor.Create(DType.F32, wHh.Shape);
                Dense.Gemm(true, false, h4, h, rows, dgv, h4, hPrev.InternalData<float>(), h, dw.MutableData<float>(), h);
                Engine.Emit(r, 2, dw, owned: true);
            }

            for (int bi = 3; bi <= 4; bi++)
            {
                if (r.NeedsInputGrad(bi))
                {
                    Engine.Emit(r, bi, ColumnSums(dgv, rows, h4), owned: true);
                }
            }

            if (r.NeedsInputGrad(5))
            {
                Engine.Emit(r, 5, dhn.Clone(), owned: true);
            }

            if (r.NeedsInputGrad(6))
            {
                Engine.Emit(r, 6, dcn.Clone(), owned: true);
            }
        }
        finally
        {
            dG.Dispose();
            dhn.Dispose();
            dcn.Dispose();
            zeros.Dispose();
            hPrev.Dispose();
        }
    }

    private static void GruBackward(BackwardRecord r, in Tensor gy)
    {
        Tensor x = r.Saved(0).Unpack(), wIh = r.Saved(1).Unpack(), wHh = r.Saved(2).Unpack();
        Tensor gates = r.Saved(3).Unpack(), hn = r.Saved(4).Unpack(), y = r.Saved(5).Unpack();
        Tensor h0 = r.Saved(6).IsEmpty ? default : r.Saved(6).Unpack();
        bool reverse = r.Ints[0] != 0;
        int steps = (int)r.Ints[1], batch = (int)r.Ints[2], inSize = (int)r.Ints[3], h = (int)r.Ints[4];
        int h3 = 3 * h, bh = batch * h, rows = steps * batch;

        Tensor dX = Tensor.Create(DType.F32, Shape.Create(rows, h3));   // gradients at the input-side pre-activations (r, z, n)
        Tensor dH = Tensor.Create(DType.F32, Shape.Create(rows, h3));   // the same on the hidden side (r, z, r * n')
        Tensor dhn = Tensor.Zeros(DType.F32, Shape.Create(batch, h));
        Tensor direct = Tensor.Create(DType.F32, Shape.Create(batch, h));
        Tensor zeros = Tensor.Zeros(DType.F32, Shape.Create(batch, h));
        Tensor hPrev = Tensor.Create(DType.F32, Shape.Create(rows, h));
        try
        {
            ReadOnlySpan<float> gv = gates.InternalData<float>(), nv = hn.InternalData<float>(), yv = y.InternalData<float>(), gyv = gy.InternalData<float>(), zv = zeros.InternalData<float>();
            Span<float> dxv = dX.MutableData<float>(), dhv = dH.MutableData<float>(), dn = dhn.MutableData<float>(), dr = direct.MutableData<float>();
            Span<float> hp = hPrev.MutableData<float>();
            for (int s = 0; s < steps; s++)
            {
                int t = reverse ? s : steps - 1 - s;
                int pt = reverse ? t + 1 : t - 1;
                bool first = reverse ? t == steps - 1 : t == 0;
                ReadOnlySpan<float> hpv = first ? (h0.IsEmpty ? zv : h0.InternalData<float>()) : yv.Slice(pt * bh, bh);
                hpv.CopyTo(hp.Slice(t * bh, bh));
                for (int b = 0; b < batch; b++)
                {
                    int rowT = (t * batch) + b;
                    ReadOnlySpan<float> g = gv.Slice(rowT * h3, h3);
                    Span<float> dx = dxv.Slice(rowT * h3, h3), dhh = dhv.Slice(rowT * h3, h3);
                    for (int j = 0; j < h; j++)
                    {
                        int k = (b * h) + j;
                        float rg = g[j], zg = g[h + j], ng = g[(2 * h) + j];
                        float dHt = gyv[(rowT * h) + j] + dn[k];
                        float dz = dHt * (hpv[k] - ng);
                        float dNew = dHt * (1f - zg);
                        dr[k] = dHt * zg;
                        float dan = dNew * (1f - (ng * ng));
                        float drg = dan * nv[(rowT * h) + j];
                        float dar = (drg * rg) * (1f - rg);
                        float daz = (dz * zg) * (1f - zg);
                        dx[j] = dar;
                        dx[h + j] = daz;
                        dx[(2 * h) + j] = dan;
                        dhh[j] = dar;
                        dhh[h + j] = daz;
                        dhh[(2 * h) + j] = dan * rg;
                    }
                }

                // dh' = dh * z + dH_t · W_hh  ([B,3H] x [3H,H])
                Dense.Gemm(false, false, batch, h, h3, dhv.Slice(t * batch * h3, batch * h3), h3, wHh.InternalData<float>(), h, dn, h);
                for (int i = 0; i < bh; i++)
                {
                    dn[i] += dr[i];
                }
            }

            if (r.NeedsInputGrad(0))
            {
                Tensor dxo = Tensor.Create(DType.F32, x.Shape);
                Dense.Gemm(false, false, rows, inSize, h3, dxv, h3, wIh.InternalData<float>(), inSize, dxo.MutableData<float>(), inSize);
                Engine.Emit(r, 0, dxo, owned: true);
            }

            if (r.NeedsInputGrad(1))
            {
                Tensor dw = Tensor.Create(DType.F32, wIh.Shape);
                Dense.Gemm(true, false, h3, inSize, rows, dxv, h3, x.InternalData<float>(), inSize, dw.MutableData<float>(), inSize);
                Engine.Emit(r, 1, dw, owned: true);
            }

            if (r.NeedsInputGrad(2))
            {
                Tensor dw = Tensor.Create(DType.F32, wHh.Shape);
                Dense.Gemm(true, false, h3, h, rows, dhv, h3, hPrev.InternalData<float>(), h, dw.MutableData<float>(), h);
                Engine.Emit(r, 2, dw, owned: true);
            }

            if (r.NeedsInputGrad(3))
            {
                Engine.Emit(r, 3, ColumnSums(dxv, rows, h3), owned: true);
            }

            if (r.NeedsInputGrad(4))
            {
                Engine.Emit(r, 4, ColumnSums(dhv, rows, h3), owned: true);
            }

            if (r.NeedsInputGrad(5))
            {
                Engine.Emit(r, 5, dhn.Clone(), owned: true);
            }
        }
        finally
        {
            dX.Dispose();
            dH.Dispose();
            dhn.Dispose();
            direct.Dispose();
            zeros.Dispose();
            hPrev.Dispose();
        }
    }

    // out[c] = sum over rows of m[row, c], accumulated in double in increasing row order and rounded once.
    private static Tensor ColumnSums(ReadOnlySpan<float> m, int rows, int cols)
    {
        Tensor acc = Tensor.Zeros(DType.F64, Shape.Create(cols));
        Span<double> a = acc.MutableData<double>();
        for (int i = 0; i < rows; i++)
        {
            ReadOnlySpan<float> row = m.Slice(i * cols, cols);
            for (int c = 0; c < cols; c++)
            {
                a[c] += row[c];
            }
        }

        Tensor f = Tensor.Create(DType.F32, Shape.Create(cols));
        Span<float> d = f.MutableData<float>();
        for (int c = 0; c < cols; c++)
        {
            d[c] = (float)a[c];
        }

        acc.Dispose();
        return f;
    }
}

internal static partial class BackwardOps
{
    // Gradient of the CTC loss with respect to the log-probabilities: for every sequence the true partial derivative of its loss
    // (CtcLossMath.Gradient), times the incoming gradient and the reduction's weight (1/B/max(1,targetLength) for Mean).
    private static void CtcBackward(BackwardRecord r, in Tensor g)
    {
        Tensor lp = r.Saved(0).Unpack();
        ReadOnlySpan<long> ints = r.Ints;
        int blank = (int)ints[0];
        var reduction = (CtcReduction)ints[2];
        int batch = (int)ints[3];
        int steps = lp.Shape[0], classes = lp.Shape[2];
        int maxLen = 1;
        int offset = 0;
        var tgt = new int[ints.Length - 4 - (2 * batch)];
        for (int i = 0; i < tgt.Length; i++)
        {
            tgt[i] = (int)ints[4 + (2 * batch) + i];
        }

        for (int b = 0; b < batch; b++)
        {
            maxLen = Math.Max(maxLen, (2 * (int)ints[4 + batch + b]) + 1);
        }

        var alpha = new double[Math.Max(1, steps) * maxLen];
        var beta = new double[alpha.Length];
        var lse = new double[classes];
        Tensor dx = Tensor.Zeros(DType.F32, lp.Shape);
        Span<float> d = dx.MutableData<float>();
        ReadOnlySpan<float> gv = g.InternalData<float>();
        ReadOnlySpan<float> lpv = lp.InternalData<float>();
        for (int b = 0; b < batch; b++)
        {
            int len = (int)ints[4 + b], tl = (int)ints[4 + batch + b];
            ReadOnlySpan<int> t = tgt.AsSpan(offset, tl);
            offset += tl;
            double scale = reduction == CtcReduction.None ? gv[b] : gv[0];
            if (reduction == CtcReduction.Mean)
            {
                scale = scale / Math.Max(1, tl) / batch;
            }

            CtcLossMath.Gradient(lpv, d, steps, batch, classes, b, len, t, blank, scale, alpha, beta, lse);
        }

        Engine.Emit(r, 0, dx, owned: true);
    }
}

internal static partial class BackwardOps
{
    // The transposed convolution y = convT(x, W) is the adjoint of the convolution c = conv(y, W) with the same weights, strides and
    // padding, so dx = conv(g, W). The weight gradient is the convolution's weight gradient with the roles swapped: the convolution's
    // input is g and its output gradient is x (Chellapilla et al. 2006).
    private static void ConvTransposeBackward(Engine e, BackwardRecord r, in Tensor g)
    {
        Tensor x = r.Saved(0).Unpack(), w = r.Saved(1).Unpack();
        ReadOnlySpan<long> p = r.Ints;
        var cp = new Conv2dParams((int)p[0], (int)p[1], (int)p[2], (int)p[3], (int)p[6], (int)p[7], (int)p[8]);
        int n = x.Shape[0], cin = x.Shape[1], hh = x.Shape[2], ww = x.Shape[3];
        int cout = g.Shape[1], gh = g.Shape[2], gw = g.Shape[3];
        int groups = (int)p[8];
        if (r.NeedsInputGrad(0))
        {
            Engine.Emit(r, 0, Ops.Conv2d(g, w, default, cp), owned: true);
        }

        if (r.NeedsInputGrad(1))
        {
            // Convolution geometry: input g [N, Cout, gh, gw], output x-shaped [N, Cin, H, W], weights w [Cin, Cout/G, KH, KW].
            var s = new ConvShape(n, cout, gh, gw, cin, w.Shape[2], w.Shape[3], (int)p[0], (int)p[1], (int)p[2], (int)p[3], (int)p[4], (int)p[5],
                (int)p[6], (int)p[7], groups);
            int cg = cout / groups, og = cin / groups, kk = cg * s.KH * s.KW, plane = hh * ww;
            Tensor dw = Tensor.Zeros(DType.F32, w.Shape);
            Tensor cols = Tensor.Create(DType.F32, Shape.Create(plane, kk));
            Tensor tmp = Tensor.Create(DType.F32, Shape.Create(og, kk));
            ReadOnlySpan<float> gv = g.InternalData<float>(), xv = x.InternalData<float>();
            Span<float> dwv = dw.MutableData<float>();
            for (int img = 0; img < n; img++)
            {
                for (int grp = 0; grp < groups; grp++)
                {
                    Im2ColPatchMajor(s, gv.Slice(((img * cout) + (grp * cg)) * gh * gw, cg * gh * gw), cols.MutableData<float>());
                    Dense.Gemm(false, false, og, kk, plane, xv.Slice(((img * cin) + (grp * og)) * plane), plane, cols.InternalData<float>(), kk,
                        tmp.MutableData<float>(), kk);
                    Span<float> dst = dwv.Slice(grp * og * kk, og * kk);
                    ReadOnlySpan<float> t = tmp.InternalData<float>();
                    for (int i = 0; i < dst.Length; i++)
                    {
                        dst[i] += t[i];
                    }
                }
            }

            cols.Dispose();
            tmp.Dispose();
            Engine.Emit(r, 1, dw, owned: true);
        }

        if (r.NeedsInputGrad(2))
        {
            int plane = gh * gw;
            ReadOnlySpan<float> gv = g.InternalData<float>();
            Tensor acc = Tensor.Zeros(DType.F64, Shape.Create(cout));
            Span<double> a = acc.MutableData<double>();
            for (int img = 0; img < n; img++)
            {
                for (int co = 0; co < cout; co++)
                {
                    ReadOnlySpan<float> row = gv.Slice(((img * cout) + co) * plane, plane);
                    double t = 0;
                    for (int i = 0; i < row.Length; i++)
                    {
                        t += row[i];
                    }

                    a[co] += t;
                }
            }

            EmitDouble(e, r, 2, acc, true);
        }
    }

    private static void ResizeBackward(BackwardRecord r, in Tensor g)
    {
        ReadOnlySpan<long> p = r.Ints;
        var mode = (InterpolationMode)p[0];
        bool align = p[1] != 0;
        int outH = (int)p[2], outW = (int)p[3];
        Shape xs = r.InputShape(0);
        int planes = xs[0] * xs[1], h = xs[2], w = xs[3];
        Tensor dx = Tensor.Zeros(DType.F32, xs);
        Span<float> d = dx.MutableData<float>();
        ReadOnlySpan<float> gv = g.InternalData<float>();
        for (int pl = 0; pl < planes; pl++)
        {
            Span<float> dp = d.Slice(pl * h * w, h * w);
            for (int oy = 0; oy < outH; oy++)
            {
                Ops.SourceCoord(mode, align, oy, h, outH, out int y0, out int y1, out float ly);
                for (int ox = 0; ox < outW; ox++)
                {
                    Ops.SourceCoord(mode, align, ox, w, outW, out int x0, out int x1, out float lx);
                    float gg = gv[(pl * outH * outW) + (oy * outW) + ox];
                    if (mode == InterpolationMode.Nearest)
                    {
                        dp[(y0 * w) + x0] += gg;
                    }
                    else
                    {
                        float t = (1f - ly) * gg, b = ly * gg;
                        dp[(y0 * w) + x0] += (1f - lx) * t;
                        dp[(y0 * w) + x1] += lx * t;
                        dp[(y1 * w) + x0] += (1f - lx) * b;
                        dp[(y1 * w) + x1] += lx * b;
                    }
                }
            }
        }

        Engine.Emit(r, 0, dx, owned: true);
    }
}
