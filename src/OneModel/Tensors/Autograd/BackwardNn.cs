using System;
using RecursiveAI.OneModel.Kernels;
namespace RecursiveAI.OneModel.Tensors;

internal static partial class BackwardOps
{
    private static void RunNn(Engine e, BackwardRecord r, in Tensor g)
    {
        bool exact = TensorCompute.Exact;
        switch (r.Op)
        {
            case OpKind.Gelu:
            {
                Tensor x = r.Saved(0).Unpack();
                bool tanhForm = r.Ints[0] == 1;
                Tensor dx = Tensor.Create(DType.F32, x.Shape);
                ReadOnlySpan<float> xv = x.Data<float>(), gv = g.Data<float>();
                Span<float> d = dx.MutableData<float>();
                for (int i = 0; i < d.Length; i++)
                {
                    float v = xv[i];
                    float slope;
                    if (tanhForm)
                    {
                        // gelu(x) = 0.5 x (1 + tanh(u)), u = k (x + 0.044715 x^3), k = sqrt(2/pi)
                        float u = 0.79788456f * (v + (0.044715f * v * v * v));
                        float t = Fn.Tanh(exact, u);
                        slope = (0.5f * (1f + t)) + (0.5f * v * (1f - (t * t)) * 0.79788456f * (1f + (3f * 0.044715f * v * v)));
                    }
                    else
                    {
                        // gelu(x) = x Phi(x); gelu'(x) = Phi(x) + x phi(x)
                        float cdf = 0.5f * (1f + Fn.Erf(v * 0.70710677f));
                        float pdf = Fn.Exp(exact, -0.5f * v * v) * 0.3989423f;
                        slope = cdf + (v * pdf);
                    }

                    d[i] = gv[i] * slope;
                }

                Engine.Emit(r, 0, dx, owned: true);
                break;
            }

            case OpKind.Sigmoid:
                Engine.Emit(r, 0, Map2(g, r.Saved(0).Unpack(), static (gv, y) => gv * y * (1f - y)), owned: true);
                break;
            case OpKind.Tanh:
                Engine.Emit(r, 0, Map2(g, r.Saved(0).Unpack(), static (gv, y) => gv * (1f - (y * y))), owned: true);
                break;
            case OpKind.Sum:
            case OpKind.Mean:
                ReduceBackward(e, r, g);
                break;
            case OpKind.Softmax:
            {
                Tensor y = r.Saved(0).Unpack();
                Tensor dx = Tensor.Create(DType.F32, y.Shape);
                int cols = y.Shape[y.Rank - 1];
                int rows = cols == 0 ? 0 : y.Length / cols;
                ReadOnlySpan<float> yv = y.Data<float>(), gv = g.Data<float>();
                Span<float> d = dx.MutableData<float>();
                for (int row = 0; row < rows; row++)
                {
                    int o = row * cols;
                    double dot = 0;
                    for (int j = 0; j < cols; j++)
                    {
                        dot += (double)(gv[o + j] * yv[o + j]);
                    }

                    float dotf = (float)dot;
                    for (int j = 0; j < cols; j++)
                    {
                        d[o + j] = yv[o + j] * (gv[o + j] - dotf);
                    }
                }

                Engine.Emit(r, 0, dx, owned: true);
                break;
            }

            case OpKind.LogSoftmax:
            {
                Tensor y = r.Saved(0).Unpack();
                Tensor dx = Tensor.Create(DType.F32, y.Shape);
                int cols = y.Shape[y.Rank - 1];
                int rows = cols == 0 ? 0 : y.Length / cols;
                ReadOnlySpan<float> yv = y.Data<float>(), gv = g.Data<float>();
                Span<float> d = dx.MutableData<float>();
                for (int row = 0; row < rows; row++)
                {
                    int o = row * cols;
                    double sum = 0;
                    for (int j = 0; j < cols; j++)
                    {
                        sum += gv[o + j];
                    }

                    float s = (float)sum;
                    for (int j = 0; j < cols; j++)
                    {
                        d[o + j] = gv[o + j] - (Fn.Exp(exact, yv[o + j]) * s);
                    }
                }

                Engine.Emit(r, 0, dx, owned: true);
                break;
            }

            case OpKind.LayerNorm:
                LayerNormBackward(e, r, g);
                break;
            case OpKind.BatchNorm:
                BatchNormBackward(e, r, g);
                break;
            case OpKind.Conv2d:
                ConvBackward(e, r, g);
                break;
            case OpKind.AvgPool2d:
                AvgPoolBackward(e, r, g);
                break;
            case OpKind.Concat:
            {
                int axis = (int)r.Ints[0];
                int start = 0;
                for (int i = 0; i < r.InputCount; i++)
                {
                    int len = r.InputShape(i)[axis];
                    if (r.NeedsInputGrad(i))
                    {
                        Engine.Emit(r, i, g.Slice(axis, start, start + len).Clone(), owned: true);
                    }

                    start += len;
                }

                break;
            }

            case OpKind.Embedding:
                EmbeddingBackward(e, r, g);
                break;
            case OpKind.NllLoss:
                NllBackward(e, r, g);
                break;
            case OpKind.Attention:
                AttentionBackward(e, r, g);
                break;
            default:
                throw new NotSupportedException($"No backward function for {r.Op}.");
        }
    }

    private static void ReduceBackward(Engine e, BackwardRecord r, in Tensor g)
    {
        Shape inShape = r.InputShape(0);
        int axis = (int)r.Ints[0];
        Span<int> keep = stackalloc int[Shape.MaxRank];
        for (int i = 0; i < inShape.Rank; i++)
        {
            keep[i] = (axis < 0 || i == axis) ? 1 : inShape[i];
        }

        Tensor gk = g.Reshape(Shape.Create(keep.Slice(0, inShape.Rank)));
        Tensor dx = gk.Expand(inShape).Clone();
        if (r.Op == OpKind.Mean)
        {
            float n = axis < 0 ? inShape.Length : inShape[axis];
            Span<float> d = dx.MutableData<float>();
            for (int i = 0; i < d.Length; i++)
            {
                d[i] /= n;
            }
        }

        Engine.Emit(r, 0, dx, owned: true);
    }

    private static void LayerNormBackward(Engine e, BackwardRecord r, in Tensor g)
    {
        Tensor x = r.Saved(0).Unpack(), gamma = r.Saved(1).Unpack(), mean = r.Saved(2).Unpack(), inv = r.Saved(3).Unpack();
        int cols = x.Shape[x.Rank - 1];
        int rows = cols == 0 ? 0 : x.Length / cols;
        bool needX = r.NeedsInputGrad(0), needGamma = r.NeedsInputGrad(1), needBeta = r.NeedsInputGrad(2);
        Tensor dx = needX ? Tensor.Create(DType.F32, x.Shape) : default;
        Tensor accG = needGamma ? Tensor.Create(DType.F64, Shape.Create(cols), zero: true) : default;
        Tensor accB = needBeta ? Tensor.Create(DType.F64, Shape.Create(cols), zero: true) : default;
        ReadOnlySpan<float> xv = x.Data<float>(), gv = g.Data<float>(), gam = gamma.Data<float>(), mu = mean.Data<float>(), iv = inv.Data<float>();
        Span<float> d = needX ? dx.MutableData<float>() : default;
        Span<double> sg = needGamma ? accG.MutableData<double>() : default;
        Span<double> sb = needBeta ? accB.MutableData<double>() : default;
        for (int row = 0; row < rows; row++)
        {
            int o = row * cols;
            float m = mu[row], s = iv[row];
            double sum1 = 0, sum2 = 0;
            for (int j = 0; j < cols; j++)
            {
                float xhat = (xv[o + j] - m) * s;
                float dxhat = gv[o + j] * gam[j];
                sum1 += dxhat;
                sum2 += (double)(dxhat * xhat);
                if (needGamma)
                {
                    sg[j] += (double)(gv[o + j] * xhat);
                }

                if (needBeta)
                {
                    sb[j] += gv[o + j];
                }
            }

            if (needX)
            {
                float m1 = (float)(sum1 / cols), m2 = (float)(sum2 / cols);
                for (int j = 0; j < cols; j++)
                {
                    float xhat = (xv[o + j] - m) * s;
                    d[o + j] = s * ((gv[o + j] * gam[j]) - m1 - (xhat * m2));
                }
            }
        }

        if (needX)
        {
            Engine.Emit(r, 0, dx, owned: true);
        }

        EmitDouble(e, r, 1, accG, needGamma);
        EmitDouble(e, r, 2, accB, needBeta);
    }

    private static void EmitDouble(Engine e, BackwardRecord r, int input, in Tensor acc, bool need)
    {
        if (!need)
        {
            return;
        }

        Tensor f = Tensor.Create(DType.F32, acc.Shape);
        ReadOnlySpan<double> s = acc.Data<double>();
        Span<float> d = f.MutableData<float>();
        for (int i = 0; i < d.Length; i++)
        {
            d[i] = (float)s[i];
        }

        acc.Dispose();
        Engine.Emit(r, input, f, owned: true);
    }

    private static void BatchNormBackward(Engine e, BackwardRecord r, in Tensor g)
    {
        Tensor x = r.Saved(0).Unpack(), gamma = r.Saved(1).Unpack(), mean = r.Saved(2).Unpack(), inv = r.Saved(3).Unpack();
        bool training = r.Ints[0] != 0;
        int n = x.Shape[0], c = x.Shape[1], plane = x.Length / Math.Max(1, n * c);
        double m = (double)n * plane;
        Tensor dg = Tensor.Create(DType.F64, Shape.Create(c), zero: true), db = Tensor.Create(DType.F64, Shape.Create(c), zero: true);
        ReadOnlySpan<float> xv = x.Data<float>(), gv = g.Data<float>(), gam = gamma.Data<float>(), mu = mean.Data<float>(), iv = inv.Data<float>();
        Span<double> sg = dg.MutableData<double>(), sb = db.MutableData<double>();
        for (int i = 0; i < n; i++)
        {
            for (int ch = 0; ch < c; ch++)
            {
                int o = ((i * c) + ch) * plane;
                float mm = mu[ch], s = iv[ch];
                double a = 0, b = 0;
                for (int p = 0; p < plane; p++)
                {
                    float gg = gv[o + p];
                    a += (double)(gg * ((xv[o + p] - mm) * s));
                    b += gg;
                }

                sg[ch] += a;
                sb[ch] += b;
            }
        }

        if (r.NeedsInputGrad(0))
        {
            Tensor dx = Tensor.Create(DType.F32, x.Shape);
            Span<float> d = dx.MutableData<float>();
            for (int i = 0; i < n; i++)
            {
                for (int ch = 0; ch < c; ch++)
                {
                    int o = ((i * c) + ch) * plane;
                    float mm = mu[ch], s = iv[ch], k = gam[ch] * s;
                    float cb = (float)(sb[ch] / m), cg = (float)(sg[ch] / m);
                    for (int p = 0; p < plane; p++)
                    {
                        // Ioffe and Szegedy 2015, eq. for dx: gamma/sigma * (g - mean(g) - xhat * mean(g * xhat)); in eval mode the statistics are constants.
                        d[o + p] = training
                            ? k * (gv[o + p] - cb - (((xv[o + p] - mm) * s) * cg))
                            : k * gv[o + p];
                    }
                }
            }

            Engine.Emit(r, 0, dx, owned: true);
        }

        EmitDouble(e, r, 1, dg, r.NeedsInputGrad(1));
        EmitDouble(e, r, 2, db, r.NeedsInputGrad(2));
        if (!r.NeedsInputGrad(1))
        {
            dg.Dispose();
        }

        if (!r.NeedsInputGrad(2))
        {
            db.Dispose();
        }
    }

    internal static ConvShape ConvShapeOf(in Tensor x, in Tensor w, ReadOnlySpan<long> p) => new(
        x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3], w.Shape[0], w.Shape[2], w.Shape[3],
        (int)p[0], (int)p[1], (int)p[2], (int)p[3], (int)p[4], (int)p[5], (int)p[6], (int)p[7], (int)p[8]);

    private static void ConvBackward(Engine e, BackwardRecord r, in Tensor g)
    {
        Tensor x = r.Saved(0).Unpack(), w = r.Saved(1).Unpack();
        ConvShape s = ConvShapeOf(x, w, r.Ints);
        int cg = s.Cin / s.Groups, og = s.Cout / s.Groups, kk = cg * s.KH * s.KW, plane = s.OutH * s.OutW;
        ReadOnlySpan<float> xv = x.Data<float>(), wv = w.Data<float>(), gv = g.Data<float>();

        if (r.NeedsInputGrad(2))
        {
            Tensor acc = Tensor.Create(DType.F64, Shape.Create(s.Cout), zero: true);
            Span<double> a = acc.MutableData<double>();
            for (int n = 0; n < s.N; n++)
            {
                for (int co = 0; co < s.Cout; co++)
                {
                    ReadOnlySpan<float> row = gv.Slice(((n * s.Cout) + co) * plane, plane);
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

        if (r.NeedsInputGrad(1))
        {
            // dW[og, K] = sum over n of g[og, P] . cols[P, K], where cols is the patch-major unfolding of x (Chellapilla et al. 2006).
            Tensor dw = Tensor.Zeros(DType.F32, w.Shape);
            Tensor cols = Tensor.Create(DType.F32, Shape.Create(plane, kk));
            Tensor tmp = Tensor.Create(DType.F32, Shape.Create(og, kk));
            Span<float> dwv = dw.MutableData<float>();
            for (int n = 0; n < s.N; n++)
            {
                for (int grp = 0; grp < s.Groups; grp++)
                {
                    Im2ColPatchMajor(s, xv.Slice(((n * s.Cin) + (grp * cg)) * s.H * s.W, cg * s.H * s.W), cols.MutableData<float>());
                    Dense.Gemm(false, false, og, kk, plane, gv.Slice(((n * s.Cout) + (grp * og)) * plane), plane, cols.Data<float>(), kk,
                        tmp.MutableData<float>(), kk);
                    Span<float> dst = dwv.Slice(grp * og * kk, og * kk);
                    ReadOnlySpan<float> t = tmp.Data<float>();
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

        if (r.NeedsInputGrad(0))
        {
            // dcols[K, P] = W_g^T [K, og] . g[og, P], then scattered back onto x (col2im).
            Tensor dx = Tensor.Zeros(DType.F32, x.Shape);
            Tensor wt = Tensor.Create(DType.F32, Shape.Create(s.Groups * kk, og));
            Tensor dcols = Tensor.Create(DType.F32, Shape.Create(kk, plane));
            Span<float> wtv = wt.MutableData<float>();
            for (int grp = 0; grp < s.Groups; grp++)
            {
                Dense.Transpose(wv.Slice(grp * og * kk, og * kk), kk, og, kk, wtv.Slice(grp * kk * og, kk * og));
            }

            Span<float> dxv = dx.MutableData<float>();
            for (int n = 0; n < s.N; n++)
            {
                for (int grp = 0; grp < s.Groups; grp++)
                {
                    Dense.Gemm(false, false, kk, plane, og, wt.Data<float>().Slice(grp * kk * og, kk * og), og,
                        gv.Slice(((n * s.Cout) + (grp * og)) * plane), plane, dcols.MutableData<float>(), plane);
                    Col2Im(s, dcols.Data<float>(), dxv.Slice(((n * s.Cin) + (grp * cg)) * s.H * s.W, cg * s.H * s.W));
                }
            }

            wt.Dispose();
            dcols.Dispose();
            Engine.Emit(r, 0, dx, owned: true);
        }
    }

    // cols[p, (c * KH + ky) * KW + kx] = x[c, oy * stride - pad + ky * dil, ox * stride - pad + kx * dil], zero outside the image.
    internal static void Im2ColPatchMajor(in ConvShape s, ReadOnlySpan<float> x, Span<float> cols)
    {
        int cg = s.Cin / s.Groups, kk = cg * s.KH * s.KW, ow = s.OutW;
        for (int oy = 0; oy < s.OutH; oy++)
        {
            for (int ox = 0; ox < ow; ox++)
            {
                Span<float> row = cols.Slice(((oy * ow) + ox) * kk, kk);
                int k = 0;
                for (int c = 0; c < cg; c++)
                {
                    for (int ky = 0; ky < s.KH; ky++)
                    {
                        int iy = (oy * s.StrideH) - s.PadTop + (ky * s.DilH);
                        for (int kx = 0; kx < s.KW; kx++, k++)
                        {
                            int ix = (ox * s.StrideW) - s.PadLeft + (kx * s.DilW);
                            row[k] = (uint)iy < (uint)s.H && (uint)ix < (uint)s.W ? x[(((c * s.H) + iy) * s.W) + ix] : 0f;
                        }
                    }
                }
            }
        }
    }

    // The adjoint of the unfolding: dx[c, iy, ix] += dcols[(c * KH + ky) * KW + kx, oy * ow + ox]. Taps are added in increasing (row of dcols, p) order.
    internal static void Col2Im(in ConvShape s, ReadOnlySpan<float> dcols, Span<float> dx)
    {
        int cg = s.Cin / s.Groups, ow = s.OutW, plane = s.OutH * ow;
        int k = 0;
        for (int c = 0; c < cg; c++)
        {
            for (int ky = 0; ky < s.KH; ky++)
            {
                for (int kx = 0; kx < s.KW; kx++, k++)
                {
                    ReadOnlySpan<float> row = dcols.Slice(k * plane, plane);
                    for (int oy = 0; oy < s.OutH; oy++)
                    {
                        int iy = (oy * s.StrideH) - s.PadTop + (ky * s.DilH);
                        if ((uint)iy >= (uint)s.H)
                        {
                            continue;
                        }

                        for (int ox = 0; ox < ow; ox++)
                        {
                            int ix = (ox * s.StrideW) - s.PadLeft + (kx * s.DilW);
                            if ((uint)ix < (uint)s.W)
                            {
                                dx[(((c * s.H) + iy) * s.W) + ix] += row[(oy * ow) + ox];
                            }
                        }
                    }
                }
            }
        }
    }

    private static void AvgPoolBackward(Engine e, BackwardRecord r, in Tensor g)
    {
        ReadOnlySpan<long> p = r.Ints;
        Shape ish = r.InputShape(0);
        Tensor dx = Tensor.Zeros(DType.F32, ish);
        Span<float> d = dx.MutableData<float>();
        ReadOnlySpan<float> gv = g.Data<float>();
        int n = ish[0] * ish[1], h = ish[2], w = ish[3], oh = g.Shape[2], ow = g.Shape[3];
        for (int pl = 0; pl < n; pl++)
        {
            for (int oy = 0; oy < oh; oy++)
            {
                for (int ox = 0; ox < ow; ox++)
                {
                    PoolWindow(p, h, w, oy, ox, out int y0, out int y1, out int x0, out int x1, out int count);
                    float share = gv[(((pl * oh) + oy) * ow) + ox] / count;
                    for (int y = y0; y < y1; y++)
                    {
                        for (int xx = x0; xx < x1; xx++)
                        {
                            d[(((pl * h) + y) * w) + xx] += share;
                        }
                    }
                }
            }
        }

        Engine.Emit(r, 0, dx, owned: true);
    }

    // The in-bounds window of output (oy, ox) and the divisor: its size, or with countIncludePad the size of the window clipped to the padded image.
    internal static void PoolWindow(ReadOnlySpan<long> p, int h, int w, int oy, int ox, out int y0, out int y1, out int x0, out int x1, out int count)
    {
        int kh = (int)p[0], kw = (int)p[1], sh = (int)p[2], sw = (int)p[3], ph = (int)p[4], pw = (int)p[5];
        int ys = (oy * sh) - ph, xs = (ox * sw) - pw;
        int ye = Math.Min(ys + kh, h + ph), xe = Math.Min(xs + kw, w + pw);
        int padded = (ye - ys) * (xe - xs);
        y0 = Math.Max(ys, 0);
        x0 = Math.Max(xs, 0);
        y1 = Math.Min(ye, h);
        x1 = Math.Min(xe, w);
        count = p[6] != 0 ? padded : Math.Max(1, (y1 - y0) * (x1 - x0));
    }

    private static void EmbeddingBackward(Engine e, BackwardRecord r, in Tensor g)
    {
        Tensor ids = r.Saved(0).Unpack();
        Shape ts = r.InputShape(0);
        int dim = ts[1];
        long pad = r.Ints[0];
        Tensor dt = Tensor.Zeros(DType.F32, ts);
        Span<float> d = dt.MutableData<float>();
        ReadOnlySpan<float> gv = g.Data<float>();
        for (int i = 0; i < ids.Length; i++)
        {
            long id = ids.DType == DType.I32 ? ids.Data<int>()[i] : ids.Data<long>()[i];
            if (id == pad)
            {
                continue;
            }

            Span<float> row = d.Slice((int)id * dim, dim);
            ReadOnlySpan<float> src = gv.Slice(i * dim, dim);
            for (int j = 0; j < dim; j++)
            {
                row[j] += src[j];
            }
        }

        Engine.Emit(r, 0, dt, owned: true);
    }

    private static void NllBackward(Engine e, BackwardRecord r, in Tensor g)
    {
        Tensor target = r.Saved(0).Unpack();
        Shape ls = r.InputShape(0);
        int rows = ls[0], cols = ls[1];
        bool mean = r.Ints[0] != 0;
        long ignore = r.Ints[1];
        int count = 0;
        for (int i = 0; i < rows; i++)
        {
            if (TargetAt(target, i) != ignore)
            {
                count++;
            }
        }

        float gv = g.Data<float>()[0];
        float scale = mean ? gv / Math.Max(1, count) : gv;
        Tensor dx = Tensor.Zeros(DType.F32, ls);
        Span<float> d = dx.MutableData<float>();
        for (int i = 0; i < rows; i++)
        {
            long t = TargetAt(target, i);
            if (t != ignore)
            {
                d[(i * cols) + (int)t] = -scale;
            }
        }

        Engine.Emit(r, 0, dx, owned: true);
    }

    internal static long TargetAt(in Tensor target, int i) => target.DType == DType.I32 ? target.Data<int>()[i] : target.Data<long>()[i];

    private static void AttentionBackward(Engine e, BackwardRecord r, in Tensor g)
    {
        Tensor q = r.Saved(0).Unpack(), k = r.Saved(1).Unpack(), v = r.Saved(2).Unpack();
        Tensor mask = r.Saved(3).IsEmpty ? default : r.Saved(3).Unpack();
        float scale = (float)r.Floats[0];
        int batch = q.Shape[0], heads = q.Shape[1], lq = q.Shape[2], d = q.Shape[3], lk = k.Shape[2];
        bool exact = TensorCompute.Exact;
        Tensor dq = Tensor.Create(DType.F32, q.Shape), dk = Tensor.Create(DType.F32, k.Shape), dv = Tensor.Create(DType.F32, v.Shape);
        Tensor sBuf = Tensor.Create(DType.F32, Shape.Create(lq, lk)), pBuf = Tensor.Create(DType.F32, Shape.Create(lq, lk));
        Tensor dpBuf = Tensor.Create(DType.F32, Shape.Create(lq, lk));
        ReadOnlySpan<float> qv = q.Data<float>(), kv = k.Data<float>(), vv = v.Data<float>(), gv = g.Data<float>();
        ReadOnlySpan<float> mv = mask.IsEmpty ? default : mask.Data<float>();
        Span<float> dqv = dq.MutableData<float>(), dkv = dk.MutableData<float>(), dvv = dv.MutableData<float>();
        Span<float> sv = sBuf.MutableData<float>(), pv = pBuf.MutableData<float>(), dpv = dpBuf.MutableData<float>();
        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < heads; h++)
            {
                int qo = ((b * heads) + h) * lq * d, ko = ((b * heads) + h) * lk * d;
                ReadOnlySpan<float> qh = qv.Slice(qo, lq * d), kh = kv.Slice(ko, lk * d), vh = vv.Slice(ko, lk * d), gh = gv.Slice(qo, lq * d);

                // Recompute the attention weights P = softmax(scale Q K^T + mask), rather than keeping an [Lq, Lk] matrix per head
                // alive from the forward pass (Dao et al. 2022, section 3.1: the backward recomputes the scores from Q, K).
                Dense.Gemm(false, true, lq, lk, d, qh, d, kh, d, sv, lk);
                for (int i = 0; i < lq; i++)
                {
                    ReadOnlySpan<float> mrow = mv.IsEmpty ? default : mv.Slice(MaskRowOffset(mask, b, i, lq, lk), lk);
                    for (int j = 0; j < lk; j++)
                    {
                        sv[(i * lk) + j] = (sv[(i * lk) + j] * scale) + (mv.IsEmpty ? 0f : mrow[j]);
                    }
                }

                Softmax.Rows(exact, TensorCompute.Isa, sv, pv, lq, lk, lk, lk);

                // dV = P^T dO; dP = dO V^T; dS = P (dP - rowsum(dP P)); dQ = scale dS K; dK = scale dS^T Q.
                Dense.Gemm(true, false, lk, d, lq, pv, lk, gh, d, dvv.Slice(ko, lk * d), d);
                Dense.Gemm(false, true, lq, lk, d, gh, d, vh, d, dpv, lk);
                for (int i = 0; i < lq; i++)
                {
                    int o = i * lk;
                    double dot = 0;
                    for (int j = 0; j < lk; j++)
                    {
                        dot += (double)(dpv[o + j] * pv[o + j]);
                    }

                    float df = (float)dot;
                    for (int j = 0; j < lk; j++)
                    {
                        dpv[o + j] = pv[o + j] * (dpv[o + j] - df) * scale;
                    }
                }

                Dense.Gemm(false, false, lq, d, lk, dpv, lk, kh, d, dqv.Slice(qo, lq * d), d);
                Dense.Gemm(true, false, lk, d, lq, dpv, lk, qh, d, dkv.Slice(ko, lk * d), d);
            }
        }

        sBuf.Dispose();
        pBuf.Dispose();
        dpBuf.Dispose();
        EmitOrFree(e, r, 0, dq);
        EmitOrFree(e, r, 1, dk);
        EmitOrFree(e, r, 2, dv);
    }

    private static void EmitOrFree(Engine e, BackwardRecord r, int i, in Tensor t)
    {
        if (r.NeedsInputGrad(i))
        {
            Engine.Emit(r, i, t, owned: true);
        }
        else
        {
            t.Dispose();
        }
    }

    // Offset of query row i of batch b in a mask of shape [Lq, Lk], [B, Lq, Lk] or [B, 1, Lq, Lk].
    internal static int MaskRowOffset(in Tensor mask, int b, int i, int lq, int lk) =>
        (mask.Rank == 2 ? 0 : b * lq * lk) + (i * lk);
}
