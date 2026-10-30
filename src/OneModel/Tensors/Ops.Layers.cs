using System;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Kernels.Exact;
namespace RecursiveAI.OneModel.Tensors;

public static partial class Ops
{
    // ---------------------------------------------------------------------------------------------------------------- softmax

    /// <summary>
    /// <param name="x">Float32 input of any shape and strides.</param>
    /// <param name="axis">Axis; negative counts from the end (default -1, the last).</param>
    /// Softmax along <paramref name="axis"/>. Runs the softmax row kernel (fast or exact per <see cref="TensorCompute.Mode"/>).
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    /// Record: slot 0 = the output. A non-last axis is handled by swapping it with the last through views, so the result may be a strided view.
    /// </summary>
    public static Tensor Softmax(in Tensor x, int axis = -1, BufferPool? pool = null)
    {
        int ax = Shape.NormalizeAxis(axis, x.Rank);
        if (ax != x.Rank - 1)
        {
            return Softmax(x.Transpose(ax, x.Rank - 1), -1, pool).Transpose(ax, x.Rank - 1);
        }

        Tensor xc = Contig(x, out bool copied);
        Tensor r = NewOut(x.Shape, pool);
        int cols = x.Shape[ax];
        int rows = cols == 0 ? 0 : x.Length / cols;
        Kernels.Softmax.Rows(TensorCompute.Exact, TensorCompute.Isa, xc.InternalData<float>(), r.MutableData<float>(), rows, cols, cols, cols);
        Free(xc, copied);
        if (!Recording(x))
        {
            return r;
        }

        BackwardRecord rec = BackwardRecord.Create(OpKind.Softmax, 1, x);
        rec.Save(0, r);
        return rec.Attach(r);
    }

    /// <summary>
    /// <param name="x">Float32 input of any shape and strides.</param>
    /// <param name="axis">Axis; negative counts from the end (default -1, the last).</param>
    /// <c>x − log Σ eˣ</c> along <paramref name="axis"/>, computed with the row maximum subtracted. Record: slot 0 = the output.
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    /// </summary>
    public static Tensor LogSoftmax(in Tensor x, int axis = -1, BufferPool? pool = null)
    {
        int ax = Shape.NormalizeAxis(axis, x.Rank);
        if (ax != x.Rank - 1)
        {
            return LogSoftmax(x.Transpose(ax, x.Rank - 1), -1, pool).Transpose(ax, x.Rank - 1);
        }

        bool exact = TensorCompute.Exact;
        Tensor xc = Contig(x, out bool copied);
        Tensor r = NewOut(x.Shape, pool);
        int cols = x.Shape[ax];
        int rows = cols == 0 ? 0 : x.Length / cols;
        ReadOnlySpan<float> s = xc.InternalData<float>();
        Span<float> d = r.MutableData<float>();
        for (int row = 0; row < rows; row++)
        {
            int o = row * cols;
            float max = float.NegativeInfinity;
            for (int j = 0; j < cols; j++)
            {
                if (s[o + j] > max)
                {
                    max = s[o + j];
                }
            }

            double sum = 0;
            for (int j = 0; j < cols; j++)
            {
                sum += Fn.Exp(exact, s[o + j] - max);
            }

            float lse = max + Fn.Log(exact, (float)sum);
            for (int j = 0; j < cols; j++)
            {
                d[o + j] = s[o + j] - lse;
            }
        }

        Free(xc, copied);
        if (!Recording(x))
        {
            return r;
        }

        BackwardRecord rec = BackwardRecord.Create(OpKind.LogSoftmax, 1, x);
        rec.Save(0, r);
        return rec.Attach(r);
    }

    // ---------------------------------------------------------------------------------------------------------------- normalization

    /// <summary>
    /// Layer normalization over the last axis (Ba et al. 2016): <c>(x − mean)/√(var + ε)·gamma + beta</c>, on the same row kernel the
    /// graph runtime uses. Record: slots x, gamma, mean[rows], 1/std[rows] (the last two owned); Floats = [ε].
    /// </summary>
    /// <param name="x">Input, last axis of extent D.</param>
    /// <param name="gamma">Scale, shape [D].</param>
    /// <param name="beta">Shift, shape [D].</param>
    /// <param name="epsilon">Added to the variance, default 1e-5.</param>
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    public static Tensor LayerNorm(in Tensor x, in Tensor gamma, in Tensor beta, float epsilon = 1e-5f, BufferPool? pool = null)
    {
        int cols = x.Shape[x.Rank - 1];
        if (gamma.Rank != 1 || beta.Rank != 1 || gamma.Shape[0] != cols || beta.Shape[0] != cols)
        {
            throw new TensorShapeException($"LayerNorm over {cols} features needs gamma and beta of shape [{cols}], not {gamma.Shape} and {beta.Shape}.");
        }

        int rows = cols == 0 ? 0 : x.Length / cols;
        Tensor xc = Contig(x, out bool xcopied), gc = Contig(gamma, out bool gcopied), bc = Contig(beta, out bool bcopied);
        Tensor r = NewOut(x.Shape, pool);
        bool rec = Recording(x, gamma, beta);
        Tensor mean = rec ? Tensor.Create(DType.F32, Shape.Create(rows), pool) : default;
        Tensor inv = rec ? Tensor.Create(DType.F32, Shape.Create(rows), pool) : default;
        NormKernels.Rows(TensorCompute.Isa, xc.InternalData<float>(), r.MutableData<float>(), 0, rows, cols, false, epsilon,
            gc.InternalData<float>(), 1f, bc.InternalData<float>(), true, 0f,
            rec ? mean.MutableData<float>() : default, rec ? inv.MutableData<float>() : default);
        Free(bc, bcopied);
        if (!rec)
        {
            Free(xc, xcopied);
            Free(gc, gcopied);
            return r;
        }

        BackwardRecord br = BackwardRecord.Create(OpKind.LayerNorm, 4, x, gamma, beta);
        SaveContig(br, 0, xc, xcopied);
        SaveContig(br, 1, gc, gcopied);
        br.Save(2, mean, owned: true);
        br.Save(3, inv, owned: true);
        br.SetFloats(epsilon);
        return br.Attach(r);
    }

    /// <summary>
    /// Batch normalization of an NCHW tensor (Ioffe and Szegedy 2015). In training the batch statistics (biased variance over N, H, W,
    /// accumulated in double) normalize the input and the running statistics are updated in place:
    /// <c>running = (1 − momentum)·running + momentum·stat</c> with the unbiased variance. In evaluation the running statistics are used
    /// and left alone. Record: slots x, gamma, mean[C], 1/std[C] (the last two owned); Ints = [training].
    /// </summary>
    /// <param name="x">Input [N, C, H, W].</param>
    /// <param name="gamma">Scale [C].</param>
    /// <param name="beta">Shift [C].</param>
    /// <param name="runningMean">Running mean [C], updated in training; plain (non-tracked) tensor.</param>
    /// <param name="runningVar">Running variance [C], updated in training; plain (non-tracked) tensor.</param>
    /// <param name="training">Use and update batch statistics.</param>
    /// <param name="momentum">Weight of the new statistic, default 0.1.</param>
    /// <param name="epsilon">Added to the variance, default 1e-5.</param>
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    public static Tensor BatchNorm2d(in Tensor x, in Tensor gamma, in Tensor beta, in Tensor runningMean, in Tensor runningVar,
        bool training, float momentum = 0.1f, float epsilon = 1e-5f, BufferPool? pool = null)
    {
        RequireFloat32(x);
        if (x.Rank != 4)
        {
            throw new TensorShapeException($"BatchNorm2d needs an (N,C,H,W) input, not {x.Shape}.");
        }

        int n = x.Shape[0], c = x.Shape[1], plane = x.Shape[2] * x.Shape[3];
        foreach (Tensor t in (ReadOnlySpan<Tensor>)[gamma, beta, runningMean, runningVar])
        {
            if (t.Rank != 1 || t.Shape[0] != c)
            {
                throw new TensorShapeException($"BatchNorm2d over {c} channels needs per-channel tensors of shape [{c}], not {t.Shape}.");
            }
        }

        if (training && (long)n * plane < 1)
        {
            throw new TensorShapeException("BatchNorm2d in training needs at least one value per channel.");
        }

        Tensor xc = Contig(x, out bool xcopied), gc = Contig(gamma, out bool gcopied), bc = Contig(beta, out bool bcopied);
        Tensor mean = Tensor.Create(DType.F32, Shape.Create(c), pool), inv = Tensor.Create(DType.F32, Shape.Create(c), pool);
        ReadOnlySpan<float> xv = xc.InternalData<float>(), gam = gc.InternalData<float>(), bet = bc.InternalData<float>();
        Span<float> mu = mean.MutableData<float>(), iv = inv.MutableData<float>();
        if (training)
        {
            double m = (double)n * plane;
            Span<float> rm = runningMean.MutableData<float>(), rv = runningVar.MutableData<float>();
            for (int ch = 0; ch < c; ch++)
            {
                double sum = 0;
                for (int i = 0; i < n; i++)
                {
                    ReadOnlySpan<float> pl = xv.Slice(((i * c) + ch) * plane, plane);
                    for (int p = 0; p < pl.Length; p++)
                    {
                        sum += pl[p];
                    }
                }

                double mu64 = sum / m, ss = 0;
                for (int i = 0; i < n; i++)
                {
                    ReadOnlySpan<float> pl = xv.Slice(((i * c) + ch) * plane, plane);
                    for (int p = 0; p < pl.Length; p++)
                    {
                        double dd = pl[p] - mu64;
                        ss += dd * dd;
                    }
                }

                double var = ss / m;
                mu[ch] = (float)mu64;
                iv[ch] = 1f / float.Sqrt((float)var + epsilon);
                double unbiased = m > 1 ? var * m / (m - 1) : var;
                rm[ch] = (float)(((1 - momentum) * rm[ch]) + (momentum * mu64));
                rv[ch] = (float)(((1 - momentum) * rv[ch]) + (momentum * unbiased));
            }
        }
        else
        {
            ReadOnlySpan<float> rm = runningMean.InternalData<float>(), rv = runningVar.InternalData<float>();
            for (int ch = 0; ch < c; ch++)
            {
                mu[ch] = rm[ch];
                iv[ch] = 1f / float.Sqrt(rv[ch] + epsilon);
            }
        }

        Tensor r = NewOut(x.Shape, pool);
        Span<float> d = r.MutableData<float>();
        for (int i = 0; i < n; i++)
        {
            for (int ch = 0; ch < c; ch++)
            {
                int o = ((i * c) + ch) * plane;
                float mm = mu[ch], s = iv[ch], gg = gam[ch], bb = bet[ch];
                for (int p = 0; p < plane; p++)
                {
                    d[o + p] = (((xv[o + p] - mm) * s) * gg) + bb;
                }
            }
        }

        Free(bc, bcopied);
        if (!Recording(x, gamma, beta))
        {
            Free(xc, xcopied);
            Free(gc, gcopied);
            mean.Dispose();
            inv.Dispose();
            return r;
        }

        BackwardRecord br = BackwardRecord.Create(OpKind.BatchNorm, 4, x, gamma, beta);
        br.AddTraceExtras(runningMean, runningVar);
        SaveContig(br, 0, xc, xcopied);
        SaveContig(br, 1, gc, gcopied);
        br.Save(2, mean, owned: true);
        br.Save(3, inv, owned: true);
        br.SetInts(training ? 1 : 0);
        br.SetFloats(epsilon);
        return br.Attach(r);
    }

    // ---------------------------------------------------------------------------------------------------------------- convolution and pooling

    /// <summary>
    /// 2-D cross-correlation of an NCHW input with OIHW weights on the blocked-GEMM / direct-depthwise kernels, with an optional bias.
    /// Record: slots x and weight (copies, owned, if they were not contiguous); Ints = [strideH, strideW, padTop, padLeft, padBottom,
    /// padRight, dilH, dilW, groups]. Backward: dW = g·im2col(x)ᵀ, dx = col2im(Wᵀ·g), dbias = Σg (Chellapilla et al. 2006).
    /// </summary>
    /// <param name="x">Input [N, Cin, H, W].</param>
    /// <param name="weight">Weights [Cout, Cin/Groups, KH, KW].</param>
    /// <param name="bias">Bias [Cout], or an empty tensor (<c>default</c>) for none.</param>
    /// <param name="p">Stride, padding, dilation and groups; defaults 1, 0, 1, 1.</param>
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    /// <exception cref="TensorShapeException">The shapes do not fit, or the output would be empty.</exception>
    public static Tensor Conv2d(in Tensor x, in Tensor weight, in Tensor bias = default, Conv2dParams p = default, BufferPool? pool = null)
    {
        RequireFloat32(x);
        RequireFloat32(weight);
        p = p with
        {
            StrideH = p.StrideH == 0 ? 1 : p.StrideH, StrideW = p.StrideW == 0 ? 1 : p.StrideW,
            DilH = p.DilH == 0 ? 1 : p.DilH, DilW = p.DilW == 0 ? 1 : p.DilW, Groups = p.Groups == 0 ? 1 : p.Groups,
        };
        if (x.Rank != 4 || weight.Rank != 4)
        {
            throw new TensorShapeException($"Conv2d needs an (N,C,H,W) input and (O,I,KH,KW) weights, not {x.Shape} and {weight.Shape}.");
        }

        if (p.Groups < 1 || x.Shape[1] % p.Groups != 0 || weight.Shape[0] % p.Groups != 0 || weight.Shape[1] != x.Shape[1] / p.Groups)
        {
            throw new TensorShapeException($"Weights {weight.Shape} do not fit input {x.Shape} with {p.Groups} groups.");
        }

        if (!bias.IsEmpty)
        {
            RequireFloat32(bias);
            if (bias.Rank != 1 || bias.Shape[0] != weight.Shape[0])
            {
                throw new TensorShapeException($"The bias must have shape [{weight.Shape[0]}], not {bias.Shape}.");
            }
        }

        var s = new ConvShape(x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3], weight.Shape[0], weight.Shape[2], weight.Shape[3],
            p.StrideH, p.StrideW, p.PadH, p.PadW, p.PadH, p.PadW, p.DilH, p.DilW, p.Groups);
        if (p.StrideH < 1 || p.StrideW < 1 || p.DilH < 1 || p.DilW < 1 || p.PadH < 0 || p.PadW < 0 || s.OutH < 1 || s.OutW < 1)
        {
            throw new TensorShapeException($"Invalid convolution geometry for input {x.Shape} and kernel {weight.Shape}.");
        }

        Tensor xc = Contig(x, out bool xcopied), wc = Contig(weight, out bool wcopied);
        Tensor bc = bias.IsEmpty ? default : Contig(bias, out _);
        Tensor r = NewOut(Shape.Create(s.N, s.Cout, s.OutH, s.OutW), pool);
        Conv.Run(TensorCompute.KMode, TensorCompute.Path, s, xc.InternalData<float>(), wc.InternalData<float>(), r.MutableData<float>(), TensorCompute.Threads,
            bc.IsEmpty ? default : bc.InternalData<float>());
        if (!bias.IsEmpty && !bias.IsContiguous)
        {
            bc.Dispose();
        }

        if (!Recording(x, weight, bias))
        {
            Free(xc, xcopied);
            Free(wc, wcopied);
            return r;
        }

        BackwardRecord br = BackwardRecord.Create(OpKind.Conv2d, 2, x, weight, bias);
        SaveContig(br, 0, xc, xcopied);
        SaveContig(br, 1, wc, wcopied);
        br.SetInts(p.StrideH, p.StrideW, p.PadH, p.PadW, p.PadH, p.PadW, p.DilH, p.DilW, p.Groups);
        return br.Attach(r);
    }

    /// <summary>
    /// Average pooling over an NCHW input. Each output is the in-order float sum of its window divided by the window size, which counts
    /// <param name="x">Float32 input of any shape and strides.</param>
    /// <param name="p">Window geometry.</param>
    /// <param name="countIncludePad">Count the zero padding in the divisor (default true).</param>
    /// the zero padding when <paramref name="countIncludePad"/> is true (the default) and only real pixels otherwise.
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    /// Record: Ints = [kernelH, kernelW, strideH, strideW, padH, padW, countIncludePad].
    /// </summary>
    public static Tensor AvgPool2d(in Tensor x, Pool2dParams p, bool countIncludePad = true, BufferPool? pool = null)
    {
        RequireFloat32(x);
        if (x.Rank != 4)
        {
            throw new TensorShapeException($"AvgPool2d needs an (N,C,H,W) input, not {x.Shape}.");
        }

        int h = x.Shape[2], w = x.Shape[3];
        int oh = Pool2dParams.OutputSize(h, p.KernelH, p.StrideH, p.PadH), ow = Pool2dParams.OutputSize(w, p.KernelW, p.StrideW, p.PadW);
        Tensor xc = Contig(x, out bool copied);
        Tensor r = NewOut(Shape.Create(x.Shape[0], x.Shape[1], oh, ow), pool);
        Span<long> ints = [p.KernelH, p.KernelW, p.StrideH, p.StrideW, p.PadH, p.PadW, countIncludePad ? 1 : 0];
        ReadOnlySpan<float> s = xc.InternalData<float>();
        Span<float> d = r.MutableData<float>();
        int planes = x.Shape[0] * x.Shape[1];
        for (int pl = 0; pl < planes; pl++)
        {
            for (int oy = 0; oy < oh; oy++)
            {
                for (int ox = 0; ox < ow; ox++)
                {
                    BackwardOps.PoolWindow(ints, h, w, oy, ox, out int y0, out int y1, out int x0, out int x1, out int count);
                    float sum = 0f;
                    for (int y = y0; y < y1; y++)
                    {
                        for (int xx = x0; xx < x1; xx++)
                        {
                            sum += s[(((pl * h) + y) * w) + xx];
                        }
                    }

                    d[(((pl * oh) + oy) * ow) + ox] = sum / count;
                }
            }
        }

        Free(xc, copied);
        if (!Recording(x))
        {
            return r;
        }

        BackwardRecord rec = BackwardRecord.Create(OpKind.AvgPool2d, 0, x);
        rec.SetInts(ints);
        return rec.Attach(r);
    }

    /// <summary>Mean over H and W of an NCHW tensor, giving [N, C, 1, 1]. Built from <see cref="Mean(in Tensor, int, bool, BufferPool?)"/> and views.</summary>
    public static Tensor GlobalAvgPool2d(in Tensor x, BufferPool? pool = null)
    {
        if (x.Rank != 4)
        {
            throw new TensorShapeException($"GlobalAvgPool2d needs an (N,C,H,W) input, not {x.Shape}.");
        }

        Tensor flat = x.Reshape(Shape.Create(x.Shape[0], x.Shape[1], x.Shape[2] * x.Shape[3]));
        return Mean(flat, 2, keepDim: false, pool).Reshape(Shape.Create(x.Shape[0], x.Shape[1], 1, 1));
    }

    // ---------------------------------------------------------------------------------------------------------------- data movement

    /// <param name="parts">The tensors to join.</param>
    /// <param name="axis">Axis; negative counts from the end (default -1, the last).</param>
    /// <summary>Joins tensors along <paramref name="axis"/>; all other extents must match. Record: Ints = [axis]; one input per part.</summary>
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    public static Tensor Concat(ReadOnlySpan<Tensor> parts, int axis, BufferPool? pool = null)
    {
        if (parts.Length == 0)
        {
            throw new ArgumentException("Concat needs at least one tensor.", nameof(parts));
        }

        Shape first = parts[0].Shape;
        int ax = Shape.NormalizeAxis(axis, first.Rank);
        int total = 0;
        foreach (Tensor t in parts)
        {
            RequireFloat32(t);
            if (t.Rank != first.Rank)
            {
                throw new TensorShapeException("Concat inputs must have the same rank.");
            }

            for (int i = 0; i < first.Rank; i++)
            {
                if (i != ax && t.Shape[i] != first[i])
                {
                    throw new TensorShapeException($"Concat inputs differ at axis {i}: {t.Shape} against {first}.");
                }
            }

            total = checked(total + t.Shape[ax]);
        }

        Span<int> dims = stackalloc int[Shape.MaxRank];
        first.Dimensions.CopyTo(dims);
        dims[ax] = total;
        Tensor r = NewOut(Shape.Create(dims.Slice(0, first.Rank)), pool);
        using (GradMode.NoGradInternal())
        {
            int start = 0;
            foreach (Tensor t in parts)
            {
                r.Slice(ax, start, start + t.Shape[ax]).CopyFromCore(t);
                start += t.Shape[ax];
            }
        }

        bool any = false;
        foreach (Tensor t in parts)
        {
            any |= t.RequiresGrad;
        }

        if (!any || !GradMode.IsEnabled)
        {
            return r;
        }

        BackwardRecord rec = BackwardRecord.Create(OpKind.Concat, 0, parts);
        rec.SetInts(ax);
        return rec.Attach(r);
    }

    /// <summary>
    /// <param name="table">Float32 table [V, D].</param>
    /// <param name="ids">Int32 or Int64 indices of any shape.</param>
    /// <param name="paddingIndex">Id whose rows get no gradient; default -1 (none).</param>
    /// Looks up rows of <paramref name="table"/> [V, D] by integer ids (Int32 or Int64, any shape), giving [ids..., D].
    /// Record: slot 0 = the ids (not copied; do not change them before backward); Ints = [paddingIndex or -1]. The gradient adds each
    /// output row to its table row in id order, skipping <paramref name="paddingIndex"/>.
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">An id is outside [0, V).</exception>
    public static Tensor Embedding(in Tensor table, in Tensor ids, int paddingIndex = -1, BufferPool? pool = null)
    {
        RequireFloat32(table);
        if (table.Rank != 2 || (ids.DType != DType.I32 && ids.DType != DType.I64))
        {
            throw new TensorShapeException("Embedding needs a [V, D] float32 table and Int32 or Int64 ids.");
        }

        Tensor tc = Contig(table, out bool tcopied);
        Tensor ic = ids.IsContiguous ? ids : ids.Clone();
        int v = table.Shape[0], dim = table.Shape[1];
        Span<int> dims = stackalloc int[Shape.MaxRank];
        if (ids.Rank + 1 > Shape.MaxRank)
        {
            throw new TensorShapeException("Too many dimensions.");
        }

        ids.Shape.Dimensions.CopyTo(dims);
        dims[ids.Rank] = dim;
        Tensor r = NewOut(Shape.Create(dims.Slice(0, ids.Rank + 1)), pool);
        ReadOnlySpan<float> tv = tc.InternalData<float>();
        Span<float> d = r.MutableData<float>();
        for (int i = 0; i < ids.Length; i++)
        {
            long id = BackwardOps.TargetAt(ic, i);
            if ((ulong)id >= (ulong)v)
            {
                Free(tc, tcopied);
                if (!ids.IsContiguous)
                {
                    ic.Dispose();
                }

                r.Dispose();
                throw new ArgumentOutOfRangeException(nameof(ids), $"Id {id} at position {i} is outside the table of {v} rows.");
            }

            tv.Slice((int)id * dim, dim).CopyTo(d.Slice(i * dim, dim));
        }

        Free(tc, tcopied);
        if (!Recording(table, ids))
        {
            if (!ids.IsContiguous)
            {
                ic.Dispose();
            }

            return r;
        }

        BackwardRecord rec = BackwardRecord.Create(OpKind.Embedding, 1, table);
        rec.AddTraceExtras(ids);
        rec.Save(0, ic, owned: !ids.IsContiguous);
        rec.SetInts(paddingIndex);
        return rec.Attach(r);
    }

    // ---------------------------------------------------------------------------------------------------------------- losses

    /// <summary>
    /// Negative log-likelihood: for log-probabilities [N, C] and class indices [N] (Int32 or Int64), <c>−logp[n, target[n]]</c>
    /// <param name="logProbs">Log-probabilities [N, C].</param>
    /// <param name="targets">Class indices [N], Int32 or Int64.</param>
    /// <param name="mean">Average over the counted rows (default true) rather than sum.</param>
    /// <param name="ignoreIndex">Target value that is skipped; default -100.</param>
    /// averaged (default) or summed over the rows whose target is not <paramref name="ignoreIndex"/>. Result is rank 0.
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    /// Record: slot 0 = targets; Ints = [mean, ignoreIndex].
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">A target is outside [0, C) and not the ignore index.</exception>
    public static Tensor NllLoss(in Tensor logProbs, in Tensor targets, bool mean = true, int ignoreIndex = -100, BufferPool? pool = null)
    {
        RequireFloat32(logProbs);
        if (logProbs.Rank != 2 || targets.Rank != 1 || targets.Shape[0] != logProbs.Shape[0] ||
            (targets.DType != DType.I32 && targets.DType != DType.I64))
        {
            throw new TensorShapeException($"NllLoss needs [N, C] log-probabilities and [N] Int32 or Int64 targets, not {logProbs.Shape} and {targets.Shape}.");
        }

        int rows = logProbs.Shape[0], cols = logProbs.Shape[1];
        Tensor lc = Contig(logProbs, out bool copied);
        Tensor tc = targets.IsContiguous ? targets : targets.Clone();
        ReadOnlySpan<float> lv = lc.InternalData<float>();
        double sum = 0;
        int count = 0;
        for (int i = 0; i < rows; i++)
        {
            long t = BackwardOps.TargetAt(tc, i);
            if (t == ignoreIndex)
            {
                continue;
            }

            if ((ulong)t >= (ulong)cols)
            {
                Free(lc, copied);
                if (!targets.IsContiguous)
                {
                    tc.Dispose();
                }

                throw new ArgumentOutOfRangeException(nameof(targets), $"Target {t} at row {i} is outside {cols} classes.");
            }

            sum -= lv[(i * cols) + (int)t];
            count++;
        }

        Free(lc, copied);
        Tensor r = NewOut(default, pool);
        r.MutableData<float>()[0] = (float)(mean ? sum / Math.Max(1, count) : sum);
        if (!Recording(logProbs))
        {
            if (!targets.IsContiguous)
            {
                tc.Dispose();
            }

            return r;
        }

        BackwardRecord rec = BackwardRecord.Create(OpKind.NllLoss, 1, logProbs);
        rec.Save(0, tc, owned: !targets.IsContiguous);
        rec.SetInts(mean ? 1 : 0, ignoreIndex);
        return rec.Attach(r);
    }

    /// <summary>Softmax cross-entropy: <see cref="NllLoss"/> of <see cref="LogSoftmax"/> over the class axis of [N, C] logits.</summary>
    public static Tensor CrossEntropy(in Tensor logits, in Tensor targets, bool mean = true, int ignoreIndex = -100, BufferPool? pool = null) =>
        NllLoss(LogSoftmax(logits, -1, pool), targets, mean, ignoreIndex, pool);

    // ---------------------------------------------------------------------------------------------------------------- dense layers

    /// <summary>
    /// <param name="x">Float32 input of any shape and strides.</param>
    /// <param name="weight">Weights.</param>
    /// <param name="bias">Bias, or an empty tensor for none.</param>
    /// <c>x·Wᵀ + b</c> over the last axis: <paramref name="x"/> [..., In], <paramref name="weight"/> [Out, In], optional
    /// <paramref name="bias"/> [Out] (empty for none). Built from <see cref="MatMul"/> (on the blocked GEMM), a broadcast add and views.
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    /// </summary>
    public static Tensor Linear(in Tensor x, in Tensor weight, in Tensor bias = default, BufferPool? pool = null)
    {
        if (weight.Rank != 2 || x.Rank < 1 || x.Shape[x.Rank - 1] != weight.Shape[1])
        {
            throw new TensorShapeException($"Linear: input {x.Shape} does not fit weights {weight.Shape} ([out, in]).");
        }

        int inF = weight.Shape[1], outF = weight.Shape[0];
        Tensor y = MatMul(x.Reshape(Shape.Create(x.Length / Math.Max(1, inF), inF)), weight.T(), pool);
        if (!bias.IsEmpty)
        {
            y = Add(y, bias, pool);
        }

        Span<int> dims = stackalloc int[Shape.MaxRank];
        x.Shape.Dimensions.CopyTo(dims);
        dims[x.Rank - 1] = outF;
        return y.Reshape(Shape.Create(dims.Slice(0, x.Rank)));
    }

    /// <summary>
    /// Scaled dot-product attention <c>softmax(scale·Q Kᵀ + mask)·V</c> per batch and head, forward on the fused attention kernel
    /// (fast or exact per <see cref="TensorCompute.Mode"/>), backward by recomputing the weights from Q and K (Dao et al. 2022).
    /// Record: slots q, k, v, mask (copies owned if not contiguous); Floats = [scale]. The mask gets no gradient.
    /// </summary>
    /// <param name="q">Queries [B, H, Lq, D]; D must be even.</param>
    /// <param name="k">Keys [B, H, Lk, D].</param>
    /// <param name="v">Values [B, H, Lk, D].</param>
    /// <param name="mask">Additive mask [Lq, Lk] or [B, Lq, Lk] (−inf hides a key), or empty for none.</param>
    /// <param name="scale">Score multiplier; default <c>1/√D</c> (pass NaN or leave 0 for the default).</param>
    /// <param name="pool">Pool for the result; default <see cref="BufferPool.Shared"/>.</param>
    public static Tensor Attention(in Tensor q, in Tensor k, in Tensor v, in Tensor mask = default, float scale = 0f, BufferPool? pool = null)
    {
        if (q.Rank != 4 || k.Rank != 4 || v.Rank != 4 || k.Shape != v.Shape || q.Shape[0] != k.Shape[0] || q.Shape[1] != k.Shape[1] ||
            q.Shape[3] != k.Shape[3] || q.Shape[3] % 2 != 0)
        {
            throw new TensorShapeException($"Attention needs [B,H,L,D] tensors with matching batch, heads and even D, not {q.Shape}, {k.Shape}, {v.Shape}.");
        }

        int b = q.Shape[0], h = q.Shape[1], lq = q.Shape[2], lk = k.Shape[2], d = q.Shape[3];
        if (!mask.IsEmpty)
        {
            RequireFloat32(mask);
            bool ok = (mask.Rank == 2 && mask.Shape[0] == lq && mask.Shape[1] == lk) ||
                      (mask.Rank == 3 && mask.Shape[0] == b && mask.Shape[1] == lq && mask.Shape[2] == lk);
            if (!ok)
            {
                throw new TensorShapeException($"The mask must be [{lq},{lk}] or [{b},{lq},{lk}], not {mask.Shape}.");
            }
        }

        if (scale == 0f || float.IsNaN(scale))
        {
            scale = 1f / float.Sqrt(d);
        }

        Tensor qc = Contig(q, out bool qcopied), kc = Contig(k, out bool kcopied), vc = Contig(v, out bool vcopied);
        Tensor mc = mask.IsEmpty ? default : Contig(mask, out _);
        Tensor r = NewOut(q.Shape, pool);
        AttentionShape sh = AttentionShape.Contiguous(b, h, lq, lk, d, scale);
        if (!mask.IsEmpty && mask.Rank == 2)
        {
            sh.MaskBatchStride = 0;
        }

        AttentionDriver.Run(TensorCompute.Exact, TensorCompute.Path, sh, qc.InternalData<float>(), kc.InternalData<float>(), vc.InternalData<float>(),
            mc.IsEmpty ? default : mc.InternalData<float>(), r.MutableData<float>(), TensorCompute.Threads);
        if (!Recording(q, k, v, mask))
        {
            Free(qc, qcopied);
            Free(kc, kcopied);
            Free(vc, vcopied);
            if (!mask.IsEmpty && !mask.IsContiguous)
            {
                mc.Dispose();
            }

            return r;
        }

        BackwardRecord rec = BackwardRecord.Create(OpKind.Attention, 4, q, k, v);
        rec.AddTraceExtras(mask);
        SaveContig(rec, 0, qc, qcopied);
        SaveContig(rec, 1, kc, kcopied);
        SaveContig(rec, 2, vc, vcopied);
        if (!mask.IsEmpty)
        {
            rec.Save(3, mc, owned: !mask.IsContiguous);
        }

        rec.SetFloats(scale);
        return rec.Attach(r);
    }
}
