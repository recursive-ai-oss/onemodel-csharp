using System;
using System.Collections.Generic;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Vision;

/// <summary>
/// NonMaxSuppression (opset 10, 11): for each batch entry and class, boxes with a score above score_threshold are visited from the
/// highest score down (ties by lower box index) and kept unless their IoU with an already kept box exceeds iou_threshold, up to
/// max_output_boxes_per_class. The result is a (selected, 3) int64 tensor of (batch, class, box) rows in that visiting order. Boxes
/// are (y1, x1, y2, x2) with any pair of opposite corners, or (x_centre, y_centre, width, height) with center_point_box = 1.
/// </summary>
internal sealed class NonMaxSuppressionOp : OpKernel
{
    public static readonly OpDef Def = new("NonMaxSuppression", 10, int.MaxValue, static n => new NonMaxSuppressionOp(n));

    private readonly bool _center;

    private NonMaxSuppressionOp(NodeInfo n)
    {
        _center = n.GetInt("center_point_box", 0) != 0;
    }

    public override bool SupportsExact => true;

    public override bool OutputsDataDependent => true;

    public override void Infer(InferContext ctx) => throw new InvalidOperationException("NonMaxSuppression is shaped when it runs.");

    public override void Run(in OpRun run)
    {
        Tensor boxesT = run.Inputs[0], scoresT = run.Inputs[1];
        if (boxesT.DType != DType.F32 || scoresT.DType != DType.F32 || boxesT.Rank != 3 || boxesT.Shape[2] != 4 || scoresT.Rank != 3 ||
            scoresT.Shape[0] != boxesT.Shape[0] || scoresT.Shape[2] != boxesT.Shape[1])
        {
            throw new SessionException(SessionErrorKind.InvalidInput, "NonMaxSuppression: boxes must be float32 (batch, boxes, 4) and scores float32 (batch, classes, boxes)");
        }

        int batches = boxesT.Shape[0], count = boxesT.Shape[1], classes = scoresT.Shape[1];
        long maxPer = run.Inputs.Length > 2 && !run.Inputs[2].IsEmpty ? OpUtil.First(run.Inputs[2]) : 0;
        float iouThreshold = run.Inputs.Length > 3 && !run.Inputs[3].IsEmpty ? run.Inputs[3].Data<float>()[0] : 0f;
        bool hasScoreThreshold = run.Inputs.Length > 4 && !run.Inputs[4].IsEmpty;
        float scoreThreshold = hasScoreThreshold ? run.Inputs[4].Data<float>()[0] : 0f;
        ReadOnlySpan<float> boxes = boxesT.Data<float>(), scores = scoresT.Data<float>();

        var selected = new List<(long Batch, long Class, long Box)>();
        var order = new List<int>(count);
        var kept = new List<int>();
        for (int b = 0; b < batches; b++)
        {
            for (int c = 0; c < classes; c++)
            {
                float[] s = scores.Slice(((b * classes) + c) * count, count).ToArray();
                order.Clear();
                for (int i = 0; i < count; i++)
                {
                    if (!hasScoreThreshold || s[i] > scoreThreshold)
                    {
                        order.Add(i);
                    }
                }

                // Stable: equal scores keep the lower box index first.
                order.Sort((p, q) => s[q].CompareTo(s[p]) is var cmp && cmp != 0 ? cmp : p.CompareTo(q));
                kept.Clear();
                foreach (int i in order)
                {
                    if (kept.Count >= maxPer)
                    {
                        break;
                    }

                    bool suppressed = false;
                    foreach (int k in kept)
                    {
                        if (Iou(boxes.Slice(((b * count) + i) * 4, 4), boxes.Slice(((b * count) + k) * 4, 4)) > iouThreshold)
                        {
                            suppressed = true;
                            break;
                        }
                    }

                    if (!suppressed)
                    {
                        kept.Add(i);
                        selected.Add((b, c, i));
                    }
                }
            }
        }

        Tensor y = run.AllocateOutput(0, DType.I64, Shape.Create(selected.Count, 3));
        Span<long> o = y.MutableData<long>();
        for (int i = 0; i < selected.Count; i++)
        {
            o[3 * i] = selected[i].Batch;
            o[(3 * i) + 1] = selected[i].Class;
            o[(3 * i) + 2] = selected[i].Box;
        }
    }

    private float Iou(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        Corners(a, out float ay1, out float ax1, out float ay2, out float ax2);
        Corners(b, out float by1, out float bx1, out float by2, out float bx2);
        float areaA = (ay2 - ay1) * (ax2 - ax1), areaB = (by2 - by1) * (bx2 - bx1);
        if (areaA <= 0 || areaB <= 0)
        {
            return 0;
        }

        float iy1 = MathF.Max(ay1, by1), ix1 = MathF.Max(ax1, bx1), iy2 = MathF.Min(ay2, by2), ix2 = MathF.Min(ax2, bx2);
        float inter = MathF.Max(iy2 - iy1, 0) * MathF.Max(ix2 - ix1, 0);
        float union = areaA + areaB - inter;
        return union <= 0 ? 0 : inter / union;
    }

    private void Corners(ReadOnlySpan<float> v, out float y1, out float x1, out float y2, out float x2)
    {
        if (_center)
        {
            float halfW = v[2] / 2, halfH = v[3] / 2;
            x1 = v[0] - halfW;
            x2 = v[0] + halfW;
            y1 = v[1] - halfH;
            y2 = v[1] + halfH;
        }
        else
        {
            y1 = MathF.Min(v[0], v[2]);
            y2 = MathF.Max(v[0], v[2]);
            x1 = MathF.Min(v[1], v[3]);
            x2 = MathF.Max(v[1], v[3]);
        }
    }
}

/// <summary>
/// RoiAlign (opset 10, 16, 22): for each region (x1, y1, x2, y2 scaled by spatial_scale) of the batch entry batch_indices[r], an
/// output_height x output_width grid of bins, each the average (or maximum, mode = max) of sampling_ratio x sampling_ratio bilinear
/// samples at the bin's sub-cell centres; sampling_ratio 0 uses ceil(region size / bins). coordinate_transformation_mode half_pixel
/// shifts the region by half a pixel; output_half_pixel (the default, and the only mode before opset 16) does not, and then regions
/// are at least one pixel wide. Samples more than a pixel outside the image are 0; others are clamped to the edge.
/// </summary>
internal sealed class RoiAlignOp : OpKernel
{
    public static readonly OpDef Def = new("RoiAlign", 10, int.MaxValue, static n => new RoiAlignOp(n));

    private readonly bool _max, _halfPixel;
    private readonly int _outH, _outW, _ratio;
    private readonly float _scale;

    private RoiAlignOp(NodeInfo n)
    {
        string mode = n.GetString("mode", "avg");
        _max = mode == "max";
        if (mode is not ("avg" or "max"))
        {
            throw n.Error($"attribute 'mode' is '{mode}', expected avg or max");
        }

        string ct = n.GetString("coordinate_transformation_mode", "output_half_pixel");
        _halfPixel = ct == "half_pixel";
        if (ct is not ("half_pixel" or "output_half_pixel"))
        {
            throw n.Error($"coordinate_transformation_mode '{ct}' is not supported");
        }

        _outH = (int)n.GetInt("output_height", 1);
        _outW = (int)n.GetInt("output_width", 1);
        _ratio = (int)n.GetInt("sampling_ratio", 0);
        _scale = n.GetFloat("spatial_scale", 1f);
        if (_outH < 1 || _outW < 1 || _ratio < 0)
        {
            throw n.Error("output_height and output_width must be at least 1 and sampling_ratio cannot be negative");
        }
    }

    public override bool SupportsExact => false;

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 3 || !ctx.HasInput(0) || !ctx.HasInput(1) || !ctx.HasInput(2))
        {
            throw ctx.Fail("expects X, rois and batch_indices");
        }

        Shape x = ctx.InputShape(0), r = ctx.InputShape(1);
        if (ctx.InputType(0) != DType.F32 || ctx.InputType(1) != DType.F32 || ctx.InputType(2) != DType.I64 || x.Rank != 4 || r.Rank != 2 || r[1] != 4 ||
            ctx.InputShape(2).Rank != 1 || ctx.InputShape(2)[0] != r[0])
        {
            throw ctx.Fail("X must be float32 (N, C, H, W), rois float32 (R, 4) and batch_indices int64 (R)");
        }

        ctx.SetOutput(0, DType.F32, Shape.Create(r[0], x[1], _outH, _outW));
    }

    public override void Run(in OpRun run)
    {
        Tensor xt = run.Inputs[0];
        ReadOnlySpan<float> x = xt.Data<float>(), rois = run.Inputs[1].Data<float>();
        ReadOnlySpan<long> bi = run.Inputs[2].Data<long>();
        Span<float> y = run.Outputs[0].MutableData<float>();
        int n = xt.Shape[0], c = xt.Shape[1], h = xt.Shape[2], w = xt.Shape[3];
        float offset = _halfPixel ? 0.5f : 0f;
        for (int r = 0; r < bi.Length; r++)
        {
            long batch = bi[r];
            if ((ulong)batch >= (ulong)n)
            {
                throw new SessionException(SessionErrorKind.InvalidInput, $"RoiAlign: batch index {batch} is outside [0, {n})");
            }

            float x1 = (rois[4 * r] * _scale) - offset, y1 = (rois[(4 * r) + 1] * _scale) - offset;
            float x2 = (rois[(4 * r) + 2] * _scale) - offset, y2 = (rois[(4 * r) + 3] * _scale) - offset;
            float rw = x2 - x1, rh = y2 - y1;
            if (!_halfPixel)
            {
                rw = MathF.Max(rw, 1);
                rh = MathF.Max(rh, 1);
            }

            float binH = rh / _outH, binW = rw / _outW;
            int sh = _ratio > 0 ? _ratio : (int)MathF.Ceiling(rh / _outH), sw = _ratio > 0 ? _ratio : (int)MathF.Ceiling(rw / _outW);
            for (int ch = 0; ch < c; ch++)
            {
                ReadOnlySpan<float> plane = x.Slice(((((int)batch * c) + ch)) * h * w, h * w);
                for (int ph = 0; ph < _outH; ph++)
                {
                    for (int pw = 0; pw < _outW; pw++)
                    {
                        float acc = _max ? float.MinValue : 0;
                        for (int iy = 0; iy < sh; iy++)
                        {
                            float yy = y1 + (ph * binH) + ((iy + 0.5f) * binH / sh);
                            for (int ix = 0; ix < sw; ix++)
                            {
                                float xx = x1 + (pw * binW) + ((ix + 0.5f) * binW / sw);
                                float v = Bilinear(plane, h, w, yy, xx, _max);
                                acc = _max ? MathF.Max(acc, v) : acc + v;
                            }
                        }

                        if (!_max)
                        {
                            acc /= Math.Max(sh * sw, 1);
                        }

                        y[((((r * c) + ch) * _outH) + ph) * _outW + pw] = acc;
                    }
                }
            }
        }
    }

    // With max pooling each of the four weighted neighbours competes on its own (the backend cases expect this), instead of their sum.
    private static float Bilinear(ReadOnlySpan<float> plane, int h, int w, float y, float x, bool max)
    {
        if (y < -1 || y > h || x < -1 || x > w)
        {
            return 0;
        }

        y = MathF.Max(y, 0);
        x = MathF.Max(x, 0);
        int y0 = (int)y, x0 = (int)x;
        int y1 = y0 + 1, x1 = x0 + 1;
        if (y0 >= h - 1)
        {
            y0 = y1 = h - 1;
            y = y0;
        }

        if (x0 >= w - 1)
        {
            x0 = x1 = w - 1;
            x = x0;
        }

        float ly = y - y0, lx = x - x0, hy = 1 - ly, hx = 1 - lx;
        if (max)
        {
            return MathF.Max(MathF.Max(hy * hx * plane[(y0 * w) + x0], hy * lx * plane[(y0 * w) + x1]), MathF.Max(ly * hx * plane[(y1 * w) + x0], ly * lx * plane[(y1 * w) + x1]));
        }

        return (hy * hx * plane[(y0 * w) + x0]) + (hy * lx * plane[(y0 * w) + x1]) + (ly * hx * plane[(y1 * w) + x0]) + (ly * lx * plane[(y1 * w) + x1]);
    }
}
