using System;
using RecursiveAI.OneModel.Graph.Operators.Signal;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Vision;


/// <summary>AffineGrid (opset 20): the sampling grid for GridSample from batches of 2x3 (2-D) or 3x4 (3-D) affine matrices; align_corners as in GridSample.</summary>
internal sealed class AffineGridOp : OpKernel
{
    public static readonly OpDef Def = new("AffineGrid", 20, int.MaxValue, static n => new AffineGridOp(n));

    private readonly bool _alignCorners;

    private AffineGridOp(NodeInfo n)
    {
        _alignCorners = n.GetInt("align_corners", 0) != 0;
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 2 || !ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects theta and size");
        }

        if (ctx.InputType(0) is not (DType.F32 or DType.F64))
        {
            throw ctx.Unsupported($"element type {ctx.InputType(0)} is not supported yet");
        }

        OpUtil.RequireIntegers(ctx, 1, "size");
        if (!ctx.TryGetInputValue(1, out Tensor sz))
        {
            ctx.RequireRunTimeValues();
            return;
        }

        IntVec s = IntVec.Of(sz);
        Shape theta = ctx.InputShape(0);
        if ((s.Length is not (4 or 5)) || theta.Rank != 3 || theta[0] != s[0] || theta[1] != s.Length - 2 || theta[2] != s.Length - 1)
        {
            throw ctx.Fail($"theta {theta} does not fit size of {s.Length} values: expected (N, {s.Length - 2}, {s.Length - 1})");
        }

        Span<long> dims = stackalloc long[Shape.MaxRank];
        dims[0] = s[0];
        for (int i = 2; i < s.Length; i++)
        {
            dims[i - 1] = s[i];
        }

        dims[s.Length - 1] = s.Length - 2;
        ctx.SetOutput(0, ctx.InputType(0), OpUtil.MakeGrownShape(ctx, dims.Slice(0, s.Length), ctx.InputType(0)));
    }

    // Coordinate of cell i of n along an axis, in [-1, 1]: the cell centres, or with align_corners the first and last cells on the ends.
    private double Coord(int i, int n) => n <= 1 ? 0 : _alignCorners ? -1 + (2.0 * i / (n - 1)) : ((2.0 * i) + 1) / n - 1;

    public override void Run(in OpRun run)
    {
        Tensor th = run.Inputs[0], y = run.Outputs[0];
        int rank = y.Rank - 2;
        int batch = th.Shape[0];
        int d = rank == 3 ? y.Shape[1] : 1, h = y.Shape[rank == 3 ? 2 : 1], w = y.Shape[rank == 3 ? 3 : 2];
        int cols = rank + 1;
        Span<double> p = stackalloc double[4];
        for (int b = 0; b < batch; b++)
        {
            long o = (long)b * d * h * w * rank;
            for (int z = 0; z < d; z++)
            {
                for (int r = 0; r < h; r++)
                {
                    for (int c = 0; c < w; c++)
                    {
                        // The point is (x, y[, z], 1): x runs along width, y along height, z along depth.
                        p[0] = Coord(c, w);
                        p[1] = Coord(r, h);
                        if (rank == 3)
                        {
                            p[2] = Coord(z, d);
                        }

                        p[rank] = 1;
                        for (int i = 0; i < rank; i++)
                        {
                            double v = 0;
                            for (int k = 0; k < cols; k++)
                            {
                                v += DftOp.Read(th, (((long)b * rank) + i) * cols + k) * p[k];
                            }

                            DftOp.Write(y, o++, v);
                        }
                    }
                }
            }
        }
    }
}

/// <summary>
/// GridSample (opset 16, 20, 22): samples X (N, C, spatial...) at the normalised positions in grid (N, out spatial..., 2 or 3) with
/// the last grid component along the last spatial axis. Two or three spatial axes; modes linear (bilinear), nearest (ties to even)
/// and cubic (bicubic, 2-D only, cubic convolution with a = -0.75); padding zeros, border or reflection; align_corners. For linear and
/// nearest the padding rule is applied to the sample position; for cubic it is applied to each tap's index. Computed in binary64
/// and rounded once. Not exact (a portable-everywhere ordering is not defined for the weights).
/// </summary>
internal sealed class GridSampleOp : OpKernel
{
    public static readonly OpDef Def = new("GridSample", 16, int.MaxValue, static n => new GridSampleOp(n));

    private enum Mode { Linear, Nearest, Cubic }

    private enum Padding { Zeros, Border, Reflection }

    private readonly Mode _mode;
    private readonly Padding _padding;
    private readonly bool _alignCorners;

    private GridSampleOp(NodeInfo n)
    {
        _mode = n.GetString("mode", "linear") switch
        {
            "linear" or "bilinear" => Mode.Linear,
            "nearest" => Mode.Nearest,
            "cubic" or "bicubic" => Mode.Cubic,
            var other => throw n.Error($"attribute 'mode' is '{other}', expected linear, nearest or cubic"),
        };
        _padding = n.GetString("padding_mode", "zeros") switch
        {
            "zeros" => Padding.Zeros,
            "border" => Padding.Border,
            "reflection" => Padding.Reflection,
            var other => throw n.Error($"attribute 'padding_mode' is '{other}', expected zeros, border or reflection"),
        };
        _alignCorners = n.GetInt("align_corners", 0) != 0;
    }

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 2 || !ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects X and grid");
        }

        Shape x = ctx.InputShape(0), g = ctx.InputShape(1);
        if (ctx.InputType(0) is not (DType.F32 or DType.F64) || ctx.InputType(1) != ctx.InputType(0))
        {
            throw ctx.Unsupported("X and grid must both be float32 or both float64");
        }

        int r = x.Rank - 2;
        if (r is not (2 or 3) || g.Rank != x.Rank || g[0] != x[0] || g[g.Rank - 1] != r)
        {
            throw ctx.Fail($"grid {g} does not fit X {x}: expected (N, out spatial..., {r})");
        }

        if (_mode == Mode.Cubic && r != 2)
        {
            throw ctx.Unsupported("cubic mode is only defined for 2-D inputs");
        }

        Span<long> dims = stackalloc long[Shape.MaxRank];
        dims[0] = x[0];
        dims[1] = x[1];
        for (int i = 1; i <= r; i++)
        {
            dims[i + 1] = g[i];
        }

        ctx.SetOutput(0, ctx.InputType(0), OpUtil.MakeGrownShape(ctx, dims.Slice(0, x.Rank), ctx.InputType(0)));
    }

    private double Unnormalize(double v, int size) => _alignCorners ? (v + 1) / 2 * (size - 1) : (((v + 1) * size) - 1) / 2;

    private static double Reflect(double c, double twiceLow, double twiceHigh)
    {
        if (twiceLow == twiceHigh)
        {
            return 0;
        }

        double min = twiceLow / 2, span = (twiceHigh - twiceLow) / 2;
        double d = Math.Abs(c - min);
        double extra = d % span;
        long flips = (long)Math.Floor(d / span);
        return flips % 2 == 0 ? extra + min : span - extra + min;
    }

    // Applies the padding mode to a continuous sample position along an axis of 'size'.
    private double Pad(double c, int size)
    {
        if (_padding == Padding.Border)
        {
            return Math.Clamp(c, 0, size - 1);
        }

        if (_padding == Padding.Reflection)
        {
            c = _alignCorners ? Reflect(c, 0, 2 * (size - 1)) : Reflect(c, -1, (2 * size) - 1);
            return Math.Clamp(c, 0, size - 1);
        }

        return c;
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], g = run.Inputs[1], y = run.Outputs[0];
        int r = x.Rank - 2, n = x.Shape[0], c = x.Shape[1];
        Span<int> inD = stackalloc int[3], outD = stackalloc int[3];
        int inVol = 1, outVol = 1;
        for (int i = 0; i < r; i++)
        {
            inD[i] = x.Shape[2 + i];
            outD[i] = y.Shape[2 + i];
            inVol *= inD[i];
            outVol *= outD[i];
        }

        Span<double> pos = stackalloc double[3];
        for (int b = 0; b < n; b++)
        {
            for (int o = 0; o < outVol; o++)
            {
                // Sample position along each spatial axis; grid component 0 is the last axis (x), then y, then z.
                for (int a = 0; a < r; a++)
                {
                    int axis = r - 1 - a;
                    double raw = DftOp.Read(g, ((((long)b * outVol) + o) * r) + a);
                    double u = Unnormalize(raw, inD[axis]);
                    pos[axis] = _mode == Mode.Cubic ? u : Pad(u, inD[axis]);
                }

                for (int ch = 0; ch < c; ch++)
                {
                    long plane = (((long)b * c) + ch) * inVol;
                    double v = _mode switch
                    {
                        Mode.Nearest => Nearest(x, plane, pos, inD, r),
                        Mode.Linear => Linear(x, plane, pos, inD, r),
                        _ => Cubic(x, plane, pos, inD),
                    };
                    DftOp.Write(y, (((long)b * c) + ch) * outVol + o, v);
                }
            }
        }
    }

    private static double Tap(Tensor x, long plane, ReadOnlySpan<int> idx, ReadOnlySpan<int> inD, int r)
    {
        long at = 0;
        for (int a = 0; a < r; a++)
        {
            if ((uint)idx[a] >= (uint)inD[a])
            {
                return 0;
            }

            at = (at * inD[a]) + idx[a];
        }

        return DftOp.Read(x, plane + at);
    }

    private static double Nearest(Tensor x, long plane, ReadOnlySpan<double> pos, ReadOnlySpan<int> inD, int r)
    {
        Span<int> idx = stackalloc int[3];
        for (int a = 0; a < r; a++)
        {
            idx[a] = (int)Math.Round(pos[a], MidpointRounding.ToEven);
        }

        return Tap(x, plane, idx, inD, r);
    }

    private static double Linear(Tensor x, long plane, ReadOnlySpan<double> pos, ReadOnlySpan<int> inD, int r)
    {
        Span<int> lo = stackalloc int[3], idx = stackalloc int[3];
        Span<double> frac = stackalloc double[3];
        for (int a = 0; a < r; a++)
        {
            double f = Math.Floor(pos[a]);
            lo[a] = (int)f;
            frac[a] = pos[a] - f;
        }

        double sum = 0;
        for (int corner = 0; corner < 1 << r; corner++)
        {
            double w = 1;
            for (int a = 0; a < r; a++)
            {
                int bit = (corner >> a) & 1;
                idx[a] = lo[a] + bit;
                w *= bit == 1 ? frac[a] : 1 - frac[a];
            }

            if (w != 0)
            {
                sum += w * Tap(x, plane, idx, inD, r);
            }
        }

        return sum;
    }

    private double Cubic(Tensor x, long plane, ReadOnlySpan<double> pos, ReadOnlySpan<int> inD)
    {
        const double A = -0.75;
        // Axis 0 of pos is the height and axis 1 the width; x is the column.
        double fx = Math.Floor(pos[1]), fy = Math.Floor(pos[0]);
        int x0 = (int)fx, y0 = (int)fy;
        Span<double> wx = stackalloc double[4], wy = stackalloc double[4];
        Weights(pos[1] - fx, wx, A);
        Weights(pos[0] - fy, wy, A);
        double sum = 0;
        for (int j = 0; j < 4; j++)
        {
            double row = 0;
            for (int i = 0; i < 4; i++)
            {
                row += wx[i] * Bounded(x, plane, x0 - 1 + i, y0 - 1 + j, inD);
            }

            sum += wy[j] * row;
        }

        return sum;
    }

    private static void Weights(double t, Span<double> w, double a)
    {
        static double Near(double d, double a) => ((((a + 2) * d) - (a + 3)) * d * d) + 1;
        static double Far(double d, double a) => ((((a * d) - (5 * a)) * d) + (8 * a)) * d - (4 * a);
        w[0] = Far(t + 1, a);
        w[1] = Near(t, a);
        w[2] = Near(1 - t, a);
        w[3] = Far(2 - t, a);
    }

    // A tap of the cubic kernel at column ix, row iy: the padding rule applies to its integer position.
    private double Bounded(Tensor x, long plane, int ix, int iy, ReadOnlySpan<int> inD)
    {
        int height = inD[0], width = inD[1];
        double cx = ix, cy = iy;
        if (_padding == Padding.Border)
        {
            cx = Math.Clamp(cx, 0, width - 1);
            cy = Math.Clamp(cy, 0, height - 1);
        }
        else if (_padding == Padding.Reflection)
        {
            cx = Pad(cx, width);
            cy = Pad(cy, height);
        }

        int col = (int)cx, row = (int)cy;
        if ((uint)col >= (uint)width || (uint)row >= (uint)height)
        {
            return 0;
        }

        return DftOp.Read(x, plane + ((long)row * width) + col);
    }
}
