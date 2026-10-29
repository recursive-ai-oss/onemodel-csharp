using System;
using System.Collections.Immutable;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

/// <summary>
/// Resize (and the deprecated Upsample, which is Resize with asymmetric coordinates and floor rounding): nearest, linear and cubic
/// interpolation over any rank, with every coordinate transformation and nearest mode, <c>axes</c> and
/// <c>keep_aspect_ratio_policy</c> (opset 18). <c>antialias</c> is not supported. The output size comes from <c>scales</c>
/// (floor(input * scale)) or from <c>sizes</c>; both are read as values, so a model that computes them at run time is shaped when it runs.
/// </summary>
internal sealed class ResizeOp : OpKernel
{
    // Opset 10 has only the scales input and no attributes beyond the mode; from 11 the coordinate and rounding modes are attributes.
    public static readonly OpDef Def10 = new("Resize", 10, 10, static n => new ResizeOp(n, legacy: true));

    public static readonly OpDef Def = new("Resize", 11, int.MaxValue, static n => new ResizeOp(n, legacy: false));

    public static readonly OpDef UpsampleDef = new("Upsample", 7, 10, static n => new ResizeOp(n, legacy: true));

    private enum Policy
    {
        Stretch,
        NotLarger,
        NotSmaller,
    }

    private readonly ResizeOptions _options;
    private readonly ImmutableArray<long> _axes;
    private readonly Policy _policy;
    private readonly int _scalesInput;
    private readonly int _sizesInput;
    private readonly int _roiInput;
    private readonly ImmutableArray<float> _scalesAttr;

    private ResizeOp(NodeInfo node, bool legacy)
    {
        string mode = node.GetString("mode", "nearest");
        ResizeMode m = mode switch
        {
            "nearest" => ResizeMode.Nearest,
            "linear" or "bilinear" => ResizeMode.Linear,
            "cubic" when !legacy => ResizeMode.Cubic,
            _ => throw node.Error($"mode '{mode}' is not supported"),
        };
        if (legacy)
        {
            _options = new ResizeOptions(m, ResizeCoord.Asymmetric, ResizeNearest.Floor, -0.75f, false, 0f);
            _scalesInput = node.OpType == "Upsample" && node.Opset < 9 ? -1 : 1;
            _scalesAttr = node.OpType == "Upsample" && node.Opset < 9 ? node.GetFloats("scales") : [];
            _sizesInput = -1;
            _roiInput = -1;
            if (_scalesInput < 0 && _scalesAttr.IsDefaultOrEmpty)
            {
                throw node.Error("attribute scales is required");
            }
        }
        else
        {
            ResizeCoord coord = node.GetString("coordinate_transformation_mode", "half_pixel") switch
            {
                "half_pixel" => ResizeCoord.HalfPixel,
                "half_pixel_symmetric" => ResizeCoord.HalfPixelSymmetric,
                "pytorch_half_pixel" => ResizeCoord.PytorchHalfPixel,
                "align_corners" => ResizeCoord.AlignCorners,
                "asymmetric" => ResizeCoord.Asymmetric,
                "tf_crop_and_resize" => ResizeCoord.TfCropAndResize,
                var other => throw node.Error($"coordinate_transformation_mode '{other}' is not supported"),
            };
            ResizeNearest nearest = node.GetString("nearest_mode", "round_prefer_floor") switch
            {
                "round_prefer_floor" => ResizeNearest.RoundPreferFloor,
                "round_prefer_ceil" => ResizeNearest.RoundPreferCeil,
                "floor" => ResizeNearest.Floor,
                "ceil" => ResizeNearest.Ceil,
                var other => throw node.Error($"nearest_mode '{other}' is not supported"),
            };
            if (node.GetInt("antialias", 0) != 0)
            {
                throw node.Unsupported("antialias = 1 is not supported");
            }

            _options = new ResizeOptions(m, coord, nearest, node.GetFloat("cubic_coeff_a", -0.75f),
                node.GetInt("exclude_outside", 0) != 0, node.GetFloat("extrapolation_value", 0f));
            _roiInput = 1;
            _scalesInput = 2;
            _sizesInput = 3;
            _scalesAttr = [];
            _policy = node.GetString("keep_aspect_ratio_policy", "stretch") switch
            {
                "stretch" => Policy.Stretch,
                "not_larger" => Policy.NotLarger,
                "not_smaller" => Policy.NotSmaller,
                var other => throw node.Error($"keep_aspect_ratio_policy '{other}' is not supported"),
            };
            _axes = node.GetInts("axes");
        }
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects input X");
        }

        if (ctx.InputType(0) != DType.F32)
        {
            throw ctx.Unsupported($"element type {ctx.InputType(0)} is not supported yet (float32 only)");
        }

        Shape x = ctx.InputShape(0);
        if (x.Rank == 0)
        {
            throw ctx.Fail("the input must have at least one dimension");
        }

        // The scale or size values decide the output shape: wait for them if they are not constants.
        Tensor scales = default, sizes = default;
        bool haveScales = _scalesAttr.IsDefaultOrEmpty && Present(ctx, _scalesInput, out scales);
        bool haveSizes = Present(ctx, _sizesInput, out sizes);
        if (!haveScales && !haveSizes && _scalesAttr.IsDefaultOrEmpty)
        {
            if (WantsValue(ctx, _scalesInput) || WantsValue(ctx, _sizesInput))
            {
                ctx.RequireRunTimeValues();
                return;
            }

            throw ctx.Fail("expects scales or sizes");
        }

        Span<int> outDims = stackalloc int[Shape.MaxRank];
        Span<float> sc = stackalloc float[Shape.MaxRank];
        Plan(ctx, x, haveScales ? scales.Data<float>() : _scalesAttr.AsSpan(), haveSizes ? sizes.Data<long>() : default, outDims, sc);
        ctx.SetOutput(0, DType.F32, Shape.Create(outDims.Slice(0, x.Rank)));
    }

    // Whether input i is declared with a non-empty shape (so its value is needed), as opposed to omitted or empty.
    private static bool WantsValue(InferContext ctx, int i) => i >= 0 && ctx.HasInput(i) && ctx.InputShape(i).Length > 0;

    private static bool Present(InferContext ctx, int i, out Tensor value)
    {
        value = default;
        return i >= 0 && ctx.HasInput(i) && ctx.InputShape(i).Length > 0 && ctx.TryGetInputValue(i, out value);
    }

    // Output extents and per-axis scales from scales or sizes.
    private void Plan(InferContext? ctx, Shape x, ReadOnlySpan<float> scalesIn, ReadOnlySpan<long> sizesIn, Span<int> outDims, Span<float> scales)
    {
        int rank = x.Rank;
        Span<int> axes = stackalloc int[Shape.MaxRank];
        int n = rank;
        if (_axes.IsDefaultOrEmpty)
        {
            for (int i = 0; i < rank; i++)
            {
                axes[i] = i;
            }
        }
        else
        {
            n = _axes.Length;
            if (n > rank)
            {
                throw Fail(ctx, "axes has more entries than the input has dimensions");
            }

            for (int i = 0; i < n; i++)
            {
                if (_axes[i] < -rank || _axes[i] >= rank)
                {
                    throw Fail(ctx, $"axis {_axes[i]} is out of range for rank {rank}");
                }

                axes[i] = Shape.NormalizeAxis((int)_axes[i], rank);
            }
        }

        for (int i = 0; i < rank; i++)
        {
            outDims[i] = x[i];
            scales[i] = 1f;
        }

        if (!sizesIn.IsEmpty)
        {
            if (sizesIn.Length != n)
            {
                throw Fail(ctx, $"sizes has {sizesIn.Length} values; expected {n}");
            }

            float policyScale = _policy == Policy.NotLarger ? float.MaxValue : 0f;
            for (int k = 0; k < n; k++)
            {
                float s = (float)sizesIn[k] / x[axes[k]];
                policyScale = _policy == Policy.NotLarger ? MathF.Min(policyScale, s) : MathF.Max(policyScale, s);
            }

            for (int k = 0; k < n; k++)
            {
                int a = axes[k];
                if (sizesIn[k] < 0 || sizesIn[k] > int.MaxValue || (x[a] == 0 && sizesIn[k] > 0))
                {
                    throw Fail(ctx, $"size {sizesIn[k]} for axis {a} is invalid for an input extent of {x[a]}");
                }

                if (_policy == Policy.Stretch)
                {
                    outDims[a] = (int)sizesIn[k];
                    scales[a] = x[a] == 0 ? 1f : (float)sizesIn[k] / x[a];
                }
                else
                {
                    outDims[a] = (int)Math.Round((double)policyScale * x[a], MidpointRounding.AwayFromZero);
                    scales[a] = policyScale;
                }
            }
        }
        else
        {
            if (scalesIn.Length != n)
            {
                throw Fail(ctx, $"scales has {scalesIn.Length} values; expected {n}");
            }

            for (int k = 0; k < n; k++)
            {
                int a = axes[k];
                float s = scalesIn[k];
                if (!(s > 0f) || float.IsInfinity(s))
                {
                    throw Fail(ctx, $"scale {s} for axis {a} must be positive and finite");
                }

                float o = MathF.Floor(x[a] * s);
                if (o > int.MaxValue)
                {
                    throw Fail(ctx, $"axis {a} would grow to {o} elements");
                }

                outDims[a] = (int)o;
                scales[a] = s;
            }
        }

        for (int i = 0; i < rank; i++)
        {
            if (x[i] == 0 && outDims[i] > 0)
            {
                throw Fail(ctx, $"cannot resample the empty axis {i} to {outDims[i]} elements");
            }
        }
    }

    private static SessionException Fail(InferContext? ctx, string message) =>
        ctx is not null ? ctx.Fail(message) : new SessionException(SessionErrorKind.InvalidModel, message);

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], y = run.Outputs[0];
        if (y.Length == 0)
        {
            return;
        }

        int rank = x.Rank;
        Tensor scalesT = Get(run, _scalesInput), sizesT = Get(run, _sizesInput);
        Span<int> outDims = stackalloc int[Shape.MaxRank];
        Span<float> sc = stackalloc float[Shape.MaxRank];
        Plan(null, x.Shape, _scalesAttr.IsDefaultOrEmpty && !scalesT.IsEmpty ? scalesT.Data<float>() : _scalesAttr.AsSpan(),
            !sizesT.IsEmpty ? sizesT.Data<long>() : default, outDims, sc);

        // The region of interest, per axis (0..1 for axes the model did not list).
        Span<float> roi = stackalloc float[2 * Shape.MaxRank];
        bool useRoi = false;
        Tensor roiT = Get(run, _roiInput);
        if (_options.Coord == ResizeCoord.TfCropAndResize && !roiT.IsEmpty)
        {
            ReadOnlySpan<float> r = roiT.Data<float>();
            int n = _axes.IsDefaultOrEmpty ? rank : _axes.Length;
            if (r.Length == 2 * n)
            {
                for (int i = 0; i < rank; i++)
                {
                    roi[i] = 0f;
                    roi[rank + i] = 1f;
                }

                for (int k = 0; k < n; k++)
                {
                    int a = _axes.IsDefaultOrEmpty ? k : Shape.NormalizeAxis((int)_axes[k], rank);
                    roi[a] = r[k];
                    roi[rank + a] = r[n + k];
                }

                useRoi = true;
            }
        }

        long maxLen = Math.Max(x.Length, y.Length);
        int maxOut = 1;
        for (int i = 0; i < rank; i++)
        {
            maxOut = Math.Max(maxOut, outDims[i]);
        }

        int taps = Resize.TapsPer(_options.Mode);
        Tensor a1 = run.Scratch(DType.F32, Shape.Create((int)maxLen));
        Tensor b1 = run.Scratch(DType.F32, Shape.Create((int)maxLen));
        Tensor ti = run.Scratch(DType.I32, Shape.Create(taps * maxOut));
        Tensor tw = run.Scratch(DType.F32, Shape.Create(taps * maxOut));
        Shape xs = x.Shape, ys = y.Shape;
        Resize.Run(_options, xs.Dimensions, ys.Dimensions, sc.Slice(0, rank), useRoi ? roi.Slice(0, 2 * rank) : default,
            x.Data<float>(), y.MutableData<float>(), a1.MutableData<float>(), b1.MutableData<float>(), ti.MutableData<int>(),
            tw.MutableData<float>(), run.Exact);
    }

    private static Tensor Get(in OpRun run, int i) =>
        i >= 0 && i < run.Inputs.Length && !run.Inputs[i].IsEmpty && run.Inputs[i].Length > 0 ? run.Inputs[i] : default;
}
