using System;
using System.Collections.Generic;
using RecursiveAI.OneGpu;
using RecursiveAI.OneGpu.Kernels;
using RecursiveAI.OneModel.Graph.Operators;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Kernels.Exact;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu;

/// <summary>
/// The entries of the vision models: Conv (with the activation and residual the graph rewrites fuse into it), ConvTranspose, MaxPool,
/// AveragePool, the global pools, Resize and Concat. Tensors are NCHW and weights OIHW, as in ONNX, so nothing is relaid. One or two
/// spatial axes; a one-dimensional operator runs as a two-dimensional one of height 1. Every kernel has a fixed summation order and
/// no atomics, so one entry serves both modes.
/// </summary>
internal static class VisionOps
{
    public static IEnumerable<GpuOpEntry> Entries
    {
        get
        {
            yield return new GpuOpEntry("Conv", true, CheckConvNode, BindConvByKernel);
            yield return new GpuOpEntry("ConvTranspose", true, _ => null, BindConvTranspose);
            yield return new GpuOpEntry("MaxPool", true, _ => null, (GpuClaimContext c, out GpuBinding? b) => BindPool(c, GpuPoolKind.Max, out b));
            yield return new GpuOpEntry("AveragePool", true, _ => null, (GpuClaimContext c, out GpuBinding? b) => BindPool(c, GpuPoolKind.Average, out b));
            yield return new GpuOpEntry("GlobalAveragePool", true, _ => null, (GpuClaimContext c, out GpuBinding? b) => BindGlobalPool(c, GpuPoolKind.Average, out b));
            yield return new GpuOpEntry("GlobalMaxPool", true, _ => null, (GpuClaimContext c, out GpuBinding? b) => BindGlobalPool(c, GpuPoolKind.Max, out b));
            yield return new GpuOpEntry("Resize", true, CheckResize, BindResize);
            yield return new GpuOpEntry("Concat", true, _ => null, BindConcat);
            yield return new GpuOpEntry("ConvInteger", true, ConvIntegerOps.Check, ConvIntegerOps.Bind);
        }
    }

    // ---- window geometry ---------------------------------------------------------------------------------------------------

    /// <summary>A sliding window lifted to two axes (a one-dimensional operator has height 1), with the output size the ONNX rules give.</summary>
    internal struct Window
    {
        public int KernelH, KernelW, StrideH, StrideW, DilationH, DilationW, PadT, PadL, PadB, PadR, OutH, OutW;
    }

    private static GpuAutoPad AutoPadOf(NodeInfo info) => info.GetString("auto_pad", "NOTSET") switch
    {
        "SAME_UPPER" => GpuAutoPad.SameUpper,
        "SAME_LOWER" => GpuAutoPad.SameLower,
        "VALID" => GpuAutoPad.Valid,
        _ => GpuAutoPad.NotSet,
    };

    /// <summary>
    /// Resolves kernel, strides, dilations and padding of a Conv or pool node for the input <paramref name="x"/> (N, C, spatial...). The
    /// kernel comes from <c>kernel_shape</c> or else from <paramref name="weightKernel"/>. The sizes are the attributes' (the CPU
    /// checked them when the session was made); the arithmetic is the shared one, so the output extent agrees with the planned shape.
    /// </summary>
    internal static string? ResolveWindow(NodeInfo info, Shape x, ReadOnlySpan<int> weightKernel, bool ceilMode, out Window w)
    {
        w = default;
        int r = x.Rank - 2;
        if (r is not (1 or 2))
        {
            return $"{r} spatial axes (the GPU kernels take one or two)";
        }

        var kernel = info.GetInts("kernel_shape");
        var strides = info.GetInts("strides");
        var dilations = info.GetInts("dilations");
        var pads = info.GetInts("pads");
        if ((kernel.Length != 0 && kernel.Length != r) || (strides.Length != 0 && strides.Length != r) ||
            (dilations.Length != 0 && dilations.Length != r) || (pads.Length != 0 && pads.Length != 2 * r))
        {
            return "an attribute does not have one value per spatial axis";
        }

        GpuAutoPad autoPad = AutoPadOf(info);
        Span<int> k = stackalloc int[2], s = stackalloc int[2], d = stackalloc int[2], pb = stackalloc int[2], pe = stackalloc int[2], o = stackalloc int[2], inp = stackalloc int[2];
        for (int a = 0; a < 2; a++)
        {
            int j = a - (2 - r);
            if (j < 0)
            {
                inp[a] = 1;
                k[a] = 1;
                s[a] = 1;
                d[a] = 1;
                o[a] = 1;
                continue;
            }

            inp[a] = x[2 + j];
            long kv = kernel.Length > 0 ? kernel[j] : weightKernel.Length > j ? weightKernel[j] : 0;
            if (kv < 1 || kv > int.MaxValue)
            {
                return "the kernel extent is missing or out of range";
            }

            long sv = strides.Length > 0 ? strides[j] : 1, dv = dilations.Length > 0 ? dilations[j] : 1;
            long pbv = pads.Length > 0 ? pads[j] : 0, pev = pads.Length > 0 ? pads[j + r] : 0;
            if (sv < 1 || dv < 1 || pbv < 0 || pev < 0 || sv > int.MaxValue || dv > int.MaxValue || pbv > int.MaxValue || pev > int.MaxValue)
            {
                return "a stride, dilation or padding is out of range";
            }

            k[a] = (int)kv;
            s[a] = (int)sv;
            d[a] = (int)dv;
            pb[a] = (int)pbv;
            pe[a] = (int)pev;
            int b = pb[a], e = pe[a];
            o[a] = GpuWindow.OutputExtent(autoPad, inp[a], k[a], s[a], d[a], ceilMode, ref b, ref e);
            pb[a] = b;
            pe[a] = e;
            if (o[a] < 1)
            {
                return "the window does not fit the padded input";
            }
        }

        w = new Window
        {
            KernelH = k[0], KernelW = k[1], StrideH = s[0], StrideW = s[1], DilationH = d[0], DilationW = d[1],
            PadT = pb[0], PadL = pb[1], PadB = pe[0], PadR = pe[1], OutH = o[0], OutW = o[1],
        };
        return null;
    }

    internal static bool OutputMatches(GpuClaimContext ctx, int batch, int channels, int rank, in Window w)
    {
        Shape o = ctx.OutputShape(0);
        if (o.Rank != rank || o[0] != batch || o[1] != channels)
        {
            return false;
        }

        return rank == 3 ? o[2] == w.OutW : o[2] == w.OutH && o[3] == w.OutW;
    }

    /// <summary>The element type the vision kernels run in: float32 or float16 (half storage, float32 accumulation), the output matching the input.</summary>
    private static string? VisionElement(GpuClaimContext ctx, int input, int output, out DeviceElement element)
    {
        element = DeviceElement.F32;
        DType t = ctx.InputType(input);
        if (t is not (DType.F32 or DType.F16) || ctx.OutputType(output) != t)
        {
            return $"input {input} is {t} and output {output} is {ctx.OutputType(output)}; the vision kernels take float32 or float16 with the output the same";
        }

        element = t == DType.F16 ? DeviceElement.F16 : DeviceElement.F32;
        return null;
    }

    /// <summary>
    /// Scratch for a window operator: the whole batch's worth up to this many bytes, else as many whole images as fit (at least one). A smaller
    /// buffer changes no bit of the result; the arena returns the space when the step is done.
    /// </summary>
    private const long ScratchBudgetBytes = 64L << 20;

    private static long BatchScratch(GpuClaimContext ctx, long wholeBatch, long perImage, int elementBytes, int batch)
    {
        if (wholeBatch <= 0)
        {
            return 0;
        }

        long per = Math.Max(1, perImage);
        long fit = Math.Min(ScratchBudgetBytes, ctx.MaxBindingBytes) / elementBytes;
        long images = Math.Max(1, Math.Min(batch, fit / per));
        return images * per;
    }

    // ---- Conv ----------------------------------------------------------------------------------------------------------------

    private static string? CheckConvNode(NodeInfo info) =>
        info.GetString("auto_pad", "NOTSET") is "NOTSET" or "VALID" or "SAME_UPPER" or "SAME_LOWER" ? null : "unknown auto_pad";

    /// <summary>
    /// The convolution's kernel: the CPU's own (float32, with the fused activation and residual the rewrites make), or the same kernel inside
    /// the adapter that runs it on a float16 model.
    /// </summary>
    private static ConvOp? ConvKernelOf(PlanStep step) => step.Node.Kernel switch
    {
        ConvOp c => c,
        HalfAdapterKernel { Inner: ConvOp c } => c,
        _ => null,
    };

    private static string? BindConvByKernel(GpuClaimContext ctx, out GpuBinding? binding)
    {
        if (ConvKernelOf(ctx.Step) is null && ctx.InputShape(0).Rank == 3)
        {
            if (Conv1dOps.CheckConv(ctx.Info) is { } bad)
            {
                binding = null;
                return bad;
            }

            return Conv1dOps.BindConv(ctx, out binding);
        }

        return BindConv(ctx, out binding);
    }

    private static string? BindConv(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        DeviceElement element = DeviceElement.F32;
        if (ctx.HasInput(0) && VisionElement(ctx, 0, 0, out element) is { } why)
        {
            return why;
        }

        if (ConvKernelOf(ctx.Step) is not { } conv)
        {
            return "the node runs through an adapter the GPU has no kernel for";
        }

        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            return "its weights are packed for the CPU kernels (the GPU needs the plain weights)";
        }

        Shape x = ctx.InputShape(0), wt = ctx.InputShape(1);
        if (x.Rank is not (3 or 4))
        {
            return $"a {x.Rank - 2}-D convolution (the GPU kernels take one or two spatial axes)";
        }

        DType type = ctx.InputType(0);
        if (ctx.InputType(1) != type || (ctx.HasInput(2) && ctx.InputType(2) != type))
        {
            return $"weights or bias are not {type}";
        }

        ReadOnlySpan<int> weightKernel = x.Rank == 4 ? [wt[2], wt[3]] : [wt[2]];
        if (ResolveWindow(ctx.Info, x, weightKernel, false, out Window win) is { } bad)
        {
            return bad;
        }

        int groups = conv.Group;
        var p = new ConvProblem(x[0], x[1], x.Rank == 4 ? x[2] : 1, x[x.Rank - 1], wt[0], x.Rank == 4 ? wt[2] : 1, wt[wt.Rank - 1])
        {
            StrideHeight = win.StrideH, StrideWidth = win.StrideW, DilationHeight = win.DilationH, DilationWidth = win.DilationW,
            PadTop = win.PadT, PadLeft = win.PadL, PadBottom = win.PadB, PadRight = win.PadR, Groups = groups,
        };
        if (!OutputMatches(ctx, p.Batch, p.OutChannels, x.Rank, win))
        {
            return "the output shape the plan holds is not the one the window rules give";
        }

        if (conv.HasResidual)
        {
            if (!ctx.HasInput(3) || ctx.InputType(3) != type || !ctx.InputShape(3).Equals(ctx.OutputShape(0)))
            {
                return "the residual operand does not have the output's shape (it broadcasts)";
            }
        }

        if (ActivationOf(conv.Activation, out GpuActivation act) is { } noAct)
        {
            return noAct;
        }

        p.Activation = act;
        foreach (long elements in new[] { x.Length, wt.Length, ctx.OutputShape(0).Length })
        {
            if (ctx.FitsBinding(elements, element) is { } big)
            {
                return big;
            }
        }

        // Scratch lives in the arena for the length of the step, so the unfold variants cost memory only while they run.
        bool half = element == DeviceElement.F16;
        ConvScratch need = ctx.Backend.Conv.Scratch(p, half);
        if (ctx.FitsBinding(need.ColumnElementsPerImage, element) is { } tooBig)
        {
            return tooBig;
        }

        if (ctx.FitsBinding(need.AccumulatorElementsPerImage, DeviceElement.F32) is { } tooBigAcc)
        {
            return tooBigAcc;
        }

        int columnBytes = half ? 2 : 4;
        long columns = BatchScratch(ctx, need.ColumnElements, need.ColumnElementsPerImage, columnBytes, p.Batch);
        long accumulators = BatchScratch(ctx, need.AccumulatorElements, need.AccumulatorElementsPerImage, 4, p.Batch);
        binding = new ConvBinding(ctx, element, p, conv.HasResidual, columns, accumulators);
        return null;
    }

    /// <summary>The kernel's activation for a node's fused one, or the reason there is none for it.</summary>
    private static string? ActivationOf(FusedActivation? a, out GpuActivation act)
    {
        act = new GpuActivation(GpuUnaryOp.Identity);
        switch (a)
        {
            case null:
                return null;
            case ReluActivation:
                act = new GpuActivation(GpuUnaryOp.Relu);
                return null;
            case ClipActivation c:
                act = new GpuActivation(GpuUnaryOp.Clip, c.Lo, c.Hi);
                return null;
            case FloatActivation<HardSwishUnary>:
                act = new GpuActivation(GpuUnaryOp.HardSwish);
                return null;
            case FloatActivation<HardSigmoidUnary> h:
                act = new GpuActivation(GpuUnaryOp.HardSigmoid, h.Function.Alpha, h.Function.Beta);
                return null;
            case FloatActivation<SigmoidUnary>:
                act = new GpuActivation(GpuUnaryOp.Sigmoid);
                return null;
            case FloatActivation<TanhUnary>:
                act = new GpuActivation(GpuUnaryOp.Tanh);
                return null;
            case FloatActivation<GeluUnary> g:
                act = new GpuActivation(g.Function.TanhForm ? GpuUnaryOp.GeluTanh : GpuUnaryOp.GeluErf);
                return null;
            case FloatActivation<LeakyReluUnary> l:
                act = new GpuActivation(GpuUnaryOp.LeakyRelu, l.Function.Alpha);
                return null;
            default:
                return "the fused activation is one the GPU has no kernel for";
        }
    }

    private sealed class ConvBinding : GpuBinding
    {
        private readonly ConvKernels _kernels;
        private readonly DeviceElement _element;
        private readonly ConvProblem _problem;
        private readonly long _columns, _accumulators;
        private readonly Operand _x, _w;
        private readonly Operand? _bias, _residual;
        private readonly int _outId;

        public ConvBinding(GpuClaimContext ctx, DeviceElement element, ConvProblem problem, bool residual, long columns, long accumulators)
        {
            _kernels = ctx.Backend.Conv;
            _element = element;
            _problem = problem;
            _columns = columns;
            _accumulators = accumulators;
            _x = Operand.Of(ctx, 0);
            _w = Operand.Of(ctx, 1);
            _bias = ctx.HasInput(2) ? Operand.Of(ctx, 2) : null;
            _residual = residual ? Operand.Of(ctx, 3) : null;
            _outId = ctx.OutputId(0);
            Flops = problem.Flops;
        }

        public override DeviceElement OutputElement(int output) => _element;

        // Float32 convolutions keep columns (when the variant unfolds) in float32 scratch; float16 ones keep halves, and float32 accumulators.
        public override long ScratchF32Elements => _element == DeviceElement.F16 ? _accumulators : _columns;

        public override long ScratchF16Elements => _element == DeviceElement.F16 ? _columns : 0;

        public override void Encode(GpuRunContext ctx)
        {
            ConvProblem p = _problem;
            if (_element == DeviceElement.F32)
            {
                GpuBuffer<float> x = _x.F32(ctx, out int ox);
                GpuBuffer<float> w = _w.F32(ctx, out int ow);
                GpuBuffer<float>? bias = null, res = null;
                if (_bias is { } b)
                {
                    bias = b.F32(ctx, out int ob);
                    p.OffsetBias = ob;
                }

                if (_residual is { } r)
                {
                    res = r.F32(ctx, out int or);
                    p.OffsetResidual = or;
                }

                GpuBuffer<float> y = ctx.Buffer<float>(_outId, out int oy);
                p.OffsetX = ox;
                p.OffsetW = ow;
                p.OffsetY = oy;
                GpuBuffer<float>? columns = null;
                int oc = 0;
                if (_columns > 0)
                {
                    columns = ctx.Scratch(this, out oc);
                }

                _kernels.Encode(ctx.Encoder, in p, x, w, bias, res, y, columns, oc);
            }
            else
            {
                GpuBuffer<Half> x = _x.F16(ctx, out int ox);
                GpuBuffer<Half> w = _w.F16(ctx, out int ow);
                GpuBuffer<Half>? bias = null, res = null;
                if (_bias is { } b)
                {
                    bias = b.F16(ctx, out int ob);
                    p.OffsetBias = ob;
                }

                if (_residual is { } r)
                {
                    res = r.F16(ctx, out int or);
                    p.OffsetResidual = or;
                }

                GpuBuffer<Half> y = ctx.F16(_outId, out int oy);
                p.OffsetX = ox;
                p.OffsetW = ow;
                p.OffsetY = oy;
                GpuBuffer<Half>? columns = null;
                GpuBuffer<float>? accumulators = null;
                int oc = 0, oa = 0;
                if (_columns > 0)
                {
                    columns = ctx.ScratchF16(this, out oc);
                }

                if (_accumulators > 0)
                {
                    accumulators = ctx.Scratch(this, out oa);
                }

                _kernels.Encode(ctx.Encoder, in p, x, w, bias, res, y, columns, accumulators, oc, oa);
            }
        }
    }

    // ---- ConvTranspose ---------------------------------------------------------------------------------------------------------

    private static string? BindConvTranspose(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            return "needs X and W";
        }

        if (VisionElement(ctx, 0, 0, out DeviceElement element) is { } why)
        {
            return why;
        }

        Shape x = ctx.InputShape(0), wt = ctx.InputShape(1);
        if (x.Rank is not (3 or 4) || wt.Rank != x.Rank)
        {
            return $"a {x.Rank - 2}-D transposed convolution (the GPU kernels take one or two spatial axes)";
        }

        if (ctx.InputType(1) != ctx.InputType(0) || (ctx.HasInput(2) && ctx.InputType(2) != ctx.InputType(0)))
        {
            return $"weights or bias are not {ctx.InputType(0)}";
        }

        NodeInfo info = ctx.Info;
        if (AutoPadOf(info) != GpuAutoPad.NotSet || info.Has("output_shape"))
        {
            return "auto_pad and output_shape are not mapped (the explicit pads and output_padding are)";
        }

        int r = x.Rank - 2;
        var strides = info.GetInts("strides");
        var dilations = info.GetInts("dilations");
        var pads = info.GetInts("pads");
        var outPad = info.GetInts("output_padding");
        if ((strides.Length != 0 && strides.Length != r) || (dilations.Length != 0 && dilations.Length != r) ||
            (pads.Length != 0 && pads.Length != 2 * r) || (outPad.Length != 0 && outPad.Length != r))
        {
            return "an attribute does not have one value per spatial axis";
        }

        long groups = info.GetInt("group", 1);
        if (groups < 1 || groups > int.MaxValue)
        {
            return "group is out of range";
        }

        // Lift to two axes, the one-dimensional case being height 1.
        long[] st = [1, 1], di = [1, 1], pb = [0, 0], pe = [0, 0], op = [0, 0];
        for (int j = 0; j < r; j++)
        {
            int a = 2 - r + j;
            st[a] = strides.Length > 0 ? strides[j] : 1;
            di[a] = dilations.Length > 0 ? dilations[j] : 1;
            pb[a] = pads.Length > 0 ? pads[j] : 0;
            pe[a] = pads.Length > 0 ? pads[j + r] : 0;
            op[a] = outPad.Length > 0 ? outPad[j] : 0;
        }

        foreach (long v in new[] { st[0], st[1], di[0], di[1], pb[0], pb[1], pe[0], pe[1], op[0], op[1] })
        {
            if (v < 0 || v > int.MaxValue)
            {
                return "a stride, dilation or padding is out of range";
            }
        }

        var p = new ConvTransposeProblem(x[0], x[1], r == 2 ? x[2] : 1, x[x.Rank - 1], wt[1] * (int)groups, r == 2 ? wt[2] : 1, wt[wt.Rank - 1])
        {
            StrideHeight = (int)st[0], StrideWidth = (int)st[1], DilationHeight = (int)di[0], DilationWidth = (int)di[1],
            PadTop = (int)pb[0], PadLeft = (int)pb[1], PadBottom = (int)pe[0], PadRight = (int)pe[1],
            OutputPaddingHeight = (int)op[0], OutputPaddingWidth = (int)op[1], Groups = (int)groups,
        };
        int oh = p.OutHeight, ow = p.OutWidth;
        Shape o = ctx.OutputShape(0);
        bool same = o.Rank == x.Rank && o[0] == p.Batch && o[1] == p.OutChannels && o[o.Rank - 1] == ow && (r == 1 || o[2] == oh);
        if (oh < 1 || ow < 1 || !same)
        {
            return "the output shape the plan holds is not the one the window rules give";
        }

        foreach (long elements in new[] { x.Length, wt.Length, o.Length })
        {
            if (ctx.FitsBinding(elements, element) is { } big)
            {
                return big;
            }
        }

        ConvScratch need = ConvKernels.Scratch(p);
        if (ctx.FitsBinding(need.ColumnElementsPerImage, DeviceElement.F32) is { } tooBig)
        {
            return tooBig;
        }

        binding = new ConvTransposeBinding(ctx, element, p, BatchScratch(ctx, need.ColumnElements, need.ColumnElementsPerImage, 4, p.Batch));
        return null;
    }

    private sealed class ConvTransposeBinding : GpuBinding
    {
        private readonly ConvKernels _kernels;
        private readonly DeviceElement _element;
        private readonly ConvTransposeProblem _problem;
        private readonly long _columns;
        private readonly Operand _x, _w;
        private readonly Operand? _bias;
        private readonly int _outId;

        public ConvTransposeBinding(GpuClaimContext ctx, DeviceElement element, ConvTransposeProblem problem, long columns)
        {
            _kernels = ctx.Backend.Conv;
            _element = element;
            _problem = problem;
            _columns = columns;
            _x = Operand.Of(ctx, 0);
            _w = Operand.Of(ctx, 1);
            _bias = ctx.HasInput(2) ? Operand.Of(ctx, 2) : null;
            _outId = ctx.OutputId(0);
            Flops = problem.Flops;
        }

        public override DeviceElement OutputElement(int output) => _element;

        // The matrix product's result is float32 for both element types.
        public override long ScratchF32Elements => _columns;

        public override void Encode(GpuRunContext ctx)
        {
            ConvTransposeProblem p = _problem;
            GpuBuffer<float> columns = ctx.Scratch(this, out int oc);
            if (_element == DeviceElement.F32)
            {
                GpuBuffer<float> x = _x.F32(ctx, out int ox);
                GpuBuffer<float> w = _w.F32(ctx, out int ow);
                GpuBuffer<float>? bias = null;
                if (_bias is { } b)
                {
                    bias = b.F32(ctx, out int ob);
                    p.OffsetBias = ob;
                }

                GpuBuffer<float> y = ctx.Buffer<float>(_outId, out int oy);
                p.OffsetX = ox;
                p.OffsetW = ow;
                p.OffsetY = oy;
                _kernels.EncodeTranspose(ctx.Encoder, in p, x, w, bias, y, columns, oc);
            }
            else
            {
                GpuBuffer<Half> x = _x.F16(ctx, out int ox);
                GpuBuffer<Half> w = _w.F16(ctx, out int ow);
                GpuBuffer<Half>? bias = null;
                if (_bias is { } b)
                {
                    bias = b.F16(ctx, out int ob);
                    p.OffsetBias = ob;
                }

                GpuBuffer<Half> y = ctx.F16(_outId, out int oy);
                p.OffsetX = ox;
                p.OffsetW = ow;
                p.OffsetY = oy;
                _kernels.EncodeTranspose(ctx.Encoder, in p, x, w, bias, y, columns, oc);
            }
        }
    }

    // ---- pooling ---------------------------------------------------------------------------------------------------------------

    private static string? BindPool(GpuClaimContext ctx, GpuPoolKind kind, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0))
        {
            return "needs an input";
        }

        if (VisionElement(ctx, 0, 0, out DeviceElement element) is { } why)
        {
            return why;
        }

        int[] outs = ctx.Step.Node.Outputs;
        if (outs.Length > 1 && outs[1] >= 0)
        {
            return "its indices output is read, and the GPU pooling returns only the pooled values";
        }

        NodeInfo info = ctx.Info;
        Shape x = ctx.InputShape(0);
        if (x.Rank is not (3 or 4))
        {
            return $"a {x.Rank - 2}-D pool (the GPU kernels take one or two spatial axes)";
        }

        if (info.OpType == "AveragePool" && info.GetInts("dilations").Length > 0 && info.GetInts("dilations").AsSpan().IndexOfAnyExcept(1L) >= 0)
        {
            return "AveragePool with dilation";
        }

        if (ResolveWindow(info, x, default, info.GetInt("ceil_mode", 0) != 0, out Window win) is { } bad)
        {
            return bad;
        }

        var p = new PoolProblem(x[0], x[1], x.Rank == 4 ? x[2] : 1, x[x.Rank - 1], win.KernelH, win.KernelW)
        {
            StrideHeight = win.StrideH, StrideWidth = win.StrideW, DilationHeight = win.DilationH, DilationWidth = win.DilationW,
            PadTop = win.PadT, PadLeft = win.PadL, PadBottom = win.PadB, PadRight = win.PadR, CeilMode = info.GetInt("ceil_mode", 0) != 0,
        };
        if (!OutputMatches(ctx, p.Batch, p.Channels, x.Rank, win) || p.OutHeight != win.OutH || p.OutWidth != win.OutW)
        {
            return "the output shape the plan holds is not the one the window rules give";
        }

        if (kind == GpuPoolKind.Average && info.GetInt("count_include_pad", 0) != 0)
        {
            kind = GpuPoolKind.AverageIncludePad;
        }

        foreach (long elements in new[] { x.Length, ctx.OutputShape(0).Length })
        {
            if (ctx.FitsBinding(elements, element) is { } big)
            {
                return big;
            }
        }

        binding = new PoolBinding(ctx, element, p, kind);
        return null;
    }

    private sealed class PoolBinding : GpuBinding
    {
        private readonly PoolKernels _kernels;
        private readonly DeviceElement _element;
        private readonly PoolProblem _problem;
        private readonly GpuPoolKind _kind;
        private readonly Operand _x;
        private readonly int _outId;

        public PoolBinding(GpuClaimContext ctx, DeviceElement element, PoolProblem problem, GpuPoolKind kind)
        {
            _kernels = ctx.Backend.Pool;
            _element = element;
            _problem = problem;
            _kind = kind;
            _x = Operand.Of(ctx, 0);
            _outId = ctx.OutputId(0);
            Flops = (double)ctx.OutputShape(0).Length * problem.KernelHeight * problem.KernelWidth;
        }

        public override DeviceElement OutputElement(int output) => _element;

        public override void Encode(GpuRunContext ctx)
        {
            PoolProblem p = _problem;
            if (_element == DeviceElement.F32)
            {
                GpuBuffer<float> x = _x.F32(ctx, out int ox);
                GpuBuffer<float> y = ctx.Buffer<float>(_outId, out int oy);
                p.OffsetX = ox;
                p.OffsetY = oy;
                _kernels.Encode(ctx.Encoder, in p, _kind, x, y);
            }
            else
            {
                GpuBuffer<Half> x = _x.F16(ctx, out int ox);
                GpuBuffer<Half> y = ctx.F16(_outId, out int oy);
                p.OffsetX = ox;
                p.OffsetY = oy;
                _kernels.Encode(ctx.Encoder, in p, _kind, x, y);
            }
        }
    }

    private static string? BindGlobalPool(GpuClaimContext ctx, GpuPoolKind kind, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0))
        {
            return "needs an input";
        }

        if (VisionElement(ctx, 0, 0, out DeviceElement element) is { } why)
        {
            return why;
        }

        Shape x = ctx.InputShape(0);
        if (x.Rank < 3)
        {
            return "the input has no spatial axes";
        }

        long planes = (long)x[0] * x[1], size = 1;
        for (int i = 2; i < x.Rank; i++)
        {
            size *= x[i];
        }

        if (planes < 1 || size < 1 || planes >= int.MaxValue || size >= int.MaxValue || ctx.OutputShape(0).Length != planes)
        {
            return "the input is empty or the planned output does not match";
        }

        if (ctx.FitsBinding(x.Length, element) is { } big)
        {
            return big;
        }

        binding = new GlobalPoolBinding(ctx, element, kind, (int)planes, (int)size);
        return null;
    }

    private sealed class GlobalPoolBinding : GpuBinding
    {
        private readonly PoolKernels _kernels;
        private readonly DeviceElement _element;
        private readonly GpuPoolKind _kind;
        private readonly int _planes, _size;
        private readonly Operand _x;
        private readonly int _outId;

        public GlobalPoolBinding(GpuClaimContext ctx, DeviceElement element, GpuPoolKind kind, int planes, int size)
        {
            _kernels = ctx.Backend.Pool;
            _element = element;
            _kind = kind;
            _planes = planes;
            _size = size;
            _x = Operand.Of(ctx, 0);
            _outId = ctx.OutputId(0);
            Flops = (double)planes * size;
        }

        public override DeviceElement OutputElement(int output) => _element;

        public override void Encode(GpuRunContext ctx)
        {
            if (_element == DeviceElement.F32)
            {
                GpuBuffer<float> x = _x.F32(ctx, out int ox);
                GpuBuffer<float> y = ctx.Buffer<float>(_outId, out int oy);
                _kernels.EncodeGlobal(ctx.Encoder, _kind, _planes, _size, x, ox, y, oy);
            }
            else
            {
                GpuBuffer<Half> x = _x.F16(ctx, out int ox);
                GpuBuffer<Half> y = ctx.F16(_outId, out int oy);
                _kernels.EncodeGlobal(ctx.Encoder, _kind, _planes, _size, x, ox, y, oy);
            }
        }
    }

    // ---- Resize ----------------------------------------------------------------------------------------------------------------

    // The deprecated Upsample and opset 10 Resize have other inputs and fixed rules; the opset 11 form covers the models.
    private static string? CheckResize(NodeInfo info)
    {
        if (info.OpType != "Resize" || info.Opset < 11)
        {
            return $"{info.OpType} before opset 11 (the GPU kernel takes the attribute form)";
        }

        string mode = info.GetString("mode", "nearest");
        if (mode is not ("nearest" or "linear" or "bilinear"))
        {
            return $"{mode} interpolation (the GPU has nearest and linear)";
        }

        if (info.GetString("coordinate_transformation_mode", "half_pixel") == "tf_crop_and_resize")
        {
            return "tf_crop_and_resize coordinates";
        }

        if (info.GetInt("exclude_outside", 0) != 0)
        {
            return "exclude_outside";
        }

        if (info.GetInt("antialias", 0) != 0 && mode != "nearest")
        {
            return "antialias";
        }

        if (info.GetString("keep_aspect_ratio_policy", "stretch") != "stretch" || info.GetInts("axes").Length > 0)
        {
            return "keep_aspect_ratio_policy or axes";
        }

        return null;
    }

    private static string? BindResize(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0))
        {
            return "needs an input";
        }

        if (VisionElement(ctx, 0, 0, out DeviceElement element) is { } why)
        {
            return why;
        }

        Shape x = ctx.InputShape(0), o = ctx.OutputShape(0);
        if (x.Rank is not (3 or 4) || o.Rank != x.Rank || o[0] != x[0] || o[1] != x[1])
        {
            return "the GPU resizes the spatial axes of a 3-D or 4-D tensor, with batch and channels unchanged";
        }

        NodeInfo info = ctx.Info;
        GpuResizeCoordinate coord = info.GetString("coordinate_transformation_mode", "half_pixel") switch
        {
            "half_pixel_symmetric" => GpuResizeCoordinate.HalfPixelSymmetric,
            "pytorch_half_pixel" => GpuResizeCoordinate.PytorchHalfPixel,
            "align_corners" => GpuResizeCoordinate.AlignCorners,
            "asymmetric" => GpuResizeCoordinate.Asymmetric,
            _ => GpuResizeCoordinate.HalfPixel,
        };
        GpuResizeNearest nearest = info.GetString("nearest_mode", "round_prefer_floor") switch
        {
            "round_prefer_ceil" => GpuResizeNearest.RoundPreferCeil,
            "floor" => GpuResizeNearest.Floor,
            "ceil" => GpuResizeNearest.Ceil,
            _ => GpuResizeNearest.RoundPreferFloor,
        };
        bool linear = info.GetString("mode", "nearest") != "nearest";
        int r = x.Rank - 2;
        float scaleH = 0f, scaleW = 0f;
        bool haveScales = ctx.InputCount > 2 && ctx.HasInput(2) && ctx.InputShape(2).Length > 0;
        if (haveScales)
        {
            if (!ctx.InputIsConst(2) || ctx.InputType(2) != DType.F32)
            {
                return "the scales are not a constant float32 tensor";
            }

            ReadOnlySpan<float> sc = ctx.InputConst(2).Data<float>();
            if (sc.Length != x.Rank || sc[0] != 1f || sc[1] != 1f)
            {
                return "the scales change batch or channels";
            }

            scaleH = r == 2 ? sc[2] : 0f;
            scaleW = sc[x.Rank - 1];
        }

        var problem = new ResizeProblem(x[0] * x[1], r == 2 ? x[2] : 1, x[x.Rank - 1], r == 2 ? o[2] : 1, o[o.Rank - 1], linear ? GpuResizeMode.Linear : GpuResizeMode.Nearest)
        {
            Coordinate = coord, Nearest = nearest, ScaleHeight = scaleH, ScaleWidth = scaleW,
        };
        foreach (long elements in new[] { x.Length, o.Length })
        {
            if (ctx.FitsBinding(elements, element) is { } big)
            {
                return big;
            }
        }

        ResizePlan plan = ctx.Backend.Resize.CreatePlan(problem);
        ctx.OwnForPlan(new DeviceConstant(plan, (linear ? 16L : 8L) * (o[o.Rank - 1] + (r == 2 ? o[2] : 1)), 0));
        binding = new ResizeBinding(ctx, element, plan);
        return null;
    }

    private sealed class ResizeBinding : GpuBinding
    {
        private readonly ResizeKernels _kernels;
        private readonly DeviceElement _element;
        private readonly ResizePlan _plan;
        private readonly Operand _x;
        private readonly int _outId;

        public ResizeBinding(GpuClaimContext ctx, DeviceElement element, ResizePlan plan)
        {
            _kernels = ctx.Backend.Resize;
            _element = element;
            _plan = plan;
            _x = Operand.Of(ctx, 0);
            _outId = ctx.OutputId(0);
            Flops = (double)ctx.OutputShape(0).Length * 4;
        }

        public override DeviceElement OutputElement(int output) => _element;

        public override void Encode(GpuRunContext ctx)
        {
            if (_element == DeviceElement.F32)
            {
                GpuBuffer<float> x = _x.F32(ctx, out int ox);
                GpuBuffer<float> y = ctx.Buffer<float>(_outId, out int oy);
                _kernels.Encode(ctx.Encoder, _plan, x, ox, y, oy);
            }
            else
            {
                GpuBuffer<Half> x = _x.F16(ctx, out int ox);
                GpuBuffer<Half> y = ctx.F16(_outId, out int oy);
                _kernels.Encode(ctx.Encoder, _plan, x, ox, y, oy);
            }
        }
    }

    // ---- Concat ----------------------------------------------------------------------------------------------------------------

    private static string? BindConcat(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        int count = ctx.InputCount;
        if (count < 1)
        {
            return "needs inputs";
        }

        Shape o = ctx.OutputShape(0);
        if (ctx.OutputType(0) is not (DType.F32 or DType.F16) || o.Rank < 1)
        {
            return "the GPU concatenates float32 and float16 tensors";
        }

        DType type = ctx.OutputType(0);
        DeviceElement element = type == DType.F16 ? DeviceElement.F16 : DeviceElement.F32;

        int axis = (int)ctx.Info.GetInt("axis", 0);
        if (axis < -o.Rank || axis >= o.Rank)
        {
            return $"axis {axis} is outside the rank {o.Rank}";
        }

        axis = axis < 0 ? axis + o.Rank : axis;
        var lengths = new int[count];
        long sum = 0;
        for (int i = 0; i < count; i++)
        {
            if (!ctx.HasInput(i) || ctx.InputType(i) != type)
            {
                return $"input {i} is not a {type} tensor";
            }

            Shape s = ctx.InputShape(i);
            if (s.Rank != o.Rank)
            {
                return "the inputs differ in rank";
            }

            lengths[i] = s[axis];
            sum += lengths[i];
        }

        if (sum != o[axis] || o.Length == 0 || o.Length >= int.MaxValue)
        {
            return "the inputs do not add up to the planned output or it is empty";
        }

        if (ctx.FitsBinding(o.Length, element) is { } big)
        {
            return big;
        }

        binding = new ConcatBinding(ctx, element, count, lengths, axis);
        return null;
    }

    private sealed class ConcatBinding : GpuBinding
    {
        private readonly LayoutKernels _kernels;
        private readonly DeviceElement _element;
        private readonly Operand[] _inputs;
        private readonly int[] _lengths;
        private readonly int[] _shape;
        private readonly int _axis;
        private readonly int _outId;
        private readonly GpuConcatInput<float>[] _parts;
        private readonly GpuConcatInput<Half>[] _partsHalf;

        public ConcatBinding(GpuClaimContext ctx, DeviceElement element, int count, int[] lengths, int axis)
        {
            _kernels = ctx.Backend.Layout;
            _element = element;
            _inputs = new Operand[count];
            for (int i = 0; i < count; i++)
            {
                _inputs[i] = Operand.Of(ctx, i);
            }

            _lengths = lengths;
            _axis = axis;
            _parts = new GpuConcatInput<float>[count];
            _partsHalf = new GpuConcatInput<Half>[count];
            Shape o = ctx.OutputShape(0);
            _shape = new int[o.Rank];
            for (int i = 0; i < _shape.Length; i++)
            {
                _shape[i] = o[i];
            }

            _outId = ctx.OutputId(0);
            Flops = o.Length;
        }

        public override DeviceElement OutputElement(int output) => _element;

        // One dispatch per four inputs, whatever the batch: the kernel reads each input where it lies, in the output's slab or not.
        public override void Encode(GpuRunContext ctx)
        {
            if (_element == DeviceElement.F32)
            {
                GpuBuffer<float> y = ctx.Buffer<float>(_outId, out int oy);
                for (int i = 0; i < _inputs.Length; i++)
                {
                    GpuBuffer<float> x = _inputs[i].F32(ctx, out int ox);
                    _parts[i] = new GpuConcatInput<float>(x, ox, _lengths[i]);
                }

                _kernels.EncodeConcat<float>(ctx.Encoder, _parts, _shape, _axis, y, oy);
            }
            else
            {
                GpuBuffer<Half> y = ctx.F16(_outId, out int oy);
                for (int i = 0; i < _inputs.Length; i++)
                {
                    GpuBuffer<Half> x = _inputs[i].F16(ctx, out int ox);
                    _partsHalf[i] = new GpuConcatInput<Half>(x, ox, _lengths[i]);
                }

                _kernels.EncodeConcat<Half>(ctx.Encoder, _partsHalf, _shape, _axis, y, oy);
            }
        }
    }
}
