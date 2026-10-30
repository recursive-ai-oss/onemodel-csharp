using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Signal;

/// <summary>
/// The spectral operators DFT (opset 17, 20) and STFT (opset 17) over binary32 or binary64 data. Both compute the textbook sum
/// X[k] = sum_t x[t] exp(-2 pi i k t / n) in binary64 with a twiddle table indexed by (k t) mod n, then round once to the output type;
/// the inverse uses the conjugate twiddles and divides by n. The cost is O(n^2) per transform: right for the frame sizes of
/// feature extraction and for coverage, and a radix-2 path would be the place to optimise. Not exact: the table uses the
/// platform's sin and cos.
/// </summary>
internal static class Spectral
{
    /// <summary>One transform of <paramref name="nIn"/> samples (zero-padded or cut to <paramref name="n"/>) into <paramref name="bins"/> outputs.</summary>
    public static void Transform(ReadOnlySpan<double> re, ReadOnlySpan<double> im, int nIn, int n, bool inverse, int bins, Span<double> outRe, Span<double> outIm, Span<double> table)
    {
        // table[2 j] = cos(2 pi j / n), table[2 j + 1] = sin(2 pi j / n) for j in [0, n).
        for (int j = 0; j < n; j++)
        {
            double a = 2 * Math.PI * j / n;
            table[2 * j] = Math.Cos(a);
            table[(2 * j) + 1] = Math.Sin(a);
        }

        int limit = Math.Min(nIn, n);
        double sign = inverse ? 1 : -1, scale = inverse ? 1.0 / n : 1.0;
        for (int k = 0; k < bins; k++)
        {
            double sr = 0, si = 0;
            for (int t = 0; t < limit; t++)
            {
                int j = (int)((long)k * t % n);
                double c = table[2 * j], s = sign * table[(2 * j) + 1];
                sr += (re[t] * c) - (im[t] * s);
                si += (re[t] * s) + (im[t] * c);
            }

            outRe[k] = sr * scale;
            outIm[k] = si * scale;
        }
    }
}

/// <summary>DFT: the discrete Fourier transform along one axis of a (..., signal, ..., 1 or 2) tensor whose last dimension holds real and imaginary parts.</summary>
internal sealed class DftOp : OpKernel
{
    public static readonly OpDef Def = new("DFT", 17, int.MaxValue, static n => new DftOp(n));

    private readonly bool _inverse, _onesided, _axisInput;
    private readonly long _axisAttr;

    private DftOp(NodeInfo n)
    {
        _inverse = n.GetInt("inverse", 0) != 0;
        _onesided = n.GetInt("onesided", 0) != 0;
        _axisInput = n.Opset >= 20;
        _axisAttr = n.GetInt("axis", 1);
    }

    private struct Plan
    {
        public int Axis, N, NIn, Bins, OutLast, InLast;
    }

    private Plan MakePlan(Shape x, long axisValue, bool hasLength, long lengthValue, InferContext? ctx)
    {
        SessionException Fail(string m) => ctx?.Fail(m) ?? new SessionException(SessionErrorKind.InvalidInput, "DFT: " + m);
        if (x.Rank < 2 || x[x.Rank - 1] is not (1 or 2))
        {
            throw Fail($"the input {x} must end in a dimension of 1 (real) or 2 (complex)");
        }

        long a = axisValue;
        if (a < -(x.Rank - 1) || a >= x.Rank - 1)
        {
            throw Fail($"axis {a} is not one of the signal dimensions of {x}");
        }

        var p = new Plan { Axis = (int)(a < 0 ? a + x.Rank : a), InLast = x[x.Rank - 1] };
        p.NIn = x[p.Axis];
        bool halfToReal = _inverse && _onesided;
        if (halfToReal && p.InLast != 2)
        {
            throw Fail("an inverse one-sided transform needs a complex spectrum");
        }

        p.N = hasLength ? (int)lengthValue : halfToReal ? 2 * (p.NIn - 1) : p.NIn;
        if (p.N < 1)
        {
            throw Fail("dft_length must be at least 1");
        }

        p.Bins = !_inverse && _onesided ? (p.N / 2) + 1 : p.N;
        p.OutLast = halfToReal ? 1 : 2;
        return p;
    }

    public override bool SupportsExact => false;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects the signal");
        }

        if (ctx.InputType(0) is not (DType.F32 or DType.F64))
        {
            throw ctx.Unsupported($"element type {ctx.InputType(0)} is not supported yet");
        }

        bool hasLength = ctx.HasInput(1);
        long length = 0, axis = _axisAttr;
        if (hasLength)
        {
            OpUtil.RequireIntegers(ctx, 1, "dft_length", allowScalar: true);
            if (!ctx.TryGetInputValue(1, out Tensor l))
            {
                ctx.RequireRunTimeValues();
                return;
            }

            length = OpUtil.First(l);
        }

        if (_axisInput && ctx.HasInput(2))
        {
            OpUtil.RequireIntegers(ctx, 2, "axis", allowScalar: true);
            if (!ctx.TryGetInputValue(2, out Tensor a))
            {
                ctx.RequireRunTimeValues();
                return;
            }

            axis = OpUtil.First(a);
        }
        else if (_axisInput)
        {
            axis = 1;
        }

        Shape x = ctx.InputShape(0);
        Plan p = MakePlan(x, axis, hasLength, length, ctx);
        Span<long> dims = stackalloc long[Shape.MaxRank];
        for (int i = 0; i < x.Rank; i++)
        {
            dims[i] = x[i];
        }

        dims[p.Axis] = p.Bins;
        dims[x.Rank - 1] = p.OutLast;
        ctx.SetOutput(0, ctx.InputType(0), OpUtil.MakeGrownShape(ctx, dims.Slice(0, x.Rank), ctx.InputType(0)));
    }

    public override void Run(in OpRun run)
    {
        Tensor xt = run.Inputs[0], yt = run.Outputs[0];
        bool hasLength = run.Inputs.Length > 1 && !run.Inputs[1].IsEmpty;
        long length = hasLength ? OpUtil.First(run.Inputs[1]) : 0;
        long axis = _axisInput ? (run.Inputs.Length > 2 && !run.Inputs[2].IsEmpty ? OpUtil.First(run.Inputs[2]) : 1) : _axisAttr;
        Plan p = MakePlan(xt.Shape, axis, hasLength, length, null);
        ReadOnlySpan<int> d = xt.Shape.Dimensions;
        int outer = OpUtil.Product(d, 0, p.Axis), inner = OpUtil.Product(d, p.Axis + 1, d.Length - 1);
        bool half = _inverse && _onesided;
        int nIn = half ? p.N : p.NIn;
        double[] buf = new double[(2 * nIn) + (2 * p.Bins) + (2 * p.N)];
        Span<double> re = buf.AsSpan(0, nIn), im = buf.AsSpan(nIn, nIn);
        Span<double> ore = buf.AsSpan(2 * nIn, p.Bins), oim = buf.AsSpan((2 * nIn) + p.Bins, p.Bins);
        Span<double> table = buf.AsSpan((2 * nIn) + (2 * p.Bins), 2 * p.N);
        for (int o = 0; o < outer; o++)
        {
            for (int i = 0; i < inner; i++)
            {
                // Gather one signal (the half spectrum is mirrored out to n points first).
                re.Clear();
                im.Clear();
                for (int t = 0; t < p.NIn; t++)
                {
                    long at = (((((long)o * p.NIn) + t) * inner) + i) * p.InLast;
                    double r = Read(xt, at), m = p.InLast == 2 ? Read(xt, at + 1) : 0;
                    if (half)
                    {
                        if (t < nIn)
                        {
                            re[t] = r;
                            im[t] = m;
                        }

                        if (t > 0 && p.N - t > t && p.N - t < nIn && t < p.NIn)
                        {
                            re[p.N - t] = r;
                            im[p.N - t] = -m;
                        }
                    }
                    else if (t < nIn)
                    {
                        re[t] = r;
                        im[t] = m;
                    }
                }

                Spectral.Transform(re, im, nIn, p.N, _inverse, p.Bins, ore, oim, table);
                for (int k = 0; k < p.Bins; k++)
                {
                    long at = (((((long)o * p.Bins) + k) * inner) + i) * p.OutLast;
                    Write(yt, at, ore[k]);
                    if (p.OutLast == 2)
                    {
                        Write(yt, at + 1, oim[k]);
                    }
                }
            }
        }
    }

    internal static double Read(Tensor t, long at) => t.DType == DType.F32 ? t.Data<float>()[(int)at] : t.Data<double>()[(int)at];

    internal static void Write(Tensor t, long at, double v)
    {
        if (t.DType == DType.F32)
        {
            t.MutableData<float>()[(int)at] = (float)v;
        }
        else
        {
            t.MutableData<double>()[(int)at] = v;
        }
    }
}

/// <summary>STFT: windowed DFTs of consecutive frames of a (batch, length, 1 or 2) signal; the output is (batch, frames, bins, 2).</summary>
internal sealed class StftOp : OpKernel
{
    public static readonly OpDef Def = new("STFT", 17, int.MaxValue, static n => new StftOp(n));

    private readonly bool _onesided;

    private StftOp(NodeInfo n)
    {
        _onesided = n.GetInt("onesided", 1) != 0;
    }

    public override bool SupportsExact => false;

    private (int FrameLength, int Step, int Frames, int Bins) Geometry(Shape x, long step, int windowLength, bool hasFrameLength, long frameLength, InferContext? ctx)
    {
        SessionException Fail(string m) => ctx?.Fail(m) ?? new SessionException(SessionErrorKind.InvalidInput, "STFT: " + m);
        if (x.Rank != 3 || x[2] is not (1 or 2))
        {
            throw Fail($"the signal {x} must be (batch, length, 1 or 2)");
        }

        long fl = windowLength > 0 ? windowLength : hasFrameLength ? frameLength : x[1];
        if (hasFrameLength && windowLength > 0 && frameLength != windowLength)
        {
            throw Fail("frame_length and the window length differ");
        }

        if (step < 1 || fl < 1 || fl > x[1])
        {
            throw Fail($"frame_step {step} and frame length {fl} do not fit a signal of {x[1]} samples");
        }

        int frames = (int)(((x[1] - fl) / step) + 1);
        int bins = _onesided && x[2] == 1 ? (int)(fl / 2) + 1 : (int)fl;
        return ((int)fl, (int)step, frames, bins);
    }

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 2 || !ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects the signal and frame_step");
        }

        if (ctx.InputType(0) is not (DType.F32 or DType.F64))
        {
            throw ctx.Unsupported($"element type {ctx.InputType(0)} is not supported yet");
        }

        if (!ctx.TryGetInputValue(1, out Tensor step) || (ctx.HasInput(3) && !ctx.TryGetInputValue(3, out _)))
        {
            ctx.RequireRunTimeValues();
            return;
        }

        int wl = ctx.HasInput(2) ? ctx.InputShape(2).Length : 0;
        long frameLength = 0;
        if (ctx.HasInput(3))
        {
            _ = ctx.TryGetInputValue(3, out Tensor f);
            frameLength = OpUtil.First(f);
        }

        var g = Geometry(ctx.InputShape(0), OpUtil.First(step), wl, ctx.HasInput(3), frameLength, ctx);
        ctx.SetOutput(0, ctx.InputType(0), OpUtil.MakeGrownShape(ctx, [ctx.InputShape(0)[0], g.Frames, g.Bins, 2], ctx.InputType(0)));
    }

    public override void Run(in OpRun run)
    {
        Tensor xt = run.Inputs[0], yt = run.Outputs[0];
        bool hasWindow = run.Inputs.Length > 2 && !run.Inputs[2].IsEmpty;
        bool hasFl = run.Inputs.Length > 3 && !run.Inputs[3].IsEmpty;
        var g = Geometry(xt.Shape, OpUtil.First(run.Inputs[1]), hasWindow ? run.Inputs[2].Length : 0, hasFl, hasFl ? OpUtil.First(run.Inputs[3]) : 0, null);
        int batch = xt.Shape[0], len = xt.Shape[1], inLast = xt.Shape[2], fl = g.FrameLength;
        double[] buf = new double[(2 * fl) + (2 * g.Bins) + (2 * fl)];
        Span<double> re = buf.AsSpan(0, fl), im = buf.AsSpan(fl, fl), ore = buf.AsSpan(2 * fl, g.Bins), oim = buf.AsSpan((2 * fl) + g.Bins, g.Bins);
        Span<double> table = buf.AsSpan((2 * fl) + (2 * g.Bins), 2 * fl);
        for (int b = 0; b < batch; b++)
        {
            for (int f = 0; f < g.Frames; f++)
            {
                for (int t = 0; t < fl; t++)
                {
                    long at = (((long)b * len) + (f * g.Step) + t) * inLast;
                    double w = hasWindow ? DftOp.Read(run.Inputs[2], t) : 1;
                    re[t] = DftOp.Read(xt, at) * w;
                    im[t] = inLast == 2 ? DftOp.Read(xt, at + 1) * w : 0;
                }

                Spectral.Transform(re, im, fl, fl, false, g.Bins, ore, oim, table);
                for (int k = 0; k < g.Bins; k++)
                {
                    long at = ((((long)b * g.Frames) + f) * g.Bins + k) * 2;
                    DftOp.Write(yt, at, ore[k]);
                    DftOp.Write(yt, at + 1, oim[k]);
                }
            }
        }
    }
}

/// <summary>HannWindow, HammingWindow and BlackmanWindow: a window of the given size in the type output_datatype (float by default); periodic (default 1) uses size, otherwise size - 1, as the period.</summary>
internal sealed class WindowOp : OpKernel
{
    public static readonly OpDef Hann = new("HannWindow", 17, int.MaxValue, static n => new WindowOp(n, 0));
    public static readonly OpDef Hamming = new("HammingWindow", 17, int.MaxValue, static n => new WindowOp(n, 1));
    public static readonly OpDef Blackman = new("BlackmanWindow", 17, int.MaxValue, static n => new WindowOp(n, 2));

    private readonly int _kind;
    private readonly bool _periodic;
    private readonly DType _type;

    private WindowOp(NodeInfo n, int kind)
    {
        _kind = kind;
        _periodic = n.GetInt("periodic", 1) != 0;
        long code = n.GetInt("output_datatype", 1);
        if (code < 0 || code > int.MaxValue || !DTypes.TryFromOnnx((int)code, out _type) || _type is not (DType.F32 or DType.F64))
        {
            throw n.Unsupported("output_datatype must be float or double");
        }
    }

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects the size");
        }

        OpUtil.RequireIntegers(ctx, 0, "size", allowScalar: true);
        if (!ctx.TryGetInputValue(0, out Tensor s))
        {
            ctx.RequireRunTimeValues();
            return;
        }

        long size = OpUtil.First(s);
        if (size < 0)
        {
            throw ctx.Fail("size cannot be negative");
        }

        ctx.SetOutput(0, _type, OpUtil.MakeGrownShape(ctx, [size], _type));
    }

    public override void Run(in OpRun run)
    {
        Tensor y = run.Outputs[0];
        int size = y.Length;
        double period = _periodic ? size : size - 1;
        for (int i = 0; i < size; i++)
        {
            double a = period == 0 ? 0 : 2 * Math.PI * i / period;
            double w = _kind switch
            {
                0 => 0.5 - (0.5 * Math.Cos(a)),
                1 => (25.0 / 46) - ((21.0 / 46) * Math.Cos(a)),
                _ => 0.42 - (0.5 * Math.Cos(a)) + (0.08 * Math.Cos(2 * a)),
            };
            DftOp.Write(y, i, w);
        }
    }
}

/// <summary>
/// MelWeightMatrix: the (dft_length / 2 + 1) x num_mel_bins matrix of triangular mel filters with peak 1. The band edges are
/// num_mel_bins + 2 points evenly spaced on the mel scale mel(f) = 2595 log10(1 + f / 700) between the two edge frequencies,
/// mapped back to hertz and then to a spectrum bin, floor((dft_length + 1) f / sample_rate); each filter rises linearly from its first
/// bin to its second and falls to its third. The bins and the edges are computed in double precision.
/// </summary>
internal sealed class MelWeightMatrixOp : OpKernel
{
    public static readonly OpDef Def = new("MelWeightMatrix", 17, int.MaxValue, static n => new MelWeightMatrixOp(n));

    private readonly DType _type;

    private MelWeightMatrixOp(NodeInfo n)
    {
        long code = n.GetInt("output_datatype", 1);
        if (code < 0 || code > int.MaxValue || !DTypes.TryFromOnnx((int)code, out _type) || _type is not (DType.F32 or DType.F64))
        {
            throw n.Unsupported("output_datatype must be float or double");
        }
    }

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 5 || !ctx.HasInput(0) || !ctx.HasInput(1) || !ctx.HasInput(2) || !ctx.HasInput(3) || !ctx.HasInput(4))
        {
            throw ctx.Fail("expects num_mel_bins, dft_length, sample_rate, lower_edge_hertz and upper_edge_hertz");
        }

        for (int i = 0; i < 3; i++)
        {
            OpUtil.RequireIntegers(ctx, i, "an integer parameter", allowScalar: true);
        }

        if (!ctx.TryGetInputValue(0, out Tensor bins) || !ctx.TryGetInputValue(1, out Tensor dft))
        {
            ctx.RequireRunTimeValues();
            return;
        }

        long mel = OpUtil.First(bins), length = OpUtil.First(dft);
        if (mel < 1 || length < 1)
        {
            throw ctx.Fail("num_mel_bins and dft_length must be at least 1");
        }

        ctx.SetOutput(0, _type, OpUtil.MakeGrownShape(ctx, [(length / 2) + 1, mel], _type));
    }

    public override void Run(in OpRun run)
    {
        Tensor y = run.Outputs[0];
        int mel = (int)OpUtil.First(run.Inputs[0]), length = (int)OpUtil.First(run.Inputs[1]);
        double rate = OpUtil.First(run.Inputs[2]), lower = Value(run.Inputs[3]), upper = Value(run.Inputs[4]);
        int spectrum = (length / 2) + 1;
        static double ToMel(double f) => 2595.0 * Math.Log10(1.0 + (f / 700.0));
        static double ToHertz(double m) => 700.0 * (Math.Pow(10.0, m / 2595.0) - 1.0);
        double low = ToMel(lower), step = (ToMel(upper) - low) / (mel + 2);
        var edge = new int[mel + 2];
        for (int i = 0; i < edge.Length; i++)
        {
            double bin = Math.Floor((length + 1) * ToHertz(low + (step * i)) / rate);
            edge[i] = (int)Math.Clamp(bin, 0, spectrum - 1);
        }

        if (y.DType == DType.F32)
        {
            y.MutableData<float>().Clear();
        }
        else
        {
            y.MutableData<double>().Clear();
        }

        for (int i = 0; i < mel; i++)
        {
            int a = edge[i], b = edge[i + 1], c = edge[i + 2];
            if (b == a)
            {
                DftOp.Write(y, ((long)b * mel) + i, 1.0);
            }
            else
            {
                for (int j = a; j <= b; j++)
                {
                    DftOp.Write(y, ((long)j * mel) + i, (double)(j - a) / (b - a));
                }
            }

            for (int j = b; j < c; j++)
            {
                DftOp.Write(y, ((long)j * mel) + i, (double)(c - j) / (c - b));
            }
        }
    }

    private static double Value(Tensor t) => t.DType == DType.F64 ? t.Data<double>()[0] : t.DType == DType.F16 ? (double)t.Data<Half>()[0] : t.DType == DType.BF16 ? (double)(float)t.Data<BFloat16>()[0] : t.Data<float>()[0];
}
