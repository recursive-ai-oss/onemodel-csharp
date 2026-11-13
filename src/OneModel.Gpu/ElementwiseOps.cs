using System;
using System.Collections.Generic;
using RecursiveAI.OneGpu;
using RecursiveAI.OneGpu.Kernels;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu;

/// <summary>
/// The entries for elementwise operators: binary operators with ONNX multidirectional broadcasting, and unary functions. Both kinds have no
/// reduction order, so the same kernel serves fast and exact mode and gives the same bits on the same device run after run (it is not the
/// CPU's exact arithmetic for the transcendental functions, which uses fixed polynomials of its own).
/// </summary>
internal static class ElementwiseOps
{
    private const int MaxRank = 8;

    /// <summary>The table key of the one-pass Gelu the rewrite makes from the five-operator form (its node info is that of the last Mul).</summary>
    public const string FusedGeluKey = "~Gelu";

    public static readonly GpuOpEntry FusedGelu = new(FusedGeluKey, true, _ => null,
        (GpuClaimContext ctx, out GpuBinding? b) => BindUnary(ctx, "FusedGelu", GpuUnaryOp.GeluErf, out b));

    public static IEnumerable<GpuOpEntry> Entries
    {
        get
        {
            foreach ((string name, GpuBinaryOp op) in new (string, GpuBinaryOp)[]
            {
                ("Add", GpuBinaryOp.Add), ("Sub", GpuBinaryOp.Sub), ("Mul", GpuBinaryOp.Mul), ("Div", GpuBinaryOp.Div),
                ("Pow", GpuBinaryOp.Pow), ("Min", GpuBinaryOp.Min), ("Max", GpuBinaryOp.Max),
            })
            {
                GpuBinaryOp captured = op;
                yield return new GpuOpEntry(name, true, CheckBinary, (GpuClaimContext ctx, out GpuBinding? b) => BindBinary(ctx, captured, out b));
            }

            foreach ((string name, GpuUnaryOp op) in new (string, GpuUnaryOp)[]
            {
                ("Neg", GpuUnaryOp.Neg), ("Abs", GpuUnaryOp.Abs), ("Sqrt", GpuUnaryOp.Sqrt), ("Reciprocal", GpuUnaryOp.Reciprocal),
                ("Exp", GpuUnaryOp.Exp), ("Log", GpuUnaryOp.Log), ("Tanh", GpuUnaryOp.Tanh), ("Erf", GpuUnaryOp.Erf),
                ("Sigmoid", GpuUnaryOp.Sigmoid), ("Relu", GpuUnaryOp.Relu), ("LeakyRelu", GpuUnaryOp.LeakyRelu),
                ("Clip", GpuUnaryOp.Clip), ("Gelu", GpuUnaryOp.GeluErf), ("HardSwish", GpuUnaryOp.HardSwish), ("HardSigmoid", GpuUnaryOp.HardSigmoid),
            })
            {
                GpuUnaryOp captured = op;
                string opName = name;
                yield return new GpuOpEntry(name, true, info => CheckUnary(info, opName), (GpuClaimContext ctx, out GpuBinding? b) => BindUnary(ctx, opName, captured, out b));
            }
        }
    }

    private static string? CheckBinary(NodeInfo info) => info.InputCount != 2 ? $"{info.OpType} with {info.InputCount} inputs (the GPU kernel takes two)" : null;

    private static string? CheckUnary(NodeInfo info, string name) => name switch
    {
        "Gelu" when info.Opset < 20 => "Gelu exists from opset 20",
        "Gelu" when info.GetString("approximate", "none") is not ("none" or "tanh") => "unknown Gelu approximation",
        _ => null,
    };

    // ---- binary ------------------------------------------------------------------------------------------------------------

    private static string? BindBinary(GpuClaimContext ctx, GpuBinaryOp op, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            return "needs two inputs";
        }

        if (ctx.InputType(0) is DType.I32 or DType.I64)
        {
            return IntegerOps.BindIntegerBinary(ctx, op, out binding);
        }

        if (ctx.InputElement(0, out DeviceElement element) is { } why)
        {
            return why;
        }

        if (ctx.InputType(1) != ctx.InputType(0))
        {
            return $"the inputs are {ctx.InputType(0)} and {ctx.InputType(1)}; the GPU kernel needs one element type";
        }

        Shape sa = ctx.InputShape(0), sb = ctx.InputShape(1);
        if (sa.Rank > MaxRank || sb.Rank > MaxRank)
        {
            return $"an input has rank above {MaxRank}";
        }

        int[] da = GpuClaimContext.Dims(sa), db = GpuClaimContext.Dims(sb);
        int[] expected;
        try
        {
            expected = ElementwiseKernels.BroadcastShape(da, db);
        }
        catch (GpuKernelShapeException e)
        {
            return $"the shapes do not broadcast: {e.Message}";
        }

        Shape so = ctx.OutputShape(0);
        if (ctx.OutputType(0) != ctx.InputType(0) || !expected.AsSpan().SequenceEqual(GpuClaimContext.Dims(so)))
        {
            return "the planned output does not match the broadcast of the inputs";
        }

        if (ctx.FitsBinding(Math.Max(Math.Max(sa.Length, sb.Length), so.Length), element) is { } big)
        {
            return big;
        }

        binding = new BinaryBinding(ctx, op, element, da, db);
        return null;
    }

    private sealed class BinaryBinding : GpuBinding, IBiasAdd
    {
        private readonly ElementwiseKernels _kernels;
        private readonly GpuBinaryOp _op;
        private readonly DeviceElement _element;
        private readonly int[] _shapeA, _shapeB;
        private readonly Operand _a, _b;
        private readonly int _outId;
        private readonly GpuKernelMode _mode;

        public BinaryBinding(GpuClaimContext ctx, GpuBinaryOp op, DeviceElement element, int[] shapeA, int[] shapeB)
        {
            _kernels = ctx.Backend.Elementwise;
            _op = op;
            _element = element;
            _shapeA = shapeA;
            _shapeB = shapeB;
            _a = Operand.Of(ctx, 0);
            _b = Operand.Of(ctx, 1);
            _aId = ctx.InputId(0);
            _bId = ctx.InputId(1);
            _outId = ctx.OutputId(0);
            _mode = ctx.Backend.KernelMode;
            _outLength = ctx.OutputShape(0).Length;
            Flops = _outLength;
        }

        private readonly int _aId, _bId, _outLength;

        // A constant whose elements are one per column of the result (shape [n], or [1, ..., n]) added to a value of the result's shape.
        public bool TryBias(out int valueId, out DeviceConstant bias, out int biasId, out int resultId, out int vectorLength)
        {
            valueId = biasId = -1;
            bias = null!;
            resultId = _outId;
            vectorLength = 0;
            if (_op != GpuBinaryOp.Add)
            {
                return false;
            }

            for (int side = 0; side < 2; side++)
            {
                int[] vec = side == 0 ? _shapeB : _shapeA, other = side == 0 ? _shapeA : _shapeB;
                Operand v = side == 0 ? _b : _a;
                if (v.Constant is not { } c || vec.Length == 0 || other.Length < vec.Length)
                {
                    continue;
                }

                long n = vec[^1];
                bool vector = true;
                for (int i = 0; i < vec.Length - 1; i++)
                {
                    vector &= vec[i] == 1;
                }

                long otherLength = 1;
                foreach (int d in other)
                {
                    otherLength *= d;
                }

                if (!vector || n != other[^1] || otherLength != _outLength)
                {
                    continue;
                }

                valueId = side == 0 ? _aId : _bId;
                biasId = side == 0 ? _bId : _aId;
                bias = c;
                vectorLength = (int)n;
                return true;
            }

            return false;
        }

        public override DeviceElement OutputElement(int output) => _element;

        public override void Encode(GpuRunContext ctx)
        {
            if (_element == DeviceElement.F32)
            {
                Encode<float>(ctx);
            }
            else
            {
                Encode<Half>(ctx);
            }
        }

        private void Encode<T>(GpuRunContext ctx) where T : unmanaged
        {
            GpuBuffer<T> a = _a.Get<T>(ctx, out int offA);
            GpuBuffer<T> b = _b.Get<T>(ctx, out int offB);
            GpuBuffer<T> y = ctx.Buffer<T>(_outId, out int offY);
            _kernels.EncodeBinary(ctx.Encoder, _op, a, offA, _shapeA, b, offB, _shapeB, y, offY, null, _mode);
        }
    }

    // ---- unary -------------------------------------------------------------------------------------------------------------

    private static string? BindUnary(GpuClaimContext ctx, string name, GpuUnaryOp op, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0))
        {
            return "needs an input";
        }

        if (ctx.InputType(0) is DType.I32 or DType.I64)
        {
            return IntegerOps.BindIntegerUnary(ctx, op, out binding);
        }

        if (ctx.InputElement(0, out DeviceElement element) is { } why)
        {
            return why;
        }

        if (ctx.OutputType(0) != ctx.InputType(0) || ctx.OutputShape(0).Length != ctx.InputShape(0).Length)
        {
            return "the planned output does not match the input";
        }

        if (ctx.FitsBinding(ctx.InputShape(0).Length, element) is { } big)
        {
            return big;
        }

        float p0 = 0f, p1 = 0f;
        switch (name)
        {
            case "LeakyRelu":
                p0 = ctx.Info.GetFloat("alpha", 0.01f);
                break;
            case "HardSigmoid":
                p0 = ctx.Info.GetFloat("alpha", 0.2f);
                p1 = ctx.Info.GetFloat("beta", 0.5f);
                break;
            case "Gelu":
                op = ctx.Info.GetString("approximate", "none") == "tanh" ? GpuUnaryOp.GeluTanh : GpuUnaryOp.GeluErf;
                break;
            case "Clip":
                // Opset 11 and later take min and max as optional scalar inputs, earlier ones as attributes.
                p0 = float.NegativeInfinity;
                p1 = float.PositiveInfinity;
                if (ctx.Info.Opset < 11)
                {
                    p0 = ctx.Info.GetFloat("min", float.MinValue);
                    p1 = ctx.Info.GetFloat("max", float.MaxValue);
                }
                else
                {
                    if (ReadScalar(ctx, 1, ref p0) is { } w1)
                    {
                        return w1;
                    }

                    if (ReadScalar(ctx, 2, ref p1) is { } w2)
                    {
                        return w2;
                    }
                }

                break;
        }

        binding = new UnaryBinding(ctx, op, element, p0, p1);
        return null;
    }

    // A Clip bound must be a constant scalar to become a kernel parameter.
    private static string? ReadScalar(GpuClaimContext ctx, int input, ref float value)
    {
        if (!ctx.HasInput(input))
        {
            return null;
        }

        if (!ctx.InputIsConst(input) || ctx.InputShape(input).Length != 1)
        {
            return $"input {input} is not a constant scalar";
        }

        Tensor t = ctx.InputConst(input);
        if (t.DType == DType.F32)
        {
            value = t.Data<float>()[0];
        }
        else if (t.DType == DType.F16)
        {
            value = (float)t.Data<Half>()[0];
        }
        else
        {
            return $"input {input} is {t.DType}";
        }

        return null;
    }

    private sealed class UnaryBinding : GpuBinding, IProductActivation
    {
        private readonly ElementwiseKernels _kernels;
        private readonly GpuUnaryOp _op;
        private readonly DeviceElement _element;
        private readonly float _p0, _p1;
        private readonly Operand _x;
        private readonly int _outId;
        private readonly int _count;
        private readonly GpuKernelMode _mode;

        public UnaryBinding(GpuClaimContext ctx, GpuUnaryOp op, DeviceElement element, float p0, float p1)
        {
            _kernels = ctx.Backend.Elementwise;
            _op = op;
            _element = element;
            _p0 = p0;
            _p1 = p1;
            _x = Operand.Of(ctx, 0);
            _xId = ctx.InputId(0);
            _outId = ctx.OutputId(0);
            _count = ctx.InputShape(0).Length;
            _mode = ctx.Backend.KernelMode;
            Flops = _count;
        }

        private readonly int _xId;

        public bool TryActivation(out GemmActivation activation, out int valueId, out int resultId)
        {
            valueId = _xId;
            resultId = _outId;
            activation = _op switch
            {
                GpuUnaryOp.Relu => GemmActivation.Relu,
                GpuUnaryOp.GeluErf => GemmActivation.GeluErf,
                GpuUnaryOp.GeluTanh => GemmActivation.GeluTanh,
                _ => GemmActivation.None,
            };
            return activation != GemmActivation.None;
        }

        public override DeviceElement OutputElement(int output) => _element;

        public override void Encode(GpuRunContext ctx)
        {
            if (_element == DeviceElement.F32)
            {
                Encode<float>(ctx);
            }
            else
            {
                Encode<Half>(ctx);
            }
        }

        private void Encode<T>(GpuRunContext ctx) where T : unmanaged
        {
            GpuBuffer<T> x = _x.Get<T>(ctx, out int offX);
            GpuBuffer<T> y = ctx.Buffer<T>(_outId, out int offY);
            _kernels.EncodeUnary(ctx.Encoder, _op, x, offX, y, offY, _count, _p0, _p1, _mode);
        }
    }
}
