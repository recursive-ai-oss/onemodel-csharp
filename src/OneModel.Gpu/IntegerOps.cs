using System;
using System.Collections.Generic;
using RecursiveAI.OneGpu;
using RecursiveAI.OneGpu.Kernels;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Graph.Operators;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu;

/// <summary>
/// The operators around an attention mask and around shape arithmetic: integer arithmetic, comparisons, boolean logic, <c>Where</c>,
/// <c>Expand</c>, and the generators <c>Range</c> and <c>ConstantOfShape</c>. Integer and boolean results have no rounding and no
/// reduction order, so every kernel here is bit-exact against the CPU and one kernel serves fast and exact mode.
/// A boolean tensor is one byte per element, 0 or 1, as on the host.
/// </summary>
internal static class IntegerOps
{
    private const int MaxRank = 8;

    public static IEnumerable<GpuOpEntry> Entries
    {
        get
        {
            foreach ((string name, GpuCompareOp op) in new (string, GpuCompareOp)[]
            {
                ("Equal", GpuCompareOp.Equal), ("Less", GpuCompareOp.Less), ("Greater", GpuCompareOp.Greater),
                ("LessOrEqual", GpuCompareOp.LessOrEqual), ("GreaterOrEqual", GpuCompareOp.GreaterOrEqual),
            })
            {
                GpuCompareOp captured = op;
                string opName = name;
                yield return new GpuOpEntry(name, true, CheckTwoInputs, (GpuClaimContext ctx, out GpuBinding? b) => BindCompare(ctx, opName, captured, out b));
            }

            foreach ((string name, GpuLogicOp op) in new (string, GpuLogicOp)[] { ("And", GpuLogicOp.And), ("Or", GpuLogicOp.Or), ("Xor", GpuLogicOp.Xor) })
            {
                GpuLogicOp captured = op;
                yield return new GpuOpEntry(name, true, CheckTwoInputs, (GpuClaimContext ctx, out GpuBinding? b) => BindLogic(ctx, captured, out b));
            }

            yield return new GpuOpEntry("Not", true, _ => null, BindNot);
            yield return new GpuOpEntry("Where", true, info => info.InputCount != 3 ? "Where takes three inputs" : null, BindWhere);
            yield return new GpuOpEntry("Expand", true, _ => null, BindExpand);
            yield return new GpuOpEntry("Mod", true, CheckTwoInputs, BindMod);
            yield return new GpuOpEntry("Range", true, _ => null, BindRange);
            yield return new GpuOpEntry("ConstantOfShape", true, _ => null, BindConstantOfShape);
        }
    }

    private static string? CheckTwoInputs(NodeInfo info) => info.InputCount != 2 ? $"{info.OpType} with {info.InputCount} inputs (the GPU kernel takes two)" : null;

    private static string? RankLimit(Shape a, Shape b, out int[] da, out int[] db)
    {
        da = GpuClaimContext.Dims(a);
        db = GpuClaimContext.Dims(b);
        return a.Rank > MaxRank || b.Rank > MaxRank ? $"an input has rank above {MaxRank}" : null;
    }

    private static string? CheckBroadcast(ReadOnlySpan<int> shapeA, ReadOnlySpan<int> shapeB, Shape planned, out int[] result)
    {
        try
        {
            result = ElementwiseKernels.BroadcastShape(shapeA, shapeB);
        }
        catch (GpuKernelShapeException e)
        {
            result = [];
            return $"the shapes do not broadcast: {e.Message}";
        }

        return result.AsSpan().SequenceEqual(GpuClaimContext.Dims(planned)) ? null : "the planned output does not match the broadcast of the inputs";
    }

    // ---- integer arithmetic (reached from the elementwise entries when the element type is an integer) -----------------------

    /// <summary>Add, Sub, Mul, Div, Min and Max on int32 or int64 tensors.</summary>
    public static string? BindIntegerBinary(GpuClaimContext ctx, GpuBinaryOp op, out GpuBinding? binding)
    {
        GpuIntegerOp? mapped = op switch
        {
            GpuBinaryOp.Add => GpuIntegerOp.Add,
            GpuBinaryOp.Sub => GpuIntegerOp.Sub,
            GpuBinaryOp.Mul => GpuIntegerOp.Mul,
            GpuBinaryOp.Div => GpuIntegerOp.Div,
            GpuBinaryOp.Min => GpuIntegerOp.Min,
            GpuBinaryOp.Max => GpuIntegerOp.Max,
            _ => null,
        };
        binding = null;
        return mapped is null ? "this operator has no integer kernel" : BindIntegerBinary(ctx, mapped.Value, out binding);
    }

    private static string? BindMod(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        if (ctx.InputType(0) is not (DType.I32 or DType.I64))
        {
            return $"Mod on {ctx.InputType(0)}: the GPU kernel takes int32 and int64";
        }

        // fmod = 1 takes the sign of the dividend (C's fmod); fmod = 0 takes the divisor's (Python's %).
        return BindIntegerBinary(ctx, ctx.Info.GetInt("fmod", 0) == 1 ? GpuIntegerOp.Fmod : GpuIntegerOp.Mod, out binding);
    }

    private static string? BindIntegerBinary(GpuClaimContext ctx, GpuIntegerOp op, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            return "needs two inputs";
        }

        if (ctx.InputType(1) != ctx.InputType(0) || ctx.OutputType(0) != ctx.InputType(0))
        {
            return $"the inputs are {ctx.InputType(0)} and {ctx.InputType(1)}; the GPU kernel needs one element type";
        }

        ctx.InputAnyElement(0, out DeviceElement element);
        if (RankLimit(ctx.InputShape(0), ctx.InputShape(1), out int[] da, out int[] db) is { } rank)
        {
            return rank;
        }

        if (CheckBroadcast(da, db, ctx.OutputShape(0), out _) is { } shapes)
        {
            return shapes;
        }

        if (ctx.FitsBinding(Math.Max(Math.Max(ctx.InputShape(0).Length, ctx.InputShape(1).Length), ctx.OutputShape(0).Length), element) is { } big)
        {
            return big;
        }

        binding = new IntegerBinaryBinding(ctx, op, element, da, db);
        return null;
    }

    private sealed class IntegerBinaryBinding : GpuBinding
    {
        private readonly IntegerKernels _kernels;
        private readonly GpuIntegerOp _op;
        private readonly DeviceElement _element;
        private readonly int[] _shapeA, _shapeB;
        private readonly Operand _a, _b;
        private readonly int _outId;

        public IntegerBinaryBinding(GpuClaimContext ctx, GpuIntegerOp op, DeviceElement element, int[] shapeA, int[] shapeB)
        {
            _kernels = ctx.Backend.Integer;
            _op = op;
            _element = element;
            _shapeA = shapeA;
            _shapeB = shapeB;
            _a = Operand.Of(ctx, 0);
            _b = Operand.Of(ctx, 1);
            _outId = ctx.OutputId(0);
        }

        public override DeviceElement OutputElement(int output) => _element;

        public override void Encode(GpuRunContext ctx)
        {
            if (_element == DeviceElement.I64)
            {
                Encode<long>(ctx);
            }
            else
            {
                Encode<int>(ctx);
            }
        }

        private void Encode<T>(GpuRunContext ctx) where T : unmanaged
        {
            GpuBuffer<T> a = _a.Get<T>(ctx, out int offA);
            GpuBuffer<T> b = _b.Get<T>(ctx, out int offB);
            GpuBuffer<T> y = ctx.Buffer<T>(_outId, out int offY);
            _kernels.EncodeBinary(ctx.Encoder, _op, a, offA, _shapeA, b, offB, _shapeB, y, offY);
        }
    }

    /// <summary>Neg and Abs on int32 or int64 tensors.</summary>
    public static string? BindIntegerUnary(GpuClaimContext ctx, GpuUnaryOp op, out GpuBinding? binding)
    {
        binding = null;
        if (op is not (GpuUnaryOp.Neg or GpuUnaryOp.Abs))
        {
            return "this operator has no integer kernel";
        }

        if (ctx.OutputType(0) != ctx.InputType(0) || ctx.OutputShape(0).Length != ctx.InputShape(0).Length)
        {
            return "the planned output does not match the input";
        }

        ctx.InputAnyElement(0, out DeviceElement element);
        if (ctx.FitsBinding(ctx.InputShape(0).Length, element) is { } big)
        {
            return big;
        }

        binding = new IntegerUnaryBinding(ctx, op == GpuUnaryOp.Neg ? GpuIntegerUnaryOp.Neg : GpuIntegerUnaryOp.Abs, element);
        return null;
    }

    private sealed class IntegerUnaryBinding : GpuBinding
    {
        private readonly IntegerKernels _kernels;
        private readonly GpuIntegerUnaryOp _op;
        private readonly DeviceElement _element;
        private readonly Operand _x;
        private readonly int _outId, _count;

        public IntegerUnaryBinding(GpuClaimContext ctx, GpuIntegerUnaryOp op, DeviceElement element)
        {
            _kernels = ctx.Backend.Integer;
            _op = op;
            _element = element;
            _x = Operand.Of(ctx, 0);
            _outId = ctx.OutputId(0);
            _count = ctx.InputShape(0).Length;
        }

        public override DeviceElement OutputElement(int output) => _element;

        public override void Encode(GpuRunContext ctx)
        {
            if (_element == DeviceElement.I64)
            {
                GpuBuffer<long> x = _x.Get<long>(ctx, out int offX);
                _kernels.EncodeUnary(ctx.Encoder, _op, x, offX, ctx.Buffer<long>(_outId, out int offY), offY, _count);
            }
            else
            {
                GpuBuffer<int> x = _x.Get<int>(ctx, out int offX);
                _kernels.EncodeUnary(ctx.Encoder, _op, x, offX, ctx.Buffer<int>(_outId, out int offY), offY, _count);
            }
        }
    }

    // ---- comparisons ------------------------------------------------------------------------------------------------------

    private static string? BindCompare(GpuClaimContext ctx, string name, GpuCompareOp op, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            return "needs two inputs";
        }

        if (ctx.InputAnyElement(0, out DeviceElement element) is { } why)
        {
            return why;
        }

        if (ctx.InputType(1) != ctx.InputType(0))
        {
            return $"the inputs are {ctx.InputType(0)} and {ctx.InputType(1)}; the GPU kernel needs one element type";
        }

        if (element == DeviceElement.Bool && name != "Equal")
        {
            return $"{name} on booleans is not an ONNX operator the GPU takes";
        }

        if (ctx.OutputType(0) != DType.Bool)
        {
            return "the planned output is not a boolean tensor";
        }

        if (RankLimit(ctx.InputShape(0), ctx.InputShape(1), out int[] da, out int[] db) is { } rank)
        {
            return rank;
        }

        if (CheckBroadcast(da, db, ctx.OutputShape(0), out _) is { } shapes)
        {
            return shapes;
        }

        if (ctx.FitsBinding(Math.Max(Math.Max(ctx.InputShape(0).Length, ctx.InputShape(1).Length), ctx.OutputShape(0).Length), element) is { } big)
        {
            return big;
        }

        binding = new CompareBinding(ctx, op, element, da, db);
        return null;
    }

    private sealed class CompareBinding : GpuBinding
    {
        private readonly ElementwiseKernels _floats;
        private readonly IntegerKernels _integers;
        private readonly GpuCompareOp _op;
        private readonly DeviceElement _element;
        private readonly int[] _shapeA, _shapeB;
        private readonly Operand _a, _b;
        private readonly int _outId;

        public CompareBinding(GpuClaimContext ctx, GpuCompareOp op, DeviceElement element, int[] shapeA, int[] shapeB)
        {
            _floats = ctx.Backend.Elementwise;
            _integers = ctx.Backend.Integer;
            _op = op;
            _element = element;
            _shapeA = shapeA;
            _shapeB = shapeB;
            _a = Operand.Of(ctx, 0);
            _b = Operand.Of(ctx, 1);
            _outId = ctx.OutputId(0);
            Flops = ctx.OutputShape(0).Length;
        }

        public override DeviceElement OutputElement(int output) => DeviceElement.Bool;

        public override void Encode(GpuRunContext ctx)
        {
            switch (_element)
            {
                case DeviceElement.F32:
                    Encode<float>(ctx, true);
                    break;
                case DeviceElement.F16:
                    Encode<Half>(ctx, true);
                    break;
                case DeviceElement.I32:
                    Encode<int>(ctx, false);
                    break;
                case DeviceElement.I64:
                    Encode<long>(ctx, false);
                    break;
                default:
                    Encode<byte>(ctx, false);
                    break;
            }
        }

        private void Encode<T>(GpuRunContext ctx, bool floating) where T : unmanaged
        {
            GpuBuffer<T> a = _a.Get<T>(ctx, out int offA);
            GpuBuffer<T> b = _b.Get<T>(ctx, out int offB);
            GpuBuffer<byte> y = ctx.Buffer<byte>(_outId, out int offY);
            if (floating)
            {
                _floats.EncodeCompare(ctx.Encoder, _op, a, offA, _shapeA, b, offB, _shapeB, y, offY);
            }
            else
            {
                _integers.EncodeCompare(ctx.Encoder, _op, a, offA, _shapeA, b, offB, _shapeB, y, offY);
            }
        }
    }

    // ---- boolean logic ----------------------------------------------------------------------------------------------------

    private static string? BindLogic(GpuClaimContext ctx, GpuLogicOp op, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            return "needs two inputs";
        }

        if (ctx.InputType(0) != DType.Bool || ctx.InputType(1) != DType.Bool || ctx.OutputType(0) != DType.Bool)
        {
            return $"the inputs are {ctx.InputType(0)} and {ctx.InputType(1)}; the GPU kernel takes booleans";
        }

        if (RankLimit(ctx.InputShape(0), ctx.InputShape(1), out int[] da, out int[] db) is { } rank)
        {
            return rank;
        }

        if (CheckBroadcast(da, db, ctx.OutputShape(0), out _) is { } shapes)
        {
            return shapes;
        }

        if (ctx.FitsBinding(Math.Max(Math.Max(ctx.InputShape(0).Length, ctx.InputShape(1).Length), ctx.OutputShape(0).Length), DeviceElement.Bool) is { } big)
        {
            return big;
        }

        binding = new LogicBinding(ctx, op, da, db);
        return null;
    }

    private sealed class LogicBinding : GpuBinding
    {
        private readonly IntegerKernels _kernels;
        private readonly GpuLogicOp _op;
        private readonly int[] _shapeA, _shapeB;
        private readonly Operand _a, _b;
        private readonly int _outId;

        public LogicBinding(GpuClaimContext ctx, GpuLogicOp op, int[] shapeA, int[] shapeB)
        {
            _kernels = ctx.Backend.Integer;
            _op = op;
            _shapeA = shapeA;
            _shapeB = shapeB;
            _a = Operand.Of(ctx, 0);
            _b = Operand.Of(ctx, 1);
            _outId = ctx.OutputId(0);
        }

        public override DeviceElement OutputElement(int output) => DeviceElement.Bool;

        public override void Encode(GpuRunContext ctx)
        {
            GpuBuffer<byte> a = _a.Get<byte>(ctx, out int offA);
            GpuBuffer<byte> b = _b.Get<byte>(ctx, out int offB);
            GpuBuffer<byte> y = ctx.Buffer<byte>(_outId, out int offY);
            _kernels.EncodeLogic(ctx.Encoder, _op, a, offA, _shapeA, b, offB, _shapeB, y, offY);
        }
    }

    private static string? BindNot(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0))
        {
            return "needs an input";
        }

        if (ctx.InputType(0) != DType.Bool || ctx.OutputType(0) != DType.Bool)
        {
            return $"the input is {ctx.InputType(0)}; the GPU kernel takes booleans";
        }

        if (ctx.FitsBinding(ctx.InputShape(0).Length, DeviceElement.Bool) is { } big)
        {
            return big;
        }

        binding = new NotBinding(ctx);
        return null;
    }

    private sealed class NotBinding : GpuBinding
    {
        private readonly IntegerKernels _kernels;
        private readonly Operand _x;
        private readonly int _outId, _count;

        public NotBinding(GpuClaimContext ctx)
        {
            _kernels = ctx.Backend.Integer;
            _x = Operand.Of(ctx, 0);
            _outId = ctx.OutputId(0);
            _count = ctx.InputShape(0).Length;
        }

        public override DeviceElement OutputElement(int output) => DeviceElement.Bool;

        public override void Encode(GpuRunContext ctx)
        {
            GpuBuffer<byte> x = _x.Get<byte>(ctx, out int offX);
            _kernels.EncodeNot(ctx.Encoder, x, offX, ctx.Buffer<byte>(_outId, out int offY), offY, _count);
        }
    }

    // ---- Where ------------------------------------------------------------------------------------------------------------

    private static string? BindWhere(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0) || !ctx.HasInput(1) || !ctx.HasInput(2))
        {
            return "needs three inputs";
        }

        if (ctx.InputType(0) != DType.Bool)
        {
            return $"the condition is {ctx.InputType(0)}, not a boolean tensor";
        }

        if (ctx.InputAnyElement(1, out DeviceElement element) is { } why)
        {
            return why;
        }

        if (element == DeviceElement.Bool)
        {
            return "Where on boolean values has no GPU kernel";
        }

        if (ctx.InputType(2) != ctx.InputType(1) || ctx.OutputType(0) != ctx.InputType(1))
        {
            return $"the values are {ctx.InputType(1)} and {ctx.InputType(2)}; the GPU kernel needs one element type";
        }

        Shape sc = ctx.InputShape(0), sa = ctx.InputShape(1), sb = ctx.InputShape(2);
        if (sc.Rank > MaxRank || sa.Rank > MaxRank || sb.Rank > MaxRank)
        {
            return $"an input has rank above {MaxRank}";
        }

        // A 64-bit value moves as two words and so costs a dimension of the kernel's limit of six.
        int limit = element == DeviceElement.I64 ? 5 : 6;
        int[] dc = GpuClaimContext.Dims(sc), da = GpuClaimContext.Dims(sa), db = GpuClaimContext.Dims(sb);
        int[] expected;
        try
        {
            expected = ElementwiseKernels.BroadcastShape(ElementwiseKernels.BroadcastShape(dc, da), db);
        }
        catch (GpuKernelShapeException e)
        {
            return $"the shapes do not broadcast: {e.Message}";
        }

        if (!expected.AsSpan().SequenceEqual(GpuClaimContext.Dims(ctx.OutputShape(0))))
        {
            return "the planned output does not match the broadcast of the inputs";
        }

        if (expected.Length > limit)
        {
            return $"the broadcast has {expected.Length} dimensions and the Where kernel takes {limit} for this element type";
        }

        if (ctx.FitsBinding(Math.Max(Math.Max(sa.Length, sb.Length), Math.Max(sc.Length, ctx.OutputShape(0).Length)), element) is { } big)
        {
            return big;
        }

        binding = new WhereBinding(ctx, element, dc, da, db);
        return null;
    }

    private sealed class WhereBinding : GpuBinding
    {
        private readonly ElementwiseKernels _kernels;
        private readonly DeviceElement _element;
        private readonly int[] _shapeC, _shapeA, _shapeB;
        private readonly Operand _c, _a, _b;
        private readonly int _outId;

        public WhereBinding(GpuClaimContext ctx, DeviceElement element, int[] shapeC, int[] shapeA, int[] shapeB)
        {
            _kernels = ctx.Backend.Elementwise;
            _element = element;
            _shapeC = shapeC;
            _shapeA = shapeA;
            _shapeB = shapeB;
            _c = Operand.Of(ctx, 0);
            _a = Operand.Of(ctx, 1);
            _b = Operand.Of(ctx, 2);
            _outId = ctx.OutputId(0);
        }

        public override DeviceElement OutputElement(int output) => _element;

        public override void Encode(GpuRunContext ctx)
        {
            switch (_element)
            {
                case DeviceElement.F32:
                    Encode<float>(ctx);
                    break;
                case DeviceElement.F16:
                    Encode<Half>(ctx);
                    break;
                case DeviceElement.I32:
                    Encode<int>(ctx);
                    break;
                default:
                    Encode<long>(ctx);
                    break;
            }
        }

        private void Encode<T>(GpuRunContext ctx) where T : unmanaged
        {
            GpuBuffer<byte> c = _c.Get<byte>(ctx, out int offC);
            GpuBuffer<T> a = _a.Get<T>(ctx, out int offA);
            GpuBuffer<T> b = _b.Get<T>(ctx, out int offB);
            GpuBuffer<T> y = ctx.Buffer<T>(_outId, out int offY);
            _kernels.EncodeWhere(ctx.Encoder, c, offC, _shapeC, a, offA, _shapeA, b, offB, _shapeB, y, offY);
        }
    }

    // ---- Expand -----------------------------------------------------------------------------------------------------------

    // The device's Expand kernel needs its input and output in different buffers, which one arena slab cannot promise. A broadcast is
    // also a selection with a condition that is true everywhere, so the result is made from a scratch of "true" over the target shape
    // and the input: Where for numbers, And for booleans. Both copy the input's bits unchanged.
    private static string? BindExpand(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            return "Expand needs a tensor and a shape";
        }

        if (ctx.InputAnyElement(0, out DeviceElement element) is { } why)
        {
            return why;
        }

        if (!ctx.InputIsConst(1) || ctx.InputType(1) != DType.I64 || ctx.InputShape(1).Rank != 1)
        {
            return "the target shape is not a constant int64 vector, so the output shape is only known at run time";
        }

        if (ctx.OutputType(0) != ctx.InputType(0))
        {
            return "the planned output does not match the input";
        }

        Tensor shapeTensor = ctx.InputConst(1);
        ReadOnlySpan<long> wanted = shapeTensor.Data<long>();
        int[] input = GpuClaimContext.Dims(ctx.InputShape(0));
        int[] target = GpuClaimContext.Dims(ctx.OutputShape(0));
        if (input.Length > MaxRank || target.Length > MaxRank)
        {
            return $"a shape has rank above {MaxRank}";
        }

        // The target is the broadcast of the input shape and the shape input (a 1 in either keeps the other).
        int[] expected;
        try
        {
            int[] w = new int[wanted.Length];
            for (int i = 0; i < w.Length; i++)
            {
                w[i] = checked((int)wanted[i]);
            }

            expected = ElementwiseKernels.BroadcastShape(input, w);
        }
        catch (Exception e) when (e is GpuKernelShapeException or OverflowException)
        {
            return $"the shape does not broadcast with the input: {e.Message}";
        }

        if (!expected.AsSpan().SequenceEqual(target))
        {
            return "the planned output does not match the broadcast of the input and the shape";
        }

        if (target.Length > (element == DeviceElement.I64 ? 5 : 6))
        {
            return $"the target has {target.Length} dimensions, more than the kernels take for this element type";
        }

        long count = ctx.OutputShape(0).Length;
        if (count == 0 || ctx.InputShape(0).Length == 0)
        {
            return "an empty tensor";
        }

        if (ctx.FitsBinding(Math.Max(count, ctx.InputShape(0).Length), element) is { } big)
        {
            return big;
        }

        binding = new ExpandBinding(ctx, element, input, target);
        return null;
    }

    private sealed class ExpandBinding : GpuBinding
    {
        private readonly ElementwiseKernels _elementwise;
        private readonly IntegerKernels _integers;
        private readonly DeviceElement _element;
        private readonly int[] _input, _target;
        private readonly Operand _x;
        private readonly int _outId, _count;

        public ExpandBinding(GpuClaimContext ctx, DeviceElement element, int[] input, int[] target)
        {
            _elementwise = ctx.Backend.Elementwise;
            _integers = ctx.Backend.Integer;
            _element = element;
            _input = input;
            _target = target;
            _x = Operand.Of(ctx, 0);
            _outId = ctx.OutputId(0);
            _count = ctx.OutputShape(0).Length;
        }

        public override DeviceElement OutputElement(int output) => _element;

        public override long ScratchBoolElements => GpuKernelBuffers.PaddedBoolLength(_count);

        public override void Encode(GpuRunContext ctx)
        {
            GpuBuffer<byte> ones = ctx.ScratchBool(this, out int offOnes);
            _integers.EncodeFillBool(ctx.Encoder, ones, offOnes, _count, true);
            switch (_element)
            {
                case DeviceElement.F32:
                    Select<float>(ctx, ones, offOnes);
                    break;
                case DeviceElement.F16:
                    Select<Half>(ctx, ones, offOnes);
                    break;
                case DeviceElement.I32:
                    Select<int>(ctx, ones, offOnes);
                    break;
                case DeviceElement.I64:
                    Select<long>(ctx, ones, offOnes);
                    break;
                default:
                    GpuBuffer<byte> x = _x.Get<byte>(ctx, out int offX);
                    _integers.EncodeLogic(ctx.Encoder, GpuLogicOp.And, x, offX, _input, ones, offOnes, _target, ctx.Buffer<byte>(_outId, out int offY), offY);
                    break;
            }
        }

        private void Select<T>(GpuRunContext ctx, GpuBuffer<byte> ones, int offOnes) where T : unmanaged
        {
            GpuBuffer<T> x = _x.Get<T>(ctx, out int offX);
            GpuBuffer<T> y = ctx.Buffer<T>(_outId, out int offY);
            _elementwise.EncodeWhere(ctx.Encoder, ones, offOnes, _target, x, offX, _input, x, offX, _input, y, offY);
        }
    }

    // ---- generators -------------------------------------------------------------------------------------------------------

    // Range(start, limit, delta) with constant scalar inputs: the length is fixed when the plan is made, so the kernel only fills.
    private static string? BindRange(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        if (ctx.InputCount != 3 || !ctx.HasInput(0) || !ctx.HasInput(1) || !ctx.HasInput(2))
        {
            return "Range needs start, limit and delta";
        }

        for (int i = 0; i < 3; i++)
        {
            if (!ctx.InputIsConst(i) || ctx.InputShape(i).Length != 1)
            {
                return $"input {i} is not a constant scalar, so the length of the range is only known at run time";
            }
        }

        DType type = ctx.InputType(0);
        if (type is not (DType.F32 or DType.I32 or DType.I64) || ctx.InputType(1) != type || ctx.InputType(2) != type || ctx.OutputType(0) != type)
        {
            return $"Range on {type}: the GPU kernel takes float32, int32 and int64 with one element type";
        }

        long length = ctx.OutputShape(0).Length;
        if (length == 0 || ctx.OutputShape(0).Rank != 1)
        {
            return "an empty range";
        }

        Tensor s = ctx.InputConst(0), l = ctx.InputConst(1), d = ctx.InputConst(2);
        try
        {
            switch (type)
            {
                case DType.F32:
                    {
                        float start = s.Data<float>()[0], delta = d.Data<float>()[0];
                        if (IntegerKernels.RangeLength(start, l.Data<float>()[0], delta) != length)
                        {
                            return "the planned length does not match the range";
                        }

                        binding = new RangeBinding<float>(ctx, start, delta, DeviceElement.F32, (int)length);
                        break;
                    }

                case DType.I32:
                    {
                        int start = s.Data<int>()[0], delta = d.Data<int>()[0];
                        if (IntegerKernels.RangeLength(start, l.Data<int>()[0], delta) != length)
                        {
                            return "the planned length does not match the range";
                        }

                        binding = new RangeBinding<int>(ctx, start, delta, DeviceElement.I32, (int)length);
                        break;
                    }

                default:
                    {
                        long start = s.Data<long>()[0], delta = d.Data<long>()[0];
                        if (IntegerKernels.RangeLength(start, l.Data<long>()[0], delta) != length)
                        {
                            return "the planned length does not match the range";
                        }

                        binding = new RangeBinding<long>(ctx, start, delta, DeviceElement.I64, (int)length);
                        break;
                    }
            }
        }
        catch (GpuKernelShapeException e)
        {
            return $"the range is not valid: {e.Message}";
        }

        return null;
    }

    private sealed class RangeBinding<T> : GpuBinding where T : unmanaged
    {
        private readonly IntegerKernels _kernels;
        private readonly T _start, _delta;
        private readonly DeviceElement _element;
        private readonly int _outId, _count;

        public RangeBinding(GpuClaimContext ctx, T start, T delta, DeviceElement element, int count)
        {
            _kernels = ctx.Backend.Integer;
            _start = start;
            _delta = delta;
            _element = element;
            _outId = ctx.OutputId(0);
            _count = count;
        }

        public override DeviceElement OutputElement(int output) => _element;

        public override void Encode(GpuRunContext ctx) =>
            _kernels.EncodeRange(ctx.Encoder, ctx.Buffer<T>(_outId, out int off), off, _count, _start, _delta);
    }

    // ConstantOfShape with a constant shape: a fill with the value attribute (default float32 zero).
    private static string? BindConstantOfShape(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0) || !ctx.InputIsConst(0))
        {
            return "the shape is not a constant, so the output shape is only known at run time";
        }

        if (ctx.OutputShape(0).Length == 0)
        {
            return "an empty tensor";
        }

        if (ctx.OutputAnyElement(0, out DeviceElement element) is { } why)
        {
            return why;
        }

        if (ctx.FitsBinding(ctx.OutputShape(0).Length, element) is { } big)
        {
            return big;
        }

        // The kernel keeps the value it was built with; the model's attribute is gone once the session is made.
        if (ctx.Step.Node.Kernel is not ConstantOfShapeOp fill)
        {
            return "the operator is not the standard ConstantOfShape";
        }

        Tensor value = fill.FillValue;
        if (value.DType != ctx.OutputType(0) || value.Length != 1)
        {
            return "the value attribute does not match the planned output";
        }

        binding = value.DType switch
        {
            DType.F32 => new FillBinding<float>(ctx, value.Data<float>()[0], element),
            DType.F16 => new FillBinding<Half>(ctx, value.Data<Half>()[0], element),
            DType.I32 => new FillBinding<int>(ctx, value.Data<int>()[0], element),
            DType.I64 => new FillBinding<long>(ctx, value.Data<long>()[0], element),
            _ => new FillBinding<bool>(ctx, value.Data<bool>()[0], element),
        };
        return null;
    }

    private sealed class FillBinding<T> : GpuBinding where T : unmanaged
    {
        private readonly IntegerKernels _kernels;
        private readonly T _value;
        private readonly DeviceElement _element;
        private readonly int _outId, _count;

        public FillBinding(GpuClaimContext ctx, T value, DeviceElement element)
        {
            _kernels = ctx.Backend.Integer;
            _value = value;
            _element = element;
            _outId = ctx.OutputId(0);
            _count = ctx.OutputShape(0).Length;
        }

        public override DeviceElement OutputElement(int output) => _element;

        public override void Encode(GpuRunContext ctx)
        {
            if (_element == DeviceElement.Bool)
            {
                _kernels.EncodeFillBool(ctx.Encoder, ctx.Buffer<byte>(_outId, out int offB), offB, _count, (bool)(object)_value);
            }
            else
            {
                _kernels.EncodeFill(ctx.Encoder, ctx.Buffer<T>(_outId, out int off), off, _count, _value);
            }
        }
    }
}
