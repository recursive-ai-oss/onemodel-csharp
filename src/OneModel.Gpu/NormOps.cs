using System;
using System.Collections.Generic;
using RecursiveAI.OneGpu;
using RecursiveAI.OneGpu.Kernels;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu;

/// <summary>
/// The entries for operators that reduce along axes: LayerNormalization, Softmax and the Reduce family. Each has a deterministic variant
/// (a fixed reduction tree and no atomics), which exact mode uses.
/// </summary>
internal static class NormOps
{
    public static IEnumerable<GpuOpEntry> Entries
    {
        get
        {
            yield return new GpuOpEntry("LayerNormalization", true, CheckLayerNorm, BindLayerNorm);
            yield return new GpuOpEntry("Softmax", true, CheckSoftmax, BindSoftmax);
            foreach ((string name, GpuReduceOp op) in new (string, GpuReduceOp)[]
            {
                ("ReduceSum", GpuReduceOp.Sum), ("ReduceMean", GpuReduceOp.Mean), ("ReduceMax", GpuReduceOp.Max), ("ReduceMin", GpuReduceOp.Min),
                ("ReduceProd", GpuReduceOp.Prod), ("ReduceL1", GpuReduceOp.L1), ("ReduceL2", GpuReduceOp.L2),
                ("ReduceSumSquare", GpuReduceOp.SumSquare), ("ReduceLogSum", GpuReduceOp.LogSum),
            })
            {
                GpuReduceOp captured = op;
                yield return new GpuOpEntry(name, true, _ => null, (GpuClaimContext ctx, out GpuBinding? b) => BindReduce(ctx, captured, out b));
            }
        }
    }

    // ---- LayerNormalization (ONNX opset 17) ----------------------------------------------------------------------------------

    private static string? CheckLayerNorm(NodeInfo info) => info.Opset < 17 ? "LayerNormalization exists from opset 17" : null;

    private static string? BindLayerNorm(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            return "needs X and Scale";
        }

        if (ctx.InputElement(0, out DeviceElement element) is { } why)
        {
            return why;
        }

        for (int i = 1; i < 3; i++)
        {
            if (ctx.HasInput(i) && ctx.InputType(i) != ctx.InputType(0))
            {
                return $"input {i} is {ctx.InputType(i)} and X is {ctx.InputType(0)}; the GPU kernel needs one element type";
            }
        }

        int[] outs = ctx.Step.Node.Outputs;
        if ((outs.Length > 1 && outs[1] >= 0) || (outs.Length > 2 && outs[2] >= 0))
        {
            return "its Mean or InvStdDev output is read, and the GPU kernel returns only the normalized tensor";
        }

        Shape x = ctx.InputShape(0);
        int axis = (int)ctx.Info.GetInt("axis", -1);
        if (axis < -x.Rank || axis >= x.Rank)
        {
            return $"axis {axis} is outside the input's rank {x.Rank}";
        }

        axis = axis < 0 ? axis + x.Rank : axis;
        long rows = 1, length = 1;
        for (int i = 0; i < axis; i++)
        {
            rows *= x[i];
        }

        for (int i = axis; i < x.Rank; i++)
        {
            length *= x[i];
        }

        for (int i = 1; i < 3; i++)
        {
            if (ctx.HasInput(i) && ctx.InputShape(i).Length != length)
            {
                return $"input {i} of shape {ctx.InputShape(i)} does not have one value per normalized element ({length})";
            }
        }

        if (length == 0 || rows == 0 || x.Length >= int.MaxValue)
        {
            return "the input is empty or too large";
        }

        if (element == DeviceElement.F16 && length % 2 != 0)
        {
            return "a float16 row of odd length (the kernel writes rows as 32-bit words)";
        }

        if (ctx.OutputShape(0).Length != x.Length || ctx.OutputType(0) != ctx.InputType(0))
        {
            return "the planned output does not match the input";
        }

        if (ctx.FitsBinding(x.Length, element) is { } big)
        {
            return big;
        }

        binding = new LayerNormBinding(ctx, element, (int)rows, (int)length, ctx.Info.GetFloat("epsilon", 1e-5f));
        return null;
    }

    private sealed class LayerNormBinding : GpuBinding
    {
        private readonly NormKernels _kernels;
        private readonly DeviceElement _element;
        private readonly int _rows, _length;
        private readonly float _epsilon;
        private readonly Operand _x, _scale;
        private readonly Operand? _bias;
        private readonly int _outId;
        private readonly GpuKernelMode _mode;

        public LayerNormBinding(GpuClaimContext ctx, DeviceElement element, int rows, int length, float epsilon)
        {
            _kernels = ctx.Backend.Norm;
            _element = element;
            _rows = rows;
            _length = length;
            _epsilon = epsilon;
            _x = Operand.Of(ctx, 0);
            _scale = Operand.Of(ctx, 1);
            _bias = ctx.HasInput(2) ? Operand.Of(ctx, 2) : null;
            _outId = ctx.OutputId(0);
            _mode = ctx.Backend.KernelMode;
            Flops = 8.0 * rows * length;
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
            GpuBuffer<T> scale = _scale.Get<T>(ctx, out int offScale);
            GpuBuffer<T>? bias = null;
            int offBias = 0;
            if (_bias is { } b)
            {
                bias = b.Get<T>(ctx, out offBias);
            }

            GpuBuffer<T> y = ctx.Buffer<T>(_outId, out int offY);
            _kernels.Encode(ctx.Encoder, false, x, offX, _rows, _length, scale, offScale, bias, offBias, y, offY, _epsilon, _mode);
        }
    }

    // ---- Softmax (ONNX opset 13) ---------------------------------------------------------------------------------------------

    // Before opset 13 the input is flattened to two dimensions around the axis, with another default axis; not worth a second form.
    private static string? CheckSoftmax(NodeInfo info) => info.Opset < 13 ? "Softmax before opset 13 flattens its input to 2-D around the axis" : null;

    private static string? BindSoftmax(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0))
        {
            return "needs an input";
        }

        if (ctx.InputElement(0, out DeviceElement element) is { } why)
        {
            return why;
        }

        Shape x = ctx.InputShape(0);
        if (x.Rank == 0)
        {
            return "a scalar input";
        }

        int axis = (int)ctx.Info.GetInt("axis", -1);
        if (axis < -x.Rank || axis >= x.Rank)
        {
            return $"axis {axis} is outside the input's rank {x.Rank}";
        }

        axis = axis < 0 ? axis + x.Rank : axis;
        if (element == DeviceElement.F16 && (axis != x.Rank - 1 || x[axis] % 2 != 0))
        {
            return "a float16 softmax needs the last axis and an even row length (the kernel writes rows as 32-bit words)";
        }

        if (x.Length == 0 || ctx.OutputShape(0).Length != x.Length || ctx.OutputType(0) != ctx.InputType(0))
        {
            return "the input is empty, or the planned output does not match it";
        }

        if (ctx.FitsBinding(x.Length, element) is { } big)
        {
            return big;
        }

        binding = new SoftmaxBinding(ctx, element, GpuClaimContext.Dims(x), axis);
        return null;
    }

    private sealed class SoftmaxBinding : GpuBinding
    {
        private readonly SoftmaxKernels _kernels;
        private readonly DeviceElement _element;
        private readonly int[] _shape;
        private readonly int _axis;
        private readonly Operand _x;
        private readonly int _outId;
        private readonly GpuKernelMode _mode;

        public SoftmaxBinding(GpuClaimContext ctx, DeviceElement element, int[] shape, int axis)
        {
            _kernels = ctx.Backend.Softmax;
            _element = element;
            _shape = shape;
            _axis = axis;
            _x = Operand.Of(ctx, 0);
            _outId = ctx.OutputId(0);
            _mode = ctx.Backend.KernelMode;
            Flops = 5.0 * ctx.InputShape(0).Length;
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
            _kernels.Encode(ctx.Encoder, x, offX, _shape, _axis, y, offY, false, _mode);
        }
    }

    // ---- Reduce* ---------------------------------------------------------------------------------------------------------------

    // The axes come from an input (opset 13 for ReduceSum, 18 for the others) or from an attribute (before); none means all of them,
    // unless noop_with_empty_axes asks for the identity. keepdims only changes the shape the result is labelled with: the elements are the same.
    private static string? BindReduce(GpuClaimContext ctx, GpuReduceOp op, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0))
        {
            return "needs an input";
        }

        if (ctx.InputElement(0, out DeviceElement element) is { } why)
        {
            return why;
        }

        Shape x = ctx.InputShape(0);
        if (x.Rank == 0 || x.Length == 0)
        {
            return "a scalar or empty input";
        }

        int[] axes;
        if (ctx.HasInput(1))
        {
            if (!ctx.InputIsConst(1) || ctx.InputType(1) != DType.I64)
            {
                return "the axes are not a constant int64 tensor";
            }

            ReadOnlySpan<long> a = ctx.InputConst(1).Data<long>();
            axes = new int[a.Length];
            for (int i = 0; i < axes.Length; i++)
            {
                axes[i] = (int)a[i];
            }
        }
        else if (ctx.Info.GetInts("axes") is { Length: > 0 } attr)
        {
            axes = new int[attr.Length];
            for (int i = 0; i < axes.Length; i++)
            {
                axes[i] = (int)attr[i];
            }
        }
        else
        {
            axes = [];
        }

        if (axes.Length == 0)
        {
            if (ctx.Info.GetInt("noop_with_empty_axes", 0) != 0)
            {
                return "noop_with_empty_axes makes it an identity, which has no kernel";
            }

            axes = new int[x.Rank];
            for (int i = 0; i < axes.Length; i++)
            {
                axes[i] = i;
            }
        }

        foreach (int axis in axes)
        {
            if (axis < -x.Rank || axis >= x.Rank)
            {
                return $"axis {axis} is outside the input's rank {x.Rank}";
            }
        }

        if (ctx.OutputType(0) != ctx.InputType(0))
        {
            return "the planned output type differs from the input's";
        }

        if (ctx.FitsBinding(x.Length, element) is { } big)
        {
            return big;
        }

        binding = new ReduceBinding(ctx, op, element, GpuClaimContext.Dims(x), axes);
        return null;
    }

    private sealed class ReduceBinding : GpuBinding
    {
        private readonly ReduceKernels _kernels;
        private readonly GpuReduceOp _op;
        private readonly DeviceElement _element;
        private readonly int[] _shape;
        private readonly int[] _axes;
        private readonly Operand _x;
        private readonly int _outId;
        private readonly GpuKernelMode _mode;

        public ReduceBinding(GpuClaimContext ctx, GpuReduceOp op, DeviceElement element, int[] shape, int[] axes)
        {
            _kernels = ctx.Backend.Reduce;
            _op = op;
            _element = element;
            _shape = shape;
            _axes = axes;
            _x = Operand.Of(ctx, 0);
            _outId = ctx.OutputId(0);
            _mode = ctx.Backend.KernelMode;
            Flops = ctx.InputShape(0).Length;
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
            _kernels.EncodeReduce(ctx.Encoder, _op, x, offX, _shape, _axes, y, offY, _mode);
        }
    }
}
