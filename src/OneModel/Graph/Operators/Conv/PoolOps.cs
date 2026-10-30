using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

/// <summary>Shape rules shared by MaxPool and AveragePool: attributes, output size, and the lifted geometry the kernels take.</summary>
internal abstract class WindowPoolKernel : OpKernel
{
    private readonly WindowAttrs _attrs;
    private readonly bool _ceilMode;

    protected WindowPoolKernel(NodeInfo node, bool hasDilations)
    {
        _attrs = new WindowAttrs(node, hasDilations);
        _ceilMode = node.GetInt("ceil_mode", 0) != 0;
        if (!_attrs.HasKernelShape)
        {
            throw node.Error("attribute kernel_shape is required");
        }
    }

    public override bool SupportsExact => true;

    /// <summary>Resolves the geometry for input shape <paramref name="x"/> (N, C, spatial...). Spatial axes are lifted to three.</summary>
    protected PoolGeom Geometry(Shape x, InferContext? ctx)
    {
        int r = x.Rank - 2;
        Span<int> kernel = stackalloc int[3], strides = stackalloc int[3], dil = stackalloc int[3], pads = stackalloc int[6];
        _attrs.Resolve(ctx, r, default, kernel, strides, dil, pads);
        Span<int> inD = stackalloc int[3], kD = stackalloc int[3], sD = stackalloc int[3], dD = stackalloc int[3];
        Span<int> pb = stackalloc int[3], pe = stackalloc int[3], oD = stackalloc int[3];
        for (int a = 0; a < 3; a++)
        {
            int j = a - (3 - r);
            if (j < 0)
            {
                inD[a] = 1;
                kD[a] = 1;
                sD[a] = 1;
                dD[a] = 1;
                oD[a] = 1;
                continue;
            }

            int b = pads[j], e = pads[j + r];
            inD[a] = x[2 + j];
            kD[a] = kernel[j];
            sD[a] = strides[j];
            dD[a] = dil[j];
            oD[a] = _attrs.OutputExtent(ctx, j, inD[a], kD[a], sD[a], dD[a], _ceilMode, ref b, ref e);
            pb[a] = b;
            pe[a] = e;
        }

        return new PoolGeom(new Dim3(inD[0], inD[1], inD[2]), new Dim3(kD[0], kD[1], kD[2]), new Dim3(sD[0], sD[1], sD[2]),
            new Dim3(dD[0], dD[1], dD[2]), new Dim3(pb[0], pb[1], pb[2]), new Dim3(pe[0], pe[1], pe[2]), new Dim3(oD[0], oD[1], oD[2]));
    }

    /// <summary>Checks the input rank and states the output shape; returns the geometry.</summary>
    protected PoolGeom InferShape(InferContext ctx, out Shape outShape)
    {
        Shape x = ctx.InputShape(0);
        if (x.Rank < 3 || x.Rank > 5)
        {
            throw ctx.Fail($"the input {x} must have 1, 2 or 3 spatial axes (rank 3 to 5)");
        }

        PoolGeom g = Geometry(x, ctx);
        Span<int> d = stackalloc int[Shape.MaxRank];
        d[0] = x[0];
        d[1] = x[1];
        int r = x.Rank - 2;
        Dim3 o = g.Out;
        for (int j = 0; j < r; j++)
        {
            d[2 + j] = (3 - r + j) switch { 0 => o.D, 1 => o.H, _ => o.W };
        }

        outShape = Shape.Create(d.Slice(0, x.Rank));
        return g;
    }
}

/// <summary>
/// MaxPool over 1 to 3 spatial axes with padding, dilation, strides and ceil mode; the optional second output holds, for each result,
/// the flat index of the chosen input element over the whole input tensor. NaN handling follows ONNX Runtime (docs/ops-conv.md).
/// </summary>
internal sealed class MaxPoolOp : WindowPoolKernel
{
    public static readonly OpDef Def = new("MaxPool", 1, int.MaxValue, static n => new MaxPoolOp(n));

    private readonly bool _columnMajor;

    private MaxPoolOp(NodeInfo node) : base(node, hasDilations: true)
    {
        _columnMajor = node.GetInt("storage_order", 0) != 0;
    }

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        DType t = ctx.InputType(0);
        if (t is not (DType.F32 or DType.F64 or DType.I8 or DType.U8))
        {
            throw ctx.Unsupported($"element type {t} is not supported yet");
        }

        _ = InferShape(ctx, out Shape o);
        ctx.SetOutput(0, t, o);
        if (ctx.OutputCount > 1)
        {
            ctx.SetOutput(1, DType.I64, o);
        }
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], y = run.Outputs[0];
        if (y.Length == 0)
        {
            return;
        }

        PoolGeom g = Geometry(x.Shape, null);
        long planes = (long)x.Shape[0] * x.Shape[1];
        Span<long> idx = run.Outputs.Length > 1 && !run.Outputs[1].IsEmpty ? run.Outputs[1].MutableData<long>() : default;
        switch (y.DType)
        {
            case DType.F32:
                if (idx.IsEmpty && MaxPoolFast.Supports(in g))
                {
                    MaxPoolFast.Run(in g, planes, x.Data<float>(), y.MutableData<float>(), run.Threads);
                }
                else
                {
                    Pooling.Max(g, planes, x.Data<float>(), y.MutableData<float>(), idx, _columnMajor);
                }

                if (run.Exact)
                {
                    Pooling.CanonicalizeNaN(y.MutableData<float>());
                }

                break;
            case DType.F64:
                Pooling.Max(g, planes, x.Data<double>(), y.MutableData<double>(), idx, _columnMajor);
                break;
            case DType.I8:
                if (idx.IsEmpty && Int8Pool.Supports(in g))
                {
                    Int8Pool.Run(in g, planes, Int8Quantize.AsBytes(x.Data<sbyte>()), Int8Quantize.AsBytes(y.MutableData<sbyte>()), true, run.Threads);
                }
                else
                {
                    Pooling.Max(g, planes, x.Data<sbyte>(), y.MutableData<sbyte>(), idx, _columnMajor);
                }

                break;
            case DType.U8:
                if (idx.IsEmpty && Int8Pool.Supports(in g))
                {
                    Int8Pool.Run(in g, planes, x.Data<byte>(), y.MutableData<byte>(), false, run.Threads);
                }
                else
                {
                    Pooling.Max(g, planes, x.Data<byte>(), y.MutableData<byte>(), idx, _columnMajor);
                }

                break;
            default:
                throw new InvalidOperationException($"Element type {y.DType} reached the kernel although inference rejects it.");
        }
    }
}

/// <summary>AveragePool over 1 to 3 spatial axes: padding, dilation (opset 19), strides, ceil mode and count_include_pad.</summary>
internal sealed class AveragePoolOp : WindowPoolKernel
{
    public static readonly OpDef Def = new("AveragePool", 1, int.MaxValue, static n => new AveragePoolOp(n));

    private readonly bool _countIncludePad;

    private AveragePoolOp(NodeInfo node) : base(node, hasDilations: node.Opset >= 19)
    {
        _countIncludePad = node.GetInt("count_include_pad", 0) != 0;
    }

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        if (ctx.InputType(0) != DType.F32)
        {
            throw ctx.Unsupported($"element type {ctx.InputType(0)} is not supported yet (float32 only)");
        }

        _ = InferShape(ctx, out Shape o);
        ctx.SetOutput(0, DType.F32, o);
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], y = run.Outputs[0];
        if (y.Length == 0)
        {
            return;
        }

        PoolGeom g = Geometry(x.Shape, null);
        Pooling.Average(g, (long)x.Shape[0] * x.Shape[1], x.Data<float>(), y.MutableData<float>(), _countIncludePad);
        if (run.Exact)
        {
            Pooling.CanonicalizeNaN(y.MutableData<float>());
        }
    }
}

/// <summary>GlobalAveragePool: the mean over all spatial positions of each channel; output N x C x 1 x ... x 1.</summary>
internal sealed class GlobalAveragePoolOp : OpKernel
{
    public static readonly OpDef Def = new("GlobalAveragePool", 1, int.MaxValue, static _ => new GlobalAveragePoolOp());

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx) => GlobalShape.Infer(ctx);

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], y = run.Outputs[0];
        if (y.Length == 0)
        {
            return;
        }

        int plane = x.Length / (x.Shape[0] * x.Shape[1]);
        Pooling.GlobalAverage(x.Data<float>(), y.MutableData<float>(), plane);
        if (run.Exact)
        {
            Pooling.CanonicalizeNaN(y.MutableData<float>());
        }
    }
}

/// <summary>GlobalMaxPool: the maximum over all spatial positions of each channel; output N x C x 1 x ... x 1.</summary>
internal sealed class GlobalMaxPoolOp : OpKernel
{
    public static readonly OpDef Def = new("GlobalMaxPool", 1, int.MaxValue, static _ => new GlobalMaxPoolOp());

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx) => GlobalShape.Infer(ctx);

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], y = run.Outputs[0];
        if (y.Length == 0)
        {
            return;
        }

        int plane = x.Length / (x.Shape[0] * x.Shape[1]);
        Pooling.GlobalMax(x.Data<float>(), y.MutableData<float>(), plane);
        if (run.Exact)
        {
            Pooling.CanonicalizeNaN(y.MutableData<float>());
        }
    }
}

internal static class GlobalShape
{
    public static void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        if (ctx.InputType(0) != DType.F32)
        {
            throw ctx.Unsupported($"element type {ctx.InputType(0)} is not supported yet (float32 only)");
        }

        Shape x = ctx.InputShape(0);
        if (x.Rank < 2)
        {
            throw ctx.Fail($"the input {x} must have at least 2 dimensions");
        }

        Span<int> d = stackalloc int[Shape.MaxRank];
        d[0] = x[0];
        d[1] = x[1];
        for (int i = 2; i < x.Rank; i++)
        {
            d[i] = 1;
            if (x[i] == 0)
            {
                throw ctx.Fail($"the input {x} has an empty spatial axis");
            }
        }

        ctx.SetOutput(0, DType.F32, Shape.Create(d.Slice(0, x.Rank)));
    }
}
