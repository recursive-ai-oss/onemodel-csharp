using System;
using System.Collections.Generic;
using System.Threading;
namespace RecursiveAI.OneModel.Tensors;

/// <summary>
/// Eager operations on float32 tensors. Each allocates its result (from the pool argument, or <see cref="BufferPool.Shared"/>)
/// and runs the kernel set argument (or <see cref="KernelSet.Default"/>). When gradients are being tracked and an
/// input requires them, the result carries a <see cref="BackwardRecord"/> whose contents are listed per operation below and in
/// <c>docs/tensors.md</c>; otherwise nothing is recorded and nothing extra is allocated.
/// </summary>
public static partial class Ops
{
    /// <summary><c>a + b</c> with broadcasting. Record: no saved tensors; backward sums the output gradient down to each input's shape.</summary>
    public static Tensor Add(in Tensor a, in Tensor b, BufferPool? pool = null, ITensorKernels? kernels = null)
    {
        Tensor r = Binary(BinaryOp.Add, a, b, pool, kernels);
        return GradMode.ShouldRecord(a, b) ? BackwardRecord.Create(OpKind.Add, 0, a, b).Attach(r) : r;
    }

    /// <summary><c>a - b</c> with broadcasting. Record: as <see cref="Add"/>; the gradient of b is negated.</summary>
    public static Tensor Sub(in Tensor a, in Tensor b, BufferPool? pool = null, ITensorKernels? kernels = null)
    {
        Tensor r = Binary(BinaryOp.Sub, a, b, pool, kernels);
        return GradMode.ShouldRecord(a, b) ? BackwardRecord.Create(OpKind.Sub, 0, a, b).Attach(r) : r;
    }

    /// <summary>
    /// <c>a * b</c> with broadcasting. Record: slot 0 holds <c>a</c> when <c>b</c> requires a gradient, slot 1 holds <c>b</c> when
    /// <c>a</c> does (each gradient is the other operand times the output gradient, summed to the input's shape).
    /// </summary>
    public static Tensor Mul(in Tensor a, in Tensor b, BufferPool? pool = null, ITensorKernels? kernels = null)
    {
        Tensor r = Binary(BinaryOp.Mul, a, b, pool, kernels);
        if (!GradMode.ShouldRecord(a, b))
        {
            return r;
        }

        BackwardRecord rec = BackwardRecord.Create(OpKind.Mul, 2, a, b);
        if (b.RequiresGrad)
        {
            rec.Save(0, a);
        }

        if (a.RequiresGrad)
        {
            rec.Save(1, b);
        }

        return rec.Attach(r);
    }

    /// <summary>
    /// <c>a / b</c> with broadcasting. Record: slot 0 holds <c>a</c> when <c>b</c> requires a gradient; slot 1 holds <c>b</c> when
    /// either does.
    /// </summary>
    public static Tensor Div(in Tensor a, in Tensor b, BufferPool? pool = null, ITensorKernels? kernels = null)
    {
        Tensor r = Binary(BinaryOp.Div, a, b, pool, kernels);
        if (!GradMode.ShouldRecord(a, b))
        {
            return r;
        }

        BackwardRecord rec = BackwardRecord.Create(OpKind.Div, 2, a, b);
        if (b.RequiresGrad)
        {
            rec.Save(0, a);
        }

        rec.Save(1, b);
        return rec.Attach(r);
    }

    /// <summary><c>-x</c>. Record: nothing saved.</summary>
    public static Tensor Neg(in Tensor x, BufferPool? pool = null, ITensorKernels? kernels = null)
    {
        Tensor r = Unary(UnaryOp.Neg, x, pool, kernels);
        return GradMode.ShouldRecord(x) ? BackwardRecord.Create(OpKind.Neg, 0, x).Attach(r) : r;
    }

    /// <summary><c>max(x, 0)</c>. Record: slot 0 holds the output (the gradient passes where it is positive).</summary>
    public static Tensor Relu(in Tensor x, BufferPool? pool = null, ITensorKernels? kernels = null)
    {
        Tensor r = Unary(UnaryOp.Relu, x, pool, kernels);
        if (!GradMode.ShouldRecord(x))
        {
            return r;
        }

        BackwardRecord rec = BackwardRecord.Create(OpKind.Relu, 1, x);
        rec.Save(0, r);
        return rec.Attach(r);
    }

    /// <summary>
    /// Matrix product of <c>a[m,k]</c> and <c>b[k,n]</c> (either may be a transposed or sliced view, no copy is made).
    /// Record: slot 0 holds <c>a</c> when <c>b</c> requires a gradient (dB = aᵀ·g), slot 1 holds <c>b</c> when <c>a</c> does
    /// (dA = g·bᵀ); the transposes are views of the saved tensors.
    /// </summary>
    /// <exception cref="TensorShapeException">Not rank 2, or inner extents differ.</exception>
    public static Tensor MatMul(in Tensor a, in Tensor b, BufferPool? pool = null, ITensorKernels? kernels = null)
    {
        RequireFloat32(a, b);
        if (a.Rank != 2 || b.Rank != 2 || a.Shape[1] != b.Shape[0])
        {
            throw new TensorShapeException($"Cannot multiply {a.Shape} by {b.Shape}.");
        }

        Tensor r = NewOut(Shape.Create(a.Shape[0], b.Shape[1]), pool);
        (kernels ?? KernelSet.Default).MatMul(a, b, r);
        if (!GradMode.ShouldRecord(a, b))
        {
            return r;
        }

        BackwardRecord rec = BackwardRecord.Create(OpKind.MatMul, 2, a, b);
        if (b.RequiresGrad)
        {
            rec.Save(0, a);
        }

        if (a.RequiresGrad)
        {
            rec.Save(1, b);
        }

        return rec.Attach(r);
    }

    /// <summary>
    /// Max pooling over an NCHW float32 input. Record: slot 0 holds an Int32 tensor of the winning flat input indices (owned by the
    /// record), Ints = [kernelH, kernelW, strideH, strideW, padH, padW]; backward scatters each output gradient to its index and
    /// needs the input shape from <see cref="BackwardRecord.InputShape"/>. When nothing is recorded the indices are not computed.
    /// </summary>
    public static Tensor MaxPool2d(in Tensor x, Pool2dParams p, BufferPool? pool = null, ITensorKernels? kernels = null)
    {
        RequireFloat32(x);
        if (x.Rank != 4)
        {
            throw new TensorShapeException($"MaxPool2d needs an (N,C,H,W) input, not {x.Shape}.");
        }

        Shape os = Shape.Create(
            x.Shape[0], x.Shape[1],
            Pool2dParams.OutputSize(x.Shape[2], p.KernelH, p.StrideH, p.PadH),
            Pool2dParams.OutputSize(x.Shape[3], p.KernelW, p.StrideW, p.PadW));
        Tensor r = NewOut(os, pool);
        bool track = GradMode.ShouldRecord(x);
        Tensor idx = track ? Tensor.Create(DType.I32, os, pool) : default;
        (kernels ?? KernelSet.Default).MaxPool2d(x, p, r, idx);
        if (!track)
        {
            return r;
        }

        BackwardRecord rec = BackwardRecord.Create(OpKind.MaxPool2d, 1, x);
        rec.Save(0, idx, owned: true);
        rec.SetInts(p.KernelH, p.KernelW, p.StrideH, p.StrideW, p.PadH, p.PadW);
        return rec.Attach(r);
    }

    private static Tensor Binary(BinaryOp op, in Tensor a, in Tensor b, BufferPool? pool, ITensorKernels? kernels)
    {
        RequireFloat32(a, b);
        Shape s = Shape.Broadcast(a.Shape, b.Shape);
        Tensor r = NewOut(s, pool);
        using (GradMode.NoGrad())
        {
            (kernels ?? KernelSet.Default).Binary(op, a.Expand(s), b.Expand(s), r);
        }

        return r;
    }

    private static Tensor Unary(UnaryOp op, in Tensor x, BufferPool? pool, ITensorKernels? kernels)
    {
        RequireFloat32(x);
        Tensor r = NewOut(x.Shape, pool);
        (kernels ?? KernelSet.Default).Unary(op, x, r);
        return r;
    }

    // Allocates a result tensor; inside an ActivationScope it is remembered so the scope can free it.
    internal static Tensor NewOut(Shape shape, BufferPool? pool, bool zero = false)
    {
        Tensor t = Tensor.Create(DType.F32, shape, pool, zero);
        ActivationScope.Track(t);
        return t;
    }

    private static void RequireFloat32(in Tensor a)
    {
        if (a.IsEmpty || a.DType != DType.F32 || a.IsBlocked)
        {
            throw new NotSupportedException("The eager operations take plain float32 tensors.");
        }
    }

    private static void RequireFloat32(in Tensor a, in Tensor b)
    {
        RequireFloat32(a);
        RequireFloat32(b);
    }
}
