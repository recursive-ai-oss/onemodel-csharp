using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Passes;

/// <summary>
/// A matrix product with constant weights, made by <see cref="LinearFusionPass"/> from MatMul or Gemm (plus an optional bias Add
/// and an optional activation). Y = act(A * B + bias) with A of shape [..., K] and B of shape [K, N] packed once into the kernels'
/// panel layout, so a run never repacks the weights; the bias and activation are applied to each finished tile of Y.
/// </summary>
internal sealed class LinearOp : OpKernel
{
    private readonly Tensor _packed;
    private readonly int _k;
    private readonly int _n;
    private readonly int _biasRank;
    private readonly KernelPath _path;
    private readonly BiasActivationEpilogue? _epilogue;

    public LinearOp(Tensor packedB, int k, int n, float[]? bias, int biasRank, FusedActivation? activation, KernelPath path)
    {
        _packed = packedB;
        _k = k;
        _n = n;
        _biasRank = biasRank;
        _path = path;
        _epilogue = bias is null && activation is null ? null : new BiasActivationEpilogue(bias, false, 0, activation, path);
    }

    public override bool SupportsExact => true;

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

        Shape a = ctx.InputShape(0);
        if (a.Rank == 0 || a[a.Rank - 1] != _k)
        {
            throw ctx.Fail($"the last dimension of {a} must be {_k}");
        }

        // MatMul: a 1-D A gives a 1-D result. A bias of higher rank than that result would have broadcast it up.
        int rank = Math.Max(a.Rank, _biasRank);
        Span<int> d = stackalloc int[Shape.MaxRank];
        for (int i = 0; i < rank; i++)
        {
            d[i] = 1;
        }

        for (int i = 0; i < a.Rank - 1; i++)
        {
            d[rank - a.Rank + i] = a[i];
        }

        d[rank - 1] = _n;
        ctx.SetOutput(0, DType.F32, Shape.Create(d.Slice(0, rank)));
    }

    public override void Run(in OpRun run)
    {
        Tensor a = run.Inputs[0], y = run.Outputs[0];
        if (run.Path != _path)
        {
            throw new InvalidOperationException("Weights were packed for a different instruction set than the one this run uses.");
        }

        int m = a.Length / _k;
        Gemm.RunPacked(run.KernelMode, run.Path, m, _n, _k, a.Data<float>(), _k, _packed.Data<float>(), y.MutableData<float>(), _n, run.Threads, _epilogue);
    }

    public override void Dispose()
    {
        _packed.Dispose();
        base.Dispose();
    }
}
