using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

/// <summary>
/// BatchNormalization in inference form: y = (x - mean) / sqrt(var + epsilon) * scale + bias per channel (axis 1). Training mode
/// (batch statistics and the running-average outputs) and per-element statistics (<c>spatial = 0</c>, opset 7 only) are not supported.
/// </summary>
internal sealed class BatchNormalizationOp : OpKernel
{
    public static readonly OpDef Def = new("BatchNormalization", 7, int.MaxValue, static n => new BatchNormalizationOp(n));

    private readonly float _epsilon;

    private BatchNormalizationOp(NodeInfo node)
    {
        _epsilon = node.GetFloat("epsilon", 1e-5f);
        if (node.GetInt("training_mode", 0) != 0)
        {
            throw node.Unsupported("training_mode = 1 is not supported");
        }

        if (node.Opset < 9 && node.GetInt("spatial", 1) == 0)
        {
            throw node.Unsupported("spatial = 0 (per-element statistics) is not supported");
        }
    }

    public override bool SupportsExact => true;

    public override int InPlaceInput => 0;

    public override void Infer(InferContext ctx)
    {
        for (int i = 0; i < 5; i++)
        {
            if (!ctx.HasInput(i))
            {
                throw ctx.Fail("expects inputs X, scale, B, mean and var");
            }

            if (ctx.InputType(i) != DType.F32)
            {
                throw ctx.Unsupported($"element type {ctx.InputType(i)} is not supported yet (float32 only)");
            }
        }

        if (ctx.OutputCount > 1)
        {
            throw ctx.Unsupported("the running-statistics outputs (training) are not supported");
        }

        Shape x = ctx.InputShape(0);
        if (x.Rank < 2)
        {
            throw ctx.Fail($"the input {x} must have at least 2 dimensions");
        }

        for (int i = 1; i < 5; i++)
        {
            Shape p = ctx.InputShape(i);
            if (p.Rank != 1 || p[0] != x[1])
            {
                throw ctx.Fail($"input {i} has shape {p}; it must be 1-D with {x[1]} values");
            }
        }

        ctx.SetOutput(0, DType.F32, x);
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], y = run.Outputs[0];
        if (y.Length == 0)
        {
            return;
        }

        int n = x.Shape[0], c = x.Shape[1];
        BatchNorm.Inference(x.Data<float>(), y.MutableData<float>(), n, c, x.Length / (n * c), run.Inputs[1].Data<float>(),
            run.Inputs[2].Data<float>(), run.Inputs[3].Data<float>(), run.Inputs[4].Data<float>(), _epsilon, run.Exact);
    }
}
