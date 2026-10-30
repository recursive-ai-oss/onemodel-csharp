using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators.Loss;

/// <summary>
/// NegativeLogLikelihoodLoss and SoftmaxCrossEntropyLoss (opset 12 on), as Operators.md defines them. The input is (N, C, d1..dk)
/// with the class on axis 1 (or (N, C)); the labels are (N, d1..dk). Per position the loss is -x[n, label, d] * w[label], and 0
/// where the label equals ignore_index. Reduction none keeps the label shape; sum adds the losses; mean divides the sum by the
/// sum of the weights of the positions that were not ignored. Sums run in binary64 in increasing position order, so the
/// result does not depend on threads or machine.
/// </summary>
internal sealed class NllLossOp : OpKernel
{
    public static readonly OpDef Nll = new("NegativeLogLikelihoodLoss", 12, int.MaxValue, static n => new NllLossOp(n, false));
    public static readonly OpDef SoftmaxCrossEntropy = new("SoftmaxCrossEntropyLoss", 12, int.MaxValue, static n => new NllLossOp(n, true));

    private readonly bool _softmax;
    private readonly long _ignore;
    private readonly bool _hasIgnore;
    private readonly string _reduction;

    private NllLossOp(NodeInfo n, bool softmax)
    {
        _softmax = softmax;
        _hasIgnore = n.Has("ignore_index");
        _ignore = n.GetInt("ignore_index", 0);
        _reduction = n.GetString("reduction", "mean");
        if (_reduction is not ("none" or "sum" or "mean"))
        {
            throw n.Error($"attribute 'reduction' is '{_reduction}', expected 'none', 'sum' or 'mean'");
        }
    }

    // The softmax form uses the platform's exp and log; the plain loss is arithmetic only.
    public override bool SupportsExact => !_softmax;

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount < 2 || !ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects the scores and the labels");
        }

        DType t = ctx.InputType(0);
        if (t is not (DType.F32 or DType.F64))
        {
            throw ctx.Unsupported($"element type {t} is not supported yet");
        }

        if (ctx.InputType(1) is not (DType.I32 or DType.I64))
        {
            throw ctx.Fail("the labels must be int32 or int64");
        }

        Shape x = ctx.InputShape(0), target = ctx.InputShape(1);
        if (x.Rank < 2)
        {
            throw ctx.Fail("the scores need at least two dimensions (N, C)");
        }

        if (target.Rank != x.Rank - 1 || target[0] != x[0] || !SameTail(x, target))
        {
            throw ctx.Fail($"labels {target} do not fit scores {x}");
        }

        if (ctx.HasInput(2) && (ctx.InputType(2) != t || ctx.InputShape(2).Rank != 1 || ctx.InputShape(2)[0] != x[1]))
        {
            throw ctx.Fail("the weights must be a vector of one weight per class with the type of the scores");
        }

        ctx.SetOutput(0, t, _reduction == "none" ? target : default);
        if (_softmax && ctx.OutputCount > 1)
        {
            ctx.SetOutput(1, t, x);
        }
    }

    private static bool SameTail(Shape x, Shape target)
    {
        for (int i = 2; i < x.Rank; i++)
        {
            if (x[i] != target[i - 1])
            {
                return false;
            }
        }

        return true;
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0];
        Tensor weight = run.Inputs.Length > 2 ? run.Inputs[2] : default;
        if (x.DType == DType.F32)
        {
            Compute<float>(run, x, weight);
        }
        else
        {
            Compute<double>(run, x, weight);
        }
    }

    private void Compute<T>(in OpRun run, Tensor x, Tensor weight) where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>
    {
        int n = x.Shape[0], c = x.Shape[1], d = 1;
        for (int i = 2; i < x.Rank; i++)
        {
            d *= x.Shape[i];
        }

        ReadOnlySpan<T> scores = x.Data<T>();
        ReadOnlySpan<T> w = weight.IsEmpty ? default : weight.Data<T>();
        Tensor labelsT = run.Inputs[1];
        ReadOnlySpan<int> l32 = labelsT.DType == DType.I32 ? labelsT.Data<int>() : default;
        ReadOnlySpan<long> l64 = labelsT.DType == DType.I64 ? labelsT.Data<long>() : default;

        // The log-probabilities: the scores themselves for NLL, log-softmax over the class axis for the cross-entropy.
        ReadOnlySpan<T> logp = scores;
        if (_softmax)
        {
            Span<T> lp = run.Outputs.Length > 1 && !run.Outputs[1].IsEmpty ? run.Outputs[1].MutableData<T>() : run.Scratch(x.DType, x.Shape).MutableData<T>();
            for (int i = 0; i < n; i++)
            {
                for (int k = 0; k < d; k++)
                {
                    double max = double.NegativeInfinity;
                    for (int j = 0; j < c; j++)
                    {
                        max = Math.Max(max, double.CreateTruncating(scores[((i * c) + j) * d + k]));
                    }

                    double sum = 0;
                    for (int j = 0; j < c; j++)
                    {
                        sum += Math.Exp(double.CreateTruncating(scores[((i * c) + j) * d + k]) - max);
                    }

                    double lse = Math.Log(sum) + max;
                    for (int j = 0; j < c; j++)
                    {
                        int at = ((i * c) + j) * d + k;
                        lp[at] = T.CreateTruncating(double.CreateTruncating(scores[at]) - lse);
                    }
                }
            }

            logp = lp;
        }

        Tensor y = run.Outputs[0];
        Span<T> loss = y.MutableData<T>();
        bool none = _reduction == "none";
        double total = 0, weights = 0;
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < d; k++)
            {
                int pos = (i * d) + k;
                long label = l32.IsEmpty ? l64[pos] : l32[pos];
                double v = 0, wt = 0;
                if (!(_hasIgnore && label == _ignore))
                {
                    if (label < 0 || label >= c)
                    {
                        throw new SessionException(SessionErrorKind.InvalidInput, $"{(_softmax ? "SoftmaxCrossEntropyLoss" : "NegativeLogLikelihoodLoss")}: label {label} at position {pos} is outside [0, {c})");
                    }

                    wt = w.IsEmpty ? 1 : double.CreateTruncating(w[(int)label]);
                    v = -double.CreateTruncating(logp[((i * c) + (int)label) * d + k]) * wt;
                }

                if (none)
                {
                    loss[pos] = T.CreateTruncating(v);
                }

                total += v;
                weights += wt;
            }
        }

        if (!none)
        {
            loss[0] = T.CreateTruncating(_reduction == "mean" ? total / weights : total);
        }
    }
}
