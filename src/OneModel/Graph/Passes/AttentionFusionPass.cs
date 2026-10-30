using System;
using System.Threading;
using RecursiveAI.OneModel.Graph.Operators;
using RecursiveAI.OneModel.Graph.Operators.Elementwise;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Passes;

/// <summary>
/// Finds scaled dot-product attention written as separate operators and runs it as one fused node (FlashAttention tiling and
/// online softmax, Dao et al. 2022; docs/kernels-attention.md):
/// <code>MatMul(Q, Kt) → [Mul or Div by a constant scale] → [Add mask] → Softmax(last axis) → MatMul(P, V)</code>
/// where the inputs may come from head-split transposes (<c>[b,s,h,d] → [b,h,s,d]</c>, and for K the same with the last two axes
/// swapped) and the result may go through the head-merge transpose. Those transposes are absorbed: the kernel reads and writes
/// the strided layouts directly. The scores are never materialized and the softmax is blocked differently from the separate
/// operators, so results differ in the last bits and the pass is skipped in exact mode.
/// </summary>
internal sealed class AttentionFusionPass : IGraphPass
{
    public string Name => "attention";

    public OptimizationLevel MinLevel => OptimizationLevel.All;

    public bool PreservesBits => false;

    public void Run(PassGraph g, CancellationToken ct)
    {
        int[] sole = PassPatterns.SoleConsumers(g);
        int[] producer = g.Producers();
        for (int i = 0; i < g.Nodes.Count; i++)
        {
            ct.ThrowIfCancellationRequested();
            CompiledNode sm = g.Nodes[i];
            if (sm.Kernel is not SoftmaxFamilyOp { IsPlainSoftmax: true } softmax || sm.Inputs.Length != 1 || sm.Inputs[0] < 0 ||
                softmax.AxisAttribute is not (-1 or 3))
            {
                continue;
            }

            // Backwards from the softmax: [Add mask] [Mul/Div scale] MatMul(Q, Kt).
            int cur = sm.Inputs[0];
            int mask = -1;
            float scale = 1f;
            if (!Producer(g, producer, sole, cur, i, out CompiledNode? p))
            {
                continue;
            }

            if (p!.Kernel is AddOp && p.Inputs.Length == 2)
            {
                int a = p.Inputs[0], b = p.Inputs[1];
                int pi = producer[cur];
                if (a >= 0 && b >= 0 && a != b && IsScoresChain(g, producer, sole, a, pi))
                {
                    (cur, mask) = (a, b);
                }
                else if (a >= 0 && b >= 0 && a != b && IsScoresChain(g, producer, sole, b, pi))
                {
                    (cur, mask) = (b, a);
                }
                else
                {
                    continue;
                }

                if (!Producer(g, producer, sole, cur, pi, out p))
                {
                    continue;
                }
            }

            if (p!.Kernel is MulOp or DivOp && p.Inputs.Length == 2)
            {
                int pi = producer[cur];
                int data = p.Inputs[0], c = p.Inputs[1];
                Tensor t = g.Constant(c);
                if (data < 0 || t.IsEmpty || t.DType != DType.F32 || t.Length != 1)
                {
                    continue;
                }

                float cv = t.Data<float>()[0];
                scale = p.Kernel is MulOp ? cv : 1f / cv;
                cur = data;
                if (!Producer(g, producer, sole, cur, pi, out p))
                {
                    continue;
                }
            }

            if (p!.Kernel is not MatMulOp || p.Inputs.Length != 2 || p.Inputs[0] < 0 || p.Inputs[1] < 0)
            {
                continue;
            }

            int qId = p.Inputs[0], ktId = p.Inputs[1];

            // Forwards: Softmax(...) is read by exactly one MatMul(P, V).
            int pv = sole[sm.Outputs[0]];
            if (pv < 0 || g.Nodes[pv].Kernel is not MatMulOp || g.Nodes[pv].Inputs.Length != 2 || g.Nodes[pv].Inputs[0] != sm.Outputs[0] ||
                g.Nodes[pv].Inputs[1] < 0 || g.Nodes[pv].Inputs[1] == sm.Outputs[0])
            {
                continue;
            }

            CompiledNode pvNode = g.Nodes[pv];
            int vId = pvNode.Inputs[1];

            // Head-split transposes in front of Q, K and V.
            (int qSrc, AttentionLayout ql) = Source(g, producer, qId, [0, 2, 1, 3]);
            (int vSrc, AttentionLayout vl) = Source(g, producer, vId, [0, 2, 1, 3]);
            int kSrc;
            AttentionLayout kl;
            int ktProducer = producer[ktId];
            if (ktProducer >= 0 && g.Nodes[ktProducer].Kernel is TransposeOp kt && g.Nodes[ktProducer].Inputs[0] >= 0 && IsPerm(kt, [0, 2, 3, 1]))
            {
                kSrc = g.Nodes[ktProducer].Inputs[0];
                kl = AttentionLayout.PositionsFirst;
            }
            else if (ktProducer >= 0 && g.Nodes[ktProducer].Kernel is TransposeOp kt2 && g.Nodes[ktProducer].Inputs[0] >= 0 && IsPerm(kt2, [0, 1, 3, 2]))
            {
                kSrc = g.Nodes[ktProducer].Inputs[0];
                kl = AttentionLayout.HeadsFirst;
            }
            else
            {
                continue;
            }

            // Head-merge transpose after the second product.
            int outNode = pv;
            AttentionLayout ol = AttentionLayout.HeadsFirst;
            int merge = sole[pvNode.Outputs[0]];
            if (merge >= 0 && g.Nodes[merge].Kernel is TransposeOp mt && IsPerm(mt, [0, 2, 1, 3]) && g.Nodes[merge].Inputs[0] == pvNode.Outputs[0])
            {
                outNode = merge;
                ol = AttentionLayout.PositionsFirst;
            }

            var kernel = g.Own(new AttentionOp(ql, kl, vl, ol, scale));
            int[] ins = mask >= 0 ? [qSrc, kSrc, vSrc, mask] : [qSrc, kSrc, vSrc];
            g.Nodes[outNode] = new CompiledNode(pvNode.Info, kernel, ins, g.Nodes[outNode].Outputs);
        }
    }

    // The node producing value `id`, when `id` is read only by node `consumer` (so the fusion may absorb it).
    private static bool Producer(PassGraph g, int[] producer, int[] sole, int id, int consumer, out CompiledNode? node)
    {
        node = null;
        int p = producer[id];
        if (p < 0 || sole[id] != consumer)
        {
            return false;
        }

        node = g.Nodes[p];
        return true;
    }

    // Is `id` (read by node `consumer`) the output of [Mul/Div by constant] over MatMul, or of MatMul directly?
    private static bool IsScoresChain(PassGraph g, int[] producer, int[] sole, int id, int consumer)
    {
        if (!Producer(g, producer, sole, id, consumer, out CompiledNode? p))
        {
            return false;
        }

        return p!.Kernel is MatMulOp || (p.Kernel is MulOp or DivOp && p.Inputs.Length == 2 && g.IsConstant(p.Inputs[1]) && !g.IsConstant(p.Inputs[0]));
    }

    // If `id` is the output of a Transpose with the given permutation, its input in position-first layout; else `id` itself, heads first.
    private static (int Source, AttentionLayout Layout) Source(PassGraph g, int[] producer, int id, ReadOnlySpan<long> perm)
    {
        int p = producer[id];
        if (p >= 0 && g.Nodes[p].Kernel is TransposeOp t && g.Nodes[p].Inputs[0] >= 0 && IsPerm(t, perm))
        {
            return (g.Nodes[p].Inputs[0], AttentionLayout.PositionsFirst);
        }

        return (id, AttentionLayout.HeadsFirst);
    }

    private static bool IsPerm(TransposeOp t, ReadOnlySpan<long> perm) => t.Permutation.SequenceEqual(perm);
}
