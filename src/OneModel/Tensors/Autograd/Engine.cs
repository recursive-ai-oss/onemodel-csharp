using System;
using System.Collections.Generic;
using System.Threading;
namespace RecursiveAI.OneModel.Tensors;

public readonly partial struct Tensor
{
    /// <summary>
    /// Runs reverse-mode differentiation from this tensor and accumulates gradients into every leaf that requires them
    /// (<see cref="Grad"/>). The tensor must be a scalar (one element), or <paramref name="gradient"/> must be given.
    /// </summary>
    /// <param name="gradient">The gradient of the final objective with respect to this tensor; same shape. Null means ones, allowed for one-element tensors only.</param>
    /// <param name="retainGraph">Keep the saved tensors so backward can run again through the same graph. Default false: the records free what they own and a second pass throws.</param>
    /// <remarks>
    /// Gradients are added to leaf gradients that already exist, so call <see cref="Backprop.ZeroGrad"/> between steps. Runs with
    /// recording off. The result depends only on the graph and <see cref="TensorCompute"/>'s mode, never on the thread count.
    /// </remarks>
    /// <exception cref="InvalidOperationException">The tensor does not require gradients, or the graph was already freed by an earlier backward.</exception>
    /// <exception cref="TensorVersionException">A tensor saved by the forward pass was modified in place.</exception>
    /// <exception cref="TensorShapeException">The given gradient has the wrong shape, or none was given for a tensor with several elements.</exception>
    public void Backward(Tensor? gradient = null, bool retainGraph = false) => Backprop.Run(this, gradient, retainGraph);
}

/// <summary>The reverse-mode engine: topological backward over <see cref="BackwardRecord"/> graphs. See docs/autograd.md.</summary>
public static class Backprop
{
    /// <summary>Runs backward from <paramref name="root"/>; same as <see cref="Tensor.Backward"/>.</summary>
    public static void Run(in Tensor root, Tensor? gradient = null, bool retainGraph = false)
    {
        if (root.IsEmpty || !root.RequiresGrad)
        {
            throw new InvalidOperationException("Backward needs a tensor that requires gradients.");
        }

        Tensor g;
        if (gradient is { } given)
        {
            if (given.Shape != root.Shape)
            {
                throw new TensorShapeException($"The gradient has shape {given.Shape}, the tensor {root.Shape}.");
            }

            g = given.Clone();
        }
        else
        {
            if (root.Length != 1)
            {
                throw new TensorShapeException("Backward without a gradient needs a one-element tensor.");
            }

            g = Tensor.Create(DType.F32, root.Shape);
            g.Fill(1f);
        }

        AutogradMeta meta = root.Autograd!;
        ActivationScope? suspended = ActivationScope.Suspend();
        using var noGrad = GradMode.NoGradInternal();
        var engine = new Engine();
        try
        {
            if (meta.GradFn is null)
            {
                Engine.AccumulateLeaf(meta, g, owned: true);
                return;
            }

            engine.Execute(meta.GradFn, g, retainGraph);
        }
        finally
        {
            ActivationScope.Resume(suspended);
        }
    }

    /// <summary>Sets the accumulated gradient of a leaf to zeros in place (or leaves it empty if none exists yet).</summary>
    public static void ZeroGrad(in Tensor leaf)
    {
        Tensor gr = leaf.Grad;
        if (!gr.IsEmpty)
        {
            gr.Fill(0f);
        }
    }

    /// <summary>Frees a leaf's accumulated gradient and clears it.</summary>
    public static void ClearGrad(in Tensor leaf)
    {
        AutogradMeta? m = leaf.Autograd;
        if (m is not null && !m.Grad.IsEmpty)
        {
            m.Grad.Dispose();
            m.Grad = default;
        }
    }
}

/// <summary>State of one backward pass: the gradient arriving at each record, and the ownership rules for emitted gradients.</summary>
internal sealed class Engine
{
    private static int s_pass;

    [ThreadStatic]
    private static List<BackwardRecord>? t_order;

    [ThreadStatic]
    private static List<(BackwardRecord Record, int Next)>? t_stack;

    private bool _consumed;

    public void Execute(BackwardRecord root, Tensor rootGrad, bool retain)
    {
        int mark = Interlocked.Increment(ref s_pass);
        List<BackwardRecord> order = t_order ??= new List<BackwardRecord>(256);
        List<(BackwardRecord, int)> stack = t_stack ??= new List<(BackwardRecord, int)>(64);
        order.Clear();
        stack.Clear();
        root.VisitMark = mark;
        stack.Add((root, 0));
        while (stack.Count > 0)
        {
            (BackwardRecord rec, int next) = stack[^1];
            if (rec.IsReleased)
            {
                rootGrad.Dispose();
                throw new InvalidOperationException(
                    "Backward ran through a graph whose saved tensors were already freed by an earlier backward. Pass retainGraph: true to the first call.");
            }

            if (next < rec.InputCount)
            {
                stack[^1] = (rec, next + 1);
                BackwardRecord? producer = rec.InputMeta(next)?.GradFn;
                if (producer is not null && producer.VisitMark != mark)
                {
                    producer.VisitMark = mark;
                    stack.Add((producer, 0));
                }
            }
            else
            {
                order.Add(rec);
                stack.RemoveAt(stack.Count - 1);
            }
        }

        root.PendingGrad = rootGrad;
        int done = 0;
        try
        {
            for (int i = order.Count - 1; i >= 0; i--, done++)
            {
                BackwardRecord rec = order[i];
                Tensor g = rec.PendingGrad;
                rec.PendingGrad = default;
                if (g.IsEmpty)
                {
                    continue;
                }

                _consumed = false;
                try
                {
                    BackwardOps.Run(this, rec, g);
                }
                finally
                {
                    if (!_consumed)
                    {
                        g.Dispose();
                    }
                }

                if (!retain)
                {
                    rec.Release();
                }
            }
        }
        finally
        {
            for (int i = 0; i < order.Count; i++)
            {
                Tensor left = order[i].PendingGrad;
                if (!left.IsEmpty)
                {
                    left.Dispose();
                    order[i].PendingGrad = default;
                }
            }

            order.Clear();
        }
    }

    /// <summary>Sends gradient <paramref name="grad"/> to input <paramref name="i"/> of <paramref name="rec"/>. When <paramref name="owned"/> the engine frees it after use; otherwise it is copied if kept.</summary>
    public static void Emit(BackwardRecord rec, int i, in Tensor grad, bool owned)
    {
        AutogradMeta? meta = rec.InputMeta(i);
        if (meta is null)
        {
            if (owned)
            {
                grad.Dispose();
            }

            return;
        }

        if (meta.GradFn is null)
        {
            AccumulateLeaf(meta, grad, owned);
            return;
        }

        BackwardRecord p = meta.GradFn;
        if (p.PendingGrad.IsEmpty)
        {
            p.PendingGrad = owned ? grad : grad.Clone();
        }
        else
        {
            AddInto(p.PendingGrad, grad);
            if (owned)
            {
                grad.Dispose();
            }
        }
    }

    /// <summary>Emits a view of the incoming gradient (which the engine must then not free) as an owned gradient. At most once per record, after every other use of the gradient.</summary>
    public void EmitAliased(BackwardRecord rec, int i, in Tensor view)
    {
        _consumed = true;
        Emit(rec, i, view, owned: true);
    }

    public static void AccumulateLeaf(AutogradMeta leaf, in Tensor grad, bool owned)
    {
        Tensor cur = leaf.Grad;
        if (cur.IsEmpty)
        {
            // A leaf's gradient outlives the step, so it is never tied to an activation scope.
            leaf.Grad = owned ? grad : grad.Clone();
        }
        else
        {
            AddInto(cur, grad);
            if (owned)
            {
                grad.Dispose();
            }
        }
    }

    // dst += src in element order (both contiguous, same shape).
    private static void AddInto(in Tensor dst, in Tensor src)
    {
        if (dst.Shape != src.Shape)
        {
            throw new TensorShapeException($"Gradient shape {src.Shape} does not match {dst.Shape}.");
        }

        Span<float> d = dst.MutableData<float>();
        ReadOnlySpan<float> s = src.InternalData<float>();
        for (int i = 0; i < d.Length; i++)
        {
            d[i] += s[i];
        }
    }
}
