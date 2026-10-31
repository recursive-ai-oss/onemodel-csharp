using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Guide;

internal static class TensorGuide
{
    public static void Basics()
    {
        #region tensor-basics
        using Tensor a = Tensor.FromSpan<float>([1, 2, 3, 4, 5, 6], 2, 3);   // copies the values; shape 2 x 3
        using Tensor zeros = Tensor.Zeros(DType.F32, 2, 3);
        using Tensor labels = Tensor.FromSpan<long>([0L, 1L], 2);

        Tensor transposed = a.T();                  // a view: no copy, shares a's storage
        Tensor firstRow = a.Select(0, 0);           // also a view
        float corner = a.Get<float>(1, 2);          // 6
        ReadOnlySpan<float> flat = a.Data<float>(); // row-major span over a contiguous tensor
        using Tensor packed = transposed.Contiguous();   // a copy in row-major order
        #endregion

        Check.That(corner == 6f && flat.Length == 6 && firstRow.Length == 3 && packed.Shape == Shape.Create(3, 2), "tensor basics");
        Check.That(zeros.Length == 6 && labels.Length == 2, "tensor creation");
    }

    public static void Ownership()
    {
        #region ownership
        Tensor t = Tensor.Zeros(DType.F32, 4);
        Tensor alias = t;                           // another handle to the same storage
        Tensor view = t.Slice(0, 0, 2);             // a view: the same storage again

        t.Dispose();                                // releases the storage for every handle
        alias.Dispose();                            // harmless: Dispose can be called any number of times
        bool stale = view.IsReleased;               // true; view.Data<float>() would now throw StaleTensorException
        #endregion

        Check.That(stale, "ownership: handles see the release");
        Check.Throws<ObjectDisposedException>(() => view.Data<float>(), "stale read throws");

        #region ownership-using
        static float Sum(Tensor x)
        {
            using (x)                               // a method may dispose what it was given; the caller's own Dispose is still fine
            {
                float s = 0;
                foreach (float v in x.Data<float>())
                {
                    s += v;
                }

                return s;
            }
        }

        Tensor data = Tensor.FromSpan<float>([1, 2, 3], 3);
        float total = Sum(data);                    // 6
        data.Dispose();
        #endregion

        Check.That(total == 6f, "ownership: using in a callee");
    }

    public static void Arithmetic()
    {
        #region operators
        using Tensor w = Tensor.FromSpan<float>([1, 2, 3, 4], 2, 2).WithRequiresGrad();   // track gradients for w

        using (new ActivationScope())               // frees every intermediate tensor when the scope ends
        {
            Tensor loss = Ops.Sum(w * w * 0.5f - w);    // operators are the differentiable Ops functions
            loss.Backward();
            Console.WriteLine(loss.Item<float>());      // Item reads a one-element tensor
        }

        float[] grad = w.Grad.ToArray<float>();     // d/dw (w^2/2 - w) = w - 1: 0, 1, 2, 3
        #endregion

        Check.That(grad.AsSpan().SequenceEqual([0f, 1f, 2f, 3f]), "gradient");
    }

    public static void Scopes()
    {
        using Tensor a = Tensor.FromSpan<float>([1, -2], 2);

        #region keep
        Tensor kept;
        using (new ActivationScope())
        {
            Tensor scratch = Ops.Add(a, a);         // collected, released at the end of the scope
            kept = ActivationScope.Keep(Ops.Relu(scratch));   // moved out: yours to dispose
        }

        float[] result = kept.ToArray<float>();     // 2, 0
        kept.Dispose();
        #endregion

        Check.That(result.AsSpan().SequenceEqual([2f, 0f]), "keep");

        #region compute-scope
        using (TensorCompute.Use(ArithmeticMode.Exact, threads: 1))
        {
            // Everything started from here, including awaited work, computes exactly and on one thread.
            using Tensor y = Ops.Mul(a, a);
        }
        // Other code in the process keeps its own setting; TensorCompute.Mode sets the process-wide default.
        #endregion
    }
}
