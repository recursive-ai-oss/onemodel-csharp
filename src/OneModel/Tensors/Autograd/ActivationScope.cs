using System;
using System.Collections.Generic;
namespace RecursiveAI.OneModel.Tensors;

/// <summary>
/// Frees the intermediate tensors of one forward and backward step in one call. While a scope is open on a thread, every tensor an
/// eager <see cref="Ops"/> operation allocates there (and every copy made by <c>Contiguous</c> or a copying <c>Reshape</c>) is
/// remembered and returned to its pool when the scope is disposed. Views are not tracked (they share their parent's storage).
/// </summary>
/// <remarks>
/// Do not dispose tensors made inside a scope yourself, and do not keep them past it. Parameters, inputs and anything created
/// outside the scope are untouched, and so are gradients accumulated into leaves (they belong to the leaf). Scopes nest; the
/// innermost collects. A scope is per thread. Steady-state training loops open one per step, which makes a step allocate no tensor memory.
/// </remarks>
public sealed class ActivationScope : IDisposable
{
    [ThreadStatic]
    private static ActivationScope? t_current;

    private readonly ActivationScope? _outer;
    private readonly List<Tensor> _tensors = new(64);
    private bool _disposed;

    /// <summary>Opens a scope on the calling thread.</summary>
    public ActivationScope()
    {
        _outer = t_current;
        t_current = this;
    }

    /// <summary>Number of tensors collected so far.</summary>
    public int Count => _tensors.Count;

    // The backward engine makes its own temporaries and frees them itself, so it steps outside any open scope.
    internal static ActivationScope? Suspend()
    {
        ActivationScope? s = t_current;
        t_current = null;
        return s;
    }

    internal static void Resume(ActivationScope? s) => t_current = s;

    internal static void Track(in Tensor t)
    {
        ActivationScope? s = t_current;
        if (s is not null && !t.IsEmpty)
        {
            s._tensors.Add(t);
        }
    }

    /// <summary>Returns the collected tensors to their pools and closes the scope. Tensors already disposed by the caller are skipped.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        t_current = _outer;
        for (int i = _tensors.Count - 1; i >= 0; i--)
        {
            try
            {
                _tensors[i].Dispose();
            }
            catch (StaleTensorException)
            {
                // Already released by its owner.
            }
        }

        _tensors.Clear();
    }
}
