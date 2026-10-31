using System;
using System.Collections.Generic;
namespace RecursiveAI.OneModel.Tensors;

/// <summary>
/// Frees the intermediate tensors of one forward and backward step in one call. While a scope is open on a thread, every tensor an
/// eager <see cref="Ops"/> operation allocates there (and every copy made by <c>Contiguous</c> or a copying <c>Reshape</c>) is
/// remembered and returned to its pool when the scope is disposed. Views are not tracked (they share their parent's storage).
/// </summary>
/// <remarks>
/// To keep a result past the scope, call <see cref="Keep"/> on it (it then belongs to the enclosing scope, or to you when there is
/// none), or copy it out: <c>Clone()</c> and <c>ToArray()</c> make storage the scope does not track. <c>Detach()</c> shares the storage
/// of its source, so it ends with the scope like the source does. Disposing a collected tensor yourself is harmless (the scope skips it),
/// but do not use any tensor made inside a scope after it ends. Parameters, inputs and anything created
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

    /// <summary>Hands a tensor the caller created (a mask, a constant) to the innermost open scope so it is released with the others. Does nothing when no scope is open or the tensor is empty.</summary>
    public static void Track(in Tensor t)
    {
        ActivationScope? s = t_current;
        if (s is not null && !t.IsEmpty)
        {
            s._tensors.Add(t);
        }
    }

    /// <summary>
    /// Takes a tensor out of the innermost open scope so it survives the scope's end: it moves to the enclosing scope, or becomes
    /// yours to dispose when none is open. Does nothing for a tensor the scope does not hold (an input, a view, a copy you made).
    /// </summary>
    /// <param name="t">A tensor made by an operation inside the scope.</param>
    /// <returns><paramref name="t"/>, so the call can end an expression: <c>kept = ActivationScope.Keep(Ops.Relu(x))</c>.</returns>
    public static Tensor Keep(in Tensor t)
    {
        ActivationScope? s = t_current;
        if (s is null || t.IsEmpty)
        {
            return t;
        }

        bool moved = false;
        for (int i = s._tensors.Count - 1; i >= 0; i--)
        {
            if (ReferenceEquals(s._tensors[i].Buffer, t.Buffer))
            {
                s._tensors.RemoveAt(i);
                moved = true;
            }
        }

        if (moved && s._outer is { } outer)
        {
            outer._tensors.Add(t);
        }

        return t;
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
            _tensors[i].Dispose(); // harmless if the owner already released it
        }

        _tensors.Clear();
    }
}
