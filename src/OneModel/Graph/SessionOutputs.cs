using System;
using System.Collections;
using System.Collections.Generic;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph;

/// <summary>
/// The outputs of one <see cref="Session"/> run, owned by this object: dispose it (one <c>using</c>) and every output tensor is
/// released. Indexable by position (the order the outputs were asked for) or by name.
/// </summary>
/// <remarks>
/// Tensors taken from it are handles to storage the container owns, so they are valid until it is disposed; copy the data out
/// (<c>ToArray</c>) or <c>Clone()</c> a tensor to keep it longer. Disposing a taken tensor yourself is allowed and harmless.
/// </remarks>
public sealed class SessionOutputs : IReadOnlyList<Tensor>, IDisposable
{
    private readonly string[] _names;
    private readonly Tensor[] _tensors;

    internal SessionOutputs(string[] names, Tensor[] tensors)
    {
        _names = names;
        _tensors = tensors;
    }

    /// <summary>Number of outputs.</summary>
    public int Count => _tensors.Length;

    /// <summary>The names of the outputs, in order.</summary>
    public IReadOnlyList<string> Names => _names;

    /// <summary>The output at a position.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The index is outside 0 to <see cref="Count"/> - 1.</exception>
    public Tensor this[int index] => (uint)index < (uint)_tensors.Length ? _tensors[index] : throw new ArgumentOutOfRangeException(nameof(index));

    /// <summary>The output with a name.</summary>
    /// <exception cref="KeyNotFoundException">No output has that name; the message lists the names.</exception>
    public Tensor this[string name] =>
        TryGetValue(name, out Tensor t) ? t : throw new KeyNotFoundException($"There is no output named '{name}'. The outputs are {string.Join(", ", _names)}.");

    /// <summary>Looks an output up by name.</summary>
    /// <param name="name">The output name.</param>
    /// <param name="tensor">The output when found.</param>
    /// <returns>True when an output has that name.</returns>
    public bool TryGetValue(string name, out Tensor tensor)
    {
        ArgumentNullException.ThrowIfNull(name);
        int i = Array.IndexOf(_names, name);
        tensor = i >= 0 ? _tensors[i] : default;
        return i >= 0;
    }

    /// <summary>Releases every output tensor. Safe to call more than once.</summary>
    public void Dispose()
    {
        foreach (Tensor t in _tensors)
        {
            t.Dispose();
        }
    }

    /// <inheritdoc/>
    public IEnumerator<Tensor> GetEnumerator() => ((IEnumerable<Tensor>)_tensors).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
