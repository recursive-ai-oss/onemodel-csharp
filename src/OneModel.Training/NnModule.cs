using System;
using System.Collections.Generic;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Training;

/// <summary>
/// A node of a model: it owns parameters (float32 leaves that require gradients), buffers (state that is saved but not trained,
/// such as batch-norm running statistics) and named child modules. Modules are written on the eager tensor ops of the core package.
/// </summary>
/// <remarks>
/// Parameters and buffers are listed in a fixed order: a module's own parameters, its own buffers, then its children in
/// registration order, recursively. Names are dotted paths (<c>layers.0.weight</c>). The same order drives
/// <see cref="StateDict"/>, <see cref="Checkpoint"/> and the optimizers, so it is part of the checkpoint format.
/// A module is not thread-safe: run one forward/backward at a time.
/// </remarks>
public abstract class NnModule : IDisposable
{
    private readonly List<(string Name, Tensor Tensor)> _parameters = new();
    private readonly List<(string Name, Tensor Tensor)> _buffers = new();
    private readonly List<(string Name, NnModule Child)> _children = new();
    private bool _disposed;

    /// <summary>True in training mode (the default): dropout is active and batch norm uses batch statistics.</summary>
    public bool IsTraining { get; private set; } = true;

    /// <summary>Sets training or evaluation mode on this module and every child.</summary>
    /// <param name="training">True for training mode (default), false for evaluation.</param>
    /// <returns>This module.</returns>
    public NnModule Train(bool training = true)
    {
        IsTraining = training;
        foreach ((_, NnModule child) in _children)
        {
            child.Train(training);
        }

        return this;
    }

    /// <summary>Sets evaluation mode on this module and every child; the same as <c>Train(false)</c>.</summary>
    /// <returns>This module.</returns>
    public NnModule Eval() => Train(false);

    /// <summary>Computes the module's output for one input tensor.</summary>
    /// <param name="input">The input; its required shape and type are stated by each module.</param>
    /// <returns>The output tensor. It carries a backward record when an input or parameter requires gradients and recording is on.</returns>
    public abstract Tensor Forward(Tensor input);

    /// <summary>Makes <paramref name="value"/> a trainable parameter of this module.</summary>
    /// <param name="name">Name within this module; must not contain a dot and must be unused among parameters, buffers and children.</param>
    /// <param name="value">A contiguous float32 tensor; the module takes ownership.</param>
    /// <returns>The tensor to use as the parameter (it requires gradients).</returns>
    protected Tensor RegisterParameter(string name, Tensor value)
    {
        CheckName(name);
        if (value.DType != DType.F32 || !value.IsContiguous)
        {
            throw new ArgumentException($"Parameter '{name}' must be a contiguous float32 tensor.", nameof(value));
        }

        Tensor p = value.WithRequiresGrad();
        _parameters.Add((name, p));
        return p;
    }

    /// <summary>Registers saved, untrained state (float32, Int32 or Int64) of this module.</summary>
    /// <param name="name">Name within this module; same rules as for parameters.</param>
    /// <param name="value">A contiguous tensor; the module takes ownership.</param>
    /// <returns><paramref name="value"/>.</returns>
    protected Tensor RegisterBuffer(string name, Tensor value)
    {
        CheckName(name);
        if (!value.IsContiguous)
        {
            throw new ArgumentException($"Buffer '{name}' must be contiguous.", nameof(value));
        }

        _buffers.Add((name, value));
        return value;
    }

    /// <summary>Registers a child module.</summary>
    /// <param name="name">Name within this module; same rules as for parameters.</param>
    /// <param name="module">The child; this module takes ownership and disposes it.</param>
    /// <typeparam name="T">The child's type.</typeparam>
    /// <returns><paramref name="module"/>.</returns>
    protected T RegisterModule<T>(string name, T module) where T : NnModule
    {
        CheckName(name);
        _children.Add((name, module));
        return module;
    }

    // For modules whose parameter names come from a file (an imported graph's initializer names), which may contain dots.
    // Same rules as RegisterParameter / RegisterBuffer except that the name is taken as it is.
    private protected Tensor RegisterVerbatim(string name, Tensor value, bool parameter)
    {
        CheckName(name, allowDots: true);
        if (parameter)
        {
            Tensor p = value.WithRequiresGrad();
            _parameters.Add((name, p));
            return p;
        }

        _buffers.Add((name, value));
        return value;
    }

    private void CheckName(string name, bool allowDots = false)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        if (!allowDots && name.Contains('.', StringComparison.Ordinal))
        {
            throw new ArgumentException($"Name '{name}' must not contain a dot.", nameof(name));
        }

        foreach ((string n, _) in _parameters)
        {
            if (n == name) { throw new ArgumentException($"Name '{name}' is already used.", nameof(name)); }
        }

        foreach ((string n, _) in _buffers)
        {
            if (n == name) { throw new ArgumentException($"Name '{name}' is already used.", nameof(name)); }
        }

        foreach ((string n, _) in _children)
        {
            if (n == name) { throw new ArgumentException($"Name '{name}' is already used.", nameof(name)); }
        }
    }

    /// <summary>The child modules of this module with their names, in registration order.</summary>
    public IReadOnlyList<(string Name, NnModule Child)> Children => _children;

    /// <summary>All parameters of this module and its descendants, named by dotted path, in the fixed order (see the class remarks).</summary>
    public List<(string Name, Tensor Tensor)> NamedParameters()
    {
        var list = new List<(string, Tensor)>();
        Collect(list, "", parameters: true, buffers: false);
        return list;
    }

    /// <summary>All buffers of this module and its descendants, named by dotted path, in the fixed order.</summary>
    public List<(string Name, Tensor Tensor)> NamedBuffers()
    {
        var list = new List<(string, Tensor)>();
        Collect(list, "", parameters: false, buffers: true);
        return list;
    }

    /// <summary>The parameters without names, in the same order as <see cref="NamedParameters"/>.</summary>
    public Tensor[] Parameters()
    {
        List<(string Name, Tensor Tensor)> named = NamedParameters();
        var result = new Tensor[named.Count];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = named[i].Tensor;
        }

        return result;
    }

    /// <summary>The parameters that currently require gradients (not frozen), named, in the same order as <see cref="NamedParameters"/>.</summary>
    public List<(string Name, Tensor Tensor)> NamedTrainableParameters()
    {
        var list = new List<(string, Tensor)>();
        foreach ((string n, Tensor t) in NamedParameters())
        {
            if (t.RequiresGrad)
            {
                list.Add((n, t));
            }
        }

        return list;
    }

    /// <summary>The parameters that currently require gradients, without names, in parameter order.</summary>
    public Tensor[] TrainableParameters()
    {
        List<(string Name, Tensor Tensor)> named = NamedTrainableParameters();
        var result = new Tensor[named.Count];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = named[i].Tensor;
        }

        return result;
    }

    /// <summary>
    /// Freezes every parameter of this module and its descendants: they stop requiring gradients (PyTorch's <c>requires_grad_(False)</c>).
    /// A frozen parameter receives no gradient (any accumulated one is freed), is skipped by optimizers and by <see cref="GradClip"/>, keeps
    /// no optimizer state if the optimizer was built while it was frozen, and stays in <see cref="StateDict"/> and checkpoints.
    /// Freeze before the forward pass: operations record backward information only for inputs that require gradients when they run.
    /// </summary>
    /// <returns>The number of parameters whose state changed.</returns>
    public int Freeze() => SetTrainable(static (_, _) => true, false);

    /// <summary>Freezes one parameter by dotted name (<c>layer0.attn.q.weight</c>) or every parameter under a child module's dotted path (<c>layer0</c>).</summary>
    /// <param name="nameOrPrefix">A parameter name or a module path.</param>
    /// <returns>The number of parameters whose state changed.</returns>
    /// <exception cref="ArgumentException">No parameter has that name or lies under that path.</exception>
    public int Freeze(string nameOrPrefix) => SetTrainable(nameOrPrefix, false);

    /// <summary>Freezes every parameter for which the predicate is true.</summary>
    /// <param name="predicate">Called with the dotted name and the parameter.</param>
    /// <returns>The number of parameters whose state changed.</returns>
    public int Freeze(Func<string, Tensor, bool> predicate) => SetTrainable(predicate, false);

    /// <summary>Makes every parameter of this module and its descendants require gradients again. Optimizer state of parameters that were frozen when the optimizer was built is created, zeroed, at their first update.</summary>
    /// <returns>The number of parameters whose state changed.</returns>
    public int Unfreeze() => SetTrainable(static (_, _) => true, true);

    /// <summary>Unfreezes one parameter by dotted name or every parameter under a module path; see <see cref="Freeze(string)"/>.</summary>
    /// <param name="nameOrPrefix">A parameter name or a module path.</param>
    /// <returns>The number of parameters whose state changed.</returns>
    /// <exception cref="ArgumentException">No parameter has that name or lies under that path.</exception>
    public int Unfreeze(string nameOrPrefix) => SetTrainable(nameOrPrefix, true);

    /// <summary>Unfreezes every parameter for which the predicate is true.</summary>
    /// <param name="predicate">Called with the dotted name and the parameter.</param>
    /// <returns>The number of parameters whose state changed.</returns>
    public int Unfreeze(Func<string, Tensor, bool> predicate) => SetTrainable(predicate, true);

    /// <summary>True if the named parameter does not require gradients.</summary>
    /// <param name="name">The parameter's dotted name.</param>
    /// <exception cref="ArgumentException">There is no parameter of that name.</exception>
    public bool IsFrozen(string name)
    {
        foreach ((string n, Tensor t) in NamedParameters())
        {
            if (n == name)
            {
                return !t.RequiresGrad;
            }
        }

        throw new ArgumentException($"No parameter named '{name}'.", nameof(name));
    }

    private int SetTrainable(string nameOrPrefix, bool trainable)
    {
        ArgumentException.ThrowIfNullOrEmpty(nameOrPrefix);
        string prefix = nameOrPrefix + ".";
        bool found = false;
        int changed = SetTrainable((n, _) =>
        {
            bool hit = n == nameOrPrefix || n.StartsWith(prefix, StringComparison.Ordinal);
            found |= hit;
            return hit;
        }, trainable);
        if (!found)
        {
            throw new ArgumentException($"No parameter is named '{nameOrPrefix}' or lies under a module of that path.", nameof(nameOrPrefix));
        }

        return changed;
    }

    private int SetTrainable(Func<string, Tensor, bool> predicate, bool trainable)
    {
        ArgumentNullException.ThrowIfNull(predicate);
        int changed = 0;
        foreach ((string n, Tensor t) in NamedParameters())
        {
            if (predicate(n, t) && t.RequiresGrad != trainable)
            {
                t.SetRequiresGrad(trainable);
                changed++;
            }
        }

        return changed;
    }

    /// <summary>Parameters and buffers together, per module in the order own parameters, own buffers, children. This is the checkpoint order.</summary>
    public List<(string Name, Tensor Tensor)> StateDict()
    {
        var list = new List<(string, Tensor)>();
        Collect(list, "", parameters: true, buffers: true);
        return list;
    }

    private void Collect(List<(string, Tensor)> into, string prefix, bool parameters, bool buffers)
    {
        if (parameters)
        {
            foreach ((string n, Tensor t) in _parameters)
            {
                into.Add((prefix + n, t));
            }
        }

        if (buffers)
        {
            foreach ((string n, Tensor t) in _buffers)
            {
                into.Add((prefix + n, t));
            }
        }

        foreach ((string n, NnModule m) in _children)
        {
            m.Collect(into, prefix + n + ".", parameters, buffers);
        }
    }

    /// <summary>Copies values into the module's parameters and buffers by name; tensors keep their identity, so optimizers and gradients stay valid.</summary>
    /// <param name="state">Named tensors, as returned by <see cref="StateDict"/> of an equal module.</param>
    /// <exception cref="ArgumentException">A name is missing or unknown, or a shape or type differs.</exception>
    public void LoadStateDict(IEnumerable<(string Name, Tensor Tensor)> state)
    {
        ArgumentNullException.ThrowIfNull(state);
        Dictionary<string, Tensor> given = new(StringComparer.Ordinal);
        foreach ((string n, Tensor t) in state)
        {
            if (!given.TryAdd(n, t))
            {
                throw new ArgumentException($"State name '{n}' appears twice.", nameof(state));
            }
        }

        List<(string Name, Tensor Tensor)> mine = StateDict();
        foreach ((string n, Tensor dst) in mine)
        {
            if (!given.TryGetValue(n, out Tensor src))
            {
                throw new ArgumentException($"State has no entry '{n}'.", nameof(state));
            }

            if (src.DType != dst.DType || !src.Shape.Equals(dst.Shape))
            {
                throw new ArgumentException($"State entry '{n}' is {src.DType}{src.Shape}, the module has {dst.DType}{dst.Shape}.", nameof(state));
            }
        }

        if (given.Count != mine.Count)
        {
            throw new ArgumentException($"State has {given.Count} entries, the module has {mine.Count}.", nameof(state));
        }

        foreach ((string n, Tensor dst) in mine)
        {
            dst.Detach().CopyFrom(given[n].Detach());
        }
    }

    /// <summary>Sets every parameter's accumulated gradient to zero in place (parameters with no gradient yet are left alone).</summary>
    public void ZeroGrad()
    {
        foreach ((_, Tensor p) in NamedParameters())
        {
            Backprop.ZeroGrad(p);
        }
    }

    // A hook for modules that own more than tensors and child modules (a model file, scratch memory).
    private protected virtual void OnDispose()
    {
    }

    /// <summary>Releases the parameters, buffers, gradients and children. The module must not be used afterwards.</summary>
    public void Dispose()
    {
        GC.SuppressFinalize(this);
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        OnDispose();
        foreach ((_, Tensor p) in _parameters)
        {
            Backprop.ClearGrad(p);
            p.Dispose();
        }

        foreach ((_, Tensor b) in _buffers)
        {
            b.Dispose();
        }

        foreach ((_, NnModule m) in _children)
        {
            m.Dispose();
        }

        _parameters.Clear();
        _buffers.Clear();
        _children.Clear();
    }
}
