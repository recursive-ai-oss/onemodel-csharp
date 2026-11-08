using System;
using System.Collections.Generic;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Training;

/// <summary>
/// Base of the optimizers. An optimizer updates a fixed list of float32 parameters in place from their <see cref="Tensor.Grad"/>.
/// </summary>
/// <remarks>
/// <para>Updates use scalar float32 arithmetic in a fixed order within each element, with parameters visited in list order, using only IEEE
/// <c>+ − × ÷ sqrt</c>, no fused multiply-add: the result is the same in fast and exact mode, on every platform and thread count.
/// Dense Adam/AdamW can update independent elements concurrently, limited by <see cref="TensorCompute.Threads"/>.
/// Parameters without a gradient are skipped, and so are frozen parameters (<see cref="Tensor.SetRequiresGrad"/>, <see cref="NnModule.Freeze()"/>).
/// State tensors (momentum, moments) are allocated with the optimizer and kept as float32 tensors shaped like their parameter,
/// except for a parameter that is frozen when the optimizer is built: its state tensors are empty (shape [0]) until the parameter is
/// unfrozen and first updated, when they are created zeroed. A checkpoint therefore records which parameters had state; load it into
/// an optimizer built with the same parameters frozen, or into any optimizer (state shapes are adopted from the file).</para>
/// <para>Formulas follow the published algorithms (see <c>docs/training.md</c>). There is one parameter group: one learning rate and
/// one set of hyper-parameters for all parameters.</para>
/// </remarks>
public abstract class Optimizer : IDisposable
{
    private readonly Tensor[] _parameters;
    private readonly List<(string Name, Tensor Tensor)> _state = new();
    private readonly List<(Tensor[] Slot, int Start)> _slots = new();
    private readonly List<int> _stateParam = new();
    private bool _disposed;

    /// <summary>Creates the optimizer over the given parameters.</summary>
    /// <param name="parameters">Contiguous float32 tensors (for example <see cref="NnModule.Parameters"/>, or <see cref="NnModule.TrainableParameters"/>); at least one. Frozen ones are accepted and skipped.</param>
    /// <param name="learningRate">The initial learning rate, finite and not negative.</param>
    protected Optimizer(IEnumerable<Tensor> parameters, float learningRate)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        _parameters = new List<Tensor>(parameters).ToArray();
        if (_parameters.Length == 0)
        {
            throw new ArgumentException("An optimizer needs at least one parameter.", nameof(parameters));
        }

        foreach (Tensor p in _parameters)
        {
            // A frozen parameter is still a gradient leaf (it has autograd state); a plain tensor never was and never can be trained.
            if (p.DType != DType.F32 || !p.IsContiguous || p.Autograd is not { IsLeaf: true })
            {
                throw new ArgumentException("Parameters must be contiguous float32 gradient leaves (frozen ones are allowed).", nameof(parameters));
            }
        }

        CheckRate(learningRate);
        LearningRate = learningRate;
    }

    /// <summary>The parameters being optimized, in the order given.</summary>
    public IReadOnlyList<Tensor> Parameters => _parameters;

    /// <summary>The learning rate used by the next <see cref="Step"/>; a schedule sets it. Finite and not negative.</summary>
    public float LearningRate { get; set; }

    /// <summary>Number of completed <see cref="Step"/> calls (restored by <see cref="Checkpoint"/>). Schedules read it.</summary>
    public long StepCount { get; protected set; }

    /// <summary>A short name for the algorithm (<c>sgd</c>, <c>adam</c>, <c>adamw</c>), written into checkpoints.</summary>
    public abstract string Kind { get; }

    /// <summary>The hyper-parameters (including <c>lr</c>) as name and value, in a fixed order; a checkpoint stores them as float bit patterns.</summary>
    public abstract List<(string Name, float Value)> HyperParameters();

    /// <summary>Sets hyper-parameters by name from a checkpoint; every name returned by <see cref="HyperParameters"/> must be present.</summary>
    /// <param name="values">Name and value pairs.</param>
    public abstract void SetHyperParameters(IReadOnlyDictionary<string, float> values);

    /// <summary>Per-parameter state (momentum buffers, moments, flags), named <c>slot.index</c>, in a fixed order; checkpoints save these tensors.</summary>
    public IReadOnlyList<(string Name, Tensor Tensor)> StateTensors => _state;

    /// <summary>Sets <see cref="StepCount"/> (used when loading a checkpoint).</summary>
    /// <param name="steps">The step count, not negative.</param>
    public void RestoreStepCount(long steps)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(steps);
        StepCount = steps;
    }

    /// <summary>
    /// Allocates one zeroed float32 state tensor per parameter, shaped like it, named <c>slot.index</c>; empty ([0]) for a parameter that
    /// is frozen now. Read a slot only through <see cref="StateOf"/>, which creates a missing tensor.
    /// </summary>
    protected Tensor[] AddStateSlot(string slot)
    {
        var tensors = new Tensor[_parameters.Length];
        _slots.Add((tensors, _state.Count));
        for (int i = 0; i < tensors.Length; i++)
        {
            tensors[i] = Tensor.Zeros(DType.F32, _parameters[i].RequiresGrad ? _parameters[i].Shape : Shape.Create(0));
            _state.Add((slot + "." + i.ToString(System.Globalization.CultureInfo.InvariantCulture), tensors[i]));
            _stateParam.Add(i);
        }

        return tensors;
    }

    /// <summary>The state tensor of parameter <paramref name="index"/> in <paramref name="slot"/> (from <see cref="AddStateSlot"/>), created zeroed if the parameter was frozen when the optimizer was built.</summary>
    protected Tensor StateOf(Tensor[] slot, int index)
    {
        Tensor t = slot[index];
        if (t.Length != _parameters[index].Length)
        {
            SetState(slot, index, _parameters[index].Shape);
            t = slot[index];
        }

        return t;
    }

    private void SetState(Tensor[] slot, int index, Shape shape)
    {
        int start = -1;
        foreach ((Tensor[] s, int st) in _slots)
        {
            if (ReferenceEquals(s, slot))
            {
                start = st;
            }
        }

        slot[index].Dispose();
        Tensor fresh = Tensor.Zeros(DType.F32, shape);
        slot[index] = fresh;
        _state[start + index] = (_state[start + index].Name, fresh);
    }

    // Checkpoint loading: can state entry i take the shape the file has (its parameter's shape, or empty)?
    internal bool CanAdoptState(int stateIndex, ReadOnlySpan<int> dims)
    {
        int p = _stateParam[stateIndex];
        if (p < 0 || _state[stateIndex].Tensor.DType != DType.F32)
        {
            return false;
        }

        return dims.SequenceEqual(_parameters[p].Shape.Dimensions) || (dims.Length == 1 && dims[0] == 0);
    }

    internal void AdoptState(int stateIndex, ReadOnlySpan<int> dims)
    {
        Tensor current = _state[stateIndex].Tensor;
        if (dims.SequenceEqual(current.Shape.Dimensions))
        {
            return;
        }

        foreach ((Tensor[] slot, int start) in _slots)
        {
            if (stateIndex >= start && stateIndex < start + slot.Length)
            {
                SetState(slot, stateIndex - start, Shape.Create(dims));
                return;
            }
        }
    }

    /// <summary>Allocates a zeroed Int32 flag tensor [parameter count].</summary>
    protected Tensor AddFlags(string name)
    {
        Tensor t = Tensor.Zeros(DType.I32, Shape.Create(_parameters.Length));
        _state.Add((name, t));
        _stateParam.Add(-1);
        return t;
    }

    /// <summary>Throws unless <paramref name="value"/> is finite and not negative.</summary>
    protected static void CheckRate(float value)
    {
        if (!float.IsFinite(value) || value < 0f)
        {
            throw new ArgumentOutOfRangeException(nameof(value), value, "Value must be finite and not negative.");
        }
    }

    /// <summary>Applies one update from the current gradients and increments <see cref="StepCount"/>. Recording is off while it runs.</summary>
    /// <exception cref="InvalidOperationException">A parameter has a row-sparse gradient and this optimizer does not support one.</exception>
    public void Step()
    {
        CheckRate(LearningRate);
        BeginStep();
        using (GradMode.NoGrad())
        {
            for (int i = 0; i < _parameters.Length; i++)
            {
                if (!_parameters[i].RequiresGrad)
                {
                    continue;
                }

                if (_parameters[i].SparseGrad is { } sparse)
                {
                    UpdateSparse(i, _parameters[i].MutableData<float>(), sparse);
                    continue;
                }

                Tensor g = _parameters[i].Grad;
                if (g.IsEmpty)
                {
                    continue;
                }

                if (!g.Shape.Equals(_parameters[i].Shape) || !g.IsContiguous)
                {
                    throw new InvalidOperationException($"Gradient of parameter {i} is {g.Shape}, the parameter is {_parameters[i].Shape}.");
                }

                Update(i, _parameters[i].MutableData<float>(), g.Data<float>());
            }
        }

        StepCount++;
    }

    /// <summary>Sets every parameter's gradient to zero in place.</summary>
    public void ZeroGrad()
    {
        foreach (Tensor p in _parameters)
        {
            Backprop.ZeroGrad(p);
        }
    }

    /// <summary>Called at the start of <see cref="Step"/>, before any parameter is updated; computes step-wide scalars.</summary>
    protected virtual void BeginStep()
    {
    }

    /// <summary>Updates one parameter from its gradient; <see cref="StepCount"/> is still the number of steps before this one.</summary>
    /// <param name="index">Index in <see cref="Parameters"/>.</param>
    /// <param name="parameter">The parameter's values, updated in place.</param>
    /// <param name="gradient">The gradient, same length.</param>
    protected abstract void Update(int index, Span<float> parameter, ReadOnlySpan<float> gradient);

    /// <summary>Updates one parameter from a row-sparse gradient. The default throws: only optimizers that define lazy row semantics override it.</summary>
    /// <param name="index">Index in <see cref="Parameters"/>.</param>
    /// <param name="parameter">The parameter's values [rows, width], updated in place.</param>
    /// <param name="gradient">The touched rows' gradient.</param>
    protected virtual void UpdateSparse(int index, Span<float> parameter, RowSparseGradient gradient) =>
        throw new InvalidOperationException($"Parameter {index} has a row-sparse gradient, which the {Kind} optimizer does not support. Use Sgd without momentum or weight decay, or SparseAdam, for that parameter, or build the Embedding with sparse: false.");

    /// <summary>Releases the state tensors (not the parameters).</summary>
    public void Dispose()
    {
        GC.SuppressFinalize(this);
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        foreach ((_, Tensor t) in _state)
        {
            t.Dispose();
        }

        _state.Clear();
    }
}

/// <summary>
/// Stochastic gradient descent with optional momentum (Polyak), Nesterov momentum (Sutskever et al. 2013), dampening and L2 weight
/// decay. Per element: <c>g += wd·p</c>; with momentum <c>b = (first step of this parameter) ? g : μ·b + (1−d)·g</c> and
/// <c>g = nesterov ? g + μ·b : b</c>; then <c>p −= lr·g</c>.
/// A row-sparse gradient (<see cref="Tensor.SparseGrad"/>) is supported without momentum and weight decay, as in PyTorch: <c>p[row] −= lr·g[row]</c>
/// for the touched rows only, which equals the dense update bit for bit.
/// </summary>
public sealed class Sgd : Optimizer
{
    private readonly float _momentum;
    private readonly float _dampening;
    private readonly bool _nesterov;
    private readonly Tensor[]? _buffers;
    private readonly Tensor _initialized;

    /// <summary>Weight decay (L2 penalty added to the gradient); changeable.</summary>
    public float WeightDecay { get; set; }

    /// <summary>Creates the optimizer.</summary>
    /// <param name="parameters">Parameters to update.</param>
    /// <param name="learningRate">Learning rate.</param>
    /// <param name="momentum">Momentum μ, not negative (default 0: plain SGD).</param>
    /// <param name="dampening">Dampening d for the momentum buffer (default 0); must be 0 with Nesterov.</param>
    /// <param name="weightDecay">L2 weight decay, not negative (default 0).</param>
    /// <param name="nesterov">Nesterov momentum; needs momentum above 0 (default false).</param>
    public Sgd(IEnumerable<Tensor> parameters, float learningRate, float momentum = 0f, float dampening = 0f, float weightDecay = 0f, bool nesterov = false)
        : base(parameters, learningRate)
    {
        CheckRate(momentum);
        CheckRate(weightDecay);
        if (nesterov && (momentum <= 0f || dampening != 0f))
        {
            throw new ArgumentException("Nesterov momentum needs momentum above 0 and no dampening.");
        }

        _momentum = momentum;
        _dampening = dampening;
        _nesterov = nesterov;
        WeightDecay = weightDecay;
        _initialized = AddFlags("initialized");
        if (momentum != 0f)
        {
            _buffers = AddStateSlot("momentum_buffer");
        }
    }

    /// <inheritdoc/>
    public override string Kind => "sgd";

    /// <inheritdoc/>
    public override List<(string Name, float Value)> HyperParameters() =>
        [("lr", LearningRate), ("momentum", _momentum), ("dampening", _dampening), ("weight_decay", WeightDecay), ("nesterov", _nesterov ? 1f : 0f)];

    /// <inheritdoc/>
    public override void SetHyperParameters(IReadOnlyDictionary<string, float> values)
    {
        if (values["momentum"] != _momentum || values["dampening"] != _dampening || (values["nesterov"] != 0f) != _nesterov)
        {
            throw new ArgumentException("The saved momentum, dampening or nesterov setting differs from this optimizer's.");
        }

        CheckRate(values["lr"]);
        CheckRate(values["weight_decay"]);
        LearningRate = values["lr"];
        WeightDecay = values["weight_decay"];
    }

    /// <inheritdoc/>
    protected override void Update(int index, Span<float> parameter, ReadOnlySpan<float> gradient)
    {
        Span<float> p = parameter;
        ReadOnlySpan<float> grad = gradient;
        float lr = LearningRate, wd = WeightDecay, mu = _momentum, keep = 1f - _dampening;
        if (_buffers is null)
        {
            for (int i = 0; i < p.Length; i++)
            {
                float g = grad[i];
                if (wd != 0f)
                {
                    g += wd * p[i];
                }

                p[i] -= lr * g;
            }

            return;
        }

        Span<float> buf = StateOf(_buffers, index).MutableData<float>();
        Span<int> init = _initialized.MutableData<int>();
        bool first = init[index] == 0;
        init[index] = 1;
        for (int i = 0; i < p.Length; i++)
        {
            float g = grad[i];
            if (wd != 0f)
            {
                g += wd * p[i];
            }

            float b = first ? g : mu * buf[i] + keep * g;
            buf[i] = b;
            g = _nesterov ? g + mu * b : b;
            p[i] -= lr * g;
        }
    }

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException">Momentum or weight decay is set: PyTorch has no sparse semantics for them either.</exception>
    protected override void UpdateSparse(int index, Span<float> parameter, RowSparseGradient gradient)
    {
        if (_buffers is not null || WeightDecay != 0f)
        {
            throw new NotSupportedException("Sgd supports row-sparse gradients only without momentum and weight decay.");
        }

        int width = gradient.DenseShape[1];
        float lr = LearningRate;
        ReadOnlySpan<int> rows = gradient.Rows;
        ReadOnlySpan<float> g = gradient.Values.Data<float>();
        for (int k = 0; k < rows.Length; k++)
        {
            Span<float> p = parameter.Slice(rows[k] * width, width);
            ReadOnlySpan<float> gr = g.Slice(k * width, width);
            for (int j = 0; j < p.Length; j++)
            {
                p[j] -= lr * gr[j];
            }
        }
    }
}

/// <summary>
/// Adam (Kingma and Ba 2015) and, with <c>decoupledWeightDecay</c>, AdamW (Loshchilov and Hutter 2019). With t the step number
/// (1 for the first): <c>m = β₁m + (1−β₁)g</c>, <c>v = β₂v + (1−β₂)g²</c>, <c>p −= (lr/(1−β₁ᵗ))·m / (√v/√(1−β₂ᵗ) + ε)</c>.
/// Plain Adam adds <c>wd·p</c> to <c>g</c> first (L2); AdamW instead scales <c>p</c> by <c>1 − lr·wd</c> first. With AMSGrad
/// (Reddi et al. 2018) the denominator uses the running maximum of v. The powers βᵗ are computed in double by binary exponentiation.
/// </summary>
public class Adam : Optimizer
{
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _epsilon;
    private readonly bool _decoupled;
    private readonly bool _amsgrad;
    private readonly Tensor[] _m;
    private readonly Tensor[] _v;
    private readonly Tensor[]? _vMax;
    private float _stepSize;
    private float _sqrtBc2;
    private readonly UpdateJob _updateJob;

    /// <summary>Weight decay (L2, or decoupled for AdamW); changeable.</summary>
    public float WeightDecay { get; set; }

    /// <summary>Creates the optimizer.</summary>
    /// <param name="parameters">Parameters to update.</param>
    /// <param name="learningRate">Learning rate (default 1e-3).</param>
    /// <param name="beta1">Decay of the first moment, in [0, 1) (default 0.9).</param>
    /// <param name="beta2">Decay of the second moment, in [0, 1) (default 0.999).</param>
    /// <param name="epsilon">Added to the denominator, positive (default 1e-8).</param>
    /// <param name="weightDecay">Weight decay, not negative (default 0).</param>
    /// <param name="decoupledWeightDecay">Decay the weights directly (AdamW) instead of through the gradient (default false).</param>
    /// <param name="amsgrad">Use the AMSGrad variant (default false).</param>
    public Adam(IEnumerable<Tensor> parameters, float learningRate = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f,
        float weightDecay = 0f, bool decoupledWeightDecay = false, bool amsgrad = false)
        : base(parameters, learningRate)
    {
        if (!(beta1 >= 0f && beta1 < 1f) || !(beta2 >= 0f && beta2 < 1f))
        {
            throw new ArgumentOutOfRangeException(nameof(beta1), "Betas must be in [0, 1).");
        }

        if (!(epsilon > 0f) || !float.IsFinite(epsilon))
        {
            throw new ArgumentOutOfRangeException(nameof(epsilon), epsilon, "Epsilon must be positive and finite.");
        }

        CheckRate(weightDecay);
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        WeightDecay = weightDecay;
        _decoupled = decoupledWeightDecay;
        _amsgrad = amsgrad;
        _updateJob = new UpdateJob(this);
        _m = AddStateSlot("exp_avg");
        _v = AddStateSlot("exp_avg_sq");
        if (amsgrad)
        {
            _vMax = AddStateSlot("max_exp_avg_sq");
        }
    }

    /// <inheritdoc/>
    public override string Kind => _decoupled ? "adamw" : "adam";

    /// <inheritdoc/>
    public override List<(string Name, float Value)> HyperParameters() =>
        [("lr", LearningRate), ("beta1", _beta1), ("beta2", _beta2), ("epsilon", _epsilon), ("weight_decay", WeightDecay), ("amsgrad", _amsgrad ? 1f : 0f)];

    /// <inheritdoc/>
    public override void SetHyperParameters(IReadOnlyDictionary<string, float> values)
    {
        if (values["beta1"] != _beta1 || values["beta2"] != _beta2 || values["epsilon"] != _epsilon || (values["amsgrad"] != 0f) != _amsgrad)
        {
            throw new ArgumentException("The saved betas, epsilon or amsgrad setting differ from this optimizer's.");
        }

        CheckRate(values["lr"]);
        CheckRate(values["weight_decay"]);
        LearningRate = values["lr"];
        WeightDecay = values["weight_decay"];
    }

    internal static double IntPow(double b, long n)
    {
        double result = 1.0;
        while (n > 0)
        {
            if ((n & 1) != 0)
            {
                result *= b;
            }

            b *= b;
            n >>= 1;
        }

        return result;
    }

    /// <inheritdoc/>
    protected override void BeginStep()
    {
        long t = StepCount + 1;
        double bc1 = 1.0 - IntPow(_beta1, t);
        double bc2 = 1.0 - IntPow(_beta2, t);
        _stepSize = (float)((double)LearningRate / bc1);
        _sqrtBc2 = (float)Math.Sqrt(bc2);
    }

    /// <inheritdoc/>
    protected override void Update(int index, Span<float> parameter, ReadOnlySpan<float> gradient)
    {
        Tensor mt = StateOf(_m, index), vt = StateOf(_v, index);
        Tensor max = _vMax is null ? default : StateOf(_vMax, index);
        Span<float> m = mt.MutableData<float>(), v = vt.MutableData<float>();
        Span<float> vmax = max.IsEmpty ? default : max.MutableData<float>();
        int threads = TensorCompute.Threads;
        if (threads == 1 || parameter.Length < UpdateJob.ChunkSize)
        {
            UpdateRange(parameter, gradient, m, v, vmax);
            return;
        }

        _updateJob.Parameter = Parameters[index];
        _updateJob.Gradient = Parameters[index].Grad;
        _updateJob.M = mt;
        _updateJob.V = vt;
        _updateJob.Max = max;
        try
        {
            ParallelFor.Run(_updateJob, 1 + (parameter.Length - 1) / UpdateJob.ChunkSize, threads, (long)parameter.Length * 8);
        }
        finally
        {
            _updateJob.Parameter = _updateJob.Gradient = _updateJob.M = _updateJob.V = _updateJob.Max = default;
        }
    }

    private sealed class UpdateJob(Adam owner) : ChunkJob
    {
        internal const int ChunkSize = 32768;
        internal Tensor Parameter, Gradient, M, V, Max;

        public override void RunChunk(int chunk)
        {
            int start = chunk * ChunkSize, length = Math.Min(ChunkSize, Parameter.Length - start);
            // Step validated contiguous float32 storage and bumped each writable buffer's version before publishing this job.
            Span<float> Slice(Tensor t) => t.Buffer!.GetSpan<float>(t.Generation).Slice(t.Offset + start, length);
            owner.UpdateRange(Slice(Parameter), Gradient.Data<float>().Slice(start, length), Slice(M), Slice(V),
                Max.IsEmpty ? default : Slice(Max));
        }
    }

    private void UpdateRange(Span<float> p, ReadOnlySpan<float> grad, Span<float> m, Span<float> v, Span<float> vmax)
    {
        float wd = WeightDecay, lr = LearningRate;
        float b1 = _beta1, b2 = _beta2, omb1 = 1f - _beta1, omb2 = 1f - _beta2, eps = _epsilon, stepSize = _stepSize, sq = _sqrtBc2;
        float decay = 1f - lr * wd;
        for (int i = 0; i < p.Length; i++)
        {
            float g = grad[i];
            float pi = p[i];
            if (wd != 0f)
            {
                if (_decoupled)
                {
                    pi *= decay;
                }
                else
                {
                    g += wd * pi;
                }
            }

            float mi = b1 * m[i] + omb1 * g;
            float vi = b2 * v[i] + omb2 * g * g;
            m[i] = mi;
            v[i] = vi;
            float vd = vi;
            if (_amsgrad)
            {
                vd = vmax[i] > vi ? vmax[i] : vi;
                vmax[i] = vd;
            }

            float denom = MathF.Sqrt(vd) / sq + eps;
            p[i] = pi - stepSize * mi / denom;
        }
    }
}

/// <summary>AdamW (Loshchilov and Hutter 2019): <see cref="Adam"/> with decoupled weight decay and a default decay of 0.01.</summary>
public sealed class AdamW : Adam
{
    /// <summary>Creates the optimizer.</summary>
    /// <param name="parameters">Parameters to update.</param>
    /// <param name="learningRate">Learning rate (default 1e-3).</param>
    /// <param name="beta1">Decay of the first moment (default 0.9).</param>
    /// <param name="beta2">Decay of the second moment (default 0.999).</param>
    /// <param name="epsilon">Added to the denominator (default 1e-8).</param>
    /// <param name="weightDecay">Decoupled weight decay (default 0.01).</param>
    /// <param name="amsgrad">Use the AMSGrad variant (default false).</param>
    public AdamW(IEnumerable<Tensor> parameters, float learningRate = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f,
        float weightDecay = 0.01f, bool amsgrad = false)
        : base(parameters, learningRate, beta1, beta2, epsilon, weightDecay, decoupledWeightDecay: true, amsgrad: amsgrad)
    {
    }
}

/// <summary>
/// Adam with lazy moments for row-sparse gradients, matching <c>torch.optim.SparseAdam</c>: only the rows that appear in a step's
/// gradient have their moments and values updated, so a step costs time in the touched rows, not in the whole table. With t the step
/// number (1 for the first), for each touched element: <c>m = β₁m + (1−β₁)g</c>, <c>v = β₂v + (1−β₂)g²</c>,
/// <c>p −= lr·√(1−β₂ᵗ)/(1−β₁ᵗ) · m / (√v + ε)</c> (Kingma and Ba 2015, the form with ε added to the uncorrected √v). Moments of
/// untouched rows neither decay nor move. There is no weight decay and no AMSGrad. The step number is the optimizer's global one, so
/// it counts every <see cref="Optimizer.Step"/> call, also those in which a parameter received no gradient.
/// </summary>
/// <remarks>
/// Like PyTorch's, it accepts only row-sparse gradients and throws on a dense one. The moments are dense [rows, width] tensors, so
/// the memory is the same as Adam's for that table; what is saved is the gradient and the per-step work. Use it for the embedding
/// (<see cref="Embedding"/> built with <c>sparse: true</c>) and another optimizer (for example <see cref="AdamW"/>) for the other
/// parameters. There is deliberately no sparse AdamW: decoupled weight decay would have to touch every row every step, and PyTorch has none.
/// All arithmetic is scalar float32 with <c>+ − × ÷ sqrt</c> only, so results are identical in fast and exact mode on every platform.
/// </remarks>
public sealed class SparseAdam : Optimizer
{
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _epsilon;
    private readonly Tensor[] _m;
    private readonly Tensor[] _v;
    private float _stepSize;

    /// <summary>Creates the optimizer.</summary>
    /// <param name="parameters">Parameters to update; rank 2, normally embedding tables with row-sparse gradients.</param>
    /// <param name="learningRate">Learning rate (default 1e-3).</param>
    /// <param name="beta1">Decay of the first moment, in [0, 1) (default 0.9).</param>
    /// <param name="beta2">Decay of the second moment, in [0, 1) (default 0.999).</param>
    /// <param name="epsilon">Added to the denominator, positive (default 1e-8).</param>
    public SparseAdam(IEnumerable<Tensor> parameters, float learningRate = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        : base(parameters, learningRate)
    {
        if (!(beta1 >= 0f && beta1 < 1f) || !(beta2 >= 0f && beta2 < 1f))
        {
            throw new ArgumentOutOfRangeException(nameof(beta1), "Betas must be in [0, 1).");
        }

        if (!(epsilon > 0f) || !float.IsFinite(epsilon))
        {
            throw new ArgumentOutOfRangeException(nameof(epsilon), epsilon, "Epsilon must be positive and finite.");
        }

        foreach (Tensor p in Parameters)
        {
            if (p.Shape.Rank != 2)
            {
                throw new ArgumentException("SparseAdam parameters must be rank 2 ([rows, width]) tables.", nameof(parameters));
            }
        }

        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _m = AddStateSlot("exp_avg");
        _v = AddStateSlot("exp_avg_sq");
    }

    /// <inheritdoc/>
    public override string Kind => "sparseadam";

    /// <inheritdoc/>
    public override List<(string Name, float Value)> HyperParameters() =>
        [("lr", LearningRate), ("beta1", _beta1), ("beta2", _beta2), ("epsilon", _epsilon)];

    /// <inheritdoc/>
    public override void SetHyperParameters(IReadOnlyDictionary<string, float> values)
    {
        if (values["beta1"] != _beta1 || values["beta2"] != _beta2 || values["epsilon"] != _epsilon)
        {
            throw new ArgumentException("The saved betas or epsilon differ from this optimizer's.");
        }

        CheckRate(values["lr"]);
        LearningRate = values["lr"];
    }

    /// <inheritdoc/>
    protected override void BeginStep()
    {
        long t = StepCount + 1;
        double bc1 = 1.0 - Adam.IntPow(_beta1, t);
        double bc2 = 1.0 - Adam.IntPow(_beta2, t);
        _stepSize = (float)((double)LearningRate * Math.Sqrt(bc2) / bc1);
    }

    /// <inheritdoc/>
    protected override void Update(int index, Span<float> parameter, ReadOnlySpan<float> gradient) =>
        throw new InvalidOperationException($"SparseAdam got a dense gradient for parameter {index}; it only handles row-sparse gradients (build the Embedding with sparse: true, or use Adam).");

    /// <inheritdoc/>
    protected override void UpdateSparse(int index, Span<float> parameter, RowSparseGradient gradient)
    {
        int width = gradient.DenseShape[1];
        float b1 = _beta1, b2 = _beta2, omb1 = 1f - _beta1, omb2 = 1f - _beta2, eps = _epsilon, stepSize = _stepSize;
        Span<float> m = StateOf(_m, index).MutableData<float>();
        Span<float> v = StateOf(_v, index).MutableData<float>();
        ReadOnlySpan<int> rows = gradient.Rows;
        ReadOnlySpan<float> g = gradient.Values.Data<float>();
        for (int k = 0; k < rows.Length; k++)
        {
            int o = rows[k] * width;
            for (int j = 0; j < width; j++)
            {
                float gi = g[k * width + j];
                float mi = b1 * m[o + j] + omb1 * gi;
                float vi = b2 * v[o + j] + omb2 * gi * gi;
                m[o + j] = mi;
                v[o + j] = vi;
                parameter[o + j] -= stepSize * mi / (MathF.Sqrt(vi) + eps);
            }
        }
    }
}
