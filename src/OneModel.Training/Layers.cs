using System;
using System.Collections.Generic;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Tracing;

namespace RecursiveAI.OneModel.Training;

internal static class Init
{
    // A float32 tensor of the shape filled with uniform values in [-bound, bound), drawn in row-major order.
    public static Tensor Uniform(Generator g, float bound, params ReadOnlySpan<int> dims)
    {
        Tensor t = Tensor.Create(DType.F32, Shape.Create(dims));
        Span<float> d = t.MutableData<float>();
        for (int i = 0; i < d.Length; i++)
        {
            d[i] = g.Uniform(-bound, bound);
        }

        return t;
    }

    public static Tensor Constant(float value, params ReadOnlySpan<int> dims)
    {
        Tensor t = Tensor.Create(DType.F32, Shape.Create(dims));
        t.Fill(value);
        return t;
    }

    // A rank-0 float32 tensor that the open ActivationScope (if any) releases.
    public static Tensor Scalar(float value)
    {
        Tensor t = Tensor.Create(DType.F32, default);
        t.Fill(value);
        ActivationScope.Track(t);
        return t;
    }
}

/// <summary>Fully connected layer <c>y = x·Wᵀ + b</c> over the last axis.</summary>
public sealed class Linear : NnModule
{
    /// <summary>Weights [out, in].</summary>
    public Tensor Weight { get; }

    /// <summary>Bias [out], or an empty tensor when the layer has none.</summary>
    public Tensor Bias { get; }

    /// <summary>Creates the layer with weights and bias drawn uniformly from ±1/√in (the fan-in bound, in row-major order).</summary>
    /// <param name="inFeatures">Input width, at least 1.</param>
    /// <param name="outFeatures">Output width, at least 1.</param>
    /// <param name="generator">Source of the initial values.</param>
    /// <param name="bias">Whether to add a bias (default true).</param>
    public Linear(int inFeatures, int outFeatures, Generator generator, bool bias = true)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(inFeatures, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(outFeatures, 1);
        ArgumentNullException.ThrowIfNull(generator);
        float bound = 1f / MathF.Sqrt(inFeatures);
        Weight = RegisterParameter("weight", Init.Uniform(generator, bound, outFeatures, inFeatures));
        if (bias)
        {
            Bias = RegisterParameter("bias", Init.Uniform(generator, bound, outFeatures));
        }
    }

    /// <summary>Applies the layer to a float32 tensor [..., in]; the result is [..., out].</summary>
    /// <param name="input">Float32 input.</param>
    public override Tensor Forward(Tensor input) => Ops.Linear(input, Weight, Bias);
}

/// <summary>2-D convolution over NCHW float32 input.</summary>
public sealed class Conv2d : NnModule
{
    private readonly Conv2dParams _geometry;

    /// <summary>Weights [out, in/groups, kernelH, kernelW].</summary>
    public Tensor Weight { get; }

    /// <summary>Bias [out], or an empty tensor when the layer has none.</summary>
    public Tensor Bias { get; }

    /// <summary>Creates the layer with weights and bias drawn uniformly from ±1/√fanIn, fanIn = in/groups·kernelH·kernelW.</summary>
    /// <param name="inChannels">Input channels, at least 1, divisible by <paramref name="groups"/>.</param>
    /// <param name="outChannels">Output channels, at least 1, divisible by <paramref name="groups"/>.</param>
    /// <param name="kernelSize">Square kernel size, at least 1.</param>
    /// <param name="generator">Source of the initial values.</param>
    /// <param name="stride">Stride on both axes (default 1).</param>
    /// <param name="padding">Zero padding on both axes (default 0).</param>
    /// <param name="dilation">Dilation on both axes (default 1).</param>
    /// <param name="groups">Channel groups (default 1).</param>
    /// <param name="bias">Whether to add a bias (default true).</param>
    public Conv2d(int inChannels, int outChannels, int kernelSize, Generator generator, int stride = 1, int padding = 0,
        int dilation = 1, int groups = 1, bool bias = true)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(kernelSize, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(groups, 1);
        ArgumentNullException.ThrowIfNull(generator);
        if (inChannels < 1 || outChannels < 1 || inChannels % groups != 0 || outChannels % groups != 0)
        {
            throw new ArgumentException($"Channels {inChannels} -> {outChannels} must be positive and divisible by groups {groups}.");
        }

        _geometry = new Conv2dParams(stride, stride, padding, padding, dilation, dilation, groups);
        float bound = 1f / MathF.Sqrt(inChannels / groups * kernelSize * kernelSize);
        Weight = RegisterParameter("weight", Init.Uniform(generator, bound, outChannels, inChannels / groups, kernelSize, kernelSize));
        if (bias)
        {
            Bias = RegisterParameter("bias", Init.Uniform(generator, bound, outChannels));
        }
    }

    /// <summary>Convolves a float32 tensor [N, in, H, W]; the result is [N, out, H', W'].</summary>
    /// <param name="input">Float32 NCHW input.</param>
    public override Tensor Forward(Tensor input) => Ops.Conv2d(input, Weight, Bias, _geometry);
}

/// <summary>Batch normalization over the channels of an NCHW tensor, with running statistics.</summary>
public sealed class BatchNorm2d : NnModule
{
    private readonly float _momentum;
    private readonly float _epsilon;

    /// <summary>Scale γ [C], initially 1.</summary>
    public Tensor Weight { get; }

    /// <summary>Shift β [C], initially 0.</summary>
    public Tensor Bias { get; }

    /// <summary>Running mean [C] (a buffer), initially 0.</summary>
    public Tensor RunningMean { get; }

    /// <summary>Running variance [C] (a buffer), initially 1.</summary>
    public Tensor RunningVar { get; }

    /// <summary>Creates the layer.</summary>
    /// <param name="channels">Channel count, at least 1.</param>
    /// <param name="momentum">Weight of the new batch statistic in the running ones (default 0.1).</param>
    /// <param name="epsilon">Added to the variance (default 1e-5).</param>
    public BatchNorm2d(int channels, float momentum = 0.1f, float epsilon = 1e-5f)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(channels, 1);
        _momentum = momentum;
        _epsilon = epsilon;
        Weight = RegisterParameter("weight", Init.Constant(1f, channels));
        Bias = RegisterParameter("bias", Init.Constant(0f, channels));
        RunningMean = RegisterBuffer("running_mean", Init.Constant(0f, channels));
        RunningVar = RegisterBuffer("running_var", Init.Constant(1f, channels));
    }

    /// <summary>Normalizes [N, C, H, W]: batch statistics (and the running ones are updated) in training mode, the running statistics in evaluation mode.</summary>
    /// <param name="input">Float32 NCHW input.</param>
    public override Tensor Forward(Tensor input) =>
        Ops.BatchNorm2d(input, Weight, Bias, RunningMean, RunningVar, IsTraining, _momentum, _epsilon);
}

/// <summary>Layer normalization over the last axis.</summary>
public sealed class LayerNorm : NnModule
{
    private readonly float _epsilon;

    /// <summary>Scale γ [features], initially 1.</summary>
    public Tensor Weight { get; }

    /// <summary>Shift β [features], initially 0.</summary>
    public Tensor Bias { get; }

    /// <summary>Creates the layer.</summary>
    /// <param name="features">Size of the last axis, at least 1.</param>
    /// <param name="epsilon">Added to the variance (default 1e-5).</param>
    public LayerNorm(int features, float epsilon = 1e-5f)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(features, 1);
        _epsilon = epsilon;
        Weight = RegisterParameter("weight", Init.Constant(1f, features));
        Bias = RegisterParameter("bias", Init.Constant(0f, features));
    }

    /// <summary>Normalizes each vector along the last axis.</summary>
    /// <param name="input">Float32 input [..., features].</param>
    public override Tensor Forward(Tensor input) => Ops.LayerNorm(input, Weight, Bias, _epsilon);
}

/// <summary>A lookup table of vectors indexed by integer ids.</summary>
public sealed class Embedding : NnModule
{
    private readonly int _paddingIndex;

    /// <summary>The table [count, width].</summary>
    public Tensor Weight { get; }

    /// <summary>Creates the table with values uniform in ±√3 (unit variance), in row-major order.</summary>
    /// <param name="count">Number of ids, at least 1.</param>
    /// <param name="width">Vector width, at least 1.</param>
    /// <param name="generator">Source of the initial values.</param>
    /// <param name="paddingIndex">An id whose row receives no gradient, or -1 for none (default).</param>
    public Embedding(int count, int width, Generator generator, int paddingIndex = -1)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(count, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentNullException.ThrowIfNull(generator);
        _paddingIndex = paddingIndex;
        Weight = RegisterParameter("weight", Init.Uniform(generator, 1.7320508f, count, width));
    }

    /// <summary>Looks up Int32 or Int64 ids of any shape; the result has the id shape plus the vector width.</summary>
    /// <param name="input">Ids.</param>
    public override Tensor Forward(Tensor input) => Ops.Embedding(Weight, input, _paddingIndex);
}

/// <summary>
/// Dropout: in training mode each element is zeroed with probability <c>p</c> and the rest are scaled by <c>1/(1-p)</c>. The mask is a
/// pure function of the seed, the number of masks drawn so far (a saved Int64 buffer, so checkpoints resume the stream) and the
/// element index, so it does not depend on threads or platform.
/// </summary>
public sealed class Dropout : NnModule
{
    private readonly float _p;
    private readonly ulong _seed;
    private readonly Tensor _counter;

    /// <summary>Creates the module.</summary>
    /// <param name="p">Drop probability in [0, 1).</param>
    /// <param name="seed">Seed of the mask stream.</param>
    public Dropout(float p, ulong seed)
    {
        if (!(p >= 0f && p < 1f))
        {
            throw new ArgumentOutOfRangeException(nameof(p), p, "Dropout probability must be in [0, 1).");
        }

        _p = p;
        _seed = seed;
        _counter = RegisterBuffer("draws", Tensor.Zeros(DType.I64, Shape.Create(1)));
    }

    /// <summary>Number of masks drawn so far.</summary>
    public long Draws => _counter.Data<long>()[0];

    /// <summary>Drops elements in training mode (a float32 tensor of any shape); returns the input itself in evaluation mode or when p is 0.</summary>
    /// <param name="input">Float32 input.</param>
    public override Tensor Forward(Tensor input)
    {
        if (!IsTraining || _p == 0f)
        {
            return input;
        }

        if (Tracer.IsTracing)
        {
            throw new TraceException(TraceErrorKind.UnsupportedOperation,
                "Dropout in training mode draws a random mask on every call, which a traced graph would freeze. Call Eval() on the module before tracing.");
        }

        long draw = _counter.Data<long>()[0];
        _counter.MutableData<long>()[0] = draw + 1;
        Tensor mask = Tensor.Create(DType.F32, input.Shape);
        Span<float> m = mask.MutableData<float>();
        float scale = 1f / (1f - _p);
        ulong stream = Generator.Mix(_seed + (ulong)draw * 0x9E3779B97F4A7C15UL);
        for (int i = 0; i < m.Length; i++)
        {
            ulong bits = Generator.Mix(stream + (ulong)(i + 1) * 0x9E3779B97F4A7C15UL);
            float u = (bits >> 40) * (1f / 16777216f);
            m[i] = u >= _p ? scale : 0f;
        }

        ActivationScope.Track(mask);
        return Ops.Mul(input, mask);
    }
}

/// <summary>The activation functions and shape adapters needed to chain layers in a <see cref="Sequential"/>.</summary>
public sealed class Lambda : NnModule
{
    private readonly Func<Tensor, Tensor> _fn;

    private Lambda(Func<Tensor, Tensor> fn) => _fn = fn;

    /// <summary>Rectified linear unit.</summary>
    public static Lambda Relu() => new(x => Ops.Relu(x));

    /// <summary>GELU, erf form by default.</summary>
    /// <param name="approximateTanh">Use the tanh approximation.</param>
    public static Lambda Gelu(bool approximateTanh = false) => new(x => Ops.Gelu(x, approximateTanh));

    /// <summary>Logistic sigmoid.</summary>
    public static Lambda Sigmoid() => new(x => Ops.Sigmoid(x));

    /// <summary>Hyperbolic tangent.</summary>
    public static Lambda Tanh() => new(x => Ops.Tanh(x));

    /// <summary>2-D max pooling with a square window.</summary>
    /// <param name="kernel">Window size.</param>
    /// <param name="stride">Stride; 0 (default) means the window size.</param>
    /// <param name="padding">Padding on both axes (default 0).</param>
    public static Lambda MaxPool2d(int kernel, int stride = 0, int padding = 0)
    {
        int s = stride == 0 ? kernel : stride;
        return new(x => Ops.MaxPool2d(x, new Pool2dParams(kernel, kernel, s, s, padding, padding)));
    }

    /// <summary>2-D average pooling with a square window (padding counted in the average).</summary>
    /// <param name="kernel">Window size.</param>
    /// <param name="stride">Stride; 0 (default) means the window size.</param>
    /// <param name="padding">Padding on both axes (default 0).</param>
    public static Lambda AvgPool2d(int kernel, int stride = 0, int padding = 0)
    {
        int s = stride == 0 ? kernel : stride;
        return new(x => Ops.AvgPool2d(x, new Pool2dParams(kernel, kernel, s, s, padding, padding)));
    }

    /// <summary>Global average pooling of [N, C, H, W] to [N, C, 1, 1].</summary>
    public static Lambda GlobalAvgPool2d() => new(x => Ops.GlobalAvgPool2d(x));

    /// <summary>Flattens every axis after the first: [N, ...] to [N, rest].</summary>
    public static Lambda Flatten() => new(x => x.Reshape(x.Shape[0], x.Length / Math.Max(1, x.Shape[0])));

    /// <summary>Mean over one axis (negative counts from the end), removing it.</summary>
    /// <param name="axis">The axis to average.</param>
    public static Lambda Mean(int axis) => new(x => Ops.Mean(x, axis));

    /// <summary>Applies the function.</summary>
    /// <param name="input">The input.</param>
    public override Tensor Forward(Tensor input) => _fn(input);
}

/// <summary>Applies child modules one after another. Children are named "0", "1", ...</summary>
public sealed class Sequential : NnModule
{
    /// <summary>Creates the container.</summary>
    /// <param name="layers">The modules, in order; the container takes ownership.</param>
    public Sequential(params NnModule[] layers)
    {
        ArgumentNullException.ThrowIfNull(layers);
        for (int i = 0; i < layers.Length; i++)
        {
            RegisterModule(i.ToString(System.Globalization.CultureInfo.InvariantCulture), layers[i]);
        }
    }

    /// <summary>Feeds the input through every child.</summary>
    /// <param name="input">The first child's input.</param>
    public override Tensor Forward(Tensor input)
    {
        Tensor x = input;
        foreach ((_, NnModule m) in Children)
        {
            x = m.Forward(x);
        }

        return x;
    }
}

/// <summary>
/// Multi-head self-attention over [batch, length, width]: query, key, value and output projections (<c>q</c>, <c>k</c>, <c>v</c>,
/// <c>o</c>, each a <see cref="Linear"/> with bias) around <see cref="Ops.Attention"/>.
/// </summary>
public sealed class MultiHeadAttention : NnModule
{
    private readonly int _heads;

    /// <summary>Query projection.</summary>
    public Linear Q { get; }

    /// <summary>Key projection.</summary>
    public Linear K { get; }

    /// <summary>Value projection.</summary>
    public Linear V { get; }

    /// <summary>Output projection.</summary>
    public Linear O { get; }

    /// <summary>Creates the module.</summary>
    /// <param name="width">Model width, divisible by <paramref name="heads"/>.</param>
    /// <param name="heads">Number of heads, at least 1.</param>
    /// <param name="generator">Source of the initial weights (drawn in the order q, k, v, o).</param>
    public MultiHeadAttention(int width, int heads, Generator generator)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(heads, 1);
        if (width < 1 || width % heads != 0)
        {
            throw new ArgumentException($"Width {width} must be positive and divisible by the head count {heads}.", nameof(width));
        }

        _heads = heads;
        Q = RegisterModule("q", new Linear(width, width, generator));
        K = RegisterModule("k", new Linear(width, width, generator));
        V = RegisterModule("v", new Linear(width, width, generator));
        O = RegisterModule("o", new Linear(width, width, generator));
    }

    /// <summary>Self-attention without a mask.</summary>
    /// <param name="input">Float32 [batch, length, width].</param>
    public override Tensor Forward(Tensor input) => Forward(input, default);

    /// <summary>Self-attention with an additive mask.</summary>
    /// <param name="input">Float32 [batch, length, width].</param>
    /// <param name="mask">Additive score mask broadcastable to [batch, heads, length, length] (0 keeps, a large negative value hides), or an empty tensor for none. It gets no gradient.</param>
    public Tensor Forward(Tensor input, Tensor mask)
    {
        if (input.Rank != 3)
        {
            throw new TensorShapeException($"MultiHeadAttention needs [batch, length, width], not {input.Shape}.");
        }

        int b = input.Shape[0], n = input.Shape[1], d = input.Shape[2], dh = d / _heads;
        Tensor q = Split(Q.Forward(input), b, n, dh);
        Tensor k = Split(K.Forward(input), b, n, dh);
        Tensor v = Split(V.Forward(input), b, n, dh);
        Tensor a = Ops.Attention(q, k, v, mask).Permute(0, 2, 1, 3).Reshape(b, n, d);
        return O.Forward(a);
    }

    private Tensor Split(Tensor z, int b, int n, int dh) => z.Reshape(b, n, _heads, dh).Permute(0, 2, 1, 3);
}

/// <summary>
/// A post-norm transformer encoder layer (Vaswani et al. 2017): <c>x1 = LN(x + Drop(Attn(x)))</c>, <c>y = LN(x1 + Drop(W2·GELU(W1·x1)))</c>.
/// Children: <c>attn</c>, <c>norm1</c>, <c>ff1</c>, <c>ff2</c>, <c>norm2</c> and, when dropout is on, <c>drop1</c> and <c>drop2</c>.
/// </summary>
public sealed class TransformerEncoderLayer : NnModule
{
    private readonly MultiHeadAttention _attn;
    private readonly LayerNorm _norm1;
    private readonly Linear _ff1;
    private readonly Linear _ff2;
    private readonly LayerNorm _norm2;
    private readonly Dropout? _drop1;
    private readonly Dropout? _drop2;

    /// <summary>Creates the layer.</summary>
    /// <param name="width">Model width, divisible by <paramref name="heads"/>.</param>
    /// <param name="heads">Attention heads.</param>
    /// <param name="feedForwardWidth">Hidden width of the feed-forward block, at least 1.</param>
    /// <param name="generator">Source of the initial weights (attention, then ff1, then ff2).</param>
    /// <param name="dropout">Dropout probability after the attention and feed-forward blocks, in [0, 1) (default 0: no dropout modules).</param>
    /// <param name="dropoutSeed">Seed for the two dropout streams (the second uses <c>seed + 1</c>).</param>
    public TransformerEncoderLayer(int width, int heads, int feedForwardWidth, Generator generator, float dropout = 0f, ulong dropoutSeed = 0)
    {
        _attn = RegisterModule("attn", new MultiHeadAttention(width, heads, generator));
        _norm1 = RegisterModule("norm1", new LayerNorm(width));
        _ff1 = RegisterModule("ff1", new Linear(width, feedForwardWidth, generator));
        _ff2 = RegisterModule("ff2", new Linear(feedForwardWidth, width, generator));
        _norm2 = RegisterModule("norm2", new LayerNorm(width));
        if (dropout > 0f)
        {
            _drop1 = RegisterModule("drop1", new Dropout(dropout, dropoutSeed));
            _drop2 = RegisterModule("drop2", new Dropout(dropout, dropoutSeed + 1));
        }
    }

    /// <summary>Applies the layer without a mask.</summary>
    /// <param name="input">Float32 [batch, length, width].</param>
    public override Tensor Forward(Tensor input) => Forward(input, default);

    /// <summary>Applies the layer with an additive attention mask (see <see cref="MultiHeadAttention.Forward(Tensor, Tensor)"/>).</summary>
    /// <param name="input">Float32 [batch, length, width].</param>
    /// <param name="mask">Additive mask or an empty tensor.</param>
    public Tensor Forward(Tensor input, Tensor mask)
    {
        Tensor a = _attn.Forward(input, mask);
        if (_drop1 is not null)
        {
            a = _drop1.Forward(a);
        }

        Tensor x1 = _norm1.Forward(Ops.Add(input, a));
        Tensor f = _ff2.Forward(Ops.Gelu(_ff1.Forward(x1)));
        if (_drop2 is not null)
        {
            f = _drop2.Forward(f);
        }

        return _norm2.Forward(Ops.Add(x1, f));
    }
}
