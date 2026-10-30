using System;
using System.Collections.Generic;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Training;

/// <summary>
/// A multi-layer, optionally bidirectional LSTM or GRU over <see cref="Ops.Lstm"/> / <see cref="Ops.Gru"/> (PyTorch gate order and
/// semantics, zero initial states, no dropout between layers). Parameters are named as in PyTorch:
/// <c>weight_ih_l0</c>, <c>weight_hh_l0</c>, <c>bias_ih_l0</c>, <c>bias_hh_l0</c> and, for the reverse direction, the same with
/// <c>_reverse</c>, layer by layer (layer 0 forward, layer 0 reverse, layer 1 forward, ...).
/// </summary>
public sealed class Recurrent : NnModule
{
    private readonly RnnLayer[] _layers;
    private readonly bool _gru;
    private readonly bool _bidirectional;
    private readonly bool _batchFirst;

    /// <summary>Creates the module with weights drawn uniformly from ±1/√hiddenSize in registration order.</summary>
    /// <param name="gru">True for a GRU (3 gates), false for an LSTM (4 gates).</param>
    /// <param name="inputSize">Width of the input vectors, at least 1.</param>
    /// <param name="hiddenSize">Hidden width per direction, at least 1.</param>
    /// <param name="generator">Source of the initial values.</param>
    /// <param name="layers">Number of stacked layers, at least 1 (default 1).</param>
    /// <param name="bidirectional">Run each layer in both directions; outputs are concatenated (default false).</param>
    /// <param name="batchFirst">Input and output are [batch, time, width] rather than [time, batch, width] (default false).</param>
    public Recurrent(bool gru, int inputSize, int hiddenSize, Generator generator, int layers = 1, bool bidirectional = false, bool batchFirst = false)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(inputSize, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(hiddenSize, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(layers, 1);
        ArgumentNullException.ThrowIfNull(generator);
        _gru = gru;
        _bidirectional = bidirectional;
        _batchFirst = batchFirst;
        int gates = gru ? 3 : 4, dirs = bidirectional ? 2 : 1;
        float bound = 1f / MathF.Sqrt(hiddenSize);
        var list = new List<RnnLayer>();
        for (int l = 0; l < layers; l++)
        {
            for (int d = 0; d < dirs; d++)
            {
                string suffix = "_l" + l.ToString(System.Globalization.CultureInfo.InvariantCulture) + (d == 1 ? "_reverse" : "");
                int inSize = l == 0 ? inputSize : hiddenSize * dirs;
                Tensor wIh = RegisterParameter("weight_ih" + suffix, Init.Uniform(generator, bound, gates * hiddenSize, inSize));
                Tensor wHh = RegisterParameter("weight_hh" + suffix, Init.Uniform(generator, bound, gates * hiddenSize, hiddenSize));
                Tensor bIh = RegisterParameter("bias_ih" + suffix, Init.Uniform(generator, bound, gates * hiddenSize));
                Tensor bHh = RegisterParameter("bias_hh" + suffix, Init.Uniform(generator, bound, gates * hiddenSize));
                list.Add(new RnnLayer(wIh, wHh, bIh, bHh));
            }
        }

        _layers = list.ToArray();
    }

    /// <summary>Runs the stack over a float32 sequence and returns the last layer's hidden states [time, batch, directions·hidden] (or batch-first).</summary>
    /// <param name="input">Float32 [time, batch, inputSize], or [batch, time, inputSize] with batch-first.</param>
    public override Tensor Forward(Tensor input) =>
        _gru ? Ops.Gru(input, _layers, _bidirectional, _batchFirst) : Ops.Lstm(input, _layers, _bidirectional, _batchFirst);
}

/// <summary>Transposed 2-D convolution over NCHW float32 input (<see cref="Ops.ConvTranspose2d"/>).</summary>
public sealed class ConvTranspose2d : NnModule
{
    private readonly ConvTranspose2dParams _geometry;

    /// <summary>Weights [in, out/groups, kernelH, kernelW].</summary>
    public Tensor Weight { get; }

    /// <summary>Bias [out], or an empty tensor when the layer has none.</summary>
    public Tensor Bias { get; }

    /// <summary>Creates the layer with weights and bias drawn uniformly from ±1/√(out/groups·k·k).</summary>
    /// <param name="inChannels">Input channels, divisible by <paramref name="groups"/>.</param>
    /// <param name="outChannels">Output channels, divisible by <paramref name="groups"/>.</param>
    /// <param name="kernelSize">Square kernel size, at least 1.</param>
    /// <param name="generator">Source of the initial values.</param>
    /// <param name="stride">Stride on both axes (default 1).</param>
    /// <param name="padding">Padding on both axes (default 0).</param>
    /// <param name="outputPadding">Extra size added to one side of the output on both axes (default 0).</param>
    /// <param name="dilation">Dilation on both axes (default 1).</param>
    /// <param name="groups">Channel groups (default 1).</param>
    /// <param name="bias">Whether to add a bias (default true).</param>
    public ConvTranspose2d(int inChannels, int outChannels, int kernelSize, Generator generator, int stride = 1, int padding = 0,
        int outputPadding = 0, int dilation = 1, int groups = 1, bool bias = true)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(kernelSize, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(groups, 1);
        ArgumentNullException.ThrowIfNull(generator);
        if (inChannels < 1 || outChannels < 1 || inChannels % groups != 0 || outChannels % groups != 0)
        {
            throw new ArgumentException($"Channels {inChannels} -> {outChannels} must be positive and divisible by groups {groups}.");
        }

        _geometry = new ConvTranspose2dParams(stride, stride, padding, padding, outputPadding, outputPadding, dilation, dilation, groups);
        float bound = 1f / MathF.Sqrt(outChannels / groups * kernelSize * kernelSize);
        Weight = RegisterParameter("weight", Init.Uniform(generator, bound, inChannels, outChannels / groups, kernelSize, kernelSize));
        if (bias)
        {
            Bias = RegisterParameter("bias", Init.Uniform(generator, bound, outChannels));
        }
    }

    /// <summary>Applies the transposed convolution to [N, in, H, W].</summary>
    /// <param name="input">Float32 NCHW input.</param>
    public override Tensor Forward(Tensor input) => Ops.ConvTranspose2d(input, Weight, Bias, _geometry);
}
