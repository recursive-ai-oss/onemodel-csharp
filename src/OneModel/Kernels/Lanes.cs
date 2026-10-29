namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// A short vector of fp32 lanes with the few operations the direct (depthwise) convolution needs. Every kernel written over this
/// interface treats each lane as an independent output, so the result does not depend on the lane count: the one-lane, four-lane
/// and eight-lane versions produce the same bits. Implementations in Exact/ never fuse the multiply and add; those in Fast/ may.
/// </summary>
internal interface ILanes<TSelf>
    where TSelf : struct, ILanes<TSelf>
{
    /// <summary>Number of fp32 lanes.</summary>
    static abstract int Width { get; }

    /// <summary>All lanes +0.</summary>
    static abstract TSelf Zero { get; }

    /// <summary>The same value in every lane.</summary>
    static abstract TSelf Broadcast(float value);

    /// <summary>Loads <see cref="Width"/> consecutive values.</summary>
    static abstract unsafe TSelf Load(float* source);

    /// <summary>Loads <see cref="Width"/> values spaced <paramref name="stride"/> floats apart.</summary>
    static abstract unsafe TSelf Gather(float* source, nint stride);

    /// <summary>Stores <see cref="Width"/> consecutive values.</summary>
    unsafe void Store(float* destination);

    /// <summary><c>c + a * b</c> per lane: with two roundings in exact mode, possibly one in fast mode.</summary>
    static abstract TSelf MulAdd(TSelf a, TSelf b, TSelf c);
}
