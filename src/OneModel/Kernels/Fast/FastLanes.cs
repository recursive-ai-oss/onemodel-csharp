using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace RecursiveAI.OneModel.Kernels.Fast;

/// <summary>One lane.</summary>
internal readonly struct FastLanes1 : ILanes<FastLanes1>
{
    private readonly float _v;

    private FastLanes1(float v) => _v = v;

    public static int Width => 1;

    public static FastLanes1 Zero => default;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FastLanes1 Broadcast(float value) => new(value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe FastLanes1 Load(float* source) => new(*source);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe FastLanes1 Gather(float* source, nint stride) => new(*source);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe void Store(float* destination) => *destination = _v;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FastLanes1 MulAdd(FastLanes1 a, FastLanes1 b, FastLanes1 c) => new(Fma.IsSupported || AdvSimd.IsSupported ? MathF.FusedMultiplyAdd(a._v, b._v, c._v) : c._v + (a._v * b._v));
}

/// <summary>4 lanes.</summary>
internal readonly struct FastLanes4 : ILanes<FastLanes4>
{
    private readonly Vector128<float> _v;

    private FastLanes4(Vector128<float> v) => _v = v;

    public static int Width => 4;

    public static FastLanes4 Zero => default;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FastLanes4 Broadcast(float value) => new(Vector128.Create(value));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe FastLanes4 Load(float* source) => new(Vector128.Load(source));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe FastLanes4 Gather(float* source, nint stride) => new(Vector128.Create(source[0], source[1 * stride], source[2 * stride], source[3 * stride]));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe void Store(float* destination) => _v.Store(destination);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FastLanes4 MulAdd(FastLanes4 a, FastLanes4 b, FastLanes4 c) => new(Fma.IsSupported || AdvSimd.IsSupported ? Vector128.FusedMultiplyAdd(a._v, b._v, c._v) : Vector128.Add(c._v, Vector128.Multiply(a._v, b._v)));
}

/// <summary>8 lanes.</summary>
internal readonly struct FastLanes8 : ILanes<FastLanes8>
{
    private readonly Vector256<float> _v;

    private FastLanes8(Vector256<float> v) => _v = v;

    public static int Width => 8;

    public static FastLanes8 Zero => default;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FastLanes8 Broadcast(float value) => new(Vector256.Create(value));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe FastLanes8 Load(float* source) => new(Vector256.Load(source));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe FastLanes8 Gather(float* source, nint stride) => new(Vector256.Create(source[0], source[1 * stride], source[2 * stride], source[3 * stride], source[4 * stride], source[5 * stride], source[6 * stride], source[7 * stride]));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe void Store(float* destination) => _v.Store(destination);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FastLanes8 MulAdd(FastLanes8 a, FastLanes8 b, FastLanes8 c) => new(Fma.IsSupported ? Vector256.FusedMultiplyAdd(a._v, b._v, c._v) : Vector256.Add(c._v, Vector256.Multiply(a._v, b._v)));
}
