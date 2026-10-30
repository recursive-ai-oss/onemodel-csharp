using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace RecursiveAI.OneModel.Kernels.Exact;

/// <summary>One lane.</summary>
internal readonly struct ExactLanes1 : ILanes<ExactLanes1>
{
    private readonly float _v;

    private ExactLanes1(float v) => _v = v;

    public static int Width => 1;

    public static ExactLanes1 Zero => default;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ExactLanes1 Broadcast(float value) => new(value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe ExactLanes1 Load(float* source) => new(*source);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe ExactLanes1 Gather(float* source, nint stride) => new(*source);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe void Store(float* destination) => *destination = _v;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ExactLanes1 MulAdd(ExactLanes1 a, ExactLanes1 b, ExactLanes1 c) => new(c._v + (a._v * b._v));
}

/// <summary>4 lanes.</summary>
internal readonly struct ExactLanes4 : ILanes<ExactLanes4>
{
    private readonly Vector128<float> _v;

    private ExactLanes4(Vector128<float> v) => _v = v;

    public static int Width => 4;

    public static ExactLanes4 Zero => default;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ExactLanes4 Broadcast(float value) => new(Vector128.Create(value));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe ExactLanes4 Load(float* source) => new(Vector128.Load(source));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe ExactLanes4 Gather(float* source, nint stride) => new(Vector128.Create(source[0], source[1 * stride], source[2 * stride], source[3 * stride]));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe void Store(float* destination) => _v.Store(destination);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ExactLanes4 MulAdd(ExactLanes4 a, ExactLanes4 b, ExactLanes4 c) => new(Vector128.Add(c._v, Vector128.Multiply(a._v, b._v)));
}

/// <summary>8 lanes.</summary>
internal readonly struct ExactLanes8 : ILanes<ExactLanes8>
{
    private readonly Vector256<float> _v;

    private ExactLanes8(Vector256<float> v) => _v = v;

    public static int Width => 8;

    public static ExactLanes8 Zero => default;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ExactLanes8 Broadcast(float value) => new(Vector256.Create(value));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe ExactLanes8 Load(float* source) => new(Vector256.Load(source));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe ExactLanes8 Gather(float* source, nint stride) => new(Vector256.Create(source[0], source[1 * stride], source[2 * stride], source[3 * stride], source[4 * stride], source[5 * stride], source[6 * stride], source[7 * stride]));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe void Store(float* destination) => _v.Store(destination);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ExactLanes8 MulAdd(ExactLanes8 a, ExactLanes8 b, ExactLanes8 c) => new(Vector256.Add(c._v, Vector256.Multiply(a._v, b._v)));
}

/// <summary>16 lanes (AVX-512).</summary>
internal readonly struct ExactLanes16 : ILanes<ExactLanes16>
{
    private readonly Vector512<float> _v;

    private ExactLanes16(Vector512<float> v) => _v = v;

    public static int Width => 16;

    public static ExactLanes16 Zero => default;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ExactLanes16 Broadcast(float value) => new(Vector512.Create(value));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe ExactLanes16 Load(float* source) => new(Vector512.Load(source));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe ExactLanes16 Gather(float* source, nint stride) => new(Vector512.Create(source[0],source[1 * stride],source[2 * stride],source[3 * stride],source[4 * stride],source[5 * stride],source[6 * stride],source[7 * stride],source[8 * stride],source[9 * stride],source[10 * stride],source[11 * stride],source[12 * stride],source[13 * stride],source[14 * stride],source[15 * stride]));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe void Store(float* destination) => _v.Store(destination);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ExactLanes16 MulAdd(ExactLanes16 a, ExactLanes16 b, ExactLanes16 c) => new(Vector512.Add(c._v, Vector512.Multiply(a._v, b._v)));
}
