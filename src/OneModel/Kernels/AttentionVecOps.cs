using System.Runtime.Intrinsics;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// The lane operations the attention, softmax and transcendental kernels are written against. One algorithm is written once over
/// this interface and instantiated for scalar, 128-, 256- and 512-bit lanes, so every width runs the same sequence of IEEE
/// operations on each lane. The exact instantiations (in <c>Kernels/Exact</c>) never fuse a multiply and an add; the fast ones may.
/// </summary>
/// <typeparam name="TV">The lane vector: <see cref="float"/> or <c>Vector128/256/512&lt;float&gt;</c>.</typeparam>
internal interface IVecOps<TV>
    where TV : struct
{
    /// <summary>Number of fp32 lanes in <typeparamref name="TV"/>.</summary>
    static abstract int Lanes { get; }

    /// <summary>True when <see cref="Accumulate"/> rounds once (fused). Exact instantiations return false.</summary>
    static abstract bool Fused { get; }

    /// <summary>All lanes zero.</summary>
    static abstract TV Zero { get; }

    /// <summary>Broadcasts a value to every lane.</summary>
    static abstract TV Splat(float x);

    /// <summary>Loads <see cref="Lanes"/> floats starting at <c>Unsafe.Add(ref p, index)</c>. The caller has checked the range.</summary>
    static abstract TV Load(ref float p, nuint index);

    /// <summary>Stores <see cref="Lanes"/> floats starting at <c>Unsafe.Add(ref p, index)</c>. The caller has checked the range.</summary>
    static abstract void Store(ref float p, nuint index, TV v);

    /// <summary>a + b, one IEEE rounding.</summary>
    static abstract TV Add(TV a, TV b);

    /// <summary>a - b, one IEEE rounding.</summary>
    static abstract TV Sub(TV a, TV b);

    /// <summary>a * b, one IEEE rounding.</summary>
    static abstract TV Mul(TV a, TV b);

    /// <summary>a / b, one IEEE rounding.</summary>
    static abstract TV Div(TV a, TV b);

    /// <summary>acc + a * b. Exact instantiations: a product rounding then a sum rounding. Fast ones: a single fused rounding where the CPU has it.</summary>
    static abstract TV Accumulate(TV acc, TV a, TV b);

    /// <summary>Lane mask (all ones or all zeros) for a &gt; b. False for NaN operands.</summary>
    static abstract TV GreaterThan(TV a, TV b);

    /// <summary>Lane mask for a &lt; b. False for NaN operands.</summary>
    static abstract TV LessThan(TV a, TV b);

    /// <summary>Lane mask for a &gt;= b. False for NaN operands.</summary>
    static abstract TV GreaterThanOrEqual(TV a, TV b);

    /// <summary>Lane mask for a == b. False for NaN operands.</summary>
    static abstract TV Equal(TV a, TV b);

    /// <summary>Per lane: the bits of <paramref name="ifTrue"/> where the mask is set, else the bits of <paramref name="ifFalse"/>.</summary>
    static abstract TV Select(TV mask, TV ifTrue, TV ifFalse);

    /// <summary>Bitwise a &amp; b.</summary>
    static abstract TV And(TV a, TV b);

    /// <summary>Bitwise a | b.</summary>
    static abstract TV Or(TV a, TV b);

    /// <summary>Bitwise a ^ b.</summary>
    static abstract TV Xor(TV a, TV b);

    /// <summary>Shifts each lane's 32 bits left, treating the lane as an integer.</summary>
    static abstract TV ShiftLeft(TV x, int count);

    /// <summary>Arithmetic right shift of each lane's 32 bits as a signed integer.</summary>
    static abstract TV ShiftRightArithmetic(TV x, int count);

    /// <summary>Integer (wrapping) addition of the lanes' 32-bit patterns.</summary>
    static abstract TV AddInt(TV a, TV b);

    /// <summary>Integer (wrapping) subtraction of the lanes' 32-bit patterns.</summary>
    static abstract TV SubInt(TV a, TV b);

    /// <summary>The exponential the attention and softmax kernels of this instantiation use.</summary>
    static abstract TV Exp(TV x);

    /// <summary>
    /// The exponential for softmax-like arguments (x &lt;= 0, or NaN): exact instantiations return exactly <see cref="Exp"/> on
    /// [-87.33654, 0] and +0 below that, with no overflow handling; fast ones are the same as <see cref="Exp"/>.
    /// </summary>
    static abstract TV SoftmaxExp(TV x);
}

/// <summary>Which lane width a kernel instantiation uses.</summary>
internal enum VecIsa
{
    /// <summary>The widest width the CPU accelerates in hardware.</summary>
    Auto,

    /// <summary>One lane, plain scalar code.</summary>
    Scalar,

    /// <summary>128-bit lanes: SSE, NEON or WASM SIMD.</summary>
    V128,

    /// <summary>256-bit lanes: AVX2.</summary>
    V256,

    /// <summary>512-bit lanes: AVX-512.</summary>
    V512,
}

internal static class VecIsaInfo
{
    /// <summary>Resolves <see cref="VecIsa.Auto"/> to the widest hardware-accelerated width.</summary>
    public static VecIsa Resolve(VecIsa isa)
    {
        if (isa != VecIsa.Auto)
        {
            return isa;
        }

        if (Vector512.IsHardwareAccelerated)
        {
            return VecIsa.V512;
        }

        if (Vector256.IsHardwareAccelerated)
        {
            return VecIsa.V256;
        }

        return Vector128.IsHardwareAccelerated ? VecIsa.V128 : VecIsa.Scalar;
    }

    /// <summary>True when the width runs in hardware here (so a timing of it means something).</summary>
    public static bool IsAccelerated(VecIsa isa) => isa switch
    {
        VecIsa.Scalar => true,
        VecIsa.V128 => Vector128.IsHardwareAccelerated,
        VecIsa.V256 => Vector256.IsHardwareAccelerated,
        VecIsa.V512 => Vector512.IsHardwareAccelerated,
        _ => true,
    };
}
