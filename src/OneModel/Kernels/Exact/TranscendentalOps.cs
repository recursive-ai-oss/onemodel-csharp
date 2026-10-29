using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace RecursiveAI.OneModel.Kernels.Exact;

// Lane operations for the exact kernels. Every operation is a single IEEE-754 fp32 operation (+, -, *, /, sqrt), a comparison, a
// bit operation or an integer operation on the lane bits, so each lane computes what the scalar twin computes. Accumulate is a
// product rounding followed by a sum rounding: nothing here fuses a multiply and an add.

internal readonly struct ExactVec128 : IVecOps<Vector128<float>>
{
    public static int Lanes => Vector128<float>.Count;
    public static bool Fused => false;
    public static Vector128<float> Zero => Vector128<float>.Zero;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Splat(float x) => Vector128.Create(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Load(ref float p, nuint index) => Vector128.LoadUnsafe(ref p, index);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store(ref float p, nuint index, Vector128<float> v) => v.StoreUnsafe(ref p, index);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Add(Vector128<float> a, Vector128<float> b) => a + b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Sub(Vector128<float> a, Vector128<float> b) => a - b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Mul(Vector128<float> a, Vector128<float> b) => a * b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Div(Vector128<float> a, Vector128<float> b) => a / b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Accumulate(Vector128<float> acc, Vector128<float> a, Vector128<float> b)
    {
        Vector128<float> product = a * b;
        return acc + product;
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> GreaterThan(Vector128<float> a, Vector128<float> b) => Vector128.GreaterThan(a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> LessThan(Vector128<float> a, Vector128<float> b) => Vector128.LessThan(a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> GreaterThanOrEqual(Vector128<float> a, Vector128<float> b) => Vector128.GreaterThanOrEqual(a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Equal(Vector128<float> a, Vector128<float> b) => Vector128.Equals(a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Select(Vector128<float> mask, Vector128<float> ifTrue, Vector128<float> ifFalse) => Vector128.ConditionalSelect(mask, ifTrue, ifFalse);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> And(Vector128<float> a, Vector128<float> b) => a & b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Or(Vector128<float> a, Vector128<float> b) => a | b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Xor(Vector128<float> a, Vector128<float> b) => a ^ b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> ShiftLeft(Vector128<float> x, int count) => Vector128.ShiftLeft(x.AsInt32(), count).AsSingle();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> ShiftRightArithmetic(Vector128<float> x, int count) => Vector128.ShiftRightArithmetic(x.AsInt32(), count).AsSingle();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> AddInt(Vector128<float> a, Vector128<float> b) => Vector128.Add(a.AsInt32(), b.AsInt32()).AsSingle();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> SubInt(Vector128<float> a, Vector128<float> b) => Vector128.Subtract(a.AsInt32(), b.AsInt32()).AsSingle();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Exp(Vector128<float> x) => Transcendental.Exp<Vector128<float>, ExactVec128>(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> SoftmaxExp(Vector128<float> x) => Transcendental.ExpNonPositive<Vector128<float>, ExactVec128>(x);
}

internal readonly struct ExactVec256 : IVecOps<Vector256<float>>
{
    public static int Lanes => Vector256<float>.Count;
    public static bool Fused => false;
    public static Vector256<float> Zero => Vector256<float>.Zero;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Splat(float x) => Vector256.Create(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Load(ref float p, nuint index) => Vector256.LoadUnsafe(ref p, index);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store(ref float p, nuint index, Vector256<float> v) => v.StoreUnsafe(ref p, index);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Add(Vector256<float> a, Vector256<float> b) => a + b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Sub(Vector256<float> a, Vector256<float> b) => a - b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Mul(Vector256<float> a, Vector256<float> b) => a * b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Div(Vector256<float> a, Vector256<float> b) => a / b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Accumulate(Vector256<float> acc, Vector256<float> a, Vector256<float> b)
    {
        Vector256<float> product = a * b;
        return acc + product;
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> GreaterThan(Vector256<float> a, Vector256<float> b) => Vector256.GreaterThan(a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> LessThan(Vector256<float> a, Vector256<float> b) => Vector256.LessThan(a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> GreaterThanOrEqual(Vector256<float> a, Vector256<float> b) => Vector256.GreaterThanOrEqual(a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Equal(Vector256<float> a, Vector256<float> b) => Vector256.Equals(a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Select(Vector256<float> mask, Vector256<float> ifTrue, Vector256<float> ifFalse) => Vector256.ConditionalSelect(mask, ifTrue, ifFalse);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> And(Vector256<float> a, Vector256<float> b) => a & b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Or(Vector256<float> a, Vector256<float> b) => a | b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Xor(Vector256<float> a, Vector256<float> b) => a ^ b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> ShiftLeft(Vector256<float> x, int count) => Vector256.ShiftLeft(x.AsInt32(), count).AsSingle();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> ShiftRightArithmetic(Vector256<float> x, int count) => Vector256.ShiftRightArithmetic(x.AsInt32(), count).AsSingle();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> AddInt(Vector256<float> a, Vector256<float> b) => Vector256.Add(a.AsInt32(), b.AsInt32()).AsSingle();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> SubInt(Vector256<float> a, Vector256<float> b) => Vector256.Subtract(a.AsInt32(), b.AsInt32()).AsSingle();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Exp(Vector256<float> x) => Transcendental.Exp<Vector256<float>, ExactVec256>(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> SoftmaxExp(Vector256<float> x) => Transcendental.ExpNonPositive<Vector256<float>, ExactVec256>(x);
}

internal readonly struct ExactVec512 : IVecOps<Vector512<float>>
{
    public static int Lanes => Vector512<float>.Count;
    public static bool Fused => false;
    public static Vector512<float> Zero => Vector512<float>.Zero;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Splat(float x) => Vector512.Create(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Load(ref float p, nuint index) => Vector512.LoadUnsafe(ref p, index);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store(ref float p, nuint index, Vector512<float> v) => v.StoreUnsafe(ref p, index);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Add(Vector512<float> a, Vector512<float> b) => a + b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Sub(Vector512<float> a, Vector512<float> b) => a - b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Mul(Vector512<float> a, Vector512<float> b) => a * b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Div(Vector512<float> a, Vector512<float> b) => a / b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Accumulate(Vector512<float> acc, Vector512<float> a, Vector512<float> b)
    {
        Vector512<float> product = a * b;
        return acc + product;
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> GreaterThan(Vector512<float> a, Vector512<float> b) => Vector512.GreaterThan(a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> LessThan(Vector512<float> a, Vector512<float> b) => Vector512.LessThan(a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> GreaterThanOrEqual(Vector512<float> a, Vector512<float> b) => Vector512.GreaterThanOrEqual(a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Equal(Vector512<float> a, Vector512<float> b) => Vector512.Equals(a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Select(Vector512<float> mask, Vector512<float> ifTrue, Vector512<float> ifFalse) => Vector512.ConditionalSelect(mask, ifTrue, ifFalse);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> And(Vector512<float> a, Vector512<float> b) => a & b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Or(Vector512<float> a, Vector512<float> b) => a | b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Xor(Vector512<float> a, Vector512<float> b) => a ^ b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> ShiftLeft(Vector512<float> x, int count) => Vector512.ShiftLeft(x.AsInt32(), count).AsSingle();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> ShiftRightArithmetic(Vector512<float> x, int count) => Vector512.ShiftRightArithmetic(x.AsInt32(), count).AsSingle();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> AddInt(Vector512<float> a, Vector512<float> b) => Vector512.Add(a.AsInt32(), b.AsInt32()).AsSingle();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> SubInt(Vector512<float> a, Vector512<float> b) => Vector512.Subtract(a.AsInt32(), b.AsInt32()).AsSingle();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Exp(Vector512<float> x) => Transcendental.Exp<Vector512<float>, ExactVec512>(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> SoftmaxExp(Vector512<float> x) => Transcendental.ExpNonPositive<Vector512<float>, ExactVec512>(x);
}

internal readonly struct ExactVecScalar : IVecOps<float>
{
    public static int Lanes => 1;
    public static bool Fused => false;
    public static float Zero => 0f;

    private const uint AllOnes = 0xFFFFFFFFu;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint Bits(float x) => BitConverter.SingleToUInt32Bits(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float FromBits(uint b) => BitConverter.UInt32BitsToSingle(b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Mask(bool b) => FromBits(b ? AllOnes : 0u);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Splat(float x) => x;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Load(ref float p, nuint index) => Unsafe.Add(ref p, index);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store(ref float p, nuint index, float v) => Unsafe.Add(ref p, index) = v;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Add(float a, float b) => a + b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sub(float a, float b) => a - b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Mul(float a, float b) => a * b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Div(float a, float b) => a / b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Accumulate(float acc, float a, float b)
    {
        float product = a * b;
        return acc + product;
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float GreaterThan(float a, float b) => Mask(a > b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float LessThan(float a, float b) => Mask(a < b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float GreaterThanOrEqual(float a, float b) => Mask(a >= b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Equal(float a, float b) => Mask(a == b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Select(float mask, float ifTrue, float ifFalse) => FromBits((Bits(ifTrue) & Bits(mask)) | (Bits(ifFalse) & ~Bits(mask)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float And(float a, float b) => FromBits(Bits(a) & Bits(b));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Or(float a, float b) => FromBits(Bits(a) | Bits(b));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Xor(float a, float b) => FromBits(Bits(a) ^ Bits(b));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ShiftLeft(float x, int count) => FromBits(Bits(x) << count);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ShiftRightArithmetic(float x, int count) => FromBits((uint)((int)Bits(x) >> count));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float AddInt(float a, float b) => FromBits(Bits(a) + Bits(b));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float SubInt(float a, float b) => FromBits(Bits(a) - Bits(b));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Exp(float x) => Transcendental.Exp<float, ExactVecScalar>(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float SoftmaxExp(float x) => Transcendental.ExpNonPositive<float, ExactVecScalar>(x);
}
