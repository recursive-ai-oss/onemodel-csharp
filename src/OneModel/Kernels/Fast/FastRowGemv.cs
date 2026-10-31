using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace RecursiveAI.OneModel.Kernels.Fast;

/// <summary>
/// Matrix-vector microkernels over weights packed in strips of <c>mr</c> rows (element (p, r) of a strip at <c>p*mr + r</c>), for a
/// product with a single output column (a convolution whose output is 1x1, such as the squeeze-and-excitation layers of MobileNetV3).
/// The tile microkernels would waste all but one column of their tile. Here the rows are cut into slivers of four rows; a sliver's
/// four outputs are one vector, and each step of k broadcasts one input value against that vector. Several slivers run at once to
/// give independent chains for the fused multiply-add to overlap. Each output accumulates its products in increasing k with one
/// fused multiply-add per step, from +0 or from the stored partial sum of the previous k block, exactly the order of the tile
/// microkernels, so the results equal the matrix product's bit for bit (and the rounding rule is the one the portable tile uses
/// when no fused multiply-add exists).
/// Bounds the caller guarantees: each <c>a</c> pointer addresses 4 rows of a strip with <c>kc</c> steps of <c>mr</c> floats
/// (the last step ends 4 floats after its start); <c>x</c> points at kc floats; <c>c</c> at 4 floats per sliver, 8 or 4 or 1
/// slivers consecutive in memory as the kernel's name says.
/// </summary>
internal static unsafe class FastRowGemv
{
    /// <summary>Eight consecutive slivers (32 rows) whose first rows are at <c>a0..a7</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Slivers8(
        int kc, int mr, float* a0, float* a1, float* a2, float* a3, float* a4, float* a5, float* a6, float* a7, float* x, float* c, int accumulate)
    {
        Vector128<float> c0, c1, c2, c3, c4, c5, c6, c7;
        if (accumulate != 0)
        {
            c0 = Vector128.Load(c);
            c1 = Vector128.Load(c + 4);
            c2 = Vector128.Load(c + 8);
            c3 = Vector128.Load(c + 12);
            c4 = Vector128.Load(c + 16);
            c5 = Vector128.Load(c + 20);
            c6 = Vector128.Load(c + 24);
            c7 = Vector128.Load(c + 28);
        }
        else
        {
            c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = Vector128<float>.Zero;
        }

        nint step = mr;
        for (int p = 0; p < kc; p++)
        {
            Vector128<float> xv = Vector128.Create(x[p]);
            nint o = p * step;
            c0 = MulAdd(Vector128.Load(a0 + o), xv, c0);
            c1 = MulAdd(Vector128.Load(a1 + o), xv, c1);
            c2 = MulAdd(Vector128.Load(a2 + o), xv, c2);
            c3 = MulAdd(Vector128.Load(a3 + o), xv, c3);
            c4 = MulAdd(Vector128.Load(a4 + o), xv, c4);
            c5 = MulAdd(Vector128.Load(a5 + o), xv, c5);
            c6 = MulAdd(Vector128.Load(a6 + o), xv, c6);
            c7 = MulAdd(Vector128.Load(a7 + o), xv, c7);
        }

        Vector128.Store(c0, c);
        Vector128.Store(c1, c + 4);
        Vector128.Store(c2, c + 8);
        Vector128.Store(c3, c + 12);
        Vector128.Store(c4, c + 16);
        Vector128.Store(c5, c + 20);
        Vector128.Store(c6, c + 24);
        Vector128.Store(c7, c + 28);
    }

    /// <summary>Four consecutive slivers.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Slivers4(int kc, int mr, float* a0, float* a1, float* a2, float* a3, float* x, float* c, int accumulate)
    {
        Vector128<float> c0, c1, c2, c3;
        if (accumulate != 0)
        {
            c0 = Vector128.Load(c);
            c1 = Vector128.Load(c + 4);
            c2 = Vector128.Load(c + 8);
            c3 = Vector128.Load(c + 12);
        }
        else
        {
            c0 = c1 = c2 = c3 = Vector128<float>.Zero;
        }

        nint step = mr;
        for (int p = 0; p < kc; p++)
        {
            Vector128<float> xv = Vector128.Create(x[p]);
            nint o = p * step;
            c0 = MulAdd(Vector128.Load(a0 + o), xv, c0);
            c1 = MulAdd(Vector128.Load(a1 + o), xv, c1);
            c2 = MulAdd(Vector128.Load(a2 + o), xv, c2);
            c3 = MulAdd(Vector128.Load(a3 + o), xv, c3);
        }

        Vector128.Store(c0, c);
        Vector128.Store(c1, c + 4);
        Vector128.Store(c2, c + 8);
        Vector128.Store(c3, c + 12);
    }

    /// <summary>One sliver.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Sliver1(int kc, int mr, float* a0, float* x, float* c, int accumulate)
    {
        Vector128<float> c0 = accumulate != 0 ? Vector128.Load(c) : Vector128<float>.Zero;
        nint step = mr;
        for (int p = 0; p < kc; p++)
        {
            c0 = MulAdd(Vector128.Load(a0 + (p * step)), Vector128.Create(x[p]), c0);
        }

        Vector128.Store(c0, c);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> MulAdd(Vector128<float> a, Vector128<float> b, Vector128<float> c) =>
        Fma.IsSupported || AdvSimd.IsSupported ? Vector128.FusedMultiplyAdd(a, b, c) : Vector128.Add(c, Vector128.Multiply(a, b));
}
