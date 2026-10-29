using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>Which instruction set a kernel runs on. Chosen once, from the hardware checks, when a session is created.</summary>
internal enum KernelPath
{
    /// <summary>Plain scalar code. Runs everywhere.</summary>
    Scalar = 0,

    /// <summary>Four-lane vectors through the cross-platform <c>Vector128</c> API: SSE2 on x86 without AVX2, and WebAssembly SIMD.</summary>
    Portable128 = 1,

    /// <summary>AVX2 with FMA (x64).</summary>
    Avx2 = 2,

    /// <summary>AVX-512F (x64).</summary>
    Avx512 = 3,

    /// <summary>Arm64 Advanced SIMD (NEON).</summary>
    Neon = 4,
}

/// <summary>Whether each <see cref="KernelPath"/> can run here, and which is the best one.</summary>
internal static class KernelPaths
{
    /// <summary>All paths, in order of increasing preference on a machine that has them.</summary>
    public static readonly KernelPath[] All = [KernelPath.Scalar, KernelPath.Portable128, KernelPath.Neon, KernelPath.Avx2, KernelPath.Avx512];

    /// <summary>Whether <paramref name="path"/> can run on this machine under the current runtime settings.</summary>
    public static bool IsSupported(KernelPath path) => path switch
    {
        KernelPath.Scalar => true,
        KernelPath.Portable128 => Vector128.IsHardwareAccelerated,
        KernelPath.Avx2 => Avx2.IsSupported && Fma.IsSupported,
        KernelPath.Avx512 => Avx512F.IsSupported,
        KernelPath.Neon => AdvSimd.Arm64.IsSupported,
        _ => false,
    };

    /// <summary>The widest supported path. AVX-512 is preferred over AVX2 only where it is supported.</summary>
    public static KernelPath Best()
    {
        for (int i = All.Length - 1; i >= 0; i--)
        {
            if (IsSupported(All[i]))
            {
                return All[i];
            }
        }

        return KernelPath.Scalar;
    }
}

/// <summary>Fast mode may use fused multiply-add and any blocking; exact mode is bit-identical everywhere (docs/exact-mode.md).</summary>
internal enum KernelMode
{
    /// <summary>FMA allowed. Results depend only on inputs and the kernel path, never on the thread count.</summary>
    Fast = 0,

    /// <summary>No FMA; one canonical accumulation order; identical bits on every path and thread count.</summary>
    Exact = 1,
}
