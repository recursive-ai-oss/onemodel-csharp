using System;
using System.Runtime.CompilerServices;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Packing for the blocked matrix product (Goto and van de Geijn 2008, section 4): A and B are copied into the order the microkernel
/// reads them, so it streams both from contiguous memory. Copies only, so the packing is the same for fast and exact modes and
/// cannot change any bit of the result. Ragged edges are padded with +0; the padded outputs are never stored.
/// </summary>
internal static unsafe class GemmPack
{
    /// <summary>
    /// Packs an mc x kc block of A into strips of mr rows; strip s holds, for p in 0..kc, the mr values a[s*mr + r, p] consecutively.
    /// Reads rows 0..mc-1 and columns 0..kc-1 of <paramref name="a"/> (row stride <paramref name="lda"/>); writes
    /// ceil(mc/mr)*mr*kc floats at <paramref name="dst"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void PackA(float* a, nint lda, int mc, int kc, int mr, float* dst)
    {
        for (int i0 = 0; i0 < mc; i0 += mr)
        {
            int rows = Math.Min(mr, mc - i0);
            for (int r = 0; r < rows; r++)
            {
                float* src = a + ((i0 + r) * lda);
                float* d = dst + r;
                for (int p = 0; p < kc; p++)
                {
                    d[p * mr] = src[p];
                }
            }

            for (int r = rows; r < mr; r++)
            {
                float* d = dst + r;
                for (int p = 0; p < kc; p++)
                {
                    d[p * mr] = 0f;
                }
            }

            dst += kc * mr;
        }
    }

    /// <summary>
    /// Packs strips s0..s1-1 (of nr columns each) of the kc x n block of B at <paramref name="b"/> (row stride <paramref name="ldb"/>) into
    /// <paramref name="dst"/>, which is the start of strip s0's kc*nr floats. Strip s holds, for p in 0..kc, the nr values b[p, s*nr + j]
    /// consecutively. Reads columns up to min(n, s1*nr)-1 of rows 0..kc-1; writes (s1-s0)*kc*nr floats.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void PackBStrips(float* b, nint ldb, int kc, int n, int nr, int s0, int s1, float* dst)
    {
        for (int s = s0; s < s1; s++)
        {
            int j0 = s * nr;
            int cols = Math.Min(nr, n - j0);
            for (int p = 0; p < kc; p++)
            {
                float* src = b + (p * ldb) + j0;
                int j = 0;
                for (; j < cols; j++)
                {
                    dst[j] = src[j];
                }

                for (; j < nr; j++)
                {
                    dst[j] = 0f;
                }

                dst += nr;
            }
        }
    }
}
