using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using RecursiveAI.OneModel.Kernels;

namespace RecursiveAI.OneModel.Tests.Kernels;

/// <summary>Seeded test data, naive references and the hash sweep used to compare kernel paths. Also compiled into the micro-benchmark app.</summary>
internal static class KernelTestData
{
    public const uint CanonicalNaN = 0x7FC00000u;

    /// <summary>SplitMix64 (Steele, Lea and Flood 2014): a fixed, portable generator, so the same seed gives the same data everywhere.</summary>
    public struct Rng(ulong seed)
    {
        private ulong _state = seed;

        public ulong Next()
        {
            ulong z = _state += 0x9E3779B97F4A7C15UL;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }

        public int Below(int n) => (int)(Next() % (ulong)n);

        /// <summary>Uniform in [-1, 1).</summary>
        public float Unit() => (((Next() >> 40) / (float)(1 << 24)) * 2f) - 1f;
    }

    /// <summary>Fills with uniform [-1, 1); with <paramref name="special"/>, about 1 value in 8 is NaN, infinity, zero, subnormal or huge.</summary>
    public static void Fill(ref Rng rng, Span<float> data, bool special)
    {
        for (int i = 0; i < data.Length; i++)
        {
            float v = rng.Unit();
            if (special && rng.Below(8) == 0)
            {
                v = rng.Below(12) switch
                {
                    0 => float.NaN,
                    1 => BitConverter.Int32BitsToSingle(unchecked((int)0xFFC12345)),
                    2 => float.PositiveInfinity,
                    3 => float.NegativeInfinity,
                    4 => 0f,
                    5 => -0f,
                    6 => BitConverter.Int32BitsToSingle(1 + rng.Below(1000)),
                    7 => -BitConverter.Int32BitsToSingle(1 + rng.Below(1000)),
                    8 => 3e30f,
                    9 => -3e30f,
                    10 => 1e-20f,
                    _ => BitConverter.Int32BitsToSingle(0x00800000 + rng.Below(100)),
                };
            }

            data[i] = v;
        }
    }

    /// <summary>The canonical exact order, written naively: acc starts at +0 and takes acc + a*b (two roundings) for k in increasing order; NaN becomes 0x7FC00000.</summary>
    public static void ReferenceExact(int m, int n, int k, ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float acc = 0f;
                for (int p = 0; p < k; p++)
                {
                    float prod = a[(i * lda) + p] * b[(p * ldb) + j];
                    acc += prod;
                }

                c[(i * ldc) + j] = float.IsNaN(acc) ? BitConverter.UInt32BitsToSingle(CanonicalNaN) : acc;
            }
        }
    }

    /// <summary>Double-accumulating reference and the error scale sum |a||b| per output (for tolerance checks).</summary>
    public static void ReferenceDouble(int m, int n, int k, ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, double[] c, double[] scale)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double acc = 0;
                double abs = 0;
                for (int p = 0; p < k; p++)
                {
                    double prod = (double)a[(i * lda) + p] * b[(p * ldb) + j];
                    acc += prod;
                    abs += Math.Abs(prod);
                }

                c[(i * n) + j] = acc;
                scale[(i * n) + j] = abs;
            }
        }
    }

    public static bool SameBits(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        if (x.Length != y.Length)
        {
            return false;
        }

        for (int i = 0; i < x.Length; i++)
        {
            if (BitConverter.SingleToUInt32Bits(x[i]) != BitConverter.SingleToUInt32Bits(y[i]))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>One product to compute: sizes, strides and the offsets of the three arrays inside their buffers.</summary>
    public readonly record struct GemmCase(int M, int N, int K, int Lda, int Ldb, int Ldc, int OffA, int OffB, int OffC, ulong Seed, bool Special);

    /// <summary>The fixed case list of the sweep: every size 0..maxDim in each dimension in turn, random triples, and shapes that cross the k block.</summary>
    public static List<GemmCase> SweepCases(int maxDim, int randomTriples, bool special)
    {
        var cases = new List<GemmCase>();
        var rng = new Rng(0x6F6E65);
        void Add(int m, int n, int k)
        {
            int lda = k + rng.Below(5);
            int ldb = n + rng.Below(5);
            int ldc = n + rng.Below(5);
            cases.Add(new GemmCase(m, n, k, lda, ldb, ldc, rng.Below(16), rng.Below(16), rng.Below(16), rng.Next(), special));
        }

        for (int d = 0; d <= maxDim; d++)
        {
            Add(d, 37, 53);
            Add(41, d, 29);
            Add(23, 47, d);
        }

        for (int i = 0; i < randomTriples; i++)
        {
            Add(rng.Below(maxDim + 1), rng.Below(maxDim + 1), rng.Below(maxDim + 1));
        }

        Add(200, 300, 600);
        Add(97, 131, 513);
        Add(5, 1000, 257);
        Add(1000, 7, 255);
        return cases;
    }

    /// <summary>Allocates and fills A, B and a C buffer (poisoned with a sentinel) for a case; the matrices start at the case's offsets.</summary>
    public static (float[] A, float[] B, float[] C) Materialize(GemmCase g)
    {
        var rng = new Rng(g.Seed);
        var a = new float[g.OffA + (Math.Max(g.M, 1) * g.Lda) + 1];
        var b = new float[g.OffB + (Math.Max(g.K, 1) * g.Ldb) + 1];
        var c = new float[g.OffC + (Math.Max(g.M, 1) * g.Ldc) + 1];
        Fill(ref rng, a, g.Special);
        Fill(ref rng, b, g.Special);
        c.AsSpan().Fill(BitConverter.UInt32BitsToSingle(0xDEADBEEF));
        return (a, b, c);
    }

    /// <summary>Runs a case with the kernels and returns C.</summary>
    public static float[] RunKernel(GemmCase g, KernelMode mode, KernelPath path, int threads)
    {
        (float[] a, float[] b, float[] c) = Materialize(g);
        Gemm.Run(mode, path, g.M, g.N, g.K, a.AsSpan(g.OffA), g.Lda, b.AsSpan(g.OffB), g.Ldb, c.AsSpan(g.OffC), g.Ldc, threads);
        return c;
    }

    /// <summary>SHA-256 over the C buffers of all sweep cases (so values outside the output, which must stay untouched, are covered too).</summary>
    public static string SweepHash(KernelMode mode, KernelPath path, int threads, int maxDim, int randomTriples, bool special)
    {
        using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (GemmCase g in SweepCases(maxDim, randomTriples, special))
        {
            float[] c = RunKernel(g, mode, path, threads);
            h.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(c.AsSpan()));
        }

        return Convert.ToHexString(h.GetHashAndReset()).ToLowerInvariant();
    }
}
