using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Kernels.Exact;

namespace RecursiveAI.OneModel.Tests.Kernels;

public class TranscendentalTests
{
    private static readonly VecIsa[] AllIsas = [VecIsa.Scalar, VecIsa.V128, VecIsa.V256, VecIsa.V512];

    private sealed record Fn(ElementwiseFn Id, Func<double, double> Reference, float HotLo, float HotHi, float FullLo, float FullHi);

    private static readonly Fn[] Fns =
    [
        new(ElementwiseFn.Exp, TranscendentalReference.Exp, -16f, 16f, -110f, 90f),
        new(ElementwiseFn.Log, TranscendentalReference.Log, 0.0625f, 16f, 0f, float.MaxValue),
        new(ElementwiseFn.Tanh, TranscendentalReference.Tanh, -10f, 10f, -50f, 50f),
        new(ElementwiseFn.Erf, TranscendentalReference.Erf, -6f, 6f, -10f, 10f),
        new(ElementwiseFn.Sigmoid, TranscendentalReference.Sigmoid, -16f, 16f, -110f, 90f),
        new(ElementwiseFn.GeluErf, TranscendentalReference.GeluErf, -3f, 10f, -3f, 40f),
        new(ElementwiseFn.GeluTanh, TranscendentalReference.GeluTanh, -3f, 10f, -3f, 40f),
    ];

    /// <summary>Monotonic integer key of a float (negative floats get negative keys, both zeros map to 0).</summary>
    private static long Key(float f)
    {
        int bits = BitConverter.SingleToInt32Bits(f);
        return bits < 0 ? -(long)(bits & 0x7FFFFFFF) : bits;
    }

    private static float FromKey(long key) =>
        key < 0 ? BitConverter.Int32BitsToSingle(unchecked((int)0x80000000u) | (int)(-key)) : BitConverter.Int32BitsToSingle((int)key);

    /// <summary>All floats in [lo, hi] at the given stride of the ordered bit patterns, in chunks.</summary>
    private static IEnumerable<float[]> Chunks(float lo, float hi, long stride)
    {
        long a = Key(lo);
        long b = Key(hi);
        var buf = new float[1 << 16];
        int n = 0;
        for (long k = a; k <= b; k += stride)
        {
            buf[n++] = FromKey(k);
            if (n == buf.Length)
            {
                yield return buf;
                n = 0;
            }
        }

        if (n > 0)
        {
            yield return buf.AsSpan(0, n).ToArray();
        }
    }

    private static float[] Specials()
    {
        var l = new List<float>
        {
            0f, -0f, 1f, -1f, 0.5f, 2f, float.PositiveInfinity, float.NegativeInfinity, float.MaxValue, -float.MaxValue,
            float.Epsilon, -float.Epsilon, BitConverter.Int32BitsToSingle(0x007FFFFF), BitConverter.Int32BitsToSingle(0x00800000),
            BitConverter.Int32BitsToSingle(unchecked((int)0x80800000)), BitConverter.Int32BitsToSingle(0x7FC00000),
            BitConverter.Int32BitsToSingle(unchecked((int)0xFFC00000)), BitConverter.Int32BitsToSingle(0x7FA00001),
            BitConverter.Int32BitsToSingle(unchecked((int)0xFF800001)), 88.72283172607422f, 88.72284f, -103.97207641601562f,
            -103.9721f, -87.33654f, -87.4f, 0.6f, -0.6f, 0.59999996f, 0.99999994f, 3.9192f, 9.01f, 44.4f, 100f, -100f,
        };
        for (int i = -150; i <= 130; i++)
        {
            l.Add(MathF.ScaleB(1f, i));
            l.Add(-MathF.ScaleB(1f, i));
        }

        return [.. l];
    }

    private static float[] Run(ElementwiseFn fn, bool exact, VecIsa isa, float[] x)
    {
        var y = new float[x.Length];
        ElementwiseMath.Apply(fn, exact, isa, x, y);
        return y;
    }

    [Fact]
    public void ExactBitsAreEqualAtEveryLaneWidthAndLength()
    {
        foreach (Fn f in Fns)
        {
            foreach (float[] chunk in Chunks(float.NegativeInfinity, float.PositiveInfinity, 100_003).Take(40))
            {
                float[] want = Run(f.Id, true, VecIsa.Scalar, chunk);
                foreach (VecIsa isa in AllIsas.Skip(1))
                {
                    float[] got = Run(f.Id, true, isa, chunk);
                    AssertSameBits(want, got, chunk, f.Id, isa);
                }
            }

            // Lengths 0..70 at every alignment of the tail, on specials.
            float[] sp = Specials();
            for (int len = 0; len <= 70; len++)
            {
                float[] part = sp.AsSpan(len, Math.Min(len + 37, sp.Length - len)).ToArray();
                float[] want = Run(f.Id, true, VecIsa.Scalar, part);
                foreach (VecIsa isa in AllIsas.Skip(1))
                {
                    AssertSameBits(want, Run(f.Id, true, isa, part), part, f.Id, isa);
                }
            }
        }
    }

    private static void AssertSameBits(float[] want, float[] got, float[] x, ElementwiseFn fn, VecIsa isa)
    {
        for (int i = 0; i < want.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(want[i]) != BitConverter.SingleToInt32Bits(got[i]))
            {
                Assert.Fail($"{fn} at {isa}: x={x[i]:R} ({BitConverter.SingleToInt32Bits(x[i]):X8}) scalar={want[i]:R} got={got[i]:R}");
            }
        }
    }

    /// <summary>
    /// SHA-256 of the exact outputs over a fixed input sweep. These are the golden values every platform (x64 with and without
    /// AVX-512, arm64, WASM, DOTNET_EnableHWIntrinsic=0) must reproduce.
    /// </summary>
    private static string GoldenHash(ElementwiseFn fn)
    {
        using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var bytes = new byte[4 << 16];
        foreach (float[] chunk in Chunks(float.NegativeInfinity, float.PositiveInfinity, 4099).Concat([Specials()]))
        {
            float[] y = Run(fn, true, VecIsa.Auto, chunk);
            Buffer.BlockCopy(y, 0, bytes, 0, y.Length * 4);
            sha.AppendData(bytes, 0, y.Length * 4);
        }

        return Convert.ToHexString(sha.GetHashAndReset()).ToLowerInvariant();
    }

    private static readonly Dictionary<ElementwiseFn, string> Golden = new()
    {
        [ElementwiseFn.Exp] = "b6f562effed1022ea7e6c2f81f16a5072aacc34180ff4311953aa038cbc374d2",
        [ElementwiseFn.Log] = "d3e22d27ace0aa427c955481911a385b00a9fa922f436f1516facce4b68708da",
        [ElementwiseFn.Tanh] = "40b0109080fd1ed7474c8a84178d0c775687b729467ff565bd2a406660677ed5",
        [ElementwiseFn.Erf] = "3021049724b365f5503c655001070c64cca7d574c0fad1a51d7e59c44c3b5ee5",
        [ElementwiseFn.Sigmoid] = "9bd37e9ebc896ecec4ae7c3e87a47e106e9239f57096f23476712f3b7372d903",
        [ElementwiseFn.GeluErf] = "e6684097913fc5fe7121c93ab01b0ff852cf9e5b57635f560e8d9f88671cb420",
        [ElementwiseFn.GeluTanh] = "c66acfdced9c1156455b5eab101b50eaa1cccd2211d8eb91f39dd9fe30b0f92b",
    };

    [Fact]
    public void ExactBitsMatchTheGoldenHashes()
    {
        var sb = new StringBuilder();
        foreach (KeyValuePair<ElementwiseFn, string> kv in Golden)
        {
            string h = GoldenHash(kv.Key);
            if (h != kv.Value)
            {
                sb.Append(CultureInfo.InvariantCulture, $"[ElementwiseFn.{kv.Key}] = \"{h}\",\n");
            }
        }

        Assert.True(sb.Length == 0, "Golden hashes differ; current values:\n" + sb);
    }

    [Fact]
    public void SoftmaxExponentialEqualsExpOnItsDomain()
    {
        foreach (float[] chunk in Chunks(-87.33654f, 0f, 997))
        {
            foreach (float x in chunk)
            {
                float a = Transcendental.Exp<float, ExactVecScalar>(x);
                float b = Transcendental.ExpNonPositive<float, ExactVecScalar>(x);
                if (BitConverter.SingleToInt32Bits(a) != BitConverter.SingleToInt32Bits(b))
                {
                    Assert.Fail($"x={x:R}: Exp {a:R}, ExpNonPositive {b:R}");
                }
            }
        }

        Assert.Equal(0f, Transcendental.ExpNonPositive<float, ExactVecScalar>(-87.4f));
        Assert.Equal(0f, Transcendental.ExpNonPositive<float, ExactVecScalar>(float.NegativeInfinity));
        Assert.Equal(0x7FC00000u, BitConverter.SingleToUInt32Bits(Transcendental.ExpNonPositive<float, ExactVecScalar>(float.NaN)));
    }

    [Fact]
    public void ExactSpecialValues()
    {
        const uint qnan = 0x7FC00000;
        void Check(ElementwiseFn fn, float x, float want)
        {
            foreach (VecIsa isa in AllIsas)
            {
                float got = Run(fn, true, isa, [x])[0];
                if (float.IsNaN(want))
                {
                    Assert.Equal(qnan, BitConverter.SingleToUInt32Bits(got));
                }
                else
                {
                    Assert.True(BitConverter.SingleToInt32Bits(want) == BitConverter.SingleToInt32Bits(got), $"{fn}({x:R}) at {isa}: want {want:R}, got {got:R}");
                }
            }
        }

        float inf = float.PositiveInfinity;
        float nan = float.NaN;
        Check(ElementwiseFn.Exp, 0f, 1f);
        Check(ElementwiseFn.Exp, -0f, 1f);
        Check(ElementwiseFn.Exp, inf, inf);
        Check(ElementwiseFn.Exp, -inf, 0f);
        Check(ElementwiseFn.Exp, nan, nan);
        Check(ElementwiseFn.Exp, 89f, inf);
        Check(ElementwiseFn.Exp, -104f, 0f);
        Check(ElementwiseFn.Log, 1f, 0f);
        Check(ElementwiseFn.Log, 0f, -inf);
        Check(ElementwiseFn.Log, -0f, -inf);
        Check(ElementwiseFn.Log, -1f, nan);
        Check(ElementwiseFn.Log, -inf, nan);
        Check(ElementwiseFn.Log, inf, inf);
        Check(ElementwiseFn.Log, nan, nan);
        Check(ElementwiseFn.Tanh, inf, 1f);
        Check(ElementwiseFn.Tanh, -inf, -1f);
        Check(ElementwiseFn.Tanh, -0f, -0f);
        Check(ElementwiseFn.Tanh, 0f, 0f);
        Check(ElementwiseFn.Tanh, nan, nan);
        Check(ElementwiseFn.Erf, inf, 1f);
        Check(ElementwiseFn.Erf, -inf, -1f);
        Check(ElementwiseFn.Erf, -0f, -0f);
        Check(ElementwiseFn.Erf, nan, nan);
        Check(ElementwiseFn.Sigmoid, inf, 1f);
        Check(ElementwiseFn.Sigmoid, -inf, 0f);
        Check(ElementwiseFn.Sigmoid, 0f, 0.5f);
        Check(ElementwiseFn.Sigmoid, nan, nan);
        Check(ElementwiseFn.GeluErf, inf, inf);
        Check(ElementwiseFn.GeluErf, -inf, -0f);
        Check(ElementwiseFn.GeluErf, 0f, 0f);
        Check(ElementwiseFn.GeluTanh, inf, inf);
        Check(ElementwiseFn.GeluTanh, -inf, -0f);
        Check(ElementwiseFn.GeluTanh, nan, nan);
    }

    private static (double Max, float At) MaxUlp(Fn f, bool exact, VecIsa isa, float lo, float hi, long stride)
    {
        double max = 0;
        float at = 0;
        var gate = new object();
        int threads = int.Parse(Environment.GetEnvironmentVariable("ONEMODEL_ULP_THREADS") ?? "1", CultureInfo.InvariantCulture);
        Parallel.ForEach(Chunks(lo, hi, stride).Select(c => (float[])c.Clone()), new ParallelOptions { MaxDegreeOfParallelism = threads }, chunk =>
        {
            float[] y = Run(f.Id, exact, isa, chunk);
            double m = 0;
            float where = 0;
            for (int i = 0; i < chunk.Length; i++)
            {
                double e = TranscendentalReference.UlpError(y[i], f.Reference(chunk[i]));
                if (e > m)
                {
                    m = e;
                    where = chunk[i];
                }
            }

            lock (gate)
            {
                if (m > max || (m == max && BitConverter.SingleToInt32Bits(where) < BitConverter.SingleToInt32Bits(at)))
                {
                    max = m;
                    at = where;
                }
            }
        });
        return (max, at);
    }

    /// <summary>Error of the platform library function over the same sweep, for comparison with the table.</summary>
    private static double BclMaxUlp(Func<float, float> fn, Func<double, double> reference, float lo, float hi, long stride)
    {
        double max = 0;
        foreach (float[] chunk in Chunks(lo, hi, stride))
        {
            foreach (float x in chunk)
            {
                max = Math.Max(max, TranscendentalReference.UlpError(fn(x), reference(x)));
            }
        }

        return max;
    }

    [Fact]
    public void ExactAccuracyIsWithinTheDocumentedBounds()
    {
        // Documented maximum errors in ulp over the sweeps below (docs/exact-transcendentals.md).
        var bound = new Dictionary<ElementwiseFn, double>
        {
            [ElementwiseFn.Exp] = 1.2,
            [ElementwiseFn.Log] = 2.1,
            [ElementwiseFn.Tanh] = 1.5,
            [ElementwiseFn.Erf] = 3.0,
            [ElementwiseFn.Sigmoid] = 2.5,
            [ElementwiseFn.GeluErf] = 13.0,
            [ElementwiseFn.GeluTanh] = 18.0,
        };
        foreach (Fn f in Fns)
        {
            (double hot, float hotAt) = MaxUlp(f, true, VecIsa.Auto, f.HotLo, f.HotHi, 61);
            (double full, float fullAt) = MaxUlp(f, true, VecIsa.Auto, f.FullLo, f.FullHi, 4099);
            Assert.True(hot <= bound[f.Id], $"{f.Id} hot range: {hot:F3} ulp at {hotAt:R}");
            Assert.True(full <= bound[f.Id], $"{f.Id} full range: {full:F3} ulp at {fullAt:R}");
        }
    }

    [Fact]
    public void FastAccuracyIsWithinTheDocumentedBounds()
    {
        var bound = new Dictionary<ElementwiseFn, double>
        {
            [ElementwiseFn.Exp] = 2.5,
            [ElementwiseFn.Log] = 2.1,
            [ElementwiseFn.Tanh] = 1.5,
            [ElementwiseFn.Erf] = 3.0,
            [ElementwiseFn.Sigmoid] = 2.5,
            [ElementwiseFn.GeluErf] = 13.0,
            [ElementwiseFn.GeluTanh] = 18.0,
        };
        foreach (Fn f in Fns)
        {
            float lo = f.Id == ElementwiseFn.Exp ? -87f : f.HotLo;
            float hi = f.Id == ElementwiseFn.Exp ? 88f : f.HotHi;
            (double hot, float at) = MaxUlp(f, false, VecIsa.Auto, lo, hi, 61);
            Assert.True(hot <= bound[f.Id], $"{f.Id} fast: {hot:F3} ulp at {at:R}");
        }
    }

    /// <summary>Writes the ulp table for the docs. Run with ONEMODEL_ULP_TABLE set to an output path; does nothing otherwise.</summary>
    [Fact]
    public void WriteUlpTable()
    {
        string? path = Environment.GetEnvironmentVariable("ONEMODEL_ULP_TABLE");
        if (string.IsNullOrEmpty(path))
        {
            return;
        }

        long stride = long.Parse(Environment.GetEnvironmentVariable("ONEMODEL_ULP_STRIDE") ?? "1", CultureInfo.InvariantCulture);
        var sb = new StringBuilder();
        sb.AppendLine("| function | variant | sweep | values | max ulp | at x |");
        sb.AppendLine("|---|---|---|---|---|---|");
        foreach (Fn f in Fns)
        {
            foreach (bool exact in new[] { true, false })
            {
                (string, float, float, long)[] sweeps =
                [
                    ("hot", f.HotLo, f.HotHi, stride),
                    ("full", f.FullLo, f.FullHi, stride * 67),
                ];
                foreach ((string name, float lo0, float hi0, long st) in sweeps)
                {
                    float lo = lo0;
                    float hi = hi0;
                    if (!exact && f.Id == ElementwiseFn.Exp)
                    {
                        lo = Math.Max(lo, -87f);
                        hi = Math.Min(hi, 88f);
                    }

                    (double max, float at) = MaxUlp(f, exact, VecIsa.Auto, lo, hi, st);
                    long count = ((Key(hi) - Key(lo)) / st) + 1;
                    sb.AppendLine(CultureInfo.InvariantCulture, $"| {f.Id} | {(exact ? "exact" : "fast")} | {name} [{lo:G6}, {hi:G6}] stride {st} | {count} | {max:F3} | {at:R} |");
                }
            }
        }

        foreach (Fn f in Fns.Where(f => f.Id is ElementwiseFn.GeluErf or ElementwiseFn.GeluTanh))
        {
            foreach ((float lo, float hi) in new[] { (-6f, -3f), (-12f, -6f) })
            {
                (double max, float at) = MaxUlp(f, true, VecIsa.Auto, lo, hi, stride);
                sb.AppendLine(CultureInfo.InvariantCulture, $"| {f.Id} | exact | negative tail [{lo:G6}, {hi:G6}] stride {stride} | {((Key(hi) - Key(lo)) / stride) + 1} | {max:F3} | {at:R} |");
            }
        }

        foreach ((string name, Func<float, float> fn, Fn f) in new (string, Func<float, float>, Fn)[]
        {
            ("MathF.Exp", MathF.Exp, Fns[0]),
            ("MathF.Log", MathF.Log, Fns[1]),
            ("MathF.Tanh", MathF.Tanh, Fns[2]),
        })
        {
            double max = BclMaxUlp(fn, f.Reference, f.HotLo, f.HotHi, stride * 8);
            sb.AppendLine(CultureInfo.InvariantCulture, $"| {name} (platform library) | - | hot [{f.HotLo:G6}, {f.HotHi:G6}] stride {stride * 8} | - | {max:F3} | - |");
        }

        File.WriteAllText(path, sb.ToString());
    }
}
