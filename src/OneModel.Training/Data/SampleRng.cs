using System;
using System.Numerics;

namespace RecursiveAI.OneModel.Training.Data;

/// <summary>
/// A small seeded random generator (SplitMix64) for sampling and augmentation. Its output is defined by integer arithmetic alone, so a
/// given seed gives the same stream on every platform and thread (<c>docs/data-loading.md</c> section 2).
/// </summary>
public struct SampleRng
{
    private ulong _state;

    /// <summary>Starts a stream at <paramref name="seed"/>.</summary>
    public SampleRng(ulong seed) => _state = seed;

    /// <summary>The seed of the stream for one place in a run: a pure function of the arguments.</summary>
    /// <param name="seed">The run seed.</param>
    /// <param name="epoch">Epoch number, at least 0.</param>
    /// <param name="step">Step within the epoch, at least 0.</param>
    /// <param name="slot">Position within the batch.</param>
    /// <param name="stream">Which independent stream at that place (the sampler, a transform and the collator each use their own).</param>
    public static ulong Derive(ulong seed, long epoch, long step, int slot, int stream = 0)
    {
        ulong h = Mix(seed ^ 0xD1B54A32D192ED03UL);
        h = Mix(h + (ulong)epoch);
        h = Mix(h + (ulong)step);
        h = Mix(h + (ulong)(uint)slot);
        return Mix(h + (ulong)(uint)stream);
    }

    internal static ulong Mix(ulong z)
    {
        z += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    /// <summary>The next 64 random bits.</summary>
    public ulong NextUInt64()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    /// <summary>A float in [0, 1) with 24 random bits.</summary>
    public float NextSingle() => (NextUInt64() >> 40) * (1f / (1 << 24));

    /// <summary>A double in [0, 1) with 53 random bits.</summary>
    public double NextDouble() => (NextUInt64() >> 11) * (1.0 / (1L << 53));

    /// <summary>A uniform integer in [0, <paramref name="bound"/>), without modulo bias (multiply-shift with rejection).</summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="bound"/> is not positive.</exception>
    public int NextInt(int bound)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(bound);
        ulong b = (uint)bound;
        UInt128 m = (UInt128)NextUInt64() * b;
        ulong low = (ulong)m;
        if (low < b)
        {
            ulong threshold = (0UL - b) % b;
            while (low < threshold)
            {
                m = (UInt128)NextUInt64() * b;
                low = (ulong)m;
            }
        }
        return (int)(ulong)(m >> 64);
    }

    /// <summary>A float uniform in [<paramref name="low"/>, <paramref name="high"/>).</summary>
    public float NextSingle(float low, float high) => low + (high - low) * NextSingle();

    /// <summary>
    /// An approximately standard-normal float: the sum of twelve uniforms minus six (Irwin–Hall; mean 0, variance 1, bounded by ±6).
    /// Chosen over Box–Muller because it needs only <c>+</c> and <c>−</c>, so it is bit-identical everywhere.
    /// </summary>
    public float NextNormal()
    {
        float s = 0;
        for (int i = 0; i < 12; i++) s += NextSingle();
        return s - 6f;
    }
}

/// <summary>A stateless pseudo-random permutation of [0, n): a four-round Feistel network with cycle walking, keyed by a seed.</summary>
internal static class KeyedPermutation
{
    /// <summary>Maps <paramref name="position"/> in [0, n) to its place in the permutation. Bijective for fixed <paramref name="key"/> and <paramref name="n"/>.</summary>
    public static int Map(ulong key, int n, int position)
    {
        if (n <= 1) return 0;
        int bits = 32 - BitOperations.LeadingZeroCount((uint)(n - 1));
        int half = (bits + 1) / 2;
        ulong mask = (1UL << half) - 1;
        ulong x = (uint)position;
        do
        {
            ulong l = x >> half, r = x & mask;
            for (int round = 0; round < 4; round++)
            {
                ulong f = SampleRng.Mix(r ^ key ^ ((ulong)round * 0x632BE59BD9B4E019UL)) & mask;
                (l, r) = (r, l ^ f);
            }
            x = (l << half) | r;
        } while (x >= (uint)n);
        return (int)x;
    }
}
