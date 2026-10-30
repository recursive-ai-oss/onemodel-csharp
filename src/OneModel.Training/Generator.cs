using System;
using System.Runtime.CompilerServices;

namespace RecursiveAI.OneModel.Training;

/// <summary>
/// A seeded pseudo-random source that gives the same numbers on every platform: SplitMix64 (Steele, Lea and Flood 2014) over a
/// 64-bit counter. Used for weight initialisation and dropout masks; it never calls <see cref="System.Random"/>.
/// </summary>
public sealed class Generator
{
    private ulong _state;

    /// <summary>Creates a generator. Equal seeds give equal sequences.</summary>
    /// <param name="seed">Any 64-bit value.</param>
    public Generator(ulong seed) => _state = seed;

    /// <summary>The internal 64-bit state; <see cref="Restore"/> puts a generator back to it.</summary>
    public ulong State => _state;

    /// <summary>Sets the internal state, as read from <see cref="State"/>.</summary>
    /// <param name="state">A value previously returned by <see cref="State"/>.</param>
    public void Restore(ulong state) => _state = state;

    /// <summary>The next 64 random bits.</summary>
    public ulong NextUInt64()
    {
        _state += 0x9E3779B97F4A7C15UL;
        return Mix(_state);
    }

    /// <summary>A uniform float in [0, 1) with 24 random bits.</summary>
    public float NextSingle() => (NextUInt64() >> 40) * (1f / 16777216f);

    /// <summary>A uniform float in [<paramref name="low"/>, <paramref name="high"/>).</summary>
    /// <param name="low">Inclusive lower bound.</param>
    /// <param name="high">Exclusive upper bound, at least <paramref name="low"/>.</param>
    public float Uniform(float low, float high) => low + (high - low) * NextSingle();

    /// <summary>The SplitMix64 output function: a bijective scramble of 64 bits.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ulong Mix(ulong z)
    {
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}
