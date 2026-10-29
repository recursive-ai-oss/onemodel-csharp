using System;

namespace RecursiveAI.OneModel;

/// <summary>
/// A tiny deterministic self-check that smoke apps call to show the library loaded and ran.
/// </summary>
public static class OneModelInfo
{
    /// <summary>
    /// Computes the 64-bit FNV-1a hash of the bytes of <paramref name="data"/>. FNV-1a is specified by Fowler, Noll and Vo:
    /// start at 14695981039346656037 and, for each byte, XOR it in and multiply by 1099511628211 modulo 2^64.
    /// </summary>
    /// <param name="data">The bytes to hash; may be empty.</param>
    /// <returns>The hash; for empty input, 14695981039346656037.</returns>
    public static ulong Fnv1a(ReadOnlySpan<byte> data)
    {
        ulong h = 14695981039346656037UL;
        foreach (byte b in data)
        {
            h = (h ^ b) * 1099511628211UL;
        }

        return h;
    }

    /// <summary>
    /// Hashes a fixed byte pattern. Smoke apps call this to show that the library loaded and ran on a platform; the result is the
    /// same everywhere.
    /// </summary>
    /// <returns>The hash of the bytes 0 to 255.</returns>
    public static ulong SelfCheck()
    {
        ulong h = 14695981039346656037UL;
        for (int i = 0; i < 256; i++)
        {
            h = (h ^ (byte)i) * 1099511628211UL;
        }

        return h;
    }
}
