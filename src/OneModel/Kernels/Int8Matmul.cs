using System;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// A matrix packed ahead of time for <see cref="Int8Gemm"/> (a constant weight): the packed bytes in the layout of one
/// <see cref="Int8Format"/>, and the sums the zero-point correction needs (per row for an A operand, per column for a B operand).
/// Immutable after creation, so concurrent runs share it.
/// </summary>
internal sealed class Int8Packed
{
    private Int8Packed(Int8Format format, int outer, int k, byte[] data, int[] sums)
    {
        Format = format;
        Outer = outer;
        K = k;
        Data = data;
        Sums = sums;
    }

    public Int8Format Format { get; }

    /// <summary>Rows (A operand) or columns (B operand).</summary>
    public int Outer { get; }

    public int K { get; }

    public byte[] Data { get; }

    public int[] Sums { get; }

    /// <summary>Packs an A operand: <paramref name="rows"/> x <paramref name="k"/> row-major 8-bit values with row stride <paramref name="lda"/>.</summary>
    public static unsafe Int8Packed ForA(in Int8Format f, ReadOnlySpan<byte> src, long lda, bool u8, int rows, int k)
    {
        CheckSource(src.Length, lda, rows, k, "A");
        byte[] data = new byte[f.PackedABytes(rows, k)];
        int[] sums = new int[rows];
        fixed (byte* ps = src, pd = data)
        fixed (int* pr = sums)
        {
            Int8Gemm.PackA(in f, ps, lda, u8, rows, k, pd, pr);
        }

        return new Int8Packed(f, rows, k, data, sums);
    }

    /// <summary>Packs a B operand: <paramref name="k"/> x <paramref name="n"/> row-major 8-bit values with row stride <paramref name="ldb"/>.</summary>
    public static unsafe Int8Packed ForB(in Int8Format f, ReadOnlySpan<byte> src, long ldb, bool u8, int k, int n)
    {
        CheckSource(src.Length, ldb, k, n, "B");
        byte[] data = new byte[f.PackedBBytes(k, n)];
        int[] sums = new int[n];
        fixed (byte* ps = src, pd = data)
        fixed (int* pc = sums)
        {
            Int8Gemm.PackB(in f, ps, ldb, u8, k, n, pd, pc, 1);
        }

        return new Int8Packed(f, n, k, data, sums);
    }

    internal static void CheckSource(long length, long ld, int rows, int cols, string name)
    {
        if (rows < 0 || cols < 0 || ld < cols || (rows > 0 && cols > 0 && ((rows - 1) * ld) + cols > length))
        {
            throw new ArgumentException($"{name} is smaller than {rows} x {cols} with row stride {ld}.");
        }
    }
}

/// <summary>Everything one integer matrix product needs; spans are validated against the shapes before any pointer is formed.</summary>
internal ref struct Int8Call
{
    public Int8Format Format;
    public int M;
    public int N;
    public int K;

    /// <summary>A as raw 8-bit rows (row stride <see cref="Lda"/>), used when <see cref="APacked"/> is null.</summary>
    public ReadOnlySpan<byte> ARaw;
    public long Lda;
    public bool AU8;

    /// <summary>A packed ahead of time, or null.</summary>
    public Int8Packed? APacked;

    /// <summary>B packed (by <see cref="Int8Gemm.PackB"/> or ahead of time) and its column sums.</summary>
    public ReadOnlySpan<byte> BPacked;
    public ReadOnlySpan<int> BSums;

    /// <summary>A's zero point per row (<see cref="ZaVec"/> when not empty) or one value, in the shifted domain.</summary>
    public int ZaScalar;
    public ReadOnlySpan<int> ZaVec;

    /// <summary>B's zero point per column or one value, in the shifted domain.</summary>
    public int ZbScalar;
    public ReadOnlySpan<int> ZbVec;

    /// <summary>The output: int32 sums, or 8-bit quantized values when <see cref="OutInt32"/> is empty.</summary>
    public Span<int> OutInt32;
    public Span<float> OutFloat;
    public ReadOnlySpan<float> BiasFloat;
    public Span<byte> OutQuant;
    public bool QuantUnsigned;
    public int QuantZero;
    public long Ldc;

    /// <summary>int32 added before scaling (may be empty): one per row, or one per column when <see cref="BiasPerCol"/>.</summary>
    public ReadOnlySpan<int> Bias;
    public bool BiasPerCol;

    /// <summary>Scale applied to the sums when <see cref="MultVec"/> is empty; else one scale per row (<see cref="MultPerRow"/>) or column.</summary>
    public float Mult;
    public ReadOnlySpan<float> MultVec;
    public bool MultPerRow;

    public int Threads;
}

/// <summary>The validated entry to the integer matrix product.</summary>
internal static unsafe class Int8Matmul
{
    /// <summary>
    /// Runs the product described by <paramref name="c"/>.
    /// </summary>
    /// <exception cref="ArgumentException">A span is shorter than its shape needs, or A is neither raw nor packed.</exception>
    public static void Run(ref Int8Call c)
    {
        Int8Format f = c.Format;
        int m = c.M, n = c.N, k = c.K;
        if (m < 0 || n < 0 || k < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(c), "Matrix dimensions must not be negative.");
        }

        if (m == 0 || n == 0)
        {
            return;
        }

        bool packedA = c.APacked != null;
        if (packedA)
        {
            Int8Packed pa = c.APacked!;
            if (pa.Outer < m || pa.K != k || pa.Format.Isa != f.Isa)
            {
                throw new ArgumentException("The packed A does not match the product's shape or kernel flavour.");
            }
        }
        else
        {
            Int8Packed.CheckSource(c.ARaw.Length, c.Lda, m, k, "A");
        }

        if (c.BPacked.Length < f.PackedBBytes(k, n) || c.BSums.Length < n)
        {
            throw new ArgumentException("The packed B or its column sums are too small.");
        }

        if ((!c.ZaVec.IsEmpty && c.ZaVec.Length < m) || (!c.ZbVec.IsEmpty && c.ZbVec.Length < n))
        {
            throw new ArgumentException("A zero-point vector is shorter than the matrix dimension it covers.");
        }

        bool toInt = !c.OutInt32.IsEmpty;
        bool toFloat = !toInt && !c.OutFloat.IsEmpty;
        long outLen = toInt ? c.OutInt32.Length : toFloat ? c.OutFloat.Length : c.OutQuant.Length;
        if (toFloat && !c.BiasFloat.IsEmpty && c.BiasFloat.Length < n)
        {
            throw new ArgumentException("The float bias is shorter than the number of columns.");
        }

        if (c.Ldc < n || ((m - 1) * c.Ldc) + n > outLen)
        {
            throw new ArgumentException("The output is smaller than m x n with its row stride.");
        }

        if (!c.Bias.IsEmpty && c.Bias.Length < (c.BiasPerCol ? n : m))
        {
            throw new ArgumentException("The bias is shorter than the dimension it covers.");
        }

        if (!c.MultVec.IsEmpty && c.MultVec.Length < (c.MultPerRow ? m : n))
        {
            throw new ArgumentException("The scale vector is shorter than the dimension it covers.");
        }

        fixed (byte* pRaw = c.ARaw, pBp = c.BPacked, pq = c.OutQuant)
        fixed (int* pBs = c.BSums, pza = c.ZaVec, pzb = c.ZbVec, pi = c.OutInt32, pbias = c.Bias)
        fixed (float* pmv = c.MultVec, pof = c.OutFloat, pbf = c.BiasFloat)
        fixed (byte* pAp = packedA ? c.APacked!.Data : default)
        fixed (int* pAs = packedA ? c.APacked!.Sums : default)
        {
            Int8A a = default;
            if (packedA)
            {
                a.Packed = pAp;
                a.RowSums = pAs;
            }
            else
            {
                a.Raw = pRaw;
                a.Ld = c.Lda;
                a.U8 = c.AU8;
            }

            Int8Epilogue ep = default;
            ep.OutInt32 = toInt ? pi : null;
            ep.OutQuant = toInt || toFloat ? null : pq;
            ep.OutFloat = toFloat ? pof : null;
            ep.BiasFloat = c.BiasFloat.IsEmpty ? null : pbf;
            ep.QuantUnsigned = c.QuantUnsigned;
            ep.QuantZero = c.QuantZero;
            ep.Ldc = c.Ldc;
            ep.Bias = c.Bias.IsEmpty ? null : pbias;
            ep.BiasPerCol = c.BiasPerCol && !c.Bias.IsEmpty;
            ep.Mult = c.Mult;
            ep.MultVec = c.MultVec.IsEmpty ? null : pmv;
            ep.MultPerRow = c.MultPerRow;
            Int8Zero za = c.ZaVec.IsEmpty ? new Int8Zero(c.ZaScalar) : new Int8Zero(pza);
            Int8Zero zb = c.ZbVec.IsEmpty ? new Int8Zero(c.ZbScalar) : new Int8Zero(pzb);
            Int8Gemm.Run(in f, m, n, k, in a, pBp, pBs, in za, in zb, in ep, c.Threads);
        }
    }

    /// <summary>
    /// Packs B (k x n, row stride <paramref name="ldb"/>) into <paramref name="packed"/> and its column sums into <paramref name="sums"/>.
    /// </summary>
    public static void PackB(in Int8Format f, ReadOnlySpan<byte> src, long ldb, bool u8, int k, int n, Span<byte> packed, Span<int> sums, int threads)
    {
        Int8Packed.CheckSource(src.Length, ldb, k, n, "B");
        if (packed.Length < f.PackedBBytes(k, n) || sums.Length < n)
        {
            throw new ArgumentException("The packed B or its column sums are too small.");
        }

        fixed (byte* ps = src, pd = packed)
        fixed (int* pc = sums)
        {
            Int8Gemm.PackB(in f, ps, ldb, u8, k, n, pd, pc, threads);
        }
    }
}
