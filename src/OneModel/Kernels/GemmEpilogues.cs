using System;
using System.Runtime.Intrinsics;
using RecursiveAI.OneModel.Kernels.Exact;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// An element-wise activation applied in place to a row of finished results. Each subclass calls the very kernel the stand-alone
/// operator uses, so in exact mode (where every kernel is independent of lane width and of how a row is cut) the fused result is
/// bit-identical to running the operator after the product; in fast mode the same function runs, and only the split between vector
/// body and scalar tail can differ in the last bits.
/// </summary>
internal abstract class FusedActivation
{
    /// <summary>Applies the function to every element of <paramref name="row"/>, in place.</summary>
    public abstract void Apply(KernelPath path, bool exact, Span<float> row);
}

/// <summary>Relu.</summary>
internal sealed class ReluActivation : FusedActivation
{
    public override void Apply(KernelPath path, bool exact, Span<float> row) => ExactElementwise.Relu(path, exact, row, row);
}

/// <summary>Clip with both bounds present (Relu6 is Clip(0, 6)).</summary>
internal sealed class ClipActivation(float lo, float hi) : FusedActivation
{
    /// <summary>The lower bound.</summary>
    public float Lo => lo;

    /// <summary>The upper bound.</summary>
    public float Hi => hi;

    public override void Apply(KernelPath path, bool exact, Span<float> row) => UnaryLoops.Clip<float>(row, row, true, lo, true, hi);
}

/// <summary>Any <see cref="IFloatUnary"/> function (HardSwish, Sigmoid, HardSigmoid, Gelu, Tanh, LeakyRelu).</summary>
internal sealed class FloatActivation<TOp>(TOp op) : FusedActivation where TOp : struct, IFloatUnary
{
    /// <summary>The function, with its parameters.</summary>
    public TOp Function => op;

    public override void Apply(KernelPath path, bool exact, Span<float> row) => FloatUnary.Run(path, exact, op, row, row);
}

/// <summary>
/// The tile epilogue for a product followed by "add a bias vector, then an activation". The bias is indexed by column (a linear
/// layer: one value per output feature) or by row (a convolution lowered to a product: one value per output channel). The add is one
/// rounded addition per element, as the stand-alone Add is, and in exact mode a NaN result becomes 0x7FC00000.
/// </summary>
internal sealed unsafe class BiasActivationEpilogue : IGemmEpilogue
{
    private readonly float[]? _bias;
    private readonly bool _perRow;
    private readonly int _rowOffset;
    private readonly FusedActivation? _activation;
    private readonly KernelPath _path;

    /// <param name="bias">The bias values, or null for none.</param>
    /// <param name="perRow">True: row r of C adds <c>bias[rowOffset + r]</c>; false: column j adds <c>bias[j]</c>.</param>
    /// <param name="rowOffset">With <paramref name="perRow"/>, the index of C's first row in <paramref name="bias"/> (a convolution group's first channel).</param>
    /// <param name="activation">Applied after the bias, or null.</param>
    /// <param name="path">The session's instruction set, handed on to the activation kernels.</param>
    public BiasActivationEpilogue(float[]? bias, bool perRow, int rowOffset, FusedActivation? activation, KernelPath path)
    {
        _bias = bias;
        _perRow = perRow;
        _rowOffset = rowOffset;
        _activation = activation;
        _path = path;
    }

    /// <summary>The same epilogue for a product whose first row is channel <paramref name="rowOffset"/> of the bias.</summary>
    public BiasActivationEpilogue WithRowOffset(int rowOffset) => new(_bias, _perRow, rowOffset, _activation, _path);

    /// <remarks>Bounds: the tile lies inside C (the product checked m, n and ldc); the bias holds one value per column of C or per row of C plus the row offset (checked when the kernel is built).</remarks>
    public void Apply(float* tile, nint ldc, int row0, int col0, int rows, int cols, KernelMode mode)
    {
        bool exact = mode == KernelMode.Exact;
        if (!exact && _perRow && _bias is not null && _activation is null or ReluActivation)
        {
            // The common convolution epilogue in one pass over each row, on pointers: one rounded add, then the Relu rule.
            bool relu = _activation is not null;
            for (int r = 0; r < rows; r++)
            {
                float b = _bias[_rowOffset + row0 + r];
                if (relu)
                {
                    RowEpilogue.BiasRelu(tile + (r * ldc), cols, b);
                }
                else
                {
                    RowEpilogue.Add(tile + (r * ldc), cols, b);
                }
            }

            return;
        }

        for (int r = 0; r < rows; r++)
        {
            var row = new Span<float>(tile + (r * ldc), cols);
            if (_bias is not null)
            {
                if (_perRow)
                {
                    AddScalar(row, _bias[_rowOffset + row0 + r], exact);
                }
                else
                {
                    AddVector(row, _bias.AsSpan(col0, cols), exact);
                }
            }

            _activation?.Apply(_path, exact, row);
        }
    }

    internal static void AddScalar(Span<float> row, float b, bool exact)
    {
        int i = 0;
        if (Vector128.IsHardwareAccelerated)
        {
            Vector128<float> vb = Vector128.Create(b);
            Vector128<float> nan = Vector128.Create(ExactElementwise.CanonicalNaN);
            for (; i + 4 <= row.Length; i += 4)
            {
                Vector128<float> v = Vector128.Create(row.Slice(i, 4)) + vb;
                if (exact)
                {
                    v = Vector128.ConditionalSelect(Vector128.Equals(v, v), v, nan);
                }

                v.CopyTo(row.Slice(i, 4));
            }
        }

        for (; i < row.Length; i++)
        {
            float v = row[i] + b;
            row[i] = exact && float.IsNaN(v) ? ExactElementwise.CanonicalNaN : v;
        }
    }

    private static void AddVector(Span<float> row, ReadOnlySpan<float> bias, bool exact)
    {
        int i = 0;
        if (Vector128.IsHardwareAccelerated)
        {
            Vector128<float> nan = Vector128.Create(ExactElementwise.CanonicalNaN);
            for (; i + 4 <= row.Length; i += 4)
            {
                Vector128<float> v = Vector128.Create(row.Slice(i, 4)) + Vector128.Create(bias.Slice(i, 4));
                if (exact)
                {
                    v = Vector128.ConditionalSelect(Vector128.Equals(v, v), v, nan);
                }

                v.CopyTo(row.Slice(i, 4));
            }
        }

        for (; i < row.Length; i++)
        {
            float v = row[i] + bias[i];
            row[i] = exact && float.IsNaN(v) ? ExactElementwise.CanonicalNaN : v;
        }
    }
}

/// <summary>
/// Fast-mode epilogue loops over one row of finished results, on pointers and 256-bit lanes. Each element gets exactly what
/// the stand-alone operators give it (one rounded add; then Relu: positive values and NaN pass, everything else becomes +0), in one pass
/// instead of one per operator, so the bits are those of the separate passes whatever the width.
/// </summary>
internal static unsafe class RowEpilogue
{
    /// <summary><c>p[i] += b</c> for i below <paramref name="n"/>. Bounds: the caller owns <c>p[0..n)</c>.</summary>
    public static void Add(float* p, int n, float b)
    {
        int i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            Vector256<float> vb = Vector256.Create(b);
            for (; i + 8 <= n; i += 8)
            {
                (Vector256.Load(p + i) + vb).Store(p + i);
            }
        }

        if (Vector128.IsHardwareAccelerated)
        {
            Vector128<float> vb = Vector128.Create(b);
            for (; i + 4 <= n; i += 4)
            {
                (Vector128.Load(p + i) + vb).Store(p + i);
            }
        }

        for (; i < n; i++)
        {
            p[i] += b;
        }
    }

    /// <summary><c>p[i] = relu(p[i] + b)</c> for i below <paramref name="n"/>. Bounds: the caller owns <c>p[0..n)</c>.</summary>
    public static void BiasRelu(float* p, int n, float b)
    {
        int i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            Vector256<float> vb = Vector256.Create(b);
            Vector256<float> zero = Vector256<float>.Zero;
            for (; i + 8 <= n; i += 8)
            {
                Vector256<float> v = Vector256.Load(p + i) + vb;
                Vector256.ConditionalSelect(Vector256.LessThanOrEqual(v, zero), zero, v).Store(p + i);
            }
        }

        if (Vector128.IsHardwareAccelerated)
        {
            Vector128<float> vb = Vector128.Create(b);
            Vector128<float> zero = Vector128<float>.Zero;
            for (; i + 4 <= n; i += 4)
            {
                Vector128<float> v = Vector128.Load(p + i) + vb;
                Vector128.ConditionalSelect(Vector128.LessThanOrEqual(v, zero), zero, v).Store(p + i);
            }
        }

        for (; i < n; i++)
        {
            float v = p[i] + b;
            p[i] = v <= 0f ? 0f : v;
        }
    }
}
