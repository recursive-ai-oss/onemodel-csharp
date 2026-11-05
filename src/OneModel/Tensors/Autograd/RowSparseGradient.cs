using System;
namespace RecursiveAI.OneModel.Tensors;

/// <summary>
/// The gradient of a [rows, width] table that touches only some rows (an embedding lookup): the touched row numbers in ascending
/// order, each once, and a [<see cref="RowCount"/>, width] tensor of their gradients.
/// </summary>
/// <remarks>
/// The gradient of a row is the sum of the output gradients of every lookup of it, added in increasing position order starting from
/// zero, so the values are the dense gradient's rows bit for bit and do not depend on the thread count. Rows that received nothing
/// (and the padding row) are absent. The object owns <see cref="Values"/>; <see cref="Tensor.SetRequiresGrad"/>, the optimizers' zeroing
/// and <see cref="Backprop.ClearGrad"/> dispose it.
/// </remarks>
public sealed class RowSparseGradient : IDisposable
{
    private readonly int[] _rows;

    internal RowSparseGradient(Shape denseShape, int[] rows, Tensor values)
    {
        DenseShape = denseShape;
        _rows = rows;
        Values = values;
    }

    /// <summary>The shape of the table this is the gradient of: [rows, width].</summary>
    public Shape DenseShape { get; }

    /// <summary>Number of touched rows, at least 1.</summary>
    public int RowCount => _rows.Length;

    /// <summary>The touched row numbers, strictly ascending, each in [0, rows).</summary>
    public ReadOnlySpan<int> Rows => _rows;

    /// <summary>The gradients of the touched rows, float32 [<see cref="RowCount"/>, width], row k belonging to <c>Rows[k]</c>. Writable (gradient clipping scales it).</summary>
    public Tensor Values { get; }

    /// <summary>Bytes held: the values plus the row numbers.</summary>
    public long ByteCount => (long)Values.Length * sizeof(float) + (long)_rows.Length * sizeof(int);

    /// <summary>Scatters the rows into a new zeroed dense tensor of <see cref="DenseShape"/>. The caller owns (and should dispose) the result.</summary>
    public Tensor ToDense()
    {
        Tensor dense = Tensor.Zeros(DType.F32, DenseShape);
        AddTo(dense.MutableData<float>());
        return dense;
    }

    // dense[row] += values[k] in ascending row order.
    internal void AddTo(Span<float> dense)
    {
        int width = DenseShape[1];
        ReadOnlySpan<float> v = Values.InternalData<float>();
        for (int k = 0; k < _rows.Length; k++)
        {
            Span<float> d = dense.Slice(_rows[k] * width, width);
            ReadOnlySpan<float> s = v.Slice(k * width, width);
            for (int j = 0; j < width; j++)
            {
                d[j] += s[j];
            }
        }
    }

    // this + other: the union of the rows; a row in both gets this + other (this first).
    internal static RowSparseGradient Merge(RowSparseGradient a, RowSparseGradient b)
    {
        int width = a.DenseShape[1];
        int[] ra = a._rows, rb = b._rows;
        var rows = new int[ra.Length + rb.Length];
        int i = 0, j = 0, n = 0;
        while (i < ra.Length || j < rb.Length)
        {
            if (j == rb.Length || (i < ra.Length && ra[i] < rb[j]))
            {
                rows[n++] = ra[i++];
            }
            else if (i == ra.Length || rb[j] < ra[i])
            {
                rows[n++] = rb[j++];
            }
            else
            {
                rows[n++] = ra[i];
                i++;
                j++;
            }
        }

        Array.Resize(ref rows, n);
        Tensor values = Tensor.Zeros(DType.F32, Shape.Create(n, width));
        Span<float> o = values.MutableData<float>();
        ReadOnlySpan<float> va = a.Values.InternalData<float>(), vb = b.Values.InternalData<float>();
        i = 0;
        j = 0;
        for (int k = 0; k < n; k++)
        {
            Span<float> d = o.Slice(k * width, width);
            if (i < ra.Length && ra[i] == rows[k])
            {
                va.Slice(i * width, width).CopyTo(d);
                i++;
            }

            if (j < rb.Length && rb[j] == rows[k])
            {
                ReadOnlySpan<float> s = vb.Slice(j * width, width);
                for (int c = 0; c < width; c++)
                {
                    d[c] += s[c];
                }

                j++;
            }
        }

        return new RowSparseGradient(a.DenseShape, rows, values);
    }

    /// <summary>Frees the values. Safe to call more than once.</summary>
    public void Dispose() => Values.Dispose();
}
