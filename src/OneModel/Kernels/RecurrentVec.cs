using System;
using System.Runtime.Intrinsics;
using RecursiveAI.OneModel.Kernels.Exact;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// The element-wise arithmetic of a recurrent step that is not an activation function, on fixed-width lanes. Only IEEE addition and
/// multiplication, one rounding each and in the order the scalar code in <c>docs/ops-recurrent.md</c> has, so every lane width and the
/// scalar twin give the same bits (exact mode relies on that).
/// </summary>
internal static class RecurrentVec
{
    /// <summary><c>row[i] = ((xw[i] + row[i]) + wb[i]) + rb[i]</c> for i in [0, count); a missing bias is skipped.</summary>
    public static void PreActivation(VecIsa isa, ReadOnlySpan<float> xw, Span<float> row, ReadOnlySpan<float> wb, ReadOnlySpan<float> rb)
    {
        int count = row.Length;
        if (xw.Length < count || (!wb.IsEmpty && wb.Length < count) || (!rb.IsEmpty && rb.Length < count))
        {
            throw new ArgumentException("An operand is shorter than the row.");
        }

        switch (VecIsaInfo.Resolve(isa))
        {
            case VecIsa.V512: PreActivation<Vector512<float>, ExactVec512>(xw, row, wb, rb); break;
            case VecIsa.V256: PreActivation<Vector256<float>, ExactVec256>(xw, row, wb, rb); break;
            case VecIsa.V128: PreActivation<Vector128<float>, ExactVec128>(xw, row, wb, rb); break;
            default: PreActivation<float, ExactVecScalar>(xw, row, wb, rb); break;
        }
    }

    /// <summary><c>c[i] = (f[i] * c[i]) + (g[i] * u[i])</c>: the LSTM cell update as two products and one sum.</summary>
    public static void CellUpdate(VecIsa isa, Span<float> c, ReadOnlySpan<float> f, ReadOnlySpan<float> g, ReadOnlySpan<float> u)
    {
        if (f.Length < c.Length || g.Length < c.Length || u.Length < c.Length)
        {
            throw new ArgumentException("An operand is shorter than the cell state.");
        }

        switch (VecIsaInfo.Resolve(isa))
        {
            case VecIsa.V512: CellUpdate<Vector512<float>, ExactVec512>(c, f, g, u); break;
            case VecIsa.V256: CellUpdate<Vector256<float>, ExactVec256>(c, f, g, u); break;
            case VecIsa.V128: CellUpdate<Vector128<float>, ExactVec128>(c, f, g, u); break;
            default: CellUpdate<float, ExactVecScalar>(c, f, g, u); break;
        }
    }

    /// <summary><c>dst[i] = a[i] * b[i]</c>.</summary>
    public static void Multiply(VecIsa isa, Span<float> dst, ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length < dst.Length || b.Length < dst.Length)
        {
            throw new ArgumentException("An operand is shorter than the destination.");
        }

        switch (VecIsaInfo.Resolve(isa))
        {
            case VecIsa.V512: Multiply<Vector512<float>, ExactVec512>(dst, a, b); break;
            case VecIsa.V256: Multiply<Vector256<float>, ExactVec256>(dst, a, b); break;
            case VecIsa.V128: Multiply<Vector128<float>, ExactVec128>(dst, a, b); break;
            default: Multiply<float, ExactVecScalar>(dst, a, b); break;
        }
    }

    // Bounds: the public methods checked that every operand holds at least row.Length elements; each vector access below has
    // i + lanes <= count.
    private static void PreActivation<TV, TO>(ReadOnlySpan<float> xw, Span<float> row, ReadOnlySpan<float> wb, ReadOnlySpan<float> rb)
        where TV : struct
        where TO : IVecOps<TV>
    {
        int count = row.Length;
        int lanes = TO.Lanes;
        bool hasWb = !wb.IsEmpty, hasRb = !rb.IsEmpty;
        int i = 0;
        for (; i + lanes <= count; i += lanes)
        {
            TV v = TO.Add(TO.Load(xw, i), TO.Load(row, i));
            if (hasWb)
            {
                v = TO.Add(v, TO.Load(wb, i));
            }

            if (hasRb)
            {
                v = TO.Add(v, TO.Load(rb, i));
            }

            TO.Store(row, i, v);
        }

        for (; i < count; i++)
        {
            float v = xw[i] + row[i];
            if (hasWb)
            {
                v += wb[i];
            }

            if (hasRb)
            {
                v += rb[i];
            }

            row[i] = v;
        }
    }

    private static void CellUpdate<TV, TO>(Span<float> c, ReadOnlySpan<float> f, ReadOnlySpan<float> g, ReadOnlySpan<float> u)
        where TV : struct
        where TO : IVecOps<TV>
    {
        int count = c.Length;
        int lanes = TO.Lanes;
        int i = 0;
        for (; i + lanes <= count; i += lanes)
        {
            TV keep = TO.Mul(TO.Load(f, i), TO.Load(c, i));
            TV add = TO.Mul(TO.Load(g, i), TO.Load(u, i));
            TO.Store(c, i, TO.Add(keep, add));
        }

        for (; i < count; i++)
        {
            float keep = f[i] * c[i];
            float add = g[i] * u[i];
            c[i] = keep + add;
        }
    }

    private static void Multiply<TV, TO>(Span<float> dst, ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        where TV : struct
        where TO : IVecOps<TV>
    {
        int count = dst.Length;
        int lanes = TO.Lanes;
        int i = 0;
        for (; i + lanes <= count; i += lanes)
        {
            TO.Store(dst, i, TO.Mul(TO.Load(a, i), TO.Load(b, i)));
        }

        for (; i < count; i++)
        {
            dst[i] = a[i] * b[i];
        }
    }
}
