using System;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

/// <summary>Called with an unmanaged type of the size of a tensor's elements, for operators that only move elements.</summary>
internal interface ICarrierVisitor
{
    void Visit<TC>() where TC : unmanaged;
}

/// <summary>Turns an element size into a compile-time carrier type (byte, ushort, uint, ulong) without reflection.</summary>
internal static class Carriers
{
    public static void Run<TV>(int elementSize, ref TV v) where TV : struct, ICarrierVisitor, allows ref struct
    {
        switch (elementSize)
        {
            case 1: v.Visit<byte>(); break;
            case 2: v.Visit<ushort>(); break;
            case 4: v.Visit<uint>(); break;
            case 8: v.Visit<ulong>(); break;
            default: throw new InvalidOperationException($"Elements of {elementSize} bytes cannot be moved.");
        }
    }

    /// <summary>Row-major element strides of <paramref name="dims"/>.</summary>
    public static void Strides(ReadOnlySpan<int> dims, Span<int> strides) => OpUtil.RowMajor(dims, strides);
}
