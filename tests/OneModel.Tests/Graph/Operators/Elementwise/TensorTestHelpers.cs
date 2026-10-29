using System.Runtime.InteropServices;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Graph.Operators.Elementwise;

/// <summary>Moving tensors between raw little-endian bytes, tensors and doubles for any stored element type.</summary>
internal static class TensorTestHelpers
{
    private struct FromBytes(byte[] bytes, Shape shape) : ITypeVisitor
    {
        public Tensor Result;

        public void Visit<T>() where T : unmanaged => Result = Tensor.FromSpan<T>(MemoryMarshal.Cast<byte, T>(bytes), shape);
    }

    private struct ToBytes(Tensor t) : ITypeVisitor
    {
        public byte[]? Result;

        public void Visit<T>() where T : unmanaged => Result = MemoryMarshal.AsBytes(t.Data<T>()).ToArray();
    }

    internal static Tensor Make(DType type, int[] shape, byte[] data)
    {
        var v = new FromBytes(data, Shape.Create(shape));
        TypeDispatch.Any(type, ref v);
        return v.Result;
    }

    internal static byte[] BytesOf(Tensor t)
    {
        var v = new ToBytes(t);
        TypeDispatch.Any(t.DType, ref v);
        return v.Result!;
    }

    internal static double[] Doubles(DType type, byte[] b)
    {
        ReadOnlySpan<byte> s = b;
        return type switch
        {
            DType.F32 => MemoryMarshal.Cast<byte, float>(s).ToArray().Select(x => (double)x).ToArray(),
            DType.F64 => MemoryMarshal.Cast<byte, double>(s).ToArray(),
            DType.BF16 => MemoryMarshal.Cast<byte, BFloat16>(s).ToArray().Select(x => (double)(float)x).ToArray(),
            DType.F16 => MemoryMarshal.Cast<byte, Half>(s).ToArray().Select(x => (double)x).ToArray(),
            DType.I8 => MemoryMarshal.Cast<byte, sbyte>(s).ToArray().Select(x => (double)x).ToArray(),
            DType.I16 => MemoryMarshal.Cast<byte, short>(s).ToArray().Select(x => (double)x).ToArray(),
            DType.I32 => MemoryMarshal.Cast<byte, int>(s).ToArray().Select(x => (double)x).ToArray(),
            DType.I64 => MemoryMarshal.Cast<byte, long>(s).ToArray().Select(x => (double)x).ToArray(),
            DType.U8 => b.Select(x => (double)x).ToArray(),
            DType.U16 => MemoryMarshal.Cast<byte, ushort>(s).ToArray().Select(x => (double)x).ToArray(),
            DType.U32 => MemoryMarshal.Cast<byte, uint>(s).ToArray().Select(x => (double)x).ToArray(),
            DType.U64 => MemoryMarshal.Cast<byte, ulong>(s).ToArray().Select(x => (double)x).ToArray(),
            DType.Bool => b.Select(x => x == 0 ? 0.0 : 1.0).ToArray(),
            _ => throw new InvalidOperationException(type.ToString()),
        };
    }

    internal static bool IsFloat(DType t) => t is DType.F32 or DType.F64 or DType.F16 or DType.BF16;
}
