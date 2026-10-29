using System;
using System.Collections.Generic;
using System.Threading;
using System.Runtime.CompilerServices;

namespace RecursiveAI.OneModel.Tensors;

/// <summary>
/// Element types a <see cref="Tensor"/> can hold. The numeric values are ONNX's <c>TensorProto.DataType</c> codes, so importing a
/// model needs no mapping table. Types ONNX defines but this library does not store are listed by <see cref="DTypes.OnnxName"/>
/// and rejected by <see cref="DTypes.FromOnnx"/>.
/// </summary>
public enum DType : byte
{
    /// <summary>No type: the value of a default <see cref="Tensor"/>.</summary>
    Undefined = 0,
    /// <summary>IEEE 754 binary32 (<see cref="float"/>).</summary>
    F32 = 1,
    /// <summary>Unsigned 8-bit integer (<see cref="byte"/>).</summary>
    U8 = 2,
    /// <summary>Signed 8-bit integer (<see cref="sbyte"/>).</summary>
    I8 = 3,
    /// <summary>Unsigned 16-bit integer (<see cref="ushort"/>).</summary>
    U16 = 4,
    /// <summary>Signed 16-bit integer (<see cref="short"/>).</summary>
    I16 = 5,
    /// <summary>Signed 32-bit integer (<see cref="int"/>).</summary>
    I32 = 6,
    /// <summary>Signed 64-bit integer (<see cref="long"/>).</summary>
    I64 = 7,
    /// <summary>Boolean, one byte per element, 0 or 1 (<see cref="bool"/>).</summary>
    Bool = 9,
    /// <summary>IEEE 754 binary16 (<see cref="Half"/>).</summary>
    F16 = 10,
    /// <summary>IEEE 754 binary64 (<see cref="double"/>).</summary>
    F64 = 11,
    /// <summary>Unsigned 32-bit integer (<see cref="uint"/>).</summary>
    U32 = 12,
    /// <summary>Unsigned 64-bit integer (<see cref="ulong"/>).</summary>
    U64 = 13,
    /// <summary>bfloat16: the top 16 bits of binary32 (<see cref="BFloat16"/>).</summary>
    BF16 = 16,
}

/// <summary>Facts about <see cref="DType"/> values and their mapping to CLR types and ONNX codes.</summary>
public static class DTypes
{
    /// <summary>Size of one element in bytes.</summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is <see cref="DType.Undefined"/> or not a defined value.</exception>
    public static int SizeOf(DType type) => type switch
    {
        DType.U8 or DType.I8 or DType.Bool => 1,
        DType.U16 or DType.I16 or DType.F16 or DType.BF16 => 2,
        DType.F32 or DType.I32 or DType.U32 => 4,
        DType.F64 or DType.I64 or DType.U64 => 8,
        _ => throw new ArgumentOutOfRangeException(nameof(type), type, "The element type is undefined."),
    };

    /// <summary>True for Float16, BFloat16, Float32 and Float64.</summary>
    public static bool IsFloatingPoint(DType type) => type is DType.F16 or DType.BF16 or DType.F32 or DType.F64;

    /// <summary>True for the signed and unsigned integer types (not <see cref="DType.Bool"/>).</summary>
    public static bool IsInteger(DType type) =>
        type is DType.U8 or DType.I8 or DType.U16 or DType.I16 or DType.I32 or DType.I64 or DType.U32 or DType.U64;

    /// <summary>The type's name as ONNX spells it in <c>TensorProto.DataType</c>, for example <c>FLOAT</c> or <c>BFLOAT16</c>.</summary>
    public static string Name(DType type) => OnnxName((int)type);

    /// <summary>The CLR element type for <typeparamref name="T"/>.</summary>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> is not one of the stored element types.</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static DType Of<T>() where T : unmanaged
    {
        if (typeof(T) == typeof(float)) return DType.F32;
        if (typeof(T) == typeof(int)) return DType.I32;
        if (typeof(T) == typeof(long)) return DType.I64;
        if (typeof(T) == typeof(byte)) return DType.U8;
        if (typeof(T) == typeof(sbyte)) return DType.I8;
        if (typeof(T) == typeof(bool)) return DType.Bool;
        if (typeof(T) == typeof(Half)) return DType.F16;
        if (typeof(T) == typeof(BFloat16)) return DType.BF16;
        if (typeof(T) == typeof(double)) return DType.F64;
        if (typeof(T) == typeof(short)) return DType.I16;
        if (typeof(T) == typeof(ushort)) return DType.U16;
        if (typeof(T) == typeof(uint)) return DType.U32;
        if (typeof(T) == typeof(ulong)) return DType.U64;
        throw new NotSupportedException($"{typeof(T).Name} is not a tensor element type.");
    }

    /// <summary>Maps an ONNX <c>TensorProto.DataType</c> code to a stored type.</summary>
    /// <exception cref="NotSupportedException">The code is a valid ONNX type this library does not store (the message names it), or is not an ONNX code.</exception>
    public static DType FromOnnx(int code) =>
        TryFromOnnx(code, out DType t) ? t : throw new NotSupportedException($"ONNX element type {OnnxName(code)} (code {code}) is not supported.");

    /// <summary>Like <see cref="FromOnnx"/>, returning false instead of throwing.</summary>
    public static bool TryFromOnnx(int code, out DType type)
    {
        switch (code)
        {
            case 1 or 2 or 3 or 4 or 5 or 6 or 7 or 9 or 10 or 11 or 12 or 13 or 16:
                type = (DType)code;
                return true;
            default:
                type = DType.Undefined;
                return false;
        }
    }

    /// <summary>The ONNX code for a stored type (the enum's value).</summary>
    public static int ToOnnx(DType type) => (int)type;

    /// <summary>Names every ONNX <c>TensorProto.DataType</c> code, supported or not; unknown codes give <c>UNKNOWN(code)</c>.</summary>
    public static string OnnxName(int code) => code switch
    {
        0 => "UNDEFINED", 1 => "FLOAT", 2 => "UINT8", 3 => "INT8", 4 => "UINT16", 5 => "INT16", 6 => "INT32", 7 => "INT64",
        8 => "STRING", 9 => "BOOL", 10 => "FLOAT16", 11 => "DOUBLE", 12 => "UINT32", 13 => "UINT64", 14 => "COMPLEX64",
        15 => "COMPLEX128", 16 => "BFLOAT16", 17 => "FLOAT8E4M3FN", 18 => "FLOAT8E4M3FNUZ", 19 => "FLOAT8E5M2",
        20 => "FLOAT8E5M2FNUZ", 21 => "UINT4", 22 => "INT4", 23 => "FLOAT4E2M1", 24 => "FLOAT8E8M0", 25 => "UINT2",
        26 => "INT2", 27 => "FLOAT6E2M3", 28 => "FLOAT6E3M2",
        _ => $"UNKNOWN({code})",
    };

    internal static void ThrowIfNotStorable(DType type, string? paramName = null)
    {
        if (!TryFromOnnx((int)type, out _))
        {
            throw new ArgumentOutOfRangeException(paramName ?? nameof(type), type, "The element type is undefined or not stored.");
        }
    }
}
