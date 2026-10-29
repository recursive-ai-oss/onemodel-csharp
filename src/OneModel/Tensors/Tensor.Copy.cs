using System;
using System.Collections.Generic;
using System.Threading;
using System.Buffers.Binary;

namespace RecursiveAI.OneModel.Tensors;

// Copies, materialization, element-type conversion and layout conversion. These are reference implementations: plain loops that
// define the result; the kernel layer provides fast versions behind the same semantics.
public readonly partial struct Tensor
{
    private Span<T> Raw<T>() where T : unmanaged => _buffer!.GetSpan<T>(_generation);

    /// <summary>
    /// Returns this tensor if it is already contiguous, otherwise a new contiguous tensor (owned by the caller) with the same
    /// logical contents. Differentiable: the copy's backward is the identity on the logical shape.
    /// </summary>
    public Tensor Contiguous(BufferPool? pool = null)
    {
        ThrowIfEmpty();
        if (IsContiguous)
        {
            return this;
        }

        Tensor dst = Create(_dtype, _shape, pool);
        CopyStrided(this, dst);
        return GradMode.ShouldRecord(this) ? Rerecord(dst) : dst;
    }

    /// <summary>Always copies into a new contiguous tensor (owned by the caller). Not tracked by autograd.</summary>
    public Tensor Clone(BufferPool? pool = null)
    {
        ThrowIfEmptyOrBlocked();
        Tensor dst = Create(_dtype, _shape, pool);
        CopyStrided(this, dst);
        return dst;
    }

    /// <summary>
    /// Copies <paramref name="src"/>'s elements into this tensor's elements (any strides on either side; same shape and type) and
    /// bumps the storage version. The two must not partly overlap in storage unless they are the same view.
    /// </summary>
    /// <exception cref="TensorShapeException">Shapes differ.</exception>
    /// <exception cref="ArgumentException">Element types differ.</exception>
    /// <exception cref="InvalidOperationException">Either tensor is tracked by autograd while recording is on (in-place updates of tracked tensors are not supported; use <see cref="Detach"/> or <see cref="GradMode.NoGrad"/>).</exception>
    public void CopyFrom(in Tensor src)
    {
        ThrowIfEmptyOrBlocked();
        src.ThrowIfEmptyOrBlocked();
        if (_shape != src._shape)
        {
            throw new TensorShapeException($"Cannot copy shape {src._shape} into {_shape}.");
        }

        if (_dtype != src._dtype)
        {
            throw new ArgumentException($"Cannot copy {src._dtype} into {_dtype}; use Cast.");
        }

        if (GradMode.IsEnabled && (RequiresGrad || src.RequiresGrad))
        {
            throw new InvalidOperationException("In-place copies into or from tensors that require gradients are not supported.");
        }

        CopyStrided(src, this);
        _buffer!.Touch();
    }

    /// <summary>Sets every element to <paramref name="value"/>; bumps the storage version.</summary>
    public void Fill<T>(T value) where T : unmanaged
    {
        CheckType<T>();
        ThrowIfBlocked();
        if (IsContiguous)
        {
            MutableData<T>().Fill(value);
            return;
        }

        using Tensor one = Scalar(value);
        CopyFrom(one.Expand(_shape));
    }

    /// <summary>The elements in row-major logical order as a new array.</summary>
    public T[] ToArray<T>() where T : unmanaged
    {
        CheckType<T>();
        ThrowIfBlocked();
        if (IsContiguous)
        {
            return Data<T>().ToArray();
        }

        using Tensor c = Clone();
        return c.Data<T>().ToArray();
    }

    private static void CopyStrided(in Tensor src, in Tensor dst)
    {
        switch (src.ElementSize)
        {
            case 1: CopyTyped<byte>(src, dst); break;
            case 2: CopyTyped<ushort>(src, dst); break;
            case 4: CopyTyped<uint>(src, dst); break;
            default: CopyTyped<ulong>(src, dst); break;
        }
    }

    private static void CopyTyped<T>(in Tensor src, in Tensor dst) where T : unmanaged
    {
        Shape shape = src._shape;
        int len = shape.Length;
        if (len == 0)
        {
            return;
        }

        ReadOnlySpan<T> s = src.Raw<T>();
        Span<T> d = dst.Raw<T>();
        if (src.IsContiguous && dst.IsContiguous)
        {
            s.Slice(src._offset, len).CopyTo(d.Slice(dst._offset, len));
            return;
        }

        int rank = shape.Rank;
        if (rank == 0)
        {
            d[dst._offset] = s[src._offset];
            return;
        }

        Span<int> idx = stackalloc int[Shape.MaxRank];
        int inner = shape.Unchecked(rank - 1), sIn = src._strides.Unchecked(rank - 1), dIn = dst._strides.Unchecked(rank - 1);
        int sp = src._offset, dp = dst._offset;
        while (true)
        {
            for (int i = 0, a = sp, b = dp; i < inner; i++, a += sIn, b += dIn)
            {
                d[b] = s[a];
            }

            int ax = rank - 2;
            for (; ax >= 0; ax--)
            {
                idx[ax]++;
                sp += src._strides.Unchecked(ax);
                dp += dst._strides.Unchecked(ax);
                if (idx[ax] < shape.Unchecked(ax))
                {
                    break;
                }

                sp -= src._strides.Unchecked(ax) * shape.Unchecked(ax);
                dp -= dst._strides.Unchecked(ax) * shape.Unchecked(ax);
                idx[ax] = 0;
            }

            if (ax < 0)
            {
                return;
            }
        }
    }

    // ---- element type conversion -------------------------------------------------------------------------------------------

    /// <summary>
    /// Converts to another element type, into a new contiguous tensor (owned by the caller). Rules (reference semantics):
    /// float to float rounds to nearest even (float64 to float16 or bfloat16 goes through float32, so it rounds twice); integer or
    /// bool to float rounds to nearest even; float to integer truncates toward zero, saturates at the target's range, and maps NaN to 0;
    /// integer to integer wraps modulo 2^bits (two's complement); anything to bool is "not equal to zero" (NaN is true); bool to a
    /// number is 0 or 1. Not tracked by autograd.
    /// </summary>
    public Tensor Cast(DType target, BufferPool? pool = null)
    {
        ThrowIfEmptyOrBlocked();
        DTypes.ThrowIfNotStorable(target);
        using Tensor src = IsContiguous ? default : Clone(pool);
        Tensor from = IsContiguous ? this : src;
        Tensor dst = Create(target, _shape, pool);
        ReadOnlySpan<byte> s = from.Raw<byte>();
        Span<byte> d = dst.Raw<byte>();
        int se = DTypes.SizeOf(_dtype), de = DTypes.SizeOf(target), n = Length, so = from._offset * se;
        for (int i = 0; i < n; i++)
        {
            Num v = ReadNum(_dtype, s.Slice(so + (i * se), se));
            WriteNum(target, v, d.Slice(i * de, de));
        }

        if (GradMode.ShouldRecord(this) && DTypes.IsFloatingPoint(target))
        {
            // Backward casts the gradient back to InputDType(0).
            return BackwardRecord.Create(OpKind.Cast, 0, this).Attach(dst);
        }

        return dst;
    }

    // A decoded element: a float64 for floating-point types, a signed or unsigned 64-bit integer otherwise.
    private readonly record struct Num(int Kind, long I, double F); // Kind 0 = integer (signed), 1 = unsigned 64, 2 = float

    private static Num ReadNum(DType t, ReadOnlySpan<byte> b) => t switch
    {
        DType.F32 => new Num(2, 0, BinaryPrimitives.ReadSingleLittleEndian(b)),
        DType.F64 => new Num(2, 0, BinaryPrimitives.ReadDoubleLittleEndian(b)),
        DType.F16 => new Num(2, 0, Float16.ToSingle(BinaryPrimitives.ReadUInt16LittleEndian(b))),
        DType.BF16 => new Num(2, 0, BFloat16.ToSingleBits(BinaryPrimitives.ReadUInt16LittleEndian(b))),
        DType.I8 => new Num(0, (sbyte)b[0], 0),
        DType.U8 => new Num(0, b[0], 0),
        DType.Bool => new Num(0, b[0] != 0 ? 1 : 0, 0),
        DType.I16 => new Num(0, BinaryPrimitives.ReadInt16LittleEndian(b), 0),
        DType.U16 => new Num(0, BinaryPrimitives.ReadUInt16LittleEndian(b), 0),
        DType.I32 => new Num(0, BinaryPrimitives.ReadInt32LittleEndian(b), 0),
        DType.U32 => new Num(0, BinaryPrimitives.ReadUInt32LittleEndian(b), 0),
        DType.I64 => new Num(0, BinaryPrimitives.ReadInt64LittleEndian(b), 0),
        _ => new Num(1, BinaryPrimitives.ReadInt64LittleEndian(b), 0), // UInt64: bits kept in the long
    };

    private static void WriteNum(DType t, Num v, Span<byte> b)
    {
        switch (t)
        {
            case DType.F32: BinaryPrimitives.WriteSingleLittleEndian(b, ToSingle(v)); return;
            case DType.F64: BinaryPrimitives.WriteDoubleLittleEndian(b, v.Kind == 2 ? v.F : v.Kind == 1 ? (double)(ulong)v.I : v.I); return;
            case DType.F16: BinaryPrimitives.WriteUInt16LittleEndian(b, Float16.FromSingle(ToSingle(v))); return;
            case DType.BF16: BinaryPrimitives.WriteUInt16LittleEndian(b, BFloat16.FromSingleBits(ToSingle(v))); return;
            case DType.Bool: b[0] = (byte)(v.Kind == 2 ? (v.F != 0 || double.IsNaN(v.F) ? 1 : 0) : v.I != 0 ? 1 : 0); return;
        }

        // Integer targets: floats truncate and saturate (NaN -> 0); integers wrap.
        long x;
        if (v.Kind == 2)
        {
            (long lo, long hi) = t switch
            {
                DType.I8 => (sbyte.MinValue, (long)sbyte.MaxValue),
                DType.U8 => (0L, (long)byte.MaxValue),
                DType.I16 => (short.MinValue, (long)short.MaxValue),
                DType.U16 => (0L, (long)ushort.MaxValue),
                DType.I32 => (int.MinValue, (long)int.MaxValue),
                DType.U32 => (0L, (long)uint.MaxValue),
                _ => (long.MinValue, long.MaxValue),
            };
            double f = Math.Truncate(v.F);
            if (double.IsNaN(f))
            {
                x = 0;
            }
            else if (t == DType.U64)
            {
                x = f <= 0 ? 0 : f >= 18446744073709551615.0 ? -1 : (long)(ulong)f;
            }
            else
            {
                x = f <= lo ? lo : f >= hi ? hi : (long)f;
            }
        }
        else
        {
            x = v.I;
        }

        switch (DTypes.SizeOf(t))
        {
            case 1: b[0] = unchecked((byte)x); break;
            case 2: BinaryPrimitives.WriteUInt16LittleEndian(b, unchecked((ushort)x)); break;
            case 4: BinaryPrimitives.WriteUInt32LittleEndian(b, unchecked((uint)x)); break;
            default: BinaryPrimitives.WriteInt64LittleEndian(b, x); break;
        }
    }

    private static float ToSingle(Num v) => v.Kind == 2 ? (float)v.F : v.Kind == 1 ? (float)(ulong)v.I : (float)v.I;

    // ---- layout conversion ------------------------------------------------------------------------------------------------

    /// <summary>
    /// Converts a rank-4 tensor whose logical shape is (N, C, H, W) to <paramref name="target"/>. The logical shape and values never
    /// change, only where elements live. Returns this tensor when it already has the target layout, otherwise a new tensor owned by
    /// the caller. NCHW and NHWC results are ordinary strided tensors (NHWC is the logical shape with channels-last strides) and
    /// remain differentiable; blocked results are kernel-internal and cannot require gradients. Padding channels of a blocked
    /// tensor are zero.
    /// </summary>
    /// <exception cref="TensorShapeException">The tensor is not rank 4.</exception>
    /// <exception cref="InvalidOperationException">A blocked conversion was asked of a tensor that requires gradients.</exception>
    public Tensor ToLayout(TensorLayout target, BufferPool? pool = null)
    {
        ThrowIfEmpty();
        if (Rank != 4)
        {
            throw new TensorShapeException($"Layouts apply to rank-4 (N,C,H,W) tensors, not {_shape}.");
        }

        TensorLayout cur = Layout;
        if (cur == target)
        {
            return this;
        }

        if ((_block > 0 || target.Kind == LayoutKind.NchwC) && GradMode.ShouldRecord(this))
        {
            throw new InvalidOperationException("Blocked layouts are kernel-internal and do not support gradients.");
        }

        if (_block > 0)
        {
            Tensor plain = Unpack(pool);
            if (target.Kind == LayoutKind.Nchw)
            {
                return plain;
            }

            using (plain)
            {
                return plain.ToLayout(target, pool);
            }
        }

        switch (target.Kind)
        {
            case LayoutKind.Nchw:
                return Contiguous(pool);
            case LayoutKind.Nhwc:
            {
                int n = _shape.Unchecked(0), c = _shape.Unchecked(1), h = _shape.Unchecked(2), w = _shape.Unchecked(3);
                Tensor storage = Create(_dtype, Shape.Create(n, h, w, c), pool);
                bool track = GradMode.ShouldRecord(this);
                Tensor view;
                using (GradMode.NoGrad())
                {
                    view = storage.Permute(0, 3, 1, 2);
                }

                CopyStrided(this, view);
                return track ? Rerecord(view) : view;
            }

            case LayoutKind.NchwC:
                return Pack(target.BlockSize, pool);
            default:
                throw new ArgumentException("A target layout must be NCHW, NHWC or NCHWc.", nameof(target));
        }
    }

    private Tensor Pack(int block, BufferPool? pool)
    {
        int n = _shape.Unchecked(0), c = _shape.Unchecked(1), h = _shape.Unchecked(2), w = _shape.Unchecked(3);
        int cb = (c + block - 1) / block;
        long count = (long)n * cb * h * w * block;
        if (count > int.MaxValue)
        {
            throw new TensorShapeException($"The blocked tensor would have {count} elements.");
        }

        long bytes = ByteCount(_dtype, (int)count);

        TensorBuffer b = (pool ?? BufferPool.Shared).Rent(bytes, zero: true);
        var result = new Tensor(b, b.Generation, 0, _shape, default, _dtype, block, null);
        switch (ElementSize)
        {
            case 1: PackTyped<byte>(this, result, block); break;
            case 2: PackTyped<ushort>(this, result, block); break;
            case 4: PackTyped<uint>(this, result, block); break;
            default: PackTyped<ulong>(this, result, block); break;
        }

        return result;
    }

    // Storage order [n][c / B][h][w][c % B]; the padding channels of the last block stay zero from the rent.
    private static void PackTyped<T>(in Tensor src, in Tensor dst, int block) where T : unmanaged
    {
        ReadOnlySpan<T> s = src.Raw<T>();
        Span<T> d = dst.Raw<T>();
        Shape sh = src._shape;
        int n = sh.Unchecked(0), c = sh.Unchecked(1), h = sh.Unchecked(2), w = sh.Unchecked(3);
        int sn = src._strides.Unchecked(0), sc = src._strides.Unchecked(1), sy = src._strides.Unchecked(2), sx = src._strides.Unchecked(3);
        int o = 0;
        for (int in_ = 0; in_ < n; in_++)
        {
            for (int cb = 0; cb * block < c; cb++)
            {
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        for (int i = 0; i < block; i++, o++)
                        {
                            int ch = (cb * block) + i;
                            if (ch < c)
                            {
                                d[o] = s[src._offset + (in_ * sn) + (ch * sc) + (y * sy) + (x * sx)];
                            }
                        }
                    }
                }
            }
        }
    }

    private Tensor Unpack(BufferPool? pool)
    {
        Tensor dst = Create(_dtype, _shape, pool);
        switch (ElementSize)
        {
            case 1: UnpackTyped<byte>(this, dst); break;
            case 2: UnpackTyped<ushort>(this, dst); break;
            case 4: UnpackTyped<uint>(this, dst); break;
            default: UnpackTyped<ulong>(this, dst); break;
        }

        return dst;
    }

    private static void UnpackTyped<T>(in Tensor src, in Tensor dst) where T : unmanaged
    {
        ReadOnlySpan<T> s = src.Raw<T>();
        Span<T> d = dst.Raw<T>();
        int block = src._block;
        int n = src._shape.Unchecked(0), c = src._shape.Unchecked(1), h = src._shape.Unchecked(2), w = src._shape.Unchecked(3);
        int cb = (c + block - 1) / block;
        int o = 0;
        for (int in_ = 0; in_ < n; in_++)
        {
            for (int ch = 0; ch < c; ch++)
            {
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++, o++)
                    {
                        d[o] = s[(((((in_ * cb) + (ch / block)) * h + y) * w) + x) * block + (ch % block)];
                    }
                }
            }
        }
    }
}
