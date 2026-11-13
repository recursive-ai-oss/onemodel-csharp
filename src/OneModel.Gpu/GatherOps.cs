using System;
using System.Collections.Generic;
using RecursiveAI.OneGpu;
using RecursiveAI.OneGpu.Kernels;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu;

/// <summary>
/// ONNX <c>Gather</c> on the device: the embedding lookups at the head of an encoder. The data is a float32, float16 or uint8 tensor (usually a
/// weight), the indices are int32 or int64. An index outside its axis fails the run with the same <see cref="SessionErrorKind.InvalidInput"/>
/// error the CPU raises: indices that come from the host are checked in safe code before the upload (naming the value), and indices the
/// device made are recorded by the kernel in a status word that is read after the submission.
/// </summary>
internal static class GatherOps
{
    public static IEnumerable<GpuOpEntry> Entries
    {
        get
        {
            yield return new GpuOpEntry("Gather", true, _ => null, Bind);
        }
    }

    private static string? Bind(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            return "Gather needs data and indices";
        }

        DeviceElement element;
        if (ctx.InputType(0) == DType.U8)
        {
            // An 8-bit table (a quantized embedding): the rows are copied as bytes and dequantized by the next operator.
            element = DeviceElement.U8;
        }
        else if (ctx.InputAnyElement(0, out element) is { } why)
        {
            return why;
        }

        DType it = ctx.InputType(1);
        if (it is not (DType.I32 or DType.I64))
        {
            return $"the indices are {it}, and the GPU kernel takes int32 and int64";
        }

        Shape sd = ctx.InputShape(0), si = ctx.InputShape(1);
        if (sd.Rank == 0 || sd.Length == 0 || si.Length == 0)
        {
            return "a scalar or empty operand";
        }

        long axisAttr = ctx.Info.GetInt("axis", 0);
        int axis = (int)(axisAttr < 0 ? axisAttr + sd.Rank : axisAttr);
        if (axis < 0 || axis >= sd.Rank)
        {
            return $"axis {axisAttr} is outside the data's rank {sd.Rank}";
        }

        long inner = 1;
        for (int i = axis + 1; i < sd.Rank; i++)
        {
            inner *= sd[i];
        }

        if (element == DeviceElement.F16 && (inner & 1) != 0)
        {
            return "float16 rows of an odd length are not gathered on the device (they are read in pairs)";
        }

        if (ctx.OutputType(0) != ctx.InputType(0))
        {
            return "the planned output does not match the data";
        }

        // A table above one binding is held in pieces (SplitOps); the result must still fit one.
        bool oversized = ctx.FitsBinding(sd.Length, element) is not null;
        if (ctx.FitsBinding(oversized ? ctx.OutputShape(0).Length : Math.Max(sd.Length, ctx.OutputShape(0).Length), element) is { } big)
        {
            return big;
        }

        DeviceElement indexElement = it == DType.I64 ? DeviceElement.I64 : DeviceElement.I32;
        if (ctx.InputMadeOnDevice(1) is { } made && made != indexElement)
        {
            return "the indices were made on the device as another type";
        }

        int axisLength = sd[axis];
        int hostIndices = -1;
        if (ctx.InputIsConst(1))
        {
            // A constant that is out of range is left to the CPU, which raises the error with the value.
            Tensor c = ctx.InputConst(1);
            if (!InRange(c, axisLength))
            {
                return "a constant index is out of range, which the CPU reports";
            }
        }
        else if (ctx.InputMadeOnDevice(1) is null)
        {
            hostIndices = ctx.InputId(1);
        }

        if (oversized)
        {
            return SplitOps.BindGather(ctx, element, axis, GpuClaimContext.Dims(sd), si.Length, it == DType.I64, hostIndices, out binding);
        }

        binding = new GatherBinding(ctx, element, it == DType.I64, GpuClaimContext.Dims(sd), axis, si.Length, hostIndices, axisLength);
        return null;
    }

    private static bool InRange(in Tensor t, int axisLength)
    {
        Tensor src = t.IsContiguous ? t : t.Contiguous();
        try
        {
            if (src.DType == DType.I64)
            {
                foreach (long v in src.Data<long>())
                {
                    if (v < -axisLength || v >= axisLength)
                    {
                        return false;
                    }
                }
            }
            else
            {
                foreach (int v in src.Data<int>())
                {
                    if (v < -axisLength || v >= axisLength)
                    {
                        return false;
                    }
                }
            }

            return true;
        }
        finally
        {
            if (!ReferenceEquals(src.Buffer, t.Buffer))
            {
                src.Dispose();
            }
        }
    }

    private sealed class GatherBinding : GpuBinding
    {
        private readonly GatherKernels _kernels;
        private readonly bool _separate;
        private readonly DeviceElement _element;
        private readonly bool _wide;
        private readonly int[] _dataShape;
        private readonly int _axis;
        private readonly int _count;
        private readonly int _hostIndices;
        private readonly int _axisLength;
        private readonly Operand _data, _indices;
        private readonly int _outId;
        private readonly long _outCount;

        private static long Product(int[] shape)
        {
            long n = 1;
            foreach (int d in shape)
            {
                n *= d;
            }

            return n;
        }

        public GatherBinding(GpuClaimContext ctx, DeviceElement element, bool wide, int[] dataShape, int axis, int count, int hostIndices, int axisLength)
        {
            _outCount = ctx.OutputShape(0).Length;
            _kernels = ctx.Backend.Gather;
            _separate = ctx.Backend.SeparateReadWriteSlabs;
            _element = element;
            _wide = wide;
            _dataShape = dataShape;
            _axis = axis;
            _count = count;
            _hostIndices = hostIndices;
            _axisLength = axisLength;
            _data = Operand.Of(ctx, 0);
            _indices = Operand.Of(ctx, 1);
            _outId = ctx.OutputId(0);
            ctx.Backend.IndexStatusNeeded();
        }

        public override bool UsesIndexStatus => true;

        public override DeviceElement OutputElement(int output) => _element;

        // The device's gather kernel moves 16, 32 or 64-bit elements. A boolean tensor is gathered as int32 words: widened into scratch,
        // gathered, narrowed back. Both conversions are exact for 0 and 1.
        public override long ScratchI32Elements => _element != DeviceElement.Bool ? 0 : _separate ? Product(_dataShape) : Product(_dataShape) + _outCount;

        // Where the device refuses one buffer bound for reading and for writing, the gathered words go to a scratch of their own.
        public override long ScratchI32bElements => _element == DeviceElement.Bool && _separate ? _outCount : 0;

        public override void Upload(GpuRunContext ctx, Tensor[] values)
        {
            if (_hostIndices < 0)
            {
                return;
            }

            Tensor t = values[_hostIndices];
            Tensor src = t.IsContiguous ? t : t.Contiguous();
            try
            {
                if (src.DType == DType.I64)
                {
                    Check(src.Data<long>());
                }
                else
                {
                    Check(src.Data<int>());
                }
            }
            finally
            {
                if (!ReferenceEquals(src.Buffer, t.Buffer))
                {
                    src.Dispose();
                }
            }
        }

        // The CPU's rule and message: negative indices count from the end, anything else outside the axis is an input error.
        private void Check<TI>(ReadOnlySpan<TI> indices) where TI : unmanaged, System.Numerics.IBinaryInteger<TI>
        {
            foreach (TI raw in indices)
            {
                long k = long.CreateTruncating(raw);
                if (k < -_axisLength || k >= _axisLength)
                {
                    throw new SessionException(SessionErrorKind.InvalidInput, $"Gather index {k} is out of range for an extent of {_axisLength}.");
                }
            }
        }

        public override void Encode(GpuRunContext ctx)
        {
            switch (_element)
            {
                case DeviceElement.F32:
                    Encode<float>(ctx);
                    break;
                case DeviceElement.F16:
                    Encode<Half>(ctx);
                    break;
                case DeviceElement.I32:
                    Encode<int>(ctx);
                    break;
                case DeviceElement.I64:
                    Encode<long>(ctx);
                    break;
                case DeviceElement.U8:
                    Encode<byte>(ctx);
                    break;
                default:
                    EncodeBool(ctx);
                    break;
            }
        }

        private void EncodeBool(GpuRunContext ctx)
        {
            long dataCount = Product(_dataShape);
            GpuBuffer<byte> data = _data.Get<byte>(ctx, out int offData);
            GpuBuffer<int> scratch = ctx.ScratchI32(this, out int offScratch);
            GpuBuffer<int> gathered = scratch;
            int offGathered = checked((int)(offScratch + dataCount));
            if (_separate)
            {
                gathered = ctx.ScratchI32b(this, out offGathered);
            }

            ElementwiseKernels cast = ctx.Backend.Elementwise;
            cast.EncodeCast(ctx.Encoder, data, offData, scratch, offScratch, (int)dataCount);
            GatherInto(ctx, scratch, offScratch, gathered, offGathered);
            GpuBuffer<byte> y = ctx.Buffer<byte>(_outId, out int offY);
            cast.EncodeCast(ctx.Encoder, gathered, offGathered, y, offY, (int)_outCount);
        }

        private void Encode<T>(GpuRunContext ctx) where T : unmanaged
        {
            GpuBuffer<T> data = _data.Get<T>(ctx, out int offData);
            GpuBuffer<T> y = ctx.Buffer<T>(_outId, out int offY);
            GatherInto(ctx, data, offData, y, offY);
        }

        private void GatherInto<T>(GpuRunContext ctx, GpuBuffer<T> data, int offData, GpuBuffer<T> y, int offY) where T : unmanaged
        {
            if (_wide)
            {
                GpuBuffer<long> ix = _indices.Get<long>(ctx, out int offIx);
                _kernels.EncodeGather(ctx.Encoder, data, offData, _dataShape, _axis, ix, offIx, _count, y, offY, ctx.Backend.IndexStatus);
            }
            else
            {
                GpuBuffer<int> ix = _indices.Get<int>(ctx, out int offIx);
                _kernels.EncodeGather(ctx.Encoder, data, offData, _dataShape, _axis, ix, offIx, _count, y, offY, ctx.Backend.IndexStatus);
            }
        }
    }
}
