using System;
using RecursiveAI.OneGpu;
using RecursiveAI.OneGpu.Kernels;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu;

/// <summary>
/// Weights larger than one buffer binding (the vocabulary tables of language and speech decoders: 51,865 by 768 float32 is 159 MB, above the
/// 128 MiB some devices bind) are held as several device buffers. The product splits along the output columns, so every output element is summed
/// in the order the single-buffer kernel uses; the lookup copies each requested row from the piece that holds it.
/// </summary>
internal static class SplitOps
{
    /// <summary>Whether a float32 constant matrix B may be larger than one binding because the product can be split.</summary>
    public static bool CanSplitB(GpuClaimContext ctx, DeviceElement element) =>
        element == DeviceElement.F32 && ctx.InputIsConst(1) && ctx.InputShape(1).Rank == 2;

    public static bool IsOversized(GpuClaimContext ctx, int input, DeviceElement element) =>
        (long)ctx.InputShape(input).Length * DeviceArena.ElementBytes(element) > ctx.MaxBindingBytes;

    /// <summary>
    /// Binds a matrix product whose B is larger than one binding. Returns the reason it cannot, or null with the binding set.
    /// <paramref name="problem"/> carries M, N, K, the transposes and alpha.
    /// </summary>
    public static string? BindProduct(GpuClaimContext ctx, GemmProblem problem, bool hasBias, out GpuBinding? binding)
    {
        binding = null;
        if (hasBias)
        {
            return "a weight larger than one buffer binding with a bias (the product splits only without one)";
        }

        Shape sb = ctx.InputShape(1);
        ConstantChunks chunks = ctx.SplitConstantInput(1, sb[0], sb[1], byColumns: !problem.TransposeB);
        binding = new SplitProductBinding(ctx, problem, chunks);
        return null;
    }

    /// <summary>A product with B in pieces: one call of the matrix kernel per piece, each writing its columns of the output.</summary>
    private sealed class SplitProductBinding : GpuBinding
    {
        private readonly Gemm _gemm;
        private readonly GemmProblem _problem;
        private readonly Operand _a;
        private readonly ConstantChunks _chunks;
        private readonly int _outId;

        public SplitProductBinding(GpuClaimContext ctx, GemmProblem problem, ConstantChunks chunks)
        {
            _gemm = ctx.Backend.GetGemm(DeviceElement.F32);
            _problem = problem;
            _a = Operand.Of(ctx, 0);
            _chunks = chunks;
            _outId = ctx.OutputId(0);
            Flops = problem.Flops;
        }

        public override DeviceElement OutputElement(int output) => DeviceElement.F32;

        public override void Encode(GpuRunContext ctx)
        {
            GpuBuffer<float> a = _a.F32(ctx, out int offA);
            GpuBuffer<float> c = ctx.F32(_outId, out int offC);
            int n = _problem.N;
            for (int i = 0; i < _chunks.Pieces.Length; i++)
            {
                int start = i * _chunks.Per, width = Math.Min(_chunks.Per, n - start);
                GemmProblem p = _problem;
                p.N = width;
                p.OffsetA = offA;
                p.OffsetB = 0;
                p.OffsetC = checked(offC + start);
                p.Ldc = n;
                p.Ldb = p.TransposeB ? p.K : width;
                _gemm.Encode(ctx.Encoder, p, a, _chunks.Pieces[i].F32!, c);
            }
        }
    }

    /// <summary>The most rows of an oversized table one lookup copies; beyond it the operator stays on the CPU.</summary>
    private const int MaxCopiedRows = 1 << 16;

    /// <summary>Whether <c>Gather</c> can take a data constant above one binding: rows (axis 0) of float32 or float16, ids from the host or constant.</summary>
    public static string? BindGather(GpuClaimContext ctx, DeviceElement element, int axis, int[] dataShape, int indexCount, bool wide, int hostIndices, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.InputIsConst(0) || axis != 0 || element is not (DeviceElement.F32 or DeviceElement.F16))
        {
            return $"a table larger than one buffer binding on this device ({ctx.MaxBindingBytes} bytes) is gathered only as float32 or float16 rows of a weight";
        }

        if (indexCount > MaxCopiedRows)
        {
            return $"{indexCount} lookups into a table larger than one buffer binding is more than the device copies row by row";
        }

        if (hostIndices < 0 && !ctx.InputIsConst(1))
        {
            return "indices made on the device cannot be looked up in a table larger than one buffer binding";
        }

        int rows = dataShape[0], width = 1;
        for (int i = 1; i < dataShape.Length; i++)
        {
            width = checked(width * dataShape[i]);
        }

        ConstantChunks chunks = ctx.SplitConstantInput(0, rows, width, byColumns: false);
        binding = new SplitGatherBinding(ctx, element, chunks, rows, width, indexCount, wide, hostIndices);
        return null;
    }

    private sealed class SplitGatherBinding : GpuBinding
    {
        private readonly DeviceElement _element;
        private readonly ConstantChunks _chunks;
        private readonly int _rows, _width, _count, _hostIndices;
        private readonly bool _wide;
        private readonly int _outId;
        private readonly int[] _row;

        public SplitGatherBinding(GpuClaimContext ctx, DeviceElement element, ConstantChunks chunks, int rows, int width, int count, bool wide, int hostIndices)
        {
            _element = element;
            _chunks = chunks;
            _rows = rows;
            _width = width;
            _count = count;
            _wide = wide;
            _hostIndices = hostIndices;
            _outId = ctx.OutputId(0);
            _row = new int[count];
            if (hostIndices < 0)
            {
                Resolve(ctx.InputConst(1));
            }
        }

        public override DeviceElement OutputElement(int output) => _element;

        public override void Upload(GpuRunContext ctx, Tensor[] values)
        {
            if (_hostIndices >= 0)
            {
                Resolve(values[_hostIndices]);
            }
        }

        // The CPU's rule and message: negative indices count from the end, anything else outside the axis is an input error.
        private void Resolve(in Tensor t)
        {
            Tensor src = t.IsContiguous ? t : t.Contiguous();
            try
            {
                if (_wide)
                {
                    Resolve(src.Data<long>());
                }
                else
                {
                    Resolve(src.Data<int>());
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

        private void Resolve<TI>(ReadOnlySpan<TI> indices) where TI : unmanaged, System.Numerics.IBinaryInteger<TI>
        {
            for (int i = 0; i < indices.Length; i++)
            {
                long k = long.CreateTruncating(indices[i]);
                if (k < -_rows || k >= _rows)
                {
                    throw new SessionException(SessionErrorKind.InvalidInput, $"Gather index {k} is out of range for an extent of {_rows}.");
                }

                _row[i] = (int)(k < 0 ? k + _rows : k);
            }
        }

        public override void Encode(GpuRunContext ctx)
        {
            int size = _element == DeviceElement.F32 ? 4 : 2;
            GpuBuffer y = _element == DeviceElement.F32 ? ctx.F32(_outId, out int offY).Buffer : ctx.F16(_outId, out offY).Buffer;
            ulong rowBytes = (ulong)_width * (ulong)size;
            for (int i = 0; i < _count; i++)
            {
                int row = _row[i], piece = row / _chunks.Per;
                DeviceConstant c = _chunks.Pieces[piece];
                GpuBuffer src = _element == DeviceElement.F32 ? c.F32!.Buffer : c.F16!.Buffer;
                ctx.Encoder.CopyBuffer(src, (ulong)(row - piece * _chunks.Per) * rowBytes, y, ((ulong)offY + (ulong)i * (ulong)_width) * (ulong)size, rowBytes);
            }
        }
    }
}
