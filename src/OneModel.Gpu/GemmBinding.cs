using System;
using System.Collections.Generic;
using RecursiveAI.OneGpu;
using RecursiveAI.OneGpu.Kernels;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu;

/// <summary>
/// A matrix product on the device: <c>C = alpha * op(A) * op(B) + beta * C</c>. MatMul and Gemm both bind to it; a bias (Gemm's C) is
/// broadcast on the host to the output's shape and copied into the output before the kernel, which then accumulates onto it.
/// </summary>
internal sealed class GemmBinding : GpuBinding
{
    private readonly Gemm _gemm;
    private readonly DeviceElement _element;
    private readonly GemmProblem _problem;
    private readonly Operand _a;
    private readonly Operand _b;
    private int _outId;
    private GpuBuffer<float>? _biasBuffer;
    private GemmActivation _activation;
    private readonly DeviceConstant? _initFromConstant;
    private readonly int _initFromValue;
    private readonly int _cRows;
    private readonly int _cCols;
    private float[]? _scratch;

    internal GemmBinding(GpuClaimContext ctx, DeviceElement element, GemmProblem problem, DeviceConstant? cInit, int cValue, int cRows, int cCols)
    {
        _element = element;
        _problem = problem;
        _gemm = ctx.Backend.GetGemm(element);
        _a = Operand.Of(ctx, 0);
        _b = Operand.Of(ctx, 1);
        _outId = ctx.OutputId(0);
        _initFromConstant = cInit;
        _initFromValue = cValue;
        _cRows = cRows;
        _cCols = cCols;
        Flops = problem.Flops;
    }

    // The kernel accumulates in float32. A float16 result is rounded from a float32 scratch by one more pass.
    public override DeviceElement OutputElement(int output) => _element;

    public override long ScratchF32Elements => _element == DeviceElement.F16 ? (long)_problem.Batch * _problem.M * _problem.N : 0;

    public override bool InitializesOutput => _initFromConstant is not null || _initFromValue >= 0;

    public override long ExtraUploadBytes => _initFromValue >= 0 ? (long)_problem.M * _problem.N * 4 : 0;

    public override void Upload(GpuRunContext ctx, Tensor[] values)
    {
        if (_initFromValue < 0)
        {
            return;
        }

        int m = _problem.M, n = _problem.N;
        _scratch ??= new float[checked(m * n)];
        Tensor host = values[_initFromValue];
        Tensor src = host.IsContiguous ? host : host.Contiguous();
        try
        {
            MatMulOps.ExpandInto(src, _cRows, _cCols, m, n, _scratch);
        }
        finally
        {
            if (!ReferenceEquals(src.Buffer, host.Buffer))
            {
                src.Dispose();
            }
        }

        GpuBuffer<float> c = Accumulator(ctx, out int offset);
        c.Write(_scratch, offset);
    }

    /// <summary>Whether the product's own dispatch can also add a bias and apply an activation: not when it starts from a C that is already in the output.</summary>
    public bool CanTakeEpilogue => !InitializesOutput && _biasBuffer is null && _activation == GemmActivation.None;

    /// <summary>Columns of the product's result (the length a bias vector must have).</summary>
    public int Columns => _problem.N;

    public DeviceElement Element => _element;

    /// <summary>
    /// Makes the dispatch add <paramref name="bias"/> (float32, one value per column; null for none) and apply <paramref name="activation"/>
    /// to each result element, and write the result as value <paramref name="resultId"/> instead of the product's own output.
    /// </summary>
    public void TakeEpilogue(GpuBuffer<float>? bias, GemmActivation activation, int resultId, int[] inputIds)
    {
        _biasBuffer = bias ?? _biasBuffer;
        if (activation != GemmActivation.None)
        {
            _activation = activation;
        }

        _outId = resultId;
        InputIds = inputIds;
        OutputIds = [resultId];
    }

    private GpuBuffer<float> Accumulator(GpuRunContext ctx, out int offset) =>
        _element == DeviceElement.F32 ? ctx.F32(_outId, out offset) : ctx.Scratch(this, out offset);

    public override void Encode(GpuRunContext ctx)
    {
        GemmProblem p = _problem;
        GpuBuffer<float> c = Accumulator(ctx, out int offC);
        GemmEpilogue epilogue = _biasBuffer is null && _activation == GemmActivation.None ? default : new GemmEpilogue(_biasBuffer, 0, _activation);
        p.OffsetC = offC;
        if (_initFromConstant is { F32: { } init })
        {
            ctx.Encoder.CopyBuffer(init.Buffer, 0, c.Buffer, (ulong)offC * 4, (ulong)init.Length * 4);
        }

        if (_element == DeviceElement.F32)
        {
            GpuBuffer<float> a = _a.F32(ctx, out int offA);
            GpuBuffer<float> b = _b.F32(ctx, out int offB);
            p.OffsetA = offA;
            p.OffsetB = offB;
            _gemm.Encode(ctx.Encoder, p, a, b, c, epilogue);
        }
        else
        {
            GpuBuffer<Half> a = _a.F16(ctx, out int offA);
            GpuBuffer<Half> b = _b.F16(ctx, out int offB);
            p.OffsetA = offA;
            p.OffsetB = offB;
            _gemm.Encode(ctx.Encoder, p, a, b, c, epilogue);
            GpuBuffer<Half> y = ctx.F16(_outId, out int offY);
            ctx.Backend.Elementwise.EncodeCast(ctx.Encoder, c, offC, y, offY, checked(_problem.Batch * _problem.M * _problem.N));
        }
    }
}

/// <summary>What a binding sees while a region runs: the encoder being recorded and the arena slots of this plan.</summary>
internal sealed class GpuRunContext
{
    private readonly DeviceArena _arena;
    private Dictionary<int, ValueSlot> _slots = [];

    public GpuRunContext(GpuBackend backend)
    {
        Backend = backend;
        _arena = backend.Arena;
    }

    public GpuBackend Backend { get; }

    public GpuCommandEncoder Encoder { get; set; } = null!;

    public void Use(Dictionary<int, ValueSlot> slots) => _slots = slots;

    public GpuBuffer<float> F32(int valueId, out int offset)
    {
        ValueSlot s = _slots[valueId];
        offset = checked((int)s.Offset);
        return _arena.F32(s.Slab);
    }

    public GpuBuffer<Half> F16(int valueId, out int offset)
    {
        ValueSlot s = _slots[valueId];
        offset = checked((int)s.Offset);
        return _arena.F16(s.Slab);
    }

    public GpuBuffer<T> Buffer<T>(int valueId, out int offset) where T : unmanaged
    {
        ValueSlot s = _slots[valueId];
        offset = checked((int)s.Offset);
        return _arena.Get<T>(s.Element, s.Slab);
    }

    /// <summary>The float32 scratch the program placed for a binding.</summary>
    public GpuBuffer<float> Scratch(GpuBinding binding, out int offset)
    {
        ValueSlot s = binding.ScratchSlot;
        offset = checked((int)s.Offset);
        return _arena.F32(s.Slab);
    }

    /// <summary>The second float32 scratch the program placed for a binding.</summary>
    public GpuBuffer<float> ScratchF32b(GpuBinding binding, out int offset)
    {
        ValueSlot s = binding.ScratchF32bSlot;
        offset = checked((int)s.Offset);
        return _arena.F32(s.Slab);
    }

    /// <summary>The float16 scratch the program placed for a binding.</summary>
    public GpuBuffer<Half> ScratchF16(GpuBinding binding, out int offset)
    {
        ValueSlot s = binding.ScratchF16Slot;
        offset = checked((int)s.Offset);
        return _arena.F16(s.Slab);
    }

    /// <summary>The int32 scratch the program placed for a binding.</summary>
    public GpuBuffer<int> ScratchI32(GpuBinding binding, out int offset)
    {
        ValueSlot s = binding.ScratchI32Slot;
        offset = checked((int)s.Offset);
        return _arena.Get<int>(DeviceElement.I32, s.Slab);
    }

    /// <summary>The second int32 scratch the program placed for a binding.</summary>
    public GpuBuffer<int> ScratchI32b(GpuBinding binding, out int offset)
    {
        ValueSlot s = binding.ScratchI32bSlot;
        offset = checked((int)s.Offset);
        return _arena.Get<int>(DeviceElement.I32, s.Slab);
    }

    /// <summary>The bool scratch the program placed for a binding.</summary>
    public GpuBuffer<byte> ScratchBool(GpuBinding binding, out int offset)
    {
        ValueSlot s = binding.ScratchBoolSlot;
        offset = checked((int)s.Offset);
        return _arena.Get<byte>(DeviceElement.Bool, s.Slab);
    }

    /// <summary>The signed 8-bit scratch the program placed for a binding.</summary>
    public GpuBuffer<sbyte> ScratchI8(GpuBinding binding, out int offset)
    {
        ValueSlot s = binding.ScratchI8Slot;
        offset = checked((int)s.Offset);
        return _arena.Get<sbyte>(DeviceElement.I8, s.Slab);
    }

    public ValueSlot Slot(int valueId) => _slots[valueId];

    public DeviceArena Arena => _arena;
}
