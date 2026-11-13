using System;
using RecursiveAI.OneGpu;

namespace RecursiveAI.OneModel.Gpu;

/// <summary>An operand of a kernel: a weight (or other constant) on the device, or a value in the arena (made by an earlier step or uploaded as a region input).</summary>
internal readonly struct Operand
{
    private readonly int _valueId;
    private readonly DeviceConstant? _constant;

    private Operand(int valueId, DeviceConstant? constant)
    {
        _valueId = valueId;
        _constant = constant;
    }

    public static Operand Of(GpuClaimContext ctx, int input) =>
        ctx.InputIsConst(input) ? new Operand(-1, ctx.ConstantInput(input)) : new Operand(ctx.InputId(input), null);

    /// <summary>The weight this operand is, or null for a value in the arena.</summary>
    public DeviceConstant? Constant => _constant;

    public GpuBuffer<float> F32(GpuRunContext ctx, out int offset) => Get<float>(ctx, out offset);

    public GpuBuffer<Half> F16(GpuRunContext ctx, out int offset) => Get<Half>(ctx, out offset);

    /// <summary>The buffer as the element type the caller instantiates with (its host representation: float, Half, int, long, byte or sbyte).</summary>
    public GpuBuffer<T> Get<T>(GpuRunContext ctx, out int offset) where T : unmanaged
    {
        if (_constant is not null)
        {
            offset = 0;
            return _constant.As<T>() ?? throw new InvalidOperationException("A constant was bound as the wrong element type.");
        }

        return ctx.Buffer<T>(_valueId, out offset);
    }
}
