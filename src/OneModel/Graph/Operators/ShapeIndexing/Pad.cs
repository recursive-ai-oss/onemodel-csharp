using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

/// <summary>
/// Pad: grows (or, with negative pads, crops) each axis by <c>pads</c> at both ends. Modes: constant (a value, default 0), reflect
/// (mirror without repeating the edge), edge (repeat the edge) and wrap (opset 19: periodic). Pads come from attributes before
/// opset 11 and from an input after; opset 18 adds an axes input that names the axes the pads apply to.
/// </summary>
internal sealed class PadOp : OpKernel
{
    public static readonly OpDef Def = new("Pad", 1, int.MaxValue, static n => new PadOp(n));

    private readonly bool _padsAreInput;
    private readonly long[] _padsAttribute;
    private readonly float _valueAttribute;
    private readonly int _mode; // 0 constant, 1 reflect, 2 edge, 3 wrap

    private PadOp(NodeInfo node)
    {
        _padsAreInput = node.Opset >= 11;
        _padsAttribute = [];
        if (!_padsAreInput)
        {
            string name = node.Opset == 1 ? "paddings" : "pads";
            if (!node.Has(name))
            {
                throw node.Error($"has no {name} attribute");
            }

            _padsAttribute = [.. node.GetInts(name)];
            _valueAttribute = node.GetFloat("value", 0f);
        }

        _mode = node.GetString("mode", "constant") switch
        {
            "constant" => 0,
            "reflect" => 1,
            "edge" => 2,
            "wrap" => 3,
            var other => throw node.Error($"unknown mode '{other}'"),
        };
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects data");
        }

        Shape s = ctx.InputShape(0);
        int rank = s.Rank;
        if (rank == 0)
        {
            throw ctx.Fail("cannot pad a scalar");
        }

        Span<long> before = stackalloc long[Shape.MaxRank];
        Span<long> after = stackalloc long[Shape.MaxRank];
        string? error;
        if (_padsAreInput)
        {
            OpUtil.RequireIntegers(ctx, 1, "pads");
            bool hasAxes = ctx.HasInput(3);
            if (hasAxes)
            {
                OpUtil.RequireIntegers(ctx, 3, "axes");
            }

            if (ctx.HasInput(2) && (ctx.InputType(2) != ctx.InputType(0) || ctx.InputShape(2).Length != 1))
            {
                throw ctx.Fail("constant_value must be one value of the data's element type");
            }

            bool known = ctx.TryGetInputValue(1, out Tensor pads);
            Tensor axes = default;
            if (hasAxes)
            {
                known &= ctx.TryGetInputValue(3, out axes);
            }

            if (!known)
            {
                ctx.RequireRunTimeValues();
                return;
            }

            error = ResolvePads(rank, IntVec.Of(pads), hasAxes ? IntVec.Of(axes) : default, hasAxes, before, after);
        }
        else
        {
            error = ResolvePads(rank, new IntVec(_padsAttribute), default, false, before, after);
        }

        if (error is not null)
        {
            throw ctx.Fail(error);
        }

        Span<long> dims = stackalloc long[Shape.MaxRank];
        for (int a = 0; a < rank; a++)
        {
            dims[a] = s[a] + before[a] + after[a];
            if (dims[a] < 0)
            {
                throw ctx.Fail($"axis {a}: extent {s[a]} with pads {before[a]} and {after[a]} is negative");
            }

            if (_mode != 0 && s[a] == 0 && dims[a] > 0)
            {
                throw ctx.Fail($"axis {a} is empty and cannot be padded by reflecting, repeating or wrapping");
            }
        }

        ctx.SetOutput(0, ctx.InputType(0), OpUtil.MakeGrownShape(ctx, dims.Slice(0, rank), ctx.InputType(0)));
    }

    // pads = [begin of each listed axis..., end of each listed axis...]; axes not listed get 0.
    private static string? ResolvePads(int rank, IntVec pads, IntVec axes, bool hasAxes, Span<long> before, Span<long> after)
    {
        before.Slice(0, rank).Clear();
        after.Slice(0, rank).Clear();
        int n = hasAxes ? axes.Length : rank;
        if (pads.Length != 2 * n)
        {
            return $"pads has {pads.Length} values; {2 * n} are needed for {n} axes";
        }

        Span<bool> seen = stackalloc bool[Shape.MaxRank];
        for (int i = 0; i < n; i++)
        {
            long axis = hasAxes ? axes[i] : i;
            if (axis < -rank || axis >= rank)
            {
                return $"axis {axis} is outside [{-rank}, {rank - 1}]";
            }

            int a = (int)(axis < 0 ? axis + rank : axis);
            if (seen[a])
            {
                return $"axis {a} is padded twice";
            }

            seen[a] = true;
            before[a] = pads[i];
            after[a] = pads[n + i];
            if (Math.Abs(before[a]) > int.MaxValue || Math.Abs(after[a]) > int.MaxValue)
            {
                return "a pad is larger than 2^31";
            }
        }

        return null;
    }

    public override void Run(in OpRun run)
    {
        Tensor x = run.Inputs[0], y = run.Outputs[0];
        int rank = x.Rank;
        Span<long> before = stackalloc long[Shape.MaxRank];
        Span<long> after = stackalloc long[Shape.MaxRank];
        string? error;
        if (_padsAreInput)
        {
            bool hasAxes = run.Inputs.Length > 3 && !run.Inputs[3].IsEmpty;
            error = ResolvePads(rank, IntVec.Of(run.Inputs[1]), hasAxes ? IntVec.Of(run.Inputs[3]) : default, hasAxes, before, after);
        }
        else
        {
            error = ResolvePads(rank, new IntVec(_padsAttribute), default, false, before, after);
        }

        if (error is not null)
        {
            throw new SessionException(SessionErrorKind.InvalidInput, "Pad: " + error);
        }

        if (y.Length == 0)
        {
            return;
        }

        if (_mode == 0)
        {
            FillConstant(run, y);
            CopyOverlap(x, y, before);
            return;
        }

        int total = 0;
        for (int a = 0; a < rank; a++)
        {
            total += y.Shape[a];
        }

        Span<int> maps = run.Scratch(DType.I32, Shape.Create(total)).MutableData<int>();
        for (int a = 0, at = 0; a < rank; at += y.Shape[a], a++)
        {
            Movement.PadMap(maps.Slice(at, y.Shape[a]), x.Shape[a], before[a], _mode);
        }

        Movement.PadMapped(x, y, maps);
    }

    private void FillConstant(in OpRun run, in Tensor y)
    {
        if (_padsAreInput)
        {
            if (run.Inputs.Length > 2 && !run.Inputs[2].IsEmpty)
            {
                Movement.FillWith(y, run.Inputs[2]);
            }
            else
            {
                y.RawMutableBytes().Clear();
            }

            return;
        }

        // Before opset 11 the value is a float attribute; the types are float16, float and double.
        switch (y.DType)
        {
            case DType.F32: y.MutableData<float>().Fill(_valueAttribute); break;
            case DType.F64: y.MutableData<double>().Fill(_valueAttribute); break;
            case DType.F16: y.MutableData<Half>().Fill((Half)_valueAttribute); break;
            default: y.RawMutableBytes().Clear(); break;
        }
    }

    // The part of the output that holds input elements is a strided copy: output index j of an axis reads input index j - before.
    private static void CopyOverlap(in Tensor x, in Tensor y, ReadOnlySpan<long> before)
    {
        int rank = x.Rank;
        Span<int> inStr = stackalloc int[Shape.MaxRank];
        Span<int> outStr = stackalloc int[Shape.MaxRank];
        Span<int> dims = stackalloc int[Shape.MaxRank];
        OpUtil.RowMajor(x.Shape.Dimensions, inStr);
        OpUtil.RowMajor(y.Shape.Dimensions, outStr);
        long srcBase = 0, dstBase = 0;
        for (int a = 0; a < rank; a++)
        {
            long lo = Math.Max(before[a], 0);
            long hi = Math.Min(before[a] + x.Shape[a], y.Shape[a]);
            if (hi <= lo)
            {
                return;
            }

            dims[a] = (int)(hi - lo);
            srcBase += (lo - before[a]) * inStr[a];
            dstBase += lo * outStr[a];
        }

        Movement.Move(x, (int)srcBase, inStr.Slice(0, rank), y, (int)dstBase, outStr.Slice(0, rank), dims.Slice(0, rank));
    }
}
