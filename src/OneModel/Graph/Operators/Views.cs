using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

// The pattern for operators that only reinterpret their input (Reshape, Identity, Squeeze, Unsqueeze, Flatten, ...): Infer states the
// output shape, ViewOfInput names the source, and the session builds the view itself, so there is no Run and no copy.

/// <summary>Reshape: the same elements in the shape given by the second input. 0 copies the input's extent (unless allowzero), -1 is inferred.</summary>
internal sealed class ReshapeOp : OpKernel
{
    // Opset 5 made the target shape an input; 14 added allowzero.
    public static readonly OpDef Def = new("Reshape", 5, int.MaxValue, static n => new ReshapeOp(n));

    private readonly bool _allowZero;

    private ReshapeOp(NodeInfo node)
    {
        _allowZero = node.GetInt("allowzero", 0) != 0;
    }

    public override bool SupportsExact => true;

    public override int ViewOfInput => 0;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            throw ctx.Fail("expects data and shape inputs");
        }

        if (ctx.InputType(1) != DType.I64)
        {
            throw ctx.Fail($"the shape input must be int64, not {ctx.InputType(1)}");
        }

        if (ctx.InputShape(1).Rank != 1)
        {
            throw ctx.Fail($"the shape input must be 1-D, not {ctx.InputShape(1)}");
        }

        // The target shape is a value, not a type: it is known now if it is a constant, else when the run reaches this node.
        if (!ctx.TryGetInputValue(1, out Tensor target))
        {
            ctx.RequireRunTimeValues();
            return;
        }

        Shape inShape = ctx.InputShape(0);
        ReadOnlySpan<long> spec = target.Data<long>();
        if (spec.Length > Shape.MaxRank)
        {
            throw ctx.Fail($"the target has {spec.Length} dimensions; the limit is {Shape.MaxRank}");
        }

        Span<long> dims = stackalloc long[Shape.MaxRank];
        int unknown = -1;
        long known = 1;
        for (int i = 0; i < spec.Length; i++)
        {
            long v = spec[i];
            if (v == 0 && !_allowZero)
            {
                if (i >= inShape.Rank)
                {
                    throw ctx.Fail($"target dimension {i} is 0 (copy the input's extent) but the input {inShape} has only {inShape.Rank} dimensions");
                }

                v = inShape[i];
            }
            else if (v == -1)
            {
                if (unknown >= 0)
                {
                    throw ctx.Fail("at most one target dimension may be -1");
                }

                unknown = i;
            }
            else if (v < 0)
            {
                throw ctx.Fail($"target dimension {i} is {v}");
            }

            if (v > int.MaxValue)
            {
                throw ctx.Fail($"target dimension {i} is {v}, larger than 2^31-1");
            }

            dims[i] = v;
            if (v != -1)
            {
                // A product past 2^31 can never equal an element count, so it stops growing (and a later 0 still zeroes it).
                known = v == 0 ? 0 : (known > (1L << 31) ? known : known * v);
            }
        }

        long total = inShape.Length;
        if (unknown >= 0)
        {
            if (known == 0 || total % known != 0)
            {
                throw ctx.Fail($"the input {inShape} ({total} elements) cannot be reshaped to {FormatSpec(spec)}");
            }

            dims[unknown] = total / known;
        }
        else if (known != total)
        {
            throw ctx.Fail($"the input {inShape} ({total} elements) cannot be reshaped to {FormatSpec(spec)}");
        }

        ctx.SetOutput(0, ctx.InputType(0), Shape.Create((ReadOnlySpan<long>)dims.Slice(0, spec.Length)));
    }

    public override void Run(in OpRun run) => throw new InvalidOperationException("Reshape is a view; the session builds it.");

    private static string FormatSpec(ReadOnlySpan<long> spec)
    {
        var parts = new string[spec.Length];
        for (int i = 0; i < parts.Length; i++)
        {
            parts[i] = spec[i].ToString(System.Globalization.CultureInfo.InvariantCulture);
        }

        return "[" + string.Join(", ", parts) + "]";
    }
}

/// <summary>Identity: the output is the input.</summary>
internal sealed class IdentityOp : OpKernel
{
    public static readonly OpDef Def = new("Identity", 1, int.MaxValue, static _ => new IdentityOp());

    public override bool SupportsExact => true;

    public override int ViewOfInput => 0;

    public override void Infer(InferContext ctx)
    {
        if (!ctx.HasInput(0))
        {
            throw ctx.Fail("expects one input");
        }

        ctx.SetOutput(0, ctx.InputType(0), ctx.InputShape(0));
    }

    public override void Run(in OpRun run) => throw new InvalidOperationException("Identity is a view; the session builds it.");
}
