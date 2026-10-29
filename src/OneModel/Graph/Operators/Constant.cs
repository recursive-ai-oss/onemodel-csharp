using System;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

// The pattern for operators whose output is fixed by attributes (Constant, and in time ConstantOfShape with constant inputs): the
// factory builds the tensor once, Infer hands it to the plan as a constant, and the node never runs.

/// <summary>Constant: produces the tensor held in one attribute (value, value_float, value_floats, value_int, value_ints).</summary>
internal sealed class ConstantOp : OpKernel
{
    public static readonly OpDef Def = new("Constant", 1, int.MaxValue, static n => new ConstantOp(n));

    private readonly Tensor _value;

    private ConstantOp(NodeInfo node)
    {
        IrAttr? a = node.Find("value");
        if (a is not null)
        {
            IrTensorOrFail(node, a);
            _value = a.Tensor.ToTensor();
            return;
        }

        if ((a = node.Find("value_float")) is not null)
        {
            _value = Tensor.Scalar(a.Real);
            return;
        }

        if ((a = node.Find("value_int")) is not null)
        {
            _value = Tensor.Scalar(a.Whole);
            return;
        }

        if ((a = node.Find("value_floats")) is not null)
        {
            _value = Tensor.FromSpan<float>(a.Reals.AsSpan(), Shape.Create(a.Reals.Length));
            return;
        }

        if ((a = node.Find("value_ints")) is not null)
        {
            _value = Tensor.FromSpan<long>(a.Wholes.AsSpan(), Shape.Create(a.Wholes.Length));
            return;
        }

        if (node.Find("sparse_value") is not null)
        {
            throw node.Unsupported("sparse_value is not supported yet");
        }

        if (node.Find("value_string") is not null || node.Find("value_strings") is not null)
        {
            throw node.Unsupported("string constants are not supported");
        }

        throw node.Error("has no value attribute");
    }

    private static void IrTensorOrFail(NodeInfo node, IrAttr a)
    {
        if (a.Type != IrAttributeType.Tensor)
        {
            throw node.Error($"attribute '{a.Name}' is {a.Type}, expected a tensor");
        }

        if (!a.Tensor.TryGetDType(out _))
        {
            throw node.Unsupported($"element type {DTypes.OnnxName(a.Tensor.DataType)} is not supported");
        }
    }

    public override bool SupportsExact => true;

    public override void Infer(InferContext ctx) => ctx.SetConstantOutput(0, _value, planOwns: false);

    public override void Run(in OpRun run) => throw new InvalidOperationException("Constant is evaluated when the plan is made; it never runs.");

    public override void Dispose()
    {
        if (!_value.IsEmpty)
        {
            _value.Dispose();
        }

        base.Dispose();
    }
}
