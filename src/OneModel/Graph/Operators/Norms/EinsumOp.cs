using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Graph.Operators;

/// <summary>
/// Einsum (opset 12): the Einstein summation of any number of inputs, written with letters, optional "..." and an optional "-&gt;"
/// output. Without "-&gt;" the output holds the letters that occur once, in alphabetical order, after the ellipsis dimensions. A
/// letter used twice in one input takes the diagonal. Every output element is the sum, over the contracted letters in
/// row-major order, of the product of the input elements, accumulated in the element type from zero; so it is deterministic and
/// uses no fused operations. The loop is a plain odometer: it favours clarity over speed, and models that need fast contractions
/// write MatMul.
/// </summary>
internal sealed class EinsumOp : OpKernel
{
    public static readonly OpDef Def = new("Einsum", 12, int.MaxValue, static n => new EinsumOp(n));

    private const int EllipsisBase = 128; // label ids from here up are ellipsis dimensions, counted from the right
    private const int MaxLabels = 64;

    private readonly string[] _terms;
    private readonly string? _output;

    private EinsumOp(NodeInfo n)
    {
        string eq = n.GetString("equation", "").Replace(" ", "", StringComparison.Ordinal);
        if (eq.Length == 0)
        {
            throw n.Error("attribute 'equation' is required");
        }

        int arrow = eq.IndexOf("->", StringComparison.Ordinal);
        string lhs = arrow < 0 ? eq : eq[..arrow];
        _output = arrow < 0 ? null : eq[(arrow + 2)..];
        _terms = lhs.Split(',');
        foreach (string t in _terms.Concat([_output ?? ""]))
        {
            int dots = 0;
            for (int i = 0; i < t.Length; i++)
            {
                if (t[i] == '.')
                {
                    dots++;
                }
                else if (!char.IsAsciiLetter(t[i]))
                {
                    throw n.Error($"the equation '{eq}' has an unexpected character '{t[i]}'");
                }
            }

            if (dots is not (0 or 3) || (dots == 3 && !t.Contains("...", StringComparison.Ordinal)))
            {
                throw n.Error($"the equation '{eq}' has a malformed ellipsis");
            }
        }
    }

    public override bool SupportsExact => true;

    // The label of every dimension of every input: letters are their character codes, ellipsis dimensions EllipsisBase + position
    // counted from the right (so "..." of different lengths line up on their last dimension).
    private int[][] LabelsOf(ReadOnlySpan<Shape> inputs, out int maxEllipsis, InferContext? ctx)
    {
        var all = new int[inputs.Length][];
        maxEllipsis = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            string t = _terms[i];
            int letters = t.Count(char.IsAsciiLetter);
            bool dots = t.Contains("...", StringComparison.Ordinal);
            int ell = inputs[i].Rank - letters;
            if (ell < 0 || (!dots && ell != 0))
            {
                throw ctx?.Fail($"input {i} has rank {inputs[i].Rank} but its term '{t}' needs {letters}{(dots ? " or more" : "")}")
                    ?? new SessionException(SessionErrorKind.InvalidInput, "Einsum: input rank does not match the equation");
            }

            maxEllipsis = Math.Max(maxEllipsis, ell);
            var l = new int[inputs[i].Rank];
            int d = 0;
            for (int k = 0; k < t.Length; k++)
            {
                if (t[k] == '.')
                {
                    for (int e = 0; e < ell; e++)
                    {
                        l[d++] = EllipsisBase + ell - 1 - e;
                    }

                    k += 2;
                }
                else
                {
                    l[d++] = t[k];
                }
            }

            all[i] = l;
        }

        return all;
    }

    // The output labels, in order.
    private List<int> OutputLabels(int[][] labels, int maxEllipsis, InferContext? ctx)
    {
        var result = new List<int>();
        if (_output is null)
        {
            for (int e = maxEllipsis - 1; e >= 0; e--)
            {
                result.Add(EllipsisBase + e);
            }

            var count = new SortedDictionary<int, int>();
            foreach (int[] l in labels)
            {
                foreach (int x in l.Where(x => x < EllipsisBase))
                {
                    count[x] = count.GetValueOrDefault(x) + 1;
                }
            }

            result.AddRange(count.Where(kv => kv.Value == 1).Select(kv => kv.Key));
            return result;
        }

        for (int k = 0; k < _output.Length; k++)
        {
            if (_output[k] == '.')
            {
                for (int e = maxEllipsis - 1; e >= 0; e--)
                {
                    result.Add(EllipsisBase + e);
                }

                k += 2;
            }
            else
            {
                if (!labels.Any(l => l.Contains(_output[k])))
                {
                    throw ctx?.Fail($"the output letter '{_output[k]}' appears in no input") ?? new SessionException(SessionErrorKind.InvalidInput, "Einsum: bad output letter");
                }

                result.Add(_output[k]);
            }
        }

        return result;
    }

    // Extent of every label (a size-1 dimension broadcasts against a larger one).
    private static Dictionary<int, int> Extents(ReadOnlySpan<Shape> inputs, int[][] labels, InferContext? ctx)
    {
        var ext = new Dictionary<int, int>();
        for (int i = 0; i < inputs.Length; i++)
        {
            for (int d = 0; d < labels[i].Length; d++)
            {
                int size = inputs[i][d];
                if (!ext.TryGetValue(labels[i][d], out int have) || have == 1)
                {
                    ext[labels[i][d]] = size;
                }
                else if (size != 1 && size != have)
                {
                    throw ctx?.Fail($"label dimension sizes disagree ({have} and {size})") ?? new SessionException(SessionErrorKind.InvalidInput, "Einsum: label sizes disagree");
                }
            }
        }

        return ext;
    }

    public override void Infer(InferContext ctx)
    {
        if (ctx.InputCount != _terms.Length)
        {
            throw ctx.Fail($"the equation has {_terms.Length} operands but the node has {ctx.InputCount} inputs");
        }

        var shapes = new Shape[ctx.InputCount];
        DType t = ctx.InputType(0);
        for (int i = 0; i < shapes.Length; i++)
        {
            if (!ctx.HasInput(i) || ctx.InputType(i) != t)
            {
                throw ctx.Fail("all inputs must be present and of one type");
            }

            shapes[i] = ctx.InputShape(i);
        }

        if (t is not (DType.F32 or DType.F64 or DType.I32 or DType.I64))
        {
            throw ctx.Unsupported($"element type {t} is not supported yet");
        }

        int[][] labels = LabelsOf(shapes, out int maxEllipsis, ctx);
        List<int> outLabels = OutputLabels(labels, maxEllipsis, ctx);
        Dictionary<int, int> ext = Extents(shapes, labels, ctx);
        var dims = new long[outLabels.Count];
        for (int i = 0; i < dims.Length; i++)
        {
            dims[i] = ext[outLabels[i]];
        }

        ctx.SetOutput(0, t, OpUtil.MakeShape(ctx, dims));
    }

    public override void Run(in OpRun run)
    {
        var shapes = new Shape[run.Inputs.Length];
        for (int i = 0; i < shapes.Length; i++)
        {
            shapes[i] = run.Inputs[i].Shape;
        }

        int[][] labels = LabelsOf(shapes, out int maxEllipsis, null);
        List<int> outLabels = OutputLabels(labels, maxEllipsis, null);
        Dictionary<int, int> ext = Extents(shapes, labels, null);

        // Loop order: output labels first, then the contracted ones, so each output element is one sequential sum.
        var order = new List<int>(outLabels);
        foreach (int l in ext.Keys.Order())
        {
            if (!outLabels.Contains(l))
            {
                order.Add(l);
            }
        }

        if (order.Count > MaxLabels)
        {
            throw new SessionException(SessionErrorKind.InvalidInput, "Einsum: too many distinct labels");
        }

        // Per input and loop position, the element stride of that label (0 where the input lacks it or is broadcast along it).
        int n = run.Inputs.Length;
        var stride = new int[n][];
        for (int i = 0; i < n; i++)
        {
            stride[i] = new int[order.Count];
            int s = 1;
            for (int d = shapes[i].Rank - 1; d >= 0; d--)
            {
                int pos = order.IndexOf(labels[i][d]);
                if (shapes[i][d] != 1 || ext[labels[i][d]] == 1)
                {
                    stride[i][pos] += s;
                }

                s *= shapes[i][d];
            }
        }

        var outStride = new int[order.Count];
        int os = 1;
        for (int d = outLabels.Count - 1; d >= 0; d--)
        {
            outStride[d] = os;
            os *= ext[outLabels[d]];
        }

        var extent = new int[order.Count];
        for (int i = 0; i < extent.Length; i++)
        {
            extent[i] = ext[order[i]];
        }

        var job = new Job { Inputs = run.Inputs.ToArray(), Output = run.Outputs[0], Stride = stride, Extent = extent, OutLabelCount = outLabels.Count, N = n };
        TypeDispatch.Number(run.Outputs[0].DType, ref job);
    }

    private struct Job : INumberVisitor
    {
        public Tensor[] Inputs;
        public Tensor Output;
        public int[][] Stride;
        public int[] Extent;
        public int OutLabelCount, N;

        public void Visit<T>() where T : unmanaged, INumber<T>
        {
            Span<T> y = Output.MutableData<T>();
            int outTotal = y.Length;
            int labels = Extent.Length;
            int contracted = 1;
            for (int i = OutLabelCount; i < labels; i++)
            {
                contracted *= Extent[i];
            }

            Span<int> idx = stackalloc int[MaxLabels];
            Span<int> off = stackalloc int[16];
            if (N > off.Length)
            {
                throw new SessionException(SessionErrorKind.InvalidInput, "Einsum: too many inputs");
            }

            for (int o = 0; o < outTotal; o++)
            {
                // Unravel o over the output labels.
                int rest = o;
                for (int d = OutLabelCount - 1; d >= 0; d--)
                {
                    idx[d] = rest % Extent[d];
                    rest /= Extent[d];
                }

                for (int d = OutLabelCount; d < labels; d++)
                {
                    idx[d] = 0;
                }

                T acc = T.Zero;
                for (int c = 0; c < contracted; c++)
                {
                    T prod = T.One;
                    for (int i = 0; i < N; i++)
                    {
                        int at = 0;
                        for (int d = 0; d < labels; d++)
                        {
                            at += idx[d] * Stride[i][d];
                        }

                        off[i] = at;
                        prod = unchecked(prod * Element<T>(Inputs[i], at));
                    }

                    acc = unchecked(acc + prod);
                    for (int d = labels - 1; d >= OutLabelCount; d--)
                    {
                        if (++idx[d] < Extent[d])
                        {
                            break;
                        }

                        idx[d] = 0;
                    }
                }

                y[o] = acc;
            }
        }

        private static T Element<T>(Tensor t, int at) where T : unmanaged => t.Data<T>()[at];
    }
}
