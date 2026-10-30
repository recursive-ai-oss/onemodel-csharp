using System;
using System.Collections.Generic;
using System.Globalization;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tracing;

public sealed partial class Tracer
{
    // An operand of a recorded operation: a graph value, a captured constant, or absent (name "").
    private sealed class Opd
    {
        public required string Name { get; init; }

        public required int[] Shape { get; init; }

        public required SymDim?[] Sym { get; init; }

        public bool IsTraced { get; init; }
    }

    private Opd Operand(in Tensor t)
    {
        if (t.IsEmpty)
        {
            return new Opd { Name = "", Shape = [], Sym = [] };
        }

        var shape = new int[t.Rank];
        for (int i = 0; i < shape.Length; i++)
        {
            shape[i] = t.Shape[i];
        }

        if (t.Autograd?.Trace is TraceValue tv && tv.Owner == this)
        {
            return new Opd { Name = tv.Name, Shape = shape, Sym = tv.Sym, IsTraced = true };
        }

        return new Opd { Name = Constant(t), Shape = shape, Sym = new SymDim?[shape.Length] };
    }

    // ---- emission helpers ------------------------------------------------------------------------------------------------------

    private string[] EmitN(string op, string[] inputs, int outputs, params OnnxAttrSpec[] attrs)
    {
        string nodeName = FreshName(op);
        var outs = new string[outputs];
        for (int i = 0; i < outputs; i++)
        {
            outs[i] = outputs == 1 ? nodeName + "_out" : nodeName + "_out" + i.ToString(CultureInfo.InvariantCulture);
            _usedNames.Add(outs[i]);
        }

        var node = new OnnxNodeSpec { OpType = op, Name = nodeName, Inputs = inputs, Outputs = outs };
        node.Attributes.AddRange(attrs);
        _nodes.Add(node);
        return outs;
    }

    private string Emit(string op, string[] inputs, params OnnxAttrSpec[] attrs) => EmitN(op, inputs, 1, attrs)[0];

    private readonly Dictionary<string, string> _smallConstants = [];

    private string ConstInts(ReadOnlySpan<long> values, bool scalar = false)
    {
        string key = (scalar ? "s" : "v") + string.Join(',', values.ToArray());
        if (_smallConstants.TryGetValue(key, out string? existing))
        {
            return existing;
        }

        string name = FreshName("ints");
        var raw = new byte[values.Length * 8];
        for (int i = 0; i < values.Length; i++)
        {
            System.Buffers.Binary.BinaryPrimitives.WriteInt64LittleEndian(raw.AsSpan(i * 8, 8), values[i]);
        }

        _initializers.Add(new OnnxInitializerSpec { Name = name, DataType = 7, Dims = scalar ? [] : [values.Length], Raw = raw });
        _smallConstants[key] = name;
        return name;
    }

    private string ConstFloat(float v)
    {
        string key = "f" + BitConverter.SingleToInt32Bits(v).ToString(CultureInfo.InvariantCulture);
        if (_smallConstants.TryGetValue(key, out string? existing))
        {
            return existing;
        }

        string name = FreshName("scalar");
        var raw = new byte[4];
        System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(raw, v);
        _initializers.Add(new OnnxInitializerSpec { Name = name, DataType = 1, Dims = [], Raw = raw });
        _smallConstants[key] = name;
        return name;
    }

    private string ConstZeros(int count)
    {
        string key = "z" + count.ToString(CultureInfo.InvariantCulture);
        if (_smallConstants.TryGetValue(key, out string? existing))
        {
            return existing;
        }

        string name = FreshName("zeros");
        _initializers.Add(new OnnxInitializerSpec { Name = name, DataType = 1, Dims = [count], Raw = new byte[count * 4] });
        _smallConstants[key] = name;
        return name;
    }

    private TraceValue NewValue(string name, in Tensor output, SymDim?[] sym)
    {
        var shape = new int[output.Rank];
        for (int i = 0; i < shape.Length; i++)
        {
            shape[i] = output.Shape[i];
        }

        var v = new TraceValue(this, name, output.DType, shape, sym);
        _byName[name] = v;
        return v;
    }

    // A symbolic extent computed at run time: Coef × Π (extent of the atom's source axis), as a one-element int64 tensor.
    private readonly Dictionary<string, string> _exprCache = [];

    private string Expr(SymDim s)
    {
        string key = s.Coef.ToString(CultureInfo.InvariantCulture) + ":" + string.Join(',', Array.ConvertAll(s.Atoms, a => a.Id));
        if (_exprCache.TryGetValue(key, out string? cached))
        {
            return cached;
        }

        string? acc = null;
        foreach (Atom a in s.Atoms)
        {
            string shape = Emit("Shape", [a.Source.Name], OnnxAttrSpec.OfInt("start", a.Axis), OnnxAttrSpec.OfInt("end", a.Axis + 1));
            acc = acc is null ? shape : Emit("Mul", [acc, shape]);
        }

        if (s.Coef != 1)
        {
            acc = Emit("Mul", [acc!, ConstInts([s.Coef])]);
        }

        _exprCache[key] = acc!;
        return acc!;
    }

    // The int64 shape tensor for a target shape whose symbolic extents are computed at run time.
    private string ShapeTensor(int[] shape, SymDim?[] sym)
    {
        int symbolic = 0;
        for (int i = 0; i < shape.Length; i++)
        {
            if (sym[i] is not null)
            {
                symbolic++;
            }
        }

        if (symbolic == 0)
        {
            return ConstInts(Array.ConvertAll(shape, x => (long)x));
        }

        var parts = new List<string>(shape.Length);
        var pending = new List<long>();
        for (int i = 0; i < shape.Length; i++)
        {
            if (sym[i] is null)
            {
                pending.Add(shape[i]);
                continue;
            }

            if (pending.Count > 0)
            {
                parts.Add(ConstInts(pending.ToArray()));
                pending.Clear();
            }

            parts.Add(Expr(sym[i]!));
        }

        if (pending.Count > 0)
        {
            parts.Add(ConstInts(pending.ToArray()));
        }

        return parts.Count == 1 ? parts[0] : Emit("Concat", [.. parts], OnnxAttrSpec.OfInt("axis", 0));
    }

    private static SymDim?[] Copy(SymDim?[] s) => (SymDim?[])s.Clone();

    // ---- lowering --------------------------------------------------------------------------------------------------------------

    private TraceValue Lower(BackwardRecord rec, Tensor output)
    {
        Tensor[] inT = rec.TraceInputs ?? [];
        var ins = new Opd[inT.Length];
        for (int i = 0; i < ins.Length; i++)
        {
            ins[i] = Operand(inT[i]);
        }

        ReadOnlySpan<long> ints = rec.Ints;
        ReadOnlySpan<double> floats = rec.Floats;
        int outRank = output.Rank;
        TraceValue Done(string name, SymDim?[]? sym = null) => NewValue(name, output, sym ?? new SymDim?[outRank]);

        switch (rec.Op)
        {
            case OpKind.Add:
                return Done(Emit("Add", [ins[0].Name, ins[1].Name]), BroadcastSym(ins, output));
            case OpKind.Sub:
                return Done(Emit("Sub", [ins[0].Name, ins[1].Name]), BroadcastSym(ins, output));
            case OpKind.Mul:
                return Done(Emit("Mul", [ins[0].Name, ins[1].Name]), BroadcastSym(ins, output));
            case OpKind.Div:
                return Done(Emit("Div", [ins[0].Name, ins[1].Name]), BroadcastSym(ins, output));
            case OpKind.Neg:
                return Done(Emit("Neg", [ins[0].Name]), Copy(ins[0].Sym));
            case OpKind.Relu:
                return Done(Emit("Relu", [ins[0].Name]), Copy(ins[0].Sym));
            case OpKind.Sigmoid:
                return Done(Emit("Sigmoid", [ins[0].Name]), Copy(ins[0].Sym));
            case OpKind.Tanh:
                return Done(Emit("Tanh", [ins[0].Name]), Copy(ins[0].Sym));
            case OpKind.Gelu:
                return Done(Emit("Gelu", [ins[0].Name], OnnxAttrSpec.OfText("approximate", ints[0] == 1 ? "tanh" : "none")), Copy(ins[0].Sym));
            case OpKind.MatMul:
                return Done(Emit("MatMul", [ins[0].Name, ins[1].Name]), [ins[0].Sym[0], ins[1].Sym[1]]);
            case OpKind.Softmax:
                return Done(Emit("Softmax", [ins[0].Name], OnnxAttrSpec.OfInt("axis", -1)), Copy(ins[0].Sym));
            case OpKind.LogSoftmax:
                return Done(Emit("LogSoftmax", [ins[0].Name], OnnxAttrSpec.OfInt("axis", -1)), Copy(ins[0].Sym));
            case OpKind.LayerNorm:
                return Done(Emit("LayerNormalization", [ins[0].Name, ins[1].Name, ins[2].Name],
                    OnnxAttrSpec.OfInt("axis", -1), OnnxAttrSpec.OfFloat("epsilon", (float)floats[0])), Copy(ins[0].Sym));
            case OpKind.BatchNorm:
                return LowerBatchNorm(rec, ins, ints, floats, output);
            case OpKind.Cast:
                return Done(Emit("Cast", [ins[0].Name], OnnxAttrSpec.OfInt("to", DTypes.ToOnnx(output.DType))), Copy(ins[0].Sym));
            case OpKind.Sum:
            case OpKind.Mean:
                return LowerReduce(rec.Op == OpKind.Sum ? "ReduceSum" : "ReduceMean", ins[0], ints, output);
            case OpKind.Permute:
            {
                var perm = new long[ints.Length];
                var sym = new SymDim?[ints.Length];
                for (int i = 0; i < perm.Length; i++)
                {
                    perm[i] = ints[i];
                    sym[i] = ins[0].Sym[(int)ints[i]];
                }

                return Done(Emit("Transpose", [ins[0].Name], OnnxAttrSpec.OfInts("perm", perm)), sym);
            }

            case OpKind.Slice:
                return LowerSlice(ins[0], ints, output);
            case OpKind.Select:
            {
                int axis = (int)ints[0];
                var sym = new List<SymDim?>(ins[0].Sym);
                sym.RemoveAt(axis);
                string name = Emit("Gather", [ins[0].Name, ConstInts([ints[1]], scalar: true)], OnnxAttrSpec.OfInt("axis", axis));
                return Done(name, [.. sym]);
            }

            case OpKind.Flip:
            {
                int axis = (int)ints[0];
                string name = Emit("Slice", [ins[0].Name, ConstInts([-1]), ConstInts([long.MinValue]), ConstInts([axis]), ConstInts([-1])]);
                return Done(name, Copy(ins[0].Sym));
            }

            case OpKind.Reshape:
                return LowerReshape(ins[0], output);
            case OpKind.Expand:
                return LowerExpand(ins[0], output);
            case OpKind.Concat:
            {
                var names = new string[ins.Length];
                for (int i = 0; i < ins.Length; i++)
                {
                    names[i] = ins[i].Name;
                }

                int axis = (int)ints[0];
                string name = Emit("Concat", names, OnnxAttrSpec.OfInt("axis", axis));
                SymDim?[] sym = new SymDim?[outRank];
                bool derived = false;
                for (int a = 0; a < outRank; a++)
                {
                    foreach (Opd o in ins)
                    {
                        if (o.Sym[a] is { } s)
                        {
                            if (a == axis)
                            {
                                derived = true;
                            }
                            else if (sym[a] is null)
                            {
                                sym[a] = s;
                            }
                        }
                    }
                }

                TraceValue v = Done(name, sym);
                if (derived)
                {
                    v.Sym[axis] = NewDerived(v, axis);
                }

                return v;
            }

            case OpKind.Conv2d:
                return LowerConv(rec, "Conv", ins, ints, output);
            case OpKind.ConvTranspose2d:
                return LowerConv(rec, "ConvTranspose", ins, ints, output);
            case OpKind.MaxPool2d:
                return LowerPool("MaxPool", ins[0], ints, output, false);
            case OpKind.AvgPool2d:
                return LowerPool("AveragePool", ins[0], ints, output, true);
            case OpKind.Resize2d:
                return LowerResize(ins[0], ints, output);
            case OpKind.Embedding:
            {
                Opd ids = Operand(rec.TraceExtras![0]);
                var sym = new SymDim?[outRank];
                Array.Copy(ids.Sym, sym, ids.Sym.Length);
                return Done(Emit("Gather", [ins[0].Name, ids.Name], OnnxAttrSpec.OfInt("axis", 0)), sym);
            }

            case OpKind.Attention:
                return LowerAttention(rec, ins, floats, output);
            case OpKind.Lstm:
                return LowerRecurrent(true, ins, ints, output);
            case OpKind.Gru:
                return LowerRecurrent(false, ins, ints, output);
            default:
                throw Fail(TraceErrorKind.UnsupportedOperation,
                    $"{rec.Op} has no graph-layer equivalent and cannot be traced (see docs/tracing.md for the operations that can).");
        }
    }

    private SymDim NewDerived(TraceValue v, int axis)
    {
        var atom = new Atom { Id = _atoms.Count, Name = null, Size = v.Shape[axis], Source = v, Axis = axis };
        _atoms.Add(atom);
        return SymDim.Of(atom);
    }

    private static SymDim?[] BroadcastSym(Opd[] ins, in Tensor output)
    {
        int rank = output.Rank;
        var sym = new SymDim?[rank];
        for (int a = 0; a < rank; a++)
        {
            foreach (Opd o in ins)
            {
                int k = a - (rank - o.Shape.Length);
                if (k >= 0 && o.Sym[k] is { } s && o.Shape[k] == output.Shape[a])
                {
                    sym[a] = s;
                    break;
                }
            }
        }

        return sym;
    }

    private TraceValue LowerBatchNorm(BackwardRecord rec, Opd[] ins, ReadOnlySpan<long> ints, ReadOnlySpan<double> floats, in Tensor output)
    {
        if (ints[0] != 0)
        {
            throw Fail(TraceErrorKind.UnsupportedOperation,
                "BatchNorm2d in training mode cannot be traced: it updates running statistics and normalizes with the batch. Trace the module in evaluation mode (training: false).");
        }

        Opd mean = Operand(rec.TraceExtras![0]), variance = Operand(rec.TraceExtras[1]);
        if (mean.IsTraced || variance.IsTraced)
        {
            throw Fail(TraceErrorKind.UnsupportedOperation, "BatchNorm2d running statistics must be constants, not computed from the inputs.");
        }

        string name = Emit("BatchNormalization", [ins[0].Name, ins[1].Name, ins[2].Name, mean.Name, variance.Name],
            OnnxAttrSpec.OfFloat("epsilon", (float)floats[0]));
        return NewValue(name, output, Copy(ins[0].Sym));
    }

    private TraceValue LowerReduce(string op, Opd x, ReadOnlySpan<long> ints, in Tensor output)
    {
        long axis = ints[0];
        bool keep = ints[1] != 0;
        SymDim?[] sym;
        string name;
        if (axis < 0)
        {
            name = Emit(op, [x.Name], OnnxAttrSpec.OfInt("keepdims", 0));
            sym = new SymDim?[output.Rank];
        }
        else
        {
            name = Emit(op, [x.Name, ConstInts([axis])], OnnxAttrSpec.OfInt("keepdims", keep ? 1 : 0));
            var l = new List<SymDim?>(x.Sym);
            if (keep)
            {
                l[(int)axis] = null;
            }
            else
            {
                l.RemoveAt((int)axis);
            }

            sym = [.. l];
        }

        return NewValue(name, output, sym);
    }

    private TraceValue LowerSlice(Opd x, ReadOnlySpan<long> ints, in Tensor output)
    {
        int axis = (int)ints[0];
        long start = ints[1], end = ints[2], step = ints[3];
        SymDim?[] sym = Copy(x.Sym);
        bool full = start == 0 && end == x.Shape[axis] && step == 1;
        if (x.Sym[axis] is not null && !full)
        {
            throw Fail(TraceErrorKind.SymbolicShape,
                $"Slice [{start}:{end}:{step}] on axis {axis}, which is symbolic: the bounds would be frozen to the example size. Slice a fixed axis, or take the whole axis.");
        }

        if (full)
        {
            // A whole-axis slice only restrides; no node is needed. Keep the value (and its symbols).
            return NewAlias(x, output);
        }

        string name = Emit("Slice", [x.Name, ConstInts([start]), ConstInts([end]), ConstInts([axis]), ConstInts([step])]);
        return NewValue(name, output, sym);
    }

    // The output is the same data as the operand: the result shares the operand's graph value.
    private TraceValue NewAlias(Opd x, in Tensor output)
    {
        TraceValue v = _byName.TryGetValue(x.Name, out TraceValue? existing) ? existing : NewValue(x.Name, output, Copy(x.Sym));
        return v;
    }

    private TraceValue LowerReshape(Opd x, in Tensor output)
    {
        int rank = output.Rank;
        var shape = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            shape[i] = output.Shape[i];
        }

        if (shape.AsSpan().SequenceEqual(x.Shape))
        {
            return NewAlias(x, output);
        }

        SymDim?[] sym = ReshapeSym(x, shape);
        string target;
        int symCount = 0, lastSym = -1;
        for (int i = 0; i < rank; i++)
        {
            if (sym[i] is not null)
            {
                symCount++;
                lastSym = i;
            }
        }

        if (symCount == 0)
        {
            target = ConstInts(Array.ConvertAll(shape, v => (long)v));
        }
        else if (symCount == 1 && sym[lastSym]!.Coef == 1 && sym[lastSym]!.Atoms.Length == 1)
        {
            // One run-time extent among fixed ones: ONNX's own -1 infers it.
            var l = Array.ConvertAll(shape, v => (long)v);
            l[lastSym] = -1;
            target = ConstInts(l);
        }
        else
        {
            target = ShapeTensor(shape, sym);
        }

        string name = Emit("Reshape", [x.Name, target]);
        return NewValue(name, output, sym);
    }

    // Reshape keeps row-major order, so the dims split into groups whose products agree on both sides; inside a group the symbolic atoms
    // go to the output dims whose example sizes they match (exact products first, then divisors).
    private static SymDim?[] ReshapeSym(Opd x, int[] outShape)
    {
        var result = new SymDim?[outShape.Length];
        bool any = false;
        foreach (SymDim? s in x.Sym)
        {
            any |= s is not null;
        }

        if (!any)
        {
            return result;
        }

        var inIdx = new List<int>();
        for (int i = 0; i < x.Shape.Length; i++)
        {
            if (x.Shape[i] != 1)
            {
                inIdx.Add(i);
            }
        }

        var outIdx = new List<int>();
        for (int i = 0; i < outShape.Length; i++)
        {
            if (outShape[i] != 1)
            {
                outIdx.Add(i);
            }
        }

        int a = 0, b = 0;
        while (a < inIdx.Count || b < outIdx.Count)
        {
            long pi = 1, pj = 1;
            var groupIn = new List<int>();
            var groupOut = new List<int>();
            while (!(groupIn.Count > 0 && groupOut.Count > 0 && pi == pj))
            {
                bool takeIn = groupIn.Count == 0 || (groupOut.Count > 0 && pi < pj);
                if (takeIn ? a >= inIdx.Count : b >= outIdx.Count)
                {
                    throw Fail(TraceErrorKind.SymbolicShape, "Reshape does not preserve the element count.");
                }

                if (takeIn)
                {
                    pi *= x.Shape[inIdx[a]];
                    groupIn.Add(inIdx[a++]);
                }
                else
                {
                    pj *= outShape[outIdx[b]];
                    groupOut.Add(outIdx[b++]);
                }
            }

            var atoms = new List<Atom>();
            foreach (int i in groupIn)
            {
                if (x.Sym[i] is { } s)
                {
                    atoms.AddRange(s.Atoms);
                }
            }

            if (atoms.Count == 0)
            {
                continue;
            }

            AssignAtoms(atoms, groupOut, outShape, result);
        }

        return result;
    }

    private static void AssignAtoms(List<Atom> atoms, List<int> groupOut, int[] outShape, SymDim?[] result)
    {
        var taken = new Dictionary<int, List<Atom>>();
        // Exact products first: the output dim equals the product of some remaining atoms.
        foreach (int o in groupOut)
        {
            List<Atom>? subset = FindSubset(atoms, outShape[o], exact: true);
            if (subset is not null)
            {
                taken[o] = subset;
                foreach (Atom s in subset)
                {
                    atoms.Remove(s);
                }
            }
        }

        // Then divisors: an output dim that is a multiple of an atom's size (a merged head dimension, say).
        foreach (int o in groupOut)
        {
            if (atoms.Count == 0)
            {
                break;
            }

            if (taken.ContainsKey(o))
            {
                continue;
            }

            List<Atom>? subset = FindSubset(atoms, outShape[o], exact: false);
            if (subset is not null)
            {
                taken[o] = subset;
                foreach (Atom s in subset)
                {
                    atoms.Remove(s);
                }
            }
        }

        if (atoms.Count != 0)
        {
            throw Fail(TraceErrorKind.SymbolicShape,
                "A reshape mixes symbolic sizes in a way the tracer cannot follow (" + string.Join(", ", atoms.ConvertAll(a => a.Name ?? "derived extent")) +
                "). Keep each symbolic size in its own dimension, or trace with fixed sizes.");
        }

        foreach ((int o, List<Atom> subset) in taken)
        {
            long prod = 1;
            foreach (Atom s in subset)
            {
                prod *= s.Size;
            }

            subset.Sort((p, q) => p.Id.CompareTo(q.Id));
            result[o] = new SymDim(outShape[o] / prod, [.. subset]);
        }
    }

    private static List<Atom>? FindSubset(List<Atom> atoms, int target, bool exact)
    {
        int n = atoms.Count;
        List<Atom>? best = null;
        for (int mask = 1; mask < (1 << n) && n <= 6; mask++)
        {
            long prod = 1;
            var subset = new List<Atom>();
            for (int i = 0; i < n; i++)
            {
                if ((mask & (1 << i)) != 0)
                {
                    prod *= atoms[i].Size;
                    subset.Add(atoms[i]);
                }
            }

            if (exact ? prod == target : target % prod == 0)
            {
                if (best is null || subset.Count > best.Count)
                {
                    best = subset;
                }
            }
        }

        return best;
    }

    private TraceValue LowerExpand(Opd x, in Tensor output)
    {
        int rank = output.Rank;
        var shape = new int[rank];
        var sym = new SymDim?[rank];
        for (int a = 0; a < rank; a++)
        {
            shape[a] = output.Shape[a];
            int k = a - (rank - x.Shape.Length);
            if (k >= 0 && x.Shape[k] == shape[a])
            {
                sym[a] = x.Sym[k];
            }
            else
            {
                // A size taken from a symbol by broadcasting (a mask [B,1,1,T] expanded to the scores); matched by example size.
                foreach (Atom atom in _atoms)
                {
                    if (atom.Size == shape[a] && AnySymbolic(x))
                    {
                        sym[a] = SymDim.Of(atom);
                        break;
                    }
                }
            }
        }

        string name = Emit("Expand", [x.Name, ShapeTensor(shape, sym)]);
        return NewValue(name, output, sym);
    }

    private static bool AnySymbolic(Opd x)
    {
        foreach (SymDim? s in x.Sym)
        {
            if (s is not null)
            {
                return true;
            }
        }

        return false;
    }

    private TraceValue LowerConv(BackwardRecord rec, string op, Opd[] ins, ReadOnlySpan<long> ints, in Tensor output)
    {
        // ints: [strideH, strideW, padTop, padLeft, padBottom, padRight, dilH, dilW, groups]
        var attrs = new List<OnnxAttrSpec>
        {
            OnnxAttrSpec.OfInts("strides", ints[0], ints[1]),
            OnnxAttrSpec.OfInts("pads", ints[2], ints[3], ints[4], ints[5]),
            OnnxAttrSpec.OfInts("dilations", ints[6], ints[7]),
            OnnxAttrSpec.OfInt("group", ints[8]),
        };
        if (op == "ConvTranspose")
        {
            // The record keeps no output padding; it is what the output extent needs beyond the formula.
            var op2 = new long[2];
            for (int d = 0; d < 2; d++)
            {
                long inp = ins[0].Shape[2 + d], k = ins[1].Shape[2 + d];
                long full = ((inp - 1) * ints[d]) - (ints[2 + d] + ints[4 + d]) + (ints[6 + d] * (k - 1)) + 1;
                op2[d] = output.Shape[2 + d] - full;
            }

            attrs.Add(OnnxAttrSpec.OfInts("output_padding", op2));
        }

        string[] names = ins.Length > 2 && ins[2].Name.Length > 0 ? [ins[0].Name, ins[1].Name, ins[2].Name] : [ins[0].Name, ins[1].Name];
        string name = Emit(op, names, [.. attrs]);
        return NewSpatial(name, ins[0], output);
    }

    private TraceValue LowerPool(string op, Opd x, ReadOnlySpan<long> ints, in Tensor output, bool average)
    {
        var attrs = new List<OnnxAttrSpec>
        {
            OnnxAttrSpec.OfInts("kernel_shape", ints[0], ints[1]),
            OnnxAttrSpec.OfInts("strides", ints[2], ints[3]),
            OnnxAttrSpec.OfInts("pads", ints[4], ints[5], ints[4], ints[5]),
        };
        if (average)
        {
            attrs.Add(OnnxAttrSpec.OfInt("count_include_pad", ints[6]));
        }

        string name = Emit(op, [x.Name], [.. attrs]);
        return NewSpatial(name, x, output);
    }

    private TraceValue LowerResize(Opd x, ReadOnlySpan<long> ints, in Tensor output)
    {
        bool bilinear = (InterpolationMode)ints[0] == InterpolationMode.Bilinear;
        bool align = ints[1] != 0;
        var attrs = new List<OnnxAttrSpec>
        {
            OnnxAttrSpec.OfText("mode", bilinear ? "linear" : "nearest"),
            OnnxAttrSpec.OfText("coordinate_transformation_mode", align ? "align_corners" : bilinear ? "half_pixel" : "asymmetric"),
        };
        if (!bilinear)
        {
            attrs.Add(OnnxAttrSpec.OfText("nearest_mode", "floor"));
        }

        long[] sizes = [output.Shape[0], output.Shape[1], output.Shape[2], output.Shape[3]];
        string name = Emit("Resize", [x.Name, "", "", ConstInts(sizes)], [.. attrs]);
        // The output size is fixed by the record, so the spatial extents are fixed too (symbolic ones would be frozen).
        for (int a = 2; a < 4; a++)
        {
            if (x.Sym[a] is not null)
            {
                throw Fail(TraceErrorKind.SymbolicShape, "Resize2d takes a fixed output size; its input's spatial extents cannot be symbolic.");
            }
        }

        return NewValue(name, output, Copy(x.Sym));
    }

    // Batch and channels carry over; a symbolic spatial extent becomes a new extent the output value holds.
    private TraceValue NewSpatial(string name, Opd x, in Tensor output)
    {
        var sym = new SymDim?[output.Rank];
        sym[0] = x.Sym[0];
        TraceValue v = NewValue(name, output, sym);
        for (int a = 2; a < output.Rank; a++)
        {
            if (x.Sym[a] is not null)
            {
                v.Sym[a] = NewDerived(v, a);
            }
        }

        return v;
    }

    private TraceValue LowerAttention(BackwardRecord rec, Opd[] ins, ReadOnlySpan<double> floats, in Tensor output)
    {
        // softmax(scale · Q Kᵀ + mask) V, written as the operators the graph layer fuses back into one attention node.
        Opd q = ins[0], k = ins[1], v = ins[2];
        Opd mask = Operand(rec.TraceExtras![0]);
        string kt = Emit("Transpose", [k.Name], OnnxAttrSpec.OfInts("perm", 0, 1, 3, 2));
        string scores = Emit("MatMul", [q.Name, kt]);
        scores = Emit("Mul", [scores, ConstFloat((float)floats[0])]);
        if (mask.Name.Length > 0)
        {
            string m = mask.Shape.Length == 3 ? Emit("Unsqueeze", [mask.Name, ConstInts([1])]) : mask.Name;
            scores = Emit("Add", [scores, m]);
        }

        string p = Emit("Softmax", [scores], OnnxAttrSpec.OfInt("axis", -1));
        string o = Emit("MatMul", [p, v.Name]);
        var sym = Copy(q.Sym);
        if (output.Rank == 4 && v.Sym[3] is { } d)
        {
            sym[3] = d;
        }

        return NewValue(o, output, sym);
    }

    // One direction of one layer. PyTorch stacks gates (i, f, g, o) and (r, z, n); ONNX stacks (i, o, f, c) and (z, r, h), so the weight
    // rows are gathered into ONNX order (a node, so traced weights work as well as constants; constants fold).
    private TraceValue LowerRecurrent(bool lstm, Opd[] ins, ReadOnlySpan<long> ints, in Tensor output)
    {
        bool reverse = ints[0] != 0;
        int h = (int)ints[4];
        int gates = lstm ? 4 : 3;
        int[] order = lstm ? [0, 3, 1, 2] : [1, 0, 2];
        var rows = new long[gates * h];
        for (int g = 0; g < gates; g++)
        {
            for (int r = 0; r < h; r++)
            {
                rows[(g * h) + r] = ((long)order[g] * h) + r;
            }
        }

        string rowIdx = ConstInts(rows);
        string axis0 = ConstInts([0]);
        string Reorder(string w) => Emit("Unsqueeze", [Emit("Gather", [w, rowIdx], OnnxAttrSpec.OfInt("axis", 0)), axis0]);
        string wName = Reorder(ins[1].Name), rName = Reorder(ins[2].Name);
        string bName = "";
        if (ins[3].Name.Length > 0 || ins[4].Name.Length > 0)
        {
            string wb = ins[3].Name.Length > 0 ? ins[3].Name : ConstZeros(gates * h);
            string rb = ins[4].Name.Length > 0 ? ins[4].Name : ConstZeros(gates * h);
            string both = Emit("Concat", [Emit("Gather", [wb, rowIdx], OnnxAttrSpec.OfInt("axis", 0)), Emit("Gather", [rb, rowIdx], OnnxAttrSpec.OfInt("axis", 0))],
                OnnxAttrSpec.OfInt("axis", 0));
            bName = Emit("Unsqueeze", [both, axis0]);
        }

        string h0 = ins[5].Name.Length > 0 ? Emit("Unsqueeze", [ins[5].Name, axis0]) : "";
        string c0 = lstm && ins.Length > 6 && ins[6].Name.Length > 0 ? Emit("Unsqueeze", [ins[6].Name, axis0]) : "";
        var inputs = new List<string> { ins[0].Name, wName, rName, bName, "", h0 };
        if (lstm)
        {
            inputs.Add(c0);
        }

        while (inputs.Count > 0 && inputs[^1].Length == 0)
        {
            inputs.RemoveAt(inputs.Count - 1);
        }

        var attrs = new List<OnnxAttrSpec>
        {
            OnnxAttrSpec.OfInt("hidden_size", h),
            OnnxAttrSpec.OfText("direction", reverse ? "reverse" : "forward"),
        };
        if (!lstm)
        {
            attrs.Add(OnnxAttrSpec.OfInt("linear_before_reset", 1));
        }

        string[] outs = EmitN(lstm ? "LSTM" : "GRU", [.. inputs], 1, [.. attrs]);
        string y = Emit("Squeeze", [outs[0], ConstInts([1])]);
        return NewValue(y, output, [ins[0].Sym[0], ins[0].Sym[1], null]);
    }
}
