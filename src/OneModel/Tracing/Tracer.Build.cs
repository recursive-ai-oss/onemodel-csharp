using System;
using System.Collections.Generic;
using RecursiveAI.OneModel.Onnx;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tracing;

public sealed partial class Tracer
{
    private TracedModel Build(ReadOnlySpan<string> outputNames, TraceValue[] outputs)
    {
        var rename = new Dictionary<string, string>(StringComparer.Ordinal);
        var identities = new List<(string From, string To)>();
        for (int i = 0; i < outputs.Length; i++)
        {
            string wanted = outputNames[i];
            if (string.IsNullOrEmpty(wanted) || !_usedNames.Add(wanted))
            {
                throw Fail(TraceErrorKind.InvalidUsage, $"The output name '{wanted}' is empty or already used by an input or another output.");
            }

            // An output that is a graph input, or the same value as an earlier output, gets an Identity node; any other is renamed.
            if (outputs[i].IsInput || rename.ContainsKey(outputs[i].Name))
            {
                identities.Add((rename.GetValueOrDefault(outputs[i].Name, outputs[i].Name), wanted));
            }
            else
            {
                rename[outputs[i].Name] = wanted;
            }
        }

        // Keep only what the outputs need: the trace also recorded scratch work and values nothing used.
        var needed = new HashSet<string>(StringComparer.Ordinal);
        foreach (TraceValue o in outputs)
        {
            needed.Add(o.Name);
        }

        var keep = new bool[_nodes.Count];
        for (int n = _nodes.Count - 1; n >= 0; n--)
        {
            OnnxNodeSpec node = _nodes[n];
            bool used = false;
            foreach (string o in node.Outputs)
            {
                used |= needed.Contains(o);
            }

            if (!used)
            {
                continue;
            }

            keep[n] = true;
            foreach (string i in node.Inputs)
            {
                if (i.Length > 0)
                {
                    needed.Add(i);
                }
            }
        }

        var nodes = new List<OnnxNodeSpec>();
        for (int n = 0; n < _nodes.Count; n++)
        {
            if (!keep[n])
            {
                continue;
            }

            OnnxNodeSpec src = _nodes[n];
            var copy = new OnnxNodeSpec
            {
                OpType = src.OpType,
                Name = src.Name,
                Inputs = Array.ConvertAll(src.Inputs, s => rename.GetValueOrDefault(s, s)),
                Outputs = Array.ConvertAll(src.Outputs, s => rename.GetValueOrDefault(s, s)),
            };
            copy.Attributes.AddRange(src.Attributes);
            nodes.Add(copy);
        }

        foreach ((string from, string to) in identities)
        {
            nodes.Add(new OnnxNodeSpec { OpType = "Identity", Name = FreshName("Identity"), Inputs = [from], Outputs = [to] });
        }

        var initializers = new List<OnnxInitializerSpec>();
        foreach (OnnxInitializerSpec init in _initializers)
        {
            if (needed.Contains(init.Name))
            {
                initializers.Add(init);
            }
        }

        var inputSpecs = new List<OnnxValueSpec>();
        foreach (TraceValue v in _inputs)
        {
            inputSpecs.Add(ValueSpec(v.Name, v));
        }

        var outputSpecs = new List<OnnxValueSpec>();
        for (int i = 0; i < outputs.Length; i++)
        {
            outputSpecs.Add(ValueSpec(outputNames[i], outputs[i]));
        }

        byte[] onnx = OnnxWriter.Write(OutputOpset, "traced", nodes, initializers, inputSpecs, outputSpecs);
        var inNames = new string[_inputs.Count];
        for (int i = 0; i < inNames.Length; i++)
        {
            inNames[i] = _inputs[i].Name;
        }

        return new TracedModel(onnx, inNames, outputNames.ToArray(), nodes.Count);
    }

    private static OnnxValueSpec ValueSpec(string name, TraceValue v)
    {
        var dims = new OnnxDimSpec[v.Shape.Length];
        for (int a = 0; a < dims.Length; a++)
        {
            dims[a] = v.Sym[a] is { } s
                ? new OnnxDimSpec(null, s.Coef == 1 && s.Atoms.Length == 1 ? s.Atoms[0].Name : null)
                : new OnnxDimSpec(v.Shape[a], null);
        }

        return new OnnxValueSpec { Name = name, DataType = DTypes.ToOnnx(v.DType), Dims = dims };
    }
}
