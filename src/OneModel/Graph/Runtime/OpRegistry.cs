using System;
using System.Collections.Generic;

namespace RecursiveAI.OneModel.Graph.Runtime;

/// <summary>A static table of operator definitions, looked up by domain, type and opset when a session is created.</summary>
internal sealed class OpRegistry
{
    private readonly Dictionary<(string Domain, string OpType), List<OpDef>> _defs = [];

    /// <summary>The operators this library ships (<see cref="Operators.OpTable"/>).</summary>
    public static OpRegistry Default { get; } = new(Operators.OpTable.All);

    public OpRegistry(IEnumerable<OpDef> defs)
    {
        foreach (OpDef d in defs)
        {
            if (!_defs.TryGetValue((d.Domain, d.OpType), out List<OpDef>? list))
            {
                _defs[(d.Domain, d.OpType)] = list = [];
            }

            foreach (OpDef other in list)
            {
                if (d.MinOpset <= other.MaxOpset && other.MinOpset <= d.MaxOpset)
                {
                    throw new InvalidOperationException($"Operator {d.OpType} is defined twice for overlapping opsets ({d.MinOpset}-{d.MaxOpset} and {other.MinOpset}-{other.MaxOpset}).");
                }
            }

            list.Add(d);
        }
    }

    /// <summary>Finds the definition that serves <paramref name="opset"/>.</summary>
    public bool TryFind(string domain, string opType, int opset, out OpDef def)
    {
        if (domain == "ai.onnx")
        {
            domain = "";
        }

        if (_defs.TryGetValue((domain, opType), out List<OpDef>? list))
        {
            foreach (OpDef d in list)
            {
                if (opset >= d.MinOpset && opset <= d.MaxOpset)
                {
                    def = d;
                    return true;
                }
            }
        }

        def = null!;
        return false;
    }
}
