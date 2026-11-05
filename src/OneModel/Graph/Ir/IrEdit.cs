using System;
using System.Collections.Immutable;

namespace RecursiveAI.OneModel.Graph;

/// <summary>Replacing a model's weights in place, for fine-tuned parameters written back into the graph they came from.</summary>
internal static class IrEdit
{
    /// <summary>
    /// Replaces the initializer named <paramref name="name"/> in the main graph with little-endian <paramref name="raw"/> bytes of the
    /// same element type and shape. The graph keeps its structure; sessions created afterwards see the new values.
    /// </summary>
    public static void ReplaceInitializer(IrModel model, string name, byte[] raw)
    {
        IrGraph g = model.Graph;
        int index = -1;
        for (int i = 0; i < g.Initializers.Length; i++)
        {
            if (g.Initializers[i].Name == name)
            {
                index = i;
                break;
            }
        }

        if (index < 0)
        {
            throw new ArgumentException($"The graph has no initializer '{name}'.", nameof(name));
        }

        TensorData old = g.Initializers[index];
        if (raw.LongLength != TensorData.RawByteLength(old.DataType, old.ElementCount))
        {
            throw new ArgumentException($"Initializer '{name}' needs {TensorData.RawByteLength(old.DataType, old.ElementCount)} bytes, not {raw.LongLength}.", nameof(raw));
        }

        var replacement = new TensorData(old.Name, old.DataType, old.Dims, old.ElementCount, ByteSource.FromArray(raw), 0, raw.Length, [], false, old.DocString);
        g.Initializers = g.Initializers.SetItem(index, replacement);
        IrValue? v = g.FindValue(name);
        if (v is not null)
        {
            v.Initializer = replacement;
        }
    }
}
