using System;
using RecursiveAI.OneModel.Tensors;
using RecursiveAI.OneModel.Tracing;

namespace RecursiveAI.OneModel.Training;

/// <summary>Records a module into the graph layer (see <see cref="Tracer"/>).</summary>
public static class ModuleTracing
{
    /// <summary>
    /// Runs <paramref name="module"/> once in evaluation mode on <paramref name="example"/> and returns the recorded graph: input
    /// <c>"input"</c>, output <c>"output"</c>, weights and buffers named by their dotted state-dict path. The module's training flag is
    /// restored afterwards.
    /// </summary>
    /// <param name="module">The model; dropout is off and batch norm uses its running statistics while tracing.</param>
    /// <param name="example">An example input. Not modified.</param>
    /// <param name="symbolicAxes">As <see cref="Tracer.Input"/>: one name per axis (null for a fixed axis), or none.</param>
    /// <returns>The traced model; dispose it when done.</returns>
    /// <exception cref="TraceException">The module does something a static graph cannot hold.</exception>
    public static TracedModel Trace(this NnModule module, in Tensor example, params ReadOnlySpan<string?> symbolicAxes)
    {
        ArgumentNullException.ThrowIfNull(module);
        bool wasTraining = module.IsTraining;
        module.Eval();
        try
        {
            using Tracer tracer = Tracer.Begin();
            foreach ((string name, Tensor t) in module.NamedParameters())
            {
                tracer.Parameter(name, t);
            }

            foreach ((string name, Tensor t) in module.NamedBuffers())
            {
                tracer.Parameter(name, t);
            }

            Tensor x = tracer.Input("input", example, symbolicAxes);
            Tensor y = module.Forward(x);
            return tracer.Finish("output", y);
        }
        finally
        {
            module.Train(wasTraining);
        }
    }
}
