using System;
using System.Collections.Immutable;
using System.IO;
using System.Threading;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel;

/// <summary>
/// The one-line way to run a model: <c>using var m = Model.Load("model.onnx"); float[] v = m.Run(tokens);</c>.
/// </summary>
/// <remarks>
/// <para>A model's input and output names, layouts and shapes are worked out from the file; callers pass arrays or spans of values in
/// row-major order. Threading is chosen for you (one thread per core inside an operator) and results do not depend on it. The
/// <c>Run</c> methods are safe to call from any number of threads at once. For several inputs or outputs, named access, tensors,
/// exact mode or a thread limit, use <see cref="Session"/> (reachable through <see cref="Session"/>) with
/// <see cref="SessionOptions"/>.</para>
/// <para>The typed <c>Run</c> overloads need a model with exactly one input and one output.</para>
/// </remarks>
public sealed class Model : IDisposable
{
    private const int StackDims = Shape.MaxRank;

    private Model(Session session)
    {
        Session = session;
    }

    /// <summary>The graph-layer session underneath, for anything the typed overloads do not cover.</summary>
    public Session Session { get; }

    /// <summary>Loads an ONNX model from a file; weights stored in external data files next to it are read too.</summary>
    /// <param name="path">The <c>.onnx</c> file.</param>
    /// <param name="options">Defaults to fast mode and one thread per core. Pass options for exact mode or a thread limit.</param>
    /// <param name="cancellationToken">Cancels loading.</param>
    /// <exception cref="Onnx.OnnxImportException">The file is not a valid ONNX model.</exception>
    /// <exception cref="SessionException">The model uses an operator this library does not implement (the exception lists all of them), or exact mode was requested and an operator is not exact.</exception>
    /// <exception cref="OperationCanceledException">Cancelled.</exception>
    public static Model Load(string path, SessionOptions? options = null, CancellationToken cancellationToken = default) =>
        new(Session.Load(path, options, cancellationToken));

    /// <summary>Loads an ONNX model from a stream, read to its end. Models with external data files must be loaded from a path.</summary>
    /// <inheritdoc cref="Load(string, SessionOptions?, CancellationToken)"/>
    public static Model Load(Stream stream, SessionOptions? options = null, CancellationToken cancellationToken = default) =>
        new(Session.Load(stream, options, cancellationToken));

    /// <summary>Runs the model on float values. The input's shape comes from the model's declared shape (see <see cref="Run{TIn, TOut}(ReadOnlySpan{TIn}, ReadOnlySpan{int})"/>).</summary>
    /// <param name="input">The values in row-major order.</param>
    /// <returns>The output as a new array, in row-major order.</returns>
    /// <exception cref="SessionException">The model does not have exactly one input and one output of these types, the length does not fit the declared shape, or an operator cannot handle the shape.</exception>
    public float[] Run(float[] input) => Run<float, float>(input, default);

    /// <summary>Runs the model on int64 values (token ids, for example).</summary>
    /// <inheritdoc cref="Run(float[])"/>
    public float[] Run(long[] input) => Run<long, float>(input, default);

    /// <summary>Runs the model on int32 values.</summary>
    /// <inheritdoc cref="Run(float[])"/>
    public float[] Run(int[] input) => Run<int, float>(input, default);

    /// <summary>Runs the model on float values of an explicit shape.</summary>
    /// <param name="input">The values in row-major order.</param>
    /// <param name="shape">The input's shape, for example <c>[1, 3, 224, 224]</c>; its product must equal the input length.</param>
    /// <inheritdoc cref="Run(float[])"/>
    public float[] Run(float[] input, ReadOnlySpan<int> shape) => Run<float, float>(input, shape);

    /// <summary>Runs the model on int64 values of an explicit shape.</summary>
    /// <inheritdoc cref="Run(float[], ReadOnlySpan{int})"/>
    public float[] Run(long[] input, ReadOnlySpan<int> shape) => Run<long, float>(input, shape);

    /// <summary>Runs the model on int32 values of an explicit shape.</summary>
    /// <inheritdoc cref="Run(float[], ReadOnlySpan{int})"/>
    public float[] Run(int[] input, ReadOnlySpan<int> shape) => Run<int, float>(input, shape);

    /// <summary>
    /// Runs the model on values of any supported element type and returns the output as an array of any supported element type.
    /// </summary>
    /// <typeparam name="TIn">Must match the model input's element type.</typeparam>
    /// <typeparam name="TOut">Must match the model output's element type.</typeparam>
    /// <param name="input">The values in row-major order.</param>
    /// <param name="shape">The input's shape. Empty (the default) takes it from the model: a declared shape with at most one symbolic or
    /// unknown extent is completed from the input length (a batch size, say); a model that declares no shape takes a 1-D input.</param>
    /// <exception cref="SessionException">As <see cref="Run(float[])"/>; also if the shape is ambiguous and none was given.</exception>
    public TOut[] Run<TIn, TOut>(ReadOnlySpan<TIn> input, ReadOnlySpan<int> shape = default)
        where TIn : unmanaged
        where TOut : unmanaged
    {
        int primary = PrimaryInput();
        string outName = SingleOutput();
        Span<int> resolved = stackalloc int[StackDims];
        Shape s = ResolveShape(primary, input.Length, shape, resolved);
        using Tensor inT = Tensor.FromSpan(input, s);
        ImmutableArray<TensorInfo> all = Session.Inputs;
        string[] inNames = new string[all.Length];
        Tensor[] ins = new Tensor[all.Length];
        try
        {
            for (int i = 0; i < all.Length; i++)
            {
                inNames[i] = all[i].Name;
                ins[i] = i == primary ? inT : AuxiliaryInput(all[i], s);
            }

            Span<string> outNames = [outName];
            Span<Tensor> outs = [default];
            Session.Run(inNames, ins, outNames, outs);
            using Tensor outT = outs[0];
            if (outT.DType != DTypes.Of<TOut>())
            {
                throw new SessionException(SessionErrorKind.InvalidInput, $"Output '{outName}' holds {outT.DType}, not {DTypes.Of<TOut>()}.");
            }

            return outT.ToArray<TOut>();
        }
        finally
        {
            for (int i = 0; i < ins.Length; i++)
            {
                if (i != primary && !ins[i].IsEmpty)
                {
                    ins[i].Dispose();
                }
            }
        }
    }

    /// <summary>
    /// Runs the model and writes the output into a caller-provided buffer. Allocates nothing once the model has run once with this shape.
    /// </summary>
    /// <param name="input">The values in row-major order.</param>
    /// <param name="shape">The input's shape; empty takes it from the model as in <see cref="Run{TIn, TOut}(ReadOnlySpan{TIn}, ReadOnlySpan{int})"/>.</param>
    /// <param name="output">Receives the output in row-major order.</param>
    /// <returns>The number of values written.</returns>
    /// <exception cref="SessionException">As <see cref="Run(float[])"/>; also if <paramref name="output"/> is too short (the message gives the length needed).</exception>
    public int Run<TIn, TOut>(ReadOnlySpan<TIn> input, ReadOnlySpan<int> shape, Span<TOut> output)
        where TIn : unmanaged
        where TOut : unmanaged
    {
        if (Session.Inputs.Length != 1)
        {
            throw new SessionException(SessionErrorKind.InvalidInput,
                $"This model has {Session.Inputs.Length} inputs; the span-output overload needs exactly one. Use the array overloads or Model.Session.Run with named tensors.");
        }

        string inName = Session.Inputs[0].Name, outName = SingleOutput();
        Span<int> resolved = stackalloc int[StackDims];
        Shape s = ResolveShape(0, input.Length, shape, resolved);
        return Session.Run(inName, input, s.Dimensions, outName, output).Length;
    }

    /// <summary>Runs the model on float values and writes the float output into <paramref name="output"/>; the shape comes from the model.</summary>
    /// <returns>The number of values written.</returns>
    /// <inheritdoc cref="Run{TIn, TOut}(ReadOnlySpan{TIn}, ReadOnlySpan{int}, Span{TOut})"/>
    public int Run(ReadOnlySpan<float> input, Span<float> output) => Run<float, float>(input, default, output);

    /// <summary>
    /// Checks that the model can be run from one array of values. A model with several inputs (a text encoder's ids, mask and segment
    /// ids) still works when every input after the first is a mask or segment-id input that can be filled from the ids' shape:
    /// <c>attention_mask</c> is all ones (nothing is padding) and <c>token_type_ids</c> is all zeros. The values feed input 0.
    /// </summary>
    private int PrimaryInput()
    {
        ImmutableArray<TensorInfo> ins = Session.Inputs;
        for (int i = 1; i < ins.Length; i++)
        {
            if (AuxiliaryKind(ins[i]) == 0)
            {
                throw new SessionException(SessionErrorKind.InvalidInput,
                    $"This model has {ins.Length} inputs and '{ins[i].Name}' cannot be filled in for you; the typed Run overloads need one input (or token ids plus attention_mask and token_type_ids). Use Model.Session.Run with named tensors.");
            }
        }

        return 0;
    }

    // 1 = ones (attention mask), 2 = zeros (segment ids), 0 = not an auxiliary input.
    private static int AuxiliaryKind(TensorInfo i)
    {
        if (i.DType is not (DType.I64 or DType.I32))
        {
            return 0;
        }

        if (i.Name.Contains("attention_mask", StringComparison.Ordinal))
        {
            return 1;
        }

        return i.Name.Contains("token_type_ids", StringComparison.Ordinal) ? 2 : 0;
    }

    private static Tensor AuxiliaryInput(TensorInfo info, Shape like)
    {
        Tensor t = Tensor.Create(info.DType, like, zero: true);
        if (AuxiliaryKind(info) == 1)
        {
            if (info.DType == DType.I64)
            {
                t.MutableData<long>().Fill(1);
            }
            else
            {
                t.MutableData<int>().Fill(1);
            }
        }

        return t;
    }

    private string SingleOutput() =>
        Session.Outputs.Length == 1
            ? Session.Outputs[0].Name
            : throw new SessionException(SessionErrorKind.InvalidInput,
                $"This model has {Session.Outputs.Length} outputs; the typed Run overloads need exactly one. Use Model.Session.Run with named tensors.");

    private Shape ResolveShape(int inputIndex, int length, ReadOnlySpan<int> given, Span<int> scratch)
    {
        TensorInfo info = Session.Inputs[inputIndex];
        if (!given.IsEmpty)
        {
            return CreateShape(given, info);
        }

        if (!info.HasShape)
        {
            scratch[0] = length;
            return Shape.Create(scratch.Slice(0, 1));
        }

        if (info.Dims.Length > scratch.Length)
        {
            throw new SessionException(SessionErrorKind.Unsupported, $"Input '{info.Name}' has {info.Dims.Length} dimensions; the limit is {Shape.MaxRank}.");
        }

        long known = 1;
        int unknown = -1, unknownCount = 0;
        for (int i = 0; i < info.Dims.Length; i++)
        {
            if (info.Dims[i].IsFixed)
            {
                scratch[i] = checked((int)info.Dims[i].Value);
                known *= scratch[i];
            }
            else
            {
                unknown = i;
                unknownCount++;
            }
        }

        // Token ids are [batch, sequence]: with both open and no shape given, the values are one sequence.
        if (unknownCount == 2 && unknown > 0 && !info.Dims[0].IsFixed && info.Dims.Length == 2 && info.DType is DType.I64 or DType.I32)
        {
            scratch[0] = 1;
            unknownCount = 1;
        }

        if (unknownCount > 1)
        {
            throw new SessionException(SessionErrorKind.InvalidInput,
                $"Input '{info.Name}' is declared [{string.Join(", ", info.Dims)}], which leaves {unknownCount} extents open; pass the shape: Run(values, [..]).");
        }

        if (unknownCount == 1)
        {
            if (known == 0 || length % known != 0)
            {
                throw new SessionException(SessionErrorKind.InvalidInput,
                    $"Input '{info.Name}' is declared [{string.Join(", ", info.Dims)}] but {length} values do not divide into whole rows of {known}.");
            }

            scratch[unknown] = checked((int)(length / known));
        }
        else if (known != length)
        {
            throw new SessionException(SessionErrorKind.InvalidInput,
                $"Input '{info.Name}' is declared [{string.Join(", ", info.Dims)}] ({known} values) but {length} were supplied.");
        }

        return Shape.Create(scratch.Slice(0, info.Dims.Length));
    }

    private static Shape CreateShape(ReadOnlySpan<int> dims, TensorInfo info)
    {
        try
        {
            return Shape.Create(dims);
        }
        catch (TensorShapeException e)
        {
            throw new SessionException(SessionErrorKind.InvalidInput, $"Input '{info.Name}': {e.Message}");
        }
    }

    /// <summary>Releases the model's weights once every run in progress has finished.</summary>
    public void Dispose() => Session.Dispose();
}
