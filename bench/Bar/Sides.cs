using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;

namespace OneModel.Bar;

/// <summary>One engine under test. The harness times <see cref="Run"/> and nothing else.</summary>
internal interface ISide : IDisposable
{
    string Name { get; }

    string Version { get; }

    /// <summary>Loads the model with <paramref name="threads"/> intra-op threads (inter-op is always 1).</summary>
    void Load(string modelPath, int threads);

    /// <summary>Binds the inputs once, outside the timed region.</summary>
    void Bind(IReadOnlyDictionary<string, NpyArray> inputs);

    /// <summary>One inference with the bound inputs; keeps its outputs readable through <see cref="Output"/>.</summary>
    void Run();

    float[]? Output(string name);
}

/// <summary>ONNX Runtime at ORT_ENABLE_ALL, sequential execution, intra-op = threads, inter-op = 1.</summary>
internal sealed class OrtSide : ISide
{
    private InferenceSession? _session;
    private readonly List<OrtValue> _inputValues = [];
    private string[] _inputNames = [];
    private string[] _outputNames = [];
    private readonly RunOptions _runOptions = new();
    private IDisposableReadOnlyCollection<OrtValue>? _last;

    public string Name => "ort";

    public string Version => OrtEnv.Instance().GetVersionString();

    public void Load(string modelPath, int threads)
    {
        using var so = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            IntraOpNumThreads = threads,
            InterOpNumThreads = 1,
        };
        _session = new InferenceSession(modelPath, so);
        _outputNames = [.. _session.OutputNames];
    }

    public void Bind(IReadOnlyDictionary<string, NpyArray> inputs)
    {
        var names = new List<string>();
        foreach (string n in _session!.InputNames)
        {
            NpyArray a = inputs[n];
            names.Add(n);
            _inputValues.Add(a.F32 is not null
                ? OrtValue.CreateTensorValueFromMemory(a.F32, a.Shape)
                : OrtValue.CreateTensorValueFromMemory(a.I64!, a.Shape));
        }

        _inputNames = [.. names];
    }

    public void Run()
    {
        _last?.Dispose();
        _last = _session!.Run(_runOptions, _inputNames, _inputValues, _outputNames);
    }

    public float[]? Output(string name)
    {
        int i = Array.IndexOf(_outputNames, name);
        return i < 0 || _last is null ? null : _last[i].GetTensorDataAsSpan<float>().ToArray();
    }

    public void Dispose()
    {
        _last?.Dispose();
        foreach (OrtValue v in _inputValues)
        {
            v.Dispose();
        }

        _runOptions.Dispose();
        _session?.Dispose();
    }
}

/// <summary>Placeholder for this library's session; wired up when the session exists.</summary>
internal sealed class OursSide : ISide
{
    public string Name => "ours";

    public string Version => "unavailable";

    public void Load(string modelPath, int threads) =>
        throw new NotSupportedException("the library's session is not implemented yet; use --side ort");

    public void Bind(IReadOnlyDictionary<string, NpyArray> inputs) => throw new NotSupportedException();

    public void Run() => throw new NotSupportedException();

    public float[]? Output(string name) => null;

    public void Dispose()
    {
    }
}
