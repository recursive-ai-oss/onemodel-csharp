using System;
using System.Collections.Generic;
using System.Linq;
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

/// <summary>This library through its session: the same model file and inputs, intra-op threads as given, no other tuning.</summary>
internal sealed class OursSide : ISide
{
    private readonly RecursiveAI.OneModel.Graph.ExecutionMode _mode;
    private RecursiveAI.OneModel.Graph.Session? _session;
    private string[] _inputNames = [];
    private string[] _outputNames = [];
    private RecursiveAI.OneModel.Tensors.Tensor[] _inputs = [];
    private RecursiveAI.OneModel.Tensors.Tensor[] _outputs = [];

    public OursSide(string mode)
    {
        _mode = mode switch
        {
            "fast" => RecursiveAI.OneModel.Graph.ExecutionMode.Fast,
            "exact" => RecursiveAI.OneModel.Graph.ExecutionMode.Exact,
            _ => throw new ArgumentException($"unknown mode {mode} (fast or exact)"),
        };
    }

    public string Name => "ours";

    public string KernelPath => _session?.KernelPath ?? "";

    public string Version => typeof(RecursiveAI.OneModel.Graph.Session).Assembly.GetName().Version?.ToString() ?? "unknown";

    public void Load(string modelPath, int threads)
    {
        _session = RecursiveAI.OneModel.Graph.Session.Load(modelPath,
            new RecursiveAI.OneModel.Graph.SessionOptions { Mode = _mode, IntraOpThreads = threads });
        _outputNames = [.. _session.Outputs.Select(o => o.Name)];
        _outputs = new RecursiveAI.OneModel.Tensors.Tensor[_outputNames.Length];
    }

    public void Bind(IReadOnlyDictionary<string, NpyArray> inputs)
    {
        _inputNames = [.. _session!.Inputs.Select(i => i.Name)];
        _inputs = new RecursiveAI.OneModel.Tensors.Tensor[_inputNames.Length];
        for (int i = 0; i < _inputs.Length; i++)
        {
            NpyArray a = inputs[_inputNames[i]];
            var shape = RecursiveAI.OneModel.Tensors.Shape.Create(a.Shape);
            _inputs[i] = a.F32 is not null
                ? RecursiveAI.OneModel.Tensors.Tensor.FromSpan<float>(a.F32, shape)
                : RecursiveAI.OneModel.Tensors.Tensor.FromSpan<long>(a.I64!, shape);
        }
    }

    /// <summary>The first run allocates the output tensors; later runs overwrite them in place, so only compute is timed.</summary>
    public void Run() => _session!.Run(_inputNames, _inputs, _outputNames, _outputs);

    public float[]? Output(string name)
    {
        int i = Array.IndexOf(_outputNames, name);
        return i < 0 || _outputs[i].IsEmpty ? null : _outputs[i].ToArray<float>();
    }

    public void Dispose()
    {
        foreach (var t in _inputs.Concat(_outputs))
        {
            if (!t.IsEmpty)
            {
                t.Dispose();
            }
        }

        _session?.Dispose();
    }
}
