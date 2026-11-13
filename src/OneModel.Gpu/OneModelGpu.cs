using System;
using System.Collections.Generic;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu;

/// <summary>
/// The entry point of the GPU package. Call <see cref="Register(GpuOptions?)"/> once, then create sessions with
/// <see cref="ModelOptions.Device"/> set to <see cref="ComputeDevice.Gpu"/>.
/// </summary>
/// <remarks>
/// <para>Registration is explicit so that nothing is discovered by reflection, which keeps the package trimmable and NativeAOT-safe.</para>
/// <para>Thread-safety: a session's GPU regions run one at a time (the session serialises them on its device); the CPU parts of
/// concurrent runs still overlap. Each session owns its own device, memory and compiled kernels, released when the session is disposed.</para>
/// </remarks>
public static class OneModelGpu
{
    /// <summary>
    /// Makes <see cref="ComputeDevice.Gpu"/> available. Safe to call more than once: the last call's options apply to sessions created afterwards.
    /// </summary>
    /// <param name="options">Adapter choice, validation and timing; defaults when null.</param>
    public static void Register(GpuOptions? options = null)
    {
        GpuOptions captured = options ?? new GpuOptions();
        ExecutionBackends.RegisterGpu(model => GpuBackend.Create(model, captured));
    }

    /// <summary>
    /// Makes the device that the next GPU session will own. Needed in a browser, which can only create its WebGPU device asynchronously;
    /// call it once before each <see cref="Session"/> created with <see cref="ComputeDevice.Gpu"/> (the session disposes its device).
    /// Elsewhere devices are made when the session is, and this does nothing.
    /// </summary>
    /// <param name="options">Adapter choice and validation, as for <see cref="Register(GpuOptions?)"/>; defaults when null.</param>
    /// <exception cref="SessionException"><see cref="SessionErrorKind.DeviceUnsupported"/>: the browser has no usable WebGPU adapter.</exception>
    public static async System.Threading.Tasks.Task PrepareDeviceAsync(GpuOptions? options = null)
    {
        if (!OperatingSystem.IsBrowser())
        {
            return;
        }

        options ??= new GpuOptions();
        try
        {
            PreparedDevices.Add(await RecursiveAI.OneGpu.GpuBackends.CreateDeviceAsync(
                new RecursiveAI.OneGpu.GpuDeviceOptions { Validation = options.Validation, Adapter = options.Adapter, ApplicationName = "RecursiveAI.OneModel" }).ConfigureAwait(true));
        }
        catch (RecursiveAI.OneGpu.GpuException e)
        {
            throw new SessionException(SessionErrorKind.DeviceUnsupported, $"No usable GPU device: {e.Message}", null, default, e);
        }
    }

    /// <summary>
    /// Runs the model like <see cref="Session.Run(IReadOnlyDictionary{string, Tensor}, IReadOnlyList{string}?)"/> without blocking the calling thread while the
    /// device works, which a browser's only thread needs: it can only read results back by returning to the event loop. Works the same on every
    /// platform, with the same results as the blocking call.
    /// </summary>
    /// <param name="session">A session created with <see cref="ComputeDevice.Gpu"/>, or any session (a CPU one completes synchronously).</param>
    /// <param name="inputs">Every model input, by name; read, never changed.</param>
    /// <param name="outputNames">The outputs wanted, or null for all in graph order.</param>
    /// <param name="cancellationToken">Checked before the run starts; a run already handed to the device is not interrupted.</param>
    /// <returns>The outputs, owned by the result; dispose it when done.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="session"/> or <paramref name="inputs"/> is null.</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was canceled before the run started.</exception>
    /// <exception cref="ObjectDisposedException">The session was disposed.</exception>
    /// <exception cref="SessionException">As <see cref="Session.Run(IReadOnlyDictionary{string, Tensor}, IReadOnlyList{string}?)"/>.</exception>
    /// <remarks>Awaited runs of one session take turns. Do not mix them with blocking runs of the same session from other threads at the same time.</remarks>
    public static async System.Threading.Tasks.ValueTask<SessionOutputs> RunAsync(Session session, IReadOnlyDictionary<string, Tensor> inputs, IReadOnlyList<string>? outputNames = null,
        System.Threading.CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(inputs);
        cancellationToken.ThrowIfCancellationRequested();
        var inNames = new string[inputs.Count];
        var ins = new Tensor[inputs.Count];
        int i = 0;
        foreach (KeyValuePair<string, Tensor> kv in inputs)
        {
            inNames[i] = kv.Key;
            ins[i++] = kv.Value;
        }

        string[] outNames;
        if (outputNames is null)
        {
            outNames = new string[session.Outputs.Length];
            for (int k = 0; k < outNames.Length; k++)
            {
                outNames[k] = session.Outputs[k].Name;
            }
        }
        else
        {
            outNames = [.. outputNames];
        }

        var outs = new Tensor[outNames.Length];
        await session.RunAsync(inNames, ins, outNames, outs).ConfigureAwait(true);
        return new SessionOutputs(outNames, outs);
    }

    /// <summary>
    /// What the GPU session has done and holds: the adapter, the kernel variants chosen, device memory and, when
    /// <see cref="GpuOptions.CollectTimings"/> is set, the time spent. Does not build a plan.
    /// </summary>
    /// <param name="session">A session created with <see cref="ComputeDevice.Gpu"/>.</param>
    /// <returns>A snapshot; later runs do not change it.</returns>
    /// <exception cref="ArgumentException">The session was not created with <see cref="ComputeDevice.Gpu"/>.</exception>
    public static GpuSessionInfo GetInfo(Session session)
    {
        ArgumentNullException.ThrowIfNull(session);
        return GpuBackendOf(session).Snapshot(null);
    }

    /// <summary>
    /// As <see cref="GetInfo(Session)"/>, plus what one run with inputs like <paramref name="inputs"/> costs on the device: arithmetic, bytes
    /// moved and device memory the plan needs. Builds and caches the plan for those shapes if there is none yet.
    /// </summary>
    /// <param name="session">A session created with <see cref="ComputeDevice.Gpu"/>.</param>
    /// <param name="inputs">Every model input by name; only element types and shapes are used.</param>
    /// <returns>A snapshot.</returns>
    /// <exception cref="ArgumentException">The session was not created with <see cref="ComputeDevice.Gpu"/>.</exception>
    /// <exception cref="SessionException">The inputs do not fit the model, or the plan could not be built.</exception>
    public static GpuSessionInfo GetInfo(Session session, IReadOnlyDictionary<string, Tensor> inputs)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(inputs);
        GpuBackend backend = GpuBackendOf(session);
        return backend.Snapshot(session.ProgramOf(inputs) as GpuProgram);
    }

    /// <summary>Sets the accumulated timings of a session to zero.</summary>
    /// <param name="session">A session created with <see cref="ComputeDevice.Gpu"/>.</param>
    /// <exception cref="ArgumentException">The session was not created with <see cref="ComputeDevice.Gpu"/>.</exception>
    public static void ResetTimings(Session session)
    {
        ArgumentNullException.ThrowIfNull(session);
        GpuBackendOf(session).ResetTimings();
    }

    private static GpuBackend GpuBackendOf(Session session) =>
        session.Backend as GpuBackend ?? throw new ArgumentException("The session was not created with ComputeDevice.Gpu.", nameof(session));
}
