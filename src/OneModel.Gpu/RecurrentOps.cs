using System;
using System.Collections.Generic;
using RecursiveAI.OneGpu;
using RecursiveAI.OneGpu.Kernels;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Graph.Operators;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Kernels;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu;

/// <summary>
/// ONNX <c>LSTM</c>, <c>GRU</c> and <c>RNN</c> on the device: forward, reverse and bidirectional, both layouts, bias, initial states,
/// peepholes, <c>clip</c>, <c>input_forget</c>, <c>linear_before_reset</c>, sequence lengths and every activation, float32 tensors. The
/// weights are used as stored. The attributes come from the CPU operator itself (activation names with alpha and beta resolved), so both
/// executors agree on what a layer means. The kernel's accumulation order is fixed by the shapes, so one entry serves both modes.
/// </summary>
internal static class RecurrentOps
{
    public static IEnumerable<GpuOpEntry> Entries
    {
        get
        {
            foreach (string name in new[] { "LSTM", "GRU", "RNN" })
            {
                yield return new GpuOpEntry(name, true, _ => null, Bind);
            }
        }
    }

    private static GpuRecurrentActivation Map(RecurrentActivation a) => new((GpuRecurrentActivationKind)(int)a.Kind, a.Alpha, a.Beta);

    private static GpuRecurrentActivations ActivationsOf(RecurrentOp op, int dir)
    {
        int per = op.ActivationsPerDirection, a0 = dir * per;
        ReadOnlySpan<RecurrentActivation> acts = op.Activations;
        GpuRecurrentActivation f = Map(acts[a0]);
        GpuRecurrentActivation g = per > 1 ? Map(acts[a0 + 1]) : f;
        GpuRecurrentActivation h = per > 2 ? Map(acts[a0 + 2]) : g;
        return new GpuRecurrentActivations(f, g, h);
    }

    private static string? Bind(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        if (ctx.Step.Node.Kernel is not RecurrentOp op)
        {
            return "not a recurrent layer";
        }

        if (!ctx.HasInput(0) || !ctx.HasInput(1) || !ctx.HasInput(2))
        {
            return "needs X, W and R";
        }

        for (int i = 0; i < 8; i++)
        {
            if (!ctx.HasInput(i))
            {
                continue;
            }

            DType want = i == 4 ? DType.I32 : DType.F32;
            if (ctx.InputType(i) != want)
            {
                return $"input {i} is {ctx.InputType(i)}, and the GPU kernel takes {want} here (float16 layers stay on the CPU)";
            }

            if (ctx.InputMadeOnDevice(i) is { } made && made != (i == 4 ? DeviceElement.I32 : DeviceElement.F32))
            {
                return $"input {i} was made on the device as another type";
            }
        }

        Shape x = ctx.InputShape(0), r = ctx.InputShape(2);
        GpuRecurrentCell cell = op.Cell switch { RecurrentCell.Lstm => GpuRecurrentCell.Lstm, RecurrentCell.Gru => GpuRecurrentCell.Gru, _ => GpuRecurrentCell.Rnn };
        int seq = op.Layout1 ? x[1] : x[0], batch = op.Layout1 ? x[0] : x[1];
        if (seq == 0 || batch == 0)
        {
            return "an empty sequence or batch (the CPU writes the initial states)";
        }

        var problem = new RecurrentProblem(cell, seq, batch, x[2], r[2])
        {
            Direction = op.Directions == 2 ? GpuRecurrentDirection.Bidirectional : op.ReverseOnly ? GpuRecurrentDirection.Reverse : GpuRecurrentDirection.Forward,
            Layout1 = op.Layout1,
            Clip = op.Clip,
            InputForget = op.InputForget,
            LinearBeforeReset = op.LinearBeforeReset,
        };

        problem.Activations = ActivationsOf(op, 0);
        if (op.Directions == 2)
        {
            problem.ReverseActivations = ActivationsOf(op, 1);
        }

        // Lengths are not validated on the device: a constant is checked here, a value from the host on every run.
        int hostLengths = -1;
        if (ctx.HasInput(4))
        {
            if (ctx.InputIsConst(4))
            {
                ReadOnlySpan<int> lens = ctx.InputConst(4).Data<int>();
                int max = 0;
                foreach (int l in lens)
                {
                    if (l < 0 || l > seq)
                    {
                        return "a constant sequence length is out of range, which the CPU reports";
                    }

                    max = Math.Max(max, l);
                }

                problem.MaxLength = max;
            }
            else if (ctx.InputMadeOnDevice(4) is null)
            {
                hostLengths = ctx.InputId(4);
            }
            else
            {
                return "sequence_lens is made on the device, and lengths are validated on the host";
            }
        }

        long xw = RecurrentKernels.XwLength(problem), state = RecurrentKernels.StateLength(problem);
        if ((ctx.FitsBinding(xw, DeviceElement.F32) ?? ctx.FitsBinding(state, DeviceElement.F32) ?? ctx.FitsBinding((long)seq * op.Directions * batch * r[2], DeviceElement.F32)
            ?? ctx.FitsBinding(x.Length, DeviceElement.F32)) is { } big)
        {
            return big;
        }

        binding = new RecurrentBinding(ctx, problem, hostLengths, seq, xw, state);
        return null;
    }

    private sealed class RecurrentBinding : GpuBinding
    {
        private readonly RecurrentProblem _problem;
        private readonly Operand?[] _in = new Operand?[8];
        private readonly int[] _out = [-1, -1, -1];
        private readonly int _hostLengths, _seq;

        // The input projection and the state share one arena range, the state starting on a 64-byte boundary; where a buffer cannot be bound
        // for reading and writing in one dispatch, the state gets a range of its own (the projection is read while the state is written).
        private readonly long _xwElements, _stateElements;
        private readonly bool _separate;

        public RecurrentBinding(GpuClaimContext ctx, RecurrentProblem problem, int hostLengths, int seq, long xwElements, long stateElements)
        {
            _xwElements = (xwElements + 15) / 16 * 16;
            _stateElements = stateElements;
            _separate = ctx.Backend.SeparateReadWriteSlabs;
            _problem = problem;
            _hostLengths = hostLengths;
            _seq = seq;
            for (int i = 0; i < 8; i++)
            {
                if (ctx.HasInput(i))
                {
                    _in[i] = Operand.Of(ctx, i);
                }
            }

            for (int o = 0; o < 3 && o < ctx.Step.Node.Outputs.Length; o++)
            {
                _out[o] = ctx.OutputId(o);
            }

            // Projection of every step plus the per-step recurrent products.
            long gh = (long)problem.Gates * problem.Hidden;
            Flops = 2.0 * problem.Seq * problem.Batch * problem.Directions * gh * (problem.Input + problem.Hidden);
        }

        public override DeviceElement OutputElement(int output) => DeviceElement.F32;

        public override long ScratchF32Elements => _separate ? _xwElements : _xwElements + _stateElements;

        public override long ScratchF32bElements => _separate ? _stateElements : 0;

        public override void Upload(GpuRunContext ctx, Tensor[] values)
        {
            if (_hostLengths < 0)
            {
                return;
            }

            Tensor t = values[_hostLengths];
            Tensor src = t.IsContiguous ? t : t.Contiguous();
            try
            {
                foreach (int l in src.Data<int>())
                {
                    if (l < 0 || l > _seq)
                    {
                        throw new SessionException(SessionErrorKind.InvalidInput, $"sequence_lens contains {l}, outside [0, {_seq}].");
                    }
                }
            }
            finally
            {
                if (!ReferenceEquals(src.Buffer, t.Buffer))
                {
                    src.Dispose();
                }
            }
        }

        private GpuTensorRef<float>? Ref(GpuRunContext ctx, int i)
        {
            if (_in[i] is not { } op)
            {
                return null;
            }

            GpuBuffer<float> b = op.Get<float>(ctx, out int off);
            return new GpuTensorRef<float>(b, off);
        }

        private GpuTensorRef<float>? Out(GpuRunContext ctx, int o)
        {
            if (_out[o] < 0)
            {
                return null;
            }

            GpuBuffer<float> b = ctx.F32(_out[o], out int off);
            return new GpuTensorRef<float>(b, off);
        }

        public override void Encode(GpuRunContext ctx)
        {
            GpuTensorRef<int>? lens = null;
            if (_in[4] is { } l)
            {
                lens = new GpuTensorRef<int>(l.Get<int>(ctx, out int offL), offL);
            }

            var operands = new RecurrentOperands<float>
            {
                X = Ref(ctx, 0)!.Value,
                W = Ref(ctx, 1)!.Value,
                R = Ref(ctx, 2)!.Value,
                B = Ref(ctx, 3),
                SequenceLengths = lens,
                InitialH = Ref(ctx, 5),
                InitialC = Ref(ctx, 6),
                P = Ref(ctx, 7),
                Y = Out(ctx, 0),
                YH = Out(ctx, 1),
                YC = Out(ctx, 2),
            };
            GpuBuffer<float> scratch = ctx.Scratch(this, out int offScratch);
            if (_separate)
            {
                GpuBuffer<float> state = ctx.ScratchF32b(this, out int offState);
                ctx.Backend.Recurrent.Encode(ctx.Encoder, _problem, operands, scratch, state, offScratch, offState);
            }
            else
            {
                ctx.Backend.Recurrent.Encode(ctx.Encoder, _problem, operands, scratch, scratch, offScratch, checked(offScratch + (int)_xwElements));
            }
        }
    }
}
