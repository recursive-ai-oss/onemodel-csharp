using System;
using RecursiveAI.OneGpu;
using RecursiveAI.OneGpu.Kernels;
using RecursiveAI.OneModel.Graph.Runtime;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Gpu;

/// <summary>The entries for MatMul and Gemm, both run by the same matrix-product kernel.</summary>
internal static class MatMulOps
{
    // ONNX operators in the default domain, opset 7 and later, have the semantics below (opset 1 to 6 Gemm broadcast C by attribute).
    public static readonly GpuOpEntry MatMul = new("MatMul", true, CheckMatMul, BindMatMul);

    public static readonly GpuOpEntry Gemm = new("Gemm", true, CheckGemm, BindGemm);

    private static string? CheckMatMul(NodeInfo info) => info.Opset < 1 ? "unknown opset" : null;

    private static string? CheckGemm(NodeInfo info) =>
        info.Opset < 7 || info.Has("broadcast") ? "Gemm before opset 7 broadcasts C by an attribute, which the GPU kernel does not do" : null;

    // ---- shared checks ----------------------------------------------------------------------------------------------------

    // The element type of a product's inputs on the device: binary32 or binary16.
    private static string? InputElement(GpuClaimContext ctx, out DeviceElement element)
    {
        element = DeviceElement.F32;
        DType a = ctx.InputType(0), b = ctx.InputType(1);
        if (a != b)
        {
            return $"A is {a} and B is {b}; the GPU kernel needs one element type";
        }

        switch (a)
        {
            case DType.F32:
                element = DeviceElement.F32;
                break;
            case DType.F16:
                element = DeviceElement.F16;
                break;
            default:
                return $"element type {a} is not supported on the GPU (float32 and float16 are)";
        }

        for (int i = 0; i < 2; i++)
        {
            if (ctx.InputShape(i).Length == 0)
            {
                return "an input is empty";
            }

            if (!(i == 1 && SplitOps.CanSplitB(ctx, element)) && (long)ctx.InputShape(i).Length * DeviceArena.ElementBytes(element) > ctx.MaxBindingBytes)
            {
                return $"input {i} is larger than one buffer binding on this device ({ctx.MaxBindingBytes} bytes)";
            }
        }

        return null;
    }

    private static string? OutputFits(GpuClaimContext ctx, long elements)
    {
        if (ctx.OutputType(0) != ctx.InputType(0))
        {
            return $"the result type {ctx.OutputType(0)} differs from the input type";
        }

        if (elements != ctx.OutputShape(0).Length)
        {
            return "the planned output shape does not match the product's";
        }

        return elements * 4 > ctx.MaxBindingBytes ? $"the result is larger than one buffer binding on this device ({ctx.MaxBindingBytes} bytes)" : null;
    }

    // ---- MatMul ------------------------------------------------------------------------------------------------------------

    // numpy matmul: the last two dimensions multiply, the leading ones broadcast. Handled: B without batch dimensions (A's rows are
    // stacked into one tall matrix), equal batch dimensions, and A without batch dimensions (one A shared by the batch of B).
    private static string? BindMatMul(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        if (ctx.InputCount != 2 || !ctx.HasInput(0) || !ctx.HasInput(1))
        {
            return "MatMul needs two inputs";
        }

        if (InputElement(ctx, out DeviceElement element) is { } why)
        {
            return why;
        }

        Shape sa = ctx.InputShape(0), sb = ctx.InputShape(1);
        long m = sa.Rank == 1 ? 1 : sa[sa.Rank - 2];
        long k = sa[sa.Rank - 1];
        long n = sb.Rank == 1 ? 1 : sb[sb.Rank - 1];
        long k2 = sb.Rank == 1 ? sb[0] : sb[sb.Rank - 2];
        if (k != k2)
        {
            return $"the inner dimensions differ ({k} and {k2})";
        }

        int batchRankA = Math.Max(sa.Rank - 2, 0), batchRankB = Math.Max(sb.Rank - 2, 0);
        long pa = 1, pb = 1;
        for (int i = 0; i < batchRankA; i++)
        {
            pa *= sa[i];
        }

        for (int i = 0; i < batchRankB; i++)
        {
            pb *= sb[i];
        }

        var p = new GemmProblem(0, 0, 0);
        long batch;
        if (pb == 1)
        {
            batch = 1;
            m *= pa;
        }
        else if (SameBatch(sa, batchRankA, sb, batchRankB))
        {
            batch = pb;
            p.StrideA = Checked(m * k);
            p.StrideB = Checked(k * n);
            p.StrideC = Checked(m * n);
        }
        else if (pa == 1)
        {
            batch = pb;
            p.StrideA = 0;
            p.StrideB = Checked(k * n);
            p.StrideC = Checked(m * n);
        }
        else
        {
            return "the batch dimensions broadcast in a pattern the GPU kernel does not handle";
        }

        long outElements = checked(batch * m * n);
        if (OutputFits(ctx, outElements) is { } bad)
        {
            return bad;
        }

        if (Math.Max(m, Math.Max(n, k)) > int.MaxValue || batch > int.MaxValue)
        {
            return "a dimension does not fit 32 bits";
        }

        p.M = (int)m;
        p.N = (int)n;
        p.K = (int)k;
        p.Batch = (int)batch;
        p.Alpha = 1f;
        p.Beta = 0f;
        if (SplitOps.IsOversized(ctx, 1, element))
        {
            return SplitOps.BindProduct(ctx, p, false, out binding);
        }

        binding = new GemmBinding(ctx, element, p, null, -1, 0, 0);
        return null;
    }

    private static bool SameBatch(Shape a, int rankA, Shape b, int rankB)
    {
        int rank = Math.Max(rankA, rankB);
        for (int i = 0; i < rank; i++)
        {
            int da = i < rank - rankA ? 1 : a[i - (rank - rankA)];
            int db = i < rank - rankB ? 1 : b[i - (rank - rankB)];
            if (da != db)
            {
                return false;
            }
        }

        return true;
    }

    private static int Checked(long v) => v > int.MaxValue ? throw new GpuUnsupportedException("a matrix has more elements than a kernel can index", "ElementIndex32", "at most 2147483647 elements", $"{v} elements") : (int)v;

    // ---- Gemm --------------------------------------------------------------------------------------------------------------

    // Y = alpha * A' * B' + beta * C with A' and B' optionally transposed and C broadcast to [M, N] (a scalar, [N], [1, N], [M, 1] or [M, N]).
    private static string? BindGemm(GpuClaimContext ctx, out GpuBinding? binding)
    {
        binding = null;
        if (!ctx.HasInput(0) || !ctx.HasInput(1))
        {
            return "Gemm needs A and B";
        }

        if (InputElement(ctx, out DeviceElement element) is { } why)
        {
            return why;
        }

        Shape sa = ctx.InputShape(0), sb = ctx.InputShape(1);
        if (sa.Rank != 2 || sb.Rank != 2)
        {
            return "Gemm inputs must be matrices";
        }

        bool transA = ctx.Info.GetInt("transA", 0) != 0, transB = ctx.Info.GetInt("transB", 0) != 0;
        float alpha = ctx.Info.GetFloat("alpha", 1f), beta = ctx.Info.GetFloat("beta", 1f);
        int m = transA ? sa[1] : sa[0], k = transA ? sa[0] : sa[1];
        int n = transB ? sb[0] : sb[1], k2 = transB ? sb[1] : sb[0];
        if (k != k2)
        {
            return $"the inner dimensions differ ({k} and {k2})";
        }

        if (OutputFits(ctx, (long)m * n) is { } bad)
        {
            return bad;
        }

        var p = new GemmProblem(m, n, k) { Batch = 1, TransposeA = transA, TransposeB = transB, Alpha = alpha, Beta = 0f };
        bool hasC = ctx.HasInput(2) && beta != 0f;
        if (hasC && SplitOps.IsOversized(ctx, 1, element))
        {
            // Refused before the bias is uploaded, so nothing is left on the device.
            return SplitOps.BindProduct(ctx, p, true, out binding);
        }

        int cId = -1, cRows = 0, cCols = 0;
        DeviceConstant? cInit = null;
        if (hasC)
        {
            Shape sc = ctx.InputShape(2);
            if (ctx.InputType(2) != ctx.InputType(0))
            {
                return $"C is {ctx.InputType(2)} but A is {ctx.InputType(0)}";
            }

            int rows = cRows = sc.Rank == 2 ? sc[0] : 1;
            int cols = cCols = sc.Rank >= 1 ? sc[sc.Rank - 1] : 1;
            if (sc.Rank > 2 || (rows != 1 && rows != m) || (cols != 1 && cols != n) || sc.Length == 0)
            {
                return $"C of shape {sc} does not broadcast to [{m}, {n}]";
            }

            if (ctx.InputMadeOnDevice(2) is not null)
            {
                return "C is computed on the device, and the kernel can only start from a bias that is a weight or comes from the host";
            }

            p.Beta = beta;
            if (ctx.InputIsConst(2))
            {
                using Tensor c = Expand(ctx.InputConst(2), rows, cols, m, n);
                cInit = ctx.Backend.Constants.Upload(c);
                ctx.OwnForPlan(cInit);
            }
            else
            {
                cId = ctx.InputId(2);
            }
        }

        if (SplitOps.IsOversized(ctx, 1, element))
        {
            return SplitOps.BindProduct(ctx, p, hasC, out binding);
        }

        binding = new GemmBinding(ctx, element, p, cInit, cId, cRows, cCols);
        return null;
    }

    // C broadcast to [m, n] as a float32 tensor, for a kernel that starts from the old contents of its output.
    internal static Tensor Expand(in Tensor c, int rows, int cols, int m, int n)
    {
        Tensor src = c.IsContiguous ? c : c.Contiguous();
        try
        {
            var result = Tensor.Create(DType.F32, Shape.Create([m, n]));
            Span<float> dst = result.MutableData<float>();
            ExpandInto(src, rows, cols, m, n, dst);
            return result;
        }
        finally
        {
            if (!ReferenceEquals(src.Buffer, c.Buffer))
            {
                src.Dispose();
            }
        }
    }

    internal static void ExpandInto(in Tensor src, int rows, int cols, int m, int n, Span<float> dst)
    {
        for (int i = 0; i < m; i++)
        {
            int r = rows == 1 ? 0 : i;
            for (int j = 0; j < n; j++)
            {
                int col = cols == 1 ? 0 : j;
                int at = r * cols + col;
                dst[i * n + j] = src.DType == DType.F16 ? (float)src.Data<Half>()[at] : src.Data<float>()[at];
            }
        }
    }
}
