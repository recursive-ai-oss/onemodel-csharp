using System;
using System.Runtime.Intrinsics;
using RecursiveAI.OneModel.Threading;

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Nearest-neighbour Resize in one pass. Nearest only selects elements, so resampling every axis in turn and picking the source
/// element of each output position along all axes at once give the same bits; the single pass reads the input once and writes the
/// output once instead of writing an intermediate per axis. An output row (all but the last axis fixed) is a copy of one input row,
/// gathered by the last axis's index table; a row that comes from the same input row as the one before it (every second row of a
/// 2x upsample) is copied from that finished output row. Rows are split into chunks of about 16 384 output elements, a function of the
/// shape alone, and every output element is written by exactly one chunk, so the result cannot depend on the thread count.
/// </summary>
internal static unsafe class ResizeGather
{
    private const int ChunkElements = 16384;

    /// <summary>
    /// Fills <paramref name="y"/> from <paramref name="x"/> using one index table per axis.
    /// </summary>
    /// <param name="inDims">Input extents (rank 1 to 8).</param>
    /// <param name="outDims">Output extents, all positive.</param>
    /// <param name="axisIdx">The tables, concatenated in axis order: axis a has <c>outDims[a]</c> source indices in <c>[0, inDims[a])</c>.</param>
    /// <param name="x">Input, at least the product of <paramref name="inDims"/> elements.</param>
    /// <param name="y">Output, at least the product of <paramref name="outDims"/> elements.</param>
    /// <param name="threads">Upper bound on threads.</param>
    public static void Run(ReadOnlySpan<int> inDims, ReadOnlySpan<int> outDims, ReadOnlySpan<int> axisIdx, ReadOnlySpan<float> x, Span<float> y, int threads)
    {
        int rank = inDims.Length;
        if (rank < 1 || rank > 8 || outDims.Length != rank)
        {
            throw new ArgumentException("Rank must be 1 to 8 and equal for input and output.");
        }

        // Checked extents, and every table entry inside its axis, before any pointer is formed.
        long inLen = 1, outLen = 1, tableLen = 0;
        for (int a = 0; a < rank; a++)
        {
            if (inDims[a] <= 0 || outDims[a] <= 0)
            {
                throw new ArgumentException("Extents must be positive.");
            }

            inLen = checked(inLen * inDims[a]);
            outLen = checked(outLen * outDims[a]);
            tableLen += outDims[a];
        }

        if (x.Length < inLen || y.Length < outLen || axisIdx.Length < tableLen)
        {
            throw new ArgumentException("A buffer is smaller than the resize shape needs.");
        }

        int at = 0;
        for (int a = 0; a < rank; a++)
        {
            for (int j = 0; j < outDims[a]; j++)
            {
                if ((uint)axisIdx[at + j] >= (uint)inDims[a])
                {
                    throw new ArgumentException("An index table entry is outside its axis.");
                }
            }

            at += outDims[a];
        }

        int last = outDims[rank - 1];
        long rows = outLen / last;
        int rowsPerChunk = Math.Max(1, ChunkElements / last);
        int chunks = (int)((rows + rowsPerChunk - 1) / rowsPerChunk);
        fixed (float* px = x, py = y)
        fixed (int* pi = axisIdx)
        {
            GatherJob job = GatherJob.Get();
            job.Setup(rank, inDims, outDims, pi, px, py, rowsPerChunk, rows);
            // About a nanosecond per output element, a fair estimate for a gather; it only decides how many threads are worth waking.
            ParallelFor.Run(job, chunks, threads, outLen);
            job.Release();
        }
    }

    private sealed class GatherJob : ChunkJob
    {
        // Kept per thread and refilled, so a steady-state call allocates nothing; a call is done with it before it returns.
        [ThreadStatic]
        private static GatherJob? t_job;

        private readonly int[] _outDims = new int[8];
        private readonly int[] _tableAt = new int[8];
        private readonly long[] _inStride = new long[8];
        private int _rank;
        private float* _x;
        private float* _y;
        private int* _idx;
        private int _rowsPerChunk;
        private long _rows;
        private bool _lastIdentity;
        private bool _lastDouble;
        private int _inLast;

        public static GatherJob Get() => t_job ??= new GatherJob();

        public void Setup(int rank, ReadOnlySpan<int> inDims, ReadOnlySpan<int> outDims, int* idx, float* x, float* y, int rowsPerChunk, long rows)
        {
            _rank = rank;
            long stride = 1;
            int at = 0;
            for (int a = 0; a < rank; a++)
            {
                _outDims[a] = outDims[a];
                _tableAt[a] = at;
                at += outDims[a];
            }

            for (int a = rank - 1; a >= 0; a--)
            {
                _inStride[a] = stride;
                stride *= inDims[a];
            }

            _x = x;
            _y = y;
            _idx = idx;
            _rowsPerChunk = rowsPerChunk;
            _rows = rows;
            // The last axis copies whole when every output position maps to itself.
            bool identity = inDims[rank - 1] == outDims[rank - 1];
            for (int j = 0; identity && j < outDims[rank - 1]; j++)
            {
                identity = idx[_tableAt[rank - 1] + j] == j;
            }

            _lastIdentity = identity;
            _inLast = inDims[rank - 1];

            // The last axis doubles every element (output j reads input j / 2), the usual 2x upsample.
            bool doubled = outDims[rank - 1] == 2 * inDims[rank - 1];
            for (int j = 0; doubled && j < outDims[rank - 1]; j++)
            {
                doubled = idx[_tableAt[rank - 1] + j] == (j >> 1);
            }

            _lastDouble = doubled;
        }

        public void Release()
        {
            _x = null;
            _y = null;
            _idx = null;
        }

        // d[2i] = d[2i + 1] = s[i]: four inputs become eight outputs with two shuffles. Data only moves, so any width gives the same bits.
        private static void Double(float* s, float* d, int count)
        {
            int i = 0;
            for (; i + 4 <= count; i += 4)
            {
                Vector128<float> v = Vector128.Load(s + i);
                Vector128.Shuffle(v, Vector128.Create(0, 0, 1, 1)).Store(d + (2 * i));
                Vector128.Shuffle(v, Vector128.Create(2, 2, 3, 3)).Store(d + (2 * i) + 4);
            }

            for (; i < count; i++)
            {
                d[2 * i] = s[i];
                d[(2 * i) + 1] = s[i];
            }
        }

        public override void RunChunk(int chunk)
        {
            int last = _outDims[_rank - 1];
            long first = (long)chunk * _rowsPerChunk;
            long end = Math.Min(_rows, first + _rowsPerChunk);
            int* lastIdx = _idx + _tableAt[_rank - 1];
            long prevSrc = -1, prevDst = -1;
            for (long r = first; r < end; r++)
            {
                // Decode the row number into positions along the leading axes and sum the source offsets.
                long rem = r, src = 0;
                for (int a = _rank - 2; a >= 0; a--)
                {
                    long q = rem / _outDims[a];
                    int j = (int)(rem - (q * _outDims[a]));
                    rem = q;
                    src += _idx[_tableAt[a] + j] * _inStride[a];
                }

                float* d = _y + (r * last);
                if (src == prevSrc)
                {
                    new ReadOnlySpan<float>(_y + prevDst, last).CopyTo(new Span<float>(d, last));
                }
                else if (_lastIdentity)
                {
                    new ReadOnlySpan<float>(_x + src, last).CopyTo(new Span<float>(d, last));
                }
                else if (_lastDouble)
                {
                    Double(_x + src, d, _inLast);
                }
                else
                {
                    float* s = _x + src;
                    for (int j = 0; j < last; j++)
                    {
                        d[j] = s[lastIdx[j]];
                    }
                }

                prevSrc = src;
                prevDst = r * last;
            }
        }
    }
}
