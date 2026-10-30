#!/usr/bin/env python3
"""Times the ONNX files written by the tracing timing test (TraceTiming) in ONNX Runtime, for comparison with the library's session on the
same file. Warm, in-process, median of 25 runs, ORT_ENABLE_ALL, inter-op threads 1.

usage: trace_timing.py <dir>
"""
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> int:
    root = Path(sys.argv[1])
    cases = {
        "encoder": ("encoder.ids.bin", np.int64, (8, 64)),
        "convnet": ("convnet.x.bin", np.float32, (8, 3, 32, 32)),
    }
    for name, (file, dtype, shape) in cases.items():
        x = np.frombuffer((root / file).read_bytes(), dtype=dtype).reshape(shape)
        for threads in (1, 8):
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(str(root / f"{name}.onnx"), opts, providers=["CPUExecutionProvider"])
            feeds = {"input": x}
            for _ in range(5):
                sess.run(None, feeds)
            times = []
            for _ in range(25):
                t = time.perf_counter()
                sess.run(None, feeds)
                times.append((time.perf_counter() - t) * 1000)
            print(f"{name} threads={threads} ort={statistics.median(times):.2f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
