#!/usr/bin/env python3
"""Runs one case file through a released ONNX Runtime execution provider and times it.

The case file is the one tools/GpuReport writes with --export (inputs and the CPU exact outputs). ONNX Runtime is used as a black box:
the released wheel is installed and run (tools/bar/setup_ort.sh), never opened.

  ort_run.py --provider webgpu|coreml|cuda|dml|cpu --model FILE --case case.bin --out result.json [--variant NAME] [--runs N] [--warmups N]

A variant is one session configuration of the provider (see VARIANTS). The result holds the median and minimum of the timed warm runs
(wall time of session.run with host arrays in and host arrays out, which is what a caller pays and what GpuReport's wall time measures),
the first run, session creation, the scaled error against the case's outputs, and the versions and options that ran.
"""
import argparse
import os
import json
import platform
import statistics
import sys
import time

import numpy as np
import onnxruntime as ort

# Provider name as ONNX Runtime reports it, and the session configurations tried for it. The best of a provider's variants that runs the
# model is the reference time; every variant's time is recorded.
VARIANTS = {
    "webgpu": {
        "default": {},
        "nhwc": {"preferredLayout": "NHWC"},
        "nchw": {"preferredLayout": "NCHW"},
    },
    "coreml": {
        "gpu-mlprogram": {"MLComputeUnits": "CPUAndGPU", "ModelFormat": "MLProgram"},
        "gpu-neuralnetwork": {"MLComputeUnits": "CPUAndGPU", "ModelFormat": "NeuralNetwork"},
        "all-mlprogram": {"MLComputeUnits": "ALL", "ModelFormat": "MLProgram"},
        # Only nodes whose shapes are fully known at session creation: the models are exported with a dynamic batch, and the provider
        # fails at run time on some of them for batch > 1 otherwise.
        "gpu-mlprogram-static": {"MLComputeUnits": "CPUAndGPU", "ModelFormat": "MLProgram", "RequireStaticInputShapes": "1"},
        "gpu-neuralnetwork-static": {"MLComputeUnits": "CPUAndGPU", "ModelFormat": "NeuralNetwork", "RequireStaticInputShapes": "1"},
    },
    "cuda": {"default": {}},
    "dml": {"default": {}},
    "cpu": {"default": {}},
}
PROVIDER_NAME = {
    "webgpu": "WebGpuExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "dml": "DmlExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def read_case(path):
    raw = open(path, "rb").read()
    n = int.from_bytes(raw[:4], "little")
    meta = json.loads(raw[4:4 + n])
    start = 4 + n
    feeds, expected = {}, {}
    for e in meta["entries"]:
        blob = raw[start + e["offset"]:start + e["offset"] + e["bytes"]]
        arr = np.frombuffer(blob, dtype=np.int64 if e["dtype"] == "i64" else np.float32).reshape(e["dims"])
        (feeds if e["role"] == "input" else expected)[e["name"]] = arr
    return meta, feeds, expected


def make_session(provider, opts, model, verbose=False):
    so = ort.SessionOptions()
    so.log_severity_level = 0 if verbose else 3  # errors only otherwise: the CUDA provider warns about shape-op placement on every session
    available = ort.get_available_providers()
    if provider == "webgpu" and PROVIDER_NAME[provider] not in available:
        # The released plugin package: registered at run time next to the core wheel.
        import onnxruntime_ep_webgpu as ep
        ort.register_execution_provider_library("webgpu_ep_registration", ep.get_library_path())
        devices = [d for d in ort.get_ep_devices() if d.ep_name == ep.get_ep_name()]
        if not devices:
            raise RuntimeError("no WebGPU EP device found")
        # The factory takes one device. With several GPUs the discrete one is the comparison (it is the one our Vulkan run uses).
        devices = [d for d in devices if d.device.metadata.get("Discrete") == "1"] or devices[:1]
        so.add_provider_for_devices(devices[:1], opts)
        return ort.InferenceSession(model, sess_options=so), "plugin onnxruntime-ep-webgpu"
    if PROVIDER_NAME[provider] not in available:
        raise RuntimeError(f"{PROVIDER_NAME[provider]} not in {available}")
    if provider == "cuda":
        ort.preload_dlls()  # the CUDA and cuDNN libraries the wheel's extras installed
    if provider == "cpu":
        so.intra_op_num_threads = 8  # the same thread count as the onemodel CPU row, so a shared host does not oversubscribe it
    providers = [(PROVIDER_NAME[provider], opts)] if provider != "cpu" else ["CPUExecutionProvider"]
    return ort.InferenceSession(model, sess_options=so, providers=providers), "core wheel"


# Nodes that move or describe data rather than compute: a runtime may leave these on the CPU without the model having left the GPU in any way
# that matters for a timing comparison. Anything else is a compute node. ONNX names, then the Core ML program names for the compute plan.
METADATA_OPS = {"Shape", "Size", "Reshape", "Flatten", "Squeeze", "Unsqueeze", "Identity", "Constant", "ConstantOfShape", "Cast", "Concat", "Gather",
                "Slice", "Expand", "Range", "Equal", "Where", "Less", "Greater", "GreaterOrEqual", "LessOrEqual", "Not", "And", "Or", "Transpose",
                "Split", "Tile"}
METADATA_MIL = {"shape", "reshape", "squeeze", "expand_dims", "identity", "const", "cast", "concat", "slice_by_index", "slice_by_size", "transpose",
                "gather", "gather_along_axis", "gather_nd", "tile", "split", "select", "equal", "not_equal", "less", "greater", "greater_equal",
                "less_equal", "logical_and", "logical_or", "logical_not", "range_1d", "fill", "fill_like", "non_zero"}


# Mask and shape arithmetic. A node left on the CPU provider that does only arithmetic on tensor dimensions, or on the attention-mask input (batch x
# sequence) and dimensions before the mask is broadcast into attention (the exports build flat gather indices into the mask with an Add and a Mul),
# is shape or mask arithmetic, not compute: a run that leaves only those on the CPU is still a GPU run. The rule is as narrow as it can be and is decided
# from the model's graph, by the nodes' inputs and never by their names:
#   - a value is "a dimension" when it is the output of a Shape node (of any tensor) or is computed only from dimensions and constants (an initializer or
#     a Constant node) by data-movement operators (METADATA_OPS: Gather, Cast, Reshape, Concat, Range, ...) or by the arithmetic below;
#   - the mask input is a graph input whose name contains "mask" (the exports name it attention_mask); a value is "from the mask" when it is the mask
#     input or is computed from one that is, through the same operators, with all its other inputs dimensions or constants;
#   - a node is shape or mask arithmetic when its operator is Add, Sub, Mul, Div or Neg and every non-constant input is a dimension or from the mask;
#   - nothing else is: arithmetic with any other input (activations, weights, any other graph input's values) is compute.
SHAPE_ARITHMETIC = {"Add", "Sub", "Mul", "Div", "Neg"}
MASK_ARITHMETIC = SHAPE_ARITHMETIC


def read_graph(path):
    """(inputs, initializer names, nodes) of an ONNX file from its wire format: node = (op_type, name, [inputs], [outputs]). Only the structure
    is read; tensors are skipped without being copied. The ONNX file format is public (onnx.proto)."""
    import mmap

    def varint(b, i):
        n = shift = 0
        while True:
            c = b[i]
            i += 1
            n |= (c & 0x7F) << shift
            if c < 0x80:
                return n, i
            shift += 7

    def fields(b, i, end):
        while i < end:
            key, i = varint(b, i)
            f, w = key >> 3, key & 7
            if w == 0:
                v, i = varint(b, i)
                yield f, v
            elif w == 2:
                n, i = varint(b, i)
                yield f, (i, i + n)
                i += n
            elif w == 1:
                i += 8
            elif w == 5:
                i += 4
            else:
                raise ValueError("unsupported wire type")

    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as b:
        graph = next(span for f, span in fields(b, 0, len(b)) if f == 7)
        inputs, inits, nodes = [], set(), []
        for f, span in fields(b, graph[0], graph[1]):
            if f == 11:
                inputs.append(next(bytes(b[x:y]).decode() for g, (x, y) in fields(b, span[0], span[1]) if g == 1))
            elif f == 5:
                for g, v in fields(b, span[0], span[1]):
                    if g == 8:
                        inits.add(bytes(b[v[0]:v[1]]).decode())
            elif f == 1:
                ins, outs, name, op = [], [], "", ""
                for g, (x, y) in fields(b, span[0], span[1]):
                    if g not in (1, 2, 3, 4):
                        continue
                    t = bytes(b[x:y]).decode()
                    if g == 1:
                        ins.append(t)
                    elif g == 2:
                        outs.append(t)
                    elif g == 3:
                        name = t
                    elif g == 4:
                        op = t
                nodes.append((op, name, ins, outs))
        return inputs, inits, nodes


def mask_arithmetic_nodes(model):
    """Names of the nodes of `model` that are shape or mask arithmetic by the rule above."""
    inputs, inits, nodes = read_graph(model)
    constants = set(inits) | {o for op, _, _, outs in nodes if op == "Constant" for o in outs}
    mask = {n for n in inputs if "mask" in n.lower() and n not in constants}
    dims, found = set(), set()
    for op, name, ins, outs in nodes:  # ONNX nodes are in topological order
        live = [t for t in ins if t and t not in constants]
        if op == "Shape":
            dims.update(outs)
            if live and live[0] in mask:
                mask.update(outs)
        elif live and all(t in mask or t in dims for t in live) and (op in METADATA_OPS or op in SHAPE_ARITHMETIC):
            dims.update(outs)
            if any(t in mask for t in live):
                mask.update(outs)
            if op in SHAPE_ARITHMETIC:
                found.add(name)
    return found


def placement(a):
    """Where ONNX Runtime put the model, read from the session's own verbose log (in a fresh process, since it writes to stderr; the
    runtime stays a black box). Returns
      byProvider        {provider: node count}, what the session placement log states;
      cpuOps            ONNX operators left to the CPU provider, with counts (the log lists them one per line);
      computeOnCpu      how many of those are compute nodes (everything outside METADATA_OPS);
      coreMlPlan        Core ML only: the compute plan Core ML reports for its own program (option ProfileComputePlan), operations by device
                        {"GPU": n, "CPU": n, "NeuralEngine": n}, and the same for compute operations only (coreMlComputePlan).
    A provider that takes only some nodes leaves the rest to the CPU, which is what a user gets, and the reader of a ratio needs to know."""
    import re
    import subprocess
    p = subprocess.run([sys.executable, __file__, "--provider", a.provider, "--variant", a.variant_name, "--model", a.model, "--case", a.case,
                        "--out", os.devnull, "--placement-only"], capture_output=True)

    def decode(raw):  # on Windows the runtime writes its log as UTF-16 when stderr is a pipe
        return raw.decode("utf-16-le" if b"\x00" in raw[:400] else "utf-8", errors="replace")

    text = decode(p.stderr) + decode(p.stdout)
    counts, cpu_ops, current, cpu_nodes = {}, {}, None, []
    for line in text.splitlines():
        m = re.search(r"(?:Node\(s\) placed on|All nodes placed on) \[(\w+)\]\. Number of nodes: (\d+)", line)
        if m:
            current = m.group(1)
            counts[current] = counts.get(current, 0) + int(m.group(2))
            continue
        m = re.search(r"VerifyEachNodeIsAssignedToAnEp\]\s+(\w+) \((.*)\)\s*$", line)
        if m and current == "CPUExecutionProvider":
            cpu_ops[m.group(1)] = cpu_ops.get(m.group(1), 0) + 1
            cpu_nodes.append((m.group(1), m.group(2)))
    if not counts:
        raise RuntimeError("no placement lines in the log")
    compute = [(o, n) for o, n in cpu_nodes if o not in METADATA_OPS]
    mask = mask_arithmetic_nodes(a.model) if compute else set()
    # The log's node names only join the log to the graph; whether a node is mask arithmetic is decided by its inputs there.
    on_mask = [(o, n) for o, n in compute if n in mask]
    d = {"byProvider": counts, "cpuOps": cpu_ops, "computeOnCpu": len(compute) - len(on_mask),
         "maskArithmeticOnCpu": len(on_mask), "maskArithmeticOps": sorted({o for o, _ in on_mask}), "cpuComputeOps": {}}
    for o, n in compute:
        if n not in mask:
            d["cpuComputeOps"][o] = d["cpuComputeOps"].get(o, 0) + 1
    if a.provider == "coreml":
        plan, comp = {}, {}
        for m in re.finditer(r"Operation: ([\w.]+), Device Usage: <ML(\w+?)ComputeDevice", text):
            dev = m.group(2)
            plan[dev] = plan.get(dev, 0) + 1
            if m.group(1).split(".")[-1] not in METADATA_MIL:
                comp[dev] = comp.get(dev, 0) + 1
        d["coreMlPlan"], d["coreMlComputePlan"] = plan, comp
    return d


def classify(provider, r):
    """A row is a gate row only when the reference ran every compute node on its GPU device. Returns (gate, reason)."""
    pl = r.get("placement")
    if not isinstance(pl, dict):
        return False, "placement unavailable: " + str(r.get("nodesByProvider"))
    if provider == "cpu":
        return False, "CPU provider"
    if pl["computeOnCpu"] > 0:
        ops = ", ".join(f"{o} x{n}" for o, n in sorted(pl["cpuComputeOps"].items(), key=lambda kv: -kv[1]))
        return False, f"{pl['computeOnCpu']} compute nodes left to the CPU provider ({ops})"
    if provider == "coreml":
        comp = pl.get("coreMlComputePlan", {})
        total = sum(comp.values())
        if total == 0:
            return False, "no Core ML compute plan reported"
        other = total - comp.get("GPU", 0)
        if other > 0:
            where = ", ".join(f"{k} {v}" for k, v in sorted(comp.items()))
            return False, f"{other} of {total} compute operations are not on the GPU in Core ML's plan ({where})"
    extra = f", {pl['maskArithmeticOnCpu']} shape or mask arithmetic nodes on the CPU" if pl.get("maskArithmeticOnCpu") else ""
    return True, "every compute node on the provider's GPU device" + extra


def pick(a, results):
    """Reads each variant's placement and classifies it; returns (the reference variant, the fastest variant on the provider at all)."""
    ok = sorted((r for r in results if "medianMs" in r), key=lambda r: r["medianMs"])
    best = anyBest = gateBest = None
    # Fastest first. Placement is read for every variant that ran (one more process each): the reference is the fastest variant that kept every
    # compute node on the provider's GPU device when there is one (a gate row), else the fastest GPU-only variant (an information row).
    for r in ok:
        if a.provider != "cpu":
            try:
                a.variant_name = r["variant"]
                r["placement"] = placement(a)
                r["nodesByProvider"] = r["placement"]["byProvider"]
            except Exception as e:
                r["nodesByProvider"] = "unavailable: " + (str(e).strip().splitlines() or [repr(e)])[0][:200]
        nb = r.get("nodesByProvider")
        # A variant that left every node to the CPU provider (a provider can decline a whole model) is a CPU time, not a GPU one.
        r["onProvider"] = a.provider == "cpu" or bool(isinstance(nb, dict) and nb.get(PROVIDER_NAME[a.provider], 0) > 0)
        # CoreML's "ALL" compute units may use the Neural Engine, which is not the GPU: such a variant is reported, never the reference.
        r["gpuOnly"] = r["options"].get("MLComputeUnits", "CPUAndGPU") != "ALL"
        r["gateRow"], r["gateReason"] = classify(a.provider, r)
        if r["onProvider"] and anyBest is None:
            anyBest = r
        if r["onProvider"] and r["gpuOnly"]:
            best = best or r
            if r["gateRow"]:
                gateBest = r
                break
    best = gateBest or best
    return best, anyBest


def measure(a, name, feeds, expected):
    opts = VARIANTS[a.provider][name]
    r = {"variant": name, "options": opts}
    try:
        t0 = time.perf_counter()
        sess, how = make_session(a.provider, opts, a.model)
        r["createMs"] = round((time.perf_counter() - t0) * 1000, 1)
        r["installedAs"] = how
        r["sessionProviders"] = sess.get_providers()
        if PROVIDER_NAME[a.provider] not in r["sessionProviders"]:
            # ONNX Runtime silently retries on the CPU when a provider cannot be created; that must never pass for a GPU time.
            raise RuntimeError(f"{PROVIDER_NAME[a.provider]} was not created; session runs on {r['sessionProviders']}")
        want = {i.name: i.type for i in sess.get_inputs()}
        f = {k: (v.astype(np.float16) if want[k] == "tensor(float16)" else v) for k, v in feeds.items()}
        t1 = time.perf_counter()
        out = sess.run(None, f)
        r["firstRunMs"] = round((time.perf_counter() - t1) * 1000, 1)
        worst = 0.0
        for o, meta_o in zip(out, sess.get_outputs()):
            ref = expected.get(meta_o.name)
            if ref is not None and ref.shape == tuple(o.shape):
                d = np.abs(o.astype(np.float64) - ref.astype(np.float64)).max()
                m = np.abs(ref).max()
                worst = max(worst, float(d / m) if m > 0 else float(d))
        r["scaledError"] = worst
        for _ in range(a.warmups):
            sess.run(None, f)
        times = []
        for _ in range(a.runs):
            s = time.perf_counter()
            sess.run(None, f)
            times.append((time.perf_counter() - s) * 1000)
        times.sort()
        r["runs"] = len(times)
        r["medianMs"] = round(statistics.median(times), 3)
        r["minMs"] = round(times[0], 3)
    except Exception as e:  # a provider that cannot take the model is a result, not a crash
        r["error"] = (str(e).strip().splitlines() or [repr(e)])[0][:300]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=sorted(VARIANTS))
    ap.add_argument("--variant")
    ap.add_argument("--model", default="")
    ap.add_argument("--case", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmups", type=int, default=3)
    ap.add_argument("--placement-only", action="store_true")
    ap.add_argument("--reclassify", help="re-read the placement of an earlier result file's variants and classify again, without timing")
    a = ap.parse_args()

    if a.placement_only:
        opts = dict(VARIANTS[a.provider][a.variant])
        if a.provider == "coreml":
            opts["ProfileComputePlan"] = "1"  # Core ML logs the device each operation of its program runs on
        make_session(a.provider, opts, a.model, verbose=True)
        return 0
    if a.reclassify:
        doc = json.load(open(a.reclassify))
        a.provider, a.model, a.case = doc["provider"], doc["model"], a.case or "-"
        for r in doc["variants"]:
            for k in ("placement", "nodesByProvider", "onProvider", "gpuOnly", "gateRow", "gateReason"):
                r.pop(k, None)
        best, anyBest = pick(a, doc["variants"])
        doc.update(best=best["variant"] if best else None, bestMedianMs=best["medianMs"] if best else None,
                   gateRow=best.get("gateRow") if best else None, gateReason=best.get("gateReason") if best else None,
                   placement=best.get("placement") if best else None, bestAnyUnitsVariant=anyBest["variant"] if anyBest else None,
                   bestAnyUnitsMedianMs=anyBest["medianMs"] if anyBest else None)
        json.dump(doc, open(a.out, "w"), indent=1)
        return 0 if best else 3
    meta, feeds, expected = read_case(a.case)
    if a.variant:
        # One variant in its own process: a provider keeps its device and options for the life of the process, so a second
        # configuration in the same process would silently run the first one's.
        json.dump(measure(a, a.variant, feeds, expected), open(a.out, "w"))
        return 0
    import subprocess
    import tempfile
    results = []
    for name in VARIANTS[a.provider]:
        with tempfile.TemporaryDirectory() as d:
            part = os.path.join(d, "v.json")
            subprocess.run([sys.executable, __file__, "--provider", a.provider, "--variant", name, "--model", a.model, "--case", a.case,
                            "--out", part, "--runs", str(a.runs), "--warmups", str(a.warmups)], check=False)
            r = json.load(open(part)) if os.path.exists(part) else {"variant": name, "error": "worker process died"}
        results.append(r)
        print(f"{a.provider}/{name}: " + (f"{r['medianMs']} ms (min {r['minMs']}), scaled error {r['scaledError']:.2e}" if "medianMs" in r else "ERROR " + r["error"]), flush=True)

    best, anyBest = pick(a, results)
    doc = {
        "provider": a.provider,
        "providerName": PROVIDER_NAME[a.provider],
        "onnxruntime": ort.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model": a.model,
        "case": meta.get("model"),
        "batch": meta.get("batch"),
        "variants": results,
        "best": best["variant"] if best else None,
        "bestMedianMs": best["medianMs"] if best else None,
        "gateRow": best.get("gateRow") if best else None,
        "gateReason": best.get("gateReason") if best else None,
        "placement": best.get("placement") if best else None,
        "bestAnyUnitsVariant": anyBest["variant"] if anyBest else None,
        "bestAnyUnitsMedianMs": anyBest["medianMs"] if anyBest else None,
    }
    try:
        import onnxruntime_ep_webgpu  # noqa: F401
        from importlib.metadata import version
        doc["pluginVersion"] = version("onnxruntime-ep-webgpu")
    except Exception:
        pass
    json.dump(doc, open(a.out, "w"), indent=1)
    return 0 if best else 3


if __name__ == "__main__":
    sys.exit(main())
