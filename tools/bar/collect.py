#!/usr/bin/env python3
"""Joins one bar run's results into one machine-readable row per model, platform and batch (JSON lines) and a markdown table.

  collect.py DIR --platform NAME --ref webgpu|coreml|dml [--gate 1.5] [--target 1.15]

DIR is what tools/bar.sh wrote: native/ROW.txt (GpuReport), cpu/ROW.txt, ref/ROW.json (+ ref/cuda-ROW.json), machine-gpu.json, load-gpu.log, load-cpu.log.
Our time is GpuReport's wall median of the warm runs (host data in, host data out), the same thing ort_run.py times for the provider.
"""
import argparse
import glob
import json
import os
import re
import statistics


def open_utf8(path):
    return open(path, encoding="utf-8-sig")


def read_text(path):
    """Text of a GpuReport log. Windows PowerShell redirects as UTF-16, everything else writes UTF-8."""
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8-sig", errors="replace")


def parse_native(path):
    t = read_text(path)
    r = {}
    m = re.search(r'row mode=\w+ adapter="([^"]*)" api=(\S+)', t)
    if m:
        r["adapter"], r["api"] = m.group(1), m.group(2)
    m = re.search(r"median of (\d+) warm runs ([\d.]+) ms \(min ([\d.]+)\); per run: upload ([\d.]+) ms, gpu kernels ([\d.]+) ms, submit\+wait\+readback ([\d.]+) ms(?: \([^)]*\))?, inside gpu regions ([\d.]+) ms.*?dispatches/run (\d+)", t)
    if m:
        r.update(runs=int(m.group(1)), oursMedianMs=float(m.group(2)), oursMinMs=float(m.group(3)), uploadMs=float(m.group(4)),
                 gpuKernelsMs=float(m.group(5)), submitReadbackMs=float(m.group(6)), gpuRegionsMs=float(m.group(7)), dispatches=int(m.group(8)))
    m = re.search(r"parity: max scaled error vs cpu exact = ([\d.E+-]+)", t)
    if m:
        r["oursScaledError"] = float(m.group(1))
    m = re.search(r"nodes: (\d+) of (\d+) on gpu", t)
    if m:
        r["nodesOnGpu"], r["nodes"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"determinism: run1/run2 outputs (\S+)", t)
    if m:
        r["deterministic"] = m.group(1)
    return r


def parse_cpu(path):
    t = read_text(path)
    m = re.search(r"cpu fast mode, (\d+) threads: median of (\d+) runs ([\d.]+) ms \(min ([\d.]+)\)", t)
    return {"cpuThreads": int(m.group(1)), "cpuMedianMs": float(m.group(3))} if m else {}


def load_summary(path):
    """Load average (1 minute) and GPU utilisation samples: min, max and mean over the run."""
    try:
        lines = [l.split() for l in open(path) if l.strip() and not l.startswith("#")]
    except OSError:
        return {}
    loads = [float(l[1]) for l in lines if len(l) > 1]
    gpus = [float(l[2]) for l in lines if len(l) > 2 and l[2] != "-"]
    d = {}
    if loads:
        d["load1"] = {"first": loads[0], "min": min(loads), "max": max(loads), "mean": round(statistics.mean(loads), 2), "last": loads[-1], "samples": len(loads)}
    if gpus:
        d["gpuUtilPercent"] = {"first": gpus[0], "max": max(gpus), "mean": round(statistics.mean(gpus), 1), "last": gpus[-1]}
    return d


def placement_text(r):
    """One cell: the reference's nodes by provider and, for Core ML, its compute operations by device."""
    p = r.get("refPlacement")
    if not p:
        return None
    t = ", ".join(f"{k.replace('ExecutionProvider', '')} {v}" for k, v in sorted(p["byProvider"].items()))
    t += f"; compute nodes on CPU provider {p['computeOnCpu']}"
    if "coreMlComputePlan" in p:
        plan = p["coreMlComputePlan"]
        t += "; Core ML compute ops " + (", ".join(f"{k} {v}" for k, v in sorted(plan.items())) if plan else "not reported (neural network format has no plan)")
    return t


def choose_reference(a, name, row):
    """With several reference providers (macOS: Core ML and the WebGPU provider on Metal) the reference is, per row, the faster of those whose best
    configuration makes the row a gate row; when none does, the faster of all, which makes the row an information row. Each provider's own time
    and class stay in row["refProviders"]."""
    names = a.ref.split(",")
    cands = {}
    for i, n in enumerate(names):
        path = os.path.join(a.dir, "ref", name + ".json" if i == 0 else f"{n}-{name}.json")
        if os.path.exists(path):
            d = json.load(open_utf8(path))
            if d.get("bestMedianMs"):
                cands[n] = d
    if len(names) < 2 or not cands:
        return
    row["refCandidates"] = {n: {"ms": d["bestMedianMs"], "class": "gate" if d.get("gateRow") else "information", "variant": d["best"],
                               "why": d.get("gateReason"), "placement": placement_text({"refPlacement": d.get("placement")})}
                            for n, d in cands.items()}
    gate = {n: d for n, d in cands.items() if d.get("gateRow")}
    pool = gate or cands
    pick = min(pool, key=lambda n: pool[n]["bestMedianMs"])
    d = pool[pick]
    row["reference"] = pick
    row["refBestVariant"] = d["best"]
    row["refMedianMs"] = d["bestMedianMs"]
    row["refGateRow"], row["refGateReason"] = d.get("gateRow"), d.get("gateReason")
    row["refPlacement"] = d.get("placement")
    row["refVersion"] = d["onnxruntime"]
    if "pluginVersion" in d:
        row["refPluginVersion"] = d["pluginVersion"]


def cand_text(r):
    c = r.get("refCandidates")
    return "; ".join(f"{n} {v['ms']:.2f} {v['class']}" for n, v in c.items()) if c else None


def web_rows(a):
    """Browser rows: our page against onnxruntime-web in the same Chrome. Each directory in a.dir plus a.reps is one repetition; the row
    takes the median of the repetitions' medians on both sides, because both swing run to run on a shared machine."""
    dirs = [a.dir] + a.reps
    base = a.dir
    names = sorted(f[len(a.adapter) + len("-fast-"):-5] for f in os.listdir(os.path.join(base, "browser"))
                   if f.startswith(a.adapter + "-fast-") and f.endswith(".json"))
    native = {n: parse_native(os.path.join(base, "native", n + ".txt")) for n in names if os.path.exists(os.path.join(base, "native", n + ".txt"))}
    rows = []
    for n in names:
        ours, theirs, errs, first = [], [], [], None
        for d in dirs:
            b = os.path.join(d, "browser", f"{a.adapter}-fast-{n}.json")
            o = os.path.join(d, "ort", f"{a.adapter.split('+')[0]}-{n}.json")  # the reference runs once per adapter, whatever our build
            if os.path.exists(b):
                j = json.load(open_utf8(b))
                if j.get("warmMedianMs") is not None:
                    ours.append(j["warmMedianMs"])
                    first = first or j
            if os.path.exists(o):
                j = json.load(open_utf8(o))
                if j.get("warmMedianMs") is not None:
                    theirs.append(j["warmMedianMs"])
                    errs.append(j)
        row = {"row": n, "platform": a.platform, "indicative": True, "reference": "onnxruntime-web webgpu", "gate": a.gate, "target": a.target,
               "oursSamples": ours, "refSamples": theirs}
        if first:
            row["nodesOnDevice"], row["nodes"] = first.get("nodesOnDevice"), first.get("nodes")
            row["oursScaledError"] = first.get("worstScaled")
        if errs:
            row["refVersion"] = errs[0].get("version")
            row["chrome"] = errs[0].get("chrome")
            row["refAdapter"] = errs[0].get("adapter")
        if n in native:
            row["nativeMedianMs"] = native[n].get("oursMedianMs")
        if ours:
            row["oursMedianMs"] = round(statistics.median(ours), 2)
        if theirs:
            row["refMedianMs"] = round(statistics.median(theirs), 2)
        if ours and theirs:
            row["ratio"] = round(row["oursMedianMs"] / row["refMedianMs"], 2)
            row["verdict"] = "target" if row["ratio"] <= a.target else "met" if row["ratio"] <= a.gate else "missed"
        elif ours:
            row["verdict"] = "no reference result (provider could not run the model)"
        rows.append(row)
    with open(os.path.join(a.dir, f"bar-{a.platform}.jsonl"), "w") as o:
        for r in rows:
            o.write(json.dumps(r) + "\n")
    out = ["| row | ours ms | ORT-web ms | ratio | verdict | samples ours | samples ORT-web | native ms |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['row']} | {r.get('oursMedianMs','-')} | {r.get('refMedianMs','-')} | {r.get('ratio','-')} | {r.get('verdict','-')} | {r['oursSamples']} | {r['refSamples']} | {r.get('nativeMedianMs','-')} |")
    open(os.path.join(a.dir, f"bar-{a.platform}.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--web", action="store_true", help="browser rows (our page vs onnxruntime-web) instead of native rows")
    ap.add_argument("--reps", nargs="*", default=[], help="more repetition directories (browser rows)")
    ap.add_argument("--adapter", default="nvidia")
    ap.add_argument("--platform", required=True)
    ap.add_argument("--ref", default="")
    ap.add_argument("--gate", type=float, default=None)
    ap.add_argument("--target", type=float, default=None)
    a = ap.parse_args()
    a.gate = a.gate or (2.5 if a.web else 1.5)
    a.target = a.target or (1.5 if a.web else 1.15)
    if a.web:
        return web_rows(a)
    mf = os.path.join(a.dir, "machine-gpu.json")
    machine = json.load(open_utf8(mf)) if os.path.exists(mf) else {}
    load = load_summary(os.path.join(a.dir, "load-gpu.log"))
    cpuload = load_summary(os.path.join(a.dir, "load-cpu.log"))
    rows = []
    for f in sorted(glob.glob(os.path.join(a.dir, "native", "*.txt"))):
        name = os.path.basename(f)[:-4]
        row = {"row": name, "platform": a.platform, "indicative": True, "reference": a.ref, "gate": a.gate, "target": a.target}
        row.update(parse_native(f))
        cpu = os.path.join(a.dir, "cpu", name + ".txt")
        if os.path.exists(cpu):
            row.update(parse_cpu(cpu))
        ref = os.path.join(a.dir, "ref", name + ".json")
        if os.path.exists(ref):
            d = json.load(open_utf8(ref))
            row["refVersion"] = d["onnxruntime"]
            if "pluginVersion" in d:
                row["refPluginVersion"] = d["pluginVersion"]
            row["refVariants"] = {v["variant"]: (v.get("medianMs") if "medianMs" in v else "error: " + v.get("error", "")) for v in d["variants"]}
            row["refBestVariant"] = d["best"]
            if d.get("bestAnyUnitsMedianMs") and d.get("bestAnyUnitsVariant") != d["best"]:
                row["refBestAnyComputeUnitsMs"] = d["bestAnyUnitsMedianMs"]  # information: may run on the Neural Engine
            row["refMedianMs"] = d["bestMedianMs"]
            row["refGateRow"], row["refGateReason"] = d.get("gateRow"), d.get("gateReason")
            if isinstance(d.get("placement"), dict):
                row["refPlacement"] = d["placement"]
            b = next((v for v in d["variants"] if v["variant"] == d["best"]), None)
            if b:
                row["refScaledError"] = b.get("scaledError")
                row["refProviders"] = b.get("sessionProviders")
                nb = b.get("nodesByProvider")
                row["refNodesByProvider"] = nb
                if isinstance(nb, dict) and sum(nb.values()):
                    row["refNodesOnProvider"] = round(nb.get(d["providerName"], 0) / sum(nb.values()), 3)
        choose_reference(a, name, row)
        cuda = os.path.join(a.dir, "ref", "cuda-" + name + ".json")
        if os.path.exists(cuda):
            d = json.load(open_utf8(cuda))
            row["cudaMedianMs"] = d["bestMedianMs"]
            row["cudaVersion"] = d["onnxruntime"]
        oc = os.path.join(a.dir, "ref", "ortcpu-" + name + ".json")
        if os.path.exists(oc):
            row["ortCpuMedianMs"] = json.load(open_utf8(oc))["bestMedianMs"]
        if "oursMedianMs" in row and row.get("refMedianMs"):
            row["ratio"] = round(row["oursMedianMs"] / row["refMedianMs"], 2)
            row["verdict"] = "target" if row["ratio"] <= a.target else "met" if row["ratio"] <= a.gate else "missed"
            # Only a row whose reference ran every compute node on its GPU device can hold the bar; the others are reported, never gated.
            row["class"] = "gate" if row.get("refGateRow") else "information"
            if row["class"] == "information":
                row["verdict"] = "information (" + row["verdict"] + ")"
        elif "oursMedianMs" in row:
            row["verdict"] = "no reference"
        row["machine"] = machine
        row["load"] = load
        if cpuload:
            row["loadCpuPhase"] = cpuload
        rows.append(row)
    with open(os.path.join(a.dir, f"bar-{a.platform}.jsonl"), "w") as o:
        for r in rows:
            o.write(json.dumps(r) + "\n")
    cols = ["row", "ours ms", "gpu kernels ms", "ref ms", "ratio", "verdict", "class", "chosen reference", "providers (ms, class)", "reference placement", "cuda ms", "onemodel cpu ms", "ort cpu ms", "ref best variant"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    f1 = lambda v: "-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)
    for r in rows:
        out.append("| " + " | ".join(f1(x) for x in [r["row"], r.get("oursMedianMs"), r.get("gpuKernelsMs"), r.get("refMedianMs"), r.get("ratio"), r.get("verdict"),
                                                       r.get("class"), r.get("reference"), cand_text(r), placement_text(r), r.get("cudaMedianMs"), r.get("cpuMedianMs"), r.get("ortCpuMedianMs"), r.get("refBestVariant")]) + " |")
    open(os.path.join(a.dir, f"bar-{a.platform}.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
