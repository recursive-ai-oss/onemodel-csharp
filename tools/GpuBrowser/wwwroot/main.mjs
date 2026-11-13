// Query: model=<url of the model file>&greedy=1 (for a greedy decode: the model is a decoder)&case=<url of case.bin>&mode=fast|exact&runs=N&warmups=N&label=text
// Sets window.__result to the JSON report (the adapter's description is added to it).
import { dotnet } from './_framework/dotnet.js';
import * as gpu from './_content/RecursiveAI.OneGpu.WebGpu/onegpu.js';

const log = document.getElementById('log');
const q = new URLSearchParams(location.search);
// An uncaptured WebGPU error is a failed run: the report says how many there were.
const gpuErrors = [];
window.addEventListener('unhandledrejection', e => gpuErrors.push('unhandledrejection: ' + e.reason));
try {
  const adapterInfo = await (async () => {
    // The GPU process can still be starting when the page loads: ask a few times.
    let a = null;
    for (let i = 0; i < 10 && !a; i++) { a = await navigator.gpu?.requestAdapter(); if (!a) await new Promise(r => setTimeout(r, 300)); }
    if (!a) return null;
    const i = a.info ?? {};
    return { vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description, isFallback: a.isFallbackAdapter, features: [...a.features].sort(),
      limits: { maxBufferSize: a.limits.maxBufferSize, maxStorageBufferBindingSize: a.limits.maxStorageBufferBindingSize, maxComputeWorkgroupStorageSize: a.limits.maxComputeWorkgroupStorageSize } };
  })();
  const fetchBytes = async (url) => {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return new Uint8Array(await r.arrayBuffer());
  };
  const t0 = performance.now();
  const greedy = q.get('greedy') === '1';
  const [model, caseFile] = await Promise.all([fetchBytes(q.get('model')), fetchBytes(q.get('case'))]);
  const fetchMs = performance.now() - t0;
  const runtime = await dotnet.withDiagnosticTracing(false).create();
  runtime.setModuleImports('onegpu', gpu);
  const exports = await runtime.getAssemblyExports(runtime.getConfig().mainAssemblyName);
  const config = JSON.stringify({ mode: q.get('mode') ?? 'fast', runs: Number(q.get('runs') ?? 10), warmups: Number(q.get('warmups') ?? 3), label: q.get('label') ?? '', fuse: q.get('fuse') !== '0', attention: q.get('attention') ?? undefined, pair: q.get('pair') === '1' });
  // greedy=1: model is a decoder and the case holds the encoder states and the CPU's tokens: a greedy decode on the device.
  const json = greedy ? await exports.GpuBrowser.Host.RunGreedy(config, model, caseFile) : await exports.GpuBrowser.Host.RunCase(config, model, caseFile);
  const r = JSON.parse(json);
  r.fetchMs = Math.round(fetchMs);
  r.modelBytes = model.length;
  r.chrome = navigator.userAgent;
  r.webgpuAdapter = adapterInfo;
  r.pageErrors = gpuErrors;
  window.__result = JSON.stringify(r);
  log.textContent = r.error ? 'error: ' + r.error : `${r.label}: ${r.nodesOnDevice}/${r.nodes} nodes on device, scaled ${r.worstScaled}, ${r.warmMedianMs} ms`;
} catch (e) {
  window.__result = JSON.stringify({ fatal: String(e && e.stack || e) });
  log.textContent = 'fatal: ' + e;
}
