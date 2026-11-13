// Runs a case file (see tools/GpuReport --export) through onnxruntime-web's WebGPU execution provider with default settings and reports
// its timing and its difference from the reference outputs. Query: model=<url>&case=<url>&runs=N&warmups=N&label=text
const q = new URLSearchParams(location.search);
const log = document.getElementById('log');
const result = { label: q.get('label') ?? '', provider: 'onnxruntime-web webgpu', version: ort.env.versions?.web };
try {
  const fetchBytes = async (url) => { const r = await fetch(url); if (!r.ok) throw new Error(`${url}: ${r.status}`); return new Uint8Array(await r.arrayBuffer()); };
  const file = await fetchBytes(q.get('case'));
  const jsonLen = new DataView(file.buffer).getUint32(0, true);
  const meta = JSON.parse(new TextDecoder().decode(file.subarray(4, 4 + jsonLen)));
  const blobStart = 4 + jsonLen;
  const blob = (e) => file.slice(blobStart + e.offset, blobStart + e.offset + e.bytes).buffer;
  const feeds = {};
  for (const e of meta.entries.filter(x => x.role === 'input')) {
    feeds[e.name] = e.dtype === 'i64' ? new ort.Tensor('int64', new BigInt64Array(blob(e)), e.dims) : new ort.Tensor('float32', new Float32Array(blob(e)), e.dims);
  }
  const expected = Object.fromEntries(meta.entries.filter(x => x.role === 'output').map(e => [e.name, new Float32Array(blob(e))]));

  ort.env.wasm.numThreads = 1;
  // The page asks for the adapter the way the onemodel harness does (no options) and hands it over, so both runtimes use the same one.
  const requested = await navigator.gpu?.requestAdapter();
  if (requested) ort.env.webgpu.adapter = requested;
  result.configuration = { executionProviders: ['webgpu'], numThreads: 1, graphOptimizationLevel: 'default', adapter: 'requested by the page with no options and set as env.webgpu.adapter', sessionOptions: 'defaults otherwise' };
  const t0 = performance.now();
  const session = await ort.InferenceSession.create(q.get('model'), { executionProviders: ['webgpu'] });
  result.createMs = Math.round(performance.now() - t0);
  const adapter = ort.env.webgpu?.adapter;
  if (adapter) {
    const i = adapter.info ?? {};
    result.adapter = { vendor: i.vendor, architecture: i.architecture, features: [...adapter.features].sort() };
  }
  const t1 = performance.now();
  let out = await session.run(feeds);
  result.firstRunMs = Math.round((performance.now() - t1) * 10) / 10;
  let worst = 0;
  for (const [name, ref] of Object.entries(expected)) {
    const got = out[name].data;
    let max = 0, diff = 0;
    for (let i = 0; i < ref.length; i++) { max = Math.max(max, Math.abs(ref[i])); diff = Math.max(diff, Math.abs(got[i] - ref[i])); }
    worst = Math.max(worst, max === 0 ? diff : diff / max);
  }
  result.worstScaled = worst;
  for (let i = 0; i < Number(q.get('warmups') ?? 3); i++) await session.run(feeds);
  const times = [];
  for (let i = 0; i < Number(q.get('runs') ?? 10); i++) {
    const s = performance.now();
    await session.run(feeds);
    times.push(performance.now() - s);
  }
  times.sort((a, b) => a - b);
  result.runs = times.length;
  result.warmMedianMs = times.length ? Math.round(times[times.length >> 1] * 100) / 100 : null;
  result.warmMinMs = times.length ? Math.round(times[0] * 100) / 100 : null;
} catch (e) {
  result.error = String(e && e.message || e);
}
result.chrome = navigator.userAgent;
window.__result = JSON.stringify(result);
log.textContent = result.error ? 'error: ' + result.error : `${result.label}: ${result.warmMedianMs} ms`;
