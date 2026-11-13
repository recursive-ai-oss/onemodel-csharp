// Prints one table from the files tools/gpu_browser.sh wrote under a directory: node summarize.mjs <dir>
import fs from 'node:fs';
import path from 'node:path';

const dir = process.argv[2];
const read = (p) => { try { return JSON.parse(fs.readFileSync(p, 'utf8')); } catch { return null; } };
const files = (sub) => (fs.existsSync(path.join(dir, sub)) ? fs.readdirSync(path.join(dir, sub)).filter(f => f.endsWith('.json')) : []);
const nativeMs = (row) => {
  try {
    const t = fs.readFileSync(path.join(dir, 'native', row + '.txt'), 'utf8');
    const m = /median of \d+ warm runs ([\d.]+) ms/.exec(t);
    return m ? Number(m[1]) : null;
  } catch { return null; }
};
const f = (v, d = 1) => (v == null ? '-' : Number(v).toFixed(d));
const rows = [];
for (const file of files('browser')) {
  const [adapter, mode, ...rest] = file.replace(/\.json$/, '').split('-');
  const row = rest.join('-');
  const r = read(path.join(dir, 'browser', file));
  if (!r) { rows.push({ row, adapter, mode, note: 'no result' }); continue; }
  const ort = read(path.join(dir, 'ort', `${adapter.replace('+aot', '')}-${row}.json`));
  rows.push({ row, adapter, mode, r, native: nativeMs(row), ort });
}
rows.sort((a, b) => (a.row + a.adapter + a.mode).localeCompare(b.row + b.adapter + b.mode));
console.log('row | adapter | mode | on device | regions | worst scaled err | RequireGpu | 2 runs | browser warm ms (median/min) | native ms | ORT-web ms | ours/ORT-web');
for (const x of rows) {
  const r = x.r;
  if (!r || r.fatal || r.error) { console.log(`${x.row} | ${x.adapter} | ${x.mode} | ${x.note ?? (r?.fatal ?? r?.error ?? '').toString().slice(0, 300)}${r?.requireGpu ? ' [RequireGpu: ' + String(r.requireGpu).slice(0, 200) + ']' : ''}`); continue; }
  if (r.tokensEqualCpu !== undefined) {
    console.log([x.row, x.adapter, x.mode, `${r.nodesOnDevice}/${r.nodes} (decoder)`, r.regions, `tokens ${r.tokensEqualCpu ? 'equal to cpu' : 'DIFFER'}`, '-', r.bitIdentical ? 'identical' : 'DIFFER',
      `${f(r.warmMsPerToken)} ms/token (first pass ${f(r.firstPassMsPerToken)})`, f(x.native), '-', '-'].join(' | '));
    continue;
  }
  const ortMs = x.ort && !x.ort.error ? x.ort.warmMedianMs : null;
  console.log([x.row, x.adapter, x.mode, `${r.nodesOnDevice}/${r.nodes}`, r.regions, r.worstScaled?.toExponential(2), String(r.requireGpu).slice(0, 40),
    (r.bitIdentical ? 'identical' : 'DIFFER') + (r.pairSessions ? `, ${r.pairSessions} devices ${r.pairBitIdentical ? 'identical' : 'DIFFER'}` : ''), `${f(r.warmMedianMs)}/${f(r.warmMinMs)}`, f(x.native), ortMs == null ? (x.ort?.error ? 'err' : '-') : f(ortMs), ortMs ? f(r.warmMedianMs / ortMs, 2) : '-'].join(' | '));
}
