// Headless runner: loads the published browser-wasm bundle under node (no browser), runs the app's Main and exits with its exit code.
// Usage: node main.mjs <published wwwroot directory> [--copy PATH]... [-- app arguments]
// Each --copy PATH (a file or a directory tree) is copied into the app's in-memory file system at the same path, so command lines
// that name model files work unchanged on the desktop runtime and here. (The runtime has no host-directory mount.)
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import { readFileSync, readdirSync, statSync } from 'node:fs';

const argv = process.argv.slice(2);
const root = resolve(argv.shift() ?? '.');
const copies = [];
const appArgs = [];
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === '--copy') copies.push(resolve(argv[++i]));
  else if (argv[i] === '--') { appArgs.push(...argv.slice(i + 1)); break; }
  else appArgs.push(argv[i]);
}

const { dotnet } = await import(pathToFileURL(resolve(root, '_framework', 'dotnet.js')).href);
const { runMain, Module } = await dotnet.withApplicationArguments(...appArgs).create();
const FS = Module.FS;
function mkdirs(FS, path) {
  let p = '';
  for (const part of path.split('/').filter(Boolean)) {
    p += '/' + part;
    try { FS.mkdir(p); } catch { /* exists */ }
  }
}

function copyTree(FS, path) {
  if (statSync(path).isDirectory()) {
    mkdirs(FS, path);
    for (const name of readdirSync(path).sort()) copyTree(FS, path + '/' + name);
  } else {
    mkdirs(FS, path.slice(0, path.lastIndexOf('/')));
    FS.writeFile(path, readFileSync(path));
  }
}

for (const path of copies) copyTree(FS, path);
process.exit(await runMain());
