// Headless runner: loads the published browser-wasm bundle under node (no browser), runs the app's Main and exits with its exit code.
// Usage: node main.mjs <published wwwroot directory>
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

const root = resolve(process.argv[2] ?? '.');
const { dotnet } = await import(pathToFileURL(resolve(root, '_framework', 'dotnet.js')).href);
const { runMain } = await dotnet.create();
process.exit(await runMain());
