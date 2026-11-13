#!/usr/bin/env bash
# Publishes the GPU smoke app with NativeAOT (any trim or AOT warning fails the publish) and runs it on a Vulkan device with the validation
# layer on; exit 77 from the app means it found no device, which is a failure here. Same environment as tools/ci/gpu-tests.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 MSBUILDDISABLENODEREUSE=1
export GLIBC_TUNABLES="${GLIBC_TUNABLES:-glibc.rtld.optional_static_tls=4096}"
export VK_INSTANCE_LAYERS="${VK_INSTANCE_LAYERS:-VK_LAYER_KHRONOS_validation}"
out="${1:-${TMPDIR:-/tmp}/onemodel-gpu-aot-smoke}"
dotnet publish tests/GpuAotSmoke -c Release -o "$out" -warnaserror
"$out/GpuAotSmoke"
