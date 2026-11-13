#!/usr/bin/env bash
# Runs the GPU package's tests, and its NativeAOT smoke app, on a Vulkan device with the Khronos validation layer on. Every test checks the
# layer's messages and fails on any. Select the device with VK_ICD_FILENAMES (lavapipe's lvp_icd.json on a machine without a GPU) and point
# VK_ADD_LAYER_PATH at the layer folder when the system does not install it (tools/ci/fetch-validation-layer.sh makes one).
#
#   tools/ci/gpu-tests.sh [dotnet test filter arguments]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 MSBUILDDISABLENODEREUSE=1
# A machine that is supposed to have a device must not pass by skipping every test.
export ONEMODEL_REQUIRE_DEVICE=1
# The NVIDIA Vulkan driver (595 series) needs more static thread-local storage than glibc reserves once the validation layer is loaded.
export GLIBC_TUNABLES="${GLIBC_TUNABLES:-glibc.rtld.optional_static_tls=4096}"
export VK_INSTANCE_LAYERS="${VK_INSTANCE_LAYERS:-VK_LAYER_KHRONOS_validation}"
dotnet build -c Release tests/OneModel.Gpu.Tests
dotnet test --no-build -c Release --project tests/OneModel.Gpu.Tests "$@"
