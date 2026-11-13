# Installs the released ONNX Runtime wheel the Windows bar rows compare against, as a black box (installed and run, never opened), in its own
# uv venv under $env:ONEMODEL_BAR_ORT (default $HOME\onemodel\bar-ort).
#
#   tools\bar\setup_ort.ps1 dml|webgpu
#
# dml     onnxruntime-directml: the DirectML provider (the reference on Windows).
# webgpu  onnxruntime + onnxruntime-ep-webgpu: the native WebGPU EP over D3D12 or Vulkan, for information.
# Pins as in setup_ort.sh; $env:ORT_VERSION, $env:DML_ORT_VERSION and $env:WEBGPU_EP_VERSION override them. The DirectML wheel is published
# on its own schedule (1.24.4 is the newest), so its pin differs from the core wheel's.
param([Parameter(Mandatory = $true)][ValidateSet("dml", "webgpu")][string]$What)
$ErrorActionPreference = "Stop"
$ortVersion = if ($env:ORT_VERSION) { $env:ORT_VERSION } else { "1.31.1" }
$dmlVersion = if ($env:DML_ORT_VERSION) { $env:DML_ORT_VERSION } else { "1.24.4" }
$epVersion = if ($env:WEBGPU_EP_VERSION) { $env:WEBGPU_EP_VERSION } else { "0.4.0" }
$home1 = if ($env:ONEMODEL_HOME) { $env:ONEMODEL_HOME } else { Join-Path $HOME "onemodel" }
$base = if ($env:ONEMODEL_BAR_ORT) { $env:ONEMODEL_BAR_ORT } else { Join-Path $home1 "bar-ort" }
$venv = Join-Path $base $What
$py = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $py)) {
  uv venv --python 3.13 $venv
  if ($What -eq "dml") { uv pip install --python $py "onnxruntime-directml==$dmlVersion" numpy }
  else { uv pip install --python $py "onnxruntime==$ortVersion" "onnxruntime-ep-webgpu==$epVersion" numpy }
}
& $py -c "import onnxruntime as o; print('onnxruntime', o.__version__, o.get_available_providers())"
Write-Output $py
