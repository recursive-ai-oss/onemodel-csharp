# The bar harness for Windows (PowerShell 5.1 or 7): the same steps as tools/bar.sh, for a reference that runs there (DirectML, or the WebGPU
# plugin EP). It avoids Unix-only paths and tools on purpose.
#
#   tools\bar.ps1 -Ref dml|webgpu [-Phase gpu|cpu] [-Out DIR] [-Runs N] [-RefOnly] ROW...      (-Rows lists the rows)
#
# Needs: a .NET 10 SDK on PATH, uv on PATH, the reference venv from tools\bar\setup_ort.ps1, the model files under
# $env:ONEMODEL_CACHE\models (default $HOME\onemodel\cache; fetched by tools\export_models.py), and the RecursiveAI.OneGpu packages from nuget.org.
# Output files are the same as bar.sh's, so tools\bar\collect.py (run at the end, stdlib only) writes bar-windows.jsonl and .md.
# The load column is the CPU load in percent (Windows has no load average); there is no GPU utilisation sample. Indicative numbers only.
param(
  [string]$Ref = "dml",
  [string]$Phase = "gpu",
  [string]$Out = "",
  [int]$Runs = 20,
  [switch]$RefOnly,
  [switch]$Rows,
  [Parameter(ValueFromRemainingArguments = $true)] [string[]]$RowNames
)
$ErrorActionPreference = "Continue"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$home1 = if ($env:ONEMODEL_HOME) { $env:ONEMODEL_HOME } else { Join-Path $HOME "onemodel" }
$cache = if ($env:ONEMODEL_CACHE) { $env:ONEMODEL_CACHE } else { Join-Path $home1 "cache" }
$models = Join-Path $cache "models"
$ortBase = if ($env:ONEMODEL_BAR_ORT) { $env:ONEMODEL_BAR_ORT } else { Join-Path $home1 "bar-ort" }
if (-not $Out) { $Out = Join-Path $HOME "onemodel\artifacts\bar" }

# Keep in step with tools/bar_rows.sh (the same rows, GpuReport arguments included).
$crnnInput = Get-ChildItem -Path (Join-Path $cache "reference") -Recurse -Filter input.npy -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -match "crnn.fp32.b1-val-00000539.inputs" } | Select-Object -First 1
$crnnArgs = "--batch 1" + $(if ($crnnInput) { " --input $($crnnInput.FullName)" } else { "" })
$table = [ordered]@{
  minilm_fp32_b1 = @("minilm_l6\minilm_l6.onnx", "--batch 1 --seq 128")
  minilm_fp32_b8 = @("minilm_l6\minilm_l6.onnx", "--batch 8 --seq 128")
  minilm_fp16_b1 = @("minilm_l6\minilm_l6_fp16.onnx", "--batch 1 --seq 128")
  minilm_int8_b1 = @("minilm_l6\minilm_l6_int8_dyn.onnx", "--batch 1 --seq 128")
  bert_fp32_b1 = @("bert_base\bert_base.onnx", "--batch 1 --seq 128")
  bert_fp32_b8 = @("bert_base\bert_base.onnx", "--batch 8 --seq 128")
  bert_int8_b1 = @("bert_base\bert_base_int8_dyn.onnx", "--batch 1 --seq 128")
  resnet50_fp32_b1 = @("resnet50\resnet50.onnx", "--batch 1")
  resnet50_fp32_b8 = @("resnet50\resnet50.onnx", "--batch 8")
  resnet50_fp16_b1 = @("resnet50\resnet50_fp16.onnx", "--batch 1")
  resnet50_int8_b1 = @("resnet50\resnet50_int8_qdq.onnx", "--batch 1")
  mobilenet_fp32_b1 = @("mobilenet_v3_large\mobilenet_v3_large.onnx", "--batch 1")
  mobilenet_fp32_b8 = @("mobilenet_v3_large\mobilenet_v3_large.onnx", "--batch 8")
  mobilenet_int8_b1 = @("mobilenet_v3_large\mobilenet_v3_large_int8_qdq.onnx", "--batch 1")
  dbnet_fp32_b1 = @("dbnet\dbnet_lite_v1.onnx", "--batch 1 --size 640x640")
  crnn_fp32_b1 = @("crnn\crnn_v0.onnx", $crnnArgs)
  whisper_enc_b1 = @("whisper_small\whisper_small_encoder.onnx", "--batch 1")
  whisper_dec_b1 = @("whisper_small\whisper_small_decoder.onnx", "--batch 1 --seq 8")
  whisper_enc_int8_b1 = @("whisper_small\whisper_small_encoder_int8_dyn.onnx", "--batch 1")
  whisper_dec_int8_b1 = @("whisper_small\whisper_small_decoder_int8_dyn.onnx", "--batch 1 --seq 8")
}
if ($Rows) { $table.Keys -join " "; exit 0 }
if (-not $RowNames) { Write-Error "usage: bar.ps1 -Ref dml [-Phase gpu|cpu] [-Out DIR] ROW...  (-Rows lists them)"; exit 2 }

foreach ($d in "cases", "native", "cpu", "ref", "profile") { New-Item -ItemType Directory -Force -Path (Join-Path $Out $d) | Out-Null }

# Machine, driver and load ---------------------------------------------------------------------------------------------------------
$gpus = @(Get-CimInstance Win32_VideoController | ForEach-Object { "$($_.Name) (driver $($_.DriverVersion))" })
$cpuInfo = Get-CimInstance Win32_Processor | Select-Object -First 1
function Sample { $l = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average; "{0} {1} -" -f [int][DateTimeOffset]::UtcNow.ToUnixTimeSeconds(), $l }
$machine = [ordered]@{
  os = (Get-CimInstance Win32_OperatingSystem).Caption; cpu = $cpuInfo.Name; cores = $cpuInfo.NumberOfLogicalProcessors
  gpu = ($gpus -join "; "); driver = ($gpus -join "; "); phase = $Phase; ref = $Ref; runsPerRow = $Runs
  date = (Get-Date).ToUniversalTime().ToString("s") + "Z"; loadBefore = (Sample)
}
$machine | ConvertTo-Json | Set-Content -Path (Join-Path $Out "machine-$Phase.json")
$loadLog = Join-Path $Out "load-$Phase.log"
Add-Content -Path $loadLog -Value "# time cpu-load-percent gpu% (job start)"
$sampler = Start-Job -ScriptBlock {
  param($log)
  while ($true) {
    $l = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
    Add-Content -Path $log -Value ("{0} {1} -" -f [int][DateTimeOffset]::UtcNow.ToUnixTimeSeconds(), $l)
    Start-Sleep -Seconds 5
  }
} -ArgumentList $loadLog

try {
  dotnet publish -c Release (Join-Path $here "GpuReport\GpuReport.csproj") -o (Join-Path $Out "gpureport") -v q -clp:NoSummary
  if ($LASTEXITCODE -ne 0) { throw "GpuReport publish failed" }
  $py = Join-Path $ortBase "$Ref\Scripts\python.exe"
  if (-not (Test-Path $py)) { throw "no $py : run tools\bar\setup_ort.ps1 $Ref" }
  $ortRun = Join-Path $here "bar\ort_run.py"
  $report = Join-Path $Out "gpureport\GpuReport.dll"
  foreach ($r in $RowNames) {
    if (-not $table.Contains($r)) { Write-Warning "unknown row $r"; continue }
    $model = Join-Path $models $table[$r][0]
    if (-not (Test-Path $model)) { Write-Warning "skip $r : $($table[$r][0]) not fetched"; continue }
    $case = Join-Path $Out "cases\$r\case.bin"
    $rowArgs = $table[$r][1] -split ' (?=--)' | ForEach-Object { $_ -split ' ', 2 } | ForEach-Object { $_ }
    if ($Phase -eq "cpu") {
      if (-not $RefOnly) {
        & dotnet $report $model @rowArgs --modes fast --runs 3 --cpu-time 8 --export (Join-Path $Out "cases\$r") *> (Join-Path $Out "cpu\$r.txt")
      }
      if (Test-Path $case) { & $py $ortRun --provider cpu --model $model --case $case --runs 5 --out (Join-Path $Out "ref\ortcpu-$r.json") }
      continue
    }
    if (-not $RefOnly) {
      & dotnet $report $model @rowArgs --modes fast --runs $Runs --export (Join-Path $Out "cases\$r") *> (Join-Path $Out "native\$r.txt")
    }
    if (-not (Test-Path $case)) { Write-Warning "skip $r : no case file"; continue }
    & $py $ortRun --provider $Ref --model $model --case $case --runs $Runs --out (Join-Path $Out "ref\$r.json")
  }
  Add-Content -Path $loadLog -Value (Sample)
  & $py (Join-Path $here "bar\collect.py") $Out --platform windows --ref $Ref
}
finally {
  Stop-Job $sampler -ErrorAction SilentlyContinue
  Remove-Job $sampler -Force -ErrorAction SilentlyContinue
}
