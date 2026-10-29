#!/usr/bin/env bash
# Prepares what a fuzz run needs under <dir>: the libFuzzer driver for SharpFuzz and SharpFuzz's instrumenter.
#
#   setup_fuzz.sh <dir>
#
# The driver is Metalnem/libfuzzer-dotnet's release binary (MIT licence); the instrumenter is the SharpFuzz.CommandLine .NET
# tool. Both are fetched, never committed. Needs `dotnet` on PATH and network access.
set -eu
dir=$1
mkdir -p "$dir"
version=v2025.05.02.0904
curl -sSL -o "$dir/libfuzzer-dotnet" "https://github.com/Metalnem/libfuzzer-dotnet/releases/download/$version/libfuzzer-dotnet-ubuntu"
chmod +x "$dir/libfuzzer-dotnet"
sha256sum "$dir/libfuzzer-dotnet"
dotnet tool install SharpFuzz.CommandLine --version 2.2.0 --tool-path "$dir/tools" >/dev/null
ls "$dir/tools"
