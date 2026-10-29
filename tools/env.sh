#!/usr/bin/env bash
# Shared locations for the tools scripts. Source it: `. tools/env.sh`.
# Everything lives under ONEMODEL_HOME (default ~/onemodel); nothing is written into the repo.
export ONEMODEL_HOME="${ONEMODEL_HOME:-$HOME/onemodel}"
export ONEMODEL_CACHE="${ONEMODEL_CACHE:-$ONEMODEL_HOME/cache}"
export ONEMODEL_VENVS="${ONEMODEL_VENVS:-$ONEMODEL_HOME/venvs}"
export ONEMODEL_BIN="$ONEMODEL_HOME/bin"
export ONEMODEL_PY="$ONEMODEL_VENVS/oracles/bin/python"
export PATH="$ONEMODEL_BIN:$HOME/.local/bin:$PATH"
# A user-local .NET SDK, if setup-dotnet.sh installed one.
if [ -x "$ONEMODEL_HOME/dotnet/dotnet" ]; then
  export DOTNET_ROOT="$ONEMODEL_HOME/dotnet" PATH="$ONEMODEL_HOME/dotnet:$PATH"
elif [ -x "$HOME/dotnet10/dotnet" ]; then
  export DOTNET_ROOT="$HOME/dotnet10" PATH="$HOME/dotnet10:$PATH"
fi
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1
export HF_HOME="$ONEMODEL_CACHE/hf" TORCH_HOME="$ONEMODEL_CACHE/torch"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$ONEMODEL_CACHE" "$ONEMODEL_VENVS" "$ONEMODEL_BIN"
