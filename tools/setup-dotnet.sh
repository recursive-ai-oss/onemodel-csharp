#!/usr/bin/env bash
# Installs a user-local .NET SDK under $ONEMODEL_HOME/dotnet (used on hosts without one).
# Usage: tools/setup-dotnet.sh [sdk-version]   (default 10.0.400)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/env.sh"
ver="${1:-10.0.400}"
if [ -x "$ONEMODEL_HOME/dotnet/dotnet" ] && "$ONEMODEL_HOME/dotnet/dotnet" --list-sdks | grep -q "^$ver "; then
  echo "SDK $ver already installed"; exit 0
fi
curl -LsSf https://dot.net/v1/dotnet-install.sh -o "$ONEMODEL_BIN/dotnet-install.sh"
bash "$ONEMODEL_BIN/dotnet-install.sh" --version "$ver" --install-dir "$ONEMODEL_HOME/dotnet" --no-path
"$ONEMODEL_HOME/dotnet/dotnet" --list-sdks
