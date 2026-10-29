#!/usr/bin/env bash
# End to end: oracle venv, models, int8 variants, backend-test data, reference outputs.
# Idempotent. Models that are not byte-reproducible across machines (see docs/oracles.md) are
# reported by export_models.py; copy the canonical file from the machine that recorded it.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/env.sh"
"$here/setup-oracles.sh"
cd "$here"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
./light.sh "$ONEMODEL_PY" export_models.py
./light.sh "$ONEMODEL_PY" quantize_models.py
./light.sh "$ONEMODEL_PY" fetch_backend_test.py
"$ONEMODEL_PY" verify_models.py
OMP_NUM_THREADS=1 ./light.sh "$ONEMODEL_PY" make_reference.py
