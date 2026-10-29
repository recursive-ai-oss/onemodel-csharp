#!/usr/bin/env bash
# Runs bar rows under the benchmark lock.
#
#   run-bar.sh --what "minilm+resnet fp32 b1" --minutes 10 --out DIR \
#              --rows "minilm_l6:fp32:1 resnet50:fp32:1" [--threads "1 all"] [--side ort|ours] [--mode fast|exact] [--reps 3]
#
# * Builds Bar first (before taking the lock), then claims the lock atomically from inside this
#   script, and removes it on exit. It refuses to start when the lock exists or the machine is busy.
# * Each row runs in its own process under /usr/bin/time so peak RSS is that engine's alone.
# * Results append to DIR/bar.csv and DIR/bar.md.
# Environment: ONEMODEL_LOCK (default ~/WORKER-BENCH.lock), ONEMODEL_IGNORE_LOAD=1 to skip the
# busy check (the loads are still recorded in every row).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tools="$here/../../tools"
. "$tools/env.sh"

what="bar"; minutes=15; out=""; rows=""; threads="1 all"; side="ort"; mode="fast"; reps=3
while [ $# -gt 0 ]; do
  case "$1" in
    --what) what="$2";; --minutes) minutes="$2";; --out) out="$2";; --rows) rows="$2";;
    --threads) threads="$2";; --side) side="$2";; --mode) mode="$2";; --reps) reps="$2";;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
  shift 2
done
[ -n "$out" ] && [ -n "$rows" ] || { echo "need --out and --rows" >&2; exit 2; }
mkdir -p "$out"

lock="${ONEMODEL_LOCK:-$HOME/WORKER-BENCH.lock}"
if [ -e "$lock" ]; then echo "lock held: $(cat "$lock"); not starting" >&2; exit 75; fi

dotnet build -c Release "$here/Bar.csproj" -o "$out/bin" -v q -clp:NoSummary
bar="$out/bin/Bar.dll"
sdk="$(dotnet --version)"

if [ "$(uname -s)" = Darwin ]; then
  cores="$(sysctl -n hw.ncpu)"; load1="$(sysctl -n vm.loadavg | tr -d '{}' | awk '{print $1}')"
  tm=(/usr/bin/time -l -o)
else
  cores="$(nproc)"; load1="$(cut -d' ' -f1 /proc/loadavg)"
  tm=(/usr/bin/time -v -o)
fi

# The lock is claimed here, inside the job that uses it.
end="$("$ONEMODEL_PY" -c "import datetime as d;print((d.datetime.utcnow()+d.timedelta(minutes=$minutes)).strftime('%H:%MZ'))" 2>/dev/null || echo "+${minutes}min")"
label="onemodel bar $what $(date -u +%H:%MZ) $end"
set -o noclobber
echo "$label" > "$lock" || { echo "lost the race for the lock: $(cat "$lock")" >&2; exit 75; }
set +o noclobber
trap 'rm -f "$lock"' EXIT

if [ -z "${ONEMODEL_IGNORE_LOAD:-}" ] && awk -v l="$load1" -v c="$cores" 'BEGIN{exit !(l > c/2)}'; then
  echo "machine busy (load $load1 on $cores cores); not starting" >&2; exit 75
fi

for spec in $rows; do
  read -r model_file ref_dir < <("$ONEMODEL_PY" "$tools/model_path.py" "$spec")
  IFS=: read -r name variant batch <<<"$spec"
  for t in $threads; do
    n="$t"; [ "$t" = all ] && n="$cores"
    id="${name}_${variant}_b${batch}_t${t}_${side}"; [ "$side" = ours ] && id="${id}-${mode}"
    time_out="$out/$id.time"; row="$out/$id.json"
    "${tm[@]}" "$time_out" dotnet "$bar" run --model "$model_file" --ref "$ref_dir" --model-name "$name" \
      --variant "$variant" --batch "$batch" --threads "$n" --side "$side" --mode "$mode" --reps "$reps" \
      --row-out "$row" --sdk "$sdk" --lock-label "$label"
    if [ "$(uname -s)" = Darwin ]; then
      rss_mb="$(awk '/maximum resident set size/{printf "%.1f", $1/1048576}' "$time_out")"
    else
      rss_mb="$(awk -F: '/Maximum resident set size/{printf "%.1f", $2/1024}' "$time_out")"
    fi
    dotnet "$bar" emit --row "$row" --rss-mb "$rss_mb" --csv "$out/bar.csv" --md "$out/bar.md"
  done
done
echo "rows in $out/bar.csv"
