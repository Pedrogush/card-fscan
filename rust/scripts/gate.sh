#!/usr/bin/env bash
# Accuracy gate: scan one or more test sets with a given fscan binary and score
# them with tools/eval.py.
#
#   [FSCAN_ARGS="--fast-model ..."] rust/scripts/gate.sh <label> [binary] [set...]
#
# Results go to rust/results/<label>/<set>/ (gitignored); a one-line summary
# per set is printed and appended to rust/results/gate_summary.txt.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
label="$1"; shift
bin="${1:-$root/rust/target/release/fscan.exe}"; shift || true
sets=("$@"); [ ${#sets[@]} -eq 0 ] && sets=(smoke dev)
for set in "${sets[@]}"; do
  out="$root/rust/results/$label/$set"
  rm -rf "$out"; mkdir -p "$out"
  # shellcheck disable=SC2086  (FSCAN_ARGS is meant to split into words)
  "$bin" scan ${FSCAN_ARGS:-} --out "$out" "$root"/testdata/"$set"/*.jpg > "$out.log" 2>&1
  (cd "$root" && uv run --project tools tools/eval.py --manifest "testdata/$set/manifest.json" \
      --results "rust/results/$label/$set" --quiet > "$out.eval.txt" 2>&1) || true
  # Mean per-photo wall ms, OCR lines and CPU-heavy OCR thread time from the JSONs.
  stats=$(python - "$out" <<'PY'
import json, sys, pathlib
ts = [json.load(open(p, encoding="utf-8"))["timing_ms"] for p in pathlib.Path(sys.argv[1]).glob("*.json")]
m = lambda f: sum(f(t) for t in ts) / len(ts)
print(f"ms/photo {m(lambda t: t['total']):.0f}  ocr-lines/photo {m(lambda t: t.get('counters', {}).get('ocr_lines', 0)):.1f}"
      f"  ocr-px-wide/photo {m(lambda t: t.get('counters', {}).get('ocr_input_px_wide', 0)):.0f}")
PY
)
  total=$(grep -E "^TOTAL" "$out.eval.txt" | sed 's/  */ /g')
  cols=$(grep -E "^column-count errors" "$out.eval.txt")
  st=$(grep -E "^status errors" "$out.eval.txt")
  res=$(grep -E "^RESULT" "$out.eval.txt")
  line="$label $set | $total | $cols | $st | $res | $stats"
  echo "$line"
  echo "$(date +%F\ %T) $line" >> "$root/rust/results/gate_summary.txt"
done
