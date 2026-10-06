"""Average the timing_ms block over a results directory.

    python rust/scripts/profile_summary.py rust/results/<label>/<set>
"""
import json
import pathlib
import sys

ts = [json.load(open(p, encoding="utf-8"))["timing_ms"] for p in pathlib.Path(sys.argv[1]).glob("*.json")]
n = len(ts)
mean = lambda f: sum(f(t) for t in ts) / n
total = mean(lambda t: t["total"])
print(f"{n} photos, mean total {total:.0f} ms")
for k in ["decode", "gray", "detect", "crop", "ocr", "match", "retry", "json", "scan"]:
    if k in ts[0]:
        print(f"  {k:10s} {mean(lambda t: t[k]):8.1f} ms  {100 * mean(lambda t: t[k]) / total:5.1f}%")
print("  thread time per stage:")
for k in ts[0].get("stages", {}):
    print(f"  {k:10s} {mean(lambda t: t['stages'][k]):8.1f} ms")
print("  counters:")
for k in ts[0].get("counters", {}):
    print(f"  {k:18s} {mean(lambda t: t['counters'][k]):8.1f}")
