"""Score scanner output against a synthgen manifest (SPEC section 5).

  uv run --project tools tools/eval.py --manifest testdata/smoke/manifest.json --results out/

For every manifest image the result is read from ``<results>/<stem>.json``
(``c4_0001.json``) or ``<results>/<basename>.json`` (``c4_0001.jpg.json``).

Metrics
  wrong          slot reported ``auto`` whose oracle_id differs from the truth,
                 or an ``auto`` slot where no card lies (beyond n_cards). Target 0.
  auto rate      auto & correct / scored true slots. Scored = every true slot
                 except ``lang: other`` printings and slots tagged
                 ``split_sideways`` (those may legitimately be review).
                 The rate over *all* true slots is printed too.
  column count   n_cards and stop_card must match exactly (not scored on
                 negative columns that carry ``expect_column_status``).
  status         photo status must equal ``expect_status``; negative columns
                 must have the expected column status (``error``).
Retake images are scored on photo status only.

Exit code 1 if anything is wrong, a status or column-count check fails, or
a result file is missing; 0 otherwise. ``--min-auto 0.95`` also fails on a
low auto rate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


class Tally:
    def __init__(self) -> None:
        self.true = 0        # all true slots
        self.scored = 0      # true slots counted in the auto-rate denominator
        self.auto_ok = 0     # auto & correct among scored slots
        self.auto_ok_all = 0  # auto & correct among all slots
        self.wrong = 0
        self.review = 0
        self.empty = 0
        self.missing = 0     # true slot absent from the result

    def line(self) -> str:
        rate = self.auto_ok / self.scored if self.scored else float("nan")
        return (f"slots {self.true:5d}  auto-rate {100 * rate:6.2f}%  wrong {self.wrong:3d}  "
                f"review {self.review:4d}  empty {self.empty:4d}  missing {self.missing:4d}")


def load_result(results: Path, key: str) -> dict[str, Any] | None:
    p = Path(key)
    for cand in (results / f"{p.stem}.json", results / f"{p.name}.json"):
        if cand.exists():
            return json.loads(cand.read_text(encoding="utf-8"))
    return None


def is_scored(slot: dict[str, Any]) -> bool:
    return slot.get("lang") != "other" and "split_sideways" not in slot.get("tags", [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--min-auto", type=float, default=None, help="fail if auto rate is below this (0..1)")
    ap.add_argument("--quiet", action="store_true", help="aggregate only")
    ap.add_argument("--show-wrong", action="store_true", help="list every wrong slot")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    total = Tally()
    by_lang: dict[str, Tally] = defaultdict(Tally)
    by_level: dict[str, Tally] = defaultdict(Tally)
    by_cat: dict[str, Tally] = defaultdict(Tally)
    by_cfg: dict[str, Tally] = defaultdict(Tally)
    status_errors: list[str] = []
    count_errors: list[str] = []
    missing_results: list[str] = []
    wrong_list: list[str] = []
    lang_mismatch = 0

    for key, exp in manifest["images"].items():
        res = load_result(args.results, key)
        level = exp.get("level") or exp.get("difficulty", {}).get("level", "?")
        if res is None:
            missing_results.append(key)
            if not args.quiet:
                print(f"{key:24s} MISSING RESULT")
            continue
        got_status = res.get("status")
        if got_status != exp["expect_status"]:
            status_errors.append(f"{key}: photo status {got_status!r}, expected {exp['expect_status']!r}")
        if exp["expect_status"] == "retake":
            if not args.quiet:
                print(f"{key:24s} {level:6s} retake   -> {got_status}")
            continue
        if res.get("config") != exp["config"]:
            status_errors.append(f"{key}: config {res.get('config')!r}, expected {exp['config']!r}")
        img = Tally()
        rcols = {c.get("column"): c for c in res.get("columns", [])}
        n_col_err = 0
        for col in exp["columns"]:
            j = col["column"]
            rc = rcols.get(j)
            tag = f"{key} col {j}"
            if rc is None:
                count_errors.append(f"{tag}: column missing from result")
                n_col_err += 1
                rc = {"slots": []}
            if "expect_column_status" in col:
                if rc.get("status") != col["expect_column_status"]:
                    status_errors.append(f"{tag}: column status {rc.get('status')!r}, "
                                         f"expected {col['expect_column_status']!r}")
            elif rc.get("n_cards") != col["n_cards"] or bool(rc.get("stop_card")) != col["stop_card"]:
                count_errors.append(f"{tag}: n_cards/stop {rc.get('n_cards')}/{rc.get('stop_card')}, "
                                    f"expected {col['n_cards']}/{col['stop_card']}")
                n_col_err += 1
            truth = {s["slot"]: s for s in col["slots"]}
            rslots = {s.get("slot"): s for s in rc.get("slots", [])}
            # phantom autos (slots where no card lies)
            for i, rs in rslots.items():
                if i not in truth and rs.get("status") == "auto":
                    for t in (img, total, by_level[level], by_cfg[exp["config"]]):
                        t.wrong += 1
                    wrong_list.append(f"{tag} slot {i}: auto {rs.get('name')!r} but no card there")
            for i, ts in truth.items():
                rs = rslots.get(i)
                groups = (img, total, by_lang[ts["lang"]], by_level[level], by_cat[ts.get("category", "?")],
                          by_cfg[exp["config"]])
                scored = is_scored(ts)
                for t in groups:
                    t.true += 1
                    t.scored += scored
                if rs is None:
                    for t in groups:
                        t.missing += 1
                    continue
                st = rs.get("status")
                if st == "auto":
                    if rs.get("oracle_id") == ts["oracle_id"]:
                        for t in groups:
                            t.auto_ok_all += 1
                            t.auto_ok += scored
                        if rs.get("lang") not in (ts["lang"], None) and ts["lang"] != "other":
                            lang_mismatch += 1
                    else:
                        for t in groups:
                            t.wrong += 1
                        wrong_list.append(f"{tag} slot {i}: auto {rs.get('name')!r} ({rs.get('oracle_id')}) "
                                          f"truth {ts['name']!r} ({ts['oracle_id']}) raw={rs.get('raw_text')!r}")
                elif st == "review":
                    for t in groups:
                        t.review += 1
                else:
                    for t in groups:
                        t.empty += 1
        if not args.quiet:
            print(f"{key:24s} {level:6s} {got_status:6s} {img.line()}  col-err {n_col_err}")

    print()
    print(f"=== {manifest.get('set')} : {len(manifest['images'])} images, results from {args.results}")
    print(f"TOTAL         {total.line()}")
    if total.true:
        print(f"              auto & correct over all true slots: {100 * total.auto_ok_all / total.true:.2f}%")
    for title, d in (("lang", by_lang), ("level", by_level), ("config", by_cfg), ("category", by_cat)):
        for k in sorted(d):
            print(f"{title:6s} {k:12s} {d[k].line()}")
    print(f"column-count errors: {len(count_errors)}")
    for e in count_errors:
        print(f"  {e}")
    print(f"status errors: {len(status_errors)}")
    for e in status_errors:
        print(f"  {e}")
    if missing_results:
        print(f"missing results: {len(missing_results)}")
    if lang_mismatch:
        print(f"(info) correct autos reported with a different lang: {lang_mismatch}")
    if wrong_list and (args.show_wrong or len(wrong_list) <= 20):
        print("wrong slots:")
        for w in wrong_list:
            print(f"  {w}")

    rate = total.auto_ok / total.scored if total.scored else 0.0
    failed = bool(total.wrong or status_errors or count_errors or missing_results)
    if args.min_auto is not None and rate < args.min_auto:
        print(f"auto rate {100 * rate:.2f}% below --min-auto {100 * args.min_auto:.1f}%")
        failed = True
    print("RESULT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
