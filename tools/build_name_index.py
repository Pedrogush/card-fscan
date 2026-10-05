"""Build the card-fscan name index (SPEC section 3) from MTGJSON AtomicCards.

Output: testdata/names/names_v1.json

    { "version": 1, "source": "MTGJSON AtomicCards <date>",
      "entries": [ {"key", "name", "oracle_id", "lang"}, ... ] }

* English entries: every atomic card's full name (``A // B`` for multi-face
  cards) and every face name.
* Portuguese entries: ``foreignData`` with language "Portuguese (Brazil)",
  full name and face name, same oracle_id.
* Secondary faces of adventure-style layouts (adventure, prepare) are not
  indexed by face name: that name sits in the text box, not the name bar,
  and often collides with a real card (e.g. "Lightning Bolt").
* Excluded layouts: tokens, art cards, emblems and non-deck oversized cards
  (planes, schemes, vanguards). Un-set ("funny") cards are kept because they
  can turn up in donated bulk.

Entries are de-duplicated on (key, oracle_id, lang, name) and sorted by
(lang, key, oracle_id, name) so the output is byte-stable for a given input.

Usage:  uv run tools/build_name_index.py [--refresh]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_match import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "testdata" / "cache"
OUT = ROOT / "testdata" / "names" / "names_v1.json"
URL = "https://mtgjson.com/api/v5/AtomicCards.json.gz"
USER_AGENT = "card-fscan-synthgen/0.1 (+https://github.com/Pedrogush/card-fscan)"

EXCLUDED_LAYOUTS = {
    "token", "double_faced_token", "art_series", "emblem",
    "planar", "scheme", "vanguard",
}
PT_LANGUAGE = "Portuguese (Brazil)"
# Layouts whose secondary face name is printed inside the text box, never on
# the name bar. Their b-face names often reprint classic names ("Lightning
# Bolt" on Emeritus of Conflict) and would make the real card ambiguous.
INSET_FACE_LAYOUTS = {"adventure", "prepare"}


def download(refresh: bool) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "AtomicCards.json.gz"
    if path.exists() and not refresh:
        return path
    print(f"downloading {URL}", file=sys.stderr)
    with requests.get(URL, headers={"User-Agent": USER_AGENT}, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = path.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.replace(path)
    return path


def build(atomic_path: Path) -> dict:
    with gzip.open(atomic_path, "rt", encoding="utf-8") as f:
        doc = json.load(f)
    meta = doc["meta"]
    seen: set[tuple[str, str, str, str]] = set()

    def add(name: str | None, oracle_id: str, lang: str) -> None:
        if not name:
            return
        name = name.strip()
        key = normalize(name)
        if not key:
            return
        seen.add((lang, key, oracle_id, name))

    for faces in doc["data"].values():
        for face in faces:
            if face.get("layout") in EXCLUDED_LAYOUTS:
                continue
            oracle_id = face.get("identifiers", {}).get("scryfallOracleId")
            if not oracle_id:
                continue
            inset_face = face.get("layout") in INSET_FACE_LAYOUTS and face.get("side") not in (None, "a")
            add(face.get("name"), oracle_id, "en")
            if not inset_face:
                add(face.get("faceName"), oracle_id, "en")
            for fd in face.get("foreignData", []):
                if fd.get("language") != PT_LANGUAGE:
                    continue
                add(fd.get("name"), oracle_id, "pt")
                if not inset_face:
                    add(fd.get("faceName"), oracle_id, "pt")

    entries = [
        {"key": k, "name": n, "oracle_id": o, "lang": l}
        for (l, k, o, n) in sorted(seen)
    ]
    return {
        "version": 1,
        "source": f"MTGJSON AtomicCards {meta['date']} (v{meta['version']})",
        "entries": entries,
    }


def write(index: dict, out: Path) -> None:
    """One entry per line: diff-friendly and still plain JSON."""
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "{",
        f'  "version": {index["version"]},',
        f'  "source": {json.dumps(index["source"], ensure_ascii=False)},',
        '  "entries": [',
    ]
    body = [
        "    " + json.dumps(e, ensure_ascii=False, separators=(", ", ": "))
        for e in index["entries"]
    ]
    lines.append(",\n".join(body))
    lines += ["  ]", "}"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-download AtomicCards")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    index = build(download(args.refresh))
    write(index, args.out)
    en = sum(1 for e in index["entries"] if e["lang"] == "en")
    pt = sum(1 for e in index["entries"] if e["lang"] == "pt")
    oracles = len({e["oracle_id"] for e in index["entries"]})
    size = args.out.stat().st_size
    print(f"{args.out}: {en} en + {pt} pt entries, {oracles} oracle_ids, {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
