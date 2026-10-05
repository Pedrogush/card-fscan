"""Reference implementation of SPEC section 3 (clean / normalize / ratio / accept).

This is the executable definition both language tracks are tested against via
``testdata/names/match_cases.json``. Keep it small and literal.

Usage:
    uv run tools/reference_match.py "Lightning Bolt {R}" ["Relampago" ...]
    uv run tools/reference_match.py --make-cases   # regenerate match_cases.json
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "testdata" / "names" / "names_v1.json"
CASES_PATH = ROOT / "testdata" / "names" / "match_cases.json"

AUTO_MIN = 90.0
AUTO_LEAD = 5.0
SHORT_KEY_LEN = 6
SHORT_MIN = 95.0
EPS = 1e-6  # threshold comparisons tolerate float noise (SPEC 3)
TRAILING_JUNK = set("{}()[]0123456789@#*%&+=|\\/")


def normalize(s: str) -> str:
    """NFKD, drop combining marks, lowercase, non [a-z0-9 ] -> space, collapse."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = "".join(c if ("a" <= c <= "z" or "0" <= c <= "9" or c == " ") else " " for c in s)
    return " ".join(s.split())


def has_letter(s: str) -> bool:
    return any(c.isalpha() for c in s)


def clean(raw: str) -> str:
    """Strip mana-cost residue from the end of an OCR line (SPEC 3, 'Cleaning').

    1. Split on whitespace; drop trailing tokens that contain no letter.
    2. Strip a trailing run of chars from ``{}()[]0123456789@#*%&+=|\\/`` and
       whitespace.
    3. Trim.
    """
    tokens = raw.split()
    while tokens and not has_letter(tokens[-1]):
        tokens.pop()
    s = " ".join(tokens)
    end = len(s)
    while end > 0 and (s[end - 1] in TRAILING_JUNK or s[end - 1].isspace()):
        end -= 1
    return s[:end].strip()


def ratio(a: str, b: str) -> float:
    """100 * (1 - indel(a, b) / (len(a) + len(b))) == rapidfuzz.fuzz.ratio."""
    return float(fuzz.ratio(a, b))


@dataclass
class Candidate:
    name: str
    oracle_id: str
    lang: str
    key: str
    score: float


@dataclass
class MatchResult:
    status: str  # auto | review | empty
    cleaned: str
    key: str
    best: Candidate | None = None
    runner_up_score: float = 0.0
    candidates: list[Candidate] = field(default_factory=list)


class NameIndex:
    def __init__(self, entries: list[dict]):
        self.entries = entries
        self.keys = [e["key"] for e in entries]
        oracle_ids = sorted({e["oracle_id"] for e in entries})
        self.oracle_ids = oracle_ids
        pos = {o: i for i, o in enumerate(oracle_ids)}
        self.entry_oracle = np.array([pos[e["oracle_id"]] for e in entries], dtype=np.int64)

    @classmethod
    def load(cls, path: Path = INDEX_PATH) -> "NameIndex":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["entries"])

    def match(self, raw: str) -> MatchResult:
        if not has_letter(raw):
            return MatchResult("empty", "", "")
        cleaned = clean(raw)
        key = normalize(cleaned)
        if not key:
            return MatchResult("empty", cleaned, key)
        scores = process.cdist([key], self.keys, scorer=fuzz.ratio, dtype=np.float32, workers=1)[0]
        scores = scores.astype(np.float64)
        # Best score per oracle_id; ties inside an oracle_id go to the first
        # entry in index order.
        per_oracle = np.full(len(self.oracle_ids), -1.0)
        np.maximum.at(per_oracle, self.entry_oracle, scores)
        # Order oracle_ids by score desc, then oracle_id asc (stable).
        order = np.lexsort((np.arange(len(per_oracle)), -per_oracle))
        top = order[:3]
        cands: list[Candidate] = []
        for oi in top:
            idx = np.nonzero((self.entry_oracle == oi) & (scores == per_oracle[oi]))[0][0]
            e = self.entries[int(idx)]
            cands.append(Candidate(e["name"], e["oracle_id"], e["lang"], e["key"], round(float(per_oracle[oi]), 4)))
        best = cands[0]
        runner = cands[1].score if len(cands) > 1 else 0.0
        accept = best.score >= AUTO_MIN - EPS and best.score - runner >= AUTO_LEAD - EPS
        if len(best.key) <= SHORT_KEY_LEN and best.score < SHORT_MIN - EPS:
            accept = False
        return MatchResult("auto" if accept else "review", cleaned, key, best, runner, cands)


# --------------------------------------------------------------------------- #
# Test-vector generation


def _find(index: NameIndex, name: str, lang: str | None = None) -> dict:
    for e in index.entries:
        if e["name"] == name and (lang is None or e["lang"] == lang):
            return e
    raise KeyError(name)


# (raw OCR string, what the case exercises). Raw strings are hand-written to
# look like what an OCR engine returns from a name-bar crop.
RAW_CASES: list[tuple[str, str]] = [
    # exact / easy
    ("Lightning Bolt", "exact EN"),
    ("Llanowar Elves", "exact EN"),
    ("Serra Angel", "exact EN"),
    ("Counterspell", "exact EN single word"),
    ("Shivan Dragon 4RR", "mana residue glued as token with letters"),
    ("Lightning Bolt {R}", "brace mana residue containing a letter"),
    ("Giant Growth G", "single-letter mana residue"),
    ("Serra Angel 3 @@", "letterless trailing tokens"),
    ("Counterspell }}", "brace residue"),
    ("Llanowar Elves (G)", "paren residue"),
    ("Dark Ritual   ", "trailing whitespace"),
    ("  Dark Ritual", "leading whitespace"),
    # case / punctuation
    ("LIGHTNING BOLT", "all caps"),
    ("lightning bolt", "lowercase"),
    ("Jace, the Mind Sculptor", "comma"),
    ("Jace the Mind Sculptor", "missing comma"),
    ("Akroma's Vengeance", "apostrophe"),
    ("Akromas Vengeance", "missing apostrophe"),
    ("Akroma’s Vengeance", "curly apostrophe"),
    ("Will-o'-the-Wisp", "hyphens and apostrophe"),
    ("Will o the Wisp", "hyphens dropped"),
    ("Borrowing 100,000 Arrows", "digits inside name"),
    # OCR errors
    ("Lightnimg Bolt", "one substitution"),
    ("Ligthning Bolt", "transposition"),
    ("Shivan Dragcn", "one substitution near end"),
    ("Serra Angcl", "one substitution, short-ish"),
    ("Llanowar Elues", "u for v"),
    ("Wrath of Gcd", "one substitution, near-duplicate names exist"),
    ("Counterspel1", "1 for l at the end (stripped as residue)"),
    ("Lightning B", "truncated"),
    ("htning Bolt", "truncated at the left"),
    # short names (<= 6 chars need 95)
    ("Opt", "short exact"),
    ("Opt 0", "short with residue"),
    ("0pt", "short with OCR error"),
    ("Shock", "short exact"),
    ("Shack", "short one-off (Shock vs Shack?)"),
    ("Duress", "6-char exact"),
    ("Durcss", "6-char one-off"),
    ("Fog", "3-char exact"),
    ("Fug", "3-char one-off"),
    # multi-face
    ("Fire // Ice", "split full name"),
    ("Fire", "split face"),
    ("Bonecrusher Giant", "adventure creature face"),
    ("Bonecrusher Giant // Stomp", "adventure full name"),
    ("Delver of Secrets", "transform front face"),
    ("Insectile Aberration", "transform back face"),
    ("Brazen Borrower", "adventure front"),
    # near-duplicates
    ("Ajani's Pridemate", "near duplicate family"),
    ("Elvish Mystic", "near duplicate: Elvish Mystic vs Elvish Mysticism?"),
    ("Pacifism", "exact; Pacifism vs Pacifist"),
    ("Mind Rot", "short-ish; Mind Rot vs Mind Ravel etc"),
    ("Giant Spider", "Giant Spider vs Giant Spiders?"),
    ("Goblin Guide", "Goblin Guide vs Goblin Glide"),
    ("Grizzly Bears", "exact"),
    ("Grizzly Bear", "missing s: Grizzly Bears vs Grizzled Bear?"),
    # Portuguese
    ("Relâmpago", "PT with accent"),
    ("Relampago", "PT accent missing"),
    ("Raio", "PT short name (Lightning Bolt)"),
    ("Anjo de Serra", "PT"),
    ("Elfos de Llanowar", "PT"),
    ("Anulação", "PT cedilla + tilde"),
    ("Anulacao", "PT cedilla + tilde stripped"),
    ("Contramágica", "PT Counterspell"),
    ("Dragão Shivano", "PT tilde"),
    ("Crescimento Gigante", "PT"),
    ("Fogo // Gelo", "PT split full"),
    ("Gigante Esmaga-ossos", "PT adventure face with hyphen"),
    ("Ritual Sombrio", "PT"),
    # junk / empty
    ("", "empty"),
    ("   ", "whitespace only"),
    ("{2}{R}", "mana only"),
    ("12 3 @#", "symbols only"),
    ("xqzv", "garbage letters"),
    ("The", "common word"),
]


def make_cases(index: NameIndex) -> list[dict]:
    out = []
    for raw, note in RAW_CASES:
        r = index.match(raw)
        case: dict = {
            "raw": raw,
            "note": note,
            "cleaned": r.cleaned,
            "key": r.key,
            "status": r.status,
            "oracle_id": None,
            "name": None,
            "lang": None,
            "best_score": None,
            "runner_up_score": None,
            "best_oracle_id": None,
        }
        if r.best is not None:
            case["best_score"] = round(r.best.score, 2)
            case["runner_up_score"] = round(r.runner_up_score, 2)
            # Only pin best_oracle_id when it is unambiguous.
            if r.best.score > r.runner_up_score:
                case["best_oracle_id"] = r.best.oracle_id
                case["best_name"] = r.best.name
        if r.status == "auto":
            assert r.best is not None
            case["oracle_id"] = r.best.oracle_id
            case["name"] = r.best.name
            case["lang"] = r.best.lang
        out.append(case)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", nargs="*")
    ap.add_argument("--make-cases", action="store_true")
    ap.add_argument("--index", type=Path, default=INDEX_PATH)
    args = ap.parse_args()
    index = NameIndex.load(args.index)
    if args.make_cases:
        cases = make_cases(index)
        doc = {
            "spec_version": 1,
            "index": "testdata/names/names_v1.json",
            "generated_by": "tools/reference_match.py --make-cases",
            "comment": (
                "Expected values come from running tools/reference_match.py. "
                "Scores are rounded to 2 decimals (compare with tolerance 0.01). "
                "oracle_id/name/lang are set only for status 'auto'. "
                "best_oracle_id is null when the top score is tied across oracle_ids."
            ),
            "cases": cases,
        }
        CASES_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        counts: dict[str, int] = {}
        for c in cases:
            counts[c["status"]] = counts.get(c["status"], 0) + 1
        print(f"wrote {len(cases)} cases to {CASES_PATH}: {counts}")
        return
    for raw in args.raw:
        r = index.match(raw)
        print(json.dumps({
            "raw": raw, "cleaned": r.cleaned, "key": r.key, "status": r.status,
            "runner_up": r.runner_up_score,
            "candidates": [c.__dict__ for c in r.candidates],
        }, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    main()
