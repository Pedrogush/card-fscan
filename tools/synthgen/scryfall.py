"""Cached, polite Scryfall access and the card pool used by the generator.

Everything fetched (API responses, bulk files, card images) is cached under
``testdata/cache/scryfall/`` so reruns are offline. API calls are spaced
>= 100 ms apart and carry a descriptive User-Agent, per Scryfall's guidelines.

The pool is built once (``pool_v1.json`` in the cache) from:
  * EN: the ``unique_artwork`` bulk file (one printing per artwork, so old
    frames, showcase and borderless treatments are all represented);
  * PT: every ``lang:pt`` printing (paged search);
  * other languages: a few search pages per language.
Each candidate's visible name must exist in ``names_v1.json`` under the same
oracle_id and that key must not map to any other oracle_id, so every truth
slot is actually solvable by a SPEC-conforming reader.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import requests

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
from reference_match import normalize  # noqa: E402

ROOT = TOOLS.parent
CACHE = ROOT / "testdata" / "cache" / "scryfall"
NAMES = ROOT / "testdata" / "names" / "names_v1.json"
USER_AGENT = "card-fscan-synthgen/0.1 (+https://github.com/Pedrogush/card-fscan)"
API_DELAY_S = 0.12
POOL_SEED = 20261005

# Pool versions. A manifest stores every slot's printing and image URL, so
# --regen never needs the pool; the version only matters when planning a new
# set (smoke/dev were planned with v1, large with v2). Quotas are distinct
# printings per category; cards are reused across images.
POOL_VERSIONS: dict[int, dict[str, Any]] = {
    1: {
        "quota": {
            "en_old": 220,        # frame 1993 / 1997
            "en_modern": 520,     # frame 2003 / 2015, regular border
            "en_special": 140,    # borderless, showcase, extended art
            "en_multiface": 160,  # adventure, transform, modal_dfc, flip, aftermath
            "en_split": 25,       # sideways split cards (tricky: name not on top edge)
            "pt": 500,
            "other": 90,          # ja/de/fr/es/it/ru/ko/zhs printings
        },
        "other_pages": None,      # v1: first 2 pages (order=set) per language
    },
    2: {
        "quota": {
            "en_old": 1000, "en_modern": 2100, "en_special": 650, "en_multiface": 650, "en_split": 80,
            "pt": 1600, "other": 480,
        },
        "other_pages": 9,         # v2: 9 random result pages per language (many sets)
    },
}


def pool_path(version: int) -> Path:
    return CACHE / f"pool_v{version}.json"


OTHER_LANGS = ["de", "fr", "es", "it", "ja", "ru", "ko", "zhs"]
EXCLUDED_LAYOUTS = {
    "token", "double_faced_token", "emblem", "art_series", "planar", "scheme",
    "vanguard", "reversible_card", "augment", "host",
}
EXCLUDED_SET_TYPES = {"token", "memorabilia", "minigame", "alchemy", "vanguard", "planechase", "archenemy", "treasure_chest"}
MULTIFACE_LAYOUTS = {"adventure", "transform", "modal_dfc", "flip", "aftermath", "prepare"}


class Scryfall:
    def __init__(self, offline: bool = False) -> None:
        self.offline = offline
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json;q=0.9,*/*;q=0.8"})
        self._last_api = 0.0
        CACHE.mkdir(parents=True, exist_ok=True)

    # -- low level ---------------------------------------------------------
    def _wait(self) -> None:
        dt = time.monotonic() - self._last_api
        if dt < API_DELAY_S:
            time.sleep(API_DELAY_S - dt)
        self._last_api = time.monotonic()

    def _get(self, url: str, api: bool) -> bytes:
        if self.offline:
            raise RuntimeError(f"offline mode and not cached: {url}")
        for attempt in range(5):
            if api:
                self._wait()
            r = self.session.get(url, timeout=120)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.content
        raise RuntimeError(f"giving up on {url}")

    def api_json(self, url: str) -> dict[str, Any]:
        key = hashlib.sha1(url.encode()).hexdigest()[:16]
        path = CACHE / "api" / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self._get(url, api=True)
        try:
            doc = json.loads(data)
        except json.JSONDecodeError:
            raise
        if doc.get("object") == "error" and doc.get("status") != 404:
            raise RuntimeError(f"scryfall error {doc}")
        path.write_text(json.dumps(doc), encoding="utf-8")
        return doc

    def search(self, q: str, max_pages: int | None = None) -> Iterator[dict[str, Any]]:
        url: str | None = "https://api.scryfall.com/cards/search?" + urllib.parse.urlencode(
            {"q": q, "unique": "prints", "order": "set", "include_extras": "false"})
        pages = 0
        while url:
            doc = self.api_json(url)
            if doc.get("object") == "error":
                return
            yield from doc["data"]
            pages += 1
            if max_pages is not None and pages >= max_pages:
                return
            url = doc.get("next_page") if doc.get("has_more") else None

    def search_page(self, q: str, page: int) -> dict[str, Any]:
        url = "https://api.scryfall.com/cards/search?" + urllib.parse.urlencode(
            {"q": q, "unique": "prints", "order": "set", "include_extras": "false", "page": page})
        return self.api_json(url)

    def bulk_jsonl(self, kind: str) -> Path:
        """Download (once) a Scryfall bulk file as .jsonl.gz and return its path."""
        path = CACHE / f"bulk_{kind}.jsonl.gz"
        if path.exists():
            return path
        meta = self.api_json("https://api.scryfall.com/bulk-data")
        uri = next(d["jsonl_download_uri"] for d in meta["data"] if d["type"] == kind)
        print(f"downloading {uri}", file=sys.stderr)
        tmp = path.with_suffix(".part")
        with self.session.get(uri, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        tmp.replace(path)
        (CACHE / f"bulk_{kind}.source.txt").write_text(uri + "\n", encoding="utf-8")
        return path

    def image(self, url: str) -> bytes:
        """Card image from the CDN, cached by URL path (query string ignored)."""
        p = urllib.parse.urlparse(url)
        name = p.path.strip("/").replace("/", "_")
        path = CACHE / "img" / name
        if path.exists():
            return path.read_bytes()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self._get(url, api=False)
        time.sleep(0.05)  # be gentle with the CDN too
        path.write_bytes(data)
        return data


# --------------------------------------------------------------------------- #
# Pool

_STAMP = None


def is_stamped(jpeg: bytes) -> bool:
    """True for Scryfall's 'Localized Image Not Available' stand-ins: an
    English scan with a grey banner, served for some non-English printings
    whose image_status still says 'lowres'. Detected by matching the banner's
    white text at its fixed position in the 672x936 'large' image."""
    import cv2

    global _STAMP
    if _STAMP is None:
        _STAMP = cv2.imread(str(Path(__file__).with_name("stamp_template.png")), cv2.IMREAD_GRAYSCALE) > 127
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None or img.shape != (936, 672):
        return False
    crop = img[325:420, 230:530] > 200
    inter = np.logical_and(crop, _STAMP).sum()
    union = np.logical_or(crop, _STAMP).sum()
    return union > 0 and inter / union > 0.45


def _load_name_keys() -> dict[str, set[str]]:
    with open(NAMES, encoding="utf-8") as f:
        entries = json.load(f)["entries"]
    keys: dict[str, set[str]] = {}
    for e in entries:
        keys.setdefault(e["key"], set()).add(e["oracle_id"])
    return keys


def _face0(card: dict[str, Any]) -> dict[str, Any]:
    faces = card.get("card_faces")
    if faces and "image_uris" in faces[0]:
        return faces[0]
    return card


def _visible(card: dict[str, Any], lang: str) -> tuple[str, str | None]:
    """(printed name to show as truth, image URL) for the front of the card."""
    layout = card.get("layout")
    faces = card.get("card_faces") or []
    if layout == "split":
        name = card.get("printed_name") or card["name"] if lang != "en" else card["name"]
    elif faces and layout != "split":
        f0 = faces[0]
        name = f0.get("printed_name") if lang != "en" else f0["name"]
    else:
        name = card.get("printed_name") if lang != "en" else card["name"]
    if not name:
        # Non-English printing without printed_name: the scan may well be
        # localised, but the truth name cannot be verified -> unusable.
        return "", None
    img = _face0(card).get("image_uris", {}).get("large")
    return name, img


def _eligible(card: dict[str, Any], allow_lowres: bool = False) -> bool:
    # Most non-English scans on Scryfall are 'lowres' (still 672x936 'large',
    # just softer); 'placeholder' means an English stand-in image, never OK.
    ok_status = {"highres_scan", "lowres"} if allow_lowres else {"highres_scan"}
    return (
        not card.get("digital")
        and not card.get("oversized")
        and card.get("layout") not in EXCLUDED_LAYOUTS
        and card.get("set_type") not in EXCLUDED_SET_TYPES
        and card.get("image_status") in ok_status
        and bool(card.get("oracle_id"))
        and not card.get("textless")
    )


def _category_en(card: dict[str, Any]) -> str | None:
    layout = card.get("layout")
    if layout == "split":
        return "en_split"
    if layout in MULTIFACE_LAYOUTS:
        return "en_multiface"
    effects = set(card.get("frame_effects") or [])
    if card.get("border_color") == "borderless" or effects & {"showcase", "extendedart"}:
        return "en_special"
    frame = card.get("frame")
    if frame in ("1993", "1997"):
        return "en_old"
    if frame in ("2003", "2015"):
        return "en_modern"
    return None


def _record(card: dict[str, Any], cat: str, lang: str, name: str, img: str) -> dict[str, Any]:
    return {
        "category": cat,
        "scryfall_id": card["id"],
        "oracle_id": card["oracle_id"],
        "name": name,
        "lang": lang,
        "printed_lang": card.get("lang"),
        "set": card.get("set"),
        "collector_number": card.get("collector_number"),
        "frame": card.get("frame"),
        "layout": card.get("layout"),
        "border_color": card.get("border_color"),
        "image_uri": img,
    }


def build_pool(sf: Scryfall, version: int = 2) -> dict[str, list[dict[str, Any]]]:
    path = pool_path(version)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    quota_map: dict[str, int] = POOL_VERSIONS[version]["quota"]
    other_pages: int | None = POOL_VERSIONS[version]["other_pages"]
    keys = _load_name_keys()

    def solvable(name: str, oracle_id: str) -> bool:
        return keys.get(normalize(name)) == {oracle_id}

    cands: dict[str, list[dict[str, Any]]] = {c: [] for c in quota_map}
    # EN from unique_artwork
    with gzip.open(sf.bulk_jsonl("unique_artwork"), "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line.startswith("{"):
                continue
            card = json.loads(line)
            if card.get("lang") != "en" or not _eligible(card):
                continue
            cat = _category_en(card)
            if cat is None:
                continue
            name, img = _visible(card, "en")
            check = card["name"] if cat == "en_split" else name
            if img and solvable(check, card["oracle_id"]):
                cands[cat].append(_record(card, cat, "en", check, img))
    # PT: every lang:pt printing
    for card in sf.search("lang:pt"):
        if not _eligible(card, allow_lowres=True):
            continue
        name, img = _visible(card, "pt")
        if card.get("layout") == "split":
            continue
        if img and solvable(name, card["oracle_id"]):
            cands["pt"].append(_record(card, "pt", "pt", name, img))
    # other languages
    page_rng = np.random.default_rng([POOL_SEED, 7])
    for lg in OTHER_LANGS:
        if other_pages is None:
            cards: Iterator[dict[str, Any]] | list[dict[str, Any]] = sf.search(f"lang:{lg}", max_pages=2)
        else:
            first = sf.search_page(f"lang:{lg}", 1)
            n_pages = max(1, -(-int(first.get("total_cards", 0)) // 175))
            pages = sorted(int(p) + 1 for p in page_rng.permutation(n_pages)[:other_pages])
            cards = [c for pg in pages for c in sf.search_page(f"lang:{lg}", pg).get("data", [])]
        for card in cards:
            if not _eligible(card, allow_lowres=True) or card.get("layout") == "split":
                continue
            name, img = _visible(card, lg)
            if img:
                cands["other"].append(_record(card, "other", "other", name, img))

    rng = np.random.default_rng(POOL_SEED)
    pool: dict[str, list[dict[str, Any]]] = {}
    for cat, quota in quota_map.items():
        items = sorted(cands[cat], key=lambda r: r["scryfall_id"])
        if cat in ("pt", "other"):
            # one printing per oracle_id keeps the pool diverse
            seen: set[str] = set()
            uniq = []
            for i in rng.permutation(len(items)):
                r = items[int(i)]
                if r["oracle_id"] not in seen:
                    seen.add(r["oracle_id"])
                    uniq.append(r)
            items = sorted(uniq, key=lambda r: r["scryfall_id"])
        order = rng.permutation(len(items))
        if cat in ("pt", "other"):
            chosen = []
            for i in order:
                if len(chosen) >= quota:
                    break
                if not is_stamped(sf.image(items[int(i)]["image_uri"])):
                    chosen.append(int(i))
            idx = np.array(chosen, dtype=np.int64)
        else:
            idx = order[:quota]
        pool[cat] = [items[int(i)] for i in sorted(idx)]
        print(f"pool {cat}: {len(pool[cat])} of {len(cands[cat])} candidates", file=sys.stderr)
    if version == 1:
        # v1 records predate the printed_lang field
        for recs in pool.values():
            for r in recs:
                r.pop("printed_lang", None)
    path.write_text(json.dumps(pool, ensure_ascii=False, indent=0), encoding="utf-8")
    return pool
