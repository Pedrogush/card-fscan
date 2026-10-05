# card-fscan shared spec (v1)

This is the contract between the two implementations (`kotlin/`, `rust/`) and the
shared test tooling (`tools/`, `testdata/`). If an implementation and this file
disagree, the implementation is wrong. Changes here need a version bump and must
be announced to both tracks.

The design rationale lives in the handoff doc in the repo root. This file only
pins the numbers and formats.

## 1. Physical standard (from print pack v1, measured from the PDF vectors)

All units are millimetres. Strip coordinates: origin at the strip's top-left
corner, x to the right, y down the column (towards card 20).

| Constant | Value |
| --- | --- |
| Strip size | 70 x 240 |
| Column pitch (strips butted edge to edge) | 70 |
| Header height (card 1 top edge line) | y = 25 |
| Slot pitch | 10 |
| Cards per full column | 20 |
| Card size | 63 x 88 (card left edge nominally at x = 3.5) |
| ArUco dictionary | DICT_4X4_50 (OpenCV), 4x4 data bits + 1-cell black border = 6x6 cells |
| Header markers | 18 x 18 (3 mm cells). Left marker square: x 6..24, y 3.5..21.5. Right marker: x 46..64, y 3.5..21.5 |
| Column j (1-based) marker ids | left = 2(j-1), right = 2(j-1)+1 |
| Stop card | card-sized (63 x 88), ArUco id 40, 30 x 30 (5 mm cells), horizontally centred (x 16.5..46.5 card coords), top at 3 mm below the card's top edge |

Marker corner order follows OpenCV: top-left, top-right, bottom-right,
bottom-left, as seen with the strip upright (header at the top). So the 8 header
corners in strip coordinates are:

```
left  (id 2(j-1)):   (6,3.5) (24,3.5) (24,21.5) (6,21.5)
right (id 2(j-1)+1): (46,3.5) (64,3.5) (64,21.5) (46,21.5)
```

Stacking: card 1's top edge on y = 25. Card i's top edge on y = 25 + 10(i-1).
Each card lies ON TOP of the previous one, so only the top 10 mm of cards
1..19 (the name bar) and all of card 20 are visible. Volunteers place cards
within about +/- 2 mm of the tick line.

Configs: C4 = columns 1..4 (ids 0..7), C6 = columns 1..6 (ids 0..11).

## 2. Pipeline contract

1. Detect DICT_4X4_50 markers in the whole photo.
2. Column j is *present* when both of its ids are found. One id found = column
   status `error` (retake). Config = C6 if any id 8..11 is present, else C4.
   Every column of the config must be present, otherwise photo status `retake`.
3. Homography from the 8 header corners to strip mm. Warp each column to a
   canonical image at **12 px/mm**: 840 x 2880 px covering strip 0..70 x 0..240.
   Cards 18-20 bodies run past y = 240; implementations may warp a taller
   canvas (up to y = 340) to look for the stop card.
4. Stop card: if marker id 40 lies inside column j's x-range (projected through
   that column's homography), and its top edge is at y_top, then the stop card
   occupies slot k = round((y_top - 3 - 25) / 10) + 1 and the column holds
   k - 1 cards. A column with no stop card must hold 20.
5. Slot i crop (mm): x 4.5..65.5, y 25 + 10(i-1) + 0.5 .. 25 + 10i - 0.5.
   Implementations may widen y by up to 1.5 mm on each side to absorb placement
   jitter, but must report per slot i.
6. OCR the crop as a single text line. Try the inverted crop too (white-on-dark
   name bars) and keep the better-scoring result.
7. Clean and match (section 3).

## 3. Name normalisation and matching

The name index is `testdata/names/names_v1.json` (built by
`tools/build_name_index.py` from MTGJSON AtomicCards, MIT-licensed data):

```json
{ "version": 1, "source": "MTGJSON AtomicCards <date>",
  "entries": [ {"key": "lightning bolt", "name": "Lightning Bolt", "oracle_id": "…", "lang": "en"},
               {"key": "relampago", "name": "Relâmpago", "oracle_id": "…", "lang": "pt"} ] }
```

For split / adventure / double-faced cards, every face name AND the full
`A // B` name get their own entries with the same `oracle_id`.

`normalize(s)`:
1. Unicode NFKD, drop combining marks (so `ã` -> `a`, `ç` -> `c`).
2. Lowercase.
3. Map every char that is not `a-z`, `0-9` or space to a space (this includes
   `'`, `,`, `-`, `/`).
4. Collapse runs of spaces, trim.

Cleaning OCR text before normalising: strip trailing tokens that contain no
letters (mana cost residue), and strip any trailing run of chars from the set
`{}()[]0123456789@#*%&+=|\/` and whitespace.

Score: `ratio(a, b) = 100 * (1 - indel(a, b) / (len(a) + len(b)))`, where
`indel` is the insert/delete-only edit distance (this equals
`rapidfuzz.fuzz.ratio`). Compare `normalize(cleaned_ocr)` against every `key`.
When several entries share an `oracle_id`, keep that oracle_id's best score only.

Auto-accept when: best >= 90 AND best - runner_up >= 5 (runner-up = best score
of a *different* oracle_id), AND if `len(best.key) <= 6` then best >= 95.
Otherwise the slot is `review`. A crop with no OCR text (or only non-letters)
is `empty`.

## 4. Output JSON (one per photo)

```json
{
  "spec_version": 1,
  "image": "c4_0001.jpg",
  "config": "C4",
  "status": "ok",                       // ok | retake
  "columns": [
    {
      "column": 1,
      "status": "ok",                   // ok | error  (error = needs retake / review)
      "stop_card": false,
      "n_cards": 20,
      "slots": [
        { "slot": 1, "status": "auto",  // auto | review | empty
          "raw_text": "Lightning Bolt {R}", "name": "Lightning Bolt",
          "oracle_id": "…", "lang": "en", "score": 100.0,
          "candidates": [ {"name": "…", "oracle_id": "…", "score": 0.0} ] }
      ]
    }
  ],
  "timing_ms": { "total": 0 }
}
```

`name`, `oracle_id`, `lang` are null unless status is `auto`. `candidates` holds
the top 3 distinct oracle_ids (for the review UI). A column is `ok` only when it
has a full 20 or a stop card AND every slot is `auto`.

## 5. Golden archive (the test oracle)

`testdata/<set>/manifest.json` maps image path -> expected output:

```json
{ "spec_version": 1, "set": "smoke", "generator": "tools/synthgen v1",
  "images": {
    "smoke/c4_0001.jpg": {
      "seed": 1, "config": "C4", "expect_status": "ok",
      "columns": [ {"column": 1, "stop_card": false, "n_cards": 20,
                    "slots": [ {"slot": 1, "oracle_id": "…", "name": "…", "lang": "en"} ] } ],
      "difficulty": {"tilt_deg": 2.1, "blur_px": 0.8, "jitter_mm": 1.5, "...": "..."}
    } } }
```

Images themselves are NOT committed (they contain Wizards of the Coast card art
from Scryfall). They are regenerated deterministically from the manifest seeds
with `tools/synthgen` and cached in `testdata/<set>/` (gitignored).

Expected outputs only contain ground truth; scoring is done by `tools/eval.py`:

- **wrong**: slot `auto` with an oracle_id different from the truth. Target 0.
- **auto rate**: auto & correct / all true slots. Target >= 95%.
- **column count**: n_cards and stop_card must match exactly.
- **status**: `retake` images must be reported as retake.

Each implementation provides a desktop CLI
`scan --out <dir> <image>...` that writes `<dir>/<image basename>.json`, and
`tools/eval.py --manifest testdata/smoke/manifest.json --results <dir>` scores it.
