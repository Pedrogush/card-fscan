# Shingle Scanner — Bulk MTG Inventory for the Casa Durval Paiva Drive

Sep 26, 2026 · @Pedro Gushiken

## Summary

Volunteers fan donated bulk cards into printed columns of 20, with each card 10 mm below the last, so only the name bars show. A phone on a fixed-height 3D-printed stand photographs 80 or 120 cards per shot, depending on the phone. Software reads each name, logs it to a box, and later packs the pool into kid-friendly 20-card half-decks.

- **Scope:** card identity by name only. Printing, foil, condition and price are out of scope.
- **Volume:** thousands of donated bulk cards, many in Portuguese (Natal, RN). Matching must cover EN and PT names.
- **Priorities:** speed and reliability over accuracy on rare edge cases. Anything uncertain goes to a human review queue; nothing is silently guessed.
- **Core idea:** every column has printed fiducial markers and a fixed 10 mm slot grid. The software therefore knows where each name bar *should* be. Counting becomes geometry, and OCR only has to read one line at a time.

No open-source project does shingled name-bar scanning as of September 2026. Recognition pieces can be borrowed from existing repos (see Reference repos).

## Physical standard

Every column is a printed 70 x 240 mm strip holding exactly 20 cards at a 10 mm offset. Butting strips edge to edge sets the column pitch, so there are no layout measurements to get wrong. Print templates are in the companion PDF (A4, 100% scale).

| Item | Value | Why |
| --- | --- | --- |
| Card size | 63 x 88 mm (unsleeved) | Scan bulk unsleeved; sleeves add glare and 2-3 mm |
| Offset between cards | 10 mm | Exposes the full name bar on old and modern frames |
| Cards per column | 20 | Easy to count, and equals one half-deck packet |
| Strip width = column pitch | 70 mm | 3.5 mm margin on each side of the card |
| Strip header | 25 mm, two 18 mm ArUco markers | Markers give the column position, rotation and scale |
| Name zone | 204 mm from the first card's top edge | Slot i occupies 10(i-1) to 10i mm |
| Strip length | 240 mm | Card 20's body overhangs the strip end by about 63 mm |
| Marker dictionary | ArUco DICT\_4X4\_50 | Column j uses IDs 2(j-1) (left) and 2(j-1)+1 (right) |
| Stop card | Card-sized, ArUco ID 40 | Placed after the last card of a partial column |

**Stacking order.** Card 1 goes at the top line. Each next card goes 10 mm lower and *on top* of the previous one. Only card 20 is fully visible.

**Column trays (recommended).** Glue each strip to a 70 x 310 mm rigid board (5 mm foam board or 3 mm MDF). Volunteers fill trays at other tables, and the camera operator only swaps trays in and out. Each tray localises itself from its own markers, so trays don't need to sit perfectly.

**Partial columns.** A column may hold fewer than 20 cards only at the end of a box. It must then end with a stop card. The software treats a column with neither 20 read names nor a stop card as an error, never as "probably 17".

## Camera geometry

There are two standard configurations: **C4** (4 columns, 80 cards per photo) for 12 MP phones, and **C6** (6 columns, 120 cards per photo) for phones that save 24 MP or more. Both keep the name text at about 11-12 px/mm or better. At that density a 2 mm tall name is about 24 px, comfortable for OCR; the hard floor is 10 px/mm.

| Config | Min. saved resolution | Frame on the mat (4:3) | px/mm at min. MP | Cards per photo |
| --- | --- | --- | --- | --- |
| C4 | 12 MP (4032 x 3024) | 360 x 270 mm | 11.2 | 80 |
| C6 | 24 MP (5712 x 4284) | 480 x 360 mm | 11.9 | 120 |

The long side of the frame runs *across* the columns. The strips (4 x 70 = 280 mm or 6 x 70 = 420 mm across, 240 mm along) leave 15-40 mm of margin to absorb stand misalignment. A 12 MP phone cannot run C6: it would get only 8.4 px/mm.

Megapixels set the resolution, but the lens sets the height. Height depends on the main camera's 35 mm-equivalent focal length f, where W is the frame's long side and 34.6 mm is the long side of a 4:3 frame in 35 mm-equivalent terms:

```latex
h = \frac{W \cdot f}{34.6\,\text{mm}}
```

| f (35 mm equiv.) | C4 height | C6 height |
| --- | --- | --- |
| 23 mm | 240 mm | 320 mm |
| 24 mm | 250 mm | 330 mm |
| 25 mm | 260 mm | 350 mm |
| 26 mm | 270 mm | 360 mm |
| 27 mm | 280 mm | 370 mm |
| 28 mm | 290 mm | 390 mm |

Heights are measured from the mat to the phone's back surface, rounded to the stand's 10 mm pin holes. The phone's EXIF tag FocalLengthIn35mmFilm gives a starting guess. The calibration photo (see Stand) is the final authority, because vendors round these figures.

**Phone settings (reliability checklist)**

- Main 1x lens only. Turn off automatic macro or lens switching (iPhone: Settings > Camera > Macro Control).
- 4:3 aspect ratio, highest *native* resolution. A 48/50 MP phone that saves 12 MP by default is a 12 MP phone for this purpose; switch it to 24 MP or full resolution for C6.
- Lock focus and exposure (long-press on the mat). Flash off.
- Trigger with a Bluetooth remote, earbuds volume button or 2 s timer, so the stand doesn't shake.
- Screen faces up; the phone lies on its back on the plate with the main lens over the crosshair.

**Lighting.** Use two diffuse lamps at about 45 degrees on opposite sides of the column axis, not overhead, to avoid glare on glossy cards. Run the whole session under the same light.

## Stand

The stand is a 30 mm square post with pin holes every 10 mm, plus a carriage whose arm holds the phone plate 160 mm out over the centre of the columns. A 5 mm steel pin fixes the height, so it can't drift between shots. The parametric OpenSCAD model (`shingle_stand.scad`) is included; it hasn't been test-printed yet.

| Part | Qty | Size | Notes |
| --- | --- | --- | --- |
| Baseboard | 1 | 600 x 450 mm plywood or MDF | Not printed; everything screws or glues to it |
| Foot | 1 | 200 x 100 x 8 mm + 30 mm socket boss | Front edge is the tray stop; 4 x M4/M5 wood screws |
| Post segments | 3 | 30 x 30 mm, 135 / 140 / 140 mm + 20 mm peg | Pin holes pass through the pegs too, so a pin also locks a joint |
| Carriage + arm + plate | 1 | 50 mm sleeve, 160 mm arm, 120 x 120 mm plate | 70 x 70 mm lens window with crosshair notches; rubber-band hooks |
| Pin | 1 | 5 mm steel dowel or M5 x 50 bolt | Holes are 5.4 mm |

**Heights.** Each hole is engraved with the lens height above the mat surface it produces, from 230 to 400 mm, assuming 5 mm trays (the `tray_thickness` parameter). Pick the hole from the height table in Camera geometry.

**Placement.** The foot's front edge sits on the strips' top line. Its centre notch lines up with the gap between the two middle columns (2|3 for C4, 3|4 for C6). This puts the lens window over the frame centre, 120 mm down the strips.

**Printing.** PLA or PETG, 0.2 mm layers, 4 walls, 25% infill. Print post segments upright and the carriage plate-side down. The tallest part is 160 mm.

**Calibration (once per phone, repeat if the stand is moved)**

1. Put all trays for the config in place, with cards in every slot.
2. Pin the carriage at the hole from the height table. Put the phone on the plate with its main lens on the crosshair.
3. Take one photo and run `shingle calibrate photo.jpg --config C4`.
4. The tool checks that all marker IDs are found and at least 5 mm inside the frame. It measures px/mm (target 11 or more), tilt (marker scale must vary by less than 3% across the frame) and sharpness per column.
5. It then answers *OK*, *raise one hole*, *lower one hole* or *re-centre phone*. Save the passing result as that phone's profile.

## Software pipeline

Each photo is split into columns by its markers, each column is warped into flat millimetre coordinates, and each of the 20 slots is cropped at a known position and read as one line of text. Nothing depends on text *detection*, which is where overlapping layouts usually fail.

**Stack.** Python 3.12, opencv-contrib-python (ArUco), RapidOCR or PaddleOCR for recognition, rapidfuzz, FastAPI + SQLite, and a small web frontend for upload and review. Photos come from the phone's native camera app (for locked focus and full resolution) and are uploaded in batches through a local web page.

1. **Ingest.** Store the original photo with session metadata: box ID, config (C4/C6), phone profile and operator. Reject exact duplicates by file hash.
2. **Find columns.** Detect DICT\_4X4\_50 markers. Column j is present when IDs 2(j-1) and 2(j-1)+1 are both found. A missing or half-found column means *retake*, never *skip*.
3. **Rectify.** Map the 8 marker corners to strip coordinates with a homography. Warp each column to a canonical 70 x 240 mm image at 12 px/mm (840 x 2880 px). Everything after this step works in millimetres.
4. **Column length.** If ArUco ID 40 (stop card) lies in the column, the last real slot is the one above it. Otherwise the column must hold 20.
5. **Crop slots.** Slot i is the band 25 + 10(i-1) to 25 + 10i mm, inset 0.5 mm, across the card width. Enhance contrast (CLAHE), and also try an inverted copy for white-on-dark name bars.
6. **Read.** Run the recognition-only OCR model on all crops as one batch (80-120 per photo). Keep the text and confidence.
7. **Clean.** Drop trailing mana-cost junk (digits, symbols, stray braces). Normalise case, punctuation and accents for matching, but keep the raw text for review.
8. **Match.** Look up the name in an index of English oracle names plus Portuguese printed names, each mapped to Scryfall `oracle_id`. Include both faces and `A // B` forms for split and double-faced cards. Auto-accept at score 90 or more with a lead of 5+ over the runner-up; names of 6 characters or fewer need 95.
9. **Validate.** A column passes only if every expected slot has an accepted match. A blank band or an unmatched read sends that slot to review, together with the columns' count check.
10. **Review.** A grid of flagged crops, each with its top 3 suggestions and a type-ahead, driven by the keyboard. Optionally, a vision LLM pre-fills a suggestion for flagged crops; a human still confirms it.

```python
# Slot band in rectified strip coordinates (mm). Scale by PX_PER_MM = 12.
HEADER, OFFSET, CARD_W, MARGIN = 25.0, 10.0, 63.0, 3.5

def slot_box(i: int) -> tuple[float, float, float, float]:
    """Return (x0, y0, x1, y1) in mm for slot i (1-based)."""
    y0 = HEADER + OFFSET * (i - 1) + 0.5
    return (MARGIN + 1.0, y0, MARGIN + CARD_W - 1.0, y0 + OFFSET - 1.0)
```

**Target numbers for the first pilot.** At least 95% of slots auto-accepted, and no more than 0.5% wrong among auto-accepted slots (checked by spot-auditing 200 slots). Processing should take a few seconds per photo on a laptop CPU; measure this in the pilot.

## Data model and packet solver

Every card is stored with a physical address: box, bundle (one scanned column, kept together with a divider) and slot. A pick list can then say "W-03, bundle 17, card 6" instead of "somewhere in box 3".

| Table | Key fields |
| --- | --- |
| `boxes` | id, label (e.g. W-03), colour pile, location |
| `sessions` | id, box\_id, config, phone\_profile, operator, started\_at |
| `photos` | id, session\_id, path, sha256, status |
| `bundles` | id, box\_id, photo\_id, column, seq\_in\_box, n\_cards |
| `slots` | id, bundle\_id, slot, crop\_path, raw\_text, ocr\_conf, oracle\_id, lang (en/pt), score, status (auto / reviewed / rejected / empty) |
| `cards` | oracle\_id, name\_en, name\_pt, mana\_value, colors, color\_identity, type\_line, oracle\_text, keywords, power, toughness (Scryfall bulk cache) |
| `packets` | id, colour, theme, score, status |
| `packet_cards` | packet\_id, slot\_id |

**Pre-sort by colour before scanning.** Sorting into W, U, B, R, G, colourless/artifact, multicolour and lands piles is fast by hand. It means every packet is picked from a single box. Basic lands are counted, not scanned.

**Packet template (Jumpstart-style, 20 cards, mono-colour).** Each packet has 8 basic lands and 12 spells. Of those, 7-8 are creatures, 2-3 are simple removal or combat tricks, and 1-2 are other spells. The target curve by mana value is 1-2 at MV 1, 3-4 at MV 2, 3 at MV 3, 2 at MV 4 and 1-2 at MV 5+. A packet takes cards of its own colour plus colourless artifacts.

**Kid-friendly filter (all thresholds tunable).**

- Keep cards whose rules text is under about 180 characters.
- Drop mechanics that need extra bookkeeping or other products: stickers, dungeons and initiative, day/night, energy, the ring tempts you, and "Un-" sets.
- Drop nonbasic lands and multicolour cards from packets; they can go to a separate pool.
- Keep an organiser-curated denylist of names and themes. The kids are patients at a paediatric cancer support centre, so the organisers may want to exclude, for example, disease- or death-heavy cards. That is their call, and the hook is cheap to add.
- Prefer Portuguese copies when two copies are otherwise equal.

**Allocation algorithm.**

1. Score each eligible card. Combine simplicity (text length, keyword count), efficiency (power + toughness versus MV, evasion keywords) and a removal flag (regex on oracle text: `destroy target creature`, `deals N damage to`, `-N/-N`, `exile target creature`).
2. Tag themes from keywords and creature subtypes (Flyers, Goblins, Elves, Zombies and so on). A theme needs about 6 matching spells to name a packet.
3. Per colour, open K = floor(eligible spells / 12) packets, capped by the basic lands available.
4. Snake-draft allocation: each packet in turn takes the best card that fills its largest template gap. This keeps packets balanced, so no kid gets a dud.
5. Local search: swap cards between packets of the same colour to reduce the spread of packet scores. Move to OR-Tools CP-SAT only if greedy allocation plus swaps is not good enough.
6. Output a printable pick list per box, sorted by bundle then slot, plus a card list per packet in Portuguese.

A draftable cube for older or returning kids can reuse the same pool and scoring later; it is out of scope for v1.

## Volunteer workflow and throughput

The line runs at the speed of the people filling trays, so add fillers, not cameras. With three fillers and one camera operator, a rough estimate is 1,500-2,500 cards per hour. That figure is a guess to be replaced by pilot measurements.

| Role | People | Job |
| --- | --- | --- |
| Sorter | 1 | Splits donations into colour piles and fills labelled boxes (W-01, U-01 and so on) |
| Filler | 2-4 | Fills trays: 20 cards per tray on the tick lines, or a stop card at the end of a box |
| Camera operator | 1 | Swaps a full set of trays under the stand, shoots, and files each column as a bundle behind a numbered divider |
| Reviewer | 1 | Clears the review queue on the laptop and flags retakes back to the camera operator |

**Per photo cycle (camera operator)**

1. Set the box in the app (one box per session).
2. Slide 4 or 6 full trays against the foot, centred on the notch.
3. Shoot with the remote. Wait for the app's green "all columns found" before moving anything.
4. Square up each column toward its header, keeping cards face up and in order. File it into the box behind the next numbered divider.
5. Return the empty trays to the fillers.

**Tray sets.** Use at least three sets (12 trays for C4, 18 for C6): one under the camera, one being filled, and one waiting.

**Estimate basis.** Filling one tray is about 45-60 s once practised, and a camera cycle is about 30-40 s. So three fillers produce roughly 3 trays (60 cards) a minute at best, before breaks and retakes.

## Milestones, tests and open questions

Build in this order. Each milestone must pass its gate before the next starts, and the camera geometry gets validated on paper before any CAD print.

- [ ] **M0: paper pilot.** Print the strips, tape them to cardboard, and hold the phone by hand at the table height. Gate: all markers are detected and px/mm is at least 11 on 10 photos.
- [ ] **M1: reader.** Marker detection, rectification, slot crops, OCR and EN+PT matching as a CLI on saved photos. Gate: at least 95% auto-accepted and 0 wrong on a hand-labelled set of 500 slots, including 100 Portuguese cards.
- [ ] **M2: stand.** Print the stand and add the `calibrate` command. Gate: two different phones calibrate to *OK* within 2 attempts.
- [ ] **M3: intake and review app.** FastAPI + SQLite, batch upload, review grid and bundle filing. Gate: one volunteer processes 1,000 cards end to end, and the time is measured.
- [ ] **M4: packet solver and pick lists.** Gate: the organisers approve 10 sample packets, and a volunteer pulls one packet from its pick list in under 3 minutes.

**Test set to collect early:** old frames (pre-2003), modern frames, showcase and borderless cards, split, adventure and double-faced cards, and Portuguese, English and other-language cards. Include dark and white name bars, bent cards, and a deliberately misplaced card.

**Open questions**

- Which phones will the volunteers actually use, and what do they save by default (12 MP or 24 MP)? This decides C4 versus C6.
- How many basic lands can the stores donate? This caps the number of packets at 8 lands each.
- Do the organisers want a content denylist, and who curates it?
- Is there a laptop at the drive site, and is there Wi-Fi for photo upload, or will photos be copied over USB?
- What happens to non-English, non-Portuguese cards: include them in packets, or keep them for trade?

**Reference repos**

- [wmjg-alt/mtg\_scanner](https://github.com/wmjg-alt/mtg_scanner): EasyOCR title reading with CLAHE and upscaling preprocessing, and a SQLite Scryfall cache.
- [MeIsGaggy/MTG-Card-Scanner-Sorter](https://github.com/MeIsGaggy/MTG-Card-Scanner-Sorter): RapidOCR with Tesseract fallback and a manual-review threshold.
- [starstuffharvestingstarlight/tcg-ocr-scanner](https://github.com/starstuffharvestingstarlight/tcg-ocr-scanner): name-only OCR identification without card-outline detection.
- [DredBaron/OpenMTG](https://github.com/DredBaron/OpenMTG): FastAPI + React self-hosted inventory, if a full collection UI is wanted later.
