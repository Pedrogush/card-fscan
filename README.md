# card-fscan — Shingle Scanner

Bulk Magic: The Gathering inventory by photographing shingled columns of cards
(only the name bars visible) on printed fiducial strips, then reading each name
bar with OCR and matching it against English and Portuguese card names.

Built for the Casa Durval Paiva donation drive (Natal, RN): donated bulk cards
are scanned, logged by physical address (box / bundle / slot), and packed into
kid-friendly 20-card half-decks.

## Contents

- `Shingle Scanner — Bulk MTG Inventory for the Casa Durval Paiva Drive.md` — design / handoff doc
- `shingle_scanner_print_pack.pdf` — printable strips and stand drawings (A4, 100% scale)
- `shingle_stand.scad` — parametric OpenSCAD camera stand (not yet test-printed)

## Status

Active implementation: [`rust/`](rust/) (pipeline, CLI and Android app; see
[`rust/README.md`](rust/README.md)). Shared contract: [`spec/SPEC.md`](spec/SPEC.md).
Synthetic test data and scoring: [`tools/`](tools/) and [`testdata/`](testdata/).

A parallel Kotlin implementation was built for comparison and is archived on the
[`archive/kotlin`](https://github.com/Pedrogush/card-fscan/tree/archive/kotlin)
branch (Phase 1: 94.4% auto-accept, 0 wrong on the dev set).

## License

MIT — see [LICENSE](LICENSE).

Magic: The Gathering is a trademark of Wizards of the Coast. This project is
unofficial fan content and is not affiliated with or endorsed by Wizards of the
Coast. Card data comes from [Scryfall](https://scryfall.com).
