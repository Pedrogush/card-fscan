# Third-party code, models and data (Rust track)

card-fscan is MIT-licensed. Everything below is compatible with that and with
free distribution of the APK. No GPL/AGPL code is linked.

## Models (`rust/models/`)

| File | Source | License |
| --- | --- | --- |
| `en_PP-OCRv4_rec_mobile.onnx` (+ `.dict.txt`) | PaddleOCR PP-OCRv4 English recognition model, ONNX export from RapidOCR (`modelscope.cn/models/RapidAI/RapidOCR`, `onnx/PP-OCRv4/rec/`) | Apache-2.0 (PaddlePaddle/PaddleOCR, RapidAI/RapidOCR) |
| `en_PP-OCRv5_rec_mobile.onnx`, `latin_PP-OCRv5_rec_mobile.onnx` (bake-off only) | same, `onnx/PP-OCRv5/rec/` | Apache-2.0 |
| `text-recognition.rten` (bake-off only) | `ocrs` recognition model, `ocrs-models.s3-accelerate.amazonaws.com` | MIT/Apache-2.0 (robertknight/ocrs) |

The `.dict.txt` files are the character lists stored in each ONNX model's
`character` metadata entry, extracted one character per line.

## Data

- `testdata/names/names_v1.json` (bundled gzipped into the APK): built from
  MTGJSON AtomicCards, MIT. Card names are facts; no card images are shipped.

## Code

- ArUco `DICT_4X4_50` bit patterns (`core/src/aruco.rs`): from OpenCV,
  `modules/objdetect/src/aruco/predefined_dictionaries.hpp`, Apache-2.0
  (extracted with `cv2.aruco.generateImageMarker`). The detector itself is an
  original implementation following OpenCV's published approach.
- The `ocrs` alphabet string in `core/src/ocr.rs` is copied from the `ocrs`
  crate (MIT/Apache-2.0).

## Rust crates (direct dependencies)

| Crate | License |
| --- | --- |
| rten, rten-tensor (ONNX inference) | MIT OR Apache-2.0 |
| image | MIT OR Apache-2.0 |
| imageproc (contour tracing) | MIT |
| nalgebra | Apache-2.0 |
| serde, serde_json | MIT OR Apache-2.0 |
| unicode-normalization | MIT OR Apache-2.0 |
| flate2 (pure-Rust miniz_oxide backend) | MIT OR Apache-2.0 |
| rayon | MIT OR Apache-2.0 |
| thiserror, anyhow | MIT OR Apache-2.0 |

The full transitive list with licenses can be produced with
`cargo tree -e normal --prefix none --format "{p} {l}" | sort -u`
(see the README); at the time of writing every transitive crate is MIT,
Apache-2.0, BSD, Zlib, Unicode-3.0 or similar permissive licenses.
