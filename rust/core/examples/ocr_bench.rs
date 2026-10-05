//! OCR bake-off helper. Crops every slot of the given photos with the real
//! pipeline, runs one recognition model over them and scores the readings
//! against the synthgen manifest.
//!
//! ```text
//! cargo run --release -p fscan-core --example ocr_bench -- \
//!     --manifest ../testdata/smoke/manifest.json --model models/en_PP-OCRv5_rec_mobile.onnx \
//!     ../testdata/smoke/c4_0001.jpg ...
//! ```
//! Env: `OCR_HEIGHT` (model input height), `OCR_BATCH` (batch size),
//! `TEXT_MM` (crop height, 0 = whole band), `CLAHE` (clip limit).

use std::path::{Path, PathBuf};
use std::time::Instant;

use fscan_core::Scanner;
use fscan_core::matching::{MatchStatus, NameIndex, normalize};
use fscan_core::ocr::RtenRecognizer;
use serde_json::Value;

fn env<T: std::str::FromStr>(key: &str) -> Option<T> {
    std::env::var(key).ok()?.parse().ok()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut manifest = None;
    let mut model = None;
    let mut images = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--manifest" => manifest = args.next().map(PathBuf::from),
            "--model" => model = args.next().map(PathBuf::from),
            _ => images.push(PathBuf::from(a)),
        }
    }
    let manifest: Value = serde_json::from_str(&std::fs::read_to_string(manifest.ok_or("--manifest")?)?)?;
    let model = model.ok_or("--model")?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let index = NameIndex::load(&root.join("testdata/names/names_v1.json"))?;
    let mut rec = RtenRecognizer::load(&model)?;
    if let Some(h) = env("OCR_HEIGHT") {
        rec.input_height = h;
    }
    if let Some(x) = env("OCR_XSCALE") {
        rec.width_scale = x;
    }
    if let Some(b) = env("OCR_BATCH") {
        rec.batch_size = b;
    }
    let mut scanner = Scanner::new(index, Box::new(rec));
    if let Some(t) = env::<f64>("TEXT_MM") {
        scanner.params.text_height_mm = (t > 0.0).then_some(t);
    }
    scanner.params.clahe_clip = env("CLAHE");
    if let Some(t) = env::<u8>("TRIM") {
        scanner.params.trim_right = t != 0;
    }

    let (mut n, mut exact, mut auto_ok, mut wrong, mut ocr_ms) = (0, 0, 0, 0, 0u128);
    for path in &images {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let key = manifest["images"]
            .as_object()
            .unwrap()
            .keys()
            .find(|k| k.ends_with(&name))
            .ok_or("image not in manifest")?
            .clone();
        let truth = &manifest["images"][&key];
        let photo = image::open(path)?.to_luma8();
        let layout = scanner.locate(&photo);
        let crops = scanner.crops(&photo, &layout);
        let t = Instant::now();
        let lines = scanner.recognizer.recognize(&crops)?;
        ocr_ms += t.elapsed().as_millis();
        let slots = layout.columns.iter().flat_map(|c| (1..=c.n_cards).map(move |i| (c.geometry.column, i)));
        for ((j, i), line) in slots.zip(&lines) {
            let col = truth["columns"].as_array().unwrap().iter().find(|c| c["column"] == j).unwrap();
            let Some(ts) = col["slots"].as_array().unwrap().iter().find(|s| s["slot"] == i) else { continue };
            if ts["lang"] == "other" {
                continue;
            }
            n += 1;
            let want = ts["name"].as_str().unwrap();
            if normalize(&line.text) == normalize(want) {
                exact += 1;
            }
            let m = scanner.index.match_raw(&line.text);
            if m.status == MatchStatus::Auto {
                if m.best().unwrap().oracle_id == ts["oracle_id"].as_str().unwrap() {
                    auto_ok += 1;
                } else {
                    wrong += 1;
                    println!("WRONG {name} c{j} s{i}: {:?} -> {} (truth {want})", line.text, m.best().unwrap().name);
                }
            } else if std::env::var("SHOW_MISSES").is_ok() {
                println!("miss  {name} c{j} s{i}: {:?} (truth {want})", line.text);
            }
        }
    }
    println!(
        "{} xscale={:?} text_mm={:?}: h={} lines {n}  exact {:.1}%  auto-ok {:.1}%  wrong {wrong}  ocr {:.0} ms/line",
        model.file_name().unwrap().to_string_lossy(),
        env::<f64>("OCR_XSCALE"),
        scanner.params.text_height_mm,
        env::<u32>("OCR_HEIGHT").map_or("default".into(), |h| h.to_string()),
        100.0 * exact as f64 / n as f64,
        100.0 * auto_ok as f64 / n as f64,
        ocr_ms as f64 / n as f64
    );
    Ok(())
}
