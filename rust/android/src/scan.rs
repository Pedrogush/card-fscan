//! Platform-independent part of the "Run sample scan" button: given the raw
//! bytes of the assets and the photo, build the scanner, run it and format a
//! short human-readable report. Kept free of Android APIs so it can be unit
//! tested on the host.

use std::fmt::Write as _;
use std::time::Instant;

use fscan_core::{
    Scanner,
    matching::NameIndex,
    ocr::RtenRecognizer,
    output::{PhotoResult, SlotStatus},
};

/// Everything the scan needs, already loaded into memory.
pub struct ScanInputs {
    /// `names_v1.json.gz` (gzipped name index).
    pub names_gz: Vec<u8>,
    /// `rec.onnx` (text recognition model).
    pub model: Vec<u8>,
    /// `rec.dict.txt` (character dictionary for the model).
    pub dict: String,
    /// JPEG/PNG bytes of the photo and a display name for it.
    pub photo: Vec<u8>,
    pub photo_name: String,
}

/// Builds the scanner, runs it on the photo and returns a text report.
/// Every failure is returned as `Err(String)` so the UI can show it.
pub fn run(inputs: ScanInputs) -> Result<String, String> {
    let mut out = String::new();

    let t = Instant::now();
    let index = NameIndex::from_bytes(&inputs.names_gz, true)
        .map_err(|e| format!("loading names_v1.json.gz failed: {e}"))?;
    let _ = writeln!(out, "Name index: {} names ({} ms)", index.len(), t.elapsed().as_millis());

    let t = Instant::now();
    let rec = RtenRecognizer::from_bytes(inputs.model, Some(&inputs.dict))
        .map_err(|e| format!("loading OCR model failed: {e}"))?;
    let _ = writeln!(out, "OCR model loaded ({} ms)", t.elapsed().as_millis());
    let scanner = Scanner::new(index, Box::new(rec));

    let t = Instant::now();
    let gray = image::load_from_memory(&inputs.photo)
        .map_err(|e| format!("decoding photo {} failed: {e}", inputs.photo_name))?
        .to_luma8();
    let _ = writeln!(
        out,
        "Photo: {} ({}x{}, decoded in {} ms)",
        inputs.photo_name,
        gray.width(),
        gray.height(),
        t.elapsed().as_millis()
    );

    let result = scanner
        .scan(&gray, &inputs.photo_name)
        .map_err(|e| format!("{out}\nscan failed: {e}"))?;
    out.push('\n');
    out.push_str(&summarize(&result));
    Ok(out)
}

/// Short summary of a [`PhotoResult`]: config, status, counts, timing.
pub fn summarize(r: &PhotoResult) -> String {
    let slots: Vec<_> = r.columns.iter().flat_map(|c| &c.slots).collect();
    let auto = slots.iter().filter(|s| s.status == SlotStatus::Auto).count();
    let review = slots.iter().filter(|s| s.status == SlotStatus::Review).count();
    let empty = slots.iter().filter(|s| s.status == SlotStatus::Empty).count();
    let mut s = String::new();
    let _ = writeln!(s, "Config: {:?}", r.config);
    let _ = writeln!(s, "Photo status: {:?}", r.status);
    let _ = writeln!(s, "Columns: {}", r.columns.len());
    let _ = writeln!(s, "Slots: {}", slots.len());
    let _ = writeln!(s, "Auto-accepted: {auto}  (review: {review}, empty: {empty})");
    let _ = writeln!(
        s,
        "Timing: total {} ms (detect {}, crop {}, ocr {}, match {})",
        r.timing_ms.total, r.timing_ms.detect, r.timing_ms.crop, r.timing_ms.ocr, r.timing_ms.match_
    );
    for c in &r.columns {
        let _ = writeln!(
            s,
            "  column {}: {:?}, {} cards{}",
            c.column,
            c.status,
            c.n_cards,
            if c.stop_card { ", stop card" } else { "" }
        );
    }
    s
}
