//! The per-photo result, serialised to the JSON of SPEC §4 with `serde`.
//!
//! `#[derive(Serialize)]` generates the JSON writer from the struct
//! definitions; field names map 1:1 to JSON keys, enums are written in
//! lowercase via `rename_all`.

use serde::{Deserialize, Serialize};

pub const SPEC_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Config {
    C4,
    C6,
}

impl Config {
    pub fn columns(self) -> usize {
        match self {
            Config::C4 => 4,
            Config::C6 => 6,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PhotoStatus {
    Ok,
    Retake,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ColumnStatus {
    Ok,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SlotStatus {
    Auto,
    Review,
    Empty,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotoResult {
    pub spec_version: u32,
    pub image: String,
    pub config: Config,
    pub status: PhotoStatus,
    pub columns: Vec<ColumnResult>,
    pub timing_ms: Timing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnResult {
    pub column: usize,
    pub status: ColumnStatus,
    pub stop_card: bool,
    pub n_cards: usize,
    pub slots: Vec<SlotResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotResult {
    pub slot: usize,
    pub status: SlotStatus,
    pub raw_text: String,
    /// `Option<T>` serialises as `null` when `None`.
    pub name: Option<String>,
    pub oracle_id: Option<String>,
    pub lang: Option<String>,
    pub score: f64,
    pub candidates: Vec<CandidateOut>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateOut {
    pub name: String,
    pub oracle_id: String,
    pub score: f64,
}

/// Milliseconds per pipeline step (SPEC §4 requires `total`; the rest are
/// extra keys). Top-level fields are wall-clock; `stages` is summed thread
/// time of the fine-grained stages (see `profile.rs`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Timing {
    /// End to end: decode + grayscale + scan + JSON (the CLI fills decode,
    /// gray and json; `Scanner::scan` alone sets total = scan).
    pub total: f64,
    pub decode: f64,
    pub gray: f64,
    /// Everything inside `Scanner::scan`.
    pub scan: f64,
    /// Marker detection + layout.
    pub detect: f64,
    /// Primary slot crops (warp, line finding, trim, enhance).
    pub crop: f64,
    /// All OCR calls (both passes).
    pub ocr: f64,
    /// All matching (both passes).
    #[serde(rename = "match")]
    pub match_: f64,
    /// Fallback pass total (crops + OCR + match), included in the above.
    pub retry: f64,
    pub json: f64,
    pub stages: StageTimes,
    pub counters: Counters,
}

/// Summed thread time per fine-grained stage, ms.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StageTimes {
    pub detect: f64,
    pub layout: f64,
    pub warp: f64,
    pub line_find: f64,
    pub trim: f64,
    pub enhance: f64,
    pub ocr_prep: f64,
    pub ocr_infer: f64,
    pub ctc: f64,
    pub clean: f64,
    #[serde(rename = "match")]
    pub match_: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Counters {
    pub slots: u64,
    pub ocr_calls: u64,
    pub ocr_batches: u64,
    pub ocr_lines: u64,
    pub crops_per_slot: f64,
    pub retry_slots: u64,
    pub retry_crops: u64,
    /// Sum over model runs of batch_size * padded input width (pixels).
    pub ocr_input_px_wide: u64,
    /// Lines read by the fast first-pass model (part of `ocr_lines`).
    pub fast_lines: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_shape() {
        let r = PhotoResult {
            spec_version: SPEC_VERSION,
            image: "c4_0001.jpg".into(),
            config: Config::C4,
            status: PhotoStatus::Ok,
            columns: vec![ColumnResult {
                column: 1,
                status: ColumnStatus::Error,
                stop_card: true,
                n_cards: 1,
                slots: vec![SlotResult {
                    slot: 1,
                    status: SlotStatus::Review,
                    raw_text: "Lightnimg B".into(),
                    name: None,
                    oracle_id: None,
                    lang: None,
                    score: 80.0,
                    candidates: vec![CandidateOut { name: "Lightning Bolt".into(), oracle_id: "x".into(), score: 80.0 }],
                }],
            }],
            timing_ms: Timing { total: 12.0, ..Default::default() },
        };
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["spec_version"], 1);
        assert_eq!(v["config"], "C4");
        assert_eq!(v["status"], "ok");
        let col = &v["columns"][0];
        assert_eq!(col["status"], "error");
        assert_eq!(col["stop_card"], true);
        let slot = &col["slots"][0];
        assert_eq!(slot["status"], "review");
        assert!(slot["name"].is_null() && slot["oracle_id"].is_null() && slot["lang"].is_null());
        assert_eq!(slot["candidates"][0]["name"], "Lightning Bolt");
        assert_eq!(v["timing_ms"]["total"], 12.0);
        assert!(v["timing_ms"]["stages"]["ocr_infer"].is_number());
        assert!(v["timing_ms"]["counters"]["ocr_lines"].is_number());
    }
}
