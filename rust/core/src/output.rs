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

/// Wall-clock milliseconds per pipeline stage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Timing {
    pub total: u64,
    pub detect: u64,
    pub crop: u64,
    pub ocr: u64,
    #[serde(rename = "match")]
    pub match_: u64,
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
            timing_ms: Timing { total: 12, ..Default::default() },
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
        assert_eq!(v["timing_ms"]["total"], 12);
    }
}
