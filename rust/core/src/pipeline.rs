//! SPEC §2 end to end: markers -> columns -> homographies -> stop card ->
//! slot crops -> OCR -> matching -> [`PhotoResult`].

use std::time::Instant;

use image::GrayImage;

use crate::aruco::{self, DetectorParams, Marker};
use crate::geometry::{self, CARDS_PER_COLUMN, PX_PER_MM, Point, STOP_CARD_ID, STRIP_W, Side};
use crate::homography::Homography;
use crate::image_ops::{clahe, invert, warp_rect};
use crate::matching::{MatchResult, MatchStatus, NameIndex};
use crate::ocr::{OcrLine, Recognizer};
use crate::output::*;
use crate::Error;

/// Knobs of the pipeline (everything that is not fixed by the spec).
#[derive(Debug, Clone)]
pub struct ScanParams {
    pub detector: DetectorParams,
    /// Extra mm above and below each slot band (spec allows up to 1.5).
    pub slot_pad_mm: f64,
    /// CLAHE tile grid and clip limit applied to each crop.
    pub clahe_tiles: (u32, u32),
    pub clahe_clip: f64,
    /// Also OCR the inverted crop (white-on-dark name bars).
    pub try_inverted: bool,
}

impl Default for ScanParams {
    fn default() -> Self {
        ScanParams {
            detector: DetectorParams::default(),
            slot_pad_mm: 1.5,
            clahe_tiles: (8, 2),
            clahe_clip: 2.0,
            try_inverted: true,
        }
    }
}

/// Owns everything needed to scan photos. Build once, scan many.
pub struct Scanner {
    pub index: NameIndex,
    /// `Box<dyn Trait>`: any type implementing `Recognizer`, chosen at runtime.
    pub recognizer: Box<dyn Recognizer>,
    pub params: ScanParams,
}

/// A column whose two header markers were found.
struct ColumnGeometry {
    column: usize,
    /// Strip millimetres -> photo pixels.
    mm_to_img: Homography,
    /// Photo pixels -> strip millimetres.
    img_to_mm: Homography,
}

impl Scanner {
    pub fn new(index: NameIndex, recognizer: Box<dyn Recognizer>) -> Self {
        Scanner { index, recognizer, params: ScanParams::default() }
    }

    /// Scan one photo. `image_name` is copied into the result.
    pub fn scan(&self, photo: &GrayImage, image_name: &str) -> Result<PhotoResult, Error> {
        let t0 = Instant::now();
        let mut timing = Timing::default();

        let markers = aruco::detect(photo, &self.params.detector);
        timing.detect = t0.elapsed().as_millis() as u64;

        let config = if markers.iter().any(|m| (8..=11).contains(&m.id)) { Config::C6 } else { Config::C4 };
        let mut status = PhotoStatus::Ok;
        let mut columns = Vec::new();
        let mut geometries = Vec::new();
        for j in 1..=config.columns() {
            let left = find_marker(&markers, geometry::marker_id(j, Side::Left));
            let right = find_marker(&markers, geometry::marker_id(j, Side::Right));
            match (left, right) {
                (Some(l), Some(r)) => match column_geometry(j, l, r) {
                    Some(g) => geometries.push(g),
                    None => {
                        status = PhotoStatus::Retake;
                        columns.push(error_column(j));
                    }
                },
                // One or both ids missing: the column can't be read.
                _ => {
                    status = PhotoStatus::Retake;
                    columns.push(error_column(j));
                }
            }
        }

        // Crop every slot of every readable column (normal + inverted).
        let t_crop = Instant::now();
        let stops: Vec<&Marker> = markers.iter().filter(|m| m.id == STOP_CARD_ID).collect();
        let mut plans = Vec::new(); // (column, stop_card, n_cards, first crop index)
        let mut crops: Vec<GrayImage> = Vec::new();
        for g in &geometries {
            let stop_slot = stops.iter().find_map(|m| stop_slot_in_column(g, m));
            let n_cards = match stop_slot {
                Some(k) => (k - 1).clamp(0, CARDS_PER_COLUMN as i64) as usize,
                None => CARDS_PER_COLUMN,
            };
            plans.push((g.column, stop_slot.is_some(), n_cards, crops.len()));
            for i in 1..=n_cards {
                let rect = geometry::slot_box_mm(i, self.params.slot_pad_mm);
                let raw = warp_rect(photo, &g.mm_to_img, rect, PX_PER_MM);
                let (tx, ty) = self.params.clahe_tiles;
                let enhanced = clahe(&raw, tx, ty, self.params.clahe_clip);
                if self.params.try_inverted {
                    crops.push(invert(&enhanced));
                }
                crops.push(enhanced);
            }
        }
        timing.crop = t_crop.elapsed().as_millis() as u64;

        let t_ocr = Instant::now();
        let lines = self.recognizer.recognize(&crops)?;
        timing.ocr = t_ocr.elapsed().as_millis() as u64;

        let t_match = Instant::now();
        let per_slot = if self.params.try_inverted { 2 } else { 1 };
        for (column, stop_card, n_cards, first) in plans {
            let slots: Vec<SlotResult> = (0..n_cards)
                .map(|i| {
                    let reads = &lines[first + i * per_slot..first + (i + 1) * per_slot];
                    self.best_read(i + 1, reads)
                })
                .collect();
            let all_auto = slots.iter().all(|s| s.status == SlotStatus::Auto);
            let full = stop_card || n_cards == CARDS_PER_COLUMN;
            columns.push(ColumnResult {
                column,
                status: if all_auto && full { ColumnStatus::Ok } else { ColumnStatus::Error },
                stop_card,
                n_cards,
                slots,
            });
        }
        columns.sort_by_key(|c| c.column);
        timing.match_ = t_match.elapsed().as_millis() as u64;
        timing.total = t0.elapsed().as_millis() as u64;

        Ok(PhotoResult {
            spec_version: SPEC_VERSION,
            image: image_name.to_string(),
            config,
            status,
            columns,
            timing_ms: timing,
        })
    }

    /// Match every OCR reading of one slot and keep the best-scoring one
    /// (ties go to the higher OCR confidence).
    fn best_read(&self, slot: usize, reads: &[OcrLine]) -> SlotResult {
        let scored: Vec<(MatchResult, &OcrLine)> =
            reads.iter().map(|r| (self.index.match_raw(&r.text), r)).collect();
        let (m, read) = scored
            .into_iter()
            .max_by(|(ma, ra), (mb, rb)| {
                ma.best_score().total_cmp(&mb.best_score()).then(ra.confidence.total_cmp(&rb.confidence))
            })
            .expect("at least one reading per slot");
        slot_result(slot, &read.text, &m)
    }
}

fn find_marker(markers: &[Marker], id: u16) -> Option<&Marker> {
    // If a marker id shows up twice (shouldn't happen), trust the largest.
    markers.iter().filter(|m| m.id == id).max_by(|a, b| a.side().total_cmp(&b.side()))
}

fn column_geometry(column: usize, left: &Marker, right: &Marker) -> Option<ColumnGeometry> {
    let mm: Vec<Point> = geometry::header_corners_mm(Side::Left)
        .into_iter()
        .chain(geometry::header_corners_mm(Side::Right))
        .collect();
    let px: Vec<Point> = left.corners.iter().chain(&right.corners).copied().collect();
    let mm_to_img = Homography::from_points(&mm, &px)?;
    let img_to_mm = mm_to_img.inverse()?;
    Some(ColumnGeometry { column, mm_to_img, img_to_mm })
}

/// SPEC §2.4: the stop card's slot if marker `m` lies in this column.
fn stop_slot_in_column(g: &ColumnGeometry, m: &Marker) -> Option<i64> {
    let c = g.img_to_mm.apply(m.center());
    if !(0.0..STRIP_W).contains(&c.x) || c.y < geometry::HEADER_Y {
        return None;
    }
    // Top edge = mean y of the marker's top-left and top-right corners.
    let tl = g.img_to_mm.apply(m.corners[0]);
    let tr = g.img_to_mm.apply(m.corners[1]);
    Some(geometry::stop_card_slot((tl.y + tr.y) / 2.0))
}

fn error_column(column: usize) -> ColumnResult {
    ColumnResult { column, status: ColumnStatus::Error, stop_card: false, n_cards: 0, slots: Vec::new() }
}

fn slot_result(slot: usize, raw_text: &str, m: &MatchResult) -> SlotResult {
    let status = match m.status {
        MatchStatus::Auto => SlotStatus::Auto,
        MatchStatus::Review => SlotStatus::Review,
        MatchStatus::Empty => SlotStatus::Empty,
    };
    // `filter` keeps the best candidate only when the slot is auto.
    let best = m.best().filter(|_| status == SlotStatus::Auto);
    SlotResult {
        slot,
        status,
        raw_text: raw_text.to_string(),
        name: best.map(|c| c.name.clone()),
        oracle_id: best.map(|c| c.oracle_id.clone()),
        lang: best.map(|c| c.lang.clone()),
        score: m.best_score(),
        candidates: m
            .candidates
            .iter()
            .map(|c| CandidateOut { name: c.name.clone(), oracle_id: c.oracle_id.clone(), score: c.score })
            .collect(),
    }
}
