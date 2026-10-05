//! SPEC §2 end to end: markers -> columns -> homographies -> stop card ->
//! slot crops -> OCR -> matching -> [`PhotoResult`].

use std::path::PathBuf;
use std::time::Instant;

use image::GrayImage;
use rayon::prelude::*;

use crate::Error;
use crate::aruco::{self, DetectorParams, Marker};
use crate::geometry::{self, CARDS_PER_COLUMN, PX_PER_MM, Point, STOP_CARD_ID, STRIP_W, Side};
use crate::homography::Homography;
use crate::image_ops::{clahe, invert, warp_rect};
use crate::matching::{MatchResult, MatchStatus, NameIndex};
use crate::ocr::{OcrLine, Recognizer};
use crate::output::*;

/// Knobs of the pipeline (everything that is not fixed by the spec).
#[derive(Debug, Clone)]
pub struct ScanParams {
    pub detector: DetectorParams,
    /// Extra mm above and below each slot band (spec allows up to 1.5).
    pub slot_pad_mm: f64,
    /// Height (mm) of the final crop around the located name text, or `None`
    /// to OCR the whole (padded) slot band.
    pub text_height_mm: Option<f64>,
    /// Cut the crop on the right after the name (before the mana cost) when a
    /// clear gap separates them. Narrower crops make OCR proportionally faster.
    pub trim_right: bool,
    /// CLAHE tile grid and clip limit; `clahe_clip: None` disables CLAHE.
    pub clahe_tiles: (u32, u32),
    pub clahe_clip: Option<f64>,
    /// Re-read slots that did not auto-accept from the inverted crop
    /// (white-on-dark name bars) and keep the better match.
    pub try_inverted: bool,
    /// When set, every slot crop is saved here as PNG (for debugging).
    pub dump_dir: Option<PathBuf>,
}

impl Default for ScanParams {
    fn default() -> Self {
        ScanParams {
            detector: DetectorParams::default(),
            slot_pad_mm: 1.5,
            text_height_mm: Some(6.4),
            trim_right: true,
            clahe_tiles: (8, 2),
            clahe_clip: None,
            try_inverted: true,
            dump_dir: None,
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

/// What [`Scanner::locate`] found in a photo.
pub struct Layout {
    pub config: Config,
    pub status: PhotoStatus,
    /// Readable columns, in column order.
    pub columns: Vec<ColumnPlan>,
    /// Columns of the config whose markers were not both found (1-based).
    pub unreadable: Vec<usize>,
}

pub struct ColumnPlan {
    pub geometry: ColumnGeometry,
    pub stop_card: bool,
    pub n_cards: usize,
}

/// A column whose two header markers were found.
pub struct ColumnGeometry {
    pub column: usize,
    /// Strip millimetres -> photo pixels.
    pub mm_to_img: Homography,
    /// Photo pixels -> strip millimetres.
    pub img_to_mm: Homography,
}

/// One slot's OCR reading and its match.
struct Read {
    line: OcrLine,
    m: MatchResult,
}

impl Read {
    /// Higher best score wins; ties go to the higher OCR confidence.
    fn better_than(&self, other: &Read) -> bool {
        self.m
            .best_score()
            .total_cmp(&other.m.best_score())
            .then(self.line.confidence.total_cmp(&other.line.confidence))
            .is_gt()
    }
}

impl Scanner {
    pub fn new(index: NameIndex, recognizer: Box<dyn Recognizer>) -> Self {
        Scanner { index, recognizer, params: ScanParams::default() }
    }

    /// SPEC §2.1-2.4: find the markers, decide the config, and work out each
    /// readable column's geometry and length.
    pub fn locate(&self, photo: &GrayImage) -> Layout {
        let markers = aruco::detect(photo, &self.params.detector);
        // Card art occasionally decodes as a stray marker, so a header marker
        // is only trusted as part of a geometrically consistent pair, and the
        // C6 ids only count when they sit where a header marker should be.
        let pairs: Vec<Option<ColumnGeometry>> = (1..=6).map(|j| best_pair(&markers, j)).collect();
        let is_c6 = markers
            .iter()
            .filter(|m| (8..=11).contains(&m.id))
            .any(|m| !pairs.iter().any(Option::is_some) || plausible_header(m, &pairs));
        let config = if is_c6 { Config::C6 } else { Config::C4 };
        let mut layout = Layout { config, status: PhotoStatus::Ok, columns: Vec::new(), unreadable: Vec::new() };
        let stops: Vec<&Marker> = markers.iter().filter(|m| m.id == STOP_CARD_ID).collect();
        // `into_iter().take(n)` moves the first n geometries out of `pairs`.
        for (j, pair) in (1..).zip(pairs.into_iter().take(config.columns())) {
            match pair {
                Some(g) => {
                    let stop_slot = stops.iter().find_map(|m| stop_slot_in_column(&g, m));
                    let n_cards = match stop_slot {
                        Some(k) => (k - 1).clamp(0, CARDS_PER_COLUMN as i64) as usize,
                        None => CARDS_PER_COLUMN,
                    };
                    layout.columns.push(ColumnPlan { geometry: g, stop_card: stop_slot.is_some(), n_cards });
                }
                None => {
                    // One or both ids missing: the column can't be read.
                    layout.status = PhotoStatus::Retake;
                    layout.unreadable.push(j);
                }
            }
        }
        layout
    }

    /// SPEC §2.5: one OCR-ready crop per card, column by column, slot 1 first.
    pub fn crops(&self, photo: &GrayImage, layout: &Layout) -> Vec<GrayImage> {
        layout
            .columns
            .iter()
            .flat_map(|c| (1..=c.n_cards).map(move |i| (c, i)))
            .collect::<Vec<_>>()
            // Warping is independent per slot: do it on all cores.
            .into_par_iter()
            .map(|(c, i)| self.slot_crop(photo, &c.geometry, i))
            .collect()
    }

    /// Scan one photo. `image_name` is copied into the result.
    pub fn scan(&self, photo: &GrayImage, image_name: &str) -> Result<PhotoResult, Error> {
        let t0 = Instant::now();
        let mut timing = Timing::default();

        let layout = self.locate(photo);
        timing.detect = t0.elapsed().as_millis() as u64;
        let mut columns: Vec<ColumnResult> = layout.unreadable.iter().map(|&j| error_column(j)).collect();

        let t_crop = Instant::now();
        let crops = self.crops(photo, &layout);
        if let Some(dir) = &self.params.dump_dir {
            let stem = image_name.rsplit_once('.').map_or(image_name, |(s, _)| s);
            let names = layout.columns.iter().flat_map(|c| (1..=c.n_cards).map(move |i| (c.geometry.column, i)));
            for ((j, i), crop) in names.zip(&crops) {
                // Debug output only: ignore write errors.
                let _ = crop.save(dir.join(format!("{stem}_c{j}_s{i:02}.png")));
            }
        }
        timing.crop = t_crop.elapsed().as_millis() as u64;

        // SPEC §2.6-2.7, pass 1: OCR + match the crops as they are.
        let t_ocr = Instant::now();
        let lines = self.recognizer.recognize(&crops)?;
        timing.ocr = t_ocr.elapsed().as_millis() as u64;
        let t_match = Instant::now();
        let mut reads = self.match_all(lines);
        timing.match_ = t_match.elapsed().as_millis() as u64;

        // Pass 2: slots that didn't auto-accept get a second chance with the
        // inverted crop (white-on-dark name bars); the better match wins.
        if self.params.try_inverted {
            let retry: Vec<usize> = (0..reads.len()).filter(|&k| reads[k].m.status != MatchStatus::Auto).collect();
            let inverted: Vec<GrayImage> = retry.iter().map(|&k| invert(&crops[k])).collect();
            let t_ocr = Instant::now();
            let lines = self.recognizer.recognize(&inverted)?;
            timing.ocr += t_ocr.elapsed().as_millis() as u64;
            let t_match = Instant::now();
            for (k, alt) in retry.into_iter().zip(self.match_all(lines)) {
                if alt.better_than(&reads[k]) {
                    reads[k] = alt;
                }
            }
            timing.match_ += t_match.elapsed().as_millis() as u64;
        }

        // Assemble the columns. `by_ref()` lets `take` consume from the shared
        // iterator without moving it, so each column continues where the
        // previous one stopped.
        let mut reads = reads.into_iter();
        for plan in &layout.columns {
            let (column, stop_card, n_cards) = (plan.geometry.column, plan.stop_card, plan.n_cards);
            let slots: Vec<SlotResult> = reads
                .by_ref()
                .take(n_cards)
                .enumerate()
                .map(|(i, r)| slot_result(i + 1, &r.line.text, &r.m))
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
        timing.total = t0.elapsed().as_millis() as u64;

        Ok(PhotoResult {
            spec_version: SPEC_VERSION,
            image: image_name.to_string(),
            config: layout.config,
            status: layout.status,
            columns,
            timing_ms: timing,
        })
    }

    /// Match OCR lines against the name index in parallel (rayon's
    /// `into_par_iter` spreads the work over all cores; `collect` keeps order).
    fn match_all(&self, lines: Vec<OcrLine>) -> Vec<Read> {
        lines.into_par_iter().map(|line| Read { m: self.index.match_raw(&line.text), line }).collect()
    }

    /// The crop handed to OCR for slot `i`: the slot band (widened by
    /// `slot_pad_mm`), tightened vertically around the name text so the
    /// letters fill the OCR model's input height, then optionally CLAHE'd.
    fn slot_crop(&self, photo: &GrayImage, g: &ColumnGeometry, i: usize) -> GrayImage {
        let band = geometry::slot_box_mm(i, self.params.slot_pad_mm);
        let mut crop = warp_rect(photo, &g.mm_to_img, band, PX_PER_MM);
        if let Some(h) = self.params.text_height_mm {
            let card_top = geometry::HEADER_Y + geometry::SLOT_PITCH * (i as f64 - 1.0);
            let (y0, y1) = text_rows(&crop, band.y0, card_top, h);
            crop = image::imageops::crop_imm(&crop, 0, y0, crop.width(), y1 - y0).to_image();
        }
        if self.params.trim_right {
            let x1 = text_right_end(&crop);
            crop = image::imageops::crop_imm(&crop, 0, 0, x1, crop.height()).to_image();
        }
        if let Some(clip) = self.params.clahe_clip {
            crop = clahe(&crop, self.params.clahe_tiles.0, self.params.clahe_tiles.1, clip);
        }
        crop
    }
}

/// Name-bar text centre, nominally this far below a card's top edge (mm).
const TEXT_CENTER_BELOW_TOP: f64 = 5.4;
/// The text centre is searched this far (mm) either side of nominal.
const TEXT_SEARCH_MM: f64 = 2.5;
/// Height (mm) of the window scored for text energy.
const TEXT_CORE_MM: f64 = 3.0;

/// Find the rows of a slot band image (whose first row is at `band_y0` mm)
/// that hold the name text: the window with the most horizontal-gradient
/// energy (text strokes are mostly vertical edges), searched around the
/// nominal text position of a card whose top edge is at `card_top` mm.
/// Returns `height_mm` worth of rows centred on it.
fn text_rows(band: &GrayImage, band_y0: f64, card_top: f64, height_mm: f64) -> (u32, u32) {
    let (w, h) = band.dimensions();
    let raw = band.as_raw();
    // Prefix sums of per-row energy sum(|I(x+1) - I(x)|) make every window O(1).
    let mut prefix = vec![0.0f64; h as usize + 1];
    for y in 0..h as usize {
        let row = &raw[y * w as usize..(y + 1) * w as usize];
        let e: u32 = row.windows(2).map(|p| p[0].abs_diff(p[1]) as u32).sum();
        prefix[y + 1] = prefix[y] + e as f64;
    }
    let to_row = |mm: f64| ((mm - band_y0) * PX_PER_MM).round() as i64;
    let core_half = (TEXT_CORE_MM / 2.0 * PX_PER_MM) as i64;
    let nominal = card_top + TEXT_CENTER_BELOW_TOP;
    let clamp_row = |r: i64| r.clamp(0, h as i64) as usize;
    let mut best = (to_row(nominal), f64::MIN);
    for c in to_row(nominal - TEXT_SEARCH_MM)..=to_row(nominal + TEXT_SEARCH_MM) {
        let e = prefix[clamp_row(c + core_half)] - prefix[clamp_row(c - core_half)];
        if e > best.1 {
            best = (c, e);
        }
    }
    let half = (height_mm / 2.0 * PX_PER_MM) as i64;
    let y0 = (best.0 - half).clamp(0, h as i64 - 1) as u32;
    let y1 = (best.0 + half).clamp(y0 as i64 + 1, h as i64) as u32;
    (y0, y1)
}

/// Minimum blank run (mm) that separates the name from the mana cost.
const NAME_GAP_MM: f64 = 3.5;

/// Right edge (pixel column) of the name text in a crop: the name is
/// left-aligned, so we walk right from the first inked column and stop at the
/// first blank run of at least `NAME_GAP_MM`. If there is no such gap the full
/// width is kept, so a long name is never cut.
fn text_right_end(crop: &GrayImage) -> u32 {
    let (w, h) = crop.dimensions();
    let raw = crop.as_raw();
    // Per-column energy: |I(x+1) - I(x)| summed over the rows.
    let mut energy = vec![0f64; w as usize];
    for y in 0..h as usize {
        let row = &raw[y * w as usize..(y + 1) * w as usize];
        for (x, p) in row.windows(2).enumerate() {
            energy[x] += p[0].abs_diff(p[1]) as f64;
        }
    }
    // Smooth over ~0.5 mm so letters' inner gaps don't look blank.
    let r = (0.25 * PX_PER_MM) as usize;
    let smooth: Vec<f64> = (0..energy.len())
        .map(|x| {
            let (a, b) = (x.saturating_sub(r), (x + r + 1).min(energy.len()));
            energy[a..b].iter().sum::<f64>() / (b - a) as f64
        })
        .collect();
    let mut sorted = smooth.clone();
    sorted.sort_by(f64::total_cmp);
    let pct = |q: f64| sorted[((sorted.len() - 1) as f64 * q) as usize];
    let (floor, peak) = (pct(0.1), pct(0.95));
    if peak - floor < 1e-9 {
        return w;
    }
    let threshold = floor + 0.2 * (peak - floor);
    let gap = (NAME_GAP_MM * PX_PER_MM) as usize;
    let margin = PX_PER_MM as usize; // keep 1 mm after the last letter
    let Some(start) = smooth.iter().position(|&e| e > threshold) else { return w };
    let mut blank_run = 0;
    for x in start..smooth.len() {
        if smooth[x] > threshold {
            blank_run = 0;
        } else {
            blank_run += 1;
            if blank_run >= gap {
                let end = x + 1 - blank_run;
                return ((end + margin).min(w as usize) as u32).max(1);
            }
        }
    }
    w
}

/// Header marker centres are 40 mm apart and the markers are 18 mm wide.
const PAIR_RATIO: f64 = 40.0 / 18.0;

/// Pick column `j`'s left/right markers: among all detections of the two ids,
/// the pair whose spacing best matches the printed layout (within 20%).
fn best_pair(markers: &[Marker], j: usize) -> Option<ColumnGeometry> {
    let of = |side| markers.iter().filter(move |m| m.id == geometry::marker_id(j, side));
    let mut best: Option<(f64, &Marker, &Marker)> = None;
    for l in of(Side::Left) {
        for r in of(Side::Right) {
            let side = (l.side() + r.side()) / 2.0;
            let ratio = l.center().dist(r.center()) / side;
            let size_ok = l.side().max(r.side()) < 1.3 * l.side().min(r.side());
            let err = (ratio / PAIR_RATIO - 1.0).abs();
            if size_ok && err < 0.2 && best.is_none_or(|b| err < b.0) {
                best = Some((err, l, r));
            }
        }
    }
    best.and_then(|(_, l, r)| column_geometry(j, l, r))
}

/// Does marker `m` (a header id) sit where its header marker should be,
/// judged from the columns that were found? Allows 10 mm of slack, plus 3%
/// of the distance to the reference column for extrapolation error.
fn plausible_header(m: &Marker, pairs: &[Option<ColumnGeometry>]) -> bool {
    let col = (m.id / 2) as usize + 1;
    let side = if m.id % 2 == 0 { Side::Left } else { Side::Right };
    let corners = geometry::header_corners_mm(side);
    let expected_in_own = Point::new((corners[0].x + corners[1].x) / 2.0, (corners[0].y + corners[2].y) / 2.0);
    pairs.iter().flatten().any(|g| {
        let dx = (col as f64 - g.column as f64) * geometry::STRIP_W;
        let expected = Point::new(expected_in_own.x + dx, expected_in_own.y);
        let got = g.img_to_mm.apply(m.center());
        got.dist(expected) < 10.0 + 0.03 * dx.abs()
    })
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
/// Besides "inside the column's x-range" we check that it looks like the
/// printed 30 mm stop marker, centred on the card, so a stray id 40 decoded
/// from card art can't end a column early.
fn stop_slot_in_column(g: &ColumnGeometry, m: &Marker) -> Option<i64> {
    let mm = m.corners.map(|p| g.img_to_mm.apply(p));
    let c = g.img_to_mm.apply(m.center());
    let side_mm = (0..4).map(|i| mm[i].dist(mm[(i + 1) % 4])).sum::<f64>() / 4.0;
    let centred = (c.x - STOP_MARKER_CENTER_X).abs() < 10.0;
    let sized = (side_mm / geometry::STOP_MARKER_SIZE - 1.0).abs() < 0.2;
    if !(0.0..STRIP_W).contains(&c.x) || c.y < geometry::HEADER_Y || !centred || !sized {
        return None;
    }
    // Top edge = mean y of the marker's top-left and top-right corners.
    Some(geometry::stop_card_slot((mm[0].y + mm[1].y) / 2.0))
}

/// Card left edge 3.5 mm + marker offset 16.5 mm + half of 30 mm.
const STOP_MARKER_CENTER_X: f64 = 35.0;

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

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn text_rows_finds_the_textured_band() {
        // A 144-row band starting at 24 mm (card top 25 mm): plain except for
        // a striped "text" band centred at 31 mm (row 84).
        let band = GrayImage::from_fn(200, 144, |x, y| {
            let textured = (66..102).contains(&y) && x % 4 < 2;
            Luma([if textured { 30 } else { 200 }])
        });
        let (y0, y1) = text_rows(&band, 24.0, 25.0, 6.0);
        let centre = (y0 + y1) / 2;
        assert!((82..=86).contains(&centre), "centre row {centre}");
        assert_eq!(y1 - y0, 72);
    }
}
