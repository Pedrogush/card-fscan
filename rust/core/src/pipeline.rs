//! SPEC §2 end to end: markers -> columns -> homographies -> stop card ->
//! slot crops -> OCR -> matching -> [`PhotoResult`].

use std::path::PathBuf;
use std::time::Instant;

use image::GrayImage;
use rayon::prelude::*;

use crate::Error;
use crate::aruco::{self, DetectorParams, Marker};
use crate::geometry::{self, CARDS_PER_COLUMN, PX_PER_MM, Point, Rect, STOP_CARD_ID, STRIP_W, Side};
use crate::homography::Homography;
use crate::image_ops::{clahe, invert, warp_rect};
use crate::matching::{MatchResult, MatchStatus, NameIndex};
use crate::ocr::{OcrLine, Recognizer};
use crate::output::*;
use crate::profile::{Count, Profile, Stage, ms_since};

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
    /// The accurate reader.
    pub recognizer: Box<dyn Recognizer>,
    /// Optional cheap first reader: when set, it reads every slot first and
    /// `recognizer` only re-reads the slots it did not auto-accept.
    pub fast_recognizer: Option<Box<dyn Recognizer>>,
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
        Scanner { index, recognizer, fast_recognizer: None, params: ScanParams::default() }
    }

    /// SPEC §2.1-2.4: find the markers, decide the config, and work out each
    /// readable column's geometry and length.
    pub fn locate(&self, photo: &GrayImage, prof: &Profile) -> Layout {
        let markers = prof.time(Stage::Detect, || aruco::detect(photo, &self.params.detector));
        let t_layout = Instant::now();
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
        prof.add(Stage::Layout, t_layout);
        layout
    }

    /// SPEC §2.5: one OCR-ready crop per card, column by column, slot 1 first.
    ///
    /// The header-only homography is extrapolated a long way (the markers
    /// are 18 mm tall, slot 20 is 200 mm further down), so on tilted or
    /// distorted photos it can drift by several mm at the bottom of a
    /// column. We therefore walk down each column and centre every slot's
    /// text search on the drift predicted from the name lines already found
    /// above it (see [`predict_shift`]). Slot numbering is unchanged.
    pub fn crops(&self, photo: &GrayImage, layout: &Layout, prof: &Profile) -> Vec<SlotCrop> {
        layout
            .columns
            .par_iter()
            .flat_map_iter(|c| {
                let mut offsets: Vec<f64> = Vec::with_capacity(c.n_cards);
                (1..=c.n_cards)
                    .map(|i| {
                        let shift_mm = prof.time(Stage::LineFind, || predict_shift(&offsets));
                        let (image, offset) = self
                            .slot_crop(photo, &c.geometry, i, shift_mm, CropVariant::Primary, prof)
                            .expect("primary crop always exists");
                        offsets.push(offset);
                        SlotCrop { column: c.geometry.column, slot: i, shift_mm, image }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Scan one photo. `image_name` is copied into the result.
    pub fn scan(&self, photo: &GrayImage, image_name: &str) -> Result<PhotoResult, Error> {
        let t0 = Instant::now();
        let prof = Profile::new();
        let mut timing = Timing::default();

        let layout = self.locate(photo, &prof);
        timing.detect = ms_since(t0);
        let mut columns: Vec<ColumnResult> = layout.unreadable.iter().map(|&j| error_column(j)).collect();

        let t_crop = Instant::now();
        let mut crops = self.crops(photo, &layout, &prof);
        if let Some(dir) = &self.params.dump_dir {
            let stem = image_name.rsplit_once('.').map_or(image_name, |(s, _)| s);
            for c in &crops {
                // Debug output only: ignore write errors.
                let _ = c.image.save(dir.join(format!("{stem}_c{}_s{:02}.png", c.column, c.slot)));
            }
        }
        timing.crop = ms_since(t_crop);

        // SPEC §2.6-2.7, pass 1: OCR + match the crops as they are.
        // `mem::take` moves each image out of its SlotCrop (leaving an empty
        // one behind) instead of copying ~50 KB per slot.
        let images: Vec<GrayImage> = crops.iter_mut().map(|c| std::mem::take(&mut c.image)).collect();
        let first = self.fast_recognizer.as_deref().unwrap_or(self.recognizer.as_ref());
        let t_ocr = Instant::now();
        let lines = first.recognize(&images, &prof)?;
        timing.ocr = ms_since(t_ocr);
        let t_match = Instant::now();
        let mut reads = self.match_all(lines, &prof);
        timing.match_ = ms_since(t_match);

        // Pass 1b (reader cascade): the accurate model re-reads what the fast
        // one could not auto-accept; the better match wins.
        if self.fast_recognizer.is_some() {
            prof.count(Count::FastLines, images.len() as u64);
            let todo: Vec<usize> = (0..reads.len()).filter(|&k| reads[k].m.status != MatchStatus::Auto).collect();
            let sub: Vec<GrayImage> = todo.iter().map(|&k| images[k].clone()).collect();
            let t_ocr = Instant::now();
            let lines = self.recognizer.recognize(&sub, &prof)?;
            timing.ocr += ms_since(t_ocr);
            let t_match = Instant::now();
            for (k, alt) in todo.into_iter().zip(self.match_all(lines, &prof)) {
                if alt.better_than(&reads[k]) {
                    reads[k] = alt;
                }
            }
            timing.match_ += ms_since(t_match);
        }

        // Pass 2: slots that didn't auto-accept get more attempts, as a
        // cascade: one fallback crop variant per round, and only for slots
        // still not auto-accepted. A variant is skipped when it would just
        // repeat the primary crop (see `slot_crop`). Per round, the
        // better-scoring reading wins.
        let mut pending: Vec<usize> = (0..reads.len()).filter(|&k| reads[k].m.status != MatchStatus::Auto).collect();
        if !pending.is_empty() {
            let t_retry = Instant::now();
            prof.count(Count::RetrySlots, pending.len() as u64);
            let geometry_of =
                |j: usize| &layout.columns.iter().find(|c| c.geometry.column == j).expect("crop of a known column").geometry;
            let mut variants = vec![CropVariant::Untrimmed, CropVariant::SecondLine];
            if self.params.try_inverted {
                variants.push(CropVariant::Inverted);
            }
            for v in variants {
                if pending.is_empty() {
                    break;
                }
                let t_crop = Instant::now();
                // (read index, crop) for every slot where this variant exists.
                let jobs: Vec<(usize, GrayImage)> = pending
                    .par_iter()
                    .filter_map(|&k| {
                        let c = &crops[k];
                        self.slot_crop(photo, geometry_of(c.column), c.slot, c.shift_mm, v, &prof).map(|(img, _)| (k, img))
                    })
                    .collect();
                timing.crop += ms_since(t_crop);
                prof.count(Count::RetryCrops, jobs.len() as u64);
                let (owners, extra): (Vec<usize>, Vec<GrayImage>) = jobs.into_iter().unzip();
                let t_ocr = Instant::now();
                let lines = self.recognizer.recognize(&extra, &prof)?;
                timing.ocr += ms_since(t_ocr);
                let t_match = Instant::now();
                for (k, alt) in owners.into_iter().zip(self.match_all(lines, &prof)) {
                    if alt.better_than(&reads[k]) {
                        reads[k] = alt;
                    }
                }
                timing.match_ += ms_since(t_match);
                pending.retain(|&k| reads[k].m.status != MatchStatus::Auto);
            }
            timing.retry = ms_since(t_retry);
        }

        // Assemble the columns. `by_ref()` lets `take` consume from the shared
        // iterator without moving it, so each column continues where the
        // previous one stopped.
        let n_slots = reads.len();
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
        timing.scan = ms_since(t0);
        timing.total = timing.scan;
        timing.stages = prof.stage_times();
        timing.counters = prof.counters(n_slots);

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
    fn match_all(&self, lines: Vec<OcrLine>, prof: &Profile) -> Vec<Read> {
        lines.into_par_iter().map(|line| self.match_line(line, prof)).collect()
    }

    /// Match one OCR line. Mana costs read as letters (`Dig Through Time
    /// 6UU`) survive SPEC cleaning, so when the last token looks like a mana
    /// cost we also try without it and keep the better match.
    ///
    /// A reading of fewer than `MIN_AUTO_CHARS` letters/digits is almost
    /// always a fragment (one glyph of a decorated name, a mana symbol), yet it
    /// can exactly match a 1-2 letter card name such as "X". Such slots are
    /// sent to review. This only ever makes the SPEC accept rule stricter.
    fn match_line(&self, line: OcrLine, prof: &Profile) -> Read {
        let mut m = self.index.match_raw_profiled(&line.text, prof);
        if let Some(stripped) = strip_mana_token(&line.text) {
            let alt = self.index.match_raw_profiled(stripped, prof);
            if alt.best_score() > m.best_score() {
                m = alt;
            }
        }
        if m.status == MatchStatus::Auto && m.key.chars().filter(|c| *c != ' ').count() < MIN_AUTO_CHARS {
            m.status = MatchStatus::Review;
        }
        Read { m, line }
    }

    /// The crop handed to OCR for slot `i`: the slot band (widened by
    /// `slot_pad_mm` and moved down by the predicted drift `shift_mm`),
    /// tightened vertically around the name text so the letters fill the OCR
    /// model's input height, cut after the name, then optionally CLAHE'd.
    /// `variant` picks fallbacks for a second attempt. Also returns where the
    /// text line was found, in mm relative to its nominal (undrifted) place.
    fn slot_crop(
        &self,
        photo: &GrayImage,
        g: &ColumnGeometry,
        i: usize,
        shift_mm: f64,
        variant: CropVariant,
        prof: &Profile,
    ) -> Option<(GrayImage, f64)> {
        let nominal = geometry::slot_box_mm(i, self.params.slot_pad_mm);
        let band = Rect { y0: nominal.y0 + shift_mm, y1: nominal.y1 + shift_mm, ..nominal };
        let mut crop = prof.time(Stage::Warp, || warp_rect(photo, &g.mm_to_img, band, PX_PER_MM));
        let card_top = geometry::HEADER_Y + geometry::SLOT_PITCH * (i as f64 - 1.0);
        let mut offset = shift_mm;
        if let Some(h) = self.params.text_height_mm {
            let lines = prof.time(Stage::LineFind, || text_rows(&crop, band.y0, card_top + shift_mm, h));
            let line = match variant {
                CropVariant::SecondLine => *lines.get(1)?,
                _ => lines[0],
            };
            offset = line.center_mm - (card_top + TEXT_CENTER_BELOW_TOP);
            crop = image::imageops::crop_imm(&crop, 0, line.y0, crop.width(), line.y1 - line.y0).to_image();
        } else if variant == CropVariant::SecondLine {
            return None;
        }
        if self.params.trim_right {
            let x1 = prof.time(Stage::Trim, || text_right_end(&crop));
            match variant {
                // The untrimmed variant only differs from the primary crop
                // when the trim actually cut something.
                CropVariant::Untrimmed if x1 >= crop.width() => return None,
                CropVariant::Untrimmed => {}
                _ => crop = image::imageops::crop_imm(&crop, 0, 0, x1, crop.height()).to_image(),
            }
        } else if variant == CropVariant::Untrimmed {
            return None;
        }
        let t = Instant::now();
        if let Some(clip) = self.params.clahe_clip {
            crop = clahe(&crop, self.params.clahe_tiles.0, self.params.clahe_tiles.1, clip);
        }
        if variant == CropVariant::Inverted {
            crop = invert(&crop);
        }
        prof.add(Stage::Enhance, t);
        Some((crop, offset))
    }
}

/// One slot's OCR input.
pub struct SlotCrop {
    pub column: usize,
    pub slot: usize,
    /// Drift correction (mm, down the column) used for this slot.
    pub shift_mm: f64,
    pub image: GrayImage,
}

/// Predict the drift (mm) of the next slot's name line from the offsets
/// measured on the slots above it: a straight-line fit through the last few
/// offsets, ignoring outliers (a mis-located line, or a card placed 2 mm off).
pub fn predict_shift(offsets: &[f64]) -> f64 {
    const WINDOW: usize = 8;
    const OUTLIER_MM: f64 = 2.5;
    let start = offsets.len().saturating_sub(WINDOW);
    let recent: Vec<(f64, f64)> = offsets[start..].iter().enumerate().map(|(k, &o)| (k as f64, o)).collect();
    if recent.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = recent.iter().map(|&(_, o)| o).collect();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    let inliers: Vec<(f64, f64)> = recent.into_iter().filter(|&(_, o)| (o - median).abs() <= OUTLIER_MM).collect();
    if inliers.len() < 4 {
        // Too few points for a trend: just follow the median offset.
        return median;
    }
    // Least-squares line o = a + b * k, evaluated at the next index.
    let n = inliers.len() as f64;
    let (sk, so) = inliers.iter().fold((0.0, 0.0), |(sk, so), &(k, o)| (sk + k, so + o));
    let (mk, mo) = (sk / n, so / n);
    let (sxy, sxx) = inliers.iter().fold((0.0, 0.0), |(sxy, sxx), &(k, o)| (sxy + (k - mk) * (o - mo), sxx + (k - mk).powi(2)));
    let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    let next = (offsets.len() - start) as f64;
    (mo + slope * (next - mk)).clamp(median - OUTLIER_MM, median + OUTLIER_MM)
}

/// Which crop of a slot to read. `Primary` is always tried; the others are
/// fallbacks for slots whose first reading did not auto-accept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CropVariant {
    Primary,
    /// The primary crop inverted (white text on a dark name bar).
    Inverted,
    /// The second-best text line (e.g. when an ornate frame out-scored the name).
    SecondLine,
    /// The primary line without cutting off the right-hand side.
    Untrimmed,
}

/// `text` without its last token if that token looks like a mana cost read
/// as text: only digits, `X` and the colour letters `WUBRGC`, at least two
/// characters long, with a digit or only upper-case letters.
fn strip_mana_token(text: &str) -> Option<&str> {
    let trimmed = text.trim_end();
    let (head, last) = trimmed.rsplit_once(char::is_whitespace)?;
    let manaish = last.len() >= 2
        && last.chars().all(|c| c.is_ascii_digit() || "XWUBRGC{}".contains(c))
        && (last.chars().any(|c| c.is_ascii_digit()) || last.chars().all(|c| c.is_ascii_uppercase()));
    (manaish && !head.trim().is_empty()).then_some(head)
}

/// Decode a JPEG/PNG photo to 8-bit grayscale. Returns the image and the
/// milliseconds spent decoding and converting to gray.
///
/// JPEGs store brightness (Y) and colour (Cb, Cr) separately, and the
/// pipeline only needs Y. Asking zune-jpeg for `Luma` output skips the colour
/// planes' storage, upsampling and colour conversion, so the gray image comes
/// straight out of the decoder (the "gray" step then costs nothing). Other
/// formats go through the `image` crate.
pub fn decode_gray(bytes: &[u8]) -> Result<(GrayImage, f64, f64), Error> {
    let t = Instant::now();
    if bytes.starts_with(&[0xFF, 0xD8]) {
        use zune_core::{bytestream::ZCursor, colorspace::ColorSpace, options::DecoderOptions};
        let options = DecoderOptions::default()
            .jpeg_set_out_colorspace(ColorSpace::Luma)
            .set_max_width(1 << 16)
            .set_max_height(1 << 16);
        let mut decoder = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(bytes), options);
        let pixels = decoder.decode().map_err(|e| Error::Invalid(format!("JPEG decode: {e:?}")))?;
        let (w, h) = decoder.dimensions().ok_or_else(|| Error::Invalid("JPEG without dimensions".into()))?;
        let gray = GrayImage::from_raw(w as u32, h as u32, pixels)
            .ok_or_else(|| Error::Invalid("JPEG luma buffer has the wrong size".into()))?;
        return Ok((gray, ms_since(t), 0.0));
    }
    let img = image::load_from_memory(bytes)?;
    let decode = ms_since(t);
    let t = Instant::now();
    let gray = img.into_luma8();
    Ok((gray, decode, ms_since(t)))
}

/// See [`Scanner::match_line`].
const MIN_AUTO_CHARS: usize = 3;

/// Name-bar text centre, nominally this far below a card's top edge (mm).
const TEXT_CENTER_BELOW_TOP: f64 = 5.4;
/// The text centre is searched this far (mm) either side of nominal.
const TEXT_SEARCH_MM: f64 = 2.5;
/// Height (mm) of the window scored for text energy.
const TEXT_CORE_MM: f64 = 3.0;

/// A located text line inside a slot band image.
#[derive(Debug, Clone, Copy, PartialEq)]
struct TextLine {
    /// Rows `y0..y1` of the band image to crop.
    y0: u32,
    y1: u32,
    /// Strip-mm position of the line's centre.
    center_mm: f64,
}

/// Find the rows of a slot band image (whose first row is at `band_y0` mm)
/// that hold the name text: windows with the most horizontal-gradient energy
/// (text strokes are mostly vertical edges), searched around the nominal text
/// position of a card whose top edge is at `card_top` mm. Returns the best
/// window and, if there is one, the best other local peak at least 2 mm away,
/// each as `height_mm` worth of rows.
fn text_rows(band: &GrayImage, band_y0: f64, card_top: f64, height_mm: f64) -> Vec<TextLine> {
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
    let centres: Vec<i64> = (to_row(nominal - TEXT_SEARCH_MM)..=to_row(nominal + TEXT_SEARCH_MM)).collect();
    let energy: Vec<f64> =
        centres.iter().map(|&c| prefix[clamp_row(c + core_half)] - prefix[clamp_row(c - core_half)]).collect();
    let best = (0..centres.len()).max_by(|&a, &b| energy[a].total_cmp(&energy[b])).unwrap_or(0);
    // A local maximum (not just the slope next to the best peak).
    let min_sep = (2.0 * PX_PER_MM) as i64;
    let second = (1..centres.len().saturating_sub(1))
        .filter(|&k| (centres[k] - centres[best]).abs() >= min_sep)
        .filter(|&k| energy[k] >= energy[k - 1] && energy[k] >= energy[k + 1])
        .max_by(|&a, &b| energy[a].total_cmp(&energy[b]));
    let half = (height_mm / 2.0 * PX_PER_MM) as i64;
    let window = |c: i64| {
        let y0 = (c - half).clamp(0, h as i64 - 1) as u32;
        let y1 = (c + half).clamp(y0 as i64 + 1, h as i64) as u32;
        TextLine { y0, y1, center_mm: band_y0 + c as f64 / PX_PER_MM }
    };
    std::iter::once(best).chain(second).map(|k| window(centres[k])).collect()
}

/// Minimum blank run (mm) that separates the name from the mana cost.
const NAME_GAP_MM: f64 = 3.5;
/// The mana cost occupies at most this much (mm) at the right of the bar.
const MANA_ZONE_MM: f64 = 22.0;

/// Right edge (pixel column) of the name text in a crop. Walking in from the
/// right, we look for a blank run of at least `NAME_GAP_MM` that starts
/// within the rightmost `MANA_ZONE_MM` (anything right of it is the mana cost
/// or nothing) and cut just left of it. If there is no such gap the full
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
    let zone_start = (w as f64 - MANA_ZONE_MM * PX_PER_MM).max(0.0) as usize;
    let margin = PX_PER_MM as usize; // keep 1 mm after the last letter
    // Walk leftwards; `run_end` is the right end of the current blank run.
    let mut run_end: Option<usize> = None;
    for x in (0..smooth.len()).rev() {
        if smooth[x] <= threshold {
            let end = *run_end.get_or_insert(x);
            if end - x + 1 >= gap && end >= zone_start {
                // Found the gap; extend it to its left end, then cut there.
                let mut left = x;
                while left > 0 && smooth[left - 1] <= threshold {
                    left -= 1;
                }
                if left == 0 {
                    return w; // nothing inked at all: keep everything
                }
                return ((left + margin).min(w as usize) as u32).max(1);
            }
        } else {
            run_end = None;
            if x < zone_start {
                return w; // ink continues past the mana zone: a long name
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
    fn drift_prediction_follows_a_trend_and_ignores_outliers() {
        assert_eq!(predict_shift(&[]), 0.0);
        // Drift growing by 0.5 mm per slot, plus one mis-located line.
        let offsets = [0.0, 0.5, 1.0, 9.0, 2.0, 2.5, 3.0];
        let p = predict_shift(&offsets);
        assert!((p - 3.5).abs() < 0.3, "{p}");
    }

    #[test]
    fn mana_tokens() {
        assert_eq!(strip_mana_token("Dig Through Time 6UU"), Some("Dig Through Time"));
        assert_eq!(strip_mana_token("Shivan Dragon 4RR"), Some("Shivan Dragon"));
        assert_eq!(strip_mana_token("Serra Angel"), None);
        assert_eq!(strip_mana_token("Fire // Ice"), None);
        assert_eq!(strip_mana_token("Ponder U"), None);
        assert_eq!(strip_mana_token("6UU"), None);
    }

    #[test]
    fn right_trim_cuts_before_mana_cost() {
        // 61 mm wide: "name" ink 2..20 mm, mana symbols 55..60 mm.
        let w = (61.0 * PX_PER_MM) as u32;
        let crop = GrayImage::from_fn(w, 60, |x, y| {
            let mm = x as f64 / PX_PER_MM;
            let inked = (2.0..20.0).contains(&mm) || (55.0..60.0).contains(&mm);
            Luma([if inked && (x + y) % 3 == 0 { 20 } else { 220 }])
        });
        let end = text_right_end(&crop) as f64 / PX_PER_MM;
        assert!((20.0..22.5).contains(&end), "cut at {end} mm");
        // A long name running into the mana zone is not cut.
        let long = GrayImage::from_fn(w, 60, |x, y| {
            let mm = x as f64 / PX_PER_MM;
            let inked = (2.0..53.0).contains(&mm) || (55.0..60.0).contains(&mm);
            Luma([if inked && (x + y) % 3 == 0 { 20 } else { 220 }])
        });
        assert_eq!(text_right_end(&long), w);
    }

    #[test]
    fn text_rows_finds_the_textured_band() {
        // A 144-row band starting at 24 mm (card top 25 mm): plain except for
        // a striped "text" band centred at 31 mm (row 84).
        let band = GrayImage::from_fn(200, 144, |x, y| {
            let textured = (66..102).contains(&y) && x % 4 < 2;
            Luma([if textured { 30 } else { 200 }])
        });
        let line = text_rows(&band, 24.0, 25.0, 6.0)[0];
        let (y0, y1) = (line.y0, line.y1);
        let centre = (y0 + y1) / 2;
        assert!((line.center_mm - 31.0).abs() < 0.2);
        assert!((82..=86).contains(&centre), "centre row {centre}");
        assert_eq!(y1 - y0, 72);
    }
}
