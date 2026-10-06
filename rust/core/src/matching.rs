//! SPEC §3: OCR-text cleaning, name normalisation, the Indel ratio and the
//! auto-accept rule, plus the in-memory name index the scores are computed
//! against.
//!
//! The executable reference is `tools/reference_match.py`; the test vectors in
//! `testdata/names/match_cases.json` are checked by `core/tests/match_cases.rs`.

use std::io::Read;
use std::path::Path;

use serde::Deserialize;
use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::canonical_combining_class;

use crate::Error;
use crate::profile::{Profile, Stage};
use std::time::Instant;

pub const AUTO_MIN: f64 = 90.0;
pub const AUTO_LEAD: f64 = 5.0;
pub const SHORT_KEY_LEN: usize = 6;
pub const SHORT_MIN: f64 = 95.0;
/// Threshold comparisons tolerate float noise (SPEC §3, "Determinism").
pub const EPS: f64 = 1e-6;

/// Characters stripped from the end of a cleaned OCR line.
const TRAILING_JUNK: &str = "{}()[]0123456789@#*%&+=|\\/";

/// `normalize(s)` from SPEC §3: NFKD, drop combining marks, lowercase, map
/// everything outside `[a-z0-9 ]` to a space, collapse spaces, trim.
pub fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut pending_space = false;
    // Iterator chain: decompose -> drop combining marks -> lowercase. Nothing
    // is allocated per character; `flat_map` is needed because lowercasing
    // one char can yield several.
    let chars = s
        .nfkd()
        .filter(|&c| canonical_combining_class(c) == 0)
        .flat_map(char::to_lowercase);
    for c in chars {
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            if pending_space && !out.is_empty() {
                out.push(' ');
            }
            pending_space = false;
            out.push(c);
        } else {
            pending_space = true;
        }
    }
    out
}

pub fn has_letter(s: &str) -> bool {
    s.chars().any(char::is_alphabetic)
}

/// Strip mana-cost residue from the end of an OCR line (SPEC §3, "Cleaning").
pub fn clean(raw: &str) -> String {
    let mut tokens: Vec<&str> = raw.split_whitespace().collect();
    while tokens.last().is_some_and(|t| !has_letter(t)) {
        tokens.pop();
    }
    let joined = tokens.join(" ");
    joined
        .trim_end_matches(|c: char| TRAILING_JUNK.contains(c) || c.is_whitespace())
        .trim()
        .to_string()
}

/// Length of the longest common subsequence of `pattern` and `text`, using the
/// bit-parallel algorithm of Allison & Dix / Hyyrö (one machine word handles 64
/// pattern characters at once). Both strings must be ASCII.
struct LcsPattern {
    len: usize,
    /// `masks[c][w]`: bit i of word w is set when `pattern[64*w + i] == c`.
    masks: Vec<[u64; 128]>,
}

impl LcsPattern {
    fn new(pattern: &[u8]) -> Self {
        let words = pattern.len().div_ceil(64).max(1);
        let mut masks = vec![[0u64; 128]; words];
        for (i, &c) in pattern.iter().enumerate() {
            masks[i / 64][(c & 0x7f) as usize] |= 1u64 << (i % 64);
        }
        LcsPattern { len: pattern.len(), masks }
    }

    fn lcs(&self, text: &[u8]) -> usize {
        let words = self.masks.len();
        if words == 1 {
            // Fast path: the common case (names are < 64 chars).
            let m = &self.masks[0];
            let mut s = u64::MAX;
            for &c in text {
                let u = s & m[(c & 0x7f) as usize];
                s = s.wrapping_add(u) | (s - u);
            }
            let used = if self.len == 64 { u64::MAX } else { (1u64 << self.len) - 1 };
            return (!s & used).count_ones() as usize;
        }
        let mut s = vec![u64::MAX; words];
        for &c in text {
            let mut carry = 0u64;
            for (w, sw) in s.iter_mut().enumerate() {
                let u = *sw & self.masks[w][(c & 0x7f) as usize];
                let (sum, c1) = sw.overflowing_add(u);
                let (sum, c2) = sum.overflowing_add(carry);
                carry = (c1 || c2) as u64;
                *sw = sum | (*sw - u);
            }
        }
        let mut total = 0;
        for (w, sw) in s.iter().enumerate() {
            let bits = (self.len - 64 * w).min(64);
            let used = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
            total += (!sw & used).count_ones() as usize;
        }
        total
    }
}

/// Indel distance: insertions + deletions needed to turn `a` into `b`.
pub fn indel(a: &str, b: &str) -> usize {
    let lcs = LcsPattern::new(a.as_bytes()).lcs(b.as_bytes());
    a.len() + b.len() - 2 * lcs
}

/// `ratio(a, b) = 100 * (1 - indel / (len a + len b))`, computed the same way
/// as `rapidfuzz.fuzz.ratio`. ASCII inputs (i.e. normalised keys) only.
pub fn ratio(a: &str, b: &str) -> f64 {
    ratio_from_lcs(a.len(), b.len(), LcsPattern::new(a.as_bytes()).lcs(b.as_bytes()))
}

fn ratio_from_lcs(la: usize, lb: usize, lcs: usize) -> f64 {
    let lensum = la + lb;
    if lensum == 0 {
        return 100.0;
    }
    let dist = lensum - 2 * lcs;
    100.0 * (1.0 - dist as f64 / lensum as f64)
}

/// The reference scores in float32 and rounds to 4 decimals before applying
/// the accept rule; we mirror that so boundary cases agree bit for bit.
fn reference_score(raw: f64) -> f64 {
    let f32_score = raw as f32 as f64;
    (f32_score * 1e4).round() / 1e4
}

// ---------------------------------------------------------------------------
// Name index

#[derive(Debug, Deserialize)]
struct IndexFile {
    version: u32,
    entries: Vec<IndexEntry>,
}

#[derive(Debug, Deserialize)]
struct IndexEntry {
    key: String,
    name: String,
    oracle_id: String,
    lang: String,
}

/// One name of one card, as stored in memory.
#[derive(Debug, Clone)]
pub struct Entry {
    pub key: String,
    pub name: String,
    pub lang: String,
    /// Index into [`NameIndex::oracle_ids`].
    oracle: u32,
}

/// All card names, ready for fuzzy lookup.
pub struct NameIndex {
    entries: Vec<Entry>,
    /// Distinct oracle ids, sorted ascending, so that comparing indices is the
    /// same as comparing the id strings (used for tie-breaking).
    oracle_ids: Vec<String>,
    /// The hot data of the scan, laid out flat ("struct of arrays") and
    /// grouped by oracle: all keys back to back in one buffer, ordered by
    /// oracle id, then index order. Slot `j` holds the key
    /// `key_bytes[key_start[j]..key_start[j + 1]]` of entry `key_entry[j]`;
    /// oracle `o` owns slots `oracle_start[o]..oracle_start[o + 1]`.
    ///
    /// A query then reads memory strictly sequentially and finishes each
    /// oracle's best before moving to the next, so it needs no per-query
    /// scratch array. The naive layout (one heap `String` per entry and a
    /// per-oracle array written in random order) was memory-bound and
    /// several times slower; see `examples/match_bench.rs`.
    key_bytes: Vec<u8>,
    key_start: Vec<u32>,
    key_entry: Vec<u32>,
    oracle_start: Vec<u32>,
}

/// One scored oracle id.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub name: String,
    pub oracle_id: String,
    pub lang: String,
    pub key: String,
    pub score: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchStatus {
    Auto,
    Review,
    Empty,
}

#[derive(Debug, Clone)]
pub struct MatchResult {
    pub status: MatchStatus,
    pub cleaned: String,
    pub key: String,
    /// Top 3 distinct oracle ids, best first.
    pub candidates: Vec<Candidate>,
}

impl MatchResult {
    pub fn best(&self) -> Option<&Candidate> {
        self.candidates.first()
    }

    pub fn best_score(&self) -> f64 {
        self.best().map_or(0.0, |c| c.score)
    }

    pub fn runner_up_score(&self) -> f64 {
        self.candidates.get(1).map_or(0.0, |c| c.score)
    }

    fn empty(cleaned: String, key: String) -> Self {
        MatchResult { status: MatchStatus::Empty, cleaned, key, candidates: Vec::new() }
    }
}

/// Best score seen so far for one oracle id, as an exact fraction
/// `lcs2 / lensum` (score = 100 * lcs2 / lensum), so ties are exact.
#[derive(Clone, Copy)]
struct Best {
    lcs2: u32,
    lensum: u32,
    entry: u32,
}

impl Best {
    const NONE: Best = Best { lcs2: 0, lensum: 0, entry: u32::MAX };

    /// `self > other` as fractions (cross-multiplied, so no float rounding).
    fn beats(&self, other: &Best) -> bool {
        if other.lensum == 0 {
            return self.lensum != 0;
        }
        (self.lcs2 as u64) * (other.lensum as u64) > (other.lcs2 as u64) * (self.lensum as u64)
    }
}

impl NameIndex {
    /// Load `names_v1.json`, or a gzip-compressed copy if the path ends in `.gz`.
    pub fn load(path: &Path) -> Result<Self, Error> {
        let bytes = std::fs::read(path).map_err(|e| Error::Io(path.display().to_string(), e))?;
        Self::from_bytes(&bytes, path.extension().is_some_and(|e| e == "gz"))
    }

    pub fn from_bytes(bytes: &[u8], gzipped: bool) -> Result<Self, Error> {
        let file: IndexFile = if gzipped {
            let mut json = Vec::new();
            flate2::read::GzDecoder::new(bytes)
                .read_to_end(&mut json)
                .map_err(|e| Error::Io("gzip name index".into(), e))?;
            serde_json::from_slice(&json)?
        } else {
            serde_json::from_slice(bytes)?
        };
        if file.version != 1 {
            return Err(Error::Invalid(format!("unsupported name index version {}", file.version)));
        }
        Ok(Self::from_entries(file.entries))
    }

    fn from_entries(raw: Vec<IndexEntry>) -> Self {
        let mut oracle_ids: Vec<String> = raw.iter().map(|e| e.oracle_id.clone()).collect();
        oracle_ids.sort();
        oracle_ids.dedup();
        let entries = raw
            .into_iter()
            .map(|e| {
                // `binary_search` returns Ok(position) because the id is present.
                let oracle = oracle_ids.binary_search(&e.oracle_id).expect("id was collected") as u32;
                Entry { key: e.key, name: e.name, lang: e.lang, oracle }
            })
            .collect::<Vec<Entry>>();
        // Entry indices ordered by (oracle, index). `sort_by_key` is stable,
        // so entries of one oracle stay in index order.
        let mut order: Vec<u32> = (0..entries.len() as u32).collect();
        order.sort_by_key(|&i| entries[i as usize].oracle);
        let mut key_bytes = Vec::with_capacity(entries.iter().map(|e| e.key.len()).sum());
        let mut key_start = Vec::with_capacity(entries.len() + 1);
        let mut oracle_start = vec![0u32; oracle_ids.len() + 1];
        for (j, &i) in order.iter().enumerate() {
            let e = &entries[i as usize];
            key_start.push(key_bytes.len() as u32);
            key_bytes.extend_from_slice(e.key.as_bytes());
            oracle_start[e.oracle as usize + 1] = j as u32 + 1;
        }
        key_start.push(key_bytes.len() as u32);
        // Every oracle has at least one entry, so each group end was set;
        // this turns group ends into proper start offsets.
        for o in 1..oracle_start.len() {
            oracle_start[o] = oracle_start[o].max(oracle_start[o - 1]);
        }
        NameIndex { entries, oracle_ids, key_bytes, key_start, key_entry: order, oracle_start }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clean, normalise and score one raw OCR line against every key.
    pub fn match_raw(&self, raw: &str) -> MatchResult {
        self.match_raw_profiled(raw, &Profile::new())
    }

    /// [`match_raw`](Self::match_raw), charging time to `prof`.
    pub fn match_raw_profiled(&self, raw: &str, prof: &Profile) -> MatchResult {
        let t = Instant::now();
        if !has_letter(raw) {
            prof.add(Stage::Clean, t);
            return MatchResult::empty(String::new(), String::new());
        }
        let cleaned = clean(raw);
        let key = normalize(&cleaned);
        prof.add(Stage::Clean, t);
        if key.is_empty() {
            return MatchResult::empty(cleaned, key);
        }
        let t = Instant::now();
        let candidates = self.top_candidates(&key, 3);
        let status = accept(&candidates);
        prof.add(Stage::Match, t);
        MatchResult { status, cleaned, key, candidates }
    }

    /// Score `key` against every entry; return the `n` best distinct oracle
    /// ids ordered by score descending, then oracle id ascending.
    pub fn top_candidates(&self, key: &str, n: usize) -> Vec<Candidate> {
        let la = key.len() as u32;
        let pattern = LcsPattern::new(key.as_bytes());
        let mut top: Vec<Best> = Vec::with_capacity(n + 1);
        // `windows(2)` yields consecutive (start, end) offset pairs.
        for group in self.oracle_start.windows(2) {
            // This oracle's best entry: strictly better only, so on a tie the
            // earlier entry in index order stays.
            let mut best = Best::NONE;
            for j in group[0] as usize..group[1] as usize {
                let k = &self.key_bytes[self.key_start[j] as usize..self.key_start[j + 1] as usize];
                let cand = Best { lcs2: 2 * pattern.lcs(k) as u32, lensum: la + k.len() as u32, entry: self.key_entry[j] };
                if cand.beats(&best) {
                    best = cand;
                }
            }
            insert_top(&mut top, best, n);
        }
        top.iter().map(|&b| self.candidate(key, b)).collect()
    }

    /// The `n` best of the per-oracle bests (oracles in ascending order).
    fn top_n(&self, key: &str, best: &[Best], n: usize) -> Vec<Candidate> {
        let mut top: Vec<Best> = Vec::with_capacity(n + 1);
        for &b in best {
            insert_top(&mut top, b, n);
        }
        top.iter().map(|&b| self.candidate(key, b)).collect()
    }

    /// Reference version of [`top_candidates`](Self::top_candidates) that
    /// reads each entry's own `String` key (the Phase 1 code). Used by tests
    /// and `examples/match_bench.rs` to check the flat layout.
    pub fn top_candidates_exhaustive(&self, key: &str, n: usize) -> Vec<Candidate> {
        let pattern = LcsPattern::new(key.as_bytes());
        let mut best = vec![Best::NONE; self.oracle_ids.len()];
        for (i, e) in self.entries.iter().enumerate() {
            let lcs = pattern.lcs(e.key.as_bytes());
            let cand = Best { lcs2: 2 * lcs as u32, lensum: (key.len() + e.key.len()) as u32, entry: i as u32 };
            let slot = &mut best[e.oracle as usize];
            if cand.beats(slot) {
                *slot = cand;
            }
        }
        self.top_n(key, &best, n)
    }

    fn candidate(&self, key: &str, b: Best) -> Candidate {
        let e = &self.entries[b.entry as usize];
        let lcs = (b.lcs2 / 2) as usize;
        let raw_score = ratio_from_lcs(key.len(), e.key.len(), lcs);
        Candidate {
            name: e.name.clone(),
            oracle_id: self.oracle_ids[e.oracle as usize].clone(),
            lang: e.lang.clone(),
            key: e.key.clone(),
            score: reference_score(raw_score),
        }
    }
}

/// Offer one oracle's best to the sorted top-`n` list. Oracles must be
/// offered in ascending id order: inserting only on a strict win then puts
/// equal scores in oracle-id order, as the spec requires.
fn insert_top(top: &mut Vec<Best>, b: Best, n: usize) {
    if b.lensum == 0 || (top.len() == n && !top.last().is_some_and(|last| b.beats(last))) {
        return;
    }
    let pos = top.iter().position(|t| b.beats(t)).unwrap_or(top.len());
    top.insert(pos, b);
    top.truncate(n);
}

/// The SPEC §3 accept rule on an ordered candidate list.
pub fn accept(candidates: &[Candidate]) -> MatchStatus {
    let Some(best) = candidates.first() else {
        return MatchStatus::Empty;
    };
    let runner = candidates.get(1).map_or(0.0, |c| c.score);
    let mut ok = best.score >= AUTO_MIN - EPS && best.score - runner >= AUTO_LEAD - EPS;
    if best.key.len() <= SHORT_KEY_LEN && best.score < SHORT_MIN - EPS {
        ok = false;
    }
    if ok { MatchStatus::Auto } else { MatchStatus::Review }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_examples() {
        assert_eq!(normalize("Relâmpago"), "relampago");
        assert_eq!(normalize("  Jace, the  Mind-Sculptor "), "jace the mind sculptor");
        assert_eq!(normalize("Fire // Ice"), "fire ice");
        assert_eq!(normalize("Akroma’s Vengeance"), "akroma s vengeance");
        assert_eq!(normalize("Anulação"), "anulacao");
    }

    #[test]
    fn clean_examples() {
        assert_eq!(clean("Lightning Bolt {R}"), "Lightning Bolt {R");
        assert_eq!(clean("Serra Angel 3 @@"), "Serra Angel");
        assert_eq!(clean("Counterspel1"), "Counterspel");
        assert_eq!(clean("{2}{R}"), "{2}{R");
        assert_eq!(clean("12 3 @#"), "");
    }

    #[test]
    fn ratio_matches_rapidfuzz() {
        assert_eq!(ratio("lightning bolt", "lightning bolt"), 100.0);
        // rapidfuzz.fuzz.ratio("this is a test", "this is a test!") == 96.55172413793103
        assert!((ratio("this is a test", "this is a test!") - 96.551_724_137_931_03).abs() < 1e-9);
        assert_eq!(indel("abc", "axc"), 2);
    }

    #[test]
    fn multiword_lcs_agrees_with_single_word() {
        let a = "a".repeat(70) + "bcd";
        let b = "a".repeat(65) + "bxd";
        // LCS = 65 a's + "bd" = 67, indel = 73 + 68 - 134 = 7
        assert_eq!(indel(&a, &b), 7);
        assert_eq!(indel(&b, &a), 7);
    }
}
