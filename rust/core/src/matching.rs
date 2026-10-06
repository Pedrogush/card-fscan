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
    /// `by_len[l]`: indices of the entries whose key is `l` bytes long, in
    /// index order. Lets a query skip whole lengths that cannot score high
    /// enough (see `top_candidates`).
    by_len: Vec<Vec<u32>>,
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

    /// Compare scores as exact fractions; an empty `Best` ranks lowest.
    fn cmp_score(&self, other: &Best) -> std::cmp::Ordering {
        match (self.lensum, other.lensum) {
            (0, 0) => std::cmp::Ordering::Equal,
            (0, _) => std::cmp::Ordering::Less,
            (_, 0) => std::cmp::Ordering::Greater,
            _ => ((self.lcs2 as u64) * (other.lensum as u64)).cmp(&((other.lcs2 as u64) * (self.lensum as u64))),
        }
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
        let max_len = entries.iter().map(|e| e.key.len()).max().unwrap_or(0);
        let mut by_len = vec![Vec::new(); max_len + 1];
        for (i, e) in entries.iter().enumerate() {
            by_len[e.key.len()].push(i as u32);
        }
        NameIndex { entries, oracle_ids, by_len }
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

    /// Score `key` against the index; return the `n` best distinct oracle
    /// ids ordered by score descending, then oracle id ascending.
    ///
    /// Pruning (exact, results identical to scoring every entry): an entry
    /// of length `l` can share at most `min(len(key), l)` characters with the
    /// key, so its score is at most `100 * 2 min / (len(key) + l)`. Lengths
    /// are visited from the highest such bound down, and the scan stops as
    /// soon as the bound is strictly below the current n-th best score: no
    /// remaining entry could enter the top n or change a member of it.
    pub fn top_candidates(&self, key: &str, n: usize) -> Vec<Candidate> {
        let la = key.len() as u32;
        let pattern = LcsPattern::new(key.as_bytes());
        // Upper bound for each non-empty length, as a fraction lcs2 / lensum.
        let mut lengths: Vec<(u32, Best)> = (0..self.by_len.len() as u32)
            .filter(|&l| !self.by_len[l as usize].is_empty())
            .map(|l| (l, Best { lcs2: 2 * la.min(l), lensum: la + l, entry: 0 }))
            .collect();
        // `sort_by` with a closure: highest bound first.
        lengths.sort_by(|a, b| b.1.cmp_score(&a.1));

        let mut best: Vec<Best> = vec![Best::NONE; self.oracle_ids.len()];
        // The running top n as (oracle, its best), best first.
        let mut top: Vec<(u32, Best)> = Vec::with_capacity(n + 1);
        for (l, bound) in lengths {
            if n == 0 || (top.len() == n && bound.cmp_score(&top[n - 1].1).is_lt()) {
                break;
            }
            for &i in &self.by_len[l as usize] {
                let e = &self.entries[i as usize];
                let lcs = pattern.lcs(e.key.as_bytes()) as u32;
                let cand = Best { lcs2: 2 * lcs, lensum: la + l, entry: i };
                let slot = &mut best[e.oracle as usize];
                // Per oracle: higher score wins; on a tie the earlier entry
                // in index order (entries are not visited in index order).
                let better = match cand.cmp_score(slot) {
                    std::cmp::Ordering::Greater => true,
                    std::cmp::Ordering::Equal => slot.lensum == 0 || cand.entry < slot.entry,
                    std::cmp::Ordering::Less => false,
                };
                if better {
                    *slot = cand;
                    update_top(&mut top, e.oracle, cand, n);
                }
            }
        }
        top.iter().map(|&(_, b)| self.candidate(key, b)).collect()
    }

    /// Reference version of [`top_candidates`](Self::top_candidates) that
    /// scores every entry (used by tests to check the pruning).
    pub fn top_candidates_exhaustive(&self, key: &str, n: usize) -> Vec<Candidate> {
        let pattern = LcsPattern::new(key.as_bytes());
        let mut best = vec![Best::NONE; self.oracle_ids.len()];
        for (i, e) in self.entries.iter().enumerate() {
            let lcs = pattern.lcs(e.key.as_bytes());
            let cand = Best { lcs2: 2 * lcs as u32, lensum: (key.len() + e.key.len()) as u32, entry: i as u32 };
            let slot = &mut best[e.oracle as usize];
            // Strictly better only: on a tie the earlier entry (index order) stays.
            if cand.beats(slot) {
                *slot = cand;
            }
        }
        // Keep a small sorted top-n list. Iterating oracle ids in ascending
        // order and inserting only on a strict win gives the id tie-break.
        let mut top: Vec<Best> = Vec::with_capacity(n + 1);
        for b in best.iter().filter(|b| b.lensum != 0) {
            if top.len() == n && !b.beats(top.last().expect("n > 0")) {
                continue;
            }
            let pos = top.iter().position(|t| b.beats(t)).unwrap_or(top.len());
            top.insert(pos, *b);
            top.truncate(n);
        }
        top.iter().map(|&b| self.candidate(key, b)).collect()
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

/// Record that `oracle`'s best is now `b` in the running top-n list, which is
/// ordered by score descending, then oracle id ascending.
fn update_top(top: &mut Vec<(u32, Best)>, oracle: u32, b: Best, n: usize) {
    if let Some(pos) = top.iter().position(|&(o, _)| o == oracle) {
        top.remove(pos);
    }
    // Does `(oracle, b)` rank before `(o, t)`?
    let before = |&(o, t): &(u32, Best)| match b.cmp_score(&t) {
        std::cmp::Ordering::Equal => oracle < o,
        ord => ord.is_gt(),
    };
    let pos = top.iter().position(before).unwrap_or(top.len());
    if pos < n {
        top.insert(pos, (oracle, b));
        top.truncate(n);
    }
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
