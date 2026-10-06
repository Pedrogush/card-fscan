//! Runs the shared SPEC §3 test vectors (`testdata/names/match_cases.json`).

use std::path::PathBuf;

use fscan_core::matching::{MatchStatus, NameIndex};
use serde_json::Value;

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is rust/core at compile time.
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn shared_match_cases() {
    let root = repo_root();
    let index = NameIndex::load(&root.join("testdata/names/names_v1.json")).expect("load names_v1.json");
    let cases: Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("testdata/names/match_cases.json")).unwrap())
            .unwrap();
    let mut failures = Vec::new();
    for case in cases["cases"].as_array().unwrap() {
        let raw = case["raw"].as_str().unwrap();
        let r = index.match_raw(raw);
        let status = match r.status {
            MatchStatus::Auto => "auto",
            MatchStatus::Review => "review",
            MatchStatus::Empty => "empty",
        };
        let mut problems = Vec::new();
        if status != case["status"] {
            problems.push(format!("status {status} != {}", case["status"]));
        }
        if r.cleaned != case["cleaned"].as_str().unwrap() {
            problems.push(format!("cleaned {:?} != {}", r.cleaned, case["cleaned"]));
        }
        if r.key != case["key"].as_str().unwrap() {
            problems.push(format!("key {:?} != {}", r.key, case["key"]));
        }
        if let Some(best) = case["best_score"].as_f64() {
            if (r.best_score() - best).abs() > 0.01 {
                problems.push(format!("best {} != {best}", r.best_score()));
            }
            let ru = case["runner_up_score"].as_f64().unwrap();
            if (r.runner_up_score() - ru).abs() > 0.01 {
                problems.push(format!("runner-up {} != {ru}", r.runner_up_score()));
            }
        }
        if let Some(id) = case["best_oracle_id"].as_str() {
            if r.best().map(|c| c.oracle_id.as_str()) != Some(id) {
                problems.push(format!("best oracle {:?} != {id}", r.best().map(|c| &c.oracle_id)));
            }
        }
        if status == "auto" {
            let best = r.best().unwrap();
            if best.oracle_id != case["oracle_id"].as_str().unwrap()
                || best.name != case["name"].as_str().unwrap()
                || best.lang != case["lang"].as_str().unwrap()
            {
                problems.push(format!("auto result {best:?} != {case}"));
            }
        }
        if !problems.is_empty() {
            failures.push(format!("{raw:?}: {}", problems.join("; ")));
        }
    }
    assert!(failures.is_empty(), "{} failing cases:\n{}", failures.len(), failures.join("\n"));
}

/// The flat-layout search must return exactly what the straightforward
/// per-`String` search returns: same oracle ids, names and scores, in order.
#[test]
fn flat_search_equals_reference() {
    use fscan_core::matching::normalize;
    let root = repo_root();
    let index = NameIndex::load(&root.join("testdata/names/names_v1.json")).expect("load names_v1.json");
    let index_json: Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("testdata/names/names_v1.json")).unwrap()).unwrap();
    let keys: Vec<String> = index_json["entries"]
        .as_array()
        .unwrap()
        .iter()
        .step_by(331)
        .map(|e| e["key"].as_str().unwrap().to_string())
        .collect();
    // Exact keys plus OCR-like damage: truncation, a substitution, a junk
    // suffix, tiny fragments.
    let mut queries = Vec::new();
    for (i, k) in keys.iter().enumerate() {
        queries.push(k.clone());
        queries.push(k.chars().take(k.len() * 2 / 3).collect());
        let mut sub: Vec<char> = k.chars().collect();
        let j = i % sub.len();
        sub[j] = if sub[j] == 'e' { 'c' } else { 'e' };
        queries.push(sub.into_iter().collect());
        queries.push(format!("{k} xq"));
        queries.push(k.chars().take(2).collect());
    }
    for raw in ["mo", "x", "the", "a b c", "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"] {
        queries.push(raw.to_string());
    }
    for q in queries.iter().map(|q| normalize(q)).filter(|q| !q.is_empty()) {
        let fast = index.top_candidates(&q, 3);
        let slow = index.top_candidates_exhaustive(&q, 3);
        assert_eq!(fast, slow, "query {q:?}");
    }
}
