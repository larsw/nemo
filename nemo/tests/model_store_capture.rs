//! End-to-end capture of a materialized model into a store.
//!
//! Exercises the whole chain against a real program rather than a hand-built
//! fixture: run inference, write a store, reopen it, and require the manifest,
//! the per-step subtables and the dictionary to describe what was computed.
//!
//! The dictionary matters most here. A trie stores dictionary-relative ids, so
//! the encoded tables mean nothing unless the snapshot captured exactly the ids
//! they reference — and that is precisely the coupling a unit test on either half
//! alone cannot check.

use std::{
    collections::BTreeMap,
    sync::{Mutex, MutexGuard},
};

use assert_fs::TempDir;
use nemo::{
    api::{load, reason},
    model_store::{CacheKey, ImportFingerprint, ModelStore},
};

/// Transitive closure with a derived predicate, so the store has to hold several
/// steps for one predicate rather than a single subtable.
const PROGRAM: &str = r#"
edge(1, 2) .
edge(2, 3) .
edge(3, 4) .

path(?x, ?y) :- edge(?x, ?y) .
path(?x, ?z) :- path(?x, ?y), edge(?y, ?z) .

named("a", <http://example.org/a>) .
named("b", <http://example.org/b>) .
labelled(?n, ?i) :- named(?n, ?i) .
"#;

/// Serializes engine runs within this binary.
///
/// `TimedCode::instance()` is a process-global `Mutex<TimedCode>` and the engine
/// writes to it throughout reasoning, so two engines running concurrently in one
/// test binary contend on it -- and one panic poisons it for every other test,
/// which reports `PoisonError` instead of its own failure. Guarding the engine
/// runs keeps failures legible without demanding `--test-threads=1`. Everything
/// after capture touches no globals and still runs in parallel.
static ENGINE: Mutex<()> = Mutex::new(());

/// Take the engine lock, ignoring poisoning: a poisoned lock means some other
/// test failed, which is that test's business, not a reason to fail this one too.
fn engine_lock() -> MutexGuard<'static, ()> {
    ENGINE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Run `PROGRAM` and write a store, returning the temp dir that owns it.
async fn captured_store() -> (TempDir, std::path::PathBuf) {
    let _guard = engine_lock();

    let temp = TempDir::new().expect("temp dir");
    let rules = temp.path().join("program.rls");
    std::fs::write(&rules, PROGRAM).expect("write program");

    let mut engine = load(rules).await.expect("program should load");
    reason(&mut engine).await.expect("reasoning should succeed");

    let path = temp.path().join("store");
    engine
        .write_model_store(&path, BTreeMap::new(), Vec::new())
        .await
        .expect("store should be written");

    (temp, path)
}

#[tokio::test(flavor = "current_thread")]
async fn captures_predicates_steps_and_rule_history() {
    let (_temp, path) = captured_store().await;
    let store = ModelStore::open(&path).expect("store should open");

    let predicates: Vec<&str> = store
        .manifest()
        .predicates
        .iter()
        .map(|entry| entry.predicate.as_str())
        .collect();

    for expected in ["edge", "path", "named", "labelled"] {
        assert!(
            predicates.contains(&expected),
            "predicate {expected} should be recorded, got {predicates:?}"
        );
    }

    // Predicates are ordered so that capturing twice is byte-identical.
    let mut sorted = predicates.clone();
    sorted.sort_unstable();
    assert_eq!(predicates, sorted, "predicates should be name-ordered");

    assert_eq!(store.predicate("edge").expect("recorded").arity, 2);
    assert_eq!(store.predicate("labelled").expect("recorded").arity, 2);

    // The recursive rule fires more than once, so `path` must span steps. This is
    // the property a store of merged relations would lose, and with it the
    // ability to explain anything.
    assert!(
        store.steps_for("path").len() >= 2,
        "path should have several per-step subtables, got {:?}",
        store.steps_for("path")
    );

    // rule_history has to reach every step a subtable claims, or provenance
    // cannot be resolved from the store.
    let rule_history = &store.manifest().rule_history;
    for predicate in ["edge", "path", "named", "labelled"] {
        for step in store.steps_for(predicate) {
            assert!(
                step < rule_history.len(),
                "step {step} of {predicate} is outside rule_history of length {}",
                rule_history.len()
            );
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn tables_decode_and_their_ids_resolve() {
    let (_temp, path) = captured_store().await;
    let store = ModelStore::open(&path).expect("store should open");
    let dictionary = store.dictionary().expect("dictionary should decode");

    let mut tables_seen = 0;
    for entry in &store.manifest().predicates {
        for subtable in &entry.subtables {
            let trie = store
                .load_table(&entry.predicate, subtable.step)
                .expect("table should load")
                .expect("manifest promised this step");

            assert_eq!(
                trie.arity(),
                entry.arity,
                "{} subtable at step {} has the wrong arity",
                entry.predicate,
                subtable.step
            );
            tables_seen += 1;

            // The coupling under test: every id a table references must be
            // resolvable from the snapshot written beside it.
            for id in trie.referenced_dictionary_ids() {
                assert!(
                    dictionary.id_to_datavalue(id).is_some(),
                    "{} at step {} references id {id}, which the snapshot does not resolve",
                    entry.predicate,
                    subtable.step
                );
            }
        }
    }

    assert!(tables_seen > 0, "the store should contain tables");
}

#[tokio::test(flavor = "current_thread")]
async fn the_dictionary_holds_the_program_s_strings_and_iris() {
    let (_temp, path) = captured_store().await;
    let store = ModelStore::open(&path).expect("store should open");
    let dictionary = store.dictionary().expect("dictionary should decode");

    let rendered: Vec<String> = dictionary
        .iter()
        .map(|(_, value)| value.to_string())
        .collect();

    // Integers are carried inline in storage values and are deliberately absent;
    // strings and IRIs need ids and must be present.
    for expected in ["http://example.org/a", "http://example.org/b", "a", "b"] {
        assert!(
            rendered.iter().any(|value| value.contains(expected)),
            "dictionary should contain {expected}, got {rendered:?}"
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn the_cache_key_records_the_normalized_program() {
    let (_temp, path) = captured_store().await;
    let store = ModelStore::open(&path).expect("store should open");
    let key = &store.manifest().cache_key;

    assert!(
        key.program.contains("path"),
        "cache key should hold the normalized program text, got {:?}",
        key.program
    );

    assert!(store.is_applicable(key), "its own key must match");

    // A different parameter binding selects different data, so the store must not
    // be reused even though the program is identical.
    let mut parameters = BTreeMap::new();
    parameters.insert("importfile".to_string(), "\"other.nt\"".to_string());
    let other = CacheKey::new(key.program.clone(), parameters, Vec::new());
    assert!(!store.is_applicable(&other));

    // Likewise if an imported resource changed underneath us.
    let other = CacheKey::new(
        key.program.clone(),
        BTreeMap::new(),
        vec![ImportFingerprint {
            resource: "data.nt".to_string(),
            bytes: Some(17),
            modified_unix_nanos: Some(1),
        }],
    );
    assert!(!store.is_applicable(&other));
}

#[tokio::test(flavor = "current_thread")]
async fn capturing_twice_refuses_rather_than_overwriting() {
    let (_temp, path) = captured_store().await;

    let _guard = engine_lock();

    let rules = path.parent().expect("parent").join("program.rls");
    let mut engine = load(rules).await.expect("program should load");
    reason(&mut engine).await.expect("reasoning should succeed");

    // A store is immutable once published; replacing one is the caller's
    // decision, not a silent side effect of capturing again.
    assert!(
        engine
            .write_model_store(&path, BTreeMap::new(), Vec::new())
            .await
            .is_err()
    );
}
