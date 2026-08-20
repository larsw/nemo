//! Guards the property the persistence design depends on: the trie encoding is
//! reachable from *outside* `nemo-physical`.
//!
//! This matters more than it looks. Every type the encoding is built from --
//! `ColumnEnum`, `ColumnVector`, `ColumnRle`, `IntervalColumnT`, `IntervalColumn`,
//! `IntervalLookup`, `TrieScanEnum` -- is `pub(crate)`, and `TableStorage` is
//! `pub(super)`. The consumer of a persisted model is the `nemo` crate, which is
//! a different crate, so an encoding that were itself crate-private would be
//! unusable for the thing it exists to enable.
//!
//! An integration test is compiled as its own crate, so if this file builds, the
//! path is genuinely public. A unit test inside the crate could not tell.

use nemo_physical::tabular::trie::{
    Trie,
    storage::{FORMAT_VERSION, MAGIC, TrieStorageError},
};

#[test]
fn encoding_is_reachable_from_another_crate() {
    assert_eq!(MAGIC.len(), 8, "magic is a fixed-width 8-byte header");
    assert!(FORMAT_VERSION >= 1, "layout version starts at 1");
}

#[test]
fn decode_rejects_foreign_bytes_across_the_crate_boundary() {
    // Also checks that the error type is nameable externally, not just the
    // function -- a caller has to be able to match on why a load failed in order
    // to decide between recomputing and giving up.
    let error = Trie::decode(b"definitely not a trie").unwrap_err();

    assert_eq!(error, TrieStorageError::NotATrie);
    assert!(!error.to_string().is_empty(), "errors are reportable");
}

#[test]
fn empty_input_is_rejected_rather_than_decoded_as_an_empty_trie() {
    // A zero-length file is the shape a crashed writer leaves behind, and
    // decoding it as a valid empty trie would turn a partial write into silently
    // missing data.
    assert!(matches!(
        Trie::decode(&[]).unwrap_err(),
        TrieStorageError::Truncated { .. }
    ));
}
