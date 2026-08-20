//! Guards that a dictionary snapshot is usable from *outside* `nemo-physical`.
//!
//! Same reasoning as `trie_storage.rs`: an encoded trie stores dictionary-relative
//! ids, so a consumer in another crate needs both halves to render anything. An
//! integration test compiles as its own crate, so it fails if either path stops
//! being reachable.

use nemo_physical::{
    datavalues::AnyDataValue,
    dictionary::{
        DvDict,
        meta_dv_dict::MetaDvDictionary,
        storage::{DictionarySnapshot, DictionaryStorageError, FORMAT_VERSION, MAGIC},
    },
};

#[test]
fn snapshot_is_reachable_from_another_crate() {
    assert_eq!(MAGIC.len(), 8, "magic is a fixed-width 8-byte header");
    assert!(FORMAT_VERSION >= 1, "layout version starts at 1");
}

#[test]
fn capture_encode_and_resolve_across_the_crate_boundary() {
    let mut dictionary = MetaDvDictionary::new();
    let iri = AnyDataValue::new_iri("http://example.org/thing".to_string());
    let id = dictionary.add_datavalue(iri.clone()).value();

    let bytes = DictionarySnapshot::capture(&dictionary, [id]).encode();

    // The point of the whole exercise: resolving an id with no dictionary in
    // scope, only the bytes.
    let decoded = DictionarySnapshot::decode(&bytes).expect("snapshot should decode");
    assert_eq!(decoded.id_to_datavalue(id), Some(iri));
}

#[test]
fn decode_rejects_foreign_bytes() {
    let error = DictionarySnapshot::decode(b"definitely not a dictionary").unwrap_err();

    assert_eq!(error, DictionaryStorageError::NotADictionary);
    assert!(!error.to_string().is_empty(), "errors are reportable");
}

#[test]
fn empty_input_is_rejected_rather_than_decoded_as_an_empty_snapshot() {
    // A zero-length file is what a crashed writer leaves behind. Decoding it as
    // a valid empty snapshot would turn a partial write into ids that silently
    // resolve to nothing.
    assert!(matches!(
        DictionarySnapshot::decode(&[]).unwrap_err(),
        DictionaryStorageError::Truncated { .. }
    ));
}
