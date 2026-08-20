//! On-disk layout for a materialized model.
//!
//! # What a persisted model has to contain
//!
//! Explaining a fact needs more than the final relations. Provenance *is* the
//! per-step subtable layout: `find_table_row` walks the `(step, table)` pairs a
//! predicate was built from and returns the step, and `rule_history[step]` then
//! names the rule that fired. So a store that only held merged relations could
//! reproduce the model but not explain it.
//!
//! A store therefore holds three things:
//!
//! - one encoded trie per `(predicate, step)` subtable,
//! - the `id -> datavalue` snapshot the tries' storage ids refer to,
//! - a manifest tying those together with `rule_history` and a cache key.
//!
//! # Why table files are numbered rather than named
//!
//! Predicate names come from user programs and are not constrained to anything a
//! filesystem accepts. Numbering the files and recording the mapping in the
//! manifest avoids escaping, case-insensitive collisions, and length limits
//! altogether.
//!
//! # Why the cache key stores program text rather than a hash
//!
//! The key decides whether a store may be reused, so a collision means loading a
//! model belonging to a different program: silently wrong answers. Nothing
//! suitable was available to hash with — `std::hash::DefaultHasher` is explicitly
//! not stable across Rust releases, so persisting it would rot, and the hashers
//! already in the dependency tree are not collision-resistant. Programs are
//! kilobytes, so the canonical text is stored verbatim and compared exactly.
//! That has no collision risk at all, and a mismatch can be diffed rather than
//! merely reported.
//!
//! Imports are the weaker part: they are fingerprinted by length and modification
//! time, which is the usual build-system compromise and carries the usual
//! caveat, that a change preserving both goes unnoticed. Content hashing is the
//! strict version and waits on choosing a hash.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use nemo_physical::{
    dictionary::storage::{
        DictionaryRestoreError, DictionarySnapshot, DictionaryStorageError,
        FORMAT_VERSION as DICTIONARY_FORMAT_VERSION,
    },
    tabular::trie::{
        Trie,
        storage::{FORMAT_VERSION as TRIE_FORMAT_VERSION, TrieStorageError},
    },
};
use serde::{Deserialize, Serialize};

/// Layout version of the store directory itself, independent of the trie and
/// dictionary encodings it contains.
pub const STORE_LAYOUT_VERSION: u32 = 1;

/// File holding the [ModelManifest].
const MANIFEST_FILE: &str = "manifest.json";
/// File holding the encoded [DictionarySnapshot].
const DICTIONARY_FILE: &str = "dictionary.bin";
/// Directory holding the encoded subtables.
const TABLES_DIR: &str = "tables";

/// Something went wrong reading or writing a model store.
#[derive(Debug, thiserror::Error)]
pub enum ModelStoreError {
    /// The filesystem operation failed.
    #[error("model store I/O error at {path}: {source}")]
    Io {
        /// Path being operated on.
        path: PathBuf,
        /// Underlying error.
        source: std::io::Error,
    },
    /// The manifest could not be parsed.
    #[error("model store manifest at {path} is malformed: {source}")]
    MalformedManifest {
        /// Path of the manifest.
        path: PathBuf,
        /// Underlying error.
        source: serde_json::Error,
    },
    /// The store was written by a different directory layout version.
    #[error("model store has layout version {found}, this build reads {expected}")]
    LayoutVersionMismatch {
        /// Version found in the store.
        found: u32,
        /// Version this build writes and reads.
        expected: u32,
    },
    /// A trie could not be decoded.
    #[error("subtable {path} could not be decoded: {source}")]
    MalformedTable {
        /// Path of the table file.
        path: PathBuf,
        /// Underlying error.
        source: TrieStorageError,
    },
    /// The dictionary snapshot could not be decoded.
    #[error("dictionary snapshot could not be decoded: {0}")]
    MalformedDictionary(DictionaryStorageError),
    /// The dictionary snapshot could not be replayed into a dictionary.
    ///
    /// Means the stored tables can no longer be interpreted, since their storage
    /// ids would refer to a different mapping.
    #[error("dictionary snapshot could not be replayed: {0}")]
    DictionaryRestore(DictionaryRestoreError),
    /// The target directory already exists.
    #[error("cannot create a model store at {path}: it already exists")]
    AlreadyExists {
        /// Path that was requested.
        path: PathBuf,
    },
    /// The manifest refers to a table file that is not present.
    #[error("manifest refers to missing subtable file {file}")]
    MissingTableFile {
        /// Name recorded in the manifest.
        file: String,
    },
}

/// Fingerprint of one imported resource.
///
/// Length and modification time rather than content: see the module docs for why,
/// and for the caveat that comes with it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportFingerprint {
    /// Resource as written in the program, after parameter substitution.
    pub resource: String,
    /// Length in bytes, or `None` if it could not be determined.
    pub bytes: Option<u64>,
    /// Modification time in nanoseconds since the Unix epoch, or `None` if
    /// unavailable — as it is for non-file resources.
    pub modified_unix_nanos: Option<u128>,
}

impl ImportFingerprint {
    /// Fingerprint a local file.
    ///
    /// Missing metadata is recorded as `None` rather than failing: a resource
    /// that cannot be fingerprinted still has to appear in the key, and a `None`
    /// will simply never match a `Some`.
    pub fn of_file(resource: impl Into<String>, path: &Path) -> Self {
        let metadata = fs::metadata(path).ok();

        Self {
            resource: resource.into(),
            bytes: metadata.as_ref().map(fs::Metadata::len),
            modified_unix_nanos: metadata
                .as_ref()
                .and_then(|metadata| metadata.modified().ok())
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|duration| duration.as_nanos()),
        }
    }
}

/// Everything that has to agree before a store may be reused.
///
/// Compared by equality, so any field differing invalidates the store. That is
/// the intended behaviour: a stale hit is silently wrong, a spurious miss merely
/// costs a recomputation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheKey {
    /// Layout version of the store directory.
    pub store_layout_version: u32,
    /// Layout version of the trie encoding.
    pub trie_format_version: u32,
    /// Layout version of the dictionary encoding.
    pub dictionary_format_version: u32,
    /// Canonical text of the normalized program, stored verbatim.
    ///
    /// The normalized program rather than the source file: two files differing
    /// only in whitespace or comments normalize identically and should share a
    /// store, and a transformation that changes the program must not.
    pub program: String,
    /// Global parameter bindings.
    ///
    /// A [BTreeMap] so serialization is ordered and two equal bindings always
    /// produce equal manifests.
    pub parameters: BTreeMap<String, String>,
    /// Fingerprints of the imported resources.
    pub imports: Vec<ImportFingerprint>,
}

impl CacheKey {
    /// Build a key for the current build's formats.
    pub fn new(
        program: impl Into<String>,
        parameters: BTreeMap<String, String>,
        imports: Vec<ImportFingerprint>,
    ) -> Self {
        Self {
            store_layout_version: STORE_LAYOUT_VERSION,
            trie_format_version: TRIE_FORMAT_VERSION,
            dictionary_format_version: DICTIONARY_FORMAT_VERSION,
            program: program.into(),
            parameters,
            imports,
        }
    }
}

/// One `(step, file)` subtable of a predicate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubtableEntry {
    /// Execution step this subtable was derived in.
    pub step: usize,
    /// Name of the encoded trie inside the tables directory.
    pub file: String,
}

/// A predicate and the subtables it was built from.
///
/// Subtables are kept per step rather than merged, because that layout is what
/// provenance is recovered from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredicateEntry {
    /// Predicate name.
    pub predicate: String,
    /// Arity.
    pub arity: usize,
    /// Subtables, ascending by step.
    pub subtables: Vec<SubtableEntry>,
}

/// The manifest of a model store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Layout version of the store directory.
    pub layout_version: u32,
    /// What has to agree for this store to be reusable.
    pub cache_key: CacheKey,
    /// Predicates and their subtables.
    pub predicates: Vec<PredicateEntry>,
    /// Rule applied at each step, indexed by step.
    ///
    /// Half of the provenance chain: a subtable gives the step, this gives the
    /// rule. Useless without the per-step subtables, and vice versa.
    pub rule_history: Vec<usize>,
}

/// Builds a model store.
///
/// Writes into a staging directory and renames it into place on
/// [ModelStoreWriter::finish], so an interrupted write leaves a staging
/// directory rather than a store that looks complete but is not.
#[derive(Debug)]
pub struct ModelStoreWriter {
    /// Final location.
    target: PathBuf,
    /// Where content is written until `finish`.
    staging: PathBuf,
    /// Predicates recorded so far.
    predicates: Vec<PredicateEntry>,
    /// Number of table files written, which also names the next one.
    tables_written: usize,
}

impl ModelStoreWriter {
    /// Start a store at `target`, which must not exist.
    pub fn create(target: impl AsRef<Path>) -> Result<Self, ModelStoreError> {
        let target = target.as_ref().to_path_buf();

        if target.exists() {
            return Err(ModelStoreError::AlreadyExists { path: target });
        }

        // Sibling of the target so the final rename stays within one filesystem.
        // Named after the process so two concurrent writers do not collide;
        // whichever renames second fails, which is the correct outcome for a
        // cache entry that is already present.
        let staging = target.with_file_name(format!(
            "{}.staging.{}",
            target
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| "model-store".to_string()),
            std::process::id()
        ));

        if staging.exists() {
            remove_dir(&staging)?;
        }
        create_dir_all(&staging.join(TABLES_DIR))?;

        Ok(Self {
            target,
            staging,
            predicates: Vec::new(),
            tables_written: 0,
        })
    }

    /// Record one subtable of a predicate.
    ///
    /// Call once per `(predicate, step)`, in ascending step order per predicate.
    pub fn write_table(
        &mut self,
        predicate: &str,
        arity: usize,
        step: usize,
        trie: &Trie,
    ) -> Result<(), ModelStoreError> {
        self.write_table_bytes(predicate, arity, step, &trie.encode())
    }

    /// Record one subtable from already-encoded bytes.
    ///
    /// The seam `write_table` goes through, so the layout, manifest, and
    /// atomicity logic can be exercised without constructing a [Trie] — whose
    /// row-level constructors are internal to `nemo-physical`.
    pub(crate) fn write_table_bytes(
        &mut self,
        predicate: &str,
        arity: usize,
        step: usize,
        bytes: &[u8],
    ) -> Result<(), ModelStoreError> {
        let file = format!("t{:06}.bin", self.tables_written);
        write_file(&self.staging.join(TABLES_DIR).join(&file), bytes)?;
        self.tables_written += 1;

        let entry = SubtableEntry { step, file };

        match self
            .predicates
            .iter_mut()
            .find(|entry| entry.predicate == predicate)
        {
            Some(existing) => existing.subtables.push(entry),
            None => self.predicates.push(PredicateEntry {
                predicate: predicate.to_string(),
                arity,
                subtables: vec![entry],
            }),
        }

        Ok(())
    }

    /// Record the dictionary the tries' storage ids refer to.
    pub fn write_dictionary(
        &mut self,
        dictionary: &DictionarySnapshot,
    ) -> Result<(), ModelStoreError> {
        write_file(&self.staging.join(DICTIONARY_FILE), &dictionary.encode())
    }

    /// Write the manifest and move the store into place.
    pub fn finish(
        mut self,
        cache_key: CacheKey,
        rule_history: Vec<usize>,
    ) -> Result<PathBuf, ModelStoreError> {
        for entry in &mut self.predicates {
            entry.subtables.sort_by_key(|subtable| subtable.step);
        }

        let manifest = ModelManifest {
            layout_version: STORE_LAYOUT_VERSION,
            cache_key,
            predicates: std::mem::take(&mut self.predicates),
            rule_history,
        };

        let encoded = serde_json::to_vec_pretty(&manifest)
            .expect("manifest is composed of plainly serializable types");
        write_file(&self.staging.join(MANIFEST_FILE), &encoded)?;

        // The manifest is written last and the rename is the only publishing
        // step, so a store either has a manifest and is complete, or does not
        // exist.
        fs::rename(&self.staging, &self.target).map_err(|source| ModelStoreError::Io {
            path: self.target.clone(),
            source,
        })?;

        Ok(self.target)
    }

    /// Abandon the store, removing anything already written.
    pub fn abandon(self) -> Result<(), ModelStoreError> {
        remove_dir(&self.staging)
    }
}

/// A model store on disk, opened for reading.
#[derive(Debug)]
pub struct ModelStore {
    /// Location of the store.
    path: PathBuf,
    /// Manifest read at open time.
    manifest: ModelManifest,
}

impl ModelStore {
    /// Open a store and read its manifest.
    ///
    /// Checks the directory layout version but not the cache key: whether a store
    /// is *applicable* is the caller's question, and it needs the manifest in
    /// order to answer it.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ModelStoreError> {
        let path = path.as_ref().to_path_buf();
        let manifest_path = path.join(MANIFEST_FILE);

        let bytes = fs::read(&manifest_path).map_err(|source| ModelStoreError::Io {
            path: manifest_path.clone(),
            source,
        })?;

        let manifest: ModelManifest = serde_json::from_slice(&bytes).map_err(|source| {
            ModelStoreError::MalformedManifest {
                path: manifest_path,
                source,
            }
        })?;

        if manifest.layout_version != STORE_LAYOUT_VERSION {
            return Err(ModelStoreError::LayoutVersionMismatch {
                found: manifest.layout_version,
                expected: STORE_LAYOUT_VERSION,
            });
        }

        Ok(Self { path, manifest })
    }

    /// The manifest.
    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    /// Whether this store was built for the given key.
    pub fn is_applicable(&self, key: &CacheKey) -> bool {
        &self.manifest.cache_key == key
    }

    /// Steps at which the given predicate has subtables, ascending.
    pub fn steps_for(&self, predicate: &str) -> Vec<usize> {
        self.predicate(predicate)
            .map(|entry| {
                entry
                    .subtables
                    .iter()
                    .map(|subtable| subtable.step)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The recorded entry for a predicate, if it has one.
    pub fn predicate(&self, predicate: &str) -> Option<&PredicateEntry> {
        self.manifest
            .predicates
            .iter()
            .find(|entry| entry.predicate == predicate)
    }

    /// Load the dictionary snapshot.
    pub fn dictionary(&self) -> Result<DictionarySnapshot, ModelStoreError> {
        let path = self.path.join(DICTIONARY_FILE);
        let bytes = fs::read(&path).map_err(|source| ModelStoreError::Io {
            path: path.clone(),
            source,
        })?;

        DictionarySnapshot::decode(&bytes).map_err(ModelStoreError::MalformedDictionary)
    }

    /// Load one subtable, or `None` if the predicate has nothing at that step.
    pub fn load_table(
        &self,
        predicate: &str,
        step: usize,
    ) -> Result<Option<Trie>, ModelStoreError> {
        let Some(bytes) = self.load_table_bytes(predicate, step)? else {
            return Ok(None);
        };

        let path = self.path.join(TABLES_DIR);
        Trie::decode(&bytes)
            .map(Some)
            .map_err(|source| ModelStoreError::MalformedTable { path, source })
    }

    /// Load the raw bytes of one subtable.
    pub(crate) fn load_table_bytes(
        &self,
        predicate: &str,
        step: usize,
    ) -> Result<Option<Vec<u8>>, ModelStoreError> {
        let Some(entry) = self.predicate(predicate) else {
            return Ok(None);
        };

        let Some(subtable) = entry
            .subtables
            .iter()
            .find(|subtable| subtable.step == step)
        else {
            return Ok(None);
        };

        let path = self.path.join(TABLES_DIR).join(&subtable.file);
        if !path.exists() {
            return Err(ModelStoreError::MissingTableFile {
                file: subtable.file.clone(),
            });
        }

        fs::read(&path)
            .map(Some)
            .map_err(|source| ModelStoreError::Io { path, source })
    }
}

fn create_dir_all(path: &Path) -> Result<(), ModelStoreError> {
    fs::create_dir_all(path).map_err(|source| ModelStoreError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn remove_dir(path: &Path) -> Result<(), ModelStoreError> {
    fs::remove_dir_all(path).map_err(|source| ModelStoreError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), ModelStoreError> {
    fs::write(path, bytes).map_err(|source| ModelStoreError::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod test {
    use std::collections::BTreeMap;

    use assert_fs::TempDir;
    use nemo_physical::{
        datavalues::AnyDataValue,
        dictionary::{DvDict, meta_dv_dict::MetaDvDictionary, storage::DictionarySnapshot},
    };

    use super::{
        CacheKey, ImportFingerprint, ModelStore, ModelStoreError, ModelStoreWriter,
        STORE_LAYOUT_VERSION,
    };

    fn key(program: &str) -> CacheKey {
        CacheKey::new(program, BTreeMap::new(), Vec::new())
    }

    /// A snapshot with one IRI in it, plus the id it was given.
    fn dictionary_with_one_iri() -> (DictionarySnapshot, usize) {
        let mut dictionary = MetaDvDictionary::new();
        let id = dictionary
            .add_datavalue(AnyDataValue::new_iri("http://example.org/a".to_string()))
            .value();

        (DictionarySnapshot::capture(&dictionary, [id]), id)
    }

    #[test]
    fn round_trip_manifest_and_contents() {
        let temp = TempDir::new().expect("temp dir");
        let path = temp.path().join("store");
        let (dictionary, id) = dictionary_with_one_iri();

        let mut writer = ModelStoreWriter::create(&path).expect("create");
        writer
            .write_table_bytes("mapping", 3, 1, b"first subtable")
            .expect("write");
        writer
            .write_table_bytes("mapping", 3, 4, b"second subtable")
            .expect("write");
        writer
            .write_table_bytes("inferredMapping", 3, 7, b"other predicate")
            .expect("write");
        writer.write_dictionary(&dictionary).expect("dictionary");
        writer
            .finish(key("mapping(?a) :- b(?a) ."), vec![usize::MAX, 0, 1, 0, 2])
            .expect("finish");

        let store = ModelStore::open(&path).expect("open");

        assert_eq!(store.manifest().layout_version, STORE_LAYOUT_VERSION);
        assert_eq!(store.manifest().rule_history, vec![usize::MAX, 0, 1, 0, 2]);
        assert_eq!(store.steps_for("mapping"), vec![1, 4]);
        assert_eq!(store.steps_for("inferredMapping"), vec![7]);
        assert_eq!(store.steps_for("absent"), Vec::<usize>::new());
        assert_eq!(store.predicate("mapping").expect("recorded").arity, 3);

        assert_eq!(
            store.load_table_bytes("mapping", 4).expect("load"),
            Some(b"second subtable".to_vec())
        );
        assert_eq!(store.load_table_bytes("mapping", 2).expect("load"), None);
        assert_eq!(store.load_table_bytes("absent", 1).expect("load"), None);

        assert_eq!(
            store
                .dictionary()
                .expect("dictionary")
                .id_to_datavalue(id)
                .expect("captured"),
            AnyDataValue::new_iri("http://example.org/a".to_string())
        );
    }

    #[test]
    fn subtables_are_ordered_by_step_regardless_of_write_order() {
        // Provenance lookups scan a predicate's subtables, so the manifest is
        // normalized rather than trusting the caller's order.
        let temp = TempDir::new().expect("temp dir");
        let path = temp.path().join("store");

        let mut writer = ModelStoreWriter::create(&path).expect("create");
        for step in [9, 2, 5, 1] {
            writer.write_table_bytes("p", 1, step, b"x").expect("write");
        }
        writer.finish(key("p"), Vec::new()).expect("finish");

        let store = ModelStore::open(&path).expect("open");
        assert_eq!(store.steps_for("p"), vec![1, 2, 5, 9]);
    }

    #[test]
    fn a_store_is_published_only_by_finish() {
        let temp = TempDir::new().expect("temp dir");
        let path = temp.path().join("store");

        let mut writer = ModelStoreWriter::create(&path).expect("create");
        writer.write_table_bytes("p", 1, 1, b"x").expect("write");

        // An interrupted write must not leave something that opens successfully:
        // the rename is the only publishing step.
        assert!(!path.exists(), "target must not exist before finish");
        assert!(ModelStore::open(&path).is_err());

        writer.finish(key("p"), Vec::new()).expect("finish");
        assert!(path.exists());
        assert!(ModelStore::open(&path).is_ok());
    }

    #[test]
    fn abandoning_removes_the_staging_directory() {
        let temp = TempDir::new().expect("temp dir");
        let path = temp.path().join("store");

        let mut writer = ModelStoreWriter::create(&path).expect("create");
        writer.write_table_bytes("p", 1, 1, b"x").expect("write");
        writer.abandon().expect("abandon");

        assert!(!path.exists());
        let leftovers: Vec<_> = std::fs::read_dir(temp.path())
            .expect("read temp dir")
            .filter_map(Result::ok)
            .collect();
        assert!(leftovers.is_empty(), "staging directory should be gone");
    }

    #[test]
    fn refuses_to_overwrite_an_existing_store() {
        let temp = TempDir::new().expect("temp dir");
        let path = temp.path().join("store");

        ModelStoreWriter::create(&path)
            .expect("create")
            .finish(key("p"), Vec::new())
            .expect("finish");

        assert!(matches!(
            ModelStoreWriter::create(&path),
            Err(ModelStoreError::AlreadyExists { .. })
        ));
    }

    #[test]
    fn applicability_is_decided_by_the_whole_key() {
        let temp = TempDir::new().expect("temp dir");
        let path = temp.path().join("store");

        let mut parameters = BTreeMap::new();
        parameters.insert("importfile".to_string(), "\"data.nt\"".to_string());
        let original = CacheKey::new(
            "p(?a) :- q(?a) .",
            parameters.clone(),
            vec![ImportFingerprint {
                resource: "data.nt".to_string(),
                bytes: Some(1024),
                modified_unix_nanos: Some(42),
            }],
        );

        ModelStoreWriter::create(&path)
            .expect("create")
            .finish(original.clone(), Vec::new())
            .expect("finish");

        let store = ModelStore::open(&path).expect("open");
        assert!(store.is_applicable(&original));

        // A different program.
        let mut other = original.clone();
        other.program = "p(?a) :- r(?a) .".to_string();
        assert!(!store.is_applicable(&other));

        // A different parameter binding, which changes which file is imported
        // even though the program text is identical.
        let mut other = original.clone();
        other
            .parameters
            .insert("importfile".to_string(), "\"elsewhere.nt\"".to_string());
        assert!(!store.is_applicable(&other));

        // The same import, changed underneath us.
        let mut other = original.clone();
        other.imports[0].bytes = Some(2048);
        assert!(!store.is_applicable(&other));

        // A format version bump must invalidate too, or old bytes get read by a
        // build that no longer understands them.
        let mut other = original.clone();
        other.trie_format_version += 1;
        assert!(!store.is_applicable(&other));
    }

    #[test]
    fn rejects_a_different_layout_version() {
        let temp = TempDir::new().expect("temp dir");
        let path = temp.path().join("store");

        ModelStoreWriter::create(&path)
            .expect("create")
            .finish(key("p"), Vec::new())
            .expect("finish");

        let manifest_path = path.join("manifest.json");
        let manifest = std::fs::read_to_string(&manifest_path).expect("read manifest");
        std::fs::write(
            &manifest_path,
            manifest.replace(
                &format!("\"layout_version\": {STORE_LAYOUT_VERSION}"),
                &format!("\"layout_version\": {}", STORE_LAYOUT_VERSION + 1),
            ),
        )
        .expect("write manifest");

        assert!(matches!(
            ModelStore::open(&path),
            Err(ModelStoreError::LayoutVersionMismatch { .. })
        ));
    }

    #[test]
    fn rejects_a_missing_or_malformed_manifest() {
        let temp = TempDir::new().expect("temp dir");
        let path = temp.path().join("store");

        assert!(matches!(
            ModelStore::open(&path),
            Err(ModelStoreError::Io { .. })
        ));

        ModelStoreWriter::create(&path)
            .expect("create")
            .finish(key("p"), Vec::new())
            .expect("finish");
        std::fs::write(path.join("manifest.json"), b"{not json").expect("clobber");

        assert!(matches!(
            ModelStore::open(&path),
            Err(ModelStoreError::MalformedManifest { .. })
        ));
    }

    #[test]
    fn reports_a_table_file_the_manifest_promised() {
        let temp = TempDir::new().expect("temp dir");
        let path = temp.path().join("store");

        let mut writer = ModelStoreWriter::create(&path).expect("create");
        writer.write_table_bytes("p", 1, 1, b"x").expect("write");
        writer.finish(key("p"), Vec::new()).expect("finish");

        std::fs::remove_file(path.join("tables").join("t000000.bin")).expect("remove");

        assert!(matches!(
            ModelStore::open(&path)
                .expect("open")
                .load_table_bytes("p", 1),
            Err(ModelStoreError::MissingTableFile { .. })
        ));
    }

    #[test]
    fn predicate_names_needing_escaping_are_stored_safely() {
        // Table files are numbered precisely so predicate names never reach the
        // filesystem. A name with separators and reserved characters must round
        // trip regardless.
        let temp = TempDir::new().expect("temp dir");
        let path = temp.path().join("store");
        let awkward = "../../etc/passwd:<http://example.org/p>";

        let mut writer = ModelStoreWriter::create(&path).expect("create");
        writer
            .write_table_bytes(awkward, 2, 3, b"payload")
            .expect("write");
        writer.finish(key("p"), Vec::new()).expect("finish");

        let store = ModelStore::open(&path).expect("open");
        assert_eq!(store.steps_for(awkward), vec![3]);
        assert_eq!(
            store.load_table_bytes(awkward, 3).expect("load"),
            Some(b"payload".to_vec())
        );
    }

    #[test]
    fn manifest_serialization_is_stable() {
        // Two stores built from equal inputs must produce byte-identical
        // manifests, or the store cannot be content-addressed.
        let temp = TempDir::new().expect("temp dir");

        let mut parameters = BTreeMap::new();
        parameters.insert("b".to_string(), "2".to_string());
        parameters.insert("a".to_string(), "1".to_string());

        let mut manifests = Vec::new();
        for name in ["first", "second"] {
            let path = temp.path().join(name);
            let mut writer = ModelStoreWriter::create(&path).expect("create");
            writer.write_table_bytes("p", 1, 1, b"x").expect("write");
            writer
                .finish(
                    CacheKey::new("p(?a) :- q(?a) .", parameters.clone(), Vec::new()),
                    vec![usize::MAX, 0],
                )
                .expect("finish");

            manifests.push(std::fs::read(path.join("manifest.json")).expect("read"));
        }

        assert_eq!(manifests[0], manifests[1]);
    }
}
