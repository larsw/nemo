//! Persistence for the `id -> datavalue` direction of a dictionary.
//!
//! # Why only one direction, and why a subset
//!
//! A [Trie] stores dictionary-relative integer ids, so an encoded trie is
//! meaningless without the dictionary that produced them. But a consumer that
//! reads a persisted model does not need a whole dictionary:
//!
//! - **Only `id -> datavalue`.** Tables are read and joined entirely in
//!   storage-id space; the datavalue is needed only to render a result. The
//!   reverse direction, resolving a datavalue typed by a user, would need a
//!   persistent index over keys, which is a separate and much larger piece of
//!   work.
//! - **Only the ids actually used.** [MetaDvDictionary] hands out ids from
//!   2^24-address blocks per sub-dictionary, so the id space is sparse and there
//!   is no way to enumerate it: [DvDict] exposes `len` but no iterator, and
//!   `id_to_datavalue` returns `None` for gaps and for merely-marked values.
//!   Capturing the ids that appear in the tables sidesteps that entirely, and
//!   bounds the snapshot by the data rather than by the dictionary.
//!
//! # Only some values are dictionary-backed
//!
//! A dictionary holds only values that need an id in order to become an
//! `Id32`/`Id64` storage value. Values a storage value can carry directly are
//! rejected outright and given [NONEXISTING_ID_MARK]:
//!
//! | value | has an id |
//! |---|---|
//! | IRI, plain string, language-tagged string | yes |
//! | boolean, integer too large for `Int64`, [ValueDomain::Other] | yes |
//! | integer fitting `Int64`, [ValueDomain::Float], [ValueDomain::Double] | no |
//!
//! So a snapshot only ever needs to cover the `Id32` and `Id64` columns of a
//! trie; the numeric columns are self-describing. Unresolvable ids are skipped
//! by [DictionarySnapshot::capture] rather than reported, which makes it safe to
//! feed it every id found in a table without pre-filtering by storage type.
//!
//! # Encoding
//!
//! A sorted `(id, offset)` index followed by a blob of encoded values. Lookup is
//! a binary search on the index. Values are decoded eagerly on load; as with
//! [crate::tabular::trie::storage], a zero-copy reader is a later change and the
//! layout is versioned so it can arrive without migration.

use std::fmt::Display;

use crate::{
    datavalues::{
        AnyDataValue, DataValue, IriDataValue, MapDataValue, NullDataValue, TupleDataValue,
        ValueDomain,
    },
    dictionary::DvDict,
};

/// Identifies the encoding.
pub const MAGIC: &[u8; 8] = b"NMODICT\0";

/// Layout version. Bump on any layout change; consumers include it in a cache
/// key, so an increment invalidates rather than migrates.
pub const FORMAT_VERSION: u32 = 1;

/// A literal reconstructible from its lexical form and datatype IRI. Covers
/// every numeric domain, booleans, and [ValueDomain::Other].
const TAG_LITERAL: u8 = 0;
/// A plain string. Given its own tag rather than folded into [TAG_LITERAL]
/// because strings dominate real dictionaries and the shorter form is worth it.
const TAG_STRING: u8 = 1;
/// An IRI.
const TAG_IRI: u8 = 2;
/// A language-tagged string: value and tag.
const TAG_LANG_STRING: u8 = 3;
/// A named null, which is nothing but its id.
const TAG_NULL: u8 = 4;
/// A tuple: optional label followed by its elements.
const TAG_TUPLE: u8 = 5;
/// A map: key/value pairs.
const TAG_MAP: u8 = 6;

/// Something went wrong while decoding a dictionary snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DictionaryStorageError {
    /// Input does not begin with [MAGIC].
    NotADictionary,
    /// Input was written by a different layout version.
    VersionMismatch {
        /// Version found in the input.
        found: u32,
        /// Version this build writes and reads.
        expected: u32,
    },
    /// Input ended in the middle of a value.
    Truncated {
        /// Byte offset at which more input was required.
        offset: usize,
        /// Number of bytes required there.
        needed: usize,
    },
    /// A discriminant byte held a value this build does not know.
    UnknownValueTag(u8),
    /// A string field was not valid UTF-8.
    InvalidUtf8 {
        /// Byte offset of the offending field.
        offset: usize,
    },
    /// A lexical form and datatype IRI did not reconstruct a datavalue.
    ///
    /// Should not arise from a snapshot this build wrote, since the pair came
    /// from a datavalue that already existed.
    MalformedLiteral {
        /// The lexical form that could not be interpreted.
        lexical: String,
        /// The datatype IRI it was paired with.
        datatype: String,
    },
    /// A value or length did not fit in `usize` on this platform.
    ValueOutOfRange(u64),
    /// The index was not sorted by id, so binary search would be unsound.
    IndexNotSorted,
}

impl Display for DictionaryStorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotADictionary => write!(f, "input is not an encoded dictionary snapshot"),
            Self::VersionMismatch { found, expected } => write!(
                f,
                "dictionary snapshot has layout version {found}, this build reads {expected}"
            ),
            Self::Truncated { offset, needed } => write!(
                f,
                "dictionary snapshot ends prematurely: needed {needed} more byte(s) at offset {offset}"
            ),
            Self::UnknownValueTag(tag) => write!(f, "unknown datavalue tag {tag}"),
            Self::InvalidUtf8 { offset } => {
                write!(f, "string field at offset {offset} is not valid UTF-8")
            }
            Self::MalformedLiteral { lexical, datatype } => {
                write!(f, "cannot rebuild literal {lexical:?} of type {datatype:?}")
            }
            Self::ValueOutOfRange(value) => write!(
                f,
                "encoded value {value} does not fit in a pointer on this platform"
            ),
            Self::IndexNotSorted => write!(f, "dictionary snapshot index is not sorted by id"),
        }
    }
}

impl std::error::Error for DictionaryStorageError {}

/// The `id -> datavalue` mapping for a fixed set of ids.
///
/// Captured from a live dictionary, encodable to bytes, and queryable after
/// decoding without a dictionary being present at all.
#[derive(Debug, Clone, Default)]
pub struct DictionarySnapshot {
    /// `(id, value)` pairs, sorted by id so lookup is a binary search.
    entries: Vec<(usize, AnyDataValue)>,
}

impl DictionarySnapshot {
    /// Capture the given ids from a dictionary.
    ///
    /// Ids the dictionary cannot resolve are skipped rather than reported: gaps
    /// in the id space and merely-marked values both legitimately yield `None`,
    /// and a caller collecting ids out of tables has no way to tell them apart
    /// in advance.
    ///
    /// Duplicate ids collapse.
    pub fn capture<Dict, Ids>(dictionary: &Dict, ids: Ids) -> Self
    where
        Dict: DvDict + ?Sized,
        Ids: IntoIterator<Item = usize>,
    {
        let mut entries: Vec<(usize, AnyDataValue)> = ids
            .into_iter()
            .filter_map(|id| dictionary.id_to_datavalue(id).map(|value| (id, value)))
            .collect();

        entries.sort_unstable_by_key(|(id, _)| *id);
        entries.dedup_by_key(|(id, _)| *id);

        Self { entries }
    }

    /// Number of ids in this snapshot.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether this snapshot holds no ids.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Resolve an id, or `None` if it was not captured.
    pub fn id_to_datavalue(&self, id: usize) -> Option<AnyDataValue> {
        self.entries
            .binary_search_by_key(&id, |(entry_id, _)| *entry_id)
            .ok()
            .map(|index| self.entries[index].1.clone())
    }

    /// Iterate over the captured pairs in ascending id order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &AnyDataValue)> {
        self.entries.iter().map(|(id, value)| (*id, value))
    }

    /// Encode this snapshot into a byte buffer.
    pub fn encode(&self) -> Vec<u8> {
        // Values are encoded first so their offsets are known before the index
        // is written.
        let mut values = Vec::new();
        let mut offsets = Vec::with_capacity(self.entries.len());

        for (_, value) in &self.entries {
            offsets.push(values.len() as u64);
            encode_value(value, &mut values);
        }

        let mut bytes = Vec::with_capacity(values.len() + offsets.len() * 16 + 32);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes()); // reserved, keeps alignment
        bytes.extend_from_slice(&(self.entries.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());

        for ((id, _), offset) in self.entries.iter().zip(&offsets) {
            bytes.extend_from_slice(&(*id as u64).to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
        }

        bytes.extend_from_slice(&values);
        bytes
    }

    /// Decode a snapshot produced by [DictionarySnapshot::encode].
    pub fn decode(bytes: &[u8]) -> Result<Self, DictionaryStorageError> {
        let mut reader = Cursor::new(bytes);

        if reader.take(MAGIC.len())? != MAGIC {
            return Err(DictionaryStorageError::NotADictionary);
        }

        let version = reader.u32()?;
        if version != FORMAT_VERSION {
            return Err(DictionaryStorageError::VersionMismatch {
                found: version,
                expected: FORMAT_VERSION,
            });
        }

        let _reserved = reader.u32()?;
        let count = reader.length()?;
        let values_length = reader.length()?;

        let mut index = Vec::with_capacity(count);
        for _ in 0..count {
            let id = reader.length()?;
            let offset = reader.length()?;
            index.push((id, offset));
        }

        let values = reader.take(values_length)?;

        let mut entries = Vec::with_capacity(count);
        let mut previous: Option<usize> = None;
        for (id, offset) in index {
            // Sortedness is what makes binary search sound, so it is verified
            // rather than assumed: the bytes may not have been written by us.
            if previous.is_some_and(|last| last >= id) {
                return Err(DictionaryStorageError::IndexNotSorted);
            }
            previous = Some(id);

            let mut value_reader = Cursor::at(values, offset)?;
            entries.push((id, decode_value(&mut value_reader)?));
        }

        Ok(Self { entries })
    }
}

/// Cursor over encoded bytes.
struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn at(bytes: &'a [u8], offset: usize) -> Result<Self, DictionaryStorageError> {
        if offset > bytes.len() {
            return Err(DictionaryStorageError::Truncated { offset, needed: 1 });
        }
        Ok(Self { bytes, offset })
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], DictionaryStorageError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or(DictionaryStorageError::Truncated {
                offset: self.offset,
                needed: count,
            })?;

        if end > self.bytes.len() {
            return Err(DictionaryStorageError::Truncated {
                offset: self.offset,
                needed: count,
            });
        }

        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn byte(&mut self) -> Result<u8, DictionaryStorageError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, DictionaryStorageError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes(
            bytes.try_into().expect("take returned exactly 4 bytes"),
        ))
    }

    fn u64(&mut self) -> Result<u64, DictionaryStorageError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes(
            bytes.try_into().expect("take returned exactly 8 bytes"),
        ))
    }

    fn length(&mut self) -> Result<usize, DictionaryStorageError> {
        let value = self.u64()?;
        usize::try_from(value).map_err(|_| DictionaryStorageError::ValueOutOfRange(value))
    }

    fn string(&mut self) -> Result<String, DictionaryStorageError> {
        let offset = self.offset;
        let length = self.length()?;
        let bytes = self.take(length)?;

        std::str::from_utf8(bytes)
            .map(ToString::to_string)
            .map_err(|_| DictionaryStorageError::InvalidUtf8 { offset })
    }
}

/// Append a length-prefixed UTF-8 string.
fn encode_string(value: &str, out: &mut Vec<u8>) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

/// Encode a single datavalue.
///
/// The `match` is exhaustive over [ValueDomain] on purpose: a new domain must
/// fail to compile here rather than silently fall into a wrong branch.
fn encode_value(value: &AnyDataValue, out: &mut Vec<u8>) {
    match value.value_domain() {
        ValueDomain::PlainString => {
            out.push(TAG_STRING);
            encode_string(&value.to_plain_string_unchecked(), out);
        }
        ValueDomain::Iri => {
            out.push(TAG_IRI);
            encode_string(&value.to_iri_unchecked(), out);
        }
        ValueDomain::LanguageTaggedString => {
            let (text, tag) = value.to_language_tagged_string_unchecked();
            out.push(TAG_LANG_STRING);
            encode_string(&text, out);
            encode_string(&tag, out);
        }
        ValueDomain::Null => {
            // A null is its id and nothing else, so there is no lexical form to
            // preserve.
            out.push(TAG_NULL);
            out.extend_from_slice(&(value.to_null_unchecked().id() as u64).to_le_bytes());
        }
        ValueDomain::Tuple => {
            out.push(TAG_TUPLE);
            match value.label() {
                Some(label) => {
                    out.push(1);
                    encode_string(&label.to_iri_unchecked(), out);
                }
                None => out.push(0),
            }

            let length = value.len_unchecked();
            out.extend_from_slice(&(length as u64).to_le_bytes());
            for index in 0..length {
                encode_value(
                    value
                        .tuple_element(index)
                        .expect("index is below the reported length"),
                    out,
                );
            }
        }
        ValueDomain::Map => {
            out.push(TAG_MAP);
            let keys: Vec<_> = value
                .map_keys()
                .expect("value domain is Map")
                .cloned()
                .collect();

            out.extend_from_slice(&(keys.len() as u64).to_le_bytes());
            for key in &keys {
                encode_value(key, out);
                encode_value(value.map_element(key).expect("key came from map_keys"), out);
            }
        }
        // Every numeric domain, plus Boolean and Other, is reconstructible from
        // its lexical form and datatype IRI.
        ValueDomain::Float
        | ValueDomain::Double
        | ValueDomain::UnsignedLong
        | ValueDomain::NonNegativeLong
        | ValueDomain::UnsignedInt
        | ValueDomain::NonNegativeInt
        | ValueDomain::Long
        | ValueDomain::Int
        | ValueDomain::Boolean
        | ValueDomain::Other => {
            out.push(TAG_LITERAL);
            encode_string(&value.lexical_value(), out);
            encode_string(&value.datatype_iri(), out);
        }
    }
}

/// Decode a single datavalue written by [encode_value].
fn decode_value(reader: &mut Cursor<'_>) -> Result<AnyDataValue, DictionaryStorageError> {
    match reader.byte()? {
        TAG_STRING => Ok(AnyDataValue::new_plain_string(reader.string()?)),
        TAG_IRI => Ok(AnyDataValue::new_iri(reader.string()?)),
        TAG_LANG_STRING => {
            let text = reader.string()?;
            let tag = reader.string()?;
            Ok(AnyDataValue::new_language_tagged_string(text, tag))
        }
        TAG_NULL => Ok(AnyDataValue::from(NullDataValue::new(reader.length()?))),
        TAG_TUPLE => {
            let label = match reader.byte()? {
                0 => None,
                _ => Some(IriDataValue::new(reader.string()?)),
            };

            let length = reader.length()?;
            let mut elements = Vec::with_capacity(length);
            for _ in 0..length {
                elements.push(decode_value(reader)?);
            }

            Ok(AnyDataValue::from(TupleDataValue::new(label, elements)))
        }
        TAG_MAP => {
            let length = reader.length()?;
            let mut pairs = Vec::with_capacity(length);
            for _ in 0..length {
                let key = decode_value(reader)?;
                let value = decode_value(reader)?;
                pairs.push((key, value));
            }

            Ok(AnyDataValue::from(MapDataValue::new(None, pairs)))
        }
        TAG_LITERAL => {
            let lexical = reader.string()?;
            let datatype = reader.string()?;

            AnyDataValue::new_from_typed_literal(lexical.clone(), datatype.clone())
                .map_err(|_| DictionaryStorageError::MalformedLiteral { lexical, datatype })
        }
        other => Err(DictionaryStorageError::UnknownValueTag(other)),
    }
}

#[cfg(test)]
mod test {
    use crate::{
        datavalues::AnyDataValue,
        dictionary::{
            DvDict,
            meta_dv_dict::MetaDvDictionary,
            storage::{DictionarySnapshot, DictionaryStorageError, FORMAT_VERSION, MAGIC},
        },
    };

    /// Put the values into a real dictionary, snapshot the ids it assigned,
    /// round-trip the bytes, and require every value to come back.
    ///
    /// Goes through [MetaDvDictionary] rather than constructing a snapshot
    /// directly, because the sparse block-allocated id space is the thing the
    /// snapshot has to cope with.
    fn assert_round_trip(values: Vec<AnyDataValue>) {
        let mut dictionary = MetaDvDictionary::new();
        let ids: Vec<usize> = values
            .iter()
            .map(|value| dictionary.add_datavalue(value.clone()).value())
            .collect();

        let snapshot = DictionarySnapshot::capture(&dictionary, ids.iter().copied());
        assert_eq!(snapshot.len(), values.len(), "every id should be captured");

        let decoded =
            DictionarySnapshot::decode(&snapshot.encode()).expect("round trip should decode");

        for (id, value) in ids.iter().zip(&values) {
            assert_eq!(
                decoded.id_to_datavalue(*id).as_ref(),
                Some(value),
                "id {id} should resolve to the value it was added for"
            );
        }
    }

    #[test]
    fn round_trip_iris_and_strings() {
        assert_round_trip(vec![
            AnyDataValue::new_iri("http://example.org/a".to_string()),
            AnyDataValue::new_iri("http://www.w3.org/2002/07/owl#equivalentClass".to_string()),
            AnyDataValue::new_plain_string("a plain string".to_string()),
            AnyDataValue::new_plain_string(String::new()),
            AnyDataValue::new_language_tagged_string("hello".to_string(), "en".to_string()),
        ]);
    }

    #[test]
    fn round_trip_dictionary_backed_scalars() {
        assert_round_trip(vec![
            AnyDataValue::new_boolean(true),
            AnyDataValue::new_boolean(false),
            // Too large for an Int64 storage value, so it does need an id.
            AnyDataValue::new_integer_from_u64(u64::MAX),
        ]);
    }

    #[test]
    fn values_carried_inline_have_no_dictionary_id() {
        // Pins the invariant the snapshot's scope depends on: anything a storage
        // value can carry directly is rejected by the dictionary, so a snapshot
        // covers only the Id32/Id64 columns of a trie. If this ever changes, the
        // snapshot is silently incomplete rather than wrong, which is exactly the
        // kind of thing worth failing a test over.
        let mut dictionary = MetaDvDictionary::new();

        for value in [
            AnyDataValue::new_integer_from_i64(-42),
            AnyDataValue::new_integer_from_i64(0),
            AnyDataValue::new_double_from_f64(1.5).expect("finite"),
            AnyDataValue::new_float_from_f32(0.25).expect("finite"),
        ] {
            let id = dictionary.add_datavalue(value.clone()).value();
            assert_eq!(
                dictionary.id_to_datavalue(id),
                None,
                "{value:?} should not be dictionary-backed"
            );

            // Feeding such an id to capture must be harmless, since a caller
            // collecting ids from a table cannot tell them apart in advance.
            assert!(DictionarySnapshot::capture(&dictionary, [id]).is_empty());
        }
    }

    #[test]
    fn round_trip_other_typed_literals() {
        assert_round_trip(vec![AnyDataValue::new_other(
            "some lexical form".to_string(),
            "http://example.org/customType".to_string(),
        )]);
    }

    #[test]
    fn round_trip_non_ascii() {
        // The encoding is length-prefixed bytes rather than NUL-terminated, so
        // multi-byte characters and embedded control bytes have to survive.
        assert_round_trip(vec![
            AnyDataValue::new_plain_string("héllo wörld — ✓ 日本語".to_string()),
            AnyDataValue::new_plain_string("with\0an\tembedded\nnul".to_string()),
            AnyDataValue::new_iri("http://example.org/ünïcode".to_string()),
        ]);
    }

    #[test]
    fn round_trip_nulls() {
        // Nulls take ids from a dedicated sub-dictionary, so they exercise a
        // different id block from everything else.
        let mut dictionary = MetaDvDictionary::new();
        let (first, first_id) = dictionary.fresh_null();
        let (second, second_id) = dictionary.fresh_null();

        let snapshot = DictionarySnapshot::capture(&dictionary, [first_id, second_id]);
        let decoded =
            DictionarySnapshot::decode(&snapshot.encode()).expect("round trip should decode");

        assert_eq!(decoded.id_to_datavalue(first_id), Some(first));
        assert_eq!(decoded.id_to_datavalue(second_id), Some(second));
    }

    #[test]
    fn captures_only_the_requested_ids() {
        let mut dictionary = MetaDvDictionary::new();
        let kept = dictionary
            .add_datavalue(AnyDataValue::new_iri("http://example.org/kept".to_string()))
            .value();
        let _dropped = dictionary
            .add_datavalue(AnyDataValue::new_iri(
                "http://example.org/dropped".to_string(),
            ))
            .value();

        // Bounding the snapshot by the ids the tables reference is the whole
        // reason it can avoid enumerating a sparse id space.
        let snapshot = DictionarySnapshot::capture(&dictionary, [kept]);
        assert_eq!(snapshot.len(), 1);
        assert!(snapshot.id_to_datavalue(kept).is_some());
    }

    #[test]
    fn unresolvable_ids_are_skipped_not_fatal() {
        let dictionary = MetaDvDictionary::new();

        // Gaps in the block-allocated id space and merely-marked values both
        // yield None, and a caller collecting ids from tables cannot tell them
        // apart beforehand.
        let snapshot = DictionarySnapshot::capture(&dictionary, [0, 1, usize::MAX / 2]);
        assert!(snapshot.is_empty());
        assert!(snapshot.encode().len() >= MAGIC.len());
    }

    #[test]
    fn duplicate_ids_collapse() {
        let mut dictionary = MetaDvDictionary::new();
        let id = dictionary
            .add_datavalue(AnyDataValue::new_iri("http://example.org/x".to_string()))
            .value();

        let snapshot = DictionarySnapshot::capture(&dictionary, [id, id, id]);
        assert_eq!(snapshot.len(), 1);
    }

    #[test]
    fn rejects_foreign_input() {
        assert_eq!(
            DictionarySnapshot::decode(b"not a dictionary").unwrap_err(),
            DictionaryStorageError::NotADictionary
        );
    }

    #[test]
    fn rejects_other_layout_version() {
        let snapshot = DictionarySnapshot::default();
        let mut bytes = snapshot.encode();
        bytes[MAGIC.len()] = bytes[MAGIC.len()].wrapping_add(1);

        assert_eq!(
            DictionarySnapshot::decode(&bytes).unwrap_err(),
            DictionaryStorageError::VersionMismatch {
                found: FORMAT_VERSION + 1,
                expected: FORMAT_VERSION,
            }
        );
    }

    #[test]
    fn rejects_truncated_input() {
        let mut dictionary = MetaDvDictionary::new();
        let ids: Vec<usize> = ["a", "b", "c"]
            .iter()
            .map(|name| {
                dictionary
                    .add_datavalue(AnyDataValue::new_iri(format!("http://example.org/{name}")))
                    .value()
            })
            .collect();

        let bytes = DictionarySnapshot::capture(&dictionary, ids).encode();

        for length in 0..bytes.len() {
            assert!(
                DictionarySnapshot::decode(&bytes[..length]).is_err(),
                "prefix of length {length} decoded, but should not have"
            );
        }
    }

    #[test]
    fn rejects_unsorted_index() {
        let mut dictionary = MetaDvDictionary::new();
        let ids: Vec<usize> = ["a", "b"]
            .iter()
            .map(|name| {
                dictionary
                    .add_datavalue(AnyDataValue::new_iri(format!("http://example.org/{name}")))
                    .value()
            })
            .collect();

        let mut bytes = DictionarySnapshot::capture(&dictionary, ids).encode();

        // Swap the two index ids so the index descends. Binary search would
        // silently return wrong answers, so this has to be rejected outright.
        let index_start = MAGIC.len() + 4 + 4 + 8 + 8;
        for byte in 0..8 {
            bytes.swap(index_start + byte, index_start + 16 + byte);
        }

        assert_eq!(
            DictionarySnapshot::decode(&bytes).unwrap_err(),
            DictionaryStorageError::IndexNotSorted
        );
    }

    #[test]
    fn encoding_is_deterministic() {
        let mut dictionary = MetaDvDictionary::new();
        // IRIs, not integers: integers are carried inline and would produce an
        // empty snapshot, making this pass without testing anything.
        let ids: Vec<usize> = ["a", "b", "c"]
            .iter()
            .map(|name| {
                dictionary
                    .add_datavalue(AnyDataValue::new_iri(format!("http://example.org/{name}")))
                    .value()
            })
            .collect();

        let snapshot = DictionarySnapshot::capture(&dictionary, ids);
        assert_eq!(snapshot.encode(), snapshot.encode());

        // Required for use as a content-addressed cache entry.
        let decoded =
            DictionarySnapshot::decode(&snapshot.encode()).expect("round trip should decode");
        assert_eq!(decoded.encode(), snapshot.encode());
    }

    #[test]
    fn capture_order_does_not_affect_the_encoding() {
        let mut dictionary = MetaDvDictionary::new();
        let ids: Vec<usize> = ["a", "b", "c", "d"]
            .iter()
            .map(|name| {
                dictionary
                    .add_datavalue(AnyDataValue::new_iri(format!("http://example.org/{name}")))
                    .value()
            })
            .collect();

        let forward = DictionarySnapshot::capture(&dictionary, ids.iter().copied());
        let reversed = DictionarySnapshot::capture(&dictionary, ids.iter().rev().copied());

        assert_eq!(forward.encode(), reversed.encode());
    }
}
