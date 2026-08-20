//! Byte encoding for [Trie], so that a materialized table can be written out
//! and read back instead of being recomputed.
//!
//! # Why this exists
//!
//! Explaining inferences requires the model that produced them, and today the
//! only way to obtain that model is to run inference again. Being able to
//! persist a materialized table removes the recomputation.
//!
//! # Shape of the format
//!
//! A [Trie] decomposes into exactly one primitive. Every layer is an
//! [IntervalColumnT] of five typed [IntervalColumn]s; each of those holds a data
//! column, an interval column, and five interval lookups; and every one of those
//! is a [ColumnEnum], which is either a vector or a run-length encoding. So the
//! whole format is "encode a [ColumnEnum]" plus a little structure around it.
//!
//! Element types that actually occur: `u32`, `u64`, `i64`, [Float], [Double] for
//! data columns, and `usize` for interval and lookup columns.
//!
//! # Deliberate limitations
//!
//! - **Little-endian, fixed width.** The encoding is not portable across
//!   endianness. That is acceptable because the intended consumer is a
//!   content-addressed local cache whose key includes [FORMAT_VERSION]; a
//!   mismatch is a cache miss, not a migration.
//! - **Logical values, not raw run-length arrays.** A run-length column is
//!   written as the values it represents and re-encoded on read, rather than as
//!   its internal `values`/`end_indices`/`increments` arrays. This keeps the
//!   encoder independent of run-length internals at the cost of re-encoding on
//!   load. It also means the on-disk bytes are *not* yet a zero-copy image of
//!   the in-memory layout — writing raw arrays is what a future borrowed
//!   `ColumnEnum` variant would need, and is a separate change.
//! - **Arrays are 8-byte aligned** even so, precisely so that later change does
//!   not have to alter the layout.

use std::fmt::Display;

use crate::{
    columnar::{
        column::{Column, ColumnEnum, vector::ColumnVector},
        columnbuilder::{ColumnBuilder, rle::ColumnBuilderRle},
        intervalcolumn::IntervalColumnT,
    },
    datatypes::{ColumnDataType, Double, Float, RunLengthEncodable},
};

use super::Trie;

/// Identifies the encoding. Checked on read so that a stray file fails loudly
/// rather than being decoded into nonsense.
pub const MAGIC: &[u8; 8] = b"NMOTRIE\0";

/// Layout version.
///
/// Bump this whenever the byte layout changes in any way. Consumers are expected
/// to include it in a cache key, so an increment invalidates old files rather
/// than requiring them to be upgraded.
pub const FORMAT_VERSION: u32 = 1;

/// Every array in the encoding starts on a multiple of this, so that a future
/// zero-copy reader can reinterpret bytes as a slice without shifting anything.
const ALIGNMENT: usize = 8;

/// Discriminant for [ColumnEnum::ColumnVector].
const TAG_VECTOR: u8 = 0;
/// Discriminant for [ColumnEnum::ColumnRle].
const TAG_RLE: u8 = 1;

/// Something went wrong while decoding a [Trie].
///
/// Every variant means the input is not a trie this build can read. None of them
/// are recoverable, and all of them should be treated as "recompute instead".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrieStorageError {
    /// Input does not begin with [MAGIC].
    NotATrie,
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
    UnknownColumnTag(u8),
    /// A length or index did not fit in `usize` on this platform.
    ValueOutOfRange(u64),
}

impl Display for TrieStorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotATrie => write!(f, "input is not an encoded trie"),
            Self::VersionMismatch { found, expected } => write!(
                f,
                "encoded trie has layout version {found}, this build reads {expected}"
            ),
            Self::Truncated { offset, needed } => write!(
                f,
                "encoded trie ends prematurely: needed {needed} more byte(s) at offset {offset}"
            ),
            Self::UnknownColumnTag(tag) => write!(f, "unknown column tag {tag}"),
            Self::ValueOutOfRange(value) => {
                write!(
                    f,
                    "encoded value {value} does not fit in a pointer on this platform"
                )
            }
        }
    }
}

impl std::error::Error for TrieStorageError {}

/// Append-only byte sink that keeps arrays aligned.
pub(crate) struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    /// Pad with zeroes until the next [ALIGNMENT] boundary.
    pub(crate) fn align(&mut self) {
        let remainder = self.bytes.len() % ALIGNMENT;
        if remainder != 0 {
            self.bytes
                .resize(self.bytes.len() + (ALIGNMENT - remainder), 0);
        }
    }

    pub(crate) fn raw(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    pub(crate) fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub(crate) fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    /// Write a flag, then pad, so whatever follows stays aligned.
    pub(crate) fn flag(&mut self, value: bool) {
        self.bytes.push(u8::from(value));
        self.align();
    }

    /// Write a tag byte, then pad.
    pub(crate) fn tag(&mut self, value: u8) {
        self.bytes.push(value);
        self.align();
    }
}

/// Cursor over an encoded trie.
pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    pub(crate) fn take(&mut self, count: usize) -> Result<&'a [u8], TrieStorageError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or(TrieStorageError::Truncated {
                offset: self.offset,
                needed: count,
            })?;

        if end > self.bytes.len() {
            return Err(TrieStorageError::Truncated {
                offset: self.offset,
                needed: count,
            });
        }

        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    /// Skip padding inserted by [Writer::align].
    pub(crate) fn align(&mut self) -> Result<(), TrieStorageError> {
        let remainder = self.offset % ALIGNMENT;
        if remainder != 0 {
            self.take(ALIGNMENT - remainder)?;
        }
        Ok(())
    }

    pub(crate) fn u32(&mut self) -> Result<u32, TrieStorageError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes(
            bytes.try_into().expect("take returned exactly 4 bytes"),
        ))
    }

    pub(crate) fn u64(&mut self) -> Result<u64, TrieStorageError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes(
            bytes.try_into().expect("take returned exactly 8 bytes"),
        ))
    }

    /// Read a length or index, rejecting values this platform cannot address.
    pub(crate) fn length(&mut self) -> Result<usize, TrieStorageError> {
        let value = self.u64()?;
        usize::try_from(value).map_err(|_| TrieStorageError::ValueOutOfRange(value))
    }

    pub(crate) fn flag(&mut self) -> Result<bool, TrieStorageError> {
        let byte = self.take(1)?[0];
        self.align()?;
        Ok(byte != 0)
    }

    pub(crate) fn tag(&mut self) -> Result<u8, TrieStorageError> {
        let byte = self.take(1)?[0];
        self.align()?;
        Ok(byte)
    }
}

/// Fixed-width little-endian encoding for the element types a [Trie] contains.
///
/// `usize` is encoded as `u64` so that a file written on a 64-bit host is not
/// silently unreadable elsewhere; the decoder rejects values it cannot address
/// rather than truncating them.
pub(crate) trait ColumnElement: Copy {
    /// Encoded width in bytes.
    const WIDTH: usize;

    fn encode(self, writer: &mut Writer);

    fn decode(bytes: &[u8]) -> Result<Self, TrieStorageError>;
}

impl ColumnElement for u32 {
    const WIDTH: usize = 4;

    fn encode(self, writer: &mut Writer) {
        writer.raw(&self.to_le_bytes());
    }

    fn decode(bytes: &[u8]) -> Result<Self, TrieStorageError> {
        Ok(u32::from_le_bytes(bytes.try_into().expect("width checked")))
    }
}

impl ColumnElement for u64 {
    const WIDTH: usize = 8;

    fn encode(self, writer: &mut Writer) {
        writer.raw(&self.to_le_bytes());
    }

    fn decode(bytes: &[u8]) -> Result<Self, TrieStorageError> {
        Ok(u64::from_le_bytes(bytes.try_into().expect("width checked")))
    }
}

impl ColumnElement for i64 {
    const WIDTH: usize = 8;

    fn encode(self, writer: &mut Writer) {
        writer.raw(&self.to_le_bytes());
    }

    fn decode(bytes: &[u8]) -> Result<Self, TrieStorageError> {
        Ok(i64::from_le_bytes(bytes.try_into().expect("width checked")))
    }
}

impl ColumnElement for usize {
    const WIDTH: usize = 8;

    fn encode(self, writer: &mut Writer) {
        writer.raw(&(self as u64).to_le_bytes());
    }

    fn decode(bytes: &[u8]) -> Result<Self, TrieStorageError> {
        let value = u64::from_le_bytes(bytes.try_into().expect("width checked"));
        usize::try_from(value).map_err(|_| TrieStorageError::ValueOutOfRange(value))
    }
}

impl ColumnElement for Float {
    const WIDTH: usize = 4;

    fn encode(self, writer: &mut Writer) {
        writer.raw(&f32::from(self).to_bits().to_le_bytes());
    }

    fn decode(bytes: &[u8]) -> Result<Self, TrieStorageError> {
        let bits = u32::from_le_bytes(bytes.try_into().expect("width checked"));
        // Infallible rather than checked: the only values that reach here are
        // ones a Float was already constructed from, so NaN cannot appear.
        Ok(Float::from_number(f32::from_bits(bits)))
    }
}

impl ColumnElement for Double {
    const WIDTH: usize = 8;

    fn encode(self, writer: &mut Writer) {
        writer.raw(&f64::from(self).to_bits().to_le_bytes());
    }

    fn decode(bytes: &[u8]) -> Result<Self, TrieStorageError> {
        let bits = u64::from_le_bytes(bytes.try_into().expect("width checked"));
        Ok(Double::from_number(f64::from_bits(bits)))
    }
}

/// Encode a column as `tag | count | values`.
pub(crate) fn encode_column<T>(column: &ColumnEnum<T>, writer: &mut Writer)
where
    T: ColumnDataType + RunLengthEncodable + ColumnElement,
{
    let tag = match column {
        ColumnEnum::ColumnVector(_) => TAG_VECTOR,
        ColumnEnum::ColumnRle(_) => TAG_RLE,
    };

    writer.tag(tag);
    writer.u64(column.len() as u64);
    writer.align();

    for value in column.iter() {
        value.encode(writer);
    }

    writer.align();
}

/// Decode a column written by [encode_column].
///
/// The run-length variant is rebuilt from the values it represented, so the
/// reconstruction is equal in content and in variant, though its internal runs
/// are whatever `ColumnRle::new` derives rather than a byte-for-byte copy of the
/// original's.
pub(crate) fn decode_column<T>(reader: &mut Reader<'_>) -> Result<ColumnEnum<T>, TrieStorageError>
where
    T: ColumnDataType + RunLengthEncodable + ColumnElement + Default,
{
    let tag = reader.tag()?;
    let count = reader.length()?;
    reader.align()?;

    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(T::decode(reader.take(T::WIDTH)?)?);
    }

    reader.align()?;

    match tag {
        TAG_VECTOR => Ok(ColumnEnum::ColumnVector(ColumnVector::new(values))),
        TAG_RLE => {
            // Rebuilt through the run-length builder rather than from stored
            // run boundaries, which is what keeps this encoder independent of
            // the run-length internals. `ColumnRle::new` does the same thing but
            // is test-only, so the builder is used directly.
            let mut builder = ColumnBuilderRle::new();
            for value in values {
                builder.add(value);
            }
            Ok(ColumnEnum::ColumnRle(builder.finalize()))
        }
        other => Err(TrieStorageError::UnknownColumnTag(other)),
    }
}

/// A component of a [Trie] that can be written into the encoding.
///
/// Implemented in each type's own module, because encoding needs the private
/// fields and decoding needs to construct the value.
pub(crate) trait EncodeInto {
    fn encode_into(&self, writer: &mut Writer);
}

/// A component of a [Trie] that can be read back out of the encoding.
pub(crate) trait DecodeFrom: Sized {
    fn decode_from(reader: &mut Reader<'_>) -> Result<Self, TrieStorageError>;
}

impl Trie {
    /// Encode this trie into a byte buffer.
    ///
    /// The result is self-describing enough to be validated on read: it carries
    /// [MAGIC] and [FORMAT_VERSION], and every array is length-prefixed.
    pub fn encode(&self) -> Vec<u8> {
        let mut writer = Writer::new();

        writer.raw(MAGIC);
        writer.u32(FORMAT_VERSION);
        writer.u32(0); // reserved, keeps the header aligned
        writer.u64(self.columns.len() as u64);
        writer.flag(self.empty_row);

        for column in &self.columns {
            column.encode_into(&mut writer);
        }

        writer.bytes
    }

    /// Decode a trie produced by [Trie::encode].
    pub fn decode(bytes: &[u8]) -> Result<Self, TrieStorageError> {
        let mut reader = Reader::new(bytes);

        if reader.take(MAGIC.len())? != MAGIC {
            return Err(TrieStorageError::NotATrie);
        }

        let version = reader.u32()?;
        if version != FORMAT_VERSION {
            return Err(TrieStorageError::VersionMismatch {
                found: version,
                expected: FORMAT_VERSION,
            });
        }

        let _reserved = reader.u32()?;
        let arity = reader.length()?;
        let empty_row = reader.flag()?;

        let mut columns = Vec::with_capacity(arity);
        for _ in 0..arity {
            columns.push(IntervalColumnT::decode_from(&mut reader)?);
        }

        Ok(Self { columns, empty_row })
    }
}

#[cfg(test)]
mod test {
    use crate::{
        datatypes::{Double, Float, StorageValueT},
        tabular::trie::{
            Trie,
            storage::{FORMAT_VERSION, MAGIC, TrieStorageError},
        },
    };

    /// Encode, decode, and require the rows to survive unchanged.
    ///
    /// Rows rather than struct equality on purpose: what has to be preserved is
    /// the table, not the particular run-length split the encoder happened to
    /// produce.
    fn assert_round_trip(rows: Vec<Vec<StorageValueT>>) {
        let original = Trie::from_rows(rows);
        let decoded = Trie::decode(&original.encode()).expect("round trip should decode");

        assert_eq!(decoded.arity(), original.arity());
        assert_eq!(decoded.num_rows(), original.num_rows());
        assert_eq!(
            decoded.row_iterator().collect::<Vec<_>>(),
            original.row_iterator().collect::<Vec<_>>()
        );
    }

    #[test]
    fn round_trip_ids() {
        assert_round_trip(vec![
            vec![StorageValueT::Id32(1), StorageValueT::Id32(2)],
            vec![StorageValueT::Id32(1), StorageValueT::Id32(3)],
            vec![StorageValueT::Id32(4), StorageValueT::Id32(5)],
        ]);
    }

    #[test]
    fn round_trip_mixed_storage_types() {
        // Exercises every typed sub-column, which is where a fixed ordering
        // mistake between encoder and decoder would show up.
        assert_round_trip(vec![
            vec![StorageValueT::Id32(1), StorageValueT::Int64(-7)],
            vec![StorageValueT::Id64(1 << 40), StorageValueT::Id32(2)],
            vec![
                StorageValueT::Float(Float::from_number(1.5)),
                StorageValueT::Double(Double::from_number(-2.25)),
            ],
            vec![
                StorageValueT::Double(Double::from_number(1e100)),
                StorageValueT::Float(Float::from_number(0.0)),
            ],
        ]);
    }

    #[test]
    fn round_trip_run_length_friendly() {
        // A long ascending run is what pushes a column into the run-length
        // variant, so this covers the re-encoding path.
        assert_round_trip(
            (0..512)
                .map(|index| vec![StorageValueT::Id32(index), StorageValueT::Id32(index * 2)])
                .collect(),
        );
    }

    #[test]
    fn round_trip_single_column() {
        assert_round_trip((0..16).map(|i| vec![StorageValueT::Id32(i)]).collect());
    }

    #[test]
    fn rejects_foreign_input() {
        // Compares the error rather than the whole Result: Trie has no
        // PartialEq, and adding one just for a test would be the wrong tail
        // wagging the dog.
        assert_eq!(
            Trie::decode(b"not a trie at all").unwrap_err(),
            TrieStorageError::NotATrie
        );
    }

    #[test]
    fn rejects_other_layout_version() {
        let trie = Trie::from_rows(vec![vec![StorageValueT::Id32(1)]]);
        let mut bytes = trie.encode();
        bytes[MAGIC.len()] = bytes[MAGIC.len()].wrapping_add(1);

        assert_eq!(
            Trie::decode(&bytes).unwrap_err(),
            TrieStorageError::VersionMismatch {
                found: FORMAT_VERSION + 1,
                expected: FORMAT_VERSION,
            }
        );
    }

    #[test]
    fn rejects_truncated_input() {
        let trie = Trie::from_rows(vec![
            vec![StorageValueT::Id32(1), StorageValueT::Id32(2)],
            vec![StorageValueT::Id32(3), StorageValueT::Id32(4)],
        ]);
        let bytes = trie.encode();

        // Every proper prefix must be rejected rather than decoded into a
        // partial trie.
        for length in 0..bytes.len() {
            assert!(
                Trie::decode(&bytes[..length]).is_err(),
                "prefix of length {length} decoded, but should not have"
            );
        }
    }

    #[test]
    fn encoding_is_deterministic() {
        let trie = Trie::from_rows(vec![
            vec![StorageValueT::Id32(9), StorageValueT::Id32(8)],
            vec![StorageValueT::Id32(7), StorageValueT::Id32(6)],
        ]);

        assert_eq!(trie.encode(), trie.encode());
        // Re-encoding a decoded trie must reproduce the same bytes, or the
        // format would not be usable as a content-addressed cache entry.
        let decoded = Trie::decode(&trie.encode()).expect("should decode");
        assert_eq!(decoded.encode(), trie.encode());
    }
}
