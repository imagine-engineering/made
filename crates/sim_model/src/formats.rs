use crate::{
    artifact_index::VerifiedArtifactFile,
    parser_limits::{ParserLimitError, ParserLimits},
    restricted_pickle::{PickleValue, RestrictedPickleError, parse_restricted_pickle_cancellable},
};
use serde::{
    Deserialize, Serialize,
    de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor},
};
use sha2::{Digest, Sha256};
use sim_tensor::{
    CancellationToken, CpuBackend, DType, DeviceId, ExecutionContext, TensorDescriptor,
    generated_elementwise_or_runtime_operation_03::TorchArchiveValue,
    generated_elementwise_or_runtime_operation_09::{TorchArchiveLoadError, TorchArchiveLoader},
};
use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::File,
    io::{self, Cursor, Read, Seek, SeekFrom},
    mem::size_of,
    path::{Component, Path, PathBuf},
};

const SAFETENSORS_DTYPES: &[(&str, u64)] = &[
    ("BOOL", 1),
    ("U8", 1),
    ("I8", 1),
    ("U16", 2),
    ("I16", 2),
    ("F16", 2),
    ("BF16", 2),
    ("U32", 4),
    ("I32", 4),
    ("F32", 4),
    ("F8_E4M3", 1),
    ("F8_E5M2", 1),
    ("U64", 8),
    ("I64", 8),
    ("F64", 8),
    ("C64", 8),
];
const GGUF_METADATA_TYPES: std::ops::RangeInclusive<u32> = 0..=12;
const GGML_TENSOR_TYPES: &[u32] = &[
    0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 34, 35, 39, 40, 41, 42,
];
pub const MAX_EMBEDDING_ARCHIVE_VALUES: usize = 16_777_216;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFormat {
    Safetensors,
    PytorchArchive,
    Gguf,
    JsonConfig,
    JsonTokenizer,
    YamlConfig,
    SentencePiece,
    Tiktoken,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FileSlice {
    pub(crate) path: PathBuf,
    pub(crate) offset: u64,
    pub(crate) length: u64,
}

impl FileSlice {
    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn length(&self) -> u64 {
        self.length
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TensorMetadata {
    pub name: String,
    pub data_type: String,
    pub shape: Vec<u64>,
    pub storage: FileSlice,
}

impl TensorMetadata {
    pub fn native_dtype(&self) -> Option<DType> {
        canonical_model_dtype(&self.data_type)
    }

    pub fn element_count(&self) -> Result<u64, ModelFormatError> {
        checked_product(&self.shape, "tensor shape")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GgufValue {
    Unsigned(u64),
    Signed(i64),
    FloatBits(u64),
    Boolean(bool),
    String(String),
    Array(Vec<GgufValue>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum ParsedModelPayload {
    Safetensors {
        metadata: BTreeMap<String, String>,
    },
    Pytorch {
        root: PickleValue,
        archive_entries: Vec<ArchiveEntry>,
    },
    Gguf {
        version: u32,
        metadata: BTreeMap<String, GgufValue>,
    },
    Json(serde_json::Value),
    Yaml(serde_json::Value),
    SentencePiece {
        vocabulary: SentencePieceVocabulary,
    },
    Tiktoken {
        token_count: u64,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SentencePieceType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl SentencePieceType {
    fn checked(value: u64) -> Result<Self, ModelFormatError> {
        match value {
            1 => Ok(Self::Normal),
            2 => Ok(Self::Unknown),
            3 => Ok(Self::Control),
            4 => Ok(Self::UserDefined),
            5 => Ok(Self::Unused),
            6 => Ok(Self::Byte),
            _ => Err(invalid(
                "SentencePiece",
                format!("unknown sentence piece type {value}"),
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SentencePieceVocabularyEntry {
    piece: String,
    score: f32,
    piece_type: SentencePieceType,
}

impl SentencePieceVocabularyEntry {
    pub fn piece(&self) -> &str {
        &self.piece
    }

    pub const fn score(&self) -> f32 {
        self.score
    }

    pub const fn piece_type(&self) -> SentencePieceType {
        self.piece_type
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SentencePieceVocabulary {
    entries: Vec<SentencePieceVocabularyEntry>,
}

impl SentencePieceVocabulary {
    pub fn entries(&self) -> &[SentencePieceVocabularyEntry] {
        &self.entries
    }

    #[cfg(test)]
    pub(crate) fn fixture_for_test(count: usize) -> Self {
        Self {
            entries: (0..count)
                .map(|index| SentencePieceVocabularyEntry {
                    piece: format!("piece-{index}"),
                    score: -(index as f32),
                    piece_type: if index == 0 {
                        SentencePieceType::Unknown
                    } else {
                        SentencePieceType::Normal
                    },
                })
                .collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParsedModel {
    pub format: ModelFormat,
    pub tensors: Vec<TensorMetadata>,
    pub payload: ParsedModelPayload,
    pub source_size: u64,
    pub source_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArchiveEntry {
    pub name: String,
    pub data_offset: u64,
    pub length: u64,
    pub crc32: u32,
}

#[derive(Debug)]
pub struct ParsedEmbeddingArchive {
    rows: Vec<Vec<f32>>,
    width: usize,
}

impl ParsedEmbeddingArchive {
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub(crate) fn into_rows(self) -> Vec<Vec<f32>> {
        self.rows
    }
}

pub(crate) fn parse_verified_embedding_archive_file(
    mut verified: VerifiedArtifactFile,
    limits: &ParserLimits,
    cancellation: &CancellationToken,
) -> Result<Option<ParsedEmbeddingArchive>, ModelFormatError> {
    limits.validate()?;
    check_cancelled(cancellation)?;
    let path = verified.path().to_path_buf();
    let entries = parse_stored_zip(verified.file_mut(), &path, limits, cancellation)?;
    for entry in entries
        .iter()
        .rev()
        .filter(|entry| entry.name.contains("data/"))
    {
        check_cancelled(cancellation)?;
        if !entry.length.is_multiple_of(4) {
            continue;
        }
        limits.check(
            "embedding archive tensor bytes",
            entry.length,
            limits.maximum_tensor_bytes,
        )?;
        let value_count = usize::try_from(entry.length / 4)
            .map_err(|_| ModelFormatError::Overflow("embedding archive value count"))?;
        if value_count < 768 || value_count > MAX_EMBEDDING_ARCHIVE_VALUES {
            continue;
        }
        let width = if value_count.is_multiple_of(768) {
            768
        } else {
            1_024
        };
        if !value_count.is_multiple_of(width) {
            continue;
        }
        let row_count = value_count / width;
        let mut rows = Vec::new();
        rows.try_reserve_exact(row_count)
            .map_err(|_| ModelFormatError::AllocationFailed {
                context: "embedding archive rows",
                requested: row_count,
            })?;
        let row_bytes = width
            .checked_mul(4)
            .ok_or(ModelFormatError::Overflow("embedding archive row bytes"))?;
        verified
            .file_mut()
            .seek(SeekFrom::Start(entry.data_offset))
            .map_err(|error| io_error(&path, error))?;
        let bytes = read_exact_bounded(
            verified.file_mut(),
            &path,
            entry.length,
            limits.maximum_tensor_bytes,
            "embedding archive tensor",
        )?;
        if crc32(&bytes) != entry.crc32 {
            return Err(invalid(
                "embedding archive",
                format!("entry {:?} CRC32 mismatch", entry.name),
            ));
        }
        for bytes in bytes.chunks_exact(row_bytes) {
            check_cancelled(cancellation)?;
            let mut row = Vec::new();
            row.try_reserve_exact(width)
                .map_err(|_| ModelFormatError::AllocationFailed {
                    context: "embedding archive row",
                    requested: width,
                })?;
            for value in bytes.chunks_exact(4) {
                let value = f32::from_le_bytes([value[0], value[1], value[2], value[3]]);
                if !value.is_finite() {
                    return Err(invalid(
                        "embedding archive",
                        "embedding value is not finite",
                    ));
                }
                row.push(value);
            }
            rows.push(row);
        }
        check_cancelled(cancellation)?;
        verified
            .verify_unchanged()
            .map_err(|error| ModelFormatError::Io {
                path: path.clone(),
                message: error.to_string(),
            })?;
        return Ok(Some(ParsedEmbeddingArchive { rows, width }));
    }
    verified
        .verify_unchanged()
        .map_err(|error| ModelFormatError::Io {
            path,
            message: error.to_string(),
        })?;
    Ok(None)
}

pub(crate) fn has_nested_string_to_param(root: &PickleValue) -> bool {
    fn mapping_contains(entries: &[(PickleValue, PickleValue)]) -> bool {
        entries.iter().any(|(key, value)| {
            matches!(key, PickleValue::String(name) if name == "string_to_param")
                && matches!(value, PickleValue::Dictionary(_))
        })
    }

    let PickleValue::Dictionary(entries) = root else {
        return false;
    };
    mapping_contains(entries)
        || entries.iter().any(|(key, value)| {
            matches!(key, PickleValue::String(name) if name == "state_dict")
                && matches!(value, PickleValue::Dictionary(inner) if mapping_contains(inner))
        })
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ModelFormatError {
    #[error("model parsing was cancelled")]
    Cancelled,
    #[error("model I/O failed for {path}: {message}")]
    Io { path: PathBuf, message: String },
    #[error(transparent)]
    Limit(#[from] ParserLimitError),
    #[error(transparent)]
    RestrictedPickle(#[from] RestrictedPickleError),
    #[error("invalid {format} data: {reason}")]
    Invalid {
        format: &'static str,
        reason: String,
    },
    #[error("unsupported model format for {0}")]
    Unsupported(PathBuf),
    #[error("archive entry {0:?} has an unsafe path")]
    UnsafeArchivePath(String),
    #[error("archive entry {0:?} is a symbolic link")]
    ArchiveLink(String),
    #[error("archive contains duplicate canonical path {0:?}")]
    DuplicateArchivePath(String),
    #[error("archive entry {name:?} uses unsupported compression method {method}")]
    UnsupportedCompression { name: String, method: u16 },
    #[error("allocation of {requested} bytes failed while parsing {context}")]
    AllocationFailed {
        context: &'static str,
        requested: usize,
    },
    #[error("model byte arithmetic overflowed while parsing {0}")]
    Overflow(&'static str),
}

pub fn parse_model_file(
    path: &Path,
    limits: &ParserLimits,
    cancellation: &CancellationToken,
) -> Result<ParsedModel, ModelFormatError> {
    let file = open_file(path)?;
    parse_model_open_file(file, path, limits, cancellation)
}

pub fn parse_safetensors_bytes(
    bytes: &[u8],
    source_name: &Path,
    limits: &ParserLimits,
    cancellation: &CancellationToken,
) -> Result<ParsedModel, ModelFormatError> {
    limits.validate()?;
    check_cancelled(cancellation)?;
    let source_size = u64::try_from(bytes.len())
        .map_err(|_| ModelFormatError::Overflow("safetensors buffer size"))?;
    // The name labels diagnostics and tensor slices; this boundary never opens it.
    parse_safetensors_reader(
        &mut Cursor::new(bytes),
        source_name,
        source_size,
        limits,
        cancellation,
    )
}

pub struct TorchArchiveFileLoader<'a> {
    path: &'a Path,
    limits: &'a ParserLimits,
}

impl<'a> TorchArchiveFileLoader<'a> {
    pub fn new(path: &'a Path, limits: &'a ParserLimits) -> Self {
        Self { path, limits }
    }
}

impl TorchArchiveLoader for TorchArchiveFileLoader<'_> {
    fn load_weights_cpu(
        &self,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<TorchArchiveValue, TorchArchiveLoadError> {
        load_torch_archive_file(self.path, self.limits, backend, context)
    }
}

pub fn load_torch_archive_file(
    path: &Path,
    limits: &ParserLimits,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<TorchArchiveValue, TorchArchiveLoadError> {
    let parsed =
        parse_model_file(path, limits, context.cancellation).map_err(torch_archive_parse_error)?;
    let ParsedModelPayload::Pytorch {
        root,
        archive_entries,
    } = parsed.payload
    else {
        return Err(torch_archive_rejected(
            "input is not a restricted PyTorch archive",
        ));
    };
    let data_pickle = archive_entries
        .iter()
        .find(|entry| entry.name == "data.pkl" || entry.name.ends_with("/data.pkl"))
        .ok_or_else(|| torch_archive_rejected("archive data.pkl entry is missing"))?;
    let archive_root = data_pickle
        .name
        .strip_suffix("data.pkl")
        .ok_or_else(|| torch_archive_rejected("archive data.pkl root is invalid"))?;
    let mut file = open_file(path).map_err(torch_archive_parse_error)?;
    let mut tensor_count = 0_u64;
    let mut aggregate_bytes = 0_u64;
    pickle_to_torch_archive_value(
        &root,
        &archive_entries,
        archive_root,
        &mut file,
        path,
        limits,
        backend,
        context,
        &mut tensor_count,
        &mut aggregate_bytes,
        limits.maximum_depth,
    )
}

#[allow(clippy::too_many_arguments)]
fn pickle_to_torch_archive_value(
    value: &PickleValue,
    entries: &[ArchiveEntry],
    archive_root: &str,
    file: &mut File,
    path: &Path,
    limits: &ParserLimits,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    tensor_count: &mut u64,
    aggregate_bytes: &mut u64,
    remaining_depth: u32,
) -> Result<TorchArchiveValue, TorchArchiveLoadError> {
    if remaining_depth == 0 {
        return Err(torch_archive_rejected(
            "archive value exceeds the configured depth",
        ));
    }
    check_cancelled(context.cancellation).map_err(torch_archive_parse_error)?;
    if let Some(rebuild) = tensor_rebuild(value, remaining_depth) {
        let validated = validate_pytorch_rebuild(
            &rebuild,
            "torch.load value",
            entries,
            archive_root,
            limits,
            tensor_count,
            aggregate_bytes,
        )
        .map_err(torch_archive_parse_error)?;
        let offset = validated
            .entry
            .data_offset
            .checked_add(validated.tensor_start)
            .ok_or_else(|| torch_archive_rejected("tensor file offset overflowed"))?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|error| torch_archive_parse_error(io_error(path, error)))?;
        let bytes = read_exact_bounded(
            file,
            path,
            validated.tensor_bytes,
            limits.maximum_tensor_bytes,
            "torch.load tensor bytes",
        )
        .map_err(torch_archive_parse_error)?;
        let dtype = torch_storage_dtype(&rebuild.data_type)?;
        let descriptor =
            TensorDescriptor::contiguous(rebuild.shape, dtype, DeviceId::CPU, context.stream)?;
        let tensor = backend.upload_bytes(descriptor, &bytes, context)?.0;
        return Ok(TorchArchiveValue::Tensor(tensor));
    }
    let next_depth = remaining_depth - 1;
    Ok(match value {
        PickleValue::None => TorchArchiveValue::None,
        PickleValue::Boolean(value) => TorchArchiveValue::Boolean(*value),
        PickleValue::Integer(value) => TorchArchiveValue::Integer(
            i64::try_from(*value)
                .map_err(|_| torch_archive_rejected("integer exceeds the native value range"))?,
        ),
        PickleValue::FloatBits(bits) => TorchArchiveValue::Float(f64::from_bits(*bits)),
        PickleValue::String(value) => TorchArchiveValue::String(value.clone()),
        PickleValue::List(values) => TorchArchiveValue::List(pickle_sequence_to_archive(
            values,
            entries,
            archive_root,
            file,
            path,
            limits,
            backend,
            context,
            tensor_count,
            aggregate_bytes,
            next_depth,
        )?),
        PickleValue::Tuple(values) => TorchArchiveValue::Tuple(pickle_sequence_to_archive(
            values,
            entries,
            archive_root,
            file,
            path,
            limits,
            backend,
            context,
            tensor_count,
            aggregate_bytes,
            next_depth,
        )?),
        PickleValue::Dictionary(values) => {
            let mut output = BTreeMap::new();
            for (index, (key, value)) in values.iter().enumerate() {
                if index.is_multiple_of(256) {
                    check_cancelled(context.cancellation).map_err(torch_archive_parse_error)?;
                }
                let PickleValue::String(key) = key else {
                    return Err(torch_archive_rejected(
                        "native torch.load maps require string keys",
                    ));
                };
                let value = pickle_to_torch_archive_value(
                    value,
                    entries,
                    archive_root,
                    file,
                    path,
                    limits,
                    backend,
                    context,
                    tensor_count,
                    aggregate_bytes,
                    next_depth,
                )?;
                if output.insert(key.clone(), value).is_some() {
                    return Err(torch_archive_rejected(
                        "native torch.load map contains a duplicate key",
                    ));
                }
            }
            TorchArchiveValue::Map(output)
        }
        PickleValue::Reduced {
            target, arguments, ..
        } if target == "collections.OrderedDict"
            && matches!(arguments.as_ref(), PickleValue::Tuple(values) if values.is_empty()) =>
        {
            TorchArchiveValue::Map(BTreeMap::new())
        }
        PickleValue::Bytes(_)
        | PickleValue::Set(_)
        | PickleValue::Global(_)
        | PickleValue::Persistent(_)
        | PickleValue::Reduced { .. } => {
            return Err(torch_archive_rejected(
                "restricted pickle value has no native torch.load representation",
            ));
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn pickle_sequence_to_archive(
    values: &[PickleValue],
    entries: &[ArchiveEntry],
    archive_root: &str,
    file: &mut File,
    path: &Path,
    limits: &ParserLimits,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    tensor_count: &mut u64,
    aggregate_bytes: &mut u64,
    remaining_depth: u32,
) -> Result<Vec<TorchArchiveValue>, TorchArchiveLoadError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(values.len())
        .map_err(|_| torch_archive_rejected("archive sequence allocation failed"))?;
    for (index, value) in values.iter().enumerate() {
        if index.is_multiple_of(256) {
            check_cancelled(context.cancellation).map_err(torch_archive_parse_error)?;
        }
        output.push(pickle_to_torch_archive_value(
            value,
            entries,
            archive_root,
            file,
            path,
            limits,
            backend,
            context,
            tensor_count,
            aggregate_bytes,
            remaining_depth,
        )?);
    }
    Ok(output)
}

fn torch_storage_dtype(storage: &str) -> Result<DType, TorchArchiveLoadError> {
    canonical_model_dtype(storage)
        .ok_or_else(|| torch_archive_rejected("tensor storage dtype is unsupported"))
}

pub(crate) fn canonical_model_dtype(storage: &str) -> Option<DType> {
    Some(match storage {
        "BOOL" => DType::Bool,
        "U8" => DType::U8,
        "I8" => DType::I8,
        "I16" => DType::I16,
        "I32" => DType::I32,
        "I64" => DType::I64,
        "F16" => DType::F16,
        "BF16" => DType::Bf16,
        "F32" => DType::F32,
        "F64" => DType::F64,
        "F8_E4M3" => DType::Float8E4m3Fn,
        "F8_E5M2" => DType::Float8E5m2,
        "C64" => DType::Complex64,
        "C128" => DType::Complex128,
        "torch.BoolStorage" => DType::Bool,
        "torch.ByteStorage" => DType::U8,
        "torch.CharStorage" => DType::I8,
        "torch.ShortStorage" => DType::I16,
        "torch.IntStorage" => DType::I32,
        "torch.LongStorage" => DType::I64,
        "torch.HalfStorage" => DType::F16,
        "torch.BFloat16Storage" => DType::Bf16,
        "torch.FloatStorage" => DType::F32,
        "torch.DoubleStorage" => DType::F64,
        "torch.ComplexFloatStorage" => DType::Complex64,
        "torch.ComplexDoubleStorage" => DType::Complex128,
        _ => return None,
    })
}

fn torch_archive_parse_error(error: ModelFormatError) -> TorchArchiveLoadError {
    if matches!(error, ModelFormatError::Cancelled) {
        TorchArchiveLoadError::Cancelled
    } else {
        torch_archive_rejected(error.to_string())
    }
}

fn torch_archive_rejected(reason: impl Into<String>) -> TorchArchiveLoadError {
    TorchArchiveLoadError::Rejected {
        reason: reason.into(),
    }
}

pub(crate) fn parse_verified_model_file(
    verified: VerifiedArtifactFile,
    limits: &ParserLimits,
    cancellation: &CancellationToken,
) -> Result<ParsedModel, ModelFormatError> {
    let path = verified.path().to_path_buf();
    parse_model_open_file(verified.into_file(), &path, limits, cancellation)
}

fn parse_model_open_file(
    mut file: File,
    path: &Path,
    limits: &ParserLimits,
    cancellation: &CancellationToken,
) -> Result<ParsedModel, ModelFormatError> {
    limits.validate()?;
    check_cancelled(cancellation)?;
    let format = detect_model_format(path)?;
    match format {
        ModelFormat::Safetensors => parse_safetensors(&mut file, path, limits, cancellation),
        ModelFormat::PytorchArchive => parse_pytorch(&mut file, path, limits, cancellation),
        ModelFormat::Gguf => parse_gguf(&mut file, path, limits, cancellation),
        ModelFormat::JsonConfig | ModelFormat::JsonTokenizer => {
            parse_json_file(&mut file, path, format, limits, cancellation)
        }
        ModelFormat::YamlConfig => parse_yaml_file(&mut file, path, limits, cancellation),
        ModelFormat::SentencePiece => parse_sentencepiece(&mut file, path, limits, cancellation),
        ModelFormat::Tiktoken => parse_tiktoken(&mut file, path, limits, cancellation),
    }
}

pub fn detect_model_format(path: &Path) -> Result<ModelFormat, ModelFormatError> {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match extension.as_str() {
        "safetensors" | "sft" | "latent" => Ok(ModelFormat::Safetensors),
        "ckpt" | "pt" | "pt2" | "bin" | "pth" | "pkl" => Ok(ModelFormat::PytorchArchive),
        "gguf" => Ok(ModelFormat::Gguf),
        "model" if name.contains("tokenizer") => Ok(ModelFormat::SentencePiece),
        "tiktoken" => Ok(ModelFormat::Tiktoken),
        "json" if name.contains("tokenizer") || name.contains("vocab") => {
            Ok(ModelFormat::JsonTokenizer)
        }
        "json" => Ok(ModelFormat::JsonConfig),
        "yaml" | "yml" => Ok(ModelFormat::YamlConfig),
        _ => Err(ModelFormatError::Unsupported(path.to_path_buf())),
    }
}

fn parse_safetensors(
    file: &mut File,
    path: &Path,
    limits: &ParserLimits,
    cancellation: &CancellationToken,
) -> Result<ParsedModel, ModelFormatError> {
    let source_size = file_length(file, path)?;
    parse_safetensors_reader(file, path, source_size, limits, cancellation)
}

fn parse_safetensors_reader(
    file: &mut (impl Read + Seek),
    path: &Path,
    source_size: u64,
    limits: &ParserLimits,
    cancellation: &CancellationToken,
) -> Result<ParsedModel, ModelFormatError> {
    if source_size < 8 {
        return Err(invalid("safetensors", "missing 8-byte header length"));
    }
    let header_length = read_u64(file, path)?;
    limits.check(
        "safetensors header bytes",
        header_length,
        limits.manifest_bytes,
    )?;
    let header_end = 8_u64
        .checked_add(header_length)
        .ok_or(ModelFormatError::Overflow("safetensors header"))?;
    if header_end > source_size {
        return Err(invalid("safetensors", "header exceeds file length"));
    }
    let header = read_exact_bounded(
        file,
        path,
        header_length,
        limits.manifest_bytes,
        "safetensors header",
    )?;
    check_cancelled(cancellation)?;
    let value = parse_strict_json(&header, limits)?;
    let object = value
        .as_object()
        .ok_or_else(|| invalid("safetensors", "header must be an object"))?;
    let tensor_count = object
        .len()
        .checked_sub(usize::from(object.contains_key("__metadata__")))
        .ok_or(ModelFormatError::Overflow("safetensors tensor count"))?;
    limits.check(
        "safetensors tensor count",
        u64::try_from(tensor_count).unwrap_or(u64::MAX),
        limits.maximum_tensors,
    )?;

    let mut metadata = BTreeMap::new();
    let mut tensors = Vec::new();
    tensors
        .try_reserve(tensor_count)
        .map_err(|_| ModelFormatError::AllocationFailed {
            context: "safetensors tensor table",
            requested: tensor_count,
        })?;
    let data_length = source_size - header_end;
    let mut aggregate = 0_u64;
    let mut ranges = Vec::new();
    for (name, tensor_value) in object {
        check_cancelled(cancellation)?;
        check_name(name, limits, "safetensors tensor name")?;
        if name == "__metadata__" {
            let values = tensor_value
                .as_object()
                .ok_or_else(|| invalid("safetensors", "__metadata__ must be an object"))?;
            limits.check(
                "safetensors metadata values",
                u64::try_from(values.len()).unwrap_or(u64::MAX),
                limits.maximum_metadata_values,
            )?;
            for (key, value) in values {
                check_name(key, limits, "safetensors metadata name")?;
                let value = value
                    .as_str()
                    .ok_or_else(|| invalid("safetensors", "metadata values must be strings"))?;
                check_name(value, limits, "safetensors metadata value")?;
                metadata.insert(key.clone(), value.to_owned());
            }
            continue;
        }
        let tensor = tensor_value
            .as_object()
            .ok_or_else(|| invalid("safetensors", "tensor descriptor must be an object"))?;
        if tensor.len() != 3
            || !tensor.contains_key("dtype")
            || !tensor.contains_key("shape")
            || !tensor.contains_key("data_offsets")
        {
            return Err(invalid(
                "safetensors",
                "tensor descriptor must contain only dtype, shape, and data_offsets",
            ));
        }
        let data_type = tensor
            .get("dtype")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| invalid("safetensors", "tensor dtype must be a string"))?;
        let bytes_per_element = SAFETENSORS_DTYPES
            .iter()
            .find_map(|(name, bytes)| (*name == data_type).then_some(*bytes))
            .ok_or_else(|| invalid("safetensors", format!("unknown dtype {data_type:?}")))?;
        let shape = json_u64_array(
            tensor.get("shape"),
            limits.maximum_depth,
            "safetensors shape",
        )?;
        let offsets = json_u64_array(tensor.get("data_offsets"), 2, "safetensors data_offsets")?;
        if offsets.len() != 2 {
            return Err(invalid(
                "safetensors",
                "data_offsets must contain exactly two integers",
            ));
        }
        let start = offsets[0];
        let end = offsets[1];
        if start > end || end > data_length {
            return Err(invalid(
                "safetensors",
                "tensor data range is outside the file",
            ));
        }
        let elements = checked_product(&shape, "safetensors shape")?;
        let expected = elements
            .checked_mul(bytes_per_element)
            .ok_or(ModelFormatError::Overflow("safetensors tensor bytes"))?;
        let length = end - start;
        if expected != length {
            return Err(invalid(
                "safetensors",
                format!("tensor {name:?} needs {expected} bytes but range has {length}"),
            ));
        }
        limits.check(
            "safetensors tensor bytes",
            length,
            limits.maximum_tensor_bytes,
        )?;
        aggregate = aggregate
            .checked_add(length)
            .ok_or(ModelFormatError::Overflow("safetensors aggregate bytes"))?;
        limits.check(
            "safetensors aggregate tensor bytes",
            aggregate,
            limits.maximum_aggregate_tensor_bytes,
        )?;
        ranges.push((start, end));
        tensors.push(TensorMetadata {
            name: name.clone(),
            data_type: data_type.to_owned(),
            shape,
            storage: FileSlice {
                path: path.to_path_buf(),
                offset: header_end
                    .checked_add(start)
                    .ok_or(ModelFormatError::Overflow("safetensors file offset"))?,
                length,
            },
        });
    }
    validate_safetensors_ranges(&mut ranges, data_length)?;
    let source_sha256 = sha256_open_file(file, path, cancellation)?;
    Ok(ParsedModel {
        format: ModelFormat::Safetensors,
        tensors,
        payload: ParsedModelPayload::Safetensors { metadata },
        source_size,
        source_sha256,
    })
}

fn parse_pytorch(
    file: &mut File,
    path: &Path,
    limits: &ParserLimits,
    cancellation: &CancellationToken,
) -> Result<ParsedModel, ModelFormatError> {
    let source_size = file_length(file, path)?;
    let mut magic = [0_u8; 4];
    let magic_length = file
        .read(&mut magic)
        .map_err(|error| io_error(path, error))?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| io_error(path, error))?;
    let (root, archive_entries, archive_root) =
        if magic_length == magic.len() && magic == [0x50, 0x4b, 0x03, 0x04] {
            let entries = parse_stored_zip(file, path, limits, cancellation)?;
            let pickle_entries = entries
                .iter()
                .filter(|entry| entry.name == "data.pkl" || entry.name.ends_with("/data.pkl"))
                .collect::<Vec<_>>();
            let [pickle_entry] = pickle_entries.as_slice() else {
                return Err(invalid(
                    "PyTorch archive",
                    "archive must contain exactly one data.pkl",
                ));
            };
            let archive_root = pickle_entry
                .name
                .strip_suffix("data.pkl")
                .ok_or_else(|| invalid("PyTorch archive", "data.pkl root is invalid"))?
                .to_owned();
            if entries
                .iter()
                .any(|entry| !entry.name.starts_with(&archive_root))
            {
                return Err(invalid(
                    "PyTorch archive",
                    "archive contains files outside the data.pkl root",
                ));
            }
            limits.check(
                "PyTorch data.pkl bytes",
                pickle_entry.length,
                limits.manifest_bytes,
            )?;
            file.seek(SeekFrom::Start(pickle_entry.data_offset))
                .map_err(|error| io_error(path, error))?;
            let bytes = read_exact_bounded(
                file,
                path,
                pickle_entry.length,
                limits.manifest_bytes,
                "PyTorch data.pkl",
            )?;
            if crc32(&bytes) != pickle_entry.crc32 {
                return Err(invalid("PyTorch archive", "data.pkl CRC32 mismatch"));
            }
            (
                parse_pickle(&bytes, limits, cancellation)?,
                entries,
                Some(archive_root),
            )
        } else {
            limits.check("legacy pickle bytes", source_size, limits.manifest_bytes)?;
            let bytes = read_exact_bounded(
                file,
                path,
                source_size,
                limits.manifest_bytes,
                "legacy PyTorch pickle",
            )?;
            (
                parse_pickle(&bytes, limits, cancellation)?,
                Vec::new(),
                None,
            )
        };
    check_cancelled(cancellation)?;
    let tensors = tensor_metadata_from_pickle(
        path,
        &root,
        &archive_entries,
        archive_root.as_deref(),
        limits,
    )?;
    let source_sha256 = sha256_open_file(file, path, cancellation)?;
    Ok(ParsedModel {
        format: ModelFormat::PytorchArchive,
        tensors,
        payload: ParsedModelPayload::Pytorch {
            root,
            archive_entries,
        },
        source_size,
        source_sha256,
    })
}

fn parse_gguf(
    file: &mut File,
    path: &Path,
    limits: &ParserLimits,
    cancellation: &CancellationToken,
) -> Result<ParsedModel, ModelFormatError> {
    let source_size = file_length(file, path)?;
    let mut magic = [0_u8; 4];
    read_exact(file, path, &mut magic)?;
    if magic != *b"GGUF" {
        return Err(invalid("GGUF", "magic is not GGUF"));
    }
    let version = read_u32(file, path)?;
    if !matches!(version, 2 | 3) {
        return Err(invalid("GGUF", format!("unsupported version {version}")));
    }
    let tensor_count = read_u64(file, path)?;
    let metadata_count = read_u64(file, path)?;
    limits.check("GGUF tensor count", tensor_count, limits.maximum_tensors)?;
    limits.check(
        "GGUF metadata count",
        metadata_count,
        limits.maximum_metadata_values,
    )?;
    limits.check("GGUF manifest header", 24, limits.manifest_bytes)?;
    let mut budget = limits.manifest_bytes - 24;
    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count {
        check_cancelled(cancellation)?;
        let key = read_gguf_string(file, path, limits, &mut budget)?;
        if metadata.contains_key(&key) {
            return Err(invalid("GGUF", format!("duplicate metadata key {key:?}")));
        }
        consume_budget(&mut budget, 4, "GGUF manifest bytes")?;
        let value_type = read_u32(file, path)?;
        if key == "general.alignment" && value_type != 4 {
            return Err(invalid(
                "GGUF",
                "general.alignment must use the UINT32 metadata type",
            ));
        }
        let value = read_gguf_value(file, path, value_type, 0, limits, &mut budget, cancellation)?;
        metadata.insert(key, value);
    }
    let minimum_tensor_table_bytes =
        tensor_count
            .checked_mul(24)
            .ok_or(ModelFormatError::Overflow(
                "GGUF minimum tensor table bytes",
            ))?;
    if minimum_tensor_table_bytes > budget {
        return Err(invalid(
            "GGUF",
            "declared tensor table cannot fit within the manifest limit",
        ));
    }
    let mut descriptors = Vec::new();
    descriptors
        .try_reserve(usize_from_u64(tensor_count, "GGUF tensor count")?)
        .map_err(|_| ModelFormatError::AllocationFailed {
            context: "GGUF tensor table",
            requested: usize_from_u64(tensor_count, "GGUF tensor count").unwrap_or(usize::MAX),
        })?;
    let mut names = BTreeSet::new();
    for _ in 0..tensor_count {
        check_cancelled(cancellation)?;
        let name = read_gguf_string(file, path, limits, &mut budget)?;
        if !names.insert(name.clone()) {
            return Err(invalid("GGUF", format!("duplicate tensor name {name:?}")));
        }
        consume_budget(&mut budget, 4, "GGUF manifest bytes")?;
        let dimensions = read_u32(file, path)?;
        if dimensions == 0 {
            return Err(invalid(
                "GGUF",
                "tensor shape must have at least one dimension",
            ));
        }
        if dimensions > limits.maximum_depth {
            return Err(ParserLimitError::Exceeded {
                kind: "GGUF tensor dimensions",
                actual: u64::from(dimensions),
                maximum: u64::from(limits.maximum_depth),
            }
            .into());
        }
        let shape_and_descriptor_bytes = u64::from(dimensions)
            .checked_mul(8)
            .and_then(|value| value.checked_add(12))
            .ok_or(ModelFormatError::Overflow("GGUF tensor descriptor bytes"))?;
        consume_budget(
            &mut budget,
            shape_and_descriptor_bytes,
            "GGUF manifest bytes",
        )?;
        let mut shape = Vec::new();
        shape
            .try_reserve(usize::try_from(dimensions).unwrap_or(usize::MAX))
            .map_err(|_| ModelFormatError::AllocationFailed {
                context: "GGUF tensor shape",
                requested: usize::try_from(dimensions).unwrap_or(usize::MAX),
            })?;
        for _ in 0..dimensions {
            shape.push(read_u64(file, path)?);
        }
        checked_product(&shape, "GGUF tensor shape")?;
        let data_type = read_u32(file, path)?;
        if !GGML_TENSOR_TYPES.contains(&data_type) {
            return Err(invalid(
                "GGUF",
                format!("unknown GGML tensor type {data_type}"),
            ));
        }
        let offset = read_u64(file, path)?;
        descriptors.push((name, shape, data_type, offset));
    }
    let alignment = match metadata.get("general.alignment") {
        Some(GgufValue::Unsigned(value)) => *value,
        Some(_) => return Err(invalid("GGUF", "general.alignment must be unsigned")),
        None => 32,
    };
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(invalid(
            "GGUF",
            "general.alignment must be a non-zero power of two",
        ));
    }
    let descriptor_end = file
        .stream_position()
        .map_err(|error| io_error(path, error))?;
    let data_start = align_up(descriptor_end, alignment)?;
    if data_start > source_size {
        return Err(invalid("GGUF", "aligned tensor data starts after EOF"));
    }
    descriptors.sort_by_key(|entry| entry.3);
    let mut tensors = Vec::new();
    let mut aggregate = 0_u64;
    for (index, (name, shape, data_type, offset)) in descriptors.iter().enumerate() {
        if *offset % alignment != 0 {
            return Err(invalid(
                "GGUF",
                format!("tensor {name:?} offset is not aligned to {alignment}"),
            ));
        }
        if index == 0 && *offset != 0 {
            return Err(invalid(
                "GGUF",
                "the first tensor must start at offset zero",
            ));
        }
        let next_offset = descriptors
            .get(index + 1)
            .map(|entry| entry.3)
            .unwrap_or(source_size - data_start);
        if *offset > next_offset || next_offset > source_size - data_start {
            return Err(invalid("GGUF", "tensor offsets overlap or exceed EOF"));
        }
        let length = ggml_tensor_bytes(shape, *data_type)?;
        let tensor_end = offset
            .checked_add(length)
            .ok_or(ModelFormatError::Overflow("GGUF tensor range"))?;
        if tensor_end > next_offset {
            return Err(invalid(
                "GGUF",
                format!("tensor {name:?} overlaps the next tensor or exceeds EOF"),
            ));
        }
        if index + 1 < descriptors.len() {
            if align_up(tensor_end, alignment)? != next_offset {
                return Err(invalid(
                    "GGUF",
                    format!("tensor {name:?} has unaccounted data before the next tensor"),
                ));
            }
        } else {
            let aligned_end = align_up(tensor_end, alignment)?;
            if next_offset != tensor_end && next_offset != aligned_end {
                return Err(invalid(
                    "GGUF",
                    format!("tensor {name:?} has unaccounted trailing data"),
                ));
            }
        }
        limits.check("GGUF tensor bytes", length, limits.maximum_tensor_bytes)?;
        aggregate = aggregate
            .checked_add(length)
            .ok_or(ModelFormatError::Overflow("GGUF aggregate tensor bytes"))?;
        limits.check(
            "GGUF aggregate tensor bytes",
            aggregate,
            limits.maximum_aggregate_tensor_bytes,
        )?;
        tensors.push(TensorMetadata {
            name: name.clone(),
            data_type: format!("GGML_TYPE_{data_type}"),
            shape: shape.clone(),
            storage: FileSlice {
                path: path.to_path_buf(),
                offset: data_start
                    .checked_add(*offset)
                    .ok_or(ModelFormatError::Overflow("GGUF tensor offset"))?,
                length,
            },
        });
    }
    let source_sha256 = sha256_open_file(file, path, cancellation)?;
    Ok(ParsedModel {
        format: ModelFormat::Gguf,
        tensors,
        payload: ParsedModelPayload::Gguf { version, metadata },
        source_size,
        source_sha256,
    })
}

fn parse_json_file(
    file: &mut File,
    path: &Path,
    format: ModelFormat,
    limits: &ParserLimits,
    cancellation: &CancellationToken,
) -> Result<ParsedModel, ModelFormatError> {
    let source_size = file_length(file, path)?;
    let bytes = read_exact_bounded(
        file,
        path,
        source_size,
        limits.manifest_bytes,
        "JSON model document",
    )?;
    check_cancelled(cancellation)?;
    let value = parse_strict_json(&bytes, limits)?;
    let source_sha256 = hex_digest(Sha256::digest(&bytes));
    Ok(ParsedModel {
        format,
        tensors: Vec::new(),
        payload: ParsedModelPayload::Json(value),
        source_size,
        source_sha256,
    })
}

fn parse_yaml_file(
    file: &mut File,
    path: &Path,
    limits: &ParserLimits,
    cancellation: &CancellationToken,
) -> Result<ParsedModel, ModelFormatError> {
    let source_size = file_length(file, path)?;
    let bytes = read_exact_bounded(
        file,
        path,
        source_size,
        limits.manifest_bytes,
        "YAML model configuration",
    )?;
    check_cancelled(cancellation)?;
    let value = parse_bounded_yaml(&bytes, limits)?;
    Ok(ParsedModel {
        format: ModelFormat::YamlConfig,
        tensors: Vec::new(),
        payload: ParsedModelPayload::Yaml(value),
        source_size,
        source_sha256: hex_digest(Sha256::digest(&bytes)),
    })
}

fn parse_sentencepiece(
    file: &mut File,
    path: &Path,
    limits: &ParserLimits,
    cancellation: &CancellationToken,
) -> Result<ParsedModel, ModelFormatError> {
    let source_size = file_length(file, path)?;
    let bytes = read_exact_bounded(
        file,
        path,
        source_size,
        limits.manifest_bytes,
        "SentencePiece model",
    )?;
    let vocabulary = parse_sentencepiece_model(&bytes, limits, cancellation)?;
    Ok(ParsedModel {
        format: ModelFormat::SentencePiece,
        tensors: Vec::new(),
        payload: ParsedModelPayload::SentencePiece { vocabulary },
        source_size,
        source_sha256: hex_digest(Sha256::digest(&bytes)),
    })
}

fn parse_tiktoken(
    file: &mut File,
    path: &Path,
    limits: &ParserLimits,
    cancellation: &CancellationToken,
) -> Result<ParsedModel, ModelFormatError> {
    let source_size = file_length(file, path)?;
    let bytes = read_exact_bounded(
        file,
        path,
        source_size,
        limits.manifest_bytes,
        "tiktoken ranks",
    )?;
    let text =
        std::str::from_utf8(&bytes).map_err(|_| invalid("tiktoken", "rank file is not UTF-8"))?;
    let mut token_count = 0_u64;
    let mut ranks = BTreeSet::new();
    let mut tokens = BTreeSet::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        check_cancelled(cancellation)?;
        let mut fields = line.split_ascii_whitespace();
        let token = fields
            .next()
            .ok_or_else(|| invalid("tiktoken", "token is missing"))?;
        let rank = fields
            .next()
            .ok_or_else(|| invalid("tiktoken", "rank is missing"))?;
        if fields.next().is_some() {
            return Err(invalid("tiktoken", "invalid token/rank row"));
        }
        check_name(token, limits, "tiktoken token")?;
        let decoded = decode_standard_base64(token.as_bytes(), limits)?;
        if decoded.is_empty() {
            return Err(invalid("tiktoken", "decoded token is empty"));
        }
        if !tokens.insert(decoded) {
            return Err(invalid("tiktoken", "duplicate decoded token"));
        }
        let rank = rank
            .parse::<u64>()
            .map_err(|error| invalid("tiktoken", error.to_string()))?;
        if !ranks.insert(rank) {
            return Err(invalid("tiktoken", format!("duplicate rank {rank}")));
        }
        token_count = token_count
            .checked_add(1)
            .ok_or(ModelFormatError::Overflow("tiktoken token count"))?;
        limits.check(
            "tiktoken token count",
            token_count,
            limits.maximum_metadata_values,
        )?;
    }
    Ok(ParsedModel {
        format: ModelFormat::Tiktoken,
        tensors: Vec::new(),
        payload: ParsedModelPayload::Tiktoken { token_count },
        source_size,
        source_sha256: hex_digest(Sha256::digest(&bytes)),
    })
}

fn parse_stored_zip(
    file: &mut File,
    path: &Path,
    limits: &ParserLimits,
    cancellation: &CancellationToken,
) -> Result<Vec<ArchiveEntry>, ModelFormatError> {
    let length = file_length(file, path)?;
    let search_length = length.min(65_557);
    let trailer_start = length - search_length;
    file.seek(SeekFrom::End(
        -i64::try_from(search_length)
            .map_err(|_| ModelFormatError::Overflow("ZIP trailer search"))?,
    ))
    .map_err(|error| io_error(path, error))?;
    let trailer = read_exact_bounded(file, path, search_length, 65_557, "ZIP trailer")?;
    let eocd_index = trailer
        .windows(4)
        .rposition(|window| window == [0x50, 0x4b, 0x05, 0x06])
        .ok_or_else(|| invalid("PyTorch ZIP", "end-of-central-directory is missing"))?;
    let eocd = trailer
        .get(eocd_index..)
        .ok_or_else(|| invalid("PyTorch ZIP", "invalid end-of-central-directory"))?;
    if eocd.len() < 22 {
        return Err(invalid("PyTorch ZIP", "truncated end-of-central-directory"));
    }
    let disk = le_u16(eocd, 4)?;
    let central_disk = le_u16(eocd, 6)?;
    let entries_on_disk_16 = le_u16(eocd, 8)?;
    let entry_count_16 = le_u16(eocd, 10)?;
    let central_size_32 = le_u32(eocd, 12)?;
    let central_offset_32 = le_u32(eocd, 16)?;
    if disk != 0 || central_disk != 0 {
        return Err(invalid("PyTorch ZIP", "multi-disk archives are forbidden"));
    }
    let zip64 = entries_on_disk_16 == u16::MAX
        || entry_count_16 == u16::MAX
        || central_size_32 == u32::MAX
        || central_offset_32 == u32::MAX;
    let (entries_on_disk, entry_count, central_size, central_offset) = if zip64 {
        let eocd_absolute =
            trailer_start
                .checked_add(u64::try_from(eocd_index).map_err(|_| {
                    ModelFormatError::Overflow("ZIP end-of-central-directory offset")
                })?)
                .ok_or(ModelFormatError::Overflow(
                    "ZIP end-of-central-directory offset",
                ))?;
        read_zip64_directory(file, path, eocd_absolute, length)?
    } else {
        (
            u64::from(entries_on_disk_16),
            u64::from(entry_count_16),
            u64::from(central_size_32),
            u64::from(central_offset_32),
        )
    };
    if entries_on_disk != entry_count {
        return Err(invalid("PyTorch ZIP", "multi-disk ZIP64 is forbidden"));
    }
    limits.check(
        "PyTorch archive entries",
        entry_count,
        limits.maximum_archive_entries,
    )?;
    limits.check(
        "PyTorch central directory bytes",
        central_size,
        limits.manifest_bytes,
    )?;
    let central_end = central_offset
        .checked_add(central_size)
        .ok_or(ModelFormatError::Overflow("ZIP central directory"))?;
    if central_end > length {
        return Err(invalid("PyTorch ZIP", "central directory exceeds EOF"));
    }
    file.seek(SeekFrom::Start(central_offset))
        .map_err(|error| io_error(path, error))?;
    let central = read_exact_bounded(
        file,
        path,
        central_size,
        limits.manifest_bytes,
        "PyTorch central directory",
    )?;
    let mut cursor = 0_usize;
    let mut entries = Vec::new();
    let entry_capacity = usize_from_u64(entry_count, "PyTorch archive entry count")?;
    entries
        .try_reserve(entry_capacity)
        .map_err(|_| ModelFormatError::AllocationFailed {
            context: "PyTorch archive entries",
            requested: entry_capacity,
        })?;
    let mut canonical_paths = BTreeSet::new();
    let mut aggregate_entry_bytes = 0_u64;
    for _ in 0..entry_count {
        check_cancelled(cancellation)?;
        let header = central
            .get(cursor..cursor.saturating_add(46))
            .ok_or_else(|| invalid("PyTorch ZIP", "truncated central entry"))?;
        if header.get(..4) != Some(&[0x50, 0x4b, 0x01, 0x02]) {
            return Err(invalid("PyTorch ZIP", "invalid central entry signature"));
        }
        let flags = le_u16(header, 8)?;
        let method = le_u16(header, 10)?;
        let crc32 = le_u32(header, 16)?;
        let compressed_32 = le_u32(header, 20)?;
        let uncompressed_32 = le_u32(header, 24)?;
        let name_length = usize::from(le_u16(header, 28)?);
        let extra_length = usize::from(le_u16(header, 30)?);
        let comment_length = usize::from(le_u16(header, 32)?);
        let disk_start = le_u16(header, 34)?;
        let external_attributes = le_u32(header, 38)?;
        let local_offset_32 = le_u32(header, 42)?;
        if flags & 1 != 0 {
            return Err(invalid("PyTorch ZIP", "encrypted entries are forbidden"));
        }
        if disk_start != 0 {
            return Err(invalid("PyTorch ZIP", "multi-disk entries are forbidden"));
        }
        let record_length = 46_usize
            .checked_add(name_length)
            .and_then(|value| value.checked_add(extra_length))
            .and_then(|value| value.checked_add(comment_length))
            .ok_or(ModelFormatError::Overflow("ZIP entry length"))?;
        let record = central
            .get(cursor..cursor.saturating_add(record_length))
            .ok_or_else(|| invalid("PyTorch ZIP", "truncated central entry fields"))?;
        let name_bytes = record
            .get(46..46 + name_length)
            .ok_or_else(|| invalid("PyTorch ZIP", "truncated entry name"))?;
        let extra = record
            .get(46 + name_length..46 + name_length + extra_length)
            .ok_or_else(|| invalid("PyTorch ZIP", "truncated entry extra fields"))?;
        let (uncompressed, compressed, local_offset) =
            zip64_entry_values(extra, uncompressed_32, compressed_32, local_offset_32)?;
        let name = std::str::from_utf8(name_bytes)
            .map_err(|_| invalid("PyTorch ZIP", "entry name is not UTF-8"))?;
        check_name(name, limits, "PyTorch archive path")?;
        let canonical = canonical_archive_path(name)?;
        if !canonical_paths.insert(canonical.clone()) {
            return Err(ModelFormatError::DuplicateArchivePath(canonical));
        }
        let unix_mode = external_attributes >> 16;
        if unix_mode & 0o170000 == 0o120000 {
            return Err(ModelFormatError::ArchiveLink(name.to_owned()));
        }
        if name.ends_with('/') {
            cursor += record_length;
            continue;
        }
        if method != 0 {
            return Err(ModelFormatError::UnsupportedCompression {
                name: name.to_owned(),
                method,
            });
        }
        if compressed != uncompressed {
            return Err(invalid(
                "PyTorch ZIP",
                "stored entry has different compressed and uncompressed sizes",
            ));
        }
        limits.check(
            "PyTorch archive entry bytes",
            uncompressed,
            limits.maximum_aggregate_tensor_bytes,
        )?;
        aggregate_entry_bytes =
            aggregate_entry_bytes
                .checked_add(uncompressed)
                .ok_or(ModelFormatError::Overflow(
                    "PyTorch archive aggregate entry bytes",
                ))?;
        limits.check(
            "PyTorch archive aggregate entry bytes",
            aggregate_entry_bytes,
            limits.maximum_aggregate_tensor_bytes,
        )?;
        let data_offset = local_data_offset(
            file,
            path,
            local_offset,
            name_bytes,
            length,
            flags,
            method,
            crc32,
            compressed,
            uncompressed,
        )?;
        let data_end = data_offset
            .checked_add(compressed)
            .ok_or(ModelFormatError::Overflow("ZIP entry data"))?;
        if data_end > length {
            return Err(invalid("PyTorch ZIP", "entry data exceeds EOF"));
        }
        entries.push(ArchiveEntry {
            name: name.to_owned(),
            data_offset,
            length: uncompressed,
            crc32,
        });
        cursor += record_length;
    }
    if cursor != central.len() {
        return Err(invalid(
            "PyTorch ZIP",
            "central directory has unaccounted bytes",
        ));
    }
    for entry in &entries {
        check_cancelled(cancellation)?;
        let actual_crc = crc32_file_range(file, path, entry, cancellation)?;
        if actual_crc != entry.crc32 {
            return Err(invalid(
                "PyTorch ZIP",
                format!("archive entry {:?} CRC32 mismatch", entry.name),
            ));
        }
    }
    Ok(entries)
}

fn read_zip64_directory(
    file: &mut File,
    path: &Path,
    eocd_offset: u64,
    file_length: u64,
) -> Result<(u64, u64, u64, u64), ModelFormatError> {
    let locator_offset = eocd_offset
        .checked_sub(20)
        .ok_or_else(|| invalid("PyTorch ZIP", "ZIP64 locator is missing"))?;
    file.seek(SeekFrom::Start(locator_offset))
        .map_err(|error| io_error(path, error))?;
    let locator = read_array::<20>(file, path)?;
    if locator.get(..4) != Some(&[0x50, 0x4b, 0x06, 0x07]) {
        return Err(invalid("PyTorch ZIP", "ZIP64 locator signature is missing"));
    }
    if le_u32(&locator, 4)? != 0 || le_u32(&locator, 16)? != 1 {
        return Err(invalid("PyTorch ZIP", "multi-disk ZIP64 is forbidden"));
    }
    let directory_offset = le_u64(&locator, 8)?;
    let minimum_end = directory_offset
        .checked_add(56)
        .ok_or(ModelFormatError::Overflow("ZIP64 directory record"))?;
    if minimum_end > file_length {
        return Err(invalid("PyTorch ZIP", "ZIP64 directory record exceeds EOF"));
    }
    file.seek(SeekFrom::Start(directory_offset))
        .map_err(|error| io_error(path, error))?;
    let record = read_array::<56>(file, path)?;
    if record.get(..4) != Some(&[0x50, 0x4b, 0x06, 0x06]) {
        return Err(invalid(
            "PyTorch ZIP",
            "ZIP64 end-of-central-directory signature is missing",
        ));
    }
    let record_size = le_u64(&record, 4)?;
    if record_size < 44 {
        return Err(invalid(
            "PyTorch ZIP",
            "ZIP64 end-of-central-directory is too short",
        ));
    }
    let record_end = directory_offset
        .checked_add(12)
        .and_then(|value| value.checked_add(record_size))
        .ok_or(ModelFormatError::Overflow("ZIP64 directory record"))?;
    if record_end > locator_offset {
        return Err(invalid(
            "PyTorch ZIP",
            "ZIP64 directory record overlaps its locator",
        ));
    }
    if le_u32(&record, 16)? != 0 || le_u32(&record, 20)? != 0 {
        return Err(invalid("PyTorch ZIP", "multi-disk ZIP64 is forbidden"));
    }
    Ok((
        le_u64(&record, 24)?,
        le_u64(&record, 32)?,
        le_u64(&record, 40)?,
        le_u64(&record, 48)?,
    ))
}

fn zip64_entry_values(
    extra: &[u8],
    uncompressed_32: u32,
    compressed_32: u32,
    local_offset_32: u32,
) -> Result<(u64, u64, u64), ModelFormatError> {
    let needs_zip64 =
        uncompressed_32 == u32::MAX || compressed_32 == u32::MAX || local_offset_32 == u32::MAX;
    if !needs_zip64 {
        return Ok((
            u64::from(uncompressed_32),
            u64::from(compressed_32),
            u64::from(local_offset_32),
        ));
    }
    let mut cursor = 0_usize;
    while cursor < extra.len() {
        let identifier = le_u16(extra, cursor)?;
        let size = usize::from(le_u16(extra, cursor.saturating_add(2))?);
        let data_start = cursor
            .checked_add(4)
            .ok_or(ModelFormatError::Overflow("ZIP extra field"))?;
        let data_end = data_start
            .checked_add(size)
            .ok_or(ModelFormatError::Overflow("ZIP extra field"))?;
        let data = extra
            .get(data_start..data_end)
            .ok_or_else(|| invalid("PyTorch ZIP", "truncated extra field"))?;
        if identifier == 0x0001 {
            let mut value_cursor = 0_usize;
            let uncompressed = if uncompressed_32 == u32::MAX {
                let value = le_u64(data, value_cursor)?;
                value_cursor += 8;
                value
            } else {
                u64::from(uncompressed_32)
            };
            let compressed = if compressed_32 == u32::MAX {
                let value = le_u64(data, value_cursor)?;
                value_cursor += 8;
                value
            } else {
                u64::from(compressed_32)
            };
            let local_offset = if local_offset_32 == u32::MAX {
                le_u64(data, value_cursor)?
            } else {
                u64::from(local_offset_32)
            };
            return Ok((uncompressed, compressed, local_offset));
        }
        cursor = data_end;
    }
    Err(invalid("PyTorch ZIP", "ZIP64 entry values are missing"))
}

fn local_data_offset(
    file: &mut File,
    path: &Path,
    local_offset: u64,
    expected_name: &[u8],
    file_length: u64,
    expected_flags: u16,
    expected_method: u16,
    expected_crc32: u32,
    expected_compressed: u64,
    expected_uncompressed: u64,
) -> Result<u64, ModelFormatError> {
    file.seek(SeekFrom::Start(local_offset))
        .map_err(|error| io_error(path, error))?;
    let mut header = [0_u8; 30];
    read_exact(file, path, &mut header)?;
    if header.get(..4) != Some(&[0x50, 0x4b, 0x03, 0x04]) {
        return Err(invalid("PyTorch ZIP", "invalid local entry signature"));
    }
    let flags = le_u16(&header, 6)?;
    let method = le_u16(&header, 8)?;
    let crc32 = le_u32(&header, 14)?;
    let compressed = le_u32(&header, 18)?;
    let uncompressed = le_u32(&header, 22)?;
    if flags != expected_flags || method != expected_method {
        return Err(invalid(
            "PyTorch ZIP",
            "central and local entry flags or compression method differ",
        ));
    }
    let uses_data_descriptor = flags & 0x0008 != 0;
    if !uses_data_descriptor {
        if crc32 != expected_crc32
            || !zip_local_size_matches(compressed, expected_compressed)
            || !zip_local_size_matches(uncompressed, expected_uncompressed)
        {
            return Err(invalid(
                "PyTorch ZIP",
                "central and local entry integrity fields differ",
            ));
        }
    } else if !(crc32 == 0 || crc32 == expected_crc32)
        || !zip_local_descriptor_size_matches(compressed, expected_compressed)
        || !zip_local_descriptor_size_matches(uncompressed, expected_uncompressed)
    {
        return Err(invalid(
            "PyTorch ZIP",
            "local data-descriptor placeholders are invalid",
        ));
    }
    let name_length = usize::from(le_u16(&header, 26)?);
    let extra_length = usize::from(le_u16(&header, 28)?);
    let mut name = vec![0_u8; name_length];
    read_exact(file, path, &mut name)?;
    if name != expected_name {
        return Err(invalid(
            "PyTorch ZIP",
            "central and local entry names differ",
        ));
    }
    let offset = local_offset
        .checked_add(30)
        .and_then(|value| value.checked_add(u64::try_from(name_length).ok()?))
        .and_then(|value| value.checked_add(u64::try_from(extra_length).ok()?))
        .ok_or(ModelFormatError::Overflow("ZIP local data offset"))?;
    if offset > file_length {
        return Err(invalid("PyTorch ZIP", "local entry exceeds EOF"));
    }
    Ok(offset)
}

fn zip_local_size_matches(local: u32, expected: u64) -> bool {
    local == u32::MAX || u64::from(local) == expected
}

fn zip_local_descriptor_size_matches(local: u32, expected: u64) -> bool {
    local == 0 || zip_local_size_matches(local, expected)
}

fn crc32_file_range(
    file: &mut File,
    path: &Path,
    entry: &ArchiveEntry,
    cancellation: &CancellationToken,
) -> Result<u32, ModelFormatError> {
    file.seek(SeekFrom::Start(entry.data_offset))
        .map_err(|error| io_error(path, error))?;
    let mut remaining = entry.length;
    let mut state = u32::MAX;
    let mut buffer = [0_u8; 1024 * 1024];
    while remaining > 0 {
        check_cancelled(cancellation)?;
        let length = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| ModelFormatError::Overflow("ZIP CRC32 chunk"))?;
        let chunk = buffer
            .get_mut(..length)
            .ok_or(ModelFormatError::Overflow("ZIP CRC32 chunk"))?;
        file.read_exact(chunk)
            .map_err(|error| io_error(path, error))?;
        state = crc32_update(state, chunk);
        remaining -= u64::try_from(length)
            .map_err(|_| ModelFormatError::Overflow("ZIP CRC32 remaining bytes"))?;
    }
    Ok(!state)
}

fn tensor_metadata_from_pickle(
    path: &Path,
    root: &PickleValue,
    entries: &[ArchiveEntry],
    archive_root: Option<&str>,
    limits: &ParserLimits,
) -> Result<Vec<TensorMetadata>, ModelFormatError> {
    let mapping = nested_string_to_param_mapping(root).or_else(|| state_dict_mapping(root));
    let Some(mapping) = mapping else {
        return Ok(Vec::new());
    };
    let mut result = Vec::new();
    let mut aggregate = 0_u64;
    let mut rebuild_count = 0_u64;
    for (key, value) in mapping {
        let PickleValue::String(name) = key else {
            return Err(invalid(
                "PyTorch archive",
                "state_dict contains a non-string key",
            ));
        };
        let Some(rebuild) = tensor_rebuild(value, limits.maximum_depth) else {
            if value.reduction_target().is_some_and(|target| {
                target.starts_with("torch._utils._rebuild")
                    || target.starts_with("torch._tensor._rebuild")
            }) {
                return Err(invalid(
                    "PyTorch archive",
                    format!("state_dict value {name:?} is not a supported tensor rebuild"),
                ));
            }
            continue;
        };
        check_name(name, limits, "PyTorch tensor name")?;
        let archive_root = archive_root.ok_or_else(|| {
            invalid(
                "PyTorch archive",
                "tensor storage requires a rooted ZIP archive",
            )
        })?;
        let validated = validate_pytorch_rebuild(
            &rebuild,
            name,
            entries,
            archive_root,
            limits,
            &mut rebuild_count,
            &mut aggregate,
        )?;
        result.push(TensorMetadata {
            name: name.clone(),
            data_type: rebuild.data_type,
            shape: rebuild.shape,
            storage: FileSlice {
                path: path.to_path_buf(),
                offset: validated
                    .entry
                    .data_offset
                    .checked_add(validated.tensor_start)
                    .ok_or(ModelFormatError::Overflow("PyTorch tensor file offset"))?,
                length: validated.tensor_bytes,
            },
        });
    }
    limits.check(
        "PyTorch tensor count",
        u64::try_from(result.len()).unwrap_or(u64::MAX),
        limits.maximum_tensors,
    )?;
    Ok(result)
}

fn nested_string_to_param_mapping(value: &PickleValue) -> Option<&Vec<(PickleValue, PickleValue)>> {
    let PickleValue::Dictionary(entries) = value else {
        return None;
    };
    if let Some((_, PickleValue::Dictionary(mapping))) = entries
        .iter()
        .find(|(key, _)| matches!(key, PickleValue::String(name) if name == "string_to_param"))
    {
        return Some(mapping);
    }
    let state_dict = entries.iter().find_map(|(key, value)| {
        matches!(key, PickleValue::String(name) if name == "state_dict").then_some(value)
    })?;
    nested_string_to_param_mapping(state_dict)
}

fn state_dict_mapping(value: &PickleValue) -> Option<&Vec<(PickleValue, PickleValue)>> {
    let PickleValue::Dictionary(entries) = value else {
        return None;
    };
    if let Some((_, PickleValue::Dictionary(state_dict))) = entries
        .iter()
        .find(|(key, _)| matches!(key, PickleValue::String(name) if name == "state_dict"))
    {
        return Some(state_dict);
    }
    if entries.len() == 1 {
        if let Some((_, PickleValue::Dictionary(inner))) = entries.first() {
            return Some(inner);
        }
    }
    Some(entries)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PytorchTensorRebuild {
    storage_key: String,
    data_type: String,
    storage_offset: u64,
    storage_elements: u64,
    shape: Vec<u64>,
    auxiliary: Vec<PytorchTensorRebuild>,
}

struct ValidatedPytorchRebuild<'a> {
    entry: &'a ArchiveEntry,
    tensor_start: u64,
    tensor_bytes: u64,
}

fn tensor_rebuild(value: &PickleValue, maximum_depth: u32) -> Option<PytorchTensorRebuild> {
    tensor_rebuild_with_depth(value, maximum_depth)
}

fn tensor_rebuild_with_depth(
    value: &PickleValue,
    remaining_depth: u32,
) -> Option<PytorchTensorRebuild> {
    if remaining_depth == 0 {
        return None;
    }
    let PickleValue::Reduced {
        target, arguments, ..
    } = value
    else {
        return None;
    };
    let PickleValue::Tuple(arguments) = arguments.as_ref() else {
        return None;
    };
    match target.as_str() {
        "torch._utils._rebuild_tensor" => {
            tensor_rebuild_from_arguments(arguments, 4, false, remaining_depth - 1)
        }
        "torch._utils._rebuild_tensor_v2" => {
            tensor_rebuild_from_arguments(arguments, 6, false, remaining_depth - 1)
        }
        "torch._utils._rebuild_tensor_v3" if matches!(arguments.len(), 7 | 8) => {
            tensor_rebuild_from_arguments(arguments, arguments.len(), false, remaining_depth - 1)
        }
        "torch._utils._rebuild_qtensor" => {
            tensor_rebuild_from_arguments(arguments, 7, true, remaining_depth - 1)
        }
        "torch._utils._rebuild_parameter" if arguments.len() == 3 => arguments
            .first()
            .and_then(|value| tensor_rebuild_with_depth(value, remaining_depth - 1)),
        "torch._utils._rebuild_parameter_with_state" if arguments.len() == 4 => arguments
            .first()
            .and_then(|value| tensor_rebuild_with_depth(value, remaining_depth - 1)),
        "torch.nn.parameter.Parameter" if matches!(arguments.len(), 1 | 2) => arguments
            .first()
            .and_then(|value| tensor_rebuild_with_depth(value, remaining_depth - 1)),
        "torch._tensor._rebuild_from_type_v2" if arguments.len() == 4 => {
            tensor_rebuild_from_type(arguments, remaining_depth - 1)
        }
        _ => None,
    }
}

fn tensor_rebuild_from_type(
    arguments: &[PickleValue],
    remaining_depth: u32,
) -> Option<PytorchTensorRebuild> {
    let target = match arguments.first()? {
        PickleValue::Global(target) => target,
        _ => return None,
    };
    match arguments.get(1)? {
        PickleValue::Global(target)
            if matches!(
                target.as_str(),
                "torch.Tensor" | "torch.nn.parameter.Parameter"
            ) => {}
        _ => return None,
    }
    let PickleValue::Tuple(rebuild_arguments) = arguments.get(2)? else {
        return None;
    };
    match target.as_str() {
        "torch._utils._rebuild_tensor" => {
            tensor_rebuild_from_arguments(rebuild_arguments, 4, false, remaining_depth)
        }
        "torch._utils._rebuild_tensor_v2" => {
            tensor_rebuild_from_arguments(rebuild_arguments, 6, false, remaining_depth)
        }
        "torch._utils._rebuild_tensor_v3" if matches!(rebuild_arguments.len(), 7 | 8) => {
            tensor_rebuild_from_arguments(
                rebuild_arguments,
                rebuild_arguments.len(),
                false,
                remaining_depth,
            )
        }
        "torch._utils._rebuild_qtensor" => {
            tensor_rebuild_from_arguments(rebuild_arguments, 7, true, remaining_depth)
        }
        _ => None,
    }
}

fn tensor_rebuild_from_arguments(
    arguments: &[PickleValue],
    expected_arguments: usize,
    quantized: bool,
    remaining_depth: u32,
) -> Option<PytorchTensorRebuild> {
    if arguments.len() != expected_arguments {
        return None;
    }
    let storage = arguments.first()?;
    let storage_offset = match arguments.get(1)? {
        PickleValue::Integer(value) => u64::try_from(*value).ok()?,
        _ => return None,
    };
    let shape = arguments.get(2)?;
    let PickleValue::Persistent(storage) = storage else {
        return None;
    };
    let PickleValue::Tuple(storage) = storage.as_ref() else {
        return None;
    };
    if storage.len() != 5
        || !matches!(storage.first(), Some(PickleValue::String(kind)) if kind == "storage")
    {
        return None;
    }
    let data_type = match storage.get(1)? {
        PickleValue::Global(target) => target.clone(),
        _ => return None,
    };
    let storage_key = match storage.get(2)? {
        PickleValue::String(value) => value.clone(),
        _ => return None,
    };
    if !matches!(storage.get(3), Some(PickleValue::String(_))) {
        return None;
    }
    let storage_elements = match storage.get(4)? {
        PickleValue::Integer(value) => u64::try_from(*value).ok()?,
        _ => return None,
    };
    let (PickleValue::Tuple(shape) | PickleValue::List(shape)) = shape else {
        return None;
    };
    let shape = shape
        .iter()
        .map(|value| match value {
            PickleValue::Integer(value) => u64::try_from(*value).ok(),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    let stride = match arguments.get(3)? {
        PickleValue::Tuple(stride) | PickleValue::List(stride) => stride,
        _ => return None,
    };
    if !is_contiguous_pytorch_stride(&shape, stride) {
        return None;
    }
    let auxiliary = if quantized {
        if !matches!(
            data_type.as_str(),
            "torch.QInt8Storage" | "torch.QUInt8Storage" | "torch.QInt32Storage"
        ) {
            return None;
        }
        quantized_rebuild_auxiliary(arguments.get(4)?, remaining_depth)?
    } else {
        Vec::new()
    };
    Some(PytorchTensorRebuild {
        storage_key,
        data_type,
        storage_offset,
        storage_elements,
        shape,
        auxiliary,
    })
}

fn is_contiguous_pytorch_stride(shape: &[u64], stride: &[PickleValue]) -> bool {
    if shape.len() != stride.len() {
        return false;
    }
    let mut expected = 1_u64;
    for (dimension, stride) in shape.iter().zip(stride.iter()).rev() {
        let PickleValue::Integer(stride) = stride else {
            return false;
        };
        let Ok(stride) = u64::try_from(*stride) else {
            return false;
        };
        if *dimension > 1 && stride != expected {
            return false;
        }
        let Some(next) = expected.checked_mul(*dimension) else {
            return false;
        };
        expected = next;
    }
    true
}

fn quantized_rebuild_auxiliary(
    value: &PickleValue,
    remaining_depth: u32,
) -> Option<Vec<PytorchTensorRebuild>> {
    let PickleValue::Tuple(parameters) = value else {
        return None;
    };
    let qscheme = match parameters.first()? {
        PickleValue::Global(qscheme) => qscheme.as_str(),
        _ => return None,
    };
    match qscheme {
        "torch.per_tensor_affine" | "torch.per_tensor_symmetric" => {
            if parameters.len() != 3
                || !positive_finite_pickle_number(parameters.get(1)?)
                || !matches!(parameters.get(2), Some(PickleValue::Integer(value)) if i64::try_from(*value).is_ok())
            {
                return None;
            }
            Some(Vec::new())
        }
        "torch.per_channel_affine"
        | "torch.per_channel_affine_float_qparams"
        | "torch.per_channel_symmetric" => {
            if parameters.len() != 4
                || !matches!(parameters.get(3), Some(PickleValue::Integer(value)) if u64::try_from(*value).is_ok())
            {
                return None;
            }
            let scales = tensor_rebuild_with_depth(parameters.get(1)?, remaining_depth)?;
            let zero_points = tensor_rebuild_with_depth(parameters.get(2)?, remaining_depth)?;
            if scales.shape.len() != 1
                || scales.shape != zero_points.shape
                || scales.shape.first().copied().unwrap_or(0) == 0
            {
                return None;
            }
            Some(vec![scales, zero_points])
        }
        _ => None,
    }
}

fn positive_finite_pickle_number(value: &PickleValue) -> bool {
    match value {
        PickleValue::FloatBits(bits) => {
            let value = f64::from_bits(*bits);
            value.is_finite() && value > 0.0
        }
        PickleValue::Integer(value) => *value > 0,
        _ => false,
    }
}

fn validate_pytorch_rebuild<'a>(
    rebuild: &PytorchTensorRebuild,
    tensor_name: &str,
    entries: &'a [ArchiveEntry],
    archive_root: &str,
    limits: &ParserLimits,
    tensor_count: &mut u64,
    aggregate: &mut u64,
) -> Result<ValidatedPytorchRebuild<'a>, ModelFormatError> {
    for auxiliary in &rebuild.auxiliary {
        validate_pytorch_rebuild(
            auxiliary,
            tensor_name,
            entries,
            archive_root,
            limits,
            tensor_count,
            aggregate,
        )?;
    }
    *tensor_count = tensor_count
        .checked_add(1)
        .ok_or(ModelFormatError::Overflow("PyTorch tensor count"))?;
    limits.check(
        "PyTorch tensor count",
        *tensor_count,
        limits.maximum_tensors,
    )?;
    let expected_storage_name = format!("{archive_root}data/{}", rebuild.storage_key);
    let entry = entries
        .iter()
        .find(|entry| entry.name == expected_storage_name)
        .ok_or_else(|| {
            invalid(
                "PyTorch archive",
                format!(
                    "storage {:?} for tensor {tensor_name:?} is missing",
                    rebuild.storage_key
                ),
            )
        })?;
    let bytes_per_element = pytorch_storage_element_bytes(&rebuild.data_type).ok_or_else(|| {
        invalid(
            "PyTorch archive",
            format!("unsupported storage type {:?}", rebuild.data_type),
        )
    })?;
    let expected_storage_bytes = rebuild
        .storage_elements
        .checked_mul(bytes_per_element)
        .ok_or(ModelFormatError::Overflow("PyTorch storage bytes"))?;
    if expected_storage_bytes != entry.length {
        return Err(invalid(
            "PyTorch archive",
            format!(
                "storage {:?} declares {expected_storage_bytes} bytes but archive has {}",
                rebuild.storage_key, entry.length
            ),
        ));
    }
    let tensor_bytes = checked_product(&rebuild.shape, "PyTorch tensor shape")?
        .checked_mul(bytes_per_element)
        .ok_or(ModelFormatError::Overflow("PyTorch tensor bytes"))?;
    let tensor_start = rebuild
        .storage_offset
        .checked_mul(bytes_per_element)
        .ok_or(ModelFormatError::Overflow("PyTorch tensor storage offset"))?;
    let tensor_end = tensor_start
        .checked_add(tensor_bytes)
        .ok_or(ModelFormatError::Overflow("PyTorch tensor storage range"))?;
    if tensor_end > entry.length {
        return Err(invalid(
            "PyTorch archive",
            format!(
                "tensor {tensor_name:?} exceeds storage {:?}",
                rebuild.storage_key
            ),
        ));
    }
    limits.check(
        "PyTorch tensor bytes",
        tensor_bytes,
        limits.maximum_tensor_bytes,
    )?;
    *aggregate = aggregate
        .checked_add(tensor_bytes)
        .ok_or(ModelFormatError::Overflow("PyTorch aggregate tensor bytes"))?;
    limits.check(
        "PyTorch aggregate tensor bytes",
        *aggregate,
        limits.maximum_aggregate_tensor_bytes,
    )?;
    Ok(ValidatedPytorchRebuild {
        entry,
        tensor_start,
        tensor_bytes,
    })
}

fn pytorch_storage_element_bytes(data_type: &str) -> Option<u64> {
    match data_type {
        "torch.BoolStorage"
        | "torch.ByteStorage"
        | "torch.CharStorage"
        | "torch.QInt8Storage"
        | "torch.QUInt8Storage" => Some(1),
        "torch.BFloat16Storage" | "torch.HalfStorage" | "torch.ShortStorage" => Some(2),
        "torch.FloatStorage" | "torch.IntStorage" | "torch.QInt32Storage" => Some(4),
        "torch.ComplexFloatStorage" | "torch.DoubleStorage" | "torch.LongStorage" => Some(8),
        "torch.ComplexDoubleStorage" => Some(16),
        _ => None,
    }
}

fn read_gguf_value(
    file: &mut File,
    path: &Path,
    value_type: u32,
    depth: u32,
    limits: &ParserLimits,
    budget: &mut u64,
    cancellation: &CancellationToken,
) -> Result<GgufValue, ModelFormatError> {
    check_cancelled(cancellation)?;
    if !GGUF_METADATA_TYPES.contains(&value_type) {
        return Err(invalid(
            "GGUF",
            format!("unknown metadata value type {value_type}"),
        ));
    }
    if depth > limits.maximum_depth {
        return Err(ParserLimitError::Exceeded {
            kind: "GGUF metadata depth",
            actual: u64::from(depth),
            maximum: u64::from(limits.maximum_depth),
        }
        .into());
    }
    match value_type {
        0 => {
            consume_budget(budget, 1, "GGUF manifest bytes")?;
            Ok(GgufValue::Unsigned(u64::from(read_u8(file, path)?)))
        }
        1 => {
            consume_budget(budget, 1, "GGUF manifest bytes")?;
            Ok(GgufValue::Signed(i64::from(i8::from_le_bytes([read_u8(
                file, path,
            )?]))))
        }
        2 => {
            consume_budget(budget, 2, "GGUF manifest bytes")?;
            Ok(GgufValue::Unsigned(u64::from(read_u16(file, path)?)))
        }
        3 => {
            consume_budget(budget, 2, "GGUF manifest bytes")?;
            Ok(GgufValue::Signed(i64::from(i16::from_le_bytes(
                read_array(file, path)?,
            ))))
        }
        4 => {
            consume_budget(budget, 4, "GGUF manifest bytes")?;
            Ok(GgufValue::Unsigned(u64::from(read_u32(file, path)?)))
        }
        5 => {
            consume_budget(budget, 4, "GGUF manifest bytes")?;
            Ok(GgufValue::Signed(i64::from(i32::from_le_bytes(
                read_array(file, path)?,
            ))))
        }
        6 => {
            consume_budget(budget, 4, "GGUF manifest bytes")?;
            Ok(GgufValue::FloatBits(u64::from(
                f32::from_le_bytes(read_array(file, path)?).to_bits(),
            )))
        }
        7 => {
            consume_budget(budget, 1, "GGUF manifest bytes")?;
            match read_u8(file, path)? {
                0 => Ok(GgufValue::Boolean(false)),
                1 => Ok(GgufValue::Boolean(true)),
                _ => Err(invalid("GGUF", "boolean metadata must be 0 or 1")),
            }
        }
        8 => Ok(GgufValue::String(read_gguf_string(
            file, path, limits, budget,
        )?)),
        9 => {
            consume_budget(budget, 12, "GGUF manifest bytes")?;
            let element_type = read_u32(file, path)?;
            if element_type == 9 {
                return Err(invalid("GGUF", "nested metadata arrays are forbidden"));
            }
            let count = read_u64(file, path)?;
            limits.check("GGUF array values", count, limits.maximum_metadata_values)?;
            let count = usize_from_u64(count, "GGUF array values")?;
            let mut values = Vec::new();
            values
                .try_reserve(count)
                .map_err(|_| ModelFormatError::AllocationFailed {
                    context: "GGUF metadata array",
                    requested: count,
                })?;
            for _ in 0..count {
                values.push(read_gguf_value(
                    file,
                    path,
                    element_type,
                    depth + 1,
                    limits,
                    budget,
                    cancellation,
                )?);
            }
            Ok(GgufValue::Array(values))
        }
        10 => {
            consume_budget(budget, 8, "GGUF manifest bytes")?;
            Ok(GgufValue::Unsigned(read_u64(file, path)?))
        }
        11 => {
            consume_budget(budget, 8, "GGUF manifest bytes")?;
            Ok(GgufValue::Signed(i64::from_le_bytes(read_array(
                file, path,
            )?)))
        }
        12 => {
            consume_budget(budget, 8, "GGUF manifest bytes")?;
            Ok(GgufValue::FloatBits(
                f64::from_le_bytes(read_array(file, path)?).to_bits(),
            ))
        }
        _ => Err(invalid("GGUF", "unreachable metadata type")),
    }
}

fn read_gguf_string(
    file: &mut File,
    path: &Path,
    limits: &ParserLimits,
    budget: &mut u64,
) -> Result<String, ModelFormatError> {
    let length = read_u64(file, path)?;
    limits.check("GGUF string bytes", length, limits.maximum_name_bytes)?;
    let encoded_bytes = length
        .checked_add(8)
        .ok_or(ModelFormatError::Overflow("GGUF string bytes"))?;
    consume_budget(budget, encoded_bytes, "GGUF manifest bytes")?;
    let bytes = read_exact_bounded(file, path, length, limits.maximum_name_bytes, "GGUF string")?;
    String::from_utf8(bytes).map_err(|_| invalid("GGUF", "string is not UTF-8"))
}

#[derive(Clone, Debug)]
struct YamlLine {
    indent: usize,
    value: String,
}

fn parse_bounded_yaml(
    bytes: &[u8],
    limits: &ParserLimits,
) -> Result<serde_json::Value, ModelFormatError> {
    limits.check(
        "YAML bytes",
        u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        limits.manifest_bytes,
    )?;
    let text =
        std::str::from_utf8(bytes).map_err(|_| invalid("YAML", "configuration is not UTF-8"))?;
    let budget = DecodedBudget::new(limits)?;
    let mut lines = Vec::new();
    for raw in text.lines() {
        if raw.contains('\t') {
            return Err(invalid("YAML", "tab indentation is forbidden"));
        }
        let indent = raw.bytes().take_while(|byte| *byte == b' ').count();
        let value = strip_yaml_comment(raw.get(indent..).unwrap_or_default()).trim_end();
        if value.is_empty() || matches!(value, "---" | "...") {
            continue;
        }
        if contains_forbidden_yaml_indicator(value) {
            return Err(invalid(
                "YAML",
                "tags, anchors, aliases, complex keys, and merge keys are forbidden",
            ));
        }
        limits.check(
            "YAML line bytes",
            u64::try_from(value.len()).unwrap_or(u64::MAX),
            limits.maximum_name_bytes,
        )?;
        budget.charge_model_value()?;
        budget.charge_model_bytes(value.len().saturating_add(size_of::<YamlLine>()))?;
        lines.push(YamlLine {
            indent,
            value: value.to_owned(),
        });
        limits.check(
            "YAML value count",
            u64::try_from(lines.len()).unwrap_or(u64::MAX),
            limits.maximum_metadata_values,
        )?;
    }
    if lines.is_empty() {
        return Ok(serde_json::Value::Object(serde_json::Map::new()));
    }
    if lines.first().is_some_and(|line| line.indent != 0) {
        return Err(invalid("YAML", "root indentation must be zero"));
    }
    let (value, cursor) = parse_yaml_block(&lines, 0, 0, 0, limits, &budget)?;
    if cursor != lines.len() {
        return Err(invalid("YAML", "inconsistent indentation"));
    }
    Ok(value)
}

fn parse_yaml_block(
    lines: &[YamlLine],
    mut cursor: usize,
    indent: usize,
    depth: u32,
    limits: &ParserLimits,
    budget: &DecodedBudget,
) -> Result<(serde_json::Value, usize), ModelFormatError> {
    if depth > limits.maximum_depth {
        return Err(ParserLimitError::Exceeded {
            kind: "YAML depth",
            actual: u64::from(depth),
            maximum: u64::from(limits.maximum_depth),
        }
        .into());
    }
    let sequence = lines
        .get(cursor)
        .is_some_and(|line| line.value == "-" || line.value.starts_with("- "));
    if sequence {
        let mut values = Vec::new();
        while let Some(line) = lines.get(cursor) {
            if line.indent < indent {
                break;
            }
            if line.indent != indent || !(line.value == "-" || line.value.starts_with("- ")) {
                return Err(invalid("YAML", "mixed sequence and mapping indentation"));
            }
            let remainder = line.value.strip_prefix('-').unwrap_or_default().trim();
            cursor += 1;
            let value = if remainder.is_empty() {
                let child = lines
                    .get(cursor)
                    .ok_or_else(|| invalid("YAML", "sequence item is missing a value"))?;
                if child.indent <= indent {
                    return Err(invalid("YAML", "sequence item is missing a nested value"));
                }
                let (value, next) =
                    parse_yaml_block(lines, cursor, child.indent, depth + 1, limits, budget)?;
                cursor = next;
                value
            } else {
                parse_yaml_scalar(remainder, limits, budget)?
            };
            budget.charge_model_value()?;
            values.push(value);
        }
        return Ok((serde_json::Value::Array(values), cursor));
    }

    let mut values = serde_json::Map::new();
    while let Some(line) = lines.get(cursor) {
        if line.indent < indent {
            break;
        }
        if line.indent != indent || line.value == "-" || line.value.starts_with("- ") {
            return Err(invalid("YAML", "mixed mapping and sequence indentation"));
        }
        let (key, remainder) = split_yaml_mapping(&line.value)?;
        let key = parse_yaml_key(key, limits, budget)?;
        if values.contains_key(&key) {
            return Err(invalid("YAML", format!("duplicate key {key:?}")));
        }
        cursor += 1;
        let value = if matches!(remainder, "|" | ">") {
            let folded = remainder == ">";
            let mut block = String::new();
            while let Some(child) = lines.get(cursor) {
                if child.indent <= indent {
                    break;
                }
                if !block.is_empty() {
                    block.push(if folded { ' ' } else { '\n' });
                }
                block.push_str(&child.value);
                limits.check(
                    "YAML block scalar bytes",
                    u64::try_from(block.len()).unwrap_or(u64::MAX),
                    limits.manifest_bytes,
                )?;
                cursor += 1;
            }
            budget.charge_model_bytes(block.len())?;
            serde_json::Value::String(block)
        } else if remainder.is_empty() {
            match lines.get(cursor) {
                Some(child) if child.indent > indent => {
                    let (value, next) =
                        parse_yaml_block(lines, cursor, child.indent, depth + 1, limits, budget)?;
                    cursor = next;
                    value
                }
                _ => serde_json::Value::Null,
            }
        } else {
            parse_yaml_scalar(remainder, limits, budget)?
        };
        budget.charge_model_value()?;
        budget.charge_model_bytes(key.len().saturating_add(size_of::<serde_json::Value>()))?;
        values.insert(key, value);
    }
    Ok((serde_json::Value::Object(values), cursor))
}

fn split_yaml_mapping(value: &str) -> Result<(&str, &str), ModelFormatError> {
    let mut single_quote = false;
    let mut double_quote = false;
    let mut escaped = false;
    for (index, character) in value.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match character {
            '\\' if double_quote => escaped = true,
            '\'' if !double_quote => single_quote = !single_quote,
            '"' if !single_quote => double_quote = !double_quote,
            ':' if !single_quote && !double_quote => {
                let key = value.get(..index).unwrap_or_default().trim();
                let remainder = value
                    .get(index + character.len_utf8()..)
                    .unwrap_or_default()
                    .trim();
                if key.is_empty() {
                    return Err(invalid("YAML", "mapping key is empty"));
                }
                return Ok((key, remainder));
            }
            _ => {}
        }
    }
    Err(invalid("YAML", "mapping entry is missing ':'"))
}

fn parse_yaml_key(
    value: &str,
    limits: &ParserLimits,
    budget: &DecodedBudget,
) -> Result<String, ModelFormatError> {
    let parsed = parse_yaml_scalar(value, limits, budget)?;
    match parsed {
        serde_json::Value::String(value) => Ok(value),
        serde_json::Value::Number(value) => Ok(value.to_string()),
        serde_json::Value::Bool(value) => Ok(value.to_string()),
        _ => Err(invalid("YAML", "mapping key must be scalar")),
    }
}

fn parse_yaml_scalar(
    value: &str,
    limits: &ParserLimits,
    budget: &DecodedBudget,
) -> Result<serde_json::Value, ModelFormatError> {
    limits.check(
        "YAML scalar bytes",
        u64::try_from(value.len()).unwrap_or(u64::MAX),
        limits.maximum_name_bytes,
    )?;
    if value.starts_with('"') {
        let parsed = parse_strict_json(value.as_bytes(), limits)?;
        if parsed.is_string() {
            budget.charge_model_bytes(parsed.as_str().map_or(0, str::len))?;
            return Ok(parsed);
        }
        return Err(invalid("YAML", "quoted scalar is invalid"));
    }
    if value.starts_with('\'') {
        if !value.ends_with('\'') || value.len() < 2 {
            return Err(invalid("YAML", "single-quoted scalar is unterminated"));
        }
        let inner = value.get(1..value.len() - 1).unwrap_or_default();
        budget.charge_model_bytes(inner.len())?;
        return Ok(serde_json::Value::String(inner.replace("''", "'")));
    }
    match value.to_ascii_lowercase().as_str() {
        "null" | "~" => return Ok(serde_json::Value::Null),
        "true" => return Ok(serde_json::Value::Bool(true)),
        "false" => return Ok(serde_json::Value::Bool(false)),
        _ => {}
    }
    if let Ok(value) = value.parse::<i64>() {
        return Ok(serde_json::Value::Number(value.into()));
    }
    if let Ok(value) = value.parse::<u64>() {
        return Ok(serde_json::Value::Number(value.into()));
    }
    if let Ok(value) = value.parse::<f64>() {
        let number = serde_json::Number::from_f64(value)
            .ok_or_else(|| invalid("YAML", "non-finite scalar is forbidden"))?;
        return Ok(serde_json::Value::Number(number));
    }
    budget.charge_model_bytes(value.len())?;
    Ok(serde_json::Value::String(value.to_owned()))
}

fn strip_yaml_comment(value: &str) -> &str {
    let mut single_quote = false;
    let mut double_quote = false;
    let mut escaped = false;
    for (index, character) in value.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match character {
            '\\' if double_quote => escaped = true,
            '\'' if !double_quote => single_quote = !single_quote,
            '"' if !single_quote => double_quote = !double_quote,
            '#' if !single_quote && !double_quote => {
                return value.get(..index).unwrap_or_default();
            }
            _ => {}
        }
    }
    value
}

pub(crate) fn parse_strict_json(
    bytes: &[u8],
    limits: &ParserLimits,
) -> Result<serde_json::Value, ModelFormatError> {
    limits.check(
        "JSON bytes",
        u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        limits.manifest_bytes,
    )?;
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let budget = DecodedBudget::new(limits)?;
    let seed = StrictJsonSeed {
        limits,
        depth: 0,
        budget: &budget,
    };
    let value = seed
        .deserialize(&mut deserializer)
        .map_err(|error| invalid("JSON", error.to_string()))?;
    deserializer
        .end()
        .map_err(|error| invalid("JSON", error.to_string()))?;
    Ok(value)
}

pub fn checked_safetensors_metadata_json(
    bytes: &[u8],
    expected_sha256: &str,
    limits: &ParserLimits,
    cancellation: &CancellationToken,
) -> Result<BTreeMap<String, String>, ModelFormatError> {
    check_cancelled(cancellation)?;
    if expected_sha256.len() != 64
        || !expected_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        || hex_digest(Sha256::digest(bytes)) != expected_sha256
    {
        return Err(invalid(
            "safetensors metadata",
            "verified metadata digest mismatch",
        ));
    }
    let document = parse_strict_json(bytes, limits)?;
    let object = document.as_object().ok_or_else(|| {
        invalid(
            "safetensors metadata",
            "verified metadata must be a JSON object",
        )
    })?;
    limits.check(
        "safetensors metadata values",
        u64::try_from(object.len()).unwrap_or(u64::MAX),
        limits.maximum_metadata_values,
    )?;
    let mut metadata = BTreeMap::new();
    for (name, value) in object {
        check_cancelled(cancellation)?;
        check_name(name, limits, "safetensors metadata name")?;
        let value = value.as_str().ok_or_else(|| {
            invalid(
                "safetensors metadata",
                "verified metadata values must be strings",
            )
        })?;
        check_name(value, limits, "safetensors metadata value")?;
        if metadata.insert(name.clone(), value.to_owned()).is_some() {
            return Err(invalid(
                "safetensors metadata",
                "verified metadata contains a duplicate key",
            ));
        }
    }
    check_cancelled(cancellation)?;
    Ok(metadata)
}

pub fn checked_latent_upscale_configuration_json(
    bytes: &[u8],
    expected_sha256: &str,
    limits: &ParserLimits,
    cancellation: &CancellationToken,
) -> Result<Option<String>, ModelFormatError> {
    check_cancelled(cancellation)?;
    if expected_sha256.len() != 64
        || !expected_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        || hex_digest(Sha256::digest(bytes)) != expected_sha256
    {
        return Err(invalid(
            "latent upscale config",
            "verified configuration digest mismatch",
        ));
    }
    let document = parse_strict_json(bytes, limits)?;
    check_cancelled(cancellation)?;
    if document.get("weight_map").is_some() {
        return Ok(None);
    }
    if document
        .get("_class_name")
        .and_then(serde_json::Value::as_str)
        != Some("LatentUpsampler")
    {
        return Ok(None);
    }
    for required in [
        "in_channels",
        "mid_channels",
        "num_blocks_per_stage",
        "dims",
        "spatial_upsample",
        "temporal_upsample",
    ] {
        if document.get(required).is_none() {
            return Err(invalid(
                "latent upscale config",
                format!("semantic configuration omits {required}"),
            ));
        }
    }
    serde_json::to_string(&document)
        .map(Some)
        .map_err(|error| invalid("latent upscale config", error.to_string()))
}

pub fn checked_vae_configuration_json(
    bytes: &[u8],
    expected_sha256: &str,
    limits: &ParserLimits,
    cancellation: &CancellationToken,
) -> Result<Option<String>, ModelFormatError> {
    check_cancelled(cancellation)?;
    if expected_sha256.len() != 64
        || !expected_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        || hex_digest(Sha256::digest(bytes)) != expected_sha256
    {
        return Err(invalid(
            "VAE config",
            "verified configuration digest mismatch",
        ));
    }
    let document = parse_strict_json(bytes, limits)?;
    check_cancelled(cancellation)?;
    if document.get("weight_map").is_some() {
        return Ok(None);
    }
    if !document.is_object() {
        return Ok(None);
    }
    serde_json::to_string(&document)
        .map(Some)
        .map_err(|error| invalid("VAE config", error.to_string()))
}

struct StrictJsonSeed<'a> {
    limits: &'a ParserLimits,
    depth: u32,
    budget: &'a DecodedBudget,
}

impl<'de> DeserializeSeed<'de> for StrictJsonSeed<'_> {
    type Value = serde_json::Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if self.depth > self.limits.maximum_depth {
            return Err(de::Error::custom("JSON depth limit exceeded"));
        }
        self.budget.charge_json_value::<D::Error>()?;
        deserializer.deserialize_any(StrictJsonVisitor {
            limits: self.limits,
            depth: self.depth,
            budget: self.budget,
        })
    }
}

struct StrictJsonVisitor<'a> {
    limits: &'a ParserLimits,
    depth: u32,
    budget: &'a DecodedBudget,
}

impl<'de> Visitor<'de> for StrictJsonVisitor<'_> {
    type Value = serde_json::Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("bounded JSON")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(serde_json::Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(serde_json::Value::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(serde_json::Value::Number(value.into()))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        serde_json::Number::from_f64(value)
            .map(serde_json::Value::Number)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if u64::try_from(value.len()).unwrap_or(u64::MAX) > self.limits.maximum_name_bytes {
            return Err(E::custom("JSON string limit exceeded"));
        }
        self.budget.charge_json_bytes::<E>(value.len())?;
        Ok(serde_json::Value::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_str(&value)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(serde_json::Value::Null)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(serde_json::Value::Null)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element_seed(StrictJsonSeed {
            limits: self.limits,
            depth: self.depth + 1,
            budget: self.budget,
        })? {
            if u64::try_from(values.len()).unwrap_or(u64::MAX)
                >= self.limits.maximum_metadata_values
            {
                return Err(de::Error::custom("JSON value count limit exceeded"));
            }
            values.push(value);
        }
        Ok(serde_json::Value::Array(values))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = serde_json::Map::new();
        let mut keys = BTreeSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if u64::try_from(key.len()).unwrap_or(u64::MAX) > self.limits.maximum_name_bytes {
                return Err(de::Error::custom("JSON key limit exceeded"));
            }
            self.budget
                .charge_json_bytes::<A::Error>(key.len().saturating_mul(2))?;
            if !keys.insert(key.clone()) {
                return Err(de::Error::custom(format!("duplicate JSON key {key:?}")));
            }
            if u64::try_from(values.len()).unwrap_or(u64::MAX)
                >= self.limits.maximum_metadata_values
            {
                return Err(de::Error::custom("JSON value count limit exceeded"));
            }
            let value = map.next_value_seed(StrictJsonSeed {
                limits: self.limits,
                depth: self.depth + 1,
                budget: self.budget,
            })?;
            values.insert(key, value);
        }
        Ok(serde_json::Value::Object(values))
    }
}

struct DecodedBudget {
    values: Cell<u64>,
    bytes: Cell<u64>,
    maximum_values: u64,
    maximum_bytes: u64,
}

impl DecodedBudget {
    fn new(limits: &ParserLimits) -> Result<Self, ModelFormatError> {
        Ok(Self {
            values: Cell::new(0),
            bytes: Cell::new(0),
            maximum_values: limits.maximum_metadata_values,
            maximum_bytes: limits.maximum_decoded_allocation_bytes()?,
        })
    }

    fn charge_json_value<E: de::Error>(&self) -> Result<(), E> {
        let values = self
            .values
            .get()
            .checked_add(1)
            .ok_or_else(|| E::custom("JSON value count overflow"))?;
        if values > self.maximum_values {
            return Err(E::custom("JSON value count limit exceeded"));
        }
        self.values.set(values);
        self.charge_json_bytes::<E>(size_of::<serde_json::Value>())
    }

    fn charge_model_value(&self) -> Result<(), ModelFormatError> {
        let values = self
            .values
            .get()
            .checked_add(1)
            .ok_or(ModelFormatError::Overflow("decoded value count"))?;
        if values > self.maximum_values {
            return Err(ParserLimitError::Exceeded {
                kind: "decoded value count",
                actual: values,
                maximum: self.maximum_values,
            }
            .into());
        }
        self.values.set(values);
        self.charge_model_bytes(size_of::<serde_json::Value>())
    }

    fn charge_model_bytes(&self, amount: usize) -> Result<(), ModelFormatError> {
        let bytes = self
            .bytes
            .get()
            .checked_add(u64::try_from(amount).unwrap_or(u64::MAX))
            .ok_or(ModelFormatError::Overflow("decoded allocation bytes"))?;
        if bytes > self.maximum_bytes {
            return Err(ParserLimitError::Exceeded {
                kind: "decoded allocation bytes",
                actual: bytes,
                maximum: self.maximum_bytes,
            }
            .into());
        }
        self.bytes.set(bytes);
        Ok(())
    }

    fn charge_json_bytes<E: de::Error>(&self, amount: usize) -> Result<(), E> {
        let bytes = self
            .bytes
            .get()
            .checked_add(u64::try_from(amount).unwrap_or(u64::MAX))
            .ok_or_else(|| E::custom("JSON decoded allocation overflow"))?;
        if bytes > self.maximum_bytes {
            return Err(E::custom("JSON decoded allocation limit exceeded"));
        }
        self.bytes.set(bytes);
        Ok(())
    }
}

fn parse_sentencepiece_model(
    bytes: &[u8],
    limits: &ParserLimits,
    cancellation: &CancellationToken,
) -> Result<SentencePieceVocabulary, ModelFormatError> {
    let mut cursor = 0_usize;
    let mut fields = 0_u64;
    let mut pieces = Vec::<SentencePieceVocabularyEntry>::new();
    while cursor < bytes.len() {
        check_cancelled(cancellation)?;
        let key = protobuf_varint(bytes, &mut cursor)?;
        let field_number = key >> 3;
        if field_number == 0 {
            return Err(invalid("SentencePiece", "protobuf field number is zero"));
        }
        match key & 7 {
            0 => {
                protobuf_varint(bytes, &mut cursor)?;
            }
            1 => advance(&mut cursor, 8, bytes.len(), "SentencePiece fixed64")?,
            2 => {
                let length = protobuf_varint(bytes, &mut cursor)?;
                limits.check("SentencePiece field bytes", length, limits.manifest_bytes)?;
                let length = usize_from_u64(length, "SentencePiece field")?;
                let end = cursor
                    .checked_add(length)
                    .ok_or(ModelFormatError::Overflow("SentencePiece field"))?;
                let value = bytes
                    .get(cursor..end)
                    .ok_or_else(|| invalid("SentencePiece", "truncated field bytes"))?;
                if field_number == 1 {
                    let piece = parse_sentencepiece_piece(value, limits)?;
                    if pieces
                        .iter()
                        .any(|existing| existing.piece() == piece.piece())
                    {
                        return Err(invalid(
                            "SentencePiece",
                            format!("duplicate sentence piece {:?}", piece.piece()),
                        ));
                    }
                    pieces
                        .try_reserve(1)
                        .map_err(|_| ModelFormatError::AllocationFailed {
                            context: "SentencePiece vocabulary",
                            requested: std::mem::size_of::<SentencePieceVocabularyEntry>(),
                        })?;
                    pieces.push(piece);
                    limits.check(
                        "SentencePiece piece count",
                        u64::try_from(pieces.len()).unwrap_or(u64::MAX),
                        limits.maximum_metadata_values,
                    )?;
                } else if !matches!(field_number, 2..=5) {
                    return Err(invalid(
                        "SentencePiece",
                        format!("unknown ModelProto field {field_number}"),
                    ));
                }
                cursor = end;
            }
            5 => advance(&mut cursor, 4, bytes.len(), "SentencePiece fixed32")?,
            wire => {
                return Err(invalid(
                    "SentencePiece",
                    format!("unsupported protobuf wire type {wire}"),
                ));
            }
        }
        fields = fields
            .checked_add(1)
            .ok_or(ModelFormatError::Overflow("SentencePiece field count"))?;
        limits.check(
            "SentencePiece field count",
            fields,
            limits.maximum_metadata_values,
        )?;
    }
    if pieces.is_empty() {
        return Err(invalid(
            "SentencePiece",
            "ModelProto contains no sentence pieces",
        ));
    }
    if pieces.len() > u32::MAX as usize {
        return Err(ModelFormatError::Overflow("SentencePiece token ID"));
    }
    Ok(SentencePieceVocabulary { entries: pieces })
}

pub(crate) fn parse_sentencepiece_model_bytes(
    bytes: &[u8],
    limits: &ParserLimits,
    cancellation: &CancellationToken,
) -> Result<SentencePieceVocabulary, ModelFormatError> {
    limits.validate()?;
    limits.check(
        "SentencePiece model bytes",
        u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        limits.manifest_bytes,
    )?;
    parse_sentencepiece_model(bytes, limits, cancellation)
}

fn parse_sentencepiece_piece(
    bytes: &[u8],
    limits: &ParserLimits,
) -> Result<SentencePieceVocabularyEntry, ModelFormatError> {
    let mut cursor = 0_usize;
    let mut piece = None;
    let mut score = None;
    let mut piece_type = None;
    while cursor < bytes.len() {
        let key = protobuf_varint(bytes, &mut cursor)?;
        match (key >> 3, key & 7) {
            (1, 2) if piece.is_none() => {
                let length = protobuf_varint(bytes, &mut cursor)?;
                limits.check(
                    "SentencePiece piece bytes",
                    length,
                    limits.maximum_name_bytes,
                )?;
                let length = usize_from_u64(length, "SentencePiece piece")?;
                let end = cursor
                    .checked_add(length)
                    .ok_or(ModelFormatError::Overflow("SentencePiece piece"))?;
                let piece_bytes = bytes
                    .get(cursor..end)
                    .ok_or_else(|| invalid("SentencePiece", "truncated sentence piece"))?;
                let piece_text = std::str::from_utf8(piece_bytes)
                    .map_err(|_| invalid("SentencePiece", "sentence piece is not UTF-8"))?;
                if piece_text.is_empty() {
                    return Err(invalid("SentencePiece", "sentence piece is empty"));
                }
                let mut owned = String::new();
                owned.try_reserve_exact(piece_text.len()).map_err(|_| {
                    ModelFormatError::AllocationFailed {
                        context: "SentencePiece piece",
                        requested: piece_text.len(),
                    }
                })?;
                owned.push_str(piece_text);
                cursor = end;
                piece = Some(owned);
            }
            (2, 5) if score.is_none() => {
                let score_bytes = bytes
                    .get(cursor..cursor.saturating_add(4))
                    .ok_or_else(|| invalid("SentencePiece", "truncated piece score"))?;
                let parsed_score = f32::from_le_bytes(
                    score_bytes
                        .try_into()
                        .map_err(|_| invalid("SentencePiece", "truncated piece score"))?,
                );
                if !parsed_score.is_finite() {
                    return Err(invalid("SentencePiece", "piece score is not finite"));
                }
                cursor += 4;
                score = Some(parsed_score);
            }
            (3, 0) if piece_type.is_none() => {
                piece_type = Some(SentencePieceType::checked(protobuf_varint(
                    bytes,
                    &mut cursor,
                )?)?);
            }
            (field, wire) => {
                return Err(invalid(
                    "SentencePiece",
                    format!("unknown or duplicate SentencePiece field {field}/{wire}"),
                ));
            }
        }
    }
    let (Some(piece), Some(score), Some(piece_type)) = (piece, score, piece_type) else {
        return Err(invalid(
            "SentencePiece",
            "sentence piece is missing piece, score, or type",
        ));
    };
    Ok(SentencePieceVocabularyEntry {
        piece,
        score,
        piece_type,
    })
}

fn protobuf_varint(bytes: &[u8], cursor: &mut usize) -> Result<u64, ModelFormatError> {
    let mut value = 0_u64;
    for shift in (0..70).step_by(7) {
        let byte = bytes
            .get(*cursor)
            .copied()
            .ok_or_else(|| invalid("SentencePiece", "truncated protobuf varint"))?;
        *cursor += 1;
        if shift == 63 && byte > 1 {
            return Err(invalid("SentencePiece", "protobuf varint overflow"));
        }
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(invalid("SentencePiece", "protobuf varint is too long"))
}

fn advance(
    cursor: &mut usize,
    amount: usize,
    length: usize,
    context: &'static str,
) -> Result<(), ModelFormatError> {
    let end = cursor
        .checked_add(amount)
        .ok_or(ModelFormatError::Overflow(context))?;
    if end > length {
        return Err(invalid("SentencePiece", format!("truncated {context}")));
    }
    *cursor = end;
    Ok(())
}

fn contains_forbidden_yaml_indicator(value: &str) -> bool {
    let mut single_quote = false;
    let mut double_quote = false;
    let mut escaped = false;
    for character in value.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        match character {
            '\\' if double_quote => escaped = true,
            '\'' if !double_quote => single_quote = !single_quote,
            '"' if !single_quote => double_quote = !double_quote,
            '!' | '&' | '*' | '?' if !single_quote && !double_quote => return true,
            _ => {}
        }
    }
    value
        .split_once(':')
        .is_some_and(|(key, _)| key.trim() == "<<")
}

fn decode_standard_base64(
    encoded: &[u8],
    limits: &ParserLimits,
) -> Result<Vec<u8>, ModelFormatError> {
    if encoded.is_empty() || !encoded.len().is_multiple_of(4) {
        return Err(invalid("tiktoken", "invalid base64 token length"));
    }
    let padding = encoded
        .iter()
        .rev()
        .take_while(|byte| **byte == b'=')
        .count();
    if padding > 2
        || encoded
            .get(..encoded.len().saturating_sub(padding))
            .is_some_and(|bytes| bytes.contains(&b'='))
    {
        return Err(invalid("tiktoken", "invalid base64 token padding"));
    }
    let decoded_length = encoded
        .len()
        .checked_div(4)
        .and_then(|groups| groups.checked_mul(3))
        .and_then(|length| length.checked_sub(padding))
        .ok_or(ModelFormatError::Overflow("tiktoken decoded token"))?;
    limits.check(
        "tiktoken decoded token bytes",
        u64::try_from(decoded_length).unwrap_or(u64::MAX),
        limits.maximum_name_bytes,
    )?;
    let mut decoded = Vec::new();
    decoded
        .try_reserve_exact(decoded_length)
        .map_err(|_| ModelFormatError::AllocationFailed {
            context: "tiktoken decoded token",
            requested: decoded_length,
        })?;
    for (group_index, group) in encoded.chunks_exact(4).enumerate() {
        let last = group_index + 1 == encoded.len() / 4;
        let first = base64_value(group[0])?;
        let second = base64_value(group[1])?;
        let third = if group[2] == b'=' {
            if !last || group[3] != b'=' {
                return Err(invalid("tiktoken", "invalid base64 token padding"));
            }
            0
        } else {
            base64_value(group[2])?
        };
        let fourth = if group[3] == b'=' {
            if !last {
                return Err(invalid("tiktoken", "invalid base64 token padding"));
            }
            0
        } else {
            base64_value(group[3])?
        };
        if group[2] == b'=' && second & 0x0f != 0
            || group[3] == b'=' && group[2] != b'=' && third & 0x03 != 0
        {
            return Err(invalid("tiktoken", "non-canonical base64 token"));
        }
        decoded.push((first << 2) | (second >> 4));
        if group[2] != b'=' {
            decoded.push((second << 4) | (third >> 2));
        }
        if group[3] != b'=' {
            decoded.push((third << 6) | fourth);
        }
    }
    if decoded.len() != decoded_length {
        return Err(invalid("tiktoken", "base64 token length mismatch"));
    }
    Ok(decoded)
}

fn base64_value(byte: u8) -> Result<u8, ModelFormatError> {
    match byte {
        b'A'..=b'Z' => Ok(byte - b'A'),
        b'a'..=b'z' => Ok(byte - b'a' + 26),
        b'0'..=b'9' => Ok(byte - b'0' + 52),
        b'+' => Ok(62),
        b'/' => Ok(63),
        _ => Err(invalid("tiktoken", "invalid base64 token character")),
    }
}

fn validate_safetensors_ranges(
    ranges: &mut [(u64, u64)],
    data_length: u64,
) -> Result<(), ModelFormatError> {
    ranges.sort_unstable();
    let mut position = 0_u64;
    for (start, end) in ranges {
        if *start != position {
            return Err(invalid(
                "safetensors",
                "tensor ranges contain a hole or overlap",
            ));
        }
        position = *end;
    }
    if position != data_length {
        return Err(invalid(
            "safetensors",
            "tensor ranges do not cover the data section",
        ));
    }
    Ok(())
}

fn canonical_archive_path(name: &str) -> Result<String, ModelFormatError> {
    if name.contains('\\') || name.starts_with('/') || name.contains('\0') {
        return Err(ModelFormatError::UnsafeArchivePath(name.to_owned()));
    }
    let path = Path::new(name);
    let mut values = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(value) => values.push(value.to_string_lossy().into_owned()),
            _ => return Err(ModelFormatError::UnsafeArchivePath(name.to_owned())),
        }
    }
    if values.is_empty() {
        return Err(ModelFormatError::UnsafeArchivePath(name.to_owned()));
    }
    Ok(values.join("/"))
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<(), ModelFormatError> {
    if cancellation.is_cancelled() {
        Err(ModelFormatError::Cancelled)
    } else {
        Ok(())
    }
}

fn parse_pickle(
    bytes: &[u8],
    limits: &ParserLimits,
    cancellation: &CancellationToken,
) -> Result<PickleValue, ModelFormatError> {
    match parse_restricted_pickle_cancellable(bytes, limits, cancellation) {
        Err(RestrictedPickleError::Cancelled) => Err(ModelFormatError::Cancelled),
        Err(error) => Err(ModelFormatError::RestrictedPickle(error)),
        Ok(value) => Ok(value),
    }
}

fn check_name(
    value: &str,
    limits: &ParserLimits,
    kind: &'static str,
) -> Result<(), ModelFormatError> {
    limits.check(
        kind,
        u64::try_from(value.len()).unwrap_or(u64::MAX),
        limits.maximum_name_bytes,
    )?;
    if value.contains('\0') {
        return Err(invalid("model", format!("{kind} contains NUL")));
    }
    Ok(())
}

fn json_u64_array(
    value: Option<&serde_json::Value>,
    maximum: u32,
    context: &'static str,
) -> Result<Vec<u64>, ModelFormatError> {
    let values = value
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| invalid("safetensors", format!("{context} must be an array")))?;
    if values.len() > usize::try_from(maximum).unwrap_or(usize::MAX) {
        return Err(invalid(
            "safetensors",
            format!("{context} has too many values"),
        ));
    }
    values
        .iter()
        .map(|value| {
            value.as_u64().ok_or_else(|| {
                invalid(
                    "safetensors",
                    format!("{context} values must be unsigned integers"),
                )
            })
        })
        .collect()
}

fn checked_product(values: &[u64], context: &'static str) -> Result<u64, ModelFormatError> {
    values.iter().try_fold(1_u64, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(ModelFormatError::Overflow(context))
    })
}

fn ggml_tensor_bytes(shape: &[u64], data_type: u32) -> Result<u64, ModelFormatError> {
    let first_dimension = shape
        .first()
        .copied()
        .ok_or_else(|| invalid("GGUF", "tensor shape must have at least one dimension"))?;
    let (block_elements, block_bytes) = ggml_type_layout(data_type)
        .ok_or_else(|| invalid("GGUF", format!("unknown GGML tensor type {data_type}")))?;
    if first_dimension % block_elements != 0 {
        return Err(invalid(
            "GGUF",
            format!(
                "tensor first dimension {first_dimension} is not divisible by type {data_type} block size {block_elements}"
            ),
        ));
    }
    let rows = checked_product(shape.get(1..).unwrap_or_default(), "GGUF tensor row count")?;
    first_dimension
        .checked_div(block_elements)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .and_then(|row_bytes| row_bytes.checked_mul(rows))
        .ok_or(ModelFormatError::Overflow("GGUF tensor bytes"))
}

fn ggml_type_layout(data_type: u32) -> Option<(u64, u64)> {
    match data_type {
        0 => Some((1, 4)),
        1 => Some((1, 2)),
        2 => Some((32, 18)),
        3 => Some((32, 20)),
        6 => Some((32, 22)),
        7 => Some((32, 24)),
        8 => Some((32, 34)),
        9 => Some((32, 36)),
        10 => Some((256, 84)),
        11 => Some((256, 110)),
        12 => Some((256, 144)),
        13 => Some((256, 176)),
        14 => Some((256, 210)),
        15 => Some((256, 292)),
        16 => Some((256, 66)),
        17 => Some((256, 74)),
        18 => Some((256, 98)),
        19 => Some((256, 50)),
        20 => Some((32, 18)),
        21 => Some((256, 110)),
        22 => Some((256, 82)),
        23 => Some((256, 136)),
        24 => Some((1, 1)),
        25 => Some((1, 2)),
        26 => Some((1, 4)),
        27 => Some((1, 8)),
        28 => Some((1, 8)),
        29 => Some((256, 56)),
        30 => Some((1, 2)),
        34 => Some((256, 54)),
        35 => Some((256, 66)),
        39 => Some((32, 17)),
        40 => Some((64, 36)),
        41 => Some((128, 18)),
        42 => Some((64, 18)),
        _ => None,
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64, ModelFormatError> {
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|value| value & !mask)
        .ok_or(ModelFormatError::Overflow("alignment"))
}

fn consume_budget(
    budget: &mut u64,
    amount: u64,
    context: &'static str,
) -> Result<(), ModelFormatError> {
    *budget = budget
        .checked_sub(amount)
        .ok_or_else(|| invalid("model metadata", format!("{context} limit exceeded")))?;
    Ok(())
}

fn open_file(path: &Path) -> Result<File, ModelFormatError> {
    File::open(path).map_err(|error| io_error(path, error))
}

fn file_length(file: &File, path: &Path) -> Result<u64, ModelFormatError> {
    file.metadata()
        .map(|metadata| metadata.len())
        .map_err(|error| io_error(path, error))
}

fn read_exact(file: &mut impl Read, path: &Path, bytes: &mut [u8]) -> Result<(), ModelFormatError> {
    file.read_exact(bytes)
        .map_err(|error| io_error(path, error))
}

fn read_array<const LENGTH: usize>(
    file: &mut impl Read,
    path: &Path,
) -> Result<[u8; LENGTH], ModelFormatError> {
    let mut bytes = [0_u8; LENGTH];
    read_exact(file, path, &mut bytes)?;
    Ok(bytes)
}

fn read_u8(file: &mut File, path: &Path) -> Result<u8, ModelFormatError> {
    Ok(read_array::<1>(file, path)?[0])
}

fn read_u16(file: &mut File, path: &Path) -> Result<u16, ModelFormatError> {
    Ok(u16::from_le_bytes(read_array(file, path)?))
}

fn read_u32(file: &mut File, path: &Path) -> Result<u32, ModelFormatError> {
    Ok(u32::from_le_bytes(read_array(file, path)?))
}

fn read_u64(file: &mut impl Read, path: &Path) -> Result<u64, ModelFormatError> {
    Ok(u64::from_le_bytes(read_array(file, path)?))
}

fn read_exact_bounded(
    file: &mut impl Read,
    path: &Path,
    length: u64,
    maximum: u64,
    context: &'static str,
) -> Result<Vec<u8>, ModelFormatError> {
    if length > maximum {
        return Err(ParserLimitError::Exceeded {
            kind: context,
            actual: length,
            maximum,
        }
        .into());
    }
    let length = usize_from_u64(length, context)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| ModelFormatError::AllocationFailed {
            context,
            requested: length,
        })?;
    bytes.resize(length, 0);
    read_exact(file, path, &mut bytes)?;
    Ok(bytes)
}

fn sha256_open_file(
    file: &mut (impl Read + Seek),
    path: &Path,
    cancellation: &CancellationToken,
) -> Result<String, ModelFormatError> {
    file.seek(SeekFrom::Start(0))
        .map_err(|error| io_error(path, error))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        check_cancelled(cancellation)?;
        let read = file
            .read(&mut buffer)
            .map_err(|error| io_error(path, error))?;
        if read == 0 {
            break;
        }
        hasher.update(
            buffer
                .get(..read)
                .ok_or(ModelFormatError::Overflow("hash read buffer"))?,
        );
    }
    Ok(hex_digest(hasher.finalize()))
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let bytes = bytes.as_ref();
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(char::from(HEX[usize::from(byte >> 4)]));
        result.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    result
}

fn crc32(bytes: &[u8]) -> u32 {
    !crc32_update(u32::MAX, bytes)
}

fn crc32_update(mut crc: u32, bytes: &[u8]) -> u32 {
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (0_u32.wrapping_sub(crc & 1)));
        }
    }
    crc
}

fn le_u16(bytes: &[u8], offset: usize) -> Result<u16, ModelFormatError> {
    let value = bytes
        .get(offset..offset.saturating_add(2))
        .ok_or_else(|| invalid("ZIP", "truncated u16"))?;
    value
        .try_into()
        .map(u16::from_le_bytes)
        .map_err(|_| invalid("ZIP", "truncated u16"))
}

fn le_u32(bytes: &[u8], offset: usize) -> Result<u32, ModelFormatError> {
    let value = bytes
        .get(offset..offset.saturating_add(4))
        .ok_or_else(|| invalid("ZIP", "truncated u32"))?;
    value
        .try_into()
        .map(u32::from_le_bytes)
        .map_err(|_| invalid("ZIP", "truncated u32"))
}

fn le_u64(bytes: &[u8], offset: usize) -> Result<u64, ModelFormatError> {
    let value = bytes
        .get(offset..offset.saturating_add(8))
        .ok_or_else(|| invalid("ZIP", "truncated u64"))?;
    value
        .try_into()
        .map(u64::from_le_bytes)
        .map_err(|_| invalid("ZIP", "truncated u64"))
}

fn usize_from_u64(value: u64, context: &'static str) -> Result<usize, ModelFormatError> {
    usize::try_from(value).map_err(|_| ModelFormatError::Overflow(context))
}

fn io_error(path: &Path, error: io::Error) -> ModelFormatError {
    ModelFormatError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    }
}

fn invalid(format: &'static str, reason: impl Into<String>) -> ModelFormatError {
    ModelFormatError::Invalid {
        format,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_safetensors(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let header = br#"{"__metadata__":{"format":"pt"},"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut file = File::create(path)?;
        file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
        file.write_all(header)?;
        file.write_all(&[0, 0, 128, 63, 0, 0, 0, 64])?;
        Ok(())
    }

    #[test]
    fn safetensors_metadata_and_file_slices_parse() -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("model.safetensors");
        write_safetensors(&path)?;
        let parsed = parse_model_file(
            &path,
            &ParserLimits::default(),
            &CancellationToken::default(),
        )?;
        assert_eq!(parsed.tensors.len(), 1);
        assert_eq!(parsed.tensors[0].shape, vec![2]);
        assert_eq!(parsed.tensors[0].storage.length, 8);
        Ok(())
    }

    #[test]
    fn safetensors_float8_metadata_uses_canonical_dtype_projection()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("float8.latent");
        for (source_dtype, expected) in [
            ("F8_E4M3", DType::Float8E4m3Fn),
            ("F8_E5M2", DType::Float8E5m2),
        ] {
            let header = format!(
                r#"{{"latent_tensor":{{"dtype":"{source_dtype}","shape":[1,1,2,2],"data_offsets":[0,4]}}}}"#
            );
            let mut bytes = u64::try_from(header.len())?.to_le_bytes().to_vec();
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend_from_slice(&[0, 1, 2, 3]);
            let cancellation = CancellationToken::default();
            let limits = ParserLimits::default();
            std::fs::write(&path, &bytes)?;
            let file = parse_model_file(&path, &limits, &cancellation)?;
            let buffer = parse_safetensors_bytes(&bytes, &path, &limits, &cancellation)?;
            assert_eq!(file.tensors, buffer.tensors);
            let metadata = buffer.tensors.first().ok_or("float8 tensor missing")?;
            assert_eq!(metadata.native_dtype(), Some(expected));
            assert_eq!(metadata.storage.length, 4);
            assert_eq!(metadata.shape, [1, 1, 2, 2]);
        }
        assert_eq!(canonical_model_dtype("UNSUPPORTED"), None);
        Ok(())
    }

    #[test]
    fn safetensors_buffer_uses_file_parser_without_opening_source_name()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("source.latent");
        write_safetensors(&path)?;
        let bytes = std::fs::read(&path)?;
        let limits = ParserLimits::default();
        let cancellation = CancellationToken::default();
        let file = parse_model_file(&path, &limits, &cancellation)?;
        std::fs::remove_file(&path)?;
        let buffer = parse_safetensors_bytes(&bytes, &path, &limits, &cancellation)?;
        assert_eq!(buffer.format, file.format);
        assert_eq!(buffer.tensors, file.tensors);
        assert_eq!(
            buffer
                .tensors
                .first()
                .and_then(TensorMetadata::native_dtype),
            Some(DType::F32)
        );
        assert_eq!(buffer.source_size, file.source_size);
        assert_eq!(buffer.source_sha256, file.source_sha256);
        assert_eq!(buffer.payload, file.payload);
        Ok(())
    }

    #[test]
    fn safetensors_buffer_preserves_hostile_input_limits_and_cancellation()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("rejected.latent");
        let limits = ParserLimits::default();
        let cancellation = CancellationToken::default();
        for header in [
            r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
            r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,1]}}"#,
            r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#,
        ] {
            let mut bytes = u64::try_from(header.len())?.to_le_bytes().to_vec();
            bytes.extend_from_slice(header.as_bytes());
            bytes.push(0);
            std::fs::write(&path, &bytes)?;
            let file = parse_model_file(&path, &limits, &cancellation)
                .err()
                .ok_or("hostile file accepted")?;
            let buffer = parse_safetensors_bytes(&bytes, &path, &limits, &cancellation)
                .err()
                .ok_or("hostile buffer accepted")?;
            assert_eq!(file.to_string(), buffer.to_string());
        }
        write_safetensors(&path)?;
        let bytes = std::fs::read(&path)?;
        let bounded = ParserLimits {
            maximum_tensor_bytes: 7,
            ..limits
        };
        assert!(matches!(
            parse_safetensors_bytes(&bytes, &path, &bounded, &cancellation),
            Err(ModelFormatError::Limit(_))
        ));
        assert!(parse_safetensors_bytes(&bytes[..7], &path, &limits, &cancellation).is_err());
        cancellation.cancel();
        assert!(matches!(
            parse_safetensors_bytes(&bytes, &path, &limits, &cancellation),
            Err(ModelFormatError::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn safetensors_duplicates_and_ranges_fail_closed() -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("duplicate.safetensors");
        let header = br#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        let mut file = File::create(&path)?;
        file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
        file.write_all(header)?;
        file.write_all(&[1])?;
        let error = parse_model_file(
            &path,
            &ParserLimits::default(),
            &CancellationToken::default(),
        );
        assert!(matches!(error, Err(ModelFormatError::Invalid { .. })));
        Ok(())
    }

    #[test]
    fn safetensors_tensor_and_metadata_counts_are_exact() -> Result<(), Box<dyn std::error::Error>>
    {
        let directory = tempfile::tempdir()?;
        let tensors = directory.path().join("count.safetensors");
        let header = br#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#;
        let mut file = File::create(&tensors)?;
        file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
        file.write_all(header)?;
        file.write_all(&[1, 2])?;
        let tensor_limits = ParserLimits {
            maximum_tensors: 1,
            ..ParserLimits::default()
        };
        assert!(matches!(
            parse_model_file(&tensors, &tensor_limits, &CancellationToken::default()),
            Err(ModelFormatError::Limit(ParserLimitError::Exceeded {
                kind: "safetensors tensor count",
                ..
            }))
        ));

        let metadata = directory.path().join("metadata.safetensors");
        let header = br#"{"__metadata__":{"one":"1","two":"2"},"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        let mut file = File::create(&metadata)?;
        file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
        file.write_all(header)?;
        file.write_all(&[1])?;
        let metadata_limits = ParserLimits {
            maximum_metadata_values: 1,
            ..ParserLimits::default()
        };
        assert!(
            parse_model_file(&metadata, &metadata_limits, &CancellationToken::default()).is_err()
        );
        Ok(())
    }

    #[test]
    fn gguf_quantized_slices_exclude_alignment_padding() -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("quantized.gguf");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        for (name, shape, data_type, offset) in
            [("quant", 32_u64, 2_u32, 0_u64), ("tail", 1, 24, 32)]
        {
            bytes.extend_from_slice(&u64::try_from(name.len())?.to_le_bytes());
            bytes.extend_from_slice(name.as_bytes());
            bytes.extend_from_slice(&1_u32.to_le_bytes());
            bytes.extend_from_slice(&shape.to_le_bytes());
            bytes.extend_from_slice(&data_type.to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        bytes.extend_from_slice(&[7; 18]);
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        bytes.push(9);
        std::fs::write(&path, bytes)?;

        let parsed = parse_model_file(
            &path,
            &ParserLimits::default(),
            &CancellationToken::default(),
        )?;
        assert_eq!(parsed.tensors[0].storage.length, 18);
        assert_eq!(parsed.tensors[1].storage.length, 1);
        Ok(())
    }

    #[test]
    fn gguf_removed_types_and_incomplete_blocks_fail_closed() {
        assert!(ggml_type_layout(31).is_none());
        assert!(ggml_type_layout(36).is_none());
        assert!(ggml_tensor_bytes(&[31], 2).is_err());
        assert_eq!(ggml_tensor_bytes(&[64], 42), Ok(18));
    }

    #[test]
    fn every_accepted_ggml_layout_is_versioned_exactly() {
        let expected = [
            (0, 1, 4),
            (1, 1, 2),
            (2, 32, 18),
            (3, 32, 20),
            (6, 32, 22),
            (7, 32, 24),
            (8, 32, 34),
            (9, 32, 36),
            (10, 256, 84),
            (11, 256, 110),
            (12, 256, 144),
            (13, 256, 176),
            (14, 256, 210),
            (15, 256, 292),
            (16, 256, 66),
            (17, 256, 74),
            (18, 256, 98),
            (19, 256, 50),
            (20, 32, 18),
            (21, 256, 110),
            (22, 256, 82),
            (23, 256, 136),
            (24, 1, 1),
            (25, 1, 2),
            (26, 1, 4),
            (27, 1, 8),
            (28, 1, 8),
            (29, 256, 56),
            (30, 1, 2),
            (34, 256, 54),
            (35, 256, 66),
            (39, 32, 17),
            (40, 64, 36),
            (41, 128, 18),
            (42, 64, 18),
        ];
        assert_eq!(GGML_TENSOR_TYPES.len(), expected.len());
        for (data_type, block_elements, block_bytes) in expected {
            assert!(GGML_TENSOR_TYPES.contains(&data_type));
            assert_eq!(
                ggml_type_layout(data_type),
                Some((block_elements, block_bytes))
            );
            assert_eq!(
                ggml_tensor_bytes(&[block_elements], data_type),
                Ok(block_bytes)
            );
        }
    }

    #[test]
    fn json_depth_and_duplicate_keys_are_rejected() {
        let limits = ParserLimits {
            maximum_depth: 1,
            ..ParserLimits::default()
        };
        assert!(parse_strict_json(br#"{"a":{"b":1}}"#, &limits).is_err());
        assert!(parse_strict_json(br#"{"a":1,"a":2}"#, &ParserLimits::default()).is_err());
        let global_count_limits = ParserLimits {
            maximum_metadata_values: 4,
            ..ParserLimits::default()
        };
        assert!(parse_strict_json(br#"[[1,2],[3,4]]"#, &global_count_limits).is_err());
        let decoded_limits = ParserLimits {
            manifest_bytes: 10,
            ..ParserLimits::default()
        };
        assert!(parse_strict_json(b"[[[[[]]]]]", &decoded_limits).is_err());
    }

    #[test]
    fn yaml_and_tokenizer_security_syntax_is_strict() {
        let limits = ParserLimits::default();
        assert!(parse_bounded_yaml(b"base_path: &root /models\n", &limits).is_err());
        assert!(parse_bounded_yaml(b"base_path: *root\n", &limits).is_err());
        assert!(parse_bounded_yaml(b"base_path: !path /models\n", &limits).is_err());
        assert_eq!(decode_standard_base64(b"YQ==", &limits), Ok(vec![b'a']));
        assert!(decode_standard_base64(b"Y===", &limits).is_err());
        assert!(decode_standard_base64(b"YQ=Z", &limits).is_err());

        let valid_piece = [
            0x0a, 0x0a, 0x0a, 0x01, b'a', 0x15, 0x00, 0x00, 0x00, 0x00, 0x18, 0x01,
        ];
        let vocabulary =
            parse_sentencepiece_model(&valid_piece, &limits, &CancellationToken::default())
                .expect("valid SentencePiece vocabulary");
        assert_eq!(vocabulary.entries().len(), 1);
        assert_eq!(vocabulary.entries()[0].piece(), "a");
        assert_eq!(vocabulary.entries()[0].score(), 0.0);
        assert_eq!(
            vocabulary.entries()[0].piece_type(),
            SentencePieceType::Normal
        );
        assert!(
            parse_sentencepiece_model(
                &[0x0a, 0x03, b'a', b'b', b'c'],
                &limits,
                &CancellationToken::default(),
            )
            .is_err()
        );
        let duplicated = [valid_piece.as_slice(), valid_piece.as_slice()].concat();
        assert!(
            parse_sentencepiece_model(&duplicated, &limits, &CancellationToken::default()).is_err()
        );
        let limited = ParserLimits {
            maximum_metadata_values: 1,
            ..ParserLimits::default()
        };
        let second_piece = [
            0x0a, 0x0a, 0x0a, 0x01, b'b', 0x15, 0x00, 0x00, 0x80, 0x3f, 0x18, 0x01,
        ];
        let two_pieces = [valid_piece.as_slice(), second_piece.as_slice()].concat();
        assert!(
            parse_sentencepiece_model(&two_pieces, &limited, &CancellationToken::default())
                .is_err()
        );
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert!(matches!(
            parse_sentencepiece_model(&valid_piece, &limits, &cancelled),
            Err(ModelFormatError::Cancelled)
        ));
    }

    fn persistent_storage(
        storage_type: &str,
        storage_key: &str,
        storage_elements: i128,
    ) -> PickleValue {
        PickleValue::Persistent(Box::new(PickleValue::Tuple(vec![
            PickleValue::String("storage".to_owned()),
            PickleValue::Global(storage_type.to_owned()),
            PickleValue::String(storage_key.to_owned()),
            PickleValue::String("cpu".to_owned()),
            PickleValue::Integer(storage_elements),
        ])))
    }

    fn plain_tensor_rebuild(
        target: &str,
        storage_type: &str,
        storage_key: &str,
        shape: &[i128],
        stride: &[i128],
    ) -> PickleValue {
        let mut arguments = vec![
            persistent_storage(storage_type, storage_key, 4),
            PickleValue::Integer(0),
            PickleValue::Tuple(shape.iter().copied().map(PickleValue::Integer).collect()),
            PickleValue::Tuple(stride.iter().copied().map(PickleValue::Integer).collect()),
        ];
        match target {
            "torch._utils._rebuild_tensor_v2" => {
                arguments.push(PickleValue::Boolean(false));
                arguments.push(PickleValue::Dictionary(Vec::new()));
            }
            "torch._utils._rebuild_tensor_v3" => {
                arguments.push(PickleValue::Boolean(false));
                arguments.push(PickleValue::Dictionary(Vec::new()));
                arguments.push(PickleValue::Global("torch.float32".to_owned()));
            }
            _ => {}
        }
        PickleValue::Reduced {
            target: target.to_owned(),
            arguments: Box::new(PickleValue::Tuple(arguments)),
            state: None,
        }
    }

    fn parameter_rebuild(target: &str, tensor: PickleValue) -> PickleValue {
        let arguments = match target {
            "torch.nn.parameter.Parameter" => vec![tensor, PickleValue::Boolean(false)],
            "torch._utils._rebuild_parameter" => vec![
                tensor,
                PickleValue::Boolean(false),
                PickleValue::Dictionary(Vec::new()),
            ],
            "torch._utils._rebuild_parameter_with_state" => vec![
                tensor,
                PickleValue::Boolean(false),
                PickleValue::Dictionary(Vec::new()),
                PickleValue::Dictionary(Vec::new()),
            ],
            _ => Vec::new(),
        };
        PickleValue::Reduced {
            target: target.to_owned(),
            arguments: Box::new(PickleValue::Tuple(arguments)),
            state: Some(Box::new(PickleValue::Dictionary(Vec::new()))),
        }
    }

    fn quantized_tensor_rebuild(
        storage_type: &str,
        qscheme: &str,
        parameters: Vec<PickleValue>,
    ) -> PickleValue {
        let mut quantizer_parameters = vec![PickleValue::Global(qscheme.to_owned())];
        quantizer_parameters.extend(parameters);
        PickleValue::Reduced {
            target: "torch._utils._rebuild_qtensor".to_owned(),
            arguments: Box::new(PickleValue::Tuple(vec![
                persistent_storage(storage_type, "quantized", 4),
                PickleValue::Integer(0),
                PickleValue::Tuple(vec![PickleValue::Integer(2), PickleValue::Integer(2)]),
                PickleValue::Tuple(vec![PickleValue::Integer(2), PickleValue::Integer(1)]),
                PickleValue::Tuple(quantizer_parameters),
                PickleValue::Boolean(false),
                PickleValue::Dictionary(Vec::new()),
            ])),
            state: None,
        }
    }

    fn extract_tensor_rebuild(value: &PickleValue) -> Option<PytorchTensorRebuild> {
        tensor_rebuild(value, ParserLimits::default().maximum_depth)
    }

    #[test]
    fn pytorch_parameter_and_tensor_rebuild_adapters_are_exact() {
        for target in [
            "torch._utils._rebuild_tensor",
            "torch._utils._rebuild_tensor_v2",
            "torch._utils._rebuild_tensor_v3",
        ] {
            let parsed = extract_tensor_rebuild(&plain_tensor_rebuild(
                target,
                "torch.FloatStorage",
                "tensor",
                &[2, 2],
                &[2, 1],
            ));
            assert!(matches!(
                parsed,
                Some(PytorchTensorRebuild {
                    storage_key,
                    data_type,
                    shape,
                    ..
                }) if storage_key == "tensor"
                    && data_type == "torch.FloatStorage"
                    && shape == [2, 2]
            ));
        }

        for target in [
            "torch.nn.parameter.Parameter",
            "torch._utils._rebuild_parameter",
            "torch._utils._rebuild_parameter_with_state",
        ] {
            let tensor = plain_tensor_rebuild(
                "torch._utils._rebuild_tensor_v2",
                "torch.FloatStorage",
                "parameter",
                &[4],
                &[1],
            );
            let parsed = extract_tensor_rebuild(&parameter_rebuild(target, tensor));
            assert!(matches!(
                parsed,
                Some(PytorchTensorRebuild {
                    storage_key,
                    shape,
                    ..
                }) if storage_key == "parameter" && shape == [4]
            ));
        }

        let underlying = plain_tensor_rebuild(
            "torch._utils._rebuild_tensor_v2",
            "torch.FloatStorage",
            "subclass",
            &[4],
            &[1],
        );
        let PickleValue::Reduced { arguments, .. } = underlying else {
            panic!("fixture must be a reduction")
        };
        let from_type = PickleValue::Reduced {
            target: "torch._tensor._rebuild_from_type_v2".to_owned(),
            arguments: Box::new(PickleValue::Tuple(vec![
                PickleValue::Global("torch._utils._rebuild_tensor_v2".to_owned()),
                PickleValue::Global("torch.Tensor".to_owned()),
                *arguments,
                PickleValue::Dictionary(Vec::new()),
            ])),
            state: None,
        };
        assert!(matches!(
            extract_tensor_rebuild(&from_type),
            Some(PytorchTensorRebuild { storage_key, .. }) if storage_key == "subclass"
        ));
    }

    #[test]
    fn pytorch_cataloged_quantized_rebuild_adapters_are_exact() {
        for qscheme in ["torch.per_tensor_affine", "torch.per_tensor_symmetric"] {
            let parsed = extract_tensor_rebuild(&quantized_tensor_rebuild(
                "torch.QInt8Storage",
                qscheme,
                vec![
                    PickleValue::FloatBits(0.25_f64.to_bits()),
                    PickleValue::Integer(0),
                ],
            ));
            assert!(matches!(
                parsed,
                Some(PytorchTensorRebuild {
                    data_type,
                    auxiliary,
                    ..
                }) if data_type == "torch.QInt8Storage" && auxiliary.is_empty()
            ));
        }

        for qscheme in [
            "torch.per_channel_affine",
            "torch.per_channel_affine_float_qparams",
            "torch.per_channel_symmetric",
        ] {
            let scales = plain_tensor_rebuild(
                "torch._utils._rebuild_tensor_v2",
                "torch.FloatStorage",
                "scales",
                &[2],
                &[1],
            );
            let zero_points = plain_tensor_rebuild(
                "torch._utils._rebuild_tensor_v2",
                "torch.LongStorage",
                "zero_points",
                &[2],
                &[1],
            );
            let parsed = extract_tensor_rebuild(&quantized_tensor_rebuild(
                "torch.QUInt8Storage",
                qscheme,
                vec![scales, zero_points, PickleValue::Integer(0)],
            ));
            assert!(matches!(
                parsed,
                Some(PytorchTensorRebuild { auxiliary, .. }) if auxiliary.len() == 2
            ));
        }
    }

    #[test]
    fn pytorch_rebuild_adapters_reject_hostile_wrappers_and_layouts() {
        let spoofed = plain_tensor_rebuild(
            "torch._utils._rebuild_tensor_hostile",
            "torch.FloatStorage",
            "tensor",
            &[4],
            &[1],
        );
        assert!(extract_tensor_rebuild(&spoofed).is_none());

        let non_contiguous = plain_tensor_rebuild(
            "torch._utils._rebuild_tensor_v2",
            "torch.FloatStorage",
            "tensor",
            &[2, 2],
            &[1, 2],
        );
        assert!(extract_tensor_rebuild(&non_contiguous).is_none());
        assert!(
            extract_tensor_rebuild(&parameter_rebuild(
                "torch._utils._rebuild_parameter",
                PickleValue::None,
            ))
            .is_none()
        );

        let invalid_scale = quantized_tensor_rebuild(
            "torch.QInt8Storage",
            "torch.per_tensor_affine",
            vec![
                PickleValue::FloatBits(f64::NAN.to_bits()),
                PickleValue::Integer(0),
            ],
        );
        assert!(extract_tensor_rebuild(&invalid_scale).is_none());
        let invalid_storage = quantized_tensor_rebuild(
            "torch.FloatStorage",
            "torch.per_tensor_affine",
            vec![
                PickleValue::FloatBits(0.25_f64.to_bits()),
                PickleValue::Integer(0),
            ],
        );
        assert!(extract_tensor_rebuild(&invalid_storage).is_none());
        let unknown_scheme = quantized_tensor_rebuild(
            "torch.QInt8Storage",
            "torch.device",
            vec![
                PickleValue::FloatBits(0.25_f64.to_bits()),
                PickleValue::Integer(0),
            ],
        );
        assert!(extract_tensor_rebuild(&unknown_scheme).is_none());

        for target in [
            "torch._utils._rebuild_sparse_tensor",
            "torch._utils._rebuild_meta_tensor_no_storage",
        ] {
            let unsupported = PickleValue::Reduced {
                target: target.to_owned(),
                arguments: Box::new(PickleValue::Tuple(Vec::new())),
                state: None,
            };
            assert!(extract_tensor_rebuild(&unsupported).is_none());
        }

        let mut nested = plain_tensor_rebuild(
            "torch._utils._rebuild_tensor_v2",
            "torch.FloatStorage",
            "tensor",
            &[4],
            &[1],
        );
        for _ in 0..64 {
            nested = parameter_rebuild("torch._utils._rebuild_parameter", nested);
        }
        assert!(extract_tensor_rebuild(&nested).is_none());
    }

    #[test]
    fn recipe_semantic_configs_are_digest_bound_bounded_and_not_weight_indexes()
    -> Result<(), Box<dyn std::error::Error>> {
        let cancellation = CancellationToken::default();
        let limits = ParserLimits::default();
        let weight_index = br#"{"weight_map":{"weight":"model-00001-of-00001.safetensors"}}"#;
        let weight_index_sha256 = hex_digest(Sha256::digest(weight_index));
        assert_eq!(
            checked_latent_upscale_configuration_json(
                weight_index,
                &weight_index_sha256,
                &limits,
                &cancellation,
            )?,
            None
        );
        assert_eq!(
            checked_vae_configuration_json(
                weight_index,
                &weight_index_sha256,
                &limits,
                &cancellation,
            )?,
            None
        );

        let latent = br#"{"_class_name":"LatentUpsampler","in_channels":4,"mid_channels":8,"num_blocks_per_stage":1,"dims":3,"spatial_upsample":true,"temporal_upsample":false}"#;
        let latent_sha256 = hex_digest(Sha256::digest(latent));
        assert!(
            checked_latent_upscale_configuration_json(
                latent,
                &latent_sha256,
                &limits,
                &cancellation,
            )?
            .is_some()
        );
        let vae = br#"{"params":{"embed_dim":4}}"#;
        let vae_sha256 = hex_digest(Sha256::digest(vae));
        assert!(
            checked_vae_configuration_json(vae, &vae_sha256, &limits, &cancellation)?.is_some()
        );
        let ltx_vae = br#"{"vae":{"timestep_conditioning":false}}"#;
        let ltx_vae_sha256 = hex_digest(Sha256::digest(ltx_vae));
        assert!(
            checked_vae_configuration_json(ltx_vae, &ltx_vae_sha256, &limits, &cancellation)?
                .is_some()
        );
        assert!(
            checked_vae_configuration_json(vae, &"f".repeat(64), &limits, &cancellation).is_err()
        );

        let mut bounded = limits;
        bounded.maximum_metadata_values = 1;
        assert!(
            checked_latent_upscale_configuration_json(
                latent,
                &latent_sha256,
                &bounded,
                &cancellation,
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn cancellation_precedes_model_work() -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("model.safetensors");
        write_safetensors(&path)?;
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        assert_eq!(
            parse_model_file(&path, &ParserLimits::default(), &cancellation),
            Err(ModelFormatError::Cancelled)
        );
        Ok(())
    }
}
