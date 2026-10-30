use crate::{
    artifact_index::{
        ArtifactAvailability, ArtifactIndex, ArtifactIndexError, ArtifactKey, VerifiedArtifactFile,
    },
    formats::{
        GgufValue, ModelFormat, ModelFormatError, ParsedModel, ParsedModelPayload,
        SentencePieceVocabulary, TensorMetadata, parse_verified_embedding_archive_file,
        parse_verified_model_file,
    },
    model_family::{
        ModelFamilyError, ModelParsedFacts, ModelParsedFormatFact, ModelParsedTensorFact,
        ModelProbe, ModelWeightStatisticObservation, ModelWeightStatisticRequest,
        model_weight_statistic_dtype, observe_loaded_model_weight_tensor,
        validate_model_weight_statistic_requests,
    },
    parser_limits::ParserLimits,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceVec, DeviceId, ExecutionContext, TensorDescriptor,
    TensorError,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
    sync::{Arc, Weak},
};

const MAX_MODEL_OPERATION_RECORDS: usize = 4_096;
const MAX_MODEL_CACHE_ENTRIES: usize = 64;
const MODEL_STREAM_MAXIMUM_READ_BYTES: u64 = 1024 * 1024;

#[cfg(unix)]
use std::{ffi::c_void, os::fd::AsRawFd, ptr::NonNull, slice};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelOperationStage {
    Resolving,
    Parsing,
    Verifying,
    Ready,
    Cancelled,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelOperationRecord {
    pub key: ArtifactKey,
    pub attempt: u8,
    pub stage: ModelOperationStage,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelLoadAccounting {
    pub source_bytes: u64,
    pub tensor_bytes: u64,
    pub tensor_count: u64,
    pub resident_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct LoadedModel {
    identity: String,
    ordered_artifact_keys: Vec<ArtifactKey>,
    artifact_identities: BTreeMap<ArtifactKey, String>,
    documents: Vec<Arc<ParsedModel>>,
    tensors: BTreeMap<String, TensorMetadata>,
    tensor_sources: BTreeMap<String, ArtifactKey>,
    accounting: ModelLoadAccounting,
    store_identity: Arc<()>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelStreamNestedStateDisposition {
    Flat,
    NestedStringToParam,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ModelStreamSemanticArtifact {
    None,
    SafetensorsMetadata { source_artifact_ordinal: u32 },
    ClipConfiguration { target_ordinal: u8 },
    ClipTokenizerJson { target_ordinal: u8 },
    ClipSentencePiece { target_ordinal: u8 },
    ClipTiktoken { target_ordinal: u8 },
}

pub struct VerifiedModelStreamArtifact {
    key: ArtifactKey,
    sha256: String,
    byte_size: u64,
    format: ModelFormat,
    nested_state_disposition: ModelStreamNestedStateDisposition,
    semantic: ModelStreamSemanticArtifact,
    backing: ModelStreamArtifactBacking,
}

#[derive(Clone)]
enum ModelStreamArtifactBacking {
    Indexed,
    Inline(Arc<[u8]>),
}

impl VerifiedModelStreamArtifact {
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub const fn byte_size(&self) -> u64 {
        self.byte_size
    }

    pub const fn format(&self) -> &ModelFormat {
        &self.format
    }

    pub const fn nested_state_disposition(&self) -> ModelStreamNestedStateDisposition {
        self.nested_state_disposition
    }

    pub const fn semantic(&self) -> ModelStreamSemanticArtifact {
        self.semantic
    }
}

pub struct VerifiedModelStreamTensor {
    name: String,
    data_type: String,
    shape: Vec<u64>,
    artifact_ordinal: usize,
    artifact_key: ArtifactKey,
    byte_offset: u64,
    byte_length: u64,
}

impl VerifiedModelStreamTensor {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn data_type(&self) -> &str {
        &self.data_type
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    pub const fn artifact_ordinal(&self) -> usize {
        self.artifact_ordinal
    }

    pub const fn byte_offset(&self) -> u64 {
        self.byte_offset
    }

    pub const fn byte_length(&self) -> u64 {
        self.byte_length
    }
}

pub struct VerifiedModelStreamSource {
    model_identity: String,
    artifacts: Vec<VerifiedModelStreamArtifact>,
    tensors: Vec<VerifiedModelStreamTensor>,
    aggregate_tensor_bytes: u64,
    maximum_read_bytes: u64,
    limits: ParserLimits,
    store_identity: Weak<()>,
}

impl VerifiedModelStreamSource {
    pub fn model_identity(&self) -> &str {
        &self.model_identity
    }

    pub fn artifacts(&self) -> &[VerifiedModelStreamArtifact] {
        &self.artifacts
    }

    pub fn tensors(&self) -> &[VerifiedModelStreamTensor] {
        &self.tensors
    }

    pub const fn aggregate_tensor_bytes(&self) -> u64 {
        self.aggregate_tensor_bytes
    }

    pub const fn maximum_read_bytes(&self) -> u64 {
        self.maximum_read_bytes
    }

    pub const fn limits(&self) -> &ParserLimits {
        &self.limits
    }
}

impl LoadedModel {
    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn documents(&self) -> &[Arc<ParsedModel>] {
        &self.documents
    }

    pub fn tensors(&self) -> &BTreeMap<String, TensorMetadata> {
        &self.tensors
    }

    pub fn accounting(&self) -> &ModelLoadAccounting {
        &self.accounting
    }
}

#[derive(Clone, Debug)]
pub struct VerifiedModelTensor {
    name: String,
    data_type: String,
    shape: Vec<u64>,
    bytes: Arc<[u8]>,
}

impl VerifiedModelTensor {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn data_type(&self) -> &str {
        &self.data_type
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn native_dtype(&self) -> Option<sim_tensor::DType> {
        crate::formats::canonical_model_dtype(&self.data_type)
    }
}

#[derive(Clone, Debug)]
pub struct VerifiedModelTensorPayload {
    artifact_key: ArtifactKey,
    artifact_sha256: String,
    tensors: Vec<VerifiedModelTensor>,
    _store_identity: Arc<()>,
    nested_string_to_param: bool,
}

impl VerifiedModelTensorPayload {
    pub fn artifact_key(&self) -> &ArtifactKey {
        &self.artifact_key
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    pub fn tensors(&self) -> &[VerifiedModelTensor] {
        &self.tensors
    }

    pub(crate) fn store_identity(&self) -> Arc<()> {
        self._store_identity.clone()
    }

    pub(crate) const fn has_nested_string_to_param(&self) -> bool {
        self.nested_string_to_param
    }
}

#[derive(Clone, Debug)]
pub struct VerifiedSentencePieceVocabulary {
    artifact_key: ArtifactKey,
    artifact_sha256: String,
    vocabulary: SentencePieceVocabulary,
    _store_identity: Arc<()>,
}

impl VerifiedSentencePieceVocabulary {
    pub fn artifact_key(&self) -> &ArtifactKey {
        &self.artifact_key
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    pub fn vocabulary(&self) -> &SentencePieceVocabulary {
        &self.vocabulary
    }

    pub(crate) fn store_identity(&self) -> Arc<()> {
        self._store_identity.clone()
    }
}

#[derive(Clone, Debug)]
pub struct VerifiedEmbeddingArchivePayload {
    artifact_key: ArtifactKey,
    artifact_sha256: String,
    width: usize,
    rows: Vec<Arc<[f32]>>,
    _store_identity: Arc<()>,
}

impl VerifiedEmbeddingArchivePayload {
    pub fn artifact_key(&self) -> &ArtifactKey {
        &self.artifact_key
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub fn rows(&self) -> &[Arc<[f32]>] {
        &self.rows
    }

    pub(crate) fn store_identity(&self) -> Arc<()> {
        self._store_identity.clone()
    }
}

#[derive(Default)]
struct ModelCache {
    models: BTreeMap<ArtifactKey, Arc<LoadedModel>>,
}

pub struct ModelStore {
    limits: ParserLimits,
    cache: ModelCache,
    operations: Vec<ModelOperationRecord>,
    identity: Arc<()>,
}

#[cfg(unix)]
pub struct ReadOnlyTensorMapping {
    _file: File,
    allocation: NonNull<c_void>,
    allocation_length: usize,
    view_offset: usize,
    view_length: usize,
}

#[cfg(unix)]
impl ReadOnlyTensorMapping {
    pub fn as_bytes(&self) -> &[u8] {
        // The allocation is a read-only MAP_PRIVATE region kept alive by this object, and
        // construction verifies that the requested view lies completely inside it.
        unsafe {
            slice::from_raw_parts(
                self.allocation.as_ptr().cast::<u8>().add(self.view_offset),
                self.view_length,
            )
        }
    }
}

#[cfg(unix)]
impl Drop for ReadOnlyTensorMapping {
    fn drop(&mut self) {
        // This pointer/length pair is returned unchanged from mmap and is unmapped once.
        let result = unsafe { munmap(self.allocation.as_ptr(), self.allocation_length) };
        if result != 0 {
            eprintln!("failed to release a read-only model tensor mapping");
        }
    }
}

impl ModelStore {
    pub fn new(limits: ParserLimits) -> Result<Self, ModelStoreError> {
        limits.validate()?;
        Ok(Self {
            limits,
            cache: ModelCache::default(),
            operations: Vec::new(),
            identity: Arc::new(()),
        })
    }

    pub fn limits(&self) -> &ParserLimits {
        &self.limits
    }

    pub fn operations(&self) -> &[ModelOperationRecord] {
        &self.operations
    }

    pub fn verified_embedding_archive(
        &self,
        index: &ArtifactIndex,
        key: &ArtifactKey,
        cancellation: &CancellationToken,
    ) -> Result<Option<VerifiedEmbeddingArchivePayload>, ModelStoreError> {
        cancellation
            .check()
            .map_err(|_| ModelStoreError::Cancelled)?;
        let record = index
            .record(key)
            .ok_or_else(|| ModelStoreError::UnknownArtifact(key.clone()))?;
        if record.availability != ArtifactAvailability::Present {
            return Err(ModelStoreError::MissingArtifact(key.clone()));
        }
        let verified = index.open_verified(key, cancellation)?;
        let Some(parsed) =
            parse_verified_embedding_archive_file(verified, &self.limits, cancellation)?
        else {
            return Ok(None);
        };
        let width = parsed.width();
        let row_count = parsed.row_count();
        let parsed_rows = parsed.into_rows();
        let mut rows = Vec::new();
        rows.try_reserve_exact(row_count)
            .map_err(|_| ModelStoreError::AllocationFailed {
                requested: u64::try_from(row_count).unwrap_or(u64::MAX),
            })?;
        rows.extend(parsed_rows.into_iter().map(Arc::<[f32]>::from));
        cancellation
            .check()
            .map_err(|_| ModelStoreError::Cancelled)?;
        Ok(Some(VerifiedEmbeddingArchivePayload {
            artifact_key: key.clone(),
            artifact_sha256: record.sha256.clone(),
            width,
            rows,
            _store_identity: self.identity.clone(),
        }))
    }

    pub fn verified_sentencepiece_vocabulary(
        &self,
        index: &ArtifactIndex,
        model: &LoadedModel,
        key: &ArtifactKey,
        cancellation: &CancellationToken,
    ) -> Result<VerifiedSentencePieceVocabulary, ModelStoreError> {
        if !Arc::ptr_eq(&self.identity, &model.store_identity) {
            return Err(ModelStoreError::ForeignModelHandle);
        }
        cancellation
            .check()
            .map_err(|_| ModelStoreError::Cancelled)?;
        let artifact_sha256 = model
            .artifact_identities
            .get(key)
            .cloned()
            .ok_or_else(|| ModelStoreError::MissingTensorPayloadSource(key.clone()))?;
        self.verify_model_source(index, model, key)?;
        drop(index.open_verified(key, cancellation)?);
        let vocabulary = model
            .documents
            .iter()
            .find_map(|document| {
                if document.source_sha256 != artifact_sha256 {
                    return None;
                }
                match &document.payload {
                    ParsedModelPayload::SentencePiece { vocabulary } => Some(vocabulary.clone()),
                    _ => None,
                }
            })
            .ok_or_else(|| ModelStoreError::MissingSentencePieceVocabulary(key.clone()))?;
        cancellation
            .check()
            .map_err(|_| ModelStoreError::Cancelled)?;
        Ok(VerifiedSentencePieceVocabulary {
            artifact_key: key.clone(),
            artifact_sha256,
            vocabulary,
            _store_identity: self.identity.clone(),
        })
    }

    pub fn family_probe(
        &self,
        model: &LoadedModel,
        cancellation: &CancellationToken,
    ) -> Result<ModelProbe, ModelStoreError> {
        if !Arc::ptr_eq(&self.identity, &model.store_identity) {
            return Err(ModelStoreError::ForeignModelHandle);
        }
        if cancellation.is_cancelled() {
            return Err(ModelStoreError::Cancelled);
        }

        let mut tensors = BTreeMap::new();
        for (name, tensor) in &model.tensors {
            if cancellation.is_cancelled() {
                return Err(ModelStoreError::Cancelled);
            }
            tensors.insert(
                name.clone(),
                ModelParsedTensorFact {
                    shape: tensor.shape.clone(),
                    storage_dtype: tensor.data_type.clone(),
                },
            );
        }

        let mut formats = Vec::new();
        formats
            .try_reserve_exact(model.documents.len())
            .map_err(|_| ModelStoreError::AllocationFailed {
                requested: u64::try_from(model.documents.len()).unwrap_or(u64::MAX),
            })?;
        for document in &model.documents {
            if cancellation.is_cancelled() {
                return Err(ModelStoreError::Cancelled);
            }
            formats.push(parsed_format_fact(document)?);
        }
        match ModelProbe::from_parsed_facts_cancellable(
            ModelParsedFacts { tensors, formats },
            cancellation,
        ) {
            Ok(probe) => Ok(probe),
            Err(ModelFamilyError::Cancelled(_)) => Err(ModelStoreError::Cancelled),
            Err(error) => Err(ModelStoreError::FamilyProbe(ModelFamilyProbeError::from(
                error,
            ))),
        }
    }

    pub fn validate_loaded_artifact_identity(
        &self,
        model: &LoadedModel,
        key: &ArtifactKey,
        expected_sha256: &str,
        cancellation: &CancellationToken,
    ) -> Result<(), ModelStoreError> {
        if !Arc::ptr_eq(&self.identity, &model.store_identity) {
            return Err(ModelStoreError::ForeignModelHandle);
        }
        if cancellation.is_cancelled() {
            return Err(ModelStoreError::Cancelled);
        }
        match model.artifact_identities.get(key) {
            Some(actual_sha256) if actual_sha256 == expected_sha256 => Ok(()),
            Some(actual_sha256) => Err(ModelStoreError::LoadedArtifactIdentityMismatch {
                key: key.clone(),
                expected_sha256: expected_sha256.to_owned(),
                actual_sha256: Some(actual_sha256.clone()),
            }),
            None => Err(ModelStoreError::LoadedArtifactIdentityMismatch {
                key: key.clone(),
                expected_sha256: expected_sha256.to_owned(),
                actual_sha256: None,
            }),
        }
    }

    pub fn observe_weight_statistics_with_context(
        &self,
        backend: &CpuBackend,
        index: &ArtifactIndex,
        model: &LoadedModel,
        requests: &[ModelWeightStatisticRequest],
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<ModelWeightStatisticObservation>, ModelWeightStatisticError> {
        context.check()?;
        validate_model_weight_statistic_requests(requests)?;

        let mut staged = Vec::new();
        staged.try_reserve_exact(requests.len()).map_err(|_| {
            ModelWeightStatisticError::AllocationFailed {
                requested: u64::try_from(requests.len()).unwrap_or(u64::MAX),
            }
        })?;
        for request in requests {
            context.check()?;
            let (metadata, _) = self.checked_tensor(model, request.tensor_name())?;
            let dtype = model_weight_statistic_dtype(request.tensor_name(), &metadata.data_type)?;
            let source =
                self.read_tensor(index, model, request.tensor_name(), context.cancellation)?;
            let native_bytes = model_weight_statistic_native_bytes(
                backend,
                context,
                &source,
                dtype,
                request.tensor_name(),
            )?;
            let descriptor = TensorDescriptor::contiguous(
                metadata.shape.clone(),
                dtype,
                DeviceId::CPU,
                context.stream,
            )?;
            let (tensor, _) = backend.upload_bytes(descriptor, &native_bytes, context)?;
            drop(native_bytes);
            staged.push(observe_loaded_model_weight_tensor(
                request, backend, &tensor, context,
            )?);
        }
        context.check()?;
        Ok(staged)
    }

    pub fn load(
        &mut self,
        index: &ArtifactIndex,
        key: &ArtifactKey,
        cancellation: &CancellationToken,
    ) -> Result<Arc<LoadedModel>, ModelStoreError> {
        self.record(key, 0, ModelOperationStage::Resolving, None);
        if cancellation.is_cancelled() {
            self.record(key, 0, ModelOperationStage::Cancelled, None);
            return Err(ModelStoreError::Cancelled);
        }
        let record = index
            .record(key)
            .cloned()
            .ok_or_else(|| ModelStoreError::UnknownArtifact(key.clone()))?;
        if record.availability == ArtifactAvailability::Missing {
            return Err(ModelStoreError::MissingArtifact(key.clone()));
        }
        if let Some(cached) = self.cache.models.get(key) {
            if cached.artifact_identities.iter().all(|(source, digest)| {
                index.record(source).is_some_and(|record| {
                    record.availability == ArtifactAvailability::Present && &record.sha256 == digest
                })
            }) {
                let cached = cached.clone();
                for source in cached.artifact_identities.keys() {
                    index.open_verified(source, cancellation)?;
                }
                self.record(key, 0, ModelOperationStage::Ready, None);
                return Ok(cached);
            }
        }

        let attempt = 0;
        self.record(key, attempt, ModelOperationStage::Parsing, None);
        let result = if is_shard_index(&key.relative_path) {
            self.load_sharded(index, key, attempt, cancellation)
        } else {
            self.load_single(index, key, attempt, cancellation)
        };
        match result {
            Ok(model) => {
                self.record(key, attempt, ModelOperationStage::Ready, None);
                if self.cache.models.len() == MAX_MODEL_CACHE_ENTRIES
                    && let Some(oldest_key) = self.cache.models.keys().next().cloned()
                {
                    self.cache.models.remove(&oldest_key);
                }
                self.cache.models.insert(key.clone(), model.clone());
                Ok(model)
            }
            Err(ModelStoreError::Cancelled) => {
                self.record(key, attempt, ModelOperationStage::Cancelled, None);
                Err(ModelStoreError::Cancelled)
            }
            Err(error) => {
                self.record(
                    key,
                    attempt,
                    ModelOperationStage::Failed,
                    Some(error.to_string()),
                );
                Err(error)
            }
        }
    }

    pub fn read_tensor(
        &self,
        index: &ArtifactIndex,
        model: &LoadedModel,
        tensor_name: &str,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u8>, ModelStoreError> {
        self.read_tensors(index, model, [tensor_name], cancellation)?
            .remove(tensor_name)
            .ok_or_else(|| ModelStoreError::UnknownTensor(tensor_name.to_owned()))
    }

    pub fn verified_stream_source(
        &mut self,
        index: &ArtifactIndex,
        model: &LoadedModel,
        cancellation: &CancellationToken,
    ) -> Result<VerifiedModelStreamSource, ModelStoreError> {
        if !Arc::ptr_eq(&self.identity, &model.store_identity) {
            return Err(ModelStoreError::ForeignModelHandle);
        }
        cancellation
            .check()
            .map_err(|_| ModelStoreError::Cancelled)?;
        if model.ordered_artifact_keys.len() != model.documents.len() {
            return Err(ModelStoreError::InvalidModelStreamSourceMetadata(
                "ordered artifact and parsed document counts differ".to_owned(),
            ));
        }

        let companions = self.load_stream_companions(index, model, cancellation)?;
        let mut artifacts = Vec::new();
        artifacts
            .try_reserve_exact(model.documents.len().saturating_add(companions.len()))
            .map_err(|_| ModelStoreError::AllocationFailed {
                requested: u64::try_from(model.documents.len()).unwrap_or(u64::MAX),
            })?;
        let mut tensors = Vec::new();
        tensors
            .try_reserve_exact(model.tensors.len())
            .map_err(|_| ModelStoreError::AllocationFailed {
                requested: u64::try_from(model.tensors.len()).unwrap_or(u64::MAX),
            })?;

        for (key, document) in model
            .ordered_artifact_keys
            .iter()
            .zip(model.documents.iter())
        {
            cancellation
                .check()
                .map_err(|_| ModelStoreError::Cancelled)?;
            let expected_sha256 = model.artifact_identities.get(key).ok_or_else(|| {
                ModelStoreError::InvalidModelStreamSourceMetadata(format!(
                    "ordered artifact {key:?} has no loaded identity"
                ))
            })?;
            if expected_sha256 != &document.source_sha256 {
                return Err(ModelStoreError::InvalidModelStreamSourceMetadata(format!(
                    "ordered artifact {key:?} does not match its parsed document"
                )));
            }
            self.verify_stream_artifact_record(
                index,
                key,
                &document.source_sha256,
                document.source_size,
            )?;
            let verified = self.open_stream_artifact(index, key, cancellation)?;
            let source_path = verified.path().to_path_buf();
            self.verify_opened_stream_artifact(&verified, key)?;
            let nested_state_disposition = match &document.payload {
                ParsedModelPayload::Pytorch { root, .. }
                    if crate::formats::has_nested_string_to_param(root) =>
                {
                    ModelStreamNestedStateDisposition::NestedStringToParam
                }
                _ => ModelStreamNestedStateDisposition::Flat,
            };
            let artifact_ordinal = artifacts.len();
            artifacts.push(VerifiedModelStreamArtifact {
                key: key.clone(),
                sha256: document.source_sha256.clone(),
                byte_size: document.source_size,
                format: document.format.clone(),
                nested_state_disposition,
                semantic: ModelStreamSemanticArtifact::None,
                backing: ModelStreamArtifactBacking::Indexed,
            });
            for tensor in &document.tensors {
                if model.tensor_sources.get(&tensor.name) != Some(key) {
                    continue;
                }
                if tensor.storage.path != source_path {
                    return Err(ModelStoreError::StorageSourceChanged { key: key.clone() });
                }
                tensors.push(VerifiedModelStreamTensor {
                    name: tensor.name.clone(),
                    data_type: tensor.data_type.clone(),
                    shape: tensor.shape.clone(),
                    artifact_ordinal,
                    artifact_key: key.clone(),
                    byte_offset: tensor.storage.offset,
                    byte_length: tensor.storage.length,
                });
            }
            if let ParsedModelPayload::Safetensors { metadata } = &document.payload
                && !metadata.is_empty()
            {
                let metadata_bytes = serde_json::to_vec(metadata).map_err(|error| {
                    ModelStoreError::InvalidModelStreamSourceMetadata(format!(
                        "safetensors metadata serialization failed: {error}"
                    ))
                })?;
                self.limits.check(
                    "safetensors stream metadata bytes",
                    u64::try_from(metadata_bytes.len())
                        .map_err(|_| ModelStoreError::Overflow("safetensors metadata bytes"))?,
                    self.limits.manifest_bytes,
                )?;
                let metadata_sha256 = format!("{:x}", Sha256::digest(&metadata_bytes));
                artifacts.push(VerifiedModelStreamArtifact {
                    key: key.clone(),
                    sha256: metadata_sha256,
                    byte_size: u64::try_from(metadata_bytes.len())
                        .map_err(|_| ModelStoreError::Overflow("safetensors metadata bytes"))?,
                    format: ModelFormat::JsonConfig,
                    nested_state_disposition: ModelStreamNestedStateDisposition::Flat,
                    semantic: ModelStreamSemanticArtifact::SafetensorsMetadata {
                        source_artifact_ordinal: u32::try_from(artifact_ordinal).map_err(|_| {
                            ModelStoreError::Overflow("safetensors source artifact ordinal")
                        })?,
                    },
                    backing: ModelStreamArtifactBacking::Inline(Arc::from(metadata_bytes)),
                });
            }
        }
        for (semantic, companion) in companions {
            cancellation
                .check()
                .map_err(|_| ModelStoreError::Cancelled)?;
            if companion.documents.len() != 1
                || companion.ordered_artifact_keys.len() != 1
                || !companion.tensors.is_empty()
            {
                return Err(ModelStoreError::InvalidModelStreamSourceMetadata(
                    "semantic companion is not one tensor-free artifact".to_owned(),
                ));
            }
            let key = companion
                .ordered_artifact_keys
                .first()
                .ok_or_else(|| {
                    ModelStoreError::InvalidModelStreamSourceMetadata(
                        "semantic companion omitted its artifact key".to_owned(),
                    )
                })?
                .clone();
            let document = companion.documents.first().ok_or_else(|| {
                ModelStoreError::InvalidModelStreamSourceMetadata(
                    "semantic companion omitted its parsed document".to_owned(),
                )
            })?;
            self.verify_stream_artifact_record(
                index,
                &key,
                &document.source_sha256,
                document.source_size,
            )?;
            artifacts.push(VerifiedModelStreamArtifact {
                key,
                sha256: document.source_sha256.clone(),
                byte_size: document.source_size,
                format: document.format.clone(),
                nested_state_disposition: ModelStreamNestedStateDisposition::Flat,
                semantic,
                backing: ModelStreamArtifactBacking::Indexed,
            });
        }

        let source = VerifiedModelStreamSource {
            model_identity: verified_model_stream_source_identity(&artifacts)?,
            artifacts,
            tensors,
            aggregate_tensor_bytes: model.accounting.tensor_bytes,
            maximum_read_bytes: MODEL_STREAM_MAXIMUM_READ_BYTES,
            limits: self.limits.clone(),
            store_identity: Arc::downgrade(&self.identity),
        };
        self.validate_stream_source(&source)?;
        for artifact in &source.artifacts {
            cancellation
                .check()
                .map_err(|_| ModelStoreError::Cancelled)?;
            if matches!(artifact.backing, ModelStreamArtifactBacking::Inline(_)) {
                continue;
            }
            self.verify_stream_artifact_record(
                index,
                &artifact.key,
                &artifact.sha256,
                artifact.byte_size,
            )?;
            drop(self.open_stream_artifact(index, &artifact.key, cancellation)?);
        }
        cancellation
            .check()
            .map_err(|_| ModelStoreError::Cancelled)?;
        Ok(source)
    }

    fn load_stream_companions(
        &mut self,
        index: &ArtifactIndex,
        model: &LoadedModel,
        cancellation: &CancellationToken,
    ) -> Result<Vec<(ModelStreamSemanticArtifact, Arc<LoadedModel>)>, ModelStoreError> {
        let primary = model.ordered_artifact_keys.first().ok_or_else(|| {
            ModelStoreError::InvalidModelStreamSourceMetadata(
                "loaded model has no primary artifact".to_owned(),
            )
        })?;
        let parent = primary
            .relative_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new(""));
        let file_name = primary
            .relative_path
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| {
                ModelStoreError::InvalidModelStreamSourceMetadata(
                    "loaded model primary artifact name is not UTF-8".to_owned(),
                )
            })?;
        let mut companions = Vec::new();
        for target_ordinal in 0_u8..4 {
            for (suffix, expected_format, semantic) in [
                (
                    format!("clip-config-{target_ordinal}.json"),
                    ModelFormat::JsonConfig,
                    ModelStreamSemanticArtifact::ClipConfiguration { target_ordinal },
                ),
                (
                    format!("clip-tokenizer-{target_ordinal}.json"),
                    ModelFormat::JsonTokenizer,
                    ModelStreamSemanticArtifact::ClipTokenizerJson { target_ordinal },
                ),
                (
                    format!("clip-tokenizer-{target_ordinal}.model"),
                    ModelFormat::SentencePiece,
                    ModelStreamSemanticArtifact::ClipSentencePiece { target_ordinal },
                ),
                (
                    format!("clip-tokenizer-{target_ordinal}.tiktoken"),
                    ModelFormat::Tiktoken,
                    ModelStreamSemanticArtifact::ClipTiktoken { target_ordinal },
                ),
            ] {
                let key = ArtifactKey::new(
                    primary.root_id.clone(),
                    parent.join(format!("{file_name}.{suffix}")),
                )?;
                let Some(record) = index.record(&key) else {
                    continue;
                };
                if record.availability != ArtifactAvailability::Present {
                    return Err(ModelStoreError::MissingArtifact(key));
                }
                let companion = self.load(index, &key, cancellation)?;
                let actual_format = companion.documents.first().map(|document| &document.format);
                if actual_format != Some(&expected_format) {
                    return Err(ModelStoreError::InvalidModelStreamSourceMetadata(format!(
                        "semantic companion {key:?} has format {actual_format:?}, expected {expected_format:?}"
                    )));
                }
                companions.push((semantic, companion));
            }
        }
        Ok(companions)
    }

    pub fn read_verified_tensor_range(
        &self,
        index: &ArtifactIndex,
        source: &VerifiedModelStreamSource,
        tensor_name: &str,
        byte_offset: u64,
        byte_length: u64,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u8>, ModelStoreError> {
        self.validate_stream_source_identity(source)?;
        cancellation
            .check()
            .map_err(|_| ModelStoreError::Cancelled)?;
        self.validate_stream_source(source)?;
        self.limits.check(
            "model stream tensor range bytes",
            byte_length,
            source.maximum_read_bytes,
        )?;
        let tensor = source
            .tensors
            .iter()
            .find(|tensor| tensor.name == tensor_name)
            .ok_or_else(|| ModelStoreError::UnknownTensor(tensor_name.to_owned()))?;
        let range_end = byte_offset
            .checked_add(byte_length)
            .ok_or(ModelStoreError::Overflow("model stream tensor range"))?;
        if range_end > tensor.byte_length {
            return Err(ModelStoreError::ModelStreamTensorRangeOutOfBounds {
                tensor: tensor_name.to_owned(),
                byte_offset,
                byte_length,
                tensor_byte_length: tensor.byte_length,
            });
        }

        let mut verified_source = None;
        for artifact in &source.artifacts {
            cancellation
                .check()
                .map_err(|_| ModelStoreError::Cancelled)?;
            if matches!(artifact.backing, ModelStreamArtifactBacking::Inline(_)) {
                continue;
            }
            self.verify_stream_artifact_record(
                index,
                &artifact.key,
                &artifact.sha256,
                artifact.byte_size,
            )?;
            let verified = self.open_stream_artifact(index, &artifact.key, cancellation)?;
            if artifact.key == tensor.artifact_key {
                verified_source = Some(verified);
            }
        }
        let mut verified = verified_source.ok_or_else(|| {
            ModelStoreError::InvalidModelStreamSourceMetadata(format!(
                "tensor {tensor_name:?} has no ordered artifact"
            ))
        })?;
        let absolute_offset = tensor
            .byte_offset
            .checked_add(byte_offset)
            .ok_or(ModelStoreError::Overflow("model stream source range"))?;
        let length = usize::try_from(byte_length)
            .map_err(|_| ModelStoreError::Overflow("model stream tensor range length"))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| ModelStoreError::AllocationFailed {
                requested: byte_length,
            })?;
        cancellation
            .check()
            .map_err(|_| ModelStoreError::Cancelled)?;
        bytes.resize(length, 0);
        verified
            .file_mut()
            .seek(SeekFrom::Start(absolute_offset))
            .map_err(|error| ModelStoreError::ModelStreamIo {
                key: tensor.artifact_key.clone(),
                message: error.to_string(),
            })?;
        const STREAM_CHUNK_BYTES: usize = 1024 * 1024;
        let mut position = 0_usize;
        while position < bytes.len() {
            cancellation
                .check()
                .map_err(|_| ModelStoreError::Cancelled)?;
            let end = position.saturating_add(STREAM_CHUNK_BYTES).min(bytes.len());
            let chunk = bytes
                .get_mut(position..end)
                .ok_or(ModelStoreError::Overflow("model stream tensor range chunk"))?;
            verified.file_mut().read_exact(chunk).map_err(|error| {
                ModelStoreError::ModelStreamIo {
                    key: tensor.artifact_key.clone(),
                    message: error.to_string(),
                }
            })?;
            position = end;
        }
        cancellation
            .check()
            .map_err(|_| ModelStoreError::Cancelled)?;
        self.verify_opened_stream_artifact(&verified, &tensor.artifact_key)?;
        drop(verified);
        for artifact in &source.artifacts {
            cancellation
                .check()
                .map_err(|_| ModelStoreError::Cancelled)?;
            if matches!(artifact.backing, ModelStreamArtifactBacking::Inline(_)) {
                continue;
            }
            self.verify_stream_artifact_record(
                index,
                &artifact.key,
                &artifact.sha256,
                artifact.byte_size,
            )?;
            drop(self.open_stream_artifact(index, &artifact.key, cancellation)?);
        }
        cancellation
            .check()
            .map_err(|_| ModelStoreError::Cancelled)?;
        Ok(bytes)
    }

    pub fn read_verified_artifact_range(
        &self,
        index: &ArtifactIndex,
        source: &VerifiedModelStreamSource,
        artifact_ordinal: usize,
        expected_sha256: &str,
        byte_offset: u64,
        byte_length: u64,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u8>, ModelStoreError> {
        self.validate_stream_source_identity(source)?;
        cancellation
            .check()
            .map_err(|_| ModelStoreError::Cancelled)?;
        self.validate_stream_source(source)?;
        self.limits.check(
            "model stream artifact range bytes",
            byte_length,
            source.maximum_read_bytes,
        )?;
        let artifact = source.artifacts.get(artifact_ordinal).ok_or(
            ModelStoreError::UnknownModelStreamArtifactOrdinal(artifact_ordinal),
        )?;
        if artifact.sha256 != expected_sha256 {
            return Err(ModelStoreError::ModelStreamArtifactIdentityMismatch { artifact_ordinal });
        }
        if !matches!(
            artifact.format,
            ModelFormat::JsonConfig
                | ModelFormat::JsonTokenizer
                | ModelFormat::YamlConfig
                | ModelFormat::SentencePiece
                | ModelFormat::Tiktoken
        ) {
            return Err(ModelStoreError::UnsupportedModelStreamArtifactFormat {
                artifact_ordinal,
                format: artifact.format.clone(),
            });
        }
        let range_end = byte_offset
            .checked_add(byte_length)
            .ok_or(ModelStoreError::Overflow("model stream artifact range"))?;
        if byte_length == 0 || range_end > artifact.byte_size {
            return Err(ModelStoreError::ModelStreamArtifactRangeOutOfBounds {
                artifact_ordinal,
                byte_offset,
                byte_length,
                artifact_byte_length: artifact.byte_size,
            });
        }

        if let ModelStreamArtifactBacking::Inline(inline) = &artifact.backing {
            for candidate in &source.artifacts {
                cancellation
                    .check()
                    .map_err(|_| ModelStoreError::Cancelled)?;
                if matches!(candidate.backing, ModelStreamArtifactBacking::Inline(_)) {
                    continue;
                }
                self.verify_stream_artifact_record(
                    index,
                    &candidate.key,
                    &candidate.sha256,
                    candidate.byte_size,
                )?;
                drop(self.open_stream_artifact(index, &candidate.key, cancellation)?);
            }
            let start = usize::try_from(byte_offset)
                .map_err(|_| ModelStoreError::Overflow("model stream artifact range offset"))?;
            let end = usize::try_from(range_end)
                .map_err(|_| ModelStoreError::Overflow("model stream artifact range end"))?;
            let bytes = inline.get(start..end).ok_or(
                ModelStoreError::ModelStreamArtifactRangeOutOfBounds {
                    artifact_ordinal,
                    byte_offset,
                    byte_length,
                    artifact_byte_length: artifact.byte_size,
                },
            )?;
            let mut result = Vec::new();
            result.try_reserve_exact(bytes.len()).map_err(|_| {
                ModelStoreError::AllocationFailed {
                    requested: byte_length,
                }
            })?;
            result.extend_from_slice(bytes);
            for candidate in &source.artifacts {
                cancellation
                    .check()
                    .map_err(|_| ModelStoreError::Cancelled)?;
                if matches!(candidate.backing, ModelStreamArtifactBacking::Inline(_)) {
                    continue;
                }
                self.verify_stream_artifact_record(
                    index,
                    &candidate.key,
                    &candidate.sha256,
                    candidate.byte_size,
                )?;
                drop(self.open_stream_artifact(index, &candidate.key, cancellation)?);
            }
            cancellation
                .check()
                .map_err(|_| ModelStoreError::Cancelled)?;
            return Ok(result);
        }

        let mut verified_source = None;
        for (ordinal, candidate) in source.artifacts.iter().enumerate() {
            cancellation
                .check()
                .map_err(|_| ModelStoreError::Cancelled)?;
            if matches!(candidate.backing, ModelStreamArtifactBacking::Inline(_)) {
                continue;
            }
            self.verify_stream_artifact_record(
                index,
                &candidate.key,
                &candidate.sha256,
                candidate.byte_size,
            )?;
            let verified = self.open_stream_artifact(index, &candidate.key, cancellation)?;
            if ordinal == artifact_ordinal {
                verified_source = Some(verified);
            }
        }
        let mut verified = verified_source.ok_or(
            ModelStoreError::UnknownModelStreamArtifactOrdinal(artifact_ordinal),
        )?;
        let length = usize::try_from(byte_length)
            .map_err(|_| ModelStoreError::Overflow("model stream artifact range length"))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| ModelStoreError::AllocationFailed {
                requested: byte_length,
            })?;
        cancellation
            .check()
            .map_err(|_| ModelStoreError::Cancelled)?;
        bytes.resize(length, 0);
        verified
            .file_mut()
            .seek(SeekFrom::Start(byte_offset))
            .map_err(|error| ModelStoreError::ModelStreamIo {
                key: artifact.key.clone(),
                message: error.to_string(),
            })?;
        const STREAM_CHUNK_BYTES: usize = 1024 * 1024;
        let mut position = 0_usize;
        while position < bytes.len() {
            cancellation
                .check()
                .map_err(|_| ModelStoreError::Cancelled)?;
            let end = position.saturating_add(STREAM_CHUNK_BYTES).min(bytes.len());
            let chunk = bytes
                .get_mut(position..end)
                .ok_or(ModelStoreError::Overflow(
                    "model stream artifact range chunk",
                ))?;
            verified.file_mut().read_exact(chunk).map_err(|error| {
                ModelStoreError::ModelStreamIo {
                    key: artifact.key.clone(),
                    message: error.to_string(),
                }
            })?;
            position = end;
        }
        cancellation
            .check()
            .map_err(|_| ModelStoreError::Cancelled)?;
        self.verify_opened_stream_artifact(&verified, &artifact.key)?;
        drop(verified);
        for candidate in &source.artifacts {
            cancellation
                .check()
                .map_err(|_| ModelStoreError::Cancelled)?;
            if matches!(candidate.backing, ModelStreamArtifactBacking::Inline(_)) {
                continue;
            }
            self.verify_stream_artifact_record(
                index,
                &candidate.key,
                &candidate.sha256,
                candidate.byte_size,
            )?;
            drop(self.open_stream_artifact(index, &candidate.key, cancellation)?);
        }
        cancellation
            .check()
            .map_err(|_| ModelStoreError::Cancelled)?;
        Ok(bytes)
    }

    pub fn verified_tensor_payload(
        &self,
        index: &ArtifactIndex,
        model: &LoadedModel,
        key: &ArtifactKey,
        cancellation: &CancellationToken,
    ) -> Result<VerifiedModelTensorPayload, ModelStoreError> {
        if !Arc::ptr_eq(&self.identity, &model.store_identity) {
            return Err(ModelStoreError::ForeignModelHandle);
        }
        if cancellation.is_cancelled() {
            return Err(ModelStoreError::Cancelled);
        }
        let artifact_sha256 = model
            .artifact_identities
            .get(key)
            .cloned()
            .ok_or_else(|| ModelStoreError::MissingTensorPayloadSource(key.clone()))?;
        self.verify_model_source(index, model, key)?;

        let mut metadata = Vec::<TensorMetadata>::new();
        for document in &model.documents {
            for tensor in &document.tensors {
                if model.tensor_sources.get(&tensor.name) == Some(key) {
                    metadata
                        .try_reserve(1)
                        .map_err(|_| ModelStoreError::AllocationFailed {
                            requested: u64::try_from(std::mem::size_of::<TensorMetadata>())
                                .unwrap_or(u64::MAX),
                        })?;
                    metadata.push(tensor.clone());
                }
            }
        }
        if metadata.is_empty() {
            return Err(ModelStoreError::EmptyTensorPayload(key.clone()));
        }

        let mut aggregate_bytes = 0_u64;
        for tensor in &metadata {
            aggregate_bytes = aggregate_bytes
                .checked_add(tensor.storage.length)
                .ok_or(ModelStoreError::Overflow("verified tensor payload bytes"))?;
        }
        self.limits.check(
            "verified tensor payload bytes",
            aggregate_bytes,
            self.limits.maximum_aggregate_tensor_bytes,
        )?;
        let mut names = Vec::new();
        names
            .try_reserve_exact(metadata.len())
            .map_err(|_| ModelStoreError::AllocationFailed {
                requested: u64::try_from(metadata.len()).unwrap_or(u64::MAX),
            })?;
        names.extend(metadata.iter().map(|tensor| tensor.name.as_str()));
        let mut bytes_by_name = self.read_tensors(index, model, names, cancellation)?;
        let mut tensors = Vec::new();
        tensors.try_reserve_exact(metadata.len()).map_err(|_| {
            ModelStoreError::AllocationFailed {
                requested: u64::try_from(metadata.len()).unwrap_or(u64::MAX),
            }
        })?;
        for tensor in metadata {
            if cancellation.is_cancelled() {
                return Err(ModelStoreError::Cancelled);
            }
            let bytes = bytes_by_name
                .remove(&tensor.name)
                .ok_or_else(|| ModelStoreError::UnknownTensor(tensor.name.clone()))?;
            tensors.push(VerifiedModelTensor {
                name: tensor.name,
                data_type: tensor.data_type,
                shape: tensor.shape,
                bytes: Arc::from(bytes),
            });
        }
        if !bytes_by_name.is_empty() {
            return Err(ModelStoreError::Overflow(
                "verified tensor payload source ordering",
            ));
        }
        cancellation
            .check()
            .map_err(|_| ModelStoreError::Cancelled)?;
        let nested_string_to_param = model.documents.iter().any(|document| {
            document.source_sha256 == artifact_sha256
                && matches!(
                    &document.payload,
                    ParsedModelPayload::Pytorch { root, .. }
                        if crate::formats::has_nested_string_to_param(root)
                )
        });
        Ok(VerifiedModelTensorPayload {
            artifact_key: key.clone(),
            artifact_sha256,
            tensors,
            _store_identity: self.identity.clone(),
            nested_string_to_param,
        })
    }

    pub fn read_tensors<'name>(
        &self,
        index: &ArtifactIndex,
        model: &LoadedModel,
        tensor_names: impl IntoIterator<Item = &'name str>,
        cancellation: &CancellationToken,
    ) -> Result<BTreeMap<String, Vec<u8>>, ModelStoreError> {
        let mut by_source: BTreeMap<ArtifactKey, Vec<(String, TensorMetadata)>> = BTreeMap::new();
        for tensor_name in tensor_names {
            let (tensor, source_key) = self.checked_tensor(model, tensor_name)?;
            by_source
                .entry(source_key.clone())
                .or_default()
                .push((tensor_name.to_owned(), tensor.clone()));
        }
        let mut output = BTreeMap::new();
        for (source_key, tensors) in by_source {
            self.verify_model_source(index, model, &source_key)?;
            let verified = index.open_verified(&source_key, cancellation)?;
            let source_path = verified.path().to_path_buf();
            if tensors
                .iter()
                .any(|(_, tensor)| tensor.storage.path != source_path)
            {
                return Err(ModelStoreError::StorageSourceChanged { key: source_key });
            }
            let mut file = verified.into_file();
            for (tensor_name, tensor) in tensors {
                if cancellation.is_cancelled() {
                    return Err(ModelStoreError::Cancelled);
                }
                self.limits.check(
                    "resident tensor bytes",
                    tensor.storage.length,
                    self.limits.maximum_tensor_bytes,
                )?;
                let length = usize::try_from(tensor.storage.length)
                    .map_err(|_| ModelStoreError::Overflow("resident tensor length"))?;
                let mut bytes = Vec::new();
                bytes
                    .try_reserve_exact(length)
                    .map_err(|_| ModelStoreError::AllocationFailed {
                        requested: tensor.storage.length,
                    })?;
                bytes.resize(length, 0);
                file.seek(SeekFrom::Start(tensor.storage.offset))
                    .map_err(|error| ModelStoreError::Io {
                        path: source_path.clone(),
                        message: error.to_string(),
                    })?;
                const CHUNK: usize = 1024 * 1024;
                let mut position = 0_usize;
                while position < bytes.len() {
                    if cancellation.is_cancelled() {
                        return Err(ModelStoreError::Cancelled);
                    }
                    let end = position.saturating_add(CHUNK).min(bytes.len());
                    let chunk = bytes
                        .get_mut(position..end)
                        .ok_or(ModelStoreError::Overflow("resident tensor chunk"))?;
                    file.read_exact(chunk)
                        .map_err(|error| ModelStoreError::Io {
                            path: source_path.clone(),
                            message: error.to_string(),
                        })?;
                    position = end;
                }
                output.insert(tensor_name, bytes);
            }
        }
        Ok(output)
    }

    #[cfg(unix)]
    pub fn map_tensor(
        &self,
        index: &ArtifactIndex,
        model: &LoadedModel,
        tensor_name: &str,
        cancellation: &CancellationToken,
    ) -> Result<ReadOnlyTensorMapping, ModelStoreError> {
        let (tensor, source_key) = self.checked_tensor(model, tensor_name)?;
        self.verify_model_source(index, model, source_key)?;
        let verified = index.open_verified(source_key, cancellation)?;
        let source_path = verified.path().to_path_buf();
        if source_path != tensor.storage.path {
            return Err(ModelStoreError::StorageSourceChanged {
                key: source_key.clone(),
            });
        }
        if cancellation.is_cancelled() {
            return Err(ModelStoreError::Cancelled);
        }
        self.limits.check(
            "mapped tensor bytes",
            tensor.storage.length,
            self.limits.maximum_tensor_bytes,
        )?;
        let file = verified.into_file();
        let file_length = file
            .metadata()
            .map_err(|error| ModelStoreError::Io {
                path: source_path.clone(),
                message: error.to_string(),
            })?
            .len();
        let end = tensor
            .storage
            .offset
            .checked_add(tensor.storage.length)
            .ok_or(ModelStoreError::Overflow("mapped tensor range"))?;
        if end > file_length {
            return Err(ModelStoreError::StorageRangeChanged(
                tensor.storage.path.clone(),
            ));
        }
        let page_size = unsafe { getpagesize() };
        if page_size <= 0 {
            return Err(ModelStoreError::MappingUnavailable(
                "operating system returned an invalid page size".to_owned(),
            ));
        }
        let page_size =
            u64::try_from(page_size).map_err(|_| ModelStoreError::Overflow("mapping page size"))?;
        let aligned_offset = tensor.storage.offset / page_size * page_size;
        let view_offset = tensor.storage.offset - aligned_offset;
        let allocation_length = view_offset
            .checked_add(tensor.storage.length)
            .ok_or(ModelStoreError::Overflow("mapping allocation length"))?;
        let allocation_length = usize::try_from(allocation_length)
            .map_err(|_| ModelStoreError::Overflow("mapping allocation length"))?;
        let view_offset = usize::try_from(view_offset)
            .map_err(|_| ModelStoreError::Overflow("mapping view offset"))?;
        let view_length = usize::try_from(tensor.storage.length)
            .map_err(|_| ModelStoreError::Overflow("mapping view length"))?;
        if allocation_length == 0 {
            return Err(ModelStoreError::MappingUnavailable(
                "zero-length tensors do not require a mapping".to_owned(),
            ));
        }
        // The file is opened read-only, the aligned range was bounds-checked, and MAP_PRIVATE
        // prevents model bytes from granting write authority to the artifact.
        let pointer = unsafe {
            mmap(
                std::ptr::null_mut(),
                allocation_length,
                PROT_READ,
                MAP_PRIVATE,
                file.as_raw_fd(),
                i64::try_from(aligned_offset)
                    .map_err(|_| ModelStoreError::Overflow("mapping file offset"))?,
            )
        };
        if pointer as isize == -1 {
            return Err(ModelStoreError::MappingUnavailable(
                std::io::Error::last_os_error().to_string(),
            ));
        }
        let allocation = NonNull::new(pointer).ok_or_else(|| {
            ModelStoreError::MappingUnavailable("mmap returned a null pointer".to_owned())
        })?;
        Ok(ReadOnlyTensorMapping {
            _file: file,
            allocation,
            allocation_length,
            view_offset,
            view_length,
        })
    }

    fn load_single(
        &mut self,
        index: &ArtifactIndex,
        key: &ArtifactKey,
        attempt: u8,
        cancellation: &CancellationToken,
    ) -> Result<Arc<LoadedModel>, ModelStoreError> {
        let verified = index.open_verified(key, cancellation)?;
        let parsed = Arc::new(parse_verified_model_file(
            verified,
            &self.limits,
            cancellation,
        )?);
        self.record(key, attempt, ModelOperationStage::Verifying, None);
        let expected = index
            .record(key)
            .ok_or_else(|| ModelStoreError::UnknownArtifact(key.clone()))?;
        if parsed.source_sha256 != expected.sha256 || parsed.source_size != expected.byte_size {
            return Err(ModelStoreError::ArtifactChanged { key: key.clone() });
        }
        let tensors = unique_tensors(std::slice::from_ref(&parsed))?;
        let accounting = accounting(std::slice::from_ref(&parsed), &tensors, &self.limits)?;
        Ok(Arc::new(LoadedModel {
            identity: parsed.source_sha256.clone(),
            ordered_artifact_keys: vec![key.clone()],
            artifact_identities: BTreeMap::from([(key.clone(), parsed.source_sha256.clone())]),
            tensor_sources: parsed
                .tensors
                .iter()
                .map(|tensor| (tensor.name.clone(), key.clone()))
                .collect(),
            documents: vec![parsed],
            tensors,
            accounting,
            store_identity: self.identity.clone(),
        }))
    }

    fn load_sharded(
        &mut self,
        index: &ArtifactIndex,
        key: &ArtifactKey,
        attempt: u8,
        cancellation: &CancellationToken,
    ) -> Result<Arc<LoadedModel>, ModelStoreError> {
        let verified = index.open_verified(key, cancellation)?;
        let index_document = Arc::new(parse_verified_model_file(
            verified,
            &self.limits,
            cancellation,
        )?);
        self.record(key, attempt, ModelOperationStage::Verifying, None);
        let index_record = index
            .record(key)
            .ok_or_else(|| ModelStoreError::UnknownArtifact(key.clone()))?;
        if index_document.source_sha256 != index_record.sha256
            || index_document.source_size != index_record.byte_size
        {
            return Err(ModelStoreError::ArtifactChanged { key: key.clone() });
        }
        let ParsedModelPayload::Json(value) = &index_document.payload else {
            return Err(ModelStoreError::InvalidShardIndex(
                "shard index is not JSON".to_owned(),
            ));
        };
        let object = value.as_object().ok_or_else(|| {
            ModelStoreError::InvalidShardIndex("root must be an object".to_owned())
        })?;
        let weight_map = object
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                ModelStoreError::InvalidShardIndex("weight_map must be an object".to_owned())
            })?;
        self.limits.check(
            "sharded tensor count",
            u64::try_from(weight_map.len()).unwrap_or(u64::MAX),
            self.limits.maximum_tensors,
        )?;
        let mut shard_names = BTreeSet::new();
        let mut expected = BTreeMap::<PathBuf, BTreeSet<String>>::new();
        for (tensor_name, shard) in weight_map {
            let shard = shard.as_str().ok_or_else(|| {
                ModelStoreError::InvalidShardIndex(format!(
                    "shard for tensor {tensor_name:?} is not a string"
                ))
            })?;
            let shard_path = normalize_shard_path(shard)?;
            shard_names.insert(shard_path.clone());
            expected
                .entry(shard_path)
                .or_default()
                .insert(tensor_name.clone());
        }
        self.limits.check(
            "model shard count",
            u64::try_from(shard_names.len()).unwrap_or(u64::MAX),
            self.limits.maximum_archive_entries,
        )?;
        let parent = key
            .relative_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new(""));
        let mut documents = Vec::new();
        documents.push(index_document.clone());
        let mut ordered_artifact_keys = vec![key.clone()];
        let mut tensor_sources = BTreeMap::new();
        let mut artifact_identities =
            BTreeMap::from([(key.clone(), index_document.source_sha256.clone())]);
        let mut identity = index_document.source_sha256.clone();
        for shard in shard_names {
            if cancellation.is_cancelled() {
                return Err(ModelStoreError::Cancelled);
            }
            let relative = parent.join(&shard);
            let shard_key = ArtifactKey::new(key.root_id.clone(), relative)?;
            let shard_record = index
                .record(&shard_key)
                .ok_or_else(|| ModelStoreError::MissingShard(shard.clone()))?;
            if shard_record.availability == ArtifactAvailability::Missing {
                return Err(ModelStoreError::MissingShard(shard));
            }
            let verified = index.open_verified(&shard_key, cancellation)?;
            let parsed = Arc::new(parse_verified_model_file(
                verified,
                &self.limits,
                cancellation,
            )?);
            if parsed.source_sha256 != shard_record.sha256
                || parsed.source_size != shard_record.byte_size
            {
                return Err(ModelStoreError::ArtifactChanged {
                    key: shard_key.clone(),
                });
            }
            let expected_names = expected.remove(&shard).ok_or_else(|| {
                ModelStoreError::InvalidShardIndex(format!(
                    "shard {shard:?} has no declared tensor set"
                ))
            })?;
            let actual_names = parsed
                .tensors
                .iter()
                .map(|tensor| tensor.name.clone())
                .collect::<BTreeSet<_>>();
            if actual_names != expected_names {
                return Err(ModelStoreError::InvalidShardIndex(format!(
                    "shard {shard:?} tensors do not match its weight_map declarations"
                )));
            }
            for tensor_name in actual_names {
                tensor_sources.insert(tensor_name, shard_key.clone());
            }
            artifact_identities.insert(shard_key.clone(), parsed.source_sha256.clone());
            ordered_artifact_keys.push(shard_key);
            identity.push(':');
            identity.push_str(&parsed.source_sha256);
            documents.push(parsed);
        }
        let tensors = unique_tensors(&documents)?;
        if !expected.is_empty() {
            return Err(ModelStoreError::InvalidShardIndex(
                "weight_map contains an unvisited shard".to_owned(),
            ));
        }
        if tensors.len() != weight_map.len() {
            return Err(ModelStoreError::InvalidShardIndex(
                "shards contain tensors absent from weight_map".to_owned(),
            ));
        }
        let accounting = accounting(&documents, &tensors, &self.limits)?;
        Ok(Arc::new(LoadedModel {
            identity,
            ordered_artifact_keys,
            artifact_identities,
            documents,
            tensors,
            tensor_sources,
            accounting,
            store_identity: self.identity.clone(),
        }))
    }

    fn record(
        &mut self,
        key: &ArtifactKey,
        attempt: u8,
        stage: ModelOperationStage,
        error: Option<String>,
    ) {
        if self.operations.len() == MAX_MODEL_OPERATION_RECORDS {
            self.operations.remove(0);
        }
        self.operations.push(ModelOperationRecord {
            key: key.clone(),
            attempt,
            stage,
            error,
        });
    }

    fn checked_tensor<'model>(
        &self,
        model: &'model LoadedModel,
        tensor_name: &str,
    ) -> Result<(&'model TensorMetadata, &'model ArtifactKey), ModelStoreError> {
        if !Arc::ptr_eq(&self.identity, &model.store_identity) {
            return Err(ModelStoreError::ForeignModelHandle);
        }
        let tensor = model
            .tensors
            .get(tensor_name)
            .ok_or_else(|| ModelStoreError::UnknownTensor(tensor_name.to_owned()))?;
        let source = model
            .tensor_sources
            .get(tensor_name)
            .ok_or_else(|| ModelStoreError::MissingTensorSource(tensor_name.to_owned()))?;
        Ok((tensor, source))
    }

    fn verify_model_source(
        &self,
        index: &ArtifactIndex,
        model: &LoadedModel,
        source: &ArtifactKey,
    ) -> Result<(), ModelStoreError> {
        let expected = model.artifact_identities.get(source).ok_or_else(|| {
            ModelStoreError::MissingTensorSource(
                source.relative_path.to_string_lossy().into_owned(),
            )
        })?;
        let current = index
            .record(source)
            .filter(|record| record.availability == ArtifactAvailability::Present)
            .ok_or_else(|| ModelStoreError::MissingArtifact(source.clone()))?;
        if &current.sha256 != expected {
            return Err(ModelStoreError::ArtifactChanged {
                key: source.clone(),
            });
        }
        Ok(())
    }

    fn validate_stream_source_identity(
        &self,
        source: &VerifiedModelStreamSource,
    ) -> Result<(), ModelStoreError> {
        let Some(identity) = source.store_identity.upgrade() else {
            return Err(ModelStoreError::StaleModelStreamSource);
        };
        if !Arc::ptr_eq(&self.identity, &identity) {
            return Err(ModelStoreError::ForeignModelStreamSource);
        }
        Ok(())
    }

    fn validate_stream_source(
        &self,
        source: &VerifiedModelStreamSource,
    ) -> Result<(), ModelStoreError> {
        self.validate_stream_source_identity(source)?;
        if source.model_identity.is_empty() || source.artifacts.is_empty() {
            return Err(ModelStoreError::InvalidModelStreamSourceMetadata(
                "model or artifact identity is empty".to_owned(),
            ));
        }
        if source.limits != self.limits {
            return Err(ModelStoreError::InvalidModelStreamSourceMetadata(
                "parser limits differ from the owning model store".to_owned(),
            ));
        }
        if source.maximum_read_bytes != MODEL_STREAM_MAXIMUM_READ_BYTES {
            return Err(ModelStoreError::InvalidModelStreamSourceMetadata(
                "per-call read limit differs from the model-store stream contract".to_owned(),
            ));
        }
        self.limits.check(
            "model stream artifact count",
            u64::try_from(source.artifacts.len()).unwrap_or(u64::MAX),
            self.limits.maximum_archive_entries,
        )?;
        self.limits.check(
            "model stream tensor count",
            u64::try_from(source.tensors.len()).unwrap_or(u64::MAX),
            self.limits.maximum_tensors,
        )?;

        let mut artifact_keys = BTreeSet::new();
        let mut semantic_artifacts = BTreeSet::new();
        for artifact in &source.artifacts {
            let backing_valid = match &artifact.backing {
                ModelStreamArtifactBacking::Indexed => {
                    !matches!(
                        artifact.semantic,
                        ModelStreamSemanticArtifact::SafetensorsMetadata { .. }
                    ) && artifact_keys.insert(artifact.key.clone())
                }
                ModelStreamArtifactBacking::Inline(bytes) => {
                    matches!(
                        artifact.semantic,
                        ModelStreamSemanticArtifact::SafetensorsMetadata { .. }
                    ) && artifact.byte_size == u64::try_from(bytes.len()).unwrap_or(u64::MAX)
                        && artifact.sha256 == format!("{:x}", Sha256::digest(bytes))
                }
            };
            if !backing_valid
                || artifact.sha256.len() != 64
                || !artifact.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
                || !model_stream_semantic_format_matches(artifact.semantic, &artifact.format)
                || (artifact.semantic != ModelStreamSemanticArtifact::None
                    && !semantic_artifacts.insert(artifact.semantic))
            {
                return Err(ModelStoreError::InvalidModelStreamSourceMetadata(format!(
                    "artifact {:?} has a duplicate key or malformed identity",
                    artifact.key
                )));
            }
            if let ModelStreamSemanticArtifact::SafetensorsMetadata {
                source_artifact_ordinal,
            } = artifact.semantic
            {
                let source_artifact = usize::try_from(source_artifact_ordinal)
                    .ok()
                    .and_then(|ordinal| source.artifacts.get(ordinal));
                if !source_artifact.is_some_and(|source_artifact| {
                    source_artifact.format == ModelFormat::Safetensors
                        && matches!(source_artifact.backing, ModelStreamArtifactBacking::Indexed)
                }) {
                    return Err(ModelStoreError::InvalidModelStreamSourceMetadata(
                        "safetensors metadata names a missing source artifact".to_owned(),
                    ));
                }
            }
        }

        let mut tensor_names = BTreeSet::new();
        let mut aggregate_tensor_bytes = 0_u64;
        let mut ranges = BTreeMap::<ArtifactKey, Vec<(&str, u64, u64)>>::new();
        for tensor in &source.tensors {
            if tensor.name.is_empty()
                || tensor.data_type.is_empty()
                || !tensor_names.insert(tensor.name.as_str())
                || !artifact_keys.contains(&tensor.artifact_key)
                || source
                    .artifacts
                    .get(tensor.artifact_ordinal)
                    .map(|artifact| &artifact.key)
                    != Some(&tensor.artifact_key)
            {
                return Err(ModelStoreError::InvalidModelStreamSourceMetadata(format!(
                    "tensor {:?} has malformed or duplicate metadata",
                    tensor.name
                )));
            }
            self.limits.check(
                "model stream tensor bytes",
                tensor.byte_length,
                self.limits.maximum_tensor_bytes,
            )?;
            let end = tensor
                .byte_offset
                .checked_add(tensor.byte_length)
                .ok_or_else(|| {
                    ModelStoreError::InvalidModelStreamSourceMetadata(format!(
                        "tensor {:?} range overflows",
                        tensor.name
                    ))
                })?;
            let artifact = source
                .artifacts
                .iter()
                .find(|artifact| artifact.key == tensor.artifact_key)
                .ok_or_else(|| {
                    ModelStoreError::InvalidModelStreamSourceMetadata(format!(
                        "tensor {:?} has no source artifact",
                        tensor.name
                    ))
                })?;
            if end > artifact.byte_size {
                return Err(ModelStoreError::InvalidModelStreamSourceMetadata(format!(
                    "tensor {:?} exceeds its source artifact",
                    tensor.name
                )));
            }
            aggregate_tensor_bytes = aggregate_tensor_bytes
                .checked_add(tensor.byte_length)
                .ok_or(ModelStoreError::Overflow(
                    "model stream aggregate tensor bytes",
                ))?;
            ranges
                .entry(tensor.artifact_key.clone())
                .or_default()
                .push((tensor.name.as_str(), tensor.byte_offset, end));
        }
        self.limits.check(
            "model stream aggregate tensor bytes",
            aggregate_tensor_bytes,
            self.limits.maximum_aggregate_tensor_bytes,
        )?;
        if aggregate_tensor_bytes != source.aggregate_tensor_bytes {
            return Err(ModelStoreError::InvalidModelStreamSourceMetadata(
                "aggregate tensor bytes do not match the manifest".to_owned(),
            ));
        }
        for (key, mut source_ranges) in ranges {
            source_ranges.sort_unstable_by_key(|(_, start, end)| (*start, *end));
            for ranges in source_ranges.windows(2) {
                let [first, second] = ranges else {
                    continue;
                };
                if second.1 < first.2 {
                    return Err(ModelStoreError::OverlappingModelStreamTensorRanges {
                        key,
                        first_tensor: first.0.to_owned(),
                        second_tensor: second.0.to_owned(),
                    });
                }
            }
        }
        Ok(())
    }

    fn verify_stream_artifact_record<'index>(
        &self,
        index: &'index ArtifactIndex,
        key: &ArtifactKey,
        expected_sha256: &str,
        expected_byte_size: u64,
    ) -> Result<&'index crate::artifact_index::ArtifactRecord, ModelStoreError> {
        let record = index
            .record(key)
            .filter(|record| record.availability == ArtifactAvailability::Present)
            .ok_or_else(|| ModelStoreError::MissingArtifact(key.clone()))?;
        if record.sha256 != expected_sha256 || record.byte_size != expected_byte_size {
            return Err(ModelStoreError::ArtifactChanged { key: key.clone() });
        }
        Ok(record)
    }

    fn open_stream_artifact(
        &self,
        index: &ArtifactIndex,
        key: &ArtifactKey,
        cancellation: &CancellationToken,
    ) -> Result<VerifiedArtifactFile, ModelStoreError> {
        index
            .open_verified(key, cancellation)
            .map_err(|error| match error {
                ArtifactIndexError::Cancelled => ModelStoreError::Cancelled,
                ArtifactIndexError::Missing(_) => ModelStoreError::MissingArtifact(key.clone()),
                ArtifactIndexError::Limit(error) => ModelStoreError::Limit(error),
                _ => ModelStoreError::ArtifactChanged { key: key.clone() },
            })
    }

    fn verify_opened_stream_artifact(
        &self,
        verified: &VerifiedArtifactFile,
        key: &ArtifactKey,
    ) -> Result<(), ModelStoreError> {
        verified
            .verify_unchanged()
            .map_err(|_| ModelStoreError::ArtifactChanged { key: key.clone() })
    }
}

fn verified_model_stream_source_identity(
    artifacts: &[VerifiedModelStreamArtifact],
) -> Result<String, ModelStoreError> {
    let mut hasher = Sha256::new();
    hasher.update(b"zed-sim-verified-model-stream-source-v1");
    hasher.update(
        u64::try_from(artifacts.len())
            .map_err(|_| ModelStoreError::Overflow("model stream artifact count"))?
            .to_le_bytes(),
    );
    for (ordinal, artifact) in artifacts.iter().enumerate() {
        hasher.update(
            u64::try_from(ordinal)
                .map_err(|_| ModelStoreError::Overflow("model stream artifact ordinal"))?
                .to_le_bytes(),
        );
        hasher.update(artifact.sha256.as_bytes());
        hasher.update(artifact.byte_size.to_le_bytes());
        hasher.update([match artifact.format {
            ModelFormat::Safetensors => 0,
            ModelFormat::PytorchArchive => 1,
            ModelFormat::Gguf => 2,
            ModelFormat::JsonConfig => 3,
            ModelFormat::JsonTokenizer => 4,
            ModelFormat::YamlConfig => 5,
            ModelFormat::SentencePiece => 6,
            ModelFormat::Tiktoken => 7,
        }]);
        hasher.update([match artifact.nested_state_disposition {
            ModelStreamNestedStateDisposition::Flat => 0,
            ModelStreamNestedStateDisposition::NestedStringToParam => 1,
        }]);
        match artifact.semantic {
            ModelStreamSemanticArtifact::None => {}
            ModelStreamSemanticArtifact::SafetensorsMetadata {
                source_artifact_ordinal,
            } => {
                hasher.update([0x7f]);
                hasher.update(source_artifact_ordinal.to_le_bytes());
            }
            ModelStreamSemanticArtifact::ClipConfiguration { target_ordinal } => {
                hasher.update([0x80, target_ordinal]);
            }
            ModelStreamSemanticArtifact::ClipTokenizerJson { target_ordinal } => {
                hasher.update([0x81, target_ordinal]);
            }
            ModelStreamSemanticArtifact::ClipSentencePiece { target_ordinal } => {
                hasher.update([0x82, target_ordinal]);
            }
            ModelStreamSemanticArtifact::ClipTiktoken { target_ordinal } => {
                hasher.update([0x83, target_ordinal]);
            }
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn model_stream_semantic_format_matches(
    semantic: ModelStreamSemanticArtifact,
    format: &ModelFormat,
) -> bool {
    match semantic {
        ModelStreamSemanticArtifact::None => true,
        ModelStreamSemanticArtifact::SafetensorsMetadata { .. } => {
            format == &ModelFormat::JsonConfig
        }
        ModelStreamSemanticArtifact::ClipConfiguration { target_ordinal } => {
            target_ordinal < 4 && format == &ModelFormat::JsonConfig
        }
        ModelStreamSemanticArtifact::ClipTokenizerJson { target_ordinal } => {
            target_ordinal < 4 && format == &ModelFormat::JsonTokenizer
        }
        ModelStreamSemanticArtifact::ClipSentencePiece { target_ordinal } => {
            target_ordinal < 4 && format == &ModelFormat::SentencePiece
        }
        ModelStreamSemanticArtifact::ClipTiktoken { target_ordinal } => {
            target_ordinal < 4 && format == &ModelFormat::Tiktoken
        }
    }
}

fn model_weight_statistic_native_bytes(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    source: &[u8],
    dtype: sim_tensor::DType,
    tensor_name: &str,
) -> Result<CpuWorkspaceVec<u8>, ModelWeightStatisticError> {
    let width = usize::try_from(dtype.byte_width()).map_err(|_| TensorError::ShapeOverflow)?;
    if !source.len().is_multiple_of(width) {
        return Err(ModelWeightStatisticError::UnalignedStorage {
            tensor: tensor_name.to_owned(),
            storage_dtype: dtype,
            byte_length: source.len(),
        });
    }
    let mut native = backend.workspace_vec(context, source.len())?;
    for encoded in source.chunks_exact(width) {
        context.check()?;
        if cfg!(target_endian = "little") || width == 1 {
            for byte in encoded {
                native.try_push(*byte)?;
            }
        } else {
            for byte in encoded.iter().rev() {
                native.try_push(*byte)?;
            }
        }
    }
    Ok(native)
}

#[derive(Debug, thiserror::Error)]
pub enum ModelWeightStatisticError {
    #[error(transparent)]
    Store(#[from] ModelStoreError),
    #[error(transparent)]
    Family(#[from] ModelFamilyError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(
        "model weight statistic tensor {tensor} has {byte_length} bytes, which is not aligned to {storage_dtype:?}"
    )]
    UnalignedStorage {
        tensor: String,
        storage_dtype: sim_tensor::DType,
        byte_length: usize,
    },
    #[error("model weight statistic allocation of {requested} records failed")]
    AllocationFailed { requested: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ModelStoreError {
    #[error("model loading was cancelled")]
    Cancelled,
    #[error(transparent)]
    Index(ArtifactIndexError),
    #[error(transparent)]
    Format(#[from] ModelFormatError),
    #[error(transparent)]
    Limit(#[from] crate::parser_limits::ParserLimitError),
    #[error(transparent)]
    FamilyProbe(#[from] ModelFamilyProbeError),
    #[error("artifact {0:?} is not indexed")]
    UnknownArtifact(ArtifactKey),
    #[error("artifact {0:?} is missing")]
    MissingArtifact(ArtifactKey),
    #[error("model artifact changed during loading: {key:?}")]
    ArtifactChanged { key: ArtifactKey },
    #[error("model handle belongs to another model store")]
    ForeignModelHandle,
    #[error("verified model stream source belongs to another model store")]
    ForeignModelStreamSource,
    #[error("verified model stream source belongs to a model store that no longer exists")]
    StaleModelStreamSource,
    #[error("verified model stream source metadata is invalid: {0}")]
    InvalidModelStreamSourceMetadata(String),
    #[error(
        "verified model stream source has overlapping tensor ranges in {key:?}: {first_tensor:?} and {second_tensor:?}"
    )]
    OverlappingModelStreamTensorRanges {
        key: ArtifactKey,
        first_tensor: String,
        second_tensor: String,
    },
    #[error(
        "model stream tensor {tensor:?} range {byte_offset}+{byte_length} exceeds {tensor_byte_length} bytes"
    )]
    ModelStreamTensorRangeOutOfBounds {
        tensor: String,
        byte_offset: u64,
        byte_length: u64,
        tensor_byte_length: u64,
    },
    #[error("model stream artifact ordinal {0} is unknown")]
    UnknownModelStreamArtifactOrdinal(usize),
    #[error("model stream artifact {artifact_ordinal} identity does not match the opened source")]
    ModelStreamArtifactIdentityMismatch { artifact_ordinal: usize },
    #[error(
        "model stream artifact {artifact_ordinal} format {format:?} cannot be streamed as auxiliary bytes"
    )]
    UnsupportedModelStreamArtifactFormat {
        artifact_ordinal: usize,
        format: ModelFormat,
    },
    #[error(
        "model stream artifact {artifact_ordinal} range {byte_offset}+{byte_length} exceeds {artifact_byte_length} bytes"
    )]
    ModelStreamArtifactRangeOutOfBounds {
        artifact_ordinal: usize,
        byte_offset: u64,
        byte_length: u64,
        artifact_byte_length: u64,
    },
    #[error(
        "loaded model artifact {key:?} identity mismatch: expected {expected_sha256}, actual {actual_sha256:?}"
    )]
    LoadedArtifactIdentityMismatch {
        key: ArtifactKey,
        expected_sha256: String,
        actual_sha256: Option<String>,
    },
    #[error("model tensor {0:?} is unknown")]
    UnknownTensor(String),
    #[error("model tensor {0:?} has no canonical artifact source")]
    MissingTensorSource(String),
    #[error("loaded model does not contain artifact {0:?} as a tensor payload source")]
    MissingTensorPayloadSource(ArtifactKey),
    #[error("loaded model artifact {0:?} contains no parsed tensors")]
    EmptyTensorPayload(ArtifactKey),
    #[error("loaded model artifact {0:?} contains no verified SentencePiece vocabulary")]
    MissingSentencePieceVocabulary(ArtifactKey),
    #[error("model tensor source {key:?} no longer resolves to its validated file")]
    StorageSourceChanged { key: ArtifactKey },
    #[error("model tensor storage range changed: {0}")]
    StorageRangeChanged(PathBuf),
    #[error("model shard {0} is missing")]
    MissingShard(PathBuf),
    #[error("invalid model shard index: {0}")]
    InvalidShardIndex(String),
    #[error("duplicate tensor name {0:?} across model shards")]
    DuplicateTensor(String),
    #[error("model I/O failed for {path}: {message}")]
    Io { path: PathBuf, message: String },
    #[error("model stream I/O failed for {key:?}: {message}")]
    ModelStreamIo { key: ArtifactKey, message: String },
    #[error("model allocation of {requested} bytes failed")]
    AllocationFailed { requested: u64 },
    #[error("model byte arithmetic overflowed while computing {0}")]
    Overflow(&'static str),
    #[error("parsed model metadata normalization failed: {0}")]
    MetadataNormalization(String),
    #[error("read-only model mapping is unavailable: {0}")]
    MappingUnavailable(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelFamilyProbeErrorKind {
    TensorLimit,
    FormatLimit,
    TensorName,
    Shape,
    Dimension,
    StorageDType,
    Format,
    Metadata,
    BlockPattern,
    Configuration,
    Other,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("model-family probe projection failed ({kind:?}): {message}")]
pub struct ModelFamilyProbeError {
    kind: ModelFamilyProbeErrorKind,
    message: String,
}

impl ModelFamilyProbeError {
    pub fn kind(&self) -> ModelFamilyProbeErrorKind {
        self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl From<ModelFamilyError> for ModelFamilyProbeError {
    fn from(error: ModelFamilyError) -> Self {
        let kind = match &error {
            ModelFamilyError::ProbeTensorLimit { .. } => ModelFamilyProbeErrorKind::TensorLimit,
            ModelFamilyError::ProbeFormatLimit { .. } => ModelFamilyProbeErrorKind::FormatLimit,
            ModelFamilyError::InvalidProbeTensorName(_) => ModelFamilyProbeErrorKind::TensorName,
            ModelFamilyError::InvalidProbeShape { .. } => ModelFamilyProbeErrorKind::Shape,
            ModelFamilyError::ProbeDimensionOverflow
            | ModelFamilyError::ProbeDimensionOutOfBounds { .. } => {
                ModelFamilyProbeErrorKind::Dimension
            }
            ModelFamilyError::UnknownStorageDType(_) => ModelFamilyProbeErrorKind::StorageDType,
            ModelFamilyError::UnsupportedProbeFormat(_) => ModelFamilyProbeErrorKind::Format,
            ModelFamilyError::InvalidProbeMetadata(_)
            | ModelFamilyError::ConflictingProbeMetadata(_)
            | ModelFamilyError::ProbeMetadataOverflow
            | ModelFamilyError::ProbeMetadataLimit { .. } => ModelFamilyProbeErrorKind::Metadata,
            ModelFamilyError::InvalidBlockPattern(_) => ModelFamilyProbeErrorKind::BlockPattern,
            ModelFamilyError::InvalidProbeConfiguration(_) => {
                ModelFamilyProbeErrorKind::Configuration
            }
            _ => ModelFamilyProbeErrorKind::Other,
        };
        Self {
            kind,
            message: error.to_string(),
        }
    }
}

impl From<ArtifactIndexError> for ModelStoreError {
    fn from(error: ArtifactIndexError) -> Self {
        match error {
            ArtifactIndexError::Cancelled => Self::Cancelled,
            error => Self::Index(error),
        }
    }
}

#[cfg(unix)]
const PROT_READ: i32 = 0x1;
#[cfg(unix)]
const MAP_PRIVATE: i32 = 0x2;

#[cfg(unix)]
unsafe extern "C" {
    fn mmap(
        address: *mut c_void,
        length: usize,
        protection: i32,
        flags: i32,
        file_descriptor: i32,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(address: *mut c_void, length: usize) -> i32;
    fn getpagesize() -> i32;
}

fn unique_tensors(
    documents: &[Arc<ParsedModel>],
) -> Result<BTreeMap<String, TensorMetadata>, ModelStoreError> {
    let mut result = BTreeMap::new();
    for document in documents {
        for tensor in &document.tensors {
            if result.insert(tensor.name.clone(), tensor.clone()).is_some() {
                return Err(ModelStoreError::DuplicateTensor(tensor.name.clone()));
            }
        }
    }
    Ok(result)
}

fn parsed_format_fact(document: &ParsedModel) -> Result<ModelParsedFormatFact, ModelStoreError> {
    let identity = match document.format {
        ModelFormat::Safetensors => "safetensors",
        ModelFormat::PytorchArchive => "pytorch_archive",
        ModelFormat::Gguf => "gguf",
        ModelFormat::JsonConfig => "json_config",
        ModelFormat::JsonTokenizer => "json_tokenizer",
        ModelFormat::YamlConfig => "yaml_config",
        ModelFormat::SentencePiece => "sentence_piece",
        ModelFormat::Tiktoken => "tiktoken",
    };
    let metadata = match &document.payload {
        ParsedModelPayload::Safetensors { metadata } => metadata.clone(),
        ParsedModelPayload::Gguf { version, metadata } => {
            let mut values = BTreeMap::from([("gguf.version".to_owned(), version.to_string())]);
            for (key, value) in metadata {
                values.insert(key.clone(), normalize_gguf_value(value)?);
            }
            values
        }
        ParsedModelPayload::Json(value) | ParsedModelPayload::Yaml(value) => {
            normalize_structured_metadata(value)?
        }
        ParsedModelPayload::SentencePiece { vocabulary } => BTreeMap::from([(
            "piece_count".to_owned(),
            vocabulary.entries().len().to_string(),
        )]),
        ParsedModelPayload::Tiktoken { token_count } => {
            BTreeMap::from([("token_count".to_owned(), token_count.to_string())])
        }
        ParsedModelPayload::Pytorch {
            archive_entries, ..
        } => BTreeMap::from([(
            "archive_entry_count".to_owned(),
            archive_entries.len().to_string(),
        )]),
    };
    Ok(ModelParsedFormatFact {
        identity: identity.to_owned(),
        metadata,
    })
}

fn normalize_structured_metadata(
    value: &serde_json::Value,
) -> Result<BTreeMap<String, String>, ModelStoreError> {
    let mut metadata = BTreeMap::new();
    match value {
        serde_json::Value::Object(values) => {
            for (key, value) in values {
                metadata.insert(
                    key.clone(),
                    serde_json::to_string(value).map_err(|error| {
                        ModelStoreError::MetadataNormalization(error.to_string())
                    })?,
                );
            }
        }
        value => {
            metadata.insert(
                "document".to_owned(),
                serde_json::to_string(value)
                    .map_err(|error| ModelStoreError::MetadataNormalization(error.to_string()))?,
            );
        }
    }
    Ok(metadata)
}

fn normalize_gguf_value(value: &GgufValue) -> Result<String, ModelStoreError> {
    let value = match value {
        GgufValue::Unsigned(value) => json_scalar("unsigned", value.to_string()),
        GgufValue::Signed(value) => json_scalar("signed", value.to_string()),
        GgufValue::FloatBits(value) => json_scalar("float_bits", format!("{value:016x}")),
        GgufValue::Boolean(value) => json_scalar("boolean", value.to_string()),
        GgufValue::String(value) => return Ok(value.clone()),
        GgufValue::Array(values) => {
            let values = values
                .iter()
                .map(normalize_gguf_value)
                .collect::<Result<Vec<_>, _>>()?;
            serde_json::to_string(&values)
                .map_err(|error| ModelStoreError::MetadataNormalization(error.to_string()))?
        }
    };
    Ok(value)
}

fn json_scalar(kind: &str, value: String) -> String {
    format!("{kind}:{value}")
}

fn accounting(
    documents: &[Arc<ParsedModel>],
    tensors: &BTreeMap<String, TensorMetadata>,
    limits: &ParserLimits,
) -> Result<ModelLoadAccounting, ModelStoreError> {
    let source_bytes = documents.iter().try_fold(0_u64, |total, document| {
        total
            .checked_add(document.source_size)
            .ok_or(ModelStoreError::Overflow("source bytes"))
    })?;
    let tensor_bytes = tensors.values().try_fold(0_u64, |total, tensor| {
        total
            .checked_add(tensor.storage.length)
            .ok_or(ModelStoreError::Overflow("tensor bytes"))
    })?;
    limits.check(
        "model tensor bytes",
        tensor_bytes,
        limits.maximum_aggregate_tensor_bytes,
    )?;
    Ok(ModelLoadAccounting {
        source_bytes,
        tensor_bytes,
        tensor_count: u64::try_from(tensors.len()).unwrap_or(u64::MAX),
        resident_bytes: 0,
    })
}

fn is_shard_index(path: &std::path::Path) -> bool {
    path.file_name()
        .and_then(|value| value.to_str())
        .is_some_and(|name| name.to_ascii_lowercase().ends_with(".index.json"))
}

fn normalize_shard_path(value: &str) -> Result<PathBuf, ModelStoreError> {
    if value.contains('\\') || value.contains('\0') {
        return Err(ModelStoreError::InvalidShardIndex(format!(
            "unsafe shard path {value:?}"
        )));
    }
    let path = PathBuf::from(value);
    if path
        .components()
        .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(ModelStoreError::InvalidShardIndex(format!(
            "unsafe shard path {value:?}"
        )));
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact_index::ArtifactRoot;
    use std::{fs, io::Write};

    #[test]
    fn sentence_piece_format_identity_matches_the_family_probe_allowlist()
    -> Result<(), ModelStoreError> {
        let document = ParsedModel {
            format: ModelFormat::SentencePiece,
            tensors: Vec::new(),
            payload: ParsedModelPayload::SentencePiece {
                vocabulary: crate::formats::SentencePieceVocabulary::fixture_for_test(7),
            },
            source_size: 0,
            source_sha256: "0".repeat(64),
        };
        let fact = parsed_format_fact(&document)?;
        assert_eq!(fact.identity, "sentence_piece");
        assert_eq!(
            fact.metadata.get("piece_count").map(String::as_str),
            Some("7")
        );
        ModelProbe::from_parsed_facts(ModelParsedFacts {
            tensors: BTreeMap::new(),
            formats: vec![fact],
        })
        .map_err(ModelFamilyProbeError::from)?;
        Ok(())
    }

    fn write_safetensors(
        path: &std::path::Path,
        tensor_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let header =
            format!(r#"{{"{tensor_name}":{{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}}}"#);
        let mut file = File::create(path)?;
        file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
        file.write_all(header.as_bytes())?;
        file.write_all(&[7])?;
        Ok(())
    }

    fn write_stream_safetensors(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let header = r#"{"first":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"second":{"dtype":"U8","shape":[4],"data_offsets":[4,8]}}"#;
        let mut file = File::create(path)?;
        file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
        file.write_all(header.as_bytes())?;
        file.write_all(&[1, 2, 3, 4, 5, 6, 7, 8])?;
        Ok(())
    }

    fn write_stream_safetensors_with_metadata(
        path: &PathBuf,
        metadata: &BTreeMap<String, String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let header = serde_json::to_vec(&serde_json::json!({
            "__metadata__": metadata,
            "weight": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
        }))?;
        let mut file = File::create(path)?;
        file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
        file.write_all(&header)?;
        file.write_all(&[1, 2, 3, 4])?;
        Ok(())
    }

    fn write_maximum_stream_safetensors(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let header = format!(
            r#"{{"maximum":{{"dtype":"U8","shape":[{MODEL_STREAM_MAXIMUM_READ_BYTES}],"data_offsets":[0,{MODEL_STREAM_MAXIMUM_READ_BYTES}]}}}}"#
        );
        let mut file = File::create(path)?;
        file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
        file.write_all(header.as_bytes())?;
        let length = usize::try_from(MODEL_STREAM_MAXIMUM_READ_BYTES)?;
        file.write_all(&vec![0x5a; length])?;
        Ok(())
    }

    fn stream_source_fixture() -> Result<
        (
            tempfile::TempDir,
            ArtifactIndex,
            ArtifactKey,
            ModelStore,
            Arc<LoadedModel>,
            VerifiedModelStreamSource,
        ),
        Box<dyn std::error::Error>,
    > {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("stream.safetensors");
        write_stream_safetensors(&path)?;
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "models",
            "checkpoints",
            directory.path(),
            ["safetensors"],
        )?)?;
        let cancellation = CancellationToken::default();
        index.refresh(&cancellation)?;
        let key = ArtifactKey::new("models", "stream.safetensors")?;
        let mut store = ModelStore::new(ParserLimits::default())?;
        let model = store.load(&index, &key, &cancellation)?;
        let source = store.verified_stream_source(&index, &model, &cancellation)?;
        Ok((directory, index, key, store, model, source))
    }

    #[test]
    fn model_store_stream_source_reads_only_checked_tensor_ranges()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, index, _key, store, _model, source) = stream_source_fixture()?;
        assert_eq!(source.artifacts().len(), 1);
        assert_eq!(source.artifacts()[0].format(), &ModelFormat::Safetensors);
        assert_eq!(
            source.artifacts()[0].nested_state_disposition(),
            ModelStreamNestedStateDisposition::Flat
        );
        assert_eq!(source.artifacts()[0].byte_size(), 130);
        assert_eq!(source.artifacts()[0].sha256().len(), 64);
        assert_eq!(source.tensors().len(), 2);
        assert_eq!(source.aggregate_tensor_bytes(), 8);
        assert_eq!(source.tensors()[0].name(), "first");
        assert_eq!(source.tensors()[0].data_type(), "U8");
        assert_eq!(source.tensors()[0].shape(), &[4]);
        assert_eq!(source.tensors()[0].artifact_ordinal(), 0);
        assert_eq!(source.tensors()[0].byte_length(), 4);
        assert_eq!(source.maximum_read_bytes(), MODEL_STREAM_MAXIMUM_READ_BYTES);
        assert_eq!(
            store.read_verified_tensor_range(
                &index,
                &source,
                "first",
                1,
                2,
                &CancellationToken::default(),
            )?,
            vec![2, 3]
        );
        assert_eq!(store.cache.models.len(), 1);

        assert!(matches!(
            store.read_verified_tensor_range(
                &index,
                &source,
                "unknown",
                0,
                1,
                &CancellationToken::default(),
            ),
            Err(ModelStoreError::UnknownTensor(name)) if name == "unknown"
        ));
        assert!(matches!(
            store.read_verified_tensor_range(
                &index,
                &source,
                "first",
                4,
                1,
                &CancellationToken::default(),
            ),
            Err(ModelStoreError::ModelStreamTensorRangeOutOfBounds {
                byte_offset: 4,
                byte_length: 1,
                tensor_byte_length: 4,
                ..
            })
        ));
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert!(matches!(
            store.read_verified_tensor_range(&index, &source, "first", 0, 1, &cancelled),
            Err(ModelStoreError::Cancelled)
        ));
        let other_store = ModelStore::new(ParserLimits::default())?;
        assert!(matches!(
            other_store.read_verified_tensor_range(
                &index,
                &source,
                "first",
                0,
                1,
                &CancellationToken::default(),
            ),
            Err(ModelStoreError::ForeignModelStreamSource)
        ));
        assert!(matches!(
            store.read_verified_tensor_range(
                &index,
                &source,
                "first",
                0,
                store.limits.maximum_tensor_bytes + 1,
                &CancellationToken::default(),
            ),
            Err(ModelStoreError::Limit(_))
        ));
        assert_eq!(store.cache.models.len(), 1);
        Ok(())
    }

    #[test]
    fn model_store_stream_source_rejects_changed_and_stale_sources()
    -> Result<(), Box<dyn std::error::Error>> {
        let (directory, mut index, key, store, model, source) = stream_source_fixture()?;
        write_stream_safetensors(&directory.path().join("replacement.safetensors"))?;
        fs::write(directory.path().join("stream.safetensors"), vec![0_u8; 130])?;
        assert!(matches!(
            store.read_verified_tensor_range(
                &index,
                &source,
                "first",
                0,
                1,
                &CancellationToken::default(),
            ),
            Err(ModelStoreError::ArtifactChanged { key: changed }) if changed == key
        ));
        index.refresh(&CancellationToken::default())?;
        assert!(matches!(
            store.read_verified_tensor_range(
                &index,
                &source,
                "first",
                0,
                1,
                &CancellationToken::default(),
            ),
            Err(ModelStoreError::ArtifactChanged { key: changed }) if changed == key
        ));
        drop(model);
        drop(store);
        let other_store = ModelStore::new(ParserLimits::default())?;
        assert!(matches!(
            other_store.read_verified_tensor_range(
                &index,
                &source,
                "first",
                0,
                1,
                &CancellationToken::default(),
            ),
            Err(ModelStoreError::StaleModelStreamSource)
        ));
        Ok(())
    }

    #[test]
    fn model_store_stream_source_rejects_malformed_overlap_and_aggregate_limits()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, index, _key, store, _model, mut source) = stream_source_fixture()?;
        source.tensors[1].byte_offset = source.tensors[0].byte_offset + 1;
        assert!(matches!(
            store.read_verified_tensor_range(
                &index,
                &source,
                "first",
                0,
                1,
                &CancellationToken::default(),
            ),
            Err(ModelStoreError::OverlappingModelStreamTensorRanges { .. })
        ));
        source.tensors[1].byte_offset = source.tensors[0].byte_offset + 4;
        source.tensors[0].data_type.clear();
        assert!(matches!(
            store.read_verified_tensor_range(
                &index,
                &source,
                "first",
                0,
                1,
                &CancellationToken::default(),
            ),
            Err(ModelStoreError::InvalidModelStreamSourceMetadata(_))
        ));

        let limits = ParserLimits {
            maximum_tensor_bytes: 4,
            maximum_aggregate_tensor_bytes: 7,
            ..ParserLimits::default()
        };
        let limited_store = ModelStore::new(limits.clone())?;
        let key = ArtifactKey::new("models", "aggregate.safetensors")?;
        let synthetic = VerifiedModelStreamSource {
            model_identity: "synthetic".to_owned(),
            artifacts: vec![VerifiedModelStreamArtifact {
                key: key.clone(),
                sha256: "0".repeat(64),
                byte_size: 8,
                format: ModelFormat::Safetensors,
                nested_state_disposition: ModelStreamNestedStateDisposition::Flat,
                semantic: ModelStreamSemanticArtifact::None,
                backing: ModelStreamArtifactBacking::Indexed,
            }],
            tensors: vec![
                VerifiedModelStreamTensor {
                    name: "first".to_owned(),
                    data_type: "U8".to_owned(),
                    shape: vec![4],
                    artifact_ordinal: 0,
                    artifact_key: key.clone(),
                    byte_offset: 0,
                    byte_length: 4,
                },
                VerifiedModelStreamTensor {
                    name: "second".to_owned(),
                    data_type: "U8".to_owned(),
                    shape: vec![4],
                    artifact_ordinal: 0,
                    artifact_key: key,
                    byte_offset: 4,
                    byte_length: 4,
                },
            ],
            aggregate_tensor_bytes: 8,
            maximum_read_bytes: MODEL_STREAM_MAXIMUM_READ_BYTES,
            limits,
            store_identity: Arc::downgrade(&limited_store.identity),
        };
        assert!(matches!(
            limited_store.validate_stream_source(&synthetic),
            Err(ModelStoreError::Limit(_))
        ));
        Ok(())
    }

    #[test]
    fn model_store_stream_source_preserves_ordered_shard_identity()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        write_safetensors(&directory.path().join("one.safetensors"), "first")?;
        write_safetensors(&directory.path().join("two.safetensors"), "second")?;
        fs::write(
            directory.path().join("model.safetensors.index.json"),
            br#"{"metadata":{},"weight_map":{"first":"one.safetensors","second":"two.safetensors"}}"#,
        )?;
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "models",
            "checkpoints",
            directory.path(),
            ["safetensors", "json"],
        )?)?;
        let cancellation = CancellationToken::default();
        index.refresh(&cancellation)?;
        let mut store = ModelStore::new(ParserLimits::default())?;
        let model = store.load(
            &index,
            &ArtifactKey::new("models", "model.safetensors.index.json")?,
            &cancellation,
        )?;
        let source = store.verified_stream_source(&index, &model, &cancellation)?;
        assert_eq!(
            source
                .artifacts()
                .iter()
                .map(|artifact| artifact.key.relative_path.as_path())
                .collect::<Vec<_>>(),
            vec![
                std::path::Path::new("model.safetensors.index.json"),
                std::path::Path::new("one.safetensors"),
                std::path::Path::new("two.safetensors"),
            ]
        );
        assert_eq!(source.artifacts()[0].format(), &ModelFormat::JsonConfig);
        assert_eq!(source.tensors()[0].name(), "first");
        assert_eq!(
            source.tensors()[0].artifact_key.relative_path,
            PathBuf::from("one.safetensors")
        );
        assert_eq!(
            store.read_verified_tensor_range(&index, &source, "second", 0, 1, &cancellation)?,
            vec![7]
        );
        Ok(())
    }

    #[test]
    fn model_store_recipe_semantic_companions_and_metadata_are_verified()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let model_path = directory.path().join("recipe.safetensors");
        let metadata = BTreeMap::from([(
            "config".to_owned(),
            r#"{"_class_name":"LatentUpsampler","in_channels":4,"mid_channels":8,"num_blocks_per_stage":1,"dims":2,"spatial_upsample":true,"temporal_upsample":false}"#.to_owned(),
        )]);
        write_stream_safetensors_with_metadata(&model_path, &metadata)?;
        let configuration = br#"{"hidden_size":4,"num_hidden_layers":1}"#.to_vec();
        let tokenizer = br#"{"model":{"vocab":{"a":0,"b":1},"merges":[]}}"#.to_vec();
        fs::write(
            directory
                .path()
                .join("recipe.safetensors.clip-config-0.json"),
            &configuration,
        )?;
        let tokenizer_path = directory
            .path()
            .join("recipe.safetensors.clip-tokenizer-0.json");
        fs::write(&tokenizer_path, &tokenizer)?;

        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "models",
            "text_encoders",
            directory.path(),
            ["safetensors", "json"],
        )?)?;
        let cancellation = CancellationToken::default();
        index.refresh(&cancellation)?;
        let mut store = ModelStore::new(ParserLimits::default())?;
        let model = store.load(
            &index,
            &ArtifactKey::new("models", "recipe.safetensors")?,
            &cancellation,
        )?;
        let source = store.verified_stream_source(&index, &model, &cancellation)?;
        assert_eq!(source.artifacts().len(), 4);
        assert_eq!(
            source
                .artifacts()
                .iter()
                .map(VerifiedModelStreamArtifact::semantic)
                .collect::<Vec<_>>(),
            vec![
                ModelStreamSemanticArtifact::None,
                ModelStreamSemanticArtifact::SafetensorsMetadata {
                    source_artifact_ordinal: 0,
                },
                ModelStreamSemanticArtifact::ClipConfiguration { target_ordinal: 0 },
                ModelStreamSemanticArtifact::ClipTokenizerJson { target_ordinal: 0 },
            ]
        );
        let metadata_artifact = source
            .artifacts()
            .get(1)
            .ok_or("verified source omitted safetensors metadata")?;
        let metadata_bytes = store.read_verified_artifact_range(
            &index,
            &source,
            1,
            metadata_artifact.sha256(),
            0,
            metadata_artifact.byte_size(),
            &cancellation,
        )?;
        assert_eq!(
            crate::formats::checked_safetensors_metadata_json(
                &metadata_bytes,
                metadata_artifact.sha256(),
                &ParserLimits::default(),
                &cancellation,
            )?,
            metadata
        );
        let tokenizer_artifact = source
            .artifacts()
            .get(3)
            .ok_or("verified source omitted tokenizer companion")?;
        assert_eq!(
            store.read_verified_artifact_range(
                &index,
                &source,
                3,
                tokenizer_artifact.sha256(),
                0,
                tokenizer_artifact.byte_size(),
                &cancellation,
            )?,
            tokenizer
        );
        fs::write(&tokenizer_path, b"{}")?;
        assert!(matches!(
            store.read_verified_artifact_range(
                &index,
                &source,
                1,
                metadata_artifact.sha256(),
                0,
                metadata_artifact.byte_size(),
                &cancellation,
            ),
            Err(ModelStoreError::ArtifactChanged { .. })
        ));
        Ok(())
    }

    #[test]
    fn model_source_auxiliary_artifact_stream_is_verified_and_bounded()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        write_safetensors(&directory.path().join("one.safetensors"), "first")?;
        let config = br#"{"metadata":{},"weight_map":{"first":"one.safetensors"}}"#.to_vec();
        fs::write(
            directory.path().join("model.safetensors.index.json"),
            &config,
        )?;
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "models",
            "checkpoints",
            directory.path(),
            ["safetensors", "json"],
        )?)?;
        let cancellation = CancellationToken::default();
        index.refresh(&cancellation)?;
        let mut store = ModelStore::new(ParserLimits::default())?;
        let model = store.load(
            &index,
            &ArtifactKey::new("models", "model.safetensors.index.json")?,
            &cancellation,
        )?;
        let source = store.verified_stream_source(&index, &model, &cancellation)?;
        assert!(model.identity().contains(':'));
        assert_eq!(source.model_identity().len(), 64);
        assert!(!source.model_identity().contains(':'));
        assert_ne!(source.model_identity(), model.identity());
        assert_eq!(
            source
                .artifacts()
                .iter()
                .map(VerifiedModelStreamArtifact::sha256)
                .collect::<Vec<_>>(),
            model
                .documents()
                .iter()
                .map(|document| document.source_sha256.as_str())
                .collect::<Vec<_>>()
        );
        let artifact_sha256 = source.artifacts()[0].sha256().to_owned();
        let split = 17_u64;
        let first = store.read_verified_artifact_range(
            &index,
            &source,
            0,
            &artifact_sha256,
            0,
            split,
            &cancellation,
        )?;
        let second = store.read_verified_artifact_range(
            &index,
            &source,
            0,
            &artifact_sha256,
            split,
            u64::try_from(config.len())? - split,
            &cancellation,
        )?;
        assert_eq!([first, second].concat(), config);
        assert!(matches!(
            store.read_verified_artifact_range(
                &index,
                &source,
                2,
                &artifact_sha256,
                0,
                1,
                &cancellation,
            ),
            Err(ModelStoreError::UnknownModelStreamArtifactOrdinal(2))
        ));
        assert!(matches!(
            store.read_verified_artifact_range(
                &index,
                &source,
                0,
                &"f".repeat(64),
                0,
                1,
                &cancellation,
            ),
            Err(ModelStoreError::ModelStreamArtifactIdentityMismatch {
                artifact_ordinal: 0
            })
        ));
        assert!(matches!(
            store.read_verified_artifact_range(
                &index,
                &source,
                1,
                source.artifacts()[1].sha256(),
                0,
                1,
                &cancellation,
            ),
            Err(ModelStoreError::UnsupportedModelStreamArtifactFormat {
                artifact_ordinal: 1,
                format: ModelFormat::Safetensors,
            })
        ));
        assert!(matches!(
            store.read_verified_artifact_range(
                &index,
                &source,
                0,
                &artifact_sha256,
                u64::try_from(config.len())?,
                1,
                &cancellation,
            ),
            Err(ModelStoreError::ModelStreamArtifactRangeOutOfBounds { .. })
        ));
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert!(matches!(
            store.read_verified_artifact_range(
                &index,
                &source,
                0,
                &artifact_sha256,
                0,
                1,
                &cancelled,
            ),
            Err(ModelStoreError::Cancelled)
        ));
        let mut changed = config;
        let first = changed
            .first_mut()
            .ok_or("auxiliary artifact fixture is empty")?;
        *first ^= 1;
        fs::write(
            directory.path().join("model.safetensors.index.json"),
            changed,
        )?;
        assert!(matches!(
            store.read_verified_artifact_range(
                &index,
                &source,
                0,
                &artifact_sha256,
                0,
                1,
                &cancellation,
            ),
            Err(ModelStoreError::ArtifactChanged { .. })
        ));
        Ok(())
    }

    #[test]
    fn model_store_stream_source_enforces_exact_per_call_ceiling()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        write_maximum_stream_safetensors(&directory.path().join("maximum.safetensors"))?;
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "models",
            "checkpoints",
            directory.path(),
            ["safetensors"],
        )?)?;
        let cancellation = CancellationToken::default();
        index.refresh(&cancellation)?;
        let mut store = ModelStore::new(ParserLimits::default())?;
        let model = store.load(
            &index,
            &ArtifactKey::new("models", "maximum.safetensors")?,
            &cancellation,
        )?;
        let source = store.verified_stream_source(&index, &model, &cancellation)?;
        assert_eq!(source.maximum_read_bytes(), 1024 * 1024);
        let bytes = store.read_verified_tensor_range(
            &index,
            &source,
            "maximum",
            0,
            source.maximum_read_bytes(),
            &cancellation,
        )?;
        assert_eq!(bytes.len(), usize::try_from(source.maximum_read_bytes())?);
        assert!(bytes.iter().all(|byte| *byte == 0x5a));
        assert!(matches!(
            store.read_verified_tensor_range(
                &index,
                &source,
                "maximum",
                0,
                source.maximum_read_bytes() + 1,
                &cancellation,
            ),
            Err(ModelStoreError::Limit(_))
        ));
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert!(matches!(
            store.read_verified_tensor_range(
                &index,
                &source,
                "maximum",
                0,
                source.maximum_read_bytes(),
                &cancelled,
            ),
            Err(ModelStoreError::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn canonical_index_refresh_precedes_model_store_replacement()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let model_path = directory.path().join("model.safetensors");
        write_safetensors(&model_path, "weight")?;
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "models",
            "checkpoints",
            directory.path(),
            ["safetensors"],
        )?)?;
        let cancellation = CancellationToken::default();
        index.refresh(&cancellation)?;
        let key = ArtifactKey::new("models", "model.safetensors")?;
        let mut store = ModelStore::new(ParserLimits::default())?;
        let loaded = store.load(&index, &key, &cancellation)?;
        assert_eq!(loaded.accounting().resident_bytes, 0);
        assert_eq!(
            store.read_tensor(&index, &loaded, "weight", &cancellation)?,
            vec![7]
        );

        write_safetensors(&model_path, "changed")?;
        assert!(matches!(
            store.read_tensor(&index, &loaded, "weight", &cancellation),
            Err(ModelStoreError::Index(ArtifactIndexError::ChangedSinceIndex(changed)))
                if changed == key
        ));
        let changes = index.refresh(&cancellation)?;
        assert_eq!(changes.len(), 1);
        assert!(matches!(
            store.read_tensor(&index, &loaded, "weight", &cancellation),
            Err(ModelStoreError::ArtifactChanged { key: changed }) if changed == key
        ));
        let loaded = store.load(&index, &key, &cancellation)?;
        assert!(loaded.tensors().contains_key("changed"));
        Ok(())
    }

    #[test]
    fn sharded_index_requires_exact_weight_map() -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        write_safetensors(&directory.path().join("one.safetensors"), "first")?;
        write_safetensors(&directory.path().join("two.safetensors"), "second")?;
        fs::write(
            directory.path().join("model.safetensors.index.json"),
            br#"{"metadata":{},"weight_map":{"first":"one.safetensors","second":"two.safetensors"}}"#,
        )?;
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "models",
            "checkpoints",
            directory.path(),
            ["safetensors", "json"],
        )?)?;
        let cancellation = CancellationToken::default();
        index.refresh(&cancellation)?;
        let mut store = ModelStore::new(ParserLimits::default())?;
        let loaded = store.load(
            &index,
            &ArtifactKey::new("models", "model.safetensors.index.json")?,
            &cancellation,
        )?;
        assert_eq!(loaded.tensors().len(), 2);
        assert_eq!(loaded.documents().len(), 3);
        let tensors = store.read_tensors(&index, &loaded, ["first", "second"], &cancellation)?;
        assert_eq!(tensors.get("first"), Some(&vec![7]));
        assert_eq!(tensors.get("second"), Some(&vec![7]));

        fs::write(
            directory.path().join("model.safetensors.index.json"),
            br#"{"metadata":{},"weight_map":{"first":"two.safetensors","second":"one.safetensors"}}"#,
        )?;
        index.refresh(&cancellation)?;
        let tensors = store.read_tensors(&index, &loaded, ["first", "second"], &cancellation)?;
        assert_eq!(tensors.get("first"), Some(&vec![7]));
        assert_eq!(tensors.get("second"), Some(&vec![7]));
        assert!(matches!(
            store.load(
                &index,
                &ArtifactKey::new("models", "model.safetensors.index.json")?,
                &cancellation,
            ),
            Err(ModelStoreError::InvalidShardIndex(_))
        ));
        Ok(())
    }

    #[test]
    fn cancelled_lazy_read_releases_all_resident_accounting()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let model_path = directory.path().join("model.safetensors");
        write_safetensors(&model_path, "weight")?;
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "models",
            "checkpoints",
            directory.path(),
            ["safetensors"],
        )?)?;
        let cancellation = CancellationToken::default();
        index.refresh(&cancellation)?;
        let mut store = ModelStore::new(ParserLimits::default())?;
        let loaded = store.load(
            &index,
            &ArtifactKey::new("models", "model.safetensors")?,
            &cancellation,
        )?;
        cancellation.cancel();
        assert!(matches!(
            store.read_tensor(&index, &loaded, "weight", &cancellation),
            Err(ModelStoreError::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn operation_history_is_bounded_and_preserves_latest_attempt() -> Result<(), ModelStoreError> {
        let mut store = ModelStore::new(ParserLimits::default())?;
        let key = ArtifactKey::new("models", "model.safetensors")?;
        for attempt in 0..=MAX_MODEL_OPERATION_RECORDS {
            store.record(
                &key,
                u8::try_from(attempt % 256).unwrap_or(u8::MAX),
                ModelOperationStage::Resolving,
                None,
            );
        }
        assert_eq!(store.operations().len(), MAX_MODEL_OPERATION_RECORDS);
        assert_eq!(
            store.operations().last().map(|record| record.attempt),
            Some(0)
        );
        Ok(())
    }

    #[test]
    fn model_handles_are_store_scoped_and_cache_entries_are_bounded()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        for index in 0..=MAX_MODEL_CACHE_ENTRIES {
            write_safetensors(
                &directory.path().join(format!("model-{index}.safetensors")),
                &format!("weight-{index}"),
            )?;
        }
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "models",
            "checkpoints",
            directory.path(),
            ["safetensors"],
        )?)?;
        let cancellation = CancellationToken::default();
        index.refresh(&cancellation)?;
        let mut store = ModelStore::new(ParserLimits::default())?;
        let first = store.load(
            &index,
            &ArtifactKey::new("models", "model-0.safetensors")?,
            &cancellation,
        )?;
        for model_index in 1..=MAX_MODEL_CACHE_ENTRIES {
            store.load(
                &index,
                &ArtifactKey::new("models", format!("model-{model_index}.safetensors"))?,
                &cancellation,
            )?;
        }
        assert_eq!(store.cache.models.len(), MAX_MODEL_CACHE_ENTRIES);

        let other_store = ModelStore::new(ParserLimits::default())?;
        assert!(matches!(
            other_store.read_tensor(&index, &first, "weight-0", &cancellation),
            Err(ModelStoreError::ForeignModelHandle)
        ));
        Ok(())
    }
}
