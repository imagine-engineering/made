use crate::{AssetOperation, AuthorizedCapabilities, Capability};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sim_media::{ComfyMetadata, MetadataCarrier, MetadataDocument, MetadataLimits};
pub use sim_model::ArtifactWritePolicy as AssetCollisionPolicy;
use sim_model::model_store::ModelStreamSemanticArtifact;
use sim_model::vae_audio::load_audio_vae_from_model_store_with_context;
use sim_model::vae_image::load_image_vae_from_model_store_with_context;
use sim_model::vae_structured::load_structured_vae_from_model_store_with_context;
use sim_model::vae_video::load_video_vae_from_model_store_with_context;
use sim_model::{
    ArtifactAvailability as CanonicalArtifactAvailability,
    ArtifactChangeKind as CanonicalArtifactChangeKind, ArtifactIndex, ArtifactIndexError,
    ArtifactKey, ArtifactRecord as CanonicalArtifactRecord, ArtifactRoot, AudioVaeError,
    ImageVaeError, LoadedModel, ModelFormat, ModelStore, ModelStoreError,
    ModelStreamNestedStateDisposition, NativeStructuredVae, NativeVae, PatchGraphIdentity,
    StructuredVaeError, VaeArchitectureError, VaeArchitectureRegistry, VaeBoundary, VaeDescriptor,
    VaeError, VaeExecutionTarget, VaeOperation, VaeStructuredDecodeRequest, VaeStructuredResult,
    VerifiedModelStreamSource, VideoVaeError, validate_native_vae_backend_target,
};
use sim_nodes::{
    NativeAssetNameList, NativeAssetNameListRequest, NativeAssetNameResolution,
    NativeAssetNameResolutionRequest, NativeAssetReadRequest, NativeAssetReference,
    NativeAssetResolver, NativeAssetServiceError, NativeNamedAssetReference,
    NativeNodeServiceIdentity, NativeResolvedAsset,
};
use sim_tensor::{CancellationToken, CpuBackend, ExecutionContext, Tensor};
use sim_types::{
    WorkerModelSourceArtifact, WorkerModelSourceArtifactChunk, WorkerModelSourceChunk,
    WorkerModelSourceClosed, WorkerModelSourceContext, WorkerModelSourceError,
    WorkerModelSourceFormat, WorkerModelSourceManifest, WorkerModelSourceNestedStateDisposition,
    WorkerModelSourceOpened, WorkerModelSourceOperation, WorkerModelSourceParserLimits,
    WorkerModelSourceRequest, WorkerModelSourceResponse, WorkerModelSourceTensor,
    WorkerSha256Digest, worker_model_source_selection_sha256,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use thiserror::Error;
use uuid::Uuid;

pub const DEFAULT_MAX_ASSET_BYTES: u64 = 2 * 1024 * 1024 * 1024;
pub const DEFAULT_MAX_RANGE_BYTES: u64 = 64 * 1024 * 1024;
pub const DEFAULT_PAGE_SIZE: usize = 100;
pub const MAX_PAGE_SIZE: usize = 1_000;
pub const ASSET_SERVICE_SCHEMA_VERSION: u32 = 2;
pub const DEFAULT_MAX_ASSET_INDEX_BYTES: usize = 64 * 1024 * 1024;
pub const DEFAULT_MAX_ASSET_RECORDS: usize = 100_000;
const ASSET_INDEX_FILENAME: &str = ".zed-asset-index.json";
const MAX_RETAINED_MODEL_SOURCE_SESSION_IDS: usize = 1_024;

const SOURCE_MODEL_EXTENSIONS: &[&str] = &[
    ".bin",
    ".ckpt",
    ".pkl",
    ".pt",
    ".pt2",
    ".pth",
    ".safetensors",
    ".sft",
    ".json",
    ".yaml",
    ".yml",
    ".model",
    ".tiktoken",
];

#[derive(Clone, Copy)]
struct SourceAssetFolderSpec {
    namespace: AssetNamespace,
    prefixes: &'static [&'static str],
    extensions: &'static [&'static str],
}

fn source_asset_folder_spec(category: &str) -> Option<SourceAssetFolderSpec> {
    let model = |prefixes| SourceAssetFolderSpec {
        namespace: AssetNamespace::Model,
        prefixes,
        extensions: SOURCE_MODEL_EXTENSIONS,
    };
    match category {
        "input" => Some(SourceAssetFolderSpec {
            namespace: AssetNamespace::Input,
            prefixes: &[""],
            extensions: &[],
        }),
        "output" => Some(SourceAssetFolderSpec {
            namespace: AssetNamespace::Output,
            prefixes: &[""],
            extensions: &[],
        }),
        "temp" | "temporary" => Some(SourceAssetFolderSpec {
            namespace: AssetNamespace::Temporary,
            prefixes: &[""],
            extensions: &[],
        }),
        "checkpoints" => Some(model(&["checkpoints"])),
        "configs" => Some(SourceAssetFolderSpec {
            namespace: AssetNamespace::Model,
            prefixes: &["configs"],
            extensions: &[".json", ".yaml", ".yml"],
        }),
        "loras" => Some(model(&["loras"])),
        "vae" => Some(model(&["vae"])),
        "text_encoders" | "clip" => Some(model(&["text_encoders", "clip"])),
        "diffusion_models" | "unet" => Some(model(&["unet", "diffusion_models"])),
        "clip_vision" => Some(model(&["clip_vision"])),
        "style_models" => Some(model(&["style_models"])),
        "embeddings" => Some(model(&["embeddings"])),
        "vae_approx" => Some(model(&["vae_approx"])),
        "controlnet" => Some(model(&["controlnet", "t2i_adapter"])),
        "gligen" => Some(model(&["gligen"])),
        "upscale_models" => Some(model(&["upscale_models"])),
        "latent_upscale_models" => Some(model(&["latent_upscale_models"])),
        "hypernetworks" => Some(model(&["hypernetworks"])),
        "photomaker" => Some(model(&["photomaker"])),
        "classifiers" => Some(SourceAssetFolderSpec {
            namespace: AssetNamespace::Model,
            prefixes: &["classifiers"],
            extensions: &[""],
        }),
        "model_patches" => Some(model(&["model_patches"])),
        "audio_encoders" => Some(model(&["audio_encoders"])),
        "background_removal" => Some(model(&["background_removal"])),
        "frame_interpolation" => Some(model(&["frame_interpolation"])),
        "geometry_estimation" => Some(model(&["geometry_estimation"])),
        "optical_flow" => Some(model(&["optical_flow"])),
        "detection" => Some(model(&["detection"])),
        _ => None,
    }
}

fn source_asset_name_has_supported_extension(name: &str, extensions: &[&str]) -> bool {
    if extensions.is_empty() {
        return true;
    }
    let extension = Path::new(name)
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| format!(".{}", extension.to_ascii_lowercase()))
        .unwrap_or_default();
    extensions.contains(&extension.as_str())
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetNamespace {
    Input,
    Output,
    Temporary,
    Model,
    Plugin,
}

impl AssetNamespace {
    pub const fn locator_type(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Output => "output",
            Self::Temporary => "temp",
            Self::Model => "model",
            Self::Plugin => "plugin",
        }
    }

    pub fn from_locator_type(value: &str) -> Result<Self, AssetError> {
        match value {
            "input" => Ok(Self::Input),
            "output" => Ok(Self::Output),
            "temp" => Ok(Self::Temporary),
            "model" => Ok(Self::Model),
            "plugin" => Ok(Self::Plugin),
            _ => Err(AssetError::InvalidNamespace(value.to_owned())),
        }
    }

    pub fn from_plugin_root(value: &str) -> Result<Self, AssetError> {
        match value {
            "input-root" | "input" => Ok(Self::Input),
            "output-root" | "output" => Ok(Self::Output),
            "temporary-root" | "temporary" | "temp" => Ok(Self::Temporary),
            "model-root" | "model" => Ok(Self::Model),
            "plugin-root" | "plugin" => Ok(Self::Plugin),
            _ => Err(AssetError::InvalidNamespace(value.to_owned())),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetAction {
    Read,
    Write,
    Rename,
    Tag,
    Delete,
}

impl From<AssetAction> for AssetOperation {
    fn from(value: AssetAction) -> Self {
        match value {
            AssetAction::Read => Self::Read,
            AssetAction::Write => Self::Write,
            AssetAction::Rename => Self::Rename,
            AssetAction::Tag => Self::Tag,
            AssetAction::Delete => Self::Delete,
        }
    }
}

pub(crate) fn require_asset_authorization(
    authorization: &AuthorizedCapabilities,
    profile_id: &str,
    namespace: AssetNamespace,
    action: AssetAction,
) -> Result<(), AssetError> {
    if authorization.profile_id() != profile_id {
        return Err(AssetError::ProfileMismatch {
            expected: profile_id.to_owned(),
            actual: authorization.profile_id().to_owned(),
        });
    }
    authorization
        .require(&Capability::Asset {
            namespace: namespace.locator_type().to_owned(),
            action: action.into(),
        })
        .map_err(|_| AssetError::PermissionDenied { namespace, action })
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct AssetIdentity {
    pub profile_id: String,
    pub namespace: AssetNamespace,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_id: Option<String>,
    pub relative_path: PathBuf,
}

impl AssetIdentity {
    pub fn new(
        profile_id: impl Into<String>,
        namespace: AssetNamespace,
        relative_path: impl Into<PathBuf>,
    ) -> Result<Self, AssetError> {
        let profile_id = profile_id.into();
        if profile_id.trim().is_empty() {
            return Err(AssetError::InvalidProfile(profile_id));
        }
        let canonical = ArtifactKey::new(namespace.locator_type(), relative_path.into())
            .map_err(map_artifact_error)?;
        Ok(Self {
            profile_id,
            namespace,
            root_id: None,
            relative_path: canonical.relative_path,
        })
    }

    pub fn filename(&self) -> Option<&str> {
        self.relative_path
            .file_name()
            .and_then(|name| name.to_str())
    }

    pub fn subfolder(&self) -> PathBuf {
        self.relative_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_default()
    }

    pub fn to_reference(&self) -> Result<String, AssetError> {
        if self.root_id.is_some() {
            return Err(AssetError::InvalidReference(
                "configured asset roots do not have a stable external reference".to_owned(),
            ));
        }
        let canonical = Self::new(
            self.profile_id.clone(),
            self.namespace,
            self.relative_path.clone(),
        )?;
        if canonical != *self {
            return Err(AssetError::InvalidReference(
                "asset identity is not canonical".to_owned(),
            ));
        }
        let relative_path = self.relative_path.to_str().ok_or_else(|| {
            AssetError::InvalidReference("asset reference path is not UTF-8".to_owned())
        })?;
        if relative_path.is_empty()
            || relative_path
                .chars()
                .any(|character| matches!(character, '\\' | '?' | '#') || character.is_control())
        {
            return Err(AssetError::InvalidReference(
                "asset reference path contains an unsupported character".to_owned(),
            ));
        }
        Ok(format!(
            "zed-asset://{}/{relative_path}",
            self.namespace.locator_type()
        ))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssetRoots {
    pub profile_id: String,
    roots: BTreeMap<(AssetNamespace, String), ArtifactRoot>,
    root_order: Vec<(AssetNamespace, String)>,
}

impl AssetRoots {
    pub fn new(
        profile_id: impl Into<String>,
        roots: impl IntoIterator<Item = (AssetNamespace, PathBuf)>,
    ) -> Result<Self, AssetError> {
        let profile_id = profile_id.into();
        if profile_id.trim().is_empty() {
            return Err(AssetError::InvalidProfile(profile_id));
        }
        let mut typed_roots = BTreeMap::new();
        let mut root_order = Vec::new();
        let mut canonical_index = ArtifactIndex::default();
        for (namespace, path) in roots {
            let root = ArtifactRoot::canonical(
                namespace.locator_type(),
                namespace.locator_type(),
                &path,
                std::iter::empty::<String>(),
            )
            .map_err(map_artifact_error)?;
            canonical_index
                .add_root(root.clone())
                .map_err(map_artifact_error)?;
            if typed_roots
                .insert((namespace, namespace.locator_type().to_owned()), root)
                .is_some()
            {
                return Err(AssetError::DuplicateNamespace(namespace));
            }
            root_order.push((namespace, namespace.locator_type().to_owned()));
        }
        Ok(Self {
            profile_id,
            roots: typed_roots,
            root_order,
        })
    }

    fn root_path(&self, namespace: AssetNamespace) -> Result<&Path, AssetError> {
        self.roots
            .get(&(namespace, namespace.locator_type().to_owned()))
            .map(ArtifactRoot::canonical_path)
            .ok_or(AssetError::UnknownNamespace(namespace))
    }

    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn test_root_path(&self, namespace: AssetNamespace) -> Result<&Path, AssetError> {
        self.root_path(namespace)
    }

    pub fn namespaces(&self) -> impl Iterator<Item = AssetNamespace> {
        self.roots
            .keys()
            .map(|(namespace, _)| *namespace)
            .collect::<BTreeSet<_>>()
            .into_iter()
    }

    pub fn add_configured_root(
        &mut self,
        namespace: AssetNamespace,
        path: PathBuf,
    ) -> Result<String, AssetError> {
        let root = ArtifactRoot::canonical_with_path_identity(
            &format!("{}-configured", namespace.locator_type()),
            namespace.locator_type(),
            &path,
            std::iter::empty::<String>(),
        )
        .map_err(map_artifact_error)?;
        let root_id = root.id().to_owned();
        self.insert_additional_root(namespace, root)?;
        Ok(root_id)
    }

    fn insert_additional_root(
        &mut self,
        namespace: AssetNamespace,
        root: ArtifactRoot,
    ) -> Result<(), AssetError> {
        let root_id = root.id().to_owned();
        if root_id == namespace.locator_type() {
            return Err(AssetError::DuplicateNamespace(namespace));
        }
        if self.roots.contains_key(&(namespace, root_id.clone())) {
            return Err(AssetError::InvalidIndex(
                "asset logical root identity is duplicated".to_owned(),
            ));
        }
        let mut canonical_index = ArtifactIndex::default();
        for existing in self.artifact_roots() {
            canonical_index
                .add_root(existing)
                .map_err(map_artifact_error)?;
        }
        canonical_index
            .add_root(root.clone())
            .map_err(map_artifact_error)?;
        self.roots.insert((namespace, root_id.clone()), root);
        self.root_order.push((namespace, root_id));
        Ok(())
    }

    fn root_ids_in_resolution_order(
        &self,
        namespace: AssetNamespace,
    ) -> impl Iterator<Item = &str> {
        self.root_order
            .iter()
            .filter(move |(candidate, _)| *candidate == namespace)
            .map(|(_, root_id)| root_id.as_str())
    }

    pub fn identity(
        &self,
        namespace: AssetNamespace,
        relative_path: impl Into<PathBuf>,
    ) -> Result<AssetIdentity, AssetError> {
        self.identity_in_root(namespace, namespace.locator_type(), relative_path)
    }

    pub fn identity_from_reference(&self, reference: &str) -> Result<AssetIdentity, AssetError> {
        if reference.len() > 16 * 1024
            || reference
                .chars()
                .any(|character| matches!(character, '\\' | '?' | '#') || character.is_control())
        {
            return Err(AssetError::InvalidReference(
                "asset reference is malformed".to_owned(),
            ));
        }
        let value = reference.strip_prefix("zed-asset://").ok_or_else(|| {
            AssetError::InvalidReference(
                "asset references must use the zed-asset scheme".to_owned(),
            )
        })?;
        let (namespace, relative_path) = value.split_once('/').ok_or_else(|| {
            AssetError::InvalidReference(
                "asset references require a namespace and relative path".to_owned(),
            )
        })?;
        if relative_path.is_empty() {
            return Err(AssetError::InvalidReference(
                "asset reference path is empty".to_owned(),
            ));
        }
        let namespace = AssetNamespace::from_locator_type(namespace)?;
        self.identity(namespace, relative_path)
    }

    pub fn identity_in_root(
        &self,
        namespace: AssetNamespace,
        root_id: &str,
        relative_path: impl Into<PathBuf>,
    ) -> Result<AssetIdentity, AssetError> {
        let root = self.artifact_root(namespace, root_id)?;
        let key = root.key(relative_path).map_err(map_artifact_error)?;
        Ok(AssetIdentity {
            profile_id: self.profile_id.clone(),
            namespace,
            root_id: (root_id != namespace.locator_type()).then(|| root_id.to_owned()),
            relative_path: key.relative_path,
        })
    }

    pub(crate) fn resolve_existing(&self, identity: &AssetIdentity) -> Result<PathBuf, AssetError> {
        let key = self.artifact_key(identity)?;
        self.artifact_root_for_identity(identity)?
            .resolve_existing(&key.relative_path)
            .map_err(|error| map_artifact_error_for_identity(error, identity))
    }

    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn test_resolve_existing(&self, identity: &AssetIdentity) -> Result<PathBuf, AssetError> {
        self.resolve_existing(identity)
    }

    pub(crate) fn resolve_for_create(
        &self,
        identity: &AssetIdentity,
    ) -> Result<PathBuf, AssetError> {
        let key = self.artifact_key(identity)?;
        self.artifact_root_for_identity(identity)?
            .resolve_for_create_with_parents(&key.relative_path)
            .map_err(map_artifact_error)
    }

    pub(crate) fn read_private(
        &self,
        identity: &AssetIdentity,
        maximum_bytes: usize,
    ) -> Result<Option<Vec<u8>>, AssetError> {
        self.artifact_key(identity)?;
        self.artifact_root_for_identity(identity)?
            .read_private_file(&identity.relative_path, maximum_bytes)
            .map_err(map_artifact_error)
    }

    pub(crate) fn write_private(
        &self,
        identity: &AssetIdentity,
        bytes: &[u8],
    ) -> Result<(), AssetError> {
        self.artifact_key(identity)?;
        self.artifact_root_for_identity(identity)?
            .write_private_file(&identity.relative_path, bytes)
            .map_err(map_artifact_error)
    }

    fn read_contained(
        &self,
        identity: &AssetIdentity,
        maximum_bytes: usize,
    ) -> Result<Option<Vec<u8>>, AssetError> {
        self.artifact_key(identity)?;
        self.artifact_root_for_identity(identity)?
            .read_private_file(&identity.relative_path, maximum_bytes)
            .map_err(map_artifact_error)
    }

    pub(crate) fn write_contained(
        &self,
        identity: &AssetIdentity,
        bytes: &[u8],
        policy: AssetCollisionPolicy,
    ) -> Result<(), AssetError> {
        self.artifact_key(identity)?;
        self.artifact_root_for_identity(identity)?
            .write_contained_file(&identity.relative_path, bytes, policy)
            .map_err(map_artifact_error)
    }

    pub(crate) fn remove_contained(&self, identity: &AssetIdentity) -> Result<(), AssetError> {
        self.artifact_key(identity)?;
        self.artifact_root_for_identity(identity)?
            .remove_contained_file(&identity.relative_path)
            .map(|_| ())
            .map_err(map_artifact_error)
    }

    pub(crate) fn contained_exists(&self, identity: &AssetIdentity) -> Result<bool, AssetError> {
        self.artifact_key(identity)?;
        self.artifact_root_for_identity(identity)?
            .contained_file_exists(&identity.relative_path)
            .map_err(map_artifact_error)
    }

    pub(crate) fn contained_digest(
        &self,
        identity: &AssetIdentity,
        cancellation: &CancellationToken,
    ) -> Result<Option<(String, u64)>, AssetError> {
        self.artifact_key(identity)?;
        self.artifact_root_for_identity(identity)?
            .contained_file_digest(&identity.relative_path, cancellation)
            .map_err(map_artifact_error)
    }

    pub(crate) fn move_contained(
        &self,
        source: &AssetIdentity,
        destination: &AssetIdentity,
        policy: AssetCollisionPolicy,
        expected_sha256: &str,
        expected_size: u64,
        cancellation: &CancellationToken,
    ) -> Result<(), AssetError> {
        self.artifact_key(source)?;
        self.artifact_key(destination)?;
        self.artifact_root_for_identity(source)?
            .move_verified_contained_file_to(
                &source.relative_path,
                self.artifact_root_for_identity(destination)?,
                &destination.relative_path,
                policy,
                expected_sha256,
                expected_size,
                cancellation,
            )
            .map_err(map_artifact_error)
    }

    pub(crate) fn list_direct_contained_files(
        &self,
        namespace: AssetNamespace,
        relative_directory: &Path,
    ) -> Result<Vec<AssetIdentity>, AssetError> {
        let root = self.artifact_root(namespace, namespace.locator_type())?;
        root.list_direct_contained_files(relative_directory)
            .map_err(map_artifact_error)?
            .into_iter()
            .map(|relative_path| self.identity(namespace, relative_path))
            .collect()
    }

    pub(crate) fn list_direct_contained_regular_files(
        &self,
        namespace: AssetNamespace,
        relative_directory: &Path,
    ) -> Result<Vec<AssetIdentity>, AssetError> {
        let root = self.artifact_root(namespace, namespace.locator_type())?;
        root.list_direct_contained_regular_files(relative_directory)
            .map_err(map_artifact_error)?
            .into_iter()
            .map(|relative_path| self.identity(namespace, relative_path))
            .collect()
    }

    pub(crate) fn artifact_roots(&self) -> impl Iterator<Item = ArtifactRoot> + '_ {
        self.roots.values().cloned()
    }

    pub(crate) fn artifact_key(&self, identity: &AssetIdentity) -> Result<ArtifactKey, AssetError> {
        if identity.profile_id != self.profile_id {
            return Err(AssetError::ProfileMismatch {
                expected: self.profile_id.clone(),
                actual: identity.profile_id.clone(),
            });
        }
        self.artifact_root_for_identity(identity)?
            .key(identity.relative_path.clone())
            .map_err(map_artifact_error)
    }

    fn artifact_root(
        &self,
        namespace: AssetNamespace,
        root_id: &str,
    ) -> Result<&ArtifactRoot, AssetError> {
        self.roots
            .get(&(namespace, root_id.to_owned()))
            .ok_or(AssetError::UnknownNamespace(namespace))
    }

    fn artifact_root_for_identity(
        &self,
        identity: &AssetIdentity,
    ) -> Result<&ArtifactRoot, AssetError> {
        self.artifact_root(
            identity.namespace,
            identity
                .root_id
                .as_deref()
                .unwrap_or_else(|| identity.namespace.locator_type()),
        )
    }

    fn identity_for_key(&self, key: &ArtifactKey) -> Result<AssetIdentity, AssetError> {
        let (namespace, root_id) = self
            .roots
            .iter()
            .find_map(|((namespace, root_id), root)| {
                (root.id() == key.root_id).then_some((*namespace, root_id.as_str()))
            })
            .ok_or_else(|| {
                AssetError::InvalidIndex("artifact key has an unknown root".to_owned())
            })?;
        self.identity_in_root(namespace, root_id, key.relative_path.clone())
    }
}

pub type AssetAvailability = CanonicalArtifactAvailability;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AssetRecord {
    pub identity: AssetIdentity,
    pub sha256: String,
    pub byte_size: u64,
    pub modified_nanoseconds: u128,
    pub content_type: String,
    pub metadata_carrier: MetadataCarrier,
    pub metadata: ComfyMetadata,
    pub metadata_diagnostics: Vec<String>,
    pub tags: BTreeSet<String>,
    pub availability: AssetAvailability,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetChangeKind {
    Added,
    Modified,
    Missing,
    Restored,
    Renamed,
    Deleted,
    TagsChanged,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AssetChange {
    pub identity: AssetIdentity,
    pub kind: AssetChangeKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UploadRequest {
    pub namespace: AssetNamespace,
    pub filename: String,
    pub subfolder: PathBuf,
    pub bytes: Vec<u8>,
    pub overwrite: bool,
    pub tags: BTreeSet<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct UploadResult {
    pub record: AssetRecord,
    pub duplicate: bool,
    pub collision_index: Option<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AssetByteRange {
    pub start: u64,
    pub end_inclusive: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssetViewRequest {
    pub identity: AssetIdentity,
    pub range: Option<AssetByteRange>,
    pub download: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssetView {
    pub bytes: Vec<u8>,
    pub content_type: String,
    pub content_disposition: String,
    pub total_size: u64,
    pub range: Option<AssetByteRange>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssetQuery {
    pub namespace: Option<AssetNamespace>,
    pub text: Option<String>,
    pub required_tags: BTreeSet<String>,
    pub availability: Option<AssetAvailability>,
    pub offset: usize,
    pub limit: usize,
}

impl Default for AssetQuery {
    fn default() -> Self {
        Self {
            namespace: None,
            text: None,
            required_tags: BTreeSet::new(),
            availability: None,
            offset: 0,
            limit: DEFAULT_PAGE_SIZE,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssetPage {
    pub records: Vec<AssetRecord>,
    pub total: usize,
    pub next_offset: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct AssetService {
    roots: AssetRoots,
    artifact_index: ArtifactIndex,
    enrichments: BTreeMap<AssetIdentity, AssetEnrichment>,
    state_identity: AssetIdentity,
    state_recovery: Option<AssetStateRecovery>,
    max_asset_bytes: u64,
    max_range_bytes: u64,
    max_index_bytes: usize,
    max_records: usize,
    metadata_limits: MetadataLimits,
}

pub type SharedAssetService = Arc<Mutex<AssetService>>;

#[derive(Clone, Debug)]
struct NativeAssetResolutionRecord {
    service_id: Uuid,
    attempt_id: sim_types::AttemptId,
    identity: AssetIdentity,
    source_type_id: String,
    byte_length: u64,
    sha256: String,
}

#[derive(Clone, Debug)]
struct SourceAssetResolutionRecord {
    name: String,
    identity: AssetIdentity,
    byte_length: u64,
    sha256: String,
}

pub struct NativeAssetResolverRegistry {
    assets: SharedAssetService,
    authorization: AuthorizedCapabilities,
    references: Mutex<BTreeMap<Uuid, NativeAssetResolutionRecord>>,
    model_sources: Mutex<NativeModelSourceBridgeState>,
}

#[derive(Default)]
struct NativeModelSourceBridgeState {
    model_store: Option<ModelStore>,
    used_sessions: BTreeMap<Uuid, sim_types::AttemptId>,
    sessions: BTreeMap<Uuid, NativeModelSourceBridgeSession>,
}

struct NativeModelSourceBridgeSession {
    context: WorkerModelSourceContext,
    next_call_ordinal: u64,
    sources: Vec<NativeModelSourceBridgeEntry>,
}

struct NativeModelSourceBridgeEntry {
    _model: Arc<LoadedModel>,
    source: VerifiedModelStreamSource,
}

impl std::fmt::Debug for NativeAssetResolverRegistry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeAssetResolverRegistry")
            .field(
                "reference_count",
                &self.references.lock().map(|rows| rows.len()).ok(),
            )
            .finish_non_exhaustive()
    }
}

impl NativeAssetResolverRegistry {
    pub fn new(assets: SharedAssetService, authorization: AuthorizedCapabilities) -> Arc<Self> {
        Arc::new(Self {
            assets,
            authorization,
            references: Mutex::new(BTreeMap::new()),
            model_sources: Mutex::new(NativeModelSourceBridgeState::default()),
        })
    }

    pub fn seal_for_node(
        &self,
        service_identity: &NativeNodeServiceIdentity,
        identity: AssetIdentity,
        source_type_id: impl Into<String>,
    ) -> Result<NativeAssetReference, NativeAssetServiceError> {
        let source_type_id = source_type_id.into();
        let record = self
            .assets
            .lock()
            .map_err(|_| NativeAssetServiceError::Rejected)?
            .record(&identity)
            .ok_or(NativeAssetServiceError::Missing)?;
        if record.availability == AssetAvailability::Missing || record.byte_size == 0 {
            return Err(NativeAssetServiceError::Missing);
        }
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.native-asset-reference.v1");
        hasher.update(service_identity.service_id().as_bytes());
        hasher.update(identity.profile_id.as_bytes());
        hasher.update(identity.namespace.locator_type().as_bytes());
        if let Some(root_id) = &identity.root_id {
            hasher.update(root_id.as_bytes());
        }
        hasher.update(identity.relative_path.as_os_str().as_encoded_bytes());
        hasher.update(source_type_id.as_bytes());
        hasher.update(record.byte_size.to_le_bytes());
        hasher.update(record.sha256.as_bytes());
        let digest = hasher.finalize();
        let mut reference_bytes = [0_u8; 16];
        reference_bytes.copy_from_slice(&digest[..16]);
        let reference_id = Uuid::from_bytes(reference_bytes);
        let reference = NativeAssetReference::checked(
            service_identity.service_id(),
            reference_id,
            source_type_id.clone(),
            record.byte_size,
            record.sha256.clone(),
        )?;
        let resolution = NativeAssetResolutionRecord {
            service_id: service_identity.service_id(),
            attempt_id: service_identity.attempt_id(),
            identity,
            source_type_id,
            byte_length: record.byte_size,
            sha256: record.sha256,
        };
        let mut references = self
            .references
            .lock()
            .map_err(|_| NativeAssetServiceError::Rejected)?;
        if let Some(existing) = references.get(&reference_id)
            && existing != &resolution
        {
            return Err(NativeAssetServiceError::InvalidReference);
        }
        references.insert(reference_id, resolution);
        Ok(reference)
    }

    fn list_names_for_node(
        &self,
        folder_category: &str,
        cancellation: &CancellationToken,
    ) -> Result<Vec<String>, NativeAssetServiceError> {
        if cancellation.is_cancelled() {
            return Err(NativeAssetServiceError::Cancelled);
        }
        self.assets
            .lock()
            .map_err(|_| NativeAssetServiceError::Rejected)?
            .list_source_asset_names(folder_category, &self.authorization, cancellation)
            .map_err(map_native_asset_error)
    }

    fn resolve_names_for_node(
        &self,
        service_identity: &NativeNodeServiceIdentity,
        request: &NativeAssetNameResolutionRequest,
        cancellation: &CancellationToken,
    ) -> Result<NativeAssetNameResolution, NativeAssetServiceError> {
        if cancellation.is_cancelled() {
            return Err(NativeAssetServiceError::Cancelled);
        }
        let resolved = self
            .assets
            .lock()
            .map_err(|_| NativeAssetServiceError::Rejected)?
            .resolve_source_asset_names(
                request.folder_category(),
                request.names(),
                &self.authorization,
                cancellation,
            )
            .map_err(map_native_asset_error)?;
        if cancellation.is_cancelled() {
            return Err(NativeAssetServiceError::Cancelled);
        }

        let mut prepared = Vec::new();
        prepared
            .try_reserve_exact(resolved.len())
            .map_err(|_| NativeAssetServiceError::TooLarge)?;
        for asset in resolved {
            let mut hasher = Sha256::new();
            hasher.update(b"zed.sim.native-asset-reference.v1");
            hasher.update(service_identity.service_id().as_bytes());
            hasher.update(asset.identity.profile_id.as_bytes());
            hasher.update(asset.identity.namespace.locator_type().as_bytes());
            if let Some(root_id) = &asset.identity.root_id {
                hasher.update(root_id.as_bytes());
            }
            hasher.update(asset.identity.relative_path.as_os_str().as_encoded_bytes());
            hasher.update(request.source_type_id().as_bytes());
            hasher.update(asset.byte_length.to_le_bytes());
            hasher.update(asset.sha256.as_bytes());
            let digest = hasher.finalize();
            let mut reference_bytes = [0_u8; 16];
            reference_bytes.copy_from_slice(&digest[..16]);
            let reference_id = Uuid::from_bytes(reference_bytes);
            let reference = NativeAssetReference::checked(
                service_identity.service_id(),
                reference_id,
                request.source_type_id().to_owned(),
                asset.byte_length,
                asset.sha256.clone(),
            )?;
            let record = NativeAssetResolutionRecord {
                service_id: service_identity.service_id(),
                attempt_id: service_identity.attempt_id(),
                identity: asset.identity,
                source_type_id: request.source_type_id().to_owned(),
                byte_length: asset.byte_length,
                sha256: asset.sha256,
            };
            prepared.push((asset.name, reference_id, reference, record));
        }

        let mut references = self
            .references
            .lock()
            .map_err(|_| NativeAssetServiceError::Rejected)?;
        for (_, reference_id, _, record) in &prepared {
            if let Some(existing) = references.get(reference_id)
                && existing != record
            {
                return Err(NativeAssetServiceError::InvalidReference);
            }
        }
        if cancellation.is_cancelled() {
            return Err(NativeAssetServiceError::Cancelled);
        }
        for (_, reference_id, _, record) in &prepared {
            references.insert(*reference_id, record.clone());
        }
        let assets = prepared
            .into_iter()
            .map(|(name, _, reference, _)| NativeNamedAssetReference::checked(name, reference))
            .collect::<Result<Vec<_>, _>>()?;
        NativeAssetNameResolution::checked(request.folder_category().to_owned(), assets)
    }

    pub(crate) fn serve_model_source_request(
        &self,
        expected_context: &WorkerModelSourceContext,
        request: &WorkerModelSourceRequest,
        cancellation: &CancellationToken,
    ) -> Result<WorkerModelSourceResponse, WorkerModelSourceError> {
        let result = self.serve_model_source_request_inner(expected_context, request, cancellation);
        if result.is_err()
            && let Ok(mut state) = self.model_sources.lock()
        {
            state.sessions.remove(&request.context.session_id);
        }
        result
    }

    #[cfg(feature = "test-support")]
    #[doc(hidden)]
    pub fn test_serve_model_source_request(
        &self,
        expected_context: &WorkerModelSourceContext,
        request: &WorkerModelSourceRequest,
        cancellation: &CancellationToken,
    ) -> Result<WorkerModelSourceResponse, WorkerModelSourceError> {
        self.serve_model_source_request(expected_context, request, cancellation)
    }

    fn serve_model_source_request_inner(
        &self,
        expected_context: &WorkerModelSourceContext,
        request: &WorkerModelSourceRequest,
        cancellation: &CancellationToken,
    ) -> Result<WorkerModelSourceResponse, WorkerModelSourceError> {
        check_model_source_cancellation(cancellation)?;
        validate_model_source_host_context(expected_context, &request.context)?;
        match &request.operation {
            WorkerModelSourceOperation::Open {
                folder_category,
                source_names,
            } => {
                self.open_model_source_session(request, folder_category, source_names, cancellation)
            }
            WorkerModelSourceOperation::Read {
                source_ordinal,
                tensor_ordinal,
                byte_offset,
                byte_length,
            } => self.read_model_source_session(
                request,
                *source_ordinal,
                *tensor_ordinal,
                *byte_offset,
                *byte_length,
                cancellation,
            ),
            WorkerModelSourceOperation::Close => {
                self.close_model_source_session(request, cancellation)
            }
            WorkerModelSourceOperation::ReadArtifact {
                source_ordinal,
                artifact_ordinal,
                artifact_sha256,
                byte_offset,
                byte_length,
            } => self.read_model_source_artifact_session(
                request,
                *source_ordinal,
                *artifact_ordinal,
                artifact_sha256,
                *byte_offset,
                *byte_length,
                cancellation,
            ),
        }
    }

    fn open_model_source_session(
        &self,
        request: &WorkerModelSourceRequest,
        folder_category: &str,
        source_names: &[String],
        cancellation: &CancellationToken,
    ) -> Result<WorkerModelSourceResponse, WorkerModelSourceError> {
        if request.call_ordinal != 1
            || worker_model_source_selection_sha256(folder_category, source_names)?
                != request.context.ordered_source_identity_sha256
        {
            return Err(WorkerModelSourceError::InvalidSourceSelection);
        }
        let mut state = self
            .model_sources
            .lock()
            .map_err(|_| WorkerModelSourceError::HostFailure)?;
        if state
            .used_sessions
            .contains_key(&request.context.session_id)
        {
            return Err(WorkerModelSourceError::Replay);
        }
        if state.used_sessions.len() >= MAX_RETAINED_MODEL_SOURCE_SESSION_IDS {
            return Err(WorkerModelSourceError::HostFailure);
        }
        state
            .used_sessions
            .insert(request.context.session_id, request.context.attempt_id);
        if !state.sessions.is_empty() {
            return Err(WorkerModelSourceError::InvalidOrder);
        }
        let mut model_store = state
            .model_store
            .take()
            .map(Ok)
            .unwrap_or_else(|| ModelStore::new(sim_model::ParserLimits::default()))
            .map_err(|_| WorkerModelSourceError::HostFailure)?;
        let result = (|| {
            let mut assets = self
                .assets
                .lock()
                .map_err(|_| WorkerModelSourceError::HostFailure)?;
            let resolved = assets
                .resolve_source_asset_names(
                    folder_category,
                    source_names,
                    &self.authorization,
                    cancellation,
                )
                .map_err(map_model_source_asset_error)?;
            if resolved.len() != source_names.len() {
                return Err(WorkerModelSourceError::InvalidSourceSelection);
            }
            let mut entries = Vec::new();
            entries
                .try_reserve_exact(resolved.len())
                .map_err(|_| WorkerModelSourceError::HostFailure)?;
            let mut manifests = Vec::new();
            manifests
                .try_reserve_exact(resolved.len())
                .map_err(|_| WorkerModelSourceError::HostFailure)?;
            for (source_ordinal, resolved_source) in resolved.into_iter().enumerate() {
                check_model_source_cancellation(cancellation)?;
                let model = assets
                    .load_model(
                        &resolved_source.identity,
                        &mut model_store,
                        &self.authorization,
                        cancellation,
                    )
                    .map_err(map_model_source_asset_error)?;
                let source = model_store
                    .verified_stream_source(&assets.artifact_index, &model, cancellation)
                    .map_err(map_model_source_store_error)?;
                manifests.push(project_model_source_manifest(source_ordinal, &source)?);
                entries.push(NativeModelSourceBridgeEntry {
                    _model: model,
                    source,
                });
            }
            check_model_source_cancellation(cancellation)?;
            Ok((entries, manifests))
        })();
        state.model_store = Some(model_store);
        let (sources, manifests) = result?;
        let opened = WorkerModelSourceOpened::checked(
            request.context.session_id,
            request.call_ordinal,
            request.context.ordered_source_identity_sha256.clone(),
            manifests,
        )?;
        state.sessions.insert(
            request.context.session_id,
            NativeModelSourceBridgeSession {
                context: request.context.clone(),
                next_call_ordinal: 2,
                sources,
            },
        );
        Ok(WorkerModelSourceResponse::Opened(opened))
    }

    #[allow(clippy::too_many_arguments)]
    fn read_model_source_session(
        &self,
        request: &WorkerModelSourceRequest,
        source_ordinal: u32,
        tensor_ordinal: u32,
        byte_offset: u64,
        byte_length: u32,
        cancellation: &CancellationToken,
    ) -> Result<WorkerModelSourceResponse, WorkerModelSourceError> {
        let mut state = self
            .model_sources
            .lock()
            .map_err(|_| WorkerModelSourceError::HostFailure)?;
        let mut session = state
            .sessions
            .remove(&request.context.session_id)
            .ok_or(WorkerModelSourceError::Closed)?;
        validate_model_source_host_context(&session.context, &request.context)?;
        if request.call_ordinal < session.next_call_ordinal {
            return Err(WorkerModelSourceError::Replay);
        }
        if request.call_ordinal != session.next_call_ordinal {
            return Err(WorkerModelSourceError::InvalidOrder);
        }
        let source = session
            .sources
            .get(
                usize::try_from(source_ordinal)
                    .map_err(|_| WorkerModelSourceError::InvalidSourceOrdinal)?,
            )
            .ok_or(WorkerModelSourceError::InvalidSourceOrdinal)?;
        let tensor = source
            .source
            .tensors()
            .get(
                usize::try_from(tensor_ordinal)
                    .map_err(|_| WorkerModelSourceError::InvalidTensorOrdinal)?,
            )
            .ok_or(WorkerModelSourceError::InvalidTensorOrdinal)?;
        let byte_length_u64 = u64::from(byte_length);
        if byte_length == 0
            || usize::try_from(byte_length).map_or(true, |length| {
                length > sim_types::MAX_WORKER_MODEL_SOURCE_CHUNK_BYTES
            })
            || byte_offset
                .checked_add(byte_length_u64)
                .is_none_or(|end| end > tensor.byte_length())
        {
            return Err(WorkerModelSourceError::RangeLimit);
        }
        let tensor_name = tensor.name().to_owned();
        let model_store = state
            .model_store
            .take()
            .ok_or(WorkerModelSourceError::HostFailure)?;
        let result = (|| {
            let assets = self
                .assets
                .lock()
                .map_err(|_| WorkerModelSourceError::HostFailure)?;
            model_store
                .read_verified_tensor_range(
                    &assets.artifact_index,
                    &source.source,
                    &tensor_name,
                    byte_offset,
                    byte_length_u64,
                    cancellation,
                )
                .map_err(map_model_source_store_error)
        })();
        state.model_store = Some(model_store);
        let bytes = result?;
        check_model_source_cancellation(cancellation)?;
        let chunk = WorkerModelSourceChunk::checked(
            request.context.session_id,
            request.call_ordinal,
            source_ordinal,
            tensor_ordinal,
            byte_offset,
            bytes,
        )?;
        session.next_call_ordinal = session
            .next_call_ordinal
            .checked_add(1)
            .ok_or(WorkerModelSourceError::InvalidOrder)?;
        state.sessions.insert(request.context.session_id, session);
        Ok(WorkerModelSourceResponse::Chunk(chunk))
    }

    #[allow(clippy::too_many_arguments)]
    fn read_model_source_artifact_session(
        &self,
        request: &WorkerModelSourceRequest,
        source_ordinal: u32,
        artifact_ordinal: u32,
        artifact_sha256: &WorkerSha256Digest,
        byte_offset: u64,
        byte_length: u32,
        cancellation: &CancellationToken,
    ) -> Result<WorkerModelSourceResponse, WorkerModelSourceError> {
        let mut state = self
            .model_sources
            .lock()
            .map_err(|_| WorkerModelSourceError::HostFailure)?;
        let mut session = state
            .sessions
            .remove(&request.context.session_id)
            .ok_or(WorkerModelSourceError::Closed)?;
        validate_model_source_host_context(&session.context, &request.context)?;
        if request.call_ordinal < session.next_call_ordinal {
            return Err(WorkerModelSourceError::Replay);
        }
        if request.call_ordinal != session.next_call_ordinal {
            return Err(WorkerModelSourceError::InvalidOrder);
        }
        let source = session
            .sources
            .get(
                usize::try_from(source_ordinal)
                    .map_err(|_| WorkerModelSourceError::InvalidSourceOrdinal)?,
            )
            .ok_or(WorkerModelSourceError::InvalidSourceOrdinal)?;
        let artifact_ordinal_usize = usize::try_from(artifact_ordinal)
            .map_err(|_| WorkerModelSourceError::InvalidSourceSelection)?;
        let artifact = source
            .source
            .artifacts()
            .get(artifact_ordinal_usize)
            .ok_or(WorkerModelSourceError::InvalidSourceSelection)?;
        if artifact.sha256() != artifact_sha256.as_str() {
            return Err(WorkerModelSourceError::SourceChanged);
        }
        let byte_length_u64 = u64::from(byte_length);
        if byte_length == 0
            || usize::try_from(byte_length).map_or(true, |length| {
                length > sim_types::MAX_WORKER_MODEL_SOURCE_CHUNK_BYTES
            })
            || byte_offset
                .checked_add(byte_length_u64)
                .is_none_or(|end| end > artifact.byte_size())
        {
            return Err(WorkerModelSourceError::RangeLimit);
        }
        let model_store = state
            .model_store
            .take()
            .ok_or(WorkerModelSourceError::HostFailure)?;
        let result = (|| {
            let assets = self
                .assets
                .lock()
                .map_err(|_| WorkerModelSourceError::HostFailure)?;
            model_store
                .read_verified_artifact_range(
                    &assets.artifact_index,
                    &source.source,
                    artifact_ordinal_usize,
                    artifact_sha256.as_str(),
                    byte_offset,
                    byte_length_u64,
                    cancellation,
                )
                .map_err(map_model_source_store_error)
        })();
        state.model_store = Some(model_store);
        let bytes = result?;
        check_model_source_cancellation(cancellation)?;
        let chunk = WorkerModelSourceArtifactChunk::checked(
            request.context.session_id,
            request.call_ordinal,
            source_ordinal,
            artifact_ordinal,
            artifact_sha256.clone(),
            byte_offset,
            bytes,
        )?;
        session.next_call_ordinal = session
            .next_call_ordinal
            .checked_add(1)
            .ok_or(WorkerModelSourceError::InvalidOrder)?;
        state.sessions.insert(request.context.session_id, session);
        Ok(WorkerModelSourceResponse::ArtifactChunk(chunk))
    }

    fn close_model_source_session(
        &self,
        request: &WorkerModelSourceRequest,
        cancellation: &CancellationToken,
    ) -> Result<WorkerModelSourceResponse, WorkerModelSourceError> {
        let mut state = self
            .model_sources
            .lock()
            .map_err(|_| WorkerModelSourceError::HostFailure)?;
        let session = state
            .sessions
            .remove(&request.context.session_id)
            .ok_or(WorkerModelSourceError::Closed)?;
        validate_model_source_host_context(&session.context, &request.context)?;
        if request.call_ordinal < session.next_call_ordinal {
            return Err(WorkerModelSourceError::Replay);
        }
        if request.call_ordinal != session.next_call_ordinal {
            return Err(WorkerModelSourceError::InvalidOrder);
        }
        check_model_source_cancellation(cancellation)?;
        Ok(WorkerModelSourceResponse::Closed(
            WorkerModelSourceClosed::checked(request.context.session_id, request.call_ordinal)?,
        ))
    }

    pub(crate) fn revoke_model_source_sessions(&self) {
        if let Ok(mut state) = self.model_sources.lock() {
            state.sessions.clear();
        }
    }

    pub(crate) fn revoke_model_source_session(&self, session_id: Uuid) {
        if let Ok(mut state) = self.model_sources.lock() {
            state.sessions.remove(&session_id);
        }
    }

    pub(crate) fn replace_model_source_service(&self) {
        if let Ok(mut state) = self.model_sources.lock() {
            state.sessions.clear();
            state.used_sessions.clear();
        }
    }

    #[cfg(feature = "test-support")]
    #[doc(hidden)]
    pub fn test_replace_model_source_service(&self) {
        self.replace_model_source_service();
    }

    pub fn retire_attempt(&self, attempt_id: sim_types::AttemptId) {
        if let Ok(mut references) = self.references.lock() {
            references.retain(|_, record| record.attempt_id != attempt_id);
        }
        if let Ok(mut state) = self.model_sources.lock() {
            state
                .sessions
                .retain(|_, session| session.context.attempt_id != attempt_id);
            state
                .used_sessions
                .retain(|_, retained_attempt_id| *retained_attempt_id != attempt_id);
        }
    }

    pub fn node_service(
        self: &Arc<Self>,
        identity: NativeNodeServiceIdentity,
    ) -> Arc<dyn NativeAssetResolver> {
        Arc::new(RuntimeNativeAssetResolver {
            identity,
            registry: self.clone(),
        })
    }
}

impl PartialEq for NativeAssetResolutionRecord {
    fn eq(&self, other: &Self) -> bool {
        self.service_id == other.service_id
            && self.attempt_id == other.attempt_id
            && self.identity == other.identity
            && self.source_type_id == other.source_type_id
            && self.byte_length == other.byte_length
            && self.sha256 == other.sha256
    }
}

fn check_model_source_cancellation(
    cancellation: &CancellationToken,
) -> Result<(), WorkerModelSourceError> {
    cancellation
        .check()
        .map_err(|_| WorkerModelSourceError::Cancelled)
}

fn validate_model_source_host_context(
    expected: &WorkerModelSourceContext,
    actual: &WorkerModelSourceContext,
) -> Result<(), WorkerModelSourceError> {
    if actual.session_id != expected.session_id {
        return Err(WorkerModelSourceError::ForeignSession);
    }
    if actual.attempt_id != expected.attempt_id
        || actual.attempt_generation != expected.attempt_generation
    {
        return Err(WorkerModelSourceError::StaleAttempt);
    }
    if actual.node_id != expected.node_id || actual.node_generation != expected.node_generation {
        return Err(WorkerModelSourceError::StaleNode);
    }
    if actual.service_id != expected.service_id
        || actual.service_generation != expected.service_generation
    {
        return Err(WorkerModelSourceError::StaleService);
    }
    if actual.ordered_source_identity_sha256 != expected.ordered_source_identity_sha256 {
        return Err(WorkerModelSourceError::InvalidSourceSelection);
    }
    Ok(())
}

fn project_model_source_manifest(
    source_ordinal: usize,
    source: &VerifiedModelStreamSource,
) -> Result<WorkerModelSourceManifest, WorkerModelSourceError> {
    let source_ordinal =
        u32::try_from(source_ordinal).map_err(|_| WorkerModelSourceError::InvalidSourceOrdinal)?;
    let model_identity_sha256 = WorkerSha256Digest::new(source.model_identity().to_owned())
        .map_err(|_| WorkerModelSourceError::InvalidSourceSelection)?;
    let artifacts = source
        .artifacts()
        .iter()
        .map(|artifact| {
            Ok(WorkerModelSourceArtifact {
                sha256: WorkerSha256Digest::new(artifact.sha256().to_owned())
                    .map_err(|_| WorkerModelSourceError::InvalidSourceSelection)?,
                byte_size: artifact.byte_size(),
                format: match artifact.semantic() {
                    ModelStreamSemanticArtifact::None => match artifact.format() {
                        ModelFormat::Safetensors => WorkerModelSourceFormat::Safetensors,
                        ModelFormat::PytorchArchive => WorkerModelSourceFormat::PytorchArchive,
                        ModelFormat::Gguf => WorkerModelSourceFormat::Gguf,
                        ModelFormat::JsonConfig => WorkerModelSourceFormat::JsonConfig,
                        ModelFormat::JsonTokenizer => WorkerModelSourceFormat::JsonTokenizer,
                        ModelFormat::YamlConfig => WorkerModelSourceFormat::YamlConfig,
                        ModelFormat::SentencePiece => WorkerModelSourceFormat::SentencePiece,
                        ModelFormat::Tiktoken => WorkerModelSourceFormat::Tiktoken,
                    },
                    ModelStreamSemanticArtifact::SafetensorsMetadata {
                        source_artifact_ordinal,
                    } => WorkerModelSourceFormat::SafetensorsMetadata {
                        source_artifact_ordinal,
                    },
                    ModelStreamSemanticArtifact::ClipConfiguration { target_ordinal } => {
                        WorkerModelSourceFormat::ClipConfiguration { target_ordinal }
                    }
                    ModelStreamSemanticArtifact::ClipTokenizerJson { target_ordinal } => {
                        WorkerModelSourceFormat::ClipTokenizerJson { target_ordinal }
                    }
                    ModelStreamSemanticArtifact::ClipSentencePiece { target_ordinal } => {
                        WorkerModelSourceFormat::ClipSentencePiece { target_ordinal }
                    }
                    ModelStreamSemanticArtifact::ClipTiktoken { target_ordinal } => {
                        WorkerModelSourceFormat::ClipTiktoken { target_ordinal }
                    }
                },
                nested_state_disposition: match artifact.nested_state_disposition() {
                    ModelStreamNestedStateDisposition::Flat => {
                        WorkerModelSourceNestedStateDisposition::Flat
                    }
                    ModelStreamNestedStateDisposition::NestedStringToParam => {
                        WorkerModelSourceNestedStateDisposition::NestedStringToParam
                    }
                },
            })
        })
        .collect::<Result<Vec<_>, WorkerModelSourceError>>()?;
    let tensors = source
        .tensors()
        .iter()
        .map(|tensor| {
            Ok(WorkerModelSourceTensor {
                name: tensor.name().to_owned(),
                data_type: tensor.data_type().to_owned(),
                shape: tensor.shape().to_vec(),
                artifact_ordinal: u32::try_from(tensor.artifact_ordinal())
                    .map_err(|_| WorkerModelSourceError::InvalidSourceOrdinal)?,
                byte_offset: tensor.byte_offset(),
                byte_length: tensor.byte_length(),
            })
        })
        .collect::<Result<Vec<_>, WorkerModelSourceError>>()?;
    let limits = source.limits();
    Ok(WorkerModelSourceManifest {
        source_ordinal,
        model_identity_sha256,
        artifacts,
        tensors,
        aggregate_tensor_bytes: source.aggregate_tensor_bytes(),
        maximum_read_bytes: source.maximum_read_bytes(),
        parser_limits: WorkerModelSourceParserLimits {
            version: limits.version,
            manifest_bytes: limits.manifest_bytes,
            maximum_depth: limits.maximum_depth,
            maximum_tensors: limits.maximum_tensors,
            maximum_tensor_bytes: limits.maximum_tensor_bytes,
            maximum_aggregate_tensor_bytes: limits.maximum_aggregate_tensor_bytes,
            maximum_name_bytes: limits.maximum_name_bytes,
            maximum_archive_entries: limits.maximum_archive_entries,
            maximum_metadata_values: limits.maximum_metadata_values,
        },
    })
}

fn map_model_source_asset_error(error: AssetError) -> WorkerModelSourceError {
    match error {
        AssetError::Cancelled => WorkerModelSourceError::Cancelled,
        AssetError::PermissionDenied { .. } | AssetError::ProfileMismatch { .. } => {
            WorkerModelSourceError::StaleService
        }
        AssetError::Missing(_)
        | AssetError::UnknownAsset(_)
        | AssetError::UnknownSourceAsset { .. }
        | AssetError::ChangedDuringRead(_) => WorkerModelSourceError::SourceChanged,
        AssetError::TooLarge { .. } | AssetError::AllocationFailed => {
            WorkerModelSourceError::RangeLimit
        }
        AssetError::Model(error) => map_model_source_store_error(error),
        _ => WorkerModelSourceError::HostFailure,
    }
}

fn map_model_source_store_error(error: ModelStoreError) -> WorkerModelSourceError {
    match error {
        ModelStoreError::Cancelled => WorkerModelSourceError::Cancelled,
        ModelStoreError::UnknownTensor(_) => WorkerModelSourceError::InvalidTensorOrdinal,
        ModelStoreError::UnknownModelStreamArtifactOrdinal(_) => {
            WorkerModelSourceError::InvalidSourceSelection
        }
        ModelStoreError::UnsupportedModelStreamArtifactFormat { .. } => {
            WorkerModelSourceError::InvalidSourceSelection
        }
        ModelStoreError::ModelStreamTensorRangeOutOfBounds { .. }
        | ModelStoreError::ModelStreamArtifactRangeOutOfBounds { .. }
        | ModelStoreError::Limit(_)
        | ModelStoreError::Overflow(_)
        | ModelStoreError::AllocationFailed { .. } => WorkerModelSourceError::RangeLimit,
        ModelStoreError::Index(_)
        | ModelStoreError::ArtifactChanged { .. }
        | ModelStoreError::LoadedArtifactIdentityMismatch { .. }
        | ModelStoreError::StorageSourceChanged { .. }
        | ModelStoreError::StorageRangeChanged(_)
        | ModelStoreError::ModelStreamIo { .. }
        | ModelStoreError::ModelStreamArtifactIdentityMismatch { .. }
        | ModelStoreError::StaleModelStreamSource => WorkerModelSourceError::SourceChanged,
        ModelStoreError::ForeignModelHandle | ModelStoreError::ForeignModelStreamSource => {
            WorkerModelSourceError::StaleService
        }
        _ => WorkerModelSourceError::HostFailure,
    }
}

#[derive(Debug)]
struct RuntimeNativeAssetResolver {
    identity: NativeNodeServiceIdentity,
    registry: Arc<NativeAssetResolverRegistry>,
}

impl NativeAssetResolver for RuntimeNativeAssetResolver {
    fn identity(&self) -> &NativeNodeServiceIdentity {
        &self.identity
    }

    fn list_names(
        &self,
        request: &NativeAssetNameListRequest,
        cancellation: &CancellationToken,
    ) -> Result<NativeAssetNameList, NativeAssetServiceError> {
        let names = self
            .registry
            .list_names_for_node(request.folder_category(), cancellation)?;
        NativeAssetNameList::checked(request.folder_category().to_owned(), names)
    }

    fn resolve_names(
        &self,
        request: &NativeAssetNameResolutionRequest,
        cancellation: &CancellationToken,
    ) -> Result<NativeAssetNameResolution, NativeAssetServiceError> {
        self.registry
            .resolve_names_for_node(&self.identity, request, cancellation)
    }

    fn read_verified(
        &self,
        request: &NativeAssetReadRequest,
        cancellation: &CancellationToken,
    ) -> Result<NativeResolvedAsset, NativeAssetServiceError> {
        if cancellation.is_cancelled() {
            return Err(NativeAssetServiceError::Cancelled);
        }
        let reference = request.reference();
        if reference.service_id() != self.identity.service_id() {
            return Err(NativeAssetServiceError::InvalidReference);
        }
        let record = self
            .registry
            .references
            .lock()
            .map_err(|_| NativeAssetServiceError::Rejected)?
            .get(&reference.reference_id())
            .cloned()
            .ok_or(NativeAssetServiceError::InvalidReference)?;
        if record.service_id != self.identity.service_id()
            || record.source_type_id != reference.source_type_id()
            || record.byte_length != reference.byte_length()
            || record.sha256 != reference.sha256()
            || record.byte_length > request.maximum_bytes()
        {
            return Err(NativeAssetServiceError::InvalidReference);
        }
        let bytes = self
            .registry
            .assets
            .lock()
            .map_err(|_| NativeAssetServiceError::Rejected)?
            .read_verified(
                &record.identity,
                &self.registry.authorization,
                cancellation,
                request.maximum_bytes(),
            )
            .map_err(map_native_asset_error)?;
        if cancellation.is_cancelled() {
            return Err(NativeAssetServiceError::Cancelled);
        }
        NativeResolvedAsset::checked(reference.clone(), Arc::from(bytes), record.sha256)
    }
}

fn map_native_asset_error(error: AssetError) -> NativeAssetServiceError {
    match error {
        AssetError::Cancelled => NativeAssetServiceError::Cancelled,
        AssetError::PermissionDenied { .. } | AssetError::ProfileMismatch { .. } => {
            NativeAssetServiceError::PermissionDenied
        }
        AssetError::Missing(_) | AssetError::UnknownAsset(_) => NativeAssetServiceError::Missing,
        AssetError::UnknownSourceAsset { .. } => NativeAssetServiceError::Missing,
        AssetError::UnsupportedSourceFolderCategory(_) => {
            NativeAssetServiceError::InvalidFolderCategory
        }
        AssetError::InvalidSourceAssetName(_) => NativeAssetServiceError::InvalidAssetName,
        AssetError::TooLarge { .. } | AssetError::AllocationFailed => {
            NativeAssetServiceError::TooLarge
        }
        AssetError::ChangedDuringRead(_) => NativeAssetServiceError::ChangedDuringRead,
        _ => NativeAssetServiceError::Rejected,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AssetStateRecovery {
    Migrated {
        removed_roots: usize,
        added_roots: usize,
        dropped_records: usize,
    },
    Quarantined {
        quarantine_relative_path: PathBuf,
        reason: String,
    },
}

pub fn open_native_profile_asset_service(
    profile_id: impl Into<String>,
    profile_root: &Path,
    configured_model_roots: &[PathBuf],
) -> Result<SharedAssetService, AssetError> {
    let profile_id = profile_id.into();
    let mut primary_roots = Vec::new();
    for (namespace, directory_name) in [
        (AssetNamespace::Input, "input"),
        (AssetNamespace::Output, "output"),
        (AssetNamespace::Temporary, "temporary"),
        (AssetNamespace::Model, "model"),
        (AssetNamespace::Plugin, "plugin"),
    ] {
        let path = profile_root.join(directory_name);
        fs::create_dir_all(&path).map_err(|error| AssetError::Io {
            path: path.clone(),
            message: error.to_string(),
        })?;
        primary_roots.push((namespace, path));
    }
    let mut roots = AssetRoots::new(profile_id, primary_roots)?;
    for path in configured_model_roots {
        if !path.is_dir() {
            return Err(AssetError::UnsafePath {
                path: path.clone(),
                reason: "configured model root is not an existing directory".to_owned(),
            });
        }
        roots.add_configured_root(AssetNamespace::Model, path.clone())?;
    }
    Ok(Arc::new(Mutex::new(AssetService::open(roots)?)))
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct AssetEnrichment {
    identity: AssetIdentity,
    content_type: String,
    metadata_carrier: MetadataCarrier,
    metadata: ComfyMetadata,
    metadata_diagnostics: Vec<String>,
    tags: BTreeSet<String>,
}

impl AssetService {
    pub fn open(roots: AssetRoots) -> Result<Self, AssetError> {
        Self::with_limits(
            roots,
            DEFAULT_MAX_ASSET_BYTES,
            DEFAULT_MAX_RANGE_BYTES,
            MetadataLimits::default(),
        )
    }

    pub fn with_limits(
        roots: AssetRoots,
        max_asset_bytes: u64,
        max_range_bytes: u64,
        metadata_limits: MetadataLimits,
    ) -> Result<Self, AssetError> {
        if max_asset_bytes == 0 || max_range_bytes == 0 {
            return Err(AssetError::InvalidLimit);
        }
        let state_identity = roots.identity(AssetNamespace::Temporary, ASSET_INDEX_FILENAME)?;
        let state_root = roots.artifact_root_for_identity(&state_identity)?.clone();
        let state_bytes = match state_root
            .read_private_file(&state_identity.relative_path, DEFAULT_MAX_ASSET_INDEX_BYTES)
        {
            Ok(bytes) => bytes,
            Err(error @ ArtifactIndexError::InvalidSnapshot(_)) => {
                let quarantine_relative_path = state_root
                    .quarantine_private_file(&state_identity.relative_path)
                    .map_err(map_artifact_error)?
                    .ok_or_else(|| AssetError::InvalidIndex(error.to_string()))?;
                let artifact_index = new_artifact_index(&roots)?;
                let service = Self {
                    roots,
                    artifact_index,
                    enrichments: BTreeMap::new(),
                    state_identity,
                    state_recovery: Some(AssetStateRecovery::Quarantined {
                        quarantine_relative_path,
                        reason: error.to_string(),
                    }),
                    max_asset_bytes,
                    max_range_bytes,
                    max_index_bytes: DEFAULT_MAX_ASSET_INDEX_BYTES,
                    max_records: DEFAULT_MAX_ASSET_RECORDS,
                    metadata_limits,
                };
                service.persist_state()?;
                return Ok(service);
            }
            Err(error) => return Err(map_artifact_error(error)),
        };
        let (artifact_index, enrichments, state_recovery) = match state_bytes {
            None => (new_artifact_index(&roots)?, BTreeMap::new(), None),
            Some(bytes) => {
                match read_asset_service_state(&bytes, &roots, DEFAULT_MAX_ASSET_RECORDS) {
                    Ok(state) => state,
                    Err(error) => {
                        let quarantine_relative_path = state_root
                            .quarantine_private_file(&state_identity.relative_path)
                            .map_err(map_artifact_error)?
                            .ok_or_else(|| AssetError::InvalidIndex(error.to_string()))?;
                        (
                            new_artifact_index(&roots)?,
                            BTreeMap::new(),
                            Some(AssetStateRecovery::Quarantined {
                                quarantine_relative_path,
                                reason: error.to_string(),
                            }),
                        )
                    }
                }
            }
        };
        let service = Self {
            roots,
            artifact_index,
            enrichments,
            state_identity,
            state_recovery,
            max_asset_bytes,
            max_range_bytes,
            max_index_bytes: DEFAULT_MAX_ASSET_INDEX_BYTES,
            max_records: DEFAULT_MAX_ASSET_RECORDS,
            metadata_limits,
        };
        if service.state_recovery.is_some() {
            service.persist_state()?;
        }
        Ok(service)
    }

    pub fn roots(&self) -> &AssetRoots {
        &self.roots
    }

    #[cfg(test)]
    pub(crate) fn artifact_index(&self) -> &ArtifactIndex {
        &self.artifact_index
    }

    pub fn state_recovery(&self) -> Option<&AssetStateRecovery> {
        self.state_recovery.as_ref()
    }

    pub fn read_verified(
        &self,
        identity: &AssetIdentity,
        authorization: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
        maximum_bytes: u64,
    ) -> Result<Vec<u8>, AssetError> {
        require_asset_authorization(
            authorization,
            &self.roots.profile_id,
            identity.namespace,
            AssetAction::Read,
        )?;
        if maximum_bytes == 0 {
            return Err(AssetError::InvalidLimit);
        }
        check_cancelled(cancellation)?;
        let key = self.roots.artifact_key(identity)?;
        let artifact = self
            .artifact_index
            .record(&key)
            .ok_or_else(|| AssetError::UnknownAsset(identity.clone()))?;
        if artifact.availability == AssetAvailability::Missing {
            return Err(AssetError::Missing(identity.clone()));
        }
        if artifact.byte_size > maximum_bytes {
            return Err(AssetError::TooLarge {
                actual: artifact.byte_size,
                limit: maximum_bytes,
            });
        }
        let length =
            usize::try_from(artifact.byte_size).map_err(|_| AssetError::AllocationFailed)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| AssetError::AllocationFailed)?;
        let mut verified = self
            .artifact_index
            .open_verified(&key, cancellation)
            .map_err(|error| map_artifact_error_for_identity(error, identity))?;
        verified
            .file_mut()
            .take(maximum_bytes.saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(|error| AssetError::Io {
                path: verified.path().to_path_buf(),
                message: error.to_string(),
            })?;
        check_cancelled(cancellation)?;
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) != artifact.byte_size {
            return Err(AssetError::ChangedDuringRead(identity.clone()));
        }
        verified
            .verify_unchanged()
            .map_err(|_| AssetError::ChangedDuringRead(identity.clone()))?;
        Ok(bytes)
    }

    pub fn load_model(
        &self,
        identity: &AssetIdentity,
        model_store: &mut ModelStore,
        authorization: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<Arc<LoadedModel>, AssetError> {
        if identity.namespace != AssetNamespace::Model {
            return Err(AssetError::ModelNamespaceRequired(identity.namespace));
        }
        require_asset_authorization(
            authorization,
            &self.roots.profile_id,
            identity.namespace,
            AssetAction::Read,
        )?;
        let key = self.roots.artifact_key(identity)?;
        model_store
            .load(&self.artifact_index, &key, cancellation)
            .map_err(AssetError::Model)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_image_vae_with_context(
        &self,
        identity: &AssetIdentity,
        model_store: &mut ModelStore,
        target: &VaeExecutionTarget,
        patch: PatchGraphIdentity,
        boundary: VaeBoundary,
        decode_clamp: [f32; 2],
        explicit_configuration: Option<&str>,
        backend: &CpuBackend,
        authorization: &AuthorizedCapabilities,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeVae, AssetImageVaeLoadError> {
        require_asset_authorization(
            authorization,
            &self.roots.profile_id,
            identity.namespace,
            AssetAction::Read,
        )?;
        validate_native_vae_backend_target(backend, target)?;
        let loaded = self.load_model(identity, model_store, authorization, context.cancellation)?;
        let key = self.roots.artifact_key(identity)?;
        let artifact = self
            .artifact_index
            .record(&key)
            .ok_or_else(|| AssetError::UnknownAsset(identity.clone()))?;
        let registry = VaeArchitectureRegistry::checked()?;
        let (family_registry, latent_registry) = VaeArchitectureRegistry::canonical_targets()?;
        let selection = if let Some(configuration) = explicit_configuration {
            let probe = model_store
                .family_probe(&loaded, context.cancellation)
                .map_err(AssetError::Model)?;
            let selection =
                registry.select_explicit(&probe, configuration, context.cancellation)?;
            registry.validate_target(
                &selection,
                target,
                &family_registry,
                &latent_registry,
                context.cancellation,
            )?;
            selection
        } else {
            registry.select_loaded(
                model_store,
                &loaded,
                target,
                &family_registry,
                &latent_registry,
                context.cancellation,
            )?
        };
        let latent_definition = latent_registry.get(target.latent_format()).ok_or_else(|| {
            VaeArchitectureError::UnknownLatentFormat(
                target.latent_format().identifier().to_owned(),
            )
        })?;
        let descriptor = VaeDescriptor::checked_selection(
            artifact,
            &selection,
            target,
            &family_registry,
            &latent_registry,
            patch,
            boundary,
            decode_clamp,
            context.cancellation,
        )?;
        Ok(load_image_vae_from_model_store_with_context(
            backend,
            model_store,
            &self.artifact_index,
            loaded,
            descriptor,
            latent_definition,
            context,
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_and_execute_image_vae_with_context(
        &self,
        identity: &AssetIdentity,
        model_store: &mut ModelStore,
        target: &VaeExecutionTarget,
        patch: PatchGraphIdentity,
        boundary: VaeBoundary,
        decode_clamp: [f32; 2],
        explicit_configuration: Option<&str>,
        operation: VaeOperation,
        input: &Tensor,
        backend: &CpuBackend,
        authorization: &AuthorizedCapabilities,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, AssetImageVaeLoadError> {
        let vae = self.load_image_vae_with_context(
            identity,
            model_store,
            target,
            patch,
            boundary,
            decode_clamp,
            explicit_configuration,
            backend,
            authorization,
            context,
        )?;
        match operation {
            VaeOperation::Encode => Ok(vae.encode(backend, input, context)?),
            VaeOperation::Decode => Ok(vae.decode(backend, input, context)?),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_video_vae_with_context(
        &self,
        identity: &AssetIdentity,
        model_store: &mut ModelStore,
        target: &VaeExecutionTarget,
        patch: PatchGraphIdentity,
        boundary: VaeBoundary,
        decode_clamp: [f32; 2],
        backend: &CpuBackend,
        authorization: &AuthorizedCapabilities,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeVae, AssetVideoVaeLoadError> {
        require_asset_authorization(
            authorization,
            &self.roots.profile_id,
            identity.namespace,
            AssetAction::Read,
        )?;
        validate_native_vae_backend_target(backend, target)?;
        let loaded = self.load_model(identity, model_store, authorization, context.cancellation)?;
        let key = self.roots.artifact_key(identity)?;
        let artifact = self
            .artifact_index
            .record(&key)
            .ok_or_else(|| AssetError::UnknownAsset(identity.clone()))?;
        let registry = VaeArchitectureRegistry::checked()?;
        let (family_registry, latent_registry) = VaeArchitectureRegistry::canonical_targets()?;
        let selection = registry.select_loaded(
            model_store,
            &loaded,
            target,
            &family_registry,
            &latent_registry,
            context.cancellation,
        )?;
        selection.ensure_native_builder_available()?;
        let latent_definition = latent_registry.get(target.latent_format()).ok_or_else(|| {
            VaeArchitectureError::UnknownLatentFormat(
                target.latent_format().identifier().to_owned(),
            )
        })?;
        let descriptor = VaeDescriptor::checked_selection(
            artifact,
            &selection,
            target,
            &family_registry,
            &latent_registry,
            patch,
            boundary,
            decode_clamp,
            context.cancellation,
        )?;
        Ok(load_video_vae_from_model_store_with_context(
            backend,
            model_store,
            &self.artifact_index,
            loaded,
            descriptor,
            latent_definition,
            context,
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_and_execute_video_vae_with_context(
        &self,
        identity: &AssetIdentity,
        model_store: &mut ModelStore,
        target: &VaeExecutionTarget,
        patch: PatchGraphIdentity,
        boundary: VaeBoundary,
        decode_clamp: [f32; 2],
        operation: VaeOperation,
        input: &Tensor,
        backend: &CpuBackend,
        authorization: &AuthorizedCapabilities,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, AssetVideoVaeLoadError> {
        let vae = self.load_video_vae_with_context(
            identity,
            model_store,
            target,
            patch,
            boundary,
            decode_clamp,
            backend,
            authorization,
            context,
        )?;
        match operation {
            VaeOperation::Encode => Ok(vae.encode(backend, input, context)?),
            VaeOperation::Decode => Ok(vae.decode(backend, input, context)?),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_audio_vae_with_context(
        &self,
        identity: &AssetIdentity,
        model_store: &mut ModelStore,
        target: &VaeExecutionTarget,
        patch: PatchGraphIdentity,
        boundary: VaeBoundary,
        decode_clamp: [f32; 2],
        backend: &CpuBackend,
        authorization: &AuthorizedCapabilities,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeVae, AssetAudioVaeLoadError> {
        require_asset_authorization(
            authorization,
            &self.roots.profile_id,
            identity.namespace,
            AssetAction::Read,
        )?;
        validate_native_vae_backend_target(backend, target)?;
        let loaded = self.load_model(identity, model_store, authorization, context.cancellation)?;
        let key = self.roots.artifact_key(identity)?;
        let artifact = self
            .artifact_index
            .record(&key)
            .ok_or_else(|| AssetError::UnknownAsset(identity.clone()))?;
        let registry = VaeArchitectureRegistry::checked()?;
        let (family_registry, latent_registry) = VaeArchitectureRegistry::canonical_targets()?;
        let selection = registry.select_loaded(
            model_store,
            &loaded,
            target,
            &family_registry,
            &latent_registry,
            context.cancellation,
        )?;
        selection.ensure_native_builder_available()?;
        let latent_definition = latent_registry.get(target.latent_format()).ok_or_else(|| {
            VaeArchitectureError::UnknownLatentFormat(
                target.latent_format().identifier().to_owned(),
            )
        })?;
        let descriptor = VaeDescriptor::checked_selection(
            artifact,
            &selection,
            target,
            &family_registry,
            &latent_registry,
            patch,
            boundary,
            decode_clamp,
            context.cancellation,
        )?;
        Ok(load_audio_vae_from_model_store_with_context(
            backend,
            model_store,
            &self.artifact_index,
            loaded,
            descriptor,
            latent_definition,
            context,
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_and_execute_audio_vae_with_context(
        &self,
        identity: &AssetIdentity,
        model_store: &mut ModelStore,
        target: &VaeExecutionTarget,
        patch: PatchGraphIdentity,
        boundary: VaeBoundary,
        decode_clamp: [f32; 2],
        operation: VaeOperation,
        input: &Tensor,
        backend: &CpuBackend,
        authorization: &AuthorizedCapabilities,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, AssetAudioVaeLoadError> {
        let vae = self.load_audio_vae_with_context(
            identity,
            model_store,
            target,
            patch,
            boundary,
            decode_clamp,
            backend,
            authorization,
            context,
        )?;
        match operation {
            VaeOperation::Encode => Ok(vae.encode(backend, input, context)?),
            VaeOperation::Decode => Ok(vae.decode(backend, input, context)?),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_structured_vae_with_context(
        &self,
        identity: &AssetIdentity,
        model_store: &mut ModelStore,
        target: &VaeExecutionTarget,
        patch: PatchGraphIdentity,
        boundary: VaeBoundary,
        decode_clamp: [f32; 2],
        backend: &CpuBackend,
        authorization: &AuthorizedCapabilities,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeStructuredVae, AssetStructuredVaeLoadError> {
        require_asset_authorization(
            authorization,
            &self.roots.profile_id,
            identity.namespace,
            AssetAction::Read,
        )?;
        validate_native_vae_backend_target(backend, target)?;
        let loaded = self.load_model(identity, model_store, authorization, context.cancellation)?;
        let key = self.roots.artifact_key(identity)?;
        let artifact = self
            .artifact_index
            .record(&key)
            .ok_or_else(|| AssetError::UnknownAsset(identity.clone()))?;
        let registry = VaeArchitectureRegistry::checked()?;
        let (family_registry, latent_registry) = VaeArchitectureRegistry::canonical_targets()?;
        let selection = registry.select_loaded(
            model_store,
            &loaded,
            target,
            &family_registry,
            &latent_registry,
            context.cancellation,
        )?;
        selection.ensure_native_builder_available()?;
        let latent_definition = latent_registry.get(target.latent_format()).ok_or_else(|| {
            VaeArchitectureError::UnknownLatentFormat(
                target.latent_format().identifier().to_owned(),
            )
        })?;
        let descriptor = VaeDescriptor::checked_selection(
            artifact,
            &selection,
            target,
            &family_registry,
            &latent_registry,
            patch,
            boundary,
            decode_clamp,
            context.cancellation,
        )?;
        Ok(load_structured_vae_from_model_store_with_context(
            backend,
            model_store,
            &self.artifact_index,
            loaded,
            descriptor,
            latent_definition,
            context,
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_and_decode_structured_vae_with_context(
        &self,
        identity: &AssetIdentity,
        model_store: &mut ModelStore,
        target: &VaeExecutionTarget,
        patch: PatchGraphIdentity,
        boundary: VaeBoundary,
        decode_clamp: [f32; 2],
        latent: &Tensor,
        request: &VaeStructuredDecodeRequest,
        backend: &CpuBackend,
        authorization: &AuthorizedCapabilities,
        context: &ExecutionContext<'_>,
    ) -> Result<VaeStructuredResult, AssetStructuredVaeLoadError> {
        let vae = self.load_structured_vae_with_context(
            identity,
            model_store,
            target,
            patch,
            boundary,
            decode_clamp,
            backend,
            authorization,
            context,
        )?;
        Ok(vae.decode(backend, latent, request, context)?)
    }

    pub fn record(&self, identity: &AssetIdentity) -> Option<AssetRecord> {
        let key = self.roots.artifact_key(identity).ok()?;
        let artifact = self.artifact_index.record(&key)?;
        let enrichment = self.enrichments.get(identity)?;
        Some(project_asset_record(artifact, enrichment))
    }

    fn list_source_asset_names(
        &mut self,
        folder_category: &str,
        authorization: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<Vec<String>, AssetError> {
        let spec = source_asset_folder_spec(folder_category).ok_or_else(|| {
            AssetError::UnsupportedSourceFolderCategory(folder_category.to_owned())
        })?;
        self.scan_namespaces(&[spec.namespace], authorization, cancellation)?;
        check_cancelled(cancellation)?;
        let mut names = BTreeSet::new();
        for root_id in self.roots.root_ids_in_resolution_order(spec.namespace) {
            for prefix in spec.prefixes {
                let prefix = Path::new(prefix);
                for artifact in self.artifact_index.records().filter(|artifact| {
                    artifact.key.root_id == root_id
                        && artifact.availability == CanonicalArtifactAvailability::Present
                }) {
                    let Ok(name) = artifact.key.relative_path.strip_prefix(prefix) else {
                        continue;
                    };
                    let Some(name) = name.to_str() else {
                        continue;
                    };
                    if !name.is_empty()
                        && source_asset_name_has_supported_extension(name, spec.extensions)
                    {
                        names.insert(name.replace(std::path::MAIN_SEPARATOR, "/"));
                    }
                }
            }
        }
        Ok(names.into_iter().collect())
    }

    fn resolve_source_asset_names(
        &mut self,
        folder_category: &str,
        names: &[String],
        authorization: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<Vec<SourceAssetResolutionRecord>, AssetError> {
        let spec = source_asset_folder_spec(folder_category).ok_or_else(|| {
            AssetError::UnsupportedSourceFolderCategory(folder_category.to_owned())
        })?;
        self.scan_namespaces(&[spec.namespace], authorization, cancellation)?;
        let mut resolved = Vec::new();
        resolved
            .try_reserve_exact(names.len())
            .map_err(|_| AssetError::AllocationFailed)?;
        for name in names {
            check_cancelled(cancellation)?;
            if !source_asset_name_has_supported_extension(name, spec.extensions) {
                return Err(AssetError::InvalidSourceAssetName(name.clone()));
            }
            let mut selected = None;
            'roots: for root_id in self.roots.root_ids_in_resolution_order(spec.namespace) {
                for prefix in spec.prefixes {
                    let identity = self.roots.identity_in_root(
                        spec.namespace,
                        root_id,
                        Path::new(prefix).join(name),
                    )?;
                    let Some(record) = self.record(&identity) else {
                        continue;
                    };
                    if record.availability != AssetAvailability::Missing {
                        selected = Some(SourceAssetResolutionRecord {
                            name: name.clone(),
                            identity,
                            byte_length: record.byte_size,
                            sha256: record.sha256,
                        });
                        break 'roots;
                    }
                }
            }
            resolved.push(selected.ok_or_else(|| AssetError::UnknownSourceAsset {
                folder_category: folder_category.to_owned(),
                name: name.clone(),
            })?);
        }
        check_cancelled(cancellation)?;
        Ok(resolved)
    }

    pub fn write_exact(
        &mut self,
        identity: &AssetIdentity,
        bytes: &[u8],
        tags: BTreeSet<String>,
        policy: AssetCollisionPolicy,
        authorization: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<AssetRecord, AssetError> {
        require_asset_authorization(
            authorization,
            &self.roots.profile_id,
            identity.namespace,
            AssetAction::Write,
        )?;
        self.validate_write(identity, bytes, &tags, cancellation)?;
        if identity == &self.state_identity {
            return Err(AssetError::ReservedAsset(identity.clone()));
        }
        self.commit_asset_write(identity, bytes, tags, Some(policy), cancellation)
    }

    pub fn upload(
        &mut self,
        request: UploadRequest,
        authorization: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<UploadResult, AssetError> {
        require_asset_authorization(
            authorization,
            &self.roots.profile_id,
            request.namespace,
            AssetAction::Write,
        )?;
        check_cancelled(cancellation)?;
        let filename = normalize_filename(&request.filename)?;
        let subfolder = normalize_optional_relative_path(&request.subfolder)?;
        let requested_identity = self
            .roots
            .identity(request.namespace, subfolder.join(&filename))?;
        self.validate_write(
            &requested_identity,
            &request.bytes,
            &request.tags,
            cancellation,
        )?;
        let requested = self.roots.resolve_for_create(&requested_identity)?;
        self.refresh_namespaces_internal(&[request.namespace], cancellation)?;
        let mut duplicate = false;
        let mut collision_index = None;
        let (identity, _target) = if self.roots.contained_exists(&requested_identity)?
            && !request.overwrite
        {
            let key = self.roots.artifact_key(&requested_identity)?;
            let existing = self
                .artifact_index
                .record(&key)
                .ok_or_else(|| AssetError::UnknownAsset(requested_identity.clone()))?;
            if existing.sha256 == sha256(&request.bytes) {
                duplicate = true;
                (requested_identity, requested)
            } else {
                let (identity, path, index) =
                    next_upload_collision(&self.roots, request.namespace, &subfolder, &filename)?;
                collision_index = Some(index);
                (identity, path)
            }
        } else {
            (requested_identity, requested)
        };
        let policy = if duplicate {
            None
        } else if request.overwrite {
            Some(AssetCollisionPolicy::Replace)
        } else {
            Some(AssetCollisionPolicy::Reject)
        };
        let record = self.commit_asset_write(
            &identity,
            &request.bytes,
            request.tags,
            policy,
            cancellation,
        )?;
        Ok(UploadResult {
            record,
            duplicate,
            collision_index,
        })
    }

    fn validate_write(
        &self,
        identity: &AssetIdentity,
        bytes: &[u8],
        tags: &BTreeSet<String>,
        cancellation: &CancellationToken,
    ) -> Result<(), AssetError> {
        check_cancelled(cancellation)?;
        self.roots.artifact_key(identity)?;
        if self.project_asset_is_immutable(identity)? {
            return Err(AssetError::ReservedAsset(identity.clone()));
        }
        let byte_size = u64::try_from(bytes.len()).map_err(|_| AssetError::TooLarge {
            actual: u64::MAX,
            limit: self.max_asset_bytes,
        })?;
        if byte_size > self.max_asset_bytes {
            return Err(AssetError::TooLarge {
                actual: byte_size,
                limit: self.max_asset_bytes,
            });
        }
        validate_tags(tags)
    }

    fn commit_asset_write(
        &mut self,
        identity: &AssetIdentity,
        bytes: &[u8],
        tags: BTreeSet<String>,
        policy: Option<AssetCollisionPolicy>,
        cancellation: &CancellationToken,
    ) -> Result<AssetRecord, AssetError> {
        self.refresh_namespaces_internal(&[identity.namespace], cancellation)?;
        let previous_index = self.artifact_index.clone();
        let previous_enrichments = self.enrichments.clone();
        let maximum_bytes = usize::try_from(self.max_asset_bytes).unwrap_or(usize::MAX);
        let previous_bytes = self.roots.read_contained(identity, maximum_bytes)?;
        let wrote_file = policy.is_some();
        let result = (|| {
            if let Some(policy) = policy {
                self.roots.write_contained(identity, bytes, policy)?;
            }
            check_cancelled(cancellation)?;
            self.refresh_namespaces_internal(&[identity.namespace], cancellation)?;
            let enrichment = self
                .enrichments
                .get_mut(identity)
                .ok_or_else(|| AssetError::UnknownAsset(identity.clone()))?;
            enrichment.tags = tags;
            let record = self
                .record(identity)
                .ok_or_else(|| AssetError::UnknownAsset(identity.clone()))?;
            self.persist_state()?;
            Ok(record)
        })();
        match result {
            Ok(record) => Ok(record),
            Err(primary) => {
                let mut rollback_failures = Vec::new();
                let restore_file = wrote_file && !matches!(&primary, AssetError::AlreadyExists(_));
                if restore_file {
                    let restored = match previous_bytes {
                        Some(previous_bytes) => self.roots.write_contained(
                            identity,
                            &previous_bytes,
                            AssetCollisionPolicy::Replace,
                        ),
                        None => self.roots.remove_contained(identity),
                    };
                    if let Err(error) = restored {
                        rollback_failures.push(error.to_string());
                    }
                }
                self.artifact_index = previous_index;
                self.enrichments = previous_enrichments;
                if restore_file
                    && rollback_failures.is_empty()
                    && let Err(error) = self.refresh_namespaces_internal(
                        &[identity.namespace],
                        &CancellationToken::default(),
                    )
                {
                    rollback_failures.push(error.to_string());
                }
                if let Err(error) = self.persist_state() {
                    rollback_failures.push(error.to_string());
                }
                if rollback_failures.is_empty() {
                    Err(primary)
                } else {
                    Err(AssetError::Rollback {
                        primary: primary.to_string(),
                        rollback: rollback_failures.join("; "),
                    })
                }
            }
        }
    }

    pub fn view(
        &self,
        request: &AssetViewRequest,
        authorization: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<AssetView, AssetError> {
        require_asset_authorization(
            authorization,
            &self.roots.profile_id,
            request.identity.namespace,
            AssetAction::Read,
        )?;
        check_cancelled(cancellation)?;
        let key = self.roots.artifact_key(&request.identity)?;
        let artifact = self
            .artifact_index
            .record(&key)
            .cloned()
            .ok_or_else(|| AssetError::UnknownAsset(request.identity.clone()))?;
        let total_size = artifact.byte_size;
        let range = validate_range(request.range, total_size, self.max_range_bytes)?;
        let (start, length) = match range {
            Some(range) => (
                range.start,
                range
                    .end_inclusive
                    .checked_sub(range.start)
                    .and_then(|length| length.checked_add(1))
                    .ok_or(AssetError::InvalidRange)?,
            ),
            None => {
                if total_size > self.max_asset_bytes {
                    return Err(AssetError::TooLarge {
                        actual: total_size,
                        limit: self.max_asset_bytes,
                    });
                }
                (0, total_size)
            }
        };
        let length = usize::try_from(length).map_err(|_| AssetError::AllocationFailed)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| AssetError::AllocationFailed)?;
        let mut verified = self
            .artifact_index
            .open_verified(&key, cancellation)
            .map_err(|error| map_artifact_error_for_identity(error, &request.identity))?;
        verified
            .file_mut()
            .seek(SeekFrom::Start(start))
            .map_err(|error| AssetError::Io {
                path: verified.path().to_path_buf(),
                message: error.to_string(),
            })?;
        verified
            .file_mut()
            .take(u64::try_from(length).unwrap_or(u64::MAX))
            .read_to_end(&mut bytes)
            .map_err(|error| AssetError::Io {
                path: verified.path().to_path_buf(),
                message: error.to_string(),
            })?;
        check_cancelled(cancellation)?;
        verified
            .verify_unchanged()
            .map_err(|_| AssetError::ChangedDuringRead(request.identity.clone()))?;
        let filename = request.identity.filename().unwrap_or("asset");
        let detected = detect_content_type(&bytes, filename);
        let unsafe_inline = is_unsafe_inline_content_type(&detected);
        let content_type = if unsafe_inline {
            "application/octet-stream".to_owned()
        } else {
            detected
        };
        let disposition = if request.download || unsafe_inline {
            "attachment"
        } else {
            "inline"
        };
        Ok(AssetView {
            bytes,
            content_type,
            content_disposition: format!(
                "{disposition}; filename=\"{}\"",
                escape_disposition_filename(filename)
            ),
            total_size,
            range,
        })
    }

    pub fn scan(
        &mut self,
        authorization: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<Vec<AssetChange>, AssetError> {
        let namespaces = self.roots.namespaces().collect::<Vec<_>>();
        self.scan_namespaces(&namespaces, authorization, cancellation)
    }

    pub fn scan_namespaces(
        &mut self,
        namespaces: &[AssetNamespace],
        authorization: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<Vec<AssetChange>, AssetError> {
        let selected_namespaces = namespaces.iter().copied().collect::<BTreeSet<_>>();
        for &namespace in &selected_namespaces {
            self.roots.root_path(namespace)?;
            require_asset_authorization(
                authorization,
                &self.roots.profile_id,
                namespace,
                AssetAction::Read,
            )?;
        }
        let changes = self.refresh_namespaces_internal(
            &selected_namespaces.into_iter().collect::<Vec<_>>(),
            cancellation,
        )?;
        self.persist_state()?;
        Ok(changes)
    }

    fn refresh_namespaces_internal(
        &mut self,
        namespaces: &[AssetNamespace],
        cancellation: &CancellationToken,
    ) -> Result<Vec<AssetChange>, AssetError> {
        let selected = namespaces.iter().copied().collect::<BTreeSet<_>>();
        let root_ids = self
            .roots
            .roots
            .keys()
            .filter(|(namespace, _)| selected.contains(namespace))
            .map(|(_, root_id)| root_id.clone())
            .collect::<Vec<_>>();
        let canonical_changes = self
            .artifact_index
            .refresh_selected(root_ids, cancellation)
            .map_err(map_artifact_error)?;
        let mut changes = Vec::with_capacity(canonical_changes.len());
        for change in canonical_changes {
            let identity = self.roots.identity_for_key(&change.key)?;
            let kind = match change.kind {
                CanonicalArtifactChangeKind::Added => AssetChangeKind::Added,
                CanonicalArtifactChangeKind::Modified => AssetChangeKind::Modified,
                CanonicalArtifactChangeKind::Missing => AssetChangeKind::Missing,
                CanonicalArtifactChangeKind::Restored => AssetChangeKind::Restored,
            };
            if matches!(
                change.kind,
                CanonicalArtifactChangeKind::Added
                    | CanonicalArtifactChangeKind::Modified
                    | CanonicalArtifactChangeKind::Restored
            ) {
                let tags = self
                    .enrichments
                    .get(&identity)
                    .map(|enrichment| enrichment.tags.clone())
                    .unwrap_or_default();
                let mut enrichment = inspect_enrichment(
                    &self.artifact_index,
                    &change.key,
                    &identity,
                    &self.metadata_limits,
                    self.max_asset_bytes,
                    cancellation,
                )?;
                enrichment.tags = tags;
                self.enrichments.insert(identity.clone(), enrichment);
            }
            changes.push(AssetChange { identity, kind });
        }
        changes.sort_by(|left, right| left.identity.cmp(&right.identity));
        Ok(changes)
    }

    pub fn rename(
        &mut self,
        identity: &AssetIdentity,
        new_filename: &str,
        new_subfolder: &Path,
        authorization: &AuthorizedCapabilities,
    ) -> Result<AssetRecord, AssetError> {
        require_asset_authorization(
            authorization,
            &self.roots.profile_id,
            identity.namespace,
            AssetAction::Rename,
        )?;
        if self.project_asset_is_immutable(identity)? {
            return Err(AssetError::ReservedAsset(identity.clone()));
        }
        let previous = self
            .record(identity)
            .ok_or_else(|| AssetError::UnknownAsset(identity.clone()))?;
        let key = self.roots.artifact_key(identity)?;
        self.artifact_index
            .open_verified(&key, &CancellationToken::default())
            .map_err(|error| map_artifact_error_for_identity(error, identity))?;
        let filename = normalize_filename(new_filename)?;
        let subfolder = normalize_optional_relative_path(new_subfolder)?;
        let new_identity = self
            .roots
            .identity(identity.namespace, subfolder.join(filename))?;
        if self.project_asset_is_immutable(&new_identity)? {
            return Err(AssetError::ReservedAsset(new_identity));
        }
        if self.roots.contained_exists(&new_identity)? {
            return Err(AssetError::AlreadyExists(
                self.roots.resolve_for_create(&new_identity)?,
            ));
        }
        self.roots.move_contained(
            identity,
            &new_identity,
            AssetCollisionPolicy::Reject,
            &previous.sha256,
            previous.byte_size,
            &CancellationToken::default(),
        )?;
        self.refresh_namespaces_internal(&[identity.namespace], &CancellationToken::default())?;
        let enrichment = self
            .enrichments
            .get_mut(&new_identity)
            .ok_or_else(|| AssetError::UnknownAsset(new_identity.clone()))?;
        enrichment.tags = previous.tags;
        let record = self
            .record(&new_identity)
            .ok_or_else(|| AssetError::UnknownAsset(new_identity.clone()))?;
        self.persist_state()?;
        Ok(record)
    }

    pub fn set_tags(
        &mut self,
        identity: &AssetIdentity,
        tags: BTreeSet<String>,
        authorization: &AuthorizedCapabilities,
    ) -> Result<AssetRecord, AssetError> {
        require_asset_authorization(
            authorization,
            &self.roots.profile_id,
            identity.namespace,
            AssetAction::Tag,
        )?;
        validate_tags(&tags)?;
        let enrichment = self
            .enrichments
            .get_mut(identity)
            .ok_or_else(|| AssetError::UnknownAsset(identity.clone()))?;
        enrichment.tags = tags;
        let record = self
            .record(identity)
            .ok_or_else(|| AssetError::UnknownAsset(identity.clone()))?;
        self.persist_state()?;
        Ok(record)
    }

    pub fn delete(
        &mut self,
        identity: &AssetIdentity,
        authorization: &AuthorizedCapabilities,
    ) -> Result<AssetRecord, AssetError> {
        require_asset_authorization(
            authorization,
            &self.roots.profile_id,
            identity.namespace,
            AssetAction::Delete,
        )?;
        if self.project_asset_is_immutable(identity)? {
            return Err(AssetError::ReservedAsset(identity.clone()));
        }
        if self.record(identity).is_none() {
            return Err(AssetError::UnknownAsset(identity.clone()));
        }
        let key = self.roots.artifact_key(identity)?;
        self.artifact_index
            .open_verified(&key, &CancellationToken::default())
            .map_err(|error| map_artifact_error_for_identity(error, identity))?;
        self.roots.remove_contained(identity)?;
        self.refresh_namespaces_internal(&[identity.namespace], &CancellationToken::default())?;
        let record = self
            .record(identity)
            .ok_or_else(|| AssetError::UnknownAsset(identity.clone()))?;
        self.persist_state()?;
        Ok(record)
    }

    pub fn register_removed_output(
        &mut self,
        identity: &AssetIdentity,
        authorization: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<AssetRecord, AssetError> {
        require_asset_authorization(
            authorization,
            &self.roots.profile_id,
            identity.namespace,
            AssetAction::Delete,
        )?;
        if !matches!(
            identity.namespace,
            AssetNamespace::Output | AssetNamespace::Temporary
        ) {
            return Err(AssetError::UnsupportedNamespace(identity.namespace));
        }
        let previous_index = self.artifact_index.clone();
        let previous_enrichments = self.enrichments.clone();
        let result = (|| {
            check_cancelled(cancellation)?;
            self.refresh_namespaces_internal(&[identity.namespace], cancellation)?;
            let record = self
                .record(identity)
                .ok_or_else(|| AssetError::UnknownAsset(identity.clone()))?;
            if record.availability != AssetAvailability::Missing {
                return Err(AssetError::StillPresent(identity.clone()));
            }
            self.persist_state()?;
            Ok(record)
        })();
        match result {
            Ok(record) => Ok(record),
            Err(primary) => {
                self.artifact_index = previous_index;
                self.enrichments = previous_enrichments;
                if let Err(rollback) = self.persist_state() {
                    return Err(AssetError::Rollback {
                        primary: primary.to_string(),
                        rollback: rollback.to_string(),
                    });
                }
                Err(primary)
            }
        }
    }

    pub fn list_authorized(
        &self,
        query: &AssetQuery,
        authorization: &AuthorizedCapabilities,
    ) -> Result<AssetPage, AssetError> {
        if query.limit == 0 || query.limit > MAX_PAGE_SIZE {
            return Err(AssetError::InvalidPageSize(query.limit));
        }
        let authorized_namespaces = self.authorized_namespaces(authorization, AssetAction::Read)?;
        if let Some(namespace) = query.namespace
            && !authorized_namespaces.contains(&namespace)
        {
            return Err(AssetError::PermissionDenied {
                namespace,
                action: AssetAction::Read,
            });
        }
        let text = query.text.as_ref().map(|text| text.to_ascii_lowercase());
        let matches = self
            .all_records()?
            .into_iter()
            .filter(|record| {
                authorized_namespaces.contains(&record.identity.namespace)
                    && query
                        .namespace
                        .is_none_or(|namespace| record.identity.namespace == namespace)
                    && query
                        .availability
                        .as_ref()
                        .is_none_or(|availability| &record.availability == availability)
                    && query
                        .required_tags
                        .iter()
                        .all(|tag| record.tags.contains(tag))
                    && text.as_ref().is_none_or(|text| {
                        record
                            .identity
                            .relative_path
                            .to_string_lossy()
                            .to_ascii_lowercase()
                            .contains(text)
                    })
            })
            .collect::<Vec<_>>();
        let total = matches.len();
        let records = matches
            .into_iter()
            .skip(query.offset)
            .take(query.limit)
            .collect::<Vec<_>>();
        let consumed = query.offset.saturating_add(records.len());
        Ok(AssetPage {
            records,
            total,
            next_offset: (consumed < total).then_some(consumed),
        })
    }

    /// Projects host paths only for the developer-only ComfyUI compatibility route.
    pub fn authorized_compatibility_folder_paths(
        &self,
        authorization: &AuthorizedCapabilities,
    ) -> Result<BTreeMap<AssetNamespace, PathBuf>, AssetError> {
        let namespaces = self.authorized_namespaces(authorization, AssetAction::Read)?;
        namespaces
            .into_iter()
            .map(|namespace| {
                self.roots
                    .root_path(namespace)
                    .map(|path| (namespace, path.to_path_buf()))
            })
            .collect()
    }

    fn authorized_namespaces(
        &self,
        authorization: &AuthorizedCapabilities,
        action: AssetAction,
    ) -> Result<BTreeSet<AssetNamespace>, AssetError> {
        if authorization.profile_id() != self.roots.profile_id {
            return Err(AssetError::ProfileMismatch {
                expected: self.roots.profile_id.clone(),
                actual: authorization.profile_id().to_owned(),
            });
        }
        Ok(self
            .roots
            .namespaces()
            .filter(|namespace| {
                authorization
                    .require(&Capability::Asset {
                        namespace: namespace.locator_type().to_owned(),
                        action: action.into(),
                    })
                    .is_ok()
            })
            .collect())
    }

    pub fn register_committed_output(
        &mut self,
        identity: AssetIdentity,
        authorization: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<AssetRecord, AssetError> {
        self.register_committed_outputs([identity], authorization, cancellation)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                AssetError::InvalidIndex(
                    "single committed-output registration returned no record".to_owned(),
                )
            })
    }

    pub fn register_committed_outputs(
        &mut self,
        identities: impl IntoIterator<Item = AssetIdentity>,
        authorization: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<Vec<AssetRecord>, AssetError> {
        let identities = identities.into_iter().collect::<Vec<_>>();
        let mut namespaces = BTreeSet::new();
        for identity in &identities {
            require_asset_authorization(
                authorization,
                &self.roots.profile_id,
                identity.namespace,
                AssetAction::Write,
            )?;
            self.roots.resolve_existing(identity)?;
            namespaces.insert(identity.namespace);
        }
        let previous_index = self.artifact_index.clone();
        let previous_enrichments = self.enrichments.clone();
        let result: Result<Vec<AssetRecord>, AssetError> = (|| {
            check_cancelled(cancellation)?;
            self.refresh_namespaces_internal(
                &namespaces.into_iter().collect::<Vec<_>>(),
                cancellation,
            )?;
            let records = identities
                .into_iter()
                .map(|identity| {
                    self.record(&identity)
                        .ok_or(AssetError::UnknownAsset(identity))
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.persist_state()?;
            Ok(records)
        })();
        match result {
            Ok(records) => Ok(records),
            Err(primary) => {
                self.artifact_index = previous_index;
                self.enrichments = previous_enrichments;
                if let Err(rollback) = self.persist_state() {
                    return Err(AssetError::Rollback {
                        primary: primary.to_string(),
                        rollback: rollback.to_string(),
                    });
                }
                Err(primary)
            }
        }
    }

    fn all_records(&self) -> Result<Vec<AssetRecord>, AssetError> {
        self.artifact_index
            .records()
            .map(|artifact| {
                let identity = self.roots.identity_for_key(&artifact.key)?;
                let enrichment = self.enrichments.get(&identity).ok_or_else(|| {
                    AssetError::InvalidIndex(format!(
                        "artifact {:?} has no asset enrichment",
                        artifact.key
                    ))
                })?;
                Ok(project_asset_record(artifact, enrichment))
            })
            .collect()
    }

    fn persist_state(&self) -> Result<(), AssetError> {
        if self.enrichments.len() > self.max_records {
            return Err(AssetError::IndexRecordLimit {
                actual: self.enrichments.len(),
                limit: self.max_records,
            });
        }
        let canonical_snapshot = self.artifact_index.snapshot().map_err(map_artifact_error)?;
        let snapshot = AssetServiceSnapshot {
            schema_version: ASSET_SERVICE_SCHEMA_VERSION,
            profile_id: self.roots.profile_id.clone(),
            canonical_artifact_index: String::from_utf8(canonical_snapshot)
                .map_err(|error| AssetError::IndexEncode(error.to_string()))?,
            enrichments: self.enrichments.values().cloned().collect(),
        };
        let bytes = serde_json::to_vec_pretty(&snapshot)
            .map_err(|error| AssetError::IndexEncode(error.to_string()))?;
        if bytes.len() > self.max_index_bytes {
            return Err(AssetError::IndexTooLarge {
                actual: bytes.len(),
                limit: self.max_index_bytes,
            });
        }
        self.roots
            .artifact_root_for_identity(&self.state_identity)?
            .write_private_file(&self.state_identity.relative_path, &bytes)
            .map_err(map_artifact_error)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct AssetServiceSnapshot {
    schema_version: u32,
    profile_id: String,
    canonical_artifact_index: String,
    enrichments: Vec<AssetEnrichment>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum AssetError {
    #[error("asset profile {0:?} is invalid")]
    InvalidProfile(String),
    #[error("asset identity belongs to profile {actual:?}, expected {expected:?}")]
    ProfileMismatch { expected: String, actual: String },
    #[error("asset namespace {0:?} is duplicated")]
    DuplicateNamespace(AssetNamespace),
    #[error("asset namespace {0:?} is invalid")]
    InvalidNamespace(String),
    #[error("asset reference is invalid: {0}")]
    InvalidReference(String),
    #[error("asset root {0} overlaps another typed root")]
    OverlappingRoots(PathBuf),
    #[error("asset namespace {0:?} is not configured")]
    UnknownNamespace(AssetNamespace),
    #[error("asset permission {action:?} is not granted for {namespace:?}")]
    PermissionDenied {
        namespace: AssetNamespace,
        action: AssetAction,
    },
    #[error("asset path {path} is unsafe: {reason}")]
    UnsafePath { path: PathBuf, reason: String },
    #[error("asset filename {0:?} is invalid")]
    InvalidFilename(String),
    #[error("source asset folder category {0:?} is unsupported")]
    UnsupportedSourceFolderCategory(String),
    #[error("source asset name {0:?} is invalid for its folder category")]
    InvalidSourceAssetName(String),
    #[error("source asset {name:?} is not indexed in folder category {folder_category:?}")]
    UnknownSourceAsset {
        folder_category: String,
        name: String,
    },
    #[error("asset {0:?} is missing")]
    Missing(AssetIdentity),
    #[error("asset {0:?} is not indexed")]
    UnknownAsset(AssetIdentity),
    #[error("asset path is reserved for canonical service state: {0:?}")]
    ReservedAsset(AssetIdentity),
    #[error("asset path already exists: {0}")]
    AlreadyExists(PathBuf),
    #[error("asset I/O failed for {path}: {message}")]
    Io { path: PathBuf, message: String },
    #[error("asset contains {actual} bytes, exceeding the {limit}-byte limit")]
    TooLarge { actual: u64, limit: u64 },
    #[error("asset range is invalid")]
    InvalidRange,
    #[error("asset range contains {actual} bytes, exceeding the {limit}-byte limit")]
    RangeTooLarge { actual: u64, limit: u64 },
    #[error("asset changed while it was being read: {0:?}")]
    ChangedDuringRead(AssetIdentity),
    #[error("asset operation was cancelled")]
    Cancelled,
    #[error("asset allocation failed")]
    AllocationFailed,
    #[error("asset {0:?} is still present")]
    StillPresent(AssetIdentity),
    #[error("asset namespace {0:?} is not supported by this operation")]
    UnsupportedNamespace(AssetNamespace),
    #[error("asset tag {0:?} is invalid")]
    InvalidTag(String),
    #[error("asset page size {0} is invalid")]
    InvalidPageSize(usize),
    #[error("asset limits must be non-zero")]
    InvalidLimit,
    #[error("model loading requires the model namespace, not {0:?}")]
    ModelNamespaceRequired(AssetNamespace),
    #[error("asset index schema {actual} is unsupported; expected {expected}")]
    UnsupportedIndexSchema { expected: u32, actual: u32 },
    #[error("asset index decode failed: {0}")]
    IndexDecode(String),
    #[error("asset index encode failed: {0}")]
    IndexEncode(String),
    #[error("asset index contains {actual} bytes, exceeding the {limit}-byte limit")]
    IndexTooLarge { actual: usize, limit: usize },
    #[error("asset index contains {actual} records, exceeding the {limit}-record limit")]
    IndexRecordLimit { actual: usize, limit: usize },
    #[error("asset index is invalid: {0}")]
    InvalidIndex(String),
    #[error("asset operation failed ({primary}) and state rollback failed ({rollback})")]
    Rollback { primary: String, rollback: String },
    #[error(transparent)]
    Model(#[from] ModelStoreError),
}

#[derive(Debug, Error)]
pub enum AssetImageVaeLoadError {
    #[error(transparent)]
    Asset(#[from] AssetError),
    #[error(transparent)]
    Architecture(#[from] VaeArchitectureError),
    #[error(transparent)]
    Vae(#[from] VaeError),
    #[error(transparent)]
    ImageVae(Box<ImageVaeError>),
}

impl From<ImageVaeError> for AssetImageVaeLoadError {
    fn from(error: ImageVaeError) -> Self {
        Self::ImageVae(Box::new(error))
    }
}

#[derive(Debug, Error)]
pub enum AssetVideoVaeLoadError {
    #[error(transparent)]
    Asset(#[from] AssetError),
    #[error(transparent)]
    Architecture(#[from] VaeArchitectureError),
    #[error(transparent)]
    Vae(#[from] VaeError),
    #[error(transparent)]
    VideoVae(Box<VideoVaeError>),
}

impl From<VideoVaeError> for AssetVideoVaeLoadError {
    fn from(error: VideoVaeError) -> Self {
        Self::VideoVae(Box::new(error))
    }
}

#[derive(Debug, Error)]
pub enum AssetAudioVaeLoadError {
    #[error(transparent)]
    Asset(#[from] AssetError),
    #[error(transparent)]
    Architecture(#[from] VaeArchitectureError),
    #[error(transparent)]
    Vae(#[from] VaeError),
    #[error(transparent)]
    AudioVae(Box<AudioVaeError>),
}

impl From<AudioVaeError> for AssetAudioVaeLoadError {
    fn from(error: AudioVaeError) -> Self {
        Self::AudioVae(Box::new(error))
    }
}

#[derive(Debug, Error)]
pub enum AssetStructuredVaeLoadError {
    #[error(transparent)]
    Asset(#[from] AssetError),
    #[error(transparent)]
    Architecture(#[from] VaeArchitectureError),
    #[error(transparent)]
    Vae(#[from] VaeError),
    #[error(transparent)]
    StructuredVae(Box<StructuredVaeError>),
}

impl From<StructuredVaeError> for AssetStructuredVaeLoadError {
    fn from(error: StructuredVaeError) -> Self {
        Self::StructuredVae(Box::new(error))
    }
}

fn new_artifact_index(roots: &AssetRoots) -> Result<ArtifactIndex, AssetError> {
    let mut index = ArtifactIndex::default();
    for root in roots.artifact_roots() {
        index.add_root(root).map_err(map_artifact_error)?;
    }
    Ok(index)
}

fn read_asset_service_state(
    bytes: &[u8],
    roots: &AssetRoots,
    max_records: usize,
) -> Result<
    (
        ArtifactIndex,
        BTreeMap<AssetIdentity, AssetEnrichment>,
        Option<AssetStateRecovery>,
    ),
    AssetError,
> {
    let snapshot: AssetServiceSnapshot = serde_json::from_slice(bytes)
        .map_err(|error| AssetError::IndexDecode(error.to_string()))?;
    if snapshot.schema_version != ASSET_SERVICE_SCHEMA_VERSION {
        return Err(AssetError::UnsupportedIndexSchema {
            expected: ASSET_SERVICE_SCHEMA_VERSION,
            actual: snapshot.schema_version,
        });
    }
    if snapshot.profile_id != roots.profile_id {
        return Err(AssetError::InvalidIndex(
            "snapshot profile does not match the active typed roots".to_owned(),
        ));
    }
    if snapshot.enrichments.len() > max_records {
        return Err(AssetError::IndexRecordLimit {
            actual: snapshot.enrichments.len(),
            limit: max_records,
        });
    }
    let (artifact_index, reconciliation) = ArtifactIndex::reconcile_snapshot(
        snapshot.canonical_artifact_index.as_bytes(),
        roots.artifact_roots(),
    )
    .map_err(map_artifact_error)?;
    let enrichment_count = snapshot.enrichments.len();
    let enrichments = snapshot
        .enrichments
        .into_iter()
        .filter_map(|mut enrichment| {
            if enrichment.identity.profile_id != snapshot.profile_id {
                return Some(Err(AssetError::InvalidIndex(
                    "snapshot enrichment profile is inconsistent".to_owned(),
                )));
            }
            let snapshot_root_id = enrichment
                .identity
                .root_id
                .as_deref()
                .unwrap_or_else(|| enrichment.identity.namespace.locator_type());
            let current_root_id = reconciliation.current_root_id(snapshot_root_id)?;
            let normalized = roots.identity_in_root(
                enrichment.identity.namespace,
                current_root_id,
                enrichment.identity.relative_path.clone(),
            );
            let normalized = match normalized {
                Ok(normalized) => normalized,
                Err(error) => return Some(Err(error)),
            };
            if current_root_id == snapshot_root_id && normalized != enrichment.identity {
                return Some(Err(AssetError::InvalidIndex(
                    "snapshot identity is not canonical".to_owned(),
                )));
            }
            enrichment.identity = normalized;
            let key = match roots.artifact_key(&enrichment.identity) {
                Ok(key) => key,
                Err(error) => return Some(Err(error)),
            };
            if artifact_index.record(&key).is_none() {
                return Some(Err(AssetError::InvalidIndex(
                    "asset enrichment has no canonical artifact record".to_owned(),
                )));
            }
            if let Err(error) = validate_tags(&enrichment.tags) {
                return Some(Err(error));
            }
            Some(Ok((enrichment.identity.clone(), enrichment)))
        })
        .collect::<Result<BTreeMap<_, _>, AssetError>>()?;
    if enrichments.len() != artifact_index.records().count() {
        return Err(AssetError::InvalidIndex(
            "snapshot contains duplicate or incomplete asset enrichments".to_owned(),
        ));
    }
    let dropped_enrichments = enrichment_count.saturating_sub(enrichments.len());
    if dropped_enrichments != reconciliation.dropped_record_count() {
        return Err(AssetError::InvalidIndex(
            "snapshot root migration dropped inconsistent artifact state".to_owned(),
        ));
    }
    let recovery = reconciliation
        .changed()
        .then_some(AssetStateRecovery::Migrated {
            removed_roots: reconciliation.removed_root_count(),
            added_roots: reconciliation.added_root_count(),
            dropped_records: reconciliation.dropped_record_count(),
        });
    Ok((artifact_index, enrichments, recovery))
}

fn project_asset_record(
    artifact: &CanonicalArtifactRecord,
    enrichment: &AssetEnrichment,
) -> AssetRecord {
    AssetRecord {
        identity: enrichment.identity.clone(),
        sha256: artifact.sha256.clone(),
        byte_size: artifact.byte_size,
        modified_nanoseconds: artifact.modified_nanoseconds,
        content_type: enrichment.content_type.clone(),
        metadata_carrier: enrichment.metadata_carrier,
        metadata: enrichment.metadata.clone(),
        metadata_diagnostics: enrichment.metadata_diagnostics.clone(),
        tags: enrichment.tags.clone(),
        availability: artifact.availability.clone(),
    }
}

fn map_artifact_error_for_identity(
    error: ArtifactIndexError,
    identity: &AssetIdentity,
) -> AssetError {
    match error {
        ArtifactIndexError::Missing(_) => AssetError::Missing(identity.clone()),
        ArtifactIndexError::ChangedDuringScan(_) | ArtifactIndexError::ChangedSinceIndex(_) => {
            AssetError::ChangedDuringRead(identity.clone())
        }
        error => map_artifact_error(error),
    }
}

fn map_artifact_error(error: ArtifactIndexError) -> AssetError {
    match error {
        ArtifactIndexError::Cancelled => AssetError::Cancelled,
        ArtifactIndexError::Io { path, message } => AssetError::Io { path, message },
        ArtifactIndexError::UnsafePath { path, reason } => AssetError::UnsafePath { path, reason },
        ArtifactIndexError::AlreadyExists(path) => AssetError::AlreadyExists(path),
        ArtifactIndexError::SymbolicLink(path) => AssetError::UnsafePath {
            path,
            reason: "artifact path is a symbolic link".to_owned(),
        },
        ArtifactIndexError::NotDirectory(path) => AssetError::UnsafePath {
            path,
            reason: "artifact root is not a directory".to_owned(),
        },
        ArtifactIndexError::DuplicateCanonicalPath(path) => AssetError::OverlappingRoots(path),
        ArtifactIndexError::PortablePathCollision(path) => AssetError::UnsafePath {
            path,
            reason: "artifact path collides on a case-insensitive platform".to_owned(),
        },
        ArtifactIndexError::AllocationFailed(_) => AssetError::AllocationFailed,
        ArtifactIndexError::UnsupportedVersion(actual) => AssetError::UnsupportedIndexSchema {
            expected: sim_model::ARTIFACT_INDEX_VERSION,
            actual,
        },
        ArtifactIndexError::InvalidSnapshot(message) => AssetError::InvalidIndex(message),
        ArtifactIndexError::InvalidRootId(message)
        | ArtifactIndexError::InvalidNamespace(message)
        | ArtifactIndexError::DuplicateRootId(message)
        | ArtifactIndexError::PortableRootCollision(message)
        | ArtifactIndexError::UnknownRoot(message) => AssetError::InvalidIndex(message),
        ArtifactIndexError::Missing(key) => {
            AssetError::InvalidIndex(format!("canonical artifact is missing: {key:?}"))
        }
        ArtifactIndexError::ChangedDuringScan(path) => AssetError::Io {
            path,
            message: "canonical artifact changed during verification".to_owned(),
        },
        ArtifactIndexError::ChangedSinceIndex(key) => AssetError::InvalidIndex(format!(
            "canonical artifact changed since indexing: {key:?}"
        )),
        ArtifactIndexError::PartialMove {
            source_path,
            destination_path,
            message,
        } => AssetError::Io {
            path: source_path,
            message: format!(
                "contained move published {} but did not remove its source: {message}",
                destination_path.display()
            ),
        },
        ArtifactIndexError::Limit(error) => AssetError::InvalidIndex(error.to_string()),
    }
}

fn normalize_filename(filename: &str) -> Result<String, AssetError> {
    let filename = filename.trim();
    if filename.is_empty()
        || filename == "."
        || filename == ".."
        || filename.contains(['/', '\\', '\0'])
        || Path::new(filename)
            .file_name()
            .and_then(|name| name.to_str())
            != Some(filename)
    {
        return Err(AssetError::InvalidFilename(filename.to_owned()));
    }
    Ok(filename.to_owned())
}

pub(crate) fn normalize_optional_relative_path(path: &Path) -> Result<PathBuf, AssetError> {
    if path.as_os_str().is_empty() {
        return Ok(PathBuf::new());
    }
    ArtifactKey::new("asset-relative-path", path.to_path_buf())
        .map(|key| key.relative_path)
        .map_err(map_artifact_error)
}

fn next_upload_collision(
    roots: &AssetRoots,
    namespace: AssetNamespace,
    subfolder: &Path,
    filename: &str,
) -> Result<(AssetIdentity, PathBuf, u32), AssetError> {
    let path = Path::new(filename);
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(filename);
    let extension = path.extension().and_then(|extension| extension.to_str());
    for index in 1..=u32::MAX {
        let candidate = match extension {
            Some(extension) => subfolder.join(format!("{stem} ({index}).{extension}")),
            None => subfolder.join(format!("{stem} ({index})")),
        };
        let identity = roots.identity(namespace, candidate)?;
        let resolved = roots.resolve_for_create(&identity)?;
        if !roots.contained_exists(&identity)? {
            return Ok((identity, resolved, index));
        }
    }
    Err(AssetError::AlreadyExists(
        roots.root_path(namespace)?.join(subfolder).join(filename),
    ))
}

fn inspect_enrichment(
    artifact_index: &ArtifactIndex,
    key: &ArtifactKey,
    identity: &AssetIdentity,
    metadata_limits: &MetadataLimits,
    max_asset_bytes: u64,
    cancellation: &CancellationToken,
) -> Result<AssetEnrichment, AssetError> {
    let artifact = artifact_index
        .record(key)
        .ok_or_else(|| AssetError::UnknownAsset(identity.clone()))?;
    if artifact.byte_size > max_asset_bytes {
        return Err(AssetError::TooLarge {
            actual: artifact.byte_size,
            limit: max_asset_bytes,
        });
    }
    let prefix_length = usize::try_from(artifact.byte_size)
        .unwrap_or(usize::MAX)
        .min(metadata_limits.max_input_bytes);
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(prefix_length)
        .map_err(|_| AssetError::AllocationFailed)?;
    let mut verified = artifact_index
        .open_verified(key, cancellation)
        .map_err(|error| map_artifact_error_for_identity(error, identity))?;
    verified
        .file_mut()
        .take(u64::try_from(prefix_length).unwrap_or(u64::MAX))
        .read_to_end(&mut bytes)
        .map_err(|error| AssetError::Io {
            path: verified.path().to_path_buf(),
            message: error.to_string(),
        })?;
    verified
        .verify_unchanged()
        .map_err(|_| AssetError::ChangedDuringRead(identity.clone()))?;
    let filename = identity.filename().unwrap_or("asset");
    let metadata_document =
        MetadataDocument::parse(&bytes, Some(filename), None, metadata_limits.clone()).map_err(
            |error| AssetError::Io {
                path: verified.path().to_path_buf(),
                message: error.to_string(),
            },
        )?;
    let mut metadata_diagnostics = metadata_document
        .diagnostics()
        .iter()
        .map(|diagnostic| diagnostic.message.clone())
        .collect::<Vec<_>>();
    if artifact.byte_size > u64::try_from(bytes.len()).unwrap_or(u64::MAX) {
        metadata_diagnostics.push(format!(
            "metadata inspection was bounded to the first {} of {} bytes",
            bytes.len(),
            artifact.byte_size
        ));
    }
    Ok(AssetEnrichment {
        identity: identity.clone(),
        content_type: detect_content_type(&bytes, filename),
        metadata_carrier: metadata_document.carrier(),
        metadata: metadata_document.sim_metadata(),
        metadata_diagnostics,
        tags: BTreeSet::new(),
    })
}

fn validate_range(
    range: Option<AssetByteRange>,
    total_size: u64,
    max_range_bytes: u64,
) -> Result<Option<AssetByteRange>, AssetError> {
    let Some(range) = range else {
        return Ok(None);
    };
    if range.start > range.end_inclusive || range.end_inclusive >= total_size {
        return Err(AssetError::InvalidRange);
    }
    let length = range.end_inclusive - range.start + 1;
    if length > max_range_bytes {
        return Err(AssetError::RangeTooLarge {
            actual: length,
            limit: max_range_bytes,
        });
    }
    Ok(Some(range))
}

fn detect_content_type(bytes: &[u8], filename: &str) -> String {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return "image/png".to_owned();
    }
    if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        return "image/webp".to_owned();
    }
    if bytes.starts_with(b"fLaC") {
        return "audio/flac".to_owned();
    }
    if bytes.starts_with(b"OggS") {
        return "audio/ogg".to_owned();
    }
    if bytes.starts_with(b"glTF") {
        return "model/gltf-binary".to_owned();
    }
    if bytes.starts_with(b"<svg") || bytes.windows(4).take(512).any(|part| part == b"<svg") {
        return "image/svg+xml".to_owned();
    }
    let extension = Path::new(filename)
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match extension.as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "avif" => "image/avif",
        "mp3" => "audio/mpeg",
        "mp4" | "m4v" => "video/mp4",
        "mov" => "video/quicktime",
        "webm" => "video/webm",
        "json" => "application/json",
        "html" | "htm" => "text/html",
        "js" => "text/javascript",
        "css" => "text/css",
        "safetensors" | "latent" => "application/octet-stream",
        _ => "application/octet-stream",
    }
    .to_owned()
}

fn is_unsafe_inline_content_type(content_type: &str) -> bool {
    matches!(
        content_type,
        "text/html"
            | "application/xhtml+xml"
            | "text/javascript"
            | "application/javascript"
            | "text/css"
            | "image/svg+xml"
    )
}

fn escape_disposition_filename(filename: &str) -> String {
    filename
        .chars()
        .map(|character| match character {
            '"' | '\\' | '\r' | '\n' => '_',
            character => character,
        })
        .collect()
}

fn validate_tags(tags: &BTreeSet<String>) -> Result<(), AssetError> {
    for tag in tags {
        if tag.trim().is_empty() || tag.len() > 128 || tag.chars().any(char::is_control) {
            return Err(AssetError::InvalidTag(tag.clone()));
        }
    }
    Ok(())
}

pub(crate) fn check_cancelled(cancellation: &CancellationToken) -> Result<(), AssetError> {
    if cancellation.is_cancelled() {
        Err(AssetError::Cancelled)
    } else {
        Ok(())
    }
}

pub(crate) fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CapabilitySet, PermissionGrant, PermissionPolicy};
    use serde_json::{Value, json};
    use sim_tensor::DType;
    use sim_types::{AttemptId, NodeId};
    use std::{fs, io::Write};

    fn test_authorization(
        profile_id: &str,
        capabilities: CapabilitySet,
    ) -> Result<AuthorizedCapabilities, AssetError> {
        let grant = PermissionGrant::new(
            profile_id,
            "asset-test",
            capabilities.clone(),
            "asset-test-fixture",
        )
        .map_err(|error| AssetError::InvalidProfile(error.to_string()))?;
        PermissionPolicy::new(profile_id, [grant])
            .and_then(|policy| policy.authorize("asset-test", &capabilities))
            .map_err(|error| AssetError::InvalidProfile(error.to_string()))
    }

    fn denied_authorization() -> Result<AuthorizedCapabilities, AssetError> {
        test_authorization("profile", CapabilitySet::default())
    }

    fn service() -> Result<(tempfile::TempDir, AssetService, AuthorizedCapabilities), AssetError> {
        let directory = tempfile::tempdir().map_err(|error| AssetError::Io {
            path: PathBuf::from("temporary-directory"),
            message: error.to_string(),
        })?;
        let roots = [
            AssetNamespace::Input,
            AssetNamespace::Output,
            AssetNamespace::Temporary,
            AssetNamespace::Model,
            AssetNamespace::Plugin,
        ]
        .into_iter()
        .map(|namespace| {
            let path = directory.path().join(namespace.locator_type());
            fs::create_dir(&path).map_err(|error| AssetError::Io {
                path: path.clone(),
                message: error.to_string(),
            })?;
            Ok((namespace, path))
        })
        .collect::<Result<Vec<_>, AssetError>>()?;
        let roots = AssetRoots::new("profile", roots)?;
        let capabilities = CapabilitySet::new(roots.namespaces().flat_map(|namespace| {
            [
                AssetOperation::Read,
                AssetOperation::Write,
                AssetOperation::Rename,
                AssetOperation::Tag,
                AssetOperation::Delete,
            ]
            .into_iter()
            .map(move |action| Capability::Asset {
                namespace: namespace.locator_type().to_owned(),
                action,
            })
        }));
        let authorization = test_authorization("profile", capabilities)?;
        Ok((directory, AssetService::open(roots)?, authorization))
    }

    fn upload(filename: &str, bytes: &[u8]) -> UploadRequest {
        UploadRequest {
            namespace: AssetNamespace::Input,
            filename: filename.to_owned(),
            subfolder: PathBuf::from("day one/日本語"),
            bytes: bytes.to_vec(),
            overwrite: false,
            tags: BTreeSet::from(["input".to_owned()]),
        }
    }

    #[test]
    fn native_asset_resolver_seals_paths_and_rejects_foreign_or_cancelled_reads()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, mut service, authorization) = service()?;
        let cancellation = CancellationToken::default();
        let uploaded = service.upload(
            upload("sealed.bin", b"canonical-asset"),
            &authorization,
            &cancellation,
        )?;
        let registry =
            NativeAssetResolverRegistry::new(Arc::new(Mutex::new(service)), authorization);
        let attempt_id = AttemptId(Uuid::from_u128(0xa551));
        let node_identity = NativeNodeServiceIdentity::checked(
            Uuid::from_u128(0xa552),
            attempt_id,
            NodeId::from("asset-node"),
        )?;
        let reference = registry.seal_for_node(
            &node_identity,
            uploaded.record.identity.clone(),
            "FILE_3D_PLY",
        )?;
        assert!(!format!("{reference:?}").contains("sealed.bin"));
        let request = NativeAssetReadRequest::checked(reference.clone(), 1024)?;
        let resolved = registry
            .node_service(node_identity.clone())
            .read_verified(&request, &cancellation)?;
        assert_eq!(resolved.bytes().as_ref(), b"canonical-asset");
        assert_eq!(resolved.reference(), &reference);

        registry.retire_attempt(attempt_id);
        assert!(matches!(
            registry
                .node_service(node_identity.clone())
                .read_verified(&request, &cancellation),
            Err(NativeAssetServiceError::InvalidReference)
        ));
        let reference =
            registry.seal_for_node(&node_identity, uploaded.record.identity, "FILE_3D_PLY")?;
        let request = NativeAssetReadRequest::checked(reference, 1024)?;

        let foreign_identity = NativeNodeServiceIdentity::checked(
            Uuid::from_u128(0xa553),
            attempt_id,
            NodeId::from("foreign-node"),
        )?;
        assert!(matches!(
            registry
                .node_service(foreign_identity)
                .read_verified(&request, &cancellation),
            Err(NativeAssetServiceError::InvalidReference)
        ));
        let cancelled = CancellationToken::default();
        assert!(cancelled.cancel());
        assert!(matches!(
            registry
                .node_service(node_identity)
                .read_verified(&request, &cancelled),
            Err(NativeAssetServiceError::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn native_model_source_resolver_projects_verified_recipe_companions_and_metadata()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, mut service, authorization) = service()?;
        let model_root = service.roots.root_path(AssetNamespace::Model)?;
        let encoder_root = model_root.join("text_encoders");
        fs::create_dir_all(&encoder_root)?;
        let model_path = encoder_root.join("production.safetensors");
        let metadata = BTreeMap::from([(
            "config".to_owned(),
            r#"{"_class_name":"LatentUpsampler","in_channels":4,"mid_channels":8,"num_blocks_per_stage":1,"dims":2,"spatial_upsample":true,"temporal_upsample":false}"#.to_owned(),
        )]);
        let header = serde_json::to_vec(&json!({
            "__metadata__": metadata,
            "weight": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
        }))?;
        let mut model_file = fs::File::create(&model_path)?;
        model_file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
        model_file.write_all(&header)?;
        model_file.write_all(&[1, 2, 3, 4])?;
        let configuration = br#"{"hidden_size":4,"num_hidden_layers":1}"#.to_vec();
        let tokenizer = br#"{"model":{"vocab":{"a":0,"b":1},"merges":[]}}"#.to_vec();
        fs::write(
            encoder_root.join("production.safetensors.clip-config-0.json"),
            &configuration,
        )?;
        fs::write(
            encoder_root.join("production.safetensors.clip-tokenizer-0.json"),
            &tokenizer,
        )?;
        service.scan_namespaces(
            &[AssetNamespace::Model],
            &authorization,
            &CancellationToken::default(),
        )?;
        let registry =
            NativeAssetResolverRegistry::new(Arc::new(Mutex::new(service)), authorization);
        let source_names = vec!["production.safetensors".to_owned()];
        let context = WorkerModelSourceContext {
            session_id: Uuid::from_u128(0x401_733_01),
            attempt_id: AttemptId(Uuid::from_u128(0x401_733_02)),
            attempt_generation: 2,
            node_id: "production-recipe-loader".to_owned(),
            node_generation: 3,
            service_id: Uuid::from_u128(0x401_733_03),
            service_generation: 4,
            ordered_source_identity_sha256: worker_model_source_selection_sha256(
                "text_encoders",
                &source_names,
            )?,
        };
        let open = WorkerModelSourceRequest {
            context: context.clone(),
            call_ordinal: 1,
            operation: WorkerModelSourceOperation::Open {
                folder_category: "text_encoders".to_owned(),
                source_names,
            },
        };
        let opened =
            registry.serve_model_source_request(&context, &open, &CancellationToken::default())?;
        let manifest = match opened {
            WorkerModelSourceResponse::Opened(opened) => opened
                .sources
                .into_iter()
                .next()
                .ok_or("production resolver omitted its source manifest")?,
            response => return Err(format!("unexpected open response: {response:?}").into()),
        };
        assert_eq!(manifest.artifacts.len(), 4);
        assert!(matches!(
            manifest.artifacts.get(1).map(|artifact| artifact.format),
            Some(WorkerModelSourceFormat::SafetensorsMetadata {
                source_artifact_ordinal: 0
            })
        ));
        assert!(matches!(
            manifest.artifacts.get(2).map(|artifact| artifact.format),
            Some(WorkerModelSourceFormat::ClipConfiguration { target_ordinal: 0 })
        ));
        assert!(matches!(
            manifest.artifacts.get(3).map(|artifact| artifact.format),
            Some(WorkerModelSourceFormat::ClipTokenizerJson { target_ordinal: 0 })
        ));
        let metadata_artifact = manifest
            .artifacts
            .get(1)
            .ok_or("production resolver omitted safetensors metadata")?;
        let read_metadata = WorkerModelSourceRequest {
            context: context.clone(),
            call_ordinal: 2,
            operation: WorkerModelSourceOperation::ReadArtifact {
                source_ordinal: 0,
                artifact_ordinal: 1,
                artifact_sha256: metadata_artifact.sha256.clone(),
                byte_offset: 0,
                byte_length: u32::try_from(metadata_artifact.byte_size)?,
            },
        };
        let response = registry.serve_model_source_request(
            &context,
            &read_metadata,
            &CancellationToken::default(),
        )?;
        let bytes = match response {
            WorkerModelSourceResponse::ArtifactChunk(chunk) => chunk.bytes,
            response => return Err(format!("unexpected metadata response: {response:?}").into()),
        };
        assert_eq!(
            sim_model::formats::checked_safetensors_metadata_json(
                &bytes,
                metadata_artifact.sha256.as_str(),
                &sim_model::ParserLimits::default(),
                &CancellationToken::default(),
            )?,
            metadata
        );
        let close = WorkerModelSourceRequest {
            context: context.clone(),
            call_ordinal: 3,
            operation: WorkerModelSourceOperation::Close,
        };
        assert!(matches!(
            registry.serve_model_source_request(&context, &close, &CancellationToken::default(),)?,
            WorkerModelSourceResponse::Closed(_)
        ));
        Ok(())
    }

    #[test]
    fn native_asset_resolver_lists_and_atomically_seals_source_names()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, mut service, authorization) = service()?;
        let model_root = service
            .roots()
            .test_root_path(AssetNamespace::Model)?
            .to_path_buf();
        fs::create_dir_all(model_root.join("text_encoders/nested"))?;
        fs::create_dir_all(model_root.join("clip"))?;
        fs::create_dir_all(model_root.join("checkpoints"))?;
        fs::write(model_root.join("text_encoders/alpha.safetensors"), b"alpha")?;
        fs::write(model_root.join("text_encoders/nested/beta.ckpt"), b"beta")?;
        fs::write(model_root.join("clip/gamma.pt"), b"gamma")?;
        fs::write(
            model_root.join("checkpoints/alpha.safetensors"),
            b"wrong-category",
        )?;
        service.upload(
            upload("visible.png", b"input-image"),
            &authorization,
            &CancellationToken::default(),
        )?;

        let assets = Arc::new(Mutex::new(service));
        let registry = NativeAssetResolverRegistry::new(assets.clone(), authorization.clone());
        let attempt_id = AttemptId(Uuid::from_u128(0xa561));
        let node_identity = NativeNodeServiceIdentity::checked(
            Uuid::from_u128(0xa562),
            attempt_id,
            NodeId::from("asset-name-node"),
        )?;
        let resolver = registry.node_service(node_identity.clone());
        let cancellation = CancellationToken::default();

        let listed = resolver.list_names(
            &NativeAssetNameListRequest::checked("text_encoders")?,
            &cancellation,
        )?;
        assert_eq!(
            listed.names(),
            ["alpha.safetensors", "gamma.pt", "nested/beta.ckpt"]
        );
        let alias_listed =
            resolver.list_names(&NativeAssetNameListRequest::checked("clip")?, &cancellation)?;
        assert_eq!(alias_listed.names(), listed.names());
        let input_names = resolver.list_names(
            &NativeAssetNameListRequest::checked("input")?,
            &cancellation,
        )?;
        assert_eq!(input_names.names(), ["day one/日本語/visible.png"]);
        assert!(matches!(
            resolver.list_names(
                &NativeAssetNameListRequest::checked("custom_nodes")?,
                &cancellation,
            ),
            Err(NativeAssetServiceError::InvalidFolderCategory)
        ));

        let names = vec![
            "nested/beta.ckpt".to_owned(),
            "alpha.safetensors".to_owned(),
            "gamma.pt".to_owned(),
        ];
        let resolution = resolver.resolve_names(
            &NativeAssetNameResolutionRequest::checked("text_encoders", names.clone(), "CLIP")?,
            &cancellation,
        )?;
        assert_eq!(
            resolution
                .assets()
                .iter()
                .map(NativeNamedAssetReference::name)
                .collect::<Vec<_>>(),
            names.iter().map(String::as_str).collect::<Vec<_>>()
        );
        let bytes = resolution
            .assets()
            .iter()
            .map(|asset| {
                resolver
                    .read_verified(
                        &NativeAssetReadRequest::checked(asset.reference().clone(), 1024)?,
                        &cancellation,
                    )
                    .map(|resolved| resolved.bytes().to_vec())
            })
            .collect::<Result<Vec<_>, NativeAssetServiceError>>()?;
        assert_eq!(
            bytes,
            [b"beta".to_vec(), b"alpha".to_vec(), b"gamma".to_vec()]
        );

        let reference_count = registry
            .references
            .lock()
            .map_err(|_| NativeAssetServiceError::Rejected)?
            .len();
        assert!(matches!(
            resolver.resolve_names(
                &NativeAssetNameResolutionRequest::checked(
                    "text_encoders",
                    vec!["alpha.safetensors".to_owned(), "missing.pt".to_owned()],
                    "CLIP",
                )?,
                &cancellation,
            ),
            Err(NativeAssetServiceError::Missing)
        ));
        let cancelled = CancellationToken::default();
        assert!(cancelled.cancel());
        assert!(matches!(
            resolver.resolve_names(
                &NativeAssetNameResolutionRequest::checked(
                    "text_encoders",
                    vec!["alpha.safetensors".to_owned()],
                    "CLIP",
                )?,
                &cancelled,
            ),
            Err(NativeAssetServiceError::Cancelled)
        ));
        assert_eq!(
            registry
                .references
                .lock()
                .map_err(|_| NativeAssetServiceError::Rejected)?
                .len(),
            reference_count
        );
        assert!(matches!(
            NativeAssetNameResolutionRequest::checked(
                "text_encoders",
                vec!["../alpha.safetensors".to_owned()],
                "CLIP",
            ),
            Err(NativeAssetServiceError::InvalidAssetName)
        ));
        assert!(matches!(
            resolver.resolve_names(
                &NativeAssetNameResolutionRequest::checked(
                    "checkpoints",
                    vec!["nested/beta.ckpt".to_owned()],
                    "MODEL",
                )?,
                &cancellation,
            ),
            Err(NativeAssetServiceError::Missing)
        ));

        fs::write(
            model_root.join("text_encoders/nested/beta.ckpt"),
            b"changed",
        )?;
        let changed_request =
            NativeAssetReadRequest::checked(resolution.assets()[0].reference().clone(), 1024)?;
        assert!(matches!(
            resolver.read_verified(&changed_request, &cancellation),
            Err(NativeAssetServiceError::ChangedDuringRead)
        ));

        let denied = NativeAssetResolverRegistry::new(assets.clone(), denied_authorization()?);
        assert!(matches!(
            denied.node_service(node_identity.clone()).list_names(
                &NativeAssetNameListRequest::checked("text_encoders")?,
                &cancellation,
            ),
            Err(NativeAssetServiceError::PermissionDenied)
        ));

        registry.retire_attempt(attempt_id);
        assert!(matches!(
            resolver.read_verified(&changed_request, &cancellation),
            Err(NativeAssetServiceError::InvalidReference)
        ));
        let restarted = NativeAssetResolverRegistry::new(assets, authorization);
        let restarted_resolver = restarted.node_service(node_identity);
        let restarted_resolution = restarted_resolver.resolve_names(
            &NativeAssetNameResolutionRequest::checked(
                "text_encoders",
                vec!["nested/beta.ckpt".to_owned()],
                "CLIP",
            )?,
            &cancellation,
        )?;
        assert_eq!(restarted_resolution.assets().len(), 1);
        Ok(())
    }

    fn taesd_flux2_safetensors(dtype: DType) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        fn convolution(
            state: &mut BTreeMap<String, Vec<u64>>,
            name: &str,
            output: u64,
            input: u64,
            bias: bool,
        ) {
            state.insert(format!("{name}.weight"), vec![output, input, 3, 3]);
            if bias {
                state.insert(format!("{name}.bias"), vec![output]);
            }
        }
        fn block(state: &mut BTreeMap<String, Vec<u64>>, name: &str, pooled: bool) {
            for index in [0_u64, 2, 4] {
                convolution(state, &format!("{name}.conv.{index}"), 64, 64, true);
            }
            if pooled {
                state.insert(format!("{name}.pool.0.weight"), vec![256, 64, 1, 1]);
                state.insert(format!("{name}.pool.1.weight"), vec![256]);
                state.insert(format!("{name}.pool.1.bias"), vec![256]);
                state.insert(format!("{name}.pool.3.weight"), vec![64, 256, 1, 1]);
            }
        }

        let mut state = BTreeMap::new();
        convolution(&mut state, "taesd_encoder.0", 64, 3, true);
        for index in [1_u64, 3, 4, 5, 7, 8, 9, 11, 12, 13] {
            block(&mut state, &format!("taesd_encoder.{index}"), index >= 11);
        }
        for index in [2_u64, 6, 10] {
            convolution(&mut state, &format!("taesd_encoder.{index}"), 64, 64, false);
        }
        convolution(&mut state, "taesd_encoder.14", 32, 64, true);
        convolution(&mut state, "taesd_decoder.1", 64, 32, true);
        for index in [3_u64, 4, 5, 8, 9, 10, 13, 14, 15, 18] {
            block(&mut state, &format!("taesd_decoder.{index}"), index <= 5);
        }
        for index in [7_u64, 12, 17] {
            convolution(&mut state, &format!("taesd_decoder.{index}"), 64, 64, false);
        }
        convolution(&mut state, "taesd_decoder.19", 3, 64, true);
        state.insert("vae_scale".to_owned(), Vec::new());
        state.insert("vae_shift".to_owned(), Vec::new());

        let mut header = serde_json::Map::new();
        header.insert(
            "__metadata__".to_owned(),
            json!({"tae_latent_channels": "128"}),
        );
        let mut offset = 0_u64;
        let mut scale_offset = None;
        let (storage_dtype, byte_width, encoded_one): (&str, u64, &[u8]) = match dtype {
            DType::F32 => ("F32", 4, &1.0_f32.to_le_bytes()),
            DType::F16 => ("F16", 2, &0x3c00_u16.to_le_bytes()),
            DType::Bf16 => ("BF16", 2, &0x3f80_u16.to_le_bytes()),
            _ => return Err("TAESD fixture requires a floating storage dtype".into()),
        };
        for (name, shape) in state {
            let elements = shape
                .iter()
                .try_fold(1_u64, |count, extent| count.checked_mul(*extent))
                .ok_or("TAESD tensor shape overflow")?;
            let length = elements
                .checked_mul(byte_width)
                .ok_or("TAESD tensor size overflow")?;
            let end = offset
                .checked_add(length)
                .ok_or("TAESD checkpoint size overflow")?;
            header.insert(
                name.clone(),
                json!({"dtype": storage_dtype, "shape": shape, "data_offsets": [offset, end]}),
            );
            if name == "vae_scale" {
                scale_offset = Some(offset);
            }
            offset = end;
        }
        let mut encoded_header = serde_json::to_vec(&header)?;
        while encoded_header.len() % 8 != 0 {
            encoded_header.push(b' ');
        }
        let data_length = usize::try_from(offset)?;
        let mut bytes = Vec::new();
        let total_length = 8_usize
            .checked_add(encoded_header.len())
            .and_then(|length| length.checked_add(data_length))
            .ok_or("TAESD checkpoint allocation overflow")?;
        bytes.try_reserve_exact(total_length)?;
        bytes.extend_from_slice(&u64::try_from(encoded_header.len())?.to_le_bytes());
        bytes.extend_from_slice(&encoded_header);
        bytes.resize(bytes.len() + data_length, 0);
        let scale_offset = usize::try_from(scale_offset.ok_or("missing TAESD scale state")?)?;
        let scale_start = 8_usize
            .checked_add(encoded_header.len())
            .and_then(|value| value.checked_add(scale_offset))
            .ok_or("TAESD scale offset overflow")?;
        let scale_end = scale_start
            .checked_add(encoded_one.len())
            .ok_or("TAESD scale offset overflow")?;
        bytes
            .get_mut(scale_start..scale_end)
            .ok_or("TAESD scale offset is outside the fixture")?
            .copy_from_slice(encoded_one);
        Ok(bytes)
    }

    fn music_dcae_marker_safetensors() -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut header = serde_json::Map::new();
        header.insert(
            "vocoder.backbone.channel_layers.0.0.bias".to_owned(),
            json!({"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}),
        );
        let mut encoded_header = serde_json::to_vec(&header)?;
        while encoded_header.len() % 8 != 0 {
            encoded_header.push(b' ');
        }
        let mut bytes = Vec::with_capacity(12 + encoded_header.len());
        bytes.extend_from_slice(&u64::try_from(encoded_header.len())?.to_le_bytes());
        bytes.extend_from_slice(&encoded_header);
        bytes.extend_from_slice(&0.0_f32.to_le_bytes());
        Ok(bytes)
    }

    #[test]
    fn asset_references_have_one_profile_scoped_checked_mapping() -> Result<(), AssetError> {
        let (_directory, service, _authorization) = service()?;
        let identity = service
            .roots()
            .identity(AssetNamespace::Output, "task 18/native-image.png")?;
        let reference = identity.to_reference()?;
        assert_eq!(reference, "zed-asset://output/task 18/native-image.png");
        assert_eq!(
            service.roots().identity_from_reference(&reference)?,
            identity
        );
        assert_eq!(identity.profile_id, "profile");

        for invalid in [
            "https://output/task18/native-image.png",
            "zed-asset://output/../secret",
            "zed-asset://other/task18/native-image.png",
            "zed-asset://output/",
            "zed-asset://output/task18/native-image.png?download=1",
            "zed-asset://output/task18\\native-image.png",
        ] {
            assert!(service.roots().identity_from_reference(invalid).is_err());
        }
        assert!(
            service
                .roots()
                .identity_from_reference("zed-asset://other-profile/output/native-image.png")
                .is_err()
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn asset_rename_and_delete_fail_closed_after_root_replacement()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::symlink;

        let (directory, mut service, authorization) = service()?;
        let cancellation = CancellationToken::default();
        let identity = service
            .roots()
            .identity(AssetNamespace::Output, "asset.bin")?;
        service.write_exact(
            &identity,
            b"anchored",
            BTreeSet::new(),
            AssetCollisionPolicy::Reject,
            &authorization,
            &cancellation,
        )?;
        let configured = service
            .roots()
            .test_root_path(AssetNamespace::Output)?
            .to_path_buf();
        let retained = directory.path().join("retained-output");
        let outside = tempfile::tempdir()?;
        fs::rename(&configured, &retained)?;
        symlink(outside.path(), &configured)?;

        assert!(matches!(
            service.rename(
                &identity,
                "renamed.bin",
                Path::new("nested"),
                &authorization,
            ),
            Err(AssetError::InvalidIndex(_))
        ));
        let retained_path = retained.join(&identity.relative_path);
        let outside_path = outside.path().join(&identity.relative_path);
        assert_eq!(fs::read(&retained_path)?, b"anchored");
        fs::create_dir_all(outside_path.parent().ok_or("outside path has no parent")?)?;
        fs::write(&outside_path, b"foreign")?;

        assert!(matches!(
            service.delete(&identity, &authorization),
            Err(AssetError::InvalidIndex(_))
        ));
        assert_eq!(fs::read(retained_path)?, b"anchored");
        assert_eq!(fs::read(outside_path)?, b"foreign");
        Ok(())
    }

    #[test]
    fn upload_duplicate_collision_range_and_permissions_are_typed() -> Result<(), AssetError> {
        let (_directory, mut service, capabilities) = service()?;
        let cancellation = CancellationToken::default();
        let first = service.upload(upload("image.png", b"first"), &capabilities, &cancellation)?;
        assert!(!first.duplicate);
        let duplicate =
            service.upload(upload("image.png", b"first"), &capabilities, &cancellation)?;
        assert!(duplicate.duplicate);
        assert_eq!(duplicate.record.identity, first.record.identity);
        let collision =
            service.upload(upload("image.png", b"second"), &capabilities, &cancellation)?;
        assert_eq!(collision.collision_index, Some(1));
        assert_eq!(collision.record.identity.filename(), Some("image (1).png"));

        let view = service.view(
            &AssetViewRequest {
                identity: collision.record.identity,
                range: Some(AssetByteRange {
                    start: 1,
                    end_inclusive: 3,
                }),
                download: false,
            },
            &capabilities,
            &cancellation,
        )?;
        assert_eq!(view.bytes, b"eco");
        assert_eq!(view.total_size, 6);

        let mut overwrite = upload("image.png", b"replacement");
        overwrite.overwrite = true;
        let overwritten = service.upload(overwrite, &capabilities, &cancellation)?;
        assert_eq!(overwritten.record.identity, first.record.identity);
        assert_eq!(overwritten.collision_index, None);
        assert_eq!(
            fs::read(
                service
                    .roots
                    .test_resolve_existing(&overwritten.record.identity)?
            )
            .map_err(|error| AssetError::Io {
                path: overwritten.record.identity.relative_path.clone(),
                message: error.to_string(),
            })?,
            b"replacement"
        );

        let denied = service.view(
            &AssetViewRequest {
                identity: first.record.identity,
                range: None,
                download: false,
            },
            &denied_authorization()?,
            &cancellation,
        );
        assert!(matches!(denied, Err(AssetError::PermissionDenied { .. })));
        Ok(())
    }

    #[test]
    fn exact_writes_share_canonical_collision_permissions_and_restart_state()
    -> Result<(), Box<dyn std::error::Error>> {
        let (directory, mut service, capabilities) = service()?;
        let roots = service.roots().clone();
        let cancellation = CancellationToken::default();
        let identity = roots.identity(AssetNamespace::Input, "published/exact.bin")?;
        let first_tags = BTreeSet::from(["first".to_owned()]);
        let first = service.write_exact(
            &identity,
            b"first",
            first_tags.clone(),
            AssetCollisionPolicy::Reject,
            &capabilities,
            &cancellation,
        )?;
        assert_eq!(first.tags, first_tags);
        assert_eq!(fs::read(roots.test_resolve_existing(&identity)?)?, b"first");

        assert!(matches!(
            service.write_exact(
                &identity,
                b"rejected",
                BTreeSet::new(),
                AssetCollisionPolicy::Reject,
                &capabilities,
                &cancellation,
            ),
            Err(AssetError::AlreadyExists(_))
        ));
        assert_eq!(fs::read(roots.test_resolve_existing(&identity)?)?, b"first");

        let replacement_tags = BTreeSet::from(["replacement".to_owned()]);
        let replacement = service.write_exact(
            &identity,
            b"replacement",
            replacement_tags.clone(),
            AssetCollisionPolicy::Replace,
            &capabilities,
            &cancellation,
        )?;
        assert_eq!(replacement.tags, replacement_tags);
        assert_eq!(
            fs::read(roots.test_resolve_existing(&identity)?)?,
            b"replacement"
        );
        assert!(matches!(
            service.write_exact(
                &identity,
                b"denied",
                BTreeSet::new(),
                AssetCollisionPolicy::Replace,
                &denied_authorization()?,
                &cancellation,
            ),
            Err(AssetError::PermissionDenied { .. })
        ));
        assert_eq!(
            fs::read(roots.test_resolve_existing(&identity)?)?,
            b"replacement"
        );

        drop(service);
        let reopened = AssetService::open(roots)?;
        let restarted = reopened
            .record(&identity)
            .ok_or_else(|| AssetError::UnknownAsset(identity.clone()))?;
        assert_eq!(restarted.sha256, sha256(b"replacement"));
        assert_eq!(restarted.tags, replacement_tags);
        assert!(directory.path().exists());
        Ok(())
    }

    #[test]
    fn failed_post_write_state_persistence_restores_bytes_index_and_enrichment()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, mut service, capabilities) = service()?;
        let cancellation = CancellationToken::default();
        let uploaded = service.upload(
            upload("rollback.bin", b"before"),
            &capabilities,
            &cancellation,
        )?;
        let identity = uploaded.record.identity;
        let previous_record = service
            .record(&identity)
            .ok_or_else(|| AssetError::UnknownAsset(identity.clone()))?;
        let previous_state = service
            .roots
            .read_private(&service.state_identity, service.max_index_bytes)?
            .ok_or_else(|| AssetError::InvalidIndex("asset state is missing".to_owned()))?;
        service.max_index_bytes = previous_state.len();

        let result = service.write_exact(
            &identity,
            b"after",
            BTreeSet::from(["x".repeat(128)]),
            AssetCollisionPolicy::Replace,
            &capabilities,
            &cancellation,
        );
        assert!(matches!(result, Err(AssetError::IndexTooLarge { .. })));
        assert_eq!(
            fs::read(service.roots.test_resolve_existing(&identity)?)?,
            b"before"
        );
        let restored_record = service
            .record(&identity)
            .ok_or_else(|| AssetError::UnknownAsset(identity.clone()))?;
        assert_eq!(restored_record.sha256, previous_record.sha256);
        assert_eq!(restored_record.byte_size, previous_record.byte_size);
        assert_eq!(restored_record.tags, previous_record.tags);
        assert_eq!(restored_record.availability, previous_record.availability);
        let restored_state = service
            .roots
            .read_private(&service.state_identity, service.max_index_bytes)?
            .ok_or_else(|| {
                AssetError::InvalidIndex("restored asset state is missing".to_owned())
            })?;
        assert!(restored_state.len() <= previous_state.len());
        serde_json::from_slice::<AssetServiceSnapshot>(&restored_state)
            .map_err(|error| AssetError::IndexDecode(error.to_string()))?;

        let new_identity = service
            .roots
            .identity(AssetNamespace::Input, "day one/日本語/new-rollback.bin")?;
        let result = service.write_exact(
            &new_identity,
            b"new",
            BTreeSet::new(),
            AssetCollisionPolicy::Reject,
            &capabilities,
            &cancellation,
        );
        assert!(matches!(result, Err(AssetError::IndexTooLarge { .. })));
        assert!(!service.roots.resolve_for_create(&new_identity)?.exists());
        assert!(service.record(&new_identity).is_none());

        let roots = service.roots.clone();
        drop(service);
        let reopened = AssetService::open(roots)?;
        let restarted = reopened
            .record(&identity)
            .ok_or_else(|| AssetError::UnknownAsset(identity.clone()))?;
        assert_eq!(restarted.sha256, previous_record.sha256);
        assert_eq!(restarted.tags, previous_record.tags);
        assert_eq!(
            reopened.read_verified(&identity, &capabilities, &cancellation, 64)?,
            b"before"
        );
        assert!(reopened.record(&new_identity).is_none());
        Ok(())
    }

    #[test]
    fn namespace_and_record_enumeration_is_capability_filtered()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, mut service, capabilities) = service()?;
        let cancellation = CancellationToken::default();
        service.upload(
            upload("visible.bin", b"visible"),
            &capabilities,
            &cancellation,
        )?;
        let input_reader = test_authorization(
            "profile",
            CapabilitySet::new([Capability::Asset {
                namespace: AssetNamespace::Input.locator_type().to_owned(),
                action: AssetOperation::Read,
            }]),
        )?;
        let page = service.list_authorized(&AssetQuery::default(), &input_reader)?;
        assert_eq!(page.total, 1);
        assert!(
            page.records
                .iter()
                .all(|record| record.identity.namespace == AssetNamespace::Input)
        );
        assert_eq!(
            service
                .authorized_compatibility_folder_paths(&input_reader)?
                .keys()
                .copied()
                .collect::<Vec<_>>(),
            vec![AssetNamespace::Input]
        );
        assert!(matches!(
            service.list_authorized(
                &AssetQuery {
                    namespace: Some(AssetNamespace::Output),
                    ..AssetQuery::default()
                },
                &input_reader,
            ),
            Err(AssetError::PermissionDenied {
                namespace: AssetNamespace::Output,
                action: AssetAction::Read,
            })
        ));
        assert!(
            service
                .authorized_compatibility_folder_paths(&denied_authorization()?)?
                .is_empty()
        );
        Ok(())
    }

    #[test]
    fn traversal_active_content_and_cancellation_fail_before_publication() -> Result<(), AssetError>
    {
        let (_directory, mut service, capabilities) = service()?;
        let cancellation = CancellationToken::default();
        let traversal = service.upload(
            UploadRequest {
                subfolder: PathBuf::from("../escape"),
                ..upload("image.png", b"x")
            },
            &capabilities,
            &cancellation,
        );
        assert!(matches!(traversal, Err(AssetError::UnsafePath { .. })));
        let active = service.upload(
            upload("page.html", b"<script>x()</script>"),
            &capabilities,
            &cancellation,
        )?;
        let view = service.view(
            &AssetViewRequest {
                identity: active.record.identity,
                range: None,
                download: false,
            },
            &capabilities,
            &cancellation,
        )?;
        assert_eq!(view.content_type, "application/octet-stream");
        assert!(view.content_disposition.starts_with("attachment"));

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let result = service.upload(upload("cancelled.png", b"never"), &capabilities, &cancelled);
        assert_eq!(result, Err(AssetError::Cancelled));
        assert!(
            !service
                .roots()
                .test_root_path(AssetNamespace::Input)?
                .join("day one/日本語/cancelled.png")
                .exists()
        );
        Ok(())
    }

    #[test]
    fn scan_rename_tags_delete_and_restore_keep_one_identity_record() -> Result<(), AssetError> {
        let (_directory, mut service, capabilities) = service()?;
        let cancellation = CancellationToken::default();
        let uploaded = service.upload(upload("asset.bin", b"one"), &capabilities, &cancellation)?;
        let mut tags = BTreeSet::from(["favorite".to_owned(), "input".to_owned()]);
        let tagged = service.set_tags(&uploaded.record.identity, tags.clone(), &capabilities)?;
        assert_eq!(tagged.tags, tags);
        let renamed = service.rename(
            &uploaded.record.identity,
            "renamed.bin",
            Path::new("renamed"),
            &capabilities,
        )?;
        assert_eq!(
            renamed.identity.relative_path,
            PathBuf::from("renamed/renamed.bin")
        );

        let path = service.roots().test_resolve_existing(&renamed.identity)?;
        fs::write(&path, b"two").map_err(|error| AssetError::Io {
            path: path.clone(),
            message: error.to_string(),
        })?;
        assert!(
            service
                .scan(&capabilities, &cancellation)?
                .iter()
                .any(|change| change.kind == AssetChangeKind::Modified)
        );
        fs::remove_file(&path).map_err(|error| AssetError::Io {
            path: path.clone(),
            message: error.to_string(),
        })?;
        assert!(
            service
                .scan(&capabilities, &cancellation)?
                .iter()
                .any(|change| change.kind == AssetChangeKind::Missing)
        );
        fs::write(&path, b"three").map_err(|error| AssetError::Io {
            path: path.clone(),
            message: error.to_string(),
        })?;
        assert!(
            service
                .scan(&capabilities, &cancellation)?
                .iter()
                .any(|change| change.kind == AssetChangeKind::Restored)
        );
        tags.insert("restored".to_owned());
        service.set_tags(&renamed.identity, tags, &capabilities)?;
        let deleted = service.delete(&renamed.identity, &capabilities)?;
        assert_eq!(deleted.availability, AssetAvailability::Missing);
        Ok(())
    }

    #[test]
    fn index_persists_tags_missing_state_and_metadata_across_restart() -> Result<(), AssetError> {
        let (_directory, mut service, capabilities) = service()?;
        let roots = service.roots().clone();
        let cancellation = CancellationToken::default();
        let uploaded = service.upload(
            upload("durable.bin", b"durable"),
            &capabilities,
            &cancellation,
        )?;
        let tags = BTreeSet::from(["favorite".to_owned(), "restart".to_owned()]);
        service.set_tags(&uploaded.record.identity, tags.clone(), &capabilities)?;
        drop(service);

        let mut reopened = AssetService::open(roots.clone())?;
        let restored = reopened
            .record(&uploaded.record.identity)
            .ok_or_else(|| AssetError::UnknownAsset(uploaded.record.identity.clone()))?;
        assert_eq!(restored.tags, tags);
        assert_eq!(restored.sha256, sha256(b"durable"));
        reopened.delete(&uploaded.record.identity, &capabilities)?;
        drop(reopened);

        let reopened = AssetService::open(roots)?;
        assert_eq!(
            reopened
                .record(&uploaded.record.identity)
                .map(|record| record.availability),
            Some(AssetAvailability::Missing)
        );
        Ok(())
    }

    #[test]
    fn streaming_inspection_hashes_full_asset_with_bounded_metadata_prefix()
    -> Result<(), AssetError> {
        let (directory, service, capabilities) = service()?;
        let roots = service.roots().clone();
        drop(service);
        let mut service = AssetService::with_limits(
            roots,
            1024,
            128,
            MetadataLimits {
                max_input_bytes: 8,
                ..MetadataLimits::default()
            },
        )?;
        let bytes = b"0123456789abcdef";
        let record = service
            .upload(
                upload("large.bin", bytes),
                &capabilities,
                &CancellationToken::default(),
            )?
            .record;
        assert_eq!(record.sha256, sha256(bytes));
        assert_eq!(record.byte_size, 16);
        assert!(
            record
                .metadata_diagnostics
                .iter()
                .any(|diagnostic| diagnostic.contains("bounded to the first 8 of 16 bytes"))
        );
        assert!(directory.path().exists());
        Ok(())
    }

    #[test]
    fn asset_service_is_an_enrichment_adapter_over_the_canonical_index()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, mut service, capabilities) = service()?;
        let cancellation = CancellationToken::default();
        let uploaded = service.upload(
            upload("canonical.bin", b"canonical"),
            &capabilities,
            &cancellation,
        )?;
        let key = service.roots.artifact_key(&uploaded.record.identity)?;
        let canonical = service
            .artifact_index()
            .record(&key)
            .ok_or("canonical artifact record is missing")?;
        assert_eq!(canonical.sha256, uploaded.record.sha256);
        assert_eq!(canonical.byte_size, uploaded.record.byte_size);

        let state_bytes = service
            .roots
            .artifact_root_for_identity(&service.state_identity)?
            .read_private_file(
                &service.state_identity.relative_path,
                service.max_index_bytes,
            )?
            .ok_or("asset state is missing")?;
        let state: serde_json::Value = serde_json::from_slice(&state_bytes)?;
        let canonical_snapshot = state
            .get("canonical_artifact_index")
            .and_then(serde_json::Value::as_str)
            .ok_or("canonical artifact snapshot is missing")?;
        let restarted = ArtifactIndex::from_snapshot(
            canonical_snapshot.as_bytes(),
            service.roots.artifact_roots(),
        )?;
        assert_eq!(restarted.record(&key), Some(canonical));

        let source = include_str!("assets.rs")
            .split_once("\n#[cfg(test)]\nmod tests")
            .map_or(include_str!("assets.rs"), |(production, _)| production);
        for prohibited in [
            "struct AssetRoot {",
            "records: BTreeMap<AssetIdentity, AssetRecord>",
            "fn scan_root(",
            "fn stable_file_sha256(",
            "fn stream_file_summary(",
            "fs::canonicalize(",
            "fs::symlink_metadata(",
            "fs::read_dir(",
        ] {
            assert!(
                !source.contains(prohibited),
                "asset adapter regained canonical owner behavior: {prohibited}"
            );
        }
        let upload_owner = source
            .split_once("    pub fn upload(")
            .and_then(|(_, source)| source.split_once("    fn validate_write("))
            .map(|(source, _)| source)
            .ok_or("upload implementation boundary is missing")?;
        assert!(upload_owner.contains("self.commit_asset_write("));
        assert!(!upload_owner.contains("atomic_write("));
        Ok(())
    }

    #[test]
    fn configured_model_roots_share_one_index_and_stable_logical_identities()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let profile_root = directory.path().join("profile");
        let first_root = directory.path().join("first-model-root");
        let second_root = directory.path().join("second-model-root");
        fs::create_dir_all(&first_root)?;
        fs::create_dir_all(&second_root)?;
        for (root, tensor_name, value) in
            [(&first_root, "first", 1_u8), (&second_root, "second", 2_u8)]
        {
            let header =
                format!(r#"{{"{tensor_name}":{{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}}}"#);
            let mut bytes = u64::try_from(header.len())?.to_le_bytes().to_vec();
            bytes.extend_from_slice(header.as_bytes());
            bytes.push(value);
            fs::write(root.join("shared.safetensors"), bytes)?;
        }

        let assets = open_native_profile_asset_service(
            "profile",
            &profile_root,
            &[second_root.clone(), first_root.clone()],
        )?;
        let authorization = test_authorization(
            "profile",
            CapabilitySet::new([Capability::Asset {
                namespace: AssetNamespace::Model.locator_type().to_owned(),
                action: AssetOperation::Read,
            }]),
        )?;
        let identities = {
            let mut service = assets.lock().map_err(|error| error.to_string())?;
            service.scan_namespaces(
                &[AssetNamespace::Model],
                &authorization,
                &CancellationToken::default(),
            )?;
            let records = service
                .list_authorized(
                    &AssetQuery {
                        namespace: Some(AssetNamespace::Model),
                        availability: Some(AssetAvailability::Present),
                        ..AssetQuery::default()
                    },
                    &authorization,
                )?
                .records;
            assert_eq!(records.len(), 2);
            let identities = records
                .into_iter()
                .map(|record| record.identity)
                .collect::<Vec<_>>();
            assert!(identities.iter().all(|identity| identity.root_id.is_some()));
            assert_ne!(identities[0].root_id, identities[1].root_id);

            let mut model_store = ModelStore::new(sim_model::ParserLimits::default())?;
            for identity in &identities {
                let loaded = service.load_model(
                    identity,
                    &mut model_store,
                    &authorization,
                    &CancellationToken::default(),
                )?;
                assert_eq!(loaded.tensors().len(), 1);
            }
            let mut denied_store = ModelStore::new(sim_model::ParserLimits::default())?;
            assert!(matches!(
                service.load_model(
                    &identities[0],
                    &mut denied_store,
                    &denied_authorization()?,
                    &CancellationToken::default(),
                ),
                Err(AssetError::PermissionDenied { .. })
            ));
            assert!(denied_store.operations().is_empty());
            identities
        };
        drop(assets);

        let reopened = open_native_profile_asset_service(
            "profile",
            &profile_root,
            &[first_root, second_root],
        )?;
        let reopened = reopened.lock().map_err(|error| error.to_string())?;
        for identity in identities {
            assert!(reopened.record(&identity).is_some());
        }
        Ok(())
    }

    #[test]
    fn authorized_asset_service_owns_image_vae_production_admission()
    -> Result<(), Box<dyn std::error::Error>> {
        use sim_model::{
            LatentFormatIdentity, ModelFamilyIdentity, NativeVisionModelError, PatchGraph,
            VaeKernelProfile,
        };
        use sim_tensor::{
            CpuWorkspaceAuthority, DType, DeviceId, StreamId, TensorBackend, TensorDescriptor,
            TensorError,
        };

        let (_directory, mut service, authorization) = service()?;
        let cancellation = CancellationToken::default();
        fs::write(
            service
                .roots()
                .test_root_path(AssetNamespace::Model)?
                .join("taesd-flux2.safetensors"),
            taesd_flux2_safetensors(DType::F32)?,
        )?;
        service.scan_namespaces(&[AssetNamespace::Model], &authorization, &cancellation)?;
        let identity = service
            .roots()
            .identity(AssetNamespace::Model, "taesd-flux2.safetensors")?;
        let record = service
            .record(&identity)
            .ok_or_else(|| AssetError::UnknownAsset(identity.clone()))?;
        let (family_registry, _) = VaeArchitectureRegistry::canonical_targets()?;
        let family = family_registry
            .definitions_in_source_order()
            .into_iter()
            .find(|definition| definition.identifier == "Flux2")
            .ok_or("missing canonical Flux2 model family")?;
        let target = VaeExecutionTarget::new(
            ModelFamilyIdentity::new(
                family.feature_id,
                family.identifier,
                family.architecture_version,
            )?,
            LatentFormatIdentity::new(family.latent_feature_id, family.latent_identifier)?,
            DType::F32,
            DeviceId::CPU,
        );
        let patch = PatchGraph::checked_semantic(record.sha256.clone(), Vec::new())?.identity();
        let (backend, workspace_authority) =
            CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
        let scratch = workspace_authority.authorize_workspace(64 * 1024 * 1024)?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch,
            rng_phase: None,
            cancellation: &cancellation,
        };
        for dtype in [DType::F16, DType::Bf16] {
            let low_precision_target = VaeExecutionTarget::new(
                target.family().clone(),
                target.latent_format().clone(),
                dtype,
                DeviceId::CPU,
            );
            let mut low_precision_store = ModelStore::new(sim_model::ParserLimits::default())?;
            assert!(matches!(
                service.load_image_vae_with_context(
                    &identity,
                    &mut low_precision_store,
                    &low_precision_target,
                    PatchGraph::checked_semantic(record.sha256.clone(), Vec::new())?.identity(),
                    VaeBoundary::image(3)?,
                    [-1.0, 1.0],
                    None,
                    &backend,
                    &authorization,
                    &context,
                ),
                Err(AssetImageVaeLoadError::Vae(VaeError::UnsupportedDType(
                    actual
                ))) if actual == dtype
            ));
            assert!(low_precision_store.operations().is_empty());
        }

        let metal_target = VaeExecutionTarget::new(
            target.family().clone(),
            target.latent_format().clone(),
            DType::F32,
            DeviceId::new(sim_types::DeviceKind::Metal, 0),
        );
        let mut metal_store = ModelStore::new(sim_model::ParserLimits::default())?;
        assert!(matches!(
            service.load_image_vae_with_context(
                &identity,
                &mut metal_store,
                &metal_target,
                PatchGraph::checked_semantic(record.sha256.clone(), Vec::new())?.identity(),
                VaeBoundary::image(3)?,
                [-1.0, 1.0],
                None,
                &backend,
                &authorization,
                &context,
            ),
            Err(AssetImageVaeLoadError::Vae(
                VaeError::ExecutionDeviceMismatch { .. }
            ))
        ));
        assert!(metal_store.operations().is_empty());

        let mut model_store = ModelStore::new(sim_model::ParserLimits::default())?;
        let vae = service.load_image_vae_with_context(
            &identity,
            &mut model_store,
            &target,
            patch,
            VaeBoundary::image(3)?,
            [-1.0, 1.0],
            None,
            &backend,
            &authorization,
            &context,
        )?;
        assert_eq!(
            vae.descriptor().identity().profile(),
            &VaeKernelProfile::TaesdV1
        );
        assert!(!model_store.operations().is_empty());
        assert_eq!(context.scratch.in_use_bytes(), 0);
        drop(vae);

        let pixel_descriptor = TensorDescriptor::contiguous(
            vec![1, 3, 16, 16],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let (pixels, event) =
            backend.upload_f32(pixel_descriptor, &vec![0.25; 3 * 16 * 16], &context)?;
        backend.wait_event(event, &context)?;
        let encoded = service.load_and_execute_image_vae_with_context(
            &identity,
            &mut model_store,
            &target,
            PatchGraph::checked_semantic(record.sha256.clone(), Vec::new())?.identity(),
            VaeBoundary::image(3)?,
            [-1.0, 1.0],
            None,
            VaeOperation::Encode,
            &pixels,
            &backend,
            &authorization,
            &context,
        )?;
        assert_eq!(encoded.descriptor().shape(), [1, 128, 1, 1]);
        let decoded = service.load_and_execute_image_vae_with_context(
            &identity,
            &mut model_store,
            &target,
            PatchGraph::checked_semantic(record.sha256.clone(), Vec::new())?.identity(),
            VaeBoundary::image(3)?,
            [-1.0, 1.0],
            None,
            VaeOperation::Decode,
            &encoded,
            &backend,
            &authorization,
            &context,
        )?;
        assert_eq!(decoded.descriptor().shape(), [1, 3, 16, 16]);
        assert_eq!(context.scratch.in_use_bytes(), 0);

        for (storage_dtype, filename) in [
            (DType::F16, "taesd-flux2-f16.safetensors"),
            (DType::Bf16, "taesd-flux2-bf16.safetensors"),
        ] {
            fs::write(
                service
                    .roots()
                    .test_root_path(AssetNamespace::Model)?
                    .join(filename),
                taesd_flux2_safetensors(storage_dtype)?,
            )?;
            service.scan_namespaces(&[AssetNamespace::Model], &authorization, &cancellation)?;
            let storage_identity = service.roots().identity(AssetNamespace::Model, filename)?;
            let storage_record = service
                .record(&storage_identity)
                .ok_or_else(|| AssetError::UnknownAsset(storage_identity.clone()))?;
            let mut storage_store = ModelStore::new(sim_model::ParserLimits::default())?;
            let storage_encoded = service.load_and_execute_image_vae_with_context(
                &storage_identity,
                &mut storage_store,
                &target,
                PatchGraph::checked_semantic(storage_record.sha256, Vec::new())?.identity(),
                VaeBoundary::image(3)?,
                [-1.0, 1.0],
                None,
                VaeOperation::Encode,
                &pixels,
                &backend,
                &authorization,
                &context,
            )?;
            assert_eq!(storage_encoded.descriptor().shape(), [1, 128, 1, 1]);
            assert_eq!(storage_encoded.descriptor().dtype(), DType::F32);
            assert_eq!(context.scratch.in_use_bytes(), 0);
        }

        let mut denied_store = ModelStore::new(sim_model::ParserLimits::default())?;
        assert!(matches!(
            service.load_image_vae_with_context(
                &identity,
                &mut denied_store,
                &target,
                PatchGraph::checked_semantic(record.sha256.clone(), Vec::new())?.identity(),
                VaeBoundary::image(3)?,
                [-1.0, 1.0],
                None,
                &backend,
                &denied_authorization()?,
                &context,
            ),
            Err(AssetImageVaeLoadError::Asset(
                AssetError::PermissionDenied { .. }
            ))
        ));
        assert!(denied_store.operations().is_empty());

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_scratch = workspace_authority.authorize_workspace(64 * 1024 * 1024)?;
        let cancelled_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: cancelled_scratch.clone(),
            rng_phase: None,
            cancellation: &cancelled,
        };
        let mut cancelled_store = ModelStore::new(sim_model::ParserLimits::default())?;
        assert!(matches!(
            service.load_image_vae_with_context(
                &identity,
                &mut cancelled_store,
                &target,
                PatchGraph::checked_semantic(record.sha256.clone(), Vec::new())?.identity(),
                VaeBoundary::image(3)?,
                [-1.0, 1.0],
                None,
                &backend,
                &authorization,
                &cancelled_context,
            ),
            Err(AssetImageVaeLoadError::Asset(AssetError::Model(
                ModelStoreError::Cancelled
            )))
        ));
        assert_eq!(cancelled_scratch.in_use_bytes(), 0);

        let insufficient_scratch = workspace_authority.authorize_workspace(1)?;
        let insufficient_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: insufficient_scratch.clone(),
            rng_phase: None,
            cancellation: &cancellation,
        };
        let mut insufficient_store = ModelStore::new(sim_model::ParserLimits::default())?;
        let insufficient_error = service
            .load_image_vae_with_context(
                &identity,
                &mut insufficient_store,
                &target,
                PatchGraph::checked_semantic(record.sha256, Vec::new())?.identity(),
                VaeBoundary::image(3)?,
                [-1.0, 1.0],
                None,
                &backend,
                &authorization,
                &insufficient_context,
            )
            .expect_err("one byte of scratch must reject image VAE state materialization");
        assert!(matches!(
            insufficient_error,
            AssetImageVaeLoadError::ImageVae(error) if matches!(
                error.as_ref(),
                ImageVaeError::VisionState(NativeVisionModelError::TensorStorage(
                    TensorError::WorkspaceAuthorizationExceeded { .. }
                ))
            )
        ));
        assert_eq!(insufficient_scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn audio_vae_production_admission_is_authorized_canonical_and_failure_atomic()
    -> Result<(), Box<dyn std::error::Error>> {
        use sim_model::{
            LatentFormatIdentity, ModelFamilyIdentity, PatchGraph, VaeKernelProfile,
            VaeStructuredOutputKind,
        };
        use sim_tensor::{CpuWorkspaceAuthority, DType, DeviceId, StreamId};

        let (_directory, mut service, authorization) = service()?;
        let cancellation = CancellationToken::default();
        fs::write(
            service
                .roots()
                .test_root_path(AssetNamespace::Model)?
                .join("music-dcae-marker.safetensors"),
            music_dcae_marker_safetensors()?,
        )?;
        service.scan_namespaces(&[AssetNamespace::Model], &authorization, &cancellation)?;
        let identity = service
            .roots()
            .identity(AssetNamespace::Model, "music-dcae-marker.safetensors")?;
        let record = service
            .record(&identity)
            .ok_or_else(|| AssetError::UnknownAsset(identity.clone()))?;
        let (family_registry, _) = VaeArchitectureRegistry::canonical_targets()?;
        let family = family_registry
            .definitions_in_source_order()
            .into_iter()
            .find(|definition| definition.identifier == "ACEStep")
            .ok_or("missing canonical ACEStep model family")?;
        let target = VaeExecutionTarget::new(
            ModelFamilyIdentity::new(
                family.feature_id,
                family.identifier,
                family.architecture_version,
            )?,
            LatentFormatIdentity::new(family.latent_feature_id, family.latent_identifier)?,
            DType::F32,
            DeviceId::CPU,
        );
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(1 << 20)?;
        let scratch = workspace.authorize_workspace(1 << 20)?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: scratch.clone(),
            rng_phase: None,
            cancellation: &cancellation,
        };
        let patch = || -> Result<_, Box<dyn std::error::Error>> {
            Ok(PatchGraph::checked_semantic(record.sha256.clone(), Vec::new())?.identity())
        };

        let mut denied_store = ModelStore::new(sim_model::ParserLimits::default())?;
        assert!(matches!(
            service.load_audio_vae_with_context(
                &identity,
                &mut denied_store,
                &target,
                patch()?,
                VaeBoundary::audio(2, 44_100)?,
                [-1.0, 1.0],
                &backend,
                &denied_authorization()?,
                &context,
            ),
            Err(AssetAudioVaeLoadError::Asset(
                AssetError::PermissionDenied { .. }
            ))
        ));
        assert!(denied_store.operations().is_empty());

        let mut denied_structured_store = ModelStore::new(sim_model::ParserLimits::default())?;
        assert!(matches!(
            service.load_structured_vae_with_context(
                &identity,
                &mut denied_structured_store,
                &target,
                patch()?,
                VaeBoundary::structured_output(1, VaeStructuredOutputKind::Shape)?,
                [-1.0, 1.0],
                &backend,
                &denied_authorization()?,
                &context,
            ),
            Err(AssetStructuredVaeLoadError::Asset(
                AssetError::PermissionDenied { .. }
            ))
        ));
        assert!(denied_structured_store.operations().is_empty());

        let metal_target = VaeExecutionTarget::new(
            target.family().clone(),
            target.latent_format().clone(),
            DType::F32,
            DeviceId::new(sim_types::DeviceKind::Metal, 0),
        );
        let mut metal_store = ModelStore::new(sim_model::ParserLimits::default())?;
        assert!(matches!(
            service.load_audio_vae_with_context(
                &identity,
                &mut metal_store,
                &metal_target,
                patch()?,
                VaeBoundary::audio(2, 44_100)?,
                [-1.0, 1.0],
                &backend,
                &authorization,
                &context,
            ),
            Err(AssetAudioVaeLoadError::Vae(
                VaeError::ExecutionDeviceMismatch { .. }
            ))
        ));
        assert!(metal_store.operations().is_empty());

        let mut model_store = ModelStore::new(sim_model::ParserLimits::default())?;
        let error = service
            .load_audio_vae_with_context(
                &identity,
                &mut model_store,
                &target,
                patch()?,
                VaeBoundary::audio(2, 44_100)?,
                [-1.0, 1.0],
                &backend,
                &authorization,
                &context,
            )
            .expect_err("partial MusicDCAE state must not publish a NativeVae");
        let first_error = error.to_string();
        assert!(
            matches!(
                &error,
                AssetAudioVaeLoadError::AudioVae(error)
                    if matches!(
                        error.as_ref(),
                        AudioVaeError::InvalidStateShape { .. }
                            | AudioVaeError::MissingState(_)
                    )
            ),
            "unexpected audio admission error: {error:?}"
        );
        assert!(!model_store.operations().is_empty());
        assert_eq!(scratch.in_use_bytes(), 0);

        let retry_error = service
            .load_audio_vae_with_context(
                &identity,
                &mut model_store,
                &target,
                patch()?,
                VaeBoundary::audio(2, 44_100)?,
                [-1.0, 1.0],
                &backend,
                &authorization,
                &context,
            )
            .expect_err("caller retry must remain failure atomic for partial state");
        assert_eq!(retry_error.to_string(), first_error);
        assert_eq!(scratch.in_use_bytes(), 0);
        assert_eq!(
            VaeKernelProfile::MusicDcaeV1.target_latent_channels(),
            Some(8)
        );
        Ok(())
    }

    #[test]
    fn configured_root_changes_migrate_and_persist_the_canonical_asset_state()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let profile_root = directory.path().join("profile");
        let retained_root = directory.path().join("retained-model-root");
        let removed_root = directory.path().join("removed-model-root");
        let added_root = directory.path().join("added-model-root");
        fs::create_dir_all(&retained_root)?;
        fs::create_dir_all(&removed_root)?;
        fs::create_dir_all(&added_root)?;
        fs::write(retained_root.join("retained.bin"), b"retained")?;
        fs::write(removed_root.join("removed.bin"), b"removed")?;
        fs::write(added_root.join("added.bin"), b"added")?;
        let authorization = test_authorization(
            "profile",
            CapabilitySet::new([Capability::Asset {
                namespace: AssetNamespace::Model.locator_type().to_owned(),
                action: AssetOperation::Read,
            }]),
        )?;

        let initial = open_native_profile_asset_service(
            "profile",
            &profile_root,
            &[retained_root.clone(), removed_root],
        )?;
        let (retained_identity, removed_identity) = {
            let mut service = initial.lock().map_err(|error| error.to_string())?;
            service.scan_namespaces(
                &[AssetNamespace::Model],
                &authorization,
                &CancellationToken::default(),
            )?;
            let records = service
                .list_authorized(
                    &AssetQuery {
                        namespace: Some(AssetNamespace::Model),
                        ..AssetQuery::default()
                    },
                    &authorization,
                )?
                .records;
            let retained = records
                .iter()
                .find(|record| record.identity.filename() == Some("retained.bin"))
                .ok_or("retained model was not indexed")?
                .identity
                .clone();
            let removed = records
                .iter()
                .find(|record| record.identity.filename() == Some("removed.bin"))
                .ok_or("removed model was not indexed")?
                .identity
                .clone();
            (retained, removed)
        };
        drop(initial);

        let migrated = open_native_profile_asset_service(
            "profile",
            &profile_root,
            &[added_root.clone(), retained_root.clone()],
        )?;
        {
            let service = migrated.lock().map_err(|error| error.to_string())?;
            assert_eq!(
                service.state_recovery(),
                Some(&AssetStateRecovery::Migrated {
                    removed_roots: 1,
                    added_roots: 1,
                    dropped_records: 1,
                })
            );
            assert!(service.record(&retained_identity).is_some());
            assert!(service.record(&removed_identity).is_none());
        }
        drop(migrated);

        let stable = open_native_profile_asset_service(
            "profile",
            &profile_root,
            &[retained_root, added_root],
        )?;
        let stable = stable.lock().map_err(|error| error.to_string())?;
        assert!(stable.state_recovery().is_none());
        assert!(stable.record(&retained_identity).is_some());
        assert!(stable.record(&removed_identity).is_none());
        Ok(())
    }

    #[test]
    fn corrupt_asset_state_is_quarantined_and_replaced_with_a_valid_snapshot()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let profile_root = directory.path().join("profile");
        let initial = open_native_profile_asset_service("profile", &profile_root, &[])?;
        {
            let mut service = initial.lock().map_err(|error| error.to_string())?;
            service.scan_namespaces(
                &[AssetNamespace::Temporary],
                &test_authorization(
                    "profile",
                    CapabilitySet::new([Capability::Asset {
                        namespace: AssetNamespace::Temporary.locator_type().to_owned(),
                        action: AssetOperation::Read,
                    }]),
                )?,
                &CancellationToken::default(),
            )?;
        }
        drop(initial);
        let invalid = b"{not valid asset state";
        fs::write(
            profile_root.join("temporary").join(ASSET_INDEX_FILENAME),
            invalid,
        )?;

        let recovered = open_native_profile_asset_service("profile", &profile_root, &[])?;
        let quarantine_relative_path = {
            let service = recovered.lock().map_err(|error| error.to_string())?;
            let Some(AssetStateRecovery::Quarantined {
                quarantine_relative_path,
                reason,
            }) = service.state_recovery()
            else {
                return Err("corrupt asset state was not quarantined".into());
            };
            assert!(reason.contains("asset index decode failed"));
            assert_eq!(service.artifact_index().records().count(), 0);
            quarantine_relative_path.clone()
        };
        drop(recovered);

        let temporary_root = ArtifactRoot::canonical(
            AssetNamespace::Temporary.locator_type(),
            AssetNamespace::Temporary.locator_type(),
            &profile_root.join("temporary"),
            std::iter::empty::<String>(),
        )?;
        assert_eq!(
            temporary_root.read_private_file(&quarantine_relative_path, invalid.len())?,
            Some(invalid.to_vec())
        );
        let stable = open_native_profile_asset_service("profile", &profile_root, &[])?;
        assert!(
            stable
                .lock()
                .map_err(|error| error.to_string())?
                .state_recovery()
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn unindexed_rename_and_delete_fail_before_filesystem_mutation() -> Result<(), AssetError> {
        let (_directory, mut service, capabilities) = service()?;
        let root = service.roots().test_root_path(AssetNamespace::Input)?;
        let path = root.join("external.bin");
        fs::write(&path, b"external").map_err(|error| AssetError::Io {
            path: path.clone(),
            message: error.to_string(),
        })?;
        let identity = service
            .roots()
            .identity(AssetNamespace::Input, "external.bin")?;
        assert!(matches!(
            service.rename(&identity, "renamed.bin", Path::new(""), &capabilities),
            Err(AssetError::UnknownAsset(_))
        ));
        assert!(path.is_file());
        assert!(matches!(
            service.delete(&identity, &capabilities),
            Err(AssetError::UnknownAsset(_))
        ));
        assert_eq!(
            fs::read(path).map_err(|error| AssetError::Io {
                path: PathBuf::from("external.bin"),
                message: error.to_string(),
            })?,
            b"external"
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn symbolic_link_subfolder_is_rejected() -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::symlink;
        let (directory, mut service, capabilities) = service()?;
        let outside = directory.path().join("outside");
        fs::create_dir(&outside)?;
        let input = service.roots().test_root_path(AssetNamespace::Input)?;
        symlink(&outside, input.join("linked"))?;
        let request = UploadRequest {
            subfolder: PathBuf::from("linked"),
            ..upload("escape.png", b"x")
        };
        assert!(matches!(
            service.upload(request, &capabilities, &CancellationToken::default()),
            Err(AssetError::UnsafePath { .. })
        ));
        Ok(())
    }

    #[test]
    fn val_domain_008_asset_stage() -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, mut service, capabilities) = service()?;
        let cancellation = CancellationToken::default();
        let uploaded = service.upload(
            upload("fixture.png", b"fixture"),
            &capabilities,
            &cancellation,
        )?;
        let duplicate = service.upload(
            upload("fixture.png", b"fixture"),
            &capabilities,
            &cancellation,
        )?;
        let collision = service.upload(
            upload("fixture.png", b"other"),
            &capabilities,
            &cancellation,
        )?;
        let range = service.view(
            &AssetViewRequest {
                identity: collision.record.identity,
                range: Some(AssetByteRange {
                    start: 0,
                    end_inclusive: 2,
                }),
                download: true,
            },
            &capabilities,
            &cancellation,
        )?;
        let cases = json!({
            "typed_namespace": uploaded.record.identity.namespace == AssetNamespace::Input,
            "safe_subfolder": uploaded.record.identity.relative_path == Path::new("day one/日本語/fixture.png"),
            "duplicate_hash": duplicate.duplicate,
            "collision_number": collision.collision_index == Some(1),
            "bounded_range": range.bytes == b"oth",
            "permission_gate": matches!(service.view(&AssetViewRequest { identity: uploaded.record.identity, range: None, download: false }, &denied_authorization()?, &cancellation), Err(AssetError::PermissionDenied { .. })),
            "traversal_rejected": matches!(service.upload(UploadRequest { subfolder: PathBuf::from(".."), ..upload("bad.png", b"x") }, &capabilities, &cancellation), Err(AssetError::UnsafePath { .. })),
        });
        assert!(
            cases
                .as_object()
                .is_some_and(|cases| cases.values().all(|value| value == &Value::Bool(true)))
        );
        let artifact = json!({
            "validation": "VAL-DOMAIN-008",
            "scope": "asset-namespace-stage",
            "environment": {"os": std::env::consts::OS, "arch": std::env::consts::ARCH, "backend": "native-rust"},
            "fixture_digests": {
                "upload_sha256": sha256(b"fixture"),
                "collision_sha256": sha256(b"other"),
                "traversal_sha256": sha256(b"../escape"),
            },
            "summary": {"passed": 7, "failed": 0, "skipped": 0},
            "cases": cases,
            "skipped": [],
            "subprocesses": 0,
        });
        let target = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../..")
                    .join("target")
            });
        let artifact_directory = target.join("sim-parity");
        fs::create_dir_all(&artifact_directory)?;
        fs::write(
            artifact_directory.join("val-domain-008.json"),
            serde_json::to_vec_pretty(&artifact)?,
        )?;
        Ok(())
    }
}
