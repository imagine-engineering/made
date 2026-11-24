use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use async_trait::async_trait;
use aws_sdk_s3::primitives::ByteStream;
use collaboration_domain::{AggregateId, CommunityId};
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::AsyncReadExt;
use uuid::Uuid;

use super::repository_registry::{HostedAuthority, HostedRepository, HostedRepositoryLifecycle};

const STORAGE_FORMAT_VERSION: u32 = 1;
const STORAGE_ROOT: &str = "collaboration-git/v1";

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct GitContentDigest(String);

impl GitContentDigest {
    pub fn parse(value: impl Into<String>) -> Result<Self, GitObjectStoreError> {
        let value = value.into();
        if !is_lower_hex(&value, 64) {
            return Err(GitObjectStoreError::InvalidDigest);
        }
        Ok(Self(value))
    }

    pub fn for_bytes(bytes: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        Self(hex::encode(hasher.finalize()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for GitContentDigest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::parse(String::deserialize(deserializer)?)
            .map_err(|_| serde::de::Error::custom("invalid Git content digest"))
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct GitObjectId(String);

impl GitObjectId {
    pub fn parse(value: impl Into<String>) -> Result<Self, GitObjectStoreError> {
        let value = value.into();
        if !is_lower_hex(&value, 40) && !is_lower_hex(&value, 64) {
            return Err(GitObjectStoreError::InvalidObjectId);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for GitObjectId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::parse(String::deserialize(deserializer)?)
            .map_err(|_| serde::de::Error::custom("invalid Git object ID"))
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct GitRefName(String);

impl GitRefName {
    pub fn parse(value: impl Into<String>) -> Result<Self, GitObjectStoreError> {
        let value = value.into();
        if !is_safe_ref_name(&value) {
            return Err(GitObjectStoreError::InvalidRefName);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for GitRefName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::parse(String::deserialize(deserializer)?)
            .map_err(|_| serde::de::Error::custom("invalid Git ref name"))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GitRefManifest {
    version: u32,
    head: Option<GitRefName>,
    refs: BTreeMap<GitRefName, GitObjectId>,
    objects: BTreeSet<GitContentDigest>,
    parent: Option<GitContentDigest>,
}

impl GitRefManifest {
    pub fn new(
        head: Option<GitRefName>,
        refs: BTreeMap<GitRefName, GitObjectId>,
        objects: BTreeSet<GitContentDigest>,
        parent: Option<GitContentDigest>,
    ) -> Self {
        Self {
            version: STORAGE_FORMAT_VERSION,
            head,
            refs,
            objects,
            parent,
        }
    }

    pub fn empty(parent: Option<GitContentDigest>) -> Self {
        Self::new(None, BTreeMap::new(), BTreeSet::new(), parent)
    }

    pub fn head(&self) -> Option<&GitRefName> {
        self.head.as_ref()
    }

    pub fn refs(&self) -> &BTreeMap<GitRefName, GitObjectId> {
        &self.refs
    }

    pub fn objects(&self) -> &BTreeSet<GitContentDigest> {
        &self.objects
    }

    pub fn parent(&self) -> Option<&GitContentDigest> {
        self.parent.as_ref()
    }

    fn validate(&self, limits: GitObjectStoreLimits) -> Result<(), GitObjectStoreError> {
        if self.version != STORAGE_FORMAT_VERSION
            || self.refs.len() > limits.max_refs
            || self.objects.len() > limits.max_objects_per_manifest
            || self
                .head
                .as_ref()
                .is_some_and(|head| !self.refs.contains_key(head))
            || (!self.refs.is_empty() && self.objects.is_empty())
        {
            return Err(GitObjectStoreError::InvalidManifest);
        }
        Ok(())
    }

    fn canonical_bytes(&self) -> Result<Vec<u8>, GitObjectStoreError> {
        serde_json::to_vec(self).map_err(|_| GitObjectStoreError::InvalidManifest)
    }

    fn from_bytes(bytes: &[u8], limits: GitObjectStoreLimits) -> Result<Self, GitObjectStoreError> {
        let manifest: Self =
            serde_json::from_slice(bytes).map_err(|_| GitObjectStoreError::InvalidManifest)?;
        manifest.validate(limits)?;
        Ok(manifest)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GitObjectStoreLimits {
    pub max_object_bytes: u64,
    pub max_manifest_bytes: u64,
    pub max_pointer_bytes: u64,
    pub max_refs: usize,
    pub max_objects_per_manifest: usize,
}

impl GitObjectStoreLimits {
    pub fn validate(self) -> Result<Self, GitObjectStoreError> {
        if self.max_object_bytes == 0
            || self.max_manifest_bytes == 0
            || self.max_pointer_bytes == 0
            || self.max_refs == 0
            || self.max_objects_per_manifest == 0
            || usize::try_from(self.max_object_bytes).is_err()
            || usize::try_from(self.max_manifest_bytes).is_err()
            || usize::try_from(self.max_pointer_bytes).is_err()
        {
            return Err(GitObjectStoreError::InvalidLimits);
        }
        Ok(self)
    }
}

impl Default for GitObjectStoreLimits {
    fn default() -> Self {
        Self {
            max_object_bytes: 512 * 1024 * 1024,
            max_manifest_bytes: 4 * 1024 * 1024,
            max_pointer_bytes: 4 * 1024,
            max_refs: 10_000,
            max_objects_per_manifest: 128,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EntityTag(String);

impl EntityTag {
    pub fn parse(value: impl Into<String>) -> Result<Self, GitObjectBackendError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > 1024
            || value.chars().any(|character| character.is_control())
        {
            return Err(GitObjectBackendError::Unavailable);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendObject {
    pub bytes: Vec<u8>,
    pub entity_tag: EntityTag,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BackendWriteCondition {
    CreateOnly,
    IfMatch(EntityTag),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BackendWriteOutcome {
    Stored(EntityTag),
    PreconditionFailed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum GitObjectBackendError {
    #[error("object-store response exceeded the read limit")]
    ObjectTooLarge,
    #[error("object-store backend is unavailable")]
    Unavailable,
}

#[async_trait]
pub trait GitObjectBackend: Send + Sync {
    async fn get(
        &self,
        key: &str,
        max_bytes: u64,
    ) -> Result<Option<BackendObject>, GitObjectBackendError>;

    async fn put(
        &self,
        key: &str,
        bytes: Vec<u8>,
        content_type: &'static str,
        condition: BackendWriteCondition,
    ) -> Result<BackendWriteOutcome, GitObjectBackendError>;
}

#[derive(Clone)]
pub struct AwsS3GitObjectBackend {
    client: aws_sdk_s3::Client,
    bucket: String,
}

impl AwsS3GitObjectBackend {
    pub fn new(
        client: aws_sdk_s3::Client,
        bucket: impl Into<String>,
    ) -> Result<Self, GitObjectBackendError> {
        let bucket = bucket.into();
        if bucket.is_empty() || bucket.len() > 255 || bucket.chars().any(char::is_control) {
            return Err(GitObjectBackendError::Unavailable);
        }
        Ok(Self { client, bucket })
    }
}

#[async_trait]
impl GitObjectBackend for AwsS3GitObjectBackend {
    async fn get(
        &self,
        key: &str,
        max_bytes: u64,
    ) -> Result<Option<BackendObject>, GitObjectBackendError> {
        let output = match self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
        {
            Ok(output) => output,
            Err(error)
                if error
                    .as_service_error()
                    .is_some_and(|error| error.is_no_such_key())
                    || error
                        .raw_response()
                        .is_some_and(|response| response.status().as_u16() == 404) =>
            {
                return Ok(None);
            }
            Err(_) => return Err(GitObjectBackendError::Unavailable),
        };
        if output
            .content_length()
            .and_then(|length| u64::try_from(length).ok())
            .is_some_and(|length| length > max_bytes)
        {
            return Err(GitObjectBackendError::ObjectTooLarge);
        }
        let entity_tag =
            EntityTag::parse(output.e_tag().ok_or(GitObjectBackendError::Unavailable)?)?;
        let read_limit = max_bytes
            .checked_add(1)
            .ok_or(GitObjectBackendError::ObjectTooLarge)?;
        let mut reader = output.body.into_async_read().take(read_limit);
        let mut bytes = Vec::new();
        reader
            .read_to_end(&mut bytes)
            .await
            .map_err(|_| GitObjectBackendError::Unavailable)?;
        if u64::try_from(bytes.len()).map_or(true, |length| length > max_bytes) {
            return Err(GitObjectBackendError::ObjectTooLarge);
        }
        Ok(Some(BackendObject { bytes, entity_tag }))
    }

    async fn put(
        &self,
        key: &str,
        bytes: Vec<u8>,
        content_type: &'static str,
        condition: BackendWriteCondition,
    ) -> Result<BackendWriteOutcome, GitObjectBackendError> {
        let mut request = self
            .client
            .put_object()
            .bucket(&self.bucket)
            .key(key)
            .content_type(content_type)
            .body(ByteStream::from(bytes));
        request = match condition {
            BackendWriteCondition::CreateOnly => request.if_none_match("*"),
            BackendWriteCondition::IfMatch(entity_tag) => request.if_match(entity_tag.0),
        };
        match request
            .customize()
            .config_override(
                aws_sdk_s3::config::Builder::new()
                    .retry_config(aws_config::retry::RetryConfig::disabled()),
            )
            .send()
            .await
        {
            Ok(output) => Ok(BackendWriteOutcome::Stored(EntityTag::parse(
                output.e_tag().ok_or(GitObjectBackendError::Unavailable)?,
            )?)),
            Err(error)
                if error
                    .raw_response()
                    .is_some_and(|response| response.status().as_u16() == 412) =>
            {
                Ok(BackendWriteOutcome::PreconditionFailed)
            }
            Err(_) => Err(GitObjectBackendError::Unavailable),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RepositoryStorageScope {
    community_id: CommunityId,
    repository_id: AggregateId,
    storage_handle_id: Uuid,
}

impl RepositoryStorageScope {
    fn from_authorized_repository(
        repository: &HostedRepository,
    ) -> Result<Self, GitObjectStoreError> {
        if repository.lifecycle != HostedRepositoryLifecycle::Active {
            return Err(GitObjectStoreError::RepositoryUnavailable);
        }
        if repository.community_id.as_uuid().is_nil() || repository.repository_id.as_uuid().is_nil()
        {
            return Err(GitObjectStoreError::RepositoryUnavailable);
        }
        let HostedAuthority::SimHostedNip34 { storage_handle_id } = repository.authority else {
            return Err(GitObjectStoreError::UnsupportedAuthority);
        };
        if storage_handle_id.is_nil() {
            return Err(GitObjectStoreError::RepositoryUnavailable);
        }
        Ok(Self {
            community_id: repository.community_id,
            repository_id: repository.repository_id,
            storage_handle_id,
        })
    }

    fn prefix(&self) -> String {
        format!(
            "{STORAGE_ROOT}/communities/{}/repositories/{}/storage/{}",
            self.community_id, self.repository_id, self.storage_handle_id
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GitRefSnapshot {
    scope: RepositoryStorageScope,
    entity_tag: EntityTag,
    manifest_digest: GitContentDigest,
    manifest: GitRefManifest,
}

impl GitRefSnapshot {
    pub fn manifest_digest(&self) -> &GitContentDigest {
        &self.manifest_digest
    }

    pub fn manifest(&self) -> &GitRefManifest {
        &self.manifest
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GitObjectStoreError {
    #[error("repository does not use hosted NIP-34 object storage")]
    UnsupportedAuthority,
    #[error("repository object storage is unavailable")]
    RepositoryUnavailable,
    #[error("object-store limits are invalid")]
    InvalidLimits,
    #[error("content digest is invalid")]
    InvalidDigest,
    #[error("Git object ID is invalid")]
    InvalidObjectId,
    #[error("Git ref name is invalid")]
    InvalidRefName,
    #[error("Git packed object is invalid")]
    InvalidObject,
    #[error("Git ref manifest is invalid")]
    InvalidManifest,
    #[error("Git object exceeds its configured byte limit")]
    ObjectTooLarge,
    #[error("Git object is unavailable")]
    ObjectNotFound,
    #[error("Git object integrity verification failed")]
    IntegrityMismatch,
    #[error("Git ref state is unavailable")]
    RefsNotFound,
    #[error("Git ref state changed concurrently")]
    ConcurrentRefUpdate,
    #[error("object-store backend is unavailable")]
    BackendUnavailable,
}

pub struct GitObjectStore {
    backend: Arc<dyn GitObjectBackend>,
    scope: RepositoryStorageScope,
    limits: GitObjectStoreLimits,
}

impl GitObjectStore {
    pub fn for_authorized_repository(
        backend: Arc<dyn GitObjectBackend>,
        repository: &HostedRepository,
        limits: GitObjectStoreLimits,
    ) -> Result<Self, GitObjectStoreError> {
        Ok(Self {
            backend,
            scope: RepositoryStorageScope::from_authorized_repository(repository)?,
            limits: limits.validate()?,
        })
    }

    pub async fn put_object(
        &self,
        bytes: Vec<u8>,
    ) -> Result<GitContentDigest, GitObjectStoreError> {
        if bytes.is_empty() {
            return Err(GitObjectStoreError::InvalidObject);
        }
        if u64::try_from(bytes.len()).map_or(true, |length| length > self.limits.max_object_bytes) {
            return Err(GitObjectStoreError::ObjectTooLarge);
        }
        let digest = GitContentDigest::for_bytes(&bytes);
        self.put_immutable(
            self.object_key(&digest),
            bytes,
            "application/x-git-pack",
            &digest,
            self.limits.max_object_bytes,
        )
        .await?;
        Ok(digest)
    }

    pub async fn get_object(
        &self,
        digest: &GitContentDigest,
    ) -> Result<Vec<u8>, GitObjectStoreError> {
        self.get_verified(
            &self.object_key(digest),
            digest,
            self.limits.max_object_bytes,
            GitObjectStoreError::ObjectNotFound,
        )
        .await
    }

    pub async fn read_refs(&self) -> Result<GitRefSnapshot, GitObjectStoreError> {
        let pointer = self
            .backend
            .get(&self.pointer_key(), self.limits.max_pointer_bytes)
            .await
            .map_err(map_backend_error)?
            .ok_or(GitObjectStoreError::RefsNotFound)?;
        let pointer_body: PointerBody = serde_json::from_slice(&pointer.bytes)
            .map_err(|_| GitObjectStoreError::IntegrityMismatch)?;
        if pointer_body.version != STORAGE_FORMAT_VERSION {
            return Err(GitObjectStoreError::IntegrityMismatch);
        }
        let canonical_pointer = serde_json::to_vec(&pointer_body)
            .map_err(|_| GitObjectStoreError::IntegrityMismatch)?;
        if canonical_pointer != pointer.bytes {
            return Err(GitObjectStoreError::IntegrityMismatch);
        }
        let manifest_bytes = self
            .get_verified(
                &self.manifest_key(&pointer_body.manifest),
                &pointer_body.manifest,
                self.limits.max_manifest_bytes,
                GitObjectStoreError::RefsNotFound,
            )
            .await?;
        let manifest = GitRefManifest::from_bytes(&manifest_bytes, self.limits)?;
        if manifest.canonical_bytes()? != manifest_bytes {
            return Err(GitObjectStoreError::InvalidManifest);
        }
        Ok(GitRefSnapshot {
            scope: self.scope.clone(),
            entity_tag: pointer.entity_tag,
            manifest_digest: pointer_body.manifest,
            manifest,
        })
    }

    pub async fn compare_and_swap_refs(
        &self,
        expected: Option<&GitRefSnapshot>,
        manifest: GitRefManifest,
    ) -> Result<GitRefSnapshot, GitObjectStoreError> {
        manifest.validate(self.limits)?;
        match expected {
            Some(expected) => {
                if expected.scope != self.scope
                    || manifest.parent.as_ref() != Some(&expected.manifest_digest)
                {
                    return Err(GitObjectStoreError::ConcurrentRefUpdate);
                }
            }
            None if manifest.parent.is_some() => {
                return Err(GitObjectStoreError::ConcurrentRefUpdate);
            }
            None => {}
        }
        for digest in &manifest.objects {
            self.get_object(digest).await?;
        }
        let manifest_bytes = manifest.canonical_bytes()?;
        if u64::try_from(manifest_bytes.len())
            .map_or(true, |length| length > self.limits.max_manifest_bytes)
        {
            return Err(GitObjectStoreError::InvalidManifest);
        }
        let manifest_digest = GitContentDigest::for_bytes(&manifest_bytes);
        self.put_immutable(
            self.manifest_key(&manifest_digest),
            manifest_bytes,
            "application/json",
            &manifest_digest,
            self.limits.max_manifest_bytes,
        )
        .await?;
        let pointer_bytes = serde_json::to_vec(&PointerBody {
            version: STORAGE_FORMAT_VERSION,
            manifest: manifest_digest.clone(),
        })
        .map_err(|_| GitObjectStoreError::InvalidManifest)?;
        if u64::try_from(pointer_bytes.len())
            .map_or(true, |length| length > self.limits.max_pointer_bytes)
        {
            return Err(GitObjectStoreError::InvalidManifest);
        }
        let condition = expected.map_or(BackendWriteCondition::CreateOnly, |expected| {
            BackendWriteCondition::IfMatch(expected.entity_tag.clone())
        });
        let outcome = self
            .backend
            .put(
                &self.pointer_key(),
                pointer_bytes,
                "application/json",
                condition,
            )
            .await
            .map_err(map_backend_error)?;
        let BackendWriteOutcome::Stored(entity_tag) = outcome else {
            return Err(GitObjectStoreError::ConcurrentRefUpdate);
        };
        Ok(GitRefSnapshot {
            scope: self.scope.clone(),
            entity_tag,
            manifest_digest,
            manifest,
        })
    }

    async fn put_immutable(
        &self,
        key: String,
        bytes: Vec<u8>,
        content_type: &'static str,
        digest: &GitContentDigest,
        max_bytes: u64,
    ) -> Result<(), GitObjectStoreError> {
        let outcome = self
            .backend
            .put(&key, bytes, content_type, BackendWriteCondition::CreateOnly)
            .await
            .map_err(map_backend_error)?;
        if outcome == BackendWriteOutcome::PreconditionFailed {
            self.get_verified(
                &key,
                digest,
                max_bytes,
                GitObjectStoreError::IntegrityMismatch,
            )
            .await?;
        }
        Ok(())
    }

    async fn get_verified(
        &self,
        key: &str,
        expected: &GitContentDigest,
        max_bytes: u64,
        missing_error: GitObjectStoreError,
    ) -> Result<Vec<u8>, GitObjectStoreError> {
        let object = self
            .backend
            .get(key, max_bytes)
            .await
            .map_err(map_backend_error)?
            .ok_or(missing_error)?;
        if GitContentDigest::for_bytes(&object.bytes) != *expected {
            return Err(GitObjectStoreError::IntegrityMismatch);
        }
        Ok(object.bytes)
    }

    fn object_key(&self, digest: &GitContentDigest) -> String {
        format!("{}/objects/{}", self.scope.prefix(), digest.as_str())
    }

    fn manifest_key(&self, digest: &GitContentDigest) -> String {
        format!("{}/manifests/{}", self.scope.prefix(), digest.as_str())
    }

    fn pointer_key(&self) -> String {
        format!("{}/refs/pointer", self.scope.prefix())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PointerBody {
    version: u32,
    manifest: GitContentDigest,
}

fn map_backend_error(error: GitObjectBackendError) -> GitObjectStoreError {
    match error {
        GitObjectBackendError::ObjectTooLarge => GitObjectStoreError::ObjectTooLarge,
        GitObjectBackendError::Unavailable => GitObjectStoreError::BackendUnavailable,
    }
}

fn is_lower_hex(value: &str, bytes: usize) -> bool {
    value.len() == bytes
        && value
            .chars()
            .all(|character| character.is_ascii_digit() || matches!(character, 'a'..='f'))
}

fn is_safe_ref_name(value: &str) -> bool {
    value.len() <= 1024
        && value.starts_with("refs/")
        && !value.ends_with('/')
        && !value.ends_with('.')
        && !value.contains("..")
        && !value.contains("//")
        && value.split('/').all(|component| {
            !component.is_empty() && !component.starts_with('.') && !component.ends_with(".lock")
        })
        && value.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '/' | '_' | '.' | '-')
        })
}
