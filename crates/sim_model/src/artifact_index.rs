use crate::parser_limits::{ParserLimitError, ParserLimits};
use cap_std::fs::{Dir, OpenOptions as CapabilityOpenOptions};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sim_tensor::CancellationToken;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, Metadata},
    io::{Read, Seek, Write},
    path::{Component, Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::UNIX_EPOCH,
};

pub const ARTIFACT_INDEX_VERSION: u32 = 1;
const PRIVATE_ARTIFACT_CAPTURE_CHUNK_BYTES: usize = 64 * 1024;
static PRIVATE_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapturedPrivateArtifact {
    bytes: Vec<u8>,
    digest_sha256: String,
}

impl CapturedPrivateArtifact {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn digest_sha256(&self) -> &str {
        &self.digest_sha256
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ArtifactKey {
    pub root_id: String,
    pub relative_path: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArtifactWritePolicy {
    Reject,
    Replace,
}

impl ArtifactKey {
    pub fn new(
        root_id: impl Into<String>,
        relative_path: impl Into<PathBuf>,
    ) -> Result<Self, ArtifactIndexError> {
        let root_id = root_id.into();
        if root_id.trim().is_empty() {
            return Err(ArtifactIndexError::InvalidRootId(root_id));
        }
        let limits = ParserLimits::default();
        limits.check(
            "artifact root ID bytes",
            u64::try_from(root_id.len()).unwrap_or(u64::MAX),
            limits.maximum_name_bytes,
        )?;
        let relative_path = normalize_relative_path_with_limits(&relative_path.into(), &limits)?;
        Ok(Self {
            root_id,
            relative_path,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactRoot {
    id: String,
    namespace: String,
    canonical_path: PathBuf,
    approved_extensions: BTreeSet<String>,
    imported: bool,
    #[serde(default)]
    approved_relative_path: Option<PathBuf>,
    #[serde(skip)]
    trusted_directory: Option<Arc<Dir>>,
}

impl PartialEq for ArtifactRoot {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.namespace == other.namespace
            && self.canonical_path == other.canonical_path
            && self.approved_extensions == other.approved_extensions
            && self.imported == other.imported
            && self.approved_relative_path == other.approved_relative_path
    }
}

impl Eq for ArtifactRoot {}

impl ArtifactRoot {
    pub fn canonical(
        id: impl Into<String>,
        namespace: impl Into<String>,
        path: &Path,
        extensions: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, ArtifactIndexError> {
        Self::new(id, namespace, path, extensions, false, None)
    }

    pub fn canonical_with_path_identity(
        id_prefix: &str,
        namespace: impl Into<String>,
        path: &Path,
        extensions: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, ArtifactIndexError> {
        if id_prefix.trim().is_empty() {
            return Err(ArtifactIndexError::InvalidRootId(id_prefix.to_owned()));
        }
        let mut root = Self::new(
            "canonical-path-identity",
            namespace,
            path,
            extensions,
            false,
            None,
        )?;
        let digest = Sha256::digest(root.canonical_path.as_os_str().as_encoded_bytes());
        root.id = format!("{id_prefix}-{:x}", digest);
        ParserLimits::default().check(
            "artifact root ID bytes",
            u64::try_from(root.id.len()).unwrap_or(u64::MAX),
            ParserLimits::default().maximum_name_bytes,
        )?;
        Ok(root)
    }

    pub fn approved_import(
        id: impl Into<String>,
        namespace: impl Into<String>,
        path: &Path,
    ) -> Result<Self, ArtifactIndexError> {
        let parent = path
            .parent()
            .ok_or_else(|| ArtifactIndexError::UnsafePath {
                path: path.to_path_buf(),
                reason: "approved import has no parent directory".to_owned(),
            })?;
        let extension = path
            .extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase())
            .into_iter();
        let approved_relative_path =
            path.file_name()
                .map(PathBuf::from)
                .ok_or_else(|| ArtifactIndexError::UnsafePath {
                    path: path.to_path_buf(),
                    reason: "approved import has no filename".to_owned(),
                })?;
        Self::new(
            id,
            namespace,
            parent,
            extension,
            true,
            Some(approved_relative_path),
        )
    }

    fn new(
        id: impl Into<String>,
        namespace: impl Into<String>,
        path: &Path,
        extensions: impl IntoIterator<Item = impl Into<String>>,
        imported: bool,
        approved_relative_path: Option<PathBuf>,
    ) -> Result<Self, ArtifactIndexError> {
        Self::new_with_admission_hook(
            id,
            namespace,
            path,
            extensions,
            imported,
            approved_relative_path,
            || {},
        )
    }

    fn new_with_admission_hook(
        id: impl Into<String>,
        namespace: impl Into<String>,
        path: &Path,
        extensions: impl IntoIterator<Item = impl Into<String>>,
        imported: bool,
        approved_relative_path: Option<PathBuf>,
        after_initial_metadata: impl FnOnce(),
    ) -> Result<Self, ArtifactIndexError> {
        let id = id.into();
        let namespace = namespace.into();
        if id.trim().is_empty() {
            return Err(ArtifactIndexError::InvalidRootId(id));
        }
        if namespace.trim().is_empty() {
            return Err(ArtifactIndexError::InvalidNamespace(namespace));
        }
        let limits = ParserLimits::default();
        limits.check(
            "artifact root ID bytes",
            u64::try_from(id.len()).unwrap_or(u64::MAX),
            limits.maximum_name_bytes,
        )?;
        limits.check(
            "artifact namespace bytes",
            u64::try_from(namespace.len()).unwrap_or(u64::MAX),
            limits.maximum_name_bytes,
        )?;
        reject_symbolic_link_components(path)?;
        let metadata = fs::symlink_metadata(path).map_err(|error| ArtifactIndexError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
        if metadata.file_type().is_symlink() {
            return Err(ArtifactIndexError::SymbolicLink(path.to_path_buf()));
        }
        if !metadata.is_dir() {
            return Err(ArtifactIndexError::NotDirectory(path.to_path_buf()));
        }
        after_initial_metadata();
        let canonical_path = fs::canonicalize(path).map_err(|error| ArtifactIndexError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
        reject_symbolic_link_components(&canonical_path)?;
        let trusted_directory =
            Dir::open_ambient_dir(&canonical_path, cap_std::ambient_authority()).map_err(
                |error| ArtifactIndexError::Io {
                    path: canonical_path.clone(),
                    message: error.to_string(),
                },
            )?;
        let opened_metadata = trusted_directory
            .try_clone()
            .and_then(|directory| directory.into_std_file().metadata())
            .map_err(|error| ArtifactIndexError::Io {
                path: canonical_path.clone(),
                message: error.to_string(),
            })?;
        let current_metadata =
            fs::metadata(&canonical_path).map_err(|error| ArtifactIndexError::Io {
                path: canonical_path.clone(),
                message: error.to_string(),
            })?;
        if !same_file_identity(&metadata, &opened_metadata)
            || !same_file_identity(&metadata, &current_metadata)
            || !same_file_identity(&opened_metadata, &current_metadata)
        {
            return Err(ArtifactIndexError::ChangedDuringScan(canonical_path));
        }
        let approved_extensions = extensions
            .into_iter()
            .map(Into::into)
            .map(|value: String| value.trim_start_matches('.').to_ascii_lowercase())
            .filter(|value| !value.is_empty())
            .collect();
        Ok(Self {
            id,
            namespace,
            canonical_path,
            approved_extensions,
            imported,
            approved_relative_path,
            trusted_directory: Some(Arc::new(trusted_directory)),
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn canonical_path(&self) -> &Path {
        &self.canonical_path
    }

    pub fn approved_extensions(&self) -> &BTreeSet<String> {
        &self.approved_extensions
    }

    pub fn is_imported(&self) -> bool {
        self.imported
    }

    pub fn approved_relative_path(&self) -> Option<&Path> {
        self.approved_relative_path.as_deref()
    }

    pub fn list_contained_regular_files_recursive(
        &self,
        maximum_entries: usize,
        cancellation: &CancellationToken,
    ) -> Result<Vec<PathBuf>, ArtifactIndexError> {
        check_cancelled(cancellation)?;
        if self.imported {
            return Err(ArtifactIndexError::UnsafePath {
                path: self.canonical_path.clone(),
                reason: "approved import roots cannot be traversed recursively".to_owned(),
            });
        }
        if maximum_entries == 0 {
            return Err(ParserLimitError::Zero("contained recursive entry limit").into());
        }

        let limits = ParserLimits::default();
        validate_root_snapshot(self, &limits)?;
        let trusted_directory = self.trusted_directory.as_deref().ok_or_else(|| {
            ArtifactIndexError::InvalidSnapshot(
                "artifact root has not been admitted by the trusted root constructor".to_owned(),
            )
        })?;
        let initial_directory =
            trusted_directory
                .try_clone()
                .map_err(|error| ArtifactIndexError::Io {
                    path: self.canonical_path.clone(),
                    message: error.to_string(),
                })?;
        let maximum_entries = u64::try_from(maximum_entries).unwrap_or(u64::MAX);
        let mut visited_entries = 0_u64;
        let mut pending = vec![(initial_directory, PathBuf::new())];
        let mut files = Vec::new();

        while let Some((directory, relative_directory)) = pending.pop() {
            check_cancelled(cancellation)?;
            let diagnostic_directory = self.canonical_path.join(&relative_directory);
            let entries = directory
                .entries()
                .map_err(|error| ArtifactIndexError::Io {
                    path: diagnostic_directory.clone(),
                    message: error.to_string(),
                })?;
            let mut sorted_entries = Vec::new();
            for entry in entries {
                check_cancelled(cancellation)?;
                visited_entries =
                    visited_entries
                        .checked_add(1)
                        .ok_or(ParserLimitError::Exceeded {
                            kind: "contained recursive entries",
                            actual: u64::MAX,
                            maximum: maximum_entries,
                        })?;
                limits.check(
                    "contained recursive entries",
                    visited_entries,
                    maximum_entries,
                )?;
                let entry = entry.map_err(|error| ArtifactIndexError::Io {
                    path: diagnostic_directory.clone(),
                    message: error.to_string(),
                })?;
                sorted_entries.try_reserve(1).map_err(|_| {
                    ArtifactIndexError::AllocationFailed(diagnostic_directory.clone())
                })?;
                sorted_entries.push(entry);
            }
            sorted_entries.sort_by_key(cap_std::fs::DirEntry::file_name);

            for entry in sorted_entries {
                check_cancelled(cancellation)?;
                let file_name = entry.file_name();
                let relative = relative_directory.join(&file_name);
                let path = self.canonical_path.join(&relative);
                let component =
                    file_name
                        .to_str()
                        .ok_or_else(|| ArtifactIndexError::UnsafePath {
                            path: path.clone(),
                            reason: "artifact paths must be valid UTF-8".to_owned(),
                        })?;
                if component.is_empty()
                    || component == "."
                    || component == ".."
                    || component.contains('/')
                    || component.contains('\\')
                {
                    return Err(ArtifactIndexError::UnsafePath {
                        path,
                        reason:
                            "artifact path component is not portable across supported platforms"
                                .to_owned(),
                    });
                }
                limits.check(
                    "artifact path component bytes",
                    u64::try_from(component.len()).unwrap_or(u64::MAX),
                    limits.maximum_name_bytes,
                )?;
                validate_portable_component(&relative, component)?;
                let relative = normalize_relative_path_with_limits(&relative, &limits)?;
                let path = self.canonical_path.join(&relative);
                let metadata = directory.symlink_metadata(&file_name).map_err(|error| {
                    ArtifactIndexError::Io {
                        path: path.clone(),
                        message: error.to_string(),
                    }
                })?;
                if metadata.file_type().is_symlink() {
                    return Err(ArtifactIndexError::SymbolicLink(path));
                }
                if metadata.is_dir() {
                    let child = entry.open_dir().map_err(|error| ArtifactIndexError::Io {
                        path: path.clone(),
                        message: error.to_string(),
                    })?;
                    let opened = child
                        .dir_metadata()
                        .map_err(|error| ArtifactIndexError::Io {
                            path: path.clone(),
                            message: error.to_string(),
                        })?;
                    let current = directory.symlink_metadata(&file_name).map_err(|error| {
                        ArtifactIndexError::Io {
                            path: path.clone(),
                            message: error.to_string(),
                        }
                    })?;
                    if !same_capability_file_identity(&metadata, &opened)
                        || !same_capability_file_identity(&opened, &current)
                    {
                        return Err(ArtifactIndexError::ChangedDuringScan(path));
                    }
                    pending.try_reserve(1).map_err(|_| {
                        ArtifactIndexError::AllocationFailed(self.canonical_path.join(&relative))
                    })?;
                    pending.push((child, relative));
                    continue;
                }
                if !metadata.is_file() {
                    return Err(ArtifactIndexError::UnsafePath {
                        path,
                        reason: "contained recursive entry is not a regular file or directory"
                            .to_owned(),
                    });
                }
                let file = entry.open().map_err(|error| ArtifactIndexError::Io {
                    path: path.clone(),
                    message: error.to_string(),
                })?;
                let opened = file.metadata().map_err(|error| ArtifactIndexError::Io {
                    path: path.clone(),
                    message: error.to_string(),
                })?;
                let current = directory.symlink_metadata(&file_name).map_err(|error| {
                    ArtifactIndexError::Io {
                        path: path.clone(),
                        message: error.to_string(),
                    }
                })?;
                if !opened.is_file()
                    || !same_capability_file_identity(&metadata, &opened)
                    || !same_capability_file_identity(&opened, &current)
                {
                    return Err(ArtifactIndexError::ChangedDuringScan(path));
                }
                files
                    .try_reserve(1)
                    .map_err(|_| ArtifactIndexError::AllocationFailed(path))?;
                files.push(relative);
            }
        }

        files.sort();
        check_cancelled(cancellation)?;
        Ok(files)
    }

    pub fn key(
        &self,
        relative_path: impl Into<PathBuf>,
    ) -> Result<ArtifactKey, ArtifactIndexError> {
        let key = ArtifactKey::new(self.id.clone(), relative_path)?;
        self.validate_key_scope(&key)?;
        Ok(key)
    }

    pub fn resolve_existing(
        &self,
        relative_path: impl AsRef<Path>,
    ) -> Result<PathBuf, ArtifactIndexError> {
        let key = self.key(relative_path.as_ref().to_path_buf())?;
        self.resolve_key(&key, PathResolution::Existing)
    }

    pub fn resolve_for_create(
        &self,
        relative_path: impl AsRef<Path>,
    ) -> Result<PathBuf, ArtifactIndexError> {
        let key = self.key(relative_path.as_ref().to_path_buf())?;
        self.resolve_key(&key, PathResolution::ForCreate)
    }

    pub fn resolve_for_create_with_parents(
        &self,
        relative_path: impl AsRef<Path>,
    ) -> Result<PathBuf, ArtifactIndexError> {
        let key = self.key(relative_path.as_ref().to_path_buf())?;
        let (parent, file_name) = self.open_capability_parent(&key.relative_path, true)?;
        if let Ok(metadata) = parent.symlink_metadata(&file_name)
            && metadata.file_type().is_symlink()
        {
            return Err(ArtifactIndexError::SymbolicLink(
                self.canonical_path.join(&key.relative_path),
            ));
        }
        Ok(self.canonical_path.join(key.relative_path))
    }

    pub fn read_private_file(
        &self,
        relative_path: impl AsRef<Path>,
        maximum_bytes: usize,
    ) -> Result<Option<Vec<u8>>, ArtifactIndexError> {
        if maximum_bytes == 0 {
            return Err(ArtifactIndexError::InvalidSnapshot(
                "private artifact byte limit is zero".to_owned(),
            ));
        }
        let relative_path =
            normalize_relative_path_with_limits(relative_path.as_ref(), &ParserLimits::default())?;
        let path = self.canonical_path.join(&relative_path);
        let (parent, file_name) = match self.open_capability_parent(&relative_path, false) {
            Ok(value) => value,
            Err(ArtifactIndexError::Missing(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
        let before = match parent.symlink_metadata(&file_name) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(ArtifactIndexError::Io {
                    path,
                    message: error.to_string(),
                });
            }
        };
        if before.file_type().is_symlink() {
            return Err(ArtifactIndexError::SymbolicLink(path));
        }
        if !before.is_file() {
            return Err(ArtifactIndexError::UnsafePath {
                path,
                reason: "private artifact is not a regular file".to_owned(),
            });
        }
        let file = match parent.open(&file_name) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(ArtifactIndexError::Io {
                    path,
                    message: error.to_string(),
                });
            }
        };
        let opened = file.metadata().map_err(|error| ArtifactIndexError::Io {
            path: path.clone(),
            message: error.to_string(),
        })?;
        if !opened.is_file() || !same_capability_file_identity(&before, &opened) {
            return Err(ArtifactIndexError::UnsafePath {
                path,
                reason: "private artifact changed while it was opened".to_owned(),
            });
        }
        let opened_length = usize::try_from(opened.len()).unwrap_or(usize::MAX);
        if opened_length > maximum_bytes {
            return Err(ArtifactIndexError::InvalidSnapshot(format!(
                "private artifact contains {opened_length} bytes, exceeding {maximum_bytes}"
            )));
        }
        let current = parent
            .metadata(&file_name)
            .map_err(|error| ArtifactIndexError::Io {
                path: path.clone(),
                message: error.to_string(),
            })?;
        if !same_capability_file_identity(&opened, &current) {
            return Err(ArtifactIndexError::ChangedDuringScan(path));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(opened_length)
            .map_err(|_| ArtifactIndexError::AllocationFailed(path.clone()))?;
        file.take(
            u64::try_from(maximum_bytes)
                .unwrap_or(u64::MAX)
                .saturating_add(1),
        )
        .read_to_end(&mut bytes)
        .map_err(|error| ArtifactIndexError::Io {
            path: path.clone(),
            message: error.to_string(),
        })?;
        let after = parent
            .metadata(&file_name)
            .map_err(|error| ArtifactIndexError::Io {
                path: path.clone(),
                message: error.to_string(),
            })?;
        if bytes.len() != opened_length || !same_capability_file_identity(&opened, &after) {
            return Err(ArtifactIndexError::ChangedDuringScan(path));
        }
        Ok(Some(bytes))
    }

    pub fn capture_private_file(
        &self,
        relative_path: impl AsRef<Path>,
        maximum_bytes: usize,
        cancellation: &CancellationToken,
    ) -> Result<Option<CapturedPrivateArtifact>, ArtifactIndexError> {
        self.capture_private_file_with_hook(
            relative_path.as_ref(),
            maximum_bytes,
            cancellation,
            |_| {},
        )
    }

    fn capture_private_file_with_hook(
        &self,
        relative_path: &Path,
        maximum_bytes: usize,
        cancellation: &CancellationToken,
        mut after_chunk: impl FnMut(usize),
    ) -> Result<Option<CapturedPrivateArtifact>, ArtifactIndexError> {
        check_cancelled(cancellation)?;
        if maximum_bytes == 0 {
            return Err(ArtifactIndexError::InvalidSnapshot(
                "private artifact byte limit is zero".to_owned(),
            ));
        }
        let relative_path =
            normalize_relative_path_with_limits(relative_path, &ParserLimits::default())?;
        self.validate_key_scope(&ArtifactKey {
            root_id: self.id.clone(),
            relative_path: relative_path.clone(),
        })?;
        let path = self.canonical_path.join(&relative_path);
        let (parent, file_name) = match self.open_capability_parent(&relative_path, false) {
            Ok(value) => value,
            Err(ArtifactIndexError::Missing(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
        let before = match parent.symlink_metadata(&file_name) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(ArtifactIndexError::Io {
                    path,
                    message: error.to_string(),
                });
            }
        };
        if before.file_type().is_symlink() {
            return Err(ArtifactIndexError::SymbolicLink(path));
        }
        if !before.is_file() {
            return Err(ArtifactIndexError::UnsafePath {
                path,
                reason: "private artifact is not a regular file".to_owned(),
            });
        }
        let mut file = match parent.open(&file_name) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(ArtifactIndexError::Io {
                    path,
                    message: error.to_string(),
                });
            }
        };
        let opened = file.metadata().map_err(|error| ArtifactIndexError::Io {
            path: path.clone(),
            message: error.to_string(),
        })?;
        let current =
            parent
                .symlink_metadata(&file_name)
                .map_err(|error| ArtifactIndexError::Io {
                    path: path.clone(),
                    message: error.to_string(),
                })?;
        if !opened.is_file()
            || !same_capability_file_identity(&before, &opened)
            || !same_capability_file_identity(&opened, &current)
        {
            return Err(ArtifactIndexError::ChangedDuringScan(path));
        }
        let opened_length = usize::try_from(opened.len()).unwrap_or(usize::MAX);
        if opened_length > maximum_bytes {
            return Err(ArtifactIndexError::InvalidSnapshot(format!(
                "private artifact contains {opened_length} bytes, exceeding {maximum_bytes}"
            )));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(opened_length)
            .map_err(|_| ArtifactIndexError::AllocationFailed(path.clone()))?;
        let mut digest = Sha256::new();
        let mut chunk = [0_u8; PRIVATE_ARTIFACT_CAPTURE_CHUNK_BYTES];
        loop {
            check_cancelled(cancellation)?;
            let read = file
                .read(&mut chunk)
                .map_err(|error| ArtifactIndexError::Io {
                    path: path.clone(),
                    message: error.to_string(),
                })?;
            if read == 0 {
                break;
            }
            let captured_length = bytes.len().checked_add(read).ok_or_else(|| {
                ArtifactIndexError::InvalidSnapshot(
                    "private artifact length overflowed during capture".to_owned(),
                )
            })?;
            if captured_length > maximum_bytes {
                return Err(ArtifactIndexError::InvalidSnapshot(format!(
                    "private artifact exceeds {maximum_bytes} bytes during capture"
                )));
            }
            bytes
                .try_reserve(read)
                .map_err(|_| ArtifactIndexError::AllocationFailed(path.clone()))?;
            bytes.extend_from_slice(&chunk[..read]);
            digest.update(&chunk[..read]);
            after_chunk(bytes.len());
        }
        check_cancelled(cancellation)?;
        let after =
            parent
                .symlink_metadata(&file_name)
                .map_err(|error| ArtifactIndexError::Io {
                    path: path.clone(),
                    message: error.to_string(),
                })?;
        let opened_after = file.metadata().map_err(|error| ArtifactIndexError::Io {
            path: path.clone(),
            message: error.to_string(),
        })?;
        if bytes.len() != opened_length
            || opened_after.len() != opened.len()
            || !same_capability_file_identity(&opened, &opened_after)
            || !same_capability_file_identity(&opened, &after)
        {
            return Err(ArtifactIndexError::ChangedDuringScan(path));
        }
        Ok(Some(CapturedPrivateArtifact {
            bytes,
            digest_sha256: format!("{:x}", digest.finalize()),
        }))
    }

    pub fn write_private_file(
        &self,
        relative_path: impl AsRef<Path>,
        bytes: &[u8],
    ) -> Result<(), ArtifactIndexError> {
        self.write_contained_file(relative_path, bytes, ArtifactWritePolicy::Replace)
    }

    pub fn remove_contained_file(
        &self,
        relative_path: impl AsRef<Path>,
    ) -> Result<bool, ArtifactIndexError> {
        let relative_path =
            normalize_relative_path_with_limits(relative_path.as_ref(), &ParserLimits::default())?;
        self.validate_key_scope(&ArtifactKey {
            root_id: self.id.clone(),
            relative_path: relative_path.clone(),
        })?;
        let path = self.canonical_path.join(&relative_path);
        let (parent, file_name) = match self.open_capability_parent(&relative_path, false) {
            Ok(value) => value,
            Err(ArtifactIndexError::Missing(_)) => return Ok(false),
            Err(error) => return Err(error),
        };
        let metadata = match parent.symlink_metadata(&file_name) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(error) => {
                return Err(ArtifactIndexError::Io {
                    path,
                    message: error.to_string(),
                });
            }
        };
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(ArtifactIndexError::UnsafePath {
                path,
                reason: "contained artifact removal target is not a regular file".to_owned(),
            });
        }
        parent
            .remove_file(&file_name)
            .and_then(|()| sync_capability_directory(&parent))
            .map_err(|error| ArtifactIndexError::Io {
                path,
                message: error.to_string(),
            })?;
        Ok(true)
    }

    pub fn contained_file_exists(
        &self,
        relative_path: impl AsRef<Path>,
    ) -> Result<bool, ArtifactIndexError> {
        let relative_path =
            normalize_relative_path_with_limits(relative_path.as_ref(), &ParserLimits::default())?;
        self.validate_key_scope(&ArtifactKey {
            root_id: self.id.clone(),
            relative_path: relative_path.clone(),
        })?;
        let path = self.canonical_path.join(&relative_path);
        let (parent, file_name) = match self.open_capability_parent(&relative_path, false) {
            Ok(value) => value,
            Err(ArtifactIndexError::Missing(_)) => return Ok(false),
            Err(error) => return Err(error),
        };
        match parent.symlink_metadata(&file_name) {
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
                Err(ArtifactIndexError::UnsafePath {
                    path,
                    reason: "contained artifact is not a regular file".to_owned(),
                })
            }
            Ok(_) => Ok(true),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(ArtifactIndexError::Io {
                path,
                message: error.to_string(),
            }),
        }
    }

    pub fn contained_file_digest(
        &self,
        relative_path: impl AsRef<Path>,
        cancellation: &CancellationToken,
    ) -> Result<Option<(String, u64)>, ArtifactIndexError> {
        check_cancelled(cancellation)?;
        let relative_path =
            normalize_relative_path_with_limits(relative_path.as_ref(), &ParserLimits::default())?;
        self.validate_key_scope(&ArtifactKey {
            root_id: self.id.clone(),
            relative_path: relative_path.clone(),
        })?;
        let path = self.canonical_path.join(&relative_path);
        let (parent, file_name) = match self.open_capability_parent(&relative_path, false) {
            Ok(value) => value,
            Err(ArtifactIndexError::Missing(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
        let before = match parent.symlink_metadata(&file_name) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(ArtifactIndexError::Io {
                    path,
                    message: error.to_string(),
                });
            }
        };
        if before.file_type().is_symlink() || !before.is_file() {
            return Err(ArtifactIndexError::UnsafePath {
                path,
                reason: "contained artifact is not a regular file".to_owned(),
            });
        }
        let mut file = parent
            .open(&file_name)
            .map_err(|error| ArtifactIndexError::Io {
                path: path.clone(),
                message: error.to_string(),
            })?;
        let opened = file.metadata().map_err(|error| ArtifactIndexError::Io {
            path: path.clone(),
            message: error.to_string(),
        })?;
        if !same_capability_file_identity(&before, &opened) {
            return Err(ArtifactIndexError::ChangedDuringScan(path));
        }
        let (digest, after) = hash_stable_capability_file(&mut file, &path, &opened, cancellation)?;
        let current = parent
            .metadata(&file_name)
            .map_err(|error| ArtifactIndexError::Io {
                path: path.clone(),
                message: error.to_string(),
            })?;
        if !same_capability_file_identity(&after, &current) {
            return Err(ArtifactIndexError::ChangedDuringScan(path));
        }
        Ok(Some((digest, after.len())))
    }

    pub fn list_direct_contained_files(
        &self,
        relative_directory: impl AsRef<Path>,
    ) -> Result<Vec<PathBuf>, ArtifactIndexError> {
        self.list_direct_contained_files_with_policy(relative_directory.as_ref(), true)
    }

    pub fn list_direct_contained_regular_files(
        &self,
        relative_directory: impl AsRef<Path>,
    ) -> Result<Vec<PathBuf>, ArtifactIndexError> {
        self.list_direct_contained_files_with_policy(relative_directory.as_ref(), false)
    }

    fn list_direct_contained_files_with_policy(
        &self,
        relative_directory: &Path,
        reject_directories: bool,
    ) -> Result<Vec<PathBuf>, ArtifactIndexError> {
        if self.imported {
            return Err(ArtifactIndexError::UnsafePath {
                path: relative_directory.to_path_buf(),
                reason: "approved imports cannot enumerate their parent root".to_owned(),
            });
        }
        let probe = if relative_directory.as_os_str().is_empty() {
            PathBuf::from(".zed-directory-probe")
        } else {
            normalize_relative_path_with_limits(relative_directory, &ParserLimits::default())?
                .join(".zed-directory-probe")
        };
        let (directory, _) = match self.open_capability_parent(&probe, false) {
            Ok(value) => value,
            Err(ArtifactIndexError::Missing(_)) => return Ok(Vec::new()),
            Err(error) => return Err(error),
        };
        let diagnostic_directory = self.canonical_path.join(relative_directory);
        let entries = directory
            .entries()
            .map_err(|error| ArtifactIndexError::Io {
                path: diagnostic_directory.clone(),
                message: error.to_string(),
            })?;
        let mut files = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|error| ArtifactIndexError::Io {
                path: diagnostic_directory.clone(),
                message: error.to_string(),
            })?;
            let file_name = entry.file_name();
            let path = diagnostic_directory.join(&file_name);
            let metadata =
                directory
                    .symlink_metadata(&file_name)
                    .map_err(|error| ArtifactIndexError::Io {
                        path: path.clone(),
                        message: error.to_string(),
                    })?;
            if metadata.file_type().is_symlink() {
                return Err(ArtifactIndexError::UnsafePath {
                    path,
                    reason: "contained directory entry is not a regular file".to_owned(),
                });
            }
            if !metadata.is_file() {
                if reject_directories {
                    return Err(ArtifactIndexError::UnsafePath {
                        path,
                        reason: "contained directory entry is not a regular file".to_owned(),
                    });
                }
                continue;
            }
            files
                .try_reserve(1)
                .map_err(|_| ArtifactIndexError::AllocationFailed(path.clone()))?;
            files.push(relative_directory.join(file_name));
        }
        files.sort();
        Ok(files)
    }

    pub fn move_contained_file_to(
        &self,
        source_relative_path: impl AsRef<Path>,
        destination_root: &Self,
        destination_relative_path: impl AsRef<Path>,
        policy: ArtifactWritePolicy,
    ) -> Result<(), ArtifactIndexError> {
        let cancellation = CancellationToken::default();
        let (expected_sha256, expected_size) = self
            .contained_file_digest(source_relative_path.as_ref(), &cancellation)?
            .ok_or_else(|| {
                ArtifactIndexError::Missing(ArtifactKey {
                    root_id: self.id.clone(),
                    relative_path: source_relative_path.as_ref().to_path_buf(),
                })
            })?;
        self.move_verified_contained_file_to_inner(
            source_relative_path.as_ref(),
            destination_root,
            destination_relative_path.as_ref(),
            policy,
            &expected_sha256,
            expected_size,
            &cancellation,
            || {},
            || {},
            || {},
            || {},
        )
    }

    pub fn move_verified_contained_file_to(
        &self,
        source_relative_path: impl AsRef<Path>,
        destination_root: &Self,
        destination_relative_path: impl AsRef<Path>,
        policy: ArtifactWritePolicy,
        expected_sha256: &str,
        expected_size: u64,
        cancellation: &CancellationToken,
    ) -> Result<(), ArtifactIndexError> {
        self.move_verified_contained_file_to_inner(
            source_relative_path.as_ref(),
            destination_root,
            destination_relative_path.as_ref(),
            policy,
            expected_sha256,
            expected_size,
            cancellation,
            || {},
            || {},
            || {},
            || {},
        )
    }

    fn move_verified_contained_file_to_inner(
        &self,
        source_relative_path: &Path,
        destination_root: &Self,
        destination_relative_path: &Path,
        policy: ArtifactWritePolicy,
        expected_sha256: &str,
        expected_size: u64,
        cancellation: &CancellationToken,
        after_source_validation: impl FnOnce(),
        before_temporary_verification: impl FnOnce(),
        before_name_publication: impl FnOnce(),
        after_name_publication: impl FnOnce(),
    ) -> Result<(), ArtifactIndexError> {
        check_cancelled(cancellation)?;
        validate_root_snapshot(self, &ParserLimits::default())?;
        validate_root_snapshot(destination_root, &ParserLimits::default())?;
        let source_relative_path =
            normalize_relative_path_with_limits(source_relative_path, &ParserLimits::default())?;
        let destination_relative_path = normalize_relative_path_with_limits(
            destination_relative_path,
            &ParserLimits::default(),
        )?;
        self.validate_key_scope(&ArtifactKey {
            root_id: self.id.clone(),
            relative_path: source_relative_path.clone(),
        })?;
        destination_root.validate_key_scope(&ArtifactKey {
            root_id: destination_root.id.clone(),
            relative_path: destination_relative_path.clone(),
        })?;
        let source_path = self.canonical_path.join(&source_relative_path);
        let destination_path = destination_root
            .canonical_path
            .join(&destination_relative_path);
        let (source_parent, source_name) =
            self.open_capability_parent(&source_relative_path, false)?;
        let source_metadata = source_parent
            .symlink_metadata(&source_name)
            .map_err(|error| ArtifactIndexError::Io {
                path: source_path.clone(),
                message: error.to_string(),
            })?;
        if source_metadata.file_type().is_symlink() || !source_metadata.is_file() {
            return Err(ArtifactIndexError::UnsafePath {
                path: source_path,
                reason: "contained artifact move source is not a regular file".to_owned(),
            });
        }
        let mut source_file =
            source_parent
                .open(&source_name)
                .map_err(|error| ArtifactIndexError::Io {
                    path: source_path.clone(),
                    message: error.to_string(),
                })?;
        let opened_source = source_file
            .metadata()
            .map_err(|error| ArtifactIndexError::Io {
                path: source_path.clone(),
                message: error.to_string(),
            })?;
        if !same_capability_file_identity(&source_metadata, &opened_source) {
            return Err(ArtifactIndexError::ChangedDuringScan(source_path));
        }
        let (actual_sha256, stable_source) = hash_stable_capability_file(
            &mut source_file,
            &source_path,
            &opened_source,
            cancellation,
        )?;
        if actual_sha256 != expected_sha256 || stable_source.len() != expected_size {
            return Err(ArtifactIndexError::ChangedDuringScan(source_path));
        }
        after_source_validation();
        source_file
            .seek(std::io::SeekFrom::Start(0))
            .map_err(|error| ArtifactIndexError::Io {
                path: source_path.clone(),
                message: error.to_string(),
            })?;
        let (destination_parent, destination_name) =
            destination_root.open_capability_parent(&destination_relative_path, true)?;
        match destination_parent.symlink_metadata(&destination_name) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(ArtifactIndexError::SymbolicLink(destination_path));
            }
            Ok(_) if policy == ArtifactWritePolicy::Reject => {
                return Err(ArtifactIndexError::AlreadyExists(destination_path));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(ArtifactIndexError::Io {
                    path: destination_path,
                    message: error.to_string(),
                });
            }
        }
        let destination_name_text =
            destination_name
                .to_str()
                .ok_or_else(|| ArtifactIndexError::UnsafePath {
                    path: destination_path.clone(),
                    reason: "contained artifact move destination filename is invalid".to_owned(),
                })?;
        let sequence = PRIVATE_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary_name = PathBuf::from(format!(
            "{destination_name_text}.move.{}.{}.tmp",
            std::process::id(),
            sequence
        ));
        let temporary_path = destination_path.with_file_name(&temporary_name);
        let temporary_identity = std::cell::RefCell::new(None);
        let result = (|| {
            let mut options = CapabilityOpenOptions::new();
            options.create_new(true).read(true).write(true);
            #[cfg(target_os = "windows")]
            {
                use cap_std::fs::OpenOptionsExt;
                use windows::Win32::{
                    Foundation::{GENERIC_READ, GENERIC_WRITE},
                    Storage::FileSystem::{
                        DELETE, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
                    },
                };
                options
                    .access_mode(GENERIC_READ.0 | GENERIC_WRITE.0 | DELETE.0)
                    .share_mode(FILE_SHARE_READ.0 | FILE_SHARE_WRITE.0 | FILE_SHARE_DELETE.0);
            }
            let mut temporary_file = destination_parent
                .open_with(&temporary_name, &options)
                .map_err(|error| ArtifactIndexError::Io {
                    path: temporary_path.clone(),
                    message: error.to_string(),
                })?;
            let created_temporary =
                temporary_file
                    .metadata()
                    .map_err(|error| ArtifactIndexError::Io {
                        path: temporary_path.clone(),
                        message: error.to_string(),
                    })?;
            let original_permissions = created_temporary.permissions();
            temporary_identity.replace(Some(created_temporary));
            let mut copied = 0_u64;
            let mut copied_hasher = Sha256::new();
            let mut buffer = [0_u8; 64 * 1024];
            loop {
                check_cancelled(cancellation)?;
                let read =
                    source_file
                        .read(&mut buffer)
                        .map_err(|error| ArtifactIndexError::Io {
                            path: source_path.clone(),
                            message: error.to_string(),
                        })?;
                if read == 0 {
                    break;
                }
                let bytes = buffer
                    .get(..read)
                    .ok_or_else(|| ArtifactIndexError::ChangedDuringScan(source_path.clone()))?;
                temporary_file
                    .write_all(bytes)
                    .map_err(|error| ArtifactIndexError::Io {
                        path: temporary_path.clone(),
                        message: error.to_string(),
                    })?;
                copied_hasher.update(bytes);
                let read = u64::try_from(read)
                    .map_err(|_| ArtifactIndexError::ChangedDuringScan(source_path.clone()))?;
                copied = copied
                    .checked_add(read)
                    .ok_or_else(|| ArtifactIndexError::ChangedDuringScan(source_path.clone()))?;
            }
            if copied != expected_size || hex_digest(copied_hasher.finalize()) != expected_sha256 {
                return Err(ArtifactIndexError::ChangedDuringScan(source_path.clone()));
            }
            let mut temporary_permissions = temporary_file
                .metadata()
                .map_err(|error| ArtifactIndexError::Io {
                    path: temporary_path.clone(),
                    message: error.to_string(),
                })?
                .permissions();
            temporary_permissions.set_readonly(true);
            temporary_file
                .set_permissions(temporary_permissions)
                .map_err(|error| ArtifactIndexError::Io {
                    path: temporary_path.clone(),
                    message: error.to_string(),
                })?;
            temporary_file
                .sync_all()
                .map_err(|error| ArtifactIndexError::Io {
                    path: temporary_path.clone(),
                    message: error.to_string(),
                })?;
            before_temporary_verification();
            let temporary_metadata =
                temporary_file
                    .metadata()
                    .map_err(|error| ArtifactIndexError::Io {
                        path: temporary_path.clone(),
                        message: error.to_string(),
                    })?;
            let (temporary_sha256, stable_temporary) = hash_stable_capability_file(
                &mut temporary_file,
                &temporary_path,
                &temporary_metadata,
                cancellation,
            )?;
            let current_temporary =
                destination_parent
                    .metadata(&temporary_name)
                    .map_err(|error| ArtifactIndexError::Io {
                        path: temporary_path.clone(),
                        message: error.to_string(),
                    })?;
            if temporary_sha256 != expected_sha256
                || stable_temporary.len() != expected_size
                || !same_capability_file_identity(&stable_temporary, &current_temporary)
            {
                return Err(ArtifactIndexError::ChangedDuringScan(
                    temporary_path.clone(),
                ));
            }
            before_name_publication();
            publish_opened_capability_file(
                &temporary_file,
                &destination_parent,
                &destination_name,
                &destination_path,
                policy,
            )?;
            after_name_publication();
            let mut published_file =
                destination_parent
                    .open(&destination_name)
                    .map_err(|error| ArtifactIndexError::Io {
                        path: destination_path.clone(),
                        message: error.to_string(),
                    })?;
            let published_metadata =
                published_file
                    .metadata()
                    .map_err(|error| ArtifactIndexError::Io {
                        path: destination_path.clone(),
                        message: error.to_string(),
                    })?;
            let verification = (|| {
                let (published_sha256, stable_published) = hash_stable_capability_file(
                    &mut published_file,
                    &destination_path,
                    &published_metadata,
                    cancellation,
                )?;
                let current_published =
                    destination_parent
                        .metadata(&destination_name)
                        .map_err(|error| ArtifactIndexError::Io {
                            path: destination_path.clone(),
                            message: error.to_string(),
                        })?;
                if published_sha256 != expected_sha256
                    || stable_published.len() != expected_size
                    || !same_capability_file_identity(&stable_published, &current_published)
                {
                    return Err(ArtifactIndexError::ChangedDuringScan(
                        destination_path.clone(),
                    ));
                }
                Ok(())
            })();
            if let Err(verification_error) = verification {
                if let Ok(latest_published) = destination_parent.metadata(&destination_name)
                    && same_capability_object_identity(&published_metadata, &latest_published)
                {
                    destination_parent.remove_file(&destination_name).map_err(|error| {
                        ArtifactIndexError::Io {
                            path: destination_path.clone(),
                            message: format!(
                                "published artifact verification failed and exact-inode cleanup failed: {error}"
                            ),
                        }
                    })?;
                }
                return Err(verification_error);
            }
            if let Ok(current_temporary) = destination_parent.metadata(&temporary_name)
                && same_capability_object_identity(&stable_temporary, &current_temporary)
            {
                destination_parent
                    .remove_file(&temporary_name)
                    .map_err(|error| ArtifactIndexError::Io {
                        path: temporary_path.clone(),
                        message: error.to_string(),
                    })?;
            }
            published_file
                .set_permissions(original_permissions)
                .and_then(|()| published_file.sync_all())
                .map_err(|error| ArtifactIndexError::Io {
                    path: destination_path.clone(),
                    message: error.to_string(),
                })?;
            Ok(())
        })();
        if result.is_err()
            && let Some(expected_temporary) = temporary_identity.borrow().as_ref()
            && let Ok(current_temporary) = destination_parent.metadata(&temporary_name)
            && same_capability_object_identity(expected_temporary, &current_temporary)
            && let Err(error) = destination_parent.remove_file(&temporary_name)
        {
            return Err(ArtifactIndexError::Io {
                path: temporary_path,
                message: format!("contained move cleanup failed after {result:?}: {error}"),
            });
        }
        result?;
        sync_capability_directory(&destination_parent).map_err(|error| ArtifactIndexError::Io {
            path: destination_path.clone(),
            message: error.to_string(),
        })?;
        let current_source = source_parent
            .symlink_metadata(&source_name)
            .map_err(|error| ArtifactIndexError::PartialMove {
                source_path: source_path.clone(),
                destination_path: destination_path.clone(),
                message: error.to_string(),
            })?;
        if !same_capability_file_identity(&stable_source, &current_source) {
            return Err(ArtifactIndexError::PartialMove {
                source_path,
                destination_path,
                message: "source identity changed before cleanup".to_owned(),
            });
        }
        source_parent.remove_file(&source_name).map_err(|error| {
            ArtifactIndexError::PartialMove {
                source_path: source_path.clone(),
                destination_path: destination_path.clone(),
                message: error.to_string(),
            }
        })?;
        sync_capability_directory(&source_parent).map_err(|error| ArtifactIndexError::Io {
            path: source_path,
            message: error.to_string(),
        })?;
        Ok(())
    }

    pub fn write_contained_file(
        &self,
        relative_path: impl AsRef<Path>,
        bytes: &[u8],
        policy: ArtifactWritePolicy,
    ) -> Result<(), ArtifactIndexError> {
        self.write_contained_file_inner(relative_path.as_ref(), bytes, policy, || {})
    }

    fn write_contained_file_inner(
        &self,
        relative_path: &Path,
        bytes: &[u8],
        policy: ArtifactWritePolicy,
        before_publication: impl FnOnce(),
    ) -> Result<(), ArtifactIndexError> {
        let relative_path =
            normalize_relative_path_with_limits(relative_path, &ParserLimits::default())?;
        self.validate_key_scope(&ArtifactKey {
            root_id: self.id.clone(),
            relative_path: relative_path.clone(),
        })?;
        let destination = self.canonical_path.join(&relative_path);
        let (parent, file_name) = self.open_capability_parent(&relative_path, true)?;
        if let Ok(metadata) = parent.symlink_metadata(&file_name)
            && metadata.file_type().is_symlink()
        {
            return Err(ArtifactIndexError::SymbolicLink(destination));
        }
        let file_name_text = file_name
            .to_str()
            .ok_or_else(|| ArtifactIndexError::UnsafePath {
                path: destination.clone(),
                reason: "private artifact filename is invalid".to_owned(),
            })?;
        let sequence = PRIVATE_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary_name = PathBuf::from(format!(
            "{file_name_text}.{}.{}.tmp",
            std::process::id(),
            sequence
        ));
        let temporary = destination.with_file_name(&temporary_name);
        let result = (|| {
            let mut options = CapabilityOpenOptions::new();
            options.create_new(true).write(true);
            let mut file = parent
                .open_with(&temporary_name, &options)
                .map_err(|error| ArtifactIndexError::Io {
                    path: temporary.clone(),
                    message: error.to_string(),
                })?;
            file.write_all(bytes)
                .and_then(|()| file.sync_all())
                .map_err(|error| ArtifactIndexError::Io {
                    path: temporary.clone(),
                    message: error.to_string(),
                })?;
            before_publication();
            match policy {
                ArtifactWritePolicy::Reject => {
                    parent
                        .hard_link(&temporary_name, &parent, &file_name)
                        .map_err(|error| {
                            if error.kind() == std::io::ErrorKind::AlreadyExists {
                                ArtifactIndexError::AlreadyExists(destination.clone())
                            } else {
                                ArtifactIndexError::Io {
                                    path: destination.clone(),
                                    message: error.to_string(),
                                }
                            }
                        })?;
                    parent.remove_file(&temporary_name).map_err(|error| {
                        ArtifactIndexError::Io {
                            path: temporary.clone(),
                            message: error.to_string(),
                        }
                    })?;
                }
                ArtifactWritePolicy::Replace => {
                    parent
                        .rename(&temporary_name, &parent, &file_name)
                        .map_err(|error| ArtifactIndexError::Io {
                            path: destination.clone(),
                            message: error.to_string(),
                        })?;
                }
            }
            sync_capability_directory(&parent).map_err(|error| ArtifactIndexError::Io {
                path: destination
                    .parent()
                    .map_or_else(|| self.canonical_path.clone(), Path::to_path_buf),
                message: error.to_string(),
            })?;
            Ok(())
        })();
        if result.is_err()
            && let Err(error) = parent.remove_file(&temporary_name)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            return Err(ArtifactIndexError::Io {
                path: temporary,
                message: format!("private artifact cleanup failed after {result:?}: {error}"),
            });
        }
        result
    }

    pub fn quarantine_private_file(
        &self,
        relative_path: impl AsRef<Path>,
    ) -> Result<Option<PathBuf>, ArtifactIndexError> {
        let relative_path =
            normalize_relative_path_with_limits(relative_path.as_ref(), &ParserLimits::default())?;
        let source = self.canonical_path.join(&relative_path);
        let (parent, file_name) = match self.open_capability_parent(&relative_path, false) {
            Ok(value) => value,
            Err(ArtifactIndexError::Missing(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
        let metadata = match parent.symlink_metadata(&file_name) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(ArtifactIndexError::Io {
                    path: source,
                    message: error.to_string(),
                });
            }
        };
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(ArtifactIndexError::UnsafePath {
                path: source,
                reason: "private artifact quarantine source is not a regular file".to_owned(),
            });
        }
        let file_name_text = relative_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| ArtifactIndexError::UnsafePath {
                path: relative_path.to_path_buf(),
                reason: "private artifact filename is invalid".to_owned(),
            })?;
        let parent_relative = relative_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty());

        for _ in 0..128 {
            let sequence = PRIVATE_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let quarantine_name = format!(
                "{file_name_text}.quarantine.{}.{}",
                std::process::id(),
                sequence
            );
            let quarantine_relative = parent_relative.map_or_else(
                || PathBuf::from(&quarantine_name),
                |parent| parent.join(&quarantine_name),
            );
            let quarantine = self.canonical_path.join(&quarantine_relative);
            let mut options = CapabilityOpenOptions::new();
            options.create_new(true).write(true);
            let reservation = match parent.open_with(Path::new(&quarantine_name), &options) {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(ArtifactIndexError::Io {
                        path: quarantine,
                        message: error.to_string(),
                    });
                }
            };
            reservation
                .sync_all()
                .map_err(|error| ArtifactIndexError::Io {
                    path: quarantine.clone(),
                    message: error.to_string(),
                })?;
            drop(reservation);

            let result = parent
                .rename(&file_name, &parent, Path::new(&quarantine_name))
                .and_then(|()| sync_capability_directory(&parent));
            if let Err(error) = result {
                if let Err(cleanup) = parent.remove_file(Path::new(&quarantine_name))
                    && cleanup.kind() != std::io::ErrorKind::NotFound
                {
                    return Err(ArtifactIndexError::Io {
                        path: quarantine,
                        message: format!(
                            "private artifact quarantine failed ({error}) and reservation cleanup failed ({cleanup})"
                        ),
                    });
                }
                return Err(ArtifactIndexError::Io {
                    path: source,
                    message: error.to_string(),
                });
            }
            return Ok(Some(quarantine_relative));
        }

        Err(ArtifactIndexError::Io {
            path: source,
            message: "could not reserve a private artifact quarantine name".to_owned(),
        })
    }

    fn open_capability_parent(
        &self,
        relative_path: &Path,
        create_missing: bool,
    ) -> Result<(Dir, PathBuf), ArtifactIndexError> {
        validate_root_snapshot(self, &ParserLimits::default())?;
        let trusted_directory = self.trusted_directory.as_deref().ok_or_else(|| {
            ArtifactIndexError::InvalidSnapshot(
                "artifact root has not been admitted by the trusted root constructor".to_owned(),
            )
        })?;
        let mut directory =
            trusted_directory
                .try_clone()
                .map_err(|error| ArtifactIndexError::Io {
                    path: self.canonical_path.clone(),
                    message: error.to_string(),
                })?;
        let file_name = relative_path
            .file_name()
            .map(PathBuf::from)
            .ok_or_else(|| ArtifactIndexError::UnsafePath {
                path: relative_path.to_path_buf(),
                reason: "artifact path has no filename".to_owned(),
            })?;
        let parent = relative_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty());
        let mut diagnostic_path = self.canonical_path.clone();
        if let Some(parent) = parent {
            for component in parent.components() {
                let Component::Normal(name) = component else {
                    return Err(ArtifactIndexError::UnsafePath {
                        path: relative_path.to_path_buf(),
                        reason: "artifact parent is not normalized".to_owned(),
                    });
                };
                diagnostic_path.push(name);
                let metadata = match directory.symlink_metadata(name) {
                    Ok(metadata) => metadata,
                    Err(error)
                        if create_missing && error.kind() == std::io::ErrorKind::NotFound =>
                    {
                        match directory.create_dir(name) {
                            Ok(()) => {}
                            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                            Err(error) => {
                                return Err(ArtifactIndexError::Io {
                                    path: diagnostic_path,
                                    message: error.to_string(),
                                });
                            }
                        }
                        directory.symlink_metadata(name).map_err(|error| {
                            ArtifactIndexError::Io {
                                path: diagnostic_path.clone(),
                                message: error.to_string(),
                            }
                        })?
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        return Err(ArtifactIndexError::Missing(ArtifactKey {
                            root_id: self.id.clone(),
                            relative_path: relative_path.to_path_buf(),
                        }));
                    }
                    Err(error) => {
                        return Err(ArtifactIndexError::Io {
                            path: diagnostic_path,
                            message: error.to_string(),
                        });
                    }
                };
                if metadata.file_type().is_symlink() {
                    return Err(ArtifactIndexError::SymbolicLink(diagnostic_path));
                }
                if !metadata.is_dir() {
                    return Err(ArtifactIndexError::UnsafePath {
                        path: diagnostic_path,
                        reason: "artifact parent component is not a directory".to_owned(),
                    });
                }
                let child = directory
                    .open_dir(name)
                    .map_err(|error| ArtifactIndexError::Io {
                        path: diagnostic_path.clone(),
                        message: error.to_string(),
                    })?;
                let opened = child
                    .dir_metadata()
                    .map_err(|error| ArtifactIndexError::Io {
                        path: diagnostic_path.clone(),
                        message: error.to_string(),
                    })?;
                if !same_capability_file_identity(&metadata, &opened) {
                    return Err(ArtifactIndexError::ChangedDuringScan(diagnostic_path));
                }
                directory = child;
            }
        }
        Ok((directory, file_name))
    }

    fn resolve_key(
        &self,
        key: &ArtifactKey,
        resolution: PathResolution,
    ) -> Result<PathBuf, ArtifactIndexError> {
        validate_root_snapshot(self, &ParserLimits::default())?;
        self.validate_key_scope(key)?;
        let relative_path =
            normalize_relative_path_with_limits(&key.relative_path, &ParserLimits::default())?;
        let candidate = self.canonical_path.join(&relative_path);
        let mut inspected = self.canonical_path.clone();
        let component_count = relative_path.components().count();
        for (index, component) in relative_path.components().enumerate() {
            let Component::Normal(name) = component else {
                return Err(ArtifactIndexError::UnsafePath {
                    path: relative_path,
                    reason: "only non-empty relative path components are allowed".to_owned(),
                });
            };
            inspected.push(name);
            match fs::symlink_metadata(&inspected) {
                Ok(metadata) => {
                    if metadata.file_type().is_symlink() {
                        return Err(ArtifactIndexError::SymbolicLink(inspected));
                    }
                    if index + 1 < component_count && !metadata.is_dir() {
                        return Err(ArtifactIndexError::UnsafePath {
                            path: inspected,
                            reason: "an artifact path parent is not a directory".to_owned(),
                        });
                    }
                }
                Err(error)
                    if error.kind() == std::io::ErrorKind::NotFound
                        && resolution == PathResolution::ForCreate =>
                {
                    break;
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    return Err(ArtifactIndexError::Missing(key.clone()));
                }
                Err(error) => {
                    return Err(ArtifactIndexError::Io {
                        path: inspected,
                        message: error.to_string(),
                    });
                }
            }
        }
        if !candidate.starts_with(&self.canonical_path) {
            return Err(ArtifactIndexError::UnsafePath {
                path: candidate,
                reason: "artifact path escaped its root".to_owned(),
            });
        }
        if resolution == PathResolution::ForCreate {
            return Ok(candidate);
        }
        let canonical = fs::canonicalize(&candidate).map_err(|error| ArtifactIndexError::Io {
            path: candidate.clone(),
            message: error.to_string(),
        })?;
        if !canonical.starts_with(&self.canonical_path) {
            return Err(ArtifactIndexError::UnsafePath {
                path: candidate,
                reason: "resolved artifact escaped its root".to_owned(),
            });
        }
        Ok(canonical)
    }

    fn validate_key_scope(&self, key: &ArtifactKey) -> Result<(), ArtifactIndexError> {
        if key.root_id != self.id {
            return Err(ArtifactIndexError::UnknownRoot(key.root_id.clone()));
        }
        if let Some(approved_relative_path) = &self.approved_relative_path
            && &key.relative_path != approved_relative_path
        {
            return Err(ArtifactIndexError::UnsafePath {
                path: key.relative_path.clone(),
                reason: "artifact path is outside the approved import".to_owned(),
            });
        }
        Ok(())
    }

    fn accepts(&self, path: &Path) -> bool {
        self.approved_extensions.is_empty()
            || path
                .extension()
                .and_then(|value| value.to_str())
                .map(|value| {
                    self.approved_extensions
                        .contains(&value.to_ascii_lowercase())
                })
                .unwrap_or(false)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PathResolution {
    Existing,
    ForCreate,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactAvailability {
    Present,
    Missing,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactRecord {
    pub key: ArtifactKey,
    pub namespace: String,
    pub canonical_path: PathBuf,
    pub byte_size: u64,
    pub modified_nanoseconds: u128,
    pub sha256: String,
    pub availability: ArtifactAvailability,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactChangeKind {
    Added,
    Modified,
    Missing,
    Restored,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactChange {
    pub key: ArtifactKey,
    pub kind: ArtifactChangeKind,
}

pub struct VerifiedArtifactFile {
    file: File,
    path: PathBuf,
    expected_byte_size: u64,
    expected_modified_nanoseconds: u128,
}

impl VerifiedArtifactFile {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn into_file(self) -> File {
        self.file
    }

    pub fn file_mut(&mut self) -> &mut File {
        &mut self.file
    }

    pub fn verify_unchanged(&self) -> Result<(), ArtifactIndexError> {
        let opened = self
            .file
            .metadata()
            .map_err(|error| ArtifactIndexError::Io {
                path: self.path.clone(),
                message: error.to_string(),
            })?;
        let current = fs::metadata(&self.path).map_err(|error| ArtifactIndexError::Io {
            path: self.path.clone(),
            message: error.to_string(),
        })?;
        if !same_file_identity(&opened, &current)
            || opened.len() != self.expected_byte_size
            || modified_nanoseconds(&opened) != self.expected_modified_nanoseconds
        {
            return Err(ArtifactIndexError::ChangedDuringScan(self.path.clone()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactIndex {
    version: u32,
    roots: BTreeMap<String, ArtifactRoot>,
    records: BTreeMap<ArtifactKey, ArtifactRecord>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactIndexReconciliation {
    remapped_root_ids: BTreeMap<String, String>,
    removed_root_count: usize,
    added_root_count: usize,
    original_record_count: usize,
    retained_record_count: usize,
}

impl ArtifactIndexReconciliation {
    pub fn current_root_id(&self, snapshot_root_id: &str) -> Option<&str> {
        self.remapped_root_ids
            .get(snapshot_root_id)
            .map(String::as_str)
    }

    pub fn removed_root_count(&self) -> usize {
        self.removed_root_count
    }

    pub fn added_root_count(&self) -> usize {
        self.added_root_count
    }

    pub fn dropped_record_count(&self) -> usize {
        self.original_record_count
            .saturating_sub(self.retained_record_count)
    }

    pub fn changed(&self) -> bool {
        self.removed_root_count != 0
            || self.added_root_count != 0
            || self.dropped_record_count() != 0
            || self
                .remapped_root_ids
                .iter()
                .any(|(snapshot, current)| snapshot != current)
    }
}

impl Default for ArtifactIndex {
    fn default() -> Self {
        Self {
            version: ARTIFACT_INDEX_VERSION,
            roots: BTreeMap::new(),
            records: BTreeMap::new(),
        }
    }
}

impl ArtifactIndex {
    pub fn from_snapshot(
        bytes: &[u8],
        trusted_roots: impl IntoIterator<Item = ArtifactRoot>,
    ) -> Result<Self, ArtifactIndexError> {
        let limits = ParserLimits::default();
        Self::from_snapshot_with_limits(bytes, trusted_roots, &limits)
    }

    pub fn from_snapshot_with_limits(
        bytes: &[u8],
        trusted_roots: impl IntoIterator<Item = ArtifactRoot>,
        limits: &ParserLimits,
    ) -> Result<Self, ArtifactIndexError> {
        let snapshot = decode_artifact_index_snapshot(bytes, limits)?;
        let roots = collect_trusted_roots(trusted_roots)?;
        if roots != snapshot.roots {
            return Err(ArtifactIndexError::InvalidSnapshot(
                "snapshot roots do not match the trusted configured roots".to_owned(),
            ));
        }
        let index = Self {
            version: snapshot.version,
            roots,
            records: snapshot.records,
        };
        index.validate_snapshot(limits)?;
        Ok(index)
    }

    pub fn reconcile_snapshot(
        bytes: &[u8],
        trusted_roots: impl IntoIterator<Item = ArtifactRoot>,
    ) -> Result<(Self, ArtifactIndexReconciliation), ArtifactIndexError> {
        let limits = ParserLimits::default();
        let snapshot = decode_artifact_index_snapshot(bytes, &limits)?;
        let trusted = collect_trusted_roots(trusted_roots)?;
        let mut remapped_root_ids = BTreeMap::new();
        let mut retained_current_root_ids = BTreeSet::new();

        for snapshot_root in snapshot.roots.values() {
            if let Some(current) = trusted.get(snapshot_root.id()) {
                if current != snapshot_root {
                    return Err(ArtifactIndexError::InvalidSnapshot(format!(
                        "snapshot root {:?} changed its trusted scope",
                        snapshot_root.id()
                    )));
                }
                remapped_root_ids.insert(snapshot_root.id().to_owned(), current.id().to_owned());
                retained_current_root_ids.insert(current.id().to_owned());
                continue;
            }
            if let Some(current) = trusted
                .values()
                .find(|current| roots_have_same_scope(current, snapshot_root))
            {
                if !retained_current_root_ids.insert(current.id().to_owned()) {
                    return Err(ArtifactIndexError::InvalidSnapshot(
                        "multiple snapshot roots map to one trusted root".to_owned(),
                    ));
                }
                remapped_root_ids.insert(snapshot_root.id().to_owned(), current.id().to_owned());
            }
        }

        let mut records = BTreeMap::new();
        for record in snapshot.records.values() {
            let snapshot_root = snapshot.roots.get(&record.key.root_id).ok_or_else(|| {
                ArtifactIndexError::InvalidSnapshot(
                    "artifact record refers to an absent snapshot root".to_owned(),
                )
            })?;
            validate_record_scope(record, snapshot_root, &limits)?;
            let Some(current_root_id) = remapped_root_ids.get(&record.key.root_id) else {
                continue;
            };
            let current_root = trusted.get(current_root_id).ok_or_else(|| {
                ArtifactIndexError::InvalidSnapshot(
                    "reconciled artifact root disappeared".to_owned(),
                )
            })?;
            let key = ArtifactKey::new(current_root_id.clone(), record.key.relative_path.clone())?;
            let mut migrated = record.clone();
            migrated.key = key.clone();
            migrated.namespace = current_root.namespace.clone();
            migrated.canonical_path = current_root.canonical_path.join(&key.relative_path);
            if records.insert(key, migrated).is_some() {
                return Err(ArtifactIndexError::InvalidSnapshot(
                    "reconciled artifact keys are duplicated".to_owned(),
                ));
            }
        }

        let index = Self {
            version: snapshot.version,
            roots: trusted,
            records,
        };
        index.validate_snapshot(&limits)?;
        let reconciliation = ArtifactIndexReconciliation {
            remapped_root_ids,
            removed_root_count: snapshot
                .roots
                .keys()
                .filter(|root_id| !index.roots.contains_key(*root_id))
                .count(),
            added_root_count: index
                .roots
                .keys()
                .filter(|root_id| !snapshot.roots.contains_key(*root_id))
                .count(),
            original_record_count: snapshot.records.len(),
            retained_record_count: index.records.len(),
        };
        Ok((index, reconciliation))
    }

    pub fn snapshot(&self) -> Result<Vec<u8>, ArtifactIndexError> {
        self.snapshot_with_limits(&ParserLimits::default())
    }

    pub fn snapshot_with_limits(
        &self,
        limits: &ParserLimits,
    ) -> Result<Vec<u8>, ArtifactIndexError> {
        limits.validate()?;
        let mut writer = BoundedSnapshotWriter::new(limits.manifest_bytes)?;
        serde_json::to_writer(
            &mut writer,
            &ArtifactIndexSnapshot {
                version: self.version,
                roots: self.roots.values().cloned().collect(),
                records: self.records.values().cloned().collect(),
            },
        )
        .map_err(|error| ArtifactIndexError::InvalidSnapshot(error.to_string()))?;
        Ok(writer.finish())
    }

    pub fn add_root(&mut self, root: ArtifactRoot) -> Result<(), ArtifactIndexError> {
        validate_root_snapshot(&root, &ParserLimits::default())?;
        if self.roots.contains_key(&root.id) {
            return Err(ArtifactIndexError::DuplicateRootId(root.id));
        }
        if self
            .roots
            .keys()
            .any(|existing| existing.to_lowercase() == root.id.to_lowercase())
        {
            return Err(ArtifactIndexError::PortableRootCollision(root.id));
        }
        if self
            .roots
            .values()
            .any(|existing| roots_overlap(existing, &root))
        {
            return Err(ArtifactIndexError::DuplicateCanonicalPath(
                root.canonical_path,
            ));
        }
        self.roots.insert(root.id.clone(), root);
        Ok(())
    }

    pub fn roots(&self) -> impl Iterator<Item = &ArtifactRoot> {
        self.roots.values()
    }

    pub fn root(&self, root_id: &str) -> Option<&ArtifactRoot> {
        self.roots.get(root_id)
    }

    pub fn records(&self) -> impl Iterator<Item = &ArtifactRecord> {
        self.records.values()
    }

    pub fn record(&self, key: &ArtifactKey) -> Option<&ArtifactRecord> {
        self.records.get(key)
    }

    pub fn open_verified(
        &self,
        key: &ArtifactKey,
        cancellation: &CancellationToken,
    ) -> Result<VerifiedArtifactFile, ArtifactIndexError> {
        check_cancelled(cancellation)?;
        let record = self
            .records
            .get(key)
            .ok_or_else(|| ArtifactIndexError::Missing(key.clone()))?;
        if record.availability != ArtifactAvailability::Present {
            return Err(ArtifactIndexError::Missing(key.clone()));
        }
        let root = self
            .roots
            .get(&key.root_id)
            .ok_or_else(|| ArtifactIndexError::UnknownRoot(key.root_id.clone()))?;
        root.validate_key_scope(key)?;
        let relative_path =
            normalize_relative_path_with_limits(&key.relative_path, &ParserLimits::default())?;
        let path = root.canonical_path.join(&relative_path);
        if path != record.canonical_path {
            return Err(ArtifactIndexError::ChangedSinceIndex(key.clone()));
        }
        let (parent, file_name) = root.open_capability_parent(&relative_path, false)?;
        let before =
            parent
                .symlink_metadata(&file_name)
                .map_err(|error| ArtifactIndexError::Io {
                    path: path.clone(),
                    message: error.to_string(),
                })?;
        if before.file_type().is_symlink() {
            return Err(ArtifactIndexError::SymbolicLink(path));
        }
        if !before.is_file() {
            return Err(ArtifactIndexError::ChangedSinceIndex(key.clone()));
        }
        let mut file = parent
            .open(&file_name)
            .map_err(|error| ArtifactIndexError::Io {
                path: path.clone(),
                message: error.to_string(),
            })?;
        let opened = file.metadata().map_err(|error| ArtifactIndexError::Io {
            path: path.clone(),
            message: error.to_string(),
        })?;
        if !same_capability_file_identity(&before, &opened) {
            return Err(ArtifactIndexError::ChangedDuringScan(path));
        }
        let (sha256, after) = hash_stable_capability_file(&mut file, &path, &opened, cancellation)?;
        let current = parent
            .metadata(&file_name)
            .map_err(|error| ArtifactIndexError::Io {
                path: path.clone(),
                message: error.to_string(),
            })?;
        if !same_capability_file_identity(&after, &current)
            || sha256 != record.sha256
            || after.len() != record.byte_size
            || capability_modified_nanoseconds(&after) != record.modified_nanoseconds
        {
            return Err(ArtifactIndexError::ChangedSinceIndex(key.clone()));
        }
        file.seek(std::io::SeekFrom::Start(0))
            .map_err(|error| ArtifactIndexError::Io {
                path: path.clone(),
                message: error.to_string(),
            })?;
        Ok(VerifiedArtifactFile {
            file: file.into_std(),
            path,
            expected_byte_size: record.byte_size,
            expected_modified_nanoseconds: record.modified_nanoseconds,
        })
    }

    pub fn refresh(
        &mut self,
        cancellation: &CancellationToken,
    ) -> Result<Vec<ArtifactChange>, ArtifactIndexError> {
        self.refresh_with_limits(cancellation, &ParserLimits::default())
    }

    pub fn refresh_with_limits(
        &mut self,
        cancellation: &CancellationToken,
        limits: &ParserLimits,
    ) -> Result<Vec<ArtifactChange>, ArtifactIndexError> {
        let root_ids = self.roots.keys().cloned().collect::<Vec<_>>();
        self.refresh_selected_with_limits(root_ids, cancellation, limits)
    }

    pub fn refresh_selected(
        &mut self,
        root_ids: impl IntoIterator<Item = impl Into<String>>,
        cancellation: &CancellationToken,
    ) -> Result<Vec<ArtifactChange>, ArtifactIndexError> {
        self.refresh_selected_with_limits(root_ids, cancellation, &ParserLimits::default())
    }

    pub fn refresh_selected_with_limits(
        &mut self,
        root_ids: impl IntoIterator<Item = impl Into<String>>,
        cancellation: &CancellationToken,
        limits: &ParserLimits,
    ) -> Result<Vec<ArtifactChange>, ArtifactIndexError> {
        limits.validate()?;
        check_cancelled(cancellation)?;
        let selected = root_ids
            .into_iter()
            .map(Into::into)
            .collect::<BTreeSet<String>>();
        for root_id in &selected {
            if !self.roots.contains_key(root_id) {
                return Err(ArtifactIndexError::UnknownRoot(root_id.clone()));
            }
        }
        let mut discovered = BTreeMap::new();
        let mut canonical_paths = BTreeSet::new();
        let mut portable_paths = BTreeSet::new();
        let mut scanned_entries = 0_u64;
        for root in self
            .roots
            .values()
            .filter(|root| selected.contains(root.id()))
        {
            scan_root(
                root,
                cancellation,
                limits,
                &mut scanned_entries,
                &mut discovered,
                &mut canonical_paths,
                &mut portable_paths,
            )?;
        }

        discovered.extend(
            self.records
                .iter()
                .filter(|(key, _)| !selected.contains(&key.root_id))
                .map(|(key, record)| (key.clone(), record.clone())),
        );

        let mut changes = Vec::new();
        for (key, record) in &discovered {
            if !selected.contains(&key.root_id) {
                continue;
            }
            match self.records.get(key) {
                None => changes.push(ArtifactChange {
                    key: key.clone(),
                    kind: ArtifactChangeKind::Added,
                }),
                Some(previous) if previous.availability == ArtifactAvailability::Missing => {
                    changes.push(ArtifactChange {
                        key: key.clone(),
                        kind: ArtifactChangeKind::Restored,
                    });
                }
                Some(previous)
                    if previous.sha256 != record.sha256
                        || previous.byte_size != record.byte_size
                        || previous.modified_nanoseconds != record.modified_nanoseconds =>
                {
                    changes.push(ArtifactChange {
                        key: key.clone(),
                        kind: ArtifactChangeKind::Modified,
                    });
                }
                Some(_) => {}
            }
        }

        for (key, previous) in &self.records {
            if selected.contains(&key.root_id) && !discovered.contains_key(key) {
                let mut missing = previous.clone();
                if previous.availability == ArtifactAvailability::Present {
                    changes.push(ArtifactChange {
                        key: key.clone(),
                        kind: ArtifactChangeKind::Missing,
                    });
                }
                missing.availability = ArtifactAvailability::Missing;
                discovered.insert(key.clone(), missing);
            }
        }
        changes.sort_by(|left, right| left.key.cmp(&right.key));
        self.records = discovered;
        Ok(changes)
    }

    fn validate_snapshot(&self, limits: &ParserLimits) -> Result<(), ArtifactIndexError> {
        let mut validated_roots: Vec<&ArtifactRoot> = Vec::new();
        let mut portable_root_ids = BTreeSet::new();
        for (id, root) in &self.roots {
            if id != &root.id
                || !portable_root_ids.insert(id.to_lowercase())
                || validated_roots
                    .iter()
                    .any(|existing| roots_overlap(existing, root))
            {
                return Err(ArtifactIndexError::InvalidSnapshot(
                    "root identity or canonical path is duplicated".to_owned(),
                ));
            }
            validate_root_snapshot(root, limits)?;
            validated_roots.push(root);
        }
        let mut canonical_paths = BTreeSet::new();
        let mut portable_paths = BTreeSet::new();
        for (key, record) in &self.records {
            if key != &record.key || !self.roots.contains_key(&key.root_id) {
                return Err(ArtifactIndexError::InvalidSnapshot(
                    "artifact record has an invalid key or root".to_owned(),
                ));
            }
            let root = self.roots.get(&key.root_id).ok_or_else(|| {
                ArtifactIndexError::InvalidSnapshot("artifact root disappeared".to_owned())
            })?;
            validate_record_scope(record, root, limits)?;
            if !canonical_paths.insert(record.canonical_path.clone())
                || !portable_paths.insert(portable_artifact_identity(
                    &key.root_id,
                    &key.relative_path,
                )?)
            {
                return Err(ArtifactIndexError::InvalidSnapshot(
                    "artifact record has a duplicate canonical or portable path".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

fn roots_overlap(left: &ArtifactRoot, right: &ArtifactRoot) -> bool {
    match (
        left.approved_relative_path.as_ref(),
        right.approved_relative_path.as_ref(),
    ) {
        (Some(left_relative), Some(right_relative)) => {
            left.canonical_path.join(left_relative) == right.canonical_path.join(right_relative)
        }
        (Some(left_relative), None) => left
            .canonical_path
            .join(left_relative)
            .starts_with(&right.canonical_path),
        (None, Some(right_relative)) => right
            .canonical_path
            .join(right_relative)
            .starts_with(&left.canonical_path),
        (None, None) => {
            left.canonical_path.starts_with(&right.canonical_path)
                || right.canonical_path.starts_with(&left.canonical_path)
        }
    }
}

fn roots_have_same_scope(left: &ArtifactRoot, right: &ArtifactRoot) -> bool {
    left.namespace == right.namespace
        && left.canonical_path == right.canonical_path
        && left.approved_extensions == right.approved_extensions
        && left.imported == right.imported
        && left.approved_relative_path == right.approved_relative_path
}

fn collect_trusted_roots(
    trusted_roots: impl IntoIterator<Item = ArtifactRoot>,
) -> Result<BTreeMap<String, ArtifactRoot>, ArtifactIndexError> {
    let mut index = ArtifactIndex::default();
    for root in trusted_roots {
        index.add_root(root)?;
    }
    Ok(index.roots)
}

fn validate_record_scope(
    record: &ArtifactRecord,
    root: &ArtifactRoot,
    limits: &ParserLimits,
) -> Result<(), ArtifactIndexError> {
    let relative = normalize_relative_path_with_limits(&record.key.relative_path, limits)?;
    root.validate_key_scope(&record.key)?;
    if record.namespace != root.namespace
        || record.canonical_path != root.canonical_path.join(&relative)
        || !root.accepts(&record.canonical_path)
    {
        return Err(ArtifactIndexError::InvalidSnapshot(
            "artifact record namespace, path, or extension is invalid".to_owned(),
        ));
    }
    if record.sha256.len() != 64
        || !record
            .sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ArtifactIndexError::InvalidSnapshot(
            "artifact record digest is invalid".to_owned(),
        ));
    }
    Ok(())
}

fn validate_root_snapshot(
    root: &ArtifactRoot,
    limits: &ParserLimits,
) -> Result<(), ArtifactIndexError> {
    let trusted_directory = root.trusted_directory.as_deref().ok_or_else(|| {
        ArtifactIndexError::InvalidSnapshot(
            "artifact root has not been admitted by the trusted root constructor".to_owned(),
        )
    })?;
    if root.id.trim().is_empty() || root.namespace.trim().is_empty() {
        return Err(ArtifactIndexError::InvalidSnapshot(
            "artifact root ID or namespace is empty".to_owned(),
        ));
    }
    for (kind, value) in [("root ID", &root.id), ("root namespace", &root.namespace)] {
        limits
            .check(
                kind,
                u64::try_from(value.len()).unwrap_or(u64::MAX),
                limits.maximum_name_bytes,
            )
            .map_err(|error| ArtifactIndexError::InvalidSnapshot(error.to_string()))?;
    }
    if root.imported != root.approved_relative_path.is_some() {
        return Err(ArtifactIndexError::InvalidSnapshot(
            "imported root scope is inconsistent".to_owned(),
        ));
    }
    if let Some(relative) = &root.approved_relative_path {
        if normalize_relative_path_with_limits(relative, limits)? != *relative {
            return Err(ArtifactIndexError::InvalidSnapshot(
                "approved import path is not normalized".to_owned(),
            ));
        }
    }
    let metadata = fs::symlink_metadata(&root.canonical_path).map_err(|error| {
        ArtifactIndexError::InvalidSnapshot(format!(
            "artifact root {} is unavailable: {error}",
            root.canonical_path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(ArtifactIndexError::InvalidSnapshot(
            "artifact root is not a direct directory".to_owned(),
        ));
    }
    let canonical = fs::canonicalize(&root.canonical_path).map_err(|error| {
        ArtifactIndexError::InvalidSnapshot(format!(
            "artifact root {} cannot be canonicalized: {error}",
            root.canonical_path.display()
        ))
    })?;
    if canonical != root.canonical_path {
        return Err(ArtifactIndexError::InvalidSnapshot(
            "artifact root path is not canonical".to_owned(),
        ));
    }
    reject_symbolic_link_components(&root.canonical_path).map_err(|error| {
        ArtifactIndexError::InvalidSnapshot(format!(
            "artifact root path components are not trusted: {error}"
        ))
    })?;
    let opened = trusted_directory
        .try_clone()
        .and_then(|directory| directory.into_std_file().metadata())
        .map_err(|error| {
            ArtifactIndexError::InvalidSnapshot(format!(
                "artifact root capability is unavailable: {error}"
            ))
        })?;
    let current = fs::metadata(&root.canonical_path).map_err(|error| {
        ArtifactIndexError::InvalidSnapshot(format!(
            "artifact root {} is unavailable: {error}",
            root.canonical_path.display()
        ))
    })?;
    if !same_file_identity(&opened, &current) {
        return Err(ArtifactIndexError::InvalidSnapshot(
            "artifact root changed after trusted admission".to_owned(),
        ));
    }
    Ok(())
}

#[derive(Deserialize, Serialize)]
struct ArtifactIndexSnapshot {
    version: u32,
    roots: Vec<ArtifactRoot>,
    records: Vec<ArtifactRecord>,
}

struct DecodedArtifactIndexSnapshot {
    version: u32,
    roots: BTreeMap<String, ArtifactRoot>,
    records: BTreeMap<ArtifactKey, ArtifactRecord>,
}

fn decode_artifact_index_snapshot(
    bytes: &[u8],
    limits: &ParserLimits,
) -> Result<DecodedArtifactIndexSnapshot, ArtifactIndexError> {
    limits.validate()?;
    limits
        .check(
            "artifact index snapshot bytes",
            u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            limits.manifest_bytes,
        )
        .map_err(|error| ArtifactIndexError::InvalidSnapshot(error.to_string()))?;
    let snapshot: ArtifactIndexSnapshot = serde_json::from_slice(bytes)
        .map_err(|error| ArtifactIndexError::InvalidSnapshot(error.to_string()))?;
    if snapshot.version != ARTIFACT_INDEX_VERSION {
        return Err(ArtifactIndexError::UnsupportedVersion(snapshot.version));
    }
    let root_count = snapshot.roots.len();
    let record_count = snapshot.records.len();
    limits
        .check(
            "artifact root count",
            u64::try_from(root_count).unwrap_or(u64::MAX),
            limits.maximum_metadata_values,
        )
        .map_err(|error| ArtifactIndexError::InvalidSnapshot(error.to_string()))?;
    limits
        .check(
            "artifact record count",
            u64::try_from(record_count).unwrap_or(u64::MAX),
            limits.maximum_tensors,
        )
        .map_err(|error| ArtifactIndexError::InvalidSnapshot(error.to_string()))?;
    let roots = snapshot
        .roots
        .into_iter()
        .map(|root| (root.id.clone(), root))
        .collect::<BTreeMap<_, _>>();
    let records = snapshot
        .records
        .into_iter()
        .map(|record| (record.key.clone(), record))
        .collect::<BTreeMap<_, _>>();
    if roots.len() != root_count || records.len() != record_count {
        return Err(ArtifactIndexError::InvalidSnapshot(
            "duplicate root or artifact key".to_owned(),
        ));
    }
    Ok(DecodedArtifactIndexSnapshot {
        version: snapshot.version,
        roots,
        records,
    })
}

struct BoundedSnapshotWriter {
    bytes: Vec<u8>,
    maximum: usize,
}

impl BoundedSnapshotWriter {
    fn new(maximum: u64) -> Result<Self, ArtifactIndexError> {
        let maximum = usize::try_from(maximum).map_err(|_| {
            ArtifactIndexError::InvalidSnapshot(
                "artifact snapshot limit does not fit this platform".to_owned(),
            )
        })?;
        Ok(Self {
            bytes: Vec::new(),
            maximum,
        })
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

impl Write for BoundedSnapshotWriter {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        let next_length = self
            .bytes
            .len()
            .checked_add(buffer.len())
            .ok_or_else(|| std::io::Error::other("artifact snapshot length overflow"))?;
        if next_length > self.maximum {
            return Err(std::io::Error::other(
                "artifact snapshot exceeds the configured manifest limit",
            ));
        }
        self.bytes
            .try_reserve(buffer.len())
            .map_err(|_| std::io::Error::other("artifact snapshot allocation failed"))?;
        self.bytes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ArtifactIndexError {
    #[error("artifact indexing was cancelled")]
    Cancelled,
    #[error("artifact I/O failed for {path}: {message}")]
    Io { path: PathBuf, message: String },
    #[error("artifact root ID {0:?} is invalid")]
    InvalidRootId(String),
    #[error("artifact namespace {0:?} is invalid")]
    InvalidNamespace(String),
    #[error("artifact root ID {0:?} is duplicated")]
    DuplicateRootId(String),
    #[error("artifact root ID {0:?} collides on a case-insensitive platform")]
    PortableRootCollision(String),
    #[error("artifact canonical path {0} is duplicated")]
    DuplicateCanonicalPath(PathBuf),
    #[error("artifact path {0} collides on a case-insensitive platform")]
    PortablePathCollision(PathBuf),
    #[error("artifact path {path} is unsafe: {reason}")]
    UnsafePath { path: PathBuf, reason: String },
    #[error("artifact path {0} is a symbolic link")]
    SymbolicLink(PathBuf),
    #[error("artifact root {0} is not a directory")]
    NotDirectory(PathBuf),
    #[error("artifact root {0:?} is unknown")]
    UnknownRoot(String),
    #[error("artifact {0:?} is missing")]
    Missing(ArtifactKey),
    #[error("artifact path already exists: {0}")]
    AlreadyExists(PathBuf),
    #[error("artifact changed while it was being hashed: {0}")]
    ChangedDuringScan(PathBuf),
    #[error("artifact changed since its canonical index record was created: {0:?}")]
    ChangedSinceIndex(ArtifactKey),
    #[error(
        "artifact move published {destination_path} but could not remove source {source_path}: {message}"
    )]
    PartialMove {
        source_path: PathBuf,
        destination_path: PathBuf,
        message: String,
    },
    #[error("artifact allocation failed while scanning {0}")]
    AllocationFailed(PathBuf),
    #[error("artifact index snapshot is invalid: {0}")]
    InvalidSnapshot(String),
    #[error("artifact index snapshot version {0} is unsupported")]
    UnsupportedVersion(u32),
    #[error(transparent)]
    Limit(#[from] ParserLimitError),
}

fn scan_root(
    root: &ArtifactRoot,
    cancellation: &CancellationToken,
    limits: &ParserLimits,
    scanned_entries: &mut u64,
    discovered: &mut BTreeMap<ArtifactKey, ArtifactRecord>,
    canonical_paths: &mut BTreeSet<PathBuf>,
    portable_paths: &mut BTreeSet<String>,
) -> Result<(), ArtifactIndexError> {
    validate_root_snapshot(root, limits)?;
    let trusted_directory = root.trusted_directory.as_deref().ok_or_else(|| {
        ArtifactIndexError::InvalidSnapshot(
            "artifact root has not been admitted by the trusted root constructor".to_owned(),
        )
    })?;
    let initial_directory =
        trusted_directory
            .try_clone()
            .map_err(|error| ArtifactIndexError::Io {
                path: root.canonical_path.clone(),
                message: error.to_string(),
            })?;
    let mut pending = vec![(initial_directory, PathBuf::new(), 0_u32)];
    while let Some((directory, relative_directory, depth)) = pending.pop() {
        check_cancelled(cancellation)?;
        let diagnostic_directory = root.canonical_path.join(&relative_directory);
        let entries = directory
            .entries()
            .map_err(|error| ArtifactIndexError::Io {
                path: diagnostic_directory.clone(),
                message: error.to_string(),
            })?;
        let mut entries_sorted = Vec::new();
        for entry in entries {
            check_cancelled(cancellation)?;
            *scanned_entries =
                scanned_entries
                    .checked_add(1)
                    .ok_or(ParserLimitError::Exceeded {
                        kind: "artifact scan entries",
                        actual: u64::MAX,
                        maximum: limits.maximum_archive_entries,
                    })?;
            limits.check(
                "artifact scan entries",
                *scanned_entries,
                limits.maximum_archive_entries,
            )?;
            let entry = entry.map_err(|error| ArtifactIndexError::Io {
                path: diagnostic_directory.clone(),
                message: error.to_string(),
            })?;
            entries_sorted
                .try_reserve(1)
                .map_err(|_| ArtifactIndexError::AllocationFailed(diagnostic_directory.clone()))?;
            entries_sorted.push(entry);
        }
        entries_sorted.sort_by_key(cap_std::fs::DirEntry::file_name);
        for entry in entries_sorted {
            check_cancelled(cancellation)?;
            let file_name = entry.file_name();
            if file_name
                .to_str()
                .is_some_and(|name| name.starts_with(".zed-"))
            {
                continue;
            }
            let relative = relative_directory.join(&file_name);
            let path = root.canonical_path.join(&relative);
            if let Some(approved) = &root.approved_relative_path {
                if &relative != approved {
                    continue;
                }
            }
            let metadata =
                directory
                    .symlink_metadata(&file_name)
                    .map_err(|error| ArtifactIndexError::Io {
                        path: path.clone(),
                        message: error.to_string(),
                    })?;
            if metadata.file_type().is_symlink() {
                return Err(ArtifactIndexError::SymbolicLink(path));
            }
            if metadata.is_dir() {
                let child_depth = depth.checked_add(1).ok_or(ParserLimitError::Exceeded {
                    kind: "artifact scan depth",
                    actual: u64::MAX,
                    maximum: u64::from(limits.maximum_depth),
                })?;
                limits.check(
                    "artifact scan depth",
                    u64::from(child_depth),
                    u64::from(limits.maximum_depth),
                )?;
                pending
                    .try_reserve(1)
                    .map_err(|_| ArtifactIndexError::AllocationFailed(path.clone()))?;
                let child = entry.open_dir().map_err(|error| ArtifactIndexError::Io {
                    path: path.clone(),
                    message: error.to_string(),
                })?;
                let opened = child
                    .dir_metadata()
                    .map_err(|error| ArtifactIndexError::Io {
                        path: path.clone(),
                        message: error.to_string(),
                    })?;
                if !same_capability_file_identity(&metadata, &opened) {
                    return Err(ArtifactIndexError::ChangedDuringScan(path));
                }
                pending.push((child, relative, child_depth));
                continue;
            }
            if !metadata.is_file() || !root.accepts(&path) {
                continue;
            }
            if !canonical_paths.insert(path.clone()) {
                return Err(ArtifactIndexError::DuplicateCanonicalPath(path));
            }
            let relative = normalize_relative_path_with_limits(&relative, limits)?;
            let key = ArtifactKey {
                root_id: root.id.clone(),
                relative_path: relative,
            };
            let portable_path = portable_artifact_identity(&root.id, &key.relative_path)?;
            if !portable_paths.insert(portable_path) {
                return Err(ArtifactIndexError::PortablePathCollision(key.relative_path));
            }
            let mut file = entry.open().map_err(|error| ArtifactIndexError::Io {
                path: path.clone(),
                message: error.to_string(),
            })?;
            let opened = file.metadata().map_err(|error| ArtifactIndexError::Io {
                path: path.clone(),
                message: error.to_string(),
            })?;
            if !same_capability_file_identity(&metadata, &opened) {
                return Err(ArtifactIndexError::ChangedDuringScan(path));
            }
            let (sha256, after) =
                hash_stable_capability_file(&mut file, &path, &opened, cancellation)?;
            let current =
                directory
                    .metadata(&file_name)
                    .map_err(|error| ArtifactIndexError::Io {
                        path: path.clone(),
                        message: error.to_string(),
                    })?;
            if !same_capability_file_identity(&after, &current) {
                return Err(ArtifactIndexError::ChangedDuringScan(path));
            }
            let record = ArtifactRecord {
                key: key.clone(),
                namespace: root.namespace.clone(),
                canonical_path: path,
                byte_size: after.len(),
                modified_nanoseconds: capability_modified_nanoseconds(&after),
                sha256,
                availability: ArtifactAvailability::Present,
            };
            discovered.insert(key, record);
        }
    }
    Ok(())
}

fn hash_stable_capability_file(
    file: &mut cap_std::fs::File,
    path: &Path,
    before: &cap_std::fs::Metadata,
    cancellation: &CancellationToken,
) -> Result<(String, cap_std::fs::Metadata), ArtifactIndexError> {
    file.seek(std::io::SeekFrom::Start(0))
        .map_err(|error| ArtifactIndexError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    let mut hasher = Sha256::new();
    // Native dispatch workers can have less than 1 MiB of stack available.
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(1024 * 1024)
        .map_err(|_| ArtifactIndexError::AllocationFailed(path.to_path_buf()))?;
    buffer.resize(1024 * 1024, 0_u8);
    loop {
        check_cancelled(cancellation)?;
        let read = file
            .read(&mut buffer)
            .map_err(|error| ArtifactIndexError::Io {
                path: path.to_path_buf(),
                message: error.to_string(),
            })?;
        if read == 0 {
            break;
        }
        let bytes = buffer
            .get(..read)
            .ok_or_else(|| ArtifactIndexError::ChangedDuringScan(path.to_path_buf()))?;
        hasher.update(bytes);
    }
    let after = file.metadata().map_err(|error| ArtifactIndexError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    if !same_capability_file_identity(before, &after) {
        return Err(ArtifactIndexError::ChangedDuringScan(path.to_path_buf()));
    }
    Ok((hex_digest(hasher.finalize()), after))
}

fn modified_nanoseconds(metadata: &Metadata) -> u128 {
    metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

fn capability_modified_nanoseconds(metadata: &cap_std::fs::Metadata) -> u128 {
    metadata
        .modified()
        .ok()
        .and_then(|modified| modified.into_std().duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

fn normalize_relative_path_with_limits(
    path: &Path,
    limits: &ParserLimits,
) -> Result<PathBuf, ArtifactIndexError> {
    let mut normalized = PathBuf::new();
    let path_text = path
        .to_str()
        .ok_or_else(|| ArtifactIndexError::UnsafePath {
            path: path.to_path_buf(),
            reason: "artifact paths must be valid UTF-8".to_owned(),
        })?;
    if path_text.starts_with('/') || path_text.starts_with('\\') || path_text.get(1..2) == Some(":")
    {
        return Err(ArtifactIndexError::UnsafePath {
            path: path.to_path_buf(),
            reason: "artifact paths must be relative".to_owned(),
        });
    }
    for component in path_text.split(['/', '\\']) {
        match component {
            "" | "." | ".." => {
                return Err(ArtifactIndexError::UnsafePath {
                    path: path.to_path_buf(),
                    reason: "only non-empty relative path components are allowed".to_owned(),
                });
            }
            value => {
                limits.check(
                    "artifact path component bytes",
                    u64::try_from(value.len()).unwrap_or(u64::MAX),
                    limits.maximum_name_bytes,
                )?;
                validate_portable_component(path, value)?;
                normalized.push(value);
            }
        }
    }
    if normalized.as_os_str().is_empty() {
        return Err(ArtifactIndexError::UnsafePath {
            path: path.to_path_buf(),
            reason: "relative artifact path is empty".to_owned(),
        });
    }
    Ok(normalized)
}

fn same_file_identity(left: &Metadata, right: &Metadata) -> bool {
    if left.len() != right.len()
        || modified_nanoseconds(left) != modified_nanoseconds(right)
        || left.file_type() != right.file_type()
    {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        left.dev() == right.dev() && left.ino() == right.ino()
    }
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::fs::MetadataExt;
        matches!(
            (
                left.volume_serial_number(),
                left.file_index(),
                right.volume_serial_number(),
                right.file_index(),
            ),
            (Some(left_volume), Some(left_index), Some(right_volume), Some(right_index))
                if left_volume == right_volume && left_index == right_index
        )
    }
    #[cfg(not(any(unix, target_os = "windows")))]
    {
        true
    }
}

fn same_capability_file_identity(
    left: &cap_std::fs::Metadata,
    right: &cap_std::fs::Metadata,
) -> bool {
    if left.len() != right.len()
        || left.modified().ok() != right.modified().ok()
        || left.file_type() != right.file_type()
    {
        return false;
    }
    #[cfg(unix)]
    {
        use cap_std::fs::MetadataExt;
        left.dev() == right.dev() && left.ino() == right.ino()
    }
    #[cfg(target_os = "windows")]
    {
        use cap_std::fs::MetadataExt;
        matches!(
            (
                left.volume_serial_number(),
                left.file_index(),
                right.volume_serial_number(),
                right.file_index(),
            ),
            (Some(left_volume), Some(left_index), Some(right_volume), Some(right_index))
                if left_volume == right_volume && left_index == right_index
        )
    }
    #[cfg(not(any(unix, target_os = "windows")))]
    {
        true
    }
}

fn same_capability_object_identity(
    left: &cap_std::fs::Metadata,
    right: &cap_std::fs::Metadata,
) -> bool {
    if left.file_type() != right.file_type() {
        return false;
    }
    #[cfg(unix)]
    {
        use cap_std::fs::MetadataExt;
        left.dev() == right.dev() && left.ino() == right.ino()
    }
    #[cfg(target_os = "windows")]
    {
        use cap_std::fs::MetadataExt;
        matches!(
            (
                left.volume_serial_number(),
                left.file_index(),
                right.volume_serial_number(),
                right.file_index(),
            ),
            (Some(left_volume), Some(left_index), Some(right_volume), Some(right_index))
                if left_volume == right_volume && left_index == right_index
        )
    }
    #[cfg(not(any(unix, target_os = "windows")))]
    {
        false
    }
}

#[cfg(target_os = "macos")]
fn publish_opened_capability_file(
    source: &cap_std::fs::File,
    destination_parent: &Dir,
    destination_name: &Path,
    destination_path: &Path,
    policy: ArtifactWritePolicy,
) -> Result<(), ArtifactIndexError> {
    use std::{ffi::CString, os::fd::AsRawFd, os::unix::ffi::OsStrExt};

    if policy == ArtifactWritePolicy::Replace {
        return Err(ArtifactIndexError::Io {
            path: destination_path.to_path_buf(),
            message: "atomic handle-based replacement is unavailable on macOS".to_owned(),
        });
    }
    let destination_name = CString::new(destination_name.as_os_str().as_bytes()).map_err(|_| {
        ArtifactIndexError::UnsafePath {
            path: destination_path.to_path_buf(),
            reason: "contained artifact destination contains a null byte".to_owned(),
        }
    })?;
    let result = unsafe {
        libc::fclonefileat(
            source.as_raw_fd(),
            destination_parent.as_raw_fd(),
            destination_name.as_ptr(),
            0,
        )
    };
    if result == 0 {
        return Ok(());
    }
    let error = std::io::Error::last_os_error();
    if error.kind() == std::io::ErrorKind::AlreadyExists {
        Err(ArtifactIndexError::AlreadyExists(
            destination_path.to_path_buf(),
        ))
    } else {
        Err(ArtifactIndexError::Io {
            path: destination_path.to_path_buf(),
            message: format!("atomic handle publication failed: {error}"),
        })
    }
}

#[cfg(target_os = "linux")]
fn publish_opened_capability_file(
    source: &cap_std::fs::File,
    destination_parent: &Dir,
    destination_name: &Path,
    destination_path: &Path,
    policy: ArtifactWritePolicy,
) -> Result<(), ArtifactIndexError> {
    use std::{ffi::CString, os::fd::AsRawFd, os::unix::ffi::OsStrExt};

    if policy == ArtifactWritePolicy::Replace {
        return Err(ArtifactIndexError::Io {
            path: destination_path.to_path_buf(),
            message: "atomic handle-based replacement is unavailable on Linux".to_owned(),
        });
    }
    let destination_name = CString::new(destination_name.as_os_str().as_bytes()).map_err(|_| {
        ArtifactIndexError::UnsafePath {
            path: destination_path.to_path_buf(),
            reason: "contained artifact destination contains a null byte".to_owned(),
        }
    })?;
    let direct = unsafe {
        libc::linkat(
            source.as_raw_fd(),
            c"".as_ptr(),
            destination_parent.as_raw_fd(),
            destination_name.as_ptr(),
            libc::AT_EMPTY_PATH,
        )
    };
    if direct == 0 {
        return Ok(());
    }
    let direct_error = std::io::Error::last_os_error();
    let descriptor_path =
        CString::new(format!("/proc/self/fd/{}", source.as_raw_fd())).map_err(|error| {
            ArtifactIndexError::Io {
                path: destination_path.to_path_buf(),
                message: error.to_string(),
            }
        })?;
    let fallback = unsafe {
        libc::linkat(
            libc::AT_FDCWD,
            descriptor_path.as_ptr(),
            destination_parent.as_raw_fd(),
            destination_name.as_ptr(),
            libc::AT_SYMLINK_FOLLOW,
        )
    };
    if fallback == 0 {
        return Ok(());
    }
    let error = std::io::Error::last_os_error();
    if error.kind() == std::io::ErrorKind::AlreadyExists {
        Err(ArtifactIndexError::AlreadyExists(
            destination_path.to_path_buf(),
        ))
    } else {
        Err(ArtifactIndexError::Io {
            path: destination_path.to_path_buf(),
            message: format!(
                "atomic handle publication failed directly ({direct_error}) and through procfs ({error})"
            ),
        })
    }
}

#[cfg(all(unix, not(any(target_os = "macos", target_os = "linux"))))]
fn publish_opened_capability_file(
    source: &cap_std::fs::File,
    destination_parent: &Dir,
    destination_name: &Path,
    destination_path: &Path,
    policy: ArtifactWritePolicy,
) -> Result<(), ArtifactIndexError> {
    use std::{ffi::CString, os::fd::AsRawFd, os::unix::ffi::OsStrExt};

    if policy == ArtifactWritePolicy::Replace {
        return Err(ArtifactIndexError::Io {
            path: destination_path.to_path_buf(),
            message: "atomic handle-based replacement is unavailable on this Unix platform"
                .to_owned(),
        });
    }
    let destination_name = CString::new(destination_name.as_os_str().as_bytes()).map_err(|_| {
        ArtifactIndexError::UnsafePath {
            path: destination_path.to_path_buf(),
            reason: "contained artifact destination contains a null byte".to_owned(),
        }
    })?;
    let descriptor_path =
        CString::new(format!("/dev/fd/{}", source.as_raw_fd())).map_err(|error| {
            ArtifactIndexError::Io {
                path: destination_path.to_path_buf(),
                message: error.to_string(),
            }
        })?;
    let result = unsafe {
        libc::linkat(
            libc::AT_FDCWD,
            descriptor_path.as_ptr(),
            destination_parent.as_raw_fd(),
            destination_name.as_ptr(),
            libc::AT_SYMLINK_FOLLOW,
        )
    };
    if result == 0 {
        return Ok(());
    }
    let error = std::io::Error::last_os_error();
    if error.kind() == std::io::ErrorKind::AlreadyExists {
        Err(ArtifactIndexError::AlreadyExists(
            destination_path.to_path_buf(),
        ))
    } else {
        Err(ArtifactIndexError::Io {
            path: destination_path.to_path_buf(),
            message: format!("atomic handle publication through /dev/fd failed: {error}"),
        })
    }
}

#[cfg(not(any(unix, target_os = "windows")))]
fn publish_opened_capability_file(
    _source: &cap_std::fs::File,
    _destination_parent: &Dir,
    _destination_name: &Path,
    destination_path: &Path,
    _policy: ArtifactWritePolicy,
) -> Result<(), ArtifactIndexError> {
    Err(ArtifactIndexError::Io {
        path: destination_path.to_path_buf(),
        message: "atomic handle publication is unavailable on this platform".to_owned(),
    })
}

#[cfg(target_os = "windows")]
fn publish_opened_capability_file(
    source: &cap_std::fs::File,
    destination_parent: &Dir,
    destination_name: &Path,
    destination_path: &Path,
    policy: ArtifactWritePolicy,
) -> Result<(), ArtifactIndexError> {
    use std::{mem, os::windows::ffi::OsStrExt, os::windows::io::AsRawHandle, ptr};
    use windows::Win32::{
        Foundation::{ERROR_ALREADY_EXISTS, ERROR_FILE_EXISTS, HANDLE},
        Storage::FileSystem::{
            FILE_RENAME_INFO, FILE_RENAME_INFO_0, FileRenameInfo, SetFileInformationByHandle,
        },
    };

    let wide_name = destination_name
        .as_os_str()
        .encode_wide()
        .collect::<Vec<_>>();
    let name_bytes = wide_name
        .len()
        .checked_mul(mem::size_of::<u16>())
        .ok_or_else(|| ArtifactIndexError::UnsafePath {
            path: destination_path.to_path_buf(),
            reason: "contained artifact destination name is too large".to_owned(),
        })?;
    let buffer_bytes = mem::offset_of!(FILE_RENAME_INFO, FileName)
        .checked_add(name_bytes)
        .ok_or_else(|| ArtifactIndexError::AllocationFailed(destination_path.to_path_buf()))?;
    let word_bytes = mem::size_of::<usize>();
    let words = buffer_bytes
        .checked_add(word_bytes - 1)
        .and_then(|bytes| bytes.checked_div(word_bytes))
        .ok_or_else(|| ArtifactIndexError::AllocationFailed(destination_path.to_path_buf()))?;
    let mut buffer = Vec::<usize>::new();
    buffer
        .try_reserve_exact(words)
        .map_err(|_| ArtifactIndexError::AllocationFailed(destination_path.to_path_buf()))?;
    buffer.resize(words, 0);
    let information = buffer.as_mut_ptr().cast::<FILE_RENAME_INFO>();
    let file_name_length =
        u32::try_from(name_bytes).map_err(|_| ArtifactIndexError::UnsafePath {
            path: destination_path.to_path_buf(),
            reason: "contained artifact destination name is too large".to_owned(),
        })?;
    let information_size = u32::try_from(buffer_bytes)
        .map_err(|_| ArtifactIndexError::AllocationFailed(destination_path.to_path_buf()))?;
    unsafe {
        (*information).Anonymous = FILE_RENAME_INFO_0 {
            ReplaceIfExists: policy == ArtifactWritePolicy::Replace,
        };
        (*information).RootDirectory = HANDLE(destination_parent.as_raw_handle());
        (*information).FileNameLength = file_name_length;
        ptr::copy_nonoverlapping(
            wide_name.as_ptr(),
            ptr::addr_of_mut!((*information).FileName).cast::<u16>(),
            wide_name.len(),
        );
        let result = SetFileInformationByHandle(
            HANDLE(source.as_raw_handle()),
            FileRenameInfo,
            information.cast(),
            information_size,
        );
        match result {
            Ok(()) => Ok(()),
            Err(error)
                if policy == ArtifactWritePolicy::Reject
                    && matches!(
                        (error.code().0 as u32) & 0xffff,
                        code if code == ERROR_ALREADY_EXISTS.0 || code == ERROR_FILE_EXISTS.0
                    ) =>
            {
                Err(ArtifactIndexError::AlreadyExists(
                    destination_path.to_path_buf(),
                ))
            }
            Err(error) => Err(ArtifactIndexError::Io {
                path: destination_path.to_path_buf(),
                message: format!("atomic handle rename failed: {error}"),
            }),
        }
    }
}

fn reject_symbolic_link_components(path: &Path) -> Result<(), ArtifactIndexError> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| ArtifactIndexError::Io {
                path: path.to_path_buf(),
                message: error.to_string(),
            })?
            .join(path)
    };
    #[cfg(target_os = "macos")]
    let absolute = if absolute.starts_with("/var")
        || absolute.starts_with("/tmp")
        || absolute.starts_with("/etc")
    {
        Path::new("/private").join(absolute.strip_prefix("/").map_err(|error| {
            ArtifactIndexError::UnsafePath {
                path: absolute.clone(),
                reason: error.to_string(),
            }
        })?)
    } else {
        absolute
    };
    let mut inspected = PathBuf::new();
    for component in absolute.components() {
        inspected.push(component.as_os_str());
        if matches!(component, Component::Prefix(_) | Component::RootDir) {
            continue;
        }
        let metadata =
            fs::symlink_metadata(&inspected).map_err(|error| ArtifactIndexError::Io {
                path: inspected.clone(),
                message: error.to_string(),
            })?;
        if metadata.file_type().is_symlink() {
            return Err(ArtifactIndexError::SymbolicLink(inspected));
        }
    }
    Ok(())
}

#[cfg(unix)]
fn sync_capability_directory(directory: &Dir) -> std::io::Result<()> {
    // Linux capability directories can use O_PATH, which cannot be fsynced.
    // Reopen the same directory through its handle to retain rename safety.
    directory.open(".")?.sync_all()
}

#[cfg(not(unix))]
fn sync_capability_directory(_directory: &Dir) -> std::io::Result<()> {
    Ok(())
}

fn validate_portable_component(path: &Path, component: &str) -> Result<(), ArtifactIndexError> {
    if component.contains(':')
        || component.ends_with(['.', ' '])
        || component.chars().any(char::is_control)
    {
        return Err(ArtifactIndexError::UnsafePath {
            path: path.to_path_buf(),
            reason: "artifact path component is not portable across supported platforms".to_owned(),
        });
    }
    let device_name = component
        .split_once('.')
        .map_or(component, |(name, _)| name)
        .to_ascii_uppercase();
    let reserved = matches!(device_name.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || device_name
            .strip_prefix("COM")
            .or_else(|| device_name.strip_prefix("LPT"))
            .is_some_and(|suffix| {
                suffix.len() == 1
                    && suffix
                        .as_bytes()
                        .first()
                        .is_some_and(|byte| matches!(byte, b'1'..=b'9'))
            });
    if reserved {
        return Err(ArtifactIndexError::UnsafePath {
            path: path.to_path_buf(),
            reason: "artifact path uses a reserved platform device name".to_owned(),
        });
    }
    Ok(())
}

fn portable_artifact_identity(
    root_id: &str,
    relative_path: &Path,
) -> Result<String, ArtifactIndexError> {
    let relative_path = relative_path
        .to_str()
        .ok_or_else(|| ArtifactIndexError::UnsafePath {
            path: relative_path.to_path_buf(),
            reason: "artifact paths must be valid UTF-8".to_owned(),
        })?;
    Ok(format!(
        "{}:{}",
        root_id.to_lowercase(),
        relative_path.replace('\\', "/").to_lowercase()
    ))
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<(), ArtifactIndexError> {
    if cancellation.is_cancelled() {
        Err(ArtifactIndexError::Cancelled)
    } else {
        Ok(())
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn index_detects_changes_and_retains_missing_records() -> Result<(), Box<dyn std::error::Error>>
    {
        let directory = tempfile::tempdir()?;
        let model = directory.path().join("model.safetensors");
        File::create(&model)?.write_all(b"first")?;
        let root =
            ArtifactRoot::canonical("models", "checkpoints", directory.path(), ["safetensors"])?;
        let mut index = ArtifactIndex::default();
        index.add_root(root)?;
        let cancellation = CancellationToken::default();
        assert!(matches!(
            index.refresh(&cancellation)?.as_slice(),
            [ArtifactChange {
                kind: ArtifactChangeKind::Added,
                ..
            }]
        ));
        File::create(&model)?.write_all(b"second")?;
        assert!(matches!(
            index.refresh(&cancellation)?.as_slice(),
            [ArtifactChange {
                kind: ArtifactChangeKind::Modified,
                ..
            }]
        ));
        fs::remove_file(&model)?;
        assert!(matches!(
            index.refresh(&cancellation)?.as_slice(),
            [ArtifactChange {
                kind: ArtifactChangeKind::Missing,
                ..
            }]
        ));
        assert_eq!(
            index.records().next().map(|record| &record.availability),
            Some(&ArtifactAvailability::Missing)
        );
        Ok(())
    }

    #[test]
    fn traversal_and_symlinks_are_rejected() -> Result<(), Box<dyn std::error::Error>> {
        assert!(ArtifactKey::new("models", "../outside").is_err());
        assert!(ArtifactKey::new("models", "..\\outside").is_err());
        for unsafe_path in [
            "CON",
            "aux.txt",
            "nested/LPT9.bin",
            "model.safetensors.",
            "model.safetensors ",
            "model:safetensors",
            "line\nbreak.safetensors",
        ] {
            assert!(
                ArtifactKey::new("models", unsafe_path).is_err(),
                "{unsafe_path:?} must be rejected"
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let directory = tempfile::tempdir()?;
            let outside = tempfile::NamedTempFile::new()?;
            symlink(outside.path(), directory.path().join("link.safetensors"))?;
            let root = ArtifactRoot::canonical(
                "models",
                "checkpoints",
                directory.path(),
                ["safetensors"],
            )?;
            let mut index = ArtifactIndex::default();
            index.add_root(root)?;
            assert!(matches!(
                index.refresh(&CancellationToken::default()),
                Err(ArtifactIndexError::SymbolicLink(_))
            ));

            let root_directory = tempfile::tempdir()?;
            let real_directory = root_directory.path().join("real");
            fs::create_dir(&real_directory)?;
            File::create(real_directory.join("model.safetensors"))?.write_all(b"model")?;
            symlink(&real_directory, root_directory.path().join("linked"))?;
            let root = ArtifactRoot::canonical(
                "nested",
                "checkpoints",
                root_directory.path(),
                ["safetensors"],
            )?;
            assert!(matches!(
                root.resolve_existing("linked/model.safetensors"),
                Err(ArtifactIndexError::SymbolicLink(_))
            ));
            assert!(matches!(
                root.resolve_for_create("linked/new.safetensors"),
                Err(ArtifactIndexError::SymbolicLink(_))
            ));
        }
        Ok(())
    }

    #[test]
    fn case_insensitive_artifact_aliases_are_rejected() -> Result<(), Box<dyn std::error::Error>> {
        let uppercase = ArtifactKey::new("Models", "Nested/Model.bin")?;
        let lowercase = ArtifactKey::new("models", "nested/model.bin")?;
        assert_eq!(
            portable_artifact_identity(&uppercase.root_id, &uppercase.relative_path)?,
            portable_artifact_identity(&lowercase.root_id, &lowercase.relative_path)?
        );
        Ok(())
    }

    #[test]
    fn root_owns_checked_existing_and_create_resolution() -> Result<(), Box<dyn std::error::Error>>
    {
        let directory = tempfile::tempdir()?;
        fs::create_dir(directory.path().join("nested"))?;
        let existing = directory.path().join("nested/model.safetensors");
        File::create(&existing)?.write_all(b"model")?;
        let root =
            ArtifactRoot::canonical("models", "checkpoints", directory.path(), ["safetensors"])?;
        assert_eq!(
            root.resolve_existing("nested/model.safetensors")?,
            fs::canonicalize(existing)?
        );
        assert_eq!(
            root.resolve_for_create("nested/new.safetensors")?,
            root.canonical_path().join("nested/new.safetensors")
        );
        assert!(matches!(
            root.resolve_for_create("../escape.safetensors"),
            Err(ArtifactIndexError::UnsafePath { .. })
        ));
        Ok(())
    }

    #[test]
    fn recursive_regular_file_listing_is_nested_bounded_and_sorted()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        fs::create_dir_all(directory.path().join("nested/deeper"))?;
        File::create(directory.path().join("z.bin"))?.write_all(b"z")?;
        File::create(directory.path().join("nested/b.bin"))?.write_all(b"b")?;
        File::create(directory.path().join("nested/deeper/a.bin"))?.write_all(b"a")?;
        let root = ArtifactRoot::canonical("package", "package", directory.path(), ["bin"])?;
        let cancellation = CancellationToken::default();

        assert_eq!(
            root.list_contained_regular_files_recursive(5, &cancellation)?,
            [
                PathBuf::from("nested/b.bin"),
                PathBuf::from("nested/deeper/a.bin"),
                PathBuf::from("z.bin"),
            ]
        );
        assert!(matches!(
            root.list_contained_regular_files_recursive(0, &cancellation),
            Err(ArtifactIndexError::Limit(ParserLimitError::Zero(
                "contained recursive entry limit"
            )))
        ));
        assert!(matches!(
            root.list_contained_regular_files_recursive(4, &cancellation),
            Err(ArtifactIndexError::Limit(ParserLimitError::Exceeded {
                kind: "contained recursive entries",
                actual: 5,
                maximum: 4,
            }))
        ));
        Ok(())
    }

    #[test]
    fn recursive_regular_file_listing_rejects_cancellation_and_import_roots()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let artifact = directory.path().join("model.bin");
        File::create(&artifact)?.write_all(b"model")?;
        let root = ArtifactRoot::canonical("package", "package", directory.path(), ["bin"])?;
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        assert_eq!(
            root.list_contained_regular_files_recursive(1, &cancellation),
            Err(ArtifactIndexError::Cancelled)
        );

        let imported = ArtifactRoot::approved_import("import", "package", &artifact)?;
        assert!(matches!(
            imported.list_contained_regular_files_recursive(1, &CancellationToken::default()),
            Err(ArtifactIndexError::UnsafePath { reason, .. })
                if reason == "approved import roots cannot be traversed recursively"
        ));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn recursive_regular_file_listing_rejects_links_and_special_files()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::{ffi::CString, os::unix::ffi::OsStrExt, os::unix::fs::symlink};

        let linked_directory = tempfile::tempdir()?;
        let outside = tempfile::NamedTempFile::new()?;
        let link = linked_directory.path().join("linked.bin");
        symlink(outside.path(), &link)?;
        let linked_root =
            ArtifactRoot::canonical("linked", "package", linked_directory.path(), ["bin"])?;
        let canonical_link = linked_root.canonical_path().join("linked.bin");
        assert_eq!(
            linked_root.list_contained_regular_files_recursive(1, &CancellationToken::default()),
            Err(ArtifactIndexError::SymbolicLink(canonical_link))
        );

        let special_directory = tempfile::tempdir()?;
        let fifo_path = special_directory.path().join("service.fifo");
        let fifo_path_c = CString::new(fifo_path.as_os_str().as_bytes())?;
        // SAFETY: `fifo_path_c` is a live, NUL-terminated path and the mode contains only
        // permission bits. The temporary directory owns cleanup after the assertion.
        if unsafe { libc::mkfifo(fifo_path_c.as_ptr(), 0o600) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let special_root = ArtifactRoot::canonical(
            "special",
            "package",
            special_directory.path(),
            std::iter::empty::<String>(),
        )?;
        let canonical_fifo_path = special_root.canonical_path().join("service.fifo");
        assert!(matches!(
            special_root
                .list_contained_regular_files_recursive(1, &CancellationToken::default()),
            Err(ArtifactIndexError::UnsafePath { path, reason })
                if path == canonical_fifo_path
                    && reason == "contained recursive entry is not a regular file or directory"
        ));
        Ok(())
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    #[test]
    fn recursive_regular_file_listing_rejects_non_utf8_components()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::{ffi::OsString, os::unix::ffi::OsStringExt};

        let directory = tempfile::tempdir()?;
        let invalid_name = OsString::from_vec(vec![b'm', b'o', b'd', b'e', b'l', 0xff]);
        File::create(directory.path().join(invalid_name))?.write_all(b"model")?;
        let root = ArtifactRoot::canonical(
            "package",
            "package",
            directory.path(),
            std::iter::empty::<String>(),
        )?;
        assert!(matches!(
            root.list_contained_regular_files_recursive(1, &CancellationToken::default()),
            Err(ArtifactIndexError::UnsafePath { reason, .. })
                if reason == "artifact paths must be valid UTF-8"
        ));
        Ok(())
    }

    #[test]
    fn snapshot_round_trip_is_deterministic() -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        File::create(directory.path().join("a.bin"))?.write_all(b"a")?;
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "models",
            "checkpoints",
            directory.path(),
            ["bin"],
        )?)?;
        index.refresh(&CancellationToken::default())?;
        let snapshot = index.snapshot()?;
        assert_eq!(
            ArtifactIndex::from_snapshot(&snapshot, index.roots().cloned())?.snapshot()?,
            snapshot
        );
        Ok(())
    }

    #[test]
    fn approved_imports_are_isolated_to_the_selected_files()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let approved = directory.path().join("approved.safetensors");
        let second_approved = directory.path().join("second.safetensors");
        File::create(&approved)?.write_all(b"approved")?;
        File::create(&second_approved)?.write_all(b"second")?;
        File::create(directory.path().join("sibling.safetensors"))?.write_all(b"sibling")?;

        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::approved_import(
            "first",
            "checkpoints",
            &approved,
        )?)?;
        index.add_root(ArtifactRoot::approved_import(
            "second",
            "checkpoints",
            &second_approved,
        )?)?;
        index.refresh(&CancellationToken::default())?;

        assert_eq!(index.records().count(), 2);
        assert!(
            index
                .records()
                .all(|record| record.key.relative_path != Path::new("sibling.safetensors"))
        );
        Ok(())
    }

    #[test]
    fn snapshots_are_bounded_before_deserialization() {
        let oversized = vec![
            b' ';
            usize::try_from(ParserLimits::default().manifest_bytes)
                .unwrap_or(usize::MAX)
                .saturating_add(1)
        ];
        assert!(matches!(
            ArtifactIndex::from_snapshot(&oversized, []),
            Err(ArtifactIndexError::InvalidSnapshot(_))
        ));
    }

    #[test]
    fn snapshot_roots_are_bound_to_trusted_configuration() -> Result<(), Box<dyn std::error::Error>>
    {
        let trusted_directory = tempfile::tempdir()?;
        let substituted_directory = tempfile::tempdir()?;
        File::create(trusted_directory.path().join("model.bin"))?.write_all(b"trusted")?;
        File::create(substituted_directory.path().join("model.bin"))?.write_all(b"substituted")?;
        let trusted =
            ArtifactRoot::canonical("models", "checkpoints", trusted_directory.path(), ["bin"])?;
        let substituted = ArtifactRoot::canonical(
            "models",
            "checkpoints",
            substituted_directory.path(),
            ["bin"],
        )?;
        let mut index = ArtifactIndex::default();
        index.add_root(trusted.clone())?;
        index.refresh(&CancellationToken::default())?;
        let snapshot = index.snapshot()?;

        assert!(matches!(
            ArtifactIndex::from_snapshot(&snapshot, [substituted]),
            Err(ArtifactIndexError::InvalidSnapshot(message))
                if message.contains("trusted configured roots")
        ));
        assert!(ArtifactIndex::from_snapshot(&snapshot, [trusted]).is_ok());
        Ok(())
    }

    #[test]
    fn snapshots_reject_noncanonical_records() -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        File::create(directory.path().join("model.bin"))?.write_all(b"model")?;
        let root = ArtifactRoot::canonical("models", "checkpoints", directory.path(), ["bin"])?;
        let mut index = ArtifactIndex::default();
        index.add_root(root.clone())?;
        index.refresh(&CancellationToken::default())?;
        let snapshot = index.snapshot()?;

        let mut uppercase_digest: serde_json::Value = serde_json::from_slice(&snapshot)?;
        let digest = uppercase_digest
            .pointer_mut("/records/0/sha256")
            .and_then(|value| value.as_str())
            .ok_or("snapshot digest is missing")?
            .to_ascii_uppercase();
        *uppercase_digest
            .pointer_mut("/records/0/sha256")
            .ok_or("snapshot digest is missing")? = serde_json::Value::String(digest);
        assert!(
            ArtifactIndex::from_snapshot(&serde_json::to_vec(&uppercase_digest)?, [root.clone()])
                .is_err()
        );

        let mut wrong_extension: serde_json::Value = serde_json::from_slice(&snapshot)?;
        *wrong_extension
            .pointer_mut("/records/0/key/relative_path")
            .ok_or("snapshot key path is missing")? =
            serde_json::Value::String("model.txt".to_owned());
        *wrong_extension
            .pointer_mut("/records/0/canonical_path")
            .ok_or("snapshot canonical path is missing")? = serde_json::Value::String(
            root.canonical_path()
                .join("model.txt")
                .display()
                .to_string(),
        );
        assert!(
            ArtifactIndex::from_snapshot(&serde_json::to_vec(&wrong_extension)?, [root.clone()])
                .is_err()
        );

        let mut portable_duplicate: serde_json::Value = serde_json::from_slice(&snapshot)?;
        let mut duplicate = portable_duplicate
            .pointer("/records/0")
            .cloned()
            .ok_or("snapshot record is missing")?;
        *duplicate
            .pointer_mut("/key/relative_path")
            .ok_or("snapshot key path is missing")? =
            serde_json::Value::String("MODEL.bin".to_owned());
        *duplicate
            .pointer_mut("/canonical_path")
            .ok_or("snapshot canonical path is missing")? = serde_json::Value::String(
            root.canonical_path()
                .join("MODEL.bin")
                .display()
                .to_string(),
        );
        portable_duplicate
            .get_mut("records")
            .and_then(serde_json::Value::as_array_mut)
            .ok_or("snapshot records are missing")?
            .push(duplicate);
        assert!(
            ArtifactIndex::from_snapshot(&serde_json::to_vec(&portable_duplicate)?, [root])
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn verified_handles_reject_changes_until_the_index_refreshes()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("model.bin");
        File::create(&path)?.write_all(b"first")?;
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "models",
            "checkpoints",
            directory.path(),
            ["bin"],
        )?)?;
        let cancellation = CancellationToken::default();
        index.refresh(&cancellation)?;
        let key = ArtifactKey::new("models", "model.bin")?;
        let mut verified = index.open_verified(&key, &cancellation)?.into_file();
        fs::rename(&path, directory.path().join("original.ignored"))?;
        File::create(&path)?.write_all(b"other")?;
        let mut bytes = Vec::new();
        verified.read_to_end(&mut bytes)?;
        assert_eq!(bytes, b"first");

        assert!(matches!(
            index.open_verified(&key, &cancellation),
            Err(ArtifactIndexError::ChangedSinceIndex(changed)) if changed == key
        ));
        index.refresh(&cancellation)?;
        let mut verified = index.open_verified(&key, &cancellation)?.into_file();
        bytes.clear();
        verified.read_to_end(&mut bytes)?;
        assert_eq!(bytes, b"other");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn verified_handles_reject_parent_and_final_symlink_replacement()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir()?;
        let outside = tempfile::tempdir()?;
        fs::create_dir(directory.path().join("nested"))?;
        File::create(directory.path().join("nested/model.bin"))?.write_all(b"inside")?;
        File::create(outside.path().join("model.bin"))?.write_all(b"outside")?;
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "models",
            "checkpoints",
            directory.path(),
            ["bin"],
        )?)?;
        let cancellation = CancellationToken::default();
        index.refresh(&cancellation)?;
        let key = ArtifactKey::new("models", "nested/model.bin")?;

        fs::rename(
            directory.path().join("nested"),
            directory.path().join("original.ignored"),
        )?;
        symlink(outside.path(), directory.path().join("nested"))?;
        assert!(matches!(
            index.open_verified(&key, &cancellation),
            Err(ArtifactIndexError::SymbolicLink(_))
        ));

        fs::remove_file(directory.path().join("nested"))?;
        fs::create_dir(directory.path().join("nested"))?;
        symlink(
            outside.path().join("model.bin"),
            directory.path().join("nested/model.bin"),
        )?;
        assert!(matches!(
            index.open_verified(&key, &cancellation),
            Err(ArtifactIndexError::SymbolicLink(_))
        ));
        Ok(())
    }

    #[test]
    fn root_ids_cannot_alias_on_case_insensitive_platforms()
    -> Result<(), Box<dyn std::error::Error>> {
        let first = tempfile::tempdir()?;
        let second = tempfile::tempdir()?;
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "Models",
            "first",
            first.path(),
            ["bin"],
        )?)?;
        assert!(matches!(
            index.add_root(ArtifactRoot::canonical(
                "models",
                "second",
                second.path(),
                ["bin"],
            )?),
            Err(ArtifactIndexError::PortableRootCollision(_))
        ));
        Ok(())
    }

    #[test]
    fn scan_depth_entry_count_and_snapshot_bytes_are_bounded()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        File::create(directory.path().join("one.bin"))?.write_all(b"one")?;
        File::create(directory.path().join("two.bin"))?.write_all(b"two")?;
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "models",
            "checkpoints",
            directory.path(),
            ["bin"],
        )?)?;
        let entry_limits = ParserLimits {
            maximum_tensors: 1,
            maximum_archive_entries: 1,
            ..ParserLimits::default()
        };
        assert!(matches!(
            index.refresh_with_limits(&CancellationToken::default(), &entry_limits),
            Err(ArtifactIndexError::Limit(ParserLimitError::Exceeded {
                kind: "artifact scan entries",
                ..
            }))
        ));

        fs::remove_file(directory.path().join("two.bin"))?;
        fs::create_dir_all(directory.path().join("nested/deeper"))?;
        let depth_limits = ParserLimits {
            maximum_depth: 1,
            ..ParserLimits::default()
        };
        assert!(matches!(
            index.refresh_with_limits(&CancellationToken::default(), &depth_limits),
            Err(ArtifactIndexError::Limit(ParserLimitError::Exceeded {
                kind: "artifact scan depth",
                ..
            }))
        ));

        fs::remove_dir_all(directory.path().join("nested"))?;
        index.refresh(&CancellationToken::default())?;
        let snapshot_limits = ParserLimits {
            manifest_bytes: 32,
            ..ParserLimits::default()
        };
        assert!(matches!(
            index.snapshot_with_limits(&snapshot_limits),
            Err(ArtifactIndexError::InvalidSnapshot(_))
        ));
        Ok(())
    }

    #[test]
    fn nested_directory_roots_are_rejected_but_disjoint_imports_are_allowed()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let nested = directory.path().join("nested");
        fs::create_dir(&nested)?;
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "parent",
            "models",
            directory.path(),
            ["bin"],
        )?)?;
        assert!(matches!(
            index.add_root(ArtifactRoot::canonical(
                "child",
                "models",
                &nested,
                ["bin"],
            )?),
            Err(ArtifactIndexError::DuplicateCanonicalPath(_))
        ));
        Ok(())
    }

    #[test]
    fn capability_hashing_preserves_exact_bytes_on_a_small_native_worker_stack()
    -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("input.bin");
        let bytes = vec![7_u8; 1024 * 1024 + 17];
        fs::write(&path, &bytes)?;
        let expected = hex_digest(Sha256::digest(&bytes));
        let expected_length = bytes.len() as u64;
        let worker = std::thread::Builder::new()
            .name("artifact-hash-small-stack".into())
            .stack_size(256 * 1024)
            .spawn(move || -> anyhow::Result<()> {
                let mut file = cap_std::fs::File::from_std(File::open(&path)?);
                let before = file.metadata()?;
                let cancellation = CancellationToken::default();
                let (digest, after) =
                    hash_stable_capability_file(&mut file, &path, &before, &cancellation)?;
                assert_eq!(digest, expected);
                assert_eq!(after.len(), expected_length);
                assert!(same_capability_file_identity(&before, &after));
                cancellation.cancel();
                assert!(matches!(
                    hash_stable_capability_file(&mut file, &path, &after, &cancellation),
                    Err(ArtifactIndexError::Cancelled)
                ));
                Ok(())
            })?;
        worker
            .join()
            .map_err(|_| anyhow::anyhow!("small-stack artifact hashing worker panicked"))??;
        Ok(())
    }

    #[test]
    fn selected_refresh_preserves_unselected_records_and_does_not_scan_their_roots()
    -> Result<(), Box<dyn std::error::Error>> {
        let selected = tempfile::tempdir()?;
        let unselected = tempfile::tempdir()?;
        File::create(selected.path().join("selected.bin"))?.write_all(b"selected")?;
        File::create(unselected.path().join("unselected.bin"))?.write_all(b"unselected")?;
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "selected",
            "input",
            selected.path(),
            ["bin"],
        )?)?;
        index.add_root(ArtifactRoot::canonical(
            "unselected",
            "output",
            unselected.path(),
            ["bin"],
        )?)?;
        let cancellation = CancellationToken::default();
        index.refresh(&cancellation)?;
        fs::remove_file(unselected.path().join("unselected.bin"))?;

        let changes = index.refresh_selected(["selected"], &cancellation)?;
        assert!(changes.is_empty());
        assert_eq!(
            index
                .record(&ArtifactKey::new("unselected", "unselected.bin")?)
                .map(|record| &record.availability),
            Some(&ArtifactAvailability::Present)
        );
        let changes = index.refresh_selected(["unselected"], &cancellation)?;
        let unselected_key = ArtifactKey::new("unselected", "unselected.bin")?;
        assert!(changes.iter().any(|change| {
            change.key == unselected_key && change.kind == ArtifactChangeKind::Missing
        }));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn root_owned_parent_creation_rejects_intermediate_links()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir()?;
        let outside = tempfile::tempdir()?;
        let root = ArtifactRoot::canonical("input", "input", directory.path(), ["bin"])?;
        let created = root.resolve_for_create_with_parents("nested/deeper/file.bin")?;
        assert_eq!(
            created,
            root.canonical_path().join("nested/deeper/file.bin")
        );
        fs::remove_dir_all(directory.path().join("nested"))?;
        symlink(outside.path(), directory.path().join("nested"))?;
        assert!(matches!(
            root.resolve_for_create_with_parents("nested/deeper/file.bin"),
            Err(ArtifactIndexError::SymbolicLink(_))
        ));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn trusted_root_admission_rejects_symbolic_link_ancestors()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir()?;
        let real_parent = directory.path().join("real-parent");
        let root_path = real_parent.join("root");
        fs::create_dir_all(&root_path)?;
        let linked_parent = directory.path().join("linked-parent");
        symlink(&real_parent, &linked_parent)?;

        let error = ArtifactRoot::canonical(
            "input",
            "input",
            &linked_parent.join("root"),
            std::iter::empty::<String>(),
        )
        .expect_err("a root reached through a symbolic-link ancestor must be rejected");
        assert!(matches!(
            error,
            ArtifactIndexError::SymbolicLink(path)
                if path.file_name() == linked_parent.file_name()
        ));
        Ok(())
    }

    #[test]
    fn trusted_root_admission_rejects_identity_replacement_during_construction()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let configured = directory.path().join("configured");
        let retained = directory.path().join("retained");
        let replacement = directory.path().join("replacement");
        fs::create_dir(&configured)?;
        fs::create_dir(&replacement)?;

        let error = ArtifactRoot::new_with_admission_hook(
            "input",
            "input",
            &configured,
            std::iter::empty::<String>(),
            false,
            None,
            || {
                fs::rename(&configured, &retained).expect("retain the originally inspected root");
                fs::rename(&replacement, &configured).expect("replace the configured root path");
            },
        )
        .expect_err("root identity replacement during admission must be rejected");
        assert!(matches!(
            error,
            ArtifactIndexError::ChangedDuringScan(path)
                if path.file_name() == configured.file_name()
        ));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn contained_file_publication_is_anchored_to_the_open_parent_directory()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir()?;
        let outside = tempfile::tempdir()?;
        let nested = directory.path().join("nested");
        let retained = directory.path().join("retained");
        fs::create_dir(&nested)?;
        let root = ArtifactRoot::canonical(
            "input",
            "input",
            directory.path(),
            std::iter::empty::<String>(),
        )?;

        root.write_contained_file_inner(
            Path::new("nested/asset.bin"),
            b"inside",
            ArtifactWritePolicy::Reject,
            || {
                fs::rename(&nested, &retained).expect("rename the admitted parent");
                symlink(outside.path(), &nested).expect("replace it with an escaping link");
            },
        )?;

        assert_eq!(fs::read(retained.join("asset.bin"))?, b"inside");
        assert!(!outside.path().join("asset.bin").exists());
        assert_eq!(fs::read_dir(&retained)?.count(), 1);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn contained_file_writes_fail_closed_when_the_root_path_is_replaced()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir()?;
        let configured = directory.path().join("configured");
        let retained = directory.path().join("retained");
        let outside = tempfile::tempdir()?;
        fs::create_dir(&configured)?;
        let root =
            ArtifactRoot::canonical("input", "input", &configured, std::iter::empty::<String>())?;
        fs::rename(&configured, &retained)?;
        symlink(outside.path(), &configured)?;

        let error = root
            .write_contained_file("asset.bin", b"inside", ArtifactWritePolicy::Reject)
            .expect_err("a replaced configured root must fail before publication");

        assert!(matches!(error, ArtifactIndexError::InvalidSnapshot(_)));
        assert!(!retained.join("asset.bin").exists());
        assert!(!outside.path().join("asset.bin").exists());
        Ok(())
    }

    #[test]
    fn reject_move_never_overwrites_a_destination_created_before_publication()
    -> Result<(), Box<dyn std::error::Error>> {
        let source_directory = tempfile::tempdir()?;
        let destination_directory = tempfile::tempdir()?;
        let source = ArtifactRoot::canonical(
            "source",
            "temporary",
            source_directory.path(),
            std::iter::empty::<String>(),
        )?;
        let destination = ArtifactRoot::canonical(
            "destination",
            "output",
            destination_directory.path(),
            std::iter::empty::<String>(),
        )?;
        source.write_contained_file("asset.bin", b"source", ArtifactWritePolicy::Reject)?;

        let error = source
            .move_verified_contained_file_to_inner(
                Path::new("asset.bin"),
                &destination,
                Path::new("asset.bin"),
                ArtifactWritePolicy::Reject,
                &hex_digest(Sha256::digest(b"source")),
                6,
                &CancellationToken::default(),
                || {},
                || {},
                || {
                    fs::write(destination_directory.path().join("asset.bin"), b"racer")
                        .expect("create the racing destination");
                },
                || {},
            )
            .expect_err("reject move must not overwrite a racing destination");

        assert!(matches!(error, ArtifactIndexError::AlreadyExists(_)));
        assert_eq!(
            fs::read(source_directory.path().join("asset.bin"))?,
            b"source"
        );
        assert_eq!(
            fs::read(destination_directory.path().join("asset.bin"))?,
            b"racer"
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn verified_move_replace_fails_closed_without_a_handle_replace_primitive()
    -> Result<(), Box<dyn std::error::Error>> {
        let source_directory = tempfile::tempdir()?;
        let destination_directory = tempfile::tempdir()?;
        let source = ArtifactRoot::canonical(
            "source",
            "temporary",
            source_directory.path(),
            std::iter::empty::<String>(),
        )?;
        let destination = ArtifactRoot::canonical(
            "destination",
            "output",
            destination_directory.path(),
            std::iter::empty::<String>(),
        )?;
        source.write_contained_file("asset.bin", b"source", ArtifactWritePolicy::Reject)?;
        destination.write_contained_file(
            "asset.bin",
            b"destination",
            ArtifactWritePolicy::Reject,
        )?;

        let error = source
            .move_verified_contained_file_to(
                "asset.bin",
                &destination,
                "asset.bin",
                ArtifactWritePolicy::Replace,
                &hex_digest(Sha256::digest(b"source")),
                6,
                &CancellationToken::default(),
            )
            .expect_err("Unix verified move replacement must fail closed");

        assert!(matches!(error, ArtifactIndexError::Io { .. }));
        assert_eq!(
            fs::read(source_directory.path().join("asset.bin"))?,
            b"source"
        );
        assert_eq!(
            fs::read(destination_directory.path().join("asset.bin"))?,
            b"destination"
        );
        Ok(())
    }

    #[test]
    fn verified_move_never_publishes_a_replaced_source_name()
    -> Result<(), Box<dyn std::error::Error>> {
        let source_directory = tempfile::tempdir()?;
        let destination_directory = tempfile::tempdir()?;
        let source = ArtifactRoot::canonical(
            "source",
            "temporary",
            source_directory.path(),
            std::iter::empty::<String>(),
        )?;
        let destination = ArtifactRoot::canonical(
            "destination",
            "output",
            destination_directory.path(),
            std::iter::empty::<String>(),
        )?;
        source.write_contained_file("asset.bin", b"verified", ArtifactWritePolicy::Reject)?;

        let error = source
            .move_verified_contained_file_to_inner(
                Path::new("asset.bin"),
                &destination,
                Path::new("asset.bin"),
                ArtifactWritePolicy::Reject,
                &hex_digest(Sha256::digest(b"verified")),
                8,
                &CancellationToken::default(),
                || {},
                || {},
                || {
                    fs::rename(
                        source_directory.path().join("asset.bin"),
                        source_directory.path().join("verified-retained.bin"),
                    )
                    .expect("retain the verified source");
                    fs::write(source_directory.path().join("asset.bin"), b"foreign")
                        .expect("replace the source name");
                },
                || {},
            )
            .expect_err("source identity replacement must be reported as a partial move");

        assert!(matches!(error, ArtifactIndexError::PartialMove { .. }));
        assert_eq!(
            fs::read(destination_directory.path().join("asset.bin"))?,
            b"verified"
        );
        assert_eq!(
            fs::read(source_directory.path().join("asset.bin"))?,
            b"foreign"
        );
        Ok(())
    }

    #[test]
    fn verified_move_rejects_same_inode_mutation_after_source_validation()
    -> Result<(), Box<dyn std::error::Error>> {
        let source_directory = tempfile::tempdir()?;
        let destination_directory = tempfile::tempdir()?;
        let source = ArtifactRoot::canonical(
            "source",
            "temporary",
            source_directory.path(),
            std::iter::empty::<String>(),
        )?;
        let destination = ArtifactRoot::canonical(
            "destination",
            "output",
            destination_directory.path(),
            std::iter::empty::<String>(),
        )?;
        source.write_contained_file("asset.bin", b"verified", ArtifactWritePolicy::Reject)?;

        let error = source
            .move_verified_contained_file_to_inner(
                Path::new("asset.bin"),
                &destination,
                Path::new("asset.bin"),
                ArtifactWritePolicy::Reject,
                &hex_digest(Sha256::digest(b"verified")),
                8,
                &CancellationToken::default(),
                || {
                    fs::write(source_directory.path().join("asset.bin"), b"mutated!")
                        .expect("mutate the admitted source inode");
                },
                || {},
                || {},
                || {},
            )
            .expect_err("mutated bytes must not be published");

        assert!(matches!(error, ArtifactIndexError::ChangedDuringScan(_)));
        assert_eq!(
            fs::read(source_directory.path().join("asset.bin"))?,
            b"mutated!"
        );
        assert!(!destination_directory.path().join("asset.bin").exists());
        Ok(())
    }

    #[test]
    fn verified_move_publishes_the_opened_inode_when_the_temporary_name_is_swapped()
    -> Result<(), Box<dyn std::error::Error>> {
        let source_directory = tempfile::tempdir()?;
        let destination_directory = tempfile::tempdir()?;
        let source = ArtifactRoot::canonical(
            "source",
            "temporary",
            source_directory.path(),
            std::iter::empty::<String>(),
        )?;
        let destination = ArtifactRoot::canonical(
            "destination",
            "output",
            destination_directory.path(),
            std::iter::empty::<String>(),
        )?;
        source.write_contained_file("asset.bin", b"verified", ArtifactWritePolicy::Reject)?;
        let retained_temporary = destination_directory.path().join("retained-temporary.bin");

        source.move_verified_contained_file_to_inner(
            Path::new("asset.bin"),
            &destination,
            Path::new("asset.bin"),
            ArtifactWritePolicy::Reject,
            &hex_digest(Sha256::digest(b"verified")),
            8,
            &CancellationToken::default(),
            || {},
            || {},
            || {
                let temporary_path = fs::read_dir(destination_directory.path())
                    .expect("enumerate the destination")
                    .filter_map(Result::ok)
                    .map(|entry| entry.path())
                    .find(|path| {
                        path.file_name()
                            .and_then(|name| name.to_str())
                            .is_some_and(|name| name.contains(".move.") && name.ends_with(".tmp"))
                    })
                    .expect("find the contained move temporary");
                fs::rename(&temporary_path, &retained_temporary)
                    .expect("retain the verified temporary inode");
                fs::write(&temporary_path, b"foreign!")
                    .expect("replace the temporary name with foreign bytes");
            },
            || {
                assert_eq!(
                    fs::read(destination_directory.path().join("asset.bin"))
                        .expect("read the newly published output"),
                    b"verified"
                );
            },
        )?;

        assert!(!source_directory.path().join("asset.bin").exists());
        assert_eq!(fs::read(retained_temporary)?, b"verified");
        assert_eq!(
            fs::read(destination_directory.path().join("asset.bin"))?,
            b"verified"
        );
        let foreign_temporary = fs::read_dir(destination_directory.path())?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.contains(".move.") && name.ends_with(".tmp"))
            })
            .ok_or("foreign temporary is missing")?;
        assert_eq!(fs::read(foreign_temporary)?, b"foreign!");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn verified_move_seals_temporary_bytes_before_handle_publication()
    -> Result<(), Box<dyn std::error::Error>> {
        let source_directory = tempfile::tempdir()?;
        let destination_directory = tempfile::tempdir()?;
        let source = ArtifactRoot::canonical(
            "source",
            "temporary",
            source_directory.path(),
            std::iter::empty::<String>(),
        )?;
        let destination = ArtifactRoot::canonical(
            "destination",
            "output",
            destination_directory.path(),
            std::iter::empty::<String>(),
        )?;
        source.write_contained_file("asset.bin", b"verified", ArtifactWritePolicy::Reject)?;

        source.move_verified_contained_file_to_inner(
            Path::new("asset.bin"),
            &destination,
            Path::new("asset.bin"),
            ArtifactWritePolicy::Reject,
            &hex_digest(Sha256::digest(b"verified")),
            8,
            &CancellationToken::default(),
            || {},
            || {},
            || {
                let temporary_path = fs::read_dir(destination_directory.path())
                    .expect("enumerate the destination")
                    .filter_map(Result::ok)
                    .map(|entry| entry.path())
                    .find(|path| {
                        path.file_name()
                            .and_then(|name| name.to_str())
                            .is_some_and(|name| name.contains(".move.") && name.ends_with(".tmp"))
                    })
                    .expect("find the contained move temporary");
                fs::write(temporary_path, b"foreign!")
                    .expect_err("sealed temporary bytes must reject mutation");
            },
            || {
                assert_eq!(
                    fs::read(destination_directory.path().join("asset.bin"))
                        .expect("observe the published output"),
                    b"verified"
                );
            },
        )?;

        assert_eq!(
            fs::read(destination_directory.path().join("asset.bin"))?,
            b"verified"
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn partial_reject_move_preserves_both_names_for_owner_recovery()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::PermissionsExt;

        let source_directory = tempfile::tempdir()?;
        let destination_directory = tempfile::tempdir()?;
        let source = ArtifactRoot::canonical(
            "source",
            "temporary",
            source_directory.path(),
            std::iter::empty::<String>(),
        )?;
        let destination = ArtifactRoot::canonical(
            "destination",
            "output",
            destination_directory.path(),
            std::iter::empty::<String>(),
        )?;
        source.write_contained_file("asset.bin", b"source", ArtifactWritePolicy::Reject)?;
        fs::set_permissions(source_directory.path(), fs::Permissions::from_mode(0o555))?;
        let result = source.move_contained_file_to(
            "asset.bin",
            &destination,
            "asset.bin",
            ArtifactWritePolicy::Reject,
        );
        fs::set_permissions(source_directory.path(), fs::Permissions::from_mode(0o755))?;

        assert!(matches!(
            result,
            Err(ArtifactIndexError::PartialMove { .. })
        ));
        assert_eq!(
            fs::read(source_directory.path().join("asset.bin"))?,
            b"source"
        );
        assert_eq!(
            fs::read(destination_directory.path().join("asset.bin"))?,
            b"source"
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn refresh_rejects_a_replaced_configured_root_path() -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir()?;
        let configured = directory.path().join("configured");
        let retained = directory.path().join("retained");
        let outside = tempfile::tempdir()?;
        fs::create_dir(&configured)?;
        fs::write(configured.join("asset.bin"), b"inside")?;
        let root = ArtifactRoot::canonical("input", "input", &configured, ["bin"])?;
        let mut index = ArtifactIndex::default();
        index.add_root(root)?;
        fs::rename(&configured, &retained)?;
        symlink(outside.path(), &configured)?;

        assert!(matches!(
            index.refresh(&CancellationToken::default()),
            Err(ArtifactIndexError::InvalidSnapshot(message))
                if message.contains("not a direct directory")
        ));
        assert_eq!(fs::read(retained.join("asset.bin"))?, b"inside");
        Ok(())
    }

    #[test]
    fn root_owned_private_files_are_atomic_bounded_and_path_identified()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let other = tempfile::tempdir()?;
        let root = ArtifactRoot::canonical_with_path_identity(
            "model-configured",
            "model",
            directory.path(),
            std::iter::empty::<String>(),
        )?;
        let same = ArtifactRoot::canonical_with_path_identity(
            "model-configured",
            "model",
            directory.path(),
            std::iter::empty::<String>(),
        )?;
        let different = ArtifactRoot::canonical_with_path_identity(
            "model-configured",
            "model",
            other.path(),
            std::iter::empty::<String>(),
        )?;
        assert_eq!(root.id(), same.id());
        assert_ne!(root.id(), different.id());

        assert!(root.read_private_file("absent/state.json", 6)?.is_none());
        assert!(root.quarantine_private_file("absent/state.json")?.is_none());

        root.write_private_file("nested/state.json", b"first")?;
        assert_eq!(
            root.read_private_file("nested/state.json", 5)?,
            Some(b"first".to_vec())
        );
        assert!(matches!(
            root.read_private_file("nested/state.json", 4),
            Err(ArtifactIndexError::InvalidSnapshot(_))
        ));
        root.write_private_file("nested/state.json", b"second")?;
        assert_eq!(
            root.read_private_file("nested/state.json", 6)?,
            Some(b"second".to_vec())
        );
        let quarantine = root
            .quarantine_private_file("nested/state.json")?
            .ok_or("private file was not quarantined")?;
        assert!(root.read_private_file("nested/state.json", 6)?.is_none());
        assert_eq!(
            root.read_private_file(&quarantine, 6)?,
            Some(b"second".to_vec())
        );
        assert!(
            quarantine
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("state.json.quarantine."))
        );
        assert!(root.quarantine_private_file("nested/state.json")?.is_none());
        Ok(())
    }

    #[test]
    fn artifact_root_cancellable_private_capture_is_bounded_and_race_safe()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let root =
            ArtifactRoot::canonical("package", "general-video-codec", directory.path(), ["bin"])?;
        let payload = vec![7_u8; PRIVATE_ARTIFACT_CAPTURE_CHUNK_BYTES + 17];
        fs::write(directory.path().join("library.bin"), &payload)?;

        let captured = root
            .capture_private_file("library.bin", payload.len(), &CancellationToken::default())?
            .ok_or("captured artifact is absent")?;
        assert_eq!(captured.as_bytes(), payload);
        assert_eq!(captured.len(), payload.len());
        assert!(!captured.is_empty());
        assert_eq!(
            captured.digest_sha256(),
            format!("{:x}", Sha256::digest(&payload))
        );
        assert!(matches!(
            root.capture_private_file(
                "library.bin",
                payload.len() - 1,
                &CancellationToken::default()
            ),
            Err(ArtifactIndexError::InvalidSnapshot(_))
        ));

        let cancellation = CancellationToken::default();
        cancellation.cancel();
        assert!(matches!(
            root.capture_private_file("library.bin", payload.len(), &cancellation),
            Err(ArtifactIndexError::Cancelled)
        ));

        let cancellation = CancellationToken::default();
        assert!(matches!(
            root.capture_private_file_with_hook(
                Path::new("library.bin"),
                payload.len(),
                &cancellation,
                |_| {
                    cancellation.cancel();
                },
            ),
            Err(ArtifactIndexError::Cancelled)
        ));

        let path = directory.path().join("library.bin");
        let replacement = vec![8_u8; payload.len() + 1];
        let mut mutated = false;
        assert!(matches!(
            root.capture_private_file_with_hook(
                Path::new("library.bin"),
                replacement.len(),
                &CancellationToken::default(),
                |_| {
                    if !mutated {
                        fs::write(&path, &replacement).expect("mutate captured file");
                        mutated = true;
                    }
                },
            ),
            Err(ArtifactIndexError::ChangedDuringScan(_))
        ));

        let retry = root
            .capture_private_file(
                "library.bin",
                replacement.len(),
                &CancellationToken::default(),
            )?
            .ok_or("retry artifact is absent")?;
        assert_eq!(retry.as_bytes(), replacement);
        Ok(())
    }

    #[test]
    fn contained_file_writes_have_exact_collision_and_replacement_semantics()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let root = ArtifactRoot::canonical(
            "input",
            "input",
            directory.path(),
            std::iter::empty::<String>(),
        )?;

        root.write_contained_file("nested/asset.bin", b"first", ArtifactWritePolicy::Reject)?;
        assert_eq!(
            fs::read(directory.path().join("nested/asset.bin"))?,
            b"first"
        );
        let collision = root
            .write_contained_file("nested/asset.bin", b"rejected", ArtifactWritePolicy::Reject)
            .expect_err("reject policy must preserve an existing file");
        assert!(
            matches!(
                collision,
                ArtifactIndexError::AlreadyExists(ref path)
                    if path == &root.canonical_path().join("nested/asset.bin")
            ),
            "unexpected collision error: {collision:?}"
        );
        assert_eq!(
            fs::read(directory.path().join("nested/asset.bin"))?,
            b"first"
        );

        root.write_contained_file(
            "nested/asset.bin",
            b"replacement",
            ArtifactWritePolicy::Replace,
        )?;
        assert_eq!(
            fs::read(directory.path().join("nested/asset.bin"))?,
            b"replacement"
        );
        assert!(
            root.write_contained_file("../escape.bin", b"escape", ArtifactWritePolicy::Reject)
                .is_err()
        );
        assert!(!directory.path().join("escape.bin").exists());
        assert_eq!(fs::read_dir(directory.path().join("nested"))?.count(), 1);
        assert!(root.remove_contained_file("nested/asset.bin")?);
        assert!(!root.remove_contained_file("nested/asset.bin")?);
        assert!(
            fs::read_dir(directory.path().join("nested"))?
                .next()
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn snapshot_reconciliation_migrates_stable_roots_and_drops_removed_roots()
    -> Result<(), Box<dyn std::error::Error>> {
        let retained = tempfile::tempdir()?;
        let removed = tempfile::tempdir()?;
        let added = tempfile::tempdir()?;
        fs::write(retained.path().join("retained.bin"), b"retained")?;
        fs::write(removed.path().join("removed.bin"), b"removed")?;

        let legacy_retained =
            ArtifactRoot::canonical("model-configured-0", "model", retained.path(), ["bin"])?;
        let legacy_removed =
            ArtifactRoot::canonical("model-configured-1", "model", removed.path(), ["bin"])?;
        let mut previous = ArtifactIndex::default();
        previous.add_root(legacy_retained.clone())?;
        previous.add_root(legacy_removed)?;
        previous.refresh(&CancellationToken::default())?;
        let snapshot = previous.snapshot()?;

        let current_retained = ArtifactRoot::canonical_with_path_identity(
            "model-configured",
            "model",
            retained.path(),
            ["bin"],
        )?;
        let current_added = ArtifactRoot::canonical_with_path_identity(
            "model-configured",
            "model",
            added.path(),
            ["bin"],
        )?;
        let (reconciled, evidence) = ArtifactIndex::reconcile_snapshot(
            &snapshot,
            [current_added, current_retained.clone()],
        )?;
        assert!(evidence.changed());
        assert_eq!(evidence.removed_root_count(), 2);
        assert_eq!(evidence.added_root_count(), 2);
        assert_eq!(evidence.dropped_record_count(), 1);
        assert_eq!(
            evidence.current_root_id(legacy_retained.id()),
            Some(current_retained.id())
        );
        assert!(
            reconciled
                .record(&ArtifactKey::new(current_retained.id(), "retained.bin")?)
                .is_some()
        );
        assert_eq!(reconciled.records().count(), 1);
        assert_eq!(reconciled.roots().count(), 2);
        Ok(())
    }

    #[test]
    fn snapshot_reconciliation_rejects_changed_scope_for_a_trusted_root_id()
    -> Result<(), Box<dyn std::error::Error>> {
        let trusted_directory = tempfile::tempdir()?;
        let substituted_directory = tempfile::tempdir()?;
        let trusted = ArtifactRoot::canonical("input", "input", trusted_directory.path(), ["bin"])?;
        let substituted =
            ArtifactRoot::canonical("input", "input", substituted_directory.path(), ["bin"])?;
        let snapshot = serde_json::to_vec(&ArtifactIndexSnapshot {
            version: ARTIFACT_INDEX_VERSION,
            roots: vec![substituted],
            records: Vec::new(),
        })?;
        assert!(matches!(
            ArtifactIndex::reconcile_snapshot(&snapshot, [trusted]),
            Err(ArtifactIndexError::InvalidSnapshot(message))
                if message.contains("changed its trusted scope")
        ));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn root_owned_private_files_reject_parent_and_final_links()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir()?;
        let outside = tempfile::tempdir()?;
        let root = ArtifactRoot::canonical(
            "state",
            "state",
            directory.path(),
            std::iter::empty::<String>(),
        )?;
        symlink(outside.path(), directory.path().join("linked"))?;
        assert!(matches!(
            root.write_private_file("linked/state.json", b"outside"),
            Err(ArtifactIndexError::SymbolicLink(_))
        ));

        fs::create_dir(directory.path().join("nested"))?;
        let outside_file = outside.path().join("state.json");
        fs::write(&outside_file, b"outside")?;
        symlink(&outside_file, directory.path().join("nested/state.json"))?;
        assert!(matches!(
            root.read_private_file("nested/state.json", 64),
            Err(ArtifactIndexError::SymbolicLink(_))
        ));
        assert!(matches!(
            root.write_private_file("nested/state.json", b"replacement"),
            Err(ArtifactIndexError::SymbolicLink(_))
        ));
        assert_eq!(fs::read(outside_file)?, b"outside");
        Ok(())
    }

    #[test]
    fn adding_a_deserialized_root_revalidates_the_trusted_filesystem_boundary()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let forged: ArtifactRoot = serde_json::from_value(serde_json::json!({
            "id": "forged",
            "namespace": "input",
            "canonical_path": directory.path().join("missing"),
            "approved_extensions": [],
            "imported": false,
            "approved_relative_path": null,
        }))?;
        let mut index = ArtifactIndex::default();
        assert!(matches!(
            index.add_root(forged),
            Err(ArtifactIndexError::InvalidSnapshot(_))
        ));
        Ok(())
    }
}
