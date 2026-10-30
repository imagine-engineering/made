use std::{
    collections::HashSet,
    env,
    ffi::OsStr,
    fmt, fs,
    path::{Path, PathBuf},
};

pub const AGENT_PROVIDER_PREFIX: &str = "buzz-backend-";
pub const SUPPORTED_AGENT_PROVIDER_PROTOCOL_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentProviderTrust {
    Trusted,
    Untrusted,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentProviderSearchDirectory {
    pub path: PathBuf,
    pub trust: AgentProviderTrust,
}

impl AgentProviderSearchDirectory {
    pub fn new(path: impl Into<PathBuf>, trust: AgentProviderTrust) -> Self {
        Self {
            path: path.into(),
            trust,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentProviderCandidate {
    pub provider_id: String,
    pub discovered_path: PathBuf,
    pub canonical_path: PathBuf,
    pub trust: AgentProviderTrust,
}

impl AgentProviderCandidate {
    pub fn executable_reference(&self) -> AgentProviderExecutableReference {
        AgentProviderExecutableReference {
            provider_id: self.provider_id.clone(),
            canonical_path: self.canonical_path.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscoveredAgentProvider {
    pub selected: AgentProviderCandidate,
    pub shadowed: Vec<AgentProviderCandidate>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentProviderExecutableReference {
    pub provider_id: String,
    pub canonical_path: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentProviderRejectionReason {
    MalformedProviderId,
    NonUtf8Name,
    NotAFile,
    NotExecutable,
    Inaccessible,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RejectedAgentProviderCandidate {
    pub path: PathBuf,
    pub reason: AgentProviderRejectionReason,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AgentProviderDiscoveryReport {
    providers: Vec<DiscoveredAgentProvider>,
    rejected: Vec<RejectedAgentProviderCandidate>,
}

impl AgentProviderDiscoveryReport {
    pub fn discover(
        search_directories: impl IntoIterator<Item = AgentProviderSearchDirectory>,
    ) -> Self {
        let mut report = Self::default();
        let mut searched_directories = HashSet::new();

        for search_directory in search_directories {
            let directory_identity = search_directory
                .path
                .canonicalize()
                .unwrap_or_else(|_| search_directory.path.clone());
            if !searched_directories.insert(directory_identity.clone()) {
                continue;
            }
            let Ok(entries) = fs::read_dir(&search_directory.path) else {
                continue;
            };
            let mut entries = entries
                .filter_map(Result::ok)
                .collect::<Vec<fs::DirEntry>>();
            entries.sort_by_key(fs::DirEntry::file_name);

            for entry in entries {
                let discovered_path = entry.path();
                let filename = entry.file_name();
                let Some(filename) = filename.to_str() else {
                    if filename
                        .to_string_lossy()
                        .starts_with(AGENT_PROVIDER_PREFIX)
                    {
                        report.reject(discovered_path, AgentProviderRejectionReason::NonUtf8Name);
                    }
                    continue;
                };
                let Some(provider_id) = provider_id_from_filename(filename) else {
                    if filename.starts_with(AGENT_PROVIDER_PREFIX) {
                        report.reject(
                            discovered_path,
                            AgentProviderRejectionReason::MalformedProviderId,
                        );
                    }
                    continue;
                };
                let metadata = match fs::metadata(&discovered_path) {
                    Ok(metadata) => metadata,
                    Err(_) => {
                        report.reject(discovered_path, AgentProviderRejectionReason::Inaccessible);
                        continue;
                    }
                };
                if !metadata.is_file() {
                    report.reject(discovered_path, AgentProviderRejectionReason::NotAFile);
                    continue;
                }
                if !is_executable(&discovered_path, &metadata) {
                    report.reject(discovered_path, AgentProviderRejectionReason::NotExecutable);
                    continue;
                }
                let canonical_path = match discovered_path.canonicalize() {
                    Ok(path) => path,
                    Err(_) => {
                        report.reject(discovered_path, AgentProviderRejectionReason::Inaccessible);
                        continue;
                    }
                };
                let trust = if search_directory.trust == AgentProviderTrust::Trusted
                    && canonical_path.starts_with(&directory_identity)
                {
                    AgentProviderTrust::Trusted
                } else {
                    AgentProviderTrust::Untrusted
                };
                report.add_candidate(AgentProviderCandidate {
                    provider_id: provider_id.to_owned(),
                    discovered_path,
                    canonical_path,
                    trust,
                });
            }
        }

        report
    }

    pub fn providers(&self) -> &[DiscoveredAgentProvider] {
        &self.providers
    }

    pub fn rejected(&self) -> &[RejectedAgentProviderCandidate] {
        &self.rejected
    }

    pub fn resolve(
        &self,
        provider_id: &str,
    ) -> Result<&AgentProviderCandidate, AgentProviderDiscoveryError> {
        validate_provider_id(provider_id)?;
        self.providers
            .iter()
            .find(|provider| provider.selected.provider_id == provider_id)
            .map(|provider| &provider.selected)
            .ok_or_else(|| AgentProviderDiscoveryError::MissingProvider {
                provider_id: provider_id.to_owned(),
            })
    }

    pub fn resolve_trusted(
        &self,
        provider_id: &str,
    ) -> Result<&AgentProviderCandidate, AgentProviderDiscoveryError> {
        let candidate = self.resolve(provider_id)?;
        if candidate.trust != AgentProviderTrust::Trusted {
            return Err(AgentProviderDiscoveryError::UntrustedProvider {
                provider_id: provider_id.to_owned(),
                path: candidate.canonical_path.clone(),
            });
        }
        Ok(candidate)
    }

    pub fn resolve_reference(
        &self,
        reference: &AgentProviderExecutableReference,
    ) -> Result<&AgentProviderCandidate, AgentProviderDiscoveryError> {
        let candidate = self.resolve(&reference.provider_id)?;
        if candidate.canonical_path != reference.canonical_path {
            return Err(AgentProviderDiscoveryError::StaleExecutableReference {
                provider_id: reference.provider_id.clone(),
                cached_path: reference.canonical_path.clone(),
                current_path: candidate.canonical_path.clone(),
            });
        }
        Ok(candidate)
    }

    fn add_candidate(&mut self, candidate: AgentProviderCandidate) {
        if let Some(provider) = self
            .providers
            .iter_mut()
            .find(|provider| provider.selected.provider_id == candidate.provider_id)
        {
            provider.shadowed.push(candidate);
        } else {
            self.providers.push(DiscoveredAgentProvider {
                selected: candidate,
                shadowed: Vec::new(),
            });
        }
    }

    fn reject(&mut self, path: PathBuf, reason: AgentProviderRejectionReason) {
        self.rejected
            .push(RejectedAgentProviderCandidate { path, reason });
    }
}

pub fn discover_agent_providers() -> AgentProviderDiscoveryReport {
    AgentProviderDiscoveryReport::discover(agent_provider_search_directories(
        env::current_exe().ok().as_deref(),
        env::var_os("PATH").as_deref(),
        Some(paths::home_dir()),
    ))
}

pub fn agent_provider_search_directories(
    current_executable: Option<&Path>,
    path: Option<&OsStr>,
    home_directory: Option<&Path>,
) -> Vec<AgentProviderSearchDirectory> {
    let mut directories = Vec::new();
    if let Some(executable_directory) = current_executable.and_then(Path::parent) {
        directories.push(AgentProviderSearchDirectory::new(
            executable_directory,
            AgentProviderTrust::Trusted,
        ));
    }
    if let Some(path) = path {
        directories
            .extend(env::split_paths(path).map(|path| {
                AgentProviderSearchDirectory::new(path, AgentProviderTrust::Untrusted)
            }));
    }
    if let Some(home_directory) = home_directory {
        directories.push(AgentProviderSearchDirectory::new(
            home_directory.join(".local/bin"),
            AgentProviderTrust::Untrusted,
        ));
    }
    directories
}

pub fn validate_agent_provider_protocol_version(
    candidate: &AgentProviderCandidate,
    declared_version: Option<u32>,
) -> Result<(), AgentProviderDiscoveryError> {
    let Some(declared_version) = declared_version else {
        return Err(AgentProviderDiscoveryError::MissingProtocolVersion {
            provider_id: candidate.provider_id.clone(),
            path: candidate.canonical_path.clone(),
        });
    };
    if declared_version != SUPPORTED_AGENT_PROVIDER_PROTOCOL_VERSION {
        return Err(AgentProviderDiscoveryError::IncompatibleProtocolVersion {
            provider_id: candidate.provider_id.clone(),
            path: candidate.canonical_path.clone(),
            expected: SUPPORTED_AGENT_PROVIDER_PROTOCOL_VERSION,
            actual: declared_version,
        });
    }
    Ok(())
}

pub fn validate_provider_id(provider_id: &str) -> Result<(), AgentProviderDiscoveryError> {
    if valid_provider_id(provider_id) {
        Ok(())
    } else {
        Err(AgentProviderDiscoveryError::InvalidProviderId {
            provider_id: provider_id.to_owned(),
        })
    }
}

fn provider_id_from_filename(filename: &str) -> Option<&str> {
    let raw_id = filename.strip_prefix(AGENT_PROVIDER_PREFIX)?;
    let provider_id = [".exe", ".bat", ".cmd"]
        .into_iter()
        .find_map(|extension| {
            raw_id
                .get(raw_id.len().saturating_sub(extension.len())..)
                .filter(|suffix| suffix.eq_ignore_ascii_case(extension))
                .map(|_| &raw_id[..raw_id.len() - extension.len()])
        })
        .unwrap_or(raw_id);
    valid_provider_id(provider_id).then_some(provider_id)
}

fn valid_provider_id(provider_id: &str) -> bool {
    provider_id
        .as_bytes()
        .first()
        .is_some_and(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        && provider_id.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_' || byte == b'-'
        })
}

#[cfg(unix)]
fn is_executable(_path: &Path, metadata: &fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt as _;

    metadata.permissions().mode() & 0o111 != 0
}

#[cfg(not(unix))]
fn is_executable(_path: &Path, _metadata: &fs::Metadata) -> bool {
    true
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AgentProviderDiscoveryError {
    InvalidProviderId {
        provider_id: String,
    },
    MissingProvider {
        provider_id: String,
    },
    UntrustedProvider {
        provider_id: String,
        path: PathBuf,
    },
    StaleExecutableReference {
        provider_id: String,
        cached_path: PathBuf,
        current_path: PathBuf,
    },
    MissingProtocolVersion {
        provider_id: String,
        path: PathBuf,
    },
    IncompatibleProtocolVersion {
        provider_id: String,
        path: PathBuf,
        expected: u32,
        actual: u32,
    },
}

impl fmt::Display for AgentProviderDiscoveryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProviderId { provider_id } => write!(
                formatter,
                "invalid agent provider ID {provider_id:?}; expected [a-z0-9][a-z0-9_-]*"
            ),
            Self::MissingProvider { provider_id } => {
                write!(formatter, "agent provider {provider_id:?} is not installed")
            }
            Self::UntrustedProvider { provider_id, path } => write!(
                formatter,
                "agent provider {provider_id:?} at {} is not trusted",
                path.display()
            ),
            Self::StaleExecutableReference {
                provider_id,
                cached_path,
                current_path,
            } => write!(
                formatter,
                "agent provider {provider_id:?} moved from {} to {}; rediscovery is required",
                cached_path.display(),
                current_path.display()
            ),
            Self::MissingProtocolVersion { provider_id, path } => write!(
                formatter,
                "agent provider {provider_id:?} at {} did not declare a protocol version",
                path.display()
            ),
            Self::IncompatibleProtocolVersion {
                provider_id,
                path,
                expected,
                actual,
            } => write!(
                formatter,
                "agent provider {provider_id:?} at {} uses protocol version {actual}; version {expected} is required",
                path.display()
            ),
        }
    }
}

impl std::error::Error for AgentProviderDiscoveryError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn make_executable(path: &Path) {
        use std::os::unix::fs::PermissionsExt as _;

        fs::write(path, b"#!/bin/sh\nexit 0\n").expect("write provider fixture");
        let mut permissions = fs::metadata(path)
            .expect("read provider fixture metadata")
            .permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(path, permissions).expect("mark provider fixture executable");
    }

    #[cfg(not(unix))]
    fn make_executable(path: &Path) {
        fs::write(path, b"provider fixture").expect("write provider fixture");
    }

    fn search(path: &Path, trust: AgentProviderTrust) -> AgentProviderSearchDirectory {
        AgentProviderSearchDirectory::new(path, trust)
    }

    #[test]
    fn discovery_finds_supported_provider_without_executing_it() {
        let directory = tempfile::tempdir().expect("create fixture directory");
        let provider_path = directory.path().join("buzz-backend-kubernetes");
        make_executable(&provider_path);

        let report = AgentProviderDiscoveryReport::discover([search(
            directory.path(),
            AgentProviderTrust::Trusted,
        )]);
        let provider = report
            .resolve_trusted("kubernetes")
            .expect("supported provider should resolve");
        assert_eq!(provider.discovered_path, provider_path);
        assert_eq!(
            provider.canonical_path,
            provider_path.canonicalize().expect("canonical path")
        );
        assert!(report.rejected().is_empty());
        validate_agent_provider_protocol_version(provider, Some(1))
            .expect("supported protocol should pass");
    }

    #[test]
    fn discovery_retains_shadowed_and_rejected_candidates() {
        let first = tempfile::tempdir().expect("create first fixture directory");
        let second = tempfile::tempdir().expect("create second fixture directory");
        let selected = first.path().join("buzz-backend-kubernetes");
        let shadowed = second.path().join("buzz-backend-kubernetes");
        let malformed = second.path().join("buzz-backend-Bad.Provider");
        let non_executable = second.path().join("buzz-backend-disabled");
        make_executable(&selected);
        make_executable(&shadowed);
        make_executable(&malformed);
        fs::write(&non_executable, b"disabled").expect("write non-executable fixture");

        let report = AgentProviderDiscoveryReport::discover([
            search(first.path(), AgentProviderTrust::Trusted),
            search(second.path(), AgentProviderTrust::Untrusted),
        ]);
        let provider = report
            .providers()
            .iter()
            .find(|provider| provider.selected.provider_id == "kubernetes")
            .expect("provider should be discovered");
        assert_eq!(provider.selected.discovered_path, selected);
        assert_eq!(provider.shadowed.len(), 1);
        assert_eq!(provider.shadowed[0].discovered_path, shadowed);
        assert!(report.rejected().iter().any(|candidate| {
            candidate.path == malformed
                && candidate.reason == AgentProviderRejectionReason::MalformedProviderId
        }));
        assert!(report.rejected().iter().any(|candidate| {
            candidate.path == non_executable
                && candidate.reason == AgentProviderRejectionReason::NotExecutable
        }));
    }

    #[test]
    fn resolution_rejects_invalid_missing_and_stale_provider_references() {
        let directory = tempfile::tempdir().expect("create fixture directory");
        let provider_path = directory.path().join("buzz-backend-kubernetes");
        make_executable(&provider_path);
        let report = AgentProviderDiscoveryReport::discover([search(
            directory.path(),
            AgentProviderTrust::Trusted,
        )]);
        assert!(matches!(
            report.resolve("../kubernetes"),
            Err(AgentProviderDiscoveryError::InvalidProviderId { .. })
        ));
        assert!(matches!(
            report.resolve("missing"),
            Err(AgentProviderDiscoveryError::MissingProvider { .. })
        ));
        let mut reference = report
            .resolve("kubernetes")
            .expect("provider should resolve")
            .executable_reference();
        reference.canonical_path = directory.path().join("replaced-provider");
        assert!(matches!(
            report.resolve_reference(&reference),
            Err(AgentProviderDiscoveryError::StaleExecutableReference { .. })
        ));
    }

    #[test]
    fn protocol_gate_rejects_missing_and_incompatible_versions() {
        let directory = tempfile::tempdir().expect("create fixture directory");
        let provider_path = directory.path().join("buzz-backend-kubernetes");
        make_executable(&provider_path);
        let report = AgentProviderDiscoveryReport::discover([search(
            directory.path(),
            AgentProviderTrust::Trusted,
        )]);
        let provider = report
            .resolve("kubernetes")
            .expect("provider should resolve");
        assert!(matches!(
            validate_agent_provider_protocol_version(provider, None),
            Err(AgentProviderDiscoveryError::MissingProtocolVersion { .. })
        ));
        assert!(matches!(
            validate_agent_provider_protocol_version(provider, Some(2)),
            Err(AgentProviderDiscoveryError::IncompatibleProtocolVersion {
                expected: 1,
                actual: 2,
                ..
            })
        ));
    }

    #[test]
    fn untrusted_provider_is_visible_but_cannot_cross_trusted_resolution() {
        let directory = tempfile::tempdir().expect("create fixture directory");
        let provider_path = directory.path().join("buzz-backend-kubernetes");
        make_executable(&provider_path);
        let report = AgentProviderDiscoveryReport::discover([search(
            directory.path(),
            AgentProviderTrust::Untrusted,
        )]);
        let provider = report
            .resolve("kubernetes")
            .expect("untrusted provider remains diagnosable");
        assert_eq!(provider.trust, AgentProviderTrust::Untrusted);
        assert!(matches!(
            report.resolve_trusted("kubernetes"),
            Err(AgentProviderDiscoveryError::UntrustedProvider { .. })
        ));
    }

    #[test]
    fn search_plan_preserves_bundle_path_and_local_bin_order() {
        let path = env::join_paths([Path::new("/path/one"), Path::new("/path/two")])
            .expect("join fixture path");
        let directories = agent_provider_search_directories(
            Some(Path::new("/app/Zed")),
            Some(&path),
            Some(Path::new("/home/user")),
        );
        assert_eq!(
            directories,
            vec![
                search(Path::new("/app"), AgentProviderTrust::Trusted),
                search(Path::new("/path/one"), AgentProviderTrust::Untrusted),
                search(Path::new("/path/two"), AgentProviderTrust::Untrusted),
                search(
                    Path::new("/home/user/.local/bin"),
                    AgentProviderTrust::Untrusted,
                ),
            ]
        );
    }
}
