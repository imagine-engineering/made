use crate::doctor::{
    CheckOutcome, DependencyProbe, DiagnosticText, DoctorCheck, DoctorHistoryStore, DoctorProduct,
    DoctorRepairOperationOutcome, DoctorRepairPhase, DoctorRepairReview, DoctorReport, ProbeEnd,
    ProbeExecutable, project_repair_binding, run_toolchain_command,
};
use anyhow::{Context as _, Result, ensure};
use futures::StreamExt as _;
use gpui::{App, AppContext as _, Entity, Task};
use project::{Project, ProjectPath};
use sha2::{Digest as _, Sha256};
use std::{
    io::Read as _,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

const MISE_VERSION: &str = "2026.10.1";
const MAXIMUM_METADATA: u64 = 2 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MiseJdkRequest {
    pub version: String,
}

impl MiseJdkRequest {
    fn validate(&self) -> Result<()> {
        let (vendor, version) = self
            .version
            .split_once('-')
            .context("Select an exact vendor-qualified JDK version")?;
        ensure!(
            matches!(
                vendor,
                "corretto" | "zulu" | "openjdk" | "microsoft" | "liberica" | "temurin"
            ),
            "This Java vendor has no reviewed delivery adapter"
        );
        let (release, build) = version
            .split_once('+')
            .map_or((version, None), |(release, build)| (release, Some(build)));
        ensure!(
            release.split('.').count() >= 3
                && release.split('.').count() <= 5
                && release.split('.').all(|part| !part.is_empty()
                    && part.bytes().all(|byte| byte.is_ascii_digit())
                    && part.parse::<u32>().is_ok())
                && build.is_none_or(|build| !build.is_empty()
                    && build.bytes().all(|byte| byte.is_ascii_digit())
                    && build.parse::<u32>().is_ok())
                && self.version.len() <= 100,
            "Moving versions, aliases, JREs, options and custom Java backends are unsupported"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FileStamp {
    path: PathBuf,
    inode: u64,
    modified: fs::MTime,
    length: u64,
}

async fn stamp(filesystem: &dyn fs::Fs, path: &Path) -> Result<FileStamp> {
    let canonical = filesystem.canonicalize(path).await?;
    let metadata = filesystem
        .metadata(&canonical)
        .await?
        .context("Reviewed manager input is unavailable")?;
    ensure!(
        !metadata.is_dir && !metadata.is_fifo,
        "Reviewed input is not a regular file"
    );
    Ok(FileStamp {
        path: canonical,
        inode: metadata.inode,
        modified: metadata.mtime,
        length: metadata.len,
    })
}

async fn bounded_file(
    filesystem: &dyn fs::Fs,
    root: &Path,
    path: &Path,
    maximum: u64,
) -> Result<(FileStamp, Vec<u8>)> {
    let before = stamp(filesystem, path).await?;
    ensure!(
        before.path.starts_with(root) && before.length <= maximum,
        "Manager input escapes its scope or exceeds its bound"
    );
    let mut reader = filesystem.open_sync(&before.path).await?;
    let mut bytes = Vec::new();
    std::io::Read::take(&mut reader, maximum + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= maximum && stamp(filesystem, path).await? == before,
        "Manager input changed while inspected"
    );
    Ok((before, bytes))
}

#[cfg(unix)]
fn acquire_manager_guard(directory: &Path) -> Result<std::fs::File> {
    use std::os::unix::fs::{MetadataExt as _, OpenOptionsExt as _};
    let path = directory.join(".made-doctor-install.lock");
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(&path)?;
    ensure!(
        file.metadata()?.is_file() && file.metadata()?.len() == 0,
        "Manager admission path is not an empty regular lock file"
    );
    file.try_lock().map_err(|_| anyhow::anyhow!("Another Doctor operation owns this user manager; diagnose/review again when it finishes"))?;
    ensure!(
        std::fs::symlink_metadata(&path)?.ino() == file.metadata()?.ino()
            && std::fs::canonicalize(&path)? == path,
        "Manager admission path changed while acquired"
    );
    Ok(file)
}

#[cfg(not(unix))]
fn acquire_manager_guard(_directory: &Path) -> Result<std::fs::File> {
    anyhow::bail!("Native user-manager admission is unsupported on this platform")
}

fn supported_mise_digest() -> Result<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => {
            Ok("d225d1c8ef2934a86be93692a19365fb1df1cd6958a0af7b85909514e0c608a7")
        }
        ("macos", "x86_64") => {
            Ok("723e3032f26b9bad34954286e56bc64c8901a9da2b8e3585c550f472a443f600")
        }
        ("linux", "aarch64") if cfg!(target_env = "gnu") => {
            Ok("d4785456a86d1c836f09ccfe00aa6b044b18d29d79aae46a1820c96024cf9c38")
        }
        ("linux", "x86_64") if cfg!(target_env = "gnu") => {
            Ok("31e6859cf639ed4594906da3fcd0fe2055e9daddae75e9786dbe50b3fb3c0f4a")
        }
        _ => anyhow::bail!("The native mise adapter is unavailable on this platform"),
    }
}

fn metadata_source() -> Result<String> {
    let operating_system = match std::env::consts::OS {
        "macos" => "macosx",
        "linux" => "linux",
        _ => anyhow::bail!("Unsupported Java metadata platform"),
    };
    let architecture = match std::env::consts::ARCH {
        "aarch64" => "aarch64",
        "x86_64" => "x86_64",
        _ => anyhow::bail!("Unsupported Java metadata architecture"),
    };
    Ok(format!(
        "https://mise-java.jdx.dev/jvm/ga/{operating_system}/{architecture}.json"
    ))
}

async fn confined_location(
    filesystem: &dyn fs::Fs,
    home: &Path,
    path: &Path,
    roots: &[PathBuf],
) -> Result<()> {
    let mut ancestor = path;
    loop {
        if let Some(metadata) = filesystem.metadata(ancestor).await? {
            ensure!(
                metadata.is_dir && !metadata.is_fifo,
                "Manager location is not a directory"
            );
            let canonical = filesystem.canonicalize(ancestor).await?;
            ensure!(
                canonical.starts_with(home)
                    && !roots.iter().any(|root| canonical.starts_with(root)),
                "Manager location escapes the user scope or enters the project"
            );
            return Ok(());
        }
        ancestor = ancestor
            .parent()
            .context("Manager location has no confined ancestor")?;
    }
}

#[derive(Clone)]
struct MiseSnapshot {
    home: PathBuf,
    data: PathBuf,
    home_inode: u64,
    data_inode: u64,
    executable: FileStamp,
    roots: Vec<PathBuf>,
}

impl MiseSnapshot {
    async fn inspect(
        filesystem: &dyn fs::Fs,
        home: PathBuf,
        executable: PathBuf,
        roots: Vec<PathBuf>,
        expected_digest: &str,
    ) -> Result<Self> {
        let home = filesystem.canonicalize(&home).await?;
        let data = filesystem
            .canonicalize(&home.join(".local/share/mise"))
            .await?;
        let mut canonical_roots = Vec::new();
        for root in roots {
            canonical_roots.push(filesystem.canonicalize(&root).await?);
        }
        confined_location(filesystem, &home, &data, &canonical_roots).await?;
        ensure!(
            executable.is_absolute()
                && !canonical_roots
                    .iter()
                    .any(|root| executable.starts_with(root)),
            "Project aliases cannot select a manager executable"
        );
        let executable = stamp(filesystem, &executable).await?;
        ensure!(
            executable.length <= 210 * 1024 * 1024
                && executable
                    .path
                    .file_name()
                    .is_some_and(|name| name == "mise")
                && !canonical_roots
                    .iter()
                    .any(|root| executable.path.starts_with(root))
                && filesystem
                    .metadata(&executable.path)
                    .await?
                    .is_some_and(|metadata| metadata.is_executable),
            "Select a supported native mise binary outside the project"
        );
        let mut reader = filesystem.open_sync(&executable.path).await?;
        let mut hasher = Sha256::new();
        let mut buffer = [0; 64 * 1024];
        let mut length = 0u64;
        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            length = length.saturating_add(read as u64);
            ensure!(
                length <= 210 * 1024 * 1024,
                "Native manager exceeds its bound"
            );
            hasher.update(buffer.get(..read).context("Invalid binary read length")?);
        }
        ensure!(
            format!("{:x}", hasher.finalize()) == expected_digest
                && stamp(filesystem, &executable.path).await? == executable,
            "Unknown or changed mise binary; only the inspected official release is supported"
        );
        let home_inode = filesystem
            .metadata(&home)
            .await?
            .context("User home missing")?
            .inode;
        let data_inode = filesystem
            .metadata(&data)
            .await?
            .context("Existing mise data missing")?
            .inode;
        let snapshot = Self {
            home,
            data,
            home_inode,
            data_inode,
            executable,
            roots: canonical_roots,
        };
        snapshot.revalidate(filesystem).await?;
        Ok(snapshot)
    }

    async fn revalidate(&self, filesystem: &dyn fs::Fs) -> Result<()> {
        ensure!(
            filesystem.canonicalize(&self.home).await? == self.home
                && filesystem
                    .metadata(&self.home)
                    .await?
                    .is_some_and(|metadata| metadata.is_dir && metadata.inode == self.home_inode)
                && filesystem
                    .canonicalize(&self.home.join(".local/share/mise"))
                    .await?
                    == self.data
                && filesystem
                    .metadata(&self.data)
                    .await?
                    .is_some_and(|metadata| metadata.is_dir && metadata.inode == self.data_inode)
                && stamp(filesystem, &self.executable.path).await? == self.executable,
            "Reviewed manager or user directories changed"
        );
        for relative in [
            "plugins/java",
            "tracked_config_files",
            "tracked-config-files",
            "trusted-configs",
        ] {
            ensure!(
                filesystem
                    .metadata(&self.data.join(relative))
                    .await?
                    .is_none(),
                "External Java plugins or legacy manager migrations require separate existing-manager resolution"
            );
        }
        for relative in ["installs", "downloads", "shims"] {
            confined_location(
                filesystem,
                &self.home,
                &self.data.join(relative),
                &self.roots,
            )
            .await?;
        }
        Ok(())
    }
}

struct MiseCache {
    path: PathBuf,
    directory: Mutex<Option<tempfile::TempDir>>,
}
impl MiseCache {
    fn new(directory: tempfile::TempDir) -> Self {
        Self {
            path: directory.path().to_path_buf(),
            directory: Mutex::new(Some(directory)),
        }
    }
    fn path(&self) -> &Path {
        &self.path
    }
    fn close(&self) -> Result<()> {
        if let Some(directory) = self
            .directory
            .lock()
            .map_err(|_| anyhow::anyhow!("Manager cache cleanup lock unavailable"))?
            .take()
        {
            directory.close()?;
        }
        Ok(())
    }
}
impl Drop for MiseCache {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            log::warn!("Doctor disposable manager cache cleanup failed: {error}");
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MiseStage {
    Inspect,
    Resolve,
    Install,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MiseArtifact {
    pub source: String,
    pub checksum: String,
    pub java_version: String,
}

#[derive(Clone)]
struct ReviewedMetadata {
    identity: FileStamp,
    bytes: Vec<u8>,
    artifact: MiseArtifact,
}

pub struct MiseAction {
    snapshot: MiseSnapshot,
    request: MiseJdkRequest,
    stage: MiseStage,
    cache: OnceLock<Arc<MiseCache>>,
    metadata: Option<ReviewedMetadata>,
}

impl MiseAction {
    pub fn stage(&self) -> MiseStage {
        self.stage
    }
    pub fn request(&self) -> &MiseJdkRequest {
        &self.request
    }
    pub fn executable(&self) -> &Path {
        &self.snapshot.executable.path
    }
    pub fn destination(&self) -> PathBuf {
        self.snapshot
            .data
            .join("installs/java")
            .join(&self.request.version)
    }
    pub fn metadata_source(&self) -> Result<String> {
        metadata_source()
    }
    pub fn artifact(&self) -> Option<&MiseArtifact> {
        self.metadata.as_ref().map(|metadata| &metadata.artifact)
    }
    pub fn command_arguments(&self) -> Vec<String> {
        let mut arguments = ["--no-config", "--no-env", "--no-hooks"]
            .map(str::to_owned)
            .to_vec();
        match self.stage {
            MiseStage::Inspect => arguments.push("--version".into()),
            MiseStage::Resolve => arguments.extend([
                "install".into(),
                "--dry-run".into(),
                format!("core:java@{}", self.request.version),
                "--jobs=1".into(),
            ]),
            MiseStage::Install => arguments.extend([
                "install".into(),
                format!("core:java@{}", self.request.version),
                "--jobs=1".into(),
            ]),
        }
        arguments
    }

    fn environment(&self) -> Result<Vec<(String, String)>> {
        let cache = self
            .cache
            .get()
            .context("Disposable manager scope is unavailable")?
            .path();
        let mut environment = Vec::new();
        for (name, path) in [
            ("MISE_DATA_DIR", self.snapshot.data.as_path()),
            ("MISE_CACHE_DIR", cache),
            ("MISE_CONFIG_DIR", cache.join("config").as_path()),
            ("MISE_STATE_DIR", cache.join("state").as_path()),
            ("MISE_SHIMS_DIR", cache.join("shims").as_path()),
        ] {
            environment.push((
                name.to_owned(),
                path.to_str()
                    .context("Unsupported manager path encoding")?
                    .to_owned(),
            ));
        }
        environment.extend(
            [
                ("PATH", "/usr/bin:/bin"),
                ("MISE_AUTO_UPDATE", "false"),
                ("MISE_DISABLE_UPDATE_WARNING", "true"),
                ("MISE_CACHE_PRUNE_AGE", "0s"),
                ("MISE_NETRC", "false"),
                ("MISE_ENABLE_TOOLS", "java"),
                ("MISE_DISABLE_BACKENDS", "asdf,vfox"),
                ("MISE_GITHUB_GH_CLI_TOKENS", "false"),
                ("MISE_GITHUB_USE_GIT_CREDENTIALS", "false"),
                ("MISE_REGISTRY_FLOATING", "false"),
                ("MISE_USE_VERSIONS_HOST", "false"),
                ("MISE_USE_VERSIONS_HOST_TRACK", "false"),
                ("MISE_FETCH_REMOTE_VERSIONS_CACHE", "365d"),
                ("MISE_HTTP_RETRIES", "0"),
                ("MISE_HTTP_TIMEOUT", "10s"),
                ("MISE_HTTP_DOWNLOAD_TIMEOUT", "300s"),
                ("MISE_COLOR", "false"),
                ("NO_COLOR", "1"),
            ]
            .into_iter()
            .map(|(name, value)| (name.to_owned(), value.to_owned())),
        );
        environment.push((
            "MISE_OFFLINE".into(),
            (self.stage == MiseStage::Inspect).to_string(),
        ));
        Ok(environment)
    }

    async fn revalidate(&self, filesystem: &dyn fs::Fs) -> Result<()> {
        self.snapshot.revalidate(filesystem).await?;
        if let Some(cache) = self.cache.get() {
            ensure!(
                filesystem.canonicalize(cache.path()).await? == cache.path()
                    && filesystem
                        .metadata(cache.path())
                        .await?
                        .is_some_and(|metadata| metadata.is_dir),
                "Disposable manager scope changed or closed"
            );
        }
        let destination = self.destination();
        confined_location(
            filesystem,
            &self.snapshot.home,
            destination.parent().context("Missing install parent")?,
            &self.snapshot.roots,
        )
        .await?;
        if self.stage != MiseStage::Install || self.metadata.is_none() {
            ensure!(
                filesystem.metadata(&destination).await?.is_none(),
                "Existing/partial JDK installation requires separate review; Doctor will not overwrite it"
            );
        }
        if let Some(metadata) = &self.metadata {
            let cache = self.cache.get().context("Manager cache closed")?.path();
            let (identity, bytes) =
                bounded_file(filesystem, cache, &metadata.identity.path, MAXIMUM_METADATA).await?;
            ensure!(
                identity == metadata.identity && bytes == metadata.bytes,
                "Reviewed JDK source or manager metadata changed"
            );
            ensure!(
                metadata
                    .identity
                    .modified
                    .timestamp_for_user()
                    .elapsed()
                    .is_ok_and(|age| age < Duration::from_secs(24 * 60 * 60)),
                "Reviewed manager metadata is no longer fresh"
            );
        }
        Ok(())
    }

    async fn resolved_metadata(&self, filesystem: &dyn fs::Fs) -> Result<ReviewedMetadata> {
        let cache = self.cache.get().context("No manager cache")?.path();
        let mut entries = filesystem.read_dir(&cache.join("java")).await?;
        let mut selected = None;
        let mut count = 0;
        while let Some(entry) = entries.next().await {
            count += 1;
            ensure!(count <= 64, "Manager cache inventory exceeds its bound");
            let entry = entry?;
            if entry
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.starts_with("java_metadata_ga") && name.ends_with(".msgpack.z")
                })
            {
                ensure!(
                    selected.is_none(),
                    "Manager metadata build identity is ambiguous"
                );
                selected = Some(entry);
            }
        }
        let (identity, bytes) = bounded_file(
            filesystem,
            cache,
            &selected.context("Manager did not publish complete metadata")?,
            MAXIMUM_METADATA,
        )
        .await?;
        let artifact = parse_metadata(&bytes, &self.request)?;
        ensure!(
            identity
                .modified
                .timestamp_for_user()
                .elapsed()
                .is_ok_and(|age| age < Duration::from_secs(24 * 60 * 60)),
            "Manager metadata is stale"
        );
        Ok(ReviewedMetadata {
            identity,
            bytes,
            artifact,
        })
    }
}

fn metadata_text<'a>(value: &'a rmpv::ValueRef<'_>) -> Option<&'a str> {
    match value {
        rmpv::ValueRef::String(value) => value.as_str(),
        _ => None,
    }
}

fn parse_metadata(bytes: &[u8], request: &MiseJdkRequest) -> Result<MiseArtifact> {
    request.validate()?;
    let mut decoded = Vec::new();
    flate2::read::ZlibDecoder::new(bytes)
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut decoded)
        .map_err(|_| anyhow::anyhow!("Manager metadata compression is invalid"))?;
    ensure!(
        decoded.len() <= 16 * 1024 * 1024,
        "Manager metadata exceeds its decoded bound"
    );
    let mut cursor = decoded.as_slice();
    let value = rmpv::decode::read_value_ref_with_max_depth(&mut cursor, 12)
        .map_err(|_| anyhow::anyhow!("Manager metadata encoding is invalid"))?;
    ensure!(cursor.is_empty(), "Manager metadata has trailing content");
    let rmpv::ValueRef::Map(map) = &value else {
        anyhow::bail!("Manager metadata must be a map")
    };
    ensure!(
        map.len() <= 50_000,
        "Manager metadata catalog exceeds its bound"
    );
    let matches = map
        .iter()
        .filter(|(key, _)| metadata_text(key) == Some(request.version.as_str()))
        .collect::<Vec<_>>();
    ensure!(
        matches.len() == 1,
        "The exact selected JDK is missing or ambiguous; no fuzzy version is permitted"
    );
    let rmpv::ValueRef::Map(entry) = &matches.first().context("No selected JDK")?.1 else {
        anyhow::bail!("Selected JDK metadata is invalid")
    };
    let field = |name: &str| -> Result<&str> {
        let values = entry
            .iter()
            .filter(|(key, _)| metadata_text(key) == Some(name))
            .collect::<Vec<_>>();
        ensure!(
            values.len() == 1,
            "Selected JDK metadata field is missing or ambiguous"
        );
        let value = metadata_text(&values.first().context("Missing JDK field")?.1)
            .context("Selected JDK field is not text")?;
        ensure!(value.len() <= 2048, "Selected JDK field exceeds its bound");
        Ok(value)
    };
    let (vendor, version) = request.version.split_once('-').context("Missing vendor")?;
    ensure!(
        field("vendor")? == vendor
            && field("version")? == version
            && field("image_type")? == "jdk"
            && field("jvm_impl")? == "hotspot",
        "Selected metadata does not identify the reviewed GA JDK"
    );
    let java_version = field("java_version")?;
    ensure!(
        java_version.len() <= 100
            && java_version
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric()
                    || matches!(byte, b'.' | b'+' | b'-' | b'_')),
        "JDK runtime version is invalid"
    );
    let expected_type = if cfg!(windows) { "zip" } else { "tar.gz" };
    ensure!(
        matches!(field("file_type")?, "tar.gz" | "tar.xz") && expected_type != "zip",
        "Unsupported JDK archive type"
    );
    let source = field("url")?;
    let url = url::Url::parse(source).map_err(|_| anyhow::anyhow!("JDK source URL is invalid"))?;
    ensure!(
        url.scheme() == "https"
            && url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none()
            && url.port().is_none(),
        "JDK source requires a reviewed public HTTPS URL without credentials/options"
    );
    let allowed = match vendor {
        "corretto" => {
            url.host_str() == Some("corretto.aws")
                && url.path().starts_with("/downloads/resources/")
        }
        "zulu" => url.host_str() == Some("cdn.azul.com") && url.path().starts_with("/zulu/bin/"),
        "openjdk" => {
            url.host_str() == Some("download.java.net") && url.path().starts_with("/java/")
        }
        "microsoft" => url.host_str() == Some("aka.ms") && url.path().starts_with("/download-jdk/"),
        "liberica" => {
            url.host_str() == Some("download.bell-sw.com") && url.path().starts_with("/java/")
        }
        "temurin" => {
            url.host_str() == Some("github.com")
                && url.path().starts_with("/adoptium/temurin")
                && url.path().contains("/releases/download/")
        }
        _ => false,
    };
    ensure!(
        allowed,
        "JDK delivery source has no approved vendor adapter"
    );
    let checksum = field("checksum")?;
    let checksum = checksum.strip_prefix("sha256:").unwrap_or(checksum);
    ensure!(
        checksum.len() == 64 && checksum.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "JDK source requires complete SHA-256 metadata"
    );
    Ok(MiseArtifact {
        source: source.to_owned(),
        checksum: checksum.to_ascii_lowercase(),
        java_version: java_version.to_owned(),
    })
}

pub fn prepare_mise_inspection(
    project: &Entity<Project>,
    directory: ProjectPath,
    executable: PathBuf,
    request: MiseJdkRequest,
    generation: u64,
    observed_at: u64,
    cx: &App,
) -> Task<Result<DoctorRepairReview<MiseAction>>> {
    let binding = project_repair_binding(project, directory.clone(), generation, cx);
    let local = project
        .read(cx)
        .doctor_observation(directory.worktree_id, cx)
        .host
        == project::DoctorExecutionHost::Local;
    let filesystem = project.read(cx).fs().clone();
    let roots = project
        .read(cx)
        .worktrees(cx)
        .map(|tree| tree.read(cx).abs_path().to_path_buf())
        .collect::<Vec<_>>();
    cx.background_spawn(async move {
        let binding = binding?;
        request.validate()?;
        ensure!(local, "Remote manager identity/cancellation is unsupported");
        ensure!(!std::env::vars_os().any(|(name, _)| name.to_str().is_some_and(|name| name.starts_with("MISE_") || matches!(name, "XDG_DATA_HOME" | "XDG_CACHE_HOME" | "XDG_CONFIG_HOME" | "XDG_STATE_HOME"))), "Custom manager roots/settings require a separately supported adapter");
        let snapshot = MiseSnapshot::inspect(filesystem.as_ref(), paths::home_dir().to_path_buf(), executable, roots, supported_mise_digest()?).await?;
        let action = MiseAction { snapshot, request, stage: MiseStage::Inspect, cache: OnceLock::new(), metadata: None };
        DoctorRepairReview::new(action, binding, DoctorCheck::new("jvm.mise.inspect", "Selected mise manager", CheckOutcome::Untested, "The installed manager has not been invoked", observed_at), DiagnosticText::new("Run only the pinned existing mise version query with a clean environment and private disposable cache/config/state. Offline; no project/config/hooks, credentials, default changes, cache pruning or self-update. Separate approvals are required for metadata resolution and JDK installation."))
    })
}

pub struct MiseOperationResult {
    pub review: DoctorRepairReview<MiseAction>,
    pub proposal: Option<DoctorRepairReview<MiseAction>>,
    pub cleanup_error: Option<DiagnosticText>,
}
impl DoctorRepairReview<MiseAction> {
    pub fn cancel_mise_proposal(&mut self) -> Result<()> {
        self.cancel_before_execution()?;
        if let Some(cache) = self.action().cache.get() {
            cache.close()?;
        }
        Ok(())
    }
}

impl MiseOperationResult {
    pub fn cancel_proposal(&mut self) -> Result<()> {
        if let Some(proposal) = &mut self.proposal {
            proposal.cancel_mise_proposal()?;
        }
        self.proposal = None;
        if let Some(cache) = self.review.action().cache.get() {
            cache.close()?;
        }
        Ok(())
    }
}

pub fn launch_mise_operation(
    mut review: DoctorRepairReview<MiseAction>,
    project: &Entity<Project>,
    directory: ProjectPath,
    generation: u64,
    report: DoctorReport,
    product: DoctorProduct,
    history: DoctorHistoryStore,
    now: u64,
    cancellation: async_channel::Sender<()>,
    cancelled: async_channel::Receiver<()>,
    cx: &mut App,
) -> Task<Result<MiseOperationResult>> {
    let filesystem = project.read(cx).fs().clone();
    let project = project.downgrade();
    let executor = cx.background_executor().clone();
    cx.spawn(async move |cx| {
        let started = executor.now();
        let binding = cx.update(|cx| project_repair_binding(&project.upgrade().context("Doctor project closed")?, directory.clone(), generation, cx))?;
        review.record_before_execution(&binding, report, product, &history, now, &cancelled).await?;
        let preparation = async {
            review.action().revalidate(filesystem.as_ref()).await?;
            if review.action().stage == MiseStage::Install { ensure!(filesystem.metadata(&review.action().destination()).await?.is_none(), "JDK destination changed after review"); }
            if review.action().cache.get().is_none() {
                let home = review.action().snapshot.home.clone();
                let cache = executor.spawn(async move { tempfile::Builder::new().prefix(".made-doctor-mise-").tempdir_in(home) }).await?;
                review.action().cache.set(Arc::new(MiseCache::new(cache))).map_err(|_| anyhow::anyhow!("Manager cache was already admitted"))?;
            }
            let manager_guard = if review.action().stage == MiseStage::Install {
                let directory = review.action().snapshot.data.clone();
                Some(executor.spawn(async move { acquire_manager_guard(&directory) }).await?)
            } else { None };
            review.action().revalidate(filesystem.as_ref()).await?;
            let arguments = review.action().command_arguments();
            let environment = review.action().environment()?;
            cx.update(|cx| {
                let project = project.upgrade().context("Doctor project closed")?;
                let binding = project_repair_binding(&project, directory.clone(), generation, cx)?;
                review.revalidate(&binding)?;
                ensure!(!cancelled.is_closed() && cancelled.is_empty(), "Cancelled before manager dispatch");
                project.read(cx).exec_user_toolchain_command_for_path(review.action().executable(), &arguments.iter().map(String::as_str).collect::<Vec<_>>(), &directory, &review.action().snapshot.home, &environment, cx)
            }).map(|command| (command, manager_guard))
        }.await;
        let revoked = Arc::new(AtomicBool::new(false));
        let mut monitor = None;
        let result = match preparation {
            Ok((command, manager_guard)) => {
                monitor = Some(cx.spawn({
                    let project = project.clone(); let directory = directory.clone(); let binding = review.binding().clone(); let snapshot = review.action().snapshot.clone(); let metadata = review.action().metadata.clone(); let cache = review.action().cache.get().map(|cache| cache.path().to_path_buf()); let executor = executor.clone(); let filesystem = filesystem.clone(); let cancellation = cancellation.clone(); let revoked = revoked.clone();
                    async move |cx| {
                        loop {
                            let valid = async {
                                snapshot.revalidate(filesystem.as_ref()).await?;
                                if let Some(metadata) = &metadata {
                                    let (identity, bytes) = bounded_file(filesystem.as_ref(), cache.as_ref().context("Manager cache unavailable")?, &metadata.identity.path, MAXIMUM_METADATA).await?;
                                    ensure!(identity == metadata.identity && bytes == metadata.bytes, "Manager source changed");
                                }
                                Ok::<_, anyhow::Error>(())
                            }.await.is_ok();
                            let current = cx.update(|cx| project_repair_binding(&project.upgrade().context("Doctor project closed")?, directory.clone(), generation, cx));
                            if !valid || !matches!(current, Ok(current) if current == binding) { revoked.store(true, Ordering::Release); cancellation.close(); return; }
                            executor.timer(Duration::from_millis(100)).await;
                        }
                    }
                }));
                let timeout = if review.action().stage == MiseStage::Inspect { Duration::from_secs(10) } else { Duration::from_secs(300) };
                let operation_executor = executor.clone();
                let operation_cancelled = cancelled.clone();
                executor.spawn(async move { let _manager_guard = manager_guard; run_toolchain_command(command, operation_executor, operation_cancelled, timeout).await }).await
            }
            Err(error) => Err(error),
        };
        let mut proposal = None;
        match result {
            Err(error) => { review.operation_finished(DoctorRepairOperationOutcome::Failed, DiagnosticText::new(format!("Existing manager dispatch failed; reached effects may remain: {error}")))?; review.recheck_unavailable(DiagnosticText::new("No completed manager observation"))?; }
            Ok(result) => {
                let completed = result.end == ProbeEnd::Completed;
                let operation = if completed { DoctorRepairOperationOutcome::Completed } else if result.end == ProbeEnd::Cancelled && !revoked.load(Ordering::Acquire) { DoctorRepairOperationOutcome::Cancelled } else { DoctorRepairOperationOutcome::Failed };
                review.operation_finished(operation, DiagnosticText::new(format!("Manager observed {:?}; no rollback or automatic replay", result.end)))?;
                if review.phase() == DoctorRepairPhase::Rechecking {
                    let verification = async {
                        ensure!(completed && !revoked.load(Ordering::Acquire), "Manager did not complete with retained authority");
                        review.action().revalidate(filesystem.as_ref()).await?;
                        let current = cx.update(|cx| project_repair_binding(&project.upgrade().context("Doctor project closed")?, directory.clone(), generation, cx))?;
                        ensure!(&current == review.binding(), "Selected inputs changed before verification");
                        let observed_at = now.saturating_add(executor.now().duration_since(started).as_secs());
                        let detail = match review.action().stage {
                            MiseStage::Inspect => {
                                ensure!(result.detail.as_str().split_whitespace().next() == Some(MISE_VERSION), "Supported manager version was not observed");
                                let action = MiseAction { snapshot: review.action().snapshot.clone(), request: review.action().request.clone(), stage: MiseStage::Resolve, cache: review.action().cache.clone(), metadata: None };
                                let effects = DiagnosticText::new(format!("Resolve only exact {} using existing mise core:java install --dry-run and metadata source {}. Network metadata request and private disposable manager cache writes; no JDK download/execution or installation, config/hook/default changes or credential inheritance. Five-minute/10 MiB bounds and owned cancellation. Installation requires another explicit source/destination review.", action.request.version, metadata_source()?));
                                let before = DoctorCheck::new("jvm.mise.resolve", "Exact Java source resolution", CheckOutcome::Untested, "The requested JDK source is not yet observed", observed_at);
                                proposal = Some(DoctorRepairReview::new(action, current.clone(), before, effects)?);
                                "Observed the supported manager version; metadata resolution and installation remain separately unapproved".to_owned()
                            }
                            MiseStage::Resolve => {
                                let metadata = review.action().resolved_metadata(filesystem.as_ref()).await?;
                                let effects = DiagnosticText::new(format!("Install exact {} using existing mise built-in core:java, executable {}, source {}, SHA-256 {}, metadata authority {}, user destination {}. Download, checksum verification, extraction, disposable shim/cache and user download writes and downloaded Java runtime execution are performed by the existing manager. An empty .made-doctor-install.lock is created/held in that user manager root to reject concurrent Doctor installations; it is not truncated or removed. No inherited credentials, plugins/hooks/config/default/JAVA_HOME/project pin changes, self-update or cache pruning. Five minutes/10 MiB observed output. Cancellation may leave partial manager files; only the private disposable review cache is cleaned. Fresh installed release plus Java/javac observations are required; no project/application fix is claimed.", review.action().request.version, review.action().executable().display(), metadata.artifact.source, metadata.artifact.checksum, metadata_source()?, review.action().destination().display()));
                                let action = MiseAction { snapshot: review.action().snapshot.clone(), request: review.action().request.clone(), stage: MiseStage::Install, cache: review.action().cache.clone(), metadata: Some(metadata) };
                                let before = DoctorCheck::new("jvm.mise.install", "Requested JDK installation", CheckOutcome::Blocked, "Exact requested user JDK is not installed", observed_at);
                                proposal = Some(DoctorRepairReview::new(action, current.clone(), before, effects)?);
                                "Observed the exact reviewed JDK source; installation remains separately unapproved".to_owned()
                            }
                            MiseStage::Install => {
                                let artifact = review.action().artifact().context("Reviewed source missing")?;
                                let (_, bytes) = bounded_file(filesystem.as_ref(), &review.action().snapshot.data, &review.action().destination().join("release"), 32 * 1024).await?;
                                let release = std::str::from_utf8(&bytes).map_err(|_| anyhow::anyhow!("Installed JDK release encoding invalid"))?;
                                let versions = release.lines().filter_map(|line| line.strip_prefix("JAVA_VERSION=")).collect::<Vec<_>>();
                                ensure!(versions.len() == 1 && versions.first().is_some_and(|version| version.trim_matches('"') == artifact.java_version.split('+').next().unwrap_or(&artifact.java_version)), "Installed JDK release does not match the reviewed runtime");
                                for (name, kind, arguments) in [("java", DependencyProbe::Java, &["-version"][..]), ("javac", DependencyProbe::JavaCompiler, &["-version"][..])] {
                                    let executable = ProbeExecutable::inspect(kind, review.action().destination().join("bin").join(name), &review.action().snapshot.roots, filesystem.as_ref()).await?;
                                    ensure!(executable.path().starts_with(&review.action().snapshot.data), "Installed compiler/runtime escapes the manager scope");
                                    let command = cx.update(|cx| project.upgrade().context("Doctor project closed")?.read(cx).exec_read_only_probe_for_path(executable.path(), arguments, &directory, cx))?;
                                    let observed = executor.spawn(run_toolchain_command(command, executor.clone(), cancelled.clone(), Duration::from_secs(10))).await?;
                                    let expected = artifact.java_version.split('+').next().unwrap_or(&artifact.java_version);
                                    let matches = if name == "javac" { observed.detail.as_str() == format!("javac {expected}") } else { observed.detail.as_str().split('"').nth(1) == Some(expected) };
                                    ensure!(observed.end == ProbeEnd::Completed && matches, "Installed runtime/compiler version was not observed");
                                    executable.revalidate(&review.action().snapshot.roots, filesystem.as_ref()).await?;
                                }
                                format!("Observed exact installed JDK release, Java runtime and javac compiler for {}; defaults/project configuration and application behavior were not changed or verified", review.action().request.version)
                            }
                        };
                        review.action().revalidate(filesystem.as_ref()).await?;
                        let current = cx.update(|cx| project_repair_binding(&project.upgrade().context("Doctor project closed")?, directory.clone(), generation, cx))?;
                        ensure!(&current == review.binding() && !revoked.load(Ordering::Acquire), "Authority changed during verification");
                        let check = DoctorCheck::new(review.before().id.as_str(), review.before().title.as_str(), CheckOutcome::Ready, &detail, now.saturating_add(executor.now().duration_since(started).as_secs()));
                        Ok::<_, anyhow::Error>((current, check))
                    }.await;
                    match verification {
                        Ok((current, check)) => review.observe_recheck(&current, check)?,
                        Err(error) => { proposal = None; if (cancelled.is_closed() || !cancelled.is_empty()) && !revoked.load(Ordering::Acquire) { review.cancel_recheck(DiagnosticText::new("Verification cancelled; reached files may remain"))?; } else { review.recheck_unavailable(DiagnosticText::new(format!("JDK verification incomplete: {error}")))?; } }
                    }
                }
            }
        }
        drop(monitor);
        let cleanup_error = if proposal.is_none() {
            match review.action().cache.get().map(|cache| cache.close()).transpose() {
                Ok(_) => None,
                Err(error) => {
                    let detail = DiagnosticText::new(format!("Disposable manager cache cleanup failed; inspect the retained owned cache: {error}"));
                    review.note_cleanup(detail.clone())?;
                    Some(detail)
                }
            }
        } else { None };
        review.persist_outcome(product, &history, None, now.saturating_add(executor.now().duration_since(started).as_secs())).await?;
        Ok(MiseOperationResult { review, proposal, cleanup_error })
    })
}

const SDKMAN_SOURCES: &[(&str, &str)] = &[
    (
        "src/sdkman-availability.sh",
        "3ef0bc5ac2aa01c6fa5a8169185cd6404fcf8667697d302043928c42762f51d6",
    ),
    (
        "src/sdkman-cache.sh",
        "15241839ca9b7e12b29ee59bcacef5b483d7c210b95246a6e38fa05e3bda2944",
    ),
    (
        "src/sdkman-config.sh",
        "042d77a04e4502b3915fda27af09d24c6a90396187dfad812aaeeb86258c4351",
    ),
    (
        "src/sdkman-current.sh",
        "6e22fdf07fdbb1928a5008c0225d33f44c6607586b90760b7da3061fa656d82a",
    ),
    (
        "src/sdkman-default.sh",
        "1e814d7a59013cceec9c4a5ec608b0e2b51233efb17876f0750e9c98d366e5bf",
    ),
    (
        "src/sdkman-env-helpers.sh",
        "906b598aed548e8130494b59a0c8b42a99ae47c53031796ea8739af58c345b05",
    ),
    (
        "src/sdkman-env.sh",
        "159fd4e9672665a4d8d9dbbc002a16839bb9dedbf5319310991546e3ebee3efb",
    ),
    (
        "src/sdkman-flush.sh",
        "5ef8833d6e9af28636cc21916465cf56bf650e37c17793ed6a164f158a6155ec",
    ),
    (
        "src/sdkman-help.sh",
        "247617360ddba9283ba03d2d5bb8a3fd4d1b79d0d2040264eb4ea060fef17ec1",
    ),
    (
        "src/sdkman-home.sh",
        "129e36d416fd24779b0b5f8ba08093f6d86ab8125a428d6d8a7ada1ab3344aaf",
    ),
    (
        "bin/sdkman-init.sh",
        "f35578c4ac307f6292775ebc7552aee4a5c18fd9987192b22f02023422193574",
    ),
    (
        "src/sdkman-install.sh",
        "1c9c40c554a1bd74d803c119575c9770398b6ca5f439e334731bc23344b4eee6",
    ),
    (
        "src/sdkman-list.sh",
        "4c4043c9e44e522652ab209110454b5502a29aa840b0c6a567115999b671428a",
    ),
    (
        "src/sdkman-main.sh",
        "7faec4b29440467428ef0d7900dfb23959d66b9e1993f47baaa1c68cbc96e8e9",
    ),
    (
        "src/sdkman-path-helpers.sh",
        "214ee9f94fe1500c63c6a54e78e969cf4c7b800354a06dd0d31d01a4d5e26585",
    ),
    (
        "src/sdkman-selfupdate.sh",
        "d171613a9910843c7bc90f1a8a61cc9170946335a04a103a6a9eb16f5488f208",
    ),
    (
        "src/sdkman-uninstall.sh",
        "81cc5ce816c7e237c6ec7ef38ccf8d15633d9f95067ad400a413554bf77c970d",
    ),
    (
        "src/sdkman-update.sh",
        "3e1186321f8b3ecc35b478e89df497f7db25e39452c6ae480bb9d46c8bedd89a",
    ),
    (
        "src/sdkman-upgrade.sh",
        "4e77791377fcbf289cc90b6961a1eb017cc3e4025a9df370cb22082687aa14ed",
    ),
    (
        "src/sdkman-use.sh",
        "5fbf7388780f44477dfb3555d717d8c2a5f158042ee83bd0f038b9ddb8a781bc",
    ),
    (
        "src/sdkman-utils.sh",
        "e1206db90e8b96b461098e840390244d8af66a4fd13960a962b447f49669bb49",
    ),
    (
        "src/sdkman-version.sh",
        "6f9a701dc48fb649ab22f2987af1db81d2aa91ec0d0b0a23cf036773970845fb",
    ),
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SdkmanJdkRequest {
    pub version: String,
}

impl SdkmanJdkRequest {
    fn validate(&self) -> Result<()> {
        let (release, vendor) = self
            .version
            .rsplit_once('-')
            .context("Select an exact SDKMAN Java ID")?;
        ensure!(
            matches!(vendor, "tem" | "amzn" | "zulu" | "librca" | "ms"),
            "Unsupported SDKMAN Java vendor"
        );
        ensure!(
            self.version.len() <= 50
                && release.split('.').count() == 3
                && release.split('.').all(|part| !part.is_empty()
                    && part.bytes().all(|byte| byte.is_ascii_digit())
                    && part.parse::<u32>().is_ok()),
            "Only exact released SDKMAN JDK IDs are supported; no latest, early-access, local paths or shell options"
        );
        Ok(())
    }
    pub fn runtime_version(&self) -> Result<String> {
        self.validate()?;
        let (release, _) = self
            .version
            .rsplit_once('-')
            .context("Missing Java vendor")?;
        let parts = release.split('.').collect::<Vec<_>>();
        if parts.first() == Some(&"8") {
            Ok(format!(
                "1.8.0_{}",
                parts.get(2).context("Missing Java update")?
            ))
        } else {
            Ok(release.to_owned())
        }
    }
}

fn validate_sdkman_configuration(bytes: &[u8]) -> Result<()> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| anyhow::anyhow!("SDKMAN configuration encoding is invalid"))?;
    let mut settings = std::collections::BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, value) = line
            .split_once('=')
            .context("SDKMAN config must contain safe literal assignments only")?;
        ensure!(
            name.starts_with("sdkman_")
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte == b'_')
                && (matches!(value, "true" | "false")
                    || (!value.is_empty()
                        && value.bytes().all(|byte| byte.is_ascii_digit())
                        && value.parse::<u32>().is_ok()))
                && settings.insert(name, value).is_none(),
            "SDKMAN shell expressions, custom values and duplicate settings are unsupported"
        );
    }
    for (name, value) in [
        ("sdkman_auto_answer", "false"),
        ("sdkman_auto_complete", "false"),
        ("sdkman_auto_env", "false"),
        ("sdkman_auto_update", "false"),
        ("sdkman_checksum_enable", "true"),
        ("sdkman_insecure_ssl", "false"),
        ("sdkman_debug_mode", "false"),
        ("sdkman_native_enable", "false"),
        ("sdkman_healthcheck_enable", "false"),
        ("sdkman_curl_retry", "0"),
        ("sdkman_curl_continue", "false"),
    ] {
        ensure!(
            settings.get(name).copied() == Some(value),
            "SDKMAN configuration requires a separately reviewed safe setting for {name}; Doctor does not edit it implicitly"
        );
    }
    ensure!(
        settings.keys().all(|name| matches!(
            *name,
            "sdkman_auto_answer"
                | "sdkman_auto_complete"
                | "sdkman_auto_env"
                | "sdkman_auto_update"
                | "sdkman_checksum_enable"
                | "sdkman_insecure_ssl"
                | "sdkman_debug_mode"
                | "sdkman_native_enable"
                | "sdkman_healthcheck_enable"
                | "sdkman_curl_retry"
                | "sdkman_curl_continue"
                | "sdkman_curl_connect_timeout"
                | "sdkman_curl_max_time"
                | "sdkman_curl_retry_max_time"
                | "sdkman_colour_enable"
                | "sdkman_selfupdate_feature"
                | "sdkman_rosetta2_compatible"
        )),
        "Unknown SDKMAN settings have unverified effects"
    );
    Ok(())
}

#[derive(Clone, PartialEq, Eq)]
struct SdkmanDefault {
    target: PathBuf,
    identity: FileStamp,
}

#[derive(Clone)]
struct SdkmanSnapshot {
    home: PathBuf,
    root: PathBuf,
    root_inode: u64,
    shell: FileStamp,
    sources: Vec<(FileStamp, Vec<u8>)>,
    configuration: (FileStamp, Vec<u8>),
    platform: (FileStamp, Vec<u8>),
    candidates: (FileStamp, Vec<u8>),
    version: (FileStamp, Vec<u8>),
    current: Option<SdkmanDefault>,
    project_roots: Vec<PathBuf>,
}

impl SdkmanSnapshot {
    async fn current(filesystem: &dyn fs::Fs, root: &Path) -> Result<Option<SdkmanDefault>> {
        let path = root.join("candidates/java/current");
        let Some(metadata) = filesystem.metadata(&path).await? else {
            return Ok(None);
        };
        ensure!(
            metadata.is_symlink && metadata.is_dir,
            "SDKMAN current selection is not a valid installed default link"
        );
        let target = filesystem.canonicalize(&path).await?;
        let identity = stamp(filesystem, &target.join("release")).await?;
        Ok(Some(SdkmanDefault { target, identity }))
    }

    async fn inspect(
        filesystem: &dyn fs::Fs,
        home: PathBuf,
        project_roots: Vec<PathBuf>,
        source_pins: &[(&str, &str)],
    ) -> Result<Self> {
        ensure!(
            matches!(std::env::consts::OS, "macos" | "linux"),
            "Native SDKMAN adapter is unsupported on this platform"
        );
        let home = filesystem.canonicalize(&home).await?;
        let root = filesystem.canonicalize(&home.join(".sdkman")).await?;
        ensure!(
            root.to_str().is_some_and(|value| value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric()
                    || matches!(byte, b'/' | b'.' | b'-' | b'_' | b' '))),
            "SDKMAN path cannot safely participate in the reviewed shell PATH"
        );
        confined_location(filesystem, &home, &root, &project_roots).await?;
        let root_inode = filesystem
            .metadata(&root)
            .await?
            .context("SDKMAN root unavailable")?
            .inode;
        let version = bounded_file(filesystem, &root, &root.join("var/version"), 128).await?;
        ensure!(
            std::str::from_utf8(&version.1).is_ok_and(|value| value.trim() == "5.23.1"),
            "Installed SDKMAN version is unsupported; Doctor will not source, bootstrap or upgrade it"
        );
        let shell = stamp(filesystem, Path::new("/bin/bash")).await?;
        ensure!(
            filesystem
                .metadata(&shell.path)
                .await?
                .is_some_and(|metadata| metadata.is_executable),
            "Existing system Bash is unavailable"
        );
        let configuration =
            bounded_file(filesystem, &root, &root.join("etc/config"), 16 * 1024).await?;
        validate_sdkman_configuration(&configuration.1)?;
        let platform = bounded_file(filesystem, &root, &root.join("var/platform"), 64).await?;
        let expected = match (std::env::consts::OS, std::env::consts::ARCH) {
            ("macos", "aarch64") => "darwinarm64",
            ("macos", "x86_64") => "darwinx64",
            ("linux", "aarch64") => "linuxarm64",
            ("linux", "x86_64") => "linuxx64",
            _ => anyhow::bail!("SDKMAN platform is unsupported"),
        };
        ensure!(
            std::str::from_utf8(&platform.1).is_ok_and(|value| value.trim() == expected),
            "SDKMAN platform does not match the selected native host"
        );
        let candidates =
            bounded_file(filesystem, &root, &root.join("var/candidates"), 4096).await?;
        let candidates_text = std::str::from_utf8(&candidates.1)
            .map_err(|_| anyhow::anyhow!("SDKMAN candidate cache is invalid"))?
            .trim();
        ensure!(
            candidates_text
                .split(',')
                .any(|candidate| candidate == "java")
                && candidates_text
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || matches!(byte, b',' | b'-')),
            "SDKMAN candidate cache is missing Java or has unsupported entries"
        );
        let mut sources = Vec::new();
        for (name, digest) in source_pins {
            let source = bounded_file(filesystem, &root, &root.join(name), 64 * 1024).await?;
            ensure!(
                format!("{:x}", Sha256::digest(&source.1)) == *digest,
                "SDKMAN source does not match the inspected official client"
            );
            sources.push(source);
        }
        let snapshot = Self {
            home,
            root,
            root_inode,
            shell,
            sources,
            configuration,
            platform,
            candidates,
            version,
            current: None,
            project_roots,
        };
        snapshot.revalidate(filesystem).await?;
        let current = Self::current(filesystem, &snapshot.root).await?;
        Ok(Self {
            current,
            ..snapshot
        })
    }

    async fn revalidate(&self, filesystem: &dyn fs::Fs) -> Result<()> {
        ensure!(
            filesystem.canonicalize(&self.home.join(".sdkman")).await? == self.root
                && filesystem
                    .metadata(&self.root)
                    .await?
                    .is_some_and(|metadata| metadata.is_dir && metadata.inode == self.root_inode)
                && stamp(filesystem, &self.shell.path).await? == self.shell,
            "SDKMAN manager/root/shell changed after review"
        );
        for source in self.sources.iter().chain([
            &self.configuration,
            &self.platform,
            &self.candidates,
            &self.version,
        ]) {
            let (identity, bytes) =
                bounded_file(filesystem, &self.root, &source.0.path, 64 * 1024).await?;
            ensure!(
                identity == source.0 && bytes == source.1,
                "SDKMAN source/configuration/platform changed after review"
            );
        }
        let mut entries = filesystem.read_dir(&self.root.join("src")).await?;
        let mut count = 0;
        while let Some(entry) = entries.next().await {
            count += 1;
            ensure!(count <= 64, "SDKMAN sources exceed their bound");
            let entry = entry?;
            ensure!(
                !filesystem
                    .metadata(&entry)
                    .await?
                    .is_some_and(|metadata| metadata.is_dir),
                "Nested SDKMAN script directories have unverified sourced effects"
            );
            if entry
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("sdkman-") && name.ends_with(".sh"))
            {
                ensure!(
                    self.sources.iter().any(|source| source.0.path == entry),
                    "Extra SDKMAN scripts have unverified effects"
                );
            }
        }
        if filesystem.metadata(&self.root.join("ext")).await?.is_some() {
            let mut entries = filesystem.read_dir(&self.root.join("ext")).await?;
            ensure!(
                entries.next().await.transpose()?.is_none(),
                "Custom SDKMAN extensions are unsupported"
            );
        }
        for name in [
            "candidates",
            "candidates/java",
            "tmp",
            "var",
            "var/metadata",
        ] {
            confined_location(
                filesystem,
                &self.home,
                &self.root.join(name),
                &self.project_roots,
            )
            .await?;
        }
        Ok(())
    }
}

pub struct SdkmanAction {
    snapshot: SdkmanSnapshot,
    request: SdkmanJdkRequest,
    first_default_consent: OnceLock<crate::doctor::DoctorRepairBinding>,
}

impl SdkmanAction {
    pub fn executable(&self) -> &Path {
        &self.snapshot.shell.path
    }
    pub fn initialization_script(&self) -> PathBuf {
        self.snapshot.root.join("bin/sdkman-init.sh")
    }
    pub fn command_arguments(&self) -> Vec<String> {
        [
            "--noprofile",
            "--norc",
            "-c",
            SDKMAN_INSTALL_COMMAND,
            "made-doctor-sdkman",
            &self.request.version,
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }
    pub fn request(&self) -> &SdkmanJdkRequest {
        &self.request
    }
    pub fn destination(&self) -> PathBuf {
        self.snapshot
            .root
            .join("candidates/java")
            .join(&self.request.version)
    }
    pub fn requires_first_default(&self) -> bool {
        self.snapshot.current.is_none()
    }
    pub fn has_first_default_consent(&self) -> bool {
        self.first_default_consent.get().is_some()
    }
    pub fn sources(&self) -> Result<Vec<String>> {
        let platform = std::str::from_utf8(&self.snapshot.platform.1)
            .map_err(|_| anyhow::anyhow!("Invalid reviewed SDKMAN platform"))?
            .trim();
        Ok(vec![
            format!(
                "https://api.sdkman.io/2/candidates/validate/java/{}/{}",
                self.request.version, platform
            ),
            format!(
                "https://broker.sdkman.io/download/java/{}/{}",
                self.request.version, platform
            ),
            format!(
                "https://api.sdkman.io/2/hooks/post/java/{}/{}",
                self.request.version, platform
            ),
        ])
    }
    async fn revalidate(&self, filesystem: &dyn fs::Fs, running: bool) -> Result<()> {
        self.snapshot.revalidate(filesystem).await?;
        let current = SdkmanSnapshot::current(filesystem, &self.snapshot.root).await?;
        ensure!(
            current == self.snapshot.current
                || (running
                    && self.requires_first_default()
                    && current
                        .as_ref()
                        .is_some_and(|current| current.target == self.destination())),
            "SDKMAN default changed outside the reviewed first-default operation"
        );
        if !running {
            ensure!(
                filesystem.metadata(&self.destination()).await?.is_none(),
                "Existing/partial JDK requires separate review; no overwrite"
            );
            ensure!(
                filesystem
                    .metadata(
                        &self
                            .snapshot
                            .root
                            .join("var/metadata")
                            .join(format!("java-{}.headers", self.request.version))
                    )
                    .await?
                    .is_none(),
                "Existing SDKMAN download metadata requires separate review; no overwrite"
            );
            for name in [
                "out".to_owned(),
                format!("java-{}.bin", self.request.version),
                format!("java-{}.zip", self.request.version),
                format!("java-{}.headers.tmp", self.request.version),
                format!("hook_post_java_{}.sh", self.request.version),
            ] {
                ensure!(
                    filesystem
                        .metadata(&self.snapshot.root.join("tmp").join(name))
                        .await?
                        .is_none(),
                    "SDKMAN scratch content may belong to another/interrupted operation; no implicit deletion"
                );
            }
        }
        Ok(())
    }
}

impl DoctorRepairReview<SdkmanAction> {
    pub fn approve_sdkman_first_default(
        &self,
        current: &crate::doctor::DoctorRepairBinding,
    ) -> Result<()> {
        ensure!(
            self.phase() == DoctorRepairPhase::Proposed
                && self.binding() == current
                && self.action().requires_first_default(),
            "First-default consent requires a current separate review before install approval"
        );
        self.action()
            .first_default_consent
            .set(current.clone())
            .map_err(|_| anyhow::anyhow!("First-default change was already acknowledged"))
    }
}

pub fn prepare_sdkman_installation(
    project: &Entity<Project>,
    directory: ProjectPath,
    request: SdkmanJdkRequest,
    generation: u64,
    observed_at: u64,
    cx: &App,
) -> Task<Result<DoctorRepairReview<SdkmanAction>>> {
    let binding = project_repair_binding(project, directory.clone(), generation, cx);
    let local = project
        .read(cx)
        .doctor_observation(directory.worktree_id, cx)
        .host
        == project::DoctorExecutionHost::Local;
    let filesystem = project.read(cx).fs().clone();
    let roots = project
        .read(cx)
        .worktrees(cx)
        .map(|tree| tree.read(cx).abs_path().to_path_buf())
        .collect::<Vec<_>>();
    cx.background_spawn(async move {
        let binding = binding?; ensure!(local, "SDKMAN remote identity/cancellation is unsupported"); request.validate()?;
        ensure!(!std::env::vars_os().any(|(name,_)|name.to_str().is_some_and(|name|name.starts_with("SDKMAN_"))), "Custom SDKMAN host environment has unverified effects");
        let snapshot = SdkmanSnapshot::inspect(filesystem.as_ref(), paths::home_dir().to_path_buf(), roots, SDKMAN_SOURCES).await?;
        let action = SdkmanAction { snapshot, request, first_default_consent: OnceLock::new() };
        action.revalidate(filesystem.as_ref(), false).await?;
        let effects = DiagnosticText::new(format!("Invoke only installed SDKMAN 5.23.1 through system Bash without shell startup for exact {}. Sources {:?}, destination {}. Official downloaded post-install hooks execute with user permissions; this is not a sandbox. Private disposable HOME/curl configuration permits only HTTPS/HTTPS redirects and no inherited credentials. Manager download/cache/metadata/hook/installation writes and an empty nonblocking Doctor manager-lock file; no implicit SDK bootstrap/update/auto-env/pin/JAVA_HOME/system changes. Existing default is preserved; a missing default requires separate explicit first-default acknowledgment before normal install approval. Five minutes/10 MiB; cancellation may leave partial files; no replay or destructive rollback. Fresh default/release/Java/javac recheck required.",action.request.version,action.sources()?,action.destination().display()));
        DoctorRepairReview::new(action,binding,DoctorCheck::new("jvm.sdkman.install","Requested SDKMAN JDK",CheckOutcome::Blocked,"Exact requested JDK is not installed; no manager execution occurred",observed_at),effects)
    })
}

const SDKMAN_INSTALL_COMMAND: &str = r#"
source "$SDKMAN_DIR/bin/sdkman-init.sh" || exit $?
export PATH="/usr/bin:/bin:$SDKMAN_DIR/candidates/java/current/bin"
unset JAVA_HOME
sdk install java "$1" <<< 'n'
"#;

pub struct SdkmanOperationResult {
    pub review: DoctorRepairReview<SdkmanAction>,
    pub cleanup_error: Option<DiagnosticText>,
}

pub fn launch_sdkman_installation(
    mut review: DoctorRepairReview<SdkmanAction>,
    project: &Entity<Project>,
    directory: ProjectPath,
    generation: u64,
    report: DoctorReport,
    product: DoctorProduct,
    history: DoctorHistoryStore,
    now: u64,
    cancellation: async_channel::Sender<()>,
    cancelled: async_channel::Receiver<()>,
    cx: &mut App,
) -> Task<Result<SdkmanOperationResult>> {
    let filesystem = project.read(cx).fs().clone();
    let project = project.downgrade();
    let executor = cx.background_executor().clone();
    cx.spawn(async move |cx| {
        let started = executor.now();
        let binding = cx.update(|cx| project_repair_binding(&project.upgrade().context("Doctor project closed")?, directory.clone(), generation, cx))?;
        if review.action().requires_first_default() { ensure!(review.action().first_default_consent.get() == Some(&binding), "This first SDKMAN install would change the default; its separate exact acknowledgment is missing or stale"); }
        review.record_before_execution(&binding, report, product, &history, now, &cancelled).await?;
        let mut temporary_home = None;
        let preparation = async {
            review.action().revalidate(filesystem.as_ref(), false).await?;
            let root = review.action().snapshot.root.clone();
            let manager_guard = executor.spawn(async move { acquire_manager_guard(&root) }).await?;
            review.action().revalidate(filesystem.as_ref(), false).await?;
            let home = review.action().snapshot.home.clone();
            let private_home = executor.spawn(async move { tempfile::Builder::new().prefix(".made-doctor-sdkman-").tempdir_in(home) }).await?;
            filesystem.atomic_write(private_home.path().join(".curlrc"), "proto = \"=https\"\nproto-redir = \"=https\"\nretry = 0\nconnect-timeout = 10\nmax-time = 300\n".into(), ).await?;
            let environment = vec![
                ("HOME".into(), private_home.path().to_str().context("Unsupported private command path")?.to_owned()),
                ("CURL_HOME".into(), private_home.path().to_str().context("Unsupported private curl path")?.to_owned()),
                ("SDKMAN_DIR".into(), review.action().snapshot.root.to_str().context("Unsupported SDKMAN path")?.to_owned()),
                ("SDKMAN_CANDIDATES_API".into(), "https://api.sdkman.io/2".into()),
                ("SDKMAN_BROKER_API".into(), "https://broker.sdkman.io".into()),
                ("PATH".into(), "/usr/bin:/bin".into()),
            ];
            temporary_home = Some(MiseCache::new(private_home));
            let command = cx.update(|cx| {
                let project = project.upgrade().context("Doctor project closed")?;
                let binding = project_repair_binding(&project, directory.clone(), generation, cx)?;
                review.revalidate(&binding)?;
                ensure!(!cancelled.is_closed() && cancelled.is_empty(), "SDKMAN operation cancelled before dispatch");
                let arguments = review.action().command_arguments();
                project.read(cx).exec_user_toolchain_command_for_path(review.action().executable(), &arguments.iter().map(String::as_str).collect::<Vec<_>>(), &directory, &review.action().snapshot.home, &environment, cx)
            })?;
            Ok::<_, anyhow::Error>((command, manager_guard))
        }.await;
        let revoked = Arc::new(AtomicBool::new(false));
        let mut monitor = None;
        let result = match preparation {
            Ok((command, manager_guard)) => {
                monitor = Some(cx.spawn({
                    let project = project.clone(); let directory = directory.clone(); let binding = review.binding().clone(); let snapshot = review.action().snapshot.clone(); let filesystem = filesystem.clone(); let executor = executor.clone(); let cancellation = cancellation.clone(); let revoked = revoked.clone(); let destination = review.action().destination();
                    async move |cx| {
                        loop {
                            let valid = async {
                                snapshot.revalidate(filesystem.as_ref()).await?;
                                let current = SdkmanSnapshot::current(filesystem.as_ref(), &snapshot.root).await?;
                                ensure!(current == snapshot.current || (snapshot.current.is_none() && current.as_ref().is_some_and(|current| current.target == destination)), "Reviewed default changed outside this first install");
                                Ok::<_,anyhow::Error>(())
                            }.await.is_ok();
                            let current = cx.update(|cx| project_repair_binding(&project.upgrade().context("Doctor project closed")?, directory.clone(), generation, cx));
                            if !valid || !matches!(current, Ok(current) if current == binding) { revoked.store(true, Ordering::Release); cancellation.close(); return; }
                            executor.timer(Duration::from_millis(100)).await;
                        }
                    }
                }));
                let operation_executor = executor.clone(); let operation_cancelled = cancelled.clone();
                executor.spawn(async move { let _manager_guard = manager_guard; run_toolchain_command(command, operation_executor, operation_cancelled, Duration::from_secs(300)).await }).await
            }
            Err(error) => Err(error),
        };
        match result {
            Err(error) => { review.operation_finished(DoctorRepairOperationOutcome::Failed, DiagnosticText::new(format!("SDKMAN dispatch failed; reached effects may remain: {error}")))?; review.recheck_unavailable(DiagnosticText::new("No completed owner operation"))?; }
            Ok(result) => {
                let completed = result.end == ProbeEnd::Completed;
                let outcome = if completed { DoctorRepairOperationOutcome::Completed } else if result.end == ProbeEnd::Cancelled && !revoked.load(Ordering::Acquire) { DoctorRepairOperationOutcome::Cancelled } else { DoctorRepairOperationOutcome::Failed };
                review.operation_finished(outcome, DiagnosticText::new(format!("Existing SDKMAN owner observed {:?}; official hooks had user permissions. First-default acknowledgment {}. Partial downloads/hooks/installation effects may remain; no automatic replay or rollback.", result.end, if review.action().requires_first_default() { "explicitly recorded for this exact installation" } else { "not required; existing selection must remain unchanged" })))?;
                if review.phase() == DoctorRepairPhase::Rechecking {
                    let verification = async {
                        ensure!(completed && !revoked.load(Ordering::Acquire), "SDKMAN did not complete under retained authority");
                        review.action().revalidate(filesystem.as_ref(), true).await?;
                        let current = SdkmanSnapshot::current(filesystem.as_ref(), &review.action().snapshot.root).await?;
                        ensure!(current == review.action().snapshot.current || (review.action().requires_first_default() && current.as_ref().is_some_and(|current| current.target == review.action().destination())), "Requested default preservation/creation was not verified");
                        if review.action().requires_first_default() { ensure!(current.is_some(), "Approved first default was not observed"); }
                        let expected = review.action().request.runtime_version()?;
                        let (_, bytes) = bounded_file(filesystem.as_ref(), &review.action().snapshot.root, &review.action().destination().join("release"), 32 * 1024).await?;
                        let release = std::str::from_utf8(&bytes).map_err(|_|anyhow::anyhow!("Installed JDK release encoding is invalid"))?;
                        let versions = release.lines().filter_map(|line|line.strip_prefix("JAVA_VERSION=")).collect::<Vec<_>>();
                        ensure!(versions.len()==1 && versions.first().is_some_and(|version|version.trim_matches('"') == expected), "Installed SDKMAN JDK release does not match the reviewed version");
                        for (name, kind) in [("java", DependencyProbe::Java), ("javac", DependencyProbe::JavaCompiler)] {
                            let executable = ProbeExecutable::inspect(kind, review.action().destination().join("bin").join(name), &review.action().snapshot.project_roots, filesystem.as_ref()).await?;
                            ensure!(executable.path().starts_with(&review.action().snapshot.root), "Installed runtime/compiler escapes the selected manager");
                            let command = cx.update(|cx|project.upgrade().context("Doctor project closed")?.read(cx).exec_read_only_probe_for_path(executable.path(), &["-version"], &directory, cx))?;
                            let observed = executor.spawn(run_toolchain_command(command,executor.clone(),cancelled.clone(),Duration::from_secs(10))).await?;
                            let matches = if name=="javac" {observed.detail.as_str()==format!("javac {expected}")} else {observed.detail.as_str().split('"').nth(1)==Some(expected.as_str())};
                            ensure!(observed.end==ProbeEnd::Completed && matches,"SDKMAN installed runtime/compiler was not observed");
                            executable.revalidate(&review.action().snapshot.project_roots,filesystem.as_ref()).await?;
                        }
                        review.action().revalidate(filesystem.as_ref(),true).await?;
                        let binding = cx.update(|cx|project_repair_binding(&project.upgrade().context("Doctor project closed")?,directory.clone(),generation,cx))?;
                        ensure!(&binding==review.binding() && !revoked.load(Ordering::Acquire),"Authority changed during SDKMAN verification");
                        let observed_at = now.saturating_add(executor.now().duration_since(started).as_secs());
                        Ok::<_,anyhow::Error>((binding,DoctorCheck::new("jvm.sdkman.install","Requested SDKMAN JDK",CheckOutcome::Ready,"Observed exact installed release, native Java/javac and reviewed default state. Project build/application correctness remains unverified",observed_at)))
                    }.await;
                    match verification {
                        Ok((binding,check))=>review.observe_recheck(&binding,check)?,
                        Err(error)=>{if (cancelled.is_closed()||!cancelled.is_empty())&&!revoked.load(Ordering::Acquire) {review.cancel_recheck(DiagnosticText::new("SDKMAN recheck cancelled; reached files may remain"))?;}else{review.recheck_unavailable(DiagnosticText::new(format!("SDKMAN verification incomplete: {error}")))?;}}
                    }
                }
            }
        }
        drop(monitor);
        let cleanup_error = match temporary_home.as_ref().map(MiseCache::close).transpose() {
            Ok(_)=>None,
            Err(error)=>{let detail=DiagnosticText::new(format!("Private command HOME cleanup failed: {error}"));review.note_cleanup(detail.clone())?;Some(detail)}
        };
        review.persist_outcome(product,&history,None,now.saturating_add(executor.now().duration_since(started).as_secs())).await?;
        Ok(SdkmanOperationResult{review,cleanup_error})
    })
}

#[cfg(all(test, unix))]
pub(crate) mod tests {
    use super::*;
    use crate::doctor::{DoctorScope, ReceiptOutcome};
    use project::trusted_worktrees::{self, DbTrustedPaths, PathTrust, TrustedWorktrees};
    use std::io::Write as _;
    use util::{rel_path::RelPath, test::TempTree};

    const FIXTURE: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
static void file(const char *path, const char *text) { FILE *f=fopen(path,"w"); if(!f) exit(12); fputs(text,f); fclose(f); }
static void copy(const char *from,const char *to) { FILE *input=fopen(from,"rb"),*output=fopen(to,"wb"); if(!input||!output) exit(13); int byte; while((byte=fgetc(input))!=EOF) fputc(byte,output); fclose(input); fclose(output); }
int main(int count,char **arguments) {
 const char *name=strrchr(arguments[0],'/'); name=name?name+1:arguments[0];
 char bad[8192]; snprintf(bad,sizeof(bad),"%s.bad",arguments[0]); int incorrect=access(bad,F_OK)==0;
 if(!strcmp(name,"java")) { fprintf(stderr,"openjdk version \"%s\" (native offline fixture)\n",incorrect?"99.9.8":"99.9.9"); return 0; }
 if(!strcmp(name,"javac")) { puts(incorrect?"javac 99.9.8":"javac 99.9.9"); return 0; }
 const char *data=getenv("MISE_DATA_DIR"),*cache=getenv("MISE_CACHE_DIR");
 if(!data||!cache||!getenv("MISE_NETRC")||strcmp(getenv("MISE_NETRC"),"false")||getenv("JAVA_HOME")||getenv("GITHUB_TOKEN")||getenv("LD_PRELOAD")) return 30;
 if(count<5||strcmp(arguments[1],"--no-config")||strcmp(arguments[2],"--no-env")||strcmp(arguments[3],"--no-hooks")) return 31;
 if(!strcmp(arguments[4],"--version")) { puts("2026.10.1 native fixture"); return 0; }
 char from[8192],to[8192],base[8192],mode[128]="";
 snprintf(from,sizeof(from),"%s/fixture-mode",data); FILE *f=fopen(from,"r"); if(f) { fgets(mode,sizeof(mode),f); fclose(f); }
 int dry=0; for(int i=4;i<count;i++) if(!strcmp(arguments[i],"--dry-run")) dry=1;
 if(dry) {
   snprintf(to,sizeof(to),"%s/java",cache); mkdir(to,0700);
   snprintf(from,sizeof(from),"%s/metadata.fixture",data); snprintf(to,sizeof(to),"%s/java/java_metadata_ga-fixture.msgpack.z",cache); copy(from,to);
   puts("would install exact JDK"); return 0;
 }
 snprintf(to,sizeof(to),"%s/started",data); file(to,"owned manager started");
 if(!strcmp(mode,"wait")) { for(;;) pause(); }
 if(!strcmp(mode,"failure")) { puts("partial manager failure"); return 7; }
 if(!strcmp(mode,"empty")) return 0;
 snprintf(base,sizeof(base),"%s/installs",data); mkdir(base,0700);
 snprintf(base,sizeof(base),"%s/installs/java",data); mkdir(base,0700);
 snprintf(base,sizeof(base),"%s/installs/java/corretto-99.9.9",data); mkdir(base,0700);
 snprintf(to,sizeof(to),"%s/bin",base); mkdir(to,0700);
 snprintf(to,sizeof(to),"%s/release",base); file(to,!strcmp(mode,"wrong")?"JAVA_VERSION=\"99.9.8\"\n":"JAVA_VERSION=\"99.9.9\"\n");
 snprintf(to,sizeof(to),"%s/bin/java",base); copy(arguments[0],to); chmod(to,0700);
 snprintf(to,sizeof(to),"%s/bin/javac",base); copy(arguments[0],to); chmod(to,0700);
 if(!strcmp(mode,"badcompiler")||!strcmp(mode,"badruntime")) {snprintf(to,sizeof(to),"%s/bin/%s.bad",base,!strcmp(mode,"badcompiler")?"javac":"java"); file(to,"incorrect version fixture");}
 puts("fixture install completed"); return 0;
}
"#;

    fn request() -> MiseJdkRequest {
        MiseJdkRequest {
            version: "corretto-99.9.9".into(),
        }
    }

    fn metadata_value() -> rmpv::Value {
        let text = |value: &str| rmpv::Value::from(value);
        rmpv::Value::Map(vec![(
            text("corretto-99.9.9"),
            rmpv::Value::Map(vec![
                (text("vendor"), text("corretto")),
                (text("version"), text("99.9.9")),
                (text("image_type"), text("jdk")),
                (text("jvm_impl"), text("hotspot")),
                (text("java_version"), text("99.9.9")),
                (text("file_type"), text("tar.gz")),
                (
                    text("url"),
                    text("https://corretto.aws/downloads/resources/99.9.9/jdk.tar.gz"),
                ),
                (
                    text("checksum"),
                    text(&format!("sha256:{}", "a".repeat(64))),
                ),
            ]),
        )])
    }

    fn encoded(value: &rmpv::Value) -> Vec<u8> {
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, value).expect("Fixture MessagePack");
        let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
        encoder.write_all(&bytes).expect("Fixture compression");
        encoder.finish().expect("Finish fixture")
    }

    #[test]
    fn doctor_jvm_install_manager_admission_preserves_files_and_releases_without_replay() {
        let root = tempfile::tempdir().expect("Owned manager fixture");
        let root = root.path().canonicalize().expect("Manager root");
        let first = acquire_manager_guard(&root).expect("Admit first operation");
        assert!(
            acquire_manager_guard(&root).is_err(),
            "No queue or duplicate dispatch across views/projects/products"
        );
        drop(first);
        drop(
            acquire_manager_guard(&root).expect("A fresh operation can be admitted after release"),
        );
        let path = root.join(".made-doctor-install.lock");
        std::fs::write(&path, "unrelated data").expect("Foreign sentinel");
        assert!(acquire_manager_guard(&root).is_err());
        assert_eq!(
            std::fs::read_to_string(&path).expect("Sentinel"),
            "unrelated data"
        );
        std::fs::remove_file(&path).expect("Remove only fixture file");
        let sentinel = root.join("sentinel");
        std::fs::write(&sentinel, "preserved").expect("Sentinel");
        std::os::unix::fs::symlink(&sentinel, &path).expect("Alias fixture");
        assert!(acquire_manager_guard(&root).is_err());
        assert_eq!(
            std::fs::read_to_string(&sentinel).expect("Preserved alias target"),
            "preserved"
        );
    }

    #[test]
    fn doctor_jvm_install_exact_request_and_metadata_security() {
        assert!(request().validate().is_ok());
        for version in [
            "latest",
            "corretto-21",
            "temurin-21.0",
            "java@21.0.1",
            "core:java-21.0.1",
            "corretto-21.0.1;write",
            "corretto-21.0.1[hook=x]",
            "custom-21.0.1",
            "corretto-21.0.1+",
            "corretto-21.0.1../path",
            "corretto-4294967296.0.1",
        ] {
            assert!(
                MiseJdkRequest {
                    version: version.into()
                }
                .validate()
                .is_err(),
                "{version}"
            );
        }
        let artifact = parse_metadata(&encoded(&metadata_value()), &request())
            .expect("Exact supported metadata");
        assert_eq!(artifact.java_version, "99.9.9");
        for (name, replacement) in [
            (
                "url",
                "http://corretto.aws/downloads/resources/99.9.9/jdk.tar.gz",
            ),
            ("url", "https://evil.example/jdk.tar.gz"),
            ("url", "https://corretto.aws@evil.example/jdk.tar.gz"),
            (
                "url",
                "https://token@corretto.aws/downloads/resources/99.9.9/jdk.tar.gz",
            ),
            (
                "url",
                "https://corretto.aws/downloads/resources/99.9.9/jdk.tar.gz?token=secret",
            ),
            ("version", "99.9.8"),
            ("image_type", "jre"),
            ("checksum", "sha256:bad"),
            ("jvm_impl", "openj9"),
        ] {
            let mut value = metadata_value();
            if let rmpv::Value::Map(map) = &mut value
                && let Some((_, rmpv::Value::Map(fields))) = map.first_mut()
            {
                fields
                    .iter_mut()
                    .find(|(key, _)| key.as_str() == Some(name))
                    .expect("Field")
                    .1 = replacement.into();
            }
            assert!(
                parse_metadata(&encoded(&value), &request()).is_err(),
                "{name} {replacement}"
            );
        }
        assert!(parse_metadata(b"invalid", &request()).is_err());
        let mut duplicate = metadata_value();
        if let rmpv::Value::Map(entries) = &mut duplicate {
            entries.push(entries.first().expect("Entry").clone());
        }
        assert!(parse_metadata(&encoded(&duplicate), &request()).is_err());
        let mut inflated = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
        inflated
            .write_all(&vec![0; 16 * 1024 * 1024 + 1])
            .expect("Bound fixture");
        assert!(
            parse_metadata(&inflated.finish().expect("Bound compression"), &request()).is_err()
        );
    }

    pub(crate) struct Fixture {
        tree: TempTree,
        pub(crate) home: PathBuf,
        filesystem: Arc<dyn fs::Fs>,
        pub(crate) project: Entity<Project>,
        directory: ProjectPath,
        digest: String,
    }

    pub(crate) async fn fixture(cx: &mut gpui::TestAppContext) -> Fixture {
        cx.executor().allow_parking();
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            trusted_worktrees::init(DbTrustedPaths::default(), cx);
        });
        let tree = TempTree::new(
            serde_json::json!({"home":{".local":{"share":{"mise":{}}},"config.sentinel":"existing configuration"},"project":{"pom.xml":"unrelated project","source.java":"unrelated source"}}),
        );
        let root = tree.path().canonicalize().expect("Fixture root");
        let home = root.join("home");
        let source = root.join("fixture.c");
        std::fs::write(&source, FIXTURE).expect("Fixture source");
        let executable = home.join("mise");
        let compiled = smol::process::Command::new("/usr/bin/cc")
            .arg(&source)
            .arg("-o")
            .arg(&executable)
            .output()
            .await
            .expect("Existing native compiler");
        assert!(
            compiled.status.success(),
            "{}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        let digest = format!(
            "{:x}",
            Sha256::digest(std::fs::read(&executable).expect("Native fixture"))
        );
        std::fs::write(
            home.join(".local/share/mise/metadata.fixture"),
            encoded(&metadata_value()),
        )
        .expect("Owned fixture metadata");
        let filesystem: Arc<dyn fs::Fs> = Arc::new(fs::RealFs::new(None, cx.executor()));
        let project = Project::test_with_worktree_trust(
            filesystem.clone(),
            [root.join("project").as_path()],
            cx,
        )
        .await;
        let directory = project.read_with(cx, |project, cx| ProjectPath {
            worktree_id: project
                .visible_worktrees(cx)
                .next()
                .expect("Worktree")
                .read(cx)
                .id(),
            path: RelPath::empty_arc(),
        });
        let store = project.read_with(cx, |project, _| project.worktree_store());
        let trust = cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("Trust owner"));
        trust.update(cx, |trust, cx| {
            trust.trust(
                &store,
                collections::HashSet::from_iter([PathTrust::Worktree(directory.worktree_id)]),
                cx,
            )
        });
        Fixture {
            tree,
            home,
            filesystem,
            project,
            directory,
            digest,
        }
    }

    pub(crate) async fn inspection(
        fixture: &Fixture,
        cx: &mut gpui::TestAppContext,
    ) -> DoctorRepairReview<MiseAction> {
        let roots = vec![fixture.tree.path().join("project")];
        let snapshot = MiseSnapshot::inspect(
            fixture.filesystem.as_ref(),
            fixture.home.clone(),
            fixture.home.join("mise"),
            roots,
            &fixture.digest,
        )
        .await
        .expect("Fixture-only native manager identity");
        let binding = cx
            .update(|cx| project_repair_binding(&fixture.project, fixture.directory.clone(), 1, cx))
            .expect("Selected authority");
        DoctorRepairReview::new(
            MiseAction {
                snapshot,
                request: request(),
                stage: MiseStage::Inspect,
                cache: OnceLock::new(),
                metadata: None,
            },
            binding,
            DoctorCheck::new(
                "jvm.mise.inspect",
                "Manager inspection",
                CheckOutcome::Untested,
                "Not invoked",
                10,
            ),
            DiagnosticText::new("Review exact isolated native manager version only"),
        )
        .expect("Review")
    }

    fn approve(
        review: &mut DoctorRepairReview<MiseAction>,
        fixture: &Fixture,
        cx: &mut gpui::TestAppContext,
    ) {
        let binding = cx
            .update(|cx| project_repair_binding(&fixture.project, fixture.directory.clone(), 1, cx))
            .expect("Current authority");
        review.approve(&binding).expect("Explicit action approval");
        assert!(review.approve(&binding).is_err());
    }

    fn launch(
        review: DoctorRepairReview<MiseAction>,
        fixture: &Fixture,
        history: DoctorHistoryStore,
        cancel: async_channel::Sender<()>,
        cancelled: async_channel::Receiver<()>,
        cx: &mut gpui::TestAppContext,
    ) -> Task<Result<MiseOperationResult>> {
        let mut report = DoctorReport::new(
            DoctorScope {
                product: DiagnosticText::new("Groovy"),
                project: DiagnosticText::new("fixture"),
                worktree: DiagnosticText::new("root"),
                host: DiagnosticText::new("Local"),
            },
            1,
            10,
        );
        report
            .push(review.before().clone())
            .expect("Relevant evidence");
        cx.update(|cx| {
            launch_mise_operation(
                review,
                &fixture.project,
                fixture.directory.clone(),
                1,
                report,
                DoctorProduct::Groovy,
                history,
                10,
                cancel,
                cancelled,
                cx,
            )
        })
    }

    async fn install_proposal(
        fixture: &Fixture,
        history: DoctorHistoryStore,
        cx: &mut gpui::TestAppContext,
    ) -> DoctorRepairReview<MiseAction> {
        let mut review = inspection(fixture, cx).await;
        approve(&mut review, fixture, cx);
        let (cancel, cancelled) = async_channel::bounded(1);
        let result = launch(review, fixture, history.clone(), cancel, cancelled, cx)
            .await
            .expect("Version operation");
        assert_eq!(result.review.phase(), DoctorRepairPhase::Verified);
        let mut resolve = result.proposal.expect("Separate source review");
        assert_eq!(resolve.phase(), DoctorRepairPhase::Proposed);
        assert_eq!(resolve.action().stage(), MiseStage::Resolve);
        assert!(!resolve.action().destination().exists());
        approve(&mut resolve, fixture, cx);
        let (cancel, cancelled) = async_channel::bounded(1);
        let result = launch(resolve, fixture, history, cancel, cancelled, cx)
            .await
            .expect("Metadata operation");
        assert_eq!(result.review.phase(), DoctorRepairPhase::Verified);
        assert!(!result.review.action().destination().exists());
        let proposal = result.proposal.expect("Separate install review");
        assert_eq!(proposal.phase(), DoctorRepairPhase::Proposed);
        proposal
    }

    #[gpui::test]
    async fn doctor_jvm_install_separate_approvals_inventory_compiler_and_cleanup(
        cx: &mut gpui::TestAppContext,
    ) {
        let fixture = fixture(cx).await;
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::for_test_database(&database);
        let mut proposal = install_proposal(&fixture, history.clone(), cx).await;
        assert_eq!(
            proposal
                .action()
                .artifact()
                .expect("Reviewed source")
                .source,
            "https://corretto.aws/downloads/resources/99.9.9/jdk.tar.gz"
        );
        let cache = proposal
            .action()
            .cache
            .get()
            .expect("Owned cache")
            .path()
            .to_path_buf();
        approve(&mut proposal, &fixture, cx);
        let (cancel, cancelled) = async_channel::bounded(1);
        let result = launch(proposal, &fixture, history.clone(), cancel, cancelled, cx)
            .await
            .expect("Owned installation");
        assert_eq!(
            result.review.phase(),
            DoctorRepairPhase::Verified,
            "{:?}",
            result.review.operation_detail()
        );
        assert!(result.proposal.is_none());
        assert!(
            result
                .review
                .action()
                .destination()
                .join("bin/javac")
                .exists()
        );
        assert_eq!(
            std::fs::read_to_string(fixture.tree.path().join("project/pom.xml"))
                .expect("Project sentinel"),
            "unrelated project"
        );
        assert_eq!(
            std::fs::read_to_string(fixture.home.join("config.sentinel")).expect("Config sentinel"),
            "existing configuration"
        );
        let records = history
            .records(DoctorProduct::Groovy, 20)
            .await
            .expect("History");
        assert_eq!(records.len(), 3);
        assert!(
            records
                .iter()
                .all(|record| record.outcome == ReceiptOutcome::Completed)
        );
        drop(result);
        assert!(
            !cache.exists(),
            "Only the disposable manager cache is cleaned"
        );
        assert!(
            fixture
                .home
                .join(".local/share/mise/installs/java/corretto-99.9.9/bin/java")
                .exists()
        );
    }

    #[gpui::test]
    async fn doctor_jvm_install_unapproved_stale_source_and_manager_block_effects(
        cx: &mut gpui::TestAppContext,
    ) {
        let fixture = fixture(cx).await;
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::for_test_database(&database);
        let review = inspection(&fixture, cx).await;
        let (cancel, cancelled) = async_channel::bounded(1);
        assert!(
            launch(review, &fixture, history.clone(), cancel, cancelled, cx)
                .await
                .is_err()
        );
        assert!(
            history
                .records(DoctorProduct::Groovy, 20)
                .await
                .expect("History")
                .is_empty()
        );
        let mut proposal = install_proposal(&fixture, history.clone(), cx).await;
        approve(&mut proposal, &fixture, cx);
        let metadata = &proposal
            .action()
            .metadata
            .as_ref()
            .expect("Metadata")
            .identity
            .path;
        std::fs::write(metadata, b"changed source").expect("Source race");
        let (cancel, cancelled) = async_channel::bounded(1);
        let result = launch(proposal, &fixture, history.clone(), cancel, cancelled, cx)
            .await
            .expect("Observed stale refusal");
        assert_eq!(result.review.phase(), DoctorRepairPhase::Incomplete);
        assert!(!fixture.home.join(".local/share/mise/started").exists());
        let executable = fixture.home.join("mise");
        assert!(
            MiseSnapshot::inspect(
                fixture.filesystem.as_ref(),
                fixture.home.clone(),
                executable.clone(),
                vec![],
                supported_mise_digest().expect("Supported platform")
            )
            .await
            .is_err(),
            "Fixture binary is never accepted by production release identity"
        );
        std::fs::create_dir_all(fixture.home.join(".local/share/mise/plugins/java"))
            .expect("Override");
        assert!(
            MiseSnapshot::inspect(
                fixture.filesystem.as_ref(),
                fixture.home.clone(),
                executable,
                vec![],
                &fixture.digest
            )
            .await
            .is_err()
        );
    }

    #[gpui::test]
    async fn doctor_jvm_install_success_without_inventory_or_matching_release_is_incomplete(
        cx: &mut gpui::TestAppContext,
    ) {
        for mode in ["empty", "wrong", "failure", "badcompiler", "badruntime"] {
            let fixture = fixture(cx).await;
            let database = db::AppDatabase::test_new();
            let history = DoctorHistoryStore::for_test_database(&database);
            let mut proposal = install_proposal(&fixture, history.clone(), cx).await;
            approve(&mut proposal, &fixture, cx);
            std::fs::write(fixture.home.join(".local/share/mise/fixture-mode"), mode)
                .expect("Fixture outcome");
            let (cancel, cancelled) = async_channel::bounded(1);
            let result = launch(proposal, &fixture, history.clone(), cancel, cancelled, cx)
                .await
                .expect("Observed incomplete result");
            assert_eq!(
                result.review.phase(),
                DoctorRepairPhase::Incomplete,
                "{mode}"
            );
            assert!(result.proposal.is_none());
            assert!(
                history
                    .records(DoctorProduct::Groovy, 20)
                    .await
                    .expect("History")
                    .iter()
                    .any(|record| record.outcome == ReceiptOutcome::Incomplete)
            );
        }
    }

    #[gpui::test]
    async fn doctor_jvm_install_proposal_cancellation_and_cleanup_failure_are_observed(
        cx: &mut gpui::TestAppContext,
    ) {
        let fixture = fixture(cx).await;
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::for_test_database(&database);
        let mut review = inspection(&fixture, cx).await;
        approve(&mut review, &fixture, cx);
        let (cancel, cancelled) = async_channel::bounded(1);
        let mut result = launch(review, &fixture, history.clone(), cancel, cancelled, cx)
            .await
            .expect("Manager inspection");
        let cache = result
            .review
            .action()
            .cache
            .get()
            .expect("Owned cache")
            .path()
            .to_path_buf();
        result
            .cancel_proposal()
            .expect("Cancel proposed next action and clean only private cache");
        assert!(result.proposal.is_none());
        assert!(!cache.exists());
        assert!(!result.review.action().destination().exists());
        assert_eq!(
            history
                .records(DoctorProduct::Groovy, 20)
                .await
                .expect("History")
                .len(),
            1
        );
        let mut proposal = install_proposal(&fixture, history.clone(), cx).await;
        approve(&mut proposal, &fixture, cx);
        let (cancel, cancelled) = async_channel::bounded(1);
        cancel.try_send(()).expect("Cancel before dispatch");
        assert!(
            launch(proposal, &fixture, history.clone(), cancel, cancelled, cx)
                .await
                .is_err()
        );
        assert_eq!(
            history
                .records(DoctorProduct::Groovy, 20)
                .await
                .expect("History")
                .len(),
            3
        );
        assert!(!fixture.home.join(".local/share/mise/started").exists());
        let mut cache = MiseCache::new(
            tempfile::Builder::new()
                .prefix("cleanup-fixture-")
                .tempdir_in(&fixture.home)
                .expect("Owned cache"),
        );
        std::fs::remove_dir(cache.path()).expect("Simulate external cache removal");
        assert!(
            cache.close().is_err(),
            "Cleanup errors are surfaced rather than silently discarded"
        );
        assert!(
            cache
                .directory
                .get_mut()
                .expect("Uncontended test lock")
                .is_none()
        );
    }

    async fn wait_started(fixture: &Fixture) {
        let path = fixture.home.join(".local/share/mise/started");
        smol::unblock(move || {
            for _ in 0..500 {
                if path.exists() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            panic!("Owned fixture did not start");
        })
        .await;
    }

    #[gpui::test]
    async fn doctor_jvm_install_owned_cancellation_deadline_and_revocation(
        cx: &mut gpui::TestAppContext,
    ) {
        for reason in ["cancel", "timeout", "revocation", "trust"] {
            let fixture = fixture(cx).await;
            let database = db::AppDatabase::test_new();
            let history = DoctorHistoryStore::for_test_database(&database);
            let mut proposal = install_proposal(&fixture, history.clone(), cx).await;
            approve(&mut proposal, &fixture, cx);
            std::fs::write(fixture.home.join(".local/share/mise/fixture-mode"), "wait")
                .expect("Waiting fixture");
            let (cancel, cancelled) = async_channel::bounded(1);
            let task = launch(
                proposal,
                &fixture,
                history.clone(),
                cancel.clone(),
                cancelled,
                cx,
            );
            wait_started(&fixture).await;
            match reason {
                "cancel" => {
                    cancel.try_send(()).expect("Cancel owned operation");
                }
                "timeout" => {
                    cx.executor()
                        .dispatcher()
                        .as_test()
                        .expect("Test dispatcher")
                        .scheduler()
                        .clock()
                        .advance(Duration::from_secs(301));
                }
                "trust" => {
                    let trust =
                        cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("Trust owner"));
                    let store = fixture
                        .project
                        .read_with(cx, |project, _| project.worktree_store());
                    trust.update(cx, |trust, cx| {
                        trust.restrict(
                            store.downgrade(),
                            collections::HashSet::from_iter([PathTrust::Worktree(
                                fixture.directory.worktree_id,
                            )]),
                            cx,
                        )
                    });
                    cx.executor()
                        .dispatcher()
                        .as_test()
                        .expect("Test dispatcher")
                        .scheduler()
                        .clock()
                        .advance(Duration::from_secs(1));
                }
                _ => {
                    std::fs::create_dir_all(fixture.home.join(".local/share/mise/plugins/java"))
                        .expect("Revoke reviewed backend");
                    cx.executor()
                        .dispatcher()
                        .as_test()
                        .expect("Test dispatcher")
                        .scheduler()
                        .clock()
                        .advance(Duration::from_secs(1));
                }
            }
            let result = task.await.expect("Stopped owned operation");
            assert_eq!(
                result.review.phase(),
                if reason == "cancel" {
                    DoctorRepairPhase::Cancelled
                } else {
                    DoctorRepairPhase::Incomplete
                },
                "{reason}"
            );
            assert!(result.proposal.is_none());
            assert!(!result.review.action().destination().exists());
            assert!(
                history
                    .records(DoctorProduct::Groovy, 20)
                    .await
                    .expect("Durable terminal receipt")
                    .iter()
                    .all(|record| record.outcome != ReceiptOutcome::Running)
            );
        }
    }

    #[gpui::test]
    async fn doctor_jvm_install_preserves_existing_inventory_and_blocks_aliases(
        cx: &mut gpui::TestAppContext,
    ) {
        let fixture = fixture(cx).await;
        let review = inspection(&fixture, cx).await;
        std::fs::create_dir_all(review.action().destination()).expect("Concurrent inventory");
        assert!(
            review
                .action()
                .revalidate(fixture.filesystem.as_ref())
                .await
                .is_err()
        );
        std::fs::rename(
            fixture.home.join(".local/share/mise"),
            fixture.home.join("saved-mise"),
        )
        .expect("Move owned fixture data");
        std::os::unix::fs::symlink(
            fixture.tree.path().join("project"),
            fixture.home.join(".local/share/mise"),
        )
        .expect("Escaping project alias");
        assert!(
            MiseSnapshot::inspect(
                fixture.filesystem.as_ref(),
                fixture.home.clone(),
                fixture.home.join("mise"),
                vec![fixture.tree.path().join("project")],
                &fixture.digest
            )
            .await
            .is_err()
        );
    }

    const SDKMAN_CONFIGURATION: &str = "sdkman_auto_answer=false\nsdkman_auto_complete=false\nsdkman_auto_env=false\nsdkman_auto_update=false\nsdkman_checksum_enable=true\nsdkman_insecure_ssl=false\nsdkman_debug_mode=false\nsdkman_native_enable=false\nsdkman_healthcheck_enable=false\nsdkman_curl_retry=0\nsdkman_curl_continue=false\n";
    const SDKMAN_FIXTURE: &str = r#"
function sdk() {
  [[ "$1" == install && "$2" == java && "$3" == 99.9.9-tem ]] || return 31
  [[ "$HOME" != "${SDKMAN_DIR%/.sdkman}" && "$CURL_HOME" == "$HOME" && -f "$HOME/.curlrc" && -z "$JAVA_HOME" && -z "$GITHUB_TOKEN" ]] || return 32
  local answer mode destination
  read answer
  [[ "$answer" == n ]] || return 33
  IFS= read -r mode < "$SDKMAN_DIR/mode" || [[ -n "$mode" ]] || return 34
  case "$mode" in
    success|wait|failure|empty|wrong|badcompiler|badruntime|switch) ;;
    *) return 34 ;;
  esac
  printf '%s\n' "$mode" > "$SDKMAN_DIR/started"
  [[ "$mode" == wait ]] && { sleep 3600; return 7; }
  [[ "$mode" == failure ]] && return 7
  [[ "$mode" == empty ]] && return 0
  destination="$SDKMAN_DIR/candidates/java/$3"
  mkdir -p "$destination/bin" || return 8
  printf 'JAVA_VERSION="99.9.9"\n' > "$destination/release"
  [[ "$mode" == wrong ]] && printf 'JAVA_VERSION="99.9.8"\n' > "$destination/release"
  cp "$SDKMAN_DIR/native" "$destination/bin/java" || return 9
  cp "$SDKMAN_DIR/native" "$destination/bin/javac" || return 9
  [[ "$mode" == badcompiler ]] && touch "$destination/bin/javac.bad"
  [[ "$mode" == badruntime ]] && touch "$destination/bin/java.bad"
  if [[ ! -L "$SDKMAN_DIR/candidates/java/current" ]]; then
    ln -s "$destination" "$SDKMAN_DIR/candidates/java/current" || return 10
  elif [[ "$mode" == switch ]]; then
    rm "$SDKMAN_DIR/candidates/java/current"
    ln -s "$destination" "$SDKMAN_DIR/candidates/java/current"
  fi
  echo fixture-install-completed
  return 0
}
"#;

    fn sdkman_request() -> SdkmanJdkRequest {
        SdkmanJdkRequest {
            version: "99.9.9-tem".into(),
        }
    }

    #[test]
    fn doctor_sdkman_install_request_configuration_and_first_default_semantics() {
        assert!(sdkman_request().validate().is_ok());
        assert_eq!(
            SdkmanJdkRequest {
                version: "8.0.462-tem".into()
            }
            .runtime_version()
            .expect("Java 8"),
            "1.8.0_462"
        );
        for version in [
            "latest",
            "21-tem",
            "21.0.1-ea",
            "21.0.1-local",
            "21.0.1-tem;touch",
            "../21.0.1-tem",
            "4294967296.0.1-tem",
        ] {
            assert!(
                SdkmanJdkRequest {
                    version: version.into()
                }
                .validate()
                .is_err(),
                "{version}"
            );
        }
        assert!(validate_sdkman_configuration(SDKMAN_CONFIGURATION.as_bytes()).is_ok());
        for configuration in [
            SDKMAN_CONFIGURATION.replace("sdkman_auto_env=false", "sdkman_auto_env=true"),
            SDKMAN_CONFIGURATION.replace(
                "sdkman_checksum_enable=true",
                "sdkman_checksum_enable=false",
            ),
            SDKMAN_CONFIGURATION.replace("sdkman_insecure_ssl=false", "sdkman_insecure_ssl=true"),
            SDKMAN_CONFIGURATION.replace("sdkman_native_enable=false", "sdkman_native_enable=true"),
            format!("{SDKMAN_CONFIGURATION}sdkman_auto_answer=false\n"),
            format!("{SDKMAN_CONFIGURATION}sdkman_unknown=false\n"),
            format!("{SDKMAN_CONFIGURATION}touch /tmp/forbidden\n"),
            format!("{SDKMAN_CONFIGURATION}sdkman_colour_enable=$(touch /tmp/forbidden)\n"),
        ] {
            assert!(validate_sdkman_configuration(configuration.as_bytes()).is_err());
        }
    }

    pub(crate) async fn sdkman_fixture(
        existing_default: bool,
        cx: &mut gpui::TestAppContext,
    ) -> (Fixture, DoctorRepairReview<SdkmanAction>) {
        let fixture = fixture(cx).await;
        let root = fixture.home.join(".sdkman");
        for name in [
            "bin",
            "src",
            "ext",
            "etc",
            "var/metadata",
            "tmp",
            "candidates/java",
        ] {
            std::fs::create_dir_all(root.join(name)).expect("Owned SDK fixture");
        }
        std::fs::write(root.join("bin/sdkman-init.sh"), SDKMAN_FIXTURE)
            .expect("Fixture-only manager");
        std::fs::write(root.join("etc/config"), SDKMAN_CONFIGURATION).expect("Safe literal config");
        std::fs::write(root.join("var/version"), "5.23.1").expect("Fixture version");
        let platform = match (std::env::consts::OS, std::env::consts::ARCH) {
            ("macos", "aarch64") => "darwinarm64",
            ("macos", "x86_64") => "darwinx64",
            ("linux", "aarch64") => "linuxarm64",
            _ => "linuxx64",
        };
        std::fs::write(root.join("var/platform"), platform).expect("Platform");
        std::fs::write(root.join("var/candidates"), "java,kotlin").expect("Candidates");
        std::fs::copy(fixture.home.join("mise"), root.join("native")).expect("Native JDK fixture");
        std::fs::write(root.join("mode"), "success").expect("Mode");
        if existing_default {
            let previous = root.join("candidates/java/previous");
            std::fs::create_dir_all(previous.join("bin")).expect("Existing JDK");
            std::fs::write(previous.join("release"), "JAVA_VERSION=\"21.0.1\"\n")
                .expect("Preserved release");
            std::fs::copy(root.join("native"), previous.join("bin/java"))
                .expect("Existing executable");
            std::os::unix::fs::symlink(&previous, root.join("candidates/java/current"))
                .expect("Existing selected default");
        }
        let digest = format!("{:x}", Sha256::digest(SDKMAN_FIXTURE.as_bytes()));
        let snapshot = SdkmanSnapshot::inspect(
            fixture.filesystem.as_ref(),
            fixture.home.clone(),
            vec![fixture.tree.path().join("project")],
            &[("bin/sdkman-init.sh", &digest)],
        )
        .await
        .expect("Fixture-only SDK identity");
        let binding = cx
            .update(|cx| project_repair_binding(&fixture.project, fixture.directory.clone(), 1, cx))
            .expect("Current authority");
        let action = SdkmanAction {
            snapshot,
            request: sdkman_request(),
            first_default_consent: OnceLock::new(),
        };
        action
            .revalidate(fixture.filesystem.as_ref(), false)
            .await
            .expect("Unmodified fixture");
        let review = DoctorRepairReview::new(action, binding, DoctorCheck::new("jvm.sdkman.install", "Exact requested JDK", CheckOutcome::Blocked, "Not installed", 10), DiagnosticText::new("Review exact manager, sources, official hook execution, download/cache effects, preserved default or separate first-default consent")).expect("Review");
        (fixture, review)
    }

    fn sdkman_approve(
        review: &mut DoctorRepairReview<SdkmanAction>,
        fixture: &Fixture,
        acknowledge_default: bool,
        cx: &mut gpui::TestAppContext,
    ) {
        let binding = cx
            .update(|cx| project_repair_binding(&fixture.project, fixture.directory.clone(), 1, cx))
            .expect("Authority");
        if acknowledge_default {
            review
                .approve_sdkman_first_default(&binding)
                .expect("Separate exact first-default approval");
            assert!(
                review.approve_sdkman_first_default(&binding).is_err(),
                "Duplicate acknowledgments do not create operations"
            );
        }
        review.approve(&binding).expect("Install approval");
        assert!(
            review.approve_sdkman_first_default(&binding).is_err(),
            "Consent is not inferred after install approval"
        );
    }

    fn sdkman_launch(
        review: DoctorRepairReview<SdkmanAction>,
        fixture: &Fixture,
        history: DoctorHistoryStore,
        cancel: async_channel::Sender<()>,
        cancelled: async_channel::Receiver<()>,
        cx: &mut gpui::TestAppContext,
    ) -> Task<Result<SdkmanOperationResult>> {
        let mut report = DoctorReport::new(
            DoctorScope {
                product: DiagnosticText::new("Groovy"),
                project: DiagnosticText::new("fixture"),
                worktree: DiagnosticText::new("root"),
                host: DiagnosticText::new("Local"),
            },
            1,
            10,
        );
        report.push(review.before().clone()).expect("Evidence");
        cx.update(|cx| {
            launch_sdkman_installation(
                review,
                &fixture.project,
                fixture.directory.clone(),
                1,
                report,
                DoctorProduct::Groovy,
                history,
                10,
                cancel,
                cancelled,
                cx,
            )
        })
    }

    #[gpui::test]
    async fn doctor_sdkman_install_first_default_requires_distinct_consent(
        cx: &mut gpui::TestAppContext,
    ) {
        let (fixture, mut review) = sdkman_fixture(false, cx).await;
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::for_test_database(&database);
        assert!(review.action().requires_first_default());
        assert_eq!(
            review.action().sources().expect("Explicit sources").len(),
            3
        );
        sdkman_approve(&mut review, &fixture, false, cx);
        let (cancel, cancelled) = async_channel::bounded(1);
        assert!(
            sdkman_launch(review, &fixture, history.clone(), cancel, cancelled, cx)
                .await
                .is_err()
        );
        assert!(!fixture.home.join(".sdkman/started").exists());
        assert!(
            history
                .records(DoctorProduct::Groovy, 20)
                .await
                .expect("History")
                .is_empty()
        );
        let (fixture, mut review) = sdkman_fixture(false, cx).await;
        sdkman_approve(&mut review, &fixture, true, cx);
        let destination = review.action().destination();
        let (cancel, cancelled) = async_channel::bounded(1);
        let result = sdkman_launch(review, &fixture, history.clone(), cancel, cancelled, cx)
            .await
            .expect("Approved owner operation");
        assert_eq!(
            result.review.phase(),
            DoctorRepairPhase::Verified,
            "{:?}",
            result.review.operation_detail()
        );
        assert!(result.cleanup_error.is_none());
        assert_eq!(
            std::fs::canonicalize(fixture.home.join(".sdkman/candidates/java/current"))
                .expect("Observed default"),
            destination
        );
        assert_eq!(
            history
                .records(DoctorProduct::Groovy, 20)
                .await
                .expect("History")
                .len(),
            1
        );
        assert_eq!(
            std::fs::read_to_string(fixture.tree.path().join("project/pom.xml"))
                .expect("Project sentinel"),
            "unrelated project"
        );
        assert!(
            !std::fs::read_dir(&fixture.home)
                .expect("Cleanup")
                .any(|entry| entry
                    .expect("Entry")
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".made-doctor-sdkman-"))
        );
    }

    #[gpui::test]
    async fn doctor_sdkman_install_preserves_existing_default_and_rechecks_compiler(
        cx: &mut gpui::TestAppContext,
    ) {
        for mode in [
            "success",
            "empty",
            "wrong",
            "badcompiler",
            "badruntime",
            "failure",
            "switch",
        ] {
            let (fixture, mut review) = sdkman_fixture(true, cx).await;
            assert!(!review.action().requires_first_default());
            let before =
                std::fs::canonicalize(fixture.home.join(".sdkman/candidates/java/current"))
                    .expect("Before");
            std::fs::write(fixture.home.join(".sdkman/mode"), mode).expect("Mode");
            let database = db::AppDatabase::test_new();
            let history = DoctorHistoryStore::for_test_database(&database);
            sdkman_approve(&mut review, &fixture, false, cx);
            let (cancel, cancelled) = async_channel::bounded(1);
            let result = sdkman_launch(review, &fixture, history.clone(), cancel, cancelled, cx)
                .await
                .expect("Observed manager operation");
            assert_eq!(
                result.review.phase(),
                if mode == "success" {
                    DoctorRepairPhase::Verified
                } else {
                    DoctorRepairPhase::Incomplete
                },
                "{mode}: {:?}",
                result.review.operation_detail()
            );
            if mode != "switch" {
                assert_eq!(
                    std::fs::canonicalize(fixture.home.join(".sdkman/candidates/java/current"))
                        .expect("Preserved default"),
                    before
                );
            }
            assert!(
                history
                    .records(DoctorProduct::Groovy, 20)
                    .await
                    .expect("Terminal receipt")
                    .iter()
                    .all(|record| record.outcome != ReceiptOutcome::Running)
            );
        }
    }

    #[gpui::test]
    async fn doctor_sdkman_install_rejects_sources_configuration_scratch_and_stale_approval(
        cx: &mut gpui::TestAppContext,
    ) {
        let (fixture, mut review) = sdkman_fixture(true, cx).await;
        let root = fixture.home.join(".sdkman");
        assert!(
            SdkmanSnapshot::inspect(
                fixture.filesystem.as_ref(),
                fixture.home.clone(),
                vec![],
                SDKMAN_SOURCES
            )
            .await
            .is_err(),
            "Fixture shell is never accepted as official SDKMAN"
        );
        for (path, content) in [
            ("src/sdkman-extra.sh", "echo forbidden"),
            ("tmp/out", "foreign scratch"),
            ("var/metadata/java-99.9.9-tem.headers", "foreign download"),
            ("etc/config", "touch /tmp/forbidden"),
        ] {
            let path = root.join(path);
            let previous = std::fs::read(&path).ok();
            std::fs::write(&path, content).expect("Unreviewed input");
            assert!(
                review
                    .action()
                    .revalidate(fixture.filesystem.as_ref(), false)
                    .await
                    .is_err(),
                "{path:?}"
            );
            if let Some(previous) = previous {
                std::fs::write(&path, previous).expect("Fixture restoration");
            } else {
                std::fs::remove_file(&path).expect("Remove own fixture");
            }
        }
        std::fs::create_dir(root.join("src/nested")).expect("Hidden recursive script folder");
        assert!(
            review
                .action()
                .revalidate(fixture.filesystem.as_ref(), false)
                .await
                .is_err()
        );
        std::fs::remove_dir(root.join("src/nested")).expect("Remove fixture");
        let stale = cx
            .update(|cx| project_repair_binding(&fixture.project, fixture.directory.clone(), 2, cx))
            .expect("Changed generation");
        assert!(review.approve(&stale).is_err());
        sdkman_approve(&mut review, &fixture, false, cx);
        std::fs::write(root.join("bin/sdkman-init.sh"), "echo changed")
            .expect("Post-approval source race");
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::for_test_database(&database);
        let (cancel, cancelled) = async_channel::bounded(1);
        let result = sdkman_launch(review, &fixture, history, cancel, cancelled, cx)
            .await
            .expect("Recorded refusal");
        assert_eq!(result.review.phase(), DoctorRepairPhase::Incomplete);
        assert!(!root.join("started").exists());
    }

    #[gpui::test]
    async fn doctor_sdkman_install_cancellation_deadline_and_source_revocation(
        cx: &mut gpui::TestAppContext,
    ) {
        for reason in ["cancel", "timeout", "source"] {
            let (fixture, mut review) = sdkman_fixture(true, cx).await;
            let root = fixture.home.join(".sdkman");
            std::fs::write(root.join("mode"), "wait").expect("Wait");
            let database = db::AppDatabase::test_new();
            let history = DoctorHistoryStore::for_test_database(&database);
            sdkman_approve(&mut review, &fixture, false, cx);
            let destination = review.action().destination();
            let (cancel, cancelled) = async_channel::bounded(1);
            let task = sdkman_launch(
                review,
                &fixture,
                history.clone(),
                cancel.clone(),
                cancelled,
                cx,
            );
            let started = root.join("started");
            smol::unblock(move || {
                let deadline = std::time::Instant::now() + Duration::from_secs(10);
                loop {
                    match std::fs::read_to_string(&started) {
                        Ok(mode) if mode == "wait\n" => break,
                        Ok(mode) if mode.is_empty() => {}
                        Ok(mode) => panic!("Unexpected SDKMAN fixture mode: {mode:?}"),
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                        Err(error) => panic!("SDKMAN fixture readiness failed: {error}"),
                    }
                    assert!(std::time::Instant::now() < deadline, "Native fixture start");
                    std::thread::sleep(Duration::from_millis(10));
                }
            })
            .await;
            if reason == "cancel" {
                cancel.close();
            } else {
                if reason == "source" {
                    std::fs::write(root.join("bin/sdkman-init.sh"), "changed source")
                        .expect("Revoked source");
                }
                cx.executor()
                    .dispatcher()
                    .as_test()
                    .expect("Test dispatcher")
                    .scheduler()
                    .clock()
                    .advance(if reason == "timeout" {
                        Duration::from_secs(301)
                    } else {
                        Duration::from_millis(101)
                    });
            }
            let result = task.await.expect("Observed termination");
            assert_eq!(
                result.review.phase(),
                if reason == "cancel" {
                    DoctorRepairPhase::Cancelled
                } else {
                    DoctorRepairPhase::Incomplete
                },
                "{reason}"
            );
            assert!(!destination.exists(), "{reason}");
            assert!(
                history
                    .records(DoctorProduct::Groovy, 20)
                    .await
                    .expect("Receipt")
                    .iter()
                    .all(|record| record.outcome != ReceiptOutcome::Running)
            );
            drop(acquire_manager_guard(&root).expect("Guard released only after termination"));
        }
    }
}
