use anyhow::{Context as _, Result, bail, ensure};
use futures::{AsyncReadExt, AsyncWriteExt, FutureExt as _};
use http_client::{
    AsyncBody, HttpClient, HttpRequestExt as _, RedirectPolicy, Request, github::AssetKind,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::Read as _,
    net::TcpListener,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

const GIB: u64 = 1024 * 1024 * 1024;
const OWNER: &str = "Copper managed local models v1\n";

#[derive(Clone, Debug, Deserialize)]
pub struct Catalog {
    pub runtime_version: String,
    pub runtime_revision: String,
    pub runtime_license: String,
    pub runtime_assets: Vec<RuntimeAsset>,
    pub models: Vec<ModelArtifact>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RuntimeAsset {
    pub platform: String,
    pub name: String,
    pub url: String,
    pub bytes: u64,
    pub sha256: String,
    pub backend: String,
    #[serde(default)]
    pub minimum_macos_version: Option<String>,
    #[serde(default)]
    pub minimum_glibc_version: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelArtifact {
    pub id: String,
    pub name: String,
    pub family: String,
    pub repository: String,
    pub revision: String,
    pub filename: String,
    pub quantization: String,
    pub bytes: u64,
    pub sha256: String,
    pub license: String,
    pub license_url: String,
    #[serde(default)]
    pub attribution: Option<String>,
    pub training_context: u64,
    pub context_tokens: u64,
    pub kv_layers: u64,
    pub kv_heads: u64,
    pub kv_head_dimension: u64,
    pub chat_template: String,
    pub chat_template_sha256: String,
    pub runtime_template_sha256: String,
    pub gguf_architecture: String,
    pub tool_support: String,
    pub qualified: bool,
}

pub fn catalog() -> Result<Catalog> {
    let catalog: Catalog = serde_json::from_str(include_str!("../managed_catalog.json"))?;
    for model in &catalog.models {
        ensure!(
            safe_name(&model.id) && safe_name(&model.filename),
            "Invalid managed artifact name"
        );
        ensure!(
            model.revision.len() == 40 && is_hex(&model.revision),
            "Model revision must be immutable"
        );
        ensure!(
            model.sha256.len() == 64 && is_hex(&model.sha256),
            "Model digest must be SHA-256"
        );
        ensure!(
            [&model.chat_template_sha256, &model.runtime_template_sha256]
                .into_iter()
                .all(|digest| digest.len() == 64 && is_hex(digest)),
            "Embedded and served chat template digests must be SHA-256"
        );
        ensure!(
            model.context_tokens <= model.training_context,
            "Context exceeds artifact limit"
        );
    }
    for asset in &catalog.runtime_assets {
        ensure!(
            asset
                .url
                .starts_with("https://github.com/ggml-org/llama.cpp/releases/download/")
                && asset
                    .url
                    .contains(&format!("/{}/", catalog.runtime_version))
                && asset.sha256.len() == 64
                && is_hex(&asset.sha256),
            "Runtime must have a pinned official archive and digest"
        );
    }
    Ok(catalog)
}

fn safe_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"-_.".contains(&byte))
}

fn is_hex(value: &str) -> bool {
    value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

impl ModelArtifact {
    pub fn url(&self) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            self.repository, self.revision, self.filename
        )
    }

    pub fn kv_cache_bytes(&self) -> u64 {
        // Reserve full-length FP16 keys and values, even for sliding/shared KV layers.
        // This deliberately overestimates those models rather than using active parameters.
        self.context_tokens
            .saturating_mul(self.kv_layers)
            .saturating_mul(self.kv_heads)
            .saturating_mul(self.kv_head_dimension)
            .saturating_mul(4)
    }

    pub fn runtime_overhead_bytes(&self) -> u64 {
        (self.bytes / 4).max(GIB)
    }

    pub fn estimated_memory_bytes(&self) -> u64 {
        self.bytes
            .saturating_add(self.kv_cache_bytes())
            .saturating_add(self.runtime_overhead_bytes())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecommendationLevel {
    Recommended,
    LimitedHeadroom,
    InsufficientCapacity,
}

impl RecommendationLevel {
    pub fn label(self) -> &'static str {
        match self {
            Self::Recommended => "Recommended",
            Self::LimitedHeadroom => "Limited headroom",
            Self::InsufficientCapacity => "Insufficient capacity",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ModelRecommendation {
    pub level: RecommendationLevel,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub struct Hardware {
    pub architecture: String,
    pub total_memory: u64,
    pub available_memory: u64,
    pub free_disk: u64,
    pub backend: String,
    pub unified_memory: bool,
    pub gpu_memory: Option<u64>,
    pub runtime_warning: Option<String>,
}

impl Hardware {
    pub fn detect(path: &Path) -> Result<Self> {
        let system = System::new_with_specifics(
            RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()),
        );
        let path = path
            .canonicalize()
            .context("Locate the model storage volume")?;
        Ok(Self {
            architecture: std::env::consts::ARCH.into(),
            total_memory: system.total_memory(),
            available_memory: if cfg!(target_os = "macos") {
                // sysinfo 0.37 subtracts compressed pages from reclaimable pages on macOS.
                // Its used-memory metric already includes those resident compressed pages.
                system.total_memory().saturating_sub(system.used_memory())
            } else {
                system.available_memory()
            },
            free_disk: available_disk_space(&path)?,
            backend: if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                "Metal candidate / CPU fallback; device probe required".into()
            } else {
                "CPU fallback; accelerated hardware is not yet qualified".into()
            },
            unified_memory: cfg!(all(target_os = "macos", target_arch = "aarch64")),
            gpu_memory: None,
            runtime_warning: ensure_host_runtime_compatibility()
                .err()
                .map(|error| error.to_string()),
        })
    }

    pub fn summary(&self) -> String {
        let architecture = match self.architecture.as_str() {
            "aarch64" if self.unified_memory => "Apple silicon",
            "aarch64" => "ARM64",
            "x86_64" => "x86-64",
            architecture => architecture,
        };
        format!(
            "{} · {:.0} GiB {} · {:.1} GiB available (estimate) · {:.1} GiB free disk",
            architecture,
            gib(self.total_memory),
            if self.unified_memory {
                "unified memory"
            } else {
                "RAM"
            },
            gib(self.available_memory),
            gib(self.free_disk)
        )
    }

    pub fn device_details(&self) -> Vec<String> {
        self.backend.lines().filter(|line| line.contains(':') && !line.starts_with("Available devices"))
            .map(|line| {
                if line.contains("BLAS:") && line.contains("Accelerate") {
                    "Accelerate optimizes CPU math and uses system memory; it has no separate VRAM pool.".to_owned()
                } else if self.unified_memory {
                    line.split_once(" (").map_or(line, |(device, _)| device).to_owned()
                } else { line.to_owned() }
            }).collect()
    }

    pub fn recommendation(
        &self,
        model: &ModelArtifact,
        needs_download: bool,
        loaded: bool,
        cpu_only: bool,
    ) -> ModelRecommendation {
        let result = |level, reason| ModelRecommendation { level, reason };
        if let Some(warning) = &self.runtime_warning {
            return result(RecommendationLevel::InsufficientCapacity, warning.clone());
        }
        if needs_download && model.bytes.saturating_add(GIB) > self.free_disk {
            return result(
                RecommendationLevel::InsufficientCapacity,
                format!(
                    "Downloading needs {:.1} GiB including a 1 GiB disk reserve; {:.1} GiB is free. Free space or choose a smaller model.",
                    gib(model.bytes.saturating_add(GIB)),
                    gib(self.free_disk)
                ),
            );
        }
        // A running model has already allocated its memory; charging its weights again
        // against the remaining available memory would report a false load failure.
        let required = if loaded {
            GIB
        } else {
            model.estimated_memory_bytes().saturating_add(GIB)
        };
        if required > self.available_memory {
            return result(
                if loaded {
                    RecommendationLevel::LimitedHeadroom
                } else {
                    RecommendationLevel::InsufficientCapacity
                },
                format!(
                    "{} needs {:.1} GiB including the system reserve; {:.1} GiB is currently available. Close applications{}.",
                    if loaded {
                        "Keeping headroom"
                    } else {
                        "Loading"
                    },
                    gib(required),
                    gib(self.available_memory),
                    if loaded {
                        ""
                    } else {
                        " or choose a smaller model"
                    }
                ),
            );
        }
        if !loaded
            && !cpu_only
            && !self.unified_memory
            && let Some(free) = self.gpu_memory
            && model.estimated_memory_bytes() > free
        {
            return result(
                RecommendationLevel::LimitedHeadroom,
                format!(
                    "The estimate exceeds {:.1} GiB free GPU memory. Choose CPU fallback or a smaller model; system RAM and VRAM are separate limits.",
                    gib(free)
                ),
            );
        }
        let headroom = self.available_memory.saturating_sub(required);
        let level = if headroom < GIB.max(model.estimated_memory_bytes() / 5) {
            RecommendationLevel::LimitedHeadroom
        } else {
            RecommendationLevel::Recommended
        };
        result(
            level,
            if loaded {
                format!(
                    "Loaded successfully; {:.1} GiB remains available beyond the system reserve (estimate).",
                    gib(headroom)
                )
            } else {
                format!(
                    "Estimated fit with {:.1} GiB headroom after model memory and a 1 GiB system reserve.{}",
                    gib(headroom),
                    if cpu_only {
                        " CPU processing of long prompts can take several minutes."
                    } else {
                        " Verify compatibility by loading the model."
                    }
                )
            },
        )
    }

    pub fn ensure_memory(&self, model: &ModelArtifact) -> Result<()> {
        ensure!(
            model.estimated_memory_bytes().saturating_add(GIB) <= self.available_memory,
            "Not enough available memory for this model at {} tokens. Choose a smaller model or close applications; estimated {:.1} GiB plus 1 GiB system reserve is required.",
            model.context_tokens,
            gib(model.estimated_memory_bytes())
        );
        Ok(())
    }
}

fn require_runtime_version(actual: Option<&str>, minimum: &str, component: &str) -> Result<()> {
    fn version_pair(value: &str) -> Result<(u64, u64)> {
        let mut components = value.split('.');
        Ok((
            components
                .next()
                .context("Missing runtime platform major version")?
                .parse()?,
            components
                .next()
                .context("Missing runtime platform minor version")?
                .parse()?,
        ))
    }
    let actual = actual.with_context(|| {
        format!("Cannot determine the {component} version required for managed local models")
    })?;
    ensure!(
        version_pair(actual)? >= version_pair(minimum)?,
        "The reviewed managed local runtime requires {component} {minimum} or newer; this computer reports {component} {actual}. Update the system before downloading these managed models"
    );
    Ok(())
}

fn ensure_host_runtime_compatibility() -> Result<()> {
    let platform = host_platform()?;
    if platform == "macos-aarch64" {
        let catalog = catalog()?;
        let minimum = catalog
            .runtime_assets
            .iter()
            .find(|asset| asset.platform == "macos-aarch64")
            .and_then(|asset| asset.minimum_macos_version.as_deref())
            .context("The reviewed runtime is missing its minimum macOS version")?;
        require_runtime_version(System::os_version().as_deref(), minimum, "macOS")?;
    }
    #[cfg(target_os = "linux")]
    {
        let catalog = catalog()?;
        let minimum = catalog
            .runtime_assets
            .iter()
            .find(|asset| asset.platform == platform)
            .and_then(|asset| asset.minimum_glibc_version.as_deref())
            .context("The reviewed runtime is missing its minimum glibc version")?;
        require_runtime_version(Some(&host_glibc_version()?), minimum, "glibc")?;
    }
    Ok(())
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn host_glibc_version() -> Result<String> {
    let version = unsafe { libc::gnu_get_libc_version() };
    ensure!(
        !version.is_null(),
        "Cannot determine the system glibc version"
    );
    // glibc returns a NUL-terminated static string with process lifetime.
    let version = unsafe { std::ffi::CStr::from_ptr(version) };
    Ok(version
        .to_str()
        .context("The system glibc version is not valid UTF-8")?
        .to_owned())
}

#[cfg(all(target_os = "linux", not(target_env = "gnu")))]
fn host_glibc_version() -> Result<String> {
    bail!(
        "Managed local models require glibc on Linux; this application uses a different C runtime"
    )
}

#[cfg(unix)]
fn available_disk_space(path: &Path) -> Result<u64> {
    use std::{ffi::CString, mem::MaybeUninit, os::unix::ffi::OsStrExt as _};

    let path = CString::new(path.as_os_str().as_bytes())?;
    let mut statistics = MaybeUninit::<libc::statvfs>::uninit();
    // Volume enumeration can omit hidden mounts and incorrectly select their parent.
    // Query this exact path; statvfs initializes the output only on success.
    let status = unsafe { libc::statvfs(path.as_ptr(), statistics.as_mut_ptr()) };
    if status != 0 {
        return Err(std::io::Error::last_os_error())
            .context("Cannot determine free space on the model storage volume");
    }
    let statistics = unsafe { statistics.assume_init() };
    u64::try_from(u128::from(statistics.f_bavail) * u128::from(statistics.f_frsize))
        .context("Model storage volume capacity exceeds the supported size")
}

#[cfg(windows)]
fn available_disk_space(path: &Path) -> Result<u64> {
    use windows::{Win32::Storage::FileSystem::GetDiskFreeSpaceExW, core::HSTRING};

    let path = HSTRING::from(path.as_os_str());
    let mut available = 0;
    unsafe { GetDiskFreeSpaceExW(&path, Some(&mut available), None, None) }
        .context("Cannot determine free space on the model storage volume")?;
    Ok(available)
}

pub fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB as f64
}

#[derive(Clone, Debug, PartialEq)]
pub enum ModelState {
    NotDownloaded,
    Downloading,
    Downloaded,
    Loading,
    Ready,
    Active,
    Failed(String),
}

impl ModelState {
    pub fn label(&self) -> &str {
        match self {
            Self::NotDownloaded => "Not downloaded",
            Self::Downloading => "Downloading",
            Self::Downloaded => "Downloaded",
            Self::Loading => "Loading",
            Self::Ready => "Ready",
            Self::Active => "Active",
            Self::Failed(_) => "Failed",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ManagedStore {
    root: PathBuf,
}

#[derive(Serialize, Deserialize)]
struct Receipt {
    id: String,
    revision: String,
    sha256: String,
    bytes: u64,
}

impl ManagedStore {
    pub fn open(root: PathBuf) -> Result<Self> {
        if root.exists() {
            ensure!(
                !std::fs::symlink_metadata(&root)?.file_type().is_symlink(),
                "Model directory cannot be a symbolic link"
            );
            if !root.join(".copper-owned").exists() {
                ensure!(
                    std::fs::read_dir(&root)?.next().is_none(),
                    "Existing model directory is not Copper-managed; its contents were preserved"
                );
            }
        } else {
            std::fs::create_dir_all(&root)?;
        }
        let marker = root.join(".copper-owned");
        if !marker.exists() {
            std::fs::write(&marker, OWNER)?;
        }
        ensure!(
            !std::fs::symlink_metadata(&marker)?.file_type().is_symlink()
                && std::fs::read_to_string(&marker)? == OWNER,
            "Model ownership marker is invalid"
        );
        Ok(Self {
            root: root.canonicalize()?,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn path(&self, model: &ModelArtifact, suffix: &str) -> Result<PathBuf> {
        ensure!(safe_name(&model.id), "Invalid model identifier");
        let path = self.root.join(format!("{}{}", model.id, suffix));
        match std::fs::symlink_metadata(&path) {
            Ok(metadata) => ensure!(
                metadata.is_file() && !metadata.file_type().is_symlink(),
                "Managed model path is not a regular owned file; it was preserved"
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        Ok(path)
    }

    pub fn model_path(&self, model: &ModelArtifact) -> Result<PathBuf> {
        self.path(model, ".gguf")
    }

    fn receipt(&self, model: &ModelArtifact) -> Result<PathBuf> {
        self.path(model, ".receipt.json")
    }

    fn operation_lock(&self, model: &ModelArtifact) -> Result<std::fs::File> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(self.path(model, ".lock")?)?;
        file.try_lock().context(
            "Another Copper operation is using this model. Wait for it to finish before retrying",
        )?;
        Ok(file)
    }

    fn check_receipt(&self, model: &ModelArtifact) -> Result<()> {
        let receipt: Receipt = serde_json::from_slice(&std::fs::read(self.receipt(model)?)?)?;
        ensure!(
            receipt.id == model.id
                && receipt.revision == model.revision
                && receipt.sha256 == model.sha256
                && receipt.bytes == model.bytes,
            "Artifact ownership does not match this catalog entry; files were preserved"
        );
        Ok(())
    }

    pub fn states(&self, models: &[ModelArtifact]) -> BTreeMap<String, ModelState> {
        models
            .iter()
            .map(|model| {
                let state = match self.model_path(model) {
                    Ok(path) if path.exists() => match self.check_receipt(model) {
                        Ok(()) => ModelState::Downloaded,
                        Err(error) => ModelState::Failed(error.to_string()),
                    },
                    Ok(_) => match self.path(model, ".part") {
                        Ok(path) if path.exists() => match self.check_receipt(model) {
                            Ok(()) => ModelState::Failed(
                                "Interrupted download retained. Choose Resume / retry to continue."
                                    .into(),
                            ),
                            Err(error) => ModelState::Failed(error.to_string()),
                        },
                        Ok(_) => ModelState::NotDownloaded,
                        Err(error) => ModelState::Failed(error.to_string()),
                    },
                    Err(error) => ModelState::Failed(error.to_string()),
                };
                (model.id.clone(), state)
            })
            .collect()
    }

    pub fn verify(&self, model: &ModelArtifact) -> Result<()> {
        self.check_receipt(model)?;
        verify_file(&self.model_path(model)?, model.bytes, &model.sha256, true)
    }

    pub fn remove(&self, model: &ModelArtifact) -> Result<()> {
        let _operation_lock = self.operation_lock(model)?;
        let artifact_paths = [self.model_path(model)?, self.path(model, ".part")?];
        let receipt = self.receipt(model)?;
        if !receipt.exists() && artifact_paths.iter().all(|path| !path.exists()) {
            return Ok(());
        }
        self.check_receipt(model)?;
        for path in artifact_paths {
            match std::fs::remove_file(path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        std::fs::remove_file(receipt)?;
        Ok(())
    }

    pub async fn load_model(
        &self,
        catalog: &Catalog,
        model: &ModelArtifact,
        http: &dyn HttpClient,
        local_http: &dyn HttpClient,
        cpu_only: bool,
        cancelled: Arc<AtomicBool>,
        timer: impl Fn(Duration) -> futures::future::BoxFuture<'static, ()>,
    ) -> Result<(ManagedRuntime, Hardware, Option<String>)> {
        ensure_host_runtime_compatibility()?;
        self.check_receipt(model)?;
        verify_file_cancellable(
            &self.model_path(model)?,
            model.bytes,
            &model.sha256,
            true,
            Some(&cancelled),
        )?;
        Hardware::detect(&self.root)?.ensure_memory(model)?;
        let mut warning = None;
        let attempts = if cpu_only {
            vec![true]
        } else {
            vec![false, true]
        };
        for cpu_only in attempts {
            ensure!(
                !cancelled.load(Ordering::Relaxed),
                "Model loading cancelled"
            );
            let attempt = async {
                let executable = self
                    .install_runtime_with_acceleration(catalog, http, !cpu_only)
                    .await?;
                let (devices, gpu_memory) = ManagedRuntime::probe_devices(&executable).await?;
                let mut hardware = Hardware::detect(&self.root)?;
                hardware.gpu_memory = gpu_memory;
                hardware.backend = format!(
                    "{}; {}",
                    if cpu_only {
                        "CPU requested; detected devices"
                    } else {
                        "Automatic backend; detected devices"
                    },
                    devices.trim()
                );
                let mut runtime = ManagedRuntime::launch(
                    &executable,
                    model,
                    &self.model_path(model)?,
                    &self.root,
                    cpu_only,
                )?;
                runtime.wait_ready(local_http, &timer).await?;
                let current_capacity = Hardware::detect(&self.root)?;
                hardware.available_memory = current_capacity.available_memory;
                hardware.free_disk = current_capacity.free_disk;
                Ok::<_, anyhow::Error>((runtime, hardware))
            }
            .await;
            match attempt {
                Ok((runtime, hardware)) => return Ok((runtime, hardware, warning)),
                Err(error) if !cpu_only => {
                    warning = Some(format!(
                        "Acceleration could not load this model; using the reviewed CPU fallback. Details: {error:#}"
                    ))
                }
                Err(error) => {
                    return Err(error
                        .context(warning.unwrap_or_else(|| "Local model could not load".into())));
                }
            }
        }
        bail!("No local inference backend was available")
    }

    pub async fn download(
        &self,
        model: &ModelArtifact,
        http: &dyn HttpClient,
        progress: Arc<AtomicU64>,
        cancelled: Arc<AtomicBool>,
    ) -> Result<()> {
        ensure_host_runtime_compatibility()?;
        let _operation_lock = self.operation_lock(model)?;
        let path = self.model_path(model)?;
        if path.exists() {
            self.check_receipt(model)?;
            verify_file_cancellable(&path, model.bytes, &model.sha256, true, Some(&cancelled))?;
            return Ok(());
        }
        let partial = self.path(model, ".part")?;
        if partial.exists() {
            self.check_receipt(model)?;
        }
        let mut offset = if partial.exists() {
            std::fs::metadata(&partial)?.len()
        } else {
            0
        };
        ensure!(
            offset <= model.bytes,
            "Partial download is larger than the expected artifact; remove it before retrying"
        );
        let hardware = Hardware::detect(&self.root)?;
        ensure!(
            hardware.free_disk >= model.bytes.saturating_sub(offset).saturating_add(GIB),
            "Not enough disk space. This download needs {:.1} GiB plus 1 GiB of free space.",
            gib(model.bytes.saturating_sub(offset))
        );
        let receipt = self.receipt(model)?;
        if receipt.exists() {
            self.check_receipt(model)?;
        } else {
            let content = serde_json::to_vec(&Receipt {
                id: model.id.clone(),
                revision: model.revision.clone(),
                sha256: model.sha256.clone(),
                bytes: model.bytes,
            })?;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&receipt)?;
            std::io::Write::write_all(&mut file, &content)?;
        }
        progress.store(offset, Ordering::Relaxed);
        if offset < model.bytes {
            let request = Request::get(model.url())
                .follow_redirects(RedirectPolicy::FollowAll)
                .timeout(Duration::from_secs(3600))
                .header("Range", format!("bytes={offset}-"))
                .body(AsyncBody::empty())?;
            let mut response = http
                .send(request)
                .await
                .context("Download could not connect; check the network and retry")?;
            let append = validate_range(
                response.status().as_u16(),
                response
                    .headers()
                    .get("Content-Range")
                    .and_then(|header| header.to_str().ok()),
                offset,
                model.bytes,
            )?;
            if !append {
                offset = 0;
                progress.store(0, Ordering::Relaxed);
            }
            let mut file = smol::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .append(append)
                .truncate(!append)
                .open(&partial)
                .await?;
            let mut buffer = vec![0; 256 * 1024];
            loop {
                ensure!(
                    !cancelled.load(Ordering::Relaxed),
                    "Download cancelled; choose Resume to continue"
                );
                let count = response
                    .body_mut()
                    .read(&mut buffer)
                    .await
                    .context("Download interrupted; choose Resume to continue")?;
                if count == 0 {
                    break;
                }
                offset = offset
                    .checked_add(count as u64)
                    .context("Download size overflow")?;
                ensure!(
                    offset <= model.bytes,
                    "Server sent more bytes than the pinned artifact size"
                );
                file.write_all(buffer.get(..count).context("Invalid download chunk")?)
                    .await?;
                progress.store(offset, Ordering::Relaxed);
            }
            file.flush().await?;
            file.sync_all().await?;
        }
        verify_file_cancellable(&partial, model.bytes, &model.sha256, true, Some(&cancelled)).context(
            "Downloaded model failed integrity verification. Remove the invalid download and retry",
        )?;
        ensure!(
            !cancelled.load(Ordering::Relaxed),
            "Download cancelled; verified bytes are preserved for retry"
        );
        smol::fs::rename(partial, path).await?;
        Ok(())
    }

    pub async fn install_runtime(
        &self,
        catalog: &Catalog,
        http: &dyn HttpClient,
    ) -> Result<PathBuf> {
        self.install_runtime_with_acceleration(catalog, http, false)
            .await
    }

    pub async fn install_runtime_with_acceleration(
        &self,
        catalog: &Catalog,
        http: &dyn HttpClient,
        accelerated: bool,
    ) -> Result<PathBuf> {
        ensure_host_runtime_compatibility()?;
        let platform = host_platform()?;
        let platform = if accelerated && !platform.starts_with("macos") {
            format!("{platform}-vulkan")
        } else {
            platform.to_owned()
        };
        let asset = catalog
            .runtime_assets
            .iter()
            .find(|asset| asset.platform == platform)
            .context("No reviewed runtime for this platform")?;
        if let Some(executable) = bundled_runtime(catalog, &platform, &std::env::current_exe()?)? {
            return Ok(executable);
        }
        let destination = self
            .root
            .join(format!("runtime-{}-{platform}", catalog.runtime_version));
        let receipt = destination.join(".copper-runtime.json");
        if destination.exists() {
            ensure!(
                !std::fs::symlink_metadata(&destination)?
                    .file_type()
                    .is_symlink(),
                "Runtime directory is not owned"
            );
            ensure!(
                std::fs::read_to_string(&receipt).ok().as_deref() == Some(asset.sha256.as_str()),
                "Existing runtime is not verified; its files were preserved"
            );
        } else {
            ensure!(
                Hardware::detect(&self.root)?.free_disk
                    >= asset.bytes.saturating_mul(5).saturating_add(GIB),
                "Not enough disk space to install the local runtime"
            );
            let kind = if asset.name.ends_with(".zip") {
                AssetKind::Zip
            } else {
                AssetKind::TarGz
            };
            http_client::github_download::download_server_binary(
                http,
                &asset.url,
                Some(&asset.sha256),
                &destination,
                kind,
            )
            .await
            .context("Runtime installation failed; retry after checking network and disk space")?;
            std::fs::write(receipt, &asset.sha256)?;
        }
        find_executable(&destination, 3)?
            .context("The verified archive did not contain llama-server")
    }
}

fn bundled_runtime(
    catalog: &Catalog,
    platform: &str,
    application: &Path,
) -> Result<Option<PathBuf>> {
    let parent = application
        .parent()
        .context("Application has no parent directory")?;
    let root = if cfg!(target_os = "macos") {
        parent
            .parent()
            .context("Application has no Contents directory")?
            .join("Resources/llama-cpp")
    } else {
        parent.join("llama-cpp")
    };
    if !root.exists() {
        return Ok(None);
    }
    let receipt: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join(".copper-bundled-runtime.json"))
            .context("Bundled local runtime is incomplete; reinstall Copper")?,
    )?;
    let asset = catalog
        .runtime_assets
        .iter()
        .find(|asset| asset.platform == platform)
        .context("Bundled runtime platform is not reviewed")?;
    ensure!(
        receipt["owner"] == "Copper bundled llama.cpp v1"
            && receipt["runtime_version"] == catalog.runtime_version
            && receipt["runtime_revision"] == catalog.runtime_revision
            && receipt["assets"][platform] == asset.sha256,
        "Bundled local runtime does not match the reviewed catalog; reinstall Copper"
    );
    let executable = find_executable(&root.join(platform), 3)?
        .context("Bundled local runtime is missing llama-server; reinstall Copper")?;
    Ok(Some(executable))
}

pub fn validate_range(
    status: u16,
    content_range: Option<&str>,
    offset: u64,
    total: u64,
) -> Result<bool> {
    if status == 401 || status == 403 {
        bail!(
            "Model access was denied. Review the publisher's license and access requirements before retrying"
        );
    }
    if status == 200 {
        return Ok(false);
    }
    ensure!(
        status == 206,
        "Download failed with HTTP {status}; retry when the service is available"
    );
    let range = content_range
        .and_then(|value| value.strip_prefix("bytes "))
        .context("Server omitted the download range; retry safely")?;
    let (bounds, actual_total) = range.split_once('/').context("Invalid download range")?;
    let (start, end) = bounds
        .split_once('-')
        .context("Invalid download range bounds")?;
    let start: u64 = start.parse()?;
    let end: u64 = end.parse()?;
    ensure!(
        start == offset && end >= start && end < total && actual_total.parse::<u64>()? == total,
        "Server returned a different artifact range; download was preserved without appending"
    );
    Ok(true)
}

fn verify_file(path: &Path, expected_bytes: u64, expected_sha256: &str, gguf: bool) -> Result<()> {
    verify_file_cancellable(path, expected_bytes, expected_sha256, gguf, None)
}

fn verify_file_cancellable(
    path: &Path,
    expected_bytes: u64,
    expected_sha256: &str,
    gguf: bool,
    cancelled: Option<&AtomicBool>,
) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "Artifact is not a regular file"
    );
    ensure!(
        metadata.len() == expected_bytes,
        "Artifact length mismatch: expected {expected_bytes} bytes, found {}",
        metadata.len()
    );
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 256 * 1024];
    let mut first = true;
    loop {
        ensure!(
            !cancelled.is_some_and(|cancelled| cancelled.load(Ordering::Relaxed)),
            "Integrity check cancelled; files were preserved"
        );
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        let chunk = buffer.get(..count).context("Invalid verification chunk")?;
        if first && gguf {
            ensure!(chunk.starts_with(b"GGUF"), "File is not a GGUF model");
        }
        first = false;
        hash.update(chunk);
    }
    ensure!(
        format!("{:x}", hash.finalize()) == expected_sha256,
        "Artifact SHA-256 mismatch"
    );
    Ok(())
}

fn find_executable(root: &Path, depth: usize) -> Result<Option<PathBuf>> {
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_symlink() {
            continue;
        }
        if kind.is_file()
            && entry.file_name()
                == if cfg!(windows) {
                    "llama-server.exe"
                } else {
                    "llama-server"
                }
        {
            return Ok(Some(entry.path()));
        }
        if kind.is_dir()
            && depth > 0
            && let Some(path) = find_executable(&entry.path(), depth - 1)?
        {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

pub fn host_platform() -> Result<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => Ok("macos-aarch64"),
        ("linux", "x86_64") => Ok("linux-x86_64"),
        ("windows", "x86_64") => Ok("windows-x86_64"),
        _ => bail!("Managed local inference is unavailable on this architecture"),
    }
}

pub struct ManagedRuntime {
    child: smol::process::Child,
    pub endpoint: String,
    pub api_key: Arc<str>,
    pub model_id: String,
    pub cpu_only: bool,
    template_sha256: String,
    context_tokens: u64,
}

impl ManagedRuntime {
    pub async fn probe_devices(executable: &Path) -> Result<(String, Option<u64>)> {
        let output = smol::process::Command::new(executable)
            .kill_on_drop(true)
            .arg("--list-devices")
            .output()
            .await?;
        ensure!(
            output.status.success(),
            "Runtime device discovery failed; use CPU fallback"
        );
        let description = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let memory = description
            .lines()
            .filter(|line| !line.contains("BLAS:"))
            .filter_map(|line| line.split_once("MiB, ").map(|(_, suffix)| suffix))
            .filter_map(|suffix| suffix.split_whitespace().next()?.parse::<u64>().ok())
            .max()
            .map(|mib| mib.saturating_mul(1024 * 1024));
        Ok((description, memory))
    }
    pub fn launch(
        executable: &Path,
        model: &ModelArtifact,
        model_path: &Path,
        root: &Path,
        cpu_only: bool,
    ) -> Result<Self> {
        ensure_host_runtime_compatibility()?;
        Hardware::detect(root)?.ensure_memory(model)?;
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))?;
        let port = listener.local_addr()?.port();
        drop(listener);
        let mut command = smol::process::Command::new(executable);
        let api_key: Arc<str> = uuid::Uuid::new_v4().to_string().into();
        command
            .kill_on_drop(true)
            .env_clear()
            .env("LLAMA_API_KEY", api_key.as_ref());
        for name in ["PATH", "SystemRoot", "TEMP", "TMP"] {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        command
            .args([
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "--model",
            ])
            .arg(model_path)
            .args([
                "--alias",
                &model.id,
                "--ctx-size",
                &model.context_tokens.to_string(),
                "--parallel",
                "1",
                "--jinja",
                "--no-webui",
                "--offline",
                "--no-agent",
                "--cors-origins",
                "localhost",
                "--no-cors-credentials",
                "--reasoning",
                "off",
                "--reasoning-budget",
                "0",
            ]);
        if cpu_only {
            command.args(["--n-gpu-layers", "0"]);
        }
        command.current_dir(
            executable
                .parent()
                .context("Runtime has no parent directory")?,
        );
        command.stdout(std::process::Stdio::null());
        let log_path = root.join("runtime.log");
        match std::fs::symlink_metadata(&log_path) {
            Ok(metadata) => ensure!(
                metadata.is_file() && !metadata.file_type().is_symlink(),
                "Runtime log is not an owned regular file; it was preserved"
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let stderr = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(log_path)?;
        command.stderr(stderr);
        let child = command
            .spawn()
            .context("Could not start the reviewed local runtime; check platform compatibility")?;
        Ok(Self {
            child,
            endpoint: format!("http://127.0.0.1:{port}"),
            api_key,
            model_id: model.id.clone(),
            cpu_only,
            // llama.cpp's Jinja lexer removes surrounding whitespace from its served template.
            template_sha256: model.runtime_template_sha256.clone(),
            context_tokens: model.context_tokens,
        })
    }

    pub fn is_running(&mut self) -> Result<bool> {
        Ok(self.child.try_status()?.is_none())
    }

    pub async fn wait_ready(
        &mut self,
        http: &dyn HttpClient,
        timer: impl Fn(Duration) -> futures::future::BoxFuture<'static, ()>,
    ) -> Result<()> {
        for _ in 0..240 {
            ensure!(
                self.is_running()?,
                "Local runtime exited while loading. Inspect runtime.log and try CPU fallback or a smaller model"
            );
            let probe = async {
                let response = match http
                    .get(
                        &format!("{}/health", self.endpoint),
                        AsyncBody::empty(),
                        false,
                    )
                    .await
                {
                    Ok(response) => response,
                    Err(_) => return Ok(false),
                };
                if !response.status().is_success() {
                    return Ok(false);
                }
                let models = crate::get_models(
                    http,
                    &self.endpoint,
                    Some(&self.api_key),
                    &Default::default(),
                )
                .await?;
                ensure!(
                    models.iter().any(|model| model.id == self.model_id),
                    "Runtime returned a different model"
                );
                let props = crate::get_props(
                    http,
                    &self.endpoint,
                    Some(&self.api_key),
                    None,
                    &Default::default(),
                )
                .await?;
                ensure!(
                    props.supports_tools(),
                    "This model's chat template does not advertise native tool support"
                );
                let template = props.chat_template.as_deref().context(
                    "Runtime did not report its chat template; readiness cannot be verified",
                )?;
                ensure!(
                    format!("{:x}", Sha256::digest(template.as_bytes())) == self.template_sha256,
                    "Runtime chat template differs from the reviewed GGUF artifact"
                );
                ensure!(
                    props.context_length() == Some(self.context_tokens),
                    "Runtime context differs from the reviewed memory estimate"
                );
                Ok::<_, anyhow::Error>(true)
            }
            .boxed();
            let deadline = timer(Duration::from_secs(2));
            match futures::future::select(probe, deadline).await {
                futures::future::Either::Left((Ok(true), _)) => return Ok(()),
                futures::future::Either::Left((Ok(false), _)) => {}
                futures::future::Either::Left((Err(error), _)) => return Err(error),
                futures::future::Either::Right(_) => {}
            }
            timer(Duration::from_millis(500)).await;
        }
        bail!(
            "Timed out waiting for local inference health and tool support. Check runtime.log, memory, and CPU fallback"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_runtime_requires_exact_catalog_receipt_and_preserves_missing_files() -> Result<()> {
        let temporary = tempfile::tempdir()?;
        let application = temporary.path().join("Contents/MacOS/copper");
        std::fs::create_dir_all(application.parent().context("fixture parent")?)?;
        let catalog = catalog()?;
        let platform = host_platform()?;
        assert!(bundled_runtime(&catalog, platform, &application)?.is_none());
        let root = if cfg!(target_os = "macos") {
            temporary.path().join("Contents/Resources/llama-cpp")
        } else {
            application
                .parent()
                .context("fixture parent")?
                .join("llama-cpp")
        };
        std::fs::create_dir_all(root.join(platform))?;
        assert!(bundled_runtime(&catalog, platform, &application).is_err());
        let asset = catalog
            .runtime_assets
            .iter()
            .find(|asset| asset.platform == platform)
            .context("fixture asset")?;
        let receipt = serde_json::json!({
            "owner": "Copper bundled llama.cpp v1",
            "runtime_version": catalog.runtime_version,
            "runtime_revision": catalog.runtime_revision,
            "assets": {platform: asset.sha256},
        });
        let receipt_path = root.join(".copper-bundled-runtime.json");
        std::fs::write(&receipt_path, serde_json::to_vec(&receipt)?)?;
        assert!(bundled_runtime(&catalog, platform, &application).is_err());
        let executable = root.join(platform).join(if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        });
        std::fs::write(&executable, b"fixture")?;
        assert_eq!(
            bundled_runtime(&catalog, platform, &application)?,
            Some(executable.clone())
        );
        let mut invalid = receipt;
        invalid["runtime_revision"] = "unreviewed".into();
        std::fs::write(&receipt_path, serde_json::to_vec(&invalid)?)?;
        assert!(bundled_runtime(&catalog, platform, &application).is_err());
        assert_eq!(std::fs::read(executable)?, b"fixture");
        Ok(())
    }

    #[test]
    fn managed_runtime_os_minimum_applies_to_cpu_and_accelerated_loading() -> Result<()> {
        assert!(require_runtime_version(Some("12.7"), "13.3", "macOS").is_err());
        assert!(require_runtime_version(Some("13.2.1"), "13.3", "macOS").is_err());
        assert!(require_runtime_version(None, "13.3", "macOS").is_err());
        assert!(require_runtime_version(Some("unknown"), "13.3", "macOS").is_err());
        require_runtime_version(Some("13.3"), "13.3", "macOS")?;
        require_runtime_version(Some("26.6.2"), "13.3", "macOS")?;
        let catalog = catalog()?;
        assert!(
            catalog
                .runtime_assets
                .iter()
                .filter(|asset| asset.platform == "macos-aarch64")
                .all(|asset| asset.minimum_macos_version.as_deref() == Some("13.3"))
        );
        Ok(())
    }

    #[test]
    fn managed_linux_glibc_minimum_is_pinned_for_cpu_and_vulkan() -> Result<()> {
        assert!(require_runtime_version(Some("2.31"), "2.34", "glibc").is_err());
        assert!(require_runtime_version(Some("2.33"), "2.34", "glibc").is_err());
        assert!(require_runtime_version(None, "2.34", "glibc").is_err());
        require_runtime_version(Some("2.34"), "2.34", "glibc")?;
        require_runtime_version(Some("2.35"), "2.34", "glibc")?;
        let catalog = catalog()?;
        assert!(
            catalog
                .runtime_assets
                .iter()
                .filter(|asset| asset.platform.starts_with("linux-x86_64"))
                .all(|asset| asset.minimum_glibc_version.as_deref() == Some("2.34"))
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires COPPER_CAPACITY_TEST_DIRECTORY on a disposable volume under 256 MiB"]
    fn capacity_probe_uses_the_actual_mounted_volume() -> Result<()> {
        let directory = std::env::var_os("COPPER_CAPACITY_TEST_DIRECTORY")
            .context("Set COPPER_CAPACITY_TEST_DIRECTORY to the disposable mounted volume")?;
        let hardware = Hardware::detect(Path::new(&directory))?;
        ensure!(
            hardware.free_disk < 256 * 1024 * 1024,
            "Capacity probe reported {} bytes; it may have selected the parent filesystem",
            hardware.free_disk
        );
        let model = catalog()?
            .models
            .into_iter()
            .next()
            .context("Missing model")?;
        assert!(
            hardware.recommendation(&model, true, false, true).level
                == RecommendationLevel::InsufficientCapacity
        );
        Ok(())
    }

    #[test]
    fn overlapping_model_operations_are_rejected_without_modifying_files() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = ManagedStore::open(directory.path().to_path_buf())?;
        let model = catalog()?
            .models
            .into_iter()
            .next()
            .context("Missing model")?;
        let guard = store.operation_lock(&model)?;
        assert!(store.operation_lock(&model).is_err());
        assert!(store.remove(&model).is_err());
        drop(guard);
        assert!(store.operation_lock(&model).is_ok());
        Ok(())
    }

    #[test]
    fn removing_a_failed_preflight_without_artifacts_preserves_other_files() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = ManagedStore::open(directory.path().to_path_buf())?;
        let model = catalog()?
            .models
            .into_iter()
            .next()
            .context("Missing model")?;
        let unrelated = directory.path().join("unrelated.gguf");
        std::fs::write(&unrelated, b"user file")?;
        store.remove(&model)?;
        store.remove(&model)?;
        assert_eq!(std::fs::read(&unrelated)?, b"user file");

        let artifact = store.model_path(&model)?;
        std::fs::write(&artifact, b"unowned model")?;
        assert!(store.remove(&model).is_err());
        assert_eq!(std::fs::read(&artifact)?, b"unowned model");
        assert_eq!(std::fs::read(unrelated)?, b"user file");
        Ok(())
    }

    fn tiny_model(bytes: &[u8]) -> Result<ModelArtifact> {
        let mut model = catalog()?
            .models
            .into_iter()
            .next()
            .context("Missing model")?;
        model.bytes = bytes.len() as u64;
        model.sha256 = format!("{:x}", Sha256::digest(bytes));
        Ok(model)
    }

    #[test]
    fn interrupted_download_resumes_exact_bytes_and_verifies_before_admission() -> Result<()> {
        smol::block_on(async {
            let bytes = b"GGUF complete fixture";
            let model = tiny_model(bytes)?;
            let directory = tempfile::tempdir()?;
            let store = ManagedStore::open(directory.path().to_path_buf())?;
            let progress = Arc::new(AtomicU64::new(0));
            let http = http_client::FakeHttpClient::create(move |request| async move {
                assert_eq!(
                    request
                        .headers()
                        .get("Range")
                        .and_then(|value| value.to_str().ok()),
                    Some("bytes=0-")
                );
                Ok(http_client::Response::builder()
                    .status(200)
                    .body(AsyncBody::from(
                        bytes.get(..7).context("Missing fixture prefix")?.to_vec(),
                    ))?)
            });
            assert!(
                store
                    .download(
                        &model,
                        http.as_ref(),
                        progress.clone(),
                        Arc::new(AtomicBool::new(false))
                    )
                    .await
                    .is_err()
            );
            assert!(!store.model_path(&model)?.exists());
            assert_eq!(progress.load(Ordering::Relaxed), 7);
            let http = http_client::FakeHttpClient::create(move |request| async move {
                assert_eq!(
                    request
                        .headers()
                        .get("Range")
                        .and_then(|value| value.to_str().ok()),
                    Some("bytes=7-")
                );
                Ok(http_client::Response::builder()
                    .status(206)
                    .header(
                        "Content-Range",
                        format!("bytes 7-{}/{}", bytes.len() - 1, bytes.len()),
                    )
                    .body(AsyncBody::from(
                        bytes.get(7..).context("Missing fixture suffix")?.to_vec(),
                    ))?)
            });
            store
                .download(
                    &model,
                    http.as_ref(),
                    progress,
                    Arc::new(AtomicBool::new(false)),
                )
                .await?;
            store.verify(&model)?;
            assert_eq!(
                store.states(std::slice::from_ref(&model)).get(&model.id),
                Some(&ModelState::Downloaded)
            );
            std::fs::write(directory.path().join("unrelated.gguf"), b"user file")?;
            store.remove(&model)?;
            assert_eq!(
                std::fs::read(directory.path().join("unrelated.gguf"))?,
                b"user file"
            );
            Ok(())
        })
    }

    #[test]
    fn cancellation_preserves_owned_partial_and_retry_can_restart_when_range_is_ignored()
    -> Result<()> {
        smol::block_on(async {
            let bytes = b"GGUF complete fixture";
            let model = tiny_model(bytes)?;
            let directory = tempfile::tempdir()?;
            let store = ManagedStore::open(directory.path().to_path_buf())?;
            let http = http_client::FakeHttpClient::create(move |_| async move {
                Ok(http_client::Response::builder()
                    .status(200)
                    .body(AsyncBody::from(bytes.to_vec()))?)
            });
            assert!(
                store
                    .download(
                        &model,
                        http.as_ref(),
                        Arc::new(AtomicU64::new(0)),
                        Arc::new(AtomicBool::new(true))
                    )
                    .await
                    .is_err()
            );
            assert!(!store.model_path(&model)?.exists());
            std::fs::write(store.path(&model, ".part")?, b"GGUF old")?;
            store
                .download(
                    &model,
                    http.as_ref(),
                    Arc::new(AtomicU64::new(0)),
                    Arc::new(AtomicBool::new(false)),
                )
                .await?;
            store.verify(&model)?;
            Ok(())
        })
    }

    #[test]
    fn catalog_pins_real_artifacts_and_does_not_claim_qualification() -> Result<()> {
        let catalog = catalog()?;
        assert_eq!(catalog.models.len(), 9);
        assert!(
            catalog
                .models
                .iter()
                .all(|model| !model.qualified && model.estimated_memory_bytes() > model.bytes)
        );
        assert!(
            catalog
                .runtime_assets
                .iter()
                .all(|asset| asset.sha256.len() == 64
                    && asset.url.contains(&catalog.runtime_version))
        );
        Ok(())
    }

    #[test]
    fn qwen_blend_variants_share_kv_cache_size() -> Result<()> {
        let catalog = catalog()?;
        let variant = |quantization| {
            catalog
                .models
                .iter()
                .find(|model| {
                    model.repository == "JetBrains/Qwen3.8-3.6-27B-blend-GGUF"
                        && model.quantization == quantization
                })
                .context("Missing Qwen blend variant")
        };
        let primary = variant("Q4_K_M")?;
        let smaller = variant("IQ3_S")?;
        assert_eq!(primary.context_tokens, 16_384);
        assert_eq!(smaller.context_tokens, primary.context_tokens);
        assert_eq!(primary.kv_cache_bytes(), GIB);
        assert_eq!(smaller.kv_cache_bytes(), GIB);
        assert_eq!(primary.chat_template_sha256, smaller.chat_template_sha256);
        assert!(primary.attribution.is_some() && smaller.attribution.is_some());
        assert!(smaller.bytes < primary.bytes);
        assert!(smaller.estimated_memory_bytes() < primary.estimated_memory_bytes());
        Ok(())
    }

    #[test]
    fn unsloth_qwen_variants_use_distinct_model_ids_and_a_16k_context() -> Result<()> {
        let catalog = catalog()?;
        let variant = |id| {
            catalog
                .models
                .iter()
                .find(|model| model.id == id)
                .context("Missing Unsloth Qwen variant")
        };
        let primary = variant("unsloth-qwen3-8-27b-ud-q4-k-m")?;
        let smaller = variant("unsloth-qwen3-8-27b-ud-iq3-s")?;
        for model in [primary, smaller] {
            assert!(model.name.starts_with("Unsloth Qwen3.8-27B"));
            assert_eq!(model.repository, "unsloth/Qwen3.8-27B-GGUF");
            assert_eq!(model.context_tokens, 16_384);
            assert_eq!(model.kv_cache_bytes(), GIB);
        }
        assert!(smaller.estimated_memory_bytes() < primary.estimated_memory_bytes());
        Ok(())
    }

    #[test]
    fn memory_estimates_include_context_and_runtime_overhead() -> Result<()> {
        let mut model = catalog()?
            .models
            .into_iter()
            .next()
            .context("Missing model")?;
        let original = model.estimated_memory_bytes();
        model.context_tokens *= 2;
        assert!(model.estimated_memory_bytes() > original);
        let hardware = Hardware {
            architecture: "aarch64".into(),
            total_memory: GIB,
            available_memory: GIB,
            free_disk: 0,
            backend: "CPU".into(),
            unified_memory: false,
            gpu_memory: None,
            runtime_warning: None,
        };
        assert!(hardware.ensure_memory(&model).is_err());
        assert!(
            hardware
                .recommendation(&model, true, false, true)
                .reason
                .contains("disk")
        );
        Ok(())
    }

    #[test]
    fn recommendations_preserve_memory_formula_and_resident_capacity() -> Result<()> {
        let model = catalog()?
            .models
            .into_iter()
            .next()
            .context("Catalog model")?;
        let cache =
            model.context_tokens * model.kv_layers * model.kv_heads * model.kv_head_dimension * 4;
        let runtime = (model.bytes / 4).max(GIB);
        assert_eq!(
            model.estimated_memory_bytes(),
            model.bytes + cache + runtime
        );
        let required = model.estimated_memory_bytes() + GIB;
        let mut hardware = Hardware {
            architecture: "aarch64".into(),
            total_memory: 32 * GIB,
            available_memory: required + 4 * GIB,
            free_disk: 0,
            backend: "Automatic; MTL0: Apple M5 (24000 MiB, 23000 MiB free)\nBLAS: Accelerate (0 MiB, 0 MiB free)".into(),
            unified_memory: true,
            gpu_memory: Some(23 * GIB),
            runtime_warning: None,
        };
        assert_eq!(
            hardware.recommendation(&model, true, false, false).level,
            RecommendationLevel::InsufficientCapacity
        );
        assert_eq!(
            hardware.recommendation(&model, false, false, false).level,
            RecommendationLevel::Recommended,
            "Downloaded weights do not need duplicate disk allocation"
        );
        hardware.available_memory = required + GIB / 2;
        assert_eq!(
            hardware.recommendation(&model, false, false, false).level,
            RecommendationLevel::LimitedHeadroom
        );
        hardware.available_memory = required - 1;
        assert_eq!(
            hardware.recommendation(&model, false, false, false).level,
            RecommendationLevel::InsufficientCapacity
        );
        hardware.available_memory = 4 * GIB;
        assert_eq!(
            hardware.recommendation(&model, false, true, false).level,
            RecommendationLevel::Recommended,
            "Resident model memory is not charged twice"
        );
        hardware.available_memory = GIB / 2;
        assert_eq!(
            hardware.recommendation(&model, false, true, false).level,
            RecommendationLevel::LimitedHeadroom
        );
        assert!(
            hardware
                .device_details()
                .iter()
                .all(|line| !line.contains("0 MiB"))
        );
        assert!(
            hardware
                .device_details()
                .iter()
                .any(|line| line.contains("Accelerate optimizes CPU math"))
        );
        hardware.available_memory = GIB;
        hardware.gpu_memory = Some(64 * GIB);
        assert_eq!(
            hardware.recommendation(&model, false, false, false).level,
            RecommendationLevel::InsufficientCapacity,
            "Shared GPU capacity cannot be added to available RAM"
        );
        Ok(())
    }

    #[test]
    fn cancelled_integrity_check_preserves_downloaded_bytes() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("model.part");
        let bytes = b"GGUFtest artifact";
        std::fs::write(&path, bytes)?;
        let digest = format!("{:x}", Sha256::digest(bytes));
        let cancelled = AtomicBool::new(true);
        let error =
            verify_file_cancellable(&path, bytes.len() as u64, &digest, true, Some(&cancelled))
                .err()
                .context("Cancellation must stop verification")?;
        assert!(error.to_string().contains("cancelled"));
        assert_eq!(std::fs::read(&path)?, bytes);
        verify_file(&path, bytes.len() as u64, &digest, true)?;
        Ok(())
    }

    #[test]
    fn resume_requires_exact_range_and_artifact_length() {
        assert!(validate_range(206, Some("bytes 4-9/10"), 4, 10).is_ok());
        assert!(validate_range(206, Some("bytes 0-9/10"), 4, 10).is_err());
        assert!(validate_range(206, Some("bytes 4-9/11"), 4, 10).is_err());
        assert!(validate_range(206, None, 4, 10).is_err());
        assert!(matches!(validate_range(200, None, 4, 10), Ok(false)));
        assert!(validate_range(403, None, 0, 10).is_err());
    }

    #[test]
    fn existing_unowned_directory_and_files_are_preserved() -> Result<()> {
        let directory = tempfile::tempdir()?;
        std::fs::write(directory.path().join("user.gguf"), b"user data")?;
        assert!(ManagedStore::open(directory.path().to_path_buf()).is_err());
        assert_eq!(
            std::fs::read(directory.path().join("user.gguf"))?,
            b"user data"
        );
        Ok(())
    }

    #[test]
    fn files_are_downloaded_not_ready_and_unowned_removal_is_rejected() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = ManagedStore::open(directory.path().to_path_buf())?;
        let model = catalog()?
            .models
            .into_iter()
            .next()
            .context("Missing model")?;
        std::fs::write(store.model_path(&model)?, b"GGUF invalid")?;
        assert!(store.verify(&model).is_err());
        assert!(store.remove(&model).is_err());
        assert!(matches!(
            store.states(std::slice::from_ref(&model)).get(&model.id),
            Some(ModelState::Failed(_))
        ));
        Ok(())
    }

    #[test]
    fn integrity_checks_reject_wrong_length_header_and_digest() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("test.gguf");
        let bytes = b"GGUF fixture";
        std::fs::write(&path, bytes)?;
        let hash = format!("{:x}", Sha256::digest(bytes));
        verify_file(&path, bytes.len() as u64, &hash, true)?;
        assert!(verify_file(&path, bytes.len() as u64 + 1, &hash, true).is_err());
        assert!(verify_file(&path, bytes.len() as u64, "incorrect", true).is_err());
        std::fs::write(&path, b"invalid data")?;
        assert!(verify_file(&path, 12, &hash, true).is_err());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_never_admit_or_delete_external_models() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = ManagedStore::open(directory.path().join("managed"))?;
        let model = catalog()?
            .models
            .into_iter()
            .next()
            .context("Missing model")?;
        let target = directory.path().join("user.gguf");
        std::fs::write(&target, b"user data")?;
        std::os::unix::fs::symlink(&target, store.model_path(&model)?)?;
        assert!(store.model_path(&model).is_err());
        assert!(store.remove(&model).is_err());
        assert_eq!(std::fs::read(target)?, b"user data");
        Ok(())
    }
}
