use crate::doctor::{
    CheckOutcome, DiagnosticText, DoctorCheck, DoctorHistoryStore, DoctorProduct,
    DoctorRepairOperationOutcome, DoctorRepairPhase, DoctorRepairReview, DoctorReport, ProbeEnd,
    project_repair_binding, run_toolchain_command,
};
use anyhow::{Context as _, Result, ensure};
use gpui::{App, Entity, Task};
use project::{Project, ProjectPath};
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RustupInstallRequest {
    Toolchain { version: String },
    Component { version: String, component: String },
    Target { version: String, target: String },
}

fn dated_channel(version: &str) -> Option<(&str, &str)> {
    let (channel, date) = version.split_once('-')?;
    if !matches!(channel, "nightly" | "beta") {
        return None;
    }
    let parts = date.split('-').collect::<Vec<_>>();
    if parts.len() != 3
        || parts.iter().zip([4, 2, 2]).any(|(part, length)| {
            part.len() != length || !part.bytes().all(|byte| byte.is_ascii_digit())
        })
    {
        return None;
    }
    let year = parts.first()?.parse::<u32>().ok()?;
    let month = parts.get(1)?.parse::<u32>().ok()?;
    let day = parts.get(2)?.parse::<u32>().ok()?;
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return None,
    };
    (year >= 2015 && day >= 1 && day <= days).then_some((channel, date))
}

impl RustupInstallRequest {
    fn version(&self) -> &str {
        match self {
            Self::Toolchain { version }
            | Self::Component { version, .. }
            | Self::Target { version, .. } => version,
        }
    }

    fn validate(&self) -> Result<()> {
        let version = self.version();
        let exact_release = version.split('.').count() == 3
            && version.split('.').all(|part| {
                !part.is_empty()
                    && part.bytes().all(|byte| byte.is_ascii_digit())
                    && part.parse::<u32>().is_ok()
            });
        ensure!(
            exact_release || dated_channel(version).is_some(),
            "Select an exact released Rust version or a calendar-valid dated nightly/beta; moving channels and custom toolchains are unsupported"
        );
        match self {
            Self::Component { component, .. } => ensure!(
                matches!(
                    component.as_str(),
                    "rustfmt" | "clippy" | "rust-src" | "rust-analyzer"
                ),
                "Select a supported named Rust component"
            ),
            Self::Target { target, .. } => ensure!(
                target.len() <= 100
                    && target.split('-').count() >= 3
                    && target
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
                    && !target.starts_with('-'),
                "Select one exact Rust target; all-target and command arguments are not allowed"
            ),
            Self::Toolchain { .. } => {}
        }
        Ok(())
    }

    fn arguments(&self, toolchain: &str) -> Vec<String> {
        match self {
            Self::Toolchain { .. } => vec![
                "toolchain",
                "install",
                toolchain,
                "--no-self-update",
                "--no-update",
                "--profile",
                "minimal",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            Self::Component { component, .. } => vec![
                "component".into(),
                "add".into(),
                component.clone(),
                "--toolchain".into(),
                toolchain.into(),
            ],
            Self::Target { target, .. } => vec![
                "target".into(),
                "add".into(),
                target.clone(),
                "--toolchain".into(),
                toolchain.into(),
            ],
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
struct FileIdentity {
    path: PathBuf,
    inode: u64,
    modified: fs::MTime,
    length: u64,
}

async fn confined_file(
    filesystem: &dyn fs::Fs,
    root: &Path,
    path: &Path,
    maximum: u64,
) -> Result<(FileIdentity, Vec<u8>)> {
    let canonical = filesystem.canonicalize(path).await?;
    ensure!(
        canonical.starts_with(root),
        "Manager file escapes the selected user scope"
    );
    let before = filesystem
        .metadata(&canonical)
        .await?
        .context("Manager file is unavailable")?;
    ensure!(
        !before.is_dir && !before.is_fifo && before.len <= maximum,
        "Manager file is not a bounded regular file"
    );
    let mut input = filesystem.open_sync(&canonical).await?;
    let mut bytes = Vec::new();
    std::io::Read::take(&mut input, maximum + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= maximum,
        "Manager file exceeds its bound"
    );
    let after = filesystem
        .metadata(&canonical)
        .await?
        .context("Manager file disappeared")?;
    ensure!(
        before.inode == after.inode
            && before.mtime == after.mtime
            && before.len == after.len
            && filesystem.canonicalize(path).await? == canonical,
        "Manager file changed while inspected"
    );
    Ok((
        FileIdentity {
            path: canonical,
            inode: before.inode,
            modified: before.mtime,
            length: before.len,
        },
        bytes,
    ))
}

#[derive(Clone, PartialEq, Eq)]
struct RustupSnapshot {
    user_home: PathBuf,
    cargo_home: PathBuf,
    rustup_home: PathBuf,
    executable: FileIdentity,
    settings_identity: FileIdentity,
    settings: Vec<u8>,
    home_inodes: [u64; 3],
}

impl RustupSnapshot {
    async fn inspect(
        filesystem: &dyn fs::Fs,
        user_home: &Path,
        executable: &Path,
        roots: &[PathBuf],
    ) -> Result<Self> {
        let mut canonical_roots = Vec::new();
        for root in roots {
            canonical_roots.push(filesystem.canonicalize(root).await?);
        }
        let roots = &canonical_roots;
        let user_home = filesystem.canonicalize(user_home).await?;
        let cargo_home = filesystem.canonicalize(&user_home.join(".cargo")).await?;
        let rustup_home = filesystem.canonicalize(&user_home.join(".rustup")).await?;
        ensure!(
            cargo_home.starts_with(&user_home) && rustup_home.starts_with(&user_home),
            "Toolchain homes escape the user home"
        );
        let mut home_inodes = [0; 3];
        for (index, home) in [&user_home, &cargo_home, &rustup_home]
            .into_iter()
            .enumerate()
        {
            let metadata = filesystem
                .metadata(home)
                .await?
                .context("Existing user manager directory is missing")?;
            ensure!(
                metadata.is_dir && !roots.iter().any(|root| home.starts_with(root)),
                "Manager directories must be outside the project"
            );
            home_inodes[index] = metadata.inode;
        }
        ensure!(
            filesystem
                .metadata(&cargo_home.join("bin/rustup-init"))
                .await?
                .is_none()
                && filesystem
                    .metadata(&cargo_home.join("bin/rustup-init.exe"))
                    .await?
                    .is_none(),
            "A leftover Rustup updater would be removed on startup; resolve it through the existing manager before Doctor runs it"
        );
        ensure!(
            executable.is_absolute(),
            "Select an absolute Rustup executable"
        );
        let canonical_executable = filesystem.canonicalize(executable).await?;
        ensure!(
            canonical_executable
                == cargo_home.join(if cfg!(windows) {
                    "bin/rustup.exe"
                } else {
                    "bin/rustup"
                }),
            "Select the already installed user-scoped Rustup binary"
        );
        let metadata = filesystem
            .metadata(&canonical_executable)
            .await?
            .context("Rustup is unavailable")?;
        ensure!(
            metadata.is_executable && !metadata.is_dir && !metadata.is_fifo,
            "Rustup is not an executable regular file"
        );
        let mut header = [0; 4];
        std::io::Read::read_exact(
            &mut filesystem.open_sync(&canonical_executable).await?,
            &mut header,
        )?;
        ensure!(
            header.starts_with(b"\x7fELF")
                || header.starts_with(b"MZ")
                || matches!(
                    header,
                    [0xcf, 0xfa, 0xed, 0xfe]
                        | [0xfe, 0xed, 0xfa, 0xcf]
                        | [0xca, 0xfe, 0xba, 0xbe]
                        | [0xbe, 0xba, 0xfe, 0xca]
                ),
            "Rustup scripts, shims and unknown binary formats are unsupported"
        );
        let (settings_identity, settings) = confined_file(
            filesystem,
            &rustup_home,
            &rustup_home.join("settings.toml"),
            16 * 1024,
        )
        .await?;
        let configuration: toml::Value = toml::from_str(
            std::str::from_utf8(&settings)
                .map_err(|_| anyhow::anyhow!("Rustup settings encoding is invalid"))?,
        )
        .map_err(|_| anyhow::anyhow!("Rustup settings cannot be parsed"))?;
        ensure!(
            configuration.get("version").and_then(toml::Value::as_str) == Some("12"),
            "Unsupported Rustup settings version"
        );
        ensure!(
            configuration
                .get("default_toolchain")
                .and_then(toml::Value::as_str)
                .is_some_and(|value| !value.is_empty()),
            "Rustup would select a first default implicitly; configure it separately through the existing manager"
        );
        Ok(Self {
            user_home,
            cargo_home,
            rustup_home,
            executable: FileIdentity {
                path: canonical_executable,
                inode: metadata.inode,
                modified: metadata.mtime,
                length: metadata.len,
            },
            settings_identity,
            settings,
            home_inodes,
        })
    }

    async fn revalidate(&self, filesystem: &dyn fs::Fs, roots: &[PathBuf]) -> Result<()> {
        ensure!(
            &Self::inspect(filesystem, &self.user_home, &self.executable.path, roots).await?
                == self,
            "Reviewed manager executable, homes or settings changed"
        );
        Ok(())
    }

    fn environment(&self) -> Result<Vec<(String, String)>> {
        Ok(vec![
            (
                "CARGO_HOME".into(),
                self.cargo_home
                    .to_str()
                    .context("Unsupported Cargo home encoding")?
                    .into(),
            ),
            (
                "RUSTUP_HOME".into(),
                self.rustup_home
                    .to_str()
                    .context("Unsupported Rustup home encoding")?
                    .into(),
            ),
            ("RUSTUP_AUTO_INSTALL".into(), "0".into()),
        ])
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RustupStage {
    Inspect,
    Install,
}

#[derive(Clone, PartialEq, Eq)]
struct RustupInventory {
    components: (FileIdentity, Vec<u8>),
    manifest: (FileIdentity, Vec<u8>),
}

pub struct RustupAction {
    snapshot: RustupSnapshot,
    request: RustupInstallRequest,
    toolchain: String,
    roots: Vec<PathBuf>,
    stage: RustupStage,
    reviewed_inventory: Option<Option<RustupInventory>>,
}

impl RustupAction {
    pub fn executable(&self) -> &Path {
        &self.snapshot.executable.path
    }
    pub fn destination(&self) -> PathBuf {
        self.snapshot
            .rustup_home
            .join("toolchains")
            .join(&self.toolchain)
    }
    pub fn request(&self) -> &RustupInstallRequest {
        &self.request
    }
    pub fn command_arguments(&self) -> Vec<String> {
        if self.stage == RustupStage::Inspect {
            vec!["--version".into()]
        } else {
            self.request.arguments(&self.toolchain)
        }
    }
    pub fn source(&self) -> &'static str {
        "https://static.rust-lang.org/dist/"
    }
    pub fn is_inspection(&self) -> bool {
        self.stage == RustupStage::Inspect
    }

    async fn compiler_version(&self, filesystem: &dyn fs::Fs) -> Result<String> {
        let (_, bytes) = confined_file(
            filesystem,
            &self.snapshot.rustup_home,
            &self
                .destination()
                .join("lib/rustlib/multirust-channel-manifest.toml"),
            2 * 1024 * 1024,
        )
        .await?;
        let manifest: toml::Value = toml::from_str(
            std::str::from_utf8(&bytes)
                .map_err(|_| anyhow::anyhow!("Installed manifest encoding is invalid"))?,
        )
        .map_err(|_| anyhow::anyhow!("Installed toolchain manifest cannot be parsed"))?;
        ensure!(
            manifest
                .get("manifest-version")
                .and_then(toml::Value::as_str)
                == Some("2"),
            "Unsupported installed channel manifest"
        );
        let compiler = manifest
            .get("pkg")
            .and_then(|packages| packages.get("rustc"))
            .and_then(|compiler| compiler.get("version"))
            .and_then(toml::Value::as_str)
            .context("Installed compiler version is absent from the channel manifest")?;
        ensure!(
            compiler.len() <= 160
                && compiler.bytes().all(|byte| byte.is_ascii_alphanumeric()
                    || matches!(byte, b'.' | b'-' | b' ' | b'(' | b')')),
            "Invalid installed compiler identity"
        );
        if let Some((channel, date)) = dated_channel(self.request.version()) {
            ensure!(
                manifest.get("date").and_then(toml::Value::as_str) == Some(date)
                    && compiler
                        .split_whitespace()
                        .next()
                        .is_some_and(|version| version.ends_with(&format!("-{channel}"))),
                "Installed channel date or compiler kind differs from the reviewed pin"
            );
        } else {
            ensure!(
                compiler.split_whitespace().next() == Some(self.request.version()),
                "Installed compiler release differs from the reviewed pin"
            );
        }
        Ok(compiler.into())
    }

    async fn inventory_state(&self, filesystem: &dyn fs::Fs) -> Result<Option<RustupInventory>> {
        let directory = self.destination();
        if filesystem.metadata(&directory).await?.is_none() {
            return Ok(None);
        }
        ensure!(
            filesystem
                .canonicalize(&directory)
                .await?
                .starts_with(&self.snapshot.rustup_home),
            "Toolchain escapes the selected manager home"
        );
        Ok(Some(RustupInventory {
            components: confined_file(
                filesystem,
                &self.snapshot.rustup_home,
                &directory.join("lib/rustlib/components"),
                64 * 1024,
            )
            .await?,
            manifest: confined_file(
                filesystem,
                &self.snapshot.rustup_home,
                &directory.join("lib/rustlib/multirust-channel-manifest.toml"),
                2 * 1024 * 1024,
            )
            .await?,
        }))
    }

    async fn inventory(&self, filesystem: &dyn fs::Fs, observed_at: u64) -> Result<DoctorCheck> {
        let directory = self.destination();
        let metadata = filesystem.metadata(&directory).await?;
        let available = if metadata.is_none() {
            false
        } else {
            ensure!(
                filesystem
                    .canonicalize(&directory)
                    .await?
                    .starts_with(&self.snapshot.rustup_home),
                "Installed toolchain escapes the reviewed manager home"
            );
            let (_, components) = confined_file(
                filesystem,
                &self.snapshot.rustup_home,
                &directory.join("lib/rustlib/components"),
                64 * 1024,
            )
            .await?;
            let components = std::str::from_utf8(&components)?;
            let wanted = match &self.request {
                RustupInstallRequest::Toolchain { .. } => "rustc".into(),
                RustupInstallRequest::Component { component, .. } => component.clone(),
                RustupInstallRequest::Target { target, .. } => format!("rust-std-{target}"),
            };
            let host = self
                .toolchain
                .strip_prefix(&format!("{}-", self.request.version()))
                .context("Invalid reviewed toolchain")?;
            components.lines().any(|component| {
                component == wanted
                    || component == format!("{wanted}-{host}")
                    || component == format!("{wanted}-preview-{host}")
            })
        };
        if available {
            self.compiler_version(filesystem).await?;
            use crate::doctor::{DependencyProbe, ProbeExecutable};
            let native = match &self.request {
                RustupInstallRequest::Toolchain { .. } => {
                    Some((DependencyProbe::RustCompiler, "rustc"))
                }
                RustupInstallRequest::Component { component, .. } => match component.as_str() {
                    "rustfmt" => Some((DependencyProbe::RustFormatter, "rustfmt")),
                    "clippy" => Some((DependencyProbe::ClippyDriver, "clippy-driver")),
                    "rust-analyzer" => Some((DependencyProbe::RustAnalyzer, "rust-analyzer")),
                    "rust-src" => {
                        let path = directory.join("lib/rustlib/src/rust/library/core/src/lib.rs");
                        let canonical = filesystem.canonicalize(&path).await?;
                        ensure!(
                            canonical.starts_with(&self.snapshot.rustup_home),
                            "Installed Rust sources escape the manager home"
                        );
                        let metadata = filesystem
                            .metadata(&canonical)
                            .await?
                            .context("Installed Rust source is absent")?;
                        ensure!(
                            !metadata.is_dir && !metadata.is_fifo,
                            "Installed Rust source is not a regular file"
                        );
                        None
                    }
                    _ => anyhow::bail!("Unsupported Rust component"),
                },
                RustupInstallRequest::Target { target, .. } => {
                    use futures::StreamExt as _;
                    let path = directory.join("lib/rustlib").join(target).join("lib");
                    let canonical = filesystem.canonicalize(&path).await?;
                    ensure!(
                        canonical.starts_with(&self.snapshot.rustup_home),
                        "Installed target libraries escape the manager home"
                    );
                    let mut files = filesystem.read_dir(&canonical).await?;
                    let mut found = false;
                    for _ in 0..256 {
                        let Some(path) = files.next().await else {
                            break;
                        };
                        let path = path?;
                        if path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .is_some_and(|name| {
                                name.starts_with("libstd-") && name.ends_with(".rlib")
                            })
                        {
                            let canonical = filesystem.canonicalize(&path).await?;
                            ensure!(
                                canonical.starts_with(&self.snapshot.rustup_home),
                                "Target library escapes the manager home"
                            );
                            let metadata = filesystem
                                .metadata(&canonical)
                                .await?
                                .context("Target library disappeared")?;
                            ensure!(
                                !metadata.is_dir && !metadata.is_fifo,
                                "Target library is not a regular file"
                            );
                            found = true;
                            break;
                        }
                    }
                    ensure!(
                        found,
                        "Selected target standard library was not observed in the bounded inventory"
                    );
                    None
                }
            };
            if let Some((probe, name)) = native {
                let executable = directory.join("bin").join(if cfg!(windows) {
                    format!("{name}.exe")
                } else {
                    name.into()
                });
                ProbeExecutable::inspect(probe, executable, &self.roots, filesystem).await?;
            }
        }
        Ok(DoctorCheck::new(
            "rust.install",
            "Selected Rust installation inventory",
            if available {
                CheckOutcome::Ready
            } else {
                CheckOutcome::Blocked
            },
            &format!(
                "Exact toolchain {}: requested installation recorded {}. Compiler execution and relevant project validation remain separate observations.",
                self.toolchain, available
            ),
            observed_at,
        ))
    }
}

fn local_host_triple() -> Result<&'static str> {
    match (std::env::consts::ARCH, std::env::consts::OS) {
        ("aarch64", "macos") => Ok("aarch64-apple-darwin"),
        ("x86_64", "macos") => Ok("x86_64-apple-darwin"),
        ("aarch64", "linux") if cfg!(target_env = "gnu") => Ok("aarch64-unknown-linux-gnu"),
        ("x86_64", "linux") if cfg!(target_env = "gnu") => Ok("x86_64-unknown-linux-gnu"),
        ("x86_64", "windows") => Ok("x86_64-pc-windows-msvc"),
        _ => anyhow::bail!("Native Rustup host adapter is unavailable on this platform"),
    }
}

pub fn prepare_rustup_inspection(
    project: &Entity<Project>,
    directory: ProjectPath,
    executable: PathBuf,
    request: RustupInstallRequest,
    generation: u64,
    observed_at: u64,
    cx: &App,
) -> Task<Result<DoctorRepairReview<RustupAction>>> {
    let preparation = (|| {
        request.validate()?;
        let binding = project_repair_binding(project, directory.clone(), generation, cx)?;
        ensure!(
            project
                .read(cx)
                .doctor_observation(directory.worktree_id, cx)
                .host
                == project::DoctorExecutionHost::Local,
            "Remote manager identity, user-home containment and cancellation are unavailable through this adapter"
        );
        for name in [
            "RUSTUP_DIST_SERVER",
            "RUSTUP_UPDATE_ROOT",
            "RUSTUP_VERSION",
            "RUSTUP_OVERRIDE_HOST_TRIPLE",
        ] {
            ensure!(
                std::env::var_os(name).is_none(),
                "Custom Rustup source or manager override is configured; this adapter cannot authorize it"
            );
        }
        let user_home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
            .context("User home is unavailable")?;
        let roots = project
            .read(cx)
            .worktrees(cx)
            .map(|tree| tree.read(cx).abs_path().to_path_buf())
            .collect::<Vec<_>>();
        Ok((
            binding,
            PathBuf::from(user_home),
            roots,
            project.read(cx).fs().clone(),
        ))
    })();
    let (binding, user_home, roots, filesystem) = match preparation {
        Ok(value) => value,
        Err(error) => return Task::ready(Err(error)),
    };
    let toolchain = match local_host_triple() {
        Ok(host) => format!("{}-{host}", request.version()),
        Err(error) => return Task::ready(Err(error)),
    };
    cx.background_executor().spawn(async move {
        let snapshot = RustupSnapshot::inspect(filesystem.as_ref(), &user_home, &executable, &roots).await?;
        let action = RustupAction { snapshot, request, toolchain, roots, stage: RustupStage::Inspect, reviewed_inventory: None };
        let before = DoctorCheck::new("rust.manager.inspect", "Selected Rustup manager version", CheckOutcome::Untested, "The selected native manager has not been invoked; this review only permits a bounded version inspection", observed_at);
        DoctorRepairReview::new(action, binding, before, DiagnosticText::new("Run only the selected user-scoped Rustup --version with a clean environment, no shell/project hooks, no network and no automatic toolchain install. A leftover updater or missing default blocks launch. Only supported Rustup 1.29.0 can prepare a separate installation proposal; this approval cannot install anything."))
    })
}

pub struct RustupOperationResult {
    pub review: DoctorRepairReview<RustupAction>,
    pub proposed_install: Option<DoctorRepairReview<RustupAction>>,
}

pub fn launch_rustup_operation(
    mut review: DoctorRepairReview<RustupAction>,
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
) -> Task<Result<RustupOperationResult>> {
    let filesystem = project.read(cx).fs().clone();
    let project = project.downgrade();
    let executor = cx.background_executor().clone();
    cx.spawn(async move |cx| {
        let started = executor.now();
        let current = cx.update(|cx| project_repair_binding(&project.upgrade().context("Doctor project closed")?, directory.clone(), generation, cx))?;
        review.record_before_execution(&current, report, product, &history, now, &cancelled).await?;
        let preparation = async {
            review.action().snapshot.revalidate(filesystem.as_ref(), &review.action().roots).await?;
            if let Some(inventory) = &review.action().reviewed_inventory {
                ensure!(&review.action().inventory_state(filesystem.as_ref()).await? == inventory, "Installation inventory changed after review; diagnose again");
            }
            let arguments = review.action().command_arguments();
            let environment = review.action().snapshot.environment()?;
            cx.update(|cx| {
                let project = project.upgrade().context("Doctor project closed")?;
                let current = project_repair_binding(&project, directory.clone(), generation, cx)?;
                review.revalidate(&current)?;
                ensure!(!cancelled.is_closed() && cancelled.is_empty(), "Cancelled before dispatch");
                project.read(cx).exec_user_toolchain_command_for_path(review.action().executable(), &arguments.iter().map(String::as_str).collect::<Vec<_>>(), &directory, &review.action().snapshot.user_home, &environment, cx)
            })
        }.await;
        let mut proposed_install = None;
        let authority_revoked = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut monitor = None;
        let result = match preparation {
            Ok(command) => {
                monitor = Some(cx.spawn({
                    let project = project.clone(); let directory = directory.clone(); let binding = review.binding().clone(); let cancellation = cancellation.clone(); let executor = executor.clone(); let snapshot = review.action().snapshot.clone(); let roots = review.action().roots.clone(); let filesystem = filesystem.clone(); let authority_revoked = authority_revoked.clone();
                    async move |cx| {
                        loop {
                            let manager_valid = snapshot.revalidate(filesystem.as_ref(), &roots).await.is_ok();
                            let current = cx.update(|cx| project_repair_binding(&project.upgrade().context("Doctor project closed")?, directory.clone(), generation, cx));
                            if !manager_valid || !matches!(current, Ok(current) if current == binding) { authority_revoked.store(true, std::sync::atomic::Ordering::Release); cancellation.close(); return; }
                            executor.timer(Duration::from_millis(100)).await;
                        }
                    }
                }));
                let timeout = if review.action().is_inspection() { Duration::from_secs(10) } else { Duration::from_secs(300) };
                let result = executor.spawn(run_toolchain_command(command, executor.clone(), cancelled.clone(), timeout)).await;
                result
            }
            Err(error) => Err(error),
        };
        let result = match result {
            Ok(result) => Some(result),
            Err(error) => {
                review.operation_finished(DoctorRepairOperationOutcome::Failed, DiagnosticText::new(format!("Manager dispatch or execution failed; reached effects must be inspected: {error}")))?;
                review.recheck_unavailable(DiagnosticText::new("No authoritative completed operation is available"))?;
                None
            }
        };
        if let Some(result) = result {
            let completed = result.end == ProbeEnd::Completed;
            let operation = if completed { DoctorRepairOperationOutcome::Completed } else if result.end == ProbeEnd::Cancelled && !authority_revoked.load(std::sync::atomic::Ordering::Acquire) { DoctorRepairOperationOutcome::Cancelled } else { DoctorRepairOperationOutcome::Failed };
            review.operation_finished(operation, DiagnosticText::new(format!("Existing manager operation observed {:?}; partial downloads or installations may remain. No replay or destructive cleanup.", result.end)))?;
            if review.phase() == DoctorRepairPhase::Rechecking {
                let recheck = async {
                    review.action().snapshot.revalidate(filesystem.as_ref(), &review.action().roots).await?;
                    let binding = cx.update(|cx| project_repair_binding(&project.upgrade().context("Doctor project closed")?, directory.clone(), generation, cx))?;
                    ensure!(&binding == review.binding(), "Authority or inputs changed before verification");
                    ensure!(!authority_revoked.load(std::sync::atomic::Ordering::Acquire), "Reviewed authority or manager inputs were revoked; reached effects are unverified");
                    ensure!(completed, "The manager operation did not complete; verification is incomplete");
                    let observed_at = now.saturating_add(executor.now().duration_since(started).as_secs());
                    if review.action().is_inspection() {
                        ensure!(result.detail.as_str().split_whitespace().take(2).eq(["rustup", "1.29.0"]), "Manager version is unsupported or could not be observed");
                        let check = DoctorCheck::new("rust.manager.inspect", "Selected Rustup manager version", CheckOutcome::Ready, "Observed selected Rustup 1.29.0; installation still requires a separate approval", observed_at);
                        let mut action = RustupAction { snapshot: review.action().snapshot.clone(), request: review.action().request.clone(), toolchain: review.action().toolchain.clone(), roots: review.action().roots.clone(), stage: RustupStage::Install, reviewed_inventory: None };
                        action.reviewed_inventory = Some(action.inventory_state(filesystem.as_ref()).await?);
                        let inventory = action.inventory(filesystem.as_ref(), observed_at).await?;
                        if inventory.outcome == CheckOutcome::Ready { return Ok((binding, check)); }
                        if !matches!(action.request, RustupInstallRequest::Toolchain { .. }) {
                            ensure!(filesystem.metadata(&action.destination()).await?.is_some(), "Install the exact missing toolchain in a separately approved action before adding a component or target");
                        }
                        let effects = DiagnosticText::new(format!("Invoke existing Rustup 1.29.0 at {} with arguments {:?} for exact {:?}, source {}, user-scoped destination {}. Toolchain installation uses the per-invocation minimal profile (rustc, cargo and native standard library). No self-update/default/global-profile/pin/JAVA_HOME changes, shell hooks or credential inheritance. Network downloads/cache writes and partial manager effects are possible. Maximum five minutes/10 MiB observed output; cancellation kills only this owned process tree and does not roll back reached files. Inventory and installed compiler recheck are required; no project-build fix is claimed.", action.executable().display(), action.command_arguments(), action.request, action.source(), action.destination().display()));
                        proposed_install = Some(DoctorRepairReview::new(action, binding.clone(), inventory, effects)?);
                        Ok((binding, check))
                    } else {
                        let mut check = review.action().inventory(filesystem.as_ref(), observed_at).await?;
                        ensure!(check.outcome == CheckOutcome::Ready, "Requested installation is absent from fresh manager inventory");
                        let compiler = crate::doctor::ProbeExecutable::inspect(crate::doctor::DependencyProbe::RustCompiler, review.action().destination().join(if cfg!(windows) { "bin/rustc.exe" } else { "bin/rustc" }), &review.action().roots, filesystem.as_ref()).await?;
                        let command = cx.update(|cx| project.upgrade().context("Doctor project closed")?.read(cx).exec_read_only_probe_for_path(compiler.path(), &["--version"], &directory, cx))?;
                        let expected_compiler = review.action().compiler_version(filesystem.as_ref()).await?;
                        let compiler_result = executor.spawn(run_toolchain_command(command, executor.clone(), cancelled.clone(), Duration::from_secs(10))).await?;
                        ensure!(compiler_result.end == ProbeEnd::Completed && compiler_result.detail.as_str() == format!("rustc {expected_compiler}"), "Installed compiler version was not verified");
                        compiler.revalidate(&review.action().roots, filesystem.as_ref()).await?;
                        review.action().snapshot.revalidate(filesystem.as_ref(), &review.action().roots).await?;
                        let current = cx.update(|cx| project_repair_binding(&project.upgrade().context("Doctor project closed")?, directory.clone(), generation, cx))?;
                        ensure!(current == binding, "Authority changed during compiler verification");
                        check.observed_at = now.saturating_add(executor.now().duration_since(started).as_secs());
                        check.detail = DiagnosticText::new(format!("Requested inventory and installed compiler {} were observed. Captured default/settings remain unchanged; project build and application correctness are not verified.", review.action().request.version()));
                        Ok((binding, check))
                    }
                }.await;
                match recheck {
                    Ok((binding, check)) => review.observe_recheck(&binding, check)?,
                    Err(error) => { proposed_install = None; if (cancelled.is_closed() || !cancelled.is_empty()) && !authority_revoked.load(std::sync::atomic::Ordering::Acquire) { review.cancel_recheck(DiagnosticText::new("User cancelled verification; reached installation files may remain"))?; } else { review.recheck_unavailable(DiagnosticText::new(format!("Installation or manager verification incomplete: {error}")))?; } }
                }
            }
        }
        drop(monitor);
        review.persist_outcome(product, &history, None, now.saturating_add(executor.now().duration_since(started).as_secs())).await?;
        Ok(RustupOperationResult { review, proposed_install })
    })
}

#[cfg(all(test, unix))]
pub(crate) mod tests {
    use super::*;
    use crate::doctor::{DoctorScope, ReceiptOutcome};
    use project::trusted_worktrees::{self, DbTrustedPaths, PathTrust, TrustedWorktrees};
    use std::sync::Arc;
    use util::{rel_path::RelPath, test::TempTree};

    const MANAGER_FIXTURE: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
static void directory(const char *path) { if (mkdir(path, 0700) && access(path, F_OK)) exit(71); }
static void write_file(const char *path, const char *text) { FILE *file = fopen(path, "w"); if (!file) exit(72); fputs(text, file); if (fclose(file)) exit(73); }
int main(int count, char **arguments) {
    const char *name = strrchr(arguments[0], '/'); name = name ? name + 1 : arguments[0];
    if (!strcmp(name, "rustc")) { char path[4096]; snprintf(path, sizeof(path), "%s.mode", arguments[0]); if (!access(path, F_OK)) puts("rustc 99.9.8 (fixture)"); else puts("rustc 99.9.9 (fixture)"); return 0; }
    if (getenv("GITHUB_TOKEN") || getenv("JAVA_HOME") || getenv("RUSTUP_DIST_SERVER") || getenv("LD_PRELOAD") || getenv("DYLD_INSERT_LIBRARIES")) return 74;
    if (count == 2 && !strcmp(arguments[1], "--version")) { puts("rustup 1.29.0 (isolated fixture)"); return 0; }
    if (count != 8 || strcmp(arguments[1], "toolchain") || strcmp(arguments[2], "install") || strcmp(arguments[4], "--no-self-update") || strcmp(arguments[5], "--no-update") || strcmp(arguments[6], "--profile") || strcmp(arguments[7], "minimal")) return 75;
    const char *home = getenv("RUSTUP_HOME"); if (!home) return 76;
    char path[4096], base[4096], mode_path[4096], mode[64] = "success";
    snprintf(mode_path, sizeof(mode_path), "%s/fixture-mode", home);
    FILE *mode_file = fopen(mode_path, "r"); if (mode_file) { if (!fgets(mode, sizeof(mode), mode_file)) return 77; fclose(mode_file); }
    snprintf(path, sizeof(path), "%s/started", home); write_file(path, "started");
    if (!strcmp(mode, "wait")) { for (;;) pause(); }
    if (!strcmp(mode, "fail")) { puts("partial download; fixture failure"); return 9; }
    snprintf(path, sizeof(path), "%s/toolchains", home); directory(path);
    snprintf(base, sizeof(base), "%s/toolchains/%s", home, arguments[3]); directory(base);
    snprintf(path, sizeof(path), "%s/bin", base); directory(path);
    snprintf(path, sizeof(path), "%s/bin/rustc", base);
    FILE *input = fopen(arguments[0], "rb"), *output = fopen(path, "wb"); if (!input || !output) return 78;
    int byte; while ((byte = fgetc(input)) != EOF) fputc(byte, output); fclose(input); if (fclose(output) || chmod(path, 0700)) return 79;
    if (!strcmp(mode, "wrong-version")) { snprintf(path, sizeof(path), "%s/bin/rustc.mode", base); write_file(path, "wrong"); }
    snprintf(path, sizeof(path), "%s/lib", base); directory(path);
    snprintf(path, sizeof(path), "%s/lib/rustlib", base); directory(path);
    if (strcmp(mode, "no-inventory")) { snprintf(path, sizeof(path), "%s/lib/rustlib/components", base); write_file(path, "rustc\n"); }
    snprintf(path, sizeof(path), "%s/lib/rustlib/multirust-channel-manifest.toml", base);
    write_file(path, "manifest-version = '2'\ndate = '2026-10-04'\n[pkg.rustc]\nversion = '99.9.9 (fixture)'\n");
    puts("fixture installation completed"); return 0;
}
"#;

    pub(crate) struct Fixture {
        _tree: TempTree,
        pub(crate) home: PathBuf,
        pub(crate) project: Entity<Project>,
        directory: ProjectPath,
        filesystem: Arc<dyn fs::Fs>,
    }

    pub(crate) async fn fixture(cx: &mut gpui::TestAppContext) -> Fixture {
        cx.executor().allow_parking();
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            trusted_worktrees::init(DbTrustedPaths::default(), cx);
        });
        let tree = TempTree::new(serde_json::json!({
            "home": { ".cargo": {"bin": {}}, ".rustup": {"settings.toml": "version = '12'\ndefault_toolchain = 'preserved-default'\nprofile = 'complete'\n"}},
            "project": {"Cargo.lock": "unrelated lockfile", "main.rs": "unrelated source"}
        }));
        let root = tree.path().canonicalize().expect("Fixture root");
        let home = root.join("home");
        let source = root.join("fixture.c");
        std::fs::write(&source, MANAGER_FIXTURE).expect("Native offline fixture source");
        let compiled = smol::process::Command::new("/usr/bin/cc")
            .arg(&source)
            .arg("-o")
            .arg(home.join(".cargo/bin/rustup"))
            .output()
            .await
            .expect("Existing compiler creates isolated native fixture");
        assert!(
            compiled.status.success(),
            "{}",
            String::from_utf8_lossy(&compiled.stderr)
        );
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
                .expect("Fixture worktree")
                .read(cx)
                .id(),
            path: RelPath::empty_arc(),
        });
        let store = project.read_with(cx, |project, _| project.worktree_store());
        let trust = cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("Existing trust"));
        trust.update(cx, |trust, cx| {
            trust.trust(
                &store,
                collections::HashSet::from_iter([PathTrust::Worktree(directory.worktree_id)]),
                cx,
            )
        });
        Fixture {
            _tree: tree,
            home,
            project,
            directory,
            filesystem,
        }
    }

    pub(crate) async fn inspection(
        fixture: &Fixture,
        cx: &mut gpui::TestAppContext,
    ) -> DoctorRepairReview<RustupAction> {
        let roots = fixture.project.read_with(cx, |project, cx| {
            project
                .worktrees(cx)
                .map(|tree| tree.read(cx).abs_path().to_path_buf())
                .collect::<Vec<_>>()
        });
        let snapshot = RustupSnapshot::inspect(
            fixture.filesystem.as_ref(),
            &fixture.home,
            &fixture.home.join(".cargo/bin/rustup"),
            &roots,
        )
        .await
        .expect("Native manager snapshot");
        let binding = cx
            .update(|cx| project_repair_binding(&fixture.project, fixture.directory.clone(), 1, cx))
            .expect("Selected authority");
        let request = RustupInstallRequest::Toolchain {
            version: "99.9.9".into(),
        };
        request.validate().expect("Exact fixture version");
        DoctorRepairReview::new(
            RustupAction {
                snapshot,
                request,
                toolchain: format!("99.9.9-{}", local_host_triple().expect("Supported host")),
                roots,
                stage: RustupStage::Inspect,
                reviewed_inventory: None,
            },
            binding,
            DoctorCheck::new(
                "rust.manager.inspect",
                "Manager inspection",
                CheckOutcome::Untested,
                "Not invoked",
                10,
            ),
            DiagnosticText::new("Only inspect the isolated existing manager; no install consent"),
        )
        .expect("Review")
    }

    fn report(review: &DoctorRepairReview<RustupAction>) -> DoctorReport {
        let mut report = DoctorReport::new(
            DoctorScope {
                product: DiagnosticText::new("Copper"),
                project: DiagnosticText::new("isolated fixture"),
                worktree: DiagnosticText::new("root"),
                host: DiagnosticText::new("Local"),
            },
            1,
            10,
        );
        report
            .push(review.before().clone())
            .expect("Relevant evidence");
        report
    }

    fn approve(
        review: &mut DoctorRepairReview<RustupAction>,
        fixture: &Fixture,
        cx: &mut gpui::TestAppContext,
    ) {
        let binding = cx
            .update(|cx| project_repair_binding(&fixture.project, fixture.directory.clone(), 1, cx))
            .expect("Authority");
        review.approve(&binding).expect("Explicit action approval");
        assert!(
            review.approve(&binding).is_err(),
            "Duplicate approval rejected"
        );
    }

    fn launch(
        review: DoctorRepairReview<RustupAction>,
        fixture: &Fixture,
        history: DoctorHistoryStore,
        cancellation: async_channel::Sender<()>,
        cancelled: async_channel::Receiver<()>,
        cx: &mut gpui::TestAppContext,
    ) -> Task<Result<RustupOperationResult>> {
        let report = report(&review);
        cx.update(|cx| {
            launch_rustup_operation(
                review,
                &fixture.project,
                fixture.directory.clone(),
                1,
                report,
                DoctorProduct::Copper,
                history,
                10,
                cancellation,
                cancelled,
                cx,
            )
        })
    }

    #[test]
    fn doctor_rust_install_exact_requests_exclude_implicit_effects() {
        let request = RustupInstallRequest::Toolchain {
            version: "1.97.1".into(),
        };
        assert!(request.validate().is_ok());
        assert_eq!(
            request.arguments("1.97.1-host"),
            [
                "toolchain",
                "install",
                "1.97.1-host",
                "--no-self-update",
                "--no-update",
                "--profile",
                "minimal"
            ]
        );
        for version in ["nightly-2026-10-04", "beta-2024-02-29"] {
            assert!(
                RustupInstallRequest::Toolchain {
                    version: version.into()
                }
                .validate()
                .is_ok()
            );
        }
        for version in [
            "nightly-2026-02-29",
            "nightly-2026-13-01",
            "nightly-2026-10-00",
            "beta-2026-1-01",
            "stable",
            "nightly",
            "--help",
            "1.97",
            "1.97.1;touch sentinel",
            "4294967296.1.1",
        ] {
            assert!(
                RustupInstallRequest::Toolchain {
                    version: version.into()
                }
                .validate()
                .is_err()
            );
        }
        for target in ["--all", "target;write", "../target", "x86_64"] {
            assert!(
                RustupInstallRequest::Target {
                    version: "1.97.1".into(),
                    target: target.into()
                }
                .validate()
                .is_err()
            );
        }
        assert!(
            RustupInstallRequest::Component {
                version: "1.97.1".into(),
                component: "--help".into()
            }
            .validate()
            .is_err()
        );
    }

    #[gpui::test]
    async fn doctor_rust_install_requires_two_approvals_and_observed_compiler(
        cx: &mut gpui::TestAppContext,
    ) {
        let fixture = fixture(cx).await;
        let mut review = inspection(&fixture, cx).await;
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::for_test_database(&database);
        let (cancel, cancelled) = async_channel::bounded(1);
        approve(&mut review, &fixture, cx);
        let result = launch(review, &fixture, history.clone(), cancel, cancelled, cx)
            .await
            .expect("Inspection");
        assert_eq!(result.review.phase(), DoctorRepairPhase::Verified);
        assert!(
            !fixture.home.join(".rustup/started").exists(),
            "Inspection cannot install"
        );
        let mut install = result.proposed_install.expect("Distinct install proposal");
        assert_eq!(install.phase(), DoctorRepairPhase::Proposed);
        approve(&mut install, &fixture, cx);
        let (cancel, cancelled) = async_channel::bounded(1);
        let result = launch(install, &fixture, history.clone(), cancel, cancelled, cx)
            .await
            .expect("Fixture install");
        assert_eq!(result.review.phase(), DoctorRepairPhase::Verified);
        assert!(result.proposed_install.is_none());
        assert_eq!(
            std::fs::read_to_string(fixture.home.join(".rustup/settings.toml")).expect("Settings"),
            "version = '12'\ndefault_toolchain = 'preserved-default'\nprofile = 'complete'\n"
        );
        assert_eq!(
            std::fs::read_to_string(fixture._tree.path().join("project/Cargo.lock"))
                .expect("Unrelated lockfile"),
            "unrelated lockfile"
        );
        assert_eq!(
            std::fs::read_to_string(fixture._tree.path().join("project/main.rs"))
                .expect("Unrelated source"),
            "unrelated source"
        );
        let records = history
            .records(DoctorProduct::Copper, 20)
            .await
            .expect("Durable history");
        assert_eq!(records.len(), 2);
        assert!(
            records
                .iter()
                .all(|record| record.outcome == ReceiptOutcome::Completed)
        );
    }

    #[gpui::test]
    async fn doctor_rust_install_refuses_unapproved_and_stale_inputs(
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
                .records(DoctorProduct::Copper, 10)
                .await
                .expect("History")
                .is_empty()
        );
        let mut review = inspection(&fixture, cx).await;
        approve(&mut review, &fixture, cx);
        std::fs::write(
            fixture.home.join(".rustup/settings.toml"),
            "version = '12'\ndefault_toolchain = 'intervening-edit'\n",
        )
        .expect("User edit");
        let (cancel, cancelled) = async_channel::bounded(1);
        let result = launch(review, &fixture, history, cancel, cancelled, cx)
            .await
            .expect("Rejected stale dispatch");
        assert_eq!(result.review.phase(), DoctorRepairPhase::Incomplete);
        assert!(!fixture.home.join(".rustup/started").exists());
    }

    #[gpui::test]
    async fn doctor_rust_install_exit_success_or_partial_failure_never_proves_fix(
        cx: &mut gpui::TestAppContext,
    ) {
        for mode in ["no-inventory", "wrong-version", "fail"] {
            let fixture = fixture(cx).await;
            let mut review = inspection(&fixture, cx).await;
            approve(&mut review, &fixture, cx);
            let database = db::AppDatabase::test_new();
            let history = DoctorHistoryStore::for_test_database(&database);
            let (cancel, cancelled) = async_channel::bounded(1);
            let mut install = launch(review, &fixture, history.clone(), cancel, cancelled, cx)
                .await
                .expect("Inspection")
                .proposed_install
                .expect("Install review");
            std::fs::write(fixture.home.join(".rustup/fixture-mode"), mode)
                .expect("Isolated behavior");
            approve(&mut install, &fixture, cx);
            let (cancel, cancelled) = async_channel::bounded(1);
            let result = launch(install, &fixture, history.clone(), cancel, cancelled, cx)
                .await
                .expect("Observed incomplete operation");
            assert_eq!(
                result.review.phase(),
                DoctorRepairPhase::Incomplete,
                "{mode}"
            );
            assert!(fixture.home.join(".rustup/started").exists());
            assert!(
                history
                    .records(DoctorProduct::Copper, 20)
                    .await
                    .expect("History")
                    .iter()
                    .any(|record| record.outcome == ReceiptOutcome::Incomplete)
            );
        }
    }

    #[gpui::test]
    async fn doctor_rust_install_blocks_startup_side_effects_and_path_escape(
        cx: &mut gpui::TestAppContext,
    ) {
        let fixture = fixture(cx).await;
        let executable = fixture.home.join(".cargo/bin/rustup");
        std::fs::write(
            fixture.home.join(".cargo/bin/rustup-init"),
            "preserved updater",
        )
        .expect("Existing updater");
        assert!(
            RustupSnapshot::inspect(fixture.filesystem.as_ref(), &fixture.home, &executable, &[])
                .await
                .is_err()
        );
        assert_eq!(
            std::fs::read_to_string(fixture.home.join(".cargo/bin/rustup-init"))
                .expect("Preserved updater"),
            "preserved updater"
        );
        std::fs::remove_file(fixture.home.join(".cargo/bin/rustup-init"))
            .expect("Only fixture cleanup");
        std::fs::write(
            fixture.home.join(".rustup/settings.toml"),
            "version = '12'\n",
        )
        .expect("No existing default");
        assert!(
            RustupSnapshot::inspect(fixture.filesystem.as_ref(), &fixture.home, &executable, &[])
                .await
                .is_err()
        );
        std::fs::write(
            fixture.home.join(".rustup/settings.toml"),
            "version = '12'\ndefault_toolchain = 'preserved'\n",
        )
        .expect("Existing default");
        let snapshot =
            RustupSnapshot::inspect(fixture.filesystem.as_ref(), &fixture.home, &executable, &[])
                .await
                .expect("Snapshot");
        std::fs::write(&executable, "#!/bin/sh\nexit 0\n")
            .expect("Intervening script substitution");
        assert!(
            snapshot
                .revalidate(fixture.filesystem.as_ref(), &[])
                .await
                .is_err()
        );
        assert!(
            RustupSnapshot::inspect(fixture.filesystem.as_ref(), &fixture.home, &executable, &[])
                .await
                .is_err()
        );
    }
    pub(crate) async fn wait_for_started(fixture: &Fixture) {
        let marker = fixture.home.join(".rustup/started");
        smol::unblock(move || {
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while !marker.exists() {
                assert!(
                    std::time::Instant::now() < deadline,
                    "Native fixture must actually start"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
        })
        .await;
    }

    #[gpui::test]
    async fn doctor_rust_install_cancels_owned_process_times_out_and_stops_on_revocation(
        cx: &mut gpui::TestAppContext,
    ) {
        for reason in ["cancel", "timeout", "revoke"] {
            eprintln!("Native Doctor fixture scenario: {reason}");
            let fixture = fixture(cx).await;
            let mut review = inspection(&fixture, cx).await;
            approve(&mut review, &fixture, cx);
            let database = db::AppDatabase::test_new();
            let history = DoctorHistoryStore::for_test_database(&database);
            let (cancel, cancelled) = async_channel::bounded(1);
            let mut install = launch(review, &fixture, history.clone(), cancel, cancelled, cx)
                .await
                .expect("Inspection")
                .proposed_install
                .expect("Separate install proposal");
            std::fs::write(fixture.home.join(".rustup/fixture-mode"), "wait")
                .expect("Native pending operation");
            approve(&mut install, &fixture, cx);
            let (cancel, cancelled) = async_channel::bounded(1);
            let operation = launch(
                install,
                &fixture,
                history.clone(),
                cancel.clone(),
                cancelled,
                cx,
            );
            wait_for_started(&fixture).await;
            eprintln!("Native Doctor fixture started: {reason}");
            let records = history
                .records(DoctorProduct::Copper, 10)
                .await
                .expect("Receipt before effects");
            assert!(
                records
                    .iter()
                    .any(|record| record.outcome == ReceiptOutcome::Running)
            );
            eprintln!("Native Doctor receipt observed: {reason}");
            match reason {
                "cancel" => {
                    cancel.close();
                }
                "timeout" => {
                    // Native async-process readiness can keep scheduling work. Advance
                    // GPUI's clock without first draining that external work.
                    cx.executor()
                        .dispatcher()
                        .as_test()
                        .expect("Test dispatcher")
                        .scheduler()
                        .clock()
                        .advance(Duration::from_secs(301));
                }
                "revoke" => {
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
                _ => unreachable!("Fixed fixture scenarios"),
            }
            eprintln!("Native Doctor stop requested: {reason}");
            let result = operation.await.expect("Stopped owned operation");
            assert_eq!(
                result.review.phase(),
                if reason == "cancel" {
                    DoctorRepairPhase::Cancelled
                } else {
                    DoctorRepairPhase::Incomplete
                },
                "{reason}"
            );
            assert!(result.proposed_install.is_none());
            assert!(
                !result.review.action().destination().exists(),
                "No simulated install or replay"
            );
            assert!(
                history
                    .records(DoctorProduct::Copper, 320)
                    .await
                    .expect("Terminal history")
                    .iter()
                    .all(|record| record.outcome != ReceiptOutcome::Running)
            );
        }
    }

    #[gpui::test]
    async fn doctor_rust_install_rejects_inventory_races_and_symlink_escape(
        cx: &mut gpui::TestAppContext,
    ) {
        let fixture = fixture(cx).await;
        let mut review = inspection(&fixture, cx).await;
        approve(&mut review, &fixture, cx);
        let database = db::AppDatabase::test_new();
        let history = DoctorHistoryStore::for_test_database(&database);
        let (cancel, cancelled) = async_channel::bounded(1);
        let mut install = launch(review, &fixture, history.clone(), cancel, cancelled, cx)
            .await
            .expect("Inspection")
            .proposed_install
            .expect("Install review");
        approve(&mut install, &fixture, cx);
        let inventory = install
            .action()
            .destination()
            .join("lib/rustlib/components");
        std::fs::create_dir_all(inventory.parent().expect("Parent"))
            .expect("Intervening owner install");
        std::fs::write(&inventory, "rustc\n").expect("Intervening inventory");
        let (cancel, cancelled) = async_channel::bounded(1);
        let result = launch(install, &fixture, history, cancel, cancelled, cx)
            .await
            .expect("Stale refusal");
        assert_eq!(result.review.phase(), DoctorRepairPhase::Incomplete);
        assert!(!fixture.home.join(".rustup/started").exists());
        let outside = fixture._tree.path().join("outside");
        std::fs::create_dir(&outside).expect("Outside destination");
        std::fs::write(
            outside.join("settings.toml"),
            "version = '12'\ndefault_toolchain = 'preserved'\n",
        )
        .expect("Outside settings");
        std::fs::rename(
            fixture.home.join(".rustup"),
            fixture.home.join("saved-rustup"),
        )
        .expect("Preserve fixture manager");
        std::os::unix::fs::symlink(outside, fixture.home.join(".rustup")).expect("Escaping alias");
        assert!(
            RustupSnapshot::inspect(
                fixture.filesystem.as_ref(),
                &fixture.home,
                &fixture.home.join(".cargo/bin/rustup"),
                &[]
            )
            .await
            .is_err()
        );
    }
    #[gpui::test]
    async fn doctor_rust_install_checks_channel_dates_component_assets_and_target_containment(
        cx: &mut gpui::TestAppContext,
    ) {
        let fixture = fixture(cx).await;
        let review = inspection(&fixture, cx).await;
        let mut action = RustupAction {
            snapshot: review.action().snapshot.clone(),
            request: RustupInstallRequest::Toolchain {
                version: "nightly-2026-10-04".into(),
            },
            toolchain: format!("nightly-2026-10-04-{}", local_host_triple().expect("Host")),
            roots: review.action().roots.clone(),
            stage: RustupStage::Install,
            reviewed_inventory: None,
        };
        let directory = action.destination();
        std::fs::create_dir_all(directory.join("bin")).expect("Fixture installed executables");
        std::fs::create_dir_all(directory.join("lib/rustlib")).expect("Fixture inventory");
        std::fs::copy(
            fixture.home.join(".cargo/bin/rustup"),
            directory.join("bin/rustc"),
        )
        .expect("Native fixture asset");
        let manifest = directory.join("lib/rustlib/multirust-channel-manifest.toml");
        std::fs::write(&manifest, "manifest-version = '2'\ndate = '2026-10-04'\n[pkg.rustc]\nversion = '1.99.0-nightly (fixture)'\n").expect("Exact channel metadata");
        assert_eq!(
            action
                .compiler_version(fixture.filesystem.as_ref())
                .await
                .expect("Dated channel"),
            "1.99.0-nightly (fixture)"
        );
        action.request = RustupInstallRequest::Toolchain {
            version: "nightly-2026-10-03".into(),
        };
        assert!(
            action
                .compiler_version(fixture.filesystem.as_ref())
                .await
                .is_err()
        );
        action.request = RustupInstallRequest::Component {
            version: "nightly-2026-10-04".into(),
            component: "rustfmt".into(),
        };
        let components = directory.join("lib/rustlib/components");
        std::fs::write(
            &components,
            format!("rustfmt-preview-{}\n", local_host_triple().expect("Host")),
        )
        .expect("Recorded but missing component");
        assert!(
            action
                .inventory(fixture.filesystem.as_ref(), 10)
                .await
                .is_err()
        );
        std::fs::copy(
            fixture.home.join(".cargo/bin/rustup"),
            directory.join("bin/rustfmt"),
        )
        .expect("Actual native component");
        assert_eq!(
            action
                .inventory(fixture.filesystem.as_ref(), 10)
                .await
                .expect("Component assets")
                .outcome,
            CheckOutcome::Ready
        );
        action.request = RustupInstallRequest::Target {
            version: "nightly-2026-10-04".into(),
            target: "wasm32-unknown-unknown".into(),
        };
        std::fs::write(&components, "rust-std-wasm32-unknown-unknown\n").expect("Target inventory");
        let libraries = directory.join("lib/rustlib/wasm32-unknown-unknown/lib");
        std::fs::create_dir_all(&libraries).expect("Target directory");
        assert!(
            action
                .inventory(fixture.filesystem.as_ref(), 10)
                .await
                .is_err()
        );
        std::fs::write(libraries.join("libstd-fixture.rlib"), "fixture archive")
            .expect("Observed target asset");
        assert_eq!(
            action
                .inventory(fixture.filesystem.as_ref(), 10)
                .await
                .expect("Target assets")
                .outcome,
            CheckOutcome::Ready
        );
        std::fs::remove_file(libraries.join("libstd-fixture.rlib")).expect("Fixture reset");
        std::os::unix::fs::symlink(
            fixture._tree.path().join("project/main.rs"),
            libraries.join("libstd-escape.rlib"),
        )
        .expect("Escaping asset");
        assert!(
            action
                .inventory(fixture.filesystem.as_ref(), 10)
                .await
                .is_err()
        );
    }
}
