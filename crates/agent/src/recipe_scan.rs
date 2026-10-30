use anyhow::{Context as _, Result, bail, ensure};
use cap_std::fs::{Dir, OpenOptions};
use collections::HashSet;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::{
    io::Read as _,
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};
use uuid::Uuid;

pub const MAX_COPY_FILES: usize = 10_000;
pub const MAX_COPY_BYTES: usize = 256 * 1024 * 1024;
const RESOURCE_LABEL: &str = "dev.made.recipe-test";

#[derive(Clone, Default)]
pub(crate) struct RecipeScanSessionPolicy {
    blocked_dependency: std::rc::Rc<std::cell::RefCell<Option<String>>>,
    pub(crate) interruption: std::rc::Rc<std::cell::Cell<Option<RecipeScanOutcome>>>,
    pub(crate) read_only: bool,
    pub(crate) run_id: Option<Uuid>,
    pub(crate) project_paths: Option<util::path_list::SerializedPathList>,
    pub(crate) observations: std::rc::Rc<std::cell::RefCell<Vec<String>>>,
}

impl RecipeScanSessionPolicy {
    pub(crate) fn block(&self, reason: impl Into<String>) {
        self.blocked_dependency
            .borrow_mut()
            .get_or_insert_with(|| reason.into());
    }

    pub(crate) fn blocked_dependency(&self) -> Option<String> {
        self.blocked_dependency.borrow().clone()
    }

    pub(crate) fn check(&self) -> Result<()> {
        ensure!(
            !self.read_only,
            "This isolated test transcript is read-only. Launch a new reviewed test to execute it again."
        );
        if let Some(reason) = self.blocked_dependency() {
            bail!("Recipe test is incomplete: {reason}");
        }
        if let Some(outcome) = self.interruption.get() {
            bail!("Recipe test stopped: {outcome:?}");
        }
        Ok(())
    }
}

pub struct ScanDockerClient {
    endpoint: String,
    docker_cli: PathBuf,
    configuration: tempfile::TempDir,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScanDockerPreflight {
    pub engine_id: String,
    pub image_id: String,
    pub architecture: String,
}

impl ScanDockerClient {
    pub fn from_reviewed_endpoint(endpoint: String) -> Result<Self> {
        remote::DockerRecipeTestOptions::validate_endpoint(&endpoint)?;
        Self::with_reviewed_cli(
            endpoint,
            remote::DockerRecipeTestOptions::local_docker_cli()?,
        )
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn with_cli_for_test(endpoint: String, docker_cli: PathBuf) -> Result<Self> {
        Self::with_reviewed_cli(endpoint, docker_cli)
    }

    fn with_reviewed_cli(endpoint: String, docker_cli: PathBuf) -> Result<Self> {
        remote::DockerRecipeTestOptions::validate_endpoint(&endpoint)?;
        // Docker's user configuration can inject proxy environment variables into
        // new containers. An empty temporary client configuration prevents that.
        let configuration = tempfile::Builder::new()
            .prefix("made-recipe-docker-")
            .tempdir()?;
        Ok(Self {
            endpoint,
            docker_cli,
            configuration,
        })
    }

    pub fn expected_runtime_version(cx: &gpui::App) -> Result<String> {
        remote::DockerRecipeTestOptions::expected_version(cx)
    }

    pub async fn inspect_runtime(
        &self,
        server_binary: PathBuf,
        expected_version: String,
    ) -> Result<String> {
        let runtime = remote::DockerRecipeTestOptions::prepare(
            self.endpoint.clone(),
            self.configuration.path().to_owned(),
            self.docker_cli.clone(),
            server_binary,
            expected_version,
        )
        .await?;
        Ok(runtime.server_sha256().to_owned())
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn docker_cli_path(&self) -> &Path {
        &self.docker_cli
    }

    pub fn configuration_path(&self) -> &Path {
        self.configuration.path()
    }

    pub async fn preflight(
        &self,
        identity: &ScanContainerIdentity,
        executor: &gpui::BackgroundExecutor,
    ) -> Result<ScanDockerPreflight> {
        let info = self
            .command(
                &["info".into(), "--format={{json .}}".into()],
                None,
                executor,
            )
            .await?;
        let engine_id = validate_engine(&serde_json::from_slice(&info)?)?;
        let image = self
            .command(
                &["image".into(), "inspect".into(), identity.image.clone()],
                None,
                executor,
            )
            .await?;
        let images: Vec<serde_json::Value> = serde_json::from_slice(&image)?;
        ensure!(
            images.len() == 1,
            "Docker did not return exactly one reviewed image"
        );
        let image = images
            .first()
            .context("The reviewed image is not available locally")?;
        identity.validate_image(image)?;
        let image_id = image["Id"]
            .as_str()
            .context("Missing local image identity")?;
        ensure!(
            image_id
                .strip_prefix("sha256:")
                .is_some_and(|digest| digest.len() == 64
                    && digest.bytes().all(|byte| byte.is_ascii_hexdigit())),
            "Invalid local image identity"
        );
        Ok(ScanDockerPreflight {
            engine_id,
            image_id: image_id.into(),
            architecture: image["Architecture"]
                .as_str()
                .context("Missing image architecture")?
                .into(),
        })
    }

    pub(crate) async fn create_verified(
        &self,
        identity: &ScanContainerIdentity,
        reviewed: &ScanDockerPreflight,
        executor: &gpui::BackgroundExecutor,
    ) -> Result<String> {
        ensure!(
            &self.preflight(identity, executor).await? == reviewed,
            "Docker engine or image changed after review; prepare a new test"
        );
        let created = self
            .command(&identity.create_arguments()?, None, executor)
            .await?;
        let created = std::str::from_utf8(&created)?.trim();
        let verified = self.inspect_owned(identity, reviewed, executor).await?;
        ensure!(
            created == verified,
            "Docker created an unexpected container identity"
        );
        Ok(verified)
    }

    pub(crate) async fn inspect_owned(
        &self,
        identity: &ScanContainerIdentity,
        reviewed: &ScanDockerPreflight,
        executor: &gpui::BackgroundExecutor,
    ) -> Result<String> {
        let output = self
            .command(
                &["container".into(), "inspect".into(), identity.name.clone()],
                None,
                executor,
            )
            .await?;
        let containers: Vec<serde_json::Value> = serde_json::from_slice(&output)?;
        ensure!(
            containers.len() == 1,
            "Docker did not return exactly one test container"
        );
        identity.validate_container(
            containers.first().context("Missing test container")?,
            &reviewed.image_id,
        )
    }

    pub(crate) async fn start_with_copy(
        &self,
        identity: &ScanContainerIdentity,
        reviewed: &ScanDockerPreflight,
        copy: &PreparedScanCopy,
        executor: &gpui::BackgroundExecutor,
    ) -> Result<String> {
        let identifier = self.inspect_owned(identity, reviewed, executor).await?;
        self.command(&["start".into(), identifier.clone()], None, executor)
            .await?;
        self.inspect_owned(identity, reviewed, executor).await?;
        self.command(
            &[
                "exec".into(),
                "-i".into(),
                "--user=1000:1000".into(),
                identifier.clone(),
                "/usr/bin/env".into(),
                "-i".into(),
                "PATH=/usr/local/bin:/usr/bin:/bin".into(),
                "HOME=/home/made".into(),
                "tar".into(),
                "-xf".into(),
                "-".into(),
                "-C".into(),
                "/workspace".into(),
            ],
            Some(copy.archive().await?),
            executor,
        )
        .await?;
        Ok(identifier)
    }

    pub(crate) async fn remove_owned(
        &self,
        identity: &ScanContainerIdentity,
        reviewed: &ScanDockerPreflight,
        executor: &gpui::BackgroundExecutor,
    ) -> Result<()> {
        let info = self
            .command(
                &["info".into(), "--format={{json .}}".into()],
                None,
                executor,
            )
            .await?;
        ensure!(
            validate_engine(&serde_json::from_slice(&info)?)? == reviewed.engine_id,
            "Docker engine changed; cleanup must target the original engine"
        );
        let matches = self
            .command(
                &[
                    "container".into(),
                    "ls".into(),
                    "--all".into(),
                    "--quiet".into(),
                    "--no-trunc".into(),
                    "--filter".into(),
                    format!("label={RESOURCE_LABEL}={}", identity.run_id),
                    "--filter".into(),
                    format!("name=^/{}$", identity.name),
                ],
                None,
                executor,
            )
            .await?;
        if matches.iter().all(u8::is_ascii_whitespace) {
            return Ok(());
        }
        let identifier = self.inspect_owned(identity, reviewed, executor).await?;
        ensure!(
            std::str::from_utf8(&matches)?.trim() == identifier,
            "Docker cleanup identity is ambiguous"
        );
        self.command(
            &[
                "container".into(),
                "rm".into(),
                "--force".into(),
                identifier,
            ],
            None,
            executor,
        )
        .await?;
        Ok(())
    }

    pub(crate) async fn command(
        &self,
        arguments: &[String],
        input: Option<Vec<u8>>,
        executor: &gpui::BackgroundExecutor,
    ) -> Result<Vec<u8>> {
        use futures::{AsyncReadExt as _, AsyncWriteExt as _, FutureExt as _};
        use util::command::Stdio;
        let mut command = remote::DockerRecipeTestOptions::local_command(
            &self.docker_cli,
            &self.endpoint,
            self.configuration.path(),
        )?;
        command
            .args(arguments)
            .kill_on_drop(true)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command
            .spawn()
            .context("Cannot start the local Docker CLI")?;
        let mut stdin = child
            .stdin
            .take()
            .context("Docker input pipe is unavailable")?;
        let stdout = child
            .stdout
            .take()
            .context("Docker output pipe is unavailable")?;
        let stderr = child
            .stderr
            .take()
            .context("Docker diagnostic pipe is unavailable")?;
        let operation = async {
            let send = async move {
                if let Some(input) = input {
                    stdin.write_all(&input).await?;
                }
                drop(stdin);
                anyhow::Ok(())
            };
            let read_output = async move {
                let mut bytes = Vec::new();
                stdout.take(1024 * 1024 + 1).read_to_end(&mut bytes).await?;
                ensure!(
                    bytes.len() <= 1024 * 1024,
                    "Docker output exceeded the test setup limit"
                );
                Ok(bytes)
            };
            let read_error = async move {
                let mut bytes = Vec::new();
                stderr.take(64 * 1024 + 1).read_to_end(&mut bytes).await?;
                ensure!(
                    bytes.len() <= 64 * 1024,
                    "Docker diagnostics exceeded the test setup limit"
                );
                Ok(bytes)
            };
            let ((), output, error) = futures::try_join!(send, read_output, read_error)?;
            let status = child.status().await?;
            ensure!(
                status.success(),
                "Docker test operation failed: {}",
                String::from_utf8_lossy(&error)
            );
            Ok(output)
        }
        .fuse();
        futures::pin_mut!(operation);
        let deadline = executor.timer(std::time::Duration::from_secs(30)).fuse();
        futures::pin_mut!(deadline);
        futures::select_biased! {
            result = operation => result,
            () = deadline => bail!("Docker test operation timed out; test-owned resources still require cleanup"),
        }
    }
}

fn validate_engine(info: &serde_json::Value) -> Result<String> {
    ensure!(
        info["OSType"] == "linux",
        "Recipe tests require a Linux Docker engine"
    );
    for capability in [
        "MemoryLimit",
        "SwapLimit",
        "CpuCfsPeriod",
        "CpuCfsQuota",
        "PidsLimit",
    ] {
        ensure!(
            info[capability] == true,
            "Docker cannot enforce the required test limit: {capability}"
        );
    }
    let identity = info["ID"]
        .as_str()
        .filter(|value| !value.is_empty())
        .context("Docker did not provide a stable engine identity")?;
    Ok(identity.to_owned())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScanCopyEntry {
    pub path: PathBuf,
    pub bytes: usize,
    pub sha256: String,
    pub executable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScanCopyExclusion {
    pub path: PathBuf,
    pub reason: String,
}

pub struct PreparedScanCopy {
    entries: Vec<ScanCopyEntry>,
    exclusions: Vec<ScanCopyExclusion>,
    contents: Vec<Vec<u8>>,
}

fn validate_copy_path(path: &Path) -> Result<()> {
    let text = path.to_str().context("Test copy paths must be UTF-8")?;
    ensure!(
        !text.is_empty()
            && text.len() <= 4096
            && !text.chars().any(char::is_control)
            && !text.contains(['\\', ':'])
            && path
                .components()
                .all(|part| matches!(part, Component::Normal(_))),
        "Test copy paths must be ordinary relative project paths"
    );
    Ok(())
}

fn exclusion_reason(path: &Path) -> Option<&'static str> {
    for component in path.components() {
        let name = component.as_os_str().to_string_lossy().to_ascii_lowercase();
        if matches!(
            name.as_str(),
            ".git"
                | ".zed"
                | ".made"
                | ".agents"
                | ".devcontainer"
                | "hooks"
                | "agents.md"
                | "claude.md"
                | "dockerfile"
                | "compose.yaml"
                | "compose.yml"
                | "docker-compose.yaml"
                | "docker-compose.yml"
        ) {
            return Some("Project configuration, agent instructions or setup hooks");
        }
        if name == ".env"
            || name.starts_with(".env.")
            || matches!(
                name.as_str(),
                ".ssh"
                    | ".aws"
                    | ".azure"
                    | ".kube"
                    | ".docker"
                    | ".npmrc"
                    | ".netrc"
                    | ".pypirc"
                    | ".git-credentials"
                    | "credentials"
                    | "credentials.json"
                    | "secrets.json"
                    | "id_rsa"
                    | "id_ed25519"
            )
            || name.ends_with(".pem")
            || name.ends_with(".key")
            || name.ends_with(".p12")
            || name.ends_with(".pfx")
        {
            return Some("Potential credential or secret file");
        }
    }
    None
}

impl PreparedScanCopy {
    pub fn entries(&self) -> &[ScanCopyEntry] {
        &self.entries
    }

    pub fn exclusions(&self) -> &[ScanCopyExclusion] {
        &self.exclusions
    }

    pub fn capture(root: &Path, selected: &[PathBuf]) -> Result<Self> {
        Self::capture_with_cancellation(root, selected, &AtomicBool::new(false))
    }

    pub fn capture_with_cancellation(
        root: &Path,
        selected: &[PathBuf],
        cancellation: &AtomicBool,
    ) -> Result<Self> {
        Self::capture_bounded(root, selected, MAX_COPY_FILES, MAX_COPY_BYTES, cancellation)
    }

    fn capture_bounded(
        root: &Path,
        selected: &[PathBuf],
        maximum_files: usize,
        maximum_bytes: usize,
        cancellation: &AtomicBool,
    ) -> Result<Self> {
        ensure!(
            !cancellation.load(Ordering::Relaxed),
            "Test copy preparation was cancelled"
        );
        ensure!(
            root.is_absolute(),
            "A test copy requires an absolute project root"
        );
        ensure!(
            selected.len() <= maximum_files,
            "Test copy exceeds the file count limit"
        );
        let directory = Dir::open_ambient_dir(root, cap_std::ambient_authority())
            .context("Cannot open the selected project for a test copy")?;
        let mut result = Self {
            entries: Vec::new(),
            exclusions: Vec::new(),
            contents: Vec::new(),
        };
        let mut seen = HashSet::default();
        let mut total = 0_usize;
        for path in selected {
            ensure!(
                !cancellation.load(Ordering::Relaxed),
                "Test copy preparation was cancelled"
            );
            validate_copy_path(path)?;
            ensure!(
                seen.insert(path.clone()),
                "Duplicate file in the test copy: {}",
                path.display()
            );
            if let Some(reason) = exclusion_reason(path) {
                result.exclusions.push(ScanCopyExclusion {
                    path: path.clone(),
                    reason: reason.into(),
                });
                continue;
            }
            let mut prefix = PathBuf::new();
            for component in path.components() {
                prefix.push(component);
                ensure!(
                    !directory
                        .symlink_metadata(&prefix)?
                        .file_type()
                        .is_symlink(),
                    "Test copies cannot include symlink paths: {}",
                    path.display()
                );
            }
            let mut options = OpenOptions::new();
            options.read(true);
            #[cfg(unix)]
            {
                use cap_std::fs::OpenOptionsExt as _;
                // A replaced FIFO must not block preparation, and a replaced leaf
                // symlink must not bypass the reviewed source selection.
                options.custom_flags(
                    (rustix::fs::OFlags::NONBLOCK | rustix::fs::OFlags::NOFOLLOW).bits() as i32,
                );
            }
            // The directory capability also confines intermediate symlink races.
            let mut file = directory
                .open_with(path, &options)
                .with_context(|| format!("Cannot safely capture {}", path.display()))?;
            let before = file.metadata()?;
            ensure!(
                before.is_file(),
                "Test copies require regular files: {}",
                path.display()
            );
            let available = maximum_bytes.saturating_sub(total);
            ensure!(
                before.len() <= available as u64,
                "Test copy exceeds the byte limit"
            );
            let mut contents = Vec::new();
            let mut buffer = [0_u8; 64 * 1024];
            loop {
                ensure!(
                    !cancellation.load(Ordering::Relaxed),
                    "Test copy preparation was cancelled"
                );
                let read_limit = buffer
                    .len()
                    .min(available.saturating_sub(contents.len()) + 1);
                let count = file.read(&mut buffer[..read_limit])?;
                if count == 0 {
                    break;
                }
                ensure!(
                    contents.len() + count <= available,
                    "Test copy exceeds the byte limit"
                );
                contents.extend_from_slice(&buffer[..count]);
            }
            let after = file.metadata()?;
            ensure!(
                before.len() == after.len()
                    && before.modified()? == after.modified()?
                    && contents.len() as u64 == after.len(),
                "File changed while preparing the test copy: {}. Prepare it again",
                path.display()
            );
            #[cfg(unix)]
            let executable = {
                use cap_std::fs::PermissionsExt as _;
                after.permissions().mode() & 0o111 != 0
            };
            #[cfg(not(unix))]
            let executable = false;
            total += contents.len();
            result.entries.push(ScanCopyEntry {
                path: path.clone(),
                bytes: contents.len(),
                sha256: format!("{:x}", Sha256::digest(&contents)),
                executable,
            });
            result.contents.push(contents);
        }
        Ok(result)
    }

    pub async fn archive(&self) -> Result<Vec<u8>> {
        ensure!(
            self.entries.len() == self.contents.len(),
            "Invalid test copy manifest"
        );
        let mut builder = async_tar::Builder::new(futures::io::Cursor::new(Vec::new()));
        for (entry, contents) in self.entries.iter().zip(&self.contents) {
            validate_copy_path(&entry.path)?;
            ensure!(
                entry.bytes == contents.len()
                    && entry.sha256 == format!("{:x}", Sha256::digest(contents)),
                "Test copy manifest changed after preparation"
            );
            let mut header = async_tar::Header::new_gnu();
            header.set_size(contents.len() as u64);
            header.set_mode(if entry.executable { 0o700 } else { 0o600 });
            header.set_uid(1000);
            header.set_gid(1000);
            header.set_mtime(0);
            header.set_entry_type(async_tar::EntryType::Regular);
            header.set_cksum();
            builder
                .append_data(&mut header, &entry.path, contents.as_slice())
                .await?;
        }
        builder.finish().await?;
        Ok(builder.into_inner().await?.into_inner())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScanContainerIdentity {
    pub run_id: Uuid,
    pub name: String,
    pub image: String,
}

impl ScanContainerIdentity {
    pub fn new(run_id: Uuid, image: String) -> Result<Self> {
        validate_image_digest(&image)?;
        Ok(Self {
            run_id,
            name: format!("made-recipe-test-{run_id}"),
            image,
        })
    }

    pub fn create_arguments(&self) -> Result<Vec<String>> {
        validate_image_digest(&self.image)?;
        ensure!(
            self.name == format!("made-recipe-test-{}", self.run_id),
            "Invalid test container identity"
        );
        let mut arguments: Vec<String> = [
            "create",
            "--pull=never",
            "--read-only",
            "--network=none",
            "--ipc=none",
            "--cgroupns=private",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges=true",
            "--user=1000:1000",
            "--cpus=2",
            "--memory=2147483648",
            "--memory-swap=2147483648",
            "--pids-limit=256",
            "--restart=no",
            "--log-driver=none",
            "--stop-timeout=2",
            "--no-healthcheck",
            "--workdir=/workspace",
            "--entrypoint=/bin/sh",
            "--tmpfs=/workspace:rw,exec,nosuid,nodev,size=1073741824,uid=1000,gid=1000,mode=0700",
            "--tmpfs=/home/made:rw,exec,nosuid,nodev,size=268435456,uid=1000,gid=1000,mode=0700",
            "--tmpfs=/tmp:rw,nosuid,nodev,size=134217728,mode=1777",
            "--env=HOME=/home/made",
            "--env=USER=made",
            "--env=LOGNAME=made",
            "--env=SHELL=/bin/sh",
            "--env=PATH=/usr/local/bin:/usr/bin:/bin",
            "--env=LANG=C.UTF-8",
            "--env=LC_ALL=C.UTF-8",
            "--env=TZ=UTC",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        arguments.extend([
            format!("--name={}", self.name),
            format!("--label={RESOURCE_LABEL}={}", self.run_id),
            self.image.clone(),
            "-c".into(),
            "exec sleep infinity".into(),
        ]);
        Ok(arguments)
    }

    pub fn validate_image(&self, image: &serde_json::Value) -> Result<()> {
        validate_image_digest(&self.image)?;
        ensure!(
            image["Os"].as_str() == Some("linux"),
            "Recipe tests require a Linux image"
        );
        ensure!(
            matches!(image["Architecture"].as_str(), Some("amd64" | "arm64")),
            "Unsupported test image architecture"
        );
        ensure!(
            image["RepoDigests"]
                .as_array()
                .is_some_and(|digests| digests
                    .iter()
                    .any(|digest| digest.as_str() == Some(&self.image))),
            "The local image does not match the reviewed immutable digest"
        );
        if let Some(volumes) = image["Config"]["Volumes"].as_object() {
            ensure!(
                volumes.is_empty(),
                "Test images cannot declare additional volumes"
            );
        } else {
            ensure!(
                image["Config"]["Volumes"].is_null(),
                "Invalid image volume configuration"
            );
        }
        if let Some(environment) = image["Config"]["Env"].as_array() {
            for variable in environment {
                let name = variable
                    .as_str()
                    .and_then(|value| value.split_once('=').map(|(name, _)| name))
                    .context("Invalid image environment")?;
                ensure!(
                    matches!(
                        name,
                        "HOME" | "USER" | "LOGNAME" | "SHELL" | "PATH" | "LANG" | "LC_ALL" | "TZ"
                    ),
                    "Test image declares an unsupported environment variable: {name}"
                );
            }
        } else {
            ensure!(
                image["Config"]["Env"].is_null(),
                "Invalid image environment"
            );
        }
        Ok(())
    }

    pub fn validate_container(
        &self,
        container: &serde_json::Value,
        reviewed_image_id: &str,
    ) -> Result<String> {
        ensure!(
            container["Name"].as_str() == Some(&format!("/{}", self.name))
                && container["Config"]["Labels"][RESOURCE_LABEL].as_str()
                    == Some(&self.run_id.to_string())
                && container["Config"]["Image"].as_str() == Some(&self.image)
                && container["Image"].as_str() == Some(reviewed_image_id),
            "Docker returned a container outside this test's ownership"
        );
        let identifier = container["Id"]
            .as_str()
            .context("Missing test container ID")?;
        ensure!(
            identifier.len() == 64 && identifier.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "Invalid test container ID"
        );
        let host = &container["HostConfig"];
        ensure!(
            host["ReadonlyRootfs"] == true
                && host["Privileged"] == false
                && host["NetworkMode"] == "none"
                && host["NanoCpus"] == 2_000_000_000_u64
                && host["Memory"] == 2_147_483_648_u64
                && host["MemorySwap"] == 2_147_483_648_u64
                && host["PidsLimit"] == 256
                && container["Config"]["User"] == "1000:1000",
            "Docker did not enforce the required test isolation or resource limits"
        );
        ensure!(
            host["CapDrop"]
                .as_array()
                .is_some_and(|values| values.iter().any(|value| value == "ALL")),
            "Docker did not drop all capabilities"
        );
        ensure!(
            host["SecurityOpt"]
                .as_array()
                .is_some_and(|values| values.iter().any(|value| value == "no-new-privileges=true")),
            "Docker did not enforce no-new-privileges"
        );
        for key in [
            "Binds",
            "Devices",
            "DeviceRequests",
            "CapAdd",
            "VolumesFrom",
        ] {
            ensure!(
                host[key].is_null() || host[key].as_array().is_some_and(Vec::is_empty),
                "Docker added unsupported test capability: {key}"
            );
        }
        for key in ["PidMode", "UTSMode"] {
            ensure!(
                host[key].as_str() == Some(""),
                "Docker joined an unreviewed namespace: {key}"
            );
        }
        ensure!(
            host["CgroupnsMode"] == "private",
            "Docker did not isolate cgroup visibility"
        );
        ensure!(host["IpcMode"] == "none", "Docker enabled shared IPC");
        let expected = self
            .create_arguments()?
            .into_iter()
            .filter_map(|argument| argument.strip_prefix("--tmpfs=").map(str::to_owned))
            .filter_map(|argument| {
                argument
                    .split_once(':')
                    .map(|(path, options)| (path.to_owned(), options.to_owned()))
            })
            .collect::<collections::HashMap<_, _>>();
        let actual = host["Tmpfs"]
            .as_object()
            .context("Docker omitted bounded test storage")?;
        ensure!(
            actual.len() == expected.len()
                && expected.iter().all(|(path, options)| actual
                    .get(path)
                    .and_then(serde_json::Value::as_str)
                    == Some(options.as_str())),
            "Docker changed the bounded test storage configuration"
        );
        ensure!(
            container["Mounts"]
                .as_array()
                .is_some_and(|mounts| mounts.iter().all(|mount| mount["Type"] == "tmpfs"
                    && mount["Destination"]
                        .as_str()
                        .is_some_and(|path| expected.contains_key(path)))),
            "Docker added an unreviewed test mount"
        );
        Ok(identifier.to_owned())
    }
}

fn validate_image_digest(image: &str) -> Result<()> {
    let Some((repository, digest)) = image.split_once("@sha256:") else {
        bail!("Select a local image by its immutable repository@sha256:digest");
    };
    ensure!(
        !repository.is_empty()
            && repository.len() <= 256
            && !repository.starts_with(['-', '/', '.'])
            && repository.bytes().all(|byte| byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || b"._-/:".contains(&byte))
            && !repository.contains("..")
            && digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "Invalid immutable test image digest"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[gpui::test]
    #[cfg(unix)]
    async fn doctor_docker_preflight_reuses_read_only_commands_and_reports_missing_inputs(
        cx: &mut gpui::TestAppContext,
    ) {
        use std::os::unix::fs::PermissionsExt;
        cx.background_executor.allow_parking();
        let directory = tempfile::tempdir().expect("CLI fixture");
        let executable = directory.path().join("docker");
        let marker = directory.path().join("unexpected-command");
        let engine = serde_json::json!({"ID": "fixture-engine", "OSType": "linux", "MemoryLimit": true, "SwapLimit": true, "CpuCfsPeriod": true, "CpuCfsQuota": true, "PidsLimit": true});
        let image = serde_json::json!([{"Id": format!("sha256:{}", "b".repeat(64)), "Os": "linux", "Architecture": "amd64", "RepoDigests": [image_name()], "Config": {"Volumes": null, "Env": ["PATH=/usr/bin:/bin"]}}]);
        std::fs::write(directory.path().join("engine"), engine.to_string())
            .expect("Engine fixture");
        std::fs::write(directory.path().join("image"), image.to_string()).expect("Image fixture");
        let script = format!(
            "#!/bin/sh\ncase \"$*\" in\n*'info --format='*) /bin/cat '{}' ;;\n*'image inspect '*) /bin/cat '{}' ;;\n*) /usr/bin/touch '{}'; exit 2 ;;\nesac\n",
            directory.path().join("engine").display(),
            directory.path().join("image").display(),
            marker.display()
        );
        std::fs::write(&executable, script).expect("Reviewed fixture executable");
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700))
            .expect("Executable permission");
        let client = ScanDockerClient::with_reviewed_cli(
            "unix:///tmp/doctor-fixture.sock".into(),
            executable,
        )
        .expect("Selected local fixture");
        let identity =
            ScanContainerIdentity::new(Uuid::new_v4(), image_name()).expect("Immutable image");
        let preflight = client
            .preflight(&identity, &cx.background_executor)
            .await
            .expect("Observed fixtures");
        assert_eq!(preflight.engine_id, "fixture-engine");
        assert!(
            !marker.exists(),
            "Preflight must not pull/create/start/remove"
        );
        std::fs::write(directory.path().join("image"), "[]").expect("Image unavailable");
        assert!(
            client
                .preflight(&identity, &cx.background_executor)
                .await
                .is_err()
        );
        let invalid_runtime = directory.path().join("runtime");
        std::fs::write(&invalid_runtime, "not a Linux Made runtime").expect("Invalid runtime");
        assert!(
            client
                .inspect_runtime(invalid_runtime, "fixture-version".into())
                .await
                .is_err()
        );
        assert!(!marker.exists());
    }

    fn image_name() -> String {
        format!("example.com/made-test@sha256:{}", "a".repeat(64))
    }

    fn scan_record_fixture(now: chrono::DateTime<chrono::Utc>) -> Result<RecipeScanRecord> {
        Ok(RecipeScanRecord {
            identity: ScanContainerIdentity::new(Uuid::new_v4(), image_name())?,
            preflight: ScanDockerPreflight {
                engine_id: "fixture-engine".into(),
                image_id: format!("sha256:{}", "b".repeat(64)),
                architecture: "arm64".into(),
            },
            endpoint: if cfg!(windows) {
                "npipe:////./pipe/docker_engine"
            } else {
                "unix:///tmp/made-fixture.sock"
            }
            .into(),
            runtime_digest: "c".repeat(64),
            policy: "offline-tools-provider-only-v1".into(),
            copied_files: Vec::new(),
            exclusions: Vec::new(),
            project_paths: util::path_list::PathList::new(&["/original"]).serialize(),
            worktree: "/original".into(),
            recipe: crate::recipes::Recipe::from_yaml(
                "version: 1\ntitle: Test\ndescription: Recovery\nprompt: Inspect\n",
            )?,
            parameters: Default::default(),
            provider: "fake".into(),
            model: "fake".into(),
            provider_endpoint: "https://example.com".into(),
            created_at: now,
            finished_at: None,
            session_id: Some("native-session".into()),
            outcome: None,
            observations: vec!["Step inspect: running".into()],
            detail: None,
            cleanup_pending: true,
            cleanup_error: None,
        })
    }

    #[test]
    fn recipe_scan_history_recovers_without_replay_and_validates_identity() -> Result<()> {
        let now = chrono::DateTime::parse_from_rfc3339("2026-09-19T10:00:00Z")?
            .with_timezone(&chrono::Utc);
        let record = scan_record_fixture(now)?;
        let legacy: crate::recipe_schedule::RecipeScheduleState =
            serde_json::from_str(r#"{"version":1,"schedules":[],"runs":[]}"#)?;
        assert!(legacy.scans.is_empty());
        let mut state = crate::recipe_schedule::RecipeScheduleState {
            scans: vec![record.clone()],
            ..Default::default()
        };
        state = serde_json::from_slice(&serde_json::to_vec(&state)?)?;
        state.recover(now + chrono::Duration::hours(1))?;
        let recovered = state.scans.first().expect("saved test");
        assert_eq!(recovered.outcome, Some(RecipeScanOutcome::Interrupted));
        assert_eq!(recovered.session_id.as_deref(), Some("native-session"));
        assert!(recovered.cleanup_pending);
        assert_eq!(recovered.observations, record.observations);
        assert!(state.schedules.is_empty() && state.runs.is_empty());
        let mut invalid = record.clone();
        invalid.identity.name = "unrelated-container".into();
        assert!(invalid.recover(now).is_err());
        let mut invalid = record.clone();
        invalid.observations = vec!["unbounded".into(); 257];
        assert!(invalid.recover(now).is_err());
        let mut finished = record;
        finished.outcome = Some(RecipeScanOutcome::Succeeded);
        finished.cleanup_error = Some("Docker unavailable".into());
        finished.recover(now)?;
        assert_eq!(finished.outcome, Some(RecipeScanOutcome::Succeeded));
        assert!(finished.cleanup_pending && finished.cleanup_error.is_some());
        Ok(())
    }

    #[gpui::test]
    async fn recipe_scan_cleanup_failure_is_durable_and_retryable(cx: &mut gpui::TestAppContext) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            crate::recipes::RecipeStore::init_global(cx);
        });
        cx.run_until_parked();
        let owner =
            cx.update(|cx| crate::recipes::RecipeStore::try_global(cx).expect("recipe owner"));
        let now = cx
            .background_executor
            .scheduler_executor()
            .scheduler()
            .clock()
            .utc_now();
        let mut record = scan_record_fixture(now).expect("record");
        let id = record.identity.run_id;
        record.endpoint = "tcp://unreviewed-engine:2375".into();
        record.outcome = Some(RecipeScanOutcome::Interrupted);
        owner
            .update(cx, |owner, cx| {
                owner.transact(
                    move |state, _| {
                        state.scans.push(record);
                        Ok(())
                    },
                    cx,
                )
            })
            .await
            .expect("saved interrupted fixture");
        for _ in 0..2 {
            owner
                .update(cx, |owner, cx| owner.retry_scan_cleanup(id, cx))
                .expect("explicit cleanup attempt");
            assert!(
                owner
                    .update(cx, |owner, cx| owner.retry_scan_cleanup(id, cx))
                    .is_err(),
                "no duplicate cleanup task"
            );
            owner
                .update(cx, |owner, _| {
                    owner.scan_task.take().expect("owned cleanup")
                })
                .await;
            owner.read_with(cx, |owner, _| {
                let record = owner.state.scans.first().expect("history");
                assert!(record.cleanup_pending && record.cleanup_error.is_some());
                assert_eq!(record.outcome, Some(RecipeScanOutcome::Interrupted));
                assert!(!owner.scan_cleanup_active);
            });
        }
        let database = cx
            .update(crate::ThreadsDatabase::connect)
            .await
            .expect("database");
        let (_, stored) = database
            .load_recipe_state()
            .await
            .expect("durable cleanup report");
        assert!(
            stored
                .scans
                .first()
                .expect("record")
                .cleanup_error
                .is_some()
        );
        assert!(stored.schedules.is_empty());
    }

    #[test]
    fn recipe_scan_copy_bounds_and_sensitive_exclusions() -> Result<()> {
        let project = tempfile::tempdir()?;
        std::fs::write(project.path().join("source.rs"), b"working tree changes")?;
        std::fs::write(project.path().join(".env"), b"SECRET=not copied")?;
        let paths = vec![
            "source.rs".into(),
            ".env".into(),
            ".zed/settings.json".into(),
        ];
        let cancellation = AtomicBool::new(true);
        assert!(
            PreparedScanCopy::capture_with_cancellation(project.path(), &paths, &cancellation)
                .is_err()
        );
        let copy = PreparedScanCopy::capture(project.path(), &paths)?;
        assert_eq!(copy.entries.len(), 1);
        assert_eq!(copy.exclusions.len(), 2);
        assert_eq!(
            copy.contents.first().map(Vec::as_slice),
            Some(b"working tree changes".as_slice())
        );
        assert!(
            PreparedScanCopy::capture_bounded(
                project.path(),
                &paths,
                2,
                MAX_COPY_BYTES,
                &AtomicBool::new(false)
            )
            .is_err()
        );
        assert!(
            PreparedScanCopy::capture_bounded(
                project.path(),
                &["source.rs".into()],
                1,
                2,
                &AtomicBool::new(false)
            )
            .is_err()
        );
        for invalid in [
            "../outside",
            "/outside",
            "C:outside",
            "a\\b",
            "line\nbreak",
            "",
        ] {
            assert!(
                PreparedScanCopy::capture(project.path(), &[invalid.into()]).is_err(),
                "{invalid}"
            );
        }
        assert!(
            PreparedScanCopy::capture(project.path(), &["source.rs".into(), "source.rs".into()])
                .is_err()
        );
        assert!(PreparedScanCopy::capture(project.path(), &["missing".into()]).is_err());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn recipe_scan_copy_rejects_links_and_special_files() -> Result<()> {
        let project = tempfile::tempdir()?;
        let outside = tempfile::tempdir()?;
        std::fs::write(outside.path().join("sentinel"), b"host secret")?;
        std::os::unix::fs::symlink(outside.path(), project.path().join("escape"))?;
        std::fs::write(project.path().join("file"), b"ordinary")?;
        std::os::unix::fs::symlink("file", project.path().join("alias"))?;
        let _socket = std::os::unix::net::UnixListener::bind(project.path().join("socket"))?;
        for path in ["escape/sentinel", "alias", "socket"] {
            assert!(
                PreparedScanCopy::capture(project.path(), &[path.into()]).is_err(),
                "{path}"
            );
        }
        assert_eq!(
            std::fs::read(outside.path().join("sentinel"))?,
            b"host secret"
        );
        Ok(())
    }

    #[gpui::test]
    async fn recipe_scan_archive_uses_reviewed_bytes() {
        async {
            use futures::{AsyncReadExt as _, StreamExt as _};
            let project = tempfile::tempdir()?;
            std::fs::write(project.path().join("file"), b"reviewed")?;
            let copy = PreparedScanCopy::capture(project.path(), &["file".into()])?;
            std::fs::write(project.path().join("file"), b"changed after review")?;
            let archive = async_tar::Archive::new(futures::io::Cursor::new(copy.archive().await?));
            let mut entries = archive.entries()?;
            let mut entry = entries.next().await.context("Missing captured file")??;
            assert_eq!(entry.path()?.as_os_str(), std::ffi::OsStr::new("file"));
            assert!(entry.header().entry_type().is_file());
            assert_eq!(entry.header().mode()?, 0o600);
            assert_eq!(entry.header().uid()?, 1000);
            let mut contents = String::new();
            entry.read_to_string(&mut contents).await?;
            assert_eq!(contents, "reviewed");
            assert!(entries.next().await.is_none());
            assert_eq!(
                std::fs::read(project.path().join("file"))?,
                b"changed after review"
            );
            anyhow::Ok(())
        }
        .await
        .expect("captured archive preserves reviewed bytes");
    }

    #[test]
    fn recipe_scan_engine_requires_enforceable_limits() -> Result<()> {
        let info = serde_json::json!({
            "OSType": "linux", "ID": "local-engine", "MemoryLimit": true,
            "SwapLimit": true, "CpuCfsPeriod": true, "CpuCfsQuota": true, "PidsLimit": true
        });
        assert_eq!(validate_engine(&info)?, "local-engine");
        for capability in [
            "MemoryLimit",
            "SwapLimit",
            "CpuCfsPeriod",
            "CpuCfsQuota",
            "PidsLimit",
        ] {
            let mut unsupported = info.clone();
            unsupported[capability] = serde_json::json!(false);
            assert!(validate_engine(&unsupported).is_err());
        }
        assert!(validate_engine(&serde_json::json!({})).is_err());
        Ok(())
    }

    #[test]
    fn recipe_scan_docker_endpoint_and_container_verification() -> Result<()> {
        for endpoint in [
            "tcp://localhost:2375",
            "ssh://host",
            "https://host",
            "unix://relative",
        ] {
            assert!(ScanDockerClient::from_reviewed_endpoint(endpoint.into()).is_err());
        }
        let endpoint = if cfg!(windows) {
            "npipe:////./pipe/docker_engine"
        } else {
            "unix:///tmp/recipe-test-docker.sock"
        };
        let docker_cli = tempfile::NamedTempFile::new()?;
        let client =
            ScanDockerClient::with_reviewed_cli(endpoint.into(), docker_cli.path().to_owned())?;
        assert_eq!(client.endpoint(), endpoint);
        assert_eq!(std::fs::read_dir(client.configuration_path())?.count(), 0);
        let configuration = client.configuration_path().to_owned();
        drop(client);
        assert!(!configuration.exists());
        let identity = ScanContainerIdentity::new(Uuid::new_v4(), image_name())?;
        let image_id = format!("sha256:{}", "b".repeat(64));
        let mut container = serde_json::json!({
            "Id": "c".repeat(64), "Name": format!("/{}", identity.name), "Image": image_id,
            "Config": {"Labels": {RESOURCE_LABEL: identity.run_id.to_string()}, "Image": image_name(), "User": "1000:1000"},
            "HostConfig": {
                "ReadonlyRootfs": true, "Privileged": false, "NetworkMode": "none", "IpcMode": "none", "PidMode": "", "UTSMode": "", "CgroupnsMode": "private",
                "NanoCpus": 2_000_000_000_u64, "Memory": 2_147_483_648_u64, "MemorySwap": 2_147_483_648_u64, "PidsLimit": 256,
                "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges=true"],
                "Tmpfs": {
                    "/workspace": "rw,exec,nosuid,nodev,size=1073741824,uid=1000,gid=1000,mode=0700",
                    "/home/made": "rw,exec,nosuid,nodev,size=268435456,uid=1000,gid=1000,mode=0700",
                    "/tmp": "rw,nosuid,nodev,size=134217728,mode=1777"
                }
            },
            "Mounts": []
        });
        identity.validate_container(&container, &image_id)?;
        for (key, value) in [
            ("Privileged", serde_json::json!(true)),
            ("NetworkMode", serde_json::json!("host")),
            ("PidMode", serde_json::json!("host")),
            ("MemorySwap", serde_json::json!(-1)),
            ("CapAdd", serde_json::json!(["SYS_ADMIN"])),
            ("Binds", serde_json::json!(["/home:/host"])),
            ("Tmpfs", serde_json::json!({"/workspace": "rw"})),
        ] {
            let mut unsafe_container = container.clone();
            unsafe_container["HostConfig"][key] = value;
            assert!(
                identity
                    .validate_container(&unsafe_container, &image_id)
                    .is_err(),
                "{key}"
            );
        }
        container["Config"]["Labels"][RESOURCE_LABEL] =
            serde_json::json!(Uuid::new_v4().to_string());
        assert!(identity.validate_container(&container, &image_id).is_err());
        Ok(())
    }

    #[test]
    fn recipe_scan_container_policy_rejects_implicit_capabilities() -> Result<()> {
        let identity = ScanContainerIdentity::new(Uuid::new_v4(), image_name())?;
        let image = serde_json::json!({
            "Os": "linux", "Architecture": "amd64", "RepoDigests": [image_name()],
            "Config": {"Volumes": null, "Env": ["PATH=/usr/bin:/bin"]}
        });
        identity.validate_image(&image)?;
        for (key, value) in [
            ("Volumes", serde_json::json!({"/data": {}})),
            ("Env", serde_json::json!(["PROVIDER_API_KEY=secret"])),
            ("Env", serde_json::json!(["BASH_ENV=/setup.sh"])),
        ] {
            let mut invalid = image.clone();
            invalid["Config"][key] = value;
            assert!(identity.validate_image(&invalid).is_err());
        }
        let mut wrong = image;
        wrong["RepoDigests"] = serde_json::json!([]);
        assert!(identity.validate_image(&wrong).is_err());
        for invalid in ["alpine:latest", "--privileged", "alpine@sha256:123", ""] {
            assert!(ScanContainerIdentity::new(Uuid::new_v4(), invalid.into()).is_err());
        }
        let arguments = identity.create_arguments()?;
        for restriction in [
            "--pull=never",
            "--read-only",
            "--network=none",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges=true",
            "--user=1000:1000",
            "--memory-swap=2147483648",
            "--pids-limit=256",
            "--no-healthcheck",
        ] {
            assert!(arguments.iter().any(|argument| argument == restriction));
        }
        assert!(
            !arguments
                .iter()
                .any(|argument| argument.starts_with("--volume")
                    || argument.starts_with("--mount")
                    || argument.contains("docker.sock"))
        );
        Ok(())
    }
    #[gpui::test]
    async fn recipe_scan_deadlines_use_the_executor_clock(cx: &mut gpui::TestAppContext) {
        let executor = cx.background_executor.clone();
        let task = cx.background_executor.spawn(async move {
            scan_deadline(
                futures::future::pending::<Result<()>>(),
                &executor,
                300,
                "setup",
            )
            .await
        });
        cx.run_until_parked();
        cx.executor()
            .advance_clock(std::time::Duration::from_secs(300));
        assert!(
            task.await
                .expect_err("setup must time out")
                .is::<ScanTimeout>()
        );
        let result = scan_deadline(async { Ok(42) }, &cx.background_executor, 900, "workflow")
            .await
            .expect("completed operation");
        assert_eq!(result, 42);
    }

    #[gpui::test]
    #[ignore = "requires explicitly selected local Docker endpoint and immutable image"]
    async fn recipe_scan_real_container_copy_and_cleanup(cx: &mut gpui::TestAppContext) {
        cx.background_executor.allow_parking();
        let endpoint =
            std::env::var("MADE_RECIPE_TEST_DOCKER_HOST").expect("explicit local endpoint");
        let image =
            std::env::var("MADE_RECIPE_TEST_IMAGE").expect("explicit immutable local image");
        let root = tempfile::tempdir().expect("source fixture");
        std::fs::write(root.path().join("sentinel"), "original").expect("source sentinel");
        let copy = PreparedScanCopy::capture(root.path(), &["sentinel".into()]).expect("held copy");
        let client = ScanDockerClient::from_reviewed_endpoint(endpoint).expect("local Docker");
        let identity = ScanContainerIdentity::new(Uuid::new_v4(), image).expect("test identity");
        let executor = &cx.background_executor;
        let reviewed = client
            .preflight(&identity, executor)
            .await
            .expect("reviewed local image");
        let result: Result<()> = async {
            client.create_verified(&identity, &reviewed, executor).await?;
            let identifier = client.start_with_copy(&identity, &reviewed, &copy, executor).await?;
            let script = "set -eu; test \"$(id -u)\" = 1000; test \"$(cat /workspace/sentinel)\" = original; echo changed > /workspace/sentinel; if (echo forbidden > /rootfs-sentinel) 2>/dev/null; then exit 1; fi; test ! -S /var/run/docker.sock; ! grep -q eth0 /proc/net/dev; grep -Eq '^NoNewPrivs:[[:space:]]+1$' /proc/self/status; cp /bin/true /home/made/execute-probe; chmod 700 /home/made/execute-probe; /home/made/execute-probe";
            client.command(&["exec".into(), "--user=1000:1000".into(), identifier, "/usr/bin/env".into(), "-i".into(), "PATH=/usr/bin:/bin".into(), "/bin/sh".into(), "-c".into(), script.into()], None, executor).await?;
            ensure!(std::fs::read_to_string(root.path().join("sentinel"))? == "original", "Original project changed");
            Ok(())
        }.await;
        let cleanup = client.remove_owned(&identity, &reviewed, executor).await;
        cleanup.expect("remove only this test's container");
        result.expect("real copy, executable storage and isolation checks");
        client
            .remove_owned(&identity, &reviewed, executor)
            .await
            .expect("idempotent cleanup");
    }
}

#[derive(Clone)]
pub struct RecipeScanRequest {
    pub project: gpui::Entity<project::Project>,
    pub agent: gpui::Entity<crate::NativeAgent>,
    pub worktree: PathBuf,
    pub recipe: crate::recipes::Recipe,
    pub parameters: collections::HashMap<String, String>,
    pub selected_files: Vec<PathBuf>,
    pub model: std::sync::Arc<dyn language_model::LanguageModel>,
    pub image: String,
    pub endpoint: String,
    pub server_binary: PathBuf,
}

pub struct PreparedRecipeScan {
    request: RecipeScanRequest,
    client: ScanDockerClient,
    identity: ScanContainerIdentity,
    preflight: ScanDockerPreflight,
    binary: remote::DockerRecipeTestOptions,
    copy: PreparedScanCopy,
    endpoint: http_client::Url,
}

impl PreparedRecipeScan {
    pub fn recipe(&self) -> &crate::recipes::Recipe {
        &self.request.recipe
    }
    pub fn parameters(&self) -> &collections::HashMap<String, String> {
        &self.request.parameters
    }
    pub fn files(&self) -> &[ScanCopyEntry] {
        self.copy.entries()
    }
    pub fn exclusions(&self) -> &[ScanCopyExclusion] {
        self.copy.exclusions()
    }
    pub fn image(&self) -> &str {
        &self.identity.image
    }
    pub fn server_digest(&self) -> &str {
        self.binary.server_sha256()
    }
    pub fn provider_endpoint(&self) -> &str {
        self.endpoint.as_str()
    }
    pub fn worktree(&self) -> &Path {
        &self.request.worktree
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecipeScanOutcome {
    Succeeded,
    Failed,
    Incomplete,
    Cancelled,
    TimedOut,
    Interrupted,
}

impl RecipeScanOutcome {
    pub fn label(self) -> &'static str {
        match self {
            Self::Succeeded => "Succeeded",
            Self::Failed => "Failed",
            Self::Incomplete => "Incomplete",
            Self::Cancelled => "Cancelled",
            Self::TimedOut => "Timed out",
            Self::Interrupted => "Interrupted",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeScanRecord {
    pub identity: ScanContainerIdentity,
    pub preflight: ScanDockerPreflight,
    pub endpoint: String,
    pub runtime_digest: String,
    pub policy: String,
    pub copied_files: Vec<ScanCopyEntry>,
    pub exclusions: Vec<ScanCopyExclusion>,
    pub project_paths: util::path_list::SerializedPathList,
    pub worktree: PathBuf,
    pub recipe: crate::recipes::Recipe,
    pub parameters: collections::HashMap<String, String>,
    pub provider: String,
    pub model: String,
    pub provider_endpoint: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub finished_at: Option<chrono::DateTime<chrono::Utc>>,
    pub session_id: Option<String>,
    pub outcome: Option<RecipeScanOutcome>,
    pub observations: Vec<String>,
    pub detail: Option<String>,
    pub cleanup_pending: bool,
    pub cleanup_error: Option<String>,
}

impl RecipeScanRecord {
    pub fn recover(&mut self, now: chrono::DateTime<chrono::Utc>) -> Result<()> {
        ensure!(
            self.identity
                == ScanContainerIdentity::new(self.identity.run_id, self.identity.image.clone())?,
            "Invalid saved test resource identity"
        );
        remote::DockerRecipeTestOptions::validate_endpoint(&self.endpoint)?;
        self.recipe.validate()?;
        ensure!(
            self.observations.len() <= 256
                && self.observations.iter().all(|entry| entry.len() <= 4096),
            "Saved test report exceeds its bounds"
        );
        if self.outcome.is_none() {
            self.outcome = Some(RecipeScanOutcome::Interrupted);
            self.finished_at = Some(now);
            self.detail = Some(
                "Made closed before this test finished. The workflow will not be replayed.".into(),
            );
        }
        Ok(())
    }
}

pub struct RecipeScanProgress {
    pub id: Uuid,
    pub title: String,
    pub phase: String,
    pub outcome: Option<RecipeScanOutcome>,
    pub detail: Option<String>,
    pub cleanup_error: Option<String>,
    pub session: Option<gpui::Entity<acp_thread::AcpThread>>,
    pub observations: std::rc::Rc<std::cell::RefCell<Vec<String>>>,
}

#[derive(Debug, thiserror::Error)]
#[error("Recipe test {0} timed out")]
struct ScanTimeout(&'static str);

fn validate_scan_source(request: &RecipeScanRequest, cx: &mut gpui::App) -> Result<()> {
    let project = request.project.read(cx);
    ensure!(
        project.is_local() && !project.is_disconnected(cx),
        "Choose an available local project for the test copy"
    );
    let worktree = project
        .visible_worktrees(cx)
        .find(|worktree| worktree.read(cx).abs_path().as_ref() == request.worktree.as_path())
        .context("The selected test worktree is unavailable")?;
    let worktree_id = worktree.read(cx).id();
    let store = project.worktree_store();
    ensure!(
        project::trusted_worktrees::TrustedWorktrees::try_get_global(cx).is_none_or(|trusted| {
            trusted.update(cx, |trusted, cx| trusted.can_trust(&store, worktree_id, cx))
        }),
        "The test source worktree is no longer trusted"
    );
    Ok(())
}

async fn scan_deadline<T>(
    operation: impl std::future::Future<Output = Result<T>>,
    executor: &gpui::BackgroundExecutor,
    seconds: u64,
    stage: &'static str,
) -> Result<T> {
    use futures::FutureExt as _;
    let operation = operation.fuse();
    let deadline = executor
        .timer(std::time::Duration::from_secs(seconds))
        .fuse();
    futures::pin_mut!(operation, deadline);
    futures::select_biased! {
        _ = deadline => Err(ScanTimeout(stage).into()),
        result = operation => result,
    }
}

impl crate::recipes::RecipeStore {
    pub(crate) fn recover_scans(&mut self, cx: &mut gpui::Context<Self>) {
        let records = self
            .state
            .scans
            .iter()
            .filter(|record| record.cleanup_pending)
            .cloned()
            .collect::<Vec<_>>();
        if !records.is_empty() {
            self.start_scan_cleanup(records, cx);
        }
    }

    pub fn retry_scan_cleanup(&mut self, id: Uuid, cx: &mut gpui::Context<Self>) -> Result<()> {
        ensure!(
            !self.scan_cleanup_active
                && self.scan.as_ref().is_none_or(|scan| scan.outcome.is_some()),
            "A test or cleanup is already active"
        );
        let record = self
            .state
            .scans
            .iter()
            .find(|record| record.identity.run_id == id)
            .context("Saved test is unavailable")?;
        ensure!(
            record.cleanup_pending && record.outcome.is_some(),
            "This test does not need recovery cleanup"
        );
        self.start_scan_cleanup(vec![record.clone()], cx);
        Ok(())
    }

    fn start_scan_cleanup(&mut self, records: Vec<RecipeScanRecord>, cx: &mut gpui::Context<Self>) {
        use util::ResultExt as _;
        self.scan_cleanup_active = true;
        self.scan_task = Some(cx.spawn(async move |this, cx| {
            for record in records {
                let id = record.identity.run_id;
                let cleanup = async {
                    let client = ScanDockerClient::from_reviewed_endpoint(record.endpoint.clone())?;
                    client
                        .remove_owned(
                            &record.identity,
                            &record.preflight,
                            cx.background_executor(),
                        )
                        .await
                }
                .await;
                let cleanup_error = cleanup.err().map(|error| format!("{error:#}"));
                let report_error = cleanup_error.clone();
                let saved = this.update(cx, |this, cx| {
                    this.transact(
                        move |state, _| {
                            let saved = state
                                .scans
                                .iter_mut()
                                .find(|saved| saved.identity.run_id == id)
                                .context("Saved test is unavailable")?;
                            saved.cleanup_pending = cleanup_error.is_some();
                            saved.cleanup_error = cleanup_error;
                            Ok(())
                        },
                        cx,
                    )
                });
                let saved = match saved {
                    Ok(task) => task.await,
                    Err(error) => Err(error),
                };
                if let Err(error) = saved {
                    this.update(cx, |this, cx| {
                        this.error = Some(format!("Could not save test cleanup result: {error:#}"));
                        cx.notify();
                    })
                    .log_err();
                    break;
                }
                this.update(cx, |this, cx| {
                    if let Some(scan) = &mut this.scan
                        && scan.id == id
                    {
                        scan.cleanup_error = report_error;
                    }
                    cx.notify();
                })
                .log_err();
            }
            this.update(cx, |this, cx| {
                this.scan_cleanup_active = false;
                cx.notify();
            })
            .log_err();
        }));
        cx.notify();
    }

    pub fn prepare_scan(
        &mut self,
        request: RecipeScanRequest,
        cx: &mut gpui::Context<Self>,
    ) -> gpui::Task<Result<PreparedRecipeScan>> {
        use gpui::AppContext as _;
        let validation = validate_scan_source(&request, cx)
            .and_then(|()| request.recipe.render(&request.parameters));
        if let Err(error) = validation {
            return gpui::Task::ready(Err(error));
        }
        {
            use language::File as _;
            let project = request.project.read(cx);
            for buffer in project.opened_buffers(cx) {
                let buffer = buffer.read(cx);
                if !buffer.is_dirty() {
                    continue;
                }
                if let Some(file) = project::File::from_dyn(buffer.file()) {
                    let root = project.worktree_for_id(file.worktree_id(cx), cx);
                    if root.is_some_and(|root| {
                        root.read(cx).abs_path().as_ref() == request.worktree.as_path()
                    }) && request
                        .selected_files
                        .iter()
                        .any(|path| path.to_str() == Some(file.path().as_unix_str()))
                    {
                        return gpui::Task::ready(Err(anyhow::anyhow!(
                            "Save or exclude the unsaved file {} before preparing the copy",
                            file.path()
                        )));
                    }
                }
            }
        }
        let server_version = match remote::DockerRecipeTestOptions::expected_version(cx) {
            Ok(version) => version,
            Err(error) => return gpui::Task::ready(Err(error)),
        };
        let endpoint = match request.model.network_endpoint(cx) {
            Ok(endpoint) => endpoint,
            Err(error) => return gpui::Task::ready(Err(error)),
        };
        if let Err(error) = request
            .model
            .clone()
            .with_endpoint_scope(endpoint.clone(), cx)
        {
            return gpui::Task::ready(Err(error));
        }
        cx.spawn(async move |_, cx| {
            let cancellation = std::sync::Arc::new(AtomicBool::new(false));
            let _cancel_copy = util::defer({
                let cancellation = cancellation.clone();
                move || cancellation.store(true, Ordering::Relaxed)
            });
            let root = request.worktree.clone();
            let files = request.selected_files.clone();
            let copy = cx
                .background_spawn(async move {
                    PreparedScanCopy::capture_with_cancellation(&root, &files, &cancellation)
                })
                .await?;
            let client = ScanDockerClient::from_reviewed_endpoint(request.endpoint.clone())?;
            let identity = ScanContainerIdentity::new(Uuid::new_v4(), request.image.clone())?;
            let preflight = client
                .preflight(&identity, cx.background_executor())
                .await?;
            let binary = remote::DockerRecipeTestOptions::prepare(
                request.endpoint.clone(),
                client.configuration_path().into(),
                client.docker_cli_path().into(),
                request.server_binary.clone(),
                server_version,
            )
            .await?;
            cx.update(|cx| validate_scan_source(&request, cx))?;
            Ok(PreparedRecipeScan {
                request,
                client,
                identity,
                preflight,
                binary,
                copy,
                endpoint,
            })
        })
    }

    pub fn cancel_scan(&mut self, cx: &mut gpui::Context<Self>) {
        let Some(cancellation) = self.scan_cancellation.take() else {
            return;
        };
        if cancellation.send(()).is_err() {
            log::debug!("Recipe test has already stopped");
            return;
        }
        if let Some(scan) = &mut self.scan {
            if scan.outcome.is_none() {
                scan.phase = "Cancelling and cleaning up".into();
            }
        }
        cx.notify();
    }

    pub fn run_scan(
        &mut self,
        prepared: PreparedRecipeScan,
        reviewed: bool,
        cx: &mut gpui::Context<Self>,
    ) -> Result<Uuid> {
        use acp_thread::AgentConnection as _;
        use futures::FutureExt as _;
        use util::ResultExt as _;
        ensure!(
            reviewed,
            "Review and explicitly authorize this isolated test before launch"
        );
        ensure!(
            self.scan
                .as_ref()
                .is_none_or(|scan| scan.outcome.is_some() && scan.cleanup_error.is_none()),
            "Finish the current test and its cleanup first"
        );
        ensure!(
            !self.scan_cleanup_active && !self.state.scans.iter().any(|scan| scan.cleanup_pending),
            "Finish cleanup of earlier tests before launching another"
        );
        validate_scan_source(&prepared.request, cx)?;
        let model = prepared
            .request
            .model
            .clone()
            .with_endpoint_scope(prepared.endpoint.clone(), cx)?;
        let id = prepared.identity.run_id;
        let project_paths = util::path_list::PathList::new(
            &prepared
                .request
                .project
                .read(cx)
                .visible_worktrees(cx)
                .map(|worktree| worktree.read(cx).abs_path().to_path_buf())
                .collect::<Vec<_>>(),
        )
        .serialize();
        let record = RecipeScanRecord {
            identity: prepared.identity.clone(),
            preflight: prepared.preflight.clone(),
            endpoint: prepared.client.endpoint().into(),
            runtime_digest: prepared.binary.server_sha256().into(),
            policy: "offline-tools-provider-only-v1".into(),
            copied_files: prepared.copy.entries().to_vec(),
            exclusions: prepared.copy.exclusions().to_vec(),
            project_paths: project_paths.clone(),
            worktree: prepared.request.worktree.clone(),
            recipe: prepared.request.recipe.clone(),
            parameters: prepared.request.parameters.clone(),
            provider: prepared.request.model.provider_id().to_string(),
            model: prepared.request.model.id().0.to_string(),
            provider_endpoint: prepared.endpoint.to_string(),
            created_at: cx
                .background_executor()
                .scheduler_executor()
                .scheduler()
                .clock()
                .utc_now(),
            finished_at: None,
            session_id: None,
            outcome: None,
            observations: Vec::new(),
            detail: None,
            cleanup_pending: true,
            cleanup_error: None,
        };
        let admission = self.transact(
            move |state, _| {
                ensure!(
                    !state.scans.iter().any(|scan| scan.identity.run_id == id),
                    "This test has already been admitted"
                );
                state.scans.push(record);
                Ok(())
            },
            cx,
        );
        let observations = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        self.scan = Some(RecipeScanProgress {
            id,
            title: prepared.request.recipe.title.clone(),
            phase: "Creating disposable container".into(),
            outcome: None,
            detail: None,
            cleanup_error: None,
            session: None,
            observations: observations.clone(),
        });
        let (sender, cancellation) = futures::channel::oneshot::channel();
        self.scan_cancellation = Some(sender);
        self.scan_task = Some(cx.spawn(async move |this, cx| {
            let executor = cx.background_executor().clone();
            if let Err(error) = admission.await {
                this.update(cx, |this, cx| {
                    this.scan_cancellation = None;
                    if let Some(scan) = &mut this.scan {
                        scan.phase = "Not started".into();
                        scan.outcome = Some(RecipeScanOutcome::Incomplete);
                        scan.detail = Some(format!("Test admission was not saved: {error:#}"));
                    }
                    cx.notify();
                }).log_err();
                return;
            }
            let mut remote_client = None;
            let mut native_session = None;
            let mut native_thread = None;
            let mut created = false;
            let connection =
                std::rc::Rc::new(crate::NativeAgentConnection(prepared.request.agent.clone()));
            let revoked = std::rc::Rc::new(std::cell::RefCell::new(None::<String>));
            let monitor_request = prepared.request.clone();
            let monitor_owner = this.clone();
            let monitor_revoked = revoked.clone();
            let monitor = cx.spawn(async move |cx| {
                loop {
                    cx.background_executor()
                        .timer(std::time::Duration::from_secs(1))
                        .await;
                    let validation = cx.update(|cx| validate_scan_source(&monitor_request, cx));
                    let mut stopped = validation.is_err();
                    if let Err(error) = validation {
                        *monitor_revoked.borrow_mut() = Some(format!("{error:#}"));
                    }
                    if let Err(error) = monitor_owner.update(cx, |owner, cx| {
                        let disconnected = owner.scan.as_ref().and_then(|scan| scan.session.as_ref()).is_some_and(|session| session.read(cx).project().read(cx).is_disconnected(cx));
                        if disconnected {
                            *monitor_revoked.borrow_mut() = Some("The isolated test connection was lost; execution will not reconnect or replay".into());
                            stopped = true;
                        }
                        if stopped {
                            owner.cancel_scan(cx);
                        }
                        cx.notify();
                    }) {
                        log::debug!("Recipe test owner closed: {error}");
                        break;
                    }
                    if stopped {
                        break;
                    }
                }
            });
            let result = {
                let operation = async {
                    let setup = async {
                        cx.update(|cx| validate_scan_source(&prepared.request, cx))?;
                        created = true;
                        prepared
                            .client
                            .create_verified(&prepared.identity, &prepared.preflight, &executor)
                            .await?;
                        let identifier = prepared
                            .client
                            .start_with_copy(
                                &prepared.identity,
                                &prepared.preflight,
                                &prepared.copy,
                                &executor,
                            )
                            .await?;
                        this.update(cx, |this, cx| {
                            if let Some(scan) = &mut this.scan {
                                scan.phase = "Connecting isolated Project".into();
                            }
                            cx.notify();
                        })?;
                        let options = remote::DockerConnectionOptions::for_recipe_test(
                            identifier,
                            prepared.binary.clone(),
                        )?;
                        let (_keep_connection, receiver) = futures::channel::oneshot::channel();
                        let remote = remote::connect_recipe_test(options, receiver, cx).await?;
                        remote_client = Some(remote.clone());
                        let project = cx.update(|cx| {
                            let source = prepared.request.project.read(cx);
                            project::Project::remote_recipe_test(
                                remote,
                                source.client(),
                                node_runtime::NodeRuntime::unavailable(),
                                source.user_store(),
                                source.languages().clone(),
                                source.fs().clone(),
                                cx,
                            )
                        });
                        project
                            .update(cx, |project, cx| {
                                project.find_or_create_worktree("/workspace", true, cx)
                            })
                            .await?;
                        Ok(project)
                    };
                    let project = scan_deadline(setup, &executor, 300, "setup").await?;
                    cx.update(|cx| validate_scan_source(&prepared.request, cx))?;
                    let session = prepared
                        .request
                        .agent
                        .update(cx, |agent, cx| agent.new_session(project.clone(), cx));
                    let session_id =
                        session.read_with(cx, |session, _| session.session_id().clone());
                    native_session = Some((session.clone(), session_id.clone()));
                    let thread = prepared
                        .request
                        .agent
                        .read_with(cx, |agent, _| {
                            agent
                                .sessions
                                .get(&session_id)
                                .map(|session| session.thread.clone())
                        })
                        .context("Native test session is unavailable")?;
                    native_thread = Some(thread.clone());
                    thread.update(cx, |thread, cx| {
                        thread.restrict_recipe_scan(cx)?;
                        let policy = thread.recipe_scan_policy.as_mut().context("Missing scan policy")?;
                        policy.observations = observations.clone();
                        policy.run_id = Some(id);
                        policy.project_paths = Some(project_paths.clone());
                        thread.set_model(model, cx);
                        anyhow::Ok(())
                    })?;
                    this.update(cx, |this, cx| {
                        if let Some(scan) = &mut this.scan {
                            scan.session = Some(session.clone());
                            scan.phase = "Running recipe in disposable project".into();
                        }
                        cx.notify();
                    })?;
                    crate::recipes::persist_workflow_session(&connection, &thread, &project, cx).await?;
                    let recipe = prepared.request.recipe.clone();
                    let parameters = prepared.request.parameters.clone();
                    let run_connection = connection.clone();
                    let run = session.update(cx, |session, cx| {
                        session.run_turn(cx, async move |_, cx| {
                            cx.update(|cx| {
                                crate::recipes::execute_recipe(
                                    &run_connection,
                                    session_id,
                                    acp_thread::ClientUserMessageId::new(),
                                    &recipe,
                                    &parameters,
                                    Vec::new(),
                                    cx,
                                )
                            })
                            .await
                        })
                    });
                    let response =
                        scan_deadline(run, &executor, 900, "workflow").await?;
                    if let Some(reason) = thread.read_with(cx, |thread, _| {
                        thread
                            .recipe_scan_policy
                            .as_ref()
                            .and_then(RecipeScanSessionPolicy::blocked_dependency)
                    }) {
                        bail!("Blocked dependency: {reason}");
                    }
                    match response {
                        Some(response)
                            if response.stop_reason
                                == agent_client_protocol::schema::v1::StopReason::EndTurn =>
                        {
                            let failed = if prepared.request.recipe.version == 1 {
                                thread
                                    .read_with(cx, |thread, _| thread.recipe_step_output(0))?
                                    .1
                            } else {
                                false
                            };
                            Ok(if failed {
                                RecipeScanOutcome::Failed
                            } else {
                                RecipeScanOutcome::Succeeded
                            })
                        }
                        None => Ok(RecipeScanOutcome::Cancelled),
                        Some(response)
                            if response.stop_reason
                                == agent_client_protocol::schema::v1::StopReason::Cancelled =>
                        {
                            Ok(RecipeScanOutcome::Cancelled)
                        }
                        Some(_) => Ok(RecipeScanOutcome::Failed),
                    }
                }
                .fuse();
                let cancellation = cancellation.fuse();
                futures::pin_mut!(operation, cancellation);
                futures::select_biased! {
                    _ = cancellation => Ok(RecipeScanOutcome::Cancelled),
                    result = operation => result,
                }
            };
            drop(monitor);
            let revoked_reason = revoked.borrow_mut().take();
            let was_revoked = revoked_reason.is_some();
            let result = match revoked_reason {
                Some(reason) => Err(anyhow::anyhow!(reason)),
                None => result,
            };
            this.update(cx, |this, cx| {
                if let Some(scan) = &mut this.scan {
                    scan.phase = "Cleaning up disposable container".into();
                }
                cx.notify();
            })
            .log_err();
            if let Some((session, _)) = &native_session {
                session.update(cx, |session, cx| session.cancel(cx)).await;
            }
            if let Some(thread) = &native_thread {
                thread.update(cx, |thread, cx| {
                    if let Some(policy) = &mut thread.recipe_scan_policy { policy.read_only = true; }
                    cx.notify();
                });
            }
            let mut session_save_error = if let Some(thread) = &native_thread {
                let project = thread.read_with(cx, |thread, _| thread.project().clone());
                scan_deadline(crate::recipes::persist_workflow_session(&connection, thread, &project, cx), &executor, 30, "session persistence").await.err().map(|error| format!("Could not save test transcript: {error:#}"))
            } else { None };
            if let Some(remote) = remote_client {
                match remote.update(cx, |remote, cx| remote.shutdown_recipe_test(cx)) {
                    Ok(shutdown) => shutdown.await,
                    Err(error) => log::error!("Could not stop recipe test connection: {error:#}"),
                }
            }
            let cleanup = if created {
                prepared
                    .client
                    .remove_owned(&prepared.identity, &prepared.preflight, &executor)
                    .await
            } else {
                Ok(())
            };
            if let Some((_, session_id)) = native_session {
                if let Err(error) = cx.update(|cx| connection.close_session(&session_id, cx)).await {
                    session_save_error = Some(format!("Could not close and save test session: {error:#}"));
                }
            }
            let interruption = native_thread.as_ref().and_then(|thread| {
                thread.read_with(cx, |thread, _| {
                    thread
                        .recipe_scan_policy
                        .as_ref()
                        .and_then(|policy| policy.interruption.get())
                })
            });
            let workflow_started = native_thread.is_some();
            let blocked = native_thread.as_ref().is_some_and(|thread| {
                thread.read_with(cx, |thread, _| {
                    thread
                        .recipe_scan_policy
                        .as_ref()
                        .and_then(RecipeScanSessionPolicy::blocked_dependency)
                        .is_some()
                })
            });
            drop(native_thread);
            this.update(cx, |this, cx| {
                this.scan_cancellation = None;
                if let Some(scan) = &mut this.scan {
                    scan.phase = "Finished".into();
                    scan.outcome = Some(interruption.unwrap_or_else(|| match &result {
                        Ok(outcome) => *outcome,
                        Err(error)
                            if error.is::<ScanTimeout>()
                                || error.is::<crate::recipes::RecipeWorkflowTimeout>() =>
                        {
                            RecipeScanOutcome::TimedOut
                        }
                        Err(error)
                            if was_revoked
                                || blocked
                                || !workflow_started
                                || crate::recipes::needs_attention(error) =>
                        {
                            RecipeScanOutcome::Incomplete
                        }
                        Err(_) => RecipeScanOutcome::Failed,
                    }));
                    scan.detail = result.err().map(|error| format!("{error:#}"));
                    if let Some(error) = session_save_error {
                        scan.outcome = Some(RecipeScanOutcome::Incomplete);
                        scan.detail = Some(error);
                    }
                    scan.cleanup_error = cleanup.err().map(|error| format!("{error:#}"));
                    let now = cx.background_executor().scheduler_executor().scheduler().clock().utc_now();
                    let mut observations = scan.observations.borrow_mut();
                    for step in &prepared.request.recipe.steps {
                        let prefix = format!("Step {}:", step.id);
                        if observations.iter().rev().find(|entry| entry.contains(&prefix)).is_some_and(|entry| entry.ends_with(": running")) {
                            observations.push(format!("{now} · {prefix} {}", scan.outcome.map(RecipeScanOutcome::label).unwrap_or("Interrupted")));
                        }
                    }
                    if prepared.request.recipe.version == 1 {
                        observations.push(format!("{now} · Prompt: {}", scan.outcome.map(RecipeScanOutcome::label).unwrap_or("Interrupted")));
                    }
                }
                cx.notify();
            })
            .log_err();
            let saved = this.update(cx, |this, cx| {
                let scan = this.scan.as_ref().context("Test result is unavailable")?;
                let outcome = scan.outcome;
                let detail = scan.detail.clone();
                let cleanup_error = scan.cleanup_error.clone();
                let observations = scan.observations.borrow().clone();
                Ok::<_, anyhow::Error>(this.transact(move |state, now| {
                    let record = state.scans.iter_mut().find(|record| record.identity.run_id == id).context("Saved test is unavailable")?;
                    record.outcome = outcome;
                    record.finished_at = Some(now);
                    record.detail = detail;
                    record.cleanup_pending = cleanup_error.is_some();
                    record.cleanup_error = cleanup_error;
                    record.observations = observations;
                    Ok(())
                }, cx))
            });
            let saved = match saved { Ok(Ok(task)) => task.await, Ok(Err(error)) | Err(error) => Err(error) };
            if let Err(error) = saved {
                this.update(cx, |this, cx| {
                    let message = format!("Could not save test result: {error:#}");
                    this.error = Some(message.clone());
                    if let Some(scan) = &mut this.scan {
                        scan.outcome = Some(RecipeScanOutcome::Incomplete);
                        scan.detail = Some(message);
                    }
                    cx.notify();
                }).log_err();
            }
        }));
        cx.notify();
        Ok(id)
    }
}
