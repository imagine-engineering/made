use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use async_trait::async_trait;
use collections::HashMap;
use parking_lot::Mutex;
use release_channel::{AppCommitSha, AppVersion, ReleaseChannel};
use semver::Version as SemanticVersion;
use std::collections::BTreeMap;
use std::time::Instant;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use util::ResultExt;
use util::command::Stdio;
use util::redact::{is_valid_environment_name, redact_command};
use util::shell::ShellKind;
use util::{
    paths::{PathStyle, RemotePathBuf},
    rel_path::RelPath,
};

use futures::channel::mpsc::{Sender, UnboundedReceiver, UnboundedSender};
use futures::{AsyncReadExt as _, AsyncWriteExt as _, FutureExt as _};
use gpui::{App, AppContext, AsyncApp, Task};
use rpc::proto::Envelope;
use sha2::{Digest as _, Sha256};

use crate::{
    RemoteArch, RemoteClientDelegate, RemoteConnection, RemoteConnectionOptions, RemoteOs,
    RemotePlatform,
    remote_client::{CommandTemplate, Interactive},
    transport::parse_platform,
};

#[derive(
    Debug,
    Default,
    Clone,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct DockerConnectionOptions {
    pub name: String,
    pub container_id: String,
    pub remote_user: String,
    pub upload_binary_over_docker_exec: bool,
    pub use_podman: bool,
    pub remote_env: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub isolated_recipe_test: Option<DockerRecipeTestOptions>,
}

const RECIPE_TEST_SERVER: &str = "/home/made/zed-remote-server";
const RECIPE_TEST_SERVER_LIMIT: u64 = 192 * 1024 * 1024;
const RECIPE_TEST_ENV: &[&str] = &[
    "PATH=/usr/local/bin:/usr/bin:/bin",
    "HOME=/home/made",
    "USER=made",
    "LOGNAME=made",
    "SHELL=/bin/sh",
    "LANG=C.UTF-8",
    "LC_ALL=C.UTF-8",
    "TZ=UTC",
    "PAGER=",
    "GIT_PAGER=cat",
];
const DOCKER_ENVIRONMENT: &[&str] = &[
    "DOCKER_CONTEXT",
    "DOCKER_HOST",
    "DOCKER_TLS_VERIFY",
    "DOCKER_CERT_PATH",
    "DOCKER_API_VERSION",
    "DOCKER_CUSTOM_HEADERS",
    "DOCKER_CONFIG",
];

#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(deny_unknown_fields)]
pub struct DockerRecipeTestOptions {
    endpoint: String,
    configuration_directory: PathBuf,
    docker_cli: PathBuf,
    server_binary: PathBuf,
    server_sha256: String,
    expected_version: String,
}

impl DockerRecipeTestOptions {
    pub fn expected_version(cx: &App) -> Result<String> {
        let channel = ReleaseChannel::global(cx);
        let mut version = AppVersion::global(cx);
        if matches!(channel, ReleaseChannel::Stable | ReleaseChannel::Preview) {
            anyhow::ensure!(
                version != SemanticVersion::new(0, 0, 0),
                "Made version is unavailable"
            );
            version.build = semver::BuildMetadata::EMPTY;
            return Ok(version.to_string());
        }
        let commit = AppCommitSha::try_global(cx)
            .context("Made build identity is unavailable; use a build with commit metadata")?;
        let commit = commit.full();
        let prefix = format!("{}.", channel.dev_name());
        let metadata = version
            .build
            .as_str()
            .strip_prefix(&prefix)
            .unwrap_or_default();
        let suffix = format!(".{commit}");
        Ok(
            match metadata
                .strip_suffix(&suffix)
                .filter(|build| !build.is_empty())
            {
                Some(build) => format!("{build}+{commit}"),
                None => commit,
            },
        )
    }

    pub fn local_docker_cli() -> Result<PathBuf> {
        Ok(which::which("docker")
            .context("Docker CLI is not installed")?
            .canonicalize()?)
    }

    pub fn validate_endpoint(endpoint: &str) -> Result<()> {
        let local = if cfg!(windows) {
            endpoint == "npipe:////./pipe/docker_engine"
        } else {
            endpoint.strip_prefix("unix://").is_some_and(|path| {
                Path::new(path).is_absolute()
                    && !path.chars().any(char::is_control)
                    && !path.contains(['?', '#'])
            })
        };
        anyhow::ensure!(local, "Recipe tests require a reviewed local Docker socket");
        Ok(())
    }

    pub fn local_command(
        docker_cli: &Path,
        endpoint: &str,
        configuration: &Path,
    ) -> Result<util::command::Command> {
        Self::validate_endpoint(endpoint)?;
        anyhow::ensure!(
            docker_cli.is_absolute() && docker_cli.is_file(),
            "The reviewed Docker CLI is unavailable"
        );
        anyhow::ensure!(
            configuration.is_absolute(),
            "Docker test configuration must be absolute"
        );
        let metadata = std::fs::symlink_metadata(configuration)?;
        anyhow::ensure!(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "Docker test configuration must be a private directory"
        );
        anyhow::ensure!(
            std::fs::read_dir(configuration)?
                .next()
                .transpose()?
                .is_none(),
            "Docker test configuration changed after review"
        );
        let mut command = util::command::new_command(docker_cli);
        command
            .env_clear()
            .arg("--host")
            .arg(endpoint)
            .arg("--config")
            .arg(configuration);
        #[cfg(windows)]
        if let Some(system_root) = std::env::var_os("SystemRoot") {
            command.env("SystemRoot", system_root);
        }
        Ok(command)
    }

    pub async fn prepare(
        endpoint: String,
        configuration_directory: PathBuf,
        docker_cli: PathBuf,
        server_binary: PathBuf,
        expected_version: String,
    ) -> Result<Self> {
        Self::local_command(&docker_cli, &endpoint, &configuration_directory)?;
        anyhow::ensure!(
            !expected_version.is_empty()
                && expected_version.len() <= 128
                && !expected_version.chars().any(char::is_whitespace),
            "Made runtime version is unavailable"
        );
        let source = server_binary.clone();
        let binary = smol::unblock(move || read_recipe_test_binary(&source)).await?;
        Ok(Self {
            endpoint,
            configuration_directory,
            docker_cli,
            server_binary,
            server_sha256: format!("{:x}", Sha256::digest(&binary)),
            expected_version,
        })
    }

    pub fn server_sha256(&self) -> &str {
        &self.server_sha256
    }
    pub fn server_binary(&self) -> &Path {
        &self.server_binary
    }

    fn command(&self) -> Result<util::command::Command> {
        Self::local_command(
            &self.docker_cli,
            &self.endpoint,
            &self.configuration_directory,
        )
    }

    fn validate_connection(&self, options: &DockerConnectionOptions) -> Result<()> {
        self.command()?;
        anyhow::ensure!(
            !options.use_podman
                && options.remote_user == "1000:1000"
                && options.remote_env.is_empty()
                && options.upload_binary_over_docker_exec,
            "Recipe test connections require the fixed nonroot offline policy"
        );
        anyhow::ensure!(
            options.container_id.len() == 64
                && options
                    .container_id
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
            "Recipe tests require the exact created Docker container identity"
        );
        Ok(())
    }
}

fn read_recipe_test_binary(path: &Path) -> Result<Vec<u8>> {
    use std::io::{Read as _, Seek as _};
    anyhow::ensure!(
        path.is_absolute(),
        "Choose an existing absolute Linux Made runtime path"
    );
    anyhow::ensure!(
        !std::fs::symlink_metadata(path)?.file_type().is_symlink(),
        "The selected runtime must not be a symlink"
    );
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut file = options.open(path)?;
    let metadata = file.metadata()?;
    anyhow::ensure!(
        metadata.is_file() && metadata.len() <= RECIPE_TEST_SERVER_LIMIT,
        "The selected runtime must be a regular file no larger than 192 MiB"
    );
    let mut magic = [0; 2];
    file.read_exact(&mut magic)?;
    file.rewind()?;
    let reader: Box<dyn std::io::Read> = if magic == [0x1f, 0x8b] {
        Box::new(flate2::read::GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let mut binary = Vec::new();
    reader
        .take(RECIPE_TEST_SERVER_LIMIT + 1)
        .read_to_end(&mut binary)?;
    anyhow::ensure!(
        binary.len() as u64 <= RECIPE_TEST_SERVER_LIMIT,
        "The decoded runtime exceeds 192 MiB"
    );
    anyhow::ensure!(
        binary.get(..4) == Some(b"\x7fELF")
            && binary.get(4) == Some(&2)
            && binary.get(5) == Some(&1)
            && matches!(binary.get(7), Some(0 | 3)),
        "The selected runtime must be a 64-bit Linux ELF executable"
    );
    anyhow::ensure!(
        matches!(binary.get(18..20), Some([62, 0] | [183, 0])),
        "Unsupported Linux runtime architecture"
    );
    Ok(binary)
}

impl DockerConnectionOptions {
    pub fn for_recipe_test(container_id: String, policy: DockerRecipeTestOptions) -> Result<Self> {
        let options = Self {
            name: "Made recipe test".into(),
            container_id,
            remote_user: "1000:1000".into(),
            upload_binary_over_docker_exec: true,
            use_podman: false,
            remote_env: BTreeMap::new(),
            isolated_recipe_test: Some(policy),
        };
        options
            .isolated_recipe_test
            .as_ref()
            .context("Missing test policy")?
            .validate_connection(&options)?;
        Ok(options)
    }
}

struct RecipeTestDelegate;

impl RemoteClientDelegate for RecipeTestDelegate {
    fn ask_password(
        &self,
        _prompt: String,
        sender: futures::channel::oneshot::Sender<askpass::EncryptedPassword>,
        _cx: &mut AsyncApp,
    ) {
        drop(sender);
    }

    fn get_download_url(
        &self,
        _platform: RemotePlatform,
        _channel: ReleaseChannel,
        _version: Option<SemanticVersion>,
        _cx: &mut AsyncApp,
    ) -> Task<Result<Option<String>>> {
        Task::ready(Err(anyhow!("Recipe tests cannot download a remote server")))
    }

    fn download_server_binary_locally(
        &self,
        _platform: RemotePlatform,
        _channel: ReleaseChannel,
        _version: Option<SemanticVersion>,
        _cx: &mut AsyncApp,
    ) -> Task<Result<PathBuf>> {
        Task::ready(Err(anyhow!(
            "Choose an already available Linux Made server binary"
        )))
    }

    fn set_status(&self, _status: Option<&str>, _cx: &mut AsyncApp) {}
}

pub async fn connect_recipe_test(
    options: DockerConnectionOptions,
    cancellation: futures::channel::oneshot::Receiver<()>,
    cx: &mut AsyncApp,
) -> Result<gpui::Entity<crate::RemoteClient>> {
    options
        .isolated_recipe_test
        .as_ref()
        .context("Missing isolated recipe policy")?
        .validate_connection(&options)?;
    let delegate: Arc<dyn RemoteClientDelegate> = Arc::new(RecipeTestDelegate);
    let connection = crate::remote_client::connect(
        RemoteConnectionOptions::Docker(options),
        delegate.clone(),
        cx,
    )
    .await?;
    cx.update(|cx| {
        crate::RemoteClient::new(
            crate::ConnectionIdentifier::setup(),
            connection,
            cancellation,
            delegate,
            cx,
        )
    })
    .await?
    .context("Recipe test connection was cancelled")
}

pub(crate) struct DockerExecConnection {
    proxy_process: Mutex<Option<u32>>,
    remote_dir_for_server: String,
    remote_binary_relpath: Option<Arc<RelPath>>,
    connection_options: DockerConnectionOptions,
    remote_platform: Option<RemotePlatform>,
    os_version: Option<String>,
    path_style: Option<PathStyle>,
    shell: String,
    executor: gpui::BackgroundExecutor,
}

impl DockerExecConnection {
    pub async fn new(
        connection_options: DockerConnectionOptions,
        delegate: Arc<dyn RemoteClientDelegate>,
        cx: &mut AsyncApp,
    ) -> Result<Self> {
        if let Some(policy) = &connection_options.isolated_recipe_test {
            policy.validate_connection(&connection_options)?;
        }
        let mut this = Self {
            proxy_process: Mutex::new(None),
            remote_dir_for_server: "/".to_string(),
            remote_binary_relpath: None,
            connection_options,
            remote_platform: None,
            os_version: None,
            path_style: None,
            shell: "sh".to_owned(),
            executor: cx.background_executor().clone(),
        };
        let (release_channel, version, commit) = cx.update(|cx| {
            (
                ReleaseChannel::global(cx),
                AppVersion::global(cx),
                AppCommitSha::try_global(cx),
            )
        });
        let remote_platform = this.check_remote_platform().await?;

        this.path_style = match remote_platform.os {
            RemoteOs::Windows => Some(PathStyle::Windows),
            _ => Some(PathStyle::Unix),
        };

        this.remote_platform = Some(remote_platform);
        if this.connection_options.isolated_recipe_test.is_some() {
            anyhow::ensure!(
                matches!(remote_platform.os, RemoteOs::Linux),
                "Recipe test runtimes require Linux containers"
            );
            this.shell = "/bin/sh".into();
            this.remote_dir_for_server = "/home/made".into();
            this.install_recipe_test_runtime().await?;
            this.remote_binary_relpath = Some(RelPath::from_unix_str("zed-remote-server")?.into());
            return Ok(this);
        }

        log::info!("Remote platform discovered: {:?}", this.remote_platform);

        this.os_version = this.discover_os_version(remote_platform.os).await;
        log::info!("Remote OS version discovered: {:?}", this.os_version);

        this.shell = this.discover_shell().await;
        log::info!("Remote shell discovered: {}", this.shell);

        this.remote_dir_for_server = this.docker_user_home_dir().await?.trim().to_string();

        this.remote_binary_relpath = Some(
            this.ensure_server_binary(
                &delegate,
                release_channel,
                version,
                &this.remote_dir_for_server,
                commit,
                cx,
            )
            .await?,
        );

        Ok(this)
    }

    fn recipe_test_exec_arguments(
        &self,
        program: &str,
        directory: Option<&str>,
        env: &HashMap<String, String>,
        arguments: &[impl AsRef<str>],
        interactive: Interactive,
    ) -> Result<Vec<String>> {
        let policy = self
            .connection_options
            .isolated_recipe_test
            .as_ref()
            .context("Missing recipe test policy")?;
        policy.validate_connection(&self.connection_options)?;
        anyhow::ensure!(
            env.is_empty(),
            "Tool environment credentials and overrides are unavailable in recipe tests"
        );
        anyhow::ensure!(
            !program.is_empty() && !program.starts_with('-'),
            "Invalid container program"
        );
        let mut result = vec!["exec".into(), "--user=1000:1000".into()];
        if let Some(directory) = directory {
            anyhow::ensure!(
                directory.starts_with('/') && !directory.chars().any(char::is_control),
                "Recipe test working directories must be absolute container paths"
            );
            result.extend(["--workdir".into(), directory.into()]);
        }
        result.push(
            match interactive {
                Interactive::Yes => "-it",
                Interactive::No => "-i",
            }
            .into(),
        );
        result.extend([
            self.connection_options.container_id.clone(),
            "/usr/bin/env".into(),
            "-i".into(),
        ]);
        result.extend(RECIPE_TEST_ENV.iter().map(|value| (*value).into()));
        result.push(program.into());
        result.extend(
            arguments
                .iter()
                .map(|argument| argument.as_ref().to_owned()),
        );
        Ok(result)
    }

    async fn recipe_test_output(
        &self,
        mut command: util::command::Command,
        input: Option<Vec<u8>>,
    ) -> Result<String> {
        command
            .kill_on_drop(true)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command
            .spawn()
            .context("Starting the reviewed local Docker CLI")?;
        let mut stdin = child.stdin.take().context("Missing Docker input")?;
        let stdout = child.stdout.take().context("Missing Docker output")?;
        let stderr = child.stderr.take().context("Missing Docker diagnostics")?;
        let operation = async {
            let send = async move {
                if let Some(input) = input {
                    stdin.write_all(&input).await?;
                }
                drop(stdin);
                anyhow::Ok(())
            };
            let read_output = async move {
                let mut output = Vec::new();
                stdout.take(65537).read_to_end(&mut output).await?;
                anyhow::ensure!(
                    output.len() <= 65536,
                    "Docker runtime output exceeded 64 KiB"
                );
                anyhow::Ok(output)
            };
            let read_error = async move {
                let mut error = Vec::new();
                stderr.take(65537).read_to_end(&mut error).await?;
                anyhow::ensure!(
                    error.len() <= 65536,
                    "Docker runtime diagnostics exceeded 64 KiB"
                );
                anyhow::Ok(error)
            };
            let ((), output, error) = futures::try_join!(send, read_output, read_error)?;
            let status = child.status().await?;
            anyhow::ensure!(
                status.success(),
                "Recipe test runtime operation failed: {}",
                String::from_utf8_lossy(&error)
            );
            Ok(String::from_utf8(output)?)
        }
        .fuse();
        let deadline = self
            .executor
            .timer(std::time::Duration::from_secs(30))
            .fuse();
        futures::pin_mut!(operation, deadline);
        futures::select_biased! {
            _ = deadline => anyhow::bail!("Recipe test runtime operation timed out; container cleanup is required"),
            result = operation => result,
        }
    }

    async fn install_recipe_test_runtime(&self) -> Result<()> {
        let policy = self
            .connection_options
            .isolated_recipe_test
            .as_ref()
            .context("Missing recipe test policy")?;
        let source = policy.server_binary.clone();
        let binary = smol::unblock(move || read_recipe_test_binary(&source)).await?;
        anyhow::ensure!(
            format!("{:x}", Sha256::digest(&binary)) == policy.server_sha256,
            "The selected runtime changed after review"
        );
        let expected_arch = match self
            .remote_platform
            .context("Missing container platform")?
            .arch
        {
            RemoteArch::X86_64 => [62, 0],
            RemoteArch::Aarch64 => [183, 0],
        };
        anyhow::ensure!(
            binary.get(18..20) == Some(expected_arch.as_slice()),
            "The selected runtime does not match the container architecture"
        );
        let arguments = self.recipe_test_exec_arguments(
            "/bin/sh",
            Some("/home/made"),
            &HashMap::default(),
            &[
                "-c",
                "umask 077; set -C; cat > zed-remote-server && chmod 700 zed-remote-server",
            ],
            Interactive::No,
        )?;
        let mut command = policy.command()?;
        command.args(arguments);
        self.recipe_test_output(command, Some(binary)).await?;
        let capabilities = self
            .run_docker_exec(
                RECIPE_TEST_SERVER,
                Some("/home/made"),
                &HashMap::default(),
                &["version", "--isolated-recipe-test"],
            )
            .await?;
        anyhow::ensure!(
            capabilities.trim() == format!("made-recipe-isolation-v1 {}", policy.expected_version),
            "The selected runtime does not match this Made version and its isolated test policy"
        );
        Ok(())
    }

    fn docker_cli(&self) -> &str {
        if self.connection_options.use_podman {
            "podman"
        } else {
            "docker"
        }
    }

    /// Run a shell command inside the container and reliably extract its output
    /// using unique delimiters, so that shell initialization noise (e.g. from
    /// BASH_ENV or .bashrc) does not corrupt the result.
    async fn run_docker_exec_delimited(&self, script: &str) -> Result<String> {
        const MARKER: &str = "=====ZED_DELIM_7f3a9c=====";
        let wrapped =
            format!("printf '{MARKER}'; {script}; __exit=$?; printf '{MARKER}'; exit $__exit");
        let output = self
            .run_docker_exec("sh", None, &Default::default(), &["-c", &wrapped])
            .await?;
        let start = output.find(MARKER).map(|i| i + MARKER.len()).unwrap_or(0);
        let end = output[start..]
            .find(MARKER)
            .map(|i| start + i)
            .unwrap_or(output.len());
        Ok(output[start..end].to_string())
    }

    async fn discover_shell(&self) -> String {
        let default_shell = "sh";
        match self.run_docker_exec_delimited("echo $SHELL").await {
            Ok(shell) => match shell.trim() {
                "" => {
                    log::info!("$SHELL is not set, checking passwd for user");
                }
                shell => {
                    return shell.to_owned();
                }
            },
            Err(e) => {
                log::error!("Failed to get $SHELL: {e}. Checking passwd for user");
            }
        }

        match self
            .run_docker_exec_delimited("getent passwd \"$(id -un)\" | cut -d: -f7")
            .await
        {
            Ok(shell) => match shell.trim() {
                "" => {
                    log::info!("No shell found in passwd, falling back to {default_shell}");
                }
                shell => {
                    return shell.to_owned();
                }
            },
            Err(e) => {
                log::info!("Error getting shell from passwd: {e}. Falling back to {default_shell}");
            }
        }
        default_shell.to_owned()
    }

    async fn check_remote_platform(&self) -> Result<RemotePlatform> {
        let uname = self.run_docker_exec_delimited("uname -sm").await?;
        parse_platform(&uname)
    }

    /// Best-effort detection of the container's OS version for telemetry.
    async fn discover_os_version(&self, os: RemoteOs) -> Option<String> {
        let (program, args) = super::os_version_command(os);
        match self
            .run_docker_exec(program, None, &Default::default(), args)
            .await
        {
            Ok(output) => super::parse_os_version(os, &output),
            Err(error) => {
                log::warn!("Failed to determine remote OS version: {error:#}");
                None
            }
        }
    }

    async fn ensure_server_binary(
        &self,
        delegate: &Arc<dyn RemoteClientDelegate>,
        release_channel: ReleaseChannel,
        version: SemanticVersion,
        remote_dir_for_server: &str,
        commit: Option<AppCommitSha>,
        cx: &mut AsyncApp,
    ) -> Result<Arc<RelPath>> {
        anyhow::ensure!(
            self.connection_options.isolated_recipe_test.is_none(),
            "Legacy runtime installation is unavailable in recipe tests"
        );
        let remote_platform = self
            .remote_platform
            .context("No remote platform defined; cannot proceed.")?;

        let version_str = match release_channel {
            ReleaseChannel::Nightly => {
                let commit = commit.map(|s| s.full()).unwrap_or_default();
                format!("{}-{}", version, commit)
            }
            ReleaseChannel::Dev => "build".to_string(),
            _ => version.to_string(),
        };
        let binary_name = format!(
            "zed-remote-server-{}-{}",
            release_channel.dev_name(),
            version_str
        );
        let dst_path =
            paths::remote_server_dir_relative().join(RelPath::from_unix_str(&binary_name).unwrap());

        let binary_exists_on_server = self
            .run_docker_exec(
                &dst_path.display(self.path_style()),
                Some(&remote_dir_for_server),
                &Default::default(),
                &["version"],
            )
            .await
            .is_ok();
        #[cfg(any(debug_assertions, feature = "build-remote-server-binary"))]
        if let Some(remote_server_path) = super::build_remote_server_from_source(
            &remote_platform,
            delegate.as_ref(),
            binary_exists_on_server,
            cx,
        )
        .await?
        {
            let tmp_path = paths::remote_server_dir_relative().join(
                RelPath::from_unix_str(&format!(
                    "download-{}-{}",
                    std::process::id(),
                    remote_server_path.file_name().unwrap().to_string_lossy()
                ))
                .unwrap(),
            );
            self.upload_local_server_binary(
                &remote_server_path,
                &tmp_path,
                &remote_dir_for_server,
                delegate,
                cx,
            )
            .await?;
            self.extract_server_binary(&dst_path, &tmp_path, &remote_dir_for_server, delegate, cx)
                .await?;
            return Ok(dst_path.into());
        }

        if binary_exists_on_server {
            return Ok(dst_path.into());
        }

        let wanted_version = cx.update(|cx| match release_channel {
            ReleaseChannel::Nightly => Ok(None),
            ReleaseChannel::Dev => {
                anyhow::bail!(
                    "ZED_BUILD_REMOTE_SERVER is not set and no remote server exists at ({:?})",
                    dst_path
                )
            }
            _ => Ok(Some(AppVersion::global(cx))),
        })?;

        let tmp_path_gz = paths::remote_server_dir_relative().join(
            RelPath::from_unix_str(&format!(
                "{}-download-{}.gz",
                binary_name,
                std::process::id()
            ))
            .unwrap(),
        );
        if !self.connection_options.upload_binary_over_docker_exec
            && let Some(url) = delegate
                .get_download_url(remote_platform, release_channel, wanted_version.clone(), cx)
                .await?
        {
            match self
                .download_binary_on_server(&url, &tmp_path_gz, &remote_dir_for_server, delegate, cx)
                .await
            {
                Ok(_) => {
                    self.extract_server_binary(
                        &dst_path,
                        &tmp_path_gz,
                        &remote_dir_for_server,
                        delegate,
                        cx,
                    )
                    .await
                    .context("extracting server binary")?;
                    return Ok(dst_path.into());
                }
                Err(e) => {
                    log::error!(
                        "Failed to download binary on server, attempting to download locally and then upload it the server: {e:#}",
                    )
                }
            }
        }

        let src_path = delegate
            .download_server_binary_locally(remote_platform, release_channel, wanted_version, cx)
            .await
            .context("downloading server binary locally")?;
        self.upload_local_server_binary(
            &src_path,
            &tmp_path_gz,
            &remote_dir_for_server,
            delegate,
            cx,
        )
        .await
        .context("uploading server binary")?;
        self.extract_server_binary(
            &dst_path,
            &tmp_path_gz,
            &remote_dir_for_server,
            delegate,
            cx,
        )
        .await
        .context("extracting server binary")?;
        Ok(dst_path.into())
    }

    async fn docker_user_home_dir(&self) -> Result<String> {
        self.run_docker_exec_delimited("echo $HOME").await
    }

    async fn extract_server_binary(
        &self,
        dst_path: &RelPath,
        tmp_path: &RelPath,
        remote_dir_for_server: &str,
        delegate: &Arc<dyn RemoteClientDelegate>,
        cx: &mut AsyncApp,
    ) -> Result<()> {
        anyhow::ensure!(
            self.connection_options.isolated_recipe_test.is_none(),
            "Legacy runtime installation is unavailable in recipe tests"
        );
        delegate.set_status(Some("Extracting remote development server"), cx);
        let server_mode = 0o755;

        let shell_kind = ShellKind::Posix;
        let orig_tmp_path = tmp_path.display(self.path_style());
        let server_mode = format!("{:o}", server_mode);
        let server_mode = shell_kind
            .try_quote(&server_mode)
            .context("shell quoting")?;
        let dst_path = dst_path.display(self.path_style());
        let dst_path = shell_kind.try_quote(&dst_path).context("shell quoting")?;
        let script = if let Some(tmp_path) = orig_tmp_path.strip_suffix(".gz") {
            let orig_tmp_path = shell_kind
                .try_quote(&orig_tmp_path)
                .context("shell quoting")?;
            let tmp_path = shell_kind.try_quote(&tmp_path).context("shell quoting")?;
            format!(
                "gunzip -f {orig_tmp_path} && chmod {server_mode} {tmp_path} && mv {tmp_path} {dst_path}",
            )
        } else {
            let orig_tmp_path = shell_kind
                .try_quote(&orig_tmp_path)
                .context("shell quoting")?;
            format!("chmod {server_mode} {orig_tmp_path} && mv {orig_tmp_path} {dst_path}",)
        };
        let args = shell_kind.args_for_shell(false, script.to_string());
        self.run_docker_exec(
            "sh",
            Some(&remote_dir_for_server),
            &Default::default(),
            &args,
        )
        .await
        .log_err();
        Ok(())
    }

    async fn upload_local_server_binary(
        &self,
        src_path: &Path,
        tmp_path_gz: &RelPath,
        remote_dir_for_server: &str,
        delegate: &Arc<dyn RemoteClientDelegate>,
        cx: &mut AsyncApp,
    ) -> Result<()> {
        anyhow::ensure!(
            self.connection_options.isolated_recipe_test.is_none(),
            "Legacy runtime installation is unavailable in recipe tests"
        );
        if let Some(parent) = tmp_path_gz.parent() {
            self.run_docker_exec(
                "mkdir",
                Some(remote_dir_for_server),
                &Default::default(),
                &["-p", parent.display(self.path_style()).as_ref()],
            )
            .await?;
        }

        let src_stat = smol::fs::metadata(&src_path).await?;
        let size = src_stat.len();

        let t0 = Instant::now();
        delegate.set_status(Some("Uploading remote development server"), cx);
        log::info!(
            "uploading remote development server to {:?} ({}kb)",
            tmp_path_gz,
            size / 1024
        );
        self.upload_file(src_path, tmp_path_gz, remote_dir_for_server)
            .await
            .context("failed to upload server binary")?;
        log::info!("uploaded remote development server in {:?}", t0.elapsed());
        Ok(())
    }

    async fn upload_and_chown(
        docker_cli: String,
        connection_options: DockerConnectionOptions,
        src_path: String,
        dst_path: String,
    ) -> Result<()> {
        anyhow::ensure!(
            connection_options.isolated_recipe_test.is_none(),
            "Root copy/chown is unavailable in recipe tests"
        );
        let mut command = util::command::new_command(&docker_cli);
        command.arg("cp");
        command.arg("-a");
        command.arg(&src_path);
        command.arg(format!("{}:{}", connection_options.container_id, dst_path));

        let output = command.output().await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            log::debug!("failed to upload via docker cp {src_path} -> {dst_path}: {stderr}",);
            anyhow::bail!(
                "failed to upload via docker cp {} -> {}: {}",
                src_path,
                dst_path,
                stderr,
            );
        }

        let mut chown_command = util::command::new_command(&docker_cli);
        chown_command.arg("exec");
        chown_command.arg(connection_options.container_id);
        chown_command.arg("chown");
        chown_command.arg(format!(
            "{}:{}",
            connection_options.remote_user, connection_options.remote_user,
        ));
        chown_command.arg(&dst_path);

        let output = chown_command.output().await?;

        if output.status.success() {
            return Ok(());
        }

        let stderr = String::from_utf8_lossy(&output.stderr);
        log::debug!("failed to change ownership for via chown: {stderr}",);
        anyhow::bail!(
            "failed to change ownership for zed_remote_server via chown: {}",
            stderr,
        );
    }

    async fn upload_file(
        &self,
        src_path: &Path,
        dest_path: &RelPath,
        remote_dir_for_server: &str,
    ) -> Result<()> {
        anyhow::ensure!(
            self.connection_options.isolated_recipe_test.is_none(),
            "Legacy runtime installation is unavailable in recipe tests"
        );
        log::debug!("uploading file {:?} to {:?}", src_path, dest_path);

        let src_path_display = src_path.display().to_string();
        let dest_path_str = dest_path.display(self.path_style());
        let full_server_path = format!("{}/{}", remote_dir_for_server, dest_path_str);

        Self::upload_and_chown(
            self.docker_cli().to_string(),
            self.connection_options.clone(),
            src_path_display,
            full_server_path,
        )
        .await
    }

    async fn run_docker_command(
        &self,
        subcommand: &str,
        args: &[impl AsRef<str>],
    ) -> Result<String> {
        if let Some(policy) = &self.connection_options.isolated_recipe_test {
            let mut command = policy.command()?;
            command.arg(subcommand);
            command.args(args.iter().map(AsRef::as_ref));
            return self.recipe_test_output(command, None).await;
        }
        let mut command = util::command::new_command(self.docker_cli());
        command.arg(subcommand);
        for arg in args {
            command.arg(arg.as_ref());
        }
        let output = command.output().await?;
        log::debug!(
            "{}: {:?}",
            redact_arguments(self.docker_cli(), subcommand, args),
            output.status
        );
        if !output.status.success() {
            anyhow::bail!(docker_command_error(
                self.docker_cli(),
                subcommand,
                args,
                &String::from_utf8_lossy(&output.stderr),
            ));
        }
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    fn docker_exec_arguments(
        &self,
        inner_program: &str,
        working_directory: Option<&str>,
        env: &HashMap<String, String>,
        program_args: &[impl AsRef<str>],
    ) -> Vec<String> {
        let mut args = match working_directory {
            Some(dir) => vec!["-w".to_string(), dir.to_string()],
            None => vec![],
        };

        args.push("-u".to_string());
        args.push(self.connection_options.remote_user.clone());

        push_environment(&mut args, &self.connection_options.remote_env);

        push_environment(&mut args, env);

        args.push(self.connection_options.container_id.clone());
        args.push(inner_program.to_string());

        for arg in program_args {
            args.push(arg.as_ref().to_owned());
        }
        args
    }

    async fn run_docker_exec(
        &self,
        inner_program: &str,
        working_directory: Option<&str>,
        env: &HashMap<String, String>,
        program_args: &[impl AsRef<str>],
    ) -> Result<String> {
        if let Some(policy) = &self.connection_options.isolated_recipe_test {
            let arguments = self.recipe_test_exec_arguments(
                inner_program,
                working_directory,
                env,
                program_args,
                Interactive::No,
            )?;
            let mut command = policy.command()?;
            command.args(arguments);
            return self.recipe_test_output(command, None).await;
        }
        let args = self.docker_exec_arguments(inner_program, working_directory, env, program_args);
        self.run_docker_command("exec", args.as_ref()).await
    }

    async fn download_binary_on_server(
        &self,
        url: &str,
        tmp_path_gz: &RelPath,
        remote_dir_for_server: &str,
        delegate: &Arc<dyn RemoteClientDelegate>,
        cx: &mut AsyncApp,
    ) -> Result<()> {
        anyhow::ensure!(
            self.connection_options.isolated_recipe_test.is_none(),
            "Legacy runtime installation is unavailable in recipe tests"
        );
        if let Some(parent) = tmp_path_gz.parent() {
            self.run_docker_exec(
                "mkdir",
                Some(remote_dir_for_server),
                &Default::default(),
                &["-p", parent.display(self.path_style()).as_ref()],
            )
            .await?;
        }

        delegate.set_status(Some("Downloading remote development server on host"), cx);

        match self
            .run_docker_exec(
                "curl",
                Some(remote_dir_for_server),
                &Default::default(),
                &[
                    "-f",
                    "-L",
                    url,
                    "-o",
                    &tmp_path_gz.display(self.path_style()),
                ],
            )
            .await
        {
            Ok(_) => {}
            Err(e) => {
                if self
                    .run_docker_exec("which", None, &Default::default(), &["curl"])
                    .await
                    .is_ok()
                {
                    return Err(e);
                }

                log::info!("curl is not available, trying wget");
                match self
                    .run_docker_exec(
                        "wget",
                        Some(remote_dir_for_server),
                        &Default::default(),
                        &[url, "-O", &tmp_path_gz.display(self.path_style())],
                    )
                    .await
                {
                    Ok(_) => {}
                    Err(e) => {
                        if self
                            .run_docker_exec("which", None, &Default::default(), &["wget"])
                            .await
                            .is_ok()
                        {
                            return Err(e);
                        } else {
                            anyhow::bail!("Neither curl nor wget is available");
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn kill_inner(&self) -> Result<()> {
        if let Some(pid) = self.proxy_process.lock().take() {
            if let Ok(_) = util::command::new_command("kill")
                .arg(pid.to_string())
                .spawn()
            {
                Ok(())
            } else {
                Err(anyhow::anyhow!("Failed to kill process"))
            }
        } else {
            Ok(())
        }
    }
}

#[async_trait(?Send)]
impl RemoteConnection for DockerExecConnection {
    fn has_wsl_interop(&self) -> bool {
        false
    }
    fn start_proxy(
        &self,
        unique_identifier: String,
        mode: crate::ProxyMode,
        incoming_tx: UnboundedSender<Envelope>,
        outgoing_rx: UnboundedReceiver<Envelope>,
        connection_activity_tx: Sender<()>,
        delegate: Arc<dyn RemoteClientDelegate>,
        cx: &mut AsyncApp,
    ) -> Task<Result<i32>> {
        if matches!(mode, crate::ProxyMode::StartIfStopped(_)) {
            return Task::ready(Err(anyhow!(
                "Safe Doctor server replacement is unavailable for this container transport"
            )));
        }
        if mode != crate::ProxyMode::Start && self.connection_options.isolated_recipe_test.is_some()
        {
            return Task::ready(Err(anyhow!(
                "Recipe tests cannot reconnect or replay interrupted work"
            )));
        }
        // We'll try connecting anew every time we open a devcontainer, so proactively try to kill any old connections.
        if !self.has_been_killed() {
            if let Err(e) = self.kill_inner() {
                return Task::ready(Err(e));
            };
        }

        delegate.set_status(Some("Starting proxy"), cx);

        let Some(remote_binary_relpath) = self.remote_binary_relpath.clone() else {
            return Task::ready(Err(anyhow!("Remote binary path not set")));
        };

        let mut docker_args = vec!["exec".to_string()];

        push_environment(&mut docker_args, &self.connection_options.remote_env);
        for env_var in ["RUST_LOG", "RUST_BACKTRACE", "ZED_GENERATE_MINIDUMPS"]
            .into_iter()
            .filter(|_| self.connection_options.isolated_recipe_test.is_none())
        {
            if let Some(value) = std::env::var(env_var).ok() {
                docker_args.push("-e".to_string());
                docker_args.push(format!("{env_var}={value}"));
            }
        }

        docker_args.extend([
            "-u".to_string(),
            self.connection_options.remote_user.to_string(),
            "-w".to_string(),
            self.remote_dir_for_server.clone(),
            "-i".to_string(),
            self.connection_options.container_id.to_string(),
        ]);

        let val = remote_binary_relpath
            .display(self.path_style())
            .into_owned();
        docker_args.push(val);
        docker_args.push("proxy".to_string());
        docker_args.push("--identifier".to_string());
        docker_args.push(unique_identifier.clone());
        docker_args.extend(mode.arguments());
        let mut command = if let Some(policy) = &self.connection_options.isolated_recipe_test {
            docker_args = match self.recipe_test_exec_arguments(
                RECIPE_TEST_SERVER,
                Some("/home/made"),
                &HashMap::default(),
                &[
                    "proxy",
                    "--isolated-recipe-test",
                    "--identifier",
                    &unique_identifier,
                ],
                Interactive::No,
            ) {
                Ok(arguments) => arguments,
                Err(error) => return Task::ready(Err(error)),
            };
            match policy.command() {
                Ok(command) => command,
                Err(error) => return Task::ready(Err(error)),
            }
        } else {
            util::command::new_command(self.docker_cli())
        };
        command
            .kill_on_drop(true)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .args(docker_args);

        let Ok(child) = command.spawn() else {
            return Task::ready(Err(anyhow::anyhow!(
                "Failed to start remote server process"
            )));
        };

        let mut proxy_process = self.proxy_process.lock();
        *proxy_process = Some(child.id());

        cx.spawn(async move |cx| {
            super::handle_rpc_messages_over_child_process_stdio(
                child,
                incoming_tx,
                outgoing_rx,
                connection_activity_tx,
                cx,
            )
            .await
            .and_then(|status| {
                if status != 0 {
                    anyhow::bail!("Remote server exited with status {status}");
                }
                Ok(0)
            })
        })
    }

    fn upload_directory(
        &self,
        src_path: PathBuf,
        dest_path: RemotePathBuf,
        cx: &App,
    ) -> Task<Result<()>> {
        let dest_path_str = dest_path.to_string();
        let src_path_display = src_path.display().to_string();

        let upload_task = Self::upload_and_chown(
            self.docker_cli().to_string(),
            self.connection_options.clone(),
            src_path_display,
            dest_path_str,
        );

        cx.background_spawn(upload_task)
    }

    async fn kill(&self) -> Result<()> {
        self.kill_inner()
    }

    fn has_been_killed(&self) -> bool {
        self.proxy_process.lock().is_none()
    }

    fn build_command(
        &self,
        program: Option<String>,
        args: &[String],
        env: &HashMap<String, String>,
        working_dir: Option<String>,
        _port_forward: Option<(u16, String, u16)>,
        interactive: Interactive,
    ) -> Result<CommandTemplate> {
        if let Some(policy) = &self.connection_options.isolated_recipe_test {
            anyhow::ensure!(
                _port_forward.is_none(),
                "Port forwarding is unavailable in recipe tests"
            );
            let program = program.unwrap_or_else(|| "/bin/sh".into());
            let arguments = self.recipe_test_exec_arguments(
                &program,
                working_dir.as_deref(),
                env,
                args,
                interactive,
            )?;
            let mut docker_arguments = vec![
                "--host".into(),
                policy.endpoint.clone(),
                "--config".into(),
                policy
                    .configuration_directory
                    .to_string_lossy()
                    .into_owned(),
            ];
            docker_arguments.extend(arguments);
            return Ok(CommandTemplate {
                program: policy.docker_cli.to_string_lossy().into_owned(),
                args: docker_arguments,
                env: DOCKER_ENVIRONMENT
                    .iter()
                    .map(|name| ((*name).into(), String::new()))
                    .collect(),
            });
        }
        let mut parsed_working_dir = None;

        let path_style = self.path_style();

        if let Some(working_dir) = working_dir {
            let working_dir = RemotePathBuf::new(working_dir, path_style).to_string();

            const TILDE_PREFIX: &'static str = "~/";
            if working_dir.starts_with(TILDE_PREFIX) {
                let working_dir = working_dir.trim_start_matches("~").trim_start_matches("/");
                parsed_working_dir =
                    Some(format!("{}/{}", self.remote_dir_for_server, working_dir));
            } else {
                parsed_working_dir = Some(working_dir);
            }
        }

        let mut inner_program = Vec::new();

        if let Some(program) = program {
            inner_program.push(program);
            for arg in args {
                inner_program.push(arg.clone());
            }
        } else {
            inner_program.push(self.shell());
            inner_program.push("-l".to_string());
        };

        let mut docker_args = vec![
            "exec".to_string(),
            "-u".to_string(),
            self.connection_options.remote_user.clone(),
        ];

        if let Some(parsed_working_dir) = parsed_working_dir {
            docker_args.push("-w".to_string());
            docker_args.push(parsed_working_dir);
        }

        push_environment(&mut docker_args, &self.connection_options.remote_env);

        push_environment(&mut docker_args, env);

        match interactive {
            Interactive::Yes => docker_args.push("-it".to_string()),
            Interactive::No => docker_args.push("-i".to_string()),
        }
        docker_args.push(self.connection_options.container_id.to_string());

        docker_args.append(&mut inner_program);

        Ok(CommandTemplate {
            program: self.docker_cli().to_string(),
            args: docker_args,
            // Docker-exec pipes in environment via the "-e" argument
            env: Default::default(),
        })
    }

    fn build_forward_ports_command(
        &self,
        _forwards: Vec<(u16, String, u16)>,
    ) -> Result<CommandTemplate> {
        Err(anyhow::anyhow!("Not currently supported for docker_exec"))
    }

    fn connection_options(&self) -> RemoteConnectionOptions {
        RemoteConnectionOptions::Docker(self.connection_options.clone())
    }

    fn path_style(&self) -> PathStyle {
        self.path_style.unwrap_or(PathStyle::Unix)
    }

    fn remote_platform(&self) -> RemotePlatform {
        // Docker containers are always Linux; the platform is populated during
        // setup, so this fallback is only for the brief pre-detection window.
        self.remote_platform.unwrap_or(RemotePlatform {
            os: RemoteOs::Linux,
            arch: RemoteArch::X86_64,
        })
    }

    fn remote_os_version(&self) -> Option<String> {
        self.os_version.clone()
    }

    fn shell(&self) -> String {
        self.shell.clone()
    }

    fn default_system_shell(&self) -> String {
        String::from("/bin/sh")
    }
}

fn push_environment<'a>(
    args: &mut Vec<String>,
    environment: impl IntoIterator<Item = (&'a String, &'a String)>,
) {
    for (name, value) in environment {
        if !is_valid_environment_name(name) {
            log::warn!("Skipping environment variable with invalid name");
            continue;
        }
        args.push("-e".to_string());
        args.push(format!("{name}={value}"));
    }
}

fn redact_arguments(docker_cli: &str, subcommand: &str, args: &[impl AsRef<str>]) -> String {
    let mut redacted = format!("{docker_cli:?} {subcommand:?}");
    let mut next_argument_is_environment = false;

    for arg in args {
        let arg = arg.as_ref();
        let argument = if next_argument_is_environment {
            next_argument_is_environment = false;
            redact_environment(arg)
        } else if arg == "-e" {
            next_argument_is_environment = true;
            arg.to_string()
        } else {
            redact_command(arg)
        };
        redacted.push_str(&format!(" {argument:?}"));
    }

    redacted
}

fn docker_command_error(
    docker_cli: &str,
    subcommand: &str,
    args: &[impl AsRef<str>],
    stderr: &str,
) -> String {
    format!(
        "failed to run command {}: {}",
        redact_arguments(docker_cli, subcommand, args),
        redact_command(stderr)
    )
}

fn redact_environment(assignment: &str) -> String {
    match assignment.split_once('=') {
        Some((name, _)) => format!("{name}=<redacted>"),
        None => assignment.to_string(),
    }
}

#[cfg(test)]
mod recipe_scan_tests {
    use super::*;
    use std::io::Write as _;

    fn test_binary() -> Vec<u8> {
        let mut bytes = vec![0; 64];
        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes[4] = 2;
        bytes[5] = 1;
        bytes[18] = 62;
        bytes
    }

    #[test]
    fn recipe_scan_binary_validation_and_compressed_input() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let binary = directory.path().join("server");
        let contents = test_binary();
        std::fs::write(&binary, &contents)?;
        assert_eq!(read_recipe_test_binary(&binary)?, contents);
        let compressed = directory.path().join("server.gz");
        let mut writer = flate2::write::GzEncoder::new(
            std::fs::File::create(&compressed)?,
            flate2::Compression::default(),
        );
        writer.write_all(&contents)?;
        writer.finish()?;
        assert_eq!(read_recipe_test_binary(&compressed)?, contents);
        std::fs::write(&binary, b"#!/bin/sh\necho not-a-server")?;
        assert!(read_recipe_test_binary(&binary).is_err());
        std::fs::write(&compressed, [0x1f, 0x8b, 0])?;
        assert!(read_recipe_test_binary(&compressed).is_err());
        let mut wrong_architecture = contents;
        wrong_architecture[18] = 0;
        std::fs::write(&binary, wrong_architecture)?;
        assert!(read_recipe_test_binary(&binary).is_err());
        assert!(read_recipe_test_binary(Path::new("relative-server")).is_err());
        assert!(read_recipe_test_binary(directory.path()).is_err());
        #[cfg(unix)]
        {
            let link = directory.path().join("link");
            std::os::unix::fs::symlink(&binary, &link)?;
            assert!(read_recipe_test_binary(&link).is_err());
        }
        Ok(())
    }

    #[gpui::test]
    async fn recipe_scan_connection_rejects_changed_policy(cx: &mut gpui::TestAppContext) {
        cx.background_executor.allow_parking();
        let directory = tempfile::tempdir().expect("temporary directory");
        let configuration = tempfile::tempdir().expect("private Docker configuration");
        let binary = directory.path().join("server");
        let cli = directory.path().join("docker");
        std::fs::write(&binary, test_binary()).expect("test binary");
        std::fs::write(&cli, b"not executed").expect("test CLI");
        let endpoint = if cfg!(windows) {
            "npipe:////./pipe/docker_engine"
        } else {
            "unix:///tmp/made-test-docker.sock"
        };
        let policy = DockerRecipeTestOptions::prepare(
            endpoint.into(),
            configuration.path().into(),
            cli.clone(),
            binary.clone(),
            "test-version".into(),
        )
        .await
        .expect("valid policy");
        assert_eq!(
            policy.server_sha256(),
            format!("{:x}", Sha256::digest(test_binary()))
        );
        let options = DockerConnectionOptions::for_recipe_test("a".repeat(64), policy.clone())
            .expect("test connection");
        let connection = DockerExecConnection {
            proxy_process: Mutex::new(None),
            remote_dir_for_server: "/home/made".into(),
            remote_binary_relpath: None,
            connection_options: options.clone(),
            remote_platform: Some(RemotePlatform {
                os: RemoteOs::Linux,
                arch: RemoteArch::X86_64,
            }),
            os_version: None,
            path_style: Some(PathStyle::Unix),
            shell: "/bin/sh".into(),
            executor: cx.background_executor.clone(),
        };
        let arguments = connection
            .recipe_test_exec_arguments(
                "/bin/sh",
                Some("/workspace"),
                &HashMap::default(),
                &["-c", "echo hello"],
                Interactive::No,
            )
            .expect("isolated command");
        assert!(
            arguments
                .windows(3)
                .any(|part| part == ["/usr/bin/env", "-i", "PATH=/usr/local/bin:/usr/bin:/bin"])
        );
        assert_eq!(
            arguments.get(1).map(String::as_str),
            Some("--user=1000:1000")
        );
        assert!(
            connection
                .recipe_test_exec_arguments(
                    "/bin/sh",
                    None,
                    &HashMap::from_iter([("TOKEN".into(), "secret".into())]),
                    &["-c", "true"],
                    Interactive::No
                )
                .is_err()
        );
        assert!(
            connection
                .build_command(
                    None,
                    &[],
                    &HashMap::default(),
                    None,
                    Some((80, "host".into(), 80)),
                    Interactive::No
                )
                .is_err()
        );
        let mut modified = options.clone();
        modified.remote_user = "root".into();
        assert!(policy.validate_connection(&modified).is_err());
        modified = options.clone();
        modified.remote_env.insert("TOKEN".into(), "secret".into());
        assert!(policy.validate_connection(&modified).is_err());
        assert!(DockerRecipeTestOptions::validate_endpoint("tcp://localhost:2375").is_err());
        let mut changed_binary = test_binary();
        changed_binary[20] = 1;
        std::fs::write(&binary, changed_binary).expect("changed binary");
        let changed = connection
            .install_recipe_test_runtime()
            .await
            .expect_err("changed digest");
        assert!(changed.to_string().contains("changed after review"));
        let ordinary = RemoteConnectionOptions::Docker(DockerConnectionOptions {
            isolated_recipe_test: None,
            ..options.clone()
        });
        let isolated = RemoteConnectionOptions::Docker(options.clone());
        assert!(!crate::same_remote_connection_identity(
            Some(&ordinary),
            Some(&isolated)
        ));
        std::fs::write(configuration.path().join("config.json"), "{}")
            .expect("changed configuration");
        assert!(policy.validate_connection(&options).is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[gpui::test]
    fn redacts_forwarded_env(cx: &mut gpui::TestAppContext) {
        let connection = connection(
            cx,
            &[
                ("DATABASE_URL", "postgres://user:password@host/db"),
                ("GH_TOKEN", "ghp_supersecret"),
                ("PATH", "/usr/bin"),
                ("lowercase_token", "secret"),
            ],
        );

        assert_eq!(
            redacted_docker_exec(&connection, &[], &["-c", "echo hi"]),
            concat!(
                "\"docker\" \"exec\" \"-w\" \"/workspace\" \"-u\" \"user\"",
                " \"-e\" \"DATABASE_URL=<redacted>\" \"-e\" \"GH_TOKEN=<redacted>\"",
                " \"-e\" \"PATH=<redacted>\" \"-e\" \"lowercase_token=<redacted>\"",
                " \"container_id\" \"sh\" \"-c\" \"echo hi\""
            )
        );
    }

    #[gpui::test]
    fn preserves_argument_boundaries_and_escapes_control_characters(cx: &mut gpui::TestAppContext) {
        let connection = connection(cx, &[]);

        assert_eq!(
            redacted_docker_exec(&connection, &[], &["-c", "first argument\nsecond argument"]),
            concat!(
                "\"docker\" \"exec\" \"-w\" \"/workspace\" \"-u\" \"user\"",
                " \"container_id\" \"sh\" \"-c\" \"first argument\\nsecond argument\""
            )
        );
    }

    #[test]
    fn redacts_assignments_in_command_arguments() {
        assert_eq!(
            redact_arguments(
                "docker",
                "exec",
                &["container_id", "sh", "-c", "GH_TOKEN=ghp_supersecret run"]
            ),
            concat!(
                "\"docker\" \"exec\" \"container_id\" \"sh\" \"-c\"",
                " \"GH_TOKEN=\\\"[REDACTED]\\\" run\""
            )
        );
    }

    #[gpui::test]
    fn redacts_failure_message(cx: &mut gpui::TestAppContext) {
        let connection = connection(cx, &[("API_KEY", "remote-secret")]);
        let env = [("COMMAND_SECRET".to_string(), "command-secret".to_string())]
            .into_iter()
            .collect::<HashMap<_, _>>();
        let args = connection.docker_exec_arguments("sh", None, &env, &["-c", "echo hi"]);

        assert_eq!(
            docker_command_error(
                connection.docker_cli(),
                "exec",
                &args,
                "sh: 1: /usr/local/cargo/bin/zed-remote-server: not found\nGH_TOKEN=ghp_supersecret run",
            ),
            concat!(
                "failed to run command \"docker\" \"exec\" \"-u\" \"user\"",
                " \"-e\" \"API_KEY=<redacted>\" \"-e\" \"COMMAND_SECRET=<redacted>\"",
                " \"container_id\" \"sh\" \"-c\" \"echo hi\"",
                ": sh: 1: /usr/local/cargo/bin/zed-remote-server: not found\nGH_TOKEN=\"[REDACTED]\" run"
            )
        );
    }

    #[gpui::test]
    fn skips_invalid_env_names(cx: &mut gpui::TestAppContext) {
        let mut connection = connection(
            cx,
            &[
                ("", "empty"),
                ("NAME=VALUE", "equals"),
                ("LINE\nBREAK", "newline"),
                ("NAME\0NUL", "nul"),
                ("GH_TOKEN", "ghp_supersecret"),
            ],
        );
        let persisted_environment =
            serde_json::to_string(&connection.connection_options.remote_env)
                .expect("persist environment as WorkspaceDb does");
        connection.connection_options.remote_env = serde_json::from_str(&persisted_environment)
            .expect("reload persisted environment, including invalid names");
        let env = HashMap::from_iter([
            ("LINE\nBREAK".to_string(), "dynamic-secret".to_string()),
            ("command.variable".to_string(), "command-value".to_string()),
        ]);
        let program_args = Vec::<&str>::new();
        let args = connection.docker_exec_arguments("sh", None, &env, &program_args);

        assert_eq!(
            args,
            vec![
                "-u",
                "user",
                "-e",
                "GH_TOKEN=ghp_supersecret",
                "-e",
                "command.variable=command-value",
                "container_id",
                "sh"
            ]
        );
        let command = connection
            .build_command(None, &[], &env, None, None, Interactive::No)
            .expect("ordinary command with valid surviving environment");
        assert_eq!(
            command.args,
            vec![
                "exec",
                "-u",
                "user",
                "-e",
                "GH_TOKEN=ghp_supersecret",
                "-e",
                "command.variable=command-value",
                "-i",
                "container_id",
                "/bin/sh",
                "-l"
            ]
        );
    }

    #[gpui::test]
    fn uses_podman_cli(cx: &mut gpui::TestAppContext) {
        let mut connection = connection(cx, &[("GH_TOKEN", "ghp_supersecret")]);
        connection.connection_options.use_podman = true;

        assert_eq!(
            redacted_docker_exec(&connection, &[], &[]),
            concat!(
                "\"podman\" \"exec\" \"-w\" \"/workspace\" \"-u\" \"user\"",
                " \"-e\" \"GH_TOKEN=<redacted>\" \"container_id\" \"sh\""
            )
        );
    }

    #[gpui::test]
    #[ignore = "requires a disposable container ID in MADE_AUDIT_DOCKER_CONTAINER; run alone"]
    async fn native_docker_exec_keeps_forwarded_secrets_out_of_diagnostics(
        cx: &mut gpui::TestAppContext,
    ) {
        struct Capture(Mutex<Vec<String>>);
        impl log::Log for Capture {
            fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
                metadata.target().starts_with("remote::transport::docker")
            }

            fn log(&self, record: &log::Record<'_>) {
                if self.enabled(record.metadata()) {
                    self.0
                        .lock()
                        .push(format!("{} {}", record.level(), record.args()));
                }
            }

            fn flush(&self) {}
        }
        static CAPTURE: Capture = Capture(Mutex::new(Vec::new()));

        let identifier = std::env::var("MADE_AUDIT_DOCKER_CONTAINER")
            .expect("provide the ID of a disposable, network-disabled test container");
        assert_eq!(identifier.len(), 64);
        assert!(identifier.bytes().all(|byte| byte.is_ascii_hexdigit()));
        log::set_logger(&CAPTURE).expect("run this native regression alone");
        log::set_max_level(log::LevelFilter::Debug);
        cx.background_executor.allow_parking();
        let mut connection = connection(
            cx,
            &[
                ("lowercase_token", "made-audit-remote-secret"),
                ("NAME=VALUE", "made-audit-invalid-secret"),
            ],
        );
        let persisted_environment =
            serde_json::to_string(&connection.connection_options.remote_env)
                .expect("persist environment as WorkspaceDb does");
        connection.connection_options.remote_env = serde_json::from_str(&persisted_environment)
            .expect("reload persisted environment, including invalid names");
        connection.connection_options.container_id = identifier;
        connection.connection_options.remote_user = "root".to_string();
        let environment = HashMap::from_iter([
            (
                "command_secret".to_string(),
                "made-audit-command-secret".to_string(),
            ),
            (
                "LINE\nBREAK".to_string(),
                "made-audit-dynamic-invalid-secret".to_string(),
            ),
        ]);
        let output = connection
            .run_docker_exec(
                "/bin/sh",
                None,
                &environment,
                &["-c", "printf '%s' \"$lowercase_token:$command_secret\""],
            )
            .await
            .expect("native exec must preserve actual environment values");
        assert_eq!(output, "made-audit-remote-secret:made-audit-command-secret");
        let error = connection
            .run_docker_exec(
                "/bin/sh",
                None,
                &environment,
                &[
                    "-c",
                    "printf 'FAILED_TOKEN=made-audit-failure-secret\\n' >&2; exit 7",
                ],
            )
            .await
            .expect_err("native command exits unsuccessfully")
            .to_string();
        assert!(error.contains("FAILED_TOKEN=\"[REDACTED]\""));
        let records = CAPTURE.0.lock();
        assert_eq!(
            records
                .iter()
                .filter(|record| record.starts_with("DEBUG "))
                .count(),
            2
        );
        assert!(records.iter().any(|record| record.contains("ExitStatus")));
        for secret in [
            "made-audit-remote-secret",
            "made-audit-command-secret",
            "made-audit-invalid-secret",
            "made-audit-dynamic-invalid-secret",
            "made-audit-failure-secret",
        ] {
            assert!(!error.contains(secret), "failure leaked {secret}");
            assert!(
                records.iter().all(|record| !record.contains(secret)),
                "logs leaked {secret}"
            );
        }
        log::set_max_level(log::LevelFilter::Off);
    }

    fn connection(cx: &gpui::TestAppContext, remote_env: &[(&str, &str)]) -> DockerExecConnection {
        DockerExecConnection {
            proxy_process: Mutex::new(None),
            remote_dir_for_server: "/tmp/made".to_string(),
            remote_binary_relpath: None,
            connection_options: DockerConnectionOptions {
                name: "container".to_string(),
                container_id: "container_id".to_string(),
                remote_user: "user".to_string(),
                isolated_recipe_test: None,
                upload_binary_over_docker_exec: false,
                use_podman: false,
                remote_env: remote_env
                    .iter()
                    .map(|(key, value)| (key.to_string(), value.to_string()))
                    .collect(),
            },
            remote_platform: None,
            os_version: None,
            path_style: None,
            shell: "/bin/sh".to_string(),
            executor: cx.background_executor.clone(),
        }
    }

    fn redacted_docker_exec(
        connection: &DockerExecConnection,
        env: &[(&str, &str)],
        program_args: &[&str],
    ) -> String {
        let env = env
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect::<HashMap<_, _>>();
        let args = connection.docker_exec_arguments("sh", Some("/workspace"), &env, program_args);
        redact_arguments(connection.docker_cli(), "exec", &args)
    }
}
