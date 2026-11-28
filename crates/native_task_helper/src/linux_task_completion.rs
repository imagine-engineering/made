use crate::{
    task_actor::{AuthenticatedTaskActor, CoverageTaskParent, OriginalTaskActor},
    task_completion::{
        MAX_TASK_BUILD_INVOCATION_BYTES, TASK_STDOUT_FRAME_BYTES, TaskBuildInvocation,
        validate_stdout_report_limit,
    },
};
use futures_lite::{AsyncReadExt as _, AsyncWriteExt as _, future};
use sha2::{Digest as _, Sha256};
use smol::net::unix::{UnixListener, UnixStream};
use std::{
    fs::{self, File, Metadata},
    io,
    net::Shutdown,
    os::fd::{FromRawFd as _, OwnedFd},
    os::unix::{
        fs::{FileExt as _, MetadataExt as _, OpenOptionsExt as _, PermissionsExt as _},
        process::{CommandExt as _, ExitStatusExt as _},
    },
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    time::Duration,
};

const REGISTRATION_TIMEOUT: Duration = Duration::from_secs(10);
const COMPLETION_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const CANCELLATION_TIMEOUT: Duration = Duration::from_secs(10);
const COMPLETION_ACKNOWLEDGED: u8 = 1;
const CANCEL_NATIVE_TASK: u8 = 2;
const MAX_COVERAGE_BYTES: usize = 32 * 1024 * 1024;
const MAX_PROFILE_BYTES: usize = 32 * 1024 * 1024;
const MAX_SYMBOL_BYTES: usize = 16 * 1024 * 1024;

struct TaskArtifactDirectory {
    directory: tempfile::TempDir,
    identity: Metadata,
}

impl TaskArtifactDirectory {
    fn new(prefix: &str) -> io::Result<Self> {
        let directory = tempfile::Builder::new().prefix(prefix).tempdir_in("/tmp")?;
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700))?;
        let identity = fs::symlink_metadata(directory.path())?;
        Ok(Self {
            directory,
            identity,
        })
    }

    fn path(&self, name: &str) -> PathBuf {
        self.directory.path().join(name)
    }

    fn validate(&self) -> io::Result<()> {
        let current = fs::symlink_metadata(self.directory.path())?;
        if !current.is_dir()
            || current.dev() != self.identity.dev()
            || current.ino() != self.identity.ino()
            || current.mode() & 0o077 != 0
        {
            return Err(io::Error::other("Native Task artifact directory changed"));
        }
        Ok(())
    }
}

struct TaskArtifactFile {
    file: File,
    name: &'static str,
    label: &'static str,
    byte_limit: usize,
}

impl TaskArtifactFile {
    fn reserve(
        directory: &TaskArtifactDirectory,
        name: &'static str,
        label: &'static str,
        byte_limit: usize,
    ) -> io::Result<Self> {
        let file = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(directory.path(name))?;
        Ok(Self {
            file,
            name,
            label,
            byte_limit,
        })
    }

    fn checked_metadata(&self, directory: &TaskArtifactDirectory) -> io::Result<Metadata> {
        directory.validate()?;
        let named = fs::symlink_metadata(directory.path(self.name))?;
        let held = self.file.metadata()?;
        if !named.is_file()
            || named.nlink() != 1
            || named.mode() & 0o077 != 0
            || !same_artifact_version(&named, &held)
        {
            return Err(io::Error::other(format!("{} was replaced", self.label)));
        }
        Ok(held)
    }

    fn read_version(
        &self,
        directory: &TaskArtifactDirectory,
        before: &Metadata,
    ) -> io::Result<Vec<u8>> {
        let length = usize::try_from(before.len())
            .ok()
            .filter(|length| (1..=self.byte_limit).contains(length))
            .ok_or_else(|| {
                io::Error::other(format!("{} is empty or exceeds its limit", self.label))
            })?;
        let mut bytes = vec![0; length];
        self.file.read_exact_at(&mut bytes, 0)?;
        if !same_artifact_version(before, &self.checked_metadata(directory)?) {
            return Err(io::Error::other(format!(
                "{} changed during read",
                self.label
            )));
        }
        Ok(bytes)
    }
}

fn same_artifact_version(left: &Metadata, right: &Metadata) -> bool {
    left.dev() == right.dev()
        && left.ino() == right.ino()
        && left.len() == right.len()
        && left.ctime() == right.ctime()
        && left.ctime_nsec() == right.ctime_nsec()
        && left.mtime() == right.mtime()
        && left.mtime_nsec() == right.mtime_nsec()
}

#[repr(u8)]
#[derive(Clone, Copy)]
enum TaskKind {
    Coverage = 1,
    NativeProfile = 2,
    CargoBuild = 3,
}

trait TaskOutputReservation: Send + 'static {
    const KIND: TaskKind;
    type Sealed: Send;
    fn validate_invocation(&self, invocation: &TaskBuildInvocation) -> io::Result<()>;
    fn seal(self) -> io::Result<Self::Sealed>;
}

pub struct CoverageTaskReportReservation {
    directory: TaskArtifactDirectory,
    report: TaskArtifactFile,
}

impl CoverageTaskReportReservation {
    pub fn new(byte_limit: usize) -> io::Result<Self> {
        if !(1..=MAX_COVERAGE_BYTES).contains(&byte_limit) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid Coverage Task report limit",
            ));
        }
        let directory = TaskArtifactDirectory::new("zed-coverage-report-")?;
        let report = TaskArtifactFile::reserve(
            &directory,
            "report.json",
            "Coverage Task report",
            byte_limit,
        )?;
        Ok(Self { directory, report })
    }

    pub fn path(&self) -> PathBuf {
        self.directory.path(self.report.name)
    }

    fn read_bytes(&self) -> io::Result<Vec<u8>> {
        let version = self.report.checked_metadata(&self.directory)?;
        self.report.read_version(&self.directory, &version)
    }
}

impl TaskOutputReservation for CoverageTaskReportReservation {
    const KIND: TaskKind = TaskKind::Coverage;
    type Sealed = SealedCoverageTaskReport;

    fn validate_invocation(&self, invocation: &TaskBuildInvocation) -> io::Result<()> {
        if self.report.checked_metadata(&self.directory)?.len() != 0
            || !has_unique_path_argument(invocation, &self.path())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Coverage Task invocation does not own its empty report",
            ));
        }
        Ok(())
    }

    fn seal(self) -> io::Result<Self::Sealed> {
        let digest = Sha256::digest(self.read_bytes()?).into();
        Ok(SealedCoverageTaskReport {
            reservation: self,
            digest,
        })
    }
}

fn has_unique_path_argument(invocation: &TaskBuildInvocation, path: &Path) -> bool {
    invocation
        .arguments()
        .iter()
        .filter(|argument| std::ffi::OsStr::new(argument) == path.as_os_str())
        .count()
        == 1
}

pub struct SealedCoverageTaskReport {
    reservation: CoverageTaskReportReservation,
    digest: [u8; 32],
}

impl SealedCoverageTaskReport {
    pub fn read(self) -> io::Result<Vec<u8>> {
        let bytes = self.reservation.read_bytes()?;
        let digest: [u8; 32] = Sha256::digest(&bytes).into();
        if digest != self.digest {
            return Err(io::Error::other("Sealed Coverage Task report changed"));
        }
        Ok(bytes)
    }
}

pub struct NativeProfileTaskReservation {
    directory: TaskArtifactDirectory,
    profile: TaskArtifactFile,
    symbols: TaskArtifactFile,
}

impl NativeProfileTaskReservation {
    pub fn new() -> io::Result<Self> {
        let directory = TaskArtifactDirectory::new("zed-native-profile-")?;
        let profile = TaskArtifactFile::reserve(
            &directory,
            "profile.json",
            "Native Profile Task recording",
            MAX_PROFILE_BYTES,
        )?;
        let symbols = TaskArtifactFile::reserve(
            &directory,
            "profile.syms.json",
            "Native Profile Task symbols",
            MAX_SYMBOL_BYTES,
        )?;
        Ok(Self {
            directory,
            profile,
            symbols,
        })
    }

    pub fn profile_path(&self) -> PathBuf {
        self.directory.path(self.profile.name)
    }

    pub fn symbols_path(&self) -> PathBuf {
        self.directory.path(self.symbols.name)
    }

    fn read_pair(&self) -> io::Result<NativeProfileTaskArtifacts> {
        let profile_version = self.profile.checked_metadata(&self.directory)?;
        let symbols_version = self.symbols.checked_metadata(&self.directory)?;
        let profile = self
            .profile
            .read_version(&self.directory, &profile_version)?;
        let symbols = self
            .symbols
            .read_version(&self.directory, &symbols_version)?;
        if !same_artifact_version(
            &profile_version,
            &self.profile.checked_metadata(&self.directory)?,
        ) || !same_artifact_version(
            &symbols_version,
            &self.symbols.checked_metadata(&self.directory)?,
        ) {
            return Err(io::Error::other(
                "Native Profile Task pair changed during read",
            ));
        }
        Ok(NativeProfileTaskArtifacts { profile, symbols })
    }
}

impl TaskOutputReservation for NativeProfileTaskReservation {
    const KIND: TaskKind = TaskKind::NativeProfile;
    type Sealed = SealedNativeProfileTask;

    fn validate_invocation(&self, invocation: &TaskBuildInvocation) -> io::Result<()> {
        let profile_path = self.profile_path();
        if self.profile.checked_metadata(&self.directory)?.len() != 0
            || self.symbols.checked_metadata(&self.directory)?.len() != 0
            || !has_unique_path_argument(invocation, &profile_path)
            || !invocation.arguments().windows(2).any(|pair| {
                pair[0] == "-o" && std::ffi::OsStr::new(&pair[1]) == profile_path.as_os_str()
            })
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Native Profile Task invocation does not own its empty recording pair",
            ));
        }
        Ok(())
    }

    fn seal(self) -> io::Result<Self::Sealed> {
        let digests = self.read_pair()?.digests();
        Ok(SealedNativeProfileTask {
            reservation: self,
            digests,
        })
    }
}

pub struct NativeProfileTaskArtifacts {
    pub profile: Vec<u8>,
    pub symbols: Vec<u8>,
}

impl NativeProfileTaskArtifacts {
    fn digests(&self) -> [[u8; 32]; 2] {
        [
            Sha256::digest(&self.profile).into(),
            Sha256::digest(&self.symbols).into(),
        ]
    }
}

pub struct SealedNativeProfileTask {
    reservation: NativeProfileTaskReservation,
    digests: [[u8; 32]; 2],
}

impl SealedNativeProfileTask {
    pub fn read(self) -> io::Result<NativeProfileTaskArtifacts> {
        let pair = self.reservation.read_pair()?;
        if pair.digests() != self.digests {
            return Err(io::Error::other("Sealed Native Profile Task pair changed"));
        }
        Ok(pair)
    }
}

pub struct LinuxTaskCompletion {
    listener: UnixListener,
    directory: tempfile::TempDir,
}

impl LinuxTaskCompletion {
    pub fn bind() -> io::Result<Self> {
        let directory = tempfile::Builder::new()
            .prefix("zed-coverage-")
            .tempdir_in("/tmp")?;
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700))?;
        let socket = directory.path().join("completion");
        let listener = UnixListener::bind(&socket)?;
        std::fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600))?;
        Ok(Self {
            listener,
            directory,
        })
    }

    pub fn socket(&self) -> PathBuf {
        self.directory.path().join("completion")
    }

    pub async fn observe(
        self,
        actor: OriginalTaskActor,
        invocation: TaskBuildInvocation,
        report: Option<CoverageTaskReportReservation>,
    ) -> io::Result<LinuxTaskReceipt<SealedCoverageTaskReport>> {
        self.observe_output(actor, invocation, report, future::pending())
            .await
    }

    pub async fn observe_profile(
        self,
        actor: OriginalTaskActor,
        invocation: TaskBuildInvocation,
        profile: NativeProfileTaskReservation,
    ) -> io::Result<LinuxTaskReceipt<SealedNativeProfileTask>> {
        self.observe_cancellable_profile(actor, invocation, profile, future::pending())
            .await
    }

    pub async fn observe_cancellable_profile(
        self,
        actor: OriginalTaskActor,
        invocation: TaskBuildInvocation,
        profile: NativeProfileTaskReservation,
        cancel: impl Future<Output = ()>,
    ) -> io::Result<LinuxTaskReceipt<SealedNativeProfileTask>> {
        self.observe_output(actor, invocation, Some(profile), cancel)
            .await
    }

    pub async fn observe_cargo_build(
        self,
        actor: OriginalTaskActor,
        invocation: TaskBuildInvocation,
        report_limit: usize,
        cancel: impl Future<Output = ()>,
    ) -> io::Result<LinuxTaskReceipt<Vec<u8>>> {
        validate_stdout_report_limit(report_limit)?;
        let (mut actor, mut connection, challenge) = self
            .register_task(actor, invocation, TaskKind::CargoBuild, Some(report_limit))
            .await?;
        until_deadline(COMPLETION_TIMEOUT, async {
            let (response, cancelled) =
                receive_task_response(&mut connection, challenge, Some(report_limit), cancel)
                    .await?;
            actor.revalidate()?;
            connection.write_all(&[COMPLETION_ACKNOWLEDGED]).await?;
            Ok(LinuxTaskReceipt {
                termination: response.termination,
                output: response.stdout.filter(|_| {
                    !cancelled && response.termination == LinuxTaskTermination::Exited(0)
                }),
                cancelled,
            })
        })
        .await
    }

    async fn observe_output<Output: TaskOutputReservation>(
        self,
        actor: OriginalTaskActor,
        invocation: TaskBuildInvocation,
        output: Option<Output>,
        cancel: impl Future<Output = ()>,
    ) -> io::Result<LinuxTaskReceipt<Output::Sealed>> {
        if let Some(output) = output.as_ref() {
            output.validate_invocation(&invocation)?;
        }
        let (mut actor, mut connection, challenge) = self
            .register_task(actor, invocation, Output::KIND, None)
            .await?;
        until_deadline(COMPLETION_TIMEOUT, async {
            let (response, cancelled) =
                receive_task_response(&mut connection, challenge, None, cancel).await?;
            actor.revalidate()?;
            let output = match (cancelled, response.termination, output) {
                (false, LinuxTaskTermination::Exited(0), Some(output)) => {
                    Some(smol::unblock(move || output.seal()).await?)
                }
                _ => None,
            };
            actor.revalidate()?;
            connection.write_all(&[COMPLETION_ACKNOWLEDGED]).await?;
            Ok(LinuxTaskReceipt {
                termination: response.termination,
                output,
                cancelled,
            })
        })
        .await
    }

    async fn register_task(
        &self,
        actor: OriginalTaskActor,
        invocation: TaskBuildInvocation,
        kind: TaskKind,
        stdout_limit: Option<usize>,
    ) -> io::Result<(AuthenticatedTaskActor, UnixStream, [u8; 32])> {
        let request = invocation.encode_helper_environment()?;
        let mut challenge = [0; 32];
        getrandom::fill(&mut challenge)
            .map_err(|_| io::Error::other("Cannot issue Native Task challenge"))?;

        until_deadline(REGISTRATION_TIMEOUT, async {
            let (mut connection, _) = self.listener.accept().await?;
            let mut actor = actor.authenticate_peer(&connection)?;
            actor.revalidate()?;
            connection.write_all(&[kind as u8]).await?;
            connection.write_all(&challenge).await?;
            if let Some(limit) = stdout_limit {
                connection.write_all(&(limit as u32).to_be_bytes()).await?;
            }
            connection
                .write_all(&(request.len() as u32).to_be_bytes())
                .await?;
            connection.write_all(request.as_bytes()).await?;
            Ok((actor, connection, challenge))
        })
        .await
    }
}

struct LinuxTaskResponse {
    termination: LinuxTaskTermination,
    stdout: Option<Vec<u8>>,
}

async fn receive_task_response(
    connection: &mut UnixStream,
    challenge: [u8; 32],
    stdout_limit: Option<usize>,
    cancel: impl Future<Output = ()>,
) -> io::Result<(LinuxTaskResponse, bool)> {
    let response = read_task_response(connection.clone(), challenge, stdout_limit);
    let mut response = std::pin::pin!(response);
    match future::or(
        async {
            cancel.await;
            Ok(None)
        },
        async { response.as_mut().await.map(Some) },
    )
    .await?
    {
        Some(response) => Ok((response, false)),
        None => {
            let response = until_deadline(CANCELLATION_TIMEOUT, async {
                connection.write_all(&[CANCEL_NATIVE_TASK]).await?;
                response.await
            })
            .await?;
            Ok((response, true))
        }
    }
}

async fn read_task_response(
    mut connection: UnixStream,
    challenge: [u8; 32],
    stdout_limit: Option<usize>,
) -> io::Result<LinuxTaskResponse> {
    let mut response_challenge = [0; 32];
    connection.read_exact(&mut response_challenge).await?;
    if response_challenge != challenge {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "Native Task receipt did not match its invocation",
        ));
    }
    let mut response = [0; 2];
    connection.read_exact(&mut response).await?;
    let termination = match response {
        [0, code] => LinuxTaskTermination::Exited(code),
        [1, signal @ 1..=64] => LinuxTaskTermination::Signalled(signal),
        _ => return Err(invalid_completion()),
    };
    let stdout = if let Some(limit) = stdout_limit {
        let mut length = [0; 4];
        connection.read_exact(&mut length).await?;
        let length = u32::from_be_bytes(length) as usize;
        if length > limit {
            return Err(invalid_completion());
        }
        let mut bytes = vec![0; length];
        connection.read_exact(&mut bytes).await?;
        Some(bytes)
    } else {
        None
    };
    let mut trailing = [0; 1];
    if connection.read(&mut trailing).await? != 0 {
        return Err(invalid_completion());
    }
    Ok(LinuxTaskResponse {
        termination,
        stdout,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LinuxTaskTermination {
    Exited(u8),
    Signalled(u8),
}

impl LinuxTaskTermination {
    fn from_status(status: ExitStatus) -> io::Result<Self> {
        if let Some(code) = status.code() {
            return u8::try_from(code)
                .map(Self::Exited)
                .map_err(|_| invalid_completion());
        }
        if let Some(signal @ 1..=64) = status.signal() {
            return Ok(Self::Signalled(signal as u8));
        }
        Err(invalid_completion())
    }

    fn encode(self) -> [u8; 2] {
        match self {
            Self::Exited(code) => [0, code],
            Self::Signalled(signal) => [1, signal],
        }
    }
}

pub struct LinuxTaskReceipt<Output> {
    termination: LinuxTaskTermination,
    output: Option<Output>,
    cancelled: bool,
}

impl<Output> LinuxTaskReceipt<Output> {
    pub fn was_cancelled(&self) -> bool {
        self.cancelled
    }

    pub fn verify_exit(self, status: ExitStatus) -> io::Result<(ExitStatus, Option<Output>)> {
        if self.termination != LinuxTaskTermination::from_status(status)? {
            return Err(invalid_completion());
        }
        Ok((status, self.output))
    }
}

pub async fn run_coverage_task(socket: &Path) -> io::Result<ExitStatus> {
    run_task(socket, TaskKind::Coverage).await
}

pub async fn run_native_profile_task(socket: &Path) -> io::Result<ExitStatus> {
    run_task(socket, TaskKind::NativeProfile).await
}

pub async fn run_cargo_build_task(socket: &Path) -> io::Result<ExitStatus> {
    run_task(socket, TaskKind::CargoBuild).await
}

async fn run_task(socket: &Path, kind: TaskKind) -> io::Result<ExitStatus> {
    let (mut connection, invocation, challenge, parent, stdout_limit) =
        until_deadline(REGISTRATION_TIMEOUT, async {
            let mut connection = UnixStream::connect(socket).await?;
            let parent = CoverageTaskParent::authenticate(&connection)?;
            let mut version = [0; 1];
            connection.read_exact(&mut version).await?;
            if version != [kind as u8] {
                return Err(invalid_completion());
            }
            let mut challenge = [0; 32];
            connection.read_exact(&mut challenge).await?;
            let stdout_limit = if matches!(kind, TaskKind::CargoBuild) {
                let mut limit = [0; 4];
                connection.read_exact(&mut limit).await?;
                let limit = u32::from_be_bytes(limit) as usize;
                validate_stdout_report_limit(limit)?;
                Some(limit)
            } else {
                None
            };
            let mut length = [0; 4];
            connection.read_exact(&mut length).await?;
            let length = u32::from_be_bytes(length) as usize;
            if length == 0 || length > MAX_TASK_BUILD_INVOCATION_BYTES {
                return Err(invalid_completion());
            }
            let mut request = vec![0; length];
            connection.read_exact(&mut request).await?;
            let invocation = serde_json::from_slice::<TaskBuildInvocation>(&request)
                .map_err(|_| invalid_completion())?;
            invocation.validate()?;
            parent.revalidate()?;
            Ok((connection, invocation, challenge, parent, stdout_limit))
        })
        .await?;

    let mut command = Command::new(invocation.program());
    command
        .args(invocation.arguments())
        .current_dir(invocation.working_directory())
        .env_clear()
        .envs(invocation.environment());
    let helper_process = std::process::id();
    // The terminal may kill the helper before it can handle socket closure.
    unsafe {
        command.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                return Err(io::Error::last_os_error());
            }
            if libc::getppid() as u32 != helper_process {
                return Err(io::Error::from_raw_os_error(libc::ESRCH));
            }
            Ok(())
        });
    }
    let (status, stdout) = match kind {
        TaskKind::Coverage => (run_coverage_workload(command, &mut connection).await?, None),
        TaskKind::NativeProfile => (run_profile_workload(command, &mut connection).await?, None),
        TaskKind::CargoBuild => {
            let limit = stdout_limit.ok_or_else(invalid_completion)?;
            let (status, stdout) =
                run_cargo_build_workload(command, &mut connection, limit).await?;
            (status, Some(stdout))
        }
    };

    let termination = LinuxTaskTermination::from_status(status)?;
    parent.revalidate()?;
    until_deadline(REGISTRATION_TIMEOUT, async {
        connection.write_all(&challenge).await?;
        connection.write_all(&termination.encode()).await?;
        if let Some(stdout) = stdout {
            connection
                .write_all(&(stdout.len() as u32).to_be_bytes())
                .await?;
            connection.write_all(&stdout).await?;
        }
        connection.shutdown(Shutdown::Write)?;
        let mut acknowledged = [0; 1];
        connection.read_exact(&mut acknowledged).await?;
        if !matches!(kind, TaskKind::Coverage) && acknowledged == [CANCEL_NATIVE_TASK] {
            connection.read_exact(&mut acknowledged).await?;
        }
        if acknowledged != [COMPLETION_ACKNOWLEDGED] {
            return Err(invalid_completion());
        }
        Ok(status)
    })
    .await
}

async fn observer_interruption(connection: &mut UnixStream) -> io::Error {
    let mut command = [0; 1];
    match connection.read(&mut command).await {
        Ok(0) => io::Error::new(io::ErrorKind::Interrupted, "Task observer disconnected"),
        Ok(_) => invalid_completion(),
        Err(error) => error,
    }
}

async fn wait_for_task_cancellation(connection: &mut UnixStream) -> io::Result<()> {
    let mut command = [0; 1];
    match connection.read(&mut command).await? {
        0 => Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "Task observer disconnected",
        )),
        _ if command == [CANCEL_NATIVE_TASK] => Ok(()),
        _ => Err(invalid_completion()),
    }
}

async fn run_coverage_workload(
    command: Command,
    connection: &mut UnixStream,
) -> io::Result<ExitStatus> {
    let mut child = smol::process::Command::from(command)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()?;
    match future::race(child.status(), async {
        Err(observer_interruption(connection).await)
    })
    .await
    {
        Ok(status) => Ok(status),
        Err(error) => {
            let termination = child.kill();
            child.status().await?;
            if let Err(termination) = termination
                && termination.raw_os_error() != Some(libc::ESRCH)
            {
                return Err(termination);
            }
            Err(error)
        }
    }
}

#[expect(
    clippy::disallowed_methods,
    reason = "The standalone helper owns the unreaped child and must keep its spawning thread alive for PR_SET_PDEATHSIG"
)]
fn spawn_task_group(mut command: Command) -> io::Result<Child> {
    // If the group leader exits first, the helper must reap the orphaned members.
    if unsafe { libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) } != 0 {
        return Err(io::Error::last_os_error());
    }
    command.process_group(0);
    unsafe {
        command.pre_exec(|| {
            if libc::isatty(libc::STDIN_FILENO) != 1 {
                return Ok(());
            }
            // The workload group must be foreground for normal PTY input.
            let previous = libc::signal(libc::SIGTTOU, libc::SIG_IGN);
            if previous == libc::SIG_ERR {
                return Err(io::Error::last_os_error());
            }
            let result = libc::tcsetpgrp(libc::STDIN_FILENO, libc::getpid());
            let error = io::Error::last_os_error();
            if libc::signal(libc::SIGTTOU, previous) == libc::SIG_ERR {
                return Err(io::Error::last_os_error());
            }
            if result == 0 { Ok(()) } else { Err(error) }
        });
    }
    command.spawn()
}

async fn run_profile_workload(
    command: Command,
    connection: &mut UnixStream,
) -> io::Result<ExitStatus> {
    let child = spawn_task_group(command)?;
    let group = child.id() as libc::pid_t;
    let result = future::or(
        wait_for_task_cancellation(connection),
        wait_for_task_exit(group),
    )
    .await;
    let status = smol::unblock(move || finish_task_group(child, group)).await?;
    result.map(|()| status)
}

#[expect(
    clippy::disallowed_methods,
    reason = "The native Cargo helper takes stdout from the original std child and holds the process group until cleanup"
)]
async fn run_cargo_build_workload(
    mut command: Command,
    connection: &mut UnixStream,
    limit: usize,
) -> io::Result<(ExitStatus, Vec<u8>)> {
    command.stdout(Stdio::piped());
    let mut child = spawn_task_group(command)?;
    let group = child.id() as libc::pid_t;
    let stdout = child.stdout.take();
    let read_and_wait = async {
        let stdout =
            stdout.ok_or_else(|| io::Error::other("Cargo Task stdout pipe is unavailable"))?;
        let (report, ()) = future::try_zip(collect_cargo_stdout(stdout, limit), async {
            wait_for_task_exit(group).await?;
            // Detached writers in Cargo's group must not keep the report pipe open.
            stop_task_group(group)
        })
        .await?;
        Ok(report)
    };
    let result = future::or(
        async {
            wait_for_task_cancellation(connection).await?;
            Ok(Vec::new())
        },
        read_and_wait,
    )
    .await;
    let status = smol::unblock(move || finish_task_group(child, group)).await?;
    result.map(|stdout| (status, stdout))
}

async fn collect_cargo_stdout(
    stdout: std::process::ChildStdout,
    limit: usize,
) -> io::Result<Vec<u8>> {
    let mut stdout = smol::Async::new(stdout)?;
    let mut terminal = smol::Unblock::with_capacity(TASK_STDOUT_FRAME_BYTES, std::io::stdout());
    let mut report = Vec::new();
    report
        .try_reserve_exact(limit)
        .map_err(|_| io::Error::other("Cannot allocate the bounded Cargo report"))?;
    let mut buffer = [0; TASK_STDOUT_FRAME_BYTES];
    loop {
        let remaining = (limit.saturating_sub(report.len()) + 1).min(buffer.len());
        let count = stdout.read(&mut buffer[..remaining]).await?;
        if count == 0 {
            return Ok(report);
        }
        if count > limit.saturating_sub(report.len()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Cargo Task stdout report exceeds its limit",
            ));
        }
        report.extend_from_slice(&buffer[..count]);
        terminal.write_all(&buffer[..count]).await?;
        terminal.flush().await?;
    }
}

async fn wait_for_task_exit(group: libc::pid_t) -> io::Result<()> {
    // A readable pidfd observes exit without releasing the leader's process group ID.
    let descriptor = unsafe { libc::syscall(libc::SYS_pidfd_open, group, 0) };
    if descriptor < 0 {
        return Err(io::Error::last_os_error());
    }
    smol::Async::new_nonblocking(unsafe { OwnedFd::from_raw_fd(descriptor as libc::c_int) })?
        .readable()
        .await
}

fn stop_task_group(group: libc::pid_t) -> io::Result<()> {
    if unsafe { libc::killpg(group, libc::SIGKILL) } != 0 {
        let error = io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::ESRCH) {
            return Err(error);
        }
    }
    Ok(())
}

fn finish_task_group(mut child: Child, group: libc::pid_t) -> io::Result<ExitStatus> {
    stop_task_group(group)?;
    let status = child.wait()?;
    loop {
        if unsafe { libc::waitpid(-group, std::ptr::null_mut(), 0) } < 0 {
            let error = io::Error::last_os_error();
            match error.raw_os_error() {
                Some(libc::ECHILD) => return Ok(status),
                Some(libc::EINTR) => continue,
                _ => return Err(error),
            }
        }
    }
}

#[expect(
    clippy::disallowed_methods,
    reason = "The standalone helper needs wall-clock deadlines without a GPUI executor"
)]
async fn until_deadline<T>(
    duration: Duration,
    operation: impl Future<Output = io::Result<T>>,
) -> io::Result<T> {
    future::race(operation, async {
        smol::Timer::after(duration).await;
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "Coverage Task observation timed out",
        ))
    })
    .await
}

fn invalid_completion() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "Invalid Coverage Task completion",
    )
}
