use crate::{
    task_actor::{CoverageTaskParent, OriginalTaskActor},
    task_completion::{MAX_TASK_BUILD_INVOCATION_BYTES, TaskBuildInvocation},
};
use futures_lite::{AsyncReadExt as _, AsyncWriteExt as _, future};
use sha2::{Digest as _, Sha256};
use smol::net::unix::{UnixListener, UnixStream};
use std::{
    fs::{self, File, Metadata},
    io,
    net::Shutdown,
    os::unix::{
        fs::{FileExt as _, MetadataExt as _, OpenOptionsExt as _, PermissionsExt as _},
        process::{CommandExt as _, ExitStatusExt as _},
    },
    path::{Path, PathBuf},
    process::{Command, ExitStatus, Stdio},
    time::Duration,
};

const REGISTRATION_TIMEOUT: Duration = Duration::from_secs(10);
const COMPLETION_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const PROTOCOL_VERSION: u8 = 1;
const COMPLETION_ACKNOWLEDGED: u8 = 1;
const MAX_REPORT_BYTES: usize = 32 * 1024 * 1024;

pub struct CoverageTaskReportReservation {
    directory: tempfile::TempDir,
    directory_metadata: Metadata,
    file: File,
    byte_limit: usize,
}

impl CoverageTaskReportReservation {
    pub fn new(byte_limit: usize) -> io::Result<Self> {
        if !(1..=MAX_REPORT_BYTES).contains(&byte_limit) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid Coverage Task report limit",
            ));
        }
        let directory = tempfile::Builder::new()
            .prefix("zed-coverage-report-")
            .tempdir_in("/tmp")?;
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700))?;
        let directory_metadata = fs::symlink_metadata(directory.path())?;
        let file = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(directory.path().join("report.json"))?;
        Ok(Self {
            directory,
            directory_metadata,
            file,
            byte_limit,
        })
    }

    pub fn path(&self) -> PathBuf {
        self.directory.path().join("report.json")
    }

    fn validate_invocation(&self, invocation: &TaskBuildInvocation) -> io::Result<()> {
        let path = self.path();
        if self.checked_metadata()?.len() != 0
            || invocation
                .arguments()
                .iter()
                .filter(|argument| std::ffi::OsStr::new(argument) == path.as_os_str())
                .count()
                != 1
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Coverage Task invocation does not own its empty report",
            ));
        }
        Ok(())
    }

    fn checked_metadata(&self) -> io::Result<Metadata> {
        let directory = fs::symlink_metadata(self.directory.path())?;
        if !directory.is_dir()
            || directory.dev() != self.directory_metadata.dev()
            || directory.ino() != self.directory_metadata.ino()
        {
            return Err(io::Error::other(
                "Coverage Task report directory was replaced",
            ));
        }
        let named = fs::symlink_metadata(self.path())?;
        let held = self.file.metadata()?;
        if !named.is_file() || named.nlink() != 1 || !same_report_version(&named, &held) {
            return Err(io::Error::other("Coverage Task report was replaced"));
        }
        Ok(held)
    }

    fn read_bytes(&self) -> io::Result<Vec<u8>> {
        let before = self.checked_metadata()?;
        let length = usize::try_from(before.len())
            .ok()
            .filter(|length| (1..=self.byte_limit).contains(length))
            .ok_or_else(|| {
                io::Error::other("Coverage Task report is empty or exceeds its limit")
            })?;
        let mut bytes = vec![0; length];
        self.file.read_exact_at(&mut bytes, 0)?;
        if !same_report_version(&before, &self.checked_metadata()?) {
            return Err(io::Error::other("Coverage Task report changed during read"));
        }
        Ok(bytes)
    }

    fn seal(self) -> io::Result<SealedCoverageTaskReport> {
        let digest = Sha256::digest(self.read_bytes()?).into();
        Ok(SealedCoverageTaskReport {
            reservation: self,
            digest,
        })
    }
}

fn same_report_version(left: &Metadata, right: &Metadata) -> bool {
    left.dev() == right.dev()
        && left.ino() == right.ino()
        && left.len() == right.len()
        && left.ctime() == right.ctime()
        && left.ctime_nsec() == right.ctime_nsec()
        && left.mtime() == right.mtime()
        && left.mtime_nsec() == right.mtime_nsec()
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

pub struct CoverageTaskCompletion {
    listener: UnixListener,
    directory: tempfile::TempDir,
}

impl CoverageTaskCompletion {
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
    ) -> io::Result<CoverageTaskReceipt> {
        if let Some(report) = report.as_ref() {
            report.validate_invocation(&invocation)?;
        }
        let request = invocation.encode_helper_environment()?;
        let mut challenge = [0; 32];
        getrandom::fill(&mut challenge)
            .map_err(|_| io::Error::other("Cannot issue Coverage Task challenge"))?;

        let (mut actor, mut connection) = until_deadline(REGISTRATION_TIMEOUT, async {
            let (mut connection, _) = self.listener.accept().await?;
            let mut actor = actor.authenticate_peer(&connection)?;
            actor.revalidate()?;
            connection.write_all(&[PROTOCOL_VERSION]).await?;
            connection.write_all(&challenge).await?;
            connection
                .write_all(&(request.len() as u32).to_be_bytes())
                .await?;
            connection.write_all(request.as_bytes()).await?;
            Ok((actor, connection))
        })
        .await?;

        until_deadline(COMPLETION_TIMEOUT, async {
            let mut response_challenge = [0; 32];
            connection.read_exact(&mut response_challenge).await?;
            if response_challenge != challenge {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "Coverage Task receipt did not match its invocation",
                ));
            }
            let mut response = [0; 2];
            connection.read_exact(&mut response).await?;
            let mut trailing = [0; 1];
            if connection.read(&mut trailing).await? != 0 {
                return Err(invalid_completion());
            }
            let termination = match response {
                [0, code] => CoverageTaskTermination::Exited(code),
                [1, signal @ 1..=64] => CoverageTaskTermination::Signalled(signal),
                _ => return Err(invalid_completion()),
            };
            actor.revalidate()?;
            let report = match (termination, report) {
                (CoverageTaskTermination::Exited(0), Some(report)) => {
                    Some(smol::unblock(move || report.seal()).await?)
                }
                _ => None,
            };
            actor.revalidate()?;
            connection.write_all(&[COMPLETION_ACKNOWLEDGED]).await?;
            Ok(CoverageTaskReceipt {
                termination,
                report,
            })
        })
        .await
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CoverageTaskTermination {
    Exited(u8),
    Signalled(u8),
}

impl CoverageTaskTermination {
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

pub struct CoverageTaskReceipt {
    termination: CoverageTaskTermination,
    report: Option<SealedCoverageTaskReport>,
}

impl CoverageTaskReceipt {
    pub fn verify_exit(
        self,
        status: ExitStatus,
    ) -> io::Result<(ExitStatus, Option<SealedCoverageTaskReport>)> {
        if self.termination != CoverageTaskTermination::from_status(status)? {
            return Err(invalid_completion());
        }
        Ok((status, self.report))
    }
}

pub async fn run_coverage_task(socket: &Path) -> io::Result<ExitStatus> {
    let (mut connection, invocation, challenge, parent) =
        until_deadline(REGISTRATION_TIMEOUT, async {
            let mut connection = UnixStream::connect(socket).await?;
            let parent = CoverageTaskParent::authenticate(&connection)?;
            let mut version = [0; 1];
            connection.read_exact(&mut version).await?;
            if version != [PROTOCOL_VERSION] {
                return Err(invalid_completion());
            }
            let mut challenge = [0; 32];
            connection.read_exact(&mut challenge).await?;
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
            Ok((connection, invocation, challenge, parent))
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
    let mut child = smol::process::Command::from(command)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()?;
    let status = match future::race(child.status(), async {
        let mut command = [0; 1];
        match connection.read(&mut command).await? {
            0 => Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "Coverage Task observer disconnected",
            )),
            _ => Err(invalid_completion()),
        }
    })
    .await
    {
        Ok(status) => status,
        Err(error) => {
            let termination = child.kill();
            child.status().await?;
            if let Err(termination) = termination
                && termination.raw_os_error() != Some(libc::ESRCH)
            {
                return Err(termination);
            }
            return Err(error);
        }
    };

    let termination = CoverageTaskTermination::from_status(status)?;
    parent.revalidate()?;
    until_deadline(REGISTRATION_TIMEOUT, async {
        connection.write_all(&challenge).await?;
        connection.write_all(&termination.encode()).await?;
        connection.shutdown(Shutdown::Write)?;
        let mut acknowledged = [0; 1];
        connection.read_exact(&mut acknowledged).await?;
        if acknowledged != [COMPLETION_ACKNOWLEDGED] {
            return Err(invalid_completion());
        }
        Ok(status)
    })
    .await
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
