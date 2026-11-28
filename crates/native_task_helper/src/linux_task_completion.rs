use crate::{
    linux_cargo_runtime::{
        LinuxCargoRuntimeObserver, configure_profile_workload, open_profile_image,
        validate_workload_environment,
    },
    task_actor::{AuthenticatedTaskActor, CoverageTaskParent, OriginalTaskActor},
    task_completion::{
        MAX_TASK_BUILD_INVOCATION_BYTES, ObservedTaskBuildInvocation, TASK_STDOUT_FRAME_BYTES,
        TaskBuildInvocation, validate_stdout_report_limit,
    },
};
use futures_lite::{AsyncReadExt as _, AsyncWriteExt as _, future};
use sha2::{Digest as _, Sha256};
use smol::net::unix::{UnixListener, UnixStream};
use std::{
    ffi::OsStr,
    fs::{self, File, Metadata},
    io::{self, Write as _},
    net::Shutdown,
    os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd},
    os::unix::{
        fs::{FileExt as _, MetadataExt as _, OpenOptionsExt as _, PermissionsExt as _},
        process::{CommandExt as _, ExitStatusExt as _},
    },
    path::{Component, Path, PathBuf},
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinuxTaskMode {
    Coverage = 1,
    NativeProfile = 2,
    CargoBuild = 3,
    CargoProfile = 4,
    CargoRuntime = 5,
}

impl LinuxTaskMode {
    pub fn command(self) -> &'static str {
        match self {
            Self::Coverage => "--observe-coverage-task",
            Self::NativeProfile => "--observe-native-profile-task",
            Self::CargoBuild => "--observe-cargo-build-task",
            Self::CargoProfile => "--observe-cargo-profile-task",
            Self::CargoRuntime => "--observe-cargo-runtime-task",
        }
    }
}

fn parse_task_helper(
    mut arguments: impl Iterator<Item = std::ffi::OsString>,
) -> io::Result<Option<(LinuxTaskMode, PathBuf)>> {
    let Some(command) = arguments.next() else {
        return Ok(None);
    };
    let Some(mode) = [
        LinuxTaskMode::Coverage,
        LinuxTaskMode::NativeProfile,
        LinuxTaskMode::CargoBuild,
        LinuxTaskMode::CargoProfile,
        LinuxTaskMode::CargoRuntime,
    ]
    .into_iter()
    .find(|mode| command == mode.command()) else {
        return Ok(None);
    };
    let invalid = || {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Invalid Linux Task helper arguments",
        )
    };
    if arguments
        .next()
        .is_none_or(|argument| argument != "--socket")
    {
        return Err(invalid());
    }
    let socket = PathBuf::from(arguments.next().ok_or_else(invalid)?);
    if arguments.next().is_some()
        || !socket.is_absolute()
        || socket.file_name().is_none()
        || socket.as_os_str().as_encoded_bytes().len() > 107
        || socket.as_os_str().as_encoded_bytes().contains(&0)
        || socket
            .components()
            .any(|component| component == Component::ParentDir)
    {
        return Err(invalid());
    }
    Ok(Some((mode, socket)))
}

pub fn run_linux_task_helper_if_invoked() {
    let (mode, socket) = match parse_task_helper(std::env::args_os().skip(1)) {
        Ok(Some(invocation)) => invocation,
        Ok(None) => return,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    match smol::block_on(run_task(&socket, mode)) {
        Ok(status) => crate::task_completion::exit_with_observed_task_status(status),
        Err(error) => {
            eprintln!("Linux Task helper failed: {error}");
            std::process::exit(1);
        }
    }
}

trait TaskOutputReservation: Send + 'static {
    const KIND: LinuxTaskMode;
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
    const KIND: LinuxTaskMode = LinuxTaskMode::Coverage;
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
    const KIND: LinuxTaskMode = LinuxTaskMode::NativeProfile;
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

pub struct NativeProfileTaskDestination {
    directory: File,
    parent: PathBuf,
    names: [PathBuf; 2],
}

impl NativeProfileTaskDestination {
    pub fn open(paths: [PathBuf; 2]) -> io::Result<Self> {
        let [profile, symbols] = paths;
        let parent = profile.parent().ok_or_else(invalid_completion)?;
        if !is_normalized_absolute_path(&profile)
            || !is_normalized_absolute_path(&symbols)
            || symbols.parent() != Some(parent)
            || profile == symbols
            || parent.canonicalize()?.as_os_str() != parent.as_os_str()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Native Profile Task outputs must be distinct files in one canonical directory",
            ));
        }
        let directory = File::options()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
            .open(parent)?;
        let names = [&profile, &symbols].map(|path| {
            path.file_name()
                .map(PathBuf::from)
                .ok_or_else(invalid_completion)
        });
        let [profile_name, symbols_name] = names;
        let destination = Self {
            directory,
            parent: parent.to_path_buf(),
            names: [profile_name?, symbols_name?],
        };
        destination.check_available()?;
        Ok(destination)
    }

    pub fn check_available(&self) -> io::Result<()> {
        self.check_directory()?;
        for name in &self.names {
            match fs::symlink_metadata(self.descriptor_path().join(name)) {
                Ok(_) => return Err(io::Error::from(io::ErrorKind::AlreadyExists)),
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    fn descriptor_path(&self) -> PathBuf {
        PathBuf::from(format!("/proc/self/fd/{}", self.directory.as_raw_fd()))
    }

    fn check_directory(&self) -> io::Result<()> {
        let current = fs::symlink_metadata(&self.parent)?;
        let opened = self.directory.metadata()?;
        if !current.is_dir()
            || current.dev() != opened.dev()
            || current.ino() != opened.ino()
            || self.descriptor_path().read_link()? != self.parent
        {
            return Err(io::Error::other(
                "Native Profile Task output directory changed",
            ));
        }
        Ok(())
    }
}

fn is_normalized_absolute_path(path: &Path) -> bool {
    path.is_absolute()
        && !path
            .components()
            .any(|component| component == Component::ParentDir)
        && path.components().collect::<PathBuf>().as_os_str() == path.as_os_str()
}

impl SealedNativeProfileTask {
    pub fn read(self) -> io::Result<NativeProfileTaskArtifacts> {
        let pair = self.reservation.read_pair()?;
        if pair.digests() != self.digests {
            return Err(io::Error::other("Sealed Native Profile Task pair changed"));
        }
        Ok(pair)
    }

    pub fn retain(
        self,
        destination: NativeProfileTaskDestination,
        is_current: impl Fn() -> bool,
    ) -> io::Result<[Metadata; 2]> {
        let check_destination = || {
            if !is_current() {
                return Err(io::Error::from(io::ErrorKind::Interrupted));
            }
            destination.check_directory()
        };
        check_destination()?;
        let pair = self.read()?;
        let directory = destination.descriptor_path();
        let [profile, symbols] = destination
            .names
            .each_ref()
            .map(|name| directory.join(name));
        let write = |bytes: &[u8]| -> io::Result<tempfile::NamedTempFile> {
            let mut file = tempfile::NamedTempFile::new_in(&directory)?;
            file.write_all(bytes)?;
            file.as_file().sync_all()?;
            Ok(file)
        };
        let recording = write(&pair.profile)?;
        let sidecar = write(&pair.symbols)?;
        let recording_inode = recording.as_file().metadata()?;
        let sidecar_inode = sidecar.as_file().metadata()?;
        let remove_own_file = |path: &Path, expected: &Metadata| match fs::symlink_metadata(path) {
            Ok(current) if current.dev() == expected.dev() && current.ino() == expected.ino() => {
                fs::remove_file(path)
            }
            Ok(_) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        };
        check_destination()?;
        // Publish the sidecar first so the visible recording never precedes its companion.
        let sidecar = sidecar.persist_noclobber(&symbols)?;
        let recording = match recording.persist_noclobber(&profile) {
            Ok(recording) => recording,
            Err(error) => {
                remove_own_file(&symbols, &sidecar_inode)?;
                return Err(error.into());
            }
        };
        let retained = (|| {
            check_destination()?;
            let recording = recording.metadata()?;
            let sidecar = sidecar.metadata()?;
            if !same_artifact_version(&recording, &fs::symlink_metadata(&profile)?)
                || !same_artifact_version(&sidecar, &fs::symlink_metadata(&symbols)?)
            {
                return Err(io::Error::other(
                    "Retained Native Profile Task files changed",
                ));
            }
            Ok([recording, sidecar])
        })();
        if retained.is_err() {
            let profile_cleanup = remove_own_file(&profile, &recording_inode);
            let sidecar_cleanup = remove_own_file(&symbols, &sidecar_inode);
            profile_cleanup.and(sidecar_cleanup)?;
        }
        retained
    }
}

pub struct LinuxTaskCompletion {
    listener: UnixListener,
    directory: tempfile::TempDir,
}

pub struct LinuxCargoProfileInvocation(CargoProfileRequest);

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinuxCargoRuntimeInvocation {
    cargo: TaskBuildInvocation,
    workload: TaskBuildInvocation,
    runner: PathBuf,
}

impl LinuxCargoRuntimeInvocation {
    pub fn new(
        cargo: TaskBuildInvocation,
        workload_program: PathBuf,
        workload_arguments: Vec<String>,
        runner: PathBuf,
    ) -> io::Result<Self> {
        let workload = TaskBuildInvocation::new(
            workload_program,
            workload_arguments,
            cargo.working_directory().to_path_buf(),
            Default::default(),
        )?;
        let invocation = Self {
            cargo,
            workload,
            runner,
        };
        invocation.validate()?;
        Ok(invocation)
    }

    pub fn cargo(&self) -> &TaskBuildInvocation {
        &self.cargo
    }

    fn validate(&self) -> io::Result<()> {
        self.cargo.validate()?;
        self.workload.validate()?;
        validate_workload_environment(&self.cargo)?;
        let arguments = self.cargo.arguments();
        let (cargo, workload) = arguments
            .iter()
            .position(|argument| argument == "--")
            .and_then(|separator| {
                Some((arguments.get(..separator)?, arguments.get(separator + 1..)?))
            })
            .ok_or_else(invalid_completion)?;
        let host = match std::env::consts::ARCH {
            "aarch64" => "aarch64-unknown-linux-gnu",
            "x86_64" => "x86_64-unknown-linux-gnu",
            _ => return Err(invalid_completion()),
        };
        let targets = cargo
            .iter()
            .enumerate()
            .filter(|(_, argument)| *argument == "--target")
            .map(|(index, _)| cargo.get(index + 1).map(String::as_str));
        if self.cargo.working_directory() != self.workload.working_directory()
            || !self.workload.environment().is_empty()
            || !self.runner.is_absolute()
            || self.runner.as_os_str().as_encoded_bytes().len() > 4096
            || cargo.first().is_none_or(|argument| argument != "run")
            || !targets.eq([Some(host)])
            || cargo
                .iter()
                .any(|argument| argument.starts_with("--target="))
            || workload != self.workload.arguments()
            || self.cargo.environment().keys().any(|key| {
                key == "CARGO_BUILD_TARGET"
                    || key.starts_with("CARGO_UNSTABLE_")
                    || key.starts_with("CARGO_TARGET_") && key.ends_with("_RUNNER")
            })
        {
            return Err(invalid_completion());
        }
        Ok(())
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CargoProfileRequest {
    collector: TaskBuildInvocation,
    workload: TaskBuildInvocation,
    runner: PathBuf,
    workload_digest: Option<[u8; 32]>,
    collector_digest: Option<[u8; 32]>,
}

impl LinuxCargoProfileInvocation {
    pub fn new(
        collector: TaskBuildInvocation,
        workload: ObservedTaskBuildInvocation,
        runner: PathBuf,
    ) -> io::Result<Self> {
        let invocation = CargoProfileRequest {
            collector,
            workload: (*workload.invocation).clone(),
            runner,
            workload_digest: None,
            collector_digest: None,
        };
        invocation.validate()?;
        Ok(Self(invocation))
    }

    pub fn collector(&self) -> &TaskBuildInvocation {
        &self.0.collector
    }

    pub fn with_image_digests(mut self, collector: [u8; 32], workload: [u8; 32]) -> Self {
        self.0.collector_digest = Some(collector);
        self.0.workload_digest = Some(workload);
        self
    }
}

impl CargoProfileRequest {
    fn validate(&self) -> io::Result<()> {
        self.collector.validate()?;
        validate_workload_environment(&self.workload)?;
        let arguments = self.collector.arguments();
        if self.collector.working_directory() != self.workload.working_directory()
            || self.collector_digest.is_some() != self.workload_digest.is_some()
            || !self.runner.is_absolute()
            || self.runner.as_os_str().as_encoded_bytes().len() > 4096
            || self
                .collector
                .environment()
                .keys()
                .any(|key| key.starts_with("LD_") || key.starts_with("ZED_PROFILE_"))
            || arguments
                .first()
                .is_none_or(|argument| argument != "record")
            || arguments.last().is_none_or(|argument| argument != "--")
            || arguments
                .iter()
                .filter(|argument| *argument == "--")
                .count()
                != 1
            || !arguments
                .iter()
                .filter(|argument| argument.starts_with("--include-args"))
                .map(String::as_str)
                .eq(["--include-args=0"])
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid Linux Cargo profile invocation",
            ));
        }
        Ok(())
    }
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
        self.observe_payload(
            actor,
            &invocation.encode_helper_environment()?,
            LinuxTaskMode::CargoBuild,
            report_limit,
            cancel,
        )
        .await
    }

    pub async fn observe_cargo_runtime(
        self,
        actor: OriginalTaskActor,
        invocation: LinuxCargoRuntimeInvocation,
        cancel: impl Future<Output = ()>,
    ) -> io::Result<LinuxTaskReceipt<ObservedTaskBuildInvocation>> {
        invocation.validate()?;
        let request = serde_json::to_string(&invocation).map_err(|_| invalid_completion())?;
        let receipt = self
            .observe_payload(
                actor,
                &request,
                LinuxTaskMode::CargoRuntime,
                MAX_TASK_BUILD_INVOCATION_BYTES,
                cancel,
            )
            .await?;
        let output = receipt
            .output
            .map(|bytes| {
                let workload = serde_json::from_slice::<TaskBuildInvocation>(&bytes)
                    .map_err(|_| invalid_completion())?;
                validate_workload_environment(&workload)?;
                let observed = ObservedTaskBuildInvocation {
                    invocation: std::sync::Arc::new(workload),
                };
                if !invocation.workload.matches_observed(&observed) {
                    return Err(invalid_completion());
                }
                Ok(observed)
            })
            .transpose()?;
        Ok(LinuxTaskReceipt {
            termination: receipt.termination,
            output,
            cancelled: receipt.cancelled,
        })
    }

    async fn observe_payload(
        self,
        actor: OriginalTaskActor,
        request: &str,
        mode: LinuxTaskMode,
        limit: usize,
        cancel: impl Future<Output = ()>,
    ) -> io::Result<LinuxTaskReceipt<Vec<u8>>> {
        let (mut actor, mut connection, challenge) = self
            .register_task(actor, request, mode, Some(limit))
            .await?;
        until_deadline(COMPLETION_TIMEOUT, async {
            let (response, cancelled) =
                receive_task_response(&mut connection, challenge, Some(limit), cancel).await?;
            actor.revalidate()?;
            connection.write_all(&[COMPLETION_ACKNOWLEDGED]).await?;
            Ok(LinuxTaskReceipt {
                termination: response.termination,
                output: response.payload.filter(|_| {
                    !cancelled && response.termination == LinuxTaskTermination::Exited(0)
                }),
                cancelled,
            })
        })
        .await
    }

    pub async fn observe_cargo_profile(
        self,
        actor: OriginalTaskActor,
        invocation: LinuxCargoProfileInvocation,
        profile: NativeProfileTaskReservation,
        cancel: impl Future<Output = ()>,
    ) -> io::Result<LinuxTaskReceipt<SealedNativeProfileTask>> {
        invocation.0.validate()?;
        profile.validate_invocation(invocation.collector())?;
        let request = serde_json::to_string(&invocation.0).map_err(|_| invalid_completion())?;
        self.observe_encoded_output(
            actor,
            &request,
            LinuxTaskMode::CargoProfile,
            Some(profile),
            cancel,
        )
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
        let request = invocation.encode_helper_environment()?;
        self.observe_encoded_output(actor, &request, Output::KIND, output, cancel)
            .await
    }

    async fn observe_encoded_output<Output: TaskOutputReservation>(
        self,
        actor: OriginalTaskActor,
        request: &str,
        mode: LinuxTaskMode,
        output: Option<Output>,
        cancel: impl Future<Output = ()>,
    ) -> io::Result<LinuxTaskReceipt<Output::Sealed>> {
        let (mut actor, mut connection, challenge) =
            self.register_task(actor, request, mode, None).await?;
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
        request: &str,
        kind: LinuxTaskMode,
        stdout_limit: Option<usize>,
    ) -> io::Result<(AuthenticatedTaskActor, UnixStream, [u8; 32])> {
        if request.len() > MAX_TASK_BUILD_INVOCATION_BYTES {
            return Err(invalid_completion());
        }
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
    payload: Option<Vec<u8>>,
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
    let payload = if let Some(limit) = stdout_limit {
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
        payload,
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

async fn run_task(socket: &Path, kind: LinuxTaskMode) -> io::Result<ExitStatus> {
    let (mut connection, request, challenge, parent, stdout_limit) =
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
            let stdout_limit = if matches!(
                kind,
                LinuxTaskMode::CargoBuild | LinuxTaskMode::CargoRuntime
            ) {
                let mut limit = [0; 4];
                connection.read_exact(&mut limit).await?;
                let limit = u32::from_be_bytes(limit) as usize;
                validate_stdout_report_limit(limit)?;
                if kind == LinuxTaskMode::CargoRuntime && limit != MAX_TASK_BUILD_INVOCATION_BYTES {
                    return Err(invalid_completion());
                }
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
            parent.revalidate()?;
            Ok((connection, request, challenge, parent, stdout_limit))
        })
        .await?;

    let command = || {
        let invocation = serde_json::from_slice::<TaskBuildInvocation>(&request)
            .map_err(|_| invalid_completion())?;
        invocation.validate()?;
        Ok::<_, io::Error>(task_command(&invocation, invocation.program()))
    };
    let (status, payload) = match kind {
        LinuxTaskMode::Coverage => (
            run_coverage_workload(command()?, &mut connection).await?,
            None,
        ),
        LinuxTaskMode::NativeProfile => (
            run_profile_workload(command()?, &mut connection).await?,
            None,
        ),
        LinuxTaskMode::CargoProfile => {
            let invocation = serde_json::from_slice::<CargoProfileRequest>(&request)
                .map_err(|_| invalid_completion())?;
            invocation.validate()?;
            let collector = invocation
                .collector_digest
                .map(|digest| open_profile_image(invocation.collector.program(), Some(digest)))
                .transpose()?;
            let mut command = match &collector {
                Some(image) => task_command(
                    &invocation.collector,
                    format!("/proc/self/fd/{}", image.as_raw_fd()),
                ),
                None => task_command(&invocation.collector, invocation.collector.program()),
            };
            configure_profile_workload(
                &mut command,
                &invocation.runner,
                &invocation.workload,
                invocation.workload_digest,
            )?;
            (run_profile_workload(command, &mut connection).await?, None)
        }
        LinuxTaskMode::CargoBuild => {
            let limit = stdout_limit.ok_or_else(invalid_completion)?;
            let (status, stdout) =
                run_cargo_build_workload(command()?, &mut connection, limit).await?;
            (status, Some(stdout))
        }
        LinuxTaskMode::CargoRuntime => {
            let invocation = serde_json::from_slice::<LinuxCargoRuntimeInvocation>(&request)
                .map_err(|_| invalid_completion())?;
            invocation.validate()?;
            let (status, runtime) = run_cargo_runtime_workload(invocation, &mut connection).await?;
            (status, Some(runtime))
        }
    };

    let termination = LinuxTaskTermination::from_status(status)?;
    parent.revalidate()?;
    until_deadline(REGISTRATION_TIMEOUT, async {
        connection.write_all(&challenge).await?;
        connection.write_all(&termination.encode()).await?;
        if let Some(payload) = payload {
            connection
                .write_all(&(payload.len() as u32).to_be_bytes())
                .await?;
            connection.write_all(&payload).await?;
        }
        connection.shutdown(Shutdown::Write)?;
        let mut acknowledged = [0; 1];
        connection.read_exact(&mut acknowledged).await?;
        if !matches!(kind, LinuxTaskMode::Coverage) && acknowledged == [CANCEL_NATIVE_TASK] {
            connection.read_exact(&mut acknowledged).await?;
        }
        if acknowledged != [COMPLETION_ACKNOWLEDGED] {
            return Err(invalid_completion());
        }
        Ok(status)
    })
    .await
}

fn task_command(invocation: &TaskBuildInvocation, executable: impl AsRef<OsStr>) -> Command {
    let mut command = Command::new(executable);
    command
        .arg0(invocation.program())
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
    command
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

async fn run_cargo_runtime_workload(
    invocation: LinuxCargoRuntimeInvocation,
    connection: &mut UnixStream,
) -> io::Result<(ExitStatus, Vec<u8>)> {
    let cargo_image = File::options()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(invocation.cargo.program())?;
    if !cargo_image.metadata()?.is_file() {
        return Err(invalid_completion());
    }
    let mut command = task_command(
        &invocation.cargo,
        format!("/proc/self/fd/{}", cargo_image.as_raw_fd()),
    );
    let pending = LinuxCargoRuntimeObserver::bind(&invocation.runner)?.configure(&mut command)?;
    let child = spawn_task_group(command)?;
    let group = child.id() as libc::pid_t;
    let observe = async {
        let actor = OriginalTaskActor::from_owned_unreaped_child(&child, &cargo_image)?;
        let receipt = future::or(
            async {
                pending
                    .observe(
                        actor,
                        invocation.workload.program(),
                        invocation.workload.arguments(),
                        invocation.workload.working_directory(),
                    )
                    .await
                    .map(Some)
            },
            async { wait_for_task_exit(group).await.map(|()| None) },
        )
        .await?;
        if receipt.is_some() {
            wait_for_task_exit(group).await?;
        }
        Ok::<_, io::Error>((false, receipt))
    };
    let result = future::or(
        async {
            wait_for_task_cancellation(connection).await?;
            Ok((true, None))
        },
        observe,
    )
    .await;
    let status = smol::unblock(move || finish_task_group(child, group)).await?;
    let (cancelled, receipt) = result?;
    if cancelled || !status.success() {
        return Ok((status, Vec::new()));
    }
    let runtime = receipt
        .ok_or_else(invalid_completion)?
        .verify_exit(status)?;
    Ok((
        status,
        runtime.invocation.encode_helper_environment()?.into_bytes(),
    ))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    #[test]
    fn retain_profile_publishes_only_the_sealed_pair_to_the_reviewed_directory() -> io::Result<()> {
        let seal = || -> io::Result<SealedNativeProfileTask> {
            let reservation = NativeProfileTaskReservation::new()?;
            fs::write(reservation.profile_path(), b"recording")?;
            fs::write(reservation.symbols_path(), b"symbols")?;
            reservation.seal()
        };
        let root = tempfile::tempdir_in("/tmp")?;
        let paths = ["capture.json", "capture.syms.json"].map(|name| root.path().join(name));
        seal()?.retain(NativeProfileTaskDestination::open(paths.clone())?, || true)?;
        for (path, expected) in paths.iter().zip([b"recording".as_slice(), b"symbols"]) {
            assert_eq!(fs::read(path)?, expected);
            assert_eq!(fs::metadata(path)?.mode() & 0o777, 0o600);
            fs::remove_file(path)?;
        }

        let destination = NativeProfileTaskDestination::open(paths.clone())?;
        fs::write(&paths[0], b"existing recording")?;
        assert_eq!(
            destination
                .check_available()
                .expect_err("Accepted an occupied recording")
                .kind(),
            io::ErrorKind::AlreadyExists
        );
        assert_eq!(
            seal()?
                .retain(destination, || true)
                .expect_err("Overwrote a recording")
                .kind(),
            io::ErrorKind::AlreadyExists
        );
        assert_eq!(fs::read(&paths[0])?, b"existing recording");
        assert!(!paths[1].exists());
        fs::remove_file(&paths[0])?;

        let profile = seal()?;
        fs::write(profile.reservation.symbols_path(), b"changed after seal")?;
        assert!(
            profile
                .retain(NativeProfileTaskDestination::open(paths.clone())?, || true)
                .is_err()
        );
        assert!(paths.iter().all(|path| !path.exists()));

        let destination = NativeProfileTaskDestination::open(paths.clone())?;
        assert_eq!(
            seal()?
                .retain(destination, || !paths[0].exists())
                .expect_err("Retained a recording after cancellation")
                .kind(),
            io::ErrorKind::Interrupted
        );
        assert!(paths.iter().all(|path| !path.exists()));

        let output = root.path().join("output");
        let moved = root.path().join("moved");
        fs::create_dir(&output)?;
        let paths = ["capture.json", "capture.syms.json"].map(|name| output.join(name));
        let destination = NativeProfileTaskDestination::open(paths.clone())?;
        fs::rename(&output, &moved)?;
        std::os::unix::fs::symlink(&moved, &output)?;
        assert!(NativeProfileTaskDestination::open(paths.clone()).is_err());
        assert!(seal()?.retain(destination, || true).is_err());
        assert!(paths.iter().all(|path| !path.exists()));
        Ok(())
    }

    #[test]
    fn cargo_runtime_task_requires_the_native_target_and_exact_workload_arguments() {
        let host = format!("{}-unknown-linux-gnu", std::env::consts::ARCH);
        let literal = ["two words", "", "$FOO"].map(str::to_owned).to_vec();
        let invoke = |target: &str, workload: Vec<String>, configured_runner: bool| {
            let environment = if configured_runner {
                [(
                    format!(
                        "CARGO_TARGET_{}_RUNNER",
                        host.replace('-', "_").to_uppercase()
                    ),
                    "other-runner".to_owned(),
                )]
                .into()
            } else {
                Default::default()
            };
            let cargo = TaskBuildInvocation::new(
                "/toolchain/bin/cargo".into(),
                ["run", "--target", target, "--"]
                    .map(str::to_owned)
                    .into_iter()
                    .chain(literal.clone())
                    .collect(),
                "/project".into(),
                environment,
            )?;
            LinuxCargoRuntimeInvocation::new(
                cargo,
                "/project/runner".into(),
                workload,
                "/application/helper".into(),
            )
        };
        assert!(invoke(&host, literal.clone(), false).is_ok());
        assert!(invoke("wasm32-unknown-unknown", literal.clone(), false).is_err());
        assert!(invoke(&host, vec!["changed".into()], false).is_err());
        assert!(invoke(&host, literal.clone(), true).is_err());
    }

    #[test]
    fn linux_task_helper_accepts_only_the_exact_mode_and_socket() -> io::Result<()> {
        for mode in [
            LinuxTaskMode::Coverage,
            LinuxTaskMode::NativeProfile,
            LinuxTaskMode::CargoBuild,
            LinuxTaskMode::CargoProfile,
            LinuxTaskMode::CargoRuntime,
        ] {
            let address = "/tmp/task path/completion";
            assert_eq!(
                parse_task_helper(
                    [mode.command(), "--socket", address]
                        .into_iter()
                        .map(OsString::from)
                )?,
                Some((mode, PathBuf::from(address))),
            );
            for args in [
                vec![mode.command()],
                vec![mode.command(), "--socket"],
                vec![mode.command(), "--wrong", address],
                vec![mode.command(), "--socket", "relative"],
                vec![mode.command(), "--socket", "/tmp/../completion"],
                vec![mode.command(), "--socket", "/"],
                vec![
                    mode.command(),
                    "--socket",
                    address,
                    "private-extra-argument",
                ],
            ] {
                let error = parse_task_helper(args.into_iter().map(OsString::from))
                    .expect_err("Malformed helper arguments were admitted");
                assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
                assert!(!error.to_string().contains("private-extra-argument"));
            }
            let long_address = format!("/tmp/{}", "x".repeat(108));
            assert!(
                parse_task_helper(
                    [mode.command(), "--socket", &long_address]
                        .into_iter()
                        .map(OsString::from)
                )
                .is_err()
            );
        }
        assert_eq!(parse_task_helper(std::iter::empty())?, None);
        assert_eq!(
            parse_task_helper([OsString::from("observe-cargo-build-task")].into_iter())?,
            None,
        );
        assert_eq!(
            parse_task_helper(
                ["--foreground", "--observe-cargo-build-task"]
                    .into_iter()
                    .map(OsString::from)
            )?,
            None
        );
        Ok(())
    }
}
