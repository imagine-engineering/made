use crate::{
    task_actor::{OriginalTaskActor, require_task_parent},
    task_completion::{
        MAX_TASK_BUILD_INVOCATION_BYTES, ObservedTaskBuildInvocation,
        TASK_CARGO_RUNTIME_ENDPOINT_ENV, TaskBuildInvocation,
    },
};
use futures_lite::{AsyncWriteExt as _, future};
use smol::net::unix::UnixListener;
use std::{
    collections::BTreeMap,
    ffi::{CStr, OsStr},
    fs::{File, Permissions},
    io::{self, Read as _},
    os::{
        fd::{AsRawFd as _, FromRawFd as _, RawFd},
        unix::{
            fs::{FileExt as _, MetadataExt as _, OpenOptionsExt as _, PermissionsExt as _},
            process::CommandExt as _,
        },
    },
    path::{Path, PathBuf},
    process::{Command, ExitStatus},
    sync::Arc,
    time::Duration,
};

const RUNNER_MODE: &str = "--observe-cargo-runtime";
const WORKLOAD_MODE: &str = "--exec-cargo-workload";
const MAX_RUNNER_BYTES: u64 = 32 * 1024 * 1024;
const ACKNOWLEDGED: u8 = 7;
const IMMUTABLE_FILE_SEALS: libc::c_int =
    libc::F_SEAL_WRITE | libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_SEAL;

pub struct LinuxCargoRuntimeObserver {
    runner: File,
    listener: UnixListener,
    directory: tempfile::TempDir,
}

pub struct PendingLinuxCargoRuntime {
    observer: LinuxCargoRuntimeObserver,
    runner_path: String,
    runner_key: &'static str,
    runner_value: String,
}

pub struct LinuxCargoRuntimeReceipt(ObservedTaskBuildInvocation);

impl LinuxCargoRuntimeReceipt {
    pub fn verify_exit(self, exit: ExitStatus) -> io::Result<ObservedTaskBuildInvocation> {
        if !exit.success() {
            return Err(io::Error::other(
                "The original Cargo runtime runner did not succeed",
            ));
        }
        Ok(self.0)
    }
}

impl LinuxCargoRuntimeObserver {
    /// The caller supplies its admitted helper image. The sealed copy is what Cargo executes.
    pub fn bind(image: &Path) -> io::Result<Self> {
        let runner = seal_static_runner(image)?;
        let directory = tempfile::Builder::new()
            .prefix("zed-cargo-runtime-")
            .tempdir_in("/tmp")?;
        std::fs::set_permissions(directory.path(), Permissions::from_mode(0o700))?;
        let socket = directory.path().join("runtime");
        let listener = UnixListener::bind(&socket)?;
        std::fs::set_permissions(&socket, Permissions::from_mode(0o600))?;
        Ok(Self {
            runner,
            listener,
            directory,
        })
    }

    fn socket(&self) -> PathBuf {
        self.directory.path().join("runtime")
    }

    pub fn configure(self, command: &mut Command) -> io::Result<PendingLinuxCargoRuntime> {
        let runner_key = match std::env::consts::ARCH {
            "aarch64" => "CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER",
            "x86_64" => "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER",
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "Unsupported Linux Cargo runtime architecture",
                ));
            }
        };
        if command
            .get_envs()
            .any(|(name, _)| name == runner_key || name == TASK_CARGO_RUNTIME_ENDPOINT_ENV)
        {
            return Err(invalid_runtime());
        }
        let runner = inherit_descriptor(command, &self.runner)?;
        let runner_path = descriptor_path(runner);
        let runner_value = format!("{runner_path} {RUNNER_MODE} --");
        command
            .env(runner_key, &runner_value)
            .env(TASK_CARGO_RUNTIME_ENDPOINT_ENV, self.socket());
        Ok(PendingLinuxCargoRuntime {
            observer: self,
            runner_path,
            runner_key,
            runner_value,
        })
    }
}

impl PendingLinuxCargoRuntime {
    #[expect(
        clippy::disallowed_methods,
        reason = "The independent native helper has no GPUI executor"
    )]
    pub async fn observe(
        self,
        cargo: OriginalTaskActor,
        program: &Path,
        arguments: &[String],
        working_directory: &Path,
    ) -> io::Result<LinuxCargoRuntimeReceipt> {
        let expected_directory = File::options()
            .read(true)
            .custom_flags(libc::O_DIRECTORY)
            .open(working_directory)?
            .metadata()?;
        let expected = cargo.expect_exec(self.observer.runner.try_clone()?)?;
        future::or(
            async {
                let (mut connection, _) = self.observer.listener.accept().await?;
                let mut runner = expected.authenticate(&connection)?;
                let snapshot = runner.runtime_snapshot()?;
                if !expected_directory.is_dir()
                    || expected_directory.dev() != snapshot.directory.dev()
                    || expected_directory.ino() != snapshot.directory.ino()
                {
                    return Err(invalid_runtime());
                }

                let mut actual = nul_fields(&snapshot.arguments)?;
                if !actual
                    .by_ref()
                    .take(3)
                    .eq([self.runner_path.as_str(), RUNNER_MODE, "--"])
                {
                    return Err(invalid_runtime());
                }
                let observed_program = Path::new(actual.next().ok_or_else(invalid_runtime)?);
                if working_directory.join(observed_program) != program
                    || !actual.eq(arguments.iter().map(String::as_str))
                {
                    return Err(invalid_runtime());
                }

                let socket = self.observer.socket();
                let mut observed = BTreeMap::new();
                for field in nul_fields(&snapshot.environment)? {
                    let (key, value) = field.split_once('=').ok_or_else(invalid_runtime)?;
                    if key.starts_with("LD_") && key != "LD_LIBRARY_PATH" && !value.is_empty()
                        || key.starts_with("ZED_PROFILE_") && key != TASK_CARGO_RUNTIME_ENDPOINT_ENV
                        || observed.insert(key.to_owned(), value.to_owned()).is_some()
                    {
                        return Err(invalid_runtime());
                    }
                }
                if observed.remove(self.runner_key).as_deref() != Some(self.runner_value.as_str())
                    || observed
                        .remove(TASK_CARGO_RUNTIME_ENDPOINT_ENV)
                        .as_deref()
                        .map(Path::new)
                        != Some(socket.as_path())
                {
                    return Err(invalid_runtime());
                }
                let invocation = TaskBuildInvocation::new(
                    program.to_path_buf(),
                    arguments.to_vec(),
                    working_directory.to_path_buf(),
                    observed,
                )?;
                runner.revalidate()?;
                connection.write_all(&[ACKNOWLEDGED]).await?;
                Ok(LinuxCargoRuntimeReceipt(ObservedTaskBuildInvocation {
                    invocation: Arc::new(invocation),
                }))
            },
            async {
                smol::Timer::after(Duration::from_secs(30 * 60)).await;
                Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "The original Cargo runtime runner did not register",
                ))
            },
        )
        .await
    }
}

pub fn run_linux_cargo_runtime_helper() -> ! {
    let mut arguments = std::env::args_os();
    let executable = arguments.next();
    let mode = arguments.next();
    if mode.as_deref() == Some(OsStr::new(WORKLOAD_MODE)) {
        let error = exec_cargo_workload(executable.as_deref(), arguments);
        eprintln!("Linux Cargo workload launch failed: {error}");
        std::process::exit(2);
    }
    let outcome = (|| {
        if mode.is_none_or(|arg| arg != RUNNER_MODE)
            || arguments.next().is_none_or(|arg| arg != "--")
            || arguments.next().is_none()
        {
            return Err(invalid_runtime());
        }
        let socket = std::env::var_os(TASK_CARGO_RUNTIME_ENDPOINT_ENV)
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
            .ok_or_else(invalid_runtime)?;
        let mut connection = std::os::unix::net::UnixStream::connect(socket)?;
        require_task_parent(&connection)?;
        connection.set_read_timeout(Some(Duration::from_secs(10)))?;
        let mut response = [0];
        connection.read_exact(&mut response)?;
        if response != [ACKNOWLEDGED] {
            return Err(invalid_runtime());
        }
        Ok(())
    })();
    if let Err(error) = outcome {
        eprintln!("Linux Cargo runtime observation failed: {error}");
        std::process::exit(2);
    }
    std::process::exit(0)
}

pub(crate) fn configure_profile_workload(
    command: &mut Command,
    runner_path: &Path,
    workload: &TaskBuildInvocation,
) -> io::Result<()> {
    use std::io::Write as _;

    let runner = seal_static_runner(runner_path)?;
    let program = open_regular_image(workload.program())?;
    verify_elf_image(&program, false)?;
    let payload = workload.encode_helper_environment()?;
    let payload = seal_file(c"zed-cargo-workload", 0o400, |file| {
        file.write_all(payload.as_bytes())
    })?;
    let runner = inherit_descriptor(command, &runner)?;
    let program = inherit_descriptor(command, &program)?;
    let payload = inherit_descriptor(command, &payload)?;
    command.args([
        descriptor_path(runner),
        WORKLOAD_MODE.to_owned(),
        payload.to_string(),
        program.to_string(),
    ]);
    Ok(())
}

fn exec_cargo_workload(
    executable: Option<&OsStr>,
    mut arguments: impl Iterator<Item = std::ffi::OsString>,
) -> io::Error {
    let mut run = || -> io::Result<()> {
        let runner = executable
            .and_then(OsStr::to_str)
            .and_then(|value| value.strip_prefix("/proc/self/fd/"))
            .and_then(parse_descriptor)
            .ok_or_else(invalid_runtime)?;
        let payload = arguments
            .next()
            .as_deref()
            .and_then(OsStr::to_str)
            .and_then(parse_descriptor)
            .ok_or_else(invalid_runtime)?;
        let program = arguments
            .next()
            .as_deref()
            .and_then(OsStr::to_str)
            .and_then(parse_descriptor)
            .ok_or_else(invalid_runtime)?;
        if arguments.next().is_some()
            || runner == payload
            || runner == program
            || payload == program
        {
            return Err(invalid_runtime());
        }
        for descriptor in [runner, payload, program] {
            if unsafe { libc::fcntl(descriptor, libc::F_SETFD, libc::FD_CLOEXEC) } == -1 {
                return Err(io::Error::last_os_error());
            }
        }
        let seals = unsafe { libc::fcntl(payload, libc::F_GET_SEALS) };
        if seals == -1 || seals & IMMUTABLE_FILE_SEALS != IMMUTABLE_FILE_SEALS {
            return Err(invalid_runtime());
        }
        let mut request = Vec::new();
        File::open(descriptor_path(payload))?
            .take((MAX_TASK_BUILD_INVOCATION_BYTES + 1) as u64)
            .read_to_end(&mut request)?;
        if request.len() > MAX_TASK_BUILD_INVOCATION_BYTES {
            return Err(invalid_runtime());
        }
        let invocation = serde_json::from_slice::<TaskBuildInvocation>(&request)
            .map_err(|_| invalid_runtime())?;
        validate_workload_environment(&invocation)?;
        Err(Command::new(descriptor_path(program))
            .arg0(invocation.program())
            .args(invocation.arguments())
            .current_dir(invocation.working_directory())
            .env_clear()
            .envs(invocation.environment())
            .exec())
    };
    match run() {
        Err(error) => error,
        Ok(()) => invalid_runtime(),
    }
}

pub(crate) fn validate_workload_environment(invocation: &TaskBuildInvocation) -> io::Result<()> {
    invocation.validate()?;
    if invocation.environment().iter().any(|(key, value)| {
        key.starts_with("ZED_PROFILE_")
            || key.starts_with("LD_") && key != "LD_LIBRARY_PATH" && !value.is_empty()
    }) {
        return Err(invalid_runtime());
    }
    Ok(())
}

fn parse_descriptor(value: &str) -> Option<RawFd> {
    value
        .parse::<RawFd>()
        .ok()
        .filter(|descriptor| *descriptor >= 3)
}

fn descriptor_path(descriptor: RawFd) -> String {
    format!("/proc/self/fd/{descriptor}")
}

fn inherit_descriptor(command: &mut Command, file: &File) -> io::Result<RawFd> {
    let descriptor = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 3) };
    if descriptor < 0 {
        return Err(io::Error::last_os_error());
    }
    let file = unsafe { File::from_raw_fd(descriptor) };
    // Keep the borrowed file live until the child can inherit it.
    unsafe {
        command.pre_exec(move || {
            if libc::fcntl(file.as_raw_fd(), libc::F_SETFD, 0) == -1 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
    Ok(descriptor)
}

fn nul_fields(bytes: &[u8]) -> io::Result<impl Iterator<Item = &str>> {
    if !bytes.is_empty() && bytes.last() != Some(&0) {
        return Err(invalid_runtime());
    }
    Ok(std::str::from_utf8(bytes)
        .map_err(|_| invalid_runtime())?
        .split_terminator('\0'))
}

fn seal_static_runner(path: &Path) -> io::Result<File> {
    let source = open_regular_image(path)?;
    let image = seal_file(c"zed-cargo-runtime", 0o500, |file| {
        if io::copy(&mut source.take(MAX_RUNNER_BYTES + 1), file)? > MAX_RUNNER_BYTES {
            return Err(invalid_runtime());
        }
        Ok(())
    })?;
    verify_elf_image(&image, true)?;
    Ok(image)
}

fn open_regular_image(path: &Path) -> io::Result<File> {
    let file = File::options()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    if !file.metadata()?.is_file() {
        return Err(invalid_runtime());
    }
    Ok(file)
}

fn seal_file(
    name: &CStr,
    permissions: u32,
    write: impl FnOnce(&mut File) -> io::Result<()>,
) -> io::Result<File> {
    let descriptor =
        unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING) };
    if descriptor < 0 {
        return Err(io::Error::last_os_error());
    }
    let mut image = unsafe { File::from_raw_fd(descriptor) };
    write(&mut image)?;
    image.set_permissions(Permissions::from_mode(permissions))?;
    if unsafe { libc::fcntl(image.as_raw_fd(), libc::F_ADD_SEALS, IMMUTABLE_FILE_SEALS) } == -1 {
        return Err(io::Error::last_os_error());
    }
    File::open(descriptor_path(image.as_raw_fd()))
}

fn verify_elf_image(image: &File, require_static: bool) -> io::Result<()> {
    let mut header = [0; 64];
    image.read_exact_at(&mut header, 0)?;
    let architecture = match std::env::consts::ARCH {
        "aarch64" => 183,
        "x86_64" => 62,
        _ => return Err(invalid_runtime()),
    };
    if !header.starts_with(b"\x7fELF\x02\x01\x01")
        || !matches!(u16::from_le_bytes([header[16], header[17]]), 2 | 3)
        || u16::from_le_bytes([header[18], header[19]]) != architecture
        || u16::from_le_bytes([header[54], header[55]]) != 56
    {
        return Err(invalid_runtime());
    }
    let offset = u64::from_le_bytes(header[32..40].try_into().map_err(|_| invalid_runtime())?);
    let entries = u64::from(u16::from_le_bytes([header[56], header[57]]));
    if !(1..=256).contains(&entries)
        || offset < 64
        || offset
            .checked_add(56 * entries)
            .is_none_or(|end| end > MAX_RUNNER_BYTES)
    {
        return Err(invalid_runtime());
    }
    let mut executable = false;
    for index in 0..entries {
        let mut segment = [0; 56];
        image.read_exact_at(&mut segment, offset + 56 * index)?;
        let kind = u32::from_le_bytes(segment[..4].try_into().map_err(|_| invalid_runtime())?);
        if require_static && kind == 3 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "The Linux Cargo runner requires a static executable",
            ));
        }
        executable |= kind == 1 && segment[4] & 1 == 1;
    }
    if !executable {
        return Err(invalid_runtime());
    }
    Ok(())
}

fn invalid_runtime() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "Invalid Linux Cargo runtime observation",
    )
}
