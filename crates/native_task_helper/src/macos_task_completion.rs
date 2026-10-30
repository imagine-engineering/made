use std::{
    io,
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd},
        unix::{fs::PermissionsExt, process::ExitStatusExt},
    },
    path::Path,
    process::ExitStatus,
    time::Duration,
};

use futures_lite::{AsyncReadExt, AsyncWriteExt, future};
use smol::{Async, Timer, net::unix::UnixListener};

const REGISTRATION_TIMEOUT: Duration = Duration::from_secs(30);
const COMPLETION_TIMEOUT: Duration = Duration::from_secs(60 * 60);
pub use crate::task_completion::{
    MAX_TASK_BUILD_INVOCATION_BYTES, MAX_TASK_STDOUT_REPORT_BYTES, ObservedTaskBuildInvocation,
    TASK_BUILD_INVOCATION_ENV, TASK_BUILD_INVOCATION_RESERVED_BYTES,
    TASK_CARGO_RUNTIME_ENDPOINT_ENV, TASK_CARGO_RUNTIME_TOKEN_ENV, TASK_COMPLETION_TOKEN_ENV,
    TASK_NATIVE_INVOCATION_ENV, TaskBuildInvocation,
};
use crate::task_completion::{
    TASK_STDOUT_FRAME_BYTES, capture_owned_task_stdout, read_stdout_report,
    validate_stdout_report_limit,
};
const CARGO_RUNTIME_REGISTRATION: [u8; 4] = *b"CR6\0";
const CARGO_RUNTIME_FRAME: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalTaskHelperMode {
    Exec,
    Stdout,
    CheckedBuildStdout,
    NativeProfileCapture,
    CheckedNativeProfileCapture,
    CheckedCargoRuntimeStdout,
    CargoRuntimeObserver,
}

pub struct LocalTaskHelperInvocation {
    pub socket: std::path::PathBuf,
    pub mode: LocalTaskHelperMode,
    pub program: std::ffi::OsString,
    pub arguments: Vec<std::ffi::OsString>,
}

pub fn parse_local_task_helper_invocation(
    mut arguments: impl Iterator<Item = std::ffi::OsString>,
) -> io::Result<Option<LocalTaskHelperInvocation>> {
    use std::os::unix::ffi::OsStrExt as _;
    const MAX_ARGUMENTS: usize = 512;
    const MAX_ARGUMENT_BYTES: usize = 4096;
    const MAX_TOTAL_BYTES: usize = 2 * 1024 * 1024;
    let invalid = || io::Error::new(io::ErrorKind::InvalidInput, "Invalid Task helper arguments");
    let entry = arguments.next();
    if entry.as_deref() == Some(std::ffi::OsStr::new("--internal-cargo-runtime-observer")) {
        if arguments.next().as_deref() != Some(std::ffi::OsStr::new("--")) {
            return Err(invalid());
        }
        let socket = std::env::var_os(TASK_CARGO_RUNTIME_ENDPOINT_ENV).ok_or_else(invalid)?;
        let path = Path::new(&socket);
        if !path.is_absolute()
            || !socket
                .to_str()
                .is_some_and(|value| value.len() < 104 && !value.chars().any(char::is_control))
        {
            return Err(invalid());
        }
        let program = arguments.next().ok_or_else(invalid)?;
        if !program.to_str().is_some_and(|value| {
            !value.is_empty() && value.len() <= 4096 && !value.chars().any(char::is_control)
        }) {
            return Err(invalid());
        }
        let mut values = Vec::new();
        let mut bytes = 0_usize;
        for value in arguments {
            let text = value.to_str().ok_or_else(invalid)?;
            bytes = bytes.checked_add(text.len()).ok_or_else(invalid)?;
            if values.len() >= 512 || text.len() > 4096 || text.contains('\0') || bytes > 32 * 1024
            {
                return Err(invalid());
            }
            values.push(value);
        }
        return Ok(Some(LocalTaskHelperInvocation {
            socket: socket.into(),
            mode: LocalTaskHelperMode::CargoRuntimeObserver,
            program,
            arguments: values,
        }));
    }
    if entry.as_deref() != Some(std::ffi::OsStr::new("--internal-task-observer")) {
        return Ok(None);
    }
    if arguments.next().as_deref() != Some(std::ffi::OsStr::new("--socket")) {
        return Err(invalid());
    }
    let socket = arguments.next().ok_or_else(invalid)?;
    let socket_path = Path::new(&socket);
    if !socket_path.is_absolute()
        || !socket
            .to_str()
            .is_some_and(|value| value.len() < 104 && !value.chars().any(char::is_control))
    {
        return Err(invalid());
    }
    let mut separator = arguments.next().ok_or_else(invalid)?;
    let mode = match separator.to_str() {
        Some("--capture-stdout") => LocalTaskHelperMode::Stdout,
        Some("--capture-build-stdout") => LocalTaskHelperMode::CheckedBuildStdout,
        Some("--checked-cargo-runtime-stdout") => LocalTaskHelperMode::CheckedCargoRuntimeStdout,
        Some("--native-profile-capture") => LocalTaskHelperMode::NativeProfileCapture,
        Some("--checked-native-profile-capture") => {
            LocalTaskHelperMode::CheckedNativeProfileCapture
        }
        _ => LocalTaskHelperMode::Exec,
    };
    if mode != LocalTaskHelperMode::Exec {
        separator = arguments.next().ok_or_else(invalid)?;
    }
    if separator != "--" {
        return Err(invalid());
    }
    let program = arguments.next().ok_or_else(invalid)?;
    if !Path::new(&program).is_absolute()
        || !program.to_str().is_some_and(|value| {
            value.len() <= MAX_ARGUMENT_BYTES && !value.chars().any(char::is_control)
        })
    {
        return Err(invalid());
    }
    let mut total = socket.as_bytes().len() + program.as_bytes().len();
    let mut workload_arguments = Vec::new();
    for argument in arguments {
        if workload_arguments.len() >= MAX_ARGUMENTS
            || argument.as_bytes().len() > MAX_ARGUMENT_BYTES
            || argument.as_bytes().contains(&0)
        {
            return Err(invalid());
        }
        total = total
            .checked_add(argument.as_bytes().len())
            .ok_or_else(invalid)?;
        if total > MAX_TOTAL_BYTES {
            return Err(invalid());
        }
        workload_arguments.push(argument);
    }
    Ok(Some(LocalTaskHelperInvocation {
        socket: std::path::PathBuf::from(socket),
        mode,
        program,
        arguments: workload_arguments,
    }))
}

pub fn run_local_task_helper_if_invoked() {
    let invocation = match parse_local_task_helper_invocation(std::env::args_os().skip(1)) {
        Ok(Some(invocation)) => invocation,
        Ok(None) => return,
        Err(error) => {
            eprintln!("Task helper arguments were rejected ({:?})", error.kind());
            std::process::exit(2);
        }
    };
    run_app_task_helper_invocation(invocation);
}

fn run_app_task_helper_invocation(invocation: LocalTaskHelperInvocation) -> ! {
    if matches!(
        invocation.mode,
        LocalTaskHelperMode::CheckedCargoRuntimeStdout | LocalTaskHelperMode::CargoRuntimeObserver
    ) {
        eprintln!("Native Cargo runtime modes require the dedicated helper image");
        std::process::exit(2);
    }
    run_local_task_helper_invocation(invocation)
}

pub fn run_native_cargo_runtime_helper() -> ! {
    let invocation = match parse_local_task_helper_invocation(std::env::args_os().skip(1)) {
        Ok(Some(invocation)) => invocation,
        Ok(_) => {
            eprintln!("Unsupported native Cargo runtime helper mode");
            std::process::exit(2);
        }
        Err(error) => {
            eprintln!(
                "Native Cargo runtime helper arguments were rejected ({:?})",
                error.kind()
            );
            std::process::exit(2);
        }
    };
    run_native_cargo_runtime_helper_invocation(invocation)
}

fn run_native_cargo_runtime_helper_invocation(invocation: LocalTaskHelperInvocation) -> ! {
    if !matches!(
        invocation.mode,
        LocalTaskHelperMode::CheckedCargoRuntimeStdout | LocalTaskHelperMode::CargoRuntimeObserver
    ) {
        eprintln!("Unsupported native Cargo runtime helper mode");
        std::process::exit(2);
    }
    run_local_task_helper_invocation(invocation)
}

fn run_local_task_helper_invocation(invocation: LocalTaskHelperInvocation) -> ! {
    if invocation.mode == LocalTaskHelperMode::CargoRuntimeObserver {
        match observe_cargo_runtime(
            &invocation.socket,
            &invocation.program,
            &invocation.arguments,
        ) {
            Ok(()) => std::process::exit(0),
            Err(error) => {
                eprintln!("Cargo runtime observation failed ({:?})", error.kind());
                std::process::exit(1);
            }
        }
    }
    let mut command = std::process::Command::new(invocation.program);
    command.args(invocation.arguments);
    let error = match invocation.mode {
        LocalTaskHelperMode::Stdout => {
            match register_and_capture_stdout(&invocation.socket, command) {
                Ok(status) => exit_with_observed_task_status(status),
                Err(error) => error,
            }
        }
        LocalTaskHelperMode::CheckedBuildStdout => {
            match register_and_capture_build_stdout(&invocation.socket, command) {
                Ok(status) => exit_with_observed_task_status(status),
                Err(error) => error,
            }
        }
        LocalTaskHelperMode::CheckedCargoRuntimeStdout => {
            match register_and_capture_cargo_runtime_stdout(&invocation.socket, command) {
                Ok(status) => exit_with_observed_task_status(status),
                Err(error) => error,
            }
        }
        LocalTaskHelperMode::CargoRuntimeObserver => io::Error::new(
            io::ErrorKind::InvalidInput,
            "Cargo observer cannot execute a command",
        ),
        LocalTaskHelperMode::Exec => register_and_exec(&invocation.socket, &mut command),
        LocalTaskHelperMode::NativeProfileCapture => {
            register_and_exec_native_profile_capture(&invocation.socket, &mut command)
        }
        LocalTaskHelperMode::CheckedNativeProfileCapture => {
            register_and_exec_checked_native_profile_capture(&invocation.socket, &mut command)
        }
    };
    eprintln!("Task helper failed ({:?})", error.kind());
    std::process::exit(1);
}

pub struct ObservedTaskStdout {
    status: ExitStatus,
    stdout: Vec<u8>,
    invocation: Option<ObservedTaskBuildInvocation>,
    cargo_runtime: Option<ObservedCargoRuntime>,
}

pub struct ObservedCargoRuntime {
    invocation: TaskBuildInvocation,
}

impl ObservedCargoRuntime {
    pub fn program(&self) -> &Path {
        self.invocation.program()
    }
    pub fn arguments(&self) -> &[String] {
        self.invocation.arguments()
    }
    pub fn working_directory(&self) -> &Path {
        self.invocation.working_directory()
    }
    pub fn environment(&self) -> &std::collections::BTreeMap<String, String> {
        self.invocation.environment()
    }
}

impl ObservedTaskStdout {
    pub fn status(&self) -> ExitStatus {
        self.status
    }

    pub fn stdout(&self) -> &[u8] {
        &self.stdout
    }

    pub fn into_parts(self) -> (ExitStatus, Vec<u8>) {
        (self.status, self.stdout)
    }

    pub fn into_parts_with_cargo_runtime(
        self,
    ) -> (
        ExitStatus,
        Vec<u8>,
        Option<ObservedTaskBuildInvocation>,
        Option<ObservedCargoRuntime>,
    ) {
        (
            self.status,
            self.stdout,
            self.invocation,
            self.cargo_runtime,
        )
    }

    pub fn into_parts_with_invocation(
        self,
    ) -> (ExitStatus, Vec<u8>, Option<ObservedTaskBuildInvocation>) {
        (self.status, self.stdout, self.invocation)
    }
}

#[expect(
    clippy::disallowed_methods,
    reason = "The standalone exec helper has no GPUI executor; deadlines bound real OS process observation"
)]
fn observation_deadline(duration: Duration) -> Timer {
    Timer::after(duration)
}

pub struct TaskCompletionListener {
    listener: UnixListener,
    directory: tempfile::TempDir,
    token: [u8; 32],
}

impl TaskCompletionListener {
    pub fn bind(token: [u8; 32]) -> io::Result<Self> {
        // The macOS user temporary directory can exceed sockaddr_un's path limit.
        let directory = tempfile::Builder::new()
            .prefix("zed-task-")
            .tempdir_in("/tmp")?;
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700))?;
        let path = directory.path().join("completion");
        let listener = UnixListener::bind(&path)?;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        Ok(Self {
            listener,
            directory,
            token,
        })
    }

    pub fn path(&self) -> std::path::PathBuf {
        self.directory.path().join("completion")
    }

    pub async fn observe(self) -> io::Result<ExitStatus> {
        self.observe_with_deadlines(REGISTRATION_TIMEOUT, COMPLETION_TIMEOUT)
            .await
    }

    async fn observe_with_deadlines(
        self,
        registration_timeout: Duration,
        completion_timeout: Duration,
    ) -> io::Result<ExitStatus> {
        let queue = future::race(self.register(), async {
            observation_deadline(registration_timeout).await;
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Task registration expired",
            ))
        })
        .await?;
        future::race(queue.exit(), async {
            observation_deadline(completion_timeout).await;
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Task observation expired",
            ))
        })
        .await
    }

    pub async fn observe_native_profile_with_helper_check<F, Fut>(
        self,
        check: F,
    ) -> io::Result<ExitStatus>
    where
        F: FnOnce(std::path::PathBuf) -> Fut,
        Fut: std::future::Future<Output = io::Result<()>>,
    {
        let registration = async {
            let (observation, mut connection) = self.register_connection().await?;
            let executable = observation.executable_path()?;
            check(executable.clone()).await?;
            if observation.executable_path()? != executable {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "Task helper executable changed during admission",
                ));
            }
            connection.write_all(&[3]).await?;
            Ok(observation)
        };
        let observation = future::race(registration, async {
            observation_deadline(REGISTRATION_TIMEOUT).await;
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Task registration expired",
            ))
        })
        .await?;
        future::race(observation.exit(), async {
            observation_deadline(COMPLETION_TIMEOUT).await;
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Task observation expired",
            ))
        })
        .await
    }

    pub async fn observe_checked_native_profile_with_helper_check<F, Fut>(
        self,
        expected: TaskBuildInvocation,
        check: F,
    ) -> io::Result<ExitStatus>
    where
        F: FnOnce(std::path::PathBuf) -> Fut,
        Fut: std::future::Future<Output = io::Result<()>>,
    {
        validate_checked_native_invocation(&expected)?;
        let registration = async {
            let (observation, mut connection) = self.register_connection().await?;
            let executable = observation.executable_path()?;
            let mut header = [0; 4];
            connection.read_exact(&mut header).await?;
            let count = u32::from_be_bytes(header) as usize;
            if count == 0 || count > MAX_TASK_BUILD_INVOCATION_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid native invocation frame",
                ));
            }
            let mut bytes = vec![0; count];
            connection.read_exact(&mut bytes).await?;
            let invocation: TaskBuildInvocation = serde_json::from_slice(&bytes).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid native invocation receipt",
                )
            })?;
            validate_checked_native_invocation(&invocation)?;
            let actual = ObservedTaskBuildInvocation {
                invocation: std::sync::Arc::new(invocation),
            };
            if expected.program() != actual.program()
                || expected.arguments() != actual.arguments()
                || expected.working_directory() != actual.working_directory()
                || expected.environment() != actual.environment()
            {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "Native invocation changed during admission",
                ));
            }
            // Source/consent/probe admission must finish after the observed invocation
            // is received, immediately before this same helper is authorized to exec.
            check(executable.clone()).await?;
            if observation.executable_path()? != executable {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "Task helper executable changed during admission",
                ));
            }
            connection.write_all(&[5]).await?;
            Ok(observation)
        };
        let observation = future::race(registration, async {
            observation_deadline(REGISTRATION_TIMEOUT).await;
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Task registration expired",
            ))
        })
        .await?;
        future::race(observation.exit(), async {
            observation_deadline(COMPLETION_TIMEOUT).await;
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Task observation expired",
            ))
        })
        .await
    }

    pub async fn observe_stdout(self, limit: usize) -> io::Result<ObservedTaskStdout> {
        validate_stdout_report_limit(limit)?;
        let (observation, connection) = future::race(self.register_stdout(limit), async {
            observation_deadline(REGISTRATION_TIMEOUT).await;
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Task registration expired",
            ))
        })
        .await?;
        observe_stdout_completion(observation, connection, limit, COMPLETION_TIMEOUT).await
    }

    pub async fn observe_stdout_with_helper_check<F, Fut>(
        self,
        limit: usize,
        check: F,
    ) -> io::Result<ObservedTaskStdout>
    where
        F: FnOnce(std::path::PathBuf) -> Fut,
        Fut: std::future::Future<Output = io::Result<()>>,
    {
        validate_stdout_report_limit(limit)?;
        let registration = async {
            let (observation, mut connection) = self.register_connection().await?;
            let executable = observation.executable_path()?;
            check(executable.clone()).await?;
            if observation.executable_path()? != executable {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "Task helper executable changed during admission",
                ));
            }
            connection.write_all(&[2]).await?;
            connection.write_all(&(limit as u64).to_be_bytes()).await?;
            Ok((observation, connection))
        };
        let (observation, connection) = future::race(registration, async {
            observation_deadline(REGISTRATION_TIMEOUT).await;
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Task registration expired",
            ))
        })
        .await?;
        observe_stdout_completion(observation, connection, limit, COMPLETION_TIMEOUT).await
    }

    pub async fn observe_checked_build_stdout_with_helper_check<F, Fut>(
        self,
        limit: usize,
        expected: TaskBuildInvocation,
        check: F,
    ) -> io::Result<ObservedTaskStdout>
    where
        F: FnOnce(std::path::PathBuf) -> Fut,
        Fut: std::future::Future<Output = io::Result<()>>,
    {
        validate_stdout_report_limit(limit)?;
        expected.validate()?;
        let registration = async {
            let (observation, mut connection) = self.register_connection().await?;
            let executable = observation.executable_path()?;
            check(executable.clone()).await?;
            let mut header = [0; 4];
            connection.read_exact(&mut header).await?;
            let count = u32::from_be_bytes(header) as usize;
            if count == 0 || count > MAX_TASK_BUILD_INVOCATION_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid Build invocation frame",
                ));
            }
            let mut bytes = vec![0; count];
            connection.read_exact(&mut bytes).await?;
            let invocation: TaskBuildInvocation = serde_json::from_slice(&bytes).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid Build invocation receipt",
                )
            })?;
            invocation.validate()?;
            let actual = ObservedTaskBuildInvocation {
                invocation: std::sync::Arc::new(invocation),
            };
            if !expected.matches_observed(&actual) || observation.executable_path()? != executable {
                #[cfg(any(test, feature = "test-support"))]
                eprintln!(
                    "Build admission mismatch: program={} arguments={} cwd={} environment_keys={:?}",
                    expected.program == actual.invocation.program,
                    expected.arguments == actual.invocation.arguments,
                    expected.cwd == actual.invocation.cwd,
                    expected
                        .environment
                        .iter()
                        .filter_map(|(key, value)| {
                            (actual.invocation.environment.get(key) != Some(value)).then_some(key)
                        })
                        .collect::<Vec<_>>()
                );
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "Build invocation changed during admission",
                ));
            }
            connection.write_all(&[4]).await?;
            connection.write_all(&(limit as u64).to_be_bytes()).await?;
            Ok((observation, connection, actual))
        };
        let (observation, connection, actual) = future::race(registration, async {
            observation_deadline(REGISTRATION_TIMEOUT).await;
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Task registration expired",
            ))
        })
        .await?;
        let mut report =
            observe_stdout_completion(observation, connection, limit, COMPLETION_TIMEOUT).await?;
        report.invocation = Some(actual);
        Ok(report)
    }

    pub async fn observe_checked_cargo_runtime_stdout_with_helper_check<F, Fut>(
        self,
        limit: usize,
        expected: TaskBuildInvocation,
        check: F,
    ) -> io::Result<ObservedTaskStdout>
    where
        F: FnOnce(std::path::PathBuf) -> Fut,
        Fut: std::future::Future<Output = io::Result<()>>,
    {
        validate_stdout_report_limit(limit)?;
        validate_cargo_runtime_invocation(&expected)?;
        let registration = async {
            let (observation, mut connection) = self.register_connection().await?;
            let executable = observation.executable_path()?;
            let mut mode = [0; 4];
            connection.read_exact(&mut mode).await?;
            if mode != CARGO_RUNTIME_REGISTRATION {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Cargo runtime registration mode mismatch",
                ));
            }
            let mut header = [0; 4];
            connection.read_exact(&mut header).await?;
            let count = u32::from_be_bytes(header) as usize;
            if count == 0 || count > MAX_TASK_BUILD_INVOCATION_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid Build invocation frame",
                ));
            }
            let mut bytes = vec![0; count];
            connection.read_exact(&mut bytes).await?;
            let invocation: TaskBuildInvocation = serde_json::from_slice(&bytes).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid Build invocation receipt",
                )
            })?;
            validate_cargo_runtime_invocation(&invocation)?;
            let actual = ObservedTaskBuildInvocation {
                invocation: std::sync::Arc::new(invocation),
            };
            if !cargo_runtime_invocations_match(&expected, &actual) {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "Build invocation changed during admission",
                ));
            }
            check(executable.clone()).await?;
            if observation.executable_path()? != executable {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "Cargo helper image changed before acknowledgement",
                ));
            }
            connection.write_all(&[6]).await?;
            connection.write_all(&(limit as u64).to_be_bytes()).await?;
            Ok((observation, connection, actual))
        };
        let (observation, connection, actual) = future::race(registration, async {
            observation_deadline(REGISTRATION_TIMEOUT).await;
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Task registration expired",
            ))
        })
        .await?;
        let mut report = observe_cargo_runtime_completion(observation, connection, limit).await?;
        report.invocation = Some(actual);
        Ok(report)
    }

    async fn register(self) -> io::Result<ProcessObservation> {
        let (observation, mut connection) = self.register_connection().await?;
        connection.write_all(&[1]).await?;
        Ok(observation)
    }

    async fn register_stdout(
        self,
        limit: usize,
    ) -> io::Result<(ProcessObservation, smol::net::unix::UnixStream)> {
        validate_stdout_report_limit(limit)?;
        let (observation, mut connection) = self.register_connection().await?;
        connection.write_all(&[2]).await?;
        connection.write_all(&(limit as u64).to_be_bytes()).await?;
        Ok((observation, connection))
    }

    async fn register_connection(
        self,
    ) -> io::Result<(ProcessObservation, smol::net::unix::UnixStream)> {
        let (mut connection, _) = self.listener.accept().await?;
        let mut user = 0;
        let mut group = 0;
        let mut process_id: libc::pid_t = 0;
        let mut length = std::mem::size_of_val(&process_id) as libc::socklen_t;
        // Both credentials and PID come from the connected kernel socket, never its payload.
        unsafe {
            if libc::getpeereid(connection.as_raw_fd(), &mut user, &mut group) != 0 {
                return Err(io::Error::last_os_error());
            }
            if user != libc::geteuid() {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "Task peer user mismatch",
                ));
            }
            if libc::getsockopt(
                connection.as_raw_fd(),
                libc::SOL_LOCAL,
                libc::LOCAL_PEERPID,
                (&mut process_id as *mut libc::pid_t).cast(),
                &mut length,
            ) != 0
            {
                return Err(io::Error::last_os_error());
            }
        }
        if length as usize != std::mem::size_of_val(&process_id) || process_id <= 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid Task peer PID",
            ));
        }
        let mut supplied_token = [0; 32];
        connection.read_exact(&mut supplied_token).await?;
        let difference = supplied_token
            .iter()
            .zip(self.token)
            .fold(0, |difference, (left, right)| difference | (left ^ right));
        if difference != 0 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "Task registration token mismatch",
            ));
        }
        let observation = ProcessObservation::new(process_id)?;
        Ok((observation, connection))
    }
}

async fn observe_stdout_completion(
    observation: ProcessObservation,
    connection: smol::net::unix::UnixStream,
    limit: usize,
    timeout: Duration,
) -> io::Result<ObservedTaskStdout> {
    future::race(
        async {
            let (status, stdout) =
                future::try_zip(observation.exit(), read_stdout_report(connection, limit)).await?;
            Ok(ObservedTaskStdout {
                status,
                stdout,
                invocation: None,
                cargo_runtime: None,
            })
        },
        async {
            observation_deadline(timeout).await;
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Task stdout observation expired",
            ))
        },
    )
    .await
}

struct ProcessObservation {
    queue: Async<OwnedFd>,
    process_id: libc::pid_t,
}

impl ProcessObservation {
    fn new(process_id: libc::pid_t) -> io::Result<Self> {
        let descriptor = unsafe { libc::kqueue() };
        if descriptor < 0 {
            return Err(io::Error::last_os_error());
        }
        let queue = unsafe { OwnedFd::from_raw_fd(descriptor) };
        if unsafe { libc::fcntl(queue.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) } == -1 {
            return Err(io::Error::last_os_error());
        }
        let event = libc::kevent {
            ident: process_id as libc::uintptr_t,
            filter: libc::EVFILT_PROC,
            flags: libc::EV_ADD | libc::EV_ONESHOT,
            fflags: libc::NOTE_EXIT | libc::NOTE_EXITSTATUS,
            data: 0,
            udata: std::ptr::null_mut(),
        };
        if unsafe {
            libc::kevent(
                queue.as_raw_fd(),
                &event,
                1,
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
            )
        } == -1
        {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            // kqueue rejects FIONBIO; reads use kevent with an explicit zero timeout.
            queue: Async::new_nonblocking(queue)?,
            process_id,
        })
    }

    fn executable_path(&self) -> io::Result<std::path::PathBuf> {
        let mut bytes = [0_u8; libc::PROC_PIDPATHINFO_MAXSIZE as usize];
        // Query the PID obtained from the authenticated kernel socket, never a claimed payload.
        let count = unsafe {
            libc::proc_pidpath(
                self.process_id,
                bytes.as_mut_ptr().cast(),
                bytes.len() as u32,
            )
        };
        if count <= 0 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "Cannot identify the registered Task helper executable",
            ));
        }
        let length = bytes.iter().position(|byte| *byte == 0).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Task helper executable path exceeds its limit",
            )
        })?;
        let value = std::str::from_utf8(&bytes[..length]).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Task helper executable path is not supported",
            )
        })?;
        if length == 0 || !Path::new(value).is_absolute() || value.chars().any(char::is_control) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Task helper executable path is invalid",
            ));
        }
        Ok(std::path::PathBuf::from(value))
    }

    async fn exit(&self) -> io::Result<ExitStatus> {
        self.queue
            .read_with(|queue| {
                let mut event = std::mem::MaybeUninit::<libc::kevent>::uninit();
                let timeout = libc::timespec {
                    tv_sec: 0,
                    tv_nsec: 0,
                };
                let count = unsafe {
                    libc::kevent(
                        queue.as_raw_fd(),
                        std::ptr::null(),
                        0,
                        event.as_mut_ptr(),
                        1,
                        &timeout,
                    )
                };
                if count == -1 {
                    return Err(io::Error::last_os_error());
                }
                if count == 0 {
                    return Err(io::ErrorKind::WouldBlock.into());
                }
                let event = unsafe { event.assume_init() };
                if event.flags & libc::EV_ERROR != 0 {
                    return Err(io::Error::from_raw_os_error(event.data as i32));
                }
                if event.ident != self.process_id as libc::uintptr_t
                    || event.fflags & libc::NOTE_EXITSTATUS == 0
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Missing kernel Task exit status",
                    ));
                }
                Ok(ExitStatus::from_raw(event.data as i32))
            })
            .await
    }
}

pub fn register_and_exec(socket: &Path, command: &mut std::process::Command) -> io::Error {
    register_and_exec_mode(socket, command, TaskExecMode::Generic)
}

pub fn register_and_exec_native_profile_capture(
    socket: &Path,
    command: &mut std::process::Command,
) -> io::Error {
    register_and_exec_mode(socket, command, TaskExecMode::NativeProfileCapture)
}

fn observed_checked_native_command_invocation(
    command: &mut std::process::Command,
) -> io::Result<TaskBuildInvocation> {
    let invocation = observed_exact_command_invocation(command, TASK_NATIVE_INVOCATION_ENV)?;
    validate_checked_native_invocation(&invocation)?;
    Ok(invocation)
}

fn observed_exact_command_invocation(
    command: &mut std::process::Command,
    envelope_key: &str,
) -> io::Result<TaskBuildInvocation> {
    use std::os::unix::ffi::OsStrExt as _;
    let invalid = || {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Invalid bounded exact invocation envelope",
        )
    };
    let encoded = std::env::var_os(envelope_key).ok_or_else(invalid)?;
    let bytes = encoded.as_bytes();
    if bytes.is_empty() || bytes.len() > MAX_TASK_BUILD_INVOCATION_BYTES {
        return Err(invalid());
    }
    let admitted: TaskBuildInvocation = serde_json::from_slice(bytes).map_err(|_| invalid())?;
    validate_cargo_runtime_invocation(&admitted)?;
    let cwd = match command.get_current_dir() {
        Some(path) if path.is_absolute() => path.to_path_buf(),
        Some(path) => std::env::current_dir()?.join(path),
        None => std::env::current_dir()?,
    };
    if admitted.program() != Path::new(command.get_program()) || admitted.working_directory() != cwd
    {
        return Err(invalid());
    }
    let mut arguments = command.get_args();
    for expected in admitted.arguments() {
        if arguments.next().and_then(|argument| argument.to_str()) != Some(expected.as_str()) {
            return Err(invalid());
        }
    }
    if arguments.next().is_some() {
        return Err(invalid());
    }
    // Reconstruct the exact child map after helper startup; inherited login values are not authority.
    command
        .env_clear()
        .envs(admitted.environment())
        .current_dir(admitted.working_directory());
    Ok(admitted)
}

fn validate_checked_native_invocation(invocation: &TaskBuildInvocation) -> io::Result<()> {
    validate_cargo_runtime_invocation(invocation)?;
    if invocation.environment().contains_key("_NT_SYMBOL_PATH") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Native invocation contains an unsupported symbol service environment",
        ));
    }
    Ok(())
}

pub fn register_and_exec_checked_native_profile_capture(
    socket: &Path,
    command: &mut std::process::Command,
) -> io::Error {
    use std::os::unix::process::CommandExt as _;
    if std::env::var_os(TASK_NATIVE_INVOCATION_ENV).is_none() {
        return io::Error::new(
            io::ErrorKind::InvalidInput,
            "Missing checked native invocation",
        );
    }
    let invocation = match observed_checked_native_command_invocation(command) {
        Ok(invocation) => invocation,
        Err(error) => return error,
    };
    if let Err(error) = validate_checked_native_invocation(&invocation) {
        return error;
    }
    let registration = async {
        let token = std::env::var(TASK_COMPLETION_TOKEN_ENV).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Missing Task registration token",
            )
        })?;
        if token.len() != 64 || !token.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid Task registration token",
            ));
        }
        let mut decoded = [0; 32];
        for (value, pair) in decoded.iter_mut().zip(token.as_bytes().chunks_exact(2)) {
            *value = u8::from_str_radix(std::str::from_utf8(pair).map_err(io::Error::other)?, 16)
                .map_err(io::Error::other)?;
        }
        let mut connection = smol::net::unix::UnixStream::connect(socket).await?;
        connection.write_all(&decoded).await?;
        let bytes = serde_json::to_vec(&invocation)
            .map_err(|_| io::Error::other("Cannot encode bounded native invocation"))?;
        if bytes.is_empty() || bytes.len() > MAX_TASK_BUILD_INVOCATION_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Native invocation exceeds its frame bound",
            ));
        }
        connection
            .write_all(&(bytes.len() as u32).to_be_bytes())
            .await?;
        connection.write_all(&bytes).await?;
        let mut acknowledgement = [0];
        connection.read_exact(&mut acknowledgement).await?;
        if acknowledgement != [5] {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Checked native registration refused",
            ));
        }
        io::Result::Ok(())
    };
    if let Err(error) = smol::block_on(future::race(registration, async {
        observation_deadline(REGISTRATION_TIMEOUT).await;
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "Task registration expired",
        ))
    })) {
        return error;
    }
    command.exec()
}

#[derive(Clone, Copy)]
enum TaskExecMode {
    Generic,
    NativeProfileCapture,
}

fn register_and_exec_mode(
    socket: &Path,
    command: &mut std::process::Command,
    mode: TaskExecMode,
) -> io::Error {
    use std::os::unix::process::CommandExt;
    let expected_acknowledgement = match mode {
        TaskExecMode::Generic => [1],
        TaskExecMode::NativeProfileCapture => [3],
    };
    let register = async {
        let token = std::env::var(TASK_COMPLETION_TOKEN_ENV).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Missing Task registration token",
            )
        })?;
        if token.len() != 64 || !token.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid Task registration token",
            ));
        }
        let mut decoded = [0; 32];
        for (value, pair) in decoded.iter_mut().zip(token.as_bytes().chunks_exact(2)) {
            let digits = std::str::from_utf8(pair).map_err(io::Error::other)?;
            *value = u8::from_str_radix(digits, 16).map_err(io::Error::other)?;
        }
        let mut connection = smol::net::unix::UnixStream::connect(socket).await?;
        connection.write_all(&decoded).await?;
        let mut acknowledgement = [0];
        connection.read_exact(&mut acknowledgement).await?;
        if acknowledgement != expected_acknowledgement {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Task registration refused",
            ));
        }
        io::Result::Ok(())
    };
    let result = smol::block_on(future::race(register, async {
        observation_deadline(REGISTRATION_TIMEOUT).await;
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "Task registration expired",
        ))
    }));
    if let Err(error) = result {
        return error;
    }
    if matches!(mode, TaskExecMode::NativeProfileCapture) {
        // Samply's presymbolicator consults this inherited symbol-server setting after login.
        command.env_remove("_NT_SYMBOL_PATH");
    }
    command
        .env_remove(TASK_BUILD_INVOCATION_ENV)
        .env_remove(TASK_NATIVE_INVOCATION_ENV)
        .env_remove(TASK_COMPLETION_TOKEN_ENV)
        .exec()
}

async fn register_stdout_connection(
    socket: &Path,
    invocation: Option<&TaskBuildInvocation>,
) -> io::Result<(smol::net::unix::UnixStream, usize)> {
    register_stdout_connection_mode(socket, invocation, false).await
}

async fn register_stdout_connection_mode(
    socket: &Path,
    invocation: Option<&TaskBuildInvocation>,
    cargo_runtime: bool,
) -> io::Result<(smol::net::unix::UnixStream, usize)> {
    let token = std::env::var(TASK_COMPLETION_TOKEN_ENV).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Missing Task registration token",
        )
    })?;
    if token.len() != 64 || !token.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Invalid Task registration token",
        ));
    }
    let mut decoded = [0; 32];
    for (value, pair) in decoded.iter_mut().zip(token.as_bytes().chunks_exact(2)) {
        let digits = std::str::from_utf8(pair).map_err(io::Error::other)?;
        *value = u8::from_str_radix(digits, 16).map_err(io::Error::other)?;
    }
    let mut connection = smol::net::unix::UnixStream::connect(socket).await?;
    connection.write_all(&decoded).await?;
    if cargo_runtime {
        connection.write_all(&CARGO_RUNTIME_REGISTRATION).await?;
    }
    if let Some(invocation) = invocation {
        invocation.validate()?;
        let bytes = serde_json::to_vec(invocation)
            .map_err(|_| io::Error::other("Cannot encode bounded Build invocation"))?;
        if bytes.is_empty() || bytes.len() > MAX_TASK_BUILD_INVOCATION_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Build invocation exceeds its frame bound",
            ));
        }
        connection
            .write_all(&(bytes.len() as u32).to_be_bytes())
            .await?;
        connection.write_all(&bytes).await?;
    }
    let mut acknowledgement = [0];
    connection.read_exact(&mut acknowledgement).await?;
    if acknowledgement
        != if cargo_runtime {
            [6]
        } else if invocation.is_some() {
            [4]
        } else {
            [2]
        }
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Task stdout registration refused",
        ));
    }
    let mut limit = [0; 8];
    connection.read_exact(&mut limit).await?;
    let limit = usize::try_from(u64::from_be_bytes(limit)).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid Task stdout report limit",
        )
    })?;
    validate_stdout_report_limit(limit)?;
    Ok((connection, limit))
}

pub fn register_and_capture_stdout(
    socket: &Path,
    command: std::process::Command,
) -> io::Result<ExitStatus> {
    register_and_capture_stdout_mode(socket, command, false)
}

pub fn register_and_capture_build_stdout(
    socket: &Path,
    command: std::process::Command,
) -> io::Result<ExitStatus> {
    register_and_capture_stdout_mode(socket, command, true)
}

fn observed_command_invocation(
    command: &mut std::process::Command,
) -> io::Result<TaskBuildInvocation> {
    let invalid = || {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Cannot capture bounded Build invocation",
        )
    };
    command.env_remove(TASK_COMPLETION_TOKEN_ENV);
    command.env_remove(TASK_BUILD_INVOCATION_ENV);
    command.env_remove(TASK_NATIVE_INVOCATION_ENV);
    let cwd = match command.get_current_dir() {
        Some(path) if path.is_absolute() => path.to_path_buf(),
        Some(path) => std::env::current_dir()?.join(path),
        None => std::env::current_dir()?,
    };
    let mut arguments = Vec::new();
    let mut argument_bytes = 0_usize;
    for value in command.get_args() {
        let value = value.to_str().ok_or_else(invalid)?;
        argument_bytes = argument_bytes
            .checked_add(value.len())
            .ok_or_else(invalid)?;
        if arguments.len() >= 512 || value.len() > 4096 || argument_bytes > 32 * 1024 {
            return Err(invalid());
        }
        arguments.push(value.to_owned());
    }
    if let Some(encoded) = std::env::var_os(TASK_BUILD_INVOCATION_ENV) {
        use std::os::unix::ffi::OsStrExt as _;
        let encoded = encoded.as_bytes();
        if encoded.is_empty() || encoded.len() > MAX_TASK_BUILD_INVOCATION_BYTES {
            return Err(invalid());
        }
        let admitted: TaskBuildInvocation =
            serde_json::from_slice(encoded).map_err(|_| invalid())?;
        admitted.validate().map_err(|_| invalid())?;
        if admitted.program() != Path::new(command.get_program())
            || admitted.arguments() != arguments
            || admitted.working_directory() != cwd
        {
            return Err(invalid());
        }
        // Restore the admitted Task values after interactive shell startup, before the exact child check.
        command.envs(admitted.environment());
    }

    let mut environment = std::collections::BTreeMap::new();
    let mut inherited_bytes = 0_usize;
    for (key, value) in std::env::vars_os() {
        let key = key.to_str().ok_or_else(invalid)?;
        if key == TASK_BUILD_INVOCATION_ENV || key == TASK_NATIVE_INVOCATION_ENV {
            continue;
        }
        let value = value.to_str().ok_or_else(invalid)?;
        inherited_bytes = inherited_bytes
            .checked_add(key.len() + value.len())
            .ok_or_else(invalid)?;
        if environment.len() >= 512
            || key.len() > 4096
            || value.len() > 4096
            || inherited_bytes > 64 * 1024
        {
            return Err(invalid());
        }
        environment.insert(key.to_owned(), value.to_owned());
    }
    let mut environment_bytes = environment
        .iter()
        .map(|(key, value)| key.len() + value.len())
        .sum::<usize>();
    for (key, value) in command.get_envs() {
        let key = key.to_str().ok_or_else(invalid)?;
        if key.len() > 4096 {
            return Err(invalid());
        }
        if let Some(value) = value {
            let value = value.to_str().ok_or_else(invalid)?;
            let previous_bytes = environment
                .get(key)
                .map_or(0, |previous| key.len() + previous.len());
            let next_bytes = environment_bytes
                .checked_sub(previous_bytes)
                .and_then(|bytes| bytes.checked_add(key.len() + value.len()))
                .ok_or_else(invalid)?;
            if value.len() > 4096
                || next_bytes > 64 * 1024
                || (!environment.contains_key(key) && environment.len() >= 512)
            {
                return Err(invalid());
            }
            environment.insert(key.to_owned(), value.to_owned());
            environment_bytes = next_bytes;
        } else if let Some(previous) = environment.remove(key) {
            environment_bytes = environment_bytes
                .checked_sub(key.len() + previous.len())
                .ok_or_else(invalid)?;
        }
    }
    let invocation =
        TaskBuildInvocation::new(command.get_program().into(), arguments, cwd, environment)?;
    // Freeze the observed final inherited values so the receipt describes the actual owned child.
    command
        .env_clear()
        .envs(invocation.environment())
        .current_dir(invocation.working_directory());
    Ok(invocation)
}

fn register_and_capture_stdout_mode(
    socket: &Path,
    mut command: std::process::Command,
    checked_build: bool,
) -> io::Result<ExitStatus> {
    let invocation = if checked_build {
        Some(observed_command_invocation(&mut command)?)
    } else {
        None
    };
    let capture = async {
        let (connection, limit) = future::race(
            register_stdout_connection(socket, invocation.as_ref()),
            async {
                observation_deadline(REGISTRATION_TIMEOUT).await;
                Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "Task registration expired",
                ))
            },
        )
        .await?;
        command
            .env_remove(TASK_BUILD_INVOCATION_ENV)
            .env_remove(TASK_NATIVE_INVOCATION_ENV)
            .env_remove(TASK_COMPLETION_TOKEN_ENV);
        capture_owned_task_stdout(connection, command, limit).await
    };
    smol::block_on(future::race(capture, async {
        observation_deadline(COMPLETION_TIMEOUT).await;
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "Task stdout capture expired",
        ))
    }))
}

fn cargo_runtime_invocations_match(
    expected: &TaskBuildInvocation,
    actual: &ObservedTaskBuildInvocation,
) -> bool {
    expected.program == actual.invocation.program
        && expected.arguments == actual.invocation.arguments
        && expected.cwd == actual.invocation.cwd
        && expected.environment == actual.invocation.environment
}

fn validate_cargo_runtime_invocation(invocation: &TaskBuildInvocation) -> io::Result<()> {
    invocation.validate()?;
    for (key, value) in invocation.environment() {
        if key.starts_with("ZED_PROFILE_") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Unissued private Cargo runtime environment field",
            ));
        }
        if (key.starts_with("DYLD_") && key != "DYLD_FALLBACK_LIBRARY_PATH"
            || key.starts_with("LD_"))
            && !value.is_empty()
        {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Unsupported Cargo runtime loader environment",
            ));
        }
        if key == "DYLD_FALLBACK_LIBRARY_PATH" && !value.is_empty() {
            let mut directories = 0_usize;
            for directory in value.split(':') {
                directories = directories.checked_add(1).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "Cargo fallback count overflow")
                })?;
                let path = Path::new(directory);
                if directories > 128
                    || directory.len() > 4096
                    || directory.chars().any(char::is_control)
                    || !path.is_absolute()
                    || (directory != "/" && directory.ends_with('/'))
                    || directory.contains("//")
                    || path
                        .components()
                        .any(|component| matches!(component, std::path::Component::ParentDir))
                    || directory.split('/').any(|component| component == ".")
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "Unsupported bounded Cargo fallback path",
                    ));
                }
            }
        }
    }
    Ok(())
}

async fn read_cargo_runtime_report(
    mut connection: smol::net::unix::UnixStream,
    limit: usize,
) -> io::Result<(Vec<u8>, Option<ObservedCargoRuntime>)> {
    validate_stdout_report_limit(limit)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(limit)
        .map_err(|_| io::Error::other("Cannot allocate bounded Cargo stdout report"))?;
    let mut runtime = None;
    loop {
        let mut header = [0; 4];
        connection.read_exact(&mut header).await?;
        let count = u32::from_be_bytes(header);
        if count == 0 {
            let mut trailing = [0];
            if connection.read(&mut trailing).await? != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Trailing Cargo stdout frame",
                ));
            }
            return Ok((output, runtime));
        }
        if count == CARGO_RUNTIME_FRAME {
            if runtime.is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Duplicate Cargo runtime observation",
                ));
            }
            let invocation = read_cargo_runtime_invocation(&mut connection).await?;
            runtime = Some(ObservedCargoRuntime { invocation });
            connection.write_all(&[7]).await?;
            continue;
        }
        let count = count as usize;
        if count > TASK_STDOUT_FRAME_BYTES || count > limit.saturating_sub(output.len()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Cargo stdout report exceeds its limit",
            ));
        }
        let start = output.len();
        output.resize(start + count, 0);
        connection.read_exact(&mut output[start..]).await?;
    }
}

async fn read_cargo_runtime_invocation(
    connection: &mut smol::net::unix::UnixStream,
) -> io::Result<TaskBuildInvocation> {
    let mut header = [0; 4];
    connection.read_exact(&mut header).await?;
    let count = u32::from_be_bytes(header) as usize;
    if count == 0 || count > MAX_TASK_BUILD_INVOCATION_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid Cargo runtime frame",
        ));
    }
    let mut bytes = vec![0; count];
    connection.read_exact(&mut bytes).await?;
    let invocation: TaskBuildInvocation = serde_json::from_slice(&bytes).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid bounded Cargo runtime record",
        )
    })?;
    validate_cargo_runtime_invocation(&invocation)?;
    if invocation
        .environment()
        .keys()
        .any(|key| key.starts_with("ZED_PROFILE_"))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Unissued private Cargo runtime environment field",
        ));
    }
    Ok(invocation)
}

async fn observe_cargo_runtime_completion(
    observation: ProcessObservation,
    connection: smol::net::unix::UnixStream,
    limit: usize,
) -> io::Result<ObservedTaskStdout> {
    future::race(
        async {
            let (status, (stdout, cargo_runtime)) = future::try_zip(
                observation.exit(),
                read_cargo_runtime_report(connection, limit),
            )
            .await?;
            if cargo_runtime.is_none() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Cargo preparation has no runtime observation",
                ));
            }
            Ok(ObservedTaskStdout {
                status,
                stdout,
                invocation: None,
                cargo_runtime,
            })
        },
        async {
            observation_deadline(COMPLETION_TIMEOUT).await;
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Cargo runtime observation expired",
            ))
        },
    )
    .await
}

fn bounded_cargo_runtime_invocation(
    program: &std::ffi::OsStr,
    arguments: &[std::ffi::OsString],
) -> io::Result<TaskBuildInvocation> {
    let invalid = || {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Invalid bounded Cargo runtime context",
        )
    };
    let cwd = std::env::current_dir()?;
    let program_text = program.to_str().ok_or_else(invalid)?;
    if program_text.is_empty()
        || program_text.len() > 4096
        || program_text.chars().any(char::is_control)
    {
        return Err(invalid());
    }
    let program = if Path::new(program).is_absolute() {
        program.into()
    } else {
        cwd.join(program)
    };
    let mut values = Vec::new();
    let mut bytes = 0_usize;
    for argument in arguments {
        let argument = argument.to_str().ok_or_else(invalid)?;
        bytes = bytes.checked_add(argument.len()).ok_or_else(invalid)?;
        if values.len() >= 512
            || argument.len() > 4096
            || argument.contains('\0')
            || bytes > 32 * 1024
        {
            return Err(invalid());
        }
        values.push(argument.to_owned());
    }
    let mut environment = std::collections::BTreeMap::new();
    let mut environment_bytes = 0_usize;
    for (key, value) in std::env::vars_os() {
        let key = key.to_str().ok_or_else(invalid)?;
        if key == TASK_CARGO_RUNTIME_ENDPOINT_ENV || key == TASK_CARGO_RUNTIME_TOKEN_ENV {
            continue;
        }
        if key.starts_with("ZED_PROFILE_") {
            return Err(invalid());
        }
        let value = value.to_str().ok_or_else(invalid)?;
        environment_bytes = environment_bytes
            .checked_add(key.len() + value.len())
            .ok_or_else(invalid)?;
        if environment.len() >= 512
            || key.len() > 4096
            || value.len() > 4096
            || environment_bytes > 64 * 1024
        {
            return Err(invalid());
        }
        if environment
            .insert(key.to_owned(), value.to_owned())
            .is_some()
        {
            return Err(invalid());
        }
    }
    let invocation = TaskBuildInvocation::new(program, values, cwd, environment)?;
    validate_cargo_runtime_invocation(&invocation)?;
    Ok(invocation)
}

pub fn observe_cargo_runtime(
    socket: &Path,
    program: &std::ffi::OsStr,
    arguments: &[std::ffi::OsString],
) -> io::Result<()> {
    let endpoint = std::env::var_os(TASK_CARGO_RUNTIME_ENDPOINT_ENV).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Missing Cargo runtime endpoint",
        )
    })?;
    if Path::new(&endpoint) != socket
        || !socket.is_absolute()
        || !socket
            .to_str()
            .is_some_and(|value| value.len() < 104 && !value.chars().any(char::is_control))
    {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "Cargo runtime endpoint changed",
        ));
    }
    let token = std::env::var(TASK_CARGO_RUNTIME_TOKEN_ENV)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Missing Cargo runtime nonce"))?;
    if token.len() != 64 || !token.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Invalid Cargo runtime nonce",
        ));
    }
    let mut decoded = [0_u8; 32];
    for (value, pair) in decoded.iter_mut().zip(token.as_bytes().chunks_exact(2)) {
        let pair = std::str::from_utf8(pair).map_err(io::Error::other)?;
        *value = u8::from_str_radix(pair, 16).map_err(io::Error::other)?;
    }
    let invocation = bounded_cargo_runtime_invocation(program, arguments)?;
    let bytes = invocation.encode_helper_environment()?.into_bytes();
    smol::block_on(future::race(
        async {
            let mut connection = smol::net::unix::UnixStream::connect(socket).await?;
            connection.write_all(&decoded).await?;
            connection
                .write_all(&(bytes.len() as u32).to_be_bytes())
                .await?;
            connection.write_all(&bytes).await?;
            connection.shutdown(std::net::Shutdown::Write)?;
            let mut acknowledgement = [0];
            connection.read_exact(&mut acknowledgement).await?;
            if acknowledgement != [7] {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Cargo runtime observation refused",
                ));
            }
            Ok(())
        },
        async {
            observation_deadline(REGISTRATION_TIMEOUT).await;
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Cargo runtime acknowledgement expired",
            ))
        },
    ))
}

pub fn register_and_capture_cargo_runtime_stdout(
    socket: &Path,
    mut command: std::process::Command,
) -> io::Result<ExitStatus> {
    let invocation = observed_exact_command_invocation(&mut command, TASK_BUILD_INVOCATION_ENV)?;
    let capture = async {
        let (connection, limit) = future::race(
            register_stdout_connection_mode(socket, Some(&invocation), true),
            async {
                observation_deadline(REGISTRATION_TIMEOUT).await;
                Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "Cargo runtime registration expired",
                ))
            },
        )
        .await?;
        let mut token = [0; 32];
        if unsafe { libc::getentropy(token.as_mut_ptr().cast(), token.len()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let runtime = TaskCompletionListener::bind(token)?;
        let token = token
            .iter()
            .map(|value| format!("{value:02x}"))
            .collect::<String>();
        let helper_image = std::env::current_exe()?;
        command
            .env(TASK_CARGO_RUNTIME_ENDPOINT_ENV, runtime.path())
            .env(TASK_CARGO_RUNTIME_TOKEN_ENV, token);
        let mut command = smol::process::Command::from(command);
        command
            .stdin(std::process::Stdio::inherit())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .kill_on_drop(true);
        let mut child = command.spawn()?;
        let cargo_process_id = libc::pid_t::try_from(child.id())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid owned Cargo PID"))?;
        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("Cargo stdout pipe unavailable"))?;
        let result = {
            let mut control = connection.clone();
            let writer = futures::lock::Mutex::new(connection);
            let (runtime_acknowledged, acknowledgement) = futures::channel::oneshot::channel();
            let runtime_pending = std::sync::atomic::AtomicBool::new(false);
            let (child_exit, exited) = futures::channel::oneshot::channel::<()>();
            let child_status = async {
                let status = child.status().await;
                drop(child_exit);
                status
            };
            let pump = async {
                let mut terminal_output =
                    smol::Unblock::with_capacity(TASK_STDOUT_FRAME_BYTES, std::io::stdout());
                let mut total = 0_usize;
                let mut buffer = [0_u8; TASK_STDOUT_FRAME_BYTES];
                loop {
                    let remaining = limit
                        .saturating_sub(total)
                        .saturating_add(1)
                        .min(buffer.len());
                    let count = stdout.read(&mut buffer[..remaining]).await?;
                    if count == 0 {
                        return io::Result::Ok(());
                    }
                    total = total
                        .checked_add(count)
                        .ok_or_else(|| io::Error::other("Cargo stdout size overflow"))?;
                    if total > limit {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Cargo stdout report exceeds its limit",
                        ));
                    }
                    {
                        let mut connection = writer.lock().await;
                        connection.write_all(&(count as u32).to_be_bytes()).await?;
                        connection.write_all(&buffer[..count]).await?;
                    }
                    terminal_output.write_all(&buffer[..count]).await?;
                    terminal_output.flush().await?;
                }
            };
            let runtime_record = async {
                let admitted = async {
                    let (observation, mut peer) = runtime.register_connection().await?;
                    if observation.process_id != cargo_process_id
                        || observation.executable_path()? != helper_image
                    {
                        return Err(io::Error::new(
                            io::ErrorKind::PermissionDenied,
                            "Cargo runtime peer is not the owned Cargo helper image",
                        ));
                    }
                    let invocation = read_cargo_runtime_invocation(&mut peer).await?;
                    let mut trailing = [0];
                    if peer.read(&mut trailing).await? != 0
                        || observation.executable_path()? != helper_image
                    {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Trailing or replaced Cargo runtime peer",
                        ));
                    }
                    let bytes = invocation.encode_helper_environment()?.into_bytes();
                    {
                        let mut connection = writer.lock().await;
                        runtime_pending.store(true, std::sync::atomic::Ordering::Release);
                        connection
                            .write_all(&CARGO_RUNTIME_FRAME.to_be_bytes())
                            .await?;
                        connection
                            .write_all(&(bytes.len() as u32).to_be_bytes())
                            .await?;
                        connection.write_all(&bytes).await?;
                    }
                    acknowledgement.await.map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::Interrupted,
                            "Cargo runtime host acknowledgement dropped",
                        )
                    })?;
                    if observation.executable_path()? != helper_image {
                        return Err(io::Error::new(
                            io::ErrorKind::PermissionDenied,
                            "Cargo runtime peer image changed before acknowledgement",
                        ));
                    }
                    peer.write_all(&[7]).await?;
                    Ok(true)
                };
                future::race(admitted, async {
                    // Sender Drop comes only from the actual owned Child wait or cancellation.
                    match exited.await {
                        Err(_) => Ok(false),
                        Ok(()) => Err(io::Error::other("Unexpected Cargo exit notification")),
                    }
                })
                .await
            };
            let work = async {
                let (status, ((), recorded)) =
                    future::try_zip(child_status, future::try_zip(pump, runtime_record)).await?;
                if status.success() && !recorded {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Cargo succeeded without a runtime observer",
                    ));
                }
                let mut connection = writer.lock().await;
                connection.write_all(&0_u32.to_be_bytes()).await?;
                connection.flush().await?;
                Ok(status)
            };
            future::race(
                future::race(work, async {
                    let mut byte = [0];
                    control.read_exact(&mut byte).await?;
                    if byte != [7] || !runtime_pending.load(std::sync::atomic::Ordering::Acquire) {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Unexpected Cargo runtime control frame",
                        ));
                    }
                    runtime_acknowledged.send(()).map_err(|_| {
                        io::Error::new(io::ErrorKind::Interrupted, "Cargo runtime receiver dropped")
                    })?;
                    match control.read(&mut byte).await? {
                        0 => Err(io::Error::new(
                            io::ErrorKind::Interrupted,
                            "Cargo runtime observation disconnected",
                        )),
                        _ => Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Duplicate Cargo runtime control frame",
                        )),
                    }
                }),
                async {
                    observation_deadline(COMPLETION_TIMEOUT).await;
                    Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "Cargo runtime capture expired",
                    ))
                },
            )
            .await
        };
        drop(stdout);
        if let Err(error) = result {
            if child.try_status()?.is_none() {
                if let Err(kill_error) = child.kill() {
                    if child.try_status()?.is_none() {
                        return Err(kill_error);
                    }
                }
                future::race(async { child.status().await.map(|_| ()) }, async {
                    observation_deadline(REGISTRATION_TIMEOUT).await;
                    Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "Cancelled Cargo child retirement expired",
                    ))
                })
                .await?;
            }
            return Err(error);
        }
        result
    };
    smol::block_on(capture)
}

pub fn exit_with_observed_task_status(status: ExitStatus) -> ! {
    if let Some(signal) = status.signal() {
        unsafe {
            if signal != libc::SIGKILL && libc::signal(signal, libc::SIG_DFL) == libc::SIG_ERR {
                std::process::exit(1);
            }
            let mut signals = std::mem::MaybeUninit::<libc::sigset_t>::uninit();
            if libc::sigemptyset(signals.as_mut_ptr()) != 0 {
                std::process::exit(1);
            }
            let mut signals = signals.assume_init();
            if signal != libc::SIGKILL && libc::sigaddset(&mut signals, signal) != 0 {
                std::process::exit(1);
            }
            let mut previous = std::mem::MaybeUninit::<libc::sigset_t>::uninit();
            if libc::pthread_sigmask(libc::SIG_UNBLOCK, &signals, previous.as_mut_ptr()) != 0 {
                std::process::exit(1);
            }
            let mut wait_mask = previous.assume_init();
            if libc::sigdelset(&mut wait_mask, signal) != 0 {
                std::process::exit(1);
            }
            if libc::kill(libc::getpid(), signal) != 0 {
                std::process::exit(1);
            }
            // A process-directed signal can arrive after kill returns; do not race it with exit(1).
            loop {
                if libc::sigsuspend(&wait_mask) != -1
                    || io::Error::last_os_error().kind() != io::ErrorKind::Interrupted
                {
                    std::process::exit(1);
                }
            }
        }
    }
    std::process::exit(status.code().unwrap_or(1))
}

#[cfg(test)]
mod tests {
    #[test]
    fn runtime_preparation_test_helper() {
        let Some(socket) = std::env::var_os("ZED_TEST_RUNTIME_SOCKET") else {
            return;
        };
        if let Some(directory) = std::env::var_os("ZED_TEST_RUNTIME_FRAME_HOLD") {
            let directory = std::path::PathBuf::from(directory);
            let result = smol::block_on(async {
                let mut connection =
                    smol::net::unix::UnixStream::connect(Path::new(&socket)).await?;
                connection.write_all(&[0x42; 32]).await?;
                connection.write_all(&CARGO_RUNTIME_REGISTRATION).await?;
                smol::fs::write(directory.join("mode-sent"), b"held").await?;
                let release = async {
                    while !directory.join("release-frame").exists() {
                        observation_deadline(Duration::from_millis(5)).await;
                    }
                    Ok::<(), io::Error>(())
                };
                future::race(release, async {
                    observation_deadline(REGISTRATION_TIMEOUT).await;
                    Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "Held fixture frame expired",
                    ))
                })
                .await?;
                let invocation = std::env::var(TASK_BUILD_INVOCATION_ENV)
                    .expect("genuine prepared invocation envelope");
                connection
                    .write_all(&(invocation.len() as u32).to_be_bytes())
                    .await?;
                connection.write_all(invocation.as_bytes()).await?;
                let mut acknowledgement = [0];
                connection.read_exact(&mut acknowledgement).await?;
                Ok::<_, io::Error>(acknowledgement)
            });
            assert!(
                result.is_err(),
                "revoked frame must receive no acknowledgement"
            );
            std::process::exit(1);
        }
        let mut command =
            std::process::Command::new(std::env::current_exe().expect("actual helper"));
        command.args([
            "--exact",
            "macos_task_completion::tests::runtime_preparation_test_cargo",
            "--nocapture",
        ]);
        match register_and_capture_cargo_runtime_stdout(Path::new(&socket), command) {
            Ok(status) => exit_with_observed_task_status(status),
            Err(error) => {
                eprintln!("runtime fixture refused ({:?})", error.kind());
                std::process::exit(1);
            }
        }
    }

    #[test]
    fn runtime_preparation_test_cargo() {
        use std::io::Write as _;
        use std::os::unix::process::CommandExt as _;
        let Some(marker) = std::env::var_os("ZED_TEST_RUNTIME_CARGO_STARTED") else {
            return;
        };
        std::fs::write(marker, std::process::id().to_string()).expect("owned child started");
        std::io::stdout()
            .write_all(b"{\"reason\":\"build-finished\",\"success\":true}\n")
            .expect("actual raw stdout");
        std::io::stdout().flush().expect("flush raw stdout");
        let mut runner =
            std::process::Command::new(std::env::current_exe().expect("actual runner image"));
        runner
            .args([
                "--exact",
                "macos_task_completion::tests::runtime_preparation_test_runner",
                "--nocapture",
            ])
            .env("DYLD_FALLBACK_LIBRARY_PATH", "/controlled/runtime/library")
            .env("CARGO_MANIFEST_DIR", "/controlled/package");
        match std::env::var("ZED_TEST_RUNTIME_MODE").as_deref() {
            Ok("foreign") => {
                let status =
                    smol::block_on(async { smol::process::Command::from(runner).status().await })
                        .expect("foreign process control");
                exit_with_observed_task_status(status);
            }
            Ok("nonce") => {
                runner.env(TASK_CARGO_RUNTIME_TOKEN_ENV, "00".repeat(32));
            }
            Ok("unknown-private") => {
                runner.env("ZED_PROFILE_UNISSUED_PRIVATE", "denied");
            }
            Ok("unsafe-loader") => {
                runner.env("DYLD_INSERT_LIBRARIES", "/controlled/not-a-library");
            }
            Ok("aliased-fallback") => {
                runner.env("DYLD_FALLBACK_LIBRARY_PATH", "/controlled/../alias");
            }
            _ => {}
        }
        let error = runner.exec();
        panic!("actual exec replacement failed ({:?})", error.kind());
    }

    #[test]
    fn runtime_preparation_test_runner() {
        if std::env::var_os("ZED_TEST_RUNTIME_CARGO_STARTED").is_none() {
            return;
        }
        let invocation = parse_local_task_helper_invocation(
            [
                "--internal-cargo-runtime-observer".into(),
                "--".into(),
                "never-executed-workload".into(),
                "literal $ZED_FILE ; ' ".into(),
                "".into(),
                "unicode-λ".into(),
            ]
            .into_iter(),
        )
        .expect("actual runner parser")
        .expect("runner mode");
        assert_eq!(invocation.mode, LocalTaskHelperMode::CargoRuntimeObserver);
        match observe_cargo_runtime(
            &invocation.socket,
            &invocation.program,
            &invocation.arguments,
        ) {
            Ok(()) => std::process::exit(0),
            Err(error) => {
                eprintln!("runtime runner refused ({:?})", error.kind());
                std::process::exit(1);
            }
        }
    }

    fn runtime_preparation_fixture(
        listener: &TaskCompletionListener,
        directory: &Path,
        mode: &str,
    ) -> (TaskBuildInvocation, smol::process::Child) {
        let workload = directory.join("never-executed-workload");
        std::fs::write(
            &workload,
            b"#!/bin/sh\nprintf invoked > \"$ZED_TEST_RUNTIME_WORKLOAD_MARKER\"\n",
        )
        .expect("real executable sentinel");
        std::fs::set_permissions(&workload, std::fs::Permissions::from_mode(0o755))
            .expect("owned workload mode");
        let environment = std::collections::BTreeMap::from([
            (
                "ZED_TEST_RUNTIME_WORKLOAD_MARKER".into(),
                directory
                    .join("workload-executed")
                    .to_str()
                    .expect("marker UTF8")
                    .into(),
            ),
            (
                "ZED_TEST_RUNTIME_CARGO_STARTED".into(),
                directory
                    .join("started")
                    .to_str()
                    .expect("fixture UTF8")
                    .into(),
            ),
            ("ZED_TEST_RUNTIME_MODE".into(), mode.into()),
            ("PATH".into(), "/usr/bin:/bin".into()),
            ("HOME".into(), "/admitted/home".into()),
        ]);
        let expected = TaskBuildInvocation::new(
            std::env::current_exe().expect("actual Cargo surrogate"),
            vec![
                "--exact".into(),
                "macos_task_completion::tests::runtime_preparation_test_cargo".into(),
                "--nocapture".into(),
            ],
            directory.to_path_buf(),
            environment,
        )
        .expect("bounded genuine expected invocation");
        let mut helper =
            smol::process::Command::new(std::env::current_exe().expect("actual helper image"));
        helper
            .args([
                "--exact",
                "macos_task_completion::tests::runtime_preparation_test_helper",
                "--nocapture",
            ])
            .env_clear()
            .env("ZED_TEST_RUNTIME_SOCKET", listener.path())
            .env(TASK_COMPLETION_TOKEN_ENV, "42".repeat(32))
            .env(
                TASK_BUILD_INVOCATION_ENV,
                expected
                    .encode_helper_environment()
                    .expect("private envelope"),
            )
            .current_dir(directory)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true);
        if mode == "held-frame" {
            helper.env("ZED_TEST_RUNTIME_FRAME_HOLD", directory);
        }
        (expected, helper.spawn().expect("actual outer helper"))
    }

    async fn runtime_preparation_positive(directory: &Path) {
        let listener = TaskCompletionListener::bind([0x42; 32]).expect("private original listener");
        let (expected, mut helper) = runtime_preparation_fixture(&listener, directory, "exact");
        let image = std::env::current_exe().expect("exact helper image");
        let report = listener
            .observe_checked_cargo_runtime_stdout_with_helper_check(
                4096,
                expected.clone(),
                |actual| async move {
                    assert_eq!(actual, image);
                    Ok(())
                },
            )
            .await
            .expect("genuine checked runtime observation");
        let (status, bytes, actual, runtime) = report.into_parts_with_cargo_runtime();
        assert!(status.success());
        assert_eq!(helper.status().await.expect("actual helper exit"), status);
        assert!(expected.matches_observed(&actual.expect("actual complete Cargo invocation")));
        assert!(
            bytes
                .windows(b"\"reason\":\"build-finished\"".len())
                .any(|value| value == b"\"reason\":\"build-finished\"")
        );
        let runtime = runtime.expect("one authenticated runtime record");
        assert_eq!(runtime.program(), directory.join("never-executed-workload"));
        assert_eq!(runtime.working_directory(), directory);
        assert_eq!(
            runtime.arguments(),
            ["literal $ZED_FILE ; ' ", "", "unicode-λ"]
        );
        assert_eq!(
            runtime
                .environment()
                .get("DYLD_FALLBACK_LIBRARY_PATH")
                .map(String::as_str),
            Some("/controlled/runtime/library")
        );
        assert_eq!(
            runtime
                .environment()
                .get("CARGO_MANIFEST_DIR")
                .map(String::as_str),
            Some("/controlled/package")
        );
        for key in [
            TASK_COMPLETION_TOKEN_ENV,
            TASK_BUILD_INVOCATION_ENV,
            TASK_NATIVE_INVOCATION_ENV,
            TASK_CARGO_RUNTIME_ENDPOINT_ENV,
            TASK_CARGO_RUNTIME_TOKEN_ENV,
        ] {
            assert!(!runtime.environment().contains_key(key));
        }
        assert!(directory.join("started").exists());
        assert!(directory.join("never-executed-workload").is_file());
        assert!(!directory.join("workload-executed").exists());
    }

    #[test]
    fn checked_cargo_runtime_observes_same_owned_exec_and_preserves_data() {
        smol::block_on(async {
            let directory = tempfile::tempdir_in("/private/tmp").expect("owned runtime fixture");
            runtime_preparation_positive(directory.path()).await;
        });
    }

    #[test]
    fn checked_cargo_runtime_refuses_foreign_nonce_and_unknown_private_fields() {
        smol::block_on(async {
            for mode in ["foreign", "nonce", "unknown-private"] {
                let directory =
                    tempfile::tempdir_in("/private/tmp").expect("owned negative runtime fixture");
                let listener = TaskCompletionListener::bind([0x42; 32]).expect("private listener");
                let (expected, mut helper) =
                    runtime_preparation_fixture(&listener, directory.path(), mode);
                let result = listener
                    .observe_checked_cargo_runtime_stdout_with_helper_check(
                        4096,
                        expected,
                        |_| async { Ok(()) },
                    )
                    .await;
                assert!(result.is_err(), "{mode} must not create a runtime receipt");
                assert!(
                    !helper
                        .status()
                        .await
                        .expect("actual refused helper exit")
                        .success()
                );
                assert!(directory.path().join("started").exists());
                assert!(directory.path().join("never-executed-workload").is_file());
                assert!(!directory.path().join("workload-executed").exists());
            }
        });
    }

    #[test]
    fn checked_cargo_runtime_wrong_ack_mode_prevents_cargo_start() {
        smol::block_on(async {
            let directory = tempfile::tempdir_in("/private/tmp").expect("owned mismatch fixture");
            let listener = TaskCompletionListener::bind([0x42; 32]).expect("private listener");
            let (expected, mut helper) =
                runtime_preparation_fixture(&listener, directory.path(), "exact");
            assert!(
                listener
                    .observe_checked_build_stdout_with_helper_check(4096, expected, |_| async {
                        Ok(())
                    })
                    .await
                    .is_err()
            );
            assert!(
                !helper
                    .status()
                    .await
                    .expect("actual mode-refused helper exit")
                    .success()
            );
            assert!(!directory.path().join("started").exists());
            assert!(directory.path().join("never-executed-workload").is_file());
            assert!(!directory.path().join("workload-executed").exists());

            let held = tempfile::tempdir_in("/private/tmp").expect("owned held frame fixture");
            let listener = TaskCompletionListener::bind([0x42; 32]).expect("private held listener");
            let (expected, mut helper) =
                runtime_preparation_fixture(&listener, held.path(), "held-frame");
            let current = std::sync::atomic::AtomicBool::new(true);
            let checked = std::sync::atomic::AtomicBool::new(false);
            let observation = listener.observe_checked_cargo_runtime_stdout_with_helper_check(
                4096,
                expected,
                |_| async {
                    checked.store(true, std::sync::atomic::Ordering::SeqCst);
                    if current.load(std::sync::atomic::Ordering::SeqCst) {
                        Ok(())
                    } else {
                        Err(io::Error::new(
                            io::ErrorKind::PermissionDenied,
                            "Current source revoked",
                        ))
                    }
                },
            );
            let revoke = async {
                let ready = async {
                    while !held.path().join("mode-sent").exists() {
                        observation_deadline(Duration::from_millis(5)).await;
                    }
                };
                future::race(ready, async {
                    observation_deadline(REGISTRATION_TIMEOUT).await;
                    panic!("actual helper did not hold the invocation frame");
                })
                .await;
                assert!(
                    !checked.load(std::sync::atomic::Ordering::SeqCst),
                    "check must follow the held frame"
                );
                current.store(false, std::sync::atomic::Ordering::SeqCst);
                smol::fs::write(held.path().join("release-frame"), b"released")
                    .await
                    .expect("release actual frame");
            };
            let (result, ()) = future::zip(observation, revoke).await;
            assert_eq!(
                result.err().expect("revoked current owner refused").kind(),
                io::ErrorKind::PermissionDenied
            );
            assert!(checked.load(std::sync::atomic::Ordering::SeqCst));
            assert!(
                !helper
                    .status()
                    .await
                    .expect("actual held helper exit")
                    .success()
            );
            assert!(!held.path().join("started").exists());
            assert!(!held.path().join("workload-executed").exists());
            runtime_preparation_positive(held.path()).await;
        });
    }

    #[test]
    fn checked_cargo_runtime_drop_registered_peer_blocks_workload_and_recovers() {
        smol::block_on(async {
            let directory =
                tempfile::tempdir_in("/private/tmp").expect("owned held runtime fixture");
            let listener = TaskCompletionListener::bind([0x42; 32]).expect("private listener");
            let (expected, mut helper) =
                runtime_preparation_fixture(&listener, directory.path(), "exact");
            let (observation, mut connection) = listener
                .register_connection()
                .await
                .expect("real kernel original registration");
            assert_eq!(
                observation.executable_path().expect("actual image"),
                std::env::current_exe().expect("expected image")
            );
            let mut mode = [0; 4];
            connection.read_exact(&mut mode).await.expect("real mode");
            assert_eq!(mode, CARGO_RUNTIME_REGISTRATION);
            let invocation = read_cargo_runtime_invocation(&mut connection)
                .await
                .expect("actual checked invocation");
            assert_eq!(
                invocation
                    .encode_helper_environment()
                    .expect("actual envelope"),
                expected
                    .encode_helper_environment()
                    .expect("expected envelope")
            );
            connection
                .write_all(&[6])
                .await
                .expect("actual checked ACK6");
            connection
                .write_all(&4096_u64.to_be_bytes())
                .await
                .expect("real bound");
            loop {
                let mut header = [0; 4];
                connection
                    .read_exact(&mut header)
                    .await
                    .expect("actual outer frame");
                let count = u32::from_be_bytes(header);
                if count == CARGO_RUNTIME_FRAME {
                    let runtime = read_cargo_runtime_invocation(&mut connection)
                        .await
                        .expect("actual authenticated owned-peer runtime");
                    assert_eq!(
                        runtime.program(),
                        directory.path().join("never-executed-workload")
                    );
                    break;
                }
                assert!(count > 0 && count <= TASK_STDOUT_FRAME_BYTES as u32);
                let mut bytes = vec![0; count as usize];
                connection
                    .read_exact(&mut bytes)
                    .await
                    .expect("actual raw stdout");
            }
            assert!(directory.path().join("started").exists());
            assert!(directory.path().join("never-executed-workload").is_file());
            assert!(!directory.path().join("workload-executed").exists());
            drop(connection);
            drop(observation);
            assert!(
                !helper
                    .status()
                    .await
                    .expect("actual disconnected helper exit")
                    .success()
            );
            assert!(directory.path().join("never-executed-workload").is_file());
            assert!(!directory.path().join("workload-executed").exists());
            let owned_cargo_pid = std::fs::read_to_string(directory.path().join("started"))
                .expect("actual Cargo PID marker")
                .parse::<libc::pid_t>()
                .expect("owned PID");
            assert_eq!(
                unsafe { libc::kill(owned_cargo_pid, 0) },
                -1,
                "owned Cargo child must be gone before helper retirement"
            );
            assert_eq!(io::Error::last_os_error().raw_os_error(), Some(libc::ESRCH));
            std::fs::remove_file(directory.path().join("started"))
                .expect("retire owned started marker");
            runtime_preparation_positive(directory.path()).await;
        });
    }

    #[test]
    fn checked_cargo_runtime_bounds_and_reserved_transport_are_finite() {
        let program = std::path::PathBuf::from("/bounded/workload");
        let cwd = std::path::PathBuf::from("/bounded/cwd");
        for key in [
            TASK_CARGO_RUNTIME_ENDPOINT_ENV,
            TASK_CARGO_RUNTIME_TOKEN_ENV,
        ] {
            let data = TaskBuildInvocation::new(
                program.clone(),
                Vec::new(),
                cwd.clone(),
                std::collections::BTreeMap::from([(key.into(), "private".into())]),
            );
            assert!(
                data.is_err(),
                "private transport values cannot enter a child recipe"
            );
        }
        assert!(
            TaskBuildInvocation::new(
                program.clone(),
                vec!["x".repeat(4097)],
                cwd.clone(),
                Default::default()
            )
            .is_err()
        );
        assert!(
            TaskBuildInvocation::new(
                program.clone(),
                vec![String::new(); 513],
                cwd.clone(),
                Default::default()
            )
            .is_err()
        );
        assert!(
            TaskBuildInvocation::new(
                program,
                Vec::new(),
                cwd,
                (0..513)
                    .map(|value| (format!("key{value}"), String::new()))
                    .collect()
            )
            .is_err()
        );
        let duplicate = br#"{"program":"/bounded/workload","arguments":[],"cwd":"/bounded/cwd","environment":{"same":"first","same":"second"}}"#;
        assert!(serde_json::from_slice::<TaskBuildInvocation>(duplicate).is_err());
        smol::block_on(async {
            for (count, payload) in [
                (0_u32, Vec::new()),
                ((MAX_TASK_BUILD_INVOCATION_BYTES + 1) as u32, Vec::new()),
                (8_u32, vec![b'x'; 7]),
            ] {
                let (mut sender, mut receiver) =
                    smol::net::unix::UnixStream::pair().expect("owned bounded frame channel");
                sender
                    .write_all(&count.to_be_bytes())
                    .await
                    .expect("real frame header");
                sender
                    .write_all(&payload)
                    .await
                    .expect("real malformed frame bytes");
                sender
                    .shutdown(std::net::Shutdown::Write)
                    .expect("bounded EOF");
                assert!(read_cargo_runtime_invocation(&mut receiver).await.is_err());
            }
            let unissued = TaskBuildInvocation::new(
                "/bounded/workload".into(),
                Vec::new(),
                "/bounded/cwd".into(),
                std::collections::BTreeMap::from([(
                    "ZED_PROFILE_UNISSUED_PRIVATE".into(),
                    "private".into(),
                )]),
            )
            .expect("ordinary data constructor does not confer runtime permission");
            let payload = unissued
                .encode_helper_environment()
                .expect("bounded unissued data");
            let (mut sender, mut receiver) =
                smol::net::unix::UnixStream::pair().expect("owned unknown-private frame");
            sender
                .write_all(&(payload.len() as u32).to_be_bytes())
                .await
                .expect("actual frame size");
            sender
                .write_all(payload.as_bytes())
                .await
                .expect("actual unissued frame");
            sender
                .shutdown(std::net::Shutdown::Write)
                .expect("bounded EOF");
            assert!(read_cargo_runtime_invocation(&mut receiver).await.is_err());
            let (mut sender, receiver) =
                smol::net::unix::UnixStream::pair().expect("owned stdout bound channel");
            sender
                .write_all(&2_u32.to_be_bytes())
                .await
                .expect("over-bound stdout header");
            sender
                .shutdown(std::net::Shutdown::Write)
                .expect("bounded EOF");
            assert!(read_cargo_runtime_report(receiver, 1).await.is_err());
        });
    }

    #[test]
    fn checked_cargo_runtime_unsafe_records_never_seal_and_allow_fresh_control() {
        smol::block_on(async {
            for mode in ["unsafe-loader", "aliased-fallback"] {
                let directory =
                    tempfile::tempdir_in("/private/tmp").expect("owned runtime refusal");
                let listener = TaskCompletionListener::bind([0x42; 32]).expect("owned observer");
                let (expected, mut helper) =
                    runtime_preparation_fixture(&listener, directory.path(), mode);
                let result = listener
                    .observe_checked_cargo_runtime_stdout_with_helper_check(
                        4096,
                        expected,
                        |_| async { Ok(()) },
                    )
                    .await;
                let status = helper.status().await.expect("genuine refused child exit");
                assert!(
                    !status.success(),
                    "unsafe runtime record cannot produce success"
                );
                if let Ok(report) = result {
                    let (observed, _, _, runtime) = report.into_parts_with_cargo_runtime();
                    assert_eq!(
                        observed, status,
                        "raw unsuccessful kernel outcome preserved"
                    );
                    assert!(
                        runtime.is_none(),
                        "unsafe runtime record cannot become observation"
                    );
                }
                assert!(directory.path().join("started").exists());
                assert!(!directory.path().join("workload-executed").exists());
                runtime_preparation_positive(directory.path()).await;
            }
        });
    }

    #[test]
    fn checked_cargo_runtime_exact_map_preserves_legacy_ack4_matching() {
        let expected = TaskBuildInvocation::new(
            "/bounded/cargo".into(),
            vec!["run".into()],
            "/bounded/cwd".into(),
            std::collections::BTreeMap::from([("PATH".into(), "/usr/bin:/bin".into())]),
        )
        .expect("finite original recipe");
        let mut changed = expected.clone();
        changed
            .environment
            .insert("UNISSUED_EXTRA".into(), "changed".into());
        let actual = ObservedTaskBuildInvocation {
            invocation: std::sync::Arc::new(changed),
        };
        assert!(
            expected.matches_observed(&actual),
            "legacy ACK4 subset remains compatible"
        );
        assert!(
            !cargo_runtime_invocations_match(&expected, &actual),
            "ACK6 requires complete equality"
        );
    }

    #[test]
    fn checked_cargo_runtime_loader_policy_is_finite_data_not_path_authority() {
        let context = |key: &str, value: String| {
            TaskBuildInvocation::new(
                "/bounded/cargo".into(),
                Vec::new(),
                "/bounded/cwd".into(),
                std::collections::BTreeMap::from([(key.to_owned(), value)]),
            )
            .expect("ordinary bounded data")
        };
        for key in [
            "DYLD_INSERT_LIBRARIES",
            "DYLD_LIBRARY_PATH",
            "DYLD_FRAMEWORK_PATH",
            "DYLD_ROOT_PATH",
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
        ] {
            assert!(validate_cargo_runtime_invocation(&context(key, "unsafe".into())).is_err());
            assert!(validate_cargo_runtime_invocation(&context(key, String::new())).is_ok());
        }
        for value in [
            ":/finite",
            "/finite:",
            "relative",
            "/finite/../escape",
            "/finite/./alias",
            "/finite//alias",
            "/finite/",
            "/finite\ncontrol",
        ] {
            assert!(
                validate_cargo_runtime_invocation(&context(
                    "DYLD_FALLBACK_LIBRARY_PATH",
                    value.into()
                ))
                .is_err()
            );
        }
        let inclusive = (0..128)
            .map(|index| format!("/finite/{index}"))
            .collect::<Vec<_>>()
            .join(":");
        assert!(
            validate_cargo_runtime_invocation(&context(
                "DYLD_FALLBACK_LIBRARY_PATH",
                inclusive.clone()
            ))
            .is_ok()
        );
        assert!(
            validate_cargo_runtime_invocation(&context(
                "DYLD_FALLBACK_LIBRARY_PATH",
                format!("{inclusive}:/finite/129")
            ))
            .is_err()
        );
        // Existence, original image closure and owned roots are checked by the issuer before exec.
        assert!(
            validate_cargo_runtime_invocation(&context(
                "DYLD_FALLBACK_LIBRARY_PATH",
                "/nonexistent/data-only".into()
            ))
            .is_ok()
        );
    }

    use super::*;

    #[test]
    fn task_helper_image_boundary_child() {
        let Some(case) = std::env::var_os("ZED_TEST_HELPER_IMAGE_BOUNDARY") else {
            return;
        };
        let marker =
            std::env::var_os("ZED_TEST_HELPER_IMAGE_MARKER").expect("owned prevented-child marker");
        let socket = std::env::var_os(TASK_CARGO_RUNTIME_ENDPOINT_ENV)
            .expect("owned non-authoritative endpoint");
        let case = case.to_str().expect("finite image boundary case");
        let mut arguments: Vec<std::ffi::OsString> = if case == "app-runtime-observer" {
            vec!["--internal-cargo-runtime-observer".into(), "--".into()]
        } else {
            let mut arguments = vec!["--internal-task-observer".into(), "--socket".into(), socket];
            match case {
                "app-checked-runtime" => arguments.push("--checked-cargo-runtime-stdout".into()),
                "dedicated-exec" => {}
                "dedicated-stdout" => arguments.push("--capture-stdout".into()),
                "dedicated-build" => arguments.push("--capture-build-stdout".into()),
                "dedicated-capture" => arguments.push("--native-profile-capture".into()),
                "dedicated-checked-capture" => {
                    arguments.push("--checked-native-profile-capture".into())
                }
                _ => panic!("Unexpected finite helper image case"),
            }
            arguments.push("--".into());
            arguments
        };
        arguments.extend([
            "/bin/sh".into(),
            "-c".into(),
            "printf started > \"$1\"".into(),
            "prevented-child".into(),
            marker,
        ]);
        let invocation = parse_local_task_helper_invocation(arguments.into_iter())
            .expect("bounded image route grammar")
            .expect("helper route");
        if case.starts_with("app-") {
            run_app_task_helper_invocation(invocation)
        } else {
            run_native_cargo_runtime_helper_invocation(invocation)
        }
    }

    #[test]
    fn task_helper_image_boundary_refuses_wrong_roles_before_child() {
        let directory = tempfile::TempDir::new().expect("owned image boundary controls");
        let endpoint = directory.path().join("never-issued-endpoint");
        let nonce = "73".repeat(32);
        for case in [
            "app-checked-runtime",
            "app-runtime-observer",
            "dedicated-exec",
            "dedicated-stdout",
            "dedicated-build",
            "dedicated-capture",
            "dedicated-checked-capture",
        ] {
            let marker = directory.path().join(case);
            let mut child =
                smol::process::Command::new(std::env::current_exe().expect("current test image"));
            child
                .args([
                    "--exact",
                    "macos_task_completion::tests::task_helper_image_boundary_child",
                    "--nocapture",
                ])
                .env_clear()
                .env("ZED_TEST_HELPER_IMAGE_BOUNDARY", case)
                .env("ZED_TEST_HELPER_IMAGE_MARKER", &marker)
                .env(TASK_CARGO_RUNTIME_ENDPOINT_ENV, &endpoint)
                .env(TASK_CARGO_RUNTIME_TOKEN_ENV, &nonce)
                .kill_on_drop(true);
            let output = smol::block_on(future::race(async { child.output().await }, async {
                observation_deadline(Duration::from_secs(30)).await;
                Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "Image boundary child expired",
                ))
            }))
            .expect("bounded actual helper image refusal");
            assert_eq!(
                output.status.code(),
                Some(2),
                "wrong image role must refuse before transport"
            );
            assert!(!marker.exists(), "wrong image must execute no workload");
            assert!(!endpoint.exists(), "wrong image must create no transport");
            let diagnostic = String::from_utf8(output.stderr).expect("bounded generic diagnostic");
            assert!(
                !diagnostic.contains(&nonce),
                "private nonce must not be echoed"
            );
            assert!(
                !diagnostic.contains(marker.to_str().expect("owned marker path")),
                "private path must not be echoed"
            );
            assert!(diagnostic.contains(if case.starts_with("app-") {
                "require the dedicated helper image"
            } else {
                "Unsupported native Cargo runtime helper mode"
            }));
        }
    }

    #[test]
    fn checked_native_environment_helper() {
        let Some(socket) = std::env::var_os("ZED_TEST_CHECKED_NATIVE_SOCKET") else {
            return;
        };
        let mut command = std::process::Command::new(std::env::current_exe().expect("child image"));
        command.args([
            "--exact",
            "macos_task_completion::tests::checked_native_environment_child",
            "--nocapture",
        ]);
        if std::env::var_os("ZED_TEST_CHECKED_NATIVE_BAD_ARGV").is_some() {
            command.arg("unexpected");
        }
        let error =
            register_and_exec_checked_native_profile_capture(Path::new(&socket), &mut command);
        panic!("checked native helper refused ({:?})", error.kind());
    }

    #[test]
    fn checked_native_environment_child() {
        if std::env::var_os("ZED_TEST_CHECKED_NATIVE_CHILD").is_none() {
            return;
        }
        let marker = std::env::var("ZED_TEST_CHECKED_NATIVE_MARKER").expect("marker");
        let expected = checked_native_child_environment(Path::new(&marker));
        let actual = std::env::vars().collect::<std::collections::BTreeMap<_, _>>();
        assert!(
            actual == expected,
            "actual exec child has exactly the issued map"
        );
        std::fs::write(marker, b"actual-checked-native-child").expect("actual child marker");
        std::process::exit(23);
    }

    fn checked_native_child_environment(
        marker: &Path,
    ) -> std::collections::BTreeMap<String, String> {
        std::collections::BTreeMap::from([
            ("ZED_TEST_CHECKED_NATIVE_CHILD".into(), "1".into()),
            (
                "ZED_TEST_CHECKED_NATIVE_MARKER".into(),
                marker.to_str().expect("fixture UTF8").into(),
            ),
            ("PATH".into(), "/usr/bin:/bin".into()),
            ("HOME".into(), "/admitted/home".into()),
            ("TERM".into(), "xterm-256color".into()),
            (
                "ADMITTED_LITERAL".into(),
                "space ; $RUSTFLAGS ' \" \\ literal".into(),
            ),
        ])
    }

    fn checked_native_test_invocation(marker: &Path) -> TaskBuildInvocation {
        TaskBuildInvocation::new(
            std::env::current_exe().expect("actual image"),
            vec![
                "--exact".into(),
                "macos_task_completion::tests::checked_native_environment_child".into(),
                "--nocapture".into(),
            ],
            std::env::current_dir().expect("actual cwd"),
            checked_native_child_environment(marker),
        )
        .expect("bounded exact invocation")
    }

    fn checked_native_test_helper(
        socket: &Path,
        invocation: &TaskBuildInvocation,
    ) -> smol::process::Command {
        let mut command =
            smol::process::Command::new(std::env::current_exe().expect("actual helper"));
        command
            .args([
                "--exact",
                "macos_task_completion::tests::checked_native_environment_helper",
                "--nocapture",
            ])
            .env("ZED_TEST_CHECKED_NATIVE_SOCKET", socket)
            .env(TASK_COMPLETION_TOKEN_ENV, "42".repeat(32))
            .env(
                TASK_NATIVE_INVOCATION_ENV,
                invocation
                    .encode_helper_environment()
                    .expect("bounded envelope"),
            )
            .env(
                TASK_BUILD_INVOCATION_ENV,
                "invalid legacy envelope ignored in distinct checked native mode",
            )
            .env("RUSTFLAGS", "--cfg introduced_after_login")
            .env("RUSTC", "/unadmitted/compiler")
            .env("RUSTDOC", "/unadmitted/rustdoc")
            .env("CARGO_PROFILE_DEV_DEBUG", "false")
            .env("UNADMITTED_LOADER_MARKER", "/unadmitted/dynamic/path")
            .env("_NT_SYMBOL_PATH", "srv*https://unadmitted.invalid")
            .env("ZED_TEST_UNADMITTED_SECRET", "must-not-reach-child")
            .env_remove("ZED_TEST_CHECKED_NATIVE_CHILD")
            .env_remove("ZED_TEST_CHECKED_NATIVE_BAD_ARGV")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true);
        command
    }

    #[test]
    fn task_checked_native_exact_environment_binds_actual_kernel_exit() {
        smol::block_on(async {
            let directory = tempfile::tempdir().expect("fixture");
            let marker = directory.path().join("actual-child");
            let expected = checked_native_test_invocation(&marker);
            let listener = TaskCompletionListener::bind([0x42; 32]).expect("listener");
            let mut helper = checked_native_test_helper(&listener.path(), &expected)
                .spawn()
                .expect("actual helper");
            let helper_image = std::env::current_exe().expect("helper image");
            let status = listener
                .observe_checked_native_profile_with_helper_check(expected, |image| async move {
                    assert_eq!(
                        image, helper_image,
                        "actual kernel-authenticated helper image"
                    );
                    Ok(())
                })
                .await
                .expect("actual ACK5 kernel completion");
            assert_eq!(status.code(), Some(23));
            assert_eq!(helper.status().await.expect("actual child status"), status);
            assert_eq!(
                std::fs::read(marker).expect("actual child marker"),
                b"actual-checked-native-child"
            );
        });
    }

    #[test]
    fn task_checked_native_rejects_changed_environment_argv_and_admission() {
        smol::block_on(async {
            for scenario in ["environment", "argv", "admission"] {
                let directory = tempfile::tempdir().expect("fixture");
                let marker = directory.path().join("must-not-execute");
                let expected = checked_native_test_invocation(&marker);
                let mut supplied = expected.clone();
                if scenario == "environment" {
                    supplied
                        .environment
                        .insert("RUSTFLAGS".into(), "unadmitted".into());
                }
                let listener = TaskCompletionListener::bind([0x42; 32]).expect("listener");
                let mut command = checked_native_test_helper(&listener.path(), &supplied);
                if scenario == "argv" {
                    command.env("ZED_TEST_CHECKED_NATIVE_BAD_ARGV", "1");
                }
                let mut helper = command.spawn().expect("actual helper");
                let observation = listener.observe_checked_native_profile_with_helper_check(
                    expected,
                    move |_| async move {
                        if scenario == "admission" {
                            return Err(io::Error::new(
                                io::ErrorKind::PermissionDenied,
                                "sealed owner expired before ACK5",
                            ));
                        }
                        Ok(())
                    },
                );
                if scenario == "argv" {
                    assert!(
                        !helper
                            .status()
                            .await
                            .expect("rejected helper exit")
                            .success()
                    );
                    drop(observation);
                } else {
                    assert_eq!(
                        observation
                            .await
                            .expect_err("changed invocation rejected")
                            .kind(),
                        io::ErrorKind::PermissionDenied
                    );
                    assert!(
                        !helper
                            .status()
                            .await
                            .expect("rejected helper exit")
                            .success()
                    );
                }
                assert!(!marker.exists(), "no actual child on rejected {scenario}");
            }
        });
    }

    #[test]
    fn task_checked_native_rejects_wrong_ack_and_oversized_frame() {
        smol::block_on(async {
            let directory = tempfile::tempdir().expect("fixture");
            let marker = directory.path().join("must-not-execute");
            let expected = checked_native_test_invocation(&marker);
            let listener = TaskCompletionListener::bind([0x42; 32]).expect("listener");
            let mut helper = checked_native_test_helper(&listener.path(), &expected)
                .spawn()
                .expect("actual helper");
            let (observation, mut connection) = listener
                .register_connection()
                .await
                .expect("kernel-authenticated peer");
            assert_eq!(
                observation.executable_path().expect("actual image"),
                std::env::current_exe().expect("test image")
            );
            connection
                .write_all(&[3])
                .await
                .expect("legacy ACK3 mismatch control");
            drop(connection);
            assert!(
                !helper
                    .status()
                    .await
                    .expect("wrong mode helper exit")
                    .success()
            );
            assert!(!marker.exists());
            let listener = TaskCompletionListener::bind([0x42; 32]).expect("listener");
            let socket = listener.path();
            let sending = async move {
                let mut connection = smol::net::unix::UnixStream::connect(socket)
                    .await
                    .expect("actual peer socket");
                connection
                    .write_all(&[0x42; 32])
                    .await
                    .expect("private fixture token");
                connection
                    .write_all(&((MAX_TASK_BUILD_INVOCATION_BYTES as u32) + 1).to_be_bytes())
                    .await
                    .expect("bounded invalid frame header");
                let mut acknowledgement = [0];
                assert!(
                    connection.read_exact(&mut acknowledgement).await.is_err(),
                    "no ACK on oversized frame"
                );
            };
            let observing = listener
                .observe_checked_native_profile_with_helper_check(expected, |_| async { Ok(()) });
            let (_, result) = future::zip(sending, observing).await;
            assert_eq!(
                result
                    .expect_err("oversized frame rejected before allocation")
                    .kind(),
                io::ErrorKind::InvalidData
            );
            assert!(!marker.exists());
        });
    }

    #[test]
    fn task_checked_native_flag_and_private_environment_are_separate() {
        let invocation = parse_local_task_helper_invocation(
            [
                "--internal-task-observer",
                "--socket",
                "/tmp/zed-task-test/completion",
                "--checked-native-profile-capture",
                "--",
                "/bin/sh",
                "literal ; $RUSTFLAGS",
            ]
            .into_iter()
            .map(std::ffi::OsString::from),
        )
        .expect("checked flag")
        .expect("helper invocation");
        assert_eq!(
            invocation.mode,
            LocalTaskHelperMode::CheckedNativeProfileCapture
        );
        let mut expected = checked_native_test_invocation(Path::new("/tmp/marker"));
        expected.environment.insert(
            "_NT_SYMBOL_PATH".into(),
            "srv*https://unadmitted.invalid".into(),
        );
        assert!(validate_checked_native_invocation(&expected).is_err());
        expected.environment.remove("_NT_SYMBOL_PATH");
        expected
            .environment
            .insert(TASK_NATIVE_INVOCATION_ENV.into(), "private-envelope".into());
        assert!(validate_checked_native_invocation(&expected).is_err());
        for key in [
            TASK_COMPLETION_TOKEN_ENV,
            TASK_BUILD_INVOCATION_ENV,
            TASK_NATIVE_INVOCATION_ENV,
            TASK_CARGO_RUNTIME_ENDPOINT_ENV,
            TASK_CARGO_RUNTIME_TOKEN_ENV,
        ] {
            let mut invocation = checked_native_test_invocation(Path::new("/tmp/marker"));
            invocation
                .environment
                .insert(key.into(), "unissued operational transport".into());
            assert!(
                invocation.validate().is_err(),
                "operational fields are not semantic environment"
            );
        }
    }

    #[test]
    fn task_checked_native_private_envelope_is_ignored_by_legacy_modes() {
        smol::block_on(async {
            for mode in ["exec", "native", "stdout", "build"] {
                let listener = TaskCompletionListener::bind([0x42; 32]).expect("bind");
                let command = "test -z \"${ZED_PROFILE_NATIVE_INVOCATION_ENV+x}\" && test -z \"${ZED_PROFILE_TASK_TOKEN+x}\" && printf selected";
                let mut helper =
                    smol::process::Command::new(std::env::current_exe().expect("test executable"));
                helper
                    .args([
                        "--exact",
                        "macos_task_completion::tests::task_completion_helper",
                        "--nocapture",
                    ])
                    .env("ZED_TEST_TASK_SOCKET", listener.path())
                    .env("ZED_TEST_TASK_COMMAND", command)
                    .env(
                        TASK_NATIVE_INVOCATION_ENV,
                        "not JSON; never restore outside checked native",
                    )
                    .env(TASK_COMPLETION_TOKEN_ENV, "42".repeat(32))
                    .env_remove("ZED_TEST_TASK_CAPTURE_BUILD_STDOUT")
                    .env_remove("ZED_TEST_TASK_CAPTURE_STDOUT")
                    .env_remove("ZED_TEST_NATIVE_PROFILE_CAPTURE")
                    .env_remove("ZED_TEST_NATIVE_LITERAL")
                    .env_remove("ZED_TEST_NATIVE_ENV_CHILD")
                    .env_remove("ZED_TEST_BUILD_ENVELOPE_CHILD")
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .kill_on_drop(true);
                match mode {
                    "native" => {
                        helper.env("ZED_TEST_NATIVE_PROFILE_CAPTURE", "1");
                    }
                    "stdout" => {
                        helper.env("ZED_TEST_TASK_CAPTURE_STDOUT", "1");
                    }
                    "build" => {
                        helper.env("ZED_TEST_TASK_CAPTURE_BUILD_STDOUT", "1");
                    }
                    _ => {}
                }
                let mut child = helper.spawn().expect("actual ordinary helper");
                let status = match mode {
                    "native" => listener
                        .observe_native_profile_with_helper_check(|_| async { Ok(()) })
                        .await
                        .expect("native helper kernel status"),
                    "stdout" => {
                        let report = listener
                            .observe_stdout(64)
                            .await
                            .expect("unchecked actual stdout");
                        assert_eq!(report.stdout(), b"selected");
                        report.status()
                    }
                    "build" => {
                        let expected = TaskBuildInvocation::new(
                            Path::new("/bin/sh").to_path_buf(),
                            vec!["-c".into(), command.into()],
                            std::env::current_dir().expect("actual cwd"),
                            std::collections::BTreeMap::new(),
                        )
                        .expect("bounded legacy Build recipe");
                        let report = listener
                            .observe_checked_build_stdout_with_helper_check(
                                64,
                                expected,
                                |_| async { Ok(()) },
                            )
                            .await
                            .expect(
                                "unchanged ACK4 admission after ignoring reserved native envelope",
                            );
                        assert_eq!(report.stdout(), b"selected");
                        report.status()
                    }
                    _ => listener
                        .observe()
                        .await
                        .expect("generic helper kernel status"),
                };
                assert!(
                    status.success(),
                    "private envelope is removed without parsing or applying it"
                );
                assert_eq!(child.status().await.expect("helper status"), status);
            }
        });
    }

    #[test]
    fn task_checked_native_held_registration_revocation_drop_and_fresh_control() {
        smol::block_on(async {
            for scenario in ["revoked", "dropped"] {
                let directory = tempfile::tempdir().expect("owned held native controls");
                let marker = directory.path().join("must-not-execute");
                let expected = checked_native_test_invocation(&marker);
                let listener = TaskCompletionListener::bind([0x42; 32]).expect("held listener");
                let mut helper = checked_native_test_helper(&listener.path(), &expected)
                    .spawn()
                    .expect("actual held helper");
                let (entered, after_frame) = futures::channel::oneshot::channel();
                let (release, owner_answer) = futures::channel::oneshot::channel::<()>();
                let observation = listener.observe_checked_native_profile_with_helper_check(
                    expected,
                    move |_| async move {
                        entered
                            .send(())
                            .map_err(|_| io::Error::other("Held owner check disappeared"))?;
                        if owner_answer.await.is_err() {
                            return Err(io::Error::new(
                                io::ErrorKind::PermissionDenied,
                                "Original owner expired before ACK5",
                            ));
                        }
                        Ok(())
                    },
                );
                let observation = Box::pin(observation);
                let after_frame = async {
                    after_frame.await.map_err(|_| {
                        io::Error::other("Registered helper did not reach owner renewal")
                    })
                };
                let observation_pending =
                    match futures::future::select(observation, Box::pin(after_frame)).await {
                        futures::future::Either::Right((received, pending)) => {
                            received.expect("actual complete-frame owner check entered");
                            pending
                        }
                        futures::future::Either::Left(_) => {
                            panic!("Native observation finished before held owner renewal")
                        }
                    };
                assert!(
                    !marker.exists(),
                    "held owner check must execute no collector child"
                );
                if scenario == "revoked" {
                    drop(release);
                    assert_eq!(
                        observation_pending
                            .await
                            .expect_err("expired owner denied before exec")
                            .kind(),
                        io::ErrorKind::PermissionDenied
                    );
                } else {
                    drop(observation_pending);
                    drop(release);
                }
                assert!(
                    !helper
                        .status()
                        .await
                        .expect("genuine refused helper kernel exit")
                        .success()
                );
                assert!(
                    !marker.exists(),
                    "revoked/dropped registered helper must execute no child"
                );
                let fresh =
                    TaskCompletionListener::bind([0x42; 32]).expect("fresh actual listener");
                let expected = checked_native_test_invocation(&marker);
                let mut helper = checked_native_test_helper(&fresh.path(), &expected)
                    .spawn()
                    .expect("fresh actual helper");
                let status = fresh
                    .observe_checked_native_profile_with_helper_check(expected, |_| async {
                        Ok(())
                    })
                    .await
                    .expect("fresh raw component admission");
                assert_eq!(status.code(), Some(23));
                assert_eq!(
                    helper.status().await.expect("actual fresh kernel exit"),
                    status
                );
                assert_eq!(
                    std::fs::read(marker).expect("fresh child marker"),
                    b"actual-checked-native-child"
                );
            }
        });
    }

    #[test]
    fn task_completion_helper() {
        let Some(socket) = std::env::var_os("ZED_TEST_TASK_SOCKET") else {
            return;
        };
        let mut command = std::process::Command::new("/bin/sh");
        command
            .arg("-c")
            .arg(std::env::var("ZED_TEST_TASK_COMMAND").expect("command"));
        if std::env::var_os("ZED_TEST_NATIVE_LITERAL").is_some() {
            use std::os::unix::ffi::OsStringExt as _;
            command
                .arg("fixture")
                .arg(std::ffi::OsString::from_vec(vec![
                    b';', b' ', b'$', b'{', b'Z', b'E', b'D', b'_', b'F', b'I', b'L', b'E', b'}',
                    0xff,
                ]));
        }
        if std::env::var_os("ZED_TEST_NATIVE_ENV_CHILD").is_some() {
            command = std::process::Command::new(std::env::current_exe().expect("test executable"));
            command.args([
                "--exact",
                "macos_task_completion::tests::native_profile_environment_child",
                "--nocapture",
            ]);
        }
        if std::env::var_os("ZED_TEST_BUILD_ENVELOPE_CHILD").is_some() {
            command = std::process::Command::new(std::env::current_exe().expect("test executable"));
            command.args([
                "--exact",
                "macos_task_completion::tests::checked_build_envelope_environment_child",
                "--nocapture",
            ]);
        }
        if std::env::var_os("ZED_TEST_TASK_CAPTURE_BUILD_STDOUT").is_some() {
            let status = register_and_capture_build_stdout(Path::new(&socket), command)
                .expect("actual checked Build child");
            exit_with_observed_task_status(status);
        }
        if std::env::var_os("ZED_TEST_TASK_CAPTURE_STDOUT").is_some() {
            let status = register_and_capture_stdout(Path::new(&socket), command)
                .expect("actual child stdout capture");
            if std::env::var_os("ZED_TEST_TASK_BLOCK_EXIT_SIGNAL").as_deref()
                == Some(std::ffi::OsStr::new("1"))
            {
                assert_eq!(status.signal(), Some(libc::SIGTERM));
                unsafe {
                    let mut signals = std::mem::MaybeUninit::<libc::sigset_t>::uninit();
                    assert_eq!(libc::sigemptyset(signals.as_mut_ptr()), 0);
                    let mut signals = signals.assume_init();
                    assert_eq!(libc::sigaddset(&mut signals, libc::SIGTERM), 0);
                    assert_eq!(
                        libc::pthread_sigmask(libc::SIG_BLOCK, &signals, std::ptr::null_mut()),
                        0
                    );
                    let mut current = std::mem::MaybeUninit::<libc::sigset_t>::uninit();
                    assert_eq!(
                        libc::pthread_sigmask(
                            libc::SIG_UNBLOCK,
                            std::ptr::null(),
                            current.as_mut_ptr()
                        ),
                        0
                    );
                    assert_eq!(libc::sigismember(&current.assume_init(), libc::SIGTERM), 1);
                }
            }
            exit_with_observed_task_status(status);
        }
        let error = if std::env::var_os("ZED_TEST_NATIVE_PROFILE_CAPTURE").is_some() {
            register_and_exec_native_profile_capture(Path::new(&socket), &mut command)
        } else {
            register_and_exec(Path::new(&socket), &mut command)
        };
        panic!("{error}");
    }

    fn helper(socket: &Path, token: &str, command: &str) -> smol::process::Child {
        smol::process::Command::new(std::env::current_exe().expect("test executable"))
            .args([
                "--exact",
                "macos_task_completion::tests::task_completion_helper",
                "--nocapture",
            ])
            .env("ZED_TEST_TASK_SOCKET", socket)
            .env("ZED_TEST_TASK_COMMAND", command)
            .env(TASK_COMPLETION_TOKEN_ENV, token)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .expect("spawn helper")
    }

    fn checked_build_helper(socket: &Path, command: &str) -> smol::process::Child {
        smol::process::Command::new(std::env::current_exe().expect("test executable"))
            .args([
                "--exact",
                "macos_task_completion::tests::task_completion_helper",
                "--nocapture",
            ])
            .env("ZED_TEST_TASK_SOCKET", socket)
            .env("ZED_TEST_TASK_COMMAND", command)
            .env("ZED_TEST_TASK_CAPTURE_BUILD_STDOUT", "1")
            .env("ZED_TEST_CHECKED_ENV", "retained")
            .env_remove("ZED_TEST_TASK_CAPTURE_STDOUT")
            .env_remove("ZED_TEST_NATIVE_PROFILE_CAPTURE")
            .env_remove("ZED_TEST_NATIVE_LITERAL")
            .env_remove("ZED_TEST_NATIVE_ENV_CHILD")
            .env(TASK_COMPLETION_TOKEN_ENV, "42".repeat(32))
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .expect("checked helper")
    }

    fn enveloped_build_helper(
        socket: &Path,
        command: &str,
        encoded: &str,
        environment_child: bool,
    ) -> smol::process::Child {
        let mut helper =
            smol::process::Command::new(std::env::current_exe().expect("test executable"));
        helper
            .args([
                "--exact",
                "macos_task_completion::tests::task_completion_helper",
                "--nocapture",
            ])
            .env("ZED_TEST_TASK_SOCKET", socket)
            .env("ZED_TEST_TASK_COMMAND", command)
            .env("ZED_TEST_TASK_CAPTURE_BUILD_STDOUT", "1")
            .env(TASK_BUILD_INVOCATION_ENV, encoded)
            .env("ZED_TEST_CHECKED_ENV", "retained")
            .env("SHLVL", "2")
            .env("TERM", "terminal-preparation-value")
            .env("PATH", "/changed-by-shell")
            .env_remove("ZED_TEST_TASK_CAPTURE_STDOUT")
            .env_remove("ZED_TEST_NATIVE_PROFILE_CAPTURE")
            .env_remove("ZED_TEST_NATIVE_LITERAL")
            .env_remove("ZED_TEST_NATIVE_ENV_CHILD")
            .env_remove("ZED_TEST_BUILD_ENVELOPE_CHILD")
            .env(TASK_COMPLETION_TOKEN_ENV, "42".repeat(32))
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true);
        if environment_child {
            helper.env("ZED_TEST_BUILD_ENVELOPE_CHILD", "1");
        }
        helper.spawn().expect("actual envelope helper")
    }

    #[test]
    fn checked_build_envelope_environment_child() {
        use std::io::{Read as _, Write as _};
        if std::env::var_os("ZED_TEST_BUILD_ENVELOPE_CHILD").is_none() {
            return;
        }
        assert_eq!(std::env::var("SHLVL").expect("admitted level"), "37");
        assert_eq!(
            std::env::var("TERM").expect("admitted terminal"),
            "admitted-terminal"
        );
        assert_eq!(
            std::env::var("PATH").expect("admitted path"),
            "/usr/bin:/bin"
        );
        assert!(std::env::var_os(TASK_BUILD_INVOCATION_ENV).is_none());
        assert!(std::env::var_os(TASK_COMPLETION_TOKEN_ENV).is_none());
        let mut input = [0; 3];
        std::io::stdin()
            .read_exact(&mut input)
            .expect("owned child stdin");
        assert_eq!(&input, b"go\n");
        std::io::stdout()
            .write_all(b"envelope-selected\0")
            .expect("owned child output");
        std::io::stdout().flush().expect("flush actual output");
        std::process::exit(0);
    }

    #[test]
    fn task_checked_build_envelope_restores_actual_child_environment() {
        smol::block_on(async {
            let listener = TaskCompletionListener::bind([0x42; 32]).expect("bind");
            let expected = TaskBuildInvocation::new(
                std::env::current_exe().expect("owned child"),
                vec![
                    "--exact".into(),
                    "macos_task_completion::tests::checked_build_envelope_environment_child".into(),
                    "--nocapture".into(),
                ],
                std::env::current_dir().expect("cwd"),
                std::collections::BTreeMap::from([
                    ("SHLVL".into(), "37".into()),
                    ("TERM".into(), "admitted-terminal".into()),
                    ("PATH".into(), "/usr/bin:/bin".into()),
                ]),
            )
            .expect("admitted invocation");
            let encoded = expected
                .encode_helper_environment()
                .expect("bounded private envelope");
            let mut child = enveloped_build_helper(&listener.path(), "unused", &encoded, true);
            child
                .stdin
                .as_mut()
                .expect("stdin")
                .write_all(b"go\n")
                .await
                .expect("input");
            let actual_helper = std::env::current_exe().expect("helper image");
            let observed = listener
                .observe_checked_build_stdout_with_helper_check(
                    4096,
                    expected.clone(),
                    |image| async move {
                        assert_eq!(image, actual_helper);
                        Ok(())
                    },
                )
                .await
                .expect("actual restored child receipt");
            let (status, bytes, actual) = observed.into_parts_with_invocation();
            assert!(status.success());
            assert_eq!(child.status().await.expect("helper exit"), status);
            assert_eq!(
                bytes
                    .windows(b"envelope-selected\0".len())
                    .filter(|bytes| *bytes == b"envelope-selected\0")
                    .count(),
                1
            );
            let actual = actual.expect("actual frozen invocation");
            assert!(expected.matches_observed(&actual));
            assert_eq!(
                actual.environment().get("SHLVL").map(String::as_str),
                Some("37")
            );
            assert_eq!(
                actual.environment().get("TERM").map(String::as_str),
                Some("admitted-terminal")
            );
            assert_eq!(
                actual.environment().get("PATH").map(String::as_str),
                Some("/usr/bin:/bin")
            );
            assert!(!actual.environment().contains_key(TASK_BUILD_INVOCATION_ENV));
            assert!(!actual.environment().contains_key(TASK_COMPLETION_TOKEN_ENV));
        });
    }

    #[test]
    fn task_checked_build_envelope_rejects_tampering_without_child() {
        smol::block_on(async {
            for changed in ["envelope", "program", "argv", "cwd", "environment"] {
                let directory = tempfile::tempdir().expect("fixture");
                let marker = directory.path().join("executed");
                let command = format!("/usr/bin/touch '{}'; printf selected", marker.display());
                let expected = TaskBuildInvocation::new(
                    Path::new("/bin/sh").to_path_buf(),
                    vec!["-c".into(), command.clone()],
                    std::env::current_dir().expect("cwd"),
                    std::collections::BTreeMap::from([("PATH".into(), "/usr/bin:/bin".into())]),
                )
                .expect("original invocation");
                let mut tampered = expected.clone();
                match changed {
                    "program" => tampered.program = Path::new("/usr/bin/false").to_path_buf(),
                    "argv" => tampered.arguments.push("unexpected".into()),
                    "cwd" => tampered.cwd = directory.path().to_path_buf(),
                    "environment" => {
                        tampered
                            .environment
                            .insert("PATH".into(), "/tampered".into());
                    }
                    "envelope" => {}
                    _ => unreachable!(),
                }
                let encoded = if changed == "envelope" {
                    "{\"success\":true}".to_owned()
                } else {
                    tampered
                        .encode_helper_environment()
                        .expect("bounded tampered data")
                };
                let listener = TaskCompletionListener::bind([0x42; 32]).expect("bind");
                let mut child = enveloped_build_helper(&listener.path(), &command, &encoded, false);
                if changed == "environment" {
                    let error = listener
                        .observe_checked_build_stdout_with_helper_check(64, expected, |_| async {
                            Ok(())
                        })
                        .await
                        .err()
                        .expect("original admitted environment rejects tampering");
                    assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
                }
                assert!(!child.status().await.expect("refused helper exit").success());
                assert!(
                    !marker.exists(),
                    "refused envelope cannot execute the actual child"
                );
            }
        });
        assert!(
            TaskBuildInvocation::new(
                Path::new("/bin/sh").to_path_buf(),
                vec![],
                std::env::current_dir().expect("cwd"),
                std::collections::BTreeMap::from([(
                    TASK_BUILD_INVOCATION_ENV.into(),
                    "caller value".into()
                )]),
            )
            .is_err(),
            "caller-required environment cannot contain the private envelope key"
        );
        let oversized = TaskBuildInvocation::new(
            Path::new("/bin/sh").to_path_buf(),
            vec![],
            std::env::current_dir().expect("cwd"),
            std::collections::BTreeMap::from_iter(
                (0..15).map(|index| (format!("ENTRY_{index}"), "\u{1}".repeat(4096))),
            ),
        )
        .expect("bounded raw environment");
        assert!(
            oversized.encode_helper_environment().is_err(),
            "JSON expansion cannot exceed private envelope bound"
        );
    }

    #[test]
    fn task_helper_suppresses_build_envelope_outside_checked_mode() {
        smol::block_on(async {
            for mode in ["exec", "native", "stdout"] {
                let listener = TaskCompletionListener::bind([0x42; 32]).expect("bind");
                let command = "test -z \"${ZED_PROFILE_BUILD_INVOCATION_ENV+x}\" && test -z \"${ZED_PROFILE_TASK_TOKEN+x}\" && printf selected";
                let mut helper =
                    smol::process::Command::new(std::env::current_exe().expect("test executable"));
                helper
                    .args([
                        "--exact",
                        "macos_task_completion::tests::task_completion_helper",
                        "--nocapture",
                    ])
                    .env("ZED_TEST_TASK_SOCKET", listener.path())
                    .env("ZED_TEST_TASK_COMMAND", command)
                    .env(
                        TASK_BUILD_INVOCATION_ENV,
                        "not JSON; never restore outside checked Build",
                    )
                    .env(TASK_COMPLETION_TOKEN_ENV, "42".repeat(32))
                    .env_remove("ZED_TEST_TASK_CAPTURE_BUILD_STDOUT")
                    .env_remove("ZED_TEST_TASK_CAPTURE_STDOUT")
                    .env_remove("ZED_TEST_NATIVE_PROFILE_CAPTURE")
                    .env_remove("ZED_TEST_NATIVE_LITERAL")
                    .env_remove("ZED_TEST_NATIVE_ENV_CHILD")
                    .env_remove("ZED_TEST_BUILD_ENVELOPE_CHILD")
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .kill_on_drop(true);
                match mode {
                    "native" => {
                        helper.env("ZED_TEST_NATIVE_PROFILE_CAPTURE", "1");
                    }
                    "stdout" => {
                        helper.env("ZED_TEST_TASK_CAPTURE_STDOUT", "1");
                    }
                    _ => {}
                }
                let mut child = helper.spawn().expect("actual ordinary helper");
                let status = match mode {
                    "native" => listener
                        .observe_native_profile_with_helper_check(|_| async { Ok(()) })
                        .await
                        .expect("native helper kernel status"),
                    "stdout" => {
                        let report = listener
                            .observe_stdout(64)
                            .await
                            .expect("unchecked actual stdout");
                        assert_eq!(report.stdout(), b"selected");
                        report.status()
                    }
                    _ => listener
                        .observe()
                        .await
                        .expect("generic helper kernel status"),
                };
                assert!(
                    status.success(),
                    "private envelope is removed without parsing or applying it"
                );
                assert_eq!(child.status().await.expect("helper status"), status);
            }
        });
    }

    #[test]
    fn task_checked_build_stdout_records_actual_invocation() {
        smol::block_on(async {
            let listener = TaskCompletionListener::bind([0x42; 32]).expect("bind");
            let command = "read value; test \"$value\" = go && test -z \"${ZED_PROFILE_TASK_TOKEN+x}\" && printf selected";
            let expected = TaskBuildInvocation::new(
                Path::new("/bin/sh").to_path_buf(),
                vec!["-c".into(), command.into()],
                std::env::current_dir().expect("cwd"),
                std::collections::BTreeMap::from([(
                    "ZED_TEST_CHECKED_ENV".into(),
                    "retained".into(),
                )]),
            )
            .expect("expected recipe");
            let mut child = checked_build_helper(&listener.path(), command);
            child
                .stdin
                .as_mut()
                .expect("stdin")
                .write_all(b"go\n")
                .await
                .expect("input");
            let helper = std::env::current_exe().expect("helper image");
            let observed = listener
                .observe_checked_build_stdout_with_helper_check(
                    64,
                    expected.clone(),
                    |actual| async move {
                        assert_eq!(actual, helper);
                        Ok(())
                    },
                )
                .await
                .expect("checked child receipt");
            let (status, bytes, invocation) = observed.into_parts_with_invocation();
            assert!(status.success());
            assert_eq!(child.status().await.expect("exit"), status);
            assert_eq!(bytes, b"selected");
            let actual = invocation.expect("actual child facts");
            assert!(expected.matches_observed(&actual));
            assert_eq!(actual.program(), Path::new("/bin/sh"));
            assert!(!actual.environment().contains_key(TASK_COMPLETION_TOKEN_ENV));
        });
    }

    #[test]
    fn task_checked_build_stdout_rejects_changed_recipe_before_spawn() {
        smol::block_on(async {
            for changed in ["argv", "cwd", "environment"] {
                let directory = tempfile::tempdir().expect("fixture");
                let marker = directory.path().join("executed");
                let command = format!("touch '{}'; printf selected", marker.display());
                let mut arguments = vec!["-c".into(), command.clone()];
                let mut cwd = std::env::current_dir().expect("cwd");
                let mut environment = std::collections::BTreeMap::from([(
                    "ZED_TEST_CHECKED_ENV".into(),
                    "retained".into(),
                )]);
                match changed {
                    "argv" => arguments.push("unexpected".into()),
                    "cwd" => cwd = directory.path().to_path_buf(),
                    "environment" => {
                        environment.insert("ZED_TEST_CHECKED_ENV".into(), "changed".into());
                    }
                    _ => unreachable!(),
                }
                let expected = TaskBuildInvocation::new(
                    Path::new("/bin/sh").to_path_buf(),
                    arguments,
                    cwd,
                    environment,
                )
                .expect("expected");
                let listener = TaskCompletionListener::bind([0x42; 32]).expect("bind");
                let mut child = checked_build_helper(&listener.path(), &command);
                let error = listener
                    .observe_checked_build_stdout_with_helper_check(64, expected, |_| async {
                        Ok(())
                    })
                    .await
                    .err()
                    .expect("changed recipe rejected");
                assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
                assert!(!child.status().await.expect("exit").success());
                assert!(!marker.exists());
            }
        });
    }

    #[test]
    fn task_checked_build_stdout_bounds_reject_forged_and_oversized_context() {
        let cwd = std::env::current_dir().expect("cwd");
        assert!(
            TaskBuildInvocation::new(
                Path::new("relative").to_path_buf(),
                vec![],
                cwd.clone(),
                Default::default()
            )
            .is_err()
        );
        assert!(
            TaskBuildInvocation::new(
                Path::new("/bin/sh").to_path_buf(),
                vec!["x".repeat(4097)],
                cwd.clone(),
                Default::default()
            )
            .is_err()
        );
        assert!(
            TaskBuildInvocation::new(
                Path::new("/bin/sh").to_path_buf(),
                vec!["x".repeat(4096); 9],
                cwd.clone(),
                Default::default()
            )
            .is_err()
        );
        assert!(
            TaskBuildInvocation::new(
                Path::new("/bin/sh").to_path_buf(),
                vec![],
                cwd,
                std::collections::BTreeMap::from([(
                    TASK_COMPLETION_TOKEN_ENV.into(),
                    "private".into()
                )])
            )
            .is_err()
        );
        let bytes =
            br#"{"program":"/bin/sh","arguments":[],"cwd":"/tmp","environment":{},"success":true}"#;
        assert!(
            serde_json::from_slice::<TaskBuildInvocation>(bytes).is_err(),
            "caller success fields are not invocation authority"
        );
    }

    #[test]
    fn native_profile_environment_child() {
        use std::io::Write as _;
        let Some(result_path) = std::env::var_os("ZED_TEST_NATIVE_ENV_RESULT") else {
            return;
        };
        assert!(std::env::var_os(TASK_COMPLETION_TOKEN_ENV).is_none());
        assert_eq!(
            std::env::var("ZED_TEST_ORDINARY_ENV").expect("ordinary env"),
            "retained 雪"
        );
        assert_eq!(
            std::env::var("SAMPLY_USE_DEBUGINFOD").expect("ordinary workload env"),
            "1"
        );
        let inherited = std::env::var("_NT_SYMBOL_PATH").ok();
        if let Some(value) = &inherited {
            let address = value
                .strip_prefix("srv*http://")
                .expect("controlled symbol setting");
            let mut connection =
                std::net::TcpStream::connect(address).expect("controlled loopback");
            connection
                .set_write_timeout(Some(Duration::from_secs(5)))
                .expect("bounded write");
            connection
                .write_all(b"GET /fixture.pdb/id HTTP/1.0\r\n\r\n")
                .expect("controlled request");
        }
        std::fs::write(
            result_path,
            if inherited.is_some() {
                b"inherited".as_slice()
            } else {
                b"removed".as_slice()
            },
        )
        .expect("child environment receipt");
    }

    fn environment_helper(
        socket: &Path,
        command: &str,
        native: bool,
        network: Option<(std::net::SocketAddr, &Path)>,
    ) -> smol::process::Child {
        let mut helper =
            smol::process::Command::new(std::env::current_exe().expect("test executable"));
        helper
            .args([
                "--exact",
                "macos_task_completion::tests::task_completion_helper",
                "--nocapture",
            ])
            .env("ZED_TEST_TASK_SOCKET", socket)
            .env("ZED_TEST_TASK_COMMAND", command)
            .env(TASK_COMPLETION_TOKEN_ENV, "42".repeat(32))
            .env("ZED_TEST_ORDINARY_ENV", "retained 雪")
            .env("SAMPLY_USE_DEBUGINFOD", "1")
            .env_remove("ZED_TEST_TASK_CAPTURE_STDOUT")
            .env_remove("ZED_TEST_NATIVE_PROFILE_CAPTURE")
            .env_remove("ZED_TEST_NATIVE_ENV_CHILD")
            .env_remove("ZED_TEST_NATIVE_ENV_RESULT")
            .env_remove("ZED_TEST_NATIVE_LITERAL")
            .env("_NT_SYMBOL_PATH", "srv*http://127.0.0.1:9")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true);
        if native {
            helper.env("ZED_TEST_NATIVE_PROFILE_CAPTURE", "1");
        }
        if let Some((address, result)) = network {
            helper
                .env("_NT_SYMBOL_PATH", format!("srv*http://{address}"))
                .env("ZED_TEST_NATIVE_ENV_CHILD", "1")
                .env("ZED_TEST_NATIVE_ENV_RESULT", result);
        }
        helper.spawn().expect("spawn controlled helper")
    }

    #[test]
    fn native_profile_environment_suppresses_inherited_symbol_channel_only() {
        use std::io::Read as _;
        smol::block_on(async {
            for native in [false, true] {
                let directory = tempfile::tempdir().expect("fixture");
                let result = directory.path().join("environment-result");
                let network = std::net::TcpListener::bind("127.0.0.1:0").expect("private loopback");
                network.set_nonblocking(true).expect("nonblocking listener");
                let address = network.local_addr().expect("loopback address");
                let listener = TaskCompletionListener::bind([0x42; 32]).expect("bind");
                let path = listener.path();
                let mut child =
                    environment_helper(&path, "exit 0", native, Some((address, &result)));
                let expected_helper = std::env::current_exe().expect("test executable");
                let observed = if native {
                    listener
                        .observe_native_profile_with_helper_check(|actual| async move {
                            assert_eq!(actual, expected_helper);
                            Ok(())
                        })
                        .await
                        .expect("admitted native helper")
                } else {
                    listener.observe().await.expect("ordinary helper")
                };
                assert!(observed.success());
                assert_eq!(child.status().await.expect("actual helper exit"), observed);
                assert_eq!(
                    std::fs::read(&result).expect("actual child receipt"),
                    if native {
                        b"removed".as_slice()
                    } else {
                        b"inherited".as_slice()
                    }
                );
                match network.accept() {
                    Ok((connection, _)) => {
                        assert!(
                            !native,
                            "native child must not consult inherited symbol server"
                        );
                        connection
                            .set_read_timeout(Some(Duration::from_secs(5)))
                            .expect("bounded read");
                        let mut request = Vec::new();
                        connection
                            .take(129)
                            .read_to_end(&mut request)
                            .expect("controlled bounded request");
                        assert_eq!(request, b"GET /fixture.pdb/id HTTP/1.0\r\n\r\n");
                    }
                    Err(error) => {
                        assert!(native, "ordinary positive control must connect");
                        assert_eq!(error.kind(), io::ErrorKind::WouldBlock);
                    }
                }
                assert!(!path.exists());
            }
        });
    }

    #[test]
    fn native_profile_environment_preserves_literal_argv_stdin_and_exit() {
        smol::block_on(async {
            let directory = tempfile::tempdir().expect("fixture");
            let output = directory.path().join("literal-argv");
            let listener = TaskCompletionListener::bind([0x42; 32]).expect("bind");
            let path = listener.path();
            let mut command =
                smol::process::Command::new(std::env::current_exe().expect("test executable"));
            command.args([
                "--exact",
                "macos_task_completion::tests::task_completion_helper",
                "--nocapture",
            ]).env("ZED_TEST_TASK_SOCKET", &path)
                .env("ZED_TEST_TASK_COMMAND", "read value; test \"$value\" = go || exit 8; test -z \"${_NT_SYMBOL_PATH+x}\" || exit 9; test -z \"${ZED_PROFILE_TASK_TOKEN+x}\" || exit 10; printf '%s' \"$1\" > \"$ZED_TEST_LITERAL_OUTPUT\"; exit 7")
                .env("ZED_TEST_LITERAL_OUTPUT", &output)
                .env("ZED_TEST_NATIVE_LITERAL", "1")
                .env("ZED_TEST_NATIVE_PROFILE_CAPTURE", "1")
                .env("_NT_SYMBOL_PATH", "srv*http://127.0.0.1:9")
                .env(TASK_COMPLETION_TOKEN_ENV, "42".repeat(32))
                .env_remove("ZED_TEST_TASK_CAPTURE_STDOUT")
                .env_remove("ZED_TEST_NATIVE_ENV_CHILD")
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .kill_on_drop(true);
            let mut child = command.spawn().expect("helper");
            child
                .stdin
                .as_mut()
                .expect("stdin")
                .write_all(b"go\n")
                .await
                .expect("input");
            let expected_helper = std::env::current_exe().expect("test executable");
            let observed = listener
                .observe_native_profile_with_helper_check(|actual| async move {
                    assert_eq!(actual, expected_helper);
                    Ok(())
                })
                .await
                .expect("native observed child exit");
            assert_eq!(observed.code(), Some(7));
            assert_eq!(child.status().await.expect("helper exit"), observed);
            assert_eq!(
                std::fs::read(output).expect("literal child bytes"),
                b"; ${ZED_FILE}\xff"
            );
        });
    }

    #[test]
    fn native_profile_environment_mode_and_image_mismatch_prevent_exec() {
        smol::block_on(async {
            for (native, native_admission, permit_helper) in [
                (true, false, true),
                (false, true, true),
                (true, true, false),
            ] {
                let directory = tempfile::tempdir().expect("fixture");
                let marker = directory.path().join("executed");
                let listener = TaskCompletionListener::bind([0x42; 32]).expect("bind");
                let path = listener.path();
                let mut child = environment_helper(
                    &path,
                    &format!("touch '{}'", marker.display()),
                    native,
                    None,
                );
                let observed = if native_admission {
                    listener
                        .observe_native_profile_with_helper_check(|_| async move {
                            if permit_helper {
                                Ok(())
                            } else {
                                Err(io::Error::new(
                                    io::ErrorKind::PermissionDenied,
                                    "controlled helper rejection",
                                ))
                            }
                        })
                        .await
                } else {
                    listener.observe().await
                };
                if permit_helper {
                    assert!(!observed.expect("kernel observed refused helper").success());
                } else {
                    assert_eq!(
                        observed.expect_err("image refused before ACK").kind(),
                        io::ErrorKind::PermissionDenied
                    );
                }
                assert!(!child.status().await.expect("helper exit").success());
                assert!(!marker.exists());
                assert!(!path.exists());
            }
        });
    }

    #[test]
    fn observes_exec_success_failure_and_signal() {
        smol::block_on(async {
            for (command, code, signal) in [
                ("test -z \"$ZED_PROFILE_TASK_TOKEN\"", Some(0), None),
                ("exit 7", Some(7), None),
                ("kill -TERM $$", None, Some(libc::SIGTERM)),
            ] {
                let listener = TaskCompletionListener::bind([0x42; 32]).expect("bind");
                let path = listener.path();
                let mut child = helper(&path, &"42".repeat(32), command);
                let observed = listener.observe().await.expect("kernel completion");
                assert_eq!(observed.code(), code);
                assert_eq!(observed.signal(), signal);
                assert_eq!(child.status().await.expect("wait"), observed);
                assert!(!path.exists());
            }
        });
    }

    #[test]
    fn invalid_registration_does_not_execute_command() {
        smol::block_on(async {
            let directory = tempfile::tempdir().expect("fixture directory");
            let marker = directory.path().join("executed");
            let listener = TaskCompletionListener::bind([0x42; 32]).expect("bind");
            let path = listener.path();
            let mut child = helper(
                &path,
                &"43".repeat(32),
                &format!("touch '{}'", marker.display()),
            );
            let error = listener.observe().await.expect_err("wrong nonce must fail");
            assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
            assert!(!child.status().await.expect("wait").success());
            assert!(!marker.exists());
            assert!(!path.exists());
        });
    }

    #[test]
    fn dropping_registration_removes_socket() {
        let listener = TaskCompletionListener::bind([0x42; 32]).expect("bind");
        let path = listener.path();
        let directory = path.parent().expect("parent").to_owned();
        assert_eq!(
            std::fs::metadata(&directory)
                .expect("metadata")
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
        assert_eq!(
            std::fs::metadata(&path)
                .expect("metadata")
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        drop(listener);
        assert!(!directory.exists());
    }

    #[test]
    fn dropping_observation_does_not_cancel_the_task() {
        smol::block_on(async {
            let listener = TaskCompletionListener::bind([0x42; 32]).expect("bind");
            let path = listener.path();
            let mut child = helper(&path, &"42".repeat(32), "read value; test \"$value\" = go");
            let observation = listener.register().await.expect("register");
            assert!(!path.exists());
            assert!(child.try_status().expect("check child").is_none());
            drop(observation);
            child
                .stdin
                .as_mut()
                .expect("stdin")
                .write_all(b"go\n")
                .await
                .expect("resume command");
            assert!(child.status().await.expect("wait").success());
        });
    }

    #[test]
    fn registration_and_completion_expire_without_cancelling_task() {
        smol::block_on(async {
            let listener = TaskCompletionListener::bind([0x42; 32]).expect("bind");
            let path = listener.path();
            let error = listener
                .observe_with_deadlines(Duration::from_millis(10), Duration::from_secs(5))
                .await
                .expect_err("missing registration expires");
            assert_eq!(error.kind(), io::ErrorKind::TimedOut);
            assert!(!path.exists());

            let listener = TaskCompletionListener::bind([0x42; 32]).expect("bind");
            let path = listener.path();
            let mut child = helper(&path, &"42".repeat(32), "read value; test \"$value\" = go");
            let error = listener
                .observe_with_deadlines(Duration::from_secs(5), Duration::from_millis(10))
                .await
                .expect_err("unfinished Task observation expires");
            assert_eq!(error.kind(), io::ErrorKind::TimedOut);
            assert!(!path.exists());
            assert!(child.try_status().expect("check child").is_none());
            child
                .stdin
                .as_mut()
                .expect("stdin")
                .write_all(b"go\n")
                .await
                .expect("resume command");
            assert!(child.status().await.expect("wait").success());
        });
    }

    fn stdout_helper(socket: &Path, token: &str, command: &str) -> smol::process::Child {
        stdout_helper_with_exit_mask(socket, token, command, false)
    }

    fn stdout_helper_with_exit_mask(
        socket: &Path,
        token: &str,
        command: &str,
        block_exit_signal: bool,
    ) -> smol::process::Child {
        smol::process::Command::new(std::env::current_exe().expect("test executable"))
            .args([
                "--exact",
                "macos_task_completion::tests::task_completion_helper",
                "--nocapture",
            ])
            .env("ZED_TEST_TASK_SOCKET", socket)
            .env("ZED_TEST_TASK_COMMAND", command)
            .env("ZED_TEST_TASK_CAPTURE_STDOUT", "1")
            .env(
                "ZED_TEST_TASK_BLOCK_EXIT_SIGNAL",
                if block_exit_signal { "1" } else { "0" },
            )
            .env(TASK_COMPLETION_TOKEN_ENV, token)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .expect("spawn stdout helper")
    }

    #[test]
    fn task_stdout_observes_actual_exit_stdin_stderr_and_console_bytes() {
        smol::block_on(async {
            for (ending, code, signal, block_exit_signal) in [
                ("exit 0", Some(0), None, false),
                ("exit 7", Some(7), None, false),
                ("kill -TERM $$", None, Some(libc::SIGTERM), false),
                ("kill -TERM $$", None, Some(libc::SIGTERM), true),
            ] {
                let listener = TaskCompletionListener::bind([0x42; 32]).expect("bind");
                let path = listener.path();
                let mut child = stdout_helper_with_exit_mask(
                    &path,
                    &"42".repeat(32),
                    &format!(
                        "test -z \"$ZED_PROFILE_TASK_TOKEN\" || exit 31; read value; test \"$value\" = go || exit 32; printf 'selected\\000bytes\\n'; printf 'inherited diagnostic' >&2; {ending}"
                    ),
                    block_exit_signal,
                );
                child
                    .stdin
                    .take()
                    .expect("stdin")
                    .write_all(b"go\n")
                    .await
                    .expect("input");
                let (observed, output) =
                    future::try_zip(listener.observe_stdout(4096), child.output())
                        .await
                        .expect("actual stdout/kernel completion");
                assert_eq!(observed.status().code(), code);
                assert_eq!(observed.status().signal(), signal);
                assert_eq!(output.status, observed.status());
                assert_eq!(observed.stdout(), b"selected\0bytes\n");
                assert_eq!(
                    output
                        .stdout
                        .windows(observed.stdout().len())
                        .filter(|bytes| *bytes == observed.stdout())
                        .count(),
                    1
                );
                assert!(
                    output
                        .stderr
                        .windows(b"inherited diagnostic".len())
                        .any(|bytes| bytes == b"inherited diagnostic")
                );
                assert!(!path.exists());
            }
        });
    }

    #[test]
    fn task_stdout_exact_limit_overflow_wrong_token_and_mode_do_not_seal() {
        smol::block_on(async {
            let listener = TaskCompletionListener::bind([0x42; 32]).expect("bind");
            let path = listener.path();
            let child = stdout_helper(&path, &"42".repeat(32), "/usr/bin/head -c 4096 /dev/zero");
            let (observed, output) = future::try_zip(listener.observe_stdout(4096), child.output())
                .await
                .expect("inclusive limit");
            assert!(observed.status().success());
            assert!(output.status.success());
            assert_eq!(observed.stdout(), &[0; 4096]);

            let listener = TaskCompletionListener::bind([0x42; 32]).expect("bind");
            let path = listener.path();
            let child = stdout_helper(&path, &"42".repeat(32), "/usr/bin/head -c 4097 /dev/zero");
            let (observed, output) =
                future::zip(listener.observe_stdout(4096), child.output()).await;
            assert!(observed.is_err(), "no complete report on overflow");
            assert!(!output.expect("helper retired").status.success());

            let directory = tempfile::tempdir().expect("fixture");
            for (token, stdout_mode) in [(&"43".repeat(32), true), (&"42".repeat(32), false)] {
                let marker = directory.path().join(if stdout_mode {
                    "wrong-token"
                } else {
                    "wrong-mode"
                });
                let listener = TaskCompletionListener::bind([0x42; 32]).expect("bind");
                let path = listener.path();
                let command = format!("touch '{}'", marker.display());
                if stdout_mode {
                    let child = stdout_helper(&path, token, &command);
                    let (observed, output) =
                        future::zip(listener.observe_stdout(4096), child.output()).await;
                    assert!(observed.is_err());
                    assert!(!output.expect("helper retired").status.success());
                } else {
                    let mut child = helper(&path, token, &command);
                    assert!(listener.observe_stdout(4096).await.is_err());
                    assert!(!child.status().await.expect("helper retired").success());
                }
                assert!(
                    !marker.exists(),
                    "refused mode/token must not execute workload"
                );
                assert!(!path.exists());
            }
            let marker = directory.path().join("new-helper-old-observer");
            let listener = TaskCompletionListener::bind([0x42; 32]).expect("bind");
            let path = listener.path();
            let child = stdout_helper(
                &path,
                &"42".repeat(32),
                &format!("touch '{}'", marker.display()),
            );
            let (observed, output) = future::zip(listener.observe(), child.output()).await;
            assert!(!observed.expect("actual refusal exit").success());
            assert!(!output.expect("helper retired").status.success());
            assert!(!marker.exists());
        });
    }

    async fn wait_for_stdout_child(path: &Path) -> libc::pid_t {
        future::race(
            async {
                loop {
                    // The shell creates the file before echo publishes the complete record.
                    if let Ok(contents) = std::fs::read_to_string(path)
                        && contents.ends_with('\n')
                    {
                        return contents.trim().parse().expect("actual child PID");
                    }
                    observation_deadline(Duration::from_millis(5)).await;
                }
            },
            async {
                observation_deadline(Duration::from_secs(5)).await;
                panic!("stdout child did not start");
            },
        )
        .await
    }

    #[test]
    fn task_stdout_waits_for_a_complete_pid_record() {
        smol::block_on(async {
            let process_id = libc::pid_t::try_from(std::process::id()).expect("fixture PID");
            for initial in ["1", ""] {
                let directory = tempfile::tempdir().expect("fixture");
                let path = directory.path().join("child-pid");
                std::fs::write(&path, initial).expect("incomplete PID record");
                let (observed, ()) = future::zip(wait_for_stdout_child(&path), async {
                    observation_deadline(Duration::from_millis(5)).await;
                    std::fs::write(&path, format!("{process_id}\n")).expect("complete PID record");
                })
                .await;
                assert_eq!(observed, process_id);
            }
        });
    }

    async fn require_stdout_child_retired(process_id: libc::pid_t) {
        future::race(
            async {
                loop {
                    if unsafe { libc::kill(process_id, 0) } == -1 {
                        assert_eq!(io::Error::last_os_error().raw_os_error(), Some(libc::ESRCH));
                        return;
                    }
                    observation_deadline(Duration::from_millis(5)).await;
                }
            },
            async {
                observation_deadline(Duration::from_secs(5)).await;
                panic!("owned stdout child was not retired");
            },
        )
        .await
    }

    #[test]
    fn task_stdout_disconnect_timeout_retire_owned_child() {
        smol::block_on(async {
            for timeout in [false, true] {
                let directory = tempfile::tempdir().expect("fixture");
                let pid_file = directory.path().join("child-pid");
                let listener = TaskCompletionListener::bind([0x42; 32]).expect("bind");
                let path = listener.path();
                let mut child = stdout_helper(
                    &path,
                    &"42".repeat(32),
                    &format!("echo $$ > '{}'; read value; exit 0", pid_file.display()),
                );
                let (observation, connection) =
                    listener.register_stdout(4096).await.expect("registered");
                let process_id = wait_for_stdout_child(&pid_file).await;
                if timeout {
                    let error = observe_stdout_completion(
                        observation,
                        connection,
                        4096,
                        Duration::from_millis(10),
                    )
                    .await
                    .err()
                    .expect("held report expires");
                    assert_eq!(error.kind(), io::ErrorKind::TimedOut);
                } else {
                    drop(connection);
                    drop(observation);
                }
                assert!(!child.status().await.expect("helper retires").success());
                require_stdout_child_retired(process_id).await;
                assert!(!path.exists());
            }
        });
    }

    #[test]
    fn task_stdout_frames_reject_unbounded_truncated_trailing_and_missing_end() {
        smol::block_on(async {
            for payload in [
                (TASK_STDOUT_FRAME_BYTES as u32 + 1).to_be_bytes().to_vec(),
                5_u32
                    .to_be_bytes()
                    .into_iter()
                    .chain([b'x'])
                    .collect::<Vec<_>>(),
                0_u32
                    .to_be_bytes()
                    .into_iter()
                    .chain([b'x'])
                    .collect::<Vec<_>>(),
                1_u32
                    .to_be_bytes()
                    .into_iter()
                    .chain([b'x'])
                    .collect::<Vec<_>>(),
            ] {
                let (reader, mut writer) =
                    smol::net::unix::UnixStream::pair().expect("socket pair");
                let (received, sent) = future::zip(read_stdout_report(reader, 4096), async {
                    writer.write_all(&payload).await?;
                    drop(writer);
                    io::Result::Ok(())
                })
                .await;
                sent.expect("sent malformed frame");
                assert!(received.is_err());
            }
            for limit in [0, MAX_TASK_STDOUT_REPORT_BYTES + 1] {
                let listener = TaskCompletionListener::bind([0x42; 32]).expect("bind");
                let path = listener.path();
                let error = listener
                    .observe_stdout(limit)
                    .await
                    .err()
                    .expect("bad cap denied");
                assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
                assert!(!path.exists());
            }
        });
    }

    #[test]
    fn task_stdout_registered_helper_check_runs_before_acknowledgement() {
        smol::block_on(async {
            for admitted in [true, false] {
                let fixture = tempfile::TempDir::new().expect("helper admission fixture");
                let marker = fixture.path().join("child-was-started");
                let listener = TaskCompletionListener::bind([0x51; 32]).expect("private listener");
                let token: String = [0x51_u8; 32]
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect();
                let expected =
                    std::fs::canonicalize(std::env::current_exe().expect("test executable"))
                        .expect("canonical helper");
                let child = stdout_helper(
                    &listener.path(),
                    &token,
                    &format!("printf verified; printf launched > '{}'", marker.display()),
                );
                let checked = listener.observe_stdout_with_helper_check(
                    4096,
                    async move |kernel_executable| {
                        assert_eq!(
                            std::fs::canonicalize(kernel_executable).expect("kernel helper path"),
                            expected
                        );
                        if admitted {
                            Ok(())
                        } else {
                            Err(io::Error::new(
                                io::ErrorKind::PermissionDenied,
                                "Controlled helper identity rejection",
                            ))
                        }
                    },
                );
                let (observed, output) = future::zip(checked, child.output()).await;
                let output = output.expect("real helper output");
                if admitted {
                    let observed = observed.expect("checked helper accepted");
                    assert!(observed.status().success());
                    assert_eq!(observed.stdout(), b"verified");
                    assert!(
                        marker.exists(),
                        "child starts only after helper acknowledgement"
                    );
                    assert!(output.status.success());
                } else {
                    assert_eq!(
                        observed
                            .err()
                            .expect("helper rejected before acknowledgement")
                            .kind(),
                        io::ErrorKind::PermissionDenied
                    );
                    assert!(
                        !marker.exists(),
                        "failed helper verification cannot start the child"
                    );
                    assert!(!output.status.success());
                }
            }
        });
    }
}
