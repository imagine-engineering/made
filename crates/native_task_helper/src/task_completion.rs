use std::{io, path::Path};

#[cfg(any(target_os = "macos", test))]
use std::process::ExitStatus;

#[cfg(any(target_os = "macos", test))]
use futures_lite::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, future};

pub const TASK_COMPLETION_TOKEN_ENV: &str = "ZED_PROFILE_TASK_TOKEN";
pub const TASK_BUILD_INVOCATION_ENV: &str = "ZED_PROFILE_BUILD_INVOCATION_ENV";
pub const TASK_NATIVE_INVOCATION_ENV: &str = "ZED_PROFILE_NATIVE_INVOCATION_ENV";
pub const TASK_CARGO_RUNTIME_ENDPOINT_ENV: &str = "ZED_PROFILE_CARGO_RUNTIME_ENDPOINT";
pub const TASK_CARGO_RUNTIME_TOKEN_ENV: &str = "ZED_PROFILE_CARGO_RUNTIME_TOKEN";
pub const MAX_TASK_STDOUT_REPORT_BYTES: usize = 4 * 1024 * 1024;
#[cfg(any(target_os = "macos", test))]
pub(crate) const TASK_STDOUT_FRAME_BYTES: usize = 64 * 1024;
pub const MAX_TASK_BUILD_INVOCATION_BYTES: usize = 256 * 1024;
pub const TASK_BUILD_INVOCATION_RESERVED_BYTES: usize = 2 * 1024 * 1024;

#[derive(Clone, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskBuildInvocation {
    pub(crate) program: std::path::PathBuf,
    #[serde(deserialize_with = "deserialize_build_arguments")]
    pub(crate) arguments: Vec<String>,
    pub(crate) cwd: std::path::PathBuf,
    #[serde(deserialize_with = "deserialize_build_environment")]
    pub(crate) environment: std::collections::BTreeMap<String, String>,
}

impl TaskBuildInvocation {
    pub fn new(
        program: std::path::PathBuf,
        arguments: Vec<String>,
        cwd: std::path::PathBuf,
        required_environment: std::collections::BTreeMap<String, String>,
    ) -> io::Result<Self> {
        let invocation = Self {
            program,
            arguments,
            cwd,
            environment: required_environment,
        };
        invocation.validate()?;
        Ok(invocation)
    }

    pub fn program(&self) -> &Path {
        &self.program
    }
    pub fn arguments(&self) -> &[String] {
        &self.arguments
    }
    pub fn working_directory(&self) -> &Path {
        &self.cwd
    }
    pub fn environment(&self) -> &std::collections::BTreeMap<String, String> {
        &self.environment
    }

    pub fn validate(&self) -> io::Result<()> {
        let invalid = || {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid bounded Build invocation",
            )
        };
        for path in [&self.program, &self.cwd] {
            if !path.is_absolute()
                || !path.to_str().is_some_and(|value| {
                    value.len() <= 4096 && !value.chars().any(char::is_control)
                })
            {
                return Err(invalid());
            }
        }
        if self.arguments.len() > 512 || self.environment.len() > 512 {
            return Err(invalid());
        }
        let mut argument_bytes = 0_usize;
        for argument in &self.arguments {
            if argument.len() > 4096 || argument.contains('\0') {
                return Err(invalid());
            }
            argument_bytes = argument_bytes
                .checked_add(argument.len())
                .ok_or_else(invalid)?;
            if argument_bytes > 32 * 1024 {
                return Err(invalid());
            }
        }
        let mut environment_bytes = 0_usize;
        for (key, value) in &self.environment {
            if key.is_empty()
                || key.len() > 4096
                || key.contains(['=', '\0'])
                || value.len() > 4096
                || value.contains('\0')
            {
                return Err(invalid());
            }
            environment_bytes = environment_bytes
                .checked_add(key.len() + value.len())
                .ok_or_else(invalid)?;
            if environment_bytes > 64 * 1024 {
                return Err(invalid());
            }
        }
        if self.environment.contains_key(TASK_COMPLETION_TOKEN_ENV)
            || self.environment.contains_key(TASK_BUILD_INVOCATION_ENV)
            || self.environment.contains_key(TASK_NATIVE_INVOCATION_ENV)
            || self
                .environment
                .contains_key(TASK_CARGO_RUNTIME_ENDPOINT_ENV)
            || self.environment.contains_key(TASK_CARGO_RUNTIME_TOKEN_ENV)
        {
            return Err(invalid());
        }
        Ok(())
    }

    pub fn encode_helper_environment(&self) -> io::Result<String> {
        self.validate()?;
        let encoded = serde_json::to_string(self)
            .map_err(|_| io::Error::other("Cannot encode bounded Build invocation"))?;
        if encoded.len() > MAX_TASK_BUILD_INVOCATION_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Build invocation exceeds its environment bound",
            ));
        }
        Ok(encoded)
    }

    pub fn matches_observed(&self, actual: &ObservedTaskBuildInvocation) -> bool {
        self.program == actual.invocation.program
            && self.arguments == actual.invocation.arguments
            && self.cwd == actual.invocation.cwd
            && self
                .environment
                .iter()
                .all(|(key, value)| actual.invocation.environment.get(key) == Some(value))
    }
}

fn deserialize_build_arguments<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct Arguments;
    impl<'de> serde::de::Visitor<'de> for Arguments {
        type Value = Vec<String>;
        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a bounded Build argument sequence")
        }
        fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            use serde::de::Error as _;
            let mut arguments = Vec::new();
            let mut bytes = 0_usize;
            while let Some(argument) = sequence.next_element::<String>()? {
                bytes = bytes
                    .checked_add(argument.len())
                    .ok_or_else(|| A::Error::custom("Build argument count overflow"))?;
                if arguments.len() >= 512
                    || argument.len() > 4096
                    || bytes > 32 * 1024
                    || argument.contains('\0')
                {
                    return Err(A::Error::custom("Build arguments exceed their bounds"));
                }
                arguments.push(argument);
            }
            Ok(arguments)
        }
    }
    deserializer.deserialize_seq(Arguments)
}

fn deserialize_build_environment<'de, D>(
    deserializer: D,
) -> Result<std::collections::BTreeMap<String, String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct Environment;
    impl<'de> serde::de::Visitor<'de> for Environment {
        type Value = std::collections::BTreeMap<String, String>;
        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a bounded Build environment map")
        }
        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::MapAccess<'de>,
        {
            use serde::de::Error as _;
            let mut environment = std::collections::BTreeMap::new();
            let mut bytes = 0_usize;
            while let Some((key, value)) = map.next_entry::<String, String>()? {
                bytes = bytes
                    .checked_add(key.len() + value.len())
                    .ok_or_else(|| A::Error::custom("Build environment count overflow"))?;
                if environment.len() >= 512
                    || key.is_empty()
                    || key.len() > 4096
                    || key.contains(['=', '\0'])
                    || value.len() > 4096
                    || value.contains('\0')
                    || bytes > 64 * 1024
                    || environment.contains_key(&key)
                {
                    return Err(A::Error::custom("Build environment exceeds its bounds"));
                }
                environment.insert(key, value);
            }
            Ok(environment)
        }
    }
    deserializer.deserialize_map(Environment)
}

#[derive(Clone)]
pub struct ObservedTaskBuildInvocation {
    pub(crate) invocation: std::sync::Arc<TaskBuildInvocation>,
}

impl ObservedTaskBuildInvocation {
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

#[cfg(any(target_os = "macos", test))]
pub(crate) fn validate_stdout_report_limit(limit: usize) -> io::Result<()> {
    if limit == 0 || limit > MAX_TASK_STDOUT_REPORT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Invalid Task stdout report limit",
        ));
    }
    Ok(())
}

#[cfg(any(target_os = "macos", test))]
pub(crate) async fn read_stdout_report(
    mut connection: impl AsyncRead + Unpin,
    limit: usize,
) -> io::Result<Vec<u8>> {
    validate_stdout_report_limit(limit)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(limit)
        .map_err(|_| io::Error::other("Cannot allocate the bounded Task stdout report"))?;
    loop {
        let mut header = [0; 4];
        connection.read_exact(&mut header).await?;
        let count = u32::from_be_bytes(header) as usize;
        if count == 0 {
            let mut trailing = [0];
            if connection.read(&mut trailing).await? != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Trailing Task stdout frame",
                ));
            }
            return Ok(output);
        }
        if count > TASK_STDOUT_FRAME_BYTES || count > limit.saturating_sub(output.len()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Task stdout report exceeds its limit",
            ));
        }
        let start = output.len();
        output.resize(start + count, 0);
        connection.read_exact(&mut output[start..]).await?;
    }
}

#[cfg(any(target_os = "macos", test))]
pub(crate) async fn capture_owned_task_stdout<C>(
    mut connection: C,
    command: std::process::Command,
    limit: usize,
) -> io::Result<ExitStatus>
where
    C: AsyncRead + AsyncWrite + Unpin + Clone,
{
    validate_stdout_report_limit(limit)?;
    let mut command = smol::process::Command::from(command);
    command
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit())
        .kill_on_drop(true);
    let mut child = command.spawn()?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("Task stdout pipe is unavailable"))?;
    let mut disconnection = connection.clone();
    let pump = async {
        let mut terminal_output =
            smol::Unblock::with_capacity(TASK_STDOUT_FRAME_BYTES, std::io::stdout());
        let mut total = 0_usize;
        let mut buffer = [0; TASK_STDOUT_FRAME_BYTES];
        loop {
            let remaining = limit
                .saturating_sub(total)
                .saturating_add(1)
                .min(buffer.len());
            let count = stdout.read(&mut buffer[..remaining]).await?;
            if count == 0 {
                break;
            }
            total = total
                .checked_add(count)
                .ok_or_else(|| io::Error::other("Task stdout report size overflow"))?;
            if total > limit {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Task stdout report exceeds its limit",
                ));
            }
            connection.write_all(&(count as u32).to_be_bytes()).await?;
            connection.write_all(&buffer[..count]).await?;
            terminal_output.write_all(&buffer[..count]).await?;
            terminal_output.flush().await?;
        }
        let status = child.status().await?;
        connection.write_all(&0_u32.to_be_bytes()).await?;
        connection.flush().await?;
        Ok(status)
    };
    future::race(pump, async {
        let mut control = [0];
        match disconnection.read(&mut control).await? {
            0 => Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "Task stdout observation disconnected",
            )),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Unexpected Task stdout control frame",
            )),
        }
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        pin::Pin,
        sync::{Arc, Mutex},
        task::{Context, Poll},
    };

    fn invocation() -> TaskBuildInvocation {
        TaskBuildInvocation::new(
            std::env::current_exe().expect("actual test image"),
            vec![
                String::new(),
                "space value".into(),
                "$(literal); 'quoted'".into(),
            ],
            std::env::current_dir().expect("actual test directory"),
            std::collections::BTreeMap::from([(
                "PRIVATE_TASK_VALUE".into(),
                "private value".into(),
            )]),
        )
        .expect("bounded invocation")
    }

    #[test]
    fn task_completion_invocation_preserves_empty_literal_arguments_and_private_values() {
        let expected = invocation();
        let encoded = expected
            .encode_helper_environment()
            .expect("bounded encoding");
        let decoded: TaskBuildInvocation = serde_json::from_str(&encoded).expect("exact decoding");
        decoded.validate().expect("valid decoded invocation");
        assert!(expected == decoded);
        let mut decoded = decoded;
        decoded.environment.insert(
            "OBSERVED_ONLY_VALUE".into(),
            "additional final value".into(),
        );
        let observed = ObservedTaskBuildInvocation {
            invocation: Arc::new(decoded),
        };
        assert!(expected.matches_observed(&observed));
        assert_eq!(
            observed.arguments(),
            ["", "space value", "$(literal); 'quoted'"]
        );
        assert_eq!(
            observed
                .environment()
                .get("PRIVATE_TASK_VALUE")
                .map(String::as_str),
            Some("private value")
        );
    }

    #[test]
    fn task_completion_invocation_rejects_reserved_oversized_duplicate_and_unknown_data() {
        for reserved in [
            TASK_COMPLETION_TOKEN_ENV,
            TASK_BUILD_INVOCATION_ENV,
            TASK_NATIVE_INVOCATION_ENV,
            TASK_CARGO_RUNTIME_ENDPOINT_ENV,
            TASK_CARGO_RUNTIME_TOKEN_ENV,
        ] {
            let mut value = invocation();
            value
                .environment
                .insert(reserved.into(), "hidden authority".into());
            assert_eq!(
                value
                    .validate()
                    .expect_err("authority is not workload environment")
                    .kind(),
                io::ErrorKind::InvalidInput
            );
        }
        let original = invocation();
        for arguments in [
            vec![String::new(); 513],
            vec!["x".repeat(4097)],
            vec!["x".repeat(4096); 9],
            vec!["nul\0argument".into()],
        ] {
            let mut value = original.clone();
            value.arguments = arguments;
            assert!(value.validate().is_err());
            assert!(
                serde_json::from_str::<TaskBuildInvocation>(
                    &serde_json::to_string(&value).expect("invalid fixture encoding")
                )
                .is_err()
            );
        }
        let encoded = original
            .encode_helper_environment()
            .expect("fixture encoding");
        let duplicate = encoded.replace(
            "\"PRIVATE_TASK_VALUE\":\"private value\"",
            "\"PRIVATE_TASK_VALUE\":\"private value\",\"PRIVATE_TASK_VALUE\":\"replacement\"",
        );
        assert_ne!(encoded, duplicate);
        assert!(serde_json::from_str::<TaskBuildInvocation>(&duplicate).is_err());
        let mut unknown: serde_json::Value =
            serde_json::from_str(&encoded).expect("fixture object");
        unknown["claimed_exit_code"] = 0.into();
        assert!(serde_json::from_value::<TaskBuildInvocation>(unknown).is_err());
    }

    #[test]
    fn task_completion_frames_preserve_bounded_payload_and_reject_invalid_termination() {
        smol::block_on(async {
            let valid = 3_u32
                .to_be_bytes()
                .into_iter()
                .chain(*b"abc")
                .chain(0_u32.to_be_bytes())
                .collect::<Vec<_>>();
            assert_eq!(
                read_stdout_report(futures_lite::io::Cursor::new(valid), 3)
                    .await
                    .expect("exact bounded payload"),
                b"abc"
            );
            for payload in [
                (TASK_STDOUT_FRAME_BYTES as u32 + 1).to_be_bytes().to_vec(),
                5_u32.to_be_bytes().into_iter().chain([b'x']).collect(),
                0_u32.to_be_bytes().into_iter().chain([b'x']).collect(),
                1_u32.to_be_bytes().into_iter().chain([b'x']).collect(),
                2_u32
                    .to_be_bytes()
                    .into_iter()
                    .chain(*b"ab")
                    .chain(2_u32.to_be_bytes())
                    .chain(*b"cd")
                    .chain(0_u32.to_be_bytes())
                    .collect(),
            ] {
                assert!(
                    read_stdout_report(futures_lite::io::Cursor::new(payload), 3)
                        .await
                        .is_err()
                );
            }
            for limit in [0, MAX_TASK_STDOUT_REPORT_BYTES + 1] {
                assert_eq!(
                    read_stdout_report(futures_lite::io::Cursor::new(Vec::<u8>::new()), limit)
                        .await
                        .expect_err("invalid bound")
                        .kind(),
                    io::ErrorKind::InvalidInput
                );
            }
        });
    }

    #[derive(Clone, Default)]
    struct ChildTestConnection(Arc<Mutex<Vec<u8>>>);

    impl AsyncRead for ChildTestConnection {
        fn poll_read(
            self: Pin<&mut Self>,
            _: &mut Context<'_>,
            _: &mut [u8],
        ) -> Poll<io::Result<usize>> {
            Poll::Pending
        }
    }

    impl AsyncWrite for ChildTestConnection {
        fn poll_write(
            self: Pin<&mut Self>,
            _: &mut Context<'_>,
            buffer: &[u8],
        ) -> Poll<io::Result<usize>> {
            self.0
                .lock()
                .expect("component output lock")
                .extend_from_slice(buffer);
            Poll::Ready(Ok(buffer.len()))
        }
        fn poll_flush(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
        fn poll_close(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    #[test]
    fn task_completion_owned_child_fixture() {
        let Ok(mode) = std::env::var("NATIVE_TASK_HELPER_COMMON_CHILD_TEST") else {
            return;
        };
        let arguments = std::env::args()
            .skip_while(|value| value != "--")
            .skip(1)
            .collect::<Vec<_>>();
        assert_eq!(arguments, ["", "space value", "$(literal); 'quoted'"]);
        assert_eq!(
            std::env::var("NATIVE_TASK_HELPER_LITERAL_VALUE").expect("literal environment"),
            "$(literal); 'quoted'"
        );
        assert_eq!(
            std::fs::canonicalize(std::env::current_dir().expect("actual child working directory"))
                .expect("canonical actual child working directory"),
            std::path::PathBuf::from(
                std::env::var_os("NATIVE_TASK_HELPER_EXPECTED_CWD")
                    .expect("selected working directory")
            ),
        );
        assert_eq!(
            std::env::var("NATIVE_TASK_HELPER_OBSERVED_ONLY_VALUE")
                .expect("additional final environment"),
            "additional final value"
        );
        println!("owned-child-completed");
        match mode.as_str() {
            "success" => std::process::exit(0),
            "failure" => std::process::exit(7),
            _ => panic!("unexpected finite child fixture mode"),
        }
    }

    #[test]
    fn task_completion_owned_child_waits_for_actual_exit_and_preserves_literal_arguments() {
        smol::block_on(async {
            for (mode, code) in [("success", 0), ("failure", 7)] {
                let connection = ChildTestConnection::default();
                let program = std::env::current_exe().expect("actual test image");
                let default_directory =
                    std::fs::canonicalize(std::env::current_dir().expect("default directory"))
                        .expect("canonical default directory");
                let mut selected_directory =
                    std::fs::canonicalize(program.parent().expect("test image directory"))
                        .expect("canonical image directory");
                if selected_directory == default_directory {
                    selected_directory = selected_directory
                        .parent()
                        .expect("non-root image directory")
                        .to_path_buf();
                }
                assert_ne!(
                    selected_directory, default_directory,
                    "the actual child must exercise a non-default selected cwd"
                );
                let expected = TaskBuildInvocation::new(
                    program,
                    [
                        "--exact",
                        "task_completion::tests::task_completion_owned_child_fixture",
                        "--nocapture",
                        "--",
                        "",
                        "space value",
                        "$(literal); 'quoted'",
                    ]
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
                    selected_directory.clone(),
                    std::collections::BTreeMap::from([
                        ("NATIVE_TASK_HELPER_COMMON_CHILD_TEST".into(), mode.into()),
                        (
                            "NATIVE_TASK_HELPER_LITERAL_VALUE".into(),
                            "$(literal); 'quoted'".into(),
                        ),
                        (
                            "NATIVE_TASK_HELPER_EXPECTED_CWD".into(),
                            selected_directory
                                .to_str()
                                .expect("selected cwd UTF8")
                                .into(),
                        ),
                    ]),
                )
                .expect("bounded selected child invocation");
                let mut command = std::process::Command::new(expected.program());
                command
                    .args(expected.arguments())
                    .current_dir(expected.working_directory())
                    .envs(expected.environment())
                    .env(
                        "NATIVE_TASK_HELPER_OBSERVED_ONLY_VALUE",
                        "additional final value",
                    );
                let status = capture_owned_task_stdout(connection.clone(), command, 4096)
                    .await
                    .expect("actual owned child completion");
                assert_eq!(status.code(), Some(code));
                let frames = connection
                    .0
                    .lock()
                    .expect("completed component output")
                    .clone();
                let output = read_stdout_report(futures_lite::io::Cursor::new(frames), 4096)
                    .await
                    .expect("bounded finished child stream");
                assert!(
                    output
                        .windows(b"owned-child-completed".len())
                        .any(|window| window == b"owned-child-completed")
                );
            }
        });
    }
}
