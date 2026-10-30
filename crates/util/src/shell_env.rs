use std::path::Path;

use anyhow::{Context as _, Result};
use collections::HashMap;
use serde::Deserialize;

use crate::shell::ShellKind;

fn parse_env_map_from_noisy_output(output: &str) -> Result<collections::HashMap<String, String>> {
    for (position, _) in output.match_indices('{') {
        let candidate = &output[position..];
        let mut deserializer = serde_json::Deserializer::from_str(candidate);
        if let Ok(env_map) = HashMap::<String, String>::deserialize(&mut deserializer) {
            return Ok(env_map);
        }
    }
    anyhow::bail!("Failed to find JSON in shell output: {output}")
}

pub fn print_env() {
    let env_vars: HashMap<String, String> = std::env::vars().collect();
    let json = serde_json::to_string_pretty(&env_vars).unwrap_or_else(|err| {
        eprintln!("Error serializing environment variables: {}", err);
        std::process::exit(1);
    });
    println!("{}", json);
}

/// Capture all environment variables from the login shell in the given directory.
pub async fn capture(
    shell_path: impl AsRef<Path>,
    args: &[String],
    directory: impl AsRef<Path>,
) -> Result<collections::HashMap<String, String>> {
    #[cfg(windows)]
    return capture_windows(shell_path.as_ref(), args, directory.as_ref()).await;
    #[cfg(unix)]
    return capture_unix(shell_path.as_ref(), args, directory.as_ref()).await;
}

/// Try to parse the environment output before checking the exit status.
/// The user's shell rc files may contain commands that fail (e.g. editor
/// integrations that call posix_spawnp outside a real PTY), causing a
/// non-zero exit status even though `zed --printenv` ran successfully and
/// produced valid output on its separate fd.
fn parse_env_output(
    env_output: &str,
    status: &std::process::ExitStatus,
    successful_capture_warning: impl FnOnce() -> String,
    failed_capture_error: impl FnOnce() -> String,
) -> Result<collections::HashMap<String, String>> {
    match parse_env_map_from_noisy_output(env_output) {
        Ok(env_map) => {
            if !status.success() {
                log::warn!("{}", successful_capture_warning());
            }
            Ok(env_map)
        }
        Err(parse_error) => {
            if !status.success() {
                anyhow::bail!(
                    "{}. Failed to deserialize environment variables from json: {parse_error}. output: {env_output}",
                    failed_capture_error(),
                );
            }

            anyhow::bail!(
                "Failed to deserialize environment variables from json: {parse_error}. output: {env_output}"
            );
        }
    }
}

#[cfg(unix)]
async fn capture_unix(
    shell_path: &Path,
    args: &[String],
    directory: &Path,
) -> Result<collections::HashMap<String, String>> {
    use std::os::unix::process::CommandExt;

    use crate::command::new_std_command;

    let shell_kind = ShellKind::new(shell_path, false);
    let quoted_zed_path = super::get_shell_safe_zed_path(shell_kind)?;

    let mut command_string = String::new();
    let mut command = new_std_command(shell_path);
    command.args(args);
    // In some shells, file descriptors greater than 2 cannot be used in interactive mode,
    // so file descriptor 0 (stdin) is used instead. This impacts zsh, old bash; perhaps others.
    // See: https://github.com/imagine-engineering/made/pull/32136#issuecomment-2999645482
    const FD_STDIN: std::os::fd::RawFd = 0;
    const FD_STDOUT: std::os::fd::RawFd = 1;
    const FD_STDERR: std::os::fd::RawFd = 2;

    let (fd_num, redir) = match shell_kind {
        ShellKind::Rc => (FD_STDIN, format!(">[1={}]", FD_STDIN)), // `[1=0]`
        ShellKind::Nushell | ShellKind::Tcsh => (FD_STDOUT, "".to_string()),
        // xonsh doesn't support redirecting to stdin, and control sequences are printed to
        // stdout on startup
        ShellKind::Xonsh => (FD_STDERR, "o>e".to_string()),
        ShellKind::PowerShell => (FD_STDIN, format!(">{}", FD_STDIN)),
        _ => (FD_STDIN, format!(">&{}", FD_STDIN)), // `>&0`
    };

    match shell_kind {
        ShellKind::Csh | ShellKind::Tcsh => {
            // For csh/tcsh, login shell requires passing `-` as 0th argument (instead of `-l`)
            command.arg0("-");
        }
        ShellKind::Fish => {
            // in fish, asdf, direnv attach to the `fish_prompt` event
            command_string.push_str("emit fish_prompt;");
            command.arg("-l");
        }
        _ => {
            command.arg("-l");
        }
    }

    match shell_kind {
        // Nushell does not allow non-interactive login shells.
        // Instead of doing "-l -i -c '<command>'"
        // use "-l -e '<command>; exit'" instead
        ShellKind::Nushell => command.arg("-e"),
        _ => command.args(["-i", "-c"]),
    };

    // Prefix with "./" if the path starts with "-" to prevent cd from interpreting it as a flag
    let dir_str = directory.to_string_lossy();
    let dir_str = if dir_str.starts_with('-') {
        format!("./{dir_str}").into()
    } else {
        dir_str
    };
    let quoted_dir = shell_kind
        .try_quote(&dir_str)
        .context("unexpected null in directory name")?;

    // cd into the directory, triggering directory specific side-effects (asdf, direnv, etc)
    command_string.push_str(&format!("cd {};", quoted_dir));
    if let Some(prefix) = shell_kind.command_prefix() {
        command_string.push(prefix);
    }
    command_string.push_str(&format!("{} --printenv {}", quoted_zed_path, redir));

    if let ShellKind::Nushell = shell_kind {
        command_string.push_str("; exit");
    }

    command.arg(&command_string);

    super::set_pre_exec_to_start_new_session(&mut command);

    let (env_output, process_output) = spawn_and_read_fd(command, fd_num).await?;
    let env_output = String::from_utf8_lossy(&env_output);

    parse_env_output(
        &env_output,
        &process_output.status,
        || {
            format!(
                "login shell exited with {} but environment was captured successfully. stderr: {:?}",
                process_output.status,
                String::from_utf8_lossy(&process_output.stderr),
            )
        },
        || {
            format!(
                "login shell exited with {}. stdout: {:?}, stderr: {:?}",
                process_output.status,
                String::from_utf8_lossy(&process_output.stdout),
                String::from_utf8_lossy(&process_output.stderr),
            )
        },
    )
}

#[cfg(unix)]
async fn spawn_and_read_fd(
    mut command: std::process::Command,
    child_fd: std::os::fd::RawFd,
) -> anyhow::Result<(Vec<u8>, std::process::Output)> {
    use command_fds::{CommandFdExt, FdMapping};
    use std::{io::Read, process::Stdio};

    let (mut reader, writer) = std::io::pipe()?;

    command.fd_mappings(vec![FdMapping {
        parent_fd: writer.into(),
        child_fd,
    }])?;

    let process = smol::process::Command::from(command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer)?;

    Ok((buffer, process.output().await?))
}

#[cfg(windows)]
async fn capture_windows(
    shell_path: &Path,
    args: &[String],
    directory: &Path,
) -> Result<collections::HashMap<String, String>> {
    use std::process::Stdio;

    let zed_path =
        std::env::current_exe().context("Failed to determine current zed executable path.")?;

    let shell_kind = ShellKind::new(shell_path, true);
    // Prefix with "./" if the path starts with "-" to prevent cd from interpreting it as a flag
    let directory_string = directory.display().to_string();
    let directory_string = if directory_string.starts_with('-') {
        format!("./{directory_string}")
    } else {
        directory_string
    };
    let zed_path_string = zed_path.display().to_string();
    let quote_for_shell = |value: &str| {
        shell_kind
            .try_quote(value)
            .map(|quoted| quoted.into_owned())
            .context("unexpected null in directory name")
    };
    let mut cmd = crate::command::new_command(shell_path);
    cmd.args(args);
    let quoted_directory = quote_for_shell(&directory_string)?;
    let quoted_zed_path = quote_for_shell(&zed_path_string)?;
    let cmd = match shell_kind {
        ShellKind::Csh
        | ShellKind::Tcsh
        | ShellKind::Rc
        | ShellKind::Fish
        | ShellKind::Xonsh
        | ShellKind::Posix => cmd.args([
            "-l",
            "-i",
            "-c",
            &format!("cd {}; {} --printenv", quoted_directory, quoted_zed_path),
        ]),
        ShellKind::PowerShell | ShellKind::Pwsh => cmd.args([
            "-NonInteractive",
            "-NoProfile",
            "-Command",
            &format!(
                "Set-Location {}; & {} --printenv",
                quoted_directory, quoted_zed_path
            ),
        ]),
        ShellKind::Elvish => cmd.args([
            "-c",
            &format!("cd {}; {} --printenv", quoted_directory, quoted_zed_path),
        ]),
        ShellKind::Nushell => {
            let zed_command = shell_kind
                .prepend_command_prefix(&quoted_zed_path)
                .into_owned();
            cmd.args([
                "-c",
                &format!("cd {}; {} --printenv", quoted_directory, zed_command),
            ])
        }
        ShellKind::Cmd => {
            let dir = directory_string.trim_end_matches('\\');
            cmd.args(["/d", "/c", "cd", dir, "&&", &zed_path_string, "--printenv"])
        }
    }
    .stdin(Stdio::null())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());
    let output = cmd
        .output()
        .await
        .with_context(|| format!("command {cmd:?}"))?;
    let env_output = String::from_utf8_lossy(&output.stdout);

    parse_env_output(
        &env_output,
        &output.status,
        || {
            format!(
                "Command {cmd:?} exited with {} but environment was captured successfully. stderr: {:?}",
                output.status,
                String::from_utf8_lossy(&output.stderr),
            )
        },
        || {
            format!(
                "Command {cmd:?} failed with {}. stdout: {:?}, stderr: {:?}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            )
        },
    )
}

#[cfg(any(windows, test))]
const TASK_ENVIRONMENT_START: &str = "__ZED_TASK_ENVIRONMENT_START__";
#[cfg(any(windows, test))]
const TASK_ENVIRONMENT_END: &str = "__ZED_TASK_ENVIRONMENT_END__";

#[cfg(any(windows, test))]
fn task_environment_command(
    shell: &crate::shell::Shell,
    activation_script: &[String],
    executable: &Path,
) -> Result<(ShellKind, String, Vec<String>)> {
    let builder = crate::shell_builder::ShellBuilder::new(shell, true);
    let kind = builder.kind();
    let executable = executable
        .to_str()
        .context("Task environment probe path cannot be represented")?;
    anyhow::ensure!(
        !executable.contains('\0'),
        "Task environment probe path cannot be represented"
    );
    let executable = if kind == ShellKind::Cmd {
        anyhow::ensure!(
            !executable.contains(['%', '!', '"', '\r', '\n']),
            "Task environment probe path is unsupported by the configured shell"
        );
        // Cmd command-position quotes must remain delimiters, unlike CRT argument quotes.
        format!("\"{executable}\"")
    } else {
        let quoted = kind
            .try_quote(executable)
            .context("Task environment probe path cannot be represented")?;
        kind.prepend_command_prefix(&quoted).into_owned()
    };
    let print = if kind == ShellKind::Nushell {
        "print"
    } else {
        "echo"
    };
    let separator = format!("{} ", kind.sequential_commands_separator());
    let mut statements = activation_script.to_vec();
    if matches!(kind, ShellKind::PowerShell | ShellKind::Pwsh) {
        // Windows PowerShell must decode the native probe's UTF-8 before forwarding it.
        statements
            .push("[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)".to_owned());
    }
    statements.extend([
        format!("{print} {TASK_ENVIRONMENT_START}"),
        format!("{executable} --printenv"),
        format!("{print} {TASK_ENVIRONMENT_END}"),
    ]);
    let (program, arguments) =
        builder.build_with_command_intent(Some(statements.join(&separator)), &[], false, 0);
    Ok((kind, program, arguments))
}

#[cfg(any(windows, test))]
fn normalize_task_environment(
    environment: HashMap<String, String>,
) -> Result<HashMap<String, String>> {
    task_environment_layers(&[&environment])
}

#[cfg(windows)]
pub fn merge_task_environment_layers(
    layers: &[&HashMap<String, String>],
) -> Result<HashMap<String, String>> {
    task_environment_layers(layers)
}

#[cfg(any(windows, test))]
fn task_environment_layers(layers: &[&HashMap<String, String>]) -> Result<HashMap<String, String>> {
    // Command uses the platform's environment-key comparison, including Windows casing.
    let mut command = std::process::Command::new("zed-task-environment");
    command.env_clear();
    for layer in layers {
        let mut entries = layer.iter().collect::<Vec<_>>();
        entries.sort_by(|left, right| left.0.cmp(right.0));
        for (name, value) in entries {
            anyhow::ensure!(
                !name.is_empty()
                    && !name.strip_prefix('=').unwrap_or(name).contains('=')
                    && !name.contains('\0')
                    && !value.contains('\0'),
                "Task environment contains invalid entries"
            );
            command.env(name, value);
        }
    }
    command
        .get_envs()
        .map(|(name, value)| {
            Ok((
                name.to_str()
                    .context("Task environment names cannot be represented")?
                    .to_owned(),
                value
                    .and_then(std::ffi::OsStr::to_str)
                    .context("Task environment values cannot be represented")?
                    .to_owned(),
            ))
        })
        .collect()
}

#[cfg(any(windows, test))]
fn parse_task_environment_output(output: &[u8]) -> Result<HashMap<String, String>> {
    let mut capturing = false;
    let mut current = Vec::new();
    let mut completed = None;
    // Shell startup output can use a legacy encoding; the probe's JSON must be UTF-8.
    for line in output.split_inclusive(|byte| *byte == b'\n') {
        let line = line
            .strip_suffix(b"\n")
            .map_or(line, |line| line.strip_suffix(b"\r").unwrap_or(line));
        if line == TASK_ENVIRONMENT_START.as_bytes() {
            capturing = true;
            current.clear();
        } else if capturing && line == TASK_ENVIRONMENT_END.as_bytes() {
            capturing = false;
            completed = Some(std::mem::take(&mut current));
        } else if capturing {
            current.extend_from_slice(line);
            current.push(b'\n');
        }
    }
    anyhow::ensure!(!capturing, "Task environment output framing is incomplete");
    let captured = completed.context("Task environment output framing is incomplete")?;
    let environment = serde_json::from_slice::<HashMap<String, String>>(&captured)
        .map_err(|_| anyhow::anyhow!("Task environment output is invalid"))?;
    normalize_task_environment(environment)
}

#[cfg(any(windows, test))]
async fn read_task_environment_stream(
    mut stream: impl futures_lite::AsyncRead + Unpin,
    limit: usize,
) -> Result<Vec<u8>> {
    use futures_lite::AsyncReadExt as _;
    let mut output = Vec::new();
    let mut scratch = [0; 8192];
    loop {
        let count = stream
            .read(&mut scratch)
            .await
            .map_err(|_| anyhow::anyhow!("Task environment output could not be read"))?;
        if count == 0 {
            return Ok(output);
        }
        anyhow::ensure!(
            count <= limit.saturating_sub(output.len()),
            "Task environment output exceeded its limit"
        );
        output.extend_from_slice(&scratch[..count]);
    }
}

#[cfg(windows)]
struct TaskEnvironmentChild(crate::process::Child);

#[cfg(windows)]
impl Drop for TaskEnvironmentChild {
    fn drop(&mut self) {
        match self.0.try_status() {
            Ok(Some(_)) => {}
            Ok(None) | Err(_) => {
                if self.0.kill().is_err() {
                    log::error!("Task environment shell cleanup could not be completed");
                }
            }
        }
    }
}

#[cfg(windows)]
pub async fn capture_task_environment(
    shell: &crate::shell::Shell,
    activation_script: &[String],
    directory: &Path,
    overrides: &HashMap<String, String>,
) -> Result<HashMap<String, String>> {
    let executable = std::env::current_exe()
        .map_err(|_| anyhow::anyhow!("Task environment probe executable is unavailable"))?;
    capture_task_environment_with_executable(
        shell,
        activation_script,
        directory,
        overrides,
        &executable,
    )
    .await
}

#[cfg(windows)]
async fn capture_task_environment_with_executable(
    shell: &crate::shell::Shell,
    activation_script: &[String],
    directory: &Path,
    overrides: &HashMap<String, String>,
    executable: &Path,
) -> Result<HashMap<String, String>> {
    use futures_lite::future::{race, try_zip};
    use std::{os::windows::process::CommandExt as _, process::Stdio, time::Duration};

    let (kind, program, arguments) =
        task_environment_command(shell, activation_script, executable)?;
    let mut command = crate::command::new_std_command(program);
    if kind == ShellKind::Cmd {
        for argument in arguments {
            command.raw_arg(argument);
        }
    } else {
        command.args(arguments);
    }
    command
        .current_dir(directory)
        .envs(normalize_task_environment(overrides.clone())?);
    let child =
        crate::process::Child::spawn(command, Stdio::null(), Stdio::piped(), Stdio::piped())
            .map_err(|_| anyhow::anyhow!("Task environment shell could not be started"))?;
    let mut child = TaskEnvironmentChild(child);
    let output = {
        let collect =
            async {
                let stdout = child
                    .0
                    .stdout
                    .take()
                    .context("Task environment stdout is unavailable")?;
                let stderr = child
                    .0
                    .stderr
                    .take()
                    .context("Task environment stderr is unavailable")?;
                try_zip(
                    try_zip(
                        read_task_environment_stream(stdout, 4 * 1024 * 1024),
                        read_task_environment_stream(stderr, 256 * 1024),
                    ),
                    async {
                        child.0.status().await.map_err(|_| {
                            anyhow::anyhow!("Task environment shell could not be reaped")
                        })
                    },
                )
                .await
            };
        race(collect, async {
            smol::Timer::after(Duration::from_secs(30)).await;
            Err(anyhow::anyhow!("Task environment capture timed out"))
        })
        .await
    };
    match output {
        Ok(((stdout, _stderr), _status)) => parse_task_environment_output(&stdout),
        Err(error) => {
            let kill_failed = child.0.kill().is_err();
            let reaped =
                race(
                    async {
                        child.0.status().await.map_err(|_| {
                            anyhow::anyhow!("Task environment shell could not be reaped")
                        })
                    },
                    async {
                        smol::Timer::after(Duration::from_secs(5)).await;
                        Err(anyhow::anyhow!(
                            "Task environment shell retirement timed out"
                        ))
                    },
                )
                .await;
            match reaped {
                Ok(_) => Err(error),
                Err(_) if kill_failed => {
                    anyhow::bail!("Task environment shell could not be stopped or reaped")
                }
                Err(_) => anyhow::bail!("Task environment shell retirement could not be verified"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::process::ExitStatus;

    use super::*;
    use crate::path;

    #[test]
    fn task_environment_powershell_preserves_startup_and_activation() {
        for program in ["powershell.exe", "pwsh.exe"] {
            let shell = crate::shell::Shell::WithArguments {
                program: program.into(),
                args: vec!["-NoLogo".into()],
                title_override: None,
            };
            let activation = vec!["$env:VIRTUAL_ENV = 'C:\\Python env'".into()];
            let (_, actual_program, arguments) = task_environment_command(
                &shell,
                &activation,
                Path::new("C:\\Editor files\\zed.exe"),
            )
            .expect("supported probe path");
            assert_eq!(actual_program, program);
            assert_eq!(
                arguments,
                [
                    "-NoLogo".to_owned(),
                    "-C".to_owned(),
                    format!(
                        "$env:VIRTUAL_ENV = 'C:\\Python env'; [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); echo {TASK_ENVIRONMENT_START}; &'C:\\Editor files\\zed.exe' --printenv; echo {TASK_ENVIRONMENT_END}"
                    ),
                ],
            );
        }
    }

    #[test]
    fn task_environment_cmd_preserves_startup_and_command_position_quotes() {
        let shell = crate::shell::Shell::WithArguments {
            program: "cmd.exe".into(),
            args: vec!["/Q".into()],
            title_override: None,
        };
        let (kind, program, arguments) = task_environment_command(
            &shell,
            &["call \"C:\\Python env\\activate.bat\"".into()],
            Path::new("C:\\Editor & files (dev)^\\zed.exe"),
        )
        .expect("quoted command-position path");
        assert_eq!(kind, ShellKind::Cmd);
        assert_eq!(program, "cmd.exe");
        assert_eq!(
            arguments,
            [
                "/Q".to_owned(),
                "/S".to_owned(),
                "/C".to_owned(),
                format!(
                    "\"call \"C:\\Python env\\activate.bat\"& echo {TASK_ENVIRONMENT_START}& \"C:\\Editor & files (dev)^\\zed.exe\" --printenv& echo {TASK_ENVIRONMENT_END}\""
                ),
            ],
        );
    }

    #[test]
    fn task_environment_cmd_refuses_unrepresentable_probe_paths_privately() {
        let shell = crate::shell::Shell::Program("cmd.exe".into());
        for executable in [
            "C:\\private%name\\zed.exe",
            "C:\\private!name\\zed.exe",
            "C:\\private\"name\\zed.exe",
            "C:\\private\rname\\zed.exe",
            "C:\\private\nname\\zed.exe",
            "C:\\private\0name\\zed.exe",
        ] {
            let error = task_environment_command(&shell, &[], Path::new(executable))
                .expect_err("unsupported path must not be evaluated");
            assert!(!error.to_string().contains("private"));
            assert!(!error.to_string().contains(executable));
        }
    }

    #[test]
    fn task_environment_other_shells_use_existing_task_startup() {
        for (program, print, invocation) in [
            ("bash", "echo", "'/editor files/zed.exe'"),
            ("nu.exe", "print", "^'/editor files/zed.exe'"),
        ] {
            let shell = crate::shell::Shell::WithArguments {
                program: program.into(),
                args: vec!["--configured".into()],
                title_override: None,
            };
            let (_, actual_program, arguments) = task_environment_command(
                &shell,
                &["trusted activation".into()],
                Path::new("/editor files/zed.exe"),
            )
            .expect("supported probe path");
            assert_eq!(actual_program, program);
            assert_eq!(
                arguments,
                [
                    "--configured".to_owned(),
                    "-i".to_owned(),
                    "-c".to_owned(),
                    format!(
                        "trusted activation; {print} {TASK_ENVIRONMENT_START}; {invocation} --printenv; {print} {TASK_ENVIRONMENT_END}"
                    ),
                ],
            );
        }
    }

    #[test]
    fn task_environment_parser_selects_framed_map_and_preserves_values() {
        let expected = HashMap::from_iter([
            ("UNICODE".into(), "日本語 🦀".into()),
            ("EMPTY".into(), String::new()),
            ("MULTILINE".into(), "first\nsecond".into()),
            ("=C:".into(), "C:\\active directory".into()),
        ]);
        let json = serde_json::to_string_pretty(&expected).expect("environment JSON");
        let output = format!(
            "{{\"UNRELATED\":\"startup\"}}\r\n{TASK_ENVIRONMENT_START}\r\n{json}\r\n{TASK_ENVIRONMENT_END}\r\nshutdown noise\r\n"
        );
        assert_eq!(
            parse_task_environment_output(output.as_bytes()).expect("framed environment"),
            expected
        );
        let mut noisy_output = b"\x82 shell startup\r\n".to_vec();
        noisy_output.extend_from_slice(output.as_bytes());
        noisy_output.extend_from_slice(b"\xff shell shutdown\r\n");
        assert_eq!(
            parse_task_environment_output(&noisy_output)
                .expect("legacy-encoded noise outside the UTF-8 frame"),
            expected
        );
    }

    #[test]
    fn task_environment_parser_refuses_missing_incomplete_or_invalid_frames_privately() {
        for output in [
            "{\"PRIVATE\":\"hidden\"}".to_owned(),
            format!("{TASK_ENVIRONMENT_START}\n{{\"PRIVATE\":\"hidden\"}}"),
            format!("{TASK_ENVIRONMENT_START}\n[\"hidden\"]\n{TASK_ENVIRONMENT_END}"),
            format!("{TASK_ENVIRONMENT_START}\n{{\"PRIVATE\":7}}\n{TASK_ENVIRONMENT_END}"),
            format!(
                "{TASK_ENVIRONMENT_START}\n{{}}\n{TASK_ENVIRONMENT_END}\n{TASK_ENVIRONMENT_START}\nhidden"
            ),
        ] {
            let error = parse_task_environment_output(output.as_bytes())
                .expect_err("invalid framing or environment");
            assert!(!error.to_string().contains("hidden"));
            assert!(!error.to_string().contains("PRIVATE"));
        }
        assert!(parse_task_environment_output(&[0xff]).is_err());
        let mut invalid_payload = format!("{TASK_ENVIRONMENT_START}\n").into_bytes();
        invalid_payload.extend_from_slice(b"{\"PRIVATE\":\"\xffhidden\"}\n");
        invalid_payload.extend_from_slice(format!("{TASK_ENVIRONMENT_END}\n").as_bytes());
        let error = parse_task_environment_output(&invalid_payload)
            .expect_err("invalid UTF-8 inside the environment frame");
        assert!(!error.to_string().contains("PRIVATE"));
        assert!(!error.to_string().contains("hidden"));
    }

    #[test]
    fn task_environment_layers_preserve_order_and_reject_invalid_entries_privately() {
        let initial = HashMap::from_iter([
            ("PATH".into(), "directory".into()),
            ("UNCHANGED".into(), "unchanged".into()),
        ]);
        let settings = HashMap::from_iter([("PATH".into(), "settings".into())]);
        let task = HashMap::from_iter([("PATH".into(), String::new())]);
        assert_eq!(
            task_environment_layers(&[&initial, &settings, &task])
                .expect("valid ordered environment"),
            HashMap::from_iter([
                ("PATH".into(), String::new()),
                ("UNCHANGED".into(), "unchanged".into()),
            ])
        );
        for (name, value) in [
            ("", "hidden"),
            ("PRIVATE=KEY", "hidden"),
            ("==PRIVATE", "hidden"),
            ("PRIVATE\0KEY", "hidden"),
            ("PRIVATE", "hidden\0value"),
        ] {
            let environment = HashMap::from_iter([(name.into(), value.into())]);
            let error = task_environment_layers(&[&environment])
                .expect_err("invalid environment must be rejected");
            assert!(!error.to_string().contains("hidden"));
            assert!(!error.to_string().contains("PRIVATE"));
        }
    }

    #[cfg(windows)]
    #[test]
    fn task_environment_windows_layers_respect_case_insensitive_precedence() {
        let directory = HashMap::from_iter([("Path".into(), "directory".into())]);
        let settings = HashMap::from_iter([("path".into(), "settings".into())]);
        let task = HashMap::from_iter([("PATH".into(), "task".into())]);
        let environment = merge_task_environment_layers(&[&directory, &settings, &task])
            .expect("valid Windows aliases");
        assert_eq!(environment.len(), 1);
        assert_eq!(
            environment.values().next().map(String::as_str),
            Some("task")
        );

        let duplicates = HashMap::from_iter([
            ("Path".into(), "deterministic-last".into()),
            ("PATH".into(), "deterministic-first".into()),
        ]);
        let environment = normalize_task_environment(duplicates).expect("valid Windows aliases");
        assert_eq!(environment.len(), 1);
        assert_eq!(
            environment.values().next().map(String::as_str),
            Some("deterministic-last")
        );
    }

    #[cfg(windows)]
    #[test]
    fn task_environment_windows_capture_runs_configured_shell_and_activation() -> Result<()> {
        use anyhow::Context as _;
        use futures_lite::future::{race, try_zip};
        use std::{process::Stdio, time::Duration};

        smol::block_on(async {
            let temporary = tempfile::Builder::new()
                .prefix("zed-task-environment-")
                .tempdir()?;
            let directory = temporary.path().join("work 日本語");
            let executable_directory = temporary.path().join("probe files");
            std::fs::create_dir(&directory)?;
            std::fs::create_dir(&executable_directory)?;
            let source = temporary.path().join("probe.rs");
            let executable = executable_directory.join("probe.exe");
            std::fs::write(
                &source,
                r#"
use std::fmt::Write as _;

fn quote(output: &mut String, value: &str) -> std::fmt::Result {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{0}'..='\u{1f}' => write!(output, "\\u{:04x}", u32::from(character))?,
            _ => output.push(character),
        }
    }
    output.push('"');
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().skip(1).collect::<Vec<_>>() != ["--printenv"] {
        return Err("unexpected fixture arguments".into());
    }
    let mut environment = std::env::vars().collect::<std::collections::BTreeMap<_, _>>();
    environment.insert("ZED_TASK_ENVIRONMENT_FIXTURE_PROBED".into(), "yes".into());
    environment.insert(
        "ZED_TASK_ENVIRONMENT_FIXTURE_CWD".into(),
        std::env::current_dir()?.to_str().ok_or("fixture directory is not Unicode")?.into(),
    );
    let mut output = String::from("{");
    for (index, (name, value)) in environment.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        quote(&mut output, name)?;
        output.push(':');
        quote(&mut output, value)?;
    }
    output.push('}');
    println!("{output}");
    Ok(())
}
"#,
            )?;
            let compiler = which::which("rustc")
                .context("Windows Task environment fixture requires an installed Rust compiler")?;
            let mut command = crate::command::new_std_command(compiler);
            command
                .current_dir(temporary.path())
                .env("RUSTUP_AUTO_INSTALL", "0")
                .args([
                    "--edition=2024",
                    "--crate-name",
                    "zed_task_environment_fixture",
                ])
                .arg(&source)
                .arg("-o")
                .arg(&executable);
            let compiler = crate::process::Child::spawn(
                command,
                Stdio::null(),
                Stdio::piped(),
                Stdio::piped(),
            )
            .map_err(|_| {
                anyhow::anyhow!("Windows Task environment fixture compiler could not start")
            })?;
            let mut compiler = TaskEnvironmentChild(compiler);
            let compilation = {
                let collect = async {
                    let stdout = compiler
                        .0
                        .stdout
                        .take()
                        .context("Fixture compiler stdout is unavailable")?;
                    let stderr = compiler
                        .0
                        .stderr
                        .take()
                        .context("Fixture compiler stderr is unavailable")?;
                    try_zip(
                        try_zip(
                            read_task_environment_stream(stdout, 16 * 1024),
                            read_task_environment_stream(stderr, 64 * 1024),
                        ),
                        async {
                            compiler
                                .0
                                .status()
                                .await
                                .context("Fixture compiler could not be reaped")
                        },
                    )
                    .await
                };
                race(collect, async {
                    smol::Timer::after(Duration::from_secs(60)).await;
                    Err(anyhow::anyhow!("Fixture compiler timed out"))
                })
                .await
            };
            match compilation {
                Ok(((_stdout, _stderr), status)) => {
                    anyhow::ensure!(
                        status.success(),
                        "Windows Task environment fixture compilation failed (exit {:?})",
                        status.code()
                    );
                }
                Err(error) => {
                    let kill_failed = compiler.0.kill().is_err();
                    let cleanup = race(
                        async {
                            compiler
                                .0
                                .status()
                                .await
                                .context("Fixture compiler could not be reaped")
                        },
                        async {
                            smol::Timer::after(Duration::from_secs(5)).await;
                            Err(anyhow::anyhow!("Fixture compiler retirement timed out"))
                        },
                    )
                    .await;
                    anyhow::ensure!(
                        cleanup.is_ok(),
                        "Fixture compiler retirement was not confirmed (kill failed: {kill_failed})"
                    );
                    return Err(error);
                }
            }

            let overrides = HashMap::from_iter([
                ("ZED_TASK_ENVIRONMENT_FIXTURE_BASE".into(), "base".into()),
                ("ZED_TASK_ENVIRONMENT_FIXTURE_EMPTY".into(), String::new()),
                (
                    "ZED_TASK_ENVIRONMENT_FIXTURE_UNICODE".into(),
                    "日本語 🦀".into(),
                ),
                (
                    "ZED_TASK_ENVIRONMENT_FIXTURE_REMOVED".into(),
                    "remove me".into(),
                ),
            ]);
            for (program, arguments, activation, expected_startup) in [
                (
                    "cmd.exe",
                    vec!["/Q".into(), "/V:ON".into()],
                    vec![
                        "set \"ZED_TASK_ENVIRONMENT_FIXTURE_STARTUP=!ZED_TASK_ENVIRONMENT_FIXTURE_BASE!\"".into(),
                        "set \"ZED_TASK_ENVIRONMENT_FIXTURE_ACTIVATED=日本語 🦀\"".into(),
                        "set \"ZED_TASK_ENVIRONMENT_FIXTURE_REMOVED=\"".into(),
                    ],
                    "base",
                ),
                (
                    "powershell.exe",
                    vec!["-NoLogo".into(), "-Mta".into()],
                    vec![
                        "$env:ZED_TASK_ENVIRONMENT_FIXTURE_STARTUP = [System.Threading.Thread]::CurrentThread.GetApartmentState().ToString()".into(),
                        "$env:ZED_TASK_ENVIRONMENT_FIXTURE_ACTIVATED = '日本語 🦀'".into(),
                        "Remove-Item Env:ZED_TASK_ENVIRONMENT_FIXTURE_REMOVED".into(),
                    ],
                    "MTA",
                ),
            ] {
                let shell = crate::shell::Shell::WithArguments {
                    program: program.into(),
                    args: arguments,
                    title_override: None,
                };
                let environment = capture_task_environment_with_executable(
                    &shell,
                    &activation,
                    &directory,
                    &overrides,
                    &executable,
                )
                .await?;
                for (name, expected) in [
                    ("ZED_TASK_ENVIRONMENT_FIXTURE_PROBED", "yes"),
                    ("ZED_TASK_ENVIRONMENT_FIXTURE_BASE", "base"),
                    ("ZED_TASK_ENVIRONMENT_FIXTURE_EMPTY", ""),
                    ("ZED_TASK_ENVIRONMENT_FIXTURE_UNICODE", "日本語 🦀"),
                    ("ZED_TASK_ENVIRONMENT_FIXTURE_ACTIVATED", "日本語 🦀"),
                    ("ZED_TASK_ENVIRONMENT_FIXTURE_STARTUP", expected_startup),
                ] {
                    assert_eq!(environment.get(name).map(String::as_str), Some(expected));
                }
                assert!(!environment.contains_key("ZED_TASK_ENVIRONMENT_FIXTURE_REMOVED"));
                let captured_directory = environment
                    .get("ZED_TASK_ENVIRONMENT_FIXTURE_CWD")
                    .context("Owned fixture directory was not captured")?;
                assert_eq!(
                    std::fs::canonicalize(captured_directory)?,
                    std::fs::canonicalize(&directory)?,
                );
            }
            Ok(())
        })
    }

    #[test]
    fn task_environment_stream_enforces_exact_byte_limit() {
        smol::block_on(async {
            let output = read_task_environment_stream(futures_lite::io::Cursor::new(b"1234"), 4)
                .await
                .expect("exact limit");
            assert_eq!(output, b"1234");
            let error = read_task_environment_stream(futures_lite::io::Cursor::new(b"hidden"), 5)
                .await
                .expect_err("bounded overflow");
            assert!(!error.to_string().contains("hidden"));
        });
    }

    #[cfg(unix)]
    fn exit_status(code: i32) -> ExitStatus {
        use std::os::unix::process::ExitStatusExt;

        ExitStatus::from_raw(code << 8)
    }

    #[cfg(windows)]
    fn exit_status(code: u32) -> ExitStatus {
        use std::os::windows::process::ExitStatusExt;

        ExitStatus::from_raw(code)
    }

    #[test]
    fn parse_env_output_accepts_valid_env_when_shell_exits_nonzero() {
        let env_json = serde_json::json!({
            "PATH": path!("/usr/bin"),
            "SHELL": path!("/bin/zsh"),
        });
        let env_output = format!("shell startup noise\n{env_json}\nshell shutdown noise");

        let env_map = parse_env_output(
            &env_output,
            &exit_status(1),
            || "shell exited with 1 but environment was captured successfully".to_string(),
            || panic!("failed capture error should not be evaluated for valid environment output"),
        )
        .expect("valid environment output should be returned despite non-zero shell exit");
        assert_eq!(
            env_map.get("PATH").map(String::as_str),
            Some(path!("/usr/bin"))
        );
        assert_eq!(
            env_map.get("SHELL").map(String::as_str),
            Some(path!("/bin/zsh"))
        );
    }
}
