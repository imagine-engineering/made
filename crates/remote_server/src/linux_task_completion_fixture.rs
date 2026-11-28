use anyhow::{Context as _, Result, ensure};
use futures::io::AsyncReadExt as _;
use native_task_helper::{
    linux_task_completion::{
        CoverageTaskReportReservation, LinuxTaskCompletion, LinuxTaskMode,
        NativeProfileTaskReservation,
    },
    task_actor::OriginalTaskActor,
    task_completion::TaskBuildInvocation,
};
use std::{
    collections::BTreeMap,
    fs::File,
    io,
    os::unix::process::ExitStatusExt as _,
    path::Path,
    process::{ExitStatus, Stdio},
    time::Duration,
};

fn spawn_helper(completion: &LinuxTaskCompletion) -> io::Result<smol::process::Child> {
    spawn_completion_helper(completion, LinuxTaskMode::Coverage)
}

fn spawn_completion_helper(
    completion: &LinuxTaskCompletion,
    mode: LinuxTaskMode,
) -> io::Result<smol::process::Child> {
    smol::process::Command::new(std::env::current_exe()?)
        .arg(mode.command())
        .arg("--socket")
        .arg(completion.socket())
        .env("COPPER_UNADMITTED_TEST_VALUE", "must not be inherited")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
}

fn invocation(directory: &Path, shell: &str) -> io::Result<TaskBuildInvocation> {
    shell_invocation(directory, shell, None)
}

fn shell_invocation(
    directory: &Path,
    shell: &str,
    report: Option<&Path>,
) -> io::Result<TaskBuildInvocation> {
    let mut arguments = vec![
        "-c".into(),
        shell.into(),
        "fixture".into(),
        String::new(),
        "two words".into(),
    ];
    if let Some(report) = report {
        arguments.push(
            report
                .to_str()
                .ok_or_else(|| io::Error::other("Report path is not Unicode"))?
                .into(),
        );
    }
    TaskBuildInvocation::new(
        "/bin/sh".into(),
        arguments,
        directory.to_path_buf(),
        BTreeMap::from([("COPPER_APPROVED_TEST_VALUE".into(), "literal value".into())]),
    )
}

#[expect(
    clippy::disallowed_methods,
    reason = "The fixture runs separate OS processes without a GPUI executor"
)]
fn fixture_delay(duration: Duration) -> smol::Timer {
    smol::Timer::after(duration)
}

async fn wait(helper: &mut smol::process::Child) -> io::Result<ExitStatus> {
    smol::future::race(helper.status(), async {
        fixture_delay(Duration::from_secs(5)).await;
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "Task helper did not exit",
        ))
    })
    .await
}

fn terminal_child_exit(directory: &Path) -> Result<()> {
    smol::block_on(async {
        let expected_image = File::open("/proc/self/exe")?;
        for (shell, expected_code, expected_signal) in [
            (
                r#"test -z "${COPPER_UNADMITTED_TEST_VALUE+x}" &&
                    test "$1" = "" && test "$2" = 'two words' &&
                    printf '%s' "$COPPER_APPROVED_TEST_VALUE" > report &&
                    printf '%s' "$COPPER_APPROVED_TEST_VALUE""#,
                Some(0),
                None,
            ),
            ("exit 27", Some(27), None),
            ("kill -TERM $$", None, Some(libc::SIGTERM)),
        ] {
            let completion = LinuxTaskCompletion::bind()?;
            let socket = completion.socket();
            let mut helper = spawn_helper(&completion)?;
            let actor = OriginalTaskActor::from_smol_child(&mut helper, &expected_image)?;
            let receipt = completion
                .observe(actor, invocation(directory, shell)?, None)
                .await?;
            let (actual, _) = receipt.verify_exit(wait(&mut helper).await?)?;
            let mut output = String::new();
            helper
                .stdout
                .take()
                .ok_or_else(|| io::Error::other("Missing helper output"))?
                .read_to_string(&mut output)
                .await?;
            ensure!(
                output
                    == if expected_code == Some(0) {
                        "literal value"
                    } else {
                        ""
                    },
                "Unexpected workload stdout"
            );
            ensure!(
                (actual.code(), actual.signal()) == (expected_code, expected_signal),
                "Unexpected workload exit"
            );
            ensure!(!socket.exists(), "Coverage socket was retained");
        }
        ensure!(
            std::fs::read(directory.join("report"))? == b"literal value",
            "Unexpected test report"
        );
        Ok(())
    })
}

fn observer_drop(directory: &Path) -> Result<()> {
    smol::block_on(async {
        let expected_image = File::open("/proc/self/exe")?;
        let completion = LinuxTaskCompletion::bind()?;
        let socket = completion.socket();
        let mut helper = spawn_helper(&completion)?;
        let actor = OriginalTaskActor::from_smol_child(&mut helper, &expected_image)?;
        let process_id_path = directory.join("workload");
        let process_id = smol::future::race(
            async {
                completion
                    .observe(
                        actor,
                        invocation(directory, "printf '%s' $$ > workload; exec /bin/sleep 60")?,
                        None,
                    )
                    .await?;
                Err(io::Error::other(
                    "Unexpected completion of the held workload",
                ))
            },
            async {
                for _ in 0..200 {
                    fixture_delay(Duration::from_millis(10)).await;
                    match std::fs::read_to_string(&process_id_path) {
                        Ok(process_id) if !process_id.is_empty() => return Ok(process_id),
                        Ok(_) => {}
                        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                        Err(error) => return Err(error),
                    }
                }
                Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "Workload never started",
                ))
            },
        )
        .await?;
        ensure!(
            !wait(&mut helper).await?.success(),
            "Invalid helper exited successfully"
        );
        ensure!(
            !Path::new("/proc").join(process_id).exists(),
            "Workload survived observer drop"
        );
        ensure!(!socket.exists(), "Coverage socket was retained");
        Ok(())
    })
}

fn foreign_child(directory: &Path) -> Result<()> {
    smol::block_on(async {
        let expected_image = File::open("/proc/self/exe")?;
        let unused = LinuxTaskCompletion::bind()?;
        let completion = LinuxTaskCompletion::bind()?;
        let mut original = spawn_helper(&unused)?;
        let actor = OriginalTaskActor::from_smol_child(&mut original, &expected_image)?;
        let mut other = spawn_helper(&completion)?;
        let result = completion
            .observe(actor, invocation(directory, "touch report")?, None)
            .await;
        ensure!(
            matches!(result, Err(error) if error.kind() == io::ErrorKind::PermissionDenied),
            "Foreign child was not refused"
        );
        ensure!(
            !wait(&mut other).await?.success(),
            "Foreign helper exited successfully"
        );
        ensure!(
            !directory.join("report").exists(),
            "Foreign helper started the workload"
        );
        original.kill()?;
        wait(&mut original).await?;
        Ok(())
    })
}

fn oversized_invocation(directory: &Path) -> Result<()> {
    use futures::io::AsyncWriteExt as _;
    use native_task_helper::task_completion::MAX_TASK_BUILD_INVOCATION_BYTES;

    smol::block_on(async {
        let listener = smol::net::unix::UnixListener::bind(directory.join("malformed"))?;
        let mut command = smol::process::Command::new(std::env::current_exe()?);
        let mut helper = command
            .arg(LinuxTaskMode::Coverage.command())
            .arg("--socket")
            .arg(
                listener
                    .local_addr()?
                    .as_pathname()
                    .ok_or_else(|| io::Error::other("Missing socket"))?,
            )
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()?;
        let (mut peer, _) = smol::future::race(listener.accept(), async {
            fixture_delay(Duration::from_secs(5)).await;
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Helper never connected",
            ))
        })
        .await?;
        peer.write_all(&[1]).await?;
        peer.write_all(&[0; 32]).await?;
        peer.write_all(&(MAX_TASK_BUILD_INVOCATION_BYTES as u32 + 1).to_be_bytes())
            .await?;
        ensure!(
            !wait(&mut helper).await?.success(),
            "Invalid helper exited successfully"
        );
        Ok(())
    })
}

fn terminal_handoff(
    headless: bool,
    invocation: TaskBuildInvocation,
    report: CoverageTaskReportReservation,
    verify_report: fn(&[u8]) -> Result<()>,
) -> Result<()> {
    use gpui::AppContext as _;
    use terminal::{HeadlessTerminal, TerminalBuilder};

    let (send, receive) = std::sync::mpsc::sync_channel(1);
    let report_path = report.path();
    gpui_platform::headless().run(move |cx| {
        release_channel::init(semver::Version::new(1, 16, 2), cx);
        cx.set_global(HeadlessTerminal(headless));
        let prepared = TerminalBuilder::new_with_coverage_task(
            task::SpawnInTerminal {
                label: "Coverage owner fixture".into(),
                ..Default::default()
            },
            None,
            invocation,
            report,
            1,
            cx,
        );
        cx.spawn(async move |cx| {
            let deadline = cx.background_executor().timer(Duration::from_secs(15));
            let observation = async {
                let (builder, observation) = prepared?;
                let builder = builder.await?;
                let terminal = cx.update(|cx| cx.new(|cx| builder.subscribe(cx)));
                let completion = terminal
                    .read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx));
                let (observed, ordinary_completion) = futures::future::join(
                    observation.finish(),
                    completion,
                )
                .await;
                let (observed, report) = observed?;
                ensure!(
                    observed.success() && ordinary_completion == Some(observed),
                    "The authenticated receipt and ordinary Terminal observer must share the same exit"
                );
                let report = report.context("Authenticated Task receipt has no sealed report")?;
                let bytes = cx.background_spawn(async move { report.read() }).await?;
                verify_report(&bytes)?;
                Ok(())
            };
            let result = match futures::future::select(Box::pin(observation), deadline).await {
                futures::future::Either::Left((result, _)) => result,
                futures::future::Either::Right(_) => Err(anyhow::anyhow!("Coverage Terminal fixture timed out")),
            };
            if let Err(error) = send.send(result) {
                log::error!("Coverage Terminal fixture lost its caller: {error}");
            }
            cx.update(|cx| cx.quit());
        })
        .detach();
    });
    receive.recv()??;
    ensure!(!report_path.exists(), "Coverage report was retained");
    Ok(())
}

fn shell_terminal(directory: &Path, headless: bool) -> Result<()> {
    let report = CoverageTaskReportReservation::new(1024)?;
    let invocation = shell_invocation(
        directory,
        r#"test "$1" = "" && test "$2" = 'two words' && printf '%s' "$COPPER_APPROVED_TEST_VALUE" > "$3""#,
        Some(&report.path()),
    )?;
    terminal_handoff(headless, invocation, report, |bytes| {
        ensure!(
            bytes == b"literal value",
            "Coverage Task report did not come from the approved invocation"
        );
        Ok(())
    })
}

fn report_replacement(directory: &Path) -> Result<()> {
    smol::block_on(async {
        let expected_image = File::open("/proc/self/exe")?;
        let shell = r#"printf 'valid' > "$3""#;
        for (command, attach_output, expected_error) in [
            (shell, false, Some("does not own its empty report")),
            ("true", true, Some("report is empty or exceeds its limit")),
            (
                r#"printf 'exceeds-limit' > "$3""#,
                true,
                Some("report is empty or exceeds its limit"),
            ),
            (
                r#"rm "$3" && printf 'valid' > "$3""#,
                true,
                Some("report was replaced"),
            ),
            (shell, true, None),
        ] {
            let report = CoverageTaskReportReservation::new(5)?;
            let path = report.path();
            let completion = LinuxTaskCompletion::bind()?;
            let mut helper = spawn_helper(&completion)?;
            let actor = OriginalTaskActor::from_smol_child(&mut helper, &expected_image)?;
            let invocation =
                shell_invocation(directory, command, attach_output.then_some(path.as_path()))?;
            let result = completion.observe(actor, invocation, Some(report)).await;
            let exit = wait(&mut helper).await?;
            if let Some(message) = expected_error {
                ensure!(!exit.success(), "An invalid report was acknowledged");
                let error = result
                    .err()
                    .context("Expected Coverage Task report refusal")?;
                ensure!(
                    error.to_string().contains(message),
                    "Unexpected report refusal: {error}"
                );
            } else {
                let (status, report) = result?.verify_exit(exit)?;
                ensure!(status.success(), "Expected a successful coverage workload");
                let report = report.context("Successful receipt is missing its report")?;
                std::fs::write(&path, b"other")?;
                let error = report.read().err().context("Changed report was accepted")?;
                ensure!(
                    error
                        .to_string()
                        .contains("Sealed Coverage Task report changed"),
                    "Unexpected sealed report refusal: {error}"
                );
            }
            ensure!(!path.exists(), "Invalid coverage report was retained");
        }
        Ok(())
    })
}

fn profile_pair_replacement(directory: &Path) -> Result<()> {
    enum ExpectedPair {
        Refused(&'static str),
        Unsuccessful,
        ChangedAfterSealing,
        Complete,
        WrongMode,
    }

    smol::block_on(async {
        let expected_image = File::open("/proc/self/exe")?;
        let write_pair = r#"printf 'one' > "$2" && printf 'two' > "${2%.json}.syms.json""#;
        for (command, output_flag, expected) in [
            (
                write_pair,
                "-i",
                ExpectedPair::Refused("does not own its empty recording pair"),
            ),
            (
                r#"printf 'one' > "$2""#,
                "-o",
                ExpectedPair::Refused("symbols is empty or exceeds"),
            ),
            (
                r#"printf 'one' > "$2" && rm "${2%.json}.syms.json" && printf 'two' > "${2%.json}.syms.json""#,
                "-o",
                ExpectedPair::Refused("symbols was replaced"),
            ),
            (
                r#"printf 'one' > "$2" && truncate -s 16777217 "${2%.json}.syms.json""#,
                "-o",
                ExpectedPair::Refused("symbols is empty or exceeds"),
            ),
            (
                r#"printf 'one' > "$2" && printf 'two' > "${2%.json}.syms.json" && exit 27"#,
                "-o",
                ExpectedPair::Unsuccessful,
            ),
            (write_pair, "-o", ExpectedPair::ChangedAfterSealing),
            (write_pair, "-o", ExpectedPair::Complete),
            (write_pair, "-o", ExpectedPair::WrongMode),
        ] {
            let report = NativeProfileTaskReservation::new()?;
            let profile_path = report.profile_path();
            let symbols_path = report.symbols_path();
            let invocation =
                profile_shell_invocation(directory, command, output_flag, &profile_path)?;
            let completion = LinuxTaskCompletion::bind()?;
            let helper_command = match expected {
                ExpectedPair::WrongMode => LinuxTaskMode::Coverage,
                _ => LinuxTaskMode::NativeProfile,
            };
            let mut helper = spawn_completion_helper(&completion, helper_command)?;
            let actor = OriginalTaskActor::from_smol_child(&mut helper, &expected_image)?;
            let observed = completion.observe_profile(actor, invocation, report).await;
            let status = wait(&mut helper).await?;
            if let ExpectedPair::Refused(expected_error) = expected {
                ensure!(!status.success(), "An invalid pair was acknowledged");
                let error = observed.err().context("Expected profile pair refusal")?;
                ensure!(
                    error.to_string().contains(expected_error),
                    "Unexpected pair refusal: {error}"
                );
            } else if matches!(expected, ExpectedPair::WrongMode) {
                ensure!(
                    observed.is_err(),
                    "A Coverage helper admitted a native profile"
                );
                ensure!(
                    !status.success(),
                    "A Coverage helper ran the native invocation"
                );
            } else {
                let (status, pair) = observed?.verify_exit(status)?;
                if matches!(expected, ExpectedPair::Unsuccessful) {
                    ensure!(
                        status.code() == Some(27),
                        "Unexpected profiling workload exit"
                    );
                    ensure!(pair.is_none(), "An unsuccessful profile was sealed");
                } else if matches!(expected, ExpectedPair::ChangedAfterSealing) {
                    ensure!(status.success(), "Expected a successful profiling workload");
                    let pair = pair.context("Successful receipt is missing its pair")?;
                    std::fs::write(&symbols_path, b"new")?;
                    let error = pair
                        .read()
                        .err()
                        .context("Changed profile pair was accepted")?;
                    ensure!(
                        error
                            .to_string()
                            .contains("Sealed Native Profile Task pair changed")
                    );
                } else {
                    ensure!(status.success(), "Expected a successful profiling workload");
                    let pair = pair
                        .context("Successful receipt is missing its pair")?
                        .read()?;
                    ensure!(pair.profile == b"one" && pair.symbols == b"two");
                }
            }
            ensure!(
                !profile_path.exists() && !symbols_path.exists(),
                "Profile pair was retained"
            );
        }
        Ok(())
    })
}

fn profile_shell_invocation(
    directory: &Path,
    script: &str,
    output_flag: &str,
    profile_path: &Path,
) -> Result<TaskBuildInvocation> {
    Ok(TaskBuildInvocation::new(
        "/bin/sh".into(),
        vec![
            "-c".into(),
            script.into(),
            "fixture".into(),
            output_flag.into(),
            profile_path
                .to_str()
                .context("Profile path is not Unicode")?
                .into(),
        ],
        directory.to_path_buf(),
        BTreeMap::from([("PATH".into(), "/usr/bin:/bin".into())]),
    )?)
}

fn profile_group_cancellation(directory: &Path) -> Result<()> {
    let report = NativeProfileTaskReservation::new()?;
    let invocation = profile_shell_invocation(
        directory,
        r#"printf 'partial' > "$2"
           printf 'symbols' > "${2%.json}.syms.json"
           sleep 60 &
           printf '%s %s\n' "$$" "$!" > profiled-processes
           wait"#,
        "-o",
        &report.profile_path(),
    )?;
    assert_profile_cancellation(directory, invocation, report, None)
}

fn native_profile_collector_cancellation(directory: &Path) -> Result<()> {
    let report = NativeProfileTaskReservation::new()?;
    let workload = std::env::var("RUST_TOOLS_LINUX_PROFILE_WORKLOAD")?;
    let invocation = samply_invocation(
        directory,
        &report.profile_path(),
        &[
            "/bin/bash",
            "-c",
            r#"printf '%s %s\n' "$PPID" "$$" > profiled-processes; exec "$1" 1200000"#,
            "fixture",
            &workload,
        ],
    )?;
    assert_profile_cancellation(directory, invocation, report, Some(Path::new(&workload)))
}

fn assert_profile_cancellation(
    directory: &Path,
    invocation: TaskBuildInvocation,
    report: NativeProfileTaskReservation,
    expected_workload: Option<&Path>,
) -> Result<()> {
    smol::block_on(async {
        let profile_path = report.profile_path();
        let symbols_path = report.symbols_path();
        let completion = LinuxTaskCompletion::bind()?;
        let socket = completion.socket();
        let mut helper = spawn_completion_helper(&completion, LinuxTaskMode::NativeProfile)?;
        let actor =
            OriginalTaskActor::from_smol_child(&mut helper, &File::open("/proc/self/exe")?)?;
        let (collector, workload) = smol::future::race(
            async {
                completion
                    .observe_profile(actor, invocation, report)
                    .await?;
                anyhow::bail!("The profile completed before cancellation");
            },
            async {
                for _ in 0..200 {
                    if let Some(processes) =
                        running_profile_processes(directory, expected_workload)?
                    {
                        return Ok(processes);
                    }
                    fixture_delay(Duration::from_millis(5)).await;
                }
                anyhow::bail!("The profiled workload never started");
            },
        )
        .await?;
        ensure!(
            !wait(&mut helper).await?.success(),
            "A cancelled profiling helper acknowledged completion",
        );
        ensure_profile_processes_exited(collector, workload)?;
        ensure!(
            !profile_path.exists() && !symbols_path.exists() && !socket.exists(),
            "Cancelled profiling outputs were retained",
        );
        Ok(())
    })
}

fn running_profile_processes(
    directory: &Path,
    expected_workload: Option<&Path>,
) -> Result<Option<(libc::pid_t, libc::pid_t)>> {
    let ids = match std::fs::read_to_string(directory.join("profiled-processes")) {
        Ok(ids) if ids.ends_with('\n') => ids,
        Ok(_) => return Ok(None),
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let (collector, workload) = ids
        .trim_end()
        .split_once(' ')
        .context("Malformed profiling fixture process IDs")?;
    let collector = collector.parse::<libc::pid_t>()?;
    let workload = workload.parse::<libc::pid_t>()?;
    ensure!(
        collector > 0 && workload > 0 && collector != workload,
        "Expected separately identified fixture processes"
    );
    if let Some(expected) = expected_workload
        && std::fs::read_link(format!("/proc/{workload}/exe"))? != expected
    {
        return Ok(None);
    }
    ensure!(
        unsafe { libc::getpgid(workload) } == collector,
        "The Rust workload was not in the original collector's process group",
    );
    Ok(Some((collector, workload)))
}

fn ensure_profile_processes_exited(collector: libc::pid_t, workload: libc::pid_t) -> Result<()> {
    for process_id in [collector, workload] {
        ensure!(
            !Path::new("/proc").join(process_id.to_string()).exists(),
            "The cancelled profiling process {process_id} was not reaped",
        );
    }
    Ok(())
}

fn samply_invocation(
    directory: &Path,
    profile_path: &Path,
    workload: &[&str],
) -> Result<TaskBuildInvocation> {
    let collector = std::env::var("RUST_TOOLS_LINUX_SAMPLY")?;
    let arguments = [
        "record",
        "--save-only",
        "--no-open",
        "--main-thread-only",
        "--unstable-presymbolicate",
        "--include-args=0",
        "--duration",
        "30",
        "-r",
        "100",
        "-o",
        profile_path
            .to_str()
            .context("Profile path is not Unicode")?,
        "--",
    ]
    .into_iter()
    .chain(workload.iter().copied())
    .map(str::to_owned)
    .collect();
    Ok(TaskBuildInvocation::new(
        collector.into(),
        arguments,
        directory.to_path_buf(),
        BTreeMap::from([("PATH".into(), "/usr/bin:/bin".into())]),
    )?)
}

fn native_profile_collector(directory: &Path) -> Result<()> {
    smol::block_on(async {
        let workload = std::env::var("RUST_TOOLS_LINUX_PROFILE_WORKLOAD")?;
        let expected_image = File::open("/proc/self/exe")?;
        let report = NativeProfileTaskReservation::new()?;
        let profile_path = report.profile_path();
        let symbols_path = report.symbols_path();
        let invocation = samply_invocation(directory, &profile_path, &[&workload, "1200000"])?;
        let completion = LinuxTaskCompletion::bind()?;
        let mut helper = spawn_completion_helper(&completion, LinuxTaskMode::NativeProfile)?;
        let actor = OriginalTaskActor::from_smol_child(&mut helper, &expected_image)?;
        let observed = completion
            .observe_profile(actor, invocation, report)
            .await?;
        let (status, pair) = observed.verify_exit(wait(&mut helper).await?)?;
        ensure!(status.success(), "Samply exited with {status}");
        let samples =
            verify_samply_profile(pair.context("Samply did not produce a sealed profile pair")?)?;
        ensure!(
            !profile_path.exists() && !symbols_path.exists(),
            "Profile pair was retained"
        );
        println!(
            "Linux authenticated collector retained {} samples and mapped stats::weight",
            samples
        );
        Ok(())
    })
}

fn verify_samply_profile(profile: terminal::SealedNativeProfileTask) -> Result<u64> {
    use project::profile_results::{ProfileHost, parse_samply_profile};
    let pair = profile.read()?;
    let result = parse_samply_profile(
        &pair.profile,
        &pair.symbols,
        ProfileHost::Linux,
        &Default::default(),
        |path, _| {
            path.ends_with("stats/src/lib.rs")
                .then(|| project::ProjectPath {
                    worktree_id: project::WorktreeId::from_proto(1),
                    path: util::rel_path::rel_path("stats/src/lib.rs").into_arc(),
                })
        },
    )?;
    ensure!(result.total_samples > 0, "Samply returned no samples");
    ensure!(
        result.frames.iter().any(|frame| {
            frame.label.contains("stats::weight")
                && frame.source.as_ref().is_some_and(|(_, line)| *line == 7)
        }),
        "The sealed profile did not map the sampled Rust frame"
    );
    Ok(result.total_samples)
}

fn cargo_build_report_controls(directory: &Path) -> Result<()> {
    smol::block_on(async {
        let expected_image = File::open("/proc/self/exe")?;
        for (script, limit, helper, expected_status) in [
            (
                r#"printf 'report'; printf 'diagnostic' >&2"#,
                6,
                LinuxTaskMode::CargoBuild,
                Some(0),
            ),
            (
                "printf 'report'; exit 27",
                6,
                LinuxTaskMode::CargoBuild,
                Some(27),
            ),
            ("printf 'report'", 5, LinuxTaskMode::CargoBuild, None),
            ("printf 'report'", 0, LinuxTaskMode::CargoBuild, None),
            ("printf 'report'", 6, LinuxTaskMode::NativeProfile, None),
        ] {
            let completion = LinuxTaskCompletion::bind()?;
            let mut helper = spawn_completion_helper(&completion, helper)?;
            let actor = OriginalTaskActor::from_smol_child(&mut helper, &expected_image)?;
            let receipt = completion
                .observe_cargo_build(
                    actor,
                    invocation(directory, script)?,
                    limit,
                    std::future::pending(),
                )
                .await;
            let exit = wait(&mut helper).await?;
            if let Some(expected_status) = expected_status {
                let (status, report) = receipt?.verify_exit(exit)?;
                ensure!(
                    status.code() == Some(expected_status),
                    "Cargo Task did not preserve its original child exit"
                );
                if status.success() {
                    ensure!(
                        report.as_deref() == Some(b"report"),
                        "Cargo Task stdout report mixed stdout with stderr"
                    );
                } else {
                    ensure!(
                        report.is_none(),
                        "A failed Cargo Task exposed a selectable report"
                    );
                }
            } else {
                ensure!(
                    receipt.is_err() && !exit.success(),
                    "Invalid Cargo report was accepted"
                );
            }
        }
        Ok(())
    })
}

fn cancel_cargo_build_report(directory: &Path) -> Result<()> {
    smol::block_on(async {
        let completion = LinuxTaskCompletion::bind()?;
        let mut helper = spawn_completion_helper(&completion, LinuxTaskMode::CargoBuild)?;
        let actor =
            OriginalTaskActor::from_smol_child(&mut helper, &File::open("/proc/self/exe")?)?;
        let build = invocation(
            directory,
            r#"printf 'partial report'
            sleep 60 &
            printf '%s %s\n' "$$" "$!" > profiled-processes
            wait"#,
        )?;
        let (ready, processes) = futures::channel::oneshot::channel();
        let cancel = async {
            let report = async {
                for _ in 0..200 {
                    if let Some(processes) = running_profile_processes(directory, None)? {
                        return Ok(processes);
                    }
                    fixture_delay(Duration::from_millis(5)).await;
                }
                anyhow::bail!("The Cargo Task did not start its child");
            }
            .await;
            if ready.send(report).is_err() {
                log::debug!("Cargo cancellation fixture released its process observer");
            }
        };
        let receipt = completion
            .observe_cargo_build(actor, build, 1024, cancel)
            .await?;
        let (cargo, compiler) = processes.await??;
        ensure!(
            receipt.was_cancelled(),
            "The Cargo Task did not acknowledge cancellation"
        );
        let (status, report) = receipt.verify_exit(wait(&mut helper).await?)?;
        ensure!(
            status.signal() == Some(libc::SIGKILL) && report.is_none(),
            "Cancelled Cargo output was admitted"
        );
        ensure_profile_processes_exited(cargo, compiler)
    })
}

fn write_native_cargo_fixture(directory: &Path) -> Result<()> {
    for (path, contents) in [
        (
            "Cargo.toml",
            "[workspace]\nmembers = [\"runner\", \"stats\"]\nresolver = \"3\"\n[profile.release]\ndebug = 2\n",
        ),
        (
            "runner/Cargo.toml",
            "[package]\nname = \"runner\"\nversion = \"0.1.0\"\nedition = \"2024\"\n[dependencies]\nstats = { path = \"../stats\" }\n",
        ),
        (
            "stats/Cargo.toml",
            "[package]\nname = \"stats\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        ),
        (
            "runner/src/main.rs",
            r#"fn main() {
    let values: Vec<_> = (0..4096).collect();
    let iterations = std::env::args().nth(1).and_then(|value| value.parse().ok()).unwrap_or(1);
    let total: u64 = (0..iterations).map(|_| stats::score(std::hint::black_box(&values))).sum();
    println!("Score: {total}");
}"#,
        ),
        (
            "runner/src/bin/decoy.rs",
            r#"compile_error!("The unselected binary must never be compiled");"#,
        ),
        (
            "stats/src/lib.rs",
            r#"pub fn score(values: &[u64]) -> u64 {
    values.iter().copied().map(weight).sum()
}

#[inline(never)]
fn weight(value: u64) -> u64 {
    if value > 1 { value * 2 } else { value }
}"#,
        ),
    ] {
        let path = directory.join(path);
        std::fs::create_dir_all(path.parent().context("Invalid Cargo fixture path")?)?;
        std::fs::write(path, contents)?;
    }
    Ok(())
}

fn native_cargo_profile_terminal(directory: &Path, headless: bool) -> Result<()> {
    use gpui::AppContext as _;
    use project::cargo_workspace::{
        CargoExecutableSelection, CargoNativeExecutionPolicy, CargoTargetKind,
        resolve_native_cargo_executable,
    };
    use terminal::{HeadlessTerminal, LinuxNativeTaskOutcome, TerminalBuilder};

    let directory = directory.canonicalize()?;
    write_native_cargo_fixture(&directory)?;
    let cargo = std::env::var_os("RUST_TOOLS_NATIVE_BUILD_CARGO")
        .context("The independently installed native Cargo is required")?;
    let cargo = std::path::PathBuf::from(cargo).canonicalize()?;
    let toolchain_bins = cargo
        .parent()
        .context("Native Cargo toolchain has no directory")?;
    let search_path =
        std::env::join_paths([toolchain_bins, Path::new("/usr/bin"), Path::new("/bin")])?
            .into_string()
            .map_err(|_| anyhow::anyhow!("Native Cargo toolchain path is not UTF-8"))?;
    let target = directory.join("cargo-target");
    let environment = BTreeMap::from([
        ("PATH".into(), search_path),
        ("CARGO_NET_OFFLINE".into(), "true".into()),
        (
            "CARGO_TARGET_DIR".into(),
            target.to_str().context("Invalid Cargo target path")?.into(),
        ),
        (
            "CARGO_HOME".into(),
            directory
                .join("cargo-home")
                .to_str()
                .context("Invalid Cargo home")?
                .into(),
        ),
    ]);
    let metadata = smol::block_on(
        smol::process::Command::new(&cargo)
            .args(["metadata", "--format-version=1", "--no-deps", "--offline"])
            .current_dir(&directory)
            .env_clear()
            .envs(&environment)
            .output(),
    )?;
    ensure!(
        metadata.status.success(),
        "Fixture Cargo metadata failed: {}",
        String::from_utf8_lossy(&metadata.stderr)
    );
    let metadata: serde_json::Value = serde_json::from_slice(&metadata.stdout)?;
    let package_id = metadata["packages"]
        .as_array()
        .and_then(|packages| {
            packages
                .iter()
                .find(|package| package["name"].as_str() == Some("runner"))
        })
        .and_then(|package| package["id"].as_str())
        .context("Fixture Cargo metadata is missing the runner identity")?
        .to_owned();
    let selection = CargoExecutableSelection {
        package_id,
        target_kind: CargoTargetKind::Binary,
        target_name: "runner".into(),
        source_path: directory.join("runner/src/main.rs"),
        execution_policy: CargoNativeExecutionPolicy::Program,
    };
    let build = TaskBuildInvocation::new(
        cargo,
        [
            "build",
            "--message-format=json",
            "--package",
            "runner",
            "--bin",
            "runner",
            "--release",
        ]
        .map(str::to_owned)
        .to_vec(),
        directory.clone(),
        environment,
    )?;
    let report = NativeProfileTaskReservation::new()?;
    let profile_path = report.profile_path();
    let symbols_path = report.symbols_path();
    let capture_path = profile_path.clone();
    let (send, receive) = std::sync::mpsc::sync_channel(1);
    gpui_platform::headless().run(move |cx| {
        release_channel::init(semver::Version::new(1, 16, 2), cx);
        cx.set_global(HeadlessTerminal(headless));
        let prepared = TerminalBuilder::new_with_native_cargo_build_task(
            task::SpawnInTerminal { label: "Selected Cargo build".into(), ..Default::default() },
            None, build, task::MAX_TASK_STDOUT_REPORT_BYTES, 1, cx,
        );
        cx.spawn(async move |cx| {
            let deadline = cx.background_executor().timer(Duration::from_secs(45));
            let run = async {
                let (builder, observation) = prepared?;
                let builder = builder.await?;
                let terminal = cx.update(|cx| cx.new(|cx| builder.subscribe(cx)));
                let ordinary = terminal.read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx));
                let (build, ordinary) = futures::future::join(observation.finish(), ordinary).await;
                let LinuxNativeTaskOutcome::Completed(status, Some(stdout)) = build? else {
                    anyhow::bail!("The Cargo build did not produce a verified stdout report");
                };
                ensure!(status.success() && ordinary == Some(status), "Cargo and its Terminal did not agree on the build exit");
                let executable = resolve_native_cargo_executable(&stdout, &selection, status.success())?;
                ensure!(executable == target.join("release/runner") && executable.is_file(), "Cargo report selected the wrong executable");
                let mut decoy = selection;
                decoy.target_name = "decoy".into();
                decoy.source_path = directory.join("runner/src/bin/decoy.rs");
                ensure!(resolve_native_cargo_executable(&stdout, &decoy, true).is_err(), "An unselected target was admitted");

                let executable = executable.to_str().context("Invalid Cargo executable path")?;
                let invocation = samply_invocation(&directory, &capture_path, &[executable, "1200000"])?;
                let (builder, observation) = cx.update(|cx| {
                    TerminalBuilder::new_with_native_profile_task(
                        task::SpawnInTerminal { label: "Profile the Cargo executable".into(), ..Default::default() },
                        None, invocation, report, 1, cx,
                    )
                })?;
                let builder = builder.await?;
                let terminal = cx.update(|cx| cx.new(|cx| builder.subscribe(cx)));
                let ordinary = terminal.read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx));
                let (captured, ordinary) = futures::future::join(observation.finish(), ordinary).await;
                let LinuxNativeTaskOutcome::Completed(status, Some(profile)) = captured? else {
                    anyhow::bail!("Profiling the built executable did not produce a sealed pair");
                };
                ensure!(status.success() && ordinary == Some(status), "Profile and its Terminal did not agree on the capture exit");
                let samples = cx.background_spawn(async move { verify_samply_profile(profile) }).await?;
                println!("Selected Cargo build and capture retained {samples} samples and mapped stats::weight");
                Ok(())
            };
            let result = match futures::future::select(Box::pin(run), deadline).await {
                futures::future::Either::Left((result, _)) => result,
                futures::future::Either::Right(_) => Err(anyhow::anyhow!("Selected Cargo build and capture timed out")),
            };
            if let Err(error) = send.send(result) {
                log::error!("Selected Cargo build fixture lost its caller: {error}");
            }
            cx.update(|cx| cx.quit());
        }).detach();
    });
    receive.recv()??;
    ensure!(
        !profile_path.exists() && !symbols_path.exists(),
        "Selected Cargo profile retained its reservation"
    );
    Ok(())
}

#[derive(Clone, Copy)]
enum ProfileTerminalAction {
    Complete,
    Stop,
    Close,
}

fn native_profile_terminal(
    directory: &Path,
    headless: bool,
    action: ProfileTerminalAction,
) -> Result<()> {
    use gpui::AppContext as _;
    use terminal::{HeadlessTerminal, LinuxNativeTaskOutcome, TerminalBuilder};

    let workload = std::env::var("RUST_TOOLS_LINUX_PROFILE_WORKLOAD")?;
    let report = NativeProfileTaskReservation::new()?;
    let profile_path = report.profile_path();
    let symbols_path = report.symbols_path();
    let invocation = match action {
        ProfileTerminalAction::Complete => {
            samply_invocation(directory, &profile_path, &[&workload, "1200000"])?
        }
        _ => samply_invocation(
            directory,
            &profile_path,
            &[
                "/bin/bash",
                "-c",
                r#"printf '%s %s\n' "$PPID" "$$" > profiled-processes; exec "$1" 1200000"#,
                "fixture",
                &workload,
            ],
        )?,
    };
    let directory = directory.to_path_buf();
    let (send, receive) = std::sync::mpsc::sync_channel(1);
    gpui_platform::headless().run(move |cx| {
        release_channel::init(semver::Version::new(1, 16, 2), cx);
        cx.set_global(HeadlessTerminal(headless));
        let handle = task::StructuredTaskHandle::new(task::TaskId("native-profile-terminal".into()));
        let (cancelled, confirmed) = async_channel::bounded(1);
        handle.subscribe(cx, move |event, _| {
            if matches!(event.state, task::StructuredTaskState::Cancelled { termination_confirmed: true, .. })
                && let Err(error) = cancelled.try_send(())
            {
                log::debug!("Fixture cancellation observer closed: {error}");
            }
        });
        let prepared = TerminalBuilder::new_with_native_profile_task(
            task::SpawnInTerminal { label: "Native Profile Task fixture".into(), ..Default::default() },
            Some(handle.clone()), invocation, report, 1, cx,
        );
        cx.spawn(async move |cx| {
            let deadline = cx.background_executor().timer(Duration::from_secs(15));
            let run = async {
                let (builder, observation) = prepared?;
                let builder = builder.await?;
                let terminal = cx.update(|cx| cx.new(|cx| {
                    handle.mark_running(Some(task::StructuredTerminalId(cx.entity_id().as_u64())), cx);
                    builder.subscribe(cx)
                }));
                let ordinary_completion = terminal.read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx));
                match action {
                    ProfileTerminalAction::Complete => {
                        let (observed, ordinary) = futures::future::join(observation.finish(), ordinary_completion).await;
                        let LinuxNativeTaskOutcome::Completed(status, Some(profile)) = observed? else {
                            anyhow::bail!("The Terminal did not produce a sealed profile");
                        };
                        ensure!(status.success() && ordinary == Some(status), "The native receipt and Terminal disagreed on exit");
                        let samples = cx.background_spawn(async move { verify_samply_profile(profile) }).await?;
                        println!("Native Profile Terminal retained {samples} samples and mapped stats::weight");
                    }
                    ProfileTerminalAction::Stop | ProfileTerminalAction::Close => {
                        let (collector, workload_pid) = loop {
                            if let Some(processes) = running_profile_processes(&directory, Some(Path::new(&workload)))? {
                                break processes;
                            }
                            cx.background_executor().timer(Duration::from_millis(5)).await;
                        };
                        let terminal = if matches!(action, ProfileTerminalAction::Close) {
                            cx.update(|_| drop(terminal));
                            None
                        } else {
                            terminal.update(cx, |terminal, _| terminal.kill_active_task());
                            Some(terminal)
                        };
                        ensure!(matches!(observation.finish().await?, LinuxNativeTaskOutcome::Cancelled), "The Terminal did not acknowledge cancellation");
                        confirmed.recv().await?;
                        ensure_profile_processes_exited(collector, workload_pid)?;
                        if terminal.is_some() {
                            ensure!(ordinary_completion.await.is_some(), "Terminal did not publish its stopped status");
                        }
                        println!("Native Profile Terminal confirmed cancellation and reaped both processes");
                    }
                }
                Ok(())
            };
            let result = match futures::future::select(Box::pin(run), deadline).await {
                futures::future::Either::Left((result, _)) => result,
                futures::future::Either::Right(_) => Err(anyhow::anyhow!("Native Profile Terminal fixture timed out")),
            };
            if let Err(error) = send.send(result) {
                log::error!("Native Profile Terminal fixture lost its caller: {error}");
            }
            cx.update(|cx| cx.quit());
        }).detach();
    });
    receive.recv()??;
    ensure!(
        !profile_path.exists() && !symbols_path.exists(),
        "Native Profile Terminal retained its reservation"
    );
    Ok(())
}

fn write_collector_fixture(directory: &Path) -> Result<()> {
    for (name, bytes) in [
        (
            "Cargo.toml",
            include_bytes!(
                "../../project/test_data/rust_test_provider/physical-workspace/Cargo.toml"
            )
            .as_slice(),
        ),
        (
            "Cargo.lock",
            include_bytes!(
                "../../project/test_data/rust_test_provider/physical-workspace/Cargo.lock"
            )
            .as_slice(),
        ),
        (
            "src/lib.rs",
            include_bytes!(
                "../../project/test_data/rust_test_provider/physical-workspace/src/lib.rs"
            )
            .as_slice(),
        ),
        (
            "src/bin/fixture.rs",
            include_bytes!(
                "../../project/test_data/rust_test_provider/physical-workspace/src/bin/fixture.rs"
            )
            .as_slice(),
        ),
        (
            "tests/integration.rs",
            include_bytes!(
                "../../project/test_data/rust_test_provider/physical-workspace/tests/integration.rs"
            )
            .as_slice(),
        ),
    ] {
        let path = directory.join(name);
        std::fs::create_dir_all(path.parent().context("Invalid fixture path")?)?;
        std::fs::write(path, bytes)?;
    }
    Ok(())
}

fn collector_invocation(directory: &Path, report: &Path) -> Result<TaskBuildInvocation> {
    write_collector_fixture(directory)?;
    let cargo = std::env::var_os("RUST_TOOLS_COVERAGE_CARGO")
        .context("The independently installed Cargo is required")?;
    let mut environment = BTreeMap::new();
    for name in [
        "PATH",
        "LLVM_COV",
        "LLVM_PROFDATA",
        "RUSTUP_HOME",
        "RUSTUP_TOOLCHAIN",
    ] {
        environment.insert(name.to_owned(), std::env::var(name)?);
    }
    environment.insert(
        "HOME".into(),
        directory
            .to_str()
            .context("Fixture path is not Unicode")?
            .into(),
    );
    Ok(TaskBuildInvocation::new(
        cargo.into(),
        vec![
            "llvm-cov".into(),
            "--json".into(),
            "--output-path".into(),
            report
                .to_str()
                .context("Report path is not Unicode")?
                .into(),
            "run".into(),
            "--locked".into(),
            "--offline".into(),
            "--bin".into(),
            "fixture-bin".into(),
        ],
        directory.to_path_buf(),
        environment,
    )?)
}

fn collector_terminal(directory: &Path) -> Result<()> {
    let report = CoverageTaskReportReservation::new(task::MAX_TASK_ARTIFACT_BYTES as usize)?;
    let invocation = collector_invocation(directory, &report.path())?;
    terminal_handoff(false, invocation, report, |bytes| {
        let report: serde_json::Value = serde_json::from_slice(bytes)?;
        ensure!(
            report["type"].as_str() == Some("llvm.coverage.json.export"),
            "Unexpected coverage export type"
        );
        let file = report["data"]
            .as_array()
            .context("Missing coverage data")?
            .iter()
            .filter_map(|data| data["files"].as_array())
            .flatten()
            .find(|file| {
                file["filename"]
                    .as_str()
                    .is_some_and(|name| name.ends_with("/src/bin/fixture.rs"))
            })
            .context("Coverage export did not contain the selected fixture binary")?;
        ensure!(
            file["summary"]["lines"]["count"].as_u64() == Some(3)
                && file["summary"]["lines"]["covered"].as_u64() == Some(3),
            "Expected all three fixture lines to be covered"
        );
        Ok(())
    })
}

fn headless_report_group(path: String) -> task::TaskArtifactGroup {
    task::TaskArtifactGroup {
        format: "llvm-cov-json-v1".into(),
        files: BTreeMap::from([(
            "report".into(),
            task::TaskArtifact {
                kind: task::TaskArtifactKind::Data,
                path,
                max_bytes: task::MAX_TASK_ARTIFACT_BYTES,
            },
        )]),
    }
}

async fn headless_failure_controls(
    host: &gpui::Entity<crate::HeadlessProject>,
    directory: &Path,
    cx: &mut gpui::AsyncApp,
) -> Result<()> {
    use project::source_coverage::{SourceCoverageSourceChanged, SourceCoverageStatus};

    for (command, expected) in [
        (r#"printf '{}' > "$2"; exit 5"#, SourceCoverageStatus::Error),
        (
            r#"printf '\n// source changed during coverage\n' >> src/bin/fixture.rs &&
                /bin/cat expected-coverage.json > "$2""#,
            SourceCoverageStatus::Stale,
        ),
    ] {
        let run_id = uuid::Uuid::new_v4();
        let path = format!("target/zed-coverage-{}.json", run_id.simple());
        let group = headless_report_group(path.clone());
        let launch = task::SpawnInTerminal {
            command: Some("/bin/sh".into()),
            args: vec![
                "-c".into(),
                command.into(),
                "fixture".into(),
                "--output-path".into(),
                path,
            ],
            cwd: Some(directory.to_path_buf()),
            label: "Coverage rejection fixture".into(),
            ..Default::default()
        };
        let result = host
            .update(cx, |host, cx| {
                host.run_host_coverage_task(run_id.to_string(), launch, &group, cx)
            })?
            .await;
        let error = result.err().context("A rejected Task published coverage")?;
        if expected == SourceCoverageStatus::Stale {
            ensure!(
                error.is::<SourceCoverageSourceChanged>(),
                "Expected the mapped source to expire, got: {error:#}"
            );
        } else {
            ensure!(
                error.to_string().contains("exit status: 5"),
                "The observed nonzero exit was lost: {error:#}"
            );
        }
        let status = host.read_with(cx, |host, cx| {
            host.source_coverage_store
                .read(cx)
                .cached_snapshots()
                .next()
                .map(|snapshot| snapshot.status)
        });
        ensure!(
            status == Some(expected),
            "Unexpected rejected Task status: {status:?}"
        );
    }
    Ok(())
}

async fn headless_cancel_control(
    host: &gpui::Entity<crate::HeadlessProject>,
    directory: &Path,
    cx: &mut gpui::AsyncApp,
) -> Result<()> {
    use project::source_coverage::SourceCoverageStatus;

    let marker = directory.with_extension("coverage-workload");
    File::options().write(true).create_new(true).open(&marker)?;
    let run_id = uuid::Uuid::new_v4();
    let path = format!("target/zed-coverage-{}.json", run_id.simple());
    let group = headless_report_group(path.clone());
    let launch = task::SpawnInTerminal {
        command: Some("/bin/sh".into()),
        args: vec![
            "-c".into(),
            r#"printf '%s' $$ > "$1"; exec /bin/sleep 60"#.into(),
            "fixture".into(),
            marker.to_str().context("Non-Unicode fixture path")?.into(),
            "--output-path".into(),
            path,
        ],
        cwd: Some(directory.to_path_buf()),
        label: "Coverage cancellation fixture".into(),
        ..Default::default()
    };
    let task = host.update(cx, |host, cx| {
        host.run_host_coverage_task(run_id.to_string(), launch, &group, cx)
    })?;
    let child = {
        let mut child = None;
        for _ in 0..200 {
            let value = std::fs::read_to_string(&marker)?;
            if !value.is_empty() {
                child = Some(value.parse::<u32>()?);
                break;
            }
            cx.background_executor()
                .timer(Duration::from_millis(10))
                .await;
        }
        child.context("Coverage workload never started")?
    };
    std::fs::remove_file(marker)?;
    drop(task);
    for _ in 0..200 {
        let status = host.read_with(cx, |host, cx| {
            host.source_coverage_store
                .read(cx)
                .cached_snapshots()
                .next()
                .map(|snapshot| snapshot.status)
        });
        if status == Some(SourceCoverageStatus::Cancelled)
            && !Path::new("/proc").join(child.to_string()).exists()
        {
            return Ok(());
        }
        cx.background_executor()
            .timer(Duration::from_millis(10))
            .await;
    }
    anyhow::bail!("Dropping the host capture did not cancel and reap its workload")
}

fn verify_headless_coverage(
    host: &gpui::Entity<crate::HeadlessProject>,
    snapshot: &project::source_coverage::SourceCoverageSnapshot,
    cx: &gpui::AsyncApp,
) -> Result<()> {
    use project::source_coverage::SourceCoverageStatus;

    ensure!(
        matches!(
            snapshot.status,
            SourceCoverageStatus::Current | SourceCoverageStatus::Partial
        ),
        "Unexpected coverage status: {:?}",
        snapshot.status
    );
    let file = snapshot
        .files
        .iter()
        .find(|file| file.path.path.as_std_path() == Path::new("src/bin/fixture.rs"))
        .context("Host report is missing the selected Rust source")?;
    ensure!(
        file.covered_lines == 3 && file.uncovered_lines == 0,
        "Expected three covered Rust lines, got {} covered and {} uncovered",
        file.covered_lines,
        file.uncovered_lines,
    );
    ensure!(
        !snapshot
            .files
            .iter()
            .any(|file| file.path.path.as_std_path() == Path::new("src/bin/second.rs")),
        "Coverage included an unselected binary"
    );
    let cached = host.read_with(cx, |host, cx| {
        host.source_coverage_store
            .read(cx)
            .cached_snapshots()
            .any(|cached| cached == snapshot)
    });
    ensure!(
        cached,
        "The authoritative provider did not publish its result"
    );
    Ok(())
}

fn headless_collector(directory: &Path, select_cargo_target: bool) -> Result<()> {
    use gpui::{AppContext as _, UpdateGlobal as _};
    use project::ProjectPath;
    use project::cargo_workspace::{CargoTargetKind, CargoWorkspaceKey};
    use project::cargo_workspace_store::CargoCoverageSelection;
    use project::worktree_store::{WorktreeIdCounter, WorktreeStore};
    use std::sync::Arc;
    use util::rel_path::RelPath;

    let directory = std::fs::canonicalize(directory)?;
    let run_id = uuid::Uuid::new_v4();
    let artifact_path = format!("target/zed-coverage-{}.json", run_id.simple());
    let launch = if select_cargo_target {
        write_collector_fixture(&directory)?;
        std::fs::write(directory.join("src/bin/second.rs"), "fn main() {}\n")?;
        None
    } else {
        let invocation = collector_invocation(&directory, Path::new(&artifact_path))?;
        Some(task::SpawnInTerminal {
            command: Some(invocation.program().to_string_lossy().into_owned()),
            args: invocation.arguments().to_vec(),
            cwd: Some(directory.clone()),
            env: invocation.environment().clone().into_iter().collect(),
            label: "Run with Coverage".into(),
            required_experimental_features: vec!["rust-coverage-experimental".into()],
            ..Default::default()
        })
    };
    let logical_report = directory.join(&artifact_path);
    std::fs::create_dir_all(directory.join("target"))?;
    std::fs::write(directory.join(".gitignore"), "/target/\n")?;
    std::fs::write(&logical_report, b"unrelated stale report")?;
    std::fs::write(
        directory.join("expected-coverage.json"),
        include_str!("../../project/test_data/source_coverage/llvm3-run.json").replace(
            "/workspace",
            directory.to_str().context("Non-Unicode fixture root")?,
        ),
    )?;
    let group = headless_report_group(artifact_path);

    let (send, receive) = std::sync::mpsc::sync_channel(1);
    gpui_platform::headless().run(move |cx| {
        release_channel::init(semver::Version::new(1, 16, 2), cx);
        gpui_tokio::init(cx);
        crate::HeadlessProject::init(cx);
        let opt_in = settings::SettingsStore::update_global(cx, |store, cx| {
            store.register_setting::<feature_flags::FeatureFlagsSettings>();
            store
                .set_user_settings(
                    r#"{"load_direnv":"disabled","feature_flags":{"rust-coverage-experimental":"on"}}"#,
                    cx,
                )
                .result()
        });
        project::trusted_worktrees::init(
            [(None, [directory.clone()].into_iter().collect())]
                .into_iter()
                .collect(),
            cx,
        );
        git::GitHostingProviderRegistry::set_global(
            Arc::new(git::GitHostingProviderRegistry::new()),
            cx,
        );
        git_hosting_providers::init(cx);
        dap_adapters::init(cx);
        cx.set_global(terminal::HeadlessTerminal(true));
        let (incoming, incoming_receiver) = futures::channel::mpsc::unbounded();
        let (outgoing, outgoing_receiver) = futures::channel::mpsc::unbounded();
        let session = remote::RemoteClient::proto_client_from_channels(
            incoming_receiver,
            outgoing,
            cx,
            "coverage fixture host",
            false,
        );
        let client = remote::RemoteClient::proto_client_from_channels(
            outgoing_receiver,
            incoming,
            cx,
            "coverage fixture client",
            false,
        );
        WorktreeStore::init_remote(&client);
        let client_worktrees = cx.new(|cx| {
            WorktreeStore::remote(
                true,
                client.clone(),
                proto::REMOTE_SERVER_PROJECT_ID,
                util::paths::PathStyle::local(),
                WorktreeIdCounter::get(cx),
            )
        });
        client.subscribe_to_entity(proto::REMOTE_SERVER_PROJECT_ID, &client_worktrees);
        let fs = Arc::new(fs::RealFs::new(None, cx.background_executor().clone()));
        let languages = Arc::new(language::LanguageRegistry::new(
            cx.background_executor().clone(),
        ));
        let host = cx.new(|cx| {
            crate::HeadlessProject::new(
                crate::HeadlessAppState {
                    session,
                    fs,
                    languages,
                    http_client: Arc::new(http_client::BlockedHttpClient),
                    node_runtime: node_runtime::NodeRuntime::unavailable(),
                    extension_host_proxy: Arc::new(extension::ExtensionHostProxy::new()),
                    startup_time: std::time::Instant::now(),
                },
                true,
                cx,
            )
        });
        cx.spawn(async move |cx| {
            let deadline = cx.background_executor().timer(Duration::from_secs(20));
            let stage = std::cell::Cell::new("registering the worktree");
            let run = async {
                opt_in?;
                let worktrees = host.read_with(cx, |host, _| host.worktree_store.clone());
                let (worktree, _) = worktrees
                    .update(cx, |worktrees, cx| {
                        worktrees.find_or_create_worktree(&directory, true, cx)
                    })
                    .await?;
                stage.set("scanning the worktree");
                worktree
                    .read_with(cx, |worktree, _| {
                        worktree
                            .as_local()
                            .context("Expected a local host worktree")
                            .map(|tree| tree.scan_complete())
                    })?
                    .await;
                stage.set("running the Cargo LLVM collector");
                let (snapshot, recovery) = if let Some(launch) = launch {
                    let snapshot = host.update(cx, |host, cx| {
                        host.run_host_coverage_task(run_id.to_string(), launch, &group, cx)
                    })?.await?;
                    (snapshot, None)
                } else {
                    let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id());
                    let selection = CargoCoverageSelection {
                        workspace: CargoWorkspaceKey {
                            root: ProjectPath { worktree_id, path: RelPath::empty_arc() },
                        },
                        package_manifest: ProjectPath {
                            worktree_id,
                            path: RelPath::from_unix_str("Cargo.toml")?.into_arc(),
                        },
                        target_kind: CargoTargetKind::Binary,
                        target_name: "--output-path".into(),
                        preset_id: None,
                    };
                    let rejected = host.update(cx, |host, cx| host.run_selected_host_coverage_task(selection.clone(), cx))?.await;
                    let error = rejected.err().context("A selector absent from Cargo metadata started a Task")?;
                    ensure!(format!("{error:#}").contains("Coverage target is no longer available"), "Unexpected selector refusal: {error:#}");
                    let selection = CargoCoverageSelection {
                        target_name: "fixture-bin".into(),
                        ..selection
                    };
                    let snapshot = host.update(cx, |host, cx| {
                        host.run_selected_host_coverage_task(selection.clone(), cx)
                    })?.await?;
                    (snapshot, Some(selection))
                };
                verify_headless_coverage(&host, &snapshot, cx)?;
                ensure!(
                    std::fs::read(&logical_report)? == b"unrelated stale report",
                    "The host collected through the logical workspace report"
                );
                stage.set("cancelling the active workload");
                headless_cancel_control(&host, &directory, cx).await?;
                stage.set("checking failed and stale runs");
                headless_failure_controls(&host, &directory, cx).await?;
                if let Some(selection) = recovery {
                    stage.set("recovering with a fresh selected run");
                    let source_path = RelPath::from_unix_str("src/bin/fixture.rs")?.into_arc();
                    worktree
                        .update(cx, |worktree, cx| {
                            worktree
                                .as_local()
                                .context("Expected a local host worktree")
                                .map(|tree| tree.refresh_entry(source_path, None, cx))
                        })?
                        .await?
                        .context("Modified fixture source is unavailable")?;
                    let recovered = host.update(cx, |host, cx| {
                        host.run_selected_host_coverage_task(selection, cx)
                    })?.await?;
                    ensure!(recovered.generation > snapshot.generation, "Recovery reused its original admission");
                    verify_headless_coverage(&host, &recovered, cx)?;
                }
                Ok(())
            };
            let result = match futures::future::select(Box::pin(run), deadline).await {
                futures::future::Either::Left((result, _)) => result,
                futures::future::Either::Right(_) => Err(anyhow::anyhow!(
                    "Headless Coverage Task fixture timed out while {}",
                    stage.get()
                )),
            };
            if let Err(error) = &result {
                eprintln!("Headless coverage fixture failed: {error:#}");
            }
            if let Err(error) = send.send(result) {
                log::error!("Headless Coverage Task fixture lost its caller: {error}");
            }
            drop(host);
            drop(client_worktrees);
            cx.update(|cx| {
                cx.shutdown();
                cx.quit();
            });
        })
        .detach();
    });
    receive.recv()?
}

#[derive(Clone, Copy, clap::ValueEnum)]
pub enum Scenario {
    TerminalChildExit,
    ObserverDrop,
    ForeignChild,
    OversizedInvocation,
    PtyTerminal,
    HeadlessTerminal,
    CollectorTerminal,
    ReportReplacement,
    ProfilePairReplacement,
    ProfileGroupCancellation,
    NativeProfileCollector,
    NativeProfileCollectorCancellation,
    NativeProfilePtyTerminal,
    NativeProfileHeadlessTerminal,
    NativeProfileStopPty,
    NativeProfileStopHeadless,
    NativeProfileClosePty,
    NativeProfileCloseHeadless,
    CargoBuildReportControls,
    CargoBuildCancellation,
    CargoBuildProfilePty,
    CargoBuildProfileHeadless,
    HeadlessCollector,
    SelectedHeadlessCollector,
}

pub fn run(directory: &Path, scenario: Scenario) -> Result<()> {
    ensure!(
        directory.is_dir(),
        "Native Task fixture requires an existing directory"
    );
    match scenario {
        Scenario::TerminalChildExit => terminal_child_exit(directory),
        Scenario::ObserverDrop => observer_drop(directory),
        Scenario::ForeignChild => foreign_child(directory),
        Scenario::OversizedInvocation => oversized_invocation(directory),
        Scenario::PtyTerminal => shell_terminal(directory, false),
        Scenario::HeadlessTerminal => shell_terminal(directory, true),
        Scenario::CollectorTerminal => collector_terminal(directory),
        Scenario::ReportReplacement => report_replacement(directory),
        Scenario::ProfilePairReplacement => profile_pair_replacement(directory),
        Scenario::ProfileGroupCancellation => profile_group_cancellation(directory),
        Scenario::NativeProfileCollector => native_profile_collector(directory),
        Scenario::NativeProfileCollectorCancellation => {
            native_profile_collector_cancellation(directory)
        }
        Scenario::NativeProfilePtyTerminal => {
            native_profile_terminal(directory, false, ProfileTerminalAction::Complete)
        }
        Scenario::NativeProfileHeadlessTerminal => {
            native_profile_terminal(directory, true, ProfileTerminalAction::Complete)
        }
        Scenario::NativeProfileStopPty => {
            native_profile_terminal(directory, false, ProfileTerminalAction::Stop)
        }
        Scenario::NativeProfileStopHeadless => {
            native_profile_terminal(directory, true, ProfileTerminalAction::Stop)
        }
        Scenario::NativeProfileClosePty => {
            native_profile_terminal(directory, false, ProfileTerminalAction::Close)
        }
        Scenario::NativeProfileCloseHeadless => {
            native_profile_terminal(directory, true, ProfileTerminalAction::Close)
        }
        Scenario::CargoBuildReportControls => cargo_build_report_controls(directory),
        Scenario::CargoBuildCancellation => cancel_cargo_build_report(directory),
        Scenario::CargoBuildProfilePty => native_cargo_profile_terminal(directory, false),
        Scenario::CargoBuildProfileHeadless => native_cargo_profile_terminal(directory, true),
        Scenario::HeadlessCollector => headless_collector(directory, false),
        Scenario::SelectedHeadlessCollector => headless_collector(directory, true),
    }
}
