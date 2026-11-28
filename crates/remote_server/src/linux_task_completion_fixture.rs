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
    sync::atomic::{AtomicU32, Ordering},
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
    verify_samply_profile_bytes(profile.read()?)
}

fn verify_samply_profile_bytes(
    pair: native_task_helper::linux_task_completion::NativeProfileTaskArtifacts,
) -> Result<u64> {
    use project::profile_results::{ProfileHost, parse_samply_profile};
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
    if let Some(value) = std::env::var_os("COPPER_PROJECT_PROFILE_FIXTURE") {
        assert_eq!(value, "project-only runtime sentinel");
        assert_eq!(std::env::args().skip(1).collect::<Vec<_>>(),
            ["1200000", "literal two words", "", "$PROJECT_CARGO_PROFILE"]);
        assert!(std::env::var("LD_LIBRARY_PATH").is_ok_and(|value| value.contains("fixture-loader")));
        std::fs::write("target/profiled-project-workload", "selected").expect("profile workload marker");
    }
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

fn native_cargo_runtime_transition(directory: &Path, record: bool) -> Result<()> {
    use native_task_helper::{
        linux_cargo_runtime::LinuxCargoRuntimeObserver,
        task_completion::TASK_CARGO_RUNTIME_ENDPOINT_ENV,
    };

    let directory = directory.canonicalize()?;
    write_native_cargo_fixture(&directory)?;
    let runner = std::env::var_os("RUST_TOOLS_LINUX_RUNTIME_OBSERVER")
        .context("The separately linked Linux Cargo runner is required")?;
    let source_image = directory.join("runtime-observer");
    std::fs::copy(&runner, &source_image)?;
    ensure!(
        LinuxCargoRuntimeObserver::bind(&std::env::current_exe()?).is_err(),
        "The runtime observer accepted a dynamically linked helper"
    );
    std::fs::write(
        directory.join("runner/build.rs"),
        r#"
fn main() {
    let directory = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"))
        .join("fixture-loader");
    std::fs::create_dir_all(&directory).expect("fixture loader");
    println!("cargo:rustc-link-search=native={}", directory.display());
}
"#,
    )?;
    std::fs::write(
        directory.join("runner/src/main.rs"),
        r#"
fn main() {
    let actual: Vec<_> = std::env::args().skip(1).collect();
    assert_eq!(actual, ["literal two words", "", "$FOO"]);
    assert!(std::env::var("LD_LIBRARY_PATH").is_ok_and(|value| value.contains("fixture-loader")));
    assert_eq!(std::env::var("COPPER_RUNTIME_FIXTURE").as_deref(), Ok("runtime sentinel"));
    assert!(std::env::var_os("ZED_PROFILE_CARGO_RUNTIME_ENDPOINT").is_none());
    for entry in std::fs::read_dir("/proc/self/fd").expect("process file descriptors").flatten() {
        if let Ok(path) = std::fs::read_link(entry.path()) {
            assert!(!path.to_string_lossy().contains("memfd:zed-cargo-"));
        }
    }
    if std::env::var_os("COPPER_RUNTIME_PROFILE").is_some() {
        std::fs::write("profiled-runtime-workload", "verified runtime").expect("workload marker");
        let values: Vec<_> = (0..4096).collect();
        let sum: u64 = (0..1_200_000).map(|_| stats::score(std::hint::black_box(&values))).sum();
        println!("Profiled score: {sum}");
    }
    println!("Selected native workload received the observed Cargo environment");
}
"#,
    )?;
    let cargo = std::env::var_os("RUST_TOOLS_NATIVE_BUILD_CARGO")
        .context("The independently installed native Cargo is required")?;
    let cargo = std::path::PathBuf::from(cargo).canonicalize()?;
    let cargo_image = File::open(&cargo)?;
    let cargo_path = cargo.parent().context("Missing toolchain directory")?;
    let cargo_home = directory.join("cargo-home");
    let host = format!("{}-unknown-linux-gnu", std::env::consts::ARCH);
    let build_root = directory.join("target");
    let target = build_root.join(&host).join(if record {
        "release/runner"
    } else {
        "debug/runner"
    });
    let literal = ["literal two words", "", "$FOO"].map(str::to_owned);
    if record {
        let arguments = [
            "run",
            "--offline",
            "--quiet",
            "--target",
            &host,
            "--package",
            "runner",
            "--bin",
            "runner",
            "--profile",
            "release",
            "--",
        ]
        .map(str::to_owned)
        .into_iter()
        .chain(literal.iter().cloned())
        .collect();
        let environment = BTreeMap::from([
            (
                "PATH".into(),
                std::env::join_paths([cargo_path, Path::new("/usr/bin"), Path::new("/bin")])?
                    .into_string()
                    .map_err(|_| anyhow::anyhow!("Invalid fixture PATH"))?,
            ),
            (
                "CARGO_HOME".into(),
                cargo_home
                    .to_str()
                    .context("Invalid fixture Cargo home")?
                    .into(),
            ),
            (
                "CARGO_TARGET_DIR".into(),
                build_root
                    .to_str()
                    .context("Invalid fixture build directory")?
                    .into(),
            ),
            ("COPPER_RUNTIME_FIXTURE".into(), "runtime sentinel".into()),
            ("COPPER_RUNTIME_PROFILE".into(), "1".into()),
        ]);
        let cargo = TaskBuildInvocation::new(cargo, arguments, directory.clone(), environment)?;
        let runtime = terminal::LinuxCargoRuntimeInvocation::new(
            cargo,
            target,
            literal.into(),
            runner.clone().into(),
        )?;
        return profile_observed_cargo_runtime(&directory, runtime, runner.into());
    }
    smol::block_on(async {
        for case in [
            "replaced-path",
            "changed-arguments",
            "wrong-image",
            "foreign-peer",
        ] {
            let mut command = std::process::Command::new(&cargo);
            command
                .args([
                    "run",
                    "--offline",
                    "--quiet",
                    "--target",
                    &host,
                    "--package",
                    "runner",
                    "--bin",
                    "runner",
                    "--profile",
                    "dev",
                    "--",
                ])
                .args(&literal)
                .current_dir(&directory)
                .env_clear()
                .env(
                    "PATH",
                    std::env::join_paths([cargo_path, Path::new("/usr/bin"), Path::new("/bin")])?,
                )
                .env("CARGO_HOME", &cargo_home)
                .env("CARGO_TARGET_DIR", &build_root)
                .env("COPPER_RUNTIME_FIXTURE", "runtime sentinel");
            let observer = LinuxCargoRuntimeObserver::bind(&source_image)?;
            let pending = observer.configure(&mut command)?;
            let endpoint = command
                .get_envs()
                .find_map(|(name, value)| {
                    (name == TASK_CARGO_RUNTIME_ENDPOINT_ENV)
                        .then_some(value)
                        .flatten()
                })
                .context("Missing checked runtime endpoint")?
                .to_owned();
            if case == "replaced-path" {
                std::fs::remove_file(&source_image)?;
                std::fs::write(&source_image, b"runner path changed after sealing")?;
            }
            if case == "wrong-image" {
                command.env(
                    format!(
                        "CARGO_TARGET_{}_RUNNER",
                        host.replace('-', "_").to_ascii_uppercase()
                    ),
                    format!("{} --observe-cargo-runtime --", source_image.display()),
                );
            }
            let foreign = if case == "foreign-peer" {
                Some(smol::net::unix::UnixStream::connect(&endpoint).await?)
            } else {
                None
            };
            let mut child = smol::process::Command::from(command)
                .stderr(Stdio::piped())
                .kill_on_drop(true)
                .spawn()?;
            let actor = OriginalTaskActor::from_smol_child(&mut child, &cargo_image)?;
            let mut expected = literal.clone();
            if case == "changed-arguments" {
                expected[0] = "different argument".into();
            }
            let capture = smol::future::race(
                pending.observe(actor, &target, &expected, &directory),
                async {
                    fixture_delay(Duration::from_secs(12)).await;
                    Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "Real Cargo runner did not connect",
                    ))
                },
            )
            .await;
            drop(foreign);
            let status = wait(&mut child).await?;
            if case == "replaced-path" {
                if let Some(mut diagnostics) = child.stderr.take() {
                    let mut text = String::new();
                    diagnostics.read_to_string(&mut text).await?;
                    ensure!(status.success(), "Real Cargo runner failed: {text}");
                }
                let invocation = capture?.verify_exit(status)?;
                ensure!(
                    invocation.program() == target && invocation.arguments() == literal,
                    "Wrong observed workload"
                );
                let output = smol::process::Command::new(invocation.program())
                    .args(invocation.arguments())
                    .env_clear()
                    .envs(invocation.environment())
                    .current_dir(invocation.working_directory())
                    .output()
                    .await?;
                ensure!(
                    output.status.success()
                        && output
                            .stdout
                            .starts_with(b"Selected native workload received"),
                    "The observed environment did not reproduce Cargo's launch"
                );
                ensure!(
                    !Path::new(&endpoint).exists(),
                    "The runtime observer kept its socket"
                );
                std::fs::remove_file(&source_image)?;
                std::fs::copy(&runner, &source_image)?;
            } else {
                let error = capture
                    .err()
                    .context("An unauthorised runtime runner was accepted")?;
                let (kind, diagnostic) = match case {
                    "foreign-peer" => (
                        io::ErrorKind::PermissionDenied,
                        "connection does not belong to its original actor",
                    ),
                    "wrong-image" => (
                        io::ErrorKind::PermissionDenied,
                        "Original Task actor image changed",
                    ),
                    _ => (
                        io::ErrorKind::InvalidData,
                        "Invalid Linux Cargo runtime observation",
                    ),
                };
                ensure!(
                    error.kind() == kind
                        && error.to_string().contains(diagnostic)
                        && !status.success(),
                    "Invalid runtime control did not fail at admission: {error}"
                );
            }
        }
        println!(
            "Real Cargo transitioned to the sealed Linux runner and preserved its selected runtime environment"
        );
        Ok(())
    })
}

fn profile_observed_cargo_runtime(
    directory: &Path,
    runtime: terminal::LinuxCargoRuntimeInvocation,
    runner: std::path::PathBuf,
) -> Result<()> {
    use gpui::AppContext as _;
    use std::os::unix::fs::PermissionsExt as _;
    use terminal::{
        HeadlessTerminal, LinuxCargoProfileInvocation, LinuxNativeTaskOutcome, TerminalBuilder,
    };

    let collector = directory.join("fixture-collector");
    std::fs::write(
        &collector,
        r#"#!/bin/sh
test -z "${LD_LIBRARY_PATH+x}" &&
test -z "${COPPER_RUNTIME_FIXTURE+x}" &&
test -z "${COPPER_RUNTIME_PROFILE+x}" || exit 43
for argument do
    case "$argument" in
        "literal two words"|'$FOO'|*"runtime sentinel"*) exit 44 ;;
    esac
done
exec "$COPPER_FIXTURE_SAMPLY" "$@"
"#,
    )?;
    std::fs::set_permissions(&collector, std::fs::Permissions::from_mode(0o700))?;
    let directory = directory.to_path_buf();
    let (send, receive) = std::sync::mpsc::sync_channel(1);
    gpui_platform::headless().run(move |cx| {
        release_channel::init(semver::Version::new(1, 16, 2), cx);
        cx.spawn(async move |cx| {
            let deadline = cx.background_executor().timer(Duration::from_secs(35));
            let run = async {
                for headless in [false, true] {
                    let (builder, observation) = cx.update(|cx| {
                        cx.set_global(HeadlessTerminal(headless));
                        TerminalBuilder::new_with_native_cargo_runtime_task(
                            task::SpawnInTerminal { label: "Observe the selected Cargo runtime".into(), ..Default::default() },
                            None, runtime.clone(), 1, cx,
                        )
                    })?;
                    let builder = builder.await?;
                    let terminal = cx.update(|cx| cx.new(|cx| builder.subscribe(cx)));
                    let ordinary = terminal.read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx));
                    let (observed, ordinary) = futures::future::join(observation.finish(), ordinary).await;
                    let LinuxNativeTaskOutcome::Completed(status, Some(workload)) = observed? else {
                        anyhow::bail!("Cargo Task did not yield a checked runtime");
                    };
                    ensure!(status.success() && ordinary == Some(status), "Cargo runtime observation disagreed with its Terminal exit");
                    ensure!(
                        workload.environment().get("COPPER_RUNTIME_FIXTURE").is_some_and(|value| value == "runtime sentinel")
                            && workload.environment().get("LD_LIBRARY_PATH").is_some_and(|value| value.contains("fixture-loader")),
                        "Cargo Task did not transfer the observed user and loader environment"
                    );
                    ensure!(!directory.join("profiled-runtime-workload").exists(), "Observing the Cargo Task ran the selected workload");

                    let report = NativeProfileTaskReservation::new()?;
                    let profile_path = report.profile_path();
                    let symbols_path = report.symbols_path();
                    let template = samply_invocation(&directory, &profile_path, &[])?;
                    let environment = BTreeMap::from([
                        ("PATH".into(), "/usr/bin:/bin".into()),
                        ("COPPER_FIXTURE_SAMPLY".into(), template.program().to_str().context("Invalid collector path")?.into()),
                    ]);
                    let mut contaminated = environment.clone();
                    contaminated.insert("LD_LIBRARY_PATH".into(), "/incorrect-runtime-for-collector".into());
                    let contaminated = TaskBuildInvocation::new(
                        collector.clone(), template.arguments().to_vec(), directory.clone(), contaminated,
                    )?;
                    ensure!(
                        LinuxCargoProfileInvocation::new(contaminated, workload.clone(), runner.clone()).is_err(),
                        "The collector admitted a Cargo loader override"
                    );
                    let command = TaskBuildInvocation::new(
                        collector.clone(), template.arguments().to_vec(), directory.clone(), environment,
                    )?;
                    let invocation = LinuxCargoProfileInvocation::new(command, workload.clone(), runner.clone())?;
                    let (builder, observation) = cx.update(|cx| {
                        TerminalBuilder::new_with_native_cargo_profile_task(
                            task::SpawnInTerminal { label: "Profile the observed Cargo runtime".into(), ..Default::default() },
                            None, invocation, report, 1, cx,
                        )
                    })?;
                    let builder = builder.await?;
                    let terminal = cx.update(|cx| cx.new(|cx| builder.subscribe(cx)));
                    let ordinary = terminal.read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx));
                    let (captured, ordinary) = futures::future::join(observation.finish(), ordinary).await;
                    let LinuxNativeTaskOutcome::Completed(status, Some(profile)) = captured? else {
                        anyhow::bail!("The Cargo runtime profile did not produce a sealed pair");
                    };
                    ensure!(status.success() && ordinary == Some(status), "Cargo runtime capture disagreed with its Terminal exit");
                    let pair = cx.background_spawn(async move { profile.read() }).await?;
                    for private in [b"runtime sentinel".as_slice(), b"$FOO"] {
                        for artifact in [&pair.profile, &pair.symbols] {
                            ensure!(!artifact.windows(private.len()).any(|bytes| bytes == private), "The profile leaked a private workload value");
                        }
                    }
                    let samples = cx.background_spawn(async move { verify_samply_profile_bytes(pair) }).await?;
                    let marker = directory.join("profiled-runtime-workload");
                    ensure!(std::fs::read(&marker)? == b"verified runtime", "The selected workload did not verify its environment");
                    std::fs::remove_file(marker)?;
                    ensure!(!profile_path.exists() && !symbols_path.exists(), "The runtime profile kept its reservation");
                    let backend = if headless { "headless" } else { "PTY" };
                    println!("{backend} Cargo runtime profile retained {samples} samples and mapped stats::weight");
                }
                Ok(())
            };
            let result = match futures::future::select(Box::pin(run), deadline).await {
                futures::future::Either::Left((result, _)) => result,
                futures::future::Either::Right(_) => Err(anyhow::anyhow!("Cargo runtime profiling timed out")),
            };
            if let Err(error) = send.send(result) {
                log::error!("Cargo runtime profile fixture lost its caller: {error}");
            }
            cx.update(|cx| cx.quit());
        }).detach();
    });
    receive.recv()?
}

fn native_cargo_profile_terminal(directory: &Path, headless: bool) -> Result<()> {
    use gpui::{AppContext as _, UpdateGlobal as _};
    use project::{
        LocalProjectFlags, Project, cargo_workspace::CargoTargetKind,
        worktree_store::WorktreeStoreEvent,
    };
    use std::sync::Arc;
    use terminal::HeadlessTerminal;

    let directory = directory.canonicalize()?;
    write_native_cargo_fixture(&directory)?;
    std::fs::write(
        directory.join("runner/build.rs"),
        r#"fn main() {
    let directory = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"))
        .join("fixture-loader");
    std::fs::create_dir_all(&directory).expect("fixture loader");
    println!("cargo:rustc-link-search=native={}", directory.display());
}"#,
    )?;
    let runner = std::env::var_os("RUST_TOOLS_LINUX_RUNTIME_OBSERVER")
        .context("The separately packaged Linux Cargo observer is required")?;
    let collector = std::env::var_os("RUST_TOOLS_LINUX_SAMPLY")
        .context("The installed native Samply is required")?;
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
    let target = directory.join("target");
    std::fs::create_dir_all(&target)?;
    let cargo_home = directory.with_extension("cargo-home");
    std::fs::create_dir_all(&cargo_home)?;
    std::fs::write(directory.join(".gitignore"), "/target/\n")?;
    let environment = collections::HashMap::from_iter([
        ("PATH".into(), search_path),
        ("CARGO_NET_OFFLINE".into(), "true".into()),
        (
            "COPPER_PROJECT_PROFILE_FIXTURE".into(),
            "project-only runtime sentinel".into(),
        ),
        (
            "CARGO_HOME".into(),
            cargo_home.to_str().context("Invalid Cargo home")?.into(),
        ),
    ]);
    let lock = smol::block_on(
        smol::process::Command::new(&cargo)
            .args(["generate-lockfile", "--offline"])
            .current_dir(&directory)
            .env_clear()
            .envs(&environment)
            .output(),
    )?;
    ensure!(
        lock.status.success(),
        "Fixture Cargo lock preparation failed"
    );

    let (send, receive) = std::sync::mpsc::sync_channel(1);
    gpui_platform::headless().run(move |cx| {
        release_channel::init(semver::Version::new(1, 16, 2), cx);
        gpui_tokio::init(cx);
        crate::HeadlessProject::init(cx);
        cx.set_http_client(Arc::new(http_client::BlockedHttpClient));
        let opt_in = settings::SettingsStore::update_global(cx, |store, cx| {
            store.register_setting::<feature_flags::FeatureFlagsSettings>();
            store
                .set_user_settings(
                    r#"{"load_direnv":"disabled","feature_flags":{"rust-native-profiling-experimental":"on"}}"#,
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
        cx.set_global(HeadlessTerminal(headless));
        let client = client::Client::production(cx);
        client::init(&client, cx);
        Project::init(&client, cx);
        let user_store = cx.new(|cx| client::UserStore::new(client.clone(), cx));
        let project = Project::local(
            client,
            node_runtime::NodeRuntime::unavailable(),
            user_store,
            Arc::new(language::LanguageRegistry::new(cx.background_executor().clone())),
            Arc::new(fs::RealFs::new(None, cx.background_executor().clone())),
            Some(environment),
            LocalProjectFlags { watch_global_configs: false, ..Default::default() },
            cx,
        );
        let (indexed, index_published) = async_channel::bounded(1);
        let initial_scan = cx.subscribe(
            &project.read(cx).worktree_store(),
            move |_, event, _| {
                if let WorktreeStoreEvent::WorktreeUpdatedEntries(_, entries) = event
                    && entries.iter().any(|(path, _, _)| path.as_std_path() == Path::new("Cargo.toml"))
                    && let Err(error) = indexed.try_send(())
                {
                    log::debug!("Cargo fixture finished observing its initial index: {error}");
                }
            },
        );
        cx.spawn(async move |cx| {
            let deadline = cx.background_executor().timer(Duration::from_secs(60));
            let run = async {
                opt_in?;
                let (tree, _) = project.update(cx, |project, cx| {
                    project.find_or_create_worktree(&directory, true, cx)
                }).await.context("Opening the local Cargo Project")?;
                tree.read_with(cx, |tree, _| tree.as_local().context("Expected a local Cargo worktree").map(|tree| tree.scan_complete()))?.await;
                // The scan future can resolve before Cargo receives the initial index event.
                index_published.recv().await?;
                drop(initial_scan);
                // The first Tests discovery also refreshes the shared Cargo selection.
                let tests = project.read_with(cx, |project, _| project.rust_test_provider_store().clone());
                tests.update(cx, |tests, cx| tests.refresh(cx)).await.context("Finishing initial Rust test discovery")?;
                let store = project.read_with(cx, |project, _| project.cargo_workspace_store().clone());
                let snapshot = store.update(cx, |store, cx| store.refresh(cx)).await.context("Refreshing the Cargo Project")?;
                let workspace = snapshot.workspaces.first().context("Fixture Cargo workspace is absent")?;
                let member = workspace.members.iter().find(|member| member.name == "runner").context("Fixture runner package is absent")?;
                let selected = member.targets.iter().find(|target| target.name == "runner" && target.kind == CargoTargetKind::Binary).context("Fixture selected target is absent")?;
                let recipe = store.update(cx, |store, cx| {
                    store.prepare_native_cargo_build_recipe(
                        workspace.key.clone(), member.id.clone(), selected.clone(),
                        Some("release".into()), Default::default(), cx,
                    )
                })?.await.context("Preparing the selected Cargo recipe")?;
                let launch = task::SpawnInTerminal {
                    label: "Selected Cargo build".into(),
                    command: Some(recipe.program.to_str().context("Invalid Cargo program")?.to_owned()),
                    literal_command: true,
                    literal_prefix_args: recipe.arguments.len(),
                    args: recipe.arguments,
                    cwd: Some(recipe.working_directory),
                    env: recipe.task_environment,
                    required_experimental_features: vec!["rust-native-profiling-experimental".into()],
                    ..Default::default()
                };
                let run_id = uuid::Uuid::new_v4();
                let mut redirected = launch.clone();
                *redirected.args.get_mut(7).context("Cargo recipe is missing its target directory")? =
                    target.join("other-output").to_str().context("Invalid output path")?.into();
                let rejected = project.update(cx, |project, cx| {
                    project.start_selected_linux_native_cargo_build_task(
                        uuid::Uuid::new_v4(), redirected, None, task::MAX_TASK_STDOUT_REPORT_BYTES, cx,
                    )
                })?;
                ensure!(rejected.await.is_err(), "Project accepted a redirected Cargo build recipe");
                let build = project.update(cx, |project, cx| {
                    project.start_selected_linux_native_cargo_build_task(
                        run_id, launch, None, task::MAX_TASK_STDOUT_REPORT_BYTES, cx,
                    )
                })?.await.context("Launching the confirmed Linux Cargo build")?;
                let ordinary = build.terminal.read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx));
                let (workload, status) = futures::future::join(build.completion, ordinary).await;
                let workload = workload.context("Admitting the Cargo build output")?;
                ensure!(status.is_some_and(|status| status.success()), "The authenticated Cargo Task disagreed with the Terminal exit");
                let executable = workload.executable();
                let host = format!("{}-unknown-linux-gnu", std::env::consts::ARCH);
                let expected = target.join(format!("zed-native-profile-build-{}", run_id.simple()))
                    .join(host).join("release/runner");
                ensure!(
                    executable == expected && executable.is_file() && workload.working_directory() == directory,
                    "Project admitted the wrong Cargo output"
                );
                let backend = if headless { "headless" } else { "PTY" };
                println!("{backend} Project admitted the exact fresh Cargo output");
                let marker = target.join("profiled-project-workload");
                let preparation = project.update(cx, |project, cx| {
                    project.start_selected_linux_native_cargo_runtime_task(
                        workload,
                        runner.into(),
                        ["1200000", "literal two words", "", "$PROJECT_CARGO_PROFILE"].map(str::to_owned).to_vec(),
                        None,
                        cx,
                    )
                })?.await.context("Starting the Project-owned Cargo runtime preparation")?;
                let ordinary = preparation.terminal.read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx));
                let (runtime, status) = futures::future::join(preparation.completion, ordinary).await;
                let runtime = runtime.context("Reading the admitted Cargo runtime")?;
                ensure!(status.is_some_and(|status| status.success()), "Cargo runtime did not match its Terminal exit");
                ensure!(!marker.exists(), "Cargo runtime preparation executed the selected workload");
                let capture = project.update(cx, |project, cx| {
                    project.start_selected_linux_native_cargo_profile_task(runtime, collector.into(), None, cx)
                })?.await.context("Starting the Project-owned Samply capture")?;
                let ordinary = capture.terminal.read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx));
                let (profile, status) = futures::future::join(capture.completion, ordinary).await;
                let profile = profile.context("Verifying the Project-owned Samply capture")?;
                ensure!(status.is_some_and(|status| status.success()), "Samply capture did not match its Terminal exit");
                ensure!(std::fs::read(&marker)? == b"selected", "Samply did not run the admitted Cargo workload");
                let samples = cx.background_spawn(async move {
                    let recording = target.join(format!("selected-{}.json", run_id.simple()));
                    let symbols = recording.with_extension("syms.json");
                    let destination = terminal::NativeProfileTaskDestination::open([recording.clone(), symbols.clone()])?;
                    profile.retain(destination, || true)?;
                    let artifacts = native_task_helper::linux_task_completion::NativeProfileTaskArtifacts {
                        profile: std::fs::read(recording)?,
                        symbols: std::fs::read(symbols)?,
                    };
                    for private in [b"project-only runtime sentinel".as_slice(), b"$PROJECT_CARGO_PROFILE"] {
                        for artifact in [&artifacts.profile, &artifacts.symbols] {
                            ensure!(!artifact.windows(private.len()).any(|bytes| bytes == private), "The Project profile leaked a Cargo runtime value");
                        }
                    }
                    verify_samply_profile_bytes(artifacts)
                }).await?;
                println!("Selected Cargo build, runtime and capture retained {samples} samples and mapped stats::weight");
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
    receive.recv()?
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

#[derive(Clone, Copy)]
enum HeadlessTaskScenario {
    Coverage,
    SelectedCoverage,
    Profile(HeadlessProfileStage),
}

#[derive(Clone, Copy)]
enum HeadlessProfileStage {
    Build,
    Runtime,
    Capture,
    OccupiedDestination,
    RevokedCapture,
}

#[cfg(test)]
pub(super) fn write_remote_cargo_profile_fixture(directory: &Path) -> Result<String> {
    std::fs::create_dir_all(directory.join("target"))?;
    std::fs::write(directory.join(".gitignore"), "/target/\n")?;
    let mut settings: serde_json::Value = serde_json::from_str(
        &write_headless_profile_build_fixture(directory, HeadlessProfileStage::Capture)?,
    )?;
    let preset = &mut settings["cargo"]["presets"]["host-profile"];
    preset["trailing_args"] = serde_json::json!(["literal two words", "", "$HOST_ARGUMENT"]);
    preset["environment"]["COPPER_PROFILE_EXPECT_ARGS"] = serde_json::json!("yes");
    Ok(settings.to_string())
}

fn write_headless_profile_build_fixture(
    directory: &Path,
    stage: HeadlessProfileStage,
) -> Result<String> {
    write_native_cargo_fixture(directory)?;
    std::fs::write(
        directory.join("runner/src/main.rs"),
        r#"fn main() {
    assert_eq!(std::env::var("COPPER_PROFILE_BUILD_OWNER").as_deref(), Ok("host"));
    if std::env::var_os("COPPER_PROFILE_EXPECT_ARGS").is_some() {
        assert_eq!(std::env::args().skip(1).collect::<Vec<_>>(), ["literal two words", "", "$HOST_ARGUMENT"]);
    }
    if std::env::var_os("COPPER_PROFILE_TEST_WAIT").is_some() {
        std::fs::write("target/profile-workload-pid", std::process::id().to_string()).expect("profile workload pid");
        std::thread::sleep(std::time::Duration::from_secs(20));
    }
    std::fs::write("target/profiled-project-workload", "selected").expect("profile workload marker");
    let values: Vec<_> = (0..4096).collect();
    let total: u64 = (0..1_200_000).map(|_| stats::score(std::hint::black_box(&values))).sum();
    println!("Score: {total}");
}"#,
    )?;
    let cargo = std::env::var_os("RUST_TOOLS_NATIVE_BUILD_CARGO")
        .map(std::path::PathBuf::from)
        .context("The installed Linux Cargo fixture toolchain is required")?
        .canonicalize()?;
    let bins = cargo.parent().context("Cargo has no toolchain directory")?;
    let search = std::env::join_paths([bins, Path::new("/usr/bin"), Path::new("/bin")])?;
    let cargo_home = directory.with_extension("cargo-home");
    std::fs::create_dir_all(&cargo_home)?;
    let output = smol::block_on(
        smol::process::Command::new(&cargo)
            .args(["generate-lockfile", "--offline"])
            .current_dir(directory)
            .env("PATH", &search)
            .env("CARGO_HOME", &cargo_home)
            .output(),
    )?;
    ensure!(
        output.status.success(),
        "Could not generate the fixture's offline Cargo.lock"
    );
    std::fs::write(
        directory.join("runner/build.rs"),
        r#"fn main() {
    println!("cargo:rerun-if-env-changed=COPPER_PROFILE_BUILD_OWNER");
    assert_eq!(std::env::var("COPPER_PROFILE_BUILD_OWNER").as_deref(), Ok("host"));
    let marker = std::path::Path::new(&std::env::var_os("OUT_DIR").expect("build output"))
        .join("verified-host-preset");
    std::fs::write(marker, "host").expect("record the profile build environment");
}"#,
    )?;
    let collector = if matches!(
        stage,
        HeadlessProfileStage::Capture | HeadlessProfileStage::RevokedCapture
    ) {
        std::env::var("RUST_TOOLS_LINUX_SAMPLY")
            .context("The installed Linux Samply collector is required")?
    } else {
        "/usr/bin/samply".into()
    };
    let mut settings = serde_json::json!({
        "load_direnv": "disabled",
        "feature_flags": {
            "rust-native-profiling-experimental": "on",
            "rust-named-cargo-presets-experimental": "on"
        },
        "cargo": {
            "schema_version": 2,
            "presets": {
                "host-profile": {
                    "subcommand": "run",
                    "package": "runner",
                    "target_kind": "bin",
                    "target_name": "runner",
                    "profile": "release",
                    "working_directory": "workspace",
                    "environment": {
                        "PATH": search.to_str().context("Fixture PATH is not UTF-8")?,
                        "CARGO_HOME": cargo_home,
                        "CARGO_NET_OFFLINE": "true",
                        "COPPER_PROFILE_BUILD_OWNER": "host"
                    },
                    "native_profiler": {
                        "collector_path": collector,
                        "profile_path": "target/capture.json"
                    }
                }
            }
        }
    });
    if matches!(stage, HeadlessProfileStage::RevokedCapture) {
        settings["cargo"]["presets"]["host-profile"]["environment"]["COPPER_PROFILE_TEST_WAIT"] =
            serde_json::Value::String("yes".into());
    }
    Ok(settings.to_string())
}

async fn verify_headless_profile_build(
    host: &gpui::Entity<crate::HeadlessProject>,
    worktree_id: project::WorktreeId,
    directory: &Path,
    stage: HeadlessProfileStage,
    profile_client: (&rpc::AnyProtoClient, &AtomicU32),
    cx: &mut gpui::AsyncApp,
) -> Result<()> {
    use gpui::{AppContext as _, UpdateGlobal as _};
    use project::cargo_workspace::{CargoTargetKind, CargoWorkspaceKey};
    use project::cargo_workspace_store::HostCargoProfileSelection;
    use util::rel_path::RelPath;

    let tests = host.read_with(cx, |host, _| host.rust_test_provider_store.clone());
    tests
        .update(cx, |tests, cx| tests.refresh(cx))
        .await
        .context("Finishing the host's initial test discovery")?;
    let manifest = RelPath::from_unix_str("runner/Cargo.toml")?.into_arc();
    let select = |preset_id: &str, target_name: &str| HostCargoProfileSelection {
        workspace: CargoWorkspaceKey {
            root: project::ProjectPath {
                worktree_id,
                path: RelPath::empty_arc(),
            },
        },
        package_manifest: project::ProjectPath {
            worktree_id,
            path: manifest.clone(),
        },
        target_kind: CargoTargetKind::Binary,
        target_name: target_name.into(),
        preset_id: preset_id.into(),
    };
    ensure!(
        host.update(cx, |host, cx| host.build_selected_host_cargo_profile(
            select("client-only", "runner"),
            cx
        ))
        .is_err(),
        "The host accepted a preset that does not exist in its settings"
    );
    let rejected = host
        .update(cx, |host, cx| {
            host.build_selected_host_cargo_profile(select("host-profile", "decoy"), cx)
        })?
        .await;
    let error = rejected
        .err()
        .context("Host accepted a different target from its preset")?;
    ensure!(
        format!("{error:#}").contains("does not match the producing host's preset"),
        "Unexpected target refusal: {error:#}"
    );
    let workload = host
        .update(cx, |host, cx| {
            host.build_selected_host_cargo_profile(select("host-profile", "runner"), cx)
        })?
        .await?;
    let executable = workload.executable();
    let target_root = directory.join("target");
    let output = executable.strip_prefix(&target_root)?;
    let host_triple = format!("{}-unknown-linux-gnu", std::env::consts::ARCH);
    let run_directory = output
        .components()
        .next()
        .context("Host Build output has no run directory")?;
    ensure!(
        executable.is_file()
            && output.ends_with(Path::new(&host_triple).join("release/runner"))
            && run_directory
                .as_os_str()
                .to_string_lossy()
                .starts_with("zed-native-profile-build-")
            && workload.working_directory() == directory,
        "The producing host admitted an unexpected Cargo executable"
    );
    ensure!(
        !directory.join("target/profiled-project-workload").exists(),
        "The host Build executed the workload"
    );
    println!(
        "Registered headless Cargo store built and verified the exact host-selected profile executable"
    );
    if matches!(stage, HeadlessProfileStage::Build) {
        return Ok(());
    }
    let runtime = host
        .update(cx, |host, cx| {
            host.prepare_selected_host_cargo_profile_runtime(workload, cx)
        })?
        .await?;
    ensure!(
        !directory.join("target/profiled-project-workload").exists(),
        "The runtime observation executed the selected workload"
    );
    println!(
        "Registered headless host observed the real Cargo runtime using its embedded static helper"
    );
    if matches!(stage, HeadlessProfileStage::Runtime) {
        return Ok(());
    }
    let prepared = host
        .update(cx, |host, cx| {
            host.prepare_selected_host_cargo_profile_capture(runtime, cx)
        })?
        .await?;
    let paths = prepared.paths().clone();
    let absolute = paths
        .each_ref()
        .map(|path| directory.join(path.path.as_std_path()));
    ensure!(
        paths.iter().all(|path| path.worktree_id == worktree_id)
            && absolute
                .iter()
                .all(|path| path.starts_with(&target_root) && !path.exists())
            && !directory.join("target/profiled-project-workload").exists(),
        "Preparation executed the workload or produced unapproved files"
    );
    if matches!(stage, HeadlessProfileStage::OccupiedDestination) {
        std::fs::write(&absolute[1], b"existing symbols")?;
        let error = host
            .update(cx, |host, cx| {
                host.capture_selected_host_cargo_profile(prepared, cx)
            })?
            .await
            .err()
            .context("Native capture accepted an occupied destination")?;
        ensure!(
            error
                .downcast_ref::<std::io::Error>()
                .is_some_and(|error| error.kind() == std::io::ErrorKind::AlreadyExists)
                && std::fs::read(&absolute[1])? == b"existing symbols"
                && !absolute[0].exists()
                && !target_root.join("profiled-project-workload").exists(),
            "A new destination occupant was overwritten or the workload started: {error:#}"
        );
        println!("Registered headless host refused the newly occupied destination before capture");
        return Ok(());
    }
    let capture = host.update(cx, |host, cx| {
        host.capture_selected_host_cargo_profile(prepared, cx)
    })?;
    if matches!(stage, HeadlessProfileStage::RevokedCapture) {
        let pid_path = target_root.join("profile-workload-pid");
        let running = async {
            loop {
                match std::fs::read_to_string(&pid_path) {
                    Ok(contents) if !contents.is_empty() => break Ok(contents.parse::<u32>()?),
                    Ok(_) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(anyhow::Error::from(error)),
                }
                cx.background_executor()
                    .timer(Duration::from_millis(10))
                    .await;
            }
        };
        let (pid, capture) = match futures::future::select(Box::pin(running), capture).await {
            futures::future::Either::Left((pid, capture)) => (pid?, capture),
            futures::future::Either::Right((result, _)) => {
                result?;
                anyhow::bail!("Native capture finished before host settings could be revoked");
            }
        };
        cx.update(|cx| {
            settings::SettingsStore::update_global(cx, |store, cx| {
                store.set_local_server_settings(r#"{"load_direnv":"disabled"}"#, cx)
            })
        })?;
        let error = capture
            .await
            .err()
            .context("Revoked native capture succeeded")?;
        ensure!(
            format!("{error:#}").contains("approval was revoked on the host"),
            "Unexpected capture failure: {error:#}"
        );
        let process = Path::new("/proc").join(pid.to_string());
        for _ in 0..500 {
            if !process.exists() {
                break;
            }
            cx.background_executor()
                .timer(Duration::from_millis(10))
                .await;
        }
        ensure!(
            !process.exists() && absolute.iter().all(|path| !path.exists()),
            "Revoked native capture left a workload or a retained output"
        );
        println!("Host settings revocation stopped the running workload and retained no profile");
        return Ok(());
    }
    let capture = capture.await?;
    ensure!(
        capture.paths() == &paths
            && absolute[0]
                == target_root.join(format!("capture-{}.json", capture.run_id().simple()))
            && absolute[1] == absolute[0].with_extension("syms.json")
            && std::fs::read(target_root.join("profiled-project-workload"))? == b"selected",
        "The producing host did not run the selected workload and retain its reviewed pair"
    );
    use std::os::unix::fs::MetadataExt as _;
    for (absolute, version) in absolute.iter().zip(capture.versions()) {
        let current = std::fs::symlink_metadata(absolute)?;
        ensure!(
            current.is_file()
                && current.ino() == version.ino()
                && current.len() == version.len()
                && current.mode() & 0o777 == 0o600,
            "A retained native recording file is unbound or not owner-only"
        );
    }
    let [profile, symbols] = absolute.clone();
    let samples = cx
        .background_spawn(async move {
            verify_samply_profile_bytes(
                native_task_helper::linux_task_completion::NativeProfileTaskArtifacts {
                    profile: std::fs::read(profile)?,
                    symbols: std::fs::read(symbols)?,
                },
            )
        })
        .await?;
    println!(
        "Registered headless host retained the reviewed pair with {samples} samples and mapped stats::weight"
    );
    let store = host.read_with(cx, |host, _| host.profile_results_store.clone());
    verify_bound_headless_profile(&store, profile_client, capture, absolute, samples, cx).await?;
    Ok(())
}

async fn verify_bound_headless_profile(
    store: &gpui::Entity<project::profile_results_store::ProfileResultsStore>,
    (client, sender): (&rpc::AnyProtoClient, &AtomicU32),
    capture: project::cargo_workspace_store::CapturedHostCargoProfile,
    absolute: [std::path::PathBuf; 2],
    samples: u64,
    cx: &mut gpui::AsyncApp,
) -> Result<()> {
    use project::profile_results_wire::{PROFILE_PROTOCOL_VERSION, ProfilePageAssembler};
    use rpc::proto::{self, ProfileResultStatus};
    let peer = proto::PeerId { owner_id: 0, id: 1 };
    let run_id = capture.run_id().to_string();
    let paths = capture.paths().clone();
    store
        .update(cx, |store, cx| {
            store.register_linux_task_profile(peer, capture, cx)
        })?
        .await?;
    let status = client
        .request(proto::GetProfileTaskBinding {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            protocol_version: PROFILE_PROTOCOL_VERSION,
            task_run_id: run_id.clone(),
        })
        .await?;
    ensure!(
        status.complete && status.status == ProfileResultStatus::Ready as i32,
        "The host did not register its completed Task"
    );
    let request = proto::ImportProfileResults {
        project_id: proto::REMOTE_SERVER_PROJECT_ID,
        import_id: 10,
        protocol_version: PROFILE_PROTOCOL_VERSION,
        profile_path: Some(paths[0].to_proto()),
        symbols_path: Some(paths[1].to_proto()),
        consent_to_transfer: true,
        artifact_format: "samply-0.13.1".into(),
        task_run_id: Some(run_id),
    };
    let deny = |page: proto::ProfileResultPage, status: ProfileResultStatus| -> Result<()> {
        ensure!(
            page.status == status as i32
                && page.frames.is_empty()
                && page.nodes.is_empty()
                && page.summary.is_none(),
            "A denied Linux Task import returned the wrong status or a profile payload: {:?}",
            page.status()
        );
        Ok(())
    };
    deny(
        client
            .request(proto::ImportProfileResults {
                consent_to_transfer: false,
                ..request.clone()
            })
            .await?,
        ProfileResultStatus::Restricted,
    )?;
    deny(
        client
            .request(proto::ImportProfileResults {
                profile_path: Some(paths[1].to_proto()),
                symbols_path: Some(paths[0].to_proto()),
                ..request.clone()
            })
            .await?,
        ProfileResultStatus::Stale,
    )?;
    sender.store(2, Ordering::Release);
    let foreign = client.request(request.clone()).await;
    sender.store(peer.id, Ordering::Release);
    deny(foreign?, ProfileResultStatus::Stale)?;
    deny(
        client.forward_request(peer, request.clone()).await?,
        ProfileResultStatus::Restricted,
    )?;

    let page = client.request(request.clone()).await?;
    let cancellation = Default::default();
    let mut assembler = ProfilePageAssembler::new(page.import_id, page.generation)?;
    let permits = |path: &project::ProjectPath| path.worktree_id == paths[0].worktree_id;
    assembler.push(page, &cancellation, permits)?;
    let result = assembler.finish(&cancellation, permits)?;
    ensure!(
        result.total_samples == samples
            && result
                .frames
                .iter()
                .any(|frame| frame.label == "stats::weight"
                    && frame
                        .source
                        .as_ref()
                        .is_some_and(|(path, line)| path.path.as_unix_str() == "stats/src/lib.rs"
                            && *line == 7)),
        "The verified Task import did not map the sampled Rust source"
    );

    std::fs::write(&absolute[1], b"replacement symbols")?;
    let changed = client
        .request(proto::ImportProfileResults {
            import_id: 11,
            ..request
        })
        .await?;
    ensure!(
        matches!(
            changed.status(),
            ProfileResultStatus::Stale | ProfileResultStatus::Error
        ) && changed.frames.is_empty()
            && changed.nodes.is_empty()
            && changed.summary.is_none(),
        "A replaced Task artifact returned a profile payload"
    );
    println!(
        "Registered Linux Task import returned only normalized source-mapped results to its consented native peer"
    );
    Ok(())
}

fn headless_task(directory: &Path, scenario: HeadlessTaskScenario) -> Result<()> {
    use futures::StreamExt as _;
    use gpui::{AppContext as _, UpdateGlobal as _};
    use project::ProjectPath;
    use project::cargo_workspace::{CargoTargetKind, CargoWorkspaceKey};
    use project::cargo_workspace_store::CargoCoverageSelection;
    use project::worktree_store::{WorktreeIdCounter, WorktreeStore, WorktreeStoreEvent};
    use std::sync::Arc;
    use util::rel_path::RelPath;

    let directory = std::fs::canonicalize(directory)?;
    let run_id = uuid::Uuid::new_v4();
    let artifact_path = format!("target/zed-coverage-{}.json", run_id.simple());
    let profile_settings = if let HeadlessTaskScenario::Profile(stage) = scenario {
        Some(write_headless_profile_build_fixture(&directory, stage)?)
    } else {
        None
    };
    let launch = if matches!(scenario, HeadlessTaskScenario::SelectedCoverage) {
        write_collector_fixture(&directory)?;
        std::fs::write(directory.join("src/bin/second.rs"), "fn main() {}\n")?;
        None
    } else if profile_settings.is_some() {
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
    if profile_settings.is_none() {
        std::fs::write(&logical_report, b"unrelated stale report")?;
        std::fs::write(
            directory.join("expected-coverage.json"),
            include_str!("../../project/test_data/source_coverage/llvm3-run.json").replace(
                "/workspace",
                directory.to_str().context("Non-Unicode fixture root")?,
            ),
        )?;
    }
    let group = headless_report_group(artifact_path);

    let (send, receive) = std::sync::mpsc::sync_channel(1);
    gpui_platform::headless().run(move |cx| {
        release_channel::init(semver::Version::new(1, 16, 2), cx);
        gpui_tokio::init(cx);
        crate::HeadlessProject::init(cx);
        let opt_in = settings::SettingsStore::update_global(cx, |store, cx| {
            store.register_setting::<feature_flags::FeatureFlagsSettings>();
            if let Some(settings) = profile_settings.as_ref() {
                store.set_local_server_settings(settings, cx)
            } else {
                store.set_user_settings(
                    r#"{"load_direnv":"disabled","feature_flags":{"rust-coverage-experimental":"on"}}"#,
                    cx,
                )
                .result().map(|_| ())
            }
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
        let (scoped, scoped_receiver) = futures::channel::mpsc::unbounded();
        let profile_sender = Arc::new(AtomicU32::new(1));
        let bridge = cx.background_spawn({
            let profile_sender = profile_sender.clone();
            async move {
                let mut incoming = incoming_receiver;
                while let Some(message) = incoming.next().await {
                    let sender = proto::PeerId { owner_id: 0, id: profile_sender.load(Ordering::Acquire) };
                    if scoped.unbounded_send((sender, message)).is_err() { break; }
                }
            }
        });
        let session = remote::RemoteClient::proto_client_from_scoped_channels(
            scoped_receiver,
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
        let (indexed, index_published) = async_channel::bounded(1);
        let initial_scan = profile_settings.is_some().then(|| {
            let worktrees = host.read(cx).worktree_store.clone();
            cx.subscribe(&worktrees, move |_, event, _| {
                if let WorktreeStoreEvent::WorktreeUpdatedEntries(_, entries) = event
                    && entries
                        .iter()
                        .any(|(path, _, _)| path.as_std_path() == Path::new("Cargo.toml"))
                    && let Err(error) = indexed.try_send(())
                {
                    log::debug!("Cargo fixture finished observing its initial index: {error}");
                }
            })
        });
        cx.spawn(async move |cx| {
            let _bridge = bridge;
            let deadline = cx.background_executor().timer(Duration::from_secs(
                if profile_settings.is_some() { 60 } else { 20 },
            ));
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
                if let HeadlessTaskScenario::Profile(profile_stage) = scenario {
                    index_published.recv().await?;
                    drop(initial_scan);
                    stage.set("building the host-selected profile workload");
                    let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id());
                    return verify_headless_profile_build(
                        &host, worktree_id, &directory, profile_stage, (&client, &profile_sender), cx,
                    ).await;
                }
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
                    "Headless Task fixture timed out while {}",
                    stage.get()
                )),
            };
            if let Err(error) = &result {
                eprintln!("Headless Task fixture failed: {error:#}");
            }
            if let Err(error) = send.send(result) {
                log::error!("Headless Task fixture lost its caller: {error}");
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
    RegisteredHeadlessProfileBuild,
    RegisteredHeadlessProfileRuntime,
    RegisteredHeadlessProfileCapture,
    RegisteredHeadlessProfileOccupiedDestination,
    RegisteredHeadlessProfileRevokedCapture,
    NativeCargoRuntimeTransition,
    NativeCargoRuntimeProfile,
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
        Scenario::NativeCargoRuntimeTransition => native_cargo_runtime_transition(directory, false),
        Scenario::NativeCargoRuntimeProfile => native_cargo_runtime_transition(directory, true),
        Scenario::HeadlessCollector => headless_task(directory, HeadlessTaskScenario::Coverage),
        Scenario::SelectedHeadlessCollector => {
            headless_task(directory, HeadlessTaskScenario::SelectedCoverage)
        }
        Scenario::RegisteredHeadlessProfileBuild => headless_task(
            directory,
            HeadlessTaskScenario::Profile(HeadlessProfileStage::Build),
        ),
        Scenario::RegisteredHeadlessProfileRuntime => headless_task(
            directory,
            HeadlessTaskScenario::Profile(HeadlessProfileStage::Runtime),
        ),
        Scenario::RegisteredHeadlessProfileCapture => headless_task(
            directory,
            HeadlessTaskScenario::Profile(HeadlessProfileStage::Capture),
        ),
        Scenario::RegisteredHeadlessProfileOccupiedDestination => headless_task(
            directory,
            HeadlessTaskScenario::Profile(HeadlessProfileStage::OccupiedDestination),
        ),
        Scenario::RegisteredHeadlessProfileRevokedCapture => headless_task(
            directory,
            HeadlessTaskScenario::Profile(HeadlessProfileStage::RevokedCapture),
        ),
    }
}
