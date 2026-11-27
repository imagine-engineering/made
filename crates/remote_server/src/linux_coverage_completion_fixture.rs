use anyhow::{Context as _, Result, ensure};
use futures::io::AsyncReadExt as _;
use native_task_helper::{
    linux_coverage_completion::{CoverageTaskCompletion, CoverageTaskReportReservation},
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

fn spawn_helper(completion: &CoverageTaskCompletion) -> io::Result<smol::process::Child> {
    let mut command = smol::process::Command::new(std::env::current_exe()?);
    command
        .arg("observe-coverage-task")
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
            "Coverage helper did not exit",
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
            let completion = CoverageTaskCompletion::bind()?;
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
        let completion = CoverageTaskCompletion::bind()?;
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
        let unused = CoverageTaskCompletion::bind()?;
        let completion = CoverageTaskCompletion::bind()?;
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
            .arg("observe-coverage-task")
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
            let completion = CoverageTaskCompletion::bind()?;
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

fn collector_terminal(directory: &Path) -> Result<()> {
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

    let cargo = std::env::var_os("RUST_TOOLS_COVERAGE_CARGO")
        .context("The independently installed Cargo is required")?;
    let report = CoverageTaskReportReservation::new(task::MAX_TASK_ARTIFACT_BYTES as usize)?;
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
    let invocation = TaskBuildInvocation::new(
        cargo.into(),
        vec![
            "llvm-cov".into(),
            "--json".into(),
            "--output-path".into(),
            report
                .path()
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
    )?;
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
}

pub fn run(directory: &Path, scenario: Scenario) -> Result<()> {
    ensure!(
        directory.is_dir(),
        "Coverage fixture requires an existing directory"
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
    }
}
