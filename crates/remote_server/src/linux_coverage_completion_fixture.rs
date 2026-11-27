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
    HeadlessCollector,
    SelectedHeadlessCollector,
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
        Scenario::HeadlessCollector => headless_collector(directory, false),
        Scenario::SelectedHeadlessCollector => headless_collector(directory, true),
    }
}
