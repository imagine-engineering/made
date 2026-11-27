use anyhow::{Result, ensure};
use futures::io::AsyncReadExt as _;
use native_task_helper::{
    linux_coverage_completion::CoverageTaskCompletion, task_actor::OriginalTaskActor,
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
    TaskBuildInvocation::new(
        "/bin/sh".into(),
        vec![
            "-c".into(),
            shell.into(),
            "fixture".into(),
            String::new(),
            "two words".into(),
        ],
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
                .observe(actor, invocation(directory, shell)?)
                .await?;
            let actual = receipt.verify_exit(wait(&mut helper).await?)?;
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
            .observe(actor, invocation(directory, "touch report")?)
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

#[derive(Clone, Copy, clap::ValueEnum)]
pub enum Scenario {
    TerminalChildExit,
    ObserverDrop,
    ForeignChild,
    OversizedInvocation,
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
    }
}
