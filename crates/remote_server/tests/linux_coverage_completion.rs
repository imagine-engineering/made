#![cfg(all(
    target_os = "linux",
    feature = "rust-tools-process-test",
    feature = "test-support"
))]

use std::{io, process::Command};

fn run_fixture(scenario: &str) -> io::Result<()> {
    let directory = tempfile::tempdir()?;
    let output = Command::new(env!("CARGO_BIN_EXE_remote_server"))
        .arg("coverage-completion-fixture")
        .arg("--directory")
        .arg(directory.path())
        .arg(scenario)
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

#[test]
fn coverage_helper_reports_the_original_workload_exit() -> io::Result<()> {
    run_fixture("terminal-child-exit")
}

#[test]
fn coverage_helper_reaps_the_workload_when_the_observer_drops() -> io::Result<()> {
    run_fixture("observer-drop")
}

#[test]
fn coverage_listener_refuses_a_different_child_of_the_same_image() -> io::Result<()> {
    run_fixture("foreign-child")
}

#[test]
fn coverage_helper_refuses_an_oversized_invocation() -> io::Result<()> {
    run_fixture("oversized-invocation")
}

#[test]
fn coverage_helper_refuses_a_different_launcher_image() -> io::Result<()> {
    use native_task_helper::linux_coverage_completion::CoverageTaskCompletion;

    let completion = CoverageTaskCompletion::bind()?;
    let output = Command::new(env!("CARGO_BIN_EXE_remote_server"))
        .arg("observe-coverage-task")
        .arg("--socket")
        .arg(completion.socket())
        .output()?;
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("Coverage Task launch owner image did not match the helper")
    );
    Ok(())
}
