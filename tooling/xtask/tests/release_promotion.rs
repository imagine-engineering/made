#![allow(
    clippy::disallowed_methods,
    reason = "These CLI integration tests synchronously exercise Git, Bash, and xtask in isolated repositories"
)]

use std::{fs, path::Path, process::Command};

use anyhow::{Context as _, Result, ensure};
use serde_json::Value;
use tempfile::TempDir;

struct Promotion {
    directory: TempDir,
    bootstrap: String,
    promoted: String,
    development: String,
}

fn git(directory: &Path, arguments: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .current_dir(directory)
        .args(arguments)
        .output()?;
    ensure!(
        output.status.success(),
        "git {arguments:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

impl Promotion {
    fn new() -> Result<Self> {
        let directory = tempfile::tempdir()?;
        let path = directory.path();
        git(path, &["init", "--initial-branch=main"])?;
        git(path, &["config", "user.name", "Release test"])?;
        git(path, &["config", "user.email", "release@example.invalid"])?;
        fs::create_dir_all(path.join("crates/zed"))?;
        fs::write(
            path.join("crates/zed/Cargo.toml"),
            "[package]\nname = 'zed'\nversion = '1.0.0'\n",
        )?;
        git(path, &["add", "."])?;
        git(path, &["commit", "-m", "Before migration"])?;
        git(
            path,
            &["commit", "--allow-empty", "-m", "Bootstrap migration"],
        )?;
        let bootstrap = git(path, &["rev-parse", "HEAD"])?;
        git(path, &["branch", "release"])?;
        git(path, &["commit", "--allow-empty", "-m", "Development"])?;
        let development = git(path, &["rev-parse", "HEAD"])?;
        git(path, &["checkout", "release"])?;
        git(path, &["merge", "--no-ff", "main", "-m", "Promote main"])?;
        let promoted = git(path, &["rev-parse", "HEAD"])?;
        // The local remote exercises the same fetch and ancestry commands without publishing.
        git(path, &["remote", "add", "origin", "."])?;
        git(
            path,
            &[
                "fetch",
                "origin",
                "+refs/heads/release:refs/remotes/origin/release",
            ],
        )?;
        Ok(Self {
            directory,
            bootstrap,
            promoted,
            development,
        })
    }

    fn version(&self, commit: &str, bootstrap: &str) -> Result<std::process::Output> {
        Ok(Command::new(env!("CARGO_BIN_EXE_xtask"))
            .current_dir(self.directory.path())
            .args([
                "release-version",
                "--commit",
                commit,
                "--bootstrap-commit",
                bootstrap,
            ])
            .output()?)
    }
}

#[test]
fn release_version_rejects_bootstrap_and_development_and_reuses_promotion() -> Result<()> {
    let promotion = Promotion::new()?;
    let path = promotion.directory.path();
    for commit in [
        &promotion.bootstrap,
        &promotion.development,
        &git(path, &["rev-parse", "main~1~1"])?,
    ] {
        let output = promotion.version(commit, &promotion.bootstrap)?;
        ensure!(
            !output.status.success(),
            "ineligible commit passed: {commit}"
        );
    }
    for bootstrap in [
        "",
        "main",
        &promotion.promoted,
        "0000000000000000000000000000000000000000",
    ] {
        ensure!(
            !promotion
                .version(&promotion.promoted, bootstrap)?
                .status
                .success()
        );
    }
    git(path, &["checkout", "main"])?;
    git(
        path,
        &["commit", "--allow-empty", "-m", "Unpromoted development"],
    )?;
    let unpromoted = git(path, &["rev-parse", "HEAD"])?;
    ensure!(
        !promotion
            .version(&unpromoted, &promotion.bootstrap)?
            .status
            .success()
    );
    let output = promotion.version(&promotion.promoted, &promotion.bootstrap)?;
    ensure!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let decision: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(decision["commit_sha"], promotion.promoted);
    assert_eq!(decision["tag"], "rust-v1.0.0");
    assert_eq!(git(path, &["tag", "--list"])?, "");
    git(path, &["tag", "rust-v1.0.0", &promotion.promoted])?;
    let retry = promotion.version(&promotion.promoted, &promotion.bootstrap)?;
    ensure!(retry.status.success());
    let retry: Value = serde_json::from_slice(&retry.stdout)?;
    assert_eq!(retry["tag"], decision["tag"]);
    assert_eq!(retry["reuse_existing_tag"], true);
    Ok(())
}

#[test]
fn explicit_one_zero_zero_does_not_inherit_upstream_package_version() -> Result<()> {
    let promotion = Promotion::new()?;
    fs::write(
        promotion.directory.path().join("crates/zed/Cargo.toml"),
        "[package]\nname = 'zed'\nversion = '1.16.2'\n",
    )?;
    git(
        promotion.directory.path(),
        &["add", "crates/zed/Cargo.toml"],
    )?;
    git(
        promotion.directory.path(),
        &["commit", "-m", "Upstream package metadata"],
    )?;
    let commit = git(promotion.directory.path(), &["rev-parse", "HEAD"])?;
    git(
        promotion.directory.path(),
        &[
            "fetch",
            "origin",
            "+refs/heads/release:refs/remotes/origin/release",
        ],
    )?;
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(promotion.directory.path())
        .args([
            "release-version",
            "--commit",
            &commit,
            "--bootstrap-commit",
            &promotion.bootstrap,
            "--explicit-version",
            "1.0.0",
        ])
        .output()?;
    ensure!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let decision: Value = serde_json::from_slice(&result.stdout)?;
    assert_eq!(decision["version"], "1.0.0");
    assert_eq!(decision["tag"], "rust-v1.0.0");
    assert_eq!(git(promotion.directory.path(), &["tag", "--list"])?, "");
    Ok(())
}

fn workflow() -> Result<serde_yaml::Value> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    Ok(serde_yaml::from_str(&fs::read_to_string(
        root.join(".github/workflows/release_promotion.yml"),
    )?)?)
}

#[test]
fn release_triggers_and_both_publication_gates_are_guarded() -> Result<()> {
    let workflow = workflow()?;
    assert_eq!(
        workflow["on"]["workflow_run"]["branches"],
        serde_yaml::to_value(["release"])?
    );
    let guard = workflow["jobs"]["prepare_release"]["if"]
        .as_str()
        .context("job guard")?;
    for required in [
        "COPPER_RELEASE_ENABLED == 'true'",
        "COPPER_RELEASE_BOOTSTRAP_SHA != ''",
        "head_branch == 'release'",
        "conclusion == 'success'",
        "head_repository.full_name == github.repository",
        "github.ref == 'refs/heads/release'",
        "head_sha != vars.COPPER_RELEASE_BOOTSTRAP_SHA",
    ] {
        ensure!(guard.contains(required), "missing {required}");
    }
    let mut scripts = Vec::new();
    for job in ["prepare_release", "publish_product"] {
        let steps = workflow["jobs"][job]["steps"]
            .as_sequence()
            .context("steps")?;
        let position = steps
            .iter()
            .position(|step| {
                step["run"]
                    .as_str()
                    .is_some_and(|script| script.contains("CI_RUNS="))
            })
            .context("promotion check")?;
        let version = steps
            .iter()
            .position(|step| {
                step["run"]
                    .as_str()
                    .is_some_and(|script| script.contains("ARGS=(release-version"))
            })
            .context("version check")?;
        ensure!(position < version);
        scripts.push(steps.get(position).context("guard step")?["run"].clone());
    }
    assert_eq!(scripts.first(), scripts.last());
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let ci: serde_yaml::Value = serde_yaml::from_str(&fs::read_to_string(
        root.join(".github/workflows/run_tests.yml"),
    )?)?;
    assert_eq!(
        ci["on"]["push"]["branches"],
        serde_yaml::to_value(["main", "release"])?
    );
    let retired: serde_yaml::Value = serde_yaml::from_str(&fs::read_to_string(
        root.join(".github/workflows/release.yml"),
    )?)?;
    assert_eq!(retired["name"].as_str(), Some("release"));
    assert_eq!(
        retired["on"],
        serde_yaml::from_str::<serde_yaml::Value>("workflow_dispatch: {}")?
    );
    assert_eq!(
        retired["permissions"],
        serde_yaml::from_str::<serde_yaml::Value>("contents: read")?
    );
    assert_eq!(
        retired["jobs"].as_mapping().context("retired jobs")?.len(),
        1
    );
    assert_eq!(retired["jobs"]["retired"]["if"].as_str(), Some("false"));
    assert_eq!(
        retired["jobs"]["retired"]["steps"],
        serde_yaml::from_str::<serde_yaml::Value>("- name: release::release\n  run: exit 1\n")?
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn release_promotion_gate_checks_real_history_and_ci_outcomes() -> Result<()> {
    use serde_json::json;
    use std::os::unix::fs::PermissionsExt;

    let promotion = Promotion::new()?;
    let path = promotion.directory.path();
    let workflow = workflow()?;
    let script = workflow["jobs"]["prepare_release"]["steps"]
        .as_sequence()
        .context("steps")?
        .iter()
        .filter_map(|step| step["run"].as_str())
        .find(|script| script.contains("CI_RUNS="))
        .context("guard")?;
    let executable_directory = path.join("bin");
    fs::create_dir(&executable_directory)?;
    let executable = executable_directory.join("gh");
    fs::write(
        &executable,
        "#!/bin/sh\nif [ \"$CI_API_FAIL\" = true ]; then exit 1; fi\nprintf '%s' \"$CI_FIXTURE\"\n",
    )?;
    fs::set_permissions(&executable, fs::Permissions::from_mode(0o755))?;
    let search_path = std::env::join_paths(std::iter::once(executable_directory).chain(
        std::env::split_paths(&std::env::var_os("PATH").context("PATH")?),
    ))?;
    let successful = json!({"id": 42, "path": ".github/workflows/run_tests.yml", "head_sha": promotion.promoted, "head_branch": "release", "event": "push", "head_repository": {"full_name": "owner/repository"}, "repository": {"full_name": "owner/repository"}, "status": "completed", "conclusion": "success"});
    let check = |commit: &str,
                 bootstrap: &str,
                 enabled: &str,
                 run_id: &str,
                 runs: Vec<Value>,
                 api_fail: bool|
     -> Result<bool> {
        Ok(Command::new("bash")
            .args(["-c", script])
            .current_dir(path)
            .env("PATH", &search_path)
            .env("COMMIT_SHA", commit)
            .env("BOOTSTRAP_SHA", bootstrap)
            .env("RELEASE_ENABLED", enabled)
            .env("GITHUB_REPOSITORY", "owner/repository")
            .env("CI_RUN_ID", run_id)
            .env(
                "CI_FIXTURE",
                serde_json::to_string(&json!([{"workflow_runs": runs}]))?,
            )
            .env("CI_API_FAIL", api_fail.to_string())
            .output()?
            .status
            .success())
    };
    for run_id in ["42", ""] {
        ensure!(check(
            &promotion.promoted,
            &promotion.bootstrap,
            "true",
            run_id,
            vec![successful.clone()],
            false
        )?);
        ensure!(!check(
            &promotion.bootstrap,
            &promotion.bootstrap,
            "true",
            run_id,
            vec![successful.clone()],
            false
        )?);
        ensure!(!check(
            &promotion.development,
            &promotion.bootstrap,
            "true",
            run_id,
            vec![successful.clone()],
            false
        )?);
    }
    for bootstrap in ["", "invalid", &promotion.promoted] {
        ensure!(!check(
            &promotion.promoted,
            bootstrap,
            "true",
            "",
            vec![successful.clone()],
            false
        )?);
    }
    ensure!(!check(
        &promotion.promoted,
        &promotion.bootstrap,
        "false",
        "",
        vec![successful.clone()],
        false
    )?);
    ensure!(!check(
        &promotion.promoted,
        &promotion.bootstrap,
        "true",
        "41",
        vec![successful.clone()],
        false
    )?);
    ensure!(!check(
        &promotion.promoted,
        &promotion.bootstrap,
        "true",
        "",
        vec![],
        false
    )?);
    ensure!(!check(
        &promotion.promoted,
        &promotion.bootstrap,
        "true",
        "",
        vec![successful.clone()],
        true
    )?);
    for (field, value) in [
        ("conclusion", json!("failure")),
        ("conclusion", json!("cancelled")),
        ("status", json!("in_progress")),
        ("event", json!("pull_request")),
        ("head_branch", json!("main")),
        ("head_sha", json!(promotion.development)),
        ("head_repository", json!({"full_name": "fork/repository"})),
        ("repository", json!({"full_name": "fork/repository"})),
        ("path", json!(".github/workflows/unrelated.yml")),
    ] {
        let mut rejected = successful.clone();
        rejected[field] = value;
        ensure!(
            !check(
                &promotion.promoted,
                &promotion.bootstrap,
                "true",
                "",
                vec![rejected.clone()],
                false
            )?,
            "accepted invalid {field}"
        );
        rejected["id"] = json!(43);
        ensure!(
            !check(
                &promotion.promoted,
                &promotion.bootstrap,
                "true",
                "",
                vec![rejected, successful.clone()],
                false
            )?,
            "ignored newer invalid {field}"
        );
    }
    Ok(())
}
