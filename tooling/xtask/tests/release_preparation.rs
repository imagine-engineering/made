#![allow(
    clippy::disallowed_methods,
    reason = "isolated Git fixture integration tests"
)]

use std::{fs, path::Path, process::Command};

use anyhow::{Context as _, Result, ensure};
use serde_json::Value;

fn git(directory: &Path, arguments: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .current_dir(directory)
        .args(arguments)
        .output()?;
    ensure!(
        output.status.success(),
        "git fixture failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

#[test]
fn existing_annotated_and_lightweight_tags_require_main_ancestry() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    git(root, &["init", "--initial-branch=main"])?;
    git(root, &["config", "user.name", "Release fixture"])?;
    git(root, &["config", "user.email", "fixture@example.invalid"])?;
    fs::write(root.join("source"), "fixture")?;
    fs::create_dir(root.join("products"))?;
    let catalog = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../products/flavors.toml"),
    )?
    .replacen(
        "display_name = \"Copper\"",
        "display_name = \"Tagged Copper\"",
        1,
    );
    fs::write(root.join("products/flavors.toml"), catalog)?;
    git(root, &["add", "source", "products/flavors.toml"])?;
    git(root, &["commit", "-m", "Main fixture"])?;
    let main = git(root, &["rev-parse", "HEAD"])?;
    git(root, &["update-ref", "refs/remotes/origin/main", &main])?;
    git(root, &["tag", "copper-v0.1.0"])?;
    git(
        root,
        &[
            "tag",
            "-a",
            "copper-v0.2.0-rc.1",
            "-m",
            "Prerelease fixture",
        ],
    )?;
    for tag in ["copper-v0.1.0", "copper-v0.2.0-rc.1"] {
        for scope in ["full", "macos-only"] {
            let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
                .current_dir(root)
                .args([
                    "release-version",
                    "--tag",
                    tag,
                    "--preparation-scope",
                    scope,
                ])
                .output()?;
            ensure!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let decision: Value = serde_json::from_slice(&output.stdout)?;
            assert!(decision.get("release_gate_variable").is_none());
            assert_eq!(decision["preparation_scope"], scope);
            assert_eq!(
                decision["matrix"]["include"]
                    .as_array()
                    .context("matrix")?
                    .len(),
                if scope == "full" { 3 } else { 1 }
            );
            assert_eq!(decision["commit_sha"], main);
            assert_eq!(decision["tag"], tag);
            assert_eq!(decision["product_id"], "rust");
            assert_eq!(decision["display_name"], "Tagged Copper");
            let plan = Command::new(env!("CARGO_BIN_EXE_xtask"))
                .current_dir(root)
                .env(
                    "RELEASE_VERSION",
                    decision["version"].as_str().context("version")?,
                )
                .args([
                    "bundle",
                    "--product",
                    "rust",
                    "--platform",
                    "macos",
                    "--target",
                    "aarch64-apple-darwin",
                    "--channel",
                    decision["channel"].as_str().context("channel")?,
                    "--catalog-commit",
                    &main,
                    "--dry-run",
                ])
                .output()?;
            ensure!(
                plan.status.success(),
                "{}",
                String::from_utf8_lossy(&plan.stderr)
            );
            let plan: Value = serde_json::from_slice(&plan.stdout)?;
            assert_eq!(plan["macos_minimum_system_version"], "13.3");
            assert!(
                plan["display_name"]
                    .as_str()
                    .context("display name")?
                    .starts_with("Tagged Copper")
            );
            #[cfg(unix)]
            {
                fs::create_dir_all(root.join("target/debug"))?;
                fs::copy(env!("CARGO_BIN_EXE_xtask"), root.join("target/debug/xtask"))?;
                fs::create_dir_all(root.join("release-bundle-plans"))?;
                fs::write(
                    root.join("release-bundle-plans/decision.json"),
                    serde_json::to_string(&decision)?,
                )?;
                let script =
                    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../script/product_release.py");
                let output = Command::new("python3")
                    .current_dir(root)
                    .arg(&script)
                    .arg("bundle-plans")
                    .output()?;
                ensure!(
                    output.status.success(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
                for row in decision["matrix"]["include"].as_array().context("matrix")? {
                    let plan: Value = serde_json::from_str(&fs::read_to_string(
                        root.join("release-bundle-plans")
                            .join(row["plan_file"].as_str().context("plan file")?),
                    )?)?;
                    assert_eq!(plan["artifact_name"], row["artifact_name"]);
                    assert_eq!(plan["version"], decision["version"]);
                    assert_eq!(plan["channel"], decision["channel"]);
                    assert!(
                        plan["display_name"]
                            .as_str()
                            .context("display")?
                            .starts_with("Tagged Copper")
                    );
                }
                let output = Command::new("python3")
                    .current_dir(root)
                    .env("GITHUB_OUTPUT", root.join("qualification-output"))
                    .arg(&script)
                    .args([
                        "contract",
                        "--development",
                        "--version",
                        decision["version"].as_str().context("version")?,
                        "--commit",
                        &main,
                        "--preparation-scope",
                        scope,
                    ])
                    .output()?;
                ensure!(
                    output.status.success(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
                let development: Value = serde_json::from_str(&fs::read_to_string(
                    root.join("release-bundle-plans/decision.json"),
                )?)?;
                assert_eq!(development["tag"], "");
                assert_eq!(development["signing"], "off");
                let rows = development["matrix"]["include"]
                    .as_array()
                    .context("matrix")?;
                let tagged_rows = decision["matrix"]["include"].as_array().context("matrix")?;
                assert_eq!(rows.len(), tagged_rows.len());
                for (row, tagged_row) in rows.iter().zip(tagged_rows) {
                    for key in [
                        "target",
                        "platform",
                        "artifact_name",
                        "application_features",
                        "remote_features",
                    ] {
                        assert_eq!(row[key], tagged_row[key]);
                    }
                }
                let output = Command::new("python3")
                    .current_dir(root)
                    .arg(&script)
                    .arg("bundle-plans")
                    .output()?;
                ensure!(
                    output.status.success(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
                for row in development["matrix"]["include"]
                    .as_array()
                    .context("matrix")?
                {
                    let plan: Value = serde_json::from_str(&fs::read_to_string(
                        root.join("release-bundle-plans")
                            .join(row["plan_file"].as_str().context("plan file")?),
                    )?)?;
                    assert_eq!(plan["signing"], "off");
                    assert_eq!(plan["version"], development["version"]);
                    assert_eq!(plan["artifact_name"], row["artifact_name"]);
                }
            }
        }
    }
    git(root, &["checkout", "-b", "not-main"])?;
    git(root, &["commit", "--allow-empty", "-m", "Unmerged fixture"])?;
    git(root, &["tag", "copper-v9.0.0"])?;
    for tag in [
        "copper-v9.0.0",
        "copper-v1.0.0",
        "jvm-v1.0.0",
        "unknown-v1.0.0",
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .current_dir(root)
            .args(["release-version", "--tag", tag])
            .output()?;
        assert!(!output.status.success(), "accepted {tag}");
    }
    assert_eq!(git(root, &["tag", "--list"])?.lines().count(), 3);
    Ok(())
}

#[test]
fn generated_release_workflow_preserves_permission_and_trigger_boundaries() -> Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for name in ["release", "release_promotion", "release_signing_check"] {
        assert!(!root.join(format!(".github/workflows/{name}.yml")).exists());
    }
    let workflow: Value = serde_yaml::from_str(&fs::read_to_string(
        root.join(".github/workflows/release_prepare.yml"),
    )?)?;
    assert!(workflow["on"]["release"].is_null());
    assert_eq!(
        workflow["on"]["push"]["tags"],
        serde_json::json!(["copper-v*"])
    );
    assert!(workflow["on"]["workflow_dispatch"]["inputs"]["release_tag"].is_object());
    for (name, job) in workflow["jobs"].as_object().context("jobs")? {
        assert_eq!(
            job["permissions"]["contents"],
            if name == "upload_release" {
                "write"
            } else {
                "read"
            }
        );
    }
    let website: Value = serde_yaml::from_str(&fs::read_to_string(
        root.join(".github/workflows/website_refresh.yml"),
    )?)?;
    assert_eq!(website["jobs"]["refresh_website"]["if"], "false");
    assert!(website["on"]["release"].is_null());
    assert!(website["on"]["push"].is_null());
    Ok(())
}

#[test]
fn only_the_release_workflow_has_release_write_permissions() -> Result<()> {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.github/workflows");
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.extension().is_none_or(|extension| extension != "yml") {
            continue;
        }
        let workflow: Value = serde_yaml::from_str(&fs::read_to_string(&path)?)?;
        if path
            .file_name()
            .is_some_and(|name| name == "release_prepare.yml")
        {
            continue;
        }
        assert_ne!(
            workflow["permissions"]["contents"],
            "write",
            "{}",
            path.display()
        );
        for job in workflow["jobs"].as_object().context("jobs")?.values() {
            assert_ne!(
                job["permissions"]["contents"],
                "write",
                "{}",
                path.display()
            );
        }
    }
    Ok(())
}
