#![allow(clippy::disallowed_methods, reason = "build scripts are exempt")]
use std::{path::Path, process::Command};

const ZED_MANIFEST: &str = include_str!("../zed/Cargo.toml");

fn main() {
    let zed_cargo_toml: cargo_toml::Manifest =
        toml::from_str(ZED_MANIFEST).expect("failed to parse zed Cargo.toml");
    println!(
        "cargo:rustc-env=ZED_PKG_VERSION={}",
        zed_cargo_toml.package.unwrap().version.unwrap()
    );
    println!(
        "cargo:rustc-env=TARGET={}",
        std::env::var("TARGET").unwrap()
    );

    // A worktree's .git is a file; watching a missing reflog rebuilds on every Cargo invocation.
    let mut git_paths = vec![
        "HEAD".to_string(),
        "logs/HEAD".to_string(),
        "packed-refs".to_string(),
    ];
    if let Some(branch) = git_output(&["symbolic-ref", "--quiet", "HEAD"]) {
        git_paths.push(branch);
    }
    for git_path in git_paths {
        if let Some(path) = git_output(&["rev-parse", "--git-path", &git_path]) {
            let path = Path::new(&path);
            if path.exists() {
                println!("cargo:rerun-if-changed={}", path.display());
            } else if git_path.starts_with("refs/")
                && let Some(parent) = path.ancestors().find(|parent| parent.is_dir())
            {
                // Committing a packed branch creates a loose ref even when reflogs are disabled.
                println!("cargo:rerun-if-changed={}", parent.display());
            }
        }
    }
    if let Some(output) = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
    {
        let git_sha = String::from_utf8_lossy(&output.stdout);
        let git_sha = git_sha.trim();

        println!("cargo:rustc-env=ZED_COMMIT_SHA={git_sha}");
    }
    if let Some(build_identifier) = option_env!("GITHUB_RUN_NUMBER") {
        println!("cargo:rustc-env=ZED_BUILD_ID={build_identifier}");
    }
}

fn git_output(arguments: &[&str]) -> Option<String> {
    match Command::new("git").args(arguments).output() {
        Ok(output) if output.status.success() => match String::from_utf8(output.stdout) {
            Ok(output) => Some(output.trim().to_string()),
            Err(error) => {
                eprintln!("Could not decode Git build inputs: {error}");
                None
            }
        },
        Ok(_) => None,
        Err(error) => {
            eprintln!("Could not query Git build inputs: {error}");
            None
        }
    }
}
