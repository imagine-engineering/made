use std::{fs, process::Command};

use anyhow::{Context as _, Result, bail, ensure};
use clap::{Parser, ValueEnum};
use semver::Version;
use serde::Serialize;

const TAG_PREFIX: &str = "rust-v";

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum BumpType {
    Patch,
    Minor,
    Major,
}

#[derive(Debug, Parser)]
pub struct ReleaseVersionArgs {
    /// Commit to tag. It must be a promoted commit on origin/release.
    #[arg(long)]
    commit: String,
    /// Permanent initial release-branch SHA; this commit can never be released.
    #[arg(long)]
    bootstrap_commit: String,
    /// Semantic-version component to increment for a new release.
    #[arg(long, value_enum, default_value_t = BumpType::Patch)]
    bump: BumpType,
    /// Exact X.Y.Z version for manual recovery.
    #[arg(long)]
    explicit_version: Option<Version>,
    /// Existing rust-vX.Y.Z tag supplied by a tag-push event.
    #[arg(long)]
    existing_tag: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ReleaseTag {
    name: String,
    version: Version,
    commit_sha: String,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
struct ReleaseDecision {
    commit_sha: String,
    version: Version,
    tag: String,
    reuse_existing_tag: bool,
    previous_tag: Option<String>,
}

pub fn run(args: ReleaseVersionArgs) -> Result<()> {
    let commit_expression = format!("{}^{{commit}}", args.commit);
    let commit_sha = git_output(["rev-parse", "--verify", &commit_expression])?;
    ensure_release_commit(&commit_sha, &args.bootstrap_commit)?;

    let tags = load_release_tags()?;
    if !tags.iter().any(|tag| tag.commit_sha == commit_sha)
        && let Some(previous_tag) = tags.last()
    {
        ensure!(
            git_is_ancestor(&previous_tag.commit_sha, &commit_sha)?,
            "release commit {commit_sha} precedes or diverges from latest tag {}",
            previous_tag.name
        );
    }
    let initial_version = initial_product_version()?;
    let decision = resolve_release(
        &commit_sha,
        initial_version,
        &tags,
        args.bump,
        args.explicit_version,
        args.existing_tag.as_deref(),
    )?;

    println!("{}", serde_json::to_string(&decision)?);
    Ok(())
}

fn ensure_release_commit(commit_sha: &str, bootstrap_sha: &str) -> Result<()> {
    ensure!(
        bootstrap_sha.len() == 40 && bootstrap_sha.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "bootstrap commit must be a full Git SHA"
    );
    let bootstrap_sha = git_output([
        "rev-parse",
        "--verify",
        &format!("{bootstrap_sha}^{{commit}}"),
    ])?;
    ensure!(
        commit_sha != bootstrap_sha && git_is_ancestor(&bootstrap_sha, commit_sha)?,
        "release commit must strictly descend from the permanent bootstrap commit"
    );
    let release_history = git_output(["rev-list", "--first-parent", "origin/release"])?;
    ensure!(
        release_history.lines().any(|commit| commit == commit_sha),
        "release commit {commit_sha} is not on the first-parent history of origin/release"
    );
    Ok(())
}

fn load_release_tags() -> Result<Vec<ReleaseTag>> {
    let pattern = format!("{TAG_PREFIX}*");
    let names = git_output(["tag", "--list", &pattern])?;
    let mut tags = Vec::new();
    for name in names.lines().filter(|name| !name.is_empty()) {
        let version = parse_tag(name)?;
        let commit_expression = format!("{name}^{{commit}}");
        let commit_sha = git_output(["rev-parse", "--verify", &commit_expression])?;
        tags.push(ReleaseTag {
            name: name.to_owned(),
            version,
            commit_sha,
        });
    }
    tags.sort_by(|left, right| left.version.cmp(&right.version));
    Ok(tags)
}

fn initial_product_version() -> Result<Version> {
    let manifest = fs::read_to_string("crates/zed/Cargo.toml")
        .context("failed to read crates/zed/Cargo.toml")?;
    let manifest: toml::Value =
        toml::from_str(&manifest).context("failed to parse crates/zed/Cargo.toml")?;
    let version = manifest
        .get("package")
        .and_then(|package| package.get("version"))
        .and_then(toml::Value::as_str)
        .context("crates/zed/Cargo.toml has no package.version")?;
    parse_version(version)
}

fn resolve_release(
    commit_sha: &str,
    initial_version: Version,
    tags: &[ReleaseTag],
    bump: BumpType,
    explicit_version: Option<Version>,
    existing_tag: Option<&str>,
) -> Result<ReleaseDecision> {
    validate_stable_version(&initial_version)?;
    if let Some(version) = explicit_version.as_ref() {
        validate_stable_version(version)?;
    }
    let requested_tag = existing_tag.map(parse_tag).transpose()?;
    let tags_at_commit = tags
        .iter()
        .filter(|tag| tag.commit_sha == commit_sha)
        .collect::<Vec<_>>();
    ensure!(
        tags_at_commit.len() <= 1,
        "release commit {commit_sha} has multiple rust-v* tags"
    );

    let previous_tag = tags.last();
    if let Some(tag) = tags_at_commit.first() {
        if let Some(requested_tag) = requested_tag.as_ref() {
            ensure!(
                requested_tag == &tag.version,
                "requested tag does not match the tag already attached to {commit_sha}"
            );
        }
        if let Some(explicit_version) = explicit_version.as_ref() {
            ensure!(
                explicit_version == &tag.version,
                "explicit version does not match the tag already attached to {commit_sha}"
            );
        }
        return Ok(ReleaseDecision {
            commit_sha: commit_sha.to_owned(),
            version: tag.version.clone(),
            tag: tag.name.clone(),
            reuse_existing_tag: true,
            previous_tag: previous_tag.map(|tag| tag.name.clone()),
        });
    }

    if existing_tag.is_some() {
        bail!("the tag-push release tag does not point to {commit_sha}");
    }

    let version = if let Some(version) = explicit_version {
        version
    } else if let Some(previous_tag) = previous_tag {
        bump_version(&previous_tag.version, bump)?
    } else {
        initial_version
    };
    validate_stable_version(&version)?;
    if let Some(previous_tag) = previous_tag {
        ensure!(
            version > previous_tag.version,
            "release version {version} must be newer than {}",
            previous_tag.version
        );
    }

    let tag = format!("{TAG_PREFIX}{version}");
    ensure!(
        !tags.iter().any(|existing| existing.name == tag),
        "release tag {tag} already points to another commit"
    );

    Ok(ReleaseDecision {
        commit_sha: commit_sha.to_owned(),
        version,
        tag,
        reuse_existing_tag: false,
        previous_tag: previous_tag.map(|tag| tag.name.clone()),
    })
}

#[allow(
    clippy::disallowed_methods,
    reason = "This synchronous git check runs only in the xtask CLI"
)]
fn git_is_ancestor(ancestor: &str, descendant: &str) -> Result<bool> {
    let status = Command::new("git")
        .args(["merge-base", "--is-ancestor", ancestor, descendant])
        .status()
        .context("failed to verify release history")?;
    match status.code() {
        Some(0) => Ok(true),
        Some(1) => Ok(false),
        _ => bail!("git merge-base failed while verifying release history"),
    }
}

fn bump_version(version: &Version, bump: BumpType) -> Result<Version> {
    let next = match bump {
        BumpType::Patch => Version::new(
            version.major,
            version.minor,
            version
                .patch
                .checked_add(1)
                .context("patch version overflow")?,
        ),
        BumpType::Minor => Version::new(
            version.major,
            version
                .minor
                .checked_add(1)
                .context("minor version overflow")?,
            0,
        ),
        BumpType::Major => Version::new(
            version
                .major
                .checked_add(1)
                .context("major version overflow")?,
            0,
            0,
        ),
    };
    Ok(next)
}

fn parse_tag(tag: &str) -> Result<Version> {
    let version = tag
        .strip_prefix(TAG_PREFIX)
        .with_context(|| format!("release tag must start with {TAG_PREFIX}: {tag}"))?;
    parse_version(version)
}

fn parse_version(version: &str) -> Result<Version> {
    let version = Version::parse(version)
        .with_context(|| format!("release version must use X.Y.Z syntax: {version}"))?;
    validate_stable_version(&version)?;
    Ok(version)
}

fn validate_stable_version(version: &Version) -> Result<()> {
    ensure!(
        version.pre.is_empty() && version.build.is_empty(),
        "release version must not contain prerelease or build metadata: {version}"
    );
    Ok(())
}

#[allow(
    clippy::disallowed_methods,
    reason = "This synchronous git command runs only in the xtask CLI"
)]
fn git_output<I, S>(args: I) -> Result<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    let output = Command::new("git")
        .args(args)
        .output()
        .context("failed to run git")?;
    ensure!(
        output.status.success(),
        "git command failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    String::from_utf8(output.stdout)
        .context("git command returned non-UTF-8 output")
        .map(|output| output.trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn version(value: &str) -> Version {
        Version::parse(value).expect("test version must be valid")
    }

    fn tag(value: &str, commit_sha: &str) -> ReleaseTag {
        ReleaseTag {
            name: format!("{TAG_PREFIX}{value}"),
            version: version(value),
            commit_sha: commit_sha.to_owned(),
        }
    }

    #[test]
    fn first_release_uses_application_version() -> Result<()> {
        let decision = resolve_release("new", version("1.16.2"), &[], BumpType::Patch, None, None)?;
        assert_eq!(decision.version, version("1.16.2"));
        assert_eq!(decision.tag, "rust-v1.16.2");
        Ok(())
    }

    #[test]
    fn automatic_and_manual_bumps_are_semantic() -> Result<()> {
        let tags = [tag("1.16.2", "old")];
        for (bump, expected) in [
            (BumpType::Patch, "1.16.3"),
            (BumpType::Minor, "1.17.0"),
            (BumpType::Major, "2.0.0"),
        ] {
            let decision = resolve_release("new", version("1.16.2"), &tags, bump, None, None)?;
            assert_eq!(decision.version, version(expected));
        }
        Ok(())
    }

    #[test]
    fn rerun_reuses_the_existing_commit_tag() -> Result<()> {
        let tags = [tag("1.16.2", "same")];
        let decision = resolve_release(
            "same",
            version("1.16.2"),
            &tags,
            BumpType::Patch,
            None,
            Some("rust-v1.16.2"),
        )?;
        assert!(decision.reuse_existing_tag);
        assert_eq!(decision.tag, "rust-v1.16.2");
        Ok(())
    }

    #[test]
    fn manual_explicit_version_is_used() -> Result<()> {
        let tags = [tag("1.16.2", "old")];
        let decision = resolve_release(
            "new",
            version("1.16.2"),
            &tags,
            BumpType::Patch,
            Some(version("1.20.0")),
            None,
        )?;
        assert_eq!(decision.version, version("1.20.0"));
        assert_eq!(decision.tag, "rust-v1.20.0");
        Ok(())
    }

    #[test]
    fn rejects_tag_push_for_a_different_commit() {
        let tags = [tag("1.16.2", "old")];
        let error = resolve_release(
            "new",
            version("1.16.2"),
            &tags,
            BumpType::Patch,
            None,
            Some("rust-v1.16.2"),
        )
        .expect_err("tag from a different commit must fail");
        assert!(error.to_string().contains("does not point"));
    }

    #[test]
    fn rejects_conflicting_explicit_version() {
        let tags = [tag("1.16.2", "same")];
        let error = resolve_release(
            "same",
            version("1.16.2"),
            &tags,
            BumpType::Patch,
            Some(version("1.17.0")),
            None,
        )
        .expect_err("conflicting version must fail");
        assert!(error.to_string().contains("does not match"));
    }

    #[test]
    fn rejects_duplicate_tags_for_one_commit() {
        let tags = [tag("1.16.2", "same"), tag("1.16.3", "same")];
        let error = resolve_release(
            "same",
            version("1.16.2"),
            &tags,
            BumpType::Patch,
            None,
            None,
        )
        .expect_err("ambiguous tags must fail");
        assert!(error.to_string().contains("multiple"));
    }

    #[test]
    fn rejects_non_stable_versions_and_tags() {
        assert!(parse_version("1.2.3-alpha.1").is_err());
        assert!(parse_tag("v1.2.3").is_err());
        assert!(parse_tag("rust-v1.2").is_err());
    }
}
