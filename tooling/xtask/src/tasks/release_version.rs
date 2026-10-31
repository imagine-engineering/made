#![allow(
    clippy::disallowed_methods,
    reason = "synchronous release planning CLI"
)]

use std::process::Command;

use anyhow::{Context as _, Result, ensure};
use clap::Parser;
use semver::Version;
use serde::Serialize;

use crate::product_manifest::{ProductManifest, ProductStatus};

#[derive(Debug, Parser)]
pub struct ReleaseVersionArgs {
    /// An existing product release tag. Never creates or moves a tag.
    #[arg(long)]
    tag: String,
    /// Print the catalog plan without resolving Git history (non-publishing validation only).
    #[arg(long)]
    plan_only: bool,
}

#[derive(Debug, Serialize)]
pub struct ReleaseDecision {
    pub commit_sha: String,
    pub product_id: String,
    pub display_name: String,
    pub update_namespace: String,
    pub version: Version,
    pub tag: String,
    pub channel: String,
    pub prerelease: bool,
    pub release_gate_variable: String,
    pub required_ci_workflows: Vec<String>,
    pub matrix: serde_json::Value,
}

pub fn run(args: ReleaseVersionArgs) -> Result<()> {
    let manifest = ProductManifest::load()?;
    let mut decision = resolve_tag(&manifest, &args.tag)?;
    if args.plan_only {
        println!("{}", serde_json::to_string(&decision)?);
        return Ok(());
    }
    let commit_sha = git_output(&[
        "rev-parse",
        "--verify",
        &format!("refs/tags/{}^{{commit}}", args.tag),
    ])?;
    let status = Command::new("git")
        .args(["merge-base", "--is-ancestor", &commit_sha, "origin/main"])
        .status()
        .context("checking main ancestry")?;
    ensure!(status.success(), "tag commit is not on origin/main history");
    let tagged_manifest = ProductManifest::from_commit(&commit_sha)?;
    let current_product = manifest.product(&decision.product_id)?;
    let tagged_product = tagged_manifest.product(&decision.product_id)?;
    ensure!(
        tagged_product
            .targets
            .iter()
            .all(|target| current_product.targets.contains(target)),
        "tagged product requires a platform that is no longer qualified"
    );
    let mut tagged_decision = resolve_tag(&tagged_manifest, &args.tag)?;
    ensure!(
        tagged_decision.product_id == decision.product_id,
        "tag product mapping changed"
    );
    tagged_decision.release_gate_variable = decision.release_gate_variable;
    tagged_decision
        .required_ci_workflows
        .extend(decision.required_ci_workflows);
    tagged_decision.required_ci_workflows.sort();
    tagged_decision.required_ci_workflows.dedup();
    tagged_decision.commit_sha = commit_sha;
    decision = tagged_decision;
    println!("{}", serde_json::to_string(&decision)?);
    Ok(())
}

pub fn resolve_tag(manifest: &ProductManifest, tag: &str) -> Result<ReleaseDecision> {
    let matches = manifest
        .products
        .iter()
        .filter(|product| tag.starts_with(&product.release_tag_prefix))
        .collect::<Vec<_>>();
    ensure!(
        matches.len() == 1,
        "unknown or ambiguous product release tag prefix: {tag}"
    );
    let product = matches.first().context("missing product")?;
    ensure!(
        product.status == ProductStatus::Enabled && !product.targets.is_empty(),
        "product {} is not qualified for release",
        product.id
    );
    let version = Version::parse(
        tag.strip_prefix(&product.release_tag_prefix)
            .context("tag prefix")?,
    )
    .context("release tag must contain a valid semantic version")?;
    ensure!(
        [version.major, version.minor, version.patch]
            .into_iter()
            .all(|part| part <= u16::MAX.into()),
        "version exceeds platform package numeric limits"
    );
    let prerelease = !version.pre.is_empty();
    let channel = if prerelease { "preview" } else { "stable" };
    let mut package_product = (*product).clone();
    package_product.display_name =
        super::bundle::channel_display_name(&product.display_name, channel);
    package_product.bundle_identifier =
        super::bundle::channel_identifier(&product.bundle_identifier, channel, ".");
    package_product.data_namespace =
        super::bundle::channel_identifier(&product.data_namespace, channel, "-");
    package_product.windows_installer_id =
        super::bundle::channel_windows_installer_id(&product.windows_installer_id, channel)?;
    let mut rows = super::workflows::release::product_matrix(&package_product);
    for row in &mut rows {
        let platform = row["platform"].as_str().context("platform")?.to_owned();
        let target = row["target"].as_str().context("target")?;
        let architecture = target.split('-').next().context("architecture")?;
        let extension = match platform.as_str() {
            "linux" => "tar.gz",
            "macos" => "dmg",
            "windows" => "exe",
            _ => anyhow::bail!("unsupported platform"),
        };
        let name = product.render_name(
            &product.artifact_name,
            &version.to_string(),
            &platform,
            architecture,
            extension,
        )?;
        row["artifact_name"] = serde_json::json!(name);
        row["artifact_path"] = serde_json::json!(format!(
            "target/products/{}/release/{name}{}",
            product.id,
            if platform == "linux" { "*" } else { "" }
        ));
    }
    Ok(ReleaseDecision {
        commit_sha: String::new(),
        product_id: product.id.clone(),
        display_name: product.display_name.clone(),
        update_namespace: product.update_namespace.clone(),
        version,
        tag: tag.to_owned(),
        channel: channel.into(),
        prerelease,
        release_gate_variable: product.release_gate_variable.clone(),
        required_ci_workflows: product.required_ci_workflows.clone(),
        matrix: serde_json::json!({"include": rows}),
    })
}

fn git_output(arguments: &[&str]) -> Result<String> {
    let output = Command::new("git").args(arguments).output()?;
    ensure!(
        output.status.success(),
        "existing release tag could not be resolved: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tag_controls_product_version_and_channel() -> Result<()> {
        let manifest = ProductManifest::load()?;
        for (tag, channel) in [
            ("rust-v0.1.0", "stable"),
            ("rust-v2.3.4-rc.2", "preview"),
            ("rust-v2.3.4-beta.1+build.5", "preview"),
        ] {
            let decision = resolve_tag(&manifest, tag)?;
            assert_eq!(decision.product_id, "rust");
            assert_eq!(decision.update_namespace, "rust");
            assert_eq!(decision.channel, channel);
            assert_eq!(
                decision.version.to_string(),
                tag.trim_start_matches("rust-v")
            );
            assert_eq!(
                decision.matrix["include"]
                    .as_array()
                    .context("matrix")?
                    .len(),
                3
            );
        }
        Ok(())
    }

    #[test]
    fn artifact_templates_and_preview_identities_come_from_catalog() -> Result<()> {
        let mut manifest = ProductManifest::load()?;
        let product = manifest
            .products
            .iter_mut()
            .find(|product| product.id == "rust")
            .context("Copper")?;
        product.artifact_name = "{product}-custom-{version}-{platform}-{arch}.{extension}".into();
        let decision = resolve_tag(&manifest, "rust-v4.5.6-rc.1")?;
        for row in decision.matrix["include"].as_array().context("matrix")? {
            assert!(
                row["artifact_name"]
                    .as_str()
                    .context("artifact")?
                    .starts_with("rust-custom-4.5.6-rc.1-")
            );
            assert_eq!(row["bundle_identifier"], "dev.ideflavors.rust.preview");
            assert_eq!(row["data_namespace"], "ide-rust-preview");
            assert_eq!(row["display_name"], "Copper Preview");
        }
        manifest
            .products
            .iter_mut()
            .find(|product| product.id == "jvm")
            .context("Groovy")?
            .release_tag_prefix = "rust-v".into();
        assert!(
            manifest
                .validate(&crate::product_manifest::workspace_root())
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn rejects_unknown_ambiguous_invalid_and_unqualified_products() -> Result<()> {
        let mut manifest = ProductManifest::load()?;
        for tag in [
            "v0.1.0",
            "rust-v01.2.3",
            "rust-v1.2",
            "rust-v1.2.3-01",
            "rust-v1.2.3\n",
            "rust-v65536.0.0",
            "jvm-v1.0.0",
            "game-v1.0.0",
        ] {
            assert!(resolve_tag(&manifest, tag).is_err(), "accepted {tag}");
        }
        manifest.products.push(manifest.product("rust")?.clone());
        assert!(resolve_tag(&manifest, "rust-v1.0.0").is_err());
        Ok(())
    }

    #[test]
    fn catalog_keeps_products_independent_even_if_future_groovy_is_qualified() -> Result<()> {
        let mut manifest = ProductManifest::load()?;
        let rust_targets = manifest.product("rust")?.targets.clone();
        let groovy = manifest
            .products
            .iter_mut()
            .find(|product| product.id == "jvm")
            .context("Groovy")?;
        groovy.status = ProductStatus::Enabled;
        groovy.targets = rust_targets;
        let decision = resolve_tag(&manifest, "jvm-v3.2.1-rc.1")?;
        assert_eq!(decision.product_id, "jvm");
        assert_eq!(decision.update_namespace, "jvm");
        assert!(
            decision.matrix["include"]
                .as_array()
                .context("matrix")?
                .iter()
                .all(|row| row["product"] == "jvm")
        );
        assert_eq!(resolve_tag(&manifest, "rust-v0.1.0")?.product_id, "rust");
        Ok(())
    }
}
