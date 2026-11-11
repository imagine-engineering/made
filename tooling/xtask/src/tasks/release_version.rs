#![allow(
    clippy::disallowed_methods,
    reason = "synchronous release planning CLI"
)]

use std::process::Command;

use anyhow::{Context as _, Result, ensure};
use clap::{Parser, ValueEnum};
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
    /// Full catalog release or explicitly limited Copper macOS preparation.
    #[arg(long, value_enum, default_value = "full")]
    preparation_scope: PreparationScope,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum PreparationScope {
    Full,
    MacosOnly,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub release_gate_variable: Option<String>,
    pub required_ci_workflows: Vec<String>,
    pub preparation_scope: PreparationScope,
    pub matrix: serde_json::Value,
}

pub fn run(args: ReleaseVersionArgs) -> Result<()> {
    let manifest = ProductManifest::load()?;
    let mut decision = resolve_tag_for_scope(&manifest, &args.tag, args.preparation_scope)?;
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
    let mut tagged_decision =
        resolve_tag_for_scope(&tagged_manifest, &args.tag, args.preparation_scope)?;
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

#[cfg(test)]
fn resolve_tag(manifest: &ProductManifest, tag: &str) -> Result<ReleaseDecision> {
    resolve_tag_for_scope(manifest, tag, PreparationScope::Full)
}

pub fn resolve_tag_for_scope(
    manifest: &ProductManifest,
    tag: &str,
    preparation_scope: PreparationScope,
) -> Result<ReleaseDecision> {
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
        preparation_scope == PreparationScope::Full || product.id == "rust",
        "macos-only preparation is qualified only for Copper"
    );
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
    if preparation_scope == PreparationScope::MacosOnly {
        rows.retain(|row| row["platform"] == "macos");
    }
    ensure!(
        !rows.is_empty(),
        "product has no targets for the selected scope"
    );
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
        preparation_scope,
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
    fn macos_only_is_explicit_and_does_not_change_full_release_targets() -> Result<()> {
        let mut manifest = ProductManifest::load()?;
        let decision =
            resolve_tag_for_scope(&manifest, "copper-v0.1.0", PreparationScope::MacosOnly)?;
        let rows = decision.matrix["include"].as_array().context("matrix")?;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["target"], "aarch64-apple-darwin");
        assert_eq!(decision.preparation_scope, PreparationScope::MacosOnly);
        assert!(decision.release_gate_variable.is_none());
        assert!(
            resolve_tag(&manifest, "copper-v0.1.0")?
                .release_gate_variable
                .is_none()
        );
        assert_eq!(
            resolve_tag(&manifest, "copper-v0.1.0")?.matrix["include"]
                .as_array()
                .context("matrix")?
                .len(),
            3
        );
        let copper = manifest
            .products
            .iter_mut()
            .find(|product| product.id == "rust")
            .context("Copper")?;
        copper
            .targets
            .retain(|target| !target.starts_with("macos-"));
        assert!(
            resolve_tag_for_scope(&manifest, "copper-v0.1.0", PreparationScope::MacosOnly).is_err()
        );
        assert!(
            resolve_tag_for_scope(&manifest, "jvm-v0.1.0", PreparationScope::MacosOnly).is_err()
        );
        assert!(resolve_tag(&manifest, "rust-v0.1.0").is_err());
        Ok(())
    }

    #[test]
    fn tag_controls_product_version_and_channel() -> Result<()> {
        let manifest = ProductManifest::load()?;
        for (tag, channel) in [
            ("copper-v0.1.0", "stable"),
            ("copper-v2.3.4-rc.2", "preview"),
            ("copper-v2.3.4-beta.1+build.5", "preview"),
        ] {
            let decision = resolve_tag(&manifest, tag)?;
            assert_eq!(decision.product_id, "rust");
            assert_eq!(decision.update_namespace, "rust");
            assert_eq!(decision.channel, channel);
            assert_eq!(
                decision.version.to_string(),
                tag.trim_start_matches("copper-v")
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
        let decision = resolve_tag(&manifest, "copper-v4.5.6-rc.1")?;
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
            .release_tag_prefix = "copper-v".into();
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
            "copper-v01.2.3",
            "copper-v1.2",
            "copper-v1.2.3-01",
            "copper-v1.2.3\n",
            "copper-v65536.0.0",
            "jvm-v1.0.0",
            "game-v1.0.0",
        ] {
            assert!(resolve_tag(&manifest, tag).is_err(), "accepted {tag}");
        }
        manifest.products.push(manifest.product("rust")?.clone());
        assert!(resolve_tag(&manifest, "copper-v1.0.0").is_err());
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
        assert_eq!(resolve_tag(&manifest, "copper-v0.1.0")?.product_id, "rust");
        Ok(())
    }
}
