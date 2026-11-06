use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result, bail, ensure};
use serde::{Deserialize, Serialize};

pub const PRODUCT_MANIFEST_PATH: &str = "products/flavors.toml";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProductManifest {
    pub schema_version: u32,
    pub products: Vec<Product>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Product {
    pub id: String,
    pub status: ProductStatus,
    pub release_tag_prefix: String,
    pub release_gate_variable: String,
    #[serde(default)]
    pub macos_release_gate_variable: Option<String>,
    #[serde(default)]
    pub macos_minimum_system_version: Option<String>,
    pub required_ci_workflows: Vec<String>,
    pub display_name: String,
    pub initial_version: Option<String>,
    pub appearance: Option<ProductAppearance>,
    pub executable_name: String,
    pub bundle_identifier: String,
    pub url_scheme: String,
    pub icon_set: String,
    pub data_namespace: String,
    pub update_namespace: String,
    pub instance_port_offset: u16,
    pub windows_installer_id: String,
    pub cargo_features: Vec<String>,
    pub remote_server_features: Vec<String>,
    pub default_extensions: Vec<String>,
    pub default_language_servers: Vec<String>,
    pub toolchain_onboarding: String,
    pub agent_profile: String,
    pub agent_profile_name: String,
    pub agent_instructions: String,
    pub installer_name: String,
    pub artifact_name: String,
    pub targets: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ProductStatus {
    Enabled,
    Planned,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProductAppearance {
    pub theme: String,
    pub light: String,
    pub dark: String,
    pub font_family: String,
    pub font_asset: String,
}

impl ProductAppearance {
    fn validate(&self, root: &Path) -> Result<()> {
        validate_relative_path(&self.theme)?;
        validate_relative_path(&self.font_asset)?;
        ensure!(
            self.theme.starts_with("assets/themes/") && self.theme.ends_with(".json"),
            "product theme must be a bundled theme JSON asset"
        );
        ensure!(
            self.font_asset.starts_with("assets/fonts/")
                && self.font_asset.ends_with(".ttf")
                && root.join(&self.font_asset).is_file()
                && !self.font_family.trim().is_empty(),
            "product font must name a family and bundled TrueType asset"
        );
        let family: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(root.join(&self.theme)).context("reading product theme")?,
        )?;
        let themes = family["themes"]
            .as_array()
            .context("product theme has no variants")?;
        for (name, appearance) in [(&self.light, "light"), (&self.dark, "dark")] {
            ensure!(
                themes
                    .iter()
                    .any(|theme| theme["name"] == *name && theme["appearance"] == appearance),
                "product theme is missing {appearance} variant `{name}`"
            );
        }
        Ok(())
    }
}

impl ProductManifest {
    pub fn load() -> Result<Self> {
        let root = workspace_root();
        Self::load_from(&root.join(PRODUCT_MANIFEST_PATH), &root)
    }

    fn load_from(path: &Path, root: &Path) -> Result<Self> {
        let source = fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let manifest: Self = toml::from_str(&source)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        manifest.validate(root)?;
        Ok(manifest)
    }

    #[allow(
        clippy::disallowed_methods,
        reason = "read-only Git metadata in the release CLI"
    )]
    pub fn from_commit(commit: &str) -> Result<Self> {
        ensure!(
            commit.len() == 40 && commit.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "catalog commit must be a full Git SHA"
        );
        let output = std::process::Command::new("git")
            .args(["show", &format!("{commit}:{PRODUCT_MANIFEST_PATH}")])
            .output()?;
        ensure!(output.status.success(), "tagged product catalog is missing");
        let manifest: Self = toml::from_str(&String::from_utf8(output.stdout)?)
            .context("tagged catalog does not support the current release contract")?;
        manifest.validate_catalog(&workspace_root(), false)?;
        Ok(manifest)
    }

    pub fn validate(&self, root: &Path) -> Result<()> {
        self.validate_catalog(root, true)
    }

    fn validate_catalog(&self, root: &Path, check_assets: bool) -> Result<()> {
        ensure!(
            self.schema_version == 1,
            "unsupported product schema version {}",
            self.schema_version
        );
        ensure!(
            !self.products.is_empty(),
            "product catalog must not be empty"
        );

        for (index, product) in self.products.iter().enumerate() {
            for other in self.products.iter().skip(index + 1) {
                ensure!(
                    !product
                        .release_tag_prefix
                        .starts_with(&other.release_tag_prefix)
                        && !other
                            .release_tag_prefix
                            .starts_with(&product.release_tag_prefix),
                    "ambiguous release tag prefixes for {} and {}",
                    product.id,
                    other.id
                );
            }
        }
        let mut identities = BTreeSet::new();
        let mut instance_port_offsets = BTreeSet::new();
        for product in &self.products {
            product.validate(root, check_assets)?;
            ensure!(
                instance_port_offsets.insert(product.instance_port_offset),
                "duplicate product instance port offset `{}`",
                product.instance_port_offset
            );
            for (kind, value) in [
                ("id", &product.id),
                ("executable", &product.executable_name),
                ("bundle identifier", &product.bundle_identifier),
                ("URL scheme", &product.url_scheme),
                ("data namespace", &product.data_namespace),
                ("update namespace", &product.update_namespace),
            ] {
                ensure!(
                    identities.insert((kind, value.as_str())),
                    "duplicate product {kind} `{value}`"
                );
            }
        }

        for required_id in ["rust", "jvm", "game"] {
            ensure!(
                self.products
                    .iter()
                    .any(|product| product.id == required_id),
                "missing required product `{required_id}`"
            );
        }
        let rust = self.product("rust")?;
        ensure!(
            rust.status == ProductStatus::Enabled,
            "Rust product must be enabled in Phase 1"
        );
        ensure!(
            rust.cargo_features == ["agentic-tools", "rust-tools"],
            "Rust application features must be exactly agentic-tools,rust-tools"
        );
        ensure!(
            rust.remote_server_features == ["rust-tools"],
            "Rust remote-server features must be exactly rust-tools"
        );
        ensure!(
            rust.default_extensions.is_empty()
                && rust.default_language_servers == ["rust-analyzer"]
                && rust.toolchain_onboarding == "rustup",
            "Rust Phase 1 must use built-in rust-analyzer, no external defaults, and rustup onboarding"
        );
        for planned in ["jvm", "game"] {
            ensure!(
                self.product(planned)?.status == ProductStatus::Planned,
                "product `{planned}` must remain planned in Phase 1"
            );
        }
        Ok(())
    }

    pub fn product(&self, id: &str) -> Result<&Product> {
        self.products
            .iter()
            .find(|product| product.id == id)
            .with_context(|| format!("unknown product `{id}`"))
    }

    pub fn enabled_products(&self) -> impl Iterator<Item = &Product> {
        self.products
            .iter()
            .filter(|product| product.status == ProductStatus::Enabled)
    }
}

impl Product {
    fn validate(&self, root: &Path, check_assets: bool) -> Result<()> {
        ensure!(is_slug(&self.id), "invalid product ID `{}`", self.id);
        ensure!(
            self.release_tag_prefix.ends_with("-v")
                && is_slug(self.release_tag_prefix.trim_end_matches("-v")),
            "invalid release tag prefix"
        );
        ensure!(
            !self.release_gate_variable.is_empty()
                && self
                    .release_gate_variable
                    .bytes()
                    .all(|byte| byte.is_ascii_uppercase() || byte == b'_' || byte.is_ascii_digit()),
            "invalid release gate variable"
        );
        ensure!(
            !self.required_ci_workflows.is_empty(),
            "product requires exact-commit CI"
        );
        if let Some(variable) = &self.macos_release_gate_variable {
            ensure!(
                !variable.is_empty()
                    && variable != &self.release_gate_variable
                    && variable.bytes().all(|byte| byte.is_ascii_uppercase()
                        || byte == b'_'
                        || byte.is_ascii_digit()),
                "macOS release gate must be valid and distinct from full-platform qualification"
            );
        }
        if let Some(version) = &self.macos_minimum_system_version {
            let components: Vec<_> = version.split('.').collect();
            ensure!(
                (2..=3).contains(&components.len())
                    && components.iter().all(|component| !component.is_empty()
                        && component.bytes().all(|byte| byte.is_ascii_digit())
                        && component.parse::<u32>().is_ok()),
                "invalid minimum macOS version"
            );
        }
        for workflow in &self.required_ci_workflows {
            ensure!(
                workflow.ends_with(".yml")
                    && !workflow.contains('/')
                    && workflow
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || b"_-.".contains(&byte)),
                "invalid required CI workflow"
            );
        }
        ensure!(
            !self.display_name.trim().is_empty(),
            "product `{}` has an empty display name",
            self.id
        );
        for (kind, value) in [
            ("executable name", &self.executable_name),
            ("URL scheme", &self.url_scheme),
            ("data namespace", &self.data_namespace),
            ("update namespace", &self.update_namespace),
        ] {
            ensure!(
                is_slug(value),
                "product `{}` has invalid {kind} `{value}`",
                self.id
            );
        }
        ensure!(
            is_bundle_identifier(&self.bundle_identifier),
            "product `{}` has invalid bundle identifier `{}`",
            self.id,
            self.bundle_identifier
        );
        validate_relative_path(&self.icon_set)?;
        validate_relative_path(&self.agent_instructions)?;
        validate_template(&self.installer_name)?;
        validate_template(&self.artifact_name)?;
        ensure!(
            !self.agent_profile.is_empty(),
            "product `{}` has no agent profile",
            self.id
        );
        ensure!(
            !self.agent_profile_name.trim().is_empty(),
            "product `{}` has no agent profile name",
            self.id
        );
        ensure!(
            !self.toolchain_onboarding.is_empty(),
            "product `{}` has no onboarding handler",
            self.id
        );
        ensure!(
            (1_000..=10_000).contains(&self.instance_port_offset),
            "product `{}` has an unsafe instance port offset",
            self.id
        );
        let mut targets = BTreeSet::new();
        for target in &self.targets {
            ensure!(
                ["linux-x86_64", "macos-aarch64", "windows-x86_64"].contains(&target.as_str()),
                "product `{}` has unsupported target `{target}`",
                self.id
            );
            ensure!(
                targets.insert(target),
                "product `{}` has duplicate target `{target}`",
                self.id
            );
        }
        ensure!(
            self.windows_installer_id.starts_with('{')
                && self.windows_installer_id.ends_with('}')
                && uuid::Uuid::parse_str(&self.windows_installer_id).is_ok(),
            "product `{}` has an invalid Windows installer ID",
            self.id
        );
        if let Some(version) = &self.initial_version {
            let version =
                semver::Version::parse(version).context("invalid initial product version")?;
            ensure!(
                version.pre.is_empty() && version.build.is_empty(),
                "initial product version must be X.Y.Z"
            );
        }
        if self.status == ProductStatus::Enabled {
            ensure!(
                self.initial_version.is_some(),
                "enabled product must declare its initial version"
            );
            let appearance = self
                .appearance
                .as_ref()
                .context("enabled product must select its appearance")?;
            if check_assets {
                appearance.validate(root)?;
                for icon in [
                    "app-icon.svg",
                    "app-icon.png",
                    "app-icon@2x.png",
                    "app-icon.ico",
                ] {
                    ensure!(
                        root.join(&self.icon_set).join(icon).is_file(),
                        "product `{}` icon set is missing {icon}",
                        self.id
                    );
                }
                ensure!(
                    root.join(&self.agent_instructions).is_file(),
                    "product `{}` agent instructions are missing",
                    self.id
                );
            }
            ensure!(
                !self.targets.is_empty(),
                "enabled product `{}` has no targets",
                self.id
            );
        } else {
            ensure!(
                self.targets.is_empty(),
                "planned product `{}` cannot declare release targets",
                self.id
            );
        }
        Ok(())
    }

    pub fn render_name(
        &self,
        template: &str,
        version: &str,
        platform: &str,
        arch: &str,
        extension: &str,
    ) -> Result<String> {
        validate_template(template)?;
        Ok(template
            .replace("{product}", &self.id)
            .replace("{version}", version)
            .replace("{platform}", platform)
            .replace("{arch}", arch)
            .replace("{extension}", extension))
    }
}

pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
}

fn validate_relative_path(value: &str) -> Result<()> {
    let path = Path::new(value);
    ensure!(
        !path.is_absolute(),
        "product path must be repository-relative: `{value}`"
    );
    ensure!(
        path.components()
            .all(|component| matches!(component, std::path::Component::Normal(_))),
        "unsafe product path `{value}`"
    );
    Ok(())
}

fn validate_template(value: &str) -> Result<()> {
    ensure!(
        !value.is_empty(),
        "product filename template must not be empty"
    );
    ensure!(
        !value.contains('/') && !value.contains('\\'),
        "product filename template contains a path separator: `{value}`"
    );
    let mut rest = value;
    while let Some(start) = rest.find('{') {
        let after = &rest[start + 1..];
        let Some(end) = after.find('}') else {
            bail!("unclosed variable in filename template `{value}`")
        };
        let variable = &after[..end];
        ensure!(
            ["product", "version", "platform", "arch", "extension"].contains(&variable),
            "unknown filename variable `{{{variable}}}`"
        );
        rest = &after[end + 1..];
    }
    ensure!(
        !rest.contains('}'),
        "unmatched closing brace in filename template `{value}`"
    );
    ensure!(
        value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"-_.{}".contains(&byte)),
        "unsafe character in filename template `{value}`"
    );
    Ok(())
}

fn is_slug(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}

fn is_bundle_identifier(value: &str) -> bool {
    value.split('.').count() >= 3 && value.split('.').all(is_slug)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repository_catalog_is_valid() {
        ProductManifest::load().expect("repository product catalog should be valid");
    }

    #[test]
    fn macos_package_minimum_is_product_owned() -> Result<()> {
        let manifest = ProductManifest::load()?;
        let mut product = manifest.product("rust")?.clone();
        assert_eq!(
            product.macos_minimum_system_version.as_deref(),
            Some("13.3")
        );
        assert!(
            manifest
                .product("jvm")?
                .macos_minimum_system_version
                .is_none()
        );
        for version in [
            "",
            "13",
            "13..3",
            "13.3-beta",
            "13.3.0.0",
            "999999999999999999999.0",
        ] {
            product.macos_minimum_system_version = Some(version.into());
            assert!(
                product.validate(&workspace_root(), true).is_err(),
                "{version}"
            );
        }
        Ok(())
    }

    #[test]
    fn enabled_product_requires_a_stable_initial_version() -> Result<()> {
        let manifest = ProductManifest::load()?;
        let mut product = manifest.product("rust")?.clone();
        assert_eq!(product.initial_version.as_deref(), Some("0.1.0"));
        for version in [None, Some("dev"), Some("0.1.0-rc.1"), Some("0.1.0+build")] {
            product.initial_version = version.map(str::to_owned);
            assert!(product.validate(&workspace_root(), true).is_err());
        }
        assert!(manifest.product("jvm")?.initial_version.is_none());
        Ok(())
    }

    #[test]
    fn product_appearance_rejects_missing_or_mismatched_assets() -> Result<()> {
        let manifest = ProductManifest::load()?;
        let appearance = manifest
            .product("rust")?
            .appearance
            .as_ref()
            .context("appearance")?;
        let mut invalid = appearance.clone();
        invalid.dark = appearance.light.clone();
        assert!(invalid.validate(&workspace_root()).is_err());
        invalid = appearance.clone();
        invalid.theme = "../theme.json".into();
        assert!(invalid.validate(&workspace_root()).is_err());
        invalid = appearance.clone();
        invalid.font_asset = "assets/fonts/missing.ttf".into();
        assert!(invalid.validate(&workspace_root()).is_err());
        let mut invalid_product = manifest.product("rust")?.clone();
        invalid_product.appearance = None;
        assert!(invalid_product.validate(&workspace_root(), true).is_err());
        Ok(())
    }

    #[test]
    fn filename_templates_reject_unsafe_values() {
        for value in ["../bad", "{unknown}.zip", "bad name.zip", "{product.zip"] {
            assert!(
                validate_template(value).is_err(),
                "accepted unsafe template {value}"
            );
        }
    }

    #[test]
    fn rust_features_are_explicit() {
        let manifest = ProductManifest::load().expect("repository product catalog should be valid");
        let rust = manifest.product("rust").expect("Rust product should exist");
        assert_eq!(rust.cargo_features, ["agentic-tools", "rust-tools"]);
        assert_eq!(rust.remote_server_features, ["rust-tools"]);
    }

    #[test]
    fn planned_jvm_contract_is_explicit() -> Result<()> {
        let manifest = ProductManifest::load()?;
        let jvm = manifest.product("jvm")?;
        assert_eq!(jvm.status, ProductStatus::Planned);
        assert_eq!(jvm.display_name, "Groovy");
        assert_eq!(jvm.executable_name, "groovy");
        assert_eq!(jvm.cargo_features, ["agentic-tools", "jvm-tools"]);
        assert_eq!(jvm.remote_server_features, ["jvm-tools"]);
        assert_eq!(jvm.default_extensions, ["java", "kotlin"]);
        assert_eq!(jvm.default_language_servers, ["jdtls", "kotlin-lsp"]);
        let root = workspace_root();
        for filename in [
            "app-icon.svg",
            "app-icon.png",
            "app-icon@2x.png",
            "app-icon.ico",
        ] {
            assert!(root.join(&jvm.icon_set).join(filename).is_file());
        }
        assert!(root.join(&jvm.agent_instructions).is_file());
        assert!(jvm.targets.is_empty());
        Ok(())
    }

    #[test]
    fn rust_identity_is_side_by_side_with_zed() {
        let manifest = ProductManifest::load().expect("repository product catalog should be valid");
        let rust = manifest.product("rust").expect("Rust product should exist");
        assert_ne!(rust.executable_name, "zed");
        assert_ne!(rust.bundle_identifier, "dev.zed.Zed");
        assert_ne!(rust.url_scheme, "zed");
        assert_ne!(rust.data_namespace, "zed");
        assert_ne!(
            rust.windows_installer_id,
            "{FEE8C3E6-8C06-4A9B-BF13-16D6E3CBB7A0}"
        );
    }
}
