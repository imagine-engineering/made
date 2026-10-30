#![allow(
    clippy::disallowed_methods,
    reason = "the synchronous xtask must wait for the selected platform bundler"
)]

use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context as _, Result, bail, ensure};
use clap::{Parser, ValueEnum};
use serde::Serialize;

use crate::product_manifest::{ProductManifest, ProductStatus, workspace_root};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Platform {
    Linux,
    Macos,
    Windows,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum SigningPolicy {
    Auto,
    Off,
    Required,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum BundlePhase {
    All,
    Application,
    RemoteServer,
    Package,
}

#[derive(Clone, Debug, Parser)]
pub struct BundleArgs {
    /// Stable product ID from products/flavors.toml.
    #[arg(long)]
    product: String,
    /// Packaging platform. Defaults to the current host.
    #[arg(long, value_enum)]
    platform: Option<Platform>,
    /// Explicit Rust compilation target.
    #[arg(long)]
    target: Option<String>,
    /// Release channel embedded in the application.
    #[arg(long, default_value = "stable")]
    channel: String,
    /// Production signing behavior.
    #[arg(long, value_enum, default_value = "auto")]
    signing: SigningPolicy,
    /// Release build phase. The generated release workflow uses this to compile components in parallel.
    #[arg(long, value_enum, default_value = "all")]
    phase: BundlePhase,
    /// Include the optional native Sim workflow integration and its worker.
    #[arg(long)]
    sim: bool,
    /// Validate and print the resolved plan without building.
    #[arg(long)]
    dry_run: bool,
}

#[derive(Debug, Serialize)]
struct BundlePlan<'a> {
    product_id: &'a str,
    display_name: String,
    executable_name: &'a str,
    bundle_identifier: String,
    url_scheme: &'a str,
    data_namespace: String,
    update_namespace: &'a str,
    windows_installer_id: String,
    platform: Platform,
    target: &'a str,
    channel: &'a str,
    application_features: &'a [String],
    remote_server_features: &'a [String],
    no_default_features: bool,
    sim: bool,
    target_dir: String,
    artifact_name: String,
    signing: SigningPolicy,
    phase: BundlePhase,
    signing_credentials_available: bool,
}

pub fn run(args: BundleArgs) -> Result<()> {
    let manifest = ProductManifest::load()?;
    let product = manifest.product(&args.product)?;
    ensure!(
        product.status == ProductStatus::Enabled,
        "product `{}` is planned and cannot be bundled",
        product.id
    );
    ensure!(
        ["stable", "preview", "nightly", "dev"].contains(&args.channel.as_str()),
        "unsupported release channel `{}`",
        args.channel
    );

    let platform = args.platform.unwrap_or_else(host_platform);
    let default_target = default_target(platform);
    let target = args.target.as_deref().unwrap_or(default_target);
    validate_target(platform, target)?;
    let target_key = target_key(platform, target);
    ensure!(
        product
            .targets
            .iter()
            .any(|candidate| candidate == target_key),
        "product `{}` does not support target `{target}`",
        product.id
    );

    let extension = match platform {
        Platform::Linux => "tar.gz",
        Platform::Macos => "dmg",
        Platform::Windows => "exe",
    };
    let (platform_name, architecture) = target_key.split_once('-').context("invalid target key")?;
    let version = env::var("RELEASE_VERSION").unwrap_or(
        product
            .initial_version
            .clone()
            .context("enabled product has no initial version")?,
    );
    semver::Version::parse(&version).context("bundle version must be a semantic version")?;
    let artifact_name = product.render_name(
        &product.artifact_name,
        &version,
        platform_name,
        architecture,
        extension,
    )?;
    let root = workspace_root();
    let target_dir = product_target_dir(platform, &root, &product.id);
    let display_name = channel_display_name(&product.display_name, &args.channel);
    let bundle_identifier = channel_identifier(&product.bundle_identifier, &args.channel, ".");
    let data_namespace = channel_identifier(&product.data_namespace, &args.channel, "-");
    let windows_installer_id =
        channel_windows_installer_id(&product.windows_installer_id, &args.channel)?;
    let credentials_available = signing_credentials_available(platform);
    if args.signing == SigningPolicy::Required && !args.dry_run && !credentials_available {
        bail!(
            "signing is required; missing {} credentials: {}",
            platform_name,
            signing_credential_names(platform)
                .iter()
                .filter(|name| !env::var(name).is_ok_and(|value| !value.trim().is_empty()))
                .copied()
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    let mut application_features = product.cargo_features.clone();
    if args.sim {
        application_features.push("sim".to_owned());
        let backends: &[&str] = match platform {
            Platform::Macos => &["metal"],
            Platform::Linux => &["cuda", "rocm", "mlu", "npu", "xpu"],
            Platform::Windows => &["rocm", "directml"],
        };
        application_features.extend(backends.iter().map(|backend| (*backend).to_owned()));
    }

    let plan = BundlePlan {
        product_id: &product.id,
        display_name: display_name.clone(),
        executable_name: &product.executable_name,
        bundle_identifier: bundle_identifier.clone(),
        url_scheme: &product.url_scheme,
        data_namespace: data_namespace.clone(),
        update_namespace: &product.update_namespace,
        windows_installer_id: windows_installer_id.clone(),
        platform,
        target,
        channel: &args.channel,
        application_features: &application_features,
        remote_server_features: &product.remote_server_features,
        no_default_features: true,
        sim: args.sim,
        target_dir: target_dir.display().to_string(),
        artifact_name: artifact_name.clone(),
        signing: args.signing,
        phase: args.phase,
        signing_credentials_available: credentials_available,
    };
    if args.dry_run {
        println!("{}", serde_json::to_string_pretty(&plan)?);
        return Ok(());
    }

    let mut command = platform_command(platform, target)?;
    if args.sim {
        command.arg(if platform == Platform::Windows {
            "-Sim"
        } else {
            "--sim"
        });
    }
    command.current_dir(&root);
    command.env("ZED_PRODUCT_ID", &product.id);
    command.env("ZED_PRODUCT_DISPLAY_NAME", display_name);
    command.env("ZED_PRODUCT_EXECUTABLE", &product.executable_name);
    command.env("ZED_PRODUCT_BUNDLE_ID", bundle_identifier);
    command.env("ZED_PRODUCT_URL_SCHEME", &product.url_scheme);
    command.env("ZED_PRODUCT_DATA_NAMESPACE", data_namespace);
    command.env("ZED_PRODUCT_UPDATE_NAMESPACE", &product.update_namespace);
    command.env("ZED_PRODUCT_WINDOWS_INSTALLER_ID", windows_installer_id);
    let product_icon_set = root.join(&product.icon_set);
    command.env(
        "ZED_PRODUCT_ICON_SET",
        external_tool_path(&product_icon_set),
    );
    command.env("ZED_PRODUCT_APP_FEATURES", application_features.join(","));
    command.env(
        "ZED_PRODUCT_REMOTE_FEATURES",
        product.remote_server_features.join(","),
    );
    command.env("ZED_PRODUCT_ARTIFACT_NAME", &artifact_name);
    command.env("RELEASE_VERSION", version);
    command.env(
        "ZED_PRODUCT_SIGNING",
        format!("{:?}", args.signing).to_lowercase(),
    );
    command.env(
        "ZED_BUNDLE_PHASE",
        match args.phase {
            BundlePhase::All => "all",
            BundlePhase::Application => "application",
            BundlePhase::RemoteServer => "remote-server",
            BundlePhase::Package => "package",
        },
    );
    command.env("ZED_RELEASE_CHANNEL", &args.channel);
    command.env("CARGO_TARGET_DIR", &target_dir);
    if args.signing == SigningPolicy::Off {
        command.env("ZED_DISABLE_SIGNING", "1");
    }

    let status = command
        .status()
        .context("failed to start platform bundler")?;
    ensure!(status.success(), "platform bundler failed with {status}");
    if matches!(args.phase, BundlePhase::All | BundlePhase::Package) {
        let artifact_root = if target_dir.is_absolute() {
            target_dir
        } else {
            root.join(target_dir)
        };
        let artifact_path = artifact_root.join("release").join(&artifact_name);
        ensure!(
            artifact_path.is_file(),
            "platform bundler did not produce expected artifact {}",
            artifact_path.display()
        );
    }
    Ok(())
}

fn product_target_dir(platform: Platform, root: &Path, product_id: &str) -> PathBuf {
    let relative = Path::new("target/products").join(product_id);
    if platform == Platform::Windows {
        relative
    } else {
        root.join(relative)
    }
}

fn external_tool_path(path: &Path) -> &Path {
    dunce::simplified(path)
}

fn platform_command(platform: Platform, target: &str) -> Result<Command> {
    let command = match platform {
        Platform::Linux => {
            let mut command = Command::new("bash");
            command.arg("script/bundle-linux");
            command
        }
        Platform::Macos => {
            let mut command = Command::new("bash");
            command.args(["script/bundle-mac", target]);
            command
        }
        Platform::Windows => {
            let architecture = target
                .strip_suffix("-pc-windows-msvc")
                .context("invalid Windows target")?;
            let mut command = Command::new("pwsh");
            command.args([
                "-File",
                "script/bundle-windows.ps1",
                "-Architecture",
                architecture,
            ]);
            command
        }
    };
    Ok(command)
}

fn channel_display_name(display_name: &str, channel: &str) -> String {
    if channel == "stable" {
        display_name.to_string()
    } else {
        format!("{display_name} {}", uppercase_first(channel))
    }
}

fn channel_identifier(identifier: &str, channel: &str, separator: &str) -> String {
    if channel == "stable" {
        identifier.to_string()
    } else {
        format!("{identifier}{separator}{channel}")
    }
}

fn uppercase_first(value: &str) -> String {
    let mut characters = value.chars();
    match characters.next() {
        Some(first) => first.to_uppercase().chain(characters).collect(),
        None => String::new(),
    }
}

fn channel_windows_installer_id(installer_id: &str, channel: &str) -> Result<String> {
    if channel == "stable" {
        return Ok(installer_id.to_string());
    }
    let namespace = uuid::Uuid::parse_str(installer_id)
        .with_context(|| format!("invalid Windows installer ID `{installer_id}`"))?;
    Ok(format!(
        "{{{}}}",
        uuid::Uuid::new_v5(&namespace, channel.as_bytes())
            .hyphenated()
            .to_string()
            .to_uppercase()
    ))
}

fn host_platform() -> Platform {
    if cfg!(target_os = "macos") {
        Platform::Macos
    } else if cfg!(target_os = "windows") {
        Platform::Windows
    } else {
        Platform::Linux
    }
}

fn default_target(platform: Platform) -> &'static str {
    match platform {
        Platform::Linux => "x86_64-unknown-linux-gnu",
        Platform::Macos => "aarch64-apple-darwin",
        Platform::Windows => "x86_64-pc-windows-msvc",
    }
}

fn validate_target(platform: Platform, target: &str) -> Result<()> {
    let valid = match platform {
        Platform::Linux => target == "x86_64-unknown-linux-gnu",
        Platform::Macos => target == "aarch64-apple-darwin",
        Platform::Windows => target == "x86_64-pc-windows-msvc",
    };
    ensure!(valid, "unsupported {:?} target `{target}`", platform);
    Ok(())
}

fn target_key(platform: Platform, target: &str) -> &'static str {
    match (platform, target) {
        (Platform::Linux, "x86_64-unknown-linux-gnu") => "linux-x86_64",
        (Platform::Macos, "aarch64-apple-darwin") => "macos-aarch64",
        (Platform::Windows, "x86_64-pc-windows-msvc") => "windows-x86_64",
        _ => unreachable!("target was validated before key derivation"),
    }
}

fn signing_credential_names(platform: Platform) -> &'static [&'static str] {
    match platform {
        Platform::Linux => &[
            "LINUX_SIGNING_KEY",
            "LINUX_SIGNING_KEY_PASSWORD",
            "LINUX_SIGNING_KEY_FINGERPRINT",
        ],
        Platform::Macos => &[
            "MACOS_SIGNING_IDENTITY",
            "MACOS_CERTIFICATE",
            "MACOS_CERTIFICATE_PASSWORD",
            "APPLE_NOTARIZATION_KEY",
            "APPLE_NOTARIZATION_KEY_ID",
            "APPLE_NOTARIZATION_ISSUER_ID",
        ],
        Platform::Windows => &[
            "AZURE_TENANT_ID",
            "AZURE_CLIENT_ID",
            "AZURE_CLIENT_SECRET",
            "ACCOUNT_NAME",
            "CERT_PROFILE_NAME",
            "ENDPOINT",
            "WINDOWS_SIGNING_PUBLISHER",
            "FILE_DIGEST",
            "TIMESTAMP_DIGEST",
            "TIMESTAMP_SERVER",
        ],
    }
}

fn signing_credentials_available(platform: Platform) -> bool {
    signing_credential_names(platform)
        .iter()
        .all(|name| env::var(name).is_ok_and(|value| !value.trim().is_empty()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_phase_one_targets_are_exact() {
        assert!(validate_target(Platform::Linux, "x86_64-unknown-linux-gnu").is_ok());
        assert!(validate_target(Platform::Macos, "aarch64-apple-darwin").is_ok());
        assert!(validate_target(Platform::Windows, "x86_64-pc-windows-msvc").is_ok());
        assert!(validate_target(Platform::Linux, "aarch64-unknown-linux-gnu").is_err());
    }

    #[test]
    fn windows_target_directory_is_relative_for_msvc_tools() {
        let root = Path::new("/workspace");
        assert_eq!(
            product_target_dir(Platform::Windows, root, "rust"),
            PathBuf::from("target/products/rust")
        );
        assert_eq!(
            product_target_dir(Platform::Linux, root, "rust"),
            root.join("target/products/rust")
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_external_tool_paths_are_not_verbatim() {
        assert_eq!(
            external_tool_path(Path::new(r"\\?\C:\workspace\icon.ico")),
            Path::new(r"C:\workspace\icon.ico")
        );

        let product_icon_path = workspace_root().join("assets/product_flavors/rust/app-icon.ico");
        let product_icon = external_tool_path(&product_icon_path);
        assert!(product_icon.is_file());
        assert!(!product_icon.to_string_lossy().starts_with(r"\\?\"));
    }

    #[test]
    fn channel_identity_matches_runtime_derivation() -> Result<()> {
        assert_eq!(channel_display_name("Product", "stable"), "Product");
        assert_eq!(
            channel_display_name("Product", "preview"),
            "Product Preview"
        );
        assert_eq!(
            channel_identifier("dev.ideflavors.rust", "preview", "."),
            "dev.ideflavors.rust.preview"
        );
        assert_eq!(
            channel_identifier("ide-rust", "preview", "-"),
            "ide-rust-preview"
        );
        let stable = "{6D7C1287-4A0E-5F5F-BE2C-8066A31F8761}";
        assert_eq!(channel_windows_installer_id(stable, "stable")?, stable);
        assert_ne!(channel_windows_installer_id(stable, "preview")?, stable);
        Ok(())
    }

    #[test]
    fn windows_license_tool_uses_an_isolated_short_target_directory() -> Result<()> {
        let script =
            std::fs::read_to_string(workspace_root().join("script/generate-licenses.ps1"))?;
        assert!(script.contains("[System.IO.Path]::GetTempPath()"));
        assert!(script.contains("cargo install \"cargo-about@$CARGO_ABOUT_VERSION\" --target-dir"));
        Ok(())
    }

    #[test]
    fn windows_bundle_uses_visual_studio_cmake() -> Result<()> {
        let script = std::fs::read_to_string(workspace_root().join("script/bundle-windows.ps1"))?;
        assert!(script.contains("CommonExtensions\\Microsoft\\CMake\\CMake\\bin\\cmake.exe"));
        assert!(script.contains("$env:CMAKE = $visualStudioCmake"));
        Ok(())
    }
}
