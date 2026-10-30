mod product_manifest;
mod tasks;
mod workspace;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "cargo xtask")]
struct Args {
    #[command(subcommand)]
    command: CliCommand,
}

#[derive(Subcommand)]
enum CliCommand {
    /// Builds one validated product bundle.
    Bundle(tasks::bundle::BundleArgs),
    /// Runs `cargo clippy`.
    Clippy(tasks::clippy::ClippyArgs),
    Compliance(tasks::compliance::ComplianceArgs),
    Licenses(tasks::licenses::LicensesArgs),
    /// Checks that packages conform to a set of standards.
    PackageConformity(tasks::package_conformity::PackageConformityArgs),
    /// Validates and generates compile-time product metadata.
    Products(tasks::products::ProductsArgs),
    /// Publishes GPUI and its dependencies to crates.io.
    PublishGpui(tasks::publish_gpui::PublishGpuiArgs),
    /// Resolves and validates the next Copper release version.
    ReleaseVersion(tasks::release_version::ReleaseVersionArgs),
    /// Runs the Linux Bubblewrap sandboxing NixOS VM tests.
    SandboxTests(tasks::sandbox_tests::SandboxTestsArgs),
    /// Runs the Windows WSL Bubblewrap sandbox behavior tests.
    WslSandboxTests(tasks::wsl_sandbox_tests::WslSandboxTestsArgs),
    /// Downloads the pinned `webrtc-sys` release and configures `LK_CUSTOM_WEBRTC`.
    SetupWebrtc(tasks::setup_webrtc::SetupWebrtcArgs),
    /// Builds GPUI web examples and serves them.
    WebExamples(tasks::web_examples::WebExamplesArgs),
    Workflows(tasks::workflows::GenerateWorkflowArgs),
    CheckWorkflows(tasks::workflow_checks::WorkflowValidationArgs),
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        CliCommand::Bundle(args) => tasks::bundle::run(args),
        CliCommand::Clippy(args) => tasks::clippy::run_clippy(args),
        CliCommand::Compliance(args) => tasks::compliance::check_compliance(args),
        CliCommand::Licenses(args) => tasks::licenses::run_licenses(args),
        CliCommand::PackageConformity(args) => {
            tasks::package_conformity::run_package_conformity(args)
        }
        CliCommand::Products(args) => tasks::products::run(args),
        CliCommand::PublishGpui(args) => tasks::publish_gpui::run_publish_gpui(args),
        CliCommand::ReleaseVersion(args) => tasks::release_version::run(args),
        CliCommand::SandboxTests(args) => tasks::sandbox_tests::run_sandbox_tests(args),
        CliCommand::WslSandboxTests(args) => tasks::wsl_sandbox_tests::run_wsl_sandbox_tests(args),
        CliCommand::SetupWebrtc(args) => tasks::setup_webrtc::run_setup_webrtc(args),
        CliCommand::WebExamples(args) => tasks::web_examples::run_web_examples(args),
        CliCommand::Workflows(args) => tasks::workflows::run_workflows(args),
        CliCommand::CheckWorkflows(args) => tasks::workflow_checks::validate(args),
    }
}
