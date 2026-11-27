use anyhow::Result;
use project::rust_coverage_provider::compile_coverage_task_template;

use crate::cargo_preset::CompiledCargoPreset;

pub use project::rust_coverage_provider::{
    CARGO_COVERAGE_ARTIFACT_FORMAT, CARGO_COVERAGE_ARTIFACT_PATH,
};

pub const CARGO_COVERAGE_FAILURE_GUIDANCE: &str = "Coverage failed. Verify that cargo-llvm-cov is installed on the project host with `cargo install cargo-llvm-cov`, then inspect the task output.";

pub fn compile_coverage_plan(
    mut compiled_cargo_context: CompiledCargoPreset,
) -> Result<CompiledCargoPreset> {
    compile_coverage_task_template(&mut compiled_cargo_context.task_template)?;
    Ok(compiled_cargo_context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cargo_preset::{
        CargoCompileContext, CargoPreset, CargoPresetScope, CargoSubcommand, CargoTargetSelector,
        CargoWorkingDirectoryPolicy, compile_preset,
    };
    use collections::HashMap;
    use task::{MAX_TASK_ARTIFACT_BYTES, TaskArtifactKind};

    #[test]
    fn cargo_coverage_preserves_compiled_context_and_declares_exact_host_artifact() {
        let mut preset = CargoPreset::ephemeral_default(CargoSubcommand::Run);
        preset.label = "release app".to_string();
        preset.scope = CargoPresetScope::Package;
        preset.package = Some("app".to_string());
        preset.target = Some(CargoTargetSelector::Binary("server".to_string()));
        preset.toolchain = Some("nightly".to_string());
        preset.features = vec!["telemetry".to_string()];
        preset.environment = HashMap::from_iter([
            ("SECRET_TOKEN".into(), "secret".into()),
            ("CARGO_LLVM_COV_SETUP".into(), "yes".into()),
            ("RUSTUP_AUTO_INSTALL".into(), "1".into()),
        ]);
        preset.working_directory = CargoWorkingDirectoryPolicy::Package;
        let compiled = compile_preset(
            &preset,
            &CargoCompileContext {
                workspace_cwd: Some("/workspace".to_string()),
                package_cwd: Some("/workspace/app".to_string()),
                ..CargoCompileContext::default()
            },
            None,
        )
        .expect("Cargo run should compile");

        let coverage = compile_coverage_plan(compiled).expect("coverage plan should compile");
        assert_eq!(coverage.task_template.command, "cargo");
        assert_eq!(
            coverage.task_template.args,
            [
                "+nightly",
                "llvm-cov",
                "--json",
                "--output-path",
                CARGO_COVERAGE_ARTIFACT_PATH,
                "run",
                "--package",
                "app",
                "--bin",
                "server",
                "--features",
                "telemetry",
            ]
        );
        assert_eq!(
            coverage.task_template.cwd.as_deref(),
            Some("/workspace/app")
        );
        assert_eq!(
            coverage
                .task_template
                .env
                .get("SECRET_TOKEN")
                .map(String::as_str),
            Some("secret")
        );
        assert_eq!(
            coverage
                .task_template
                .env
                .get("CARGO_LLVM_COV_SETUP")
                .map(String::as_str),
            Some("no")
        );
        assert_eq!(
            coverage
                .task_template
                .env
                .get("RUSTUP_AUTO_INSTALL")
                .map(String::as_str),
            Some("0")
        );
        let artifact = coverage
            .task_template
            .artifact_group
            .as_ref()
            .expect("coverage format intent")
            .files
            .get("report")
            .expect("coverage should declare its report");
        assert_eq!(artifact.path, CARGO_COVERAGE_ARTIFACT_PATH);
        let binding = coverage
            .task_template
            .artifact_group_argument_binding
            .as_ref()
            .expect("exact collector output argument binding");
        assert_eq!(binding.argument_index, 4);
        assert_eq!(
            coverage.task_template.args.get(binding.argument_index),
            Some(&artifact.path)
        );
        assert_eq!(artifact.kind, TaskArtifactKind::Data);
        assert_eq!(artifact.max_bytes, MAX_TASK_ARTIFACT_BYTES);
        assert!(
            coverage
                .task_template
                .tags
                .contains(&"cargo-coverage".to_string())
        );
        assert!(!format!("{coverage:?}").contains("SECRET_TOKEN=secret"));
    }

    #[test]
    fn cargo_coverage_binding_uses_exact_output_slot_for_default_and_selected_toolchains() {
        for toolchain in [None, Some("stable")] {
            let mut preset = CargoPreset::ephemeral_default(CargoSubcommand::Run);
            preset.toolchain = toolchain.map(str::to_string);
            preset.args = vec!["--release".into()];
            preset.trailing_args = vec!["workload argument".into()];
            let coverage = compile_coverage_plan(
                compile_preset(&preset, &CargoCompileContext::default(), None)
                    .expect("actual selected Cargo Run preset"),
            )
            .expect("actual coverage collector plan");
            let template = coverage.task_template;
            let resolved = template
                .resolve_task(
                    "coverage-output-slot",
                    &task::TaskContext {
                        cwd: Some("/workspace".into()),
                        ..task::TaskContext::default()
                    },
                )
                .expect("actual coverage Task resolution");
            let index = 3 + usize::from(toolchain.is_some());
            let bound = resolved
                .bind_artifact_invocation("00000000-0000-0000-0000-000000000001")
                .expect("actual scheduler binding primitive");
            let artifact = bound
                .resolved_artifact_group()
                .expect("bound coverage format")
                .files
                .get("report")
                .expect("bound report declaration");
            let mut expected = resolved.resolved.args.clone();
            *expected.get_mut(index).expect("selected output operand") = artifact.path.clone();
            assert_eq!(bound.resolved.args, expected);
            assert_eq!(
                artifact.path,
                "target/zed-coverage-00000000000000000000000000000001.json"
            );
            assert_eq!(bound.id, resolved.id);
            assert_eq!(bound.original_task(), &template);
            assert_eq!(bound.resolved.cwd, resolved.resolved.cwd);
            let rerun = bound
                .bind_artifact_invocation("00000000-0000-0000-0000-000000000002")
                .expect("cached invocation rebinds original output");
            let rerun_artifact = rerun
                .resolved_artifact_group()
                .expect("rerun coverage format")
                .files
                .get("report")
                .expect("rerun report declaration");
            assert_eq!(rerun.resolved.args.get(index), Some(&rerun_artifact.path));
            assert_ne!(rerun_artifact.path, artifact.path);
            assert_eq!(rerun.id, resolved.id);
            assert_eq!(rerun.original_task(), &template);
        }
    }

    #[test]
    fn cargo_coverage_rejects_non_run_tasks_and_has_bounded_setup_guidance() {
        let compiled = compile_preset(
            &CargoPreset::ephemeral_default(CargoSubcommand::Test),
            &CargoCompileContext::default(),
            None,
        )
        .expect("Cargo test should compile");
        assert!(compile_coverage_plan(compiled).is_err());
        assert!(CARGO_COVERAGE_FAILURE_GUIDANCE.contains("cargo install cargo-llvm-cov"));
        assert!(CARGO_COVERAGE_FAILURE_GUIDANCE.len() < 512);
    }
}
