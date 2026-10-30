use anyhow::{Result, bail};
use collections::BTreeMap;
use task::{
    MAX_TASK_ARTIFACT_BYTES, TaskArtifact, TaskArtifactGroup, TaskArtifactGroupArgumentBinding,
    TaskArtifactKind,
};

use crate::cargo_preset::CompiledCargoPreset;

pub const CARGO_COVERAGE_ARTIFACT_FORMAT: &str = "llvm-cov-json-v1";
pub const CARGO_COVERAGE_ARTIFACT_PATH: &str = "target/zed-coverage.json";
pub const CARGO_COVERAGE_FAILURE_GUIDANCE: &str = "Coverage failed. Verify that cargo-llvm-cov is installed on the project host with `cargo install cargo-llvm-cov`, then inspect the task output.";

pub fn compile_coverage_plan(
    mut compiled_cargo_context: CompiledCargoPreset,
) -> Result<CompiledCargoPreset> {
    if compiled_cargo_context.task_template.command != "cargo" {
        bail!("Run with Coverage requires the Cargo task runner");
    }
    let arguments = &mut compiled_cargo_context.task_template.args;
    let subcommand_index = usize::from(
        arguments
            .first()
            .is_some_and(|argument| argument.starts_with('+')),
    );
    if arguments.get(subcommand_index).map(String::as_str) != Some("run") {
        bail!("Run with Coverage requires a Cargo run selection");
    }

    let original_arguments = std::mem::take(arguments);
    let mut coverage_arguments = Vec::with_capacity(original_arguments.len() + 4);
    let mut original_arguments = original_arguments.into_iter();
    if subcommand_index == 1
        && let Some(toolchain) = original_arguments.next()
    {
        coverage_arguments.push(toolchain);
    }
    let _run = original_arguments.next();
    coverage_arguments.extend([
        "llvm-cov".to_string(),
        "--json".to_string(),
        "--output-path".to_string(),
        CARGO_COVERAGE_ARTIFACT_PATH.to_string(),
        "run".to_string(),
    ]);
    coverage_arguments.extend(original_arguments);

    compiled_cargo_context.task_template.label = format!(
        "Run with Coverage ({})",
        compiled_cargo_context.task_template.label
    );
    compiled_cargo_context.task_template.args = coverage_arguments;
    compiled_cargo_context
        .task_template
        .tags
        .retain(|tag| !tag.starts_with("cargo-"));
    compiled_cargo_context
        .task_template
        .tags
        .extend(["cargo-coverage".to_string(), "external-tool".to_string()]);
    compiled_cargo_context.task_template.artifact = None;
    compiled_cargo_context
        .task_template
        .artifact_argument_binding = None;
    compiled_cargo_context
        .task_template
        .artifact_group_argument_binding = Some(TaskArtifactGroupArgumentBinding {
        argument_index: subcommand_index + 3,
        role: "report".into(),
        related: Vec::new(),
    });
    compiled_cargo_context.task_template.artifact_group = Some(TaskArtifactGroup {
        format: CARGO_COVERAGE_ARTIFACT_FORMAT.into(),
        files: BTreeMap::from_iter([(
            "report".into(),
            TaskArtifact {
                path: CARGO_COVERAGE_ARTIFACT_PATH.into(),
                kind: TaskArtifactKind::Data,
                max_bytes: MAX_TASK_ARTIFACT_BYTES,
            },
        )]),
    });
    if !compiled_cargo_context
        .task_template
        .required_experimental_features
        .iter()
        .any(|flag| flag == "rust-coverage-experimental")
    {
        compiled_cargo_context
            .task_template
            .required_experimental_features
            .push("rust-coverage-experimental".into());
    }
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

    #[test]
    fn cargo_coverage_preserves_compiled_context_and_declares_exact_host_artifact() {
        let mut preset = CargoPreset::ephemeral_default(CargoSubcommand::Run);
        preset.label = "release app".to_string();
        preset.scope = CargoPresetScope::Package;
        preset.package = Some("app".to_string());
        preset.target = Some(CargoTargetSelector::Binary("server".to_string()));
        preset.toolchain = Some("nightly".to_string());
        preset.features = vec!["telemetry".to_string()];
        preset.environment =
            HashMap::from_iter([("SECRET_TOKEN".to_string(), "secret".to_string())]);
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
