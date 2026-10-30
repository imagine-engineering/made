use gh_workflow::{Container, Event, Job, Port, PullRequest, Push, Workflow, WorkflowDispatch};

use crate::tasks::workflows::{
    runners::Platform,
    steps::{self, CommonPermissionSets, FluentBuilder, named, use_clang},
    vars,
};

const HOSTED_COLLAB_PATHS: &[&str] = &[
    ".github/workflows/hosted_collab_tests.yml",
    "crates/collab/**",
    "crates/collaboration_domain/**",
    "crates/collaboration_workflow/**",
    "crates/proto/**",
    "crates/rpc/**",
    "deploy/collaboration/**",
    "tooling/xtask/src/tasks/workflows.rs",
    "tooling/xtask/src/tasks/workflows/hosted_collab_tests.rs",
];

const HOSTED_COLLAB_TEST_COMMAND: &str = "cargo nextest run --package collab --features test-support --test collab_tests --no-fail-fast --no-tests=warn";

pub(crate) fn hosted_collab_tests() -> Workflow {
    let pull_request = HOSTED_COLLAB_PATHS
        .iter()
        .fold(PullRequest::default(), |event, path| event.add_path(*path));
    let push = HOSTED_COLLAB_PATHS
        .iter()
        .fold(Push::default().add_branch("main"), |event, path| {
            event.add_path(*path)
        });

    named::workflow()
        .with_minimal_permissions()
        .on(Event::default()
            .push(push)
            .pull_request(pull_request)
            .workflow_dispatch(WorkflowDispatch::default()))
        .concurrency(vars::one_workflow_per_non_main_branch())
        .add_env(("CARGO_TERM_COLOR", "always"))
        .add_env(("RUST_BACKTRACE", 1))
        .add_env(("CARGO_INCREMENTAL", 0))
        .add_job(
            "hosted_collab_tests",
            Job::default()
                .runs_on("ubuntu-22.04")
                .timeout_minutes(90u32)
                .add_env(("CARGO_PROFILE_TEST_DEBUG", 0))
                .add_env((
                    "COLLAB_TEST_DATABASE_URL",
                    "postgres://postgres@localhost/postgres",
                ))
                .map(use_clang)
                .add_service(
                    "postgres",
                    Container::new("postgres:15@sha256:1b92e7a80c021647bf70f5d3eb66066a998e4f5cf43c07bb9dc9f729782cf88e")
                        .add_env(("POSTGRES_HOST_AUTH_METHOD", "trust"))
                        .ports(vec![Port::Name("5432:5432".into())])
                        .options(
                            "--health-cmd pg_isready \
                             --health-interval 500ms \
                             --health-timeout 5s \
                             --health-retries 10",
                        ),
                )
                .add_step(steps::checkout_repo())
                .add_step(steps::free_linux_disk_space())
                .add_step(steps::setup_cargo_config(Platform::Linux))
                .map(steps::install_linux_dependencies)
                .add_step(steps::cargo_install_nextest())
                .add_step(named::bash(HOSTED_COLLAB_TEST_COMMAND))
                .add_step(steps::cleanup_cargo_config(Platform::Linux)),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hosted_collab_tests_are_postgres_backed_and_path_scoped() -> anyhow::Result<()> {
        let workflow = hosted_collab_tests()
            .to_string()
            .map_err(|error| anyhow::anyhow!("failed to serialize workflow: {error:?}"))?;

        assert!(workflow.contains(HOSTED_COLLAB_TEST_COMMAND));
        assert_eq!(workflow.matches(HOSTED_COLLAB_TEST_COMMAND).count(), 1);
        assert!(workflow.contains("workflow_dispatch: {}"));
        assert!(workflow.contains("crates/collab/**"));
        assert!(workflow.contains("crates/collaboration_domain/**"));
        assert!(workflow.contains("crates/collaboration_workflow/**"));
        assert!(workflow.contains("crates/proto/**"));
        assert!(workflow.contains("crates/rpc/**"));
        assert!(workflow.contains("deploy/collaboration/**"));
        assert!(workflow.contains("POSTGRES_HOST_AUTH_METHOD: trust"));
        assert!(workflow.contains("COLLAB_TEST_DATABASE_URL"));
        assert!(workflow.contains("postgres://postgres@localhost/postgres"));
        assert!(!workflow.contains("USE_POSTGRES"));
        assert!(workflow.contains("5432:5432"));
        assert!(workflow.contains("--health-cmd pg_isready"));
        assert!(workflow.contains("CC: clang"));
        assert!(workflow.contains("CXX: clang++"));
        assert!(!workflow.contains("repository_owner"));
        assert!(!workflow.contains("namespace-profile"));
        assert!(!workflow.contains("multiplayer-tools"));

        Ok(())
    }
}
