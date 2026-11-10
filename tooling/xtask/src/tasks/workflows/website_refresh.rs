use gh_workflow::{
    Concurrency, Event, Expression, Job, Level, Permissions, Workflow, WorkflowDispatch,
};

use super::{
    steps::{CommonPermissionSets, named},
    vars::WorkflowInput,
};

pub(crate) fn refresh_job(tag: String) -> Job {
    Job::default()
        .runs_on("ubuntu-24.04")
        .timeout_minutes(30u32)
        .permissions(Permissions::default().contents(Level::Read))
        .concurrency(
            Concurrency::new(Expression::new("website-main-release-refresh"))
                .cancel_in_progress(false),
        )
        .add_step(super::steps::checkout_repo().with_ref("main"))
        .add_step(
            named::bash("python3 script/refresh_website.py")
                .shell("bash -euo pipefail {0}")
                .add_env(("COPPER_RELEASE_TAG", tag))
                .add_env(("GH_TOKEN", "${{ github.token }}"))
                .add_env((
                    "AMPLIFY_REBUILD_WEBHOOK",
                    "${{ secrets.AMPLIFY_REBUILD_WEBHOOK }}",
                )),
        )
}

pub(crate) fn website_refresh() -> Workflow {
    let tag = WorkflowInput::string("release_tag", None).description(
        "Existing published Copper tag (copper-vX.Y.Z); never builds or publishes Copper",
    );
    named::workflow()
        .with_minimal_permissions()
        .on(Event::default()
            .workflow_dispatch(WorkflowDispatch::default().add_input(tag.name, tag.input())))
        .add_job(
            "refresh_website",
            refresh_job("${{ inputs.release_tag }}".to_owned()).cond(Expression::new("false")),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn website_refresh_is_paused_even_for_manual_dispatch() -> anyhow::Result<()> {
        let workflow = serde_json::to_value(website_refresh())?;
        assert_eq!(workflow["jobs"].as_object().expect("jobs").len(), 1);
        let job = &workflow["jobs"]["refresh_website"];
        assert_eq!(job["permissions"]["contents"], "read");
        assert_eq!(job["if"], "false");
        assert!(workflow["on"]["release"].is_null());
        assert!(job.get("needs").is_none());
        assert_eq!(job["steps"].as_array().expect("steps").len(), 2);
        let script = job["steps"][1]["run"].as_str().expect("script");
        assert_eq!(script, "python3 script/refresh_website.py");
        assert!(!script.contains("cargo "));
        assert!(!script.contains("gh release"));
        assert!(job["steps"][1].get("uses").is_none());
        assert_eq!(job["concurrency"]["cancel-in-progress"], false);
        Ok(())
    }
}
