use std::{fs, path::PathBuf};

fn repository_root() -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .ok_or("sim_test_support has no repository root")?
        .to_path_buf())
}

fn production_source(path: &std::path::Path) -> Result<String, Box<dyn std::error::Error>> {
    let source = fs::read_to_string(path)?;
    Ok(source
        .split_once("#[cfg(test)]\nmod tests")
        .map_or(source.as_str(), |(production, _)| production)
        .to_owned())
}

#[test]
fn native_slices_share_the_worker_authorized_execution_context()
-> Result<(), Box<dyn std::error::Error>> {
    let root = repository_root()?;
    let scoped_sources = [
        ("crates/sim_tensor/src/ops/native_diffusion.rs", 0),
        ("crates/sim_model/src/alias_free_activation.rs", 0),
        ("crates/sim_model/src/native_ops.rs", 0),
        ("crates/sim_model/src/slices/native_diffusion.rs", 0),
        ("crates/sim_model/src/vision_models.rs", 0),
        ("crates/sim_sampler/src/algorithms/native_diffusion.rs", 0),
        ("crates/sim_runtime/src/native_execution_controller.rs", 0),
        ("crates/sim_media/src/png.rs", 0),
    ];
    let mut sources = Vec::new();
    for (relative, expected_compatibility_contexts) in scoped_sources {
        let source = production_source(&root.join(relative))?;
        assert!(
            !source.contains("authorize_workspace("),
            "{relative} must consume worker authority rather than mint it"
        );
        assert_eq!(
            source.matches("ScratchReservation::none()").count(),
            expected_compatibility_contexts,
            "{relative} reintroduced a zero-scratch compatibility context"
        );
        sources.push((relative, source));
    }

    let source = |relative: &str| {
        sources
            .iter()
            .find_map(|(path, source)| (*path == relative).then_some(source.as_str()))
            .ok_or_else(|| format!("missing inventoried source {relative}"))
    };

    let tensor = source("crates/sim_tensor/src/ops/native_diffusion.rs")?;
    assert!(tensor.contains("context: &ExecutionContext<'_>"));
    assert!(tensor.contains("workspace_vec"));

    let model_slice = source("crates/sim_model/src/slices/native_diffusion.rs")?;
    assert!(model_slice.contains("context: &ExecutionContext<'_>"));
    assert!(model_slice.contains("workspace_vec"));
    assert!(model_slice.contains("scaled_dot_product_attention_with_context"));

    let native_ops = source("crates/sim_model/src/native_ops.rs")?;
    for entry_point in [
        "pub fn forward_with_context(",
        "pub fn forward_if_dense_weight_is_zero_with_context(",
        "pub fn forward_with_autopad_with_context(",
    ] {
        assert!(native_ops.contains(entry_point), "missing {entry_point}");
    }

    let vision = source("crates/sim_model/src/vision_models.rs")?;
    assert!(vision.contains("load_vision_state_from_model_store_with_context"));
    assert!(vision.contains("workspace_vec"));

    let sampler = source("crates/sim_sampler/src/algorithms/native_diffusion.rs")?;
    assert!(sampler.contains("context: &ExecutionContext<'_>"));
    assert!(sampler.contains("workspace_vec"));

    let media = source("crates/sim_media/src/png.rs")?;
    assert!(media.contains("encode_png_frame_with_policy_and_context"));
    assert!(media.contains("workspace_vec::<u8>"));

    let runtime = source("crates/sim_runtime/src/native_execution_controller.rs")?;
    let bridge = runtime
        .split_once("fn native_image_tensor_context")
        .and_then(|(_, tail)| tail.split_once("\n}").map(|(body, _)| body))
        .ok_or("runtime ExecutionContext bridge is missing")?;
    assert!(bridge.contains("context.scratch.clone()"));
    assert!(bridge.contains("&context.cancellation"));
    assert!(runtime.contains("encode_png_frame_with_policy_and_context"));

    let worker = production_source(&root.join("crates/sim_worker/src/sim_worker.rs"))?;
    assert_eq!(
        worker.matches(".authorize_workspace(").count(),
        0,
        "the worker process must not authorize a caller-supplied byte count"
    );
    assert_eq!(
        worker.matches("issue_workspace_authorization()?").count(),
        1
    );
    assert_eq!(
        worker
            .matches("authorize_planned_workspace(planned_workspace)?")
            .count(),
        1
    );
    let preflight = worker
        .find("preflight")
        .ok_or("worker memory preflight is missing")?;
    let authorization = worker
        .find("issue_workspace_authorization()?")
        .ok_or("worker planned authorization issuance is missing")?;
    let dispatch = worker[authorization..]
        .find("execute")
        .map(|offset| authorization + offset)
        .ok_or("worker dispatch after authorization is missing")?;
    assert!(preflight < authorization && authorization < dispatch);

    let memory_modes = production_source(&root.join("crates/sim_worker/src/memory_modes.rs"))?;
    assert!(memory_modes.contains("pub struct PlannedWorkspaceAuthorization"));
    assert!(memory_modes.contains("workspace_authorization_issued: bool"));
    assert!(
        !memory_modes.contains("#[derive(Clone, Debug)]\npub struct PlannedWorkspaceAuthorization")
    );
    assert!(!memory_modes.contains("#[derive(Clone, Debug)]\npub struct AttemptMemoryController"));
    let supervisor = production_source(&root.join("crates/sim_worker/src/supervisor.rs"))?;
    assert_eq!(
        supervisor
            .matches("fn authorize_planned_workspace(")
            .count(),
        1
    );
    assert_eq!(
        supervisor
            .matches(".authorize_workspace(planned.bytes())")
            .count(),
        1
    );

    let packaged_fixture_worker = production_source(
        &root.join("crates/sim_test_support/src/bin/sim_native_diffusion_worker_fixture.rs"),
    )?;
    assert!(packaged_fixture_worker.contains("run_worker_process_with_diffusion_provider"));
    assert!(packaged_fixture_worker.contains("Some(Arc::new(NativeDiffusionFixture::at("));
    let provider =
        production_source(&root.join("crates/sim_test_support/src/native_diffusion_fixture.rs"))?;
    assert!(provider.contains("impl NativeDiffusionProvider for NativeDiffusionFixture"));
    assert!(provider.contains("ModelStore::new"));
    assert!(provider.contains("Sd15TinyModel::load_reduced_fixture"));
    Ok(())
}
