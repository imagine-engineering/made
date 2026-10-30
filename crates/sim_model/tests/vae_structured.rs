use serde::Deserialize;
use sha2::{Digest, Sha256};
use sim_model::{
    VaeKernelProfile, VaeStructuredDecodeRequest, VaeStructuredResult, hammersley_3d,
    level_embedding, shape_grid_coordinates, shape_output_from_logits, structured_vae_source_plan,
    structured_vae_source_state_count, systematic_sample_counts,
    tripo_gaussian_output_from_predictions,
};
use sim_tensor::generated_sim_operator_indirection_01::{
    tensor_from_f32_with_backend_exact_native, tensor_to_f32_with_backend_exact_native,
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext, RetryRngPolicy,
    RngAlgorithm, RngProfileVersion, RngStreamAddress, StreamId, Tensor,
    generated_random_number_generation_01::generator_exact_native,
};
use sim_types::CancellationToken;
use std::{collections::BTreeSet, error::Error, fs, path::Path};

#[derive(Deserialize)]
struct Fixture {
    schema_version: u16,
    fixture_id: String,
    oracle_kind: String,
    production_dependency: bool,
    provenance_sha256: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    catalog_contract_id: String,
    profile: String,
    architecture: String,
    structured_kind: String,
    equation_checkpoints: Vec<String>,
}

fn profile(name: &str) -> Result<VaeKernelProfile, Box<dyn Error>> {
    match name {
        "HunyuanShapeV1" => Ok(VaeKernelProfile::HunyuanShapeV1),
        "TripoSplatV1" => Ok(VaeKernelProfile::TripoSplatV1),
        other => Err(format!("unknown structured VAE profile {other}").into()),
    }
}

fn backend(limit: u64) -> Result<(CpuBackend, CpuWorkspaceAuthority), Box<dyn Error>> {
    Ok(CpuWorkspaceAuthority::create_backend(limit)?)
}

fn execution_context<'a>(
    backend: &CpuBackend,
    workspace: &CpuWorkspaceAuthority,
    cancellation: &'a CancellationToken,
    limit: u64,
) -> Result<ExecutionContext<'a>, Box<dyn Error>> {
    Ok(backend.execution_context(
        StreamId::DEFAULT,
        workspace.authorize_workspace(limit)?,
        cancellation,
    ))
}

fn upload(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn Error>> {
    Ok(tensor_from_f32_with_backend_exact_native(
        backend,
        shape,
        values,
        DType::F32,
        DeviceId::CPU,
        context,
    )?)
}

fn transaction(
    phase: &str,
    retry: u32,
    cancellation: &CancellationToken,
) -> Result<sim_tensor::RngTransaction, Box<dyn Error>> {
    let address = RngStreamAddress::new(
        "structured-workflow",
        "attempt",
        "structured-vae",
        0,
        phase,
        0,
        retry,
        RetryRngPolicy::Replay,
    )?;
    Ok(generator_exact_native(
        RngProfileVersion::V2,
        RngAlgorithm::Philox4x32_10,
        0,
        address,
        cancellation,
    )?
    .begin(None)?)
}

#[test]
fn val_vae_001_structured_source_ledger_covers_both_contracts() -> Result<(), Box<dyn Error>> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let root = workspace.join("crates/sim_test_support/fixtures/models/vae-structured");
    let provenance = fs::read(root.join("provenance.json"))?;
    let fixture: Fixture =
        serde_json::from_slice(&fs::read(root.join("architecture-checkpoints.json"))?)?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(
        fixture.fixture_id,
        "sim-native-structured-vae-source-checkpoints-v1"
    );
    assert_eq!(fixture.oracle_kind, "immutable-source-derived-checkpoints");
    assert!(!fixture.production_dependency);
    assert_eq!(
        fixture.provenance_sha256,
        format!("{:x}", Sha256::digest(provenance))
    );
    assert_eq!(fixture.cases.len(), 2);
    let catalog = fs::read_to_string(
        workspace
            .join("crates/sim_test_support/parity/catalogs/backend-conditioning-contracts.csv"),
    )?;
    let mut case_ids = BTreeSet::new();
    let mut kinds = BTreeSet::new();
    for case in fixture.cases {
        assert!(case_ids.insert(case.id));
        assert!(catalog.contains(&case.catalog_contract_id));
        let plan = structured_vae_source_plan(&profile(&case.profile)?)?;
        assert_eq!(plan.architecture(), case.architecture);
        assert_eq!(plan.equation_checkpoints(), case.equation_checkpoints);
        assert!(plan.state_checkpoints().len() >= 5);
        kinds.insert(case.structured_kind);
    }
    assert_eq!(
        kinds,
        BTreeSet::from(["gaussian_splats".to_owned(), "shape".to_owned()])
    );
    assert_eq!(
        structured_vae_source_state_count(&VaeKernelProfile::HunyuanShapeV1)?,
        266
    );
    assert_eq!(
        structured_vae_source_state_count(&VaeKernelProfile::TripoSplatV1)?,
        388
    );
    Ok(())
}

#[test]
fn typed_requests_reject_generic_or_mismatched_parameters() -> Result<(), Box<dyn Error>> {
    let shape =
        VaeStructuredDecodeRequest::shape([-1.01, -1.01, -1.01, 1.01, 1.01, 1.01], 256, 8_000)?;
    assert_eq!(
        shape.output_kind(),
        sim_model::VaeStructuredOutputKind::Shape
    );
    let splats = VaeStructuredDecodeRequest::gaussian_splats(100_000, 8)?;
    assert_eq!(
        splats.output_kind(),
        sim_model::VaeStructuredOutputKind::GaussianSplats
    );
    assert!(VaeStructuredDecodeRequest::shape([0.0; 6], 256, 8_000).is_err());
    assert!(VaeStructuredDecodeRequest::gaussian_splats(0, 8).is_err());
    assert!(VaeStructuredDecodeRequest::gaussian_splats(1, 9).is_err());
    Ok(())
}

#[test]
fn shape_results_require_exact_geometry_and_ascending_bounds() -> Result<(), Box<dyn Error>> {
    let (backend, workspace) = backend(1 << 20)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &workspace, &cancellation, 1 << 20)?;
    let logits = upload(&backend, &[1, 2, 2, 2], &[0.0; 8], &context)?;
    let result = shape_output_from_logits(logits.clone(), [-1.0, -2.0, -3.0, 1.0, 2.0, 3.0], 1)?;
    let VaeStructuredResult::Shape(field) = result else {
        return Err("expected shape field result".into());
    };
    assert_eq!(field.logits().descriptor().shape(), [1, 2, 2, 2]);
    assert_eq!(field.resolution(), 1);
    assert!(shape_output_from_logits(logits.clone(), [0.0; 6], 1).is_err());
    assert!(shape_output_from_logits(logits, [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0], 2).is_err());
    Ok(())
}

#[test]
fn hunyuan_volume_grid_is_inclusive_and_source_ordered() -> Result<(), Box<dyn Error>> {
    let (backend, workspace) = backend(1 << 20)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &workspace, &cancellation, 1 << 20)?;
    let grid = shape_grid_coordinates(&backend, [-1.0, -2.0, -3.0, 1.0, 2.0, 3.0], 2, &context)?;
    assert_eq!(grid.len(), 27);
    assert_eq!(grid[0], [-1.0, -2.0, -3.0]);
    assert_eq!(grid[1], [-1.0, -2.0, 0.0]);
    assert_eq!(grid[3], [-1.0, 0.0, -3.0]);
    assert_eq!(grid[9], [0.0, -2.0, -3.0]);
    assert_eq!(grid[26], [1.0, 2.0, 3.0]);
    drop(grid);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    let constrained = execution_context(&backend, &workspace, &cancellation, 1)?;
    assert!(
        shape_grid_coordinates(&backend, [-1.0, -2.0, -3.0, 1.0, 2.0, 3.0], 2, &constrained,)
            .is_err()
    );
    assert_eq!(constrained.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn tripo_quasi_random_and_level_embeddings_match_source_equations() -> Result<(), Box<dyn Error>> {
    assert_eq!(hammersley_3d(0, 32)?, [0.0, 0.0, 0.0]);
    assert_eq!(hammersley_3d(1, 32)?, [1.0 / 32.0, 0.5, 1.0 / 3.0]);
    assert_eq!(hammersley_3d(2, 32)?, [2.0 / 32.0, 0.25, 2.0 / 3.0]);
    let embedding = level_embedding(&[0, 1], 5)?;
    assert_eq!(embedding.len(), 10);
    assert_eq!(&embedding[..5], &[1.0, 1.0, 0.0, 0.0, 0.0]);
    assert!((embedding[5] - 1.0).abs() < 1.0e-6);
    Ok(())
}

#[test]
fn systematic_sampling_is_caller_addressed_and_preserves_counts() -> Result<(), Box<dyn Error>> {
    let (backend, workspace) = backend(1 << 20)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &workspace, &cancellation, 1 << 20)?;
    let probabilities = vec![vec![0.1, 0.2, 0.3, 0.4], vec![0.0, 0.0, 0.0, 0.0]];
    let counts = vec![17, 9];
    let mut first_transaction = transaction("octree:42", 0, &cancellation)?;
    let first =
        systematic_sample_counts(&probabilities, &counts, &mut first_transaction, &context)?;
    let mut replay_transaction = transaction("octree:42", 3, &cancellation)?;
    let replay =
        systematic_sample_counts(&probabilities, &counts, &mut replay_transaction, &context)?;
    assert_eq!(first, replay);
    assert_eq!(first[0].iter().sum::<u32>(), 17);
    assert_eq!(first[1].iter().sum::<u32>(), 9);
    let mut different_transaction = transaction("octree:43", 0, &cancellation)?;
    let different = systematic_sample_counts(
        &probabilities,
        &counts,
        &mut different_transaction,
        &context,
    )?;
    assert_ne!(first, different);
    Ok(())
}

#[test]
fn gaussian_prediction_layout_activates_exact_typed_fields() -> Result<(), Box<dyn Error>> {
    let (backend, workspace) = backend(1 << 20)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &workspace, &cancellation, 1 << 20)?;
    let points = upload(&backend, &[1, 1, 3], &[0.5, 0.5, 0.5], &context)?;
    let features = upload(&backend, &[1, 1, 480], &[0.0; 480], &context)?;
    let result = tripo_gaussian_output_from_predictions(&backend, &points, &features, &context)?;
    let VaeStructuredResult::GaussianSplats(batches) = result else {
        return Err("expected Gaussian splat result".into());
    };
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert_eq!(batch.positions().descriptor().shape(), [32, 3]);
    assert_eq!(batch.features_dc().descriptor().shape(), [32, 1, 3]);
    assert_eq!(batch.scales().descriptor().shape(), [32, 3]);
    assert_eq!(batch.rotations().descriptor().shape(), [32, 4]);
    assert_eq!(batch.opacities().descriptor().shape(), [32, 1]);
    let scales = tensor_to_f32_with_backend_exact_native(&backend, batch.scales(), &context)?;
    let expected_scale = (0.004_f32.powi(2) + 0.0009_f32.powi(2)).sqrt();
    assert!((scales[0] - expected_scale).abs() < 1.0e-6);
    let opacities = tensor_to_f32_with_backend_exact_native(&backend, batch.opacities(), &context)?;
    assert!((opacities[0] - 0.1).abs() < 1.0e-6);
    let rotations = tensor_to_f32_with_backend_exact_native(&backend, batch.rotations(), &context)?;
    let half = 0.5_f32.sqrt();
    assert!((rotations[0].abs() - half).abs() < 1.0e-6);
    assert!((rotations[1].abs() - half).abs() < 1.0e-6);
    assert!(rotations[2].abs() < 1.0e-6);
    assert!(rotations[3].abs() < 1.0e-6);

    let first_values = [
        tensor_to_f32_with_backend_exact_native(&backend, batch.positions(), &context)?,
        tensor_to_f32_with_backend_exact_native(&backend, batch.features_dc(), &context)?,
        scales,
        rotations,
        opacities,
    ];
    let first_peak = context.scratch.peak_bytes();
    assert!(first_peak > 0);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    drop(batches);

    let replay = tripo_gaussian_output_from_predictions(&backend, &points, &features, &context)?;
    let VaeStructuredResult::GaussianSplats(replay_batches) = replay else {
        return Err("expected replayed Gaussian splat result".into());
    };
    let replay_batch = replay_batches
        .first()
        .ok_or("expected one replayed Gaussian splat batch")?;
    let replay_values = [
        tensor_to_f32_with_backend_exact_native(&backend, replay_batch.positions(), &context)?,
        tensor_to_f32_with_backend_exact_native(&backend, replay_batch.features_dc(), &context)?,
        tensor_to_f32_with_backend_exact_native(&backend, replay_batch.scales(), &context)?,
        tensor_to_f32_with_backend_exact_native(&backend, replay_batch.rotations(), &context)?,
        tensor_to_f32_with_backend_exact_native(&backend, replay_batch.opacities(), &context)?,
    ];
    assert_eq!(replay_values, first_values);
    assert_eq!(context.scratch.peak_bytes(), first_peak);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn structured_postprocessing_cancels_and_ooms_without_result() -> Result<(), Box<dyn Error>> {
    let (backend, workspace) = backend(1 << 20)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &workspace, &cancellation, 1 << 20)?;
    let points = upload(&backend, &[1, 1, 3], &[0.5; 3], &context)?;
    let features = upload(&backend, &[1, 1, 480], &[0.0; 480], &context)?;

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &workspace, &cancelled, 1 << 20)?;
    assert!(
        tripo_gaussian_output_from_predictions(&backend, &points, &features, &cancelled_context,)
            .is_err()
    );

    let constrained = execution_context(&backend, &workspace, &cancellation, 64)?;
    assert!(
        tripo_gaussian_output_from_predictions(&backend, &points, &features, &constrained).is_err()
    );
    Ok(())
}

#[test]
fn val_ownership_001_structured_adapter_has_no_foundational_owner() -> Result<(), Box<dyn Error>> {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = fs::read_to_string(crate_root.join("src/vae_structured.rs"))?;
    for forbidden in [
        "struct StructuredModelStore",
        "struct StructuredAssetService",
        "struct StructuredCancellationToken",
        "CpuWorkspaceAuthority::create_backend",
        "Command::new",
        "python",
        "retry",
        "std::fs",
    ] {
        assert!(!source.contains(forbidden), "duplicate owner: {forbidden}");
    }
    for required in [
        "ExecutionContext",
        "RngTransaction",
        "backend.workspace_vec",
        "VaeStructuredResult",
        "VaeGaussianSplatBatch::checked",
        "VaeShapeField::checked",
        "parameter_tensor(module, \"gs.points_offset_perturbation\")",
        "parameter_tensor(module, \"gs.base_offset_scale\")",
    ] {
        assert!(
            source.contains(required),
            "missing owner delegation: {required}"
        );
    }
    Ok(())
}
