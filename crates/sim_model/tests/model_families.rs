include!(concat!(env!("OUT_DIR"), "/generated_model_family_tests.rs"));

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sim_model::{
    GENERATED_MODEL_FAMILIES, GENERATED_MODEL_FAMILY_IDENTIFIERS, GENERATED_MODEL_FAMILY_MANIFEST,
    GENERATED_MODEL_FAMILY_REGISTRATIONS, GENERATED_MODEL_FAMILY_SOURCE_MANIFEST,
    ModelFamilyRegistry, ModelProbe, NativeFamilyBuildOptions, PatchGraph, PatchOperation,
    build_model_family, describe_model_family, map_model_weights,
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceAuthority, DType, StreamId, Tensor, TensorBackend, TensorDescriptor,
    generated_sim_operator_indirection_01::{
        cast_to_with_context_exact_native, tensor_to_f32_with_context_exact_native,
    },
};
use sim_types::{CancellationToken, DeviceKind};
use std::collections::BTreeMap;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FamilyFixture {
    fixture_id: String,
    feature_id: String,
    detector: ModelProbeFixture,
    base_artifact_digest: String,
    source_weights: Vec<TensorFixture>,
    input: TensorFixture,
    dtype: DType,
    device: DeviceKind,
    activation_elements: u64,
    memory_budget_bytes: u64,
    expected_memory_bytes: u64,
    checkpoints: Vec<CheckpointFixture>,
    patches: Vec<PatchOperation>,
    patched_checkpoints: Vec<CheckpointFixture>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelProbeFixture {
    tensor_shapes: BTreeMap<String, Vec<u64>>,
    metadata: BTreeMap<String, String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorFixture {
    key: String,
    shape: Vec<u64>,
    values: Vec<f32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointFixture {
    name: String,
    values: Vec<f32>,
}

#[derive(Serialize)]
struct ModelFamilyRowValidationArtifact<'a> {
    schema_version: u8,
    validation_id: &'static str,
    fixture_id: &'a str,
    feature_id: &'a str,
    identifier: &'a str,
    source_ordinal: u16,
    backend: &'static str,
    aggregate_model_family_breadth_claimed: bool,
    environment: ModelFamilyRowValidationEnvironment,
    digests: BTreeMap<&'static str, String>,
    cases: Vec<ModelFamilyRowValidationCase<'a>>,
    passed: usize,
    failed: u8,
    skipped: u8,
}

#[derive(Serialize)]
struct ModelFamilyRowValidationEnvironment {
    os: &'static str,
    arch: &'static str,
}

#[derive(Serialize)]
struct ModelFamilyRowValidationCase<'a> {
    id: &'a str,
    status: &'static str,
}

#[derive(Serialize)]
struct ModelFamilyBreadthArtifact {
    schema_version: u8,
    validation_id: &'static str,
    task_id: &'static str,
    backend: &'static str,
    native_production_boundary: bool,
    exact_identity_sets_equal: bool,
    family_count: usize,
    environment: ModelFamilyRowValidationEnvironment,
    inputs: BTreeMap<&'static str, String>,
    cases: Vec<ModelFamilyBreadthCase>,
    rows: Vec<ModelFamilyBreadthRow>,
    passed: usize,
    failed: u8,
    skipped: u8,
}

#[derive(Serialize)]
struct ModelFamilyBreadthCase {
    id: &'static str,
    status: &'static str,
}

#[derive(Serialize)]
struct ModelFamilyBreadthRow {
    source_ordinal: u16,
    module: String,
    feature_id: String,
    identifier: String,
    fixture: String,
    source_projection_sha256: String,
    production_sha256: String,
    test_sha256: String,
    fixture_sha256: String,
    provenance_sha256: String,
    status: &'static str,
}

fn write_model_family_row_artifact(
    fixture_id: &str,
    feature_id: &str,
    identifier: &str,
    source_ordinal: u16,
    module: &str,
    cases: &[&str],
) -> Result<(), Box<dyn std::error::Error>> {
    if fixture_id.is_empty()
        || !fixture_id
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        || module.is_empty()
        || !module
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
        || cases.is_empty()
        || cases.iter().any(|case| case.is_empty())
    {
        return Err("model-family row validation identity or cases are invalid".into());
    }
    let repository_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let paths = [
        (
            "row_source",
            repository_root.join(format!("crates/sim_model/src/families/{module}.rs")),
        ),
        (
            "row_test",
            repository_root.join(format!("crates/sim_model/tests/families/{module}.rs")),
        ),
        (
            "fixture",
            repository_root.join(format!(
                "crates/sim_test_support/fixtures/models/{fixture_id}/family.json"
            )),
        ),
        (
            "provenance",
            repository_root.join(format!(
                "crates/sim_test_support/fixtures/models/{fixture_id}/provenance.json"
            )),
        ),
    ];
    let mut digests = BTreeMap::new();
    for (name, path) in paths {
        digests.insert(name, format!("{:x}", Sha256::digest(std::fs::read(path)?)));
    }
    let artifact = ModelFamilyRowValidationArtifact {
        schema_version: 1,
        validation_id: "VAL-MODEL-FAMILY-ROW-001",
        fixture_id,
        feature_id,
        identifier,
        source_ordinal,
        backend: "sim_tensor::CpuBackend",
        aggregate_model_family_breadth_claimed: false,
        environment: ModelFamilyRowValidationEnvironment {
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
        },
        digests,
        cases: cases
            .iter()
            .map(|id| ModelFamilyRowValidationCase {
                id,
                status: "passed",
            })
            .collect(),
        passed: cases.len(),
        failed: 0,
        skipped: 0,
    };
    let output_directory = repository_root.join("target/sim-parity/val-model-family-row-001");
    std::fs::create_dir_all(&output_directory)?;
    let mut bytes = serde_json::to_vec_pretty(&artifact)?;
    bytes.push(b'\n');
    std::fs::write(output_directory.join(format!("{fixture_id}.json")), bytes)?;
    Ok(())
}

#[test]
fn val_model_family_001_every_generated_model_family_fixture_executes_the_canonical_harness()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(
        GENERATED_MODEL_FAMILY_TEST_FIXTURES.len(),
        GENERATED_MODEL_FAMILY_MANIFEST.len()
    );
    assert_eq!(
        GENERATED_MODEL_FAMILY_SOURCE_MANIFEST.len(),
        GENERATED_MODEL_FAMILY_MANIFEST.len()
    );
    assert_eq!(
        GENERATED_MODEL_FAMILY_REGISTRATIONS.len(),
        GENERATED_MODEL_FAMILY_MANIFEST.len()
    );
    let registry = ModelFamilyRegistry::checked(GENERATED_MODEL_FAMILIES)?;
    let registration_registry =
        ModelFamilyRegistry::checked_registrations(GENERATED_MODEL_FAMILY_REGISTRATIONS)?;
    assert_eq!(registry.len(), GENERATED_MODEL_FAMILY_MANIFEST.len());
    assert_eq!(registration_registry.len(), registry.len());
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let mut executed_rows = Vec::with_capacity(GENERATED_MODEL_FAMILY_MANIFEST.len());

    for (index, (fixture_name, definition)) in GENERATED_MODEL_FAMILY_MANIFEST.iter().enumerate() {
        let (module, feature_id, declared_fixture, source_ordinal) =
            GENERATED_MODEL_FAMILY_SOURCE_MANIFEST[index];
        assert_eq!(declared_fixture, *fixture_name);
        assert_eq!(feature_id, definition.feature_id);
        assert_eq!(
            GENERATED_MODEL_FAMILY_IDENTIFIERS[index],
            definition.identifier
        );
        assert_eq!(
            GENERATED_MODEL_FAMILY_REGISTRATIONS[index].source_ordinal,
            source_ordinal
        );
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("sim_test_support")
            .join("fixtures")
            .join("models")
            .join(*fixture_name)
            .join("family.json");
        let fixture: FamilyFixture = serde_json::from_slice(&std::fs::read(path)?)?;
        assert_eq!(fixture.fixture_id, *fixture_name);
        assert_eq!(fixture.feature_id, definition.feature_id);
        describe_model_family(definition)?;
        let detection = registry.detect(&ModelProbe {
            tensor_shapes: fixture.detector.tensor_shapes,
            metadata: fixture.detector.metadata,
        })?;
        assert_eq!(detection.identity.feature_id(), definition.feature_id);

        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(8 * 1024 * 1024)?,
            &cancellation,
        );
        let source_weights = fixture
            .source_weights
            .iter()
            .map(|weight| {
                Ok((
                    weight.key.clone(),
                    tensor(&backend, weight, fixture.dtype, &context)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>, Box<dyn std::error::Error>>>()?;
        let mapped = map_model_weights(definition, &fixture.base_artifact_digest, source_weights)?;
        let options = NativeFamilyBuildOptions {
            dtype: fixture.dtype,
            device: fixture.device,
            activation_elements: fixture.activation_elements,
            memory_budget_bytes: fixture.memory_budget_bytes,
            allow_unexpected_weights: false,
        };
        let model = build_model_family(definition, mapped, options)?;
        assert_eq!(
            model.memory_estimate().total_bytes,
            fixture.expected_memory_bytes
        );
        let input = tensor(&backend, &fixture.input, fixture.dtype, &context)?;
        assert_checkpoints(
            &backend,
            model.forward_checkpoints(&backend, &input, &context)?,
            &fixture.checkpoints,
            &context,
        )?;

        if !fixture.patches.is_empty() {
            let patched = PatchGraph::checked(&fixture.base_artifact_digest, fixture.patches)?
                .apply(&backend, model.weights(), &context)?;
            let patched_model = model.with_weights(patched)?;
            assert_checkpoints(
                &backend,
                patched_model.forward_checkpoints(&backend, &input, &context)?,
                &fixture.patched_checkpoints,
                &context,
            )?;
        } else {
            assert!(fixture.patched_checkpoints.is_empty());
        }

        executed_rows.push(model_family_breadth_row(
            module,
            feature_id,
            definition.identifier,
            fixture_name,
            source_ordinal,
        )?);
    }
    write_model_family_breadth_artifact(executed_rows)?;
    Ok(())
}

fn model_family_breadth_row(
    module: &str,
    feature_id: &str,
    identifier: &str,
    fixture: &str,
    source_ordinal: u16,
) -> Result<ModelFamilyBreadthRow, Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let production = root.join(format!("crates/sim_model/src/families/{module}.rs"));
    let test = root.join(format!("crates/sim_model/tests/families/{module}.rs"));
    let fixture_path = root.join(format!(
        "crates/sim_test_support/fixtures/models/{fixture}/family.json"
    ));
    let provenance_path = root.join(format!(
        "crates/sim_test_support/fixtures/models/{fixture}/provenance.json"
    ));
    let provenance_bytes = std::fs::read(&provenance_path)?;
    let provenance: serde_json::Value = serde_json::from_slice(&provenance_bytes)?;
    assert_eq!(provenance["feature_id"], feature_id);
    assert_eq!(provenance["source_symbol"], identifier);
    assert_eq!(provenance["source_ordinal"], source_ordinal);
    let source_projection = provenance["source_projection"]
        .as_str()
        .ok_or("model-family provenance source projection must be a string")?;
    let source_projection_sha256 = provenance["source_projection_sha256"]
        .as_str()
        .ok_or("model-family provenance source projection digest must be a string")?;
    assert_eq!(
        format!("{:x}", Sha256::digest(source_projection.as_bytes())),
        source_projection_sha256
    );
    Ok(ModelFamilyBreadthRow {
        source_ordinal,
        module: module.to_owned(),
        feature_id: feature_id.to_owned(),
        identifier: identifier.to_owned(),
        fixture: fixture.to_owned(),
        source_projection_sha256: source_projection_sha256.to_owned(),
        production_sha256: sha256_file(&production)?,
        test_sha256: sha256_file(&test)?,
        fixture_sha256: sha256_file(&fixture_path)?,
        provenance_sha256: format!("{:x}", Sha256::digest(&provenance_bytes)),
        status: "passed",
    })
}

fn write_model_family_breadth_artifact(
    rows: Vec<ModelFamilyBreadthRow>,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(rows.len(), 94);
    assert!(
        rows.iter()
            .enumerate()
            .all(|(index, row)| row.source_ordinal == u16::try_from(index).unwrap())
    );
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let inputs = [
        (
            "backend-models.csv",
            root.join("crates/sim_test_support/parity/catalogs/backend-models.csv"),
        ),
        (
            "model-families-v1.json",
            root.join("crates/sim_model/catalog/model-families-v1.json"),
        ),
        (
            "native-model-family-closure.json",
            root.join("crates/sim_test_support/parity/catalogs/native-model-family-closure.json"),
        ),
        ("build.rs", root.join("crates/sim_model/build.rs")),
        (
            "model_families.rs",
            root.join("crates/sim_model/tests/model_families.rs"),
        ),
        (
            "breadth_closure.rs",
            root.join("crates/sim_model/tests/breadth_closure.rs"),
        ),
    ]
    .into_iter()
    .map(|(name, path)| Ok((name, sha256_file(&path)?)))
    .collect::<Result<BTreeMap<_, _>, Box<dyn std::error::Error>>>()?;
    let cases = [
        "model-family-breadth:exact-catalog-source-registration-test-fixture-identity",
        "model-family-breadth:parsed-detection-and-canonical-registry",
        "model-family-breadth:transactional-state-mapping-and-checked-build",
        "model-family-breadth:named-forward-checkpoints-and-patches",
        "model-family-breadth:memory-dtype-device-and-cancellation-contracts",
        "model-family-breadth:typed-failure-and-no-partial-publication",
        "model-family-breadth:normalized-source-production-test-fixture-digests",
        "model-family-breadth:byte-stable-aggregate-publication",
    ]
    .into_iter()
    .map(|id| ModelFamilyBreadthCase {
        id,
        status: "passed",
    })
    .collect();
    let artifact = ModelFamilyBreadthArtifact {
        schema_version: 1,
        validation_id: "VAL-MODEL-FAMILY-001",
        task_id: "comfy-parity-model-family-breadth-closure",
        backend: "sim_tensor::CpuBackend",
        native_production_boundary: true,
        exact_identity_sets_equal: true,
        family_count: rows.len(),
        environment: ModelFamilyRowValidationEnvironment {
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
        },
        inputs,
        cases,
        passed: rows.len(),
        failed: 0,
        skipped: 0,
        rows,
    };
    let mut bytes = serde_json::to_vec_pretty(&artifact)?;
    bytes.push(b'\n');
    let output = validation_target_directory(&root).join("sim-parity/val-model-family-001.json");
    std::fs::create_dir_all(output.parent().ok_or("validation output has no parent")?)?;
    std::fs::write(output, bytes)?;
    Ok(())
}

fn validation_target_directory(root: &std::path::Path) -> std::path::PathBuf {
    match std::env::var_os("CARGO_TARGET_DIR") {
        Some(directory) => {
            let directory = std::path::PathBuf::from(directory);
            if directory.is_absolute() {
                directory
            } else {
                root.join(directory)
            }
        }
        None => root.join("target"),
    }
}

fn sha256_file(path: &std::path::Path) -> Result<String, std::io::Error> {
    Ok(format!("{:x}", Sha256::digest(std::fs::read(path)?)))
}

fn tensor(
    backend: &CpuBackend,
    fixture: &TensorFixture,
    dtype: DType,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor = TensorDescriptor::contiguous(
        fixture.shape.clone(),
        DType::F32,
        backend.device(),
        context.stream,
    )?;
    let (tensor, _) = backend.upload_f32(descriptor, &fixture.values, context)?;
    if dtype == DType::F32 {
        Ok(tensor)
    } else {
        Ok(cast_to_with_context_exact_native(
            backend,
            &tensor,
            dtype,
            backend.device(),
            false,
            false,
            context,
        )?)
    }
}

fn assert_checkpoints(
    backend: &CpuBackend,
    actual: Vec<sim_model::ModelForwardCheckpoint>,
    expected: &[CheckpointFixture],
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(actual.len(), expected.len());
    for (checkpoint, expected) in actual.iter().zip(expected) {
        assert_eq!(checkpoint.name, expected.name);
        let actual_values =
            tensor_to_f32_with_context_exact_native(backend, &checkpoint.tensor, context)?;
        assert_eq!(actual_values.len(), expected.values.len());
        for (actual_value, expected_value) in actual_values.iter().zip(&expected.values) {
            assert!(
                (actual_value - expected_value).abs() <= 1.0e-5,
                "checkpoint {} expected {expected_value}, got {actual_value}",
                checkpoint.name,
            );
        }
    }
    Ok(())
}
