use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sim_model::{
    GENERATED_LATENT_FORMAT_MANIFEST, GENERATED_LATENT_FORMATS, GENERATED_MODEL_FAMILIES,
    GENERATED_MODEL_FAMILY_REGISTRATIONS, LatentFormatRegistry, ModelFamilyRegistry,
};
use sim_sampler::{
    GENERATED_MODULES, GENERATED_SAMPLER_DEFINITIONS, GENERATED_SCHEDULER_DEFINITIONS,
    SamplerRegistry, SchedulerRegistry,
};
use sim_tensor::{
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, OPERATION_CONTRACTS,
    validate_generated_operation_release_closure,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

const UPDATE_ENV: &str = "UPDATE_SIM_NATIVE_COMPUTE_CLOSURE";
const VALIDATION_IDS: [&str; 11] = [
    "VAL-AUTOGRAD-001",
    "VAL-DEVICE-001",
    "VAL-LATENT-001",
    "VAL-MODEL-FAMILY-001",
    "VAL-OWNERSHIP-001",
    "VAL-PATCH-001",
    "VAL-PATCH-ADAPTER-001",
    "VAL-RNG-001",
    "VAL-SAMPLER-001",
    "VAL-SCHEDULER-001",
    "VAL-TENSOR-001",
];

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct NativeComputeClosure {
    schema_version: u8,
    task_id: String,
    validation_ids: Vec<String>,
    generator: String,
    inputs: BTreeMap<String, String>,
    counts: NativeComputeCounts,
    tensor_resolutions: Vec<TensorResolutionRow>,
    devices: Vec<DeviceRow>,
    model_family_closure_sha256: String,
    latent_formats: Vec<LatentFormatRow>,
    samplers: Vec<ComputeRow>,
    schedulers: Vec<ComputeRow>,
    patches: Vec<PatchRow>,
    external_release_gates: Vec<ExternalReleaseGate>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct NativeComputeCounts {
    tensor_contracts: usize,
    tensor_resolutions: usize,
    tensor_reference_contracts: usize,
    external_tensor_dispositions: usize,
    device_adapters: usize,
    model_families: usize,
    latent_formats: usize,
    samplers: usize,
    schedulers: usize,
    patch_contracts: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
struct ComputeRow {
    source_ordinal: u16,
    feature_id: String,
    identity: String,
    status: String,
    row_module: String,
    implementation_module: String,
    source_sha256: String,
    test_sha256: String,
    fixture: String,
    fixture_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
struct TensorResolutionRow {
    operation_id: String,
    overload_id: String,
    module: String,
    owner_task_id: String,
    baseline_fixture_sha256: String,
    evidence_fixture: String,
    evidence_fixture_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
struct DeviceRow {
    identity: String,
    wrapper_module: String,
    wrapper_sha256: String,
    backend_module: String,
    backend_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
struct LatentFormatRow {
    feature_id: String,
    identity: String,
    module: String,
    source_sha256: String,
    test_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
struct PatchRow {
    contract_id: String,
    native_owner: String,
    source_path: String,
    source_sha256: String,
    symbol_sha256: String,
    closure_artifact: String,
    validation_artifact_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
struct ExternalReleaseGate {
    gate_id: String,
    mapping_pointer: String,
    mapping_sha256: String,
}

#[test]
fn native_compute_breadth_is_exact_and_byte_stable() -> Result<(), Box<dyn std::error::Error>> {
    let expected = build_closure()?;
    validate_closure(&expected)?;
    let serialized = stable_json(&expected)?;
    let path =
        repository_root().join(".agents/specs/comfy-parity/catalogs/native-compute-closure.json");
    if std::env::var_os(UPDATE_ENV).is_some_and(|value| value == "1") {
        fs::write(&path, &serialized)?;
    }
    let checked_in = fs::read(&path)?;
    assert_eq!(
        checked_in, serialized,
        "native compute closure is stale; rerun with {UPDATE_ENV}=1",
    );
    let decoded: NativeComputeClosure = serde_json::from_slice(&checked_in)?;
    assert_eq!(decoded, expected);
    Ok(())
}

fn build_closure() -> Result<NativeComputeClosure, Box<dyn std::error::Error>> {
    let root = repository_root();
    validate_generated_operation_release_closure(OPERATION_CONTRACTS)?;
    let tensor_resolution_count = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .map(|slice| slice.len())
        .sum::<usize>();
    let tensor_reference_count = OPERATION_CONTRACTS
        .iter()
        .filter(|contract| contract.typed_reference().is_some())
        .count();
    let mut tensor_resolutions = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .flat_map(|slice| {
            slice.iter().map(|resolution| TensorResolutionRow {
                operation_id: resolution.operation_id.to_owned(),
                overload_id: resolution.overload_id.to_owned(),
                module: resolution.resolution_module.to_owned(),
                owner_task_id: resolution.owner_task_id.to_owned(),
                baseline_fixture_sha256: resolution.baseline_fixture_sha256.to_owned(),
                evidence_fixture: resolution.evidence_fixture.to_owned(),
                evidence_fixture_sha256: resolution.evidence_fixture_sha256.to_owned(),
            })
        })
        .collect::<Vec<_>>();
    tensor_resolutions.sort();
    let devices = device_rows(&root)?;

    let model_registry = ModelFamilyRegistry::checked(GENERATED_MODEL_FAMILIES)?;
    let registration_registry =
        ModelFamilyRegistry::checked_registrations(GENERATED_MODEL_FAMILY_REGISTRATIONS)?;
    assert_eq!(model_registry.len(), registration_registry.len());
    let latent_registry = LatentFormatRegistry::checked(GENERATED_LATENT_FORMATS)?;
    assert_eq!(
        latent_registry.len(),
        GENERATED_LATENT_FORMAT_MANIFEST.len(),
    );
    let latent_formats = latent_format_rows(&root)?;

    let sampler_registry = SamplerRegistry::new(GENERATED_SAMPLER_DEFINITIONS.to_vec())?;
    let scheduler_registry = SchedulerRegistry::new(GENERATED_SCHEDULER_DEFINITIONS.to_vec())?;
    let samplers = compute_rows(
        &root,
        "samplers",
        sampler_registry.definitions().iter().map(|definition| {
            (
                definition.source_ordinal,
                definition.feature_id,
                definition.identity,
                definition.implementation_module,
            )
        }),
    )?;
    let schedulers = compute_rows(
        &root,
        "schedulers",
        scheduler_registry.definitions().iter().map(|definition| {
            (
                definition.source_ordinal,
                definition.feature_id,
                definition.identity,
                definition.implementation_module,
            )
        }),
    )?;
    let patches = patch_rows(&root)?;

    let input_paths = [
        "crates/sim_tensor/build.rs",
        "crates/sim_model/build.rs",
        "crates/sim_sampler/build.rs",
        "crates/sim_tensor/operation_contract_evidence.rs",
        "crates/sim_tensor/src/operation_contract_records.rs",
        "crates/sim_tensor/src/operation_resolutions",
        "crates/sim_test_support/fixtures/tensor_operations",
        "crates/sim_tensor/src/ops",
        "crates/sim_tensor/src/backends",
        ".agents/specs/comfy-parity/catalogs/backend-models.csv",
        ".agents/specs/comfy-parity/catalogs/native-model-family-closure.json",
        "crates/sim_model/src/latent_formats",
        "crates/sim_sampler/src/algorithms",
        "crates/sim_sampler/src/schedulers",
        ".agents/specs/comfy-parity/catalogs/backend-conditioning-contracts.csv",
        ".agents/specs/comfy-parity/catalogs/native-spec-mapping.json",
        "crates/sim_sampler/tests/breadth_closure.rs",
        "crates/sim_test_support/fixtures/samplers/lms_comfy_model_0192/source_high_orders.json",
        "crates/sim_test_support/fixtures/samplers/lms_comfy_model_0192/verify_source_high_orders.py",
    ];
    let inputs = input_paths
        .into_iter()
        .map(|relative| Ok((relative.to_owned(), digest_path(&root.join(relative))?)))
        .collect::<Result<BTreeMap<_, _>, Box<dyn std::error::Error>>>()?;

    Ok(NativeComputeClosure {
        schema_version: 1,
        task_id: "comfy-parity-native-compute-breadth-integration".to_owned(),
        validation_ids: VALIDATION_IDS
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
        generator: "crates/sim_sampler/tests/breadth_closure.rs".to_owned(),
        inputs,
        counts: NativeComputeCounts {
            tensor_contracts: OPERATION_CONTRACTS.len(),
            tensor_resolutions: tensor_resolution_count,
            tensor_reference_contracts: tensor_reference_count,
            external_tensor_dispositions: OPERATION_CONTRACTS.len()
                - tensor_resolution_count
                - tensor_reference_count,
            device_adapters: 8,
            model_families: model_registry.len(),
            latent_formats: latent_registry.len(),
            samplers: samplers.len(),
            schedulers: schedulers.len(),
            patch_contracts: patches.len(),
        },
        tensor_resolutions,
        devices,
        model_family_closure_sha256: digest_file(
            &root.join(".agents/specs/comfy-parity/catalogs/native-model-family-closure.json"),
        )?,
        latent_formats,
        samplers,
        schedulers,
        patches,
        external_release_gates: external_release_gates(&root)?,
    })
}

fn external_release_gates(
    root: &Path,
) -> Result<Vec<ExternalReleaseGate>, Box<dyn std::error::Error>> {
    let mapping_path = root.join(".agents/specs/comfy-parity/catalogs/native-spec-mapping.json");
    let mapping: serde_json::Value = serde_json::from_slice(&fs::read(mapping_path)?)?;
    [
        ("corex_enablement", "/scope_transfers/corex_enablement"),
        (
            "optional_accelerator_hardware",
            "/external_release_certification_gates/optional_accelerator_hardware",
        ),
    ]
    .into_iter()
    .map(|(gate_id, mapping_pointer)| {
        let value = mapping
            .pointer(mapping_pointer)
            .filter(|value| value.is_object())
            .ok_or("native spec mapping is missing an external release gate")?;
        Ok(ExternalReleaseGate {
            gate_id: gate_id.to_owned(),
            mapping_pointer: mapping_pointer.to_owned(),
            mapping_sha256: format!("{:x}", Sha256::digest(serde_json::to_vec(value)?)),
        })
    })
    .collect()
}

fn device_rows(root: &Path) -> Result<Vec<DeviceRow>, Box<dyn std::error::Error>> {
    let identities = [
        "amd_rocm_comfy_model_0014",
        "apple_metal_mps_comfy_model_0015",
        "cambricon_mlu_comfy_model_0017",
        "cpu_comfy_model_0016",
        "directml_comfy_model_0018",
        "huawei_ascend_npu_comfy_model_0019",
        "intel_xpu_comfy_model_0021",
        "nvidia_cuda_comfy_model_0022",
    ];
    identities
        .into_iter()
        .map(|identity| {
            let wrapper_module = format!("ops/backend_{identity}");
            let backend_module = format!("backends/{identity}");
            Ok(DeviceRow {
                identity: identity.to_owned(),
                wrapper_sha256: digest_file(
                    &root.join(format!("crates/sim_tensor/src/{wrapper_module}.rs")),
                )?,
                backend_sha256: digest_file(
                    &root.join(format!("crates/sim_tensor/src/{backend_module}.rs")),
                )?,
                wrapper_module,
                backend_module,
            })
        })
        .collect()
}

fn latent_format_rows(root: &Path) -> Result<Vec<LatentFormatRow>, Box<dyn std::error::Error>> {
    let mut rows = GENERATED_LATENT_FORMAT_MANIFEST
        .iter()
        .map(|(module, definition)| {
            Ok(LatentFormatRow {
                feature_id: definition.feature_id.to_owned(),
                identity: definition.identifier.to_owned(),
                module: (*module).to_owned(),
                source_sha256: digest_file(
                    &root.join(format!("crates/sim_model/src/latent_formats/{module}.rs")),
                )?,
                test_sha256: digest_file(
                    &root.join(format!("crates/sim_model/tests/latent_formats/{module}.rs")),
                )?,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    rows.sort();
    Ok(rows)
}

fn compute_rows<'a>(
    root: &Path,
    fixture_kind: &str,
    definitions: impl IntoIterator<Item = (u16, &'a str, &'a str, &'a str)>,
) -> Result<Vec<ComputeRow>, Box<dyn std::error::Error>> {
    let mut rows = Vec::new();
    let mut identities = BTreeSet::new();
    let mut features = BTreeSet::new();
    let mut ordinals = BTreeSet::new();
    let expected_namespace = match fixture_kind {
        "samplers" => "algorithms",
        "schedulers" => "schedulers",
        _ => return Err("unsupported native compute fixture namespace".into()),
    };
    for (source_ordinal, feature_id, identity, implementation_module) in definitions {
        assert!(identities.insert(identity));
        assert!(features.insert(feature_id));
        assert!(ordinals.insert(source_ordinal));
        let feature_suffix = feature_id
            .strip_prefix("COMFY-MODEL-")
            .ok_or("generated compute feature has an invalid identifier")?;
        if feature_suffix.len() != 4 || !feature_suffix.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err("generated compute feature has an invalid numeric suffix".into());
        }
        let expected_suffix = format!("_sim_model_{feature_suffix}");
        let row_modules = GENERATED_MODULES
            .iter()
            .copied()
            .filter(|module| {
                module.starts_with(&format!("{expected_namespace}/"))
                    && module.ends_with(&expected_suffix)
            })
            .collect::<Vec<_>>();
        if row_modules.len() != 1 {
            return Err("generated compute feature does not map to one row module".into());
        }
        let row_module = row_modules[0];
        let (_, module_name) = row_module
            .split_once('/')
            .ok_or("generated compute row module is not namespaced")?;
        let source = root.join(format!("crates/sim_sampler/src/{row_module}.rs"));
        let test = root.join(format!("crates/sim_sampler/tests/{row_module}.rs"));
        let fixture_directory = root.join(format!(
            "crates/sim_test_support/fixtures/{fixture_kind}/{module_name}"
        ));
        let fixture = if fixture_kind == "samplers" && module_name == "lms_comfy_model_0192" {
            checked_lms_fixture(&fixture_directory)?
        } else {
            let fixture_files = regular_files(&fixture_directory)?;
            assert_eq!(fixture_files.len(), 1);
            fixture_files
                .into_iter()
                .next()
                .ok_or("fixture directory is empty")?
        };
        rows.push(ComputeRow {
            source_ordinal,
            feature_id: feature_id.to_owned(),
            identity: identity.to_owned(),
            status: "compiled".to_owned(),
            row_module: row_module.to_owned(),
            implementation_module: implementation_module.to_owned(),
            source_sha256: digest_file(&source)?,
            test_sha256: digest_file(&test)?,
            fixture: fixture.strip_prefix(root)?.to_string_lossy().into_owned(),
            fixture_sha256: digest_file(&fixture)?,
        });
    }
    rows.sort();
    validate_compute_row_directory_closure(root, fixture_kind, expected_namespace, &rows)?;
    Ok(rows)
}

fn validate_compute_row_directory_closure(
    root: &Path,
    fixture_kind: &str,
    namespace: &str,
    rows: &[ComputeRow],
) -> Result<(), Box<dyn std::error::Error>> {
    let expected = rows
        .iter()
        .map(|row| row.row_module.as_str())
        .collect::<BTreeSet<_>>();
    let generated = GENERATED_MODULES
        .iter()
        .copied()
        .filter(|module| {
            module.starts_with(&format!("{namespace}/")) && is_generated_row_module(module)
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(expected, generated);

    let test_directory = root.join(format!("crates/sim_sampler/tests/{namespace}"));
    let mut tests = BTreeSet::new();
    for entry in fs::read_dir(test_directory)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            return Err("generated compute test may not be a symlink".into());
        }
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("rs") {
            continue;
        }
        let name = path
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or("generated compute test name is not UTF-8")?;
        let module = format!("{namespace}/{name}");
        if is_generated_row_module(&module) {
            tests.insert(module);
        }
    }
    assert_eq!(
        expected,
        tests.iter().map(String::as_str).collect::<BTreeSet<_>>()
    );

    let fixture_directory = root.join(format!("crates/sim_test_support/fixtures/{fixture_kind}"));
    let mut fixtures = BTreeSet::new();
    for entry in fs::read_dir(fixture_directory)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if entry.file_name().to_string_lossy().starts_with("._") {
            continue;
        }
        if file_type.is_symlink() || !file_type.is_dir() {
            return Err("generated compute fixture root may contain only row directories".into());
        }
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| "generated compute fixture name is not UTF-8")?;
        fixtures.insert(format!("{namespace}/{name}"));
    }
    assert_eq!(
        expected,
        fixtures.iter().map(String::as_str).collect::<BTreeSet<_>>()
    );
    Ok(())
}

fn is_generated_row_module(module: &str) -> bool {
    module
        .rsplit_once("_sim_model_")
        .is_some_and(|(prefix, suffix)| {
            !prefix.is_empty()
                && suffix.len() == 4
                && suffix.bytes().all(|byte| byte.is_ascii_digit())
        })
}

fn patch_rows(root: &Path) -> Result<Vec<PatchRow>, Box<dyn std::error::Error>> {
    let path = root.join(".agents/specs/comfy-parity/catalogs/backend-conditioning-contracts.csv");
    let rows = parse_csv(&fs::read_to_string(path)?)?;
    let header = rows.first().ok_or("conditioning catalog is empty")?;
    let contract = column(header, "contract_id")?;
    let owner = column(header, "native_owner")?;
    let source_path = column(header, "source_path")?;
    let source = column(header, "source_sha256")?;
    let symbol = column(header, "symbol_sha256")?;
    let artifact = column(header, "validation_artifact_sha256")?;
    let closure_artifact = column(header, "closure_artifact")?;
    let mut patches = rows
        .iter()
        .skip(1)
        .filter(|row| {
            row.get(owner).is_some_and(|owner| {
                owner == "sim_model::patch_graph" || owner == "sim_model::patches"
            })
        })
        .map(|row| {
            Ok(PatchRow {
                contract_id: row.get(contract).ok_or("missing contract id")?.clone(),
                native_owner: row.get(owner).ok_or("missing native owner")?.clone(),
                source_path: row.get(source_path).ok_or("missing source path")?.clone(),
                source_sha256: row.get(source).ok_or("missing source digest")?.clone(),
                symbol_sha256: row.get(symbol).ok_or("missing symbol digest")?.clone(),
                closure_artifact: row
                    .get(closure_artifact)
                    .ok_or("missing closure artifact")?
                    .clone(),
                validation_artifact_sha256: row
                    .get(artifact)
                    .ok_or("missing validation artifact digest")?
                    .clone(),
            })
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    patches.sort();
    let mut contract_ids = BTreeSet::new();
    let mut owner_counts = BTreeMap::new();
    for patch in &mut patches {
        assert!(contract_ids.insert(&patch.contract_id));
        *owner_counts
            .entry(patch.native_owner.as_str())
            .or_insert(0usize) += 1;
        assert_eq!(
            digest_file(&root.join(&patch.source_path))?,
            patch.source_sha256
        );
        let artifact_path = root.join(format!(
            "target/sim-parity/{}.json",
            patch.closure_artifact.to_ascii_lowercase()
        ));
        let validation_artifact_sha256 = digest_file(&artifact_path)?;
        if patch.validation_artifact_sha256.is_empty() {
            patch.validation_artifact_sha256 = validation_artifact_sha256;
        } else {
            assert_eq!(validation_artifact_sha256, patch.validation_artifact_sha256);
        }
    }
    assert_eq!(owner_counts.get("sim_model::patch_graph"), Some(&14));
    assert_eq!(owner_counts.get("sim_model::patches"), Some(&14));
    assert_eq!(owner_counts.len(), 2);
    Ok(patches)
}

fn validate_closure(value: &NativeComputeClosure) -> Result<(), &'static str> {
    if value.schema_version != 1
        || value.task_id != "comfy-parity-native-compute-breadth-integration"
        || value.validation_ids
            != VALIDATION_IDS
                .iter()
                .map(|value| (*value).to_owned())
                .collect::<Vec<_>>()
        || value.generator != "crates/sim_sampler/tests/breadth_closure.rs"
        || value.inputs.len() != 19
        || value.inputs.values().any(|digest| !valid_sha256(digest))
        || value.counts.tensor_contracts != 600
        || value.counts.tensor_resolutions != 511
        || value.counts.tensor_reference_contracts != 82
        || value.counts.external_tensor_dispositions != 7
        || value.counts.device_adapters != 8
        || value.counts.model_families != 94
        || value.counts.latent_formats != 33
        || value.counts.samplers != 44
        || value.counts.schedulers != 9
        || value.counts.patch_contracts != 28
        || value.tensor_resolutions.len() != value.counts.tensor_resolutions
        || value.devices.len() != value.counts.device_adapters
        || !valid_sha256(&value.model_family_closure_sha256)
        || value.latent_formats.len() != value.counts.latent_formats
        || value.samplers.len() != value.counts.samplers
        || value.schedulers.len() != value.counts.schedulers
        || value.patches.len() != value.counts.patch_contracts
        || value.external_release_gates.len() != 2
    {
        return Err("native compute closure is incomplete");
    }
    let tensor_operations = value
        .tensor_resolutions
        .iter()
        .map(|row| row.operation_id.as_str())
        .collect::<BTreeSet<_>>();
    let tensor_overloads = value
        .tensor_resolutions
        .iter()
        .map(|row| row.overload_id.as_str())
        .collect::<BTreeSet<_>>();
    if tensor_operations.len() != value.tensor_resolutions.len()
        || tensor_overloads.len() != value.tensor_resolutions.len()
        || value.tensor_resolutions.iter().any(|row| {
            !valid_sha256(&row.baseline_fixture_sha256)
                || !valid_sha256(&row.evidence_fixture_sha256)
                || row.baseline_fixture_sha256 == row.evidence_fixture_sha256
        })
    {
        return Err("tensor resolution closure is invalid");
    }
    let device_identities = value
        .devices
        .iter()
        .map(|row| row.identity.as_str())
        .collect::<BTreeSet<_>>();
    if device_identities.len() != value.devices.len()
        || value
            .devices
            .iter()
            .any(|row| !valid_sha256(&row.wrapper_sha256) || !valid_sha256(&row.backend_sha256))
    {
        return Err("device closure is invalid");
    }
    let latent_features = value
        .latent_formats
        .iter()
        .map(|row| row.feature_id.as_str())
        .collect::<BTreeSet<_>>();
    let latent_identities = value
        .latent_formats
        .iter()
        .map(|row| row.identity.as_str())
        .collect::<BTreeSet<_>>();
    if latent_features.len() != value.latent_formats.len()
        || latent_identities.len() != value.latent_formats.len()
        || value
            .latent_formats
            .iter()
            .any(|row| !valid_sha256(&row.source_sha256) || !valid_sha256(&row.test_sha256))
    {
        return Err("latent-format closure is invalid");
    }
    for row in value.samplers.iter().chain(&value.schedulers) {
        if row.status != "compiled"
            || !valid_sha256(&row.source_sha256)
            || !valid_sha256(&row.test_sha256)
            || !valid_sha256(&row.fixture_sha256)
        {
            return Err("native compute row has an invalid digest");
        }
    }
    for row in &value.patches {
        if !valid_sha256(&row.source_sha256)
            || !valid_sha256(&row.symbol_sha256)
            || !valid_sha256(&row.validation_artifact_sha256)
        {
            return Err("patch row has an invalid digest");
        }
    }
    let expected_external_gates = [
        ("corex_enablement", "/scope_transfers/corex_enablement"),
        (
            "optional_accelerator_hardware",
            "/external_release_certification_gates/optional_accelerator_hardware",
        ),
    ];
    for (gate, (expected_id, expected_pointer)) in value
        .external_release_gates
        .iter()
        .zip(expected_external_gates)
    {
        if gate.gate_id != expected_id
            || gate.mapping_pointer != expected_pointer
            || !valid_sha256(&gate.mapping_sha256)
        {
            return Err("external release gates are not canonical");
        }
    }
    Ok(())
}

fn checked_lms_fixture(directory: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut expected = [
        "trajectory.json",
        "source_high_orders.json",
        "verify_source_high_orders.py",
    ]
    .map(|name| directory.join(name));
    expected.sort();
    if regular_files(directory)? != expected {
        return Err(
            "LMS fixture files must match the exact trajectory and high-order references".into(),
        );
    }
    Ok(directory.join("trajectory.json"))
}

#[test]
fn lms_fixture_references_are_closed_and_digest_bound() -> Result<(), Box<dyn std::error::Error>> {
    let closure = build_closure()?;
    validate_closure(&closure)?;
    let mut missing_reference = closure.clone();
    assert!(missing_reference.inputs.remove(
        "crates/sim_test_support/fixtures/samplers/lms_comfy_model_0192/source_high_orders.json"
    ).is_some());
    assert!(validate_closure(&missing_reference).is_err());
    for relative in [
        "crates/sim_test_support/fixtures/samplers/lms_comfy_model_0192/source_high_orders.json",
        "crates/sim_test_support/fixtures/samplers/lms_comfy_model_0192/verify_source_high_orders.py",
    ] {
        assert_eq!(
            closure.inputs.get(relative),
            Some(&digest_file(&repository_root().join(relative))?)
        );
    }
    let directory = std::env::temp_dir().join(format!(
        "sim-lms-fixture-closure-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    fs::create_dir(&directory)?;
    for name in [
        "trajectory.json",
        "source_high_orders.json",
        "verify_source_high_orders.py",
    ] {
        fs::write(directory.join(name), b"reference")?;
    }
    let selected = checked_lms_fixture(&directory)?;
    fs::write(directory.join("unexpected.json"), b"unreviewed")?;
    let rejects_extra = checked_lms_fixture(&directory).is_err();
    fs::remove_file(directory.join("unexpected.json"))?;
    fs::remove_file(directory.join("source_high_orders.json"))?;
    let rejects_missing = checked_lms_fixture(&directory).is_err();
    fs::remove_dir_all(&directory)?;
    assert_eq!(selected, directory.join("trajectory.json"));
    assert!(rejects_extra);
    assert!(rejects_missing);
    Ok(())
}

fn regular_files(directory: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            return Err("compute fixture may not be a symlink".into());
        }
        if entry.file_name().to_string_lossy().starts_with("._") {
            continue;
        }
        if file_type.is_file() {
            paths.push(entry.path());
        } else {
            return Err("compute fixture directory may contain only one regular file".into());
        }
    }
    paths.sort();
    Ok(paths)
}

fn digest_path(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    if path.is_file() {
        return Ok(digest_file(path)?);
    }
    let root = path.canonicalize()?;
    let mut files = Vec::new();
    collect_files(&root, &root, &mut files)?;
    files.sort_by(|left, right| left.0.cmp(&right.0));
    let mut hasher = Sha256::new();
    for (relative, physical) in files {
        let bytes = fs::read(physical)?;
        hasher.update(relative.as_bytes());
        hasher.update([0]);
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn collect_files(
    root: &Path,
    directory: &Path,
    files: &mut Vec<(String, PathBuf)>,
) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            return Err("compute input may not contain symlinks".into());
        }
        if entry.file_name().to_string_lossy().starts_with("._") {
            continue;
        }
        if file_type.is_dir() {
            collect_files(root, &entry.path(), files)?;
        } else if file_type.is_file() {
            files.push((
                entry
                    .path()
                    .strip_prefix(root)?
                    .to_string_lossy()
                    .into_owned(),
                entry.path(),
            ));
        }
    }
    Ok(())
}

fn parse_csv(text: &str) -> Result<Vec<Vec<String>>, Box<dyn std::error::Error>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut chars = text.chars().peekable();
    let mut quoted = false;
    while let Some(character) = chars.next() {
        match character {
            '"' if quoted && chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => row.push(std::mem::take(&mut field)),
            '\n' if !quoted => {
                row.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut row));
            }
            '\r' if !quoted => {}
            other => field.push(other),
        }
    }
    if quoted {
        return Err("unterminated quoted CSV field".into());
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    Ok(rows)
}

fn column(header: &[String], name: &str) -> Result<usize, Box<dyn std::error::Error>> {
    header
        .iter()
        .position(|value| value == name)
        .ok_or_else(|| format!("missing {name} column").into())
}

fn stable_json(value: &NativeComputeClosure) -> Result<Vec<u8>, serde_json::Error> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn digest_file(path: &Path) -> Result<String, std::io::Error> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
