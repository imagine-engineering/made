use serde::Deserialize;
use sha2::{Digest as _, Sha256};
use sim_types::CancellationToken;
use std::collections::BTreeSet;

const EMBEDDED_CONTRACT: &str =
    include_str!("../../../.agents/specs/comfy-parity/catalogs/spandrel-image-model-contract.json");

pub const NATIVE_UPSCALE_CONTRACT_SCHEMA_VERSION: u32 = 1;
pub const NATIVE_UPSCALE_CONTRACT_ID: &str = "zed-sim-spandrel-image-model-contract-v1";
pub const NATIVE_UPSCALE_CONTRACT_SHA256: &str =
    "dd46b24dbbb59f90d125e0f37394478a0087f646df3a1ada07677f32d57d032e";
pub const NATIVE_UPSCALE_ARCHITECTURE_COUNT: usize = 52;
pub const NATIVE_UPSCALE_ADMITTED_ARCHITECTURE_COUNT: usize = 0;

const MAIN_ARCHITECTURE_COUNT: usize = 42;
const EXTRA_ARCHITECTURE_COUNT: usize = 10;
const MAX_CONTRACT_BYTES: usize = 1_048_576;
const MAX_CONDITION_NODES: usize = 128;
const MAX_COLLECTION_ITEMS: usize = 128;
const MAX_IDENTITY_BYTES: usize = 4_096;
const MAX_STATE_KEY_COUNT: usize = 262_144;
const MAX_STATE_KEY_BYTES: usize = 64 * 1_024 * 1_024;
const MODEL_USE_DISPOSITION: &str =
    "no-model-weights-approved; evaluate model rights independently";
const MAIN_LICENSE_DISPOSITION: &str = "rejected-missing-individual-license-artifact";
const EXTRA_LICENSE_DISPOSITION: &str = "rejected-reference-only-extra-architecture";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativeUpscaleSourceFile {
    pub path: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativeUpscaleArchitectureContract {
    pub architecture_class: String,
    pub architecture_id: String,
    pub dependency_imports: Vec<String>,
    pub dependency_sha256: String,
    pub descriptor_disposition: String,
    pub descriptor_kinds: Vec<String>,
    pub detection_predicate: String,
    pub detection_state_keys: Vec<String>,
    pub display_name: String,
    pub equation_family_id: String,
    pub equation_sha256: String,
    pub input_channel_expressions: Vec<String>,
    pub license_artifacts: Vec<String>,
    pub license_disposition: String,
    pub model_use_disposition: String,
    pub normalized_state_keys: Vec<String>,
    pub ordinal: usize,
    pub origin: String,
    pub origin_ordinal: usize,
    pub output_channel_expressions: Vec<String>,
    pub rejection_reasons: Vec<String>,
    pub scale_expressions: Vec<String>,
    pub source_files: Vec<NativeUpscaleSourceFile>,
    pub source_path: String,
    pub source_sha256: String,
    pub state_normalization: Vec<String>,
    pub support_disposition: String,
}

#[derive(Clone, Debug)]
pub struct NativeUpscaleRuntimeContract {
    architectures: Vec<NativeUpscaleArchitectureContract>,
    detection_conditions: Vec<DetectionCondition>,
}

impl NativeUpscaleRuntimeContract {
    pub fn architectures(&self) -> &[NativeUpscaleArchitectureContract] {
        &self.architectures
    }

    pub fn architecture(
        &self,
        architecture_id: &str,
    ) -> Option<&NativeUpscaleArchitectureContract> {
        self.architectures
            .iter()
            .find(|architecture| architecture.architecture_id == architecture_id)
    }

    pub fn admitted_architecture_count(&self) -> usize {
        self.architectures
            .iter()
            .filter(|architecture| architecture.support_disposition == "admitted")
            .count()
    }

    pub fn detect_state_keys<'a>(
        &'a self,
        state_keys: &BTreeSet<String>,
        cancellation: &CancellationToken,
    ) -> Result<NativeUpscaleDetection<'a>, NativeUpscaleContractError> {
        let state_keys = NativeUpscaleCanonicalStateKeys::checked(
            NativeUpscaleStateDictionaryLayout::Flat,
            state_keys,
            cancellation,
        )?;
        self.detect_canonical_state_keys(&state_keys, cancellation)
    }

    pub fn detect_wrapped_state_keys<'a>(
        &'a self,
        layout: NativeUpscaleStateDictionaryLayout,
        state_keys: &BTreeSet<String>,
        cancellation: &CancellationToken,
    ) -> Result<NativeUpscaleDetection<'a>, NativeUpscaleContractError> {
        let state_keys =
            NativeUpscaleCanonicalStateKeys::checked(layout, state_keys, cancellation)?;
        self.detect_canonical_state_keys(&state_keys, cancellation)
    }

    pub fn detect_canonical_state_keys<'a>(
        &'a self,
        state_keys: &NativeUpscaleCanonicalStateKeys,
        cancellation: &CancellationToken,
    ) -> Result<NativeUpscaleDetection<'a>, NativeUpscaleContractError> {
        cancellation
            .check()
            .map_err(|_| NativeUpscaleContractError::Cancelled)?;
        for (index, (architecture, condition)) in self
            .architectures
            .iter()
            .zip(&self.detection_conditions)
            .enumerate()
        {
            if index % 16 == 0 {
                cancellation
                    .check()
                    .map_err(|_| NativeUpscaleContractError::Cancelled)?;
            }
            if condition.matches(state_keys.keys()) {
                return Ok(NativeUpscaleDetection::Unavailable { architecture });
            }
        }
        cancellation
            .check()
            .map_err(|_| NativeUpscaleContractError::Cancelled)?;
        Err(NativeUpscaleContractError::NoArchitectureMatch)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeUpscaleStateDictionaryLayout {
    Flat,
    ModelStateDict,
    StateDict,
    ParamsEma,
    ParamsDashEma,
    Params,
    Model,
    Net,
    SingleMapping,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeUpscaleCanonicalStateKeys {
    keys: BTreeSet<String>,
}

impl NativeUpscaleCanonicalStateKeys {
    pub fn checked<I, S>(
        _layout: NativeUpscaleStateDictionaryLayout,
        state_keys: I,
        cancellation: &CancellationToken,
    ) -> Result<Self, NativeUpscaleContractError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        cancellation
            .check()
            .map_err(|_| NativeUpscaleContractError::Cancelled)?;
        let mut keys = BTreeSet::new();
        let mut total_bytes = 0usize;
        for (index, state_key) in state_keys.into_iter().enumerate() {
            if index % 64 == 0 {
                cancellation
                    .check()
                    .map_err(|_| NativeUpscaleContractError::Cancelled)?;
            }
            if index >= MAX_STATE_KEY_COUNT {
                return Err(NativeUpscaleContractError::InvalidStateKeys);
            }
            let state_key = state_key.as_ref();
            if state_key.is_empty()
                || state_key.len() > MAX_IDENTITY_BYTES
                || state_key.contains('\0')
            {
                return Err(NativeUpscaleContractError::InvalidStateKeys);
            }
            total_bytes = total_bytes
                .checked_add(state_key.len())
                .ok_or(NativeUpscaleContractError::InvalidStateKeys)?;
            if total_bytes > MAX_STATE_KEY_BYTES {
                return Err(NativeUpscaleContractError::InvalidStateKeys);
            }
            keys.insert(state_key.to_owned());
        }
        for prefix in ["module.", "netG."] {
            if !keys.is_empty() && keys.iter().all(|key| key.starts_with(prefix)) {
                cancellation
                    .check()
                    .map_err(|_| NativeUpscaleContractError::Cancelled)?;
                let mut stripped = BTreeSet::new();
                for (index, key) in keys.into_iter().enumerate() {
                    if index % 64 == 0 {
                        cancellation
                            .check()
                            .map_err(|_| NativeUpscaleContractError::Cancelled)?;
                    }
                    stripped.insert(key[prefix.len()..].to_owned());
                }
                keys = stripped;
            }
        }
        cancellation
            .check()
            .map_err(|_| NativeUpscaleContractError::Cancelled)?;
        Ok(Self { keys })
    }

    pub fn keys(&self) -> &BTreeSet<String> {
        &self.keys
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeUpscaleDetection<'a> {
    Unavailable {
        architecture: &'a NativeUpscaleArchitectureContract,
    },
}

pub fn compiled_native_upscale_contract()
-> Result<NativeUpscaleRuntimeContract, NativeUpscaleContractError> {
    compile_contract(EMBEDDED_CONTRACT)
}

pub fn validate_native_upscale_contract_candidate(
    candidate: &str,
) -> Result<(), NativeUpscaleContractError> {
    compile_contract(candidate).map(|_| ())
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum NativeUpscaleContractError {
    #[error("native upscale contract exceeds its byte bound")]
    ContractTooLarge,
    #[error("native upscale contract is malformed: {0}")]
    MalformedContract(String),
    #[error("native upscale contract schema is unsupported: {0}")]
    UnsupportedSchema(u32),
    #[error("native upscale contract identity is invalid")]
    InvalidContractIdentity,
    #[error("native upscale source snapshot is invalid: {0}")]
    InvalidSourceSnapshot(String),
    #[error("native upscale source boundary is invalid")]
    InvalidSourceBoundary,
    #[error("native upscale optional-extra outcomes are invalid")]
    InvalidOptionalExtraOutcomes,
    #[error("native upscale task projection is invalid")]
    InvalidTaskProjection,
    #[error("native upscale summary is invalid")]
    InvalidSummary,
    #[error("native upscale architecture count is invalid: {0}")]
    InvalidArchitectureCount(usize),
    #[error("native upscale architecture ordinal is invalid: expected {expected}, got {actual}")]
    InvalidOrdinal { expected: usize, actual: usize },
    #[error("native upscale origin ordering is invalid at ordinal {0}")]
    InvalidOrigin(usize),
    #[error("native upscale contract has a duplicate architecture: {0}")]
    DuplicateArchitecture(String),
    #[error("native upscale contract has an ambiguous detector: {0}")]
    AmbiguousDetection(String),
    #[error("native upscale architecture row is invalid: {0}")]
    InvalidArchitecture(String),
    #[error("native upscale detector is invalid: {0}")]
    InvalidDetection(String),
    #[error("native upscale detector key projection is invalid: {0}")]
    InvalidDetectionKeyProjection(String),
    #[error("native upscale architecture is not licensed for admission: {0}")]
    UnlicensedArchitecture(String),
    #[error("native upscale catalog digest does not match the generated contract")]
    CatalogDigestMismatch,
    #[error("native upscale state-key projection is invalid")]
    InvalidStateKeys,
    #[error("native upscale state keys do not match an architecture")]
    NoArchitectureMatch,
    #[error("native upscale contract operation was cancelled")]
    Cancelled,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContractWire {
    schema_version: u32,
    contract_id: String,
    source_snapshots: SourceSnapshotsWire,
    optional_extra_outcomes: Vec<OptionalOutcomeWire>,
    source_boundary: SourceBoundaryWire,
    architectures: Vec<NativeUpscaleArchitectureContract>,
    summary: SummaryWire,
    task_projection: TaskProjectionWire,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceSnapshotsWire {
    spandrel: SourceSnapshotWire,
    spandrel_extra_arches: SourceSnapshotWire,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceSnapshotWire {
    archive_verification: String,
    baseline_tree_sha256: String,
    code_license_disposition: String,
    commit: String,
    file_count: usize,
    included_file_count: usize,
    model_use_disposition: String,
    package: String,
    path: String,
    registry_entry_count: usize,
    registry_source: String,
    registry_source_sha256: String,
    sdist: String,
    sdist_sha256: String,
    source_authority: String,
    tag: String,
    tree_sha256: String,
    version: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OptionalOutcomeWire {
    diagnostic: String,
    outcome: String,
    registry: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceBoundaryWire {
    sim_source: String,
    sim_source_sha256: String,
    fixtures: String,
    production_runtime: String,
    requirements_source: String,
    requirements_source_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SummaryWire {
    admitted_count: usize,
    architecture_count: usize,
    extra_count: usize,
    individual_license_artifact_count: usize,
    main_count: usize,
    rejected_count: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskProjectionWire {
    final_integration_task_id: String,
    implementation_leaves: Vec<String>,
    shared_runtime_contract_task_id: String,
}

#[derive(Clone, Debug)]
enum DetectionCondition {
    Key(String),
    All(Vec<Self>),
    Any(Vec<Self>),
}

impl DetectionCondition {
    fn matches(&self, state_keys: &BTreeSet<String>) -> bool {
        match self {
            Self::Key(key) => state_keys.contains(key),
            Self::All(conditions) => conditions
                .iter()
                .all(|condition| condition.matches(state_keys)),
            Self::Any(conditions) => conditions
                .iter()
                .any(|condition| condition.matches(state_keys)),
        }
    }

    fn collect_keys<'a>(&'a self, keys: &mut BTreeSet<&'a str>) {
        match self {
            Self::Key(key) => {
                keys.insert(key);
            }
            Self::All(conditions) | Self::Any(conditions) => {
                for condition in conditions {
                    condition.collect_keys(keys);
                }
            }
        }
    }
}

fn compile_contract(raw: &str) -> Result<NativeUpscaleRuntimeContract, NativeUpscaleContractError> {
    if raw.len() > MAX_CONTRACT_BYTES {
        return Err(NativeUpscaleContractError::ContractTooLarge);
    }
    let wire: ContractWire = serde_json::from_str(raw)
        .map_err(|error| NativeUpscaleContractError::MalformedContract(error.to_string()))?;
    if wire.schema_version != NATIVE_UPSCALE_CONTRACT_SCHEMA_VERSION {
        return Err(NativeUpscaleContractError::UnsupportedSchema(
            wire.schema_version,
        ));
    }
    if wire.contract_id != NATIVE_UPSCALE_CONTRACT_ID {
        return Err(NativeUpscaleContractError::InvalidContractIdentity);
    }
    validate_source_snapshots(&wire.source_snapshots)?;
    validate_source_boundary(&wire.source_boundary)?;
    validate_optional_outcomes(&wire.optional_extra_outcomes)?;
    validate_summary(&wire.summary)?;
    validate_task_projection(&wire.task_projection)?;
    let detection_conditions = validate_architectures(&wire.architectures)?;
    let digest = format!("{:x}", Sha256::digest(raw.as_bytes()));
    if digest != NATIVE_UPSCALE_CONTRACT_SHA256 {
        return Err(NativeUpscaleContractError::CatalogDigestMismatch);
    }
    Ok(NativeUpscaleRuntimeContract {
        architectures: wire.architectures,
        detection_conditions,
    })
}

fn validate_source_snapshots(
    snapshots: &SourceSnapshotsWire,
) -> Result<(), NativeUpscaleContractError> {
    validate_source_snapshot(
        &snapshots.spandrel,
        "spandrel",
        "projects/sim/Spandrel",
        "0.4.2",
        "v0.4.2",
        "724cca389f28c38e1050689d4862a452fd644484",
        "spandrel-0.4.2.tar.gz",
        "fefa4ea966c6a5b7721dcf24f3e2062a5a96a395c8bedcb570fb55971fdcbccb",
        "e1870c42b314fddb290f4d5322a03743076d98d0c6d288fc73691e3013994bbb",
        180,
        MAIN_ARCHITECTURE_COUNT,
    )?;
    validate_source_snapshot(
        &snapshots.spandrel_extra_arches,
        "spandrel-extra-arches",
        "projects/sim/spandrel-extra-arches",
        "0.2.0",
        "v0.4.0",
        "a1db3f5debbeeacbe02fb4114c69feee56ba5e21",
        "spandrel_extra_arches-0.2.0.tar.gz",
        "9216877ecabc9c97e001ad5d49c4f8d2b1f6c6f82d1e77c8e2b350c586b6e64a",
        "7c0915d2e0df7db2131117087744fa5e73954dcad72aa785386d6bf8c1efb3aa",
        52,
        EXTRA_ARCHITECTURE_COUNT,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_source_snapshot(
    snapshot: &SourceSnapshotWire,
    package: &str,
    path: &str,
    version: &str,
    tag: &str,
    commit: &str,
    sdist: &str,
    sdist_sha256: &str,
    tree_sha256: &str,
    file_count: usize,
    registry_entry_count: usize,
) -> Result<(), NativeUpscaleContractError> {
    let valid = snapshot.package == package
        && snapshot.path == path
        && snapshot.version == version
        && snapshot.tag == tag
        && snapshot.commit == commit
        && snapshot.sdist == sdist
        && snapshot.sdist_sha256 == sdist_sha256
        && snapshot.tree_sha256 == tree_sha256
        && snapshot.baseline_tree_sha256 == tree_sha256
        && snapshot.file_count == file_count
        && snapshot.included_file_count == file_count
        && snapshot.registry_entry_count == registry_entry_count
        && snapshot.archive_verification == "official PyPI sdist SHA-256 matched before extraction"
        && snapshot.source_authority == "explicit user approval"
        && snapshot
            .model_use_disposition
            .contains("no model weights approved")
        && !snapshot.code_license_disposition.is_empty()
        && is_sha256(&snapshot.registry_source_sha256)
        && is_safe_source_path(&snapshot.registry_source);
    if valid {
        Ok(())
    } else {
        Err(NativeUpscaleContractError::InvalidSourceSnapshot(
            package.to_owned(),
        ))
    }
}

fn validate_source_boundary(
    boundary: &SourceBoundaryWire,
) -> Result<(), NativeUpscaleContractError> {
    let valid = boundary.sim_source == "projects/comfy/ComfyUI/comfy_extras/nodes_upscale_model.py"
        && boundary.sim_source_sha256
            == "da11e6f46a473fad3fdfdf068ae07a022802cf122240e4b99e639a380e9772f4"
        && boundary.requirements_source == "projects/comfy/ComfyUI/requirements.txt"
        && boundary.requirements_source_sha256
            == "48f4835af39b753fb2e637ec17813716024e08952e82e6e4e536a0fcfd944d0e"
        && boundary.production_runtime
            == "native Rust only; no Python or Spandrel import or execution"
        && boundary.fixtures
            == "JSON only; no Python, model weights, native handles, or executable payloads";
    if valid {
        Ok(())
    } else {
        Err(NativeUpscaleContractError::InvalidSourceBoundary)
    }
}

fn validate_optional_outcomes(
    outcomes: &[OptionalOutcomeWire],
) -> Result<(), NativeUpscaleContractError> {
    let expected = [
        (
            "absent-or-import-failure",
            "MAIN only",
            "typed extra import unavailable",
        ),
        (
            "successful-add",
            "MAIN followed by EXTRA in source order",
            "none",
        ),
        (
            "add-failure",
            "MAIN only",
            "typed extra registry add failure",
        ),
    ];
    let valid = outcomes.len() == expected.len()
        && outcomes.iter().zip(expected).all(|(outcome, expected)| {
            outcome.outcome == expected.0
                && outcome.registry == expected.1
                && outcome.diagnostic == expected.2
        });
    if valid {
        Ok(())
    } else {
        Err(NativeUpscaleContractError::InvalidOptionalExtraOutcomes)
    }
}

fn validate_summary(summary: &SummaryWire) -> Result<(), NativeUpscaleContractError> {
    if summary.architecture_count == NATIVE_UPSCALE_ARCHITECTURE_COUNT
        && summary.main_count == MAIN_ARCHITECTURE_COUNT
        && summary.extra_count == EXTRA_ARCHITECTURE_COUNT
        && summary.admitted_count == NATIVE_UPSCALE_ADMITTED_ARCHITECTURE_COUNT
        && summary.rejected_count == NATIVE_UPSCALE_ARCHITECTURE_COUNT
        && summary.individual_license_artifact_count == 0
    {
        Ok(())
    } else {
        Err(NativeUpscaleContractError::InvalidSummary)
    }
}

fn validate_task_projection(
    projection: &TaskProjectionWire,
) -> Result<(), NativeUpscaleContractError> {
    if projection.shared_runtime_contract_task_id
        == "comfy-parity-native-upscale-runtime-contract-foundation"
        && projection.final_integration_task_id
            == "comfy-parity-native-upscale-model-resource-foundation"
        && projection.implementation_leaves.is_empty()
    {
        Ok(())
    } else {
        Err(NativeUpscaleContractError::InvalidTaskProjection)
    }
}

fn validate_architectures(
    architectures: &[NativeUpscaleArchitectureContract],
) -> Result<Vec<DetectionCondition>, NativeUpscaleContractError> {
    if architectures.len() != NATIVE_UPSCALE_ARCHITECTURE_COUNT {
        return Err(NativeUpscaleContractError::InvalidArchitectureCount(
            architectures.len(),
        ));
    }
    let mut architecture_ids = BTreeSet::new();
    let mut equation_families = BTreeSet::new();
    let mut detection_predicates = BTreeSet::new();
    let mut conditions = Vec::new();
    conditions
        .try_reserve_exact(architectures.len())
        .map_err(|error| NativeUpscaleContractError::MalformedContract(error.to_string()))?;

    for (ordinal, architecture) in architectures.iter().enumerate() {
        if architecture.ordinal != ordinal {
            return Err(NativeUpscaleContractError::InvalidOrdinal {
                expected: ordinal,
                actual: architecture.ordinal,
            });
        }
        let (expected_origin, expected_origin_ordinal, expected_license) =
            if ordinal < MAIN_ARCHITECTURE_COUNT {
                ("main", ordinal, MAIN_LICENSE_DISPOSITION)
            } else {
                (
                    "extra",
                    ordinal - MAIN_ARCHITECTURE_COUNT,
                    EXTRA_LICENSE_DISPOSITION,
                )
            };
        if architecture.origin != expected_origin
            || architecture.origin_ordinal != expected_origin_ordinal
        {
            return Err(NativeUpscaleContractError::InvalidOrigin(ordinal));
        }
        validate_architecture(architecture, expected_license)?;
        if !architecture_ids.insert(architecture.architecture_id.as_str()) {
            return Err(NativeUpscaleContractError::DuplicateArchitecture(
                architecture.architecture_id.clone(),
            ));
        }
        if !equation_families.insert(architecture.equation_family_id.as_str())
            || !detection_predicates.insert(architecture.detection_predicate.as_str())
        {
            return Err(NativeUpscaleContractError::AmbiguousDetection(
                architecture.architecture_id.clone(),
            ));
        }
        let condition = PredicateParser::parse(&architecture.detection_predicate)?;
        let mut projected_keys = BTreeSet::new();
        condition.collect_keys(&mut projected_keys);
        if !architecture
            .detection_state_keys
            .iter()
            .map(String::as_str)
            .eq(projected_keys)
        {
            return Err(NativeUpscaleContractError::InvalidDetectionKeyProjection(
                architecture.architecture_id.clone(),
            ));
        }
        conditions.push(condition);
    }
    Ok(conditions)
}

fn validate_architecture(
    architecture: &NativeUpscaleArchitectureContract,
    expected_license: &str,
) -> Result<(), NativeUpscaleContractError> {
    for (name, value) in [
        (
            "architecture class",
            architecture.architecture_class.as_str(),
        ),
        ("architecture id", architecture.architecture_id.as_str()),
        ("display name", architecture.display_name.as_str()),
        ("equation family", architecture.equation_family_id.as_str()),
        ("source path", architecture.source_path.as_str()),
    ] {
        validate_text(name, value)?;
    }
    for (name, values) in [
        ("dependency imports", &architecture.dependency_imports),
        ("descriptor kinds", &architecture.descriptor_kinds),
        ("detection state keys", &architecture.detection_state_keys),
        (
            "input channel expressions",
            &architecture.input_channel_expressions,
        ),
        ("normalized state keys", &architecture.normalized_state_keys),
        (
            "output channel expressions",
            &architecture.output_channel_expressions,
        ),
        ("rejection reasons", &architecture.rejection_reasons),
        ("state normalization", &architecture.state_normalization),
    ] {
        validate_nonempty_collection(name, values)?;
    }
    if !is_sha256(&architecture.dependency_sha256)
        || !is_sha256(&architecture.equation_sha256)
        || !is_sha256(&architecture.source_sha256)
        || architecture.support_disposition != "rejected"
        || architecture.license_disposition != expected_license
        || architecture.model_use_disposition != MODEL_USE_DISPOSITION
        || !architecture.license_artifacts.is_empty()
    {
        return Err(NativeUpscaleContractError::UnlicensedArchitecture(
            architecture.architecture_id.clone(),
        ));
    }
    if !architecture
        .rejection_reasons
        .iter()
        .any(|reason| reason == expected_license)
    {
        return Err(NativeUpscaleContractError::UnlicensedArchitecture(
            architecture.architecture_id.clone(),
        ));
    }
    let descriptor_valid = match architecture.descriptor_disposition.as_str() {
        "single-image" => {
            architecture.descriptor_kinds == ["ImageModelDescriptor"]
                && validate_nonempty_collection(
                    "scale expressions",
                    &architecture.scale_expressions,
                )
                .is_ok()
        }
        "rejected-non-single-image" => {
            architecture.descriptor_kinds == ["MaskedImageModelDescriptor"]
                && architecture.scale_expressions.is_empty()
                && architecture
                    .rejection_reasons
                    .iter()
                    .any(|reason| reason == "rejected-non-single-image")
        }
        _ => false,
    };
    if !descriptor_valid || !is_safe_source_path(&architecture.source_path) {
        return Err(NativeUpscaleContractError::InvalidArchitecture(
            architecture.architecture_id.clone(),
        ));
    }
    if architecture.source_files.is_empty()
        || architecture.source_files.len() > MAX_COLLECTION_ITEMS
    {
        return Err(NativeUpscaleContractError::InvalidArchitecture(
            architecture.architecture_id.clone(),
        ));
    }
    let mut source_paths = BTreeSet::new();
    for source in &architecture.source_files {
        if !is_safe_source_path(&source.path)
            || !is_sha256(&source.sha256)
            || !source_paths.insert(source.path.as_str())
        {
            return Err(NativeUpscaleContractError::InvalidArchitecture(
                architecture.architecture_id.clone(),
            ));
        }
    }
    Ok(())
}

fn validate_nonempty_collection(
    name: &str,
    values: &[String],
) -> Result<(), NativeUpscaleContractError> {
    if values.is_empty() || values.len() > MAX_COLLECTION_ITEMS {
        return Err(NativeUpscaleContractError::InvalidArchitecture(
            name.to_owned(),
        ));
    }
    for value in values {
        validate_text(name, value)?;
    }
    Ok(())
}

fn validate_text(name: &str, value: &str) -> Result<(), NativeUpscaleContractError> {
    if value.is_empty()
        || value.len() > MAX_IDENTITY_BYTES
        || value.chars().any(|character| character == '\0')
    {
        Err(NativeUpscaleContractError::InvalidArchitecture(
            name.to_owned(),
        ))
    } else {
        Ok(())
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_safe_source_path(path: &str) -> bool {
    !path.is_empty()
        && path.len() <= MAX_IDENTITY_BYTES
        && !path.starts_with('/')
        && !path.split('/').any(|component| {
            component.is_empty() || component == "." || component == ".." || component == ".git"
        })
        && !path.contains('\0')
}

struct PredicateParser<'a> {
    input: &'a [u8],
    offset: usize,
    nodes: usize,
}

impl<'a> PredicateParser<'a> {
    fn parse(input: &'a str) -> Result<DetectionCondition, NativeUpscaleContractError> {
        let mut parser = Self {
            input: input.as_bytes(),
            offset: 0,
            nodes: 0,
        };
        let condition = parser.condition()?;
        parser.whitespace();
        if parser.offset != parser.input.len() {
            return Err(NativeUpscaleContractError::InvalidDetection(
                input.to_owned(),
            ));
        }
        Ok(condition)
    }

    fn condition(&mut self) -> Result<DetectionCondition, NativeUpscaleContractError> {
        self.whitespace();
        self.nodes = self.nodes.saturating_add(1);
        if self.nodes > MAX_CONDITION_NODES {
            return self.invalid();
        }
        if self.peek() == Some(b'\'') {
            return self.string().map(DetectionCondition::Key);
        }
        let kind = if self.consume(b"KeyCondition.has_all(") {
            0
        } else if self.consume(b"KeyCondition.has_any(") {
            1
        } else {
            return self.invalid();
        };
        let mut children = Vec::new();
        loop {
            if children.len() >= MAX_CONDITION_NODES {
                return self.invalid();
            }
            children.push(self.condition()?);
            self.whitespace();
            if self.consume(b",") {
                continue;
            }
            if self.consume(b")") {
                break;
            }
            return self.invalid();
        }
        if children.is_empty() {
            return self.invalid();
        }
        if kind == 0 {
            Ok(DetectionCondition::All(children))
        } else {
            Ok(DetectionCondition::Any(children))
        }
    }

    fn string(&mut self) -> Result<String, NativeUpscaleContractError> {
        if !self.consume(b"'") {
            return self.invalid();
        }
        let start = self.offset;
        while let Some(byte) = self.peek() {
            if byte == b'\'' {
                let value = std::str::from_utf8(&self.input[start..self.offset])
                    .map_err(|_| NativeUpscaleContractError::InvalidDetection("utf8".to_owned()))?;
                if value.is_empty()
                    || value.len() > MAX_IDENTITY_BYTES
                    || value.contains('\\')
                    || value.contains('\0')
                {
                    return self.invalid();
                }
                self.offset += 1;
                return Ok(value.to_owned());
            }
            self.offset += 1;
        }
        self.invalid()
    }

    fn whitespace(&mut self) {
        while self.peek().is_some_and(|byte| byte.is_ascii_whitespace()) {
            self.offset += 1;
        }
    }

    fn consume(&mut self, expected: &[u8]) -> bool {
        if self
            .input
            .get(self.offset..)
            .is_some_and(|rest| rest.starts_with(expected))
        {
            self.offset += expected.len();
            true
        } else {
            false
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.offset).copied()
    }

    fn invalid<T>(&self) -> Result<T, NativeUpscaleContractError> {
        Err(NativeUpscaleContractError::InvalidDetection(
            String::from_utf8_lossy(self.input).into_owned(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    fn contains_any(condition: &DetectionCondition) -> bool {
        match condition {
            DetectionCondition::Key(_) => false,
            DetectionCondition::All(children) => children.iter().any(contains_any),
            DetectionCondition::Any(_) => true,
        }
    }

    fn first_branch_keys(condition: &DetectionCondition, keys: &mut BTreeSet<String>) {
        match condition {
            DetectionCondition::Key(key) => {
                keys.insert(key.clone());
            }
            DetectionCondition::All(children) => {
                for child in children {
                    first_branch_keys(child, keys);
                }
            }
            DetectionCondition::Any(children) => {
                if let Some(child) = children.first() {
                    first_branch_keys(child, keys);
                }
            }
        }
    }

    #[test]
    fn every_any_detector_accepts_one_branch_including_nested_variants() -> Result<(), String> {
        let contract = compiled_native_upscale_contract().map_err(|error| error.to_string())?;
        let mut any_count = 0usize;
        for (architecture, condition) in contract
            .architectures
            .iter()
            .zip(&contract.detection_conditions)
        {
            if !contains_any(condition) {
                continue;
            }
            any_count += 1;
            let mut keys = BTreeSet::new();
            first_branch_keys(condition, &mut keys);
            assert!(
                condition.matches(&keys),
                "{} treated has_any as has_all",
                architecture.architecture_id
            );
            if matches!(architecture.architecture_id.as_str(), "GRL" | "SwinIR") {
                assert!(
                    keys.len() < architecture.detection_state_keys.len(),
                    "{} did not exercise a strict nested branch",
                    architecture.architecture_id
                );
            }
        }
        assert_eq!(any_count, 12);
        Ok(())
    }

    struct CancellingStateKey {
        key: String,
        ordinal: usize,
        cancel_at: usize,
        cancellation: CancellationToken,
        visits: Arc<AtomicUsize>,
    }

    impl AsRef<str> for CancellingStateKey {
        fn as_ref(&self) -> &str {
            self.visits.fetch_add(1, Ordering::SeqCst);
            if self.ordinal == self.cancel_at {
                self.cancellation.cancel();
            }
            &self.key
        }
    }

    #[test]
    fn state_key_canonicalization_checks_cancellation_periodically_and_finally() {
        let cancellation = CancellationToken::default();
        let visits = Arc::new(AtomicUsize::new(0));
        let keys = (0..192)
            .map(|ordinal| CancellingStateKey {
                key: format!("module.layer.{ordinal}.weight"),
                ordinal,
                cancel_at: 65,
                cancellation: cancellation.clone(),
                visits: visits.clone(),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            NativeUpscaleCanonicalStateKeys::checked(
                NativeUpscaleStateDictionaryLayout::Flat,
                keys,
                &cancellation,
            ),
            Err(NativeUpscaleContractError::Cancelled)
        );
        assert_eq!(visits.load(Ordering::SeqCst), 128);

        let cancellation = CancellationToken::default();
        let visits = Arc::new(AtomicUsize::new(0));
        let keys = vec![CancellingStateKey {
            key: "module.conv.weight".to_owned(),
            ordinal: 0,
            cancel_at: 0,
            cancellation: cancellation.clone(),
            visits: visits.clone(),
        }];
        assert_eq!(
            NativeUpscaleCanonicalStateKeys::checked(
                NativeUpscaleStateDictionaryLayout::SingleMapping,
                keys,
                &cancellation,
            ),
            Err(NativeUpscaleContractError::Cancelled)
        );
        assert_eq!(visits.load(Ordering::SeqCst), 1);
    }
}
