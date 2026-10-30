use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::HashSet, fs, path::Path};

const EVIDENCE_PREFIX: &str = "crates/sim_test_support/fixtures/tensor_operations";
const MAX_RESOLUTION_JSON_BYTES: usize = 64 * 1024;
const MAX_RESOLUTION_PARAMETERS: usize = 128;
const MAX_RESOLUTION_OUTPUTS: usize = 32;
const MAX_RESOLUTION_FIELD_BYTES: usize = 1024;
const MAX_EVIDENCE_FIXTURE_BYTES: usize = 1024 * 1024;
const MAX_SOURCE_OBSERVATIONS: usize = 64;
const MAX_OBSERVATION_CONTAINER_ITEMS: usize = 512;
const MAX_OBSERVATION_DEPTH: usize = 12;
const MAX_OBSERVATION_NODES: usize = 32 * 1024;

pub(crate) struct ResolutionExpectation<'a> {
    pub(crate) resolution_module: &'a str,
    pub(crate) operation_id: &'a str,
    pub(crate) baseline_overload_id: &'a str,
    pub(crate) baseline_fixture_sha256: &'a str,
    pub(crate) overload_id: &'a str,
    pub(crate) ordered_parameters_json: &'a str,
    pub(crate) output_arity: &'a str,
    pub(crate) output_types_json: &'a str,
    pub(crate) rust_signature: &'a str,
    pub(crate) mutation_rule: &'a str,
    pub(crate) alias_rule: &'a str,
    pub(crate) shape_rule: &'a str,
    pub(crate) dtype_rule: &'a str,
    pub(crate) accumulation_dtype: &'a str,
    pub(crate) layout_rule: &'a str,
    pub(crate) device_rule: &'a str,
    pub(crate) numeric_rule: &'a str,
    pub(crate) tolerance: &'a str,
    pub(crate) determinism: &'a str,
    pub(crate) cancellation_points: &'a str,
    pub(crate) vjp_rule: &'a str,
    pub(crate) jvp_rule: &'a str,
    pub(crate) owner_task_id: &'a str,
    pub(crate) evidence_fixture: &'a str,
    pub(crate) evidence_fixture_sha256: &'a str,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct ResolutionParameter {
    default: serde_json::Value,
    keyword_only: bool,
    kind: ResolutionParameterKind,
    name: String,
    #[serde(rename = "type")]
    type_name: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum ResolutionParameterKind {
    PositionalOnly,
    PositionalOrKeyword,
    KeywordOnly,
    VariadicPositional,
    VariadicKeyword,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolutionEvidence {
    schema_version: u32,
    resolution_module: String,
    operation_id: String,
    baseline_overload_id: String,
    baseline_fixture_sha256: String,
    overload_id: String,
    owner_task_id: String,
    ordered_parameters: Vec<ResolutionParameter>,
    output_types: Vec<String>,
    semantics: ResolutionEvidenceSemantics,
    #[serde(default)]
    source_profile: Option<ResolutionEvidenceSourceProfile>,
    #[serde(default)]
    source_observations: Vec<ResolutionEvidenceObservation>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolutionEvidenceSourceProfile {
    dependency: String,
    version: String,
    profile: String,
    fingerprint_sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ResolutionEvidenceObservationCase {
    Forward,
    Invalid,
    Alias,
    Empty,
    Cancellation,
    Jvp,
    Vjp,
    Architecture,
    NonzeroExecution,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ResolutionEvidenceObservationProvenance {
    SourceDerived,
    IndependentlyAnalytical,
    NativeValidated,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ResolutionEvidenceToleranceMode {
    BitExact,
    Absolute,
    AbsoluteRelative,
    Structural,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolutionEvidenceTolerance {
    mode: ResolutionEvidenceToleranceMode,
    #[serde(default)]
    absolute: Option<f64>,
    #[serde(default)]
    relative: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolutionEvidenceObservation {
    id: String,
    case: ResolutionEvidenceObservationCase,
    provenance: ResolutionEvidenceObservationProvenance,
    inputs: serde_json::Value,
    expected: serde_json::Value,
    tolerance: ResolutionEvidenceTolerance,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolutionEvidenceSemantics {
    rust_signature: String,
    mutation_rule: String,
    alias_rule: String,
    shape_rule: String,
    dtype_rule: String,
    accumulation_dtype: String,
    layout_rule: String,
    device_rule: String,
    numeric_rule: String,
    tolerance: String,
    determinism: String,
    cancellation_points: String,
    vjp_rule: String,
    jvp_rule: String,
}

struct ParsedResolutionSemantics {
    parameters: Vec<ResolutionParameter>,
    output_types: Vec<String>,
}

pub(crate) fn valid_module_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.as_bytes().first().is_some_and(u8::is_ascii_lowercase)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

pub(crate) fn valid_evidence_fixture_path(value: &str, module_name: &str) -> bool {
    let expected_prefix = format!("{EVIDENCE_PREFIX}/{module_name}/");
    let Some(file_name) = value.strip_prefix(&expected_prefix) else {
        return false;
    };
    valid_module_name(module_name)
        && !file_name.is_empty()
        && !file_name.contains(['/', '\\'])
        && file_name != "."
        && file_name != ".."
        && file_name.ends_with(".json")
}

pub(crate) fn validate_resolution_semantics(
    expectation: &ResolutionExpectation<'_>,
) -> Result<(), String> {
    parse_resolution_semantics(expectation).map(|_| ())
}

fn parse_resolution_semantics(
    expectation: &ResolutionExpectation<'_>,
) -> Result<ParsedResolutionSemantics, String> {
    if expectation.ordered_parameters_json.len() > MAX_RESOLUTION_JSON_BYTES
        || expectation.output_types_json.len() > MAX_RESOLUTION_JSON_BYTES
    {
        return Err("resolution JSON exceeds its bounded size".to_owned());
    }
    let parameters =
        serde_json::from_str::<Vec<ResolutionParameter>>(expectation.ordered_parameters_json)
            .map_err(|error| format!("ordered parameter JSON is invalid: {error}"))?;
    let output_types = serde_json::from_str::<Vec<String>>(expectation.output_types_json)
        .map_err(|error| format!("output type JSON is invalid: {error}"))?;
    let output_arity = expectation
        .output_arity
        .parse::<usize>()
        .map_err(|error| format!("output arity is invalid: {error}"))?;
    let mut parameter_names = HashSet::with_capacity(parameters.len());
    let parameters_are_valid = parameters.len() <= MAX_RESOLUTION_PARAMETERS
        && parameters.iter().all(|parameter| {
            let kind_matches_keyword_flag = matches!(
                parameter.kind,
                ResolutionParameterKind::KeywordOnly | ResolutionParameterKind::VariadicKeyword
            ) == parameter.keyword_only;
            kind_matches_keyword_flag
                && valid_semantic_identifier(&parameter.name)
                && parameter_names.insert(parameter.name.as_str())
                && valid_semantic_field(&parameter.type_name)
                && valid_parameter_default(&parameter.default)
        });
    let outputs_are_valid = output_types.len() <= MAX_RESOLUTION_OUTPUTS
        && output_arity == output_types.len()
        && output_types
            .iter()
            .all(|output_type| valid_semantic_field(output_type));
    let semantic_fields_are_valid = [
        expectation.overload_id,
        expectation.rust_signature,
        expectation.mutation_rule,
        expectation.alias_rule,
        expectation.shape_rule,
        expectation.dtype_rule,
        expectation.accumulation_dtype,
        expectation.layout_rule,
        expectation.device_rule,
        expectation.numeric_rule,
        expectation.tolerance,
        expectation.determinism,
        expectation.cancellation_points,
        expectation.vjp_rule,
        expectation.jvp_rule,
    ]
    .into_iter()
    .all(valid_semantic_field);
    if !parameters_are_valid || !outputs_are_valid || !semantic_fields_are_valid {
        return Err("resolution semantics are incomplete or contain a sentinel".to_owned());
    }
    Ok(ParsedResolutionSemantics {
        parameters,
        output_types,
    })
}

fn valid_semantic_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .as_bytes()
            .first()
            .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_')
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

pub(crate) fn valid_semantic_field(value: &str) -> bool {
    if value.trim().is_empty() || value.len() > MAX_RESOLUTION_FIELD_BYTES {
        return false;
    }
    !contains_semantic_sentinel(value)
}

fn valid_parameter_default(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => true,
        serde_json::Value::String(value) => {
            value.len() <= MAX_RESOLUTION_FIELD_BYTES && !contains_semantic_sentinel(value)
        }
        serde_json::Value::Array(values) => {
            values.len() <= MAX_RESOLUTION_PARAMETERS && values.iter().all(valid_parameter_default)
        }
        serde_json::Value::Object(values) => {
            values.len() <= MAX_RESOLUTION_PARAMETERS
                && values.iter().all(|(key, value)| {
                    key.len() <= MAX_RESOLUTION_FIELD_BYTES
                        && !contains_semantic_sentinel(key)
                        && valid_parameter_default(value)
                })
        }
    }
}

fn contains_semantic_sentinel(value: &str) -> bool {
    let normalized = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>();
    let words = normalized.split_whitespace().collect::<Vec<_>>();
    let sentinel_word = words.iter().any(|word| {
        matches!(
            *word,
            "unresolved"
                | "unknown"
                | "todo"
                | "tbd"
                | "placeholder"
                | "unimplemented"
                | "pending"
                | "stub"
                | "fixme"
        )
    });
    let sentinel_phrase = words
        .windows(2)
        .any(|words| matches!(words, ["not", "implemented"] | ["not", "resolved"]));
    sentinel_word || sentinel_phrase
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && value.bytes().any(|byte| byte != b'0')
}

fn validate_observation_value(value: &serde_json::Value, depth: usize, nodes: &mut usize) -> bool {
    *nodes = nodes.saturating_add(1);
    if depth > MAX_OBSERVATION_DEPTH || *nodes > MAX_OBSERVATION_NODES {
        return false;
    }
    match value {
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => true,
        serde_json::Value::String(value) => value.len() <= MAX_RESOLUTION_FIELD_BYTES,
        serde_json::Value::Array(values) => {
            values.len() <= MAX_OBSERVATION_CONTAINER_ITEMS
                && values
                    .iter()
                    .all(|value| validate_observation_value(value, depth + 1, nodes))
        }
        serde_json::Value::Object(values) => {
            values.len() <= MAX_OBSERVATION_CONTAINER_ITEMS
                && values.iter().all(|(key, value)| {
                    valid_semantic_identifier(key)
                        && validate_observation_value(value, depth + 1, nodes)
                })
        }
    }
}

fn valid_observation_tolerance(tolerance: &ResolutionEvidenceTolerance) -> bool {
    let valid_number =
        |value: Option<f64>| value.is_some_and(|value| value.is_finite() && value >= 0.0);
    match tolerance.mode {
        ResolutionEvidenceToleranceMode::BitExact | ResolutionEvidenceToleranceMode::Structural => {
            tolerance.absolute.is_none() && tolerance.relative.is_none()
        }
        ResolutionEvidenceToleranceMode::Absolute => {
            valid_number(tolerance.absolute) && tolerance.relative.is_none()
        }
        ResolutionEvidenceToleranceMode::AbsoluteRelative => {
            valid_number(tolerance.absolute) && valid_number(tolerance.relative)
        }
    }
}

fn validate_source_observations(
    evidence: &ResolutionEvidence,
    actual_evidence_digest: &str,
) -> Result<(), String> {
    let has_profile = evidence.source_profile.is_some();
    let has_observations = !evidence.source_observations.is_empty();
    if !has_profile && !has_observations {
        return Ok(());
    }
    let profile = evidence
        .source_profile
        .as_ref()
        .ok_or_else(|| "source observations require an exact source profile".to_owned())?;
    if !has_observations || evidence.source_observations.len() > MAX_SOURCE_OBSERVATIONS {
        return Err("source observations must be nonempty and bounded".to_owned());
    }
    if !valid_semantic_identifier(&profile.dependency)
        || !valid_semantic_field(&profile.version)
        || !valid_semantic_field(&profile.profile)
        || !valid_sha256(&profile.fingerprint_sha256)
        || profile.fingerprint_sha256 == actual_evidence_digest
        || profile.fingerprint_sha256 == evidence.baseline_fixture_sha256
    {
        return Err("source profile is incomplete, malformed, or self-referential".to_owned());
    }
    let mut observation_ids = HashSet::with_capacity(evidence.source_observations.len());
    for observation in &evidence.source_observations {
        let mut nodes = 0;
        if !valid_semantic_identifier(&observation.id)
            || !observation_ids.insert(observation.id.as_str())
            || !observation
                .inputs
                .as_object()
                .is_some_and(|values| !values.is_empty())
            || !observation
                .expected
                .as_object()
                .is_some_and(|values| !values.is_empty())
            || !validate_observation_value(&observation.inputs, 0, &mut nodes)
            || !validate_observation_value(&observation.expected, 0, &mut nodes)
            || !valid_observation_tolerance(&observation.tolerance)
        {
            return Err(
                "source observation is empty, duplicated, malformed, or unbounded".to_owned(),
            );
        }
        let case_tolerance_is_valid = match observation.case {
            ResolutionEvidenceObservationCase::Invalid
            | ResolutionEvidenceObservationCase::Alias
            | ResolutionEvidenceObservationCase::Cancellation
            | ResolutionEvidenceObservationCase::Architecture => {
                matches!(
                    observation.tolerance.mode,
                    ResolutionEvidenceToleranceMode::Structural
                )
            }
            ResolutionEvidenceObservationCase::Empty => matches!(
                observation.tolerance.mode,
                ResolutionEvidenceToleranceMode::BitExact
                    | ResolutionEvidenceToleranceMode::Structural
            ),
            ResolutionEvidenceObservationCase::Forward
            | ResolutionEvidenceObservationCase::Jvp
            | ResolutionEvidenceObservationCase::Vjp
            | ResolutionEvidenceObservationCase::NonzeroExecution => !matches!(
                observation.tolerance.mode,
                ResolutionEvidenceToleranceMode::Structural
            ),
        };
        let provenance_is_valid = match observation.provenance {
            ResolutionEvidenceObservationProvenance::SourceDerived
            | ResolutionEvidenceObservationProvenance::NativeValidated => true,
            ResolutionEvidenceObservationProvenance::IndependentlyAnalytical => matches!(
                observation.case,
                ResolutionEvidenceObservationCase::Jvp | ResolutionEvidenceObservationCase::Vjp
            ),
        };
        if !case_tolerance_is_valid || !provenance_is_valid {
            return Err("source observation case, provenance, and tolerance disagree".to_owned());
        }
    }
    Ok(())
}

pub(crate) fn validate_resolution_evidence(
    workspace_root: &Path,
    expectation: &ResolutionExpectation<'_>,
) -> Result<(), String> {
    let parsed_semantics = parse_resolution_semantics(expectation)?;
    if !valid_evidence_fixture_path(expectation.evidence_fixture, expectation.resolution_module) {
        return Err("evidence path is outside the exact module root".to_owned());
    }
    let workspace_root = workspace_root
        .canonicalize()
        .map_err(|error| format!("workspace root is unavailable: {error}"))?;
    let module_relative = Path::new(EVIDENCE_PREFIX).join(expectation.resolution_module);
    reject_symlinked_path(&workspace_root, &module_relative, true)?;
    let module_root = workspace_root.join(&module_relative);
    let canonical_module_root = module_root
        .canonicalize()
        .map_err(|error| format!("evidence module root is unavailable: {error}"))?;
    if canonical_module_root != module_root {
        return Err("evidence module root is not the exact canonical directory".to_owned());
    }
    let evidence_relative = Path::new(expectation.evidence_fixture);
    reject_symlinked_path(&workspace_root, evidence_relative, false)?;
    let evidence_path = workspace_root.join(evidence_relative);
    if evidence_path.parent() != Some(module_root.as_path()) {
        return Err("evidence file is not directly below its exact module root".to_owned());
    }
    let canonical_evidence_path = evidence_path
        .canonicalize()
        .map_err(|error| format!("evidence file is unavailable: {error}"))?;
    if canonical_evidence_path.parent() != Some(canonical_module_root.as_path()) {
        return Err("canonical evidence file escaped its exact module root".to_owned());
    }
    let metadata = fs::metadata(&canonical_evidence_path)
        .map_err(|error| format!("evidence metadata is unavailable: {error}"))?;
    if !metadata.is_file() || metadata.len() > MAX_EVIDENCE_FIXTURE_BYTES as u64 {
        return Err("evidence fixture is not a bounded regular file".to_owned());
    }
    let bytes = fs::read(canonical_evidence_path)
        .map_err(|error| format!("evidence file cannot be read: {error}"))?;
    if bytes.len() > MAX_EVIDENCE_FIXTURE_BYTES {
        return Err("evidence bytes exceed the post-read bound".to_owned());
    }
    let actual_digest = format!("{:x}", Sha256::digest(&bytes));
    if actual_digest != expectation.evidence_fixture_sha256 {
        return Err("evidence SHA-256 does not match the declared digest".to_owned());
    }
    let evidence = serde_json::from_slice::<ResolutionEvidence>(&bytes)
        .map_err(|error| format!("evidence JSON is invalid: {error}"))?;
    validate_source_observations(&evidence, &actual_digest)?;
    let semantics = evidence.semantics;
    let identity_matches = evidence.schema_version == 1
        && evidence.resolution_module == expectation.resolution_module
        && evidence.operation_id == expectation.operation_id
        && evidence.baseline_overload_id == expectation.baseline_overload_id
        && evidence.baseline_fixture_sha256 == expectation.baseline_fixture_sha256
        && evidence.overload_id == expectation.overload_id
        && evidence.owner_task_id == expectation.owner_task_id;
    let signature_matches = evidence.ordered_parameters == parsed_semantics.parameters
        && evidence.output_types == parsed_semantics.output_types;
    let semantic_fields_match = [
        (
            semantics.rust_signature.as_str(),
            expectation.rust_signature,
        ),
        (semantics.mutation_rule.as_str(), expectation.mutation_rule),
        (semantics.alias_rule.as_str(), expectation.alias_rule),
        (semantics.shape_rule.as_str(), expectation.shape_rule),
        (semantics.dtype_rule.as_str(), expectation.dtype_rule),
        (
            semantics.accumulation_dtype.as_str(),
            expectation.accumulation_dtype,
        ),
        (semantics.layout_rule.as_str(), expectation.layout_rule),
        (semantics.device_rule.as_str(), expectation.device_rule),
        (semantics.numeric_rule.as_str(), expectation.numeric_rule),
        (semantics.tolerance.as_str(), expectation.tolerance),
        (semantics.determinism.as_str(), expectation.determinism),
        (
            semantics.cancellation_points.as_str(),
            expectation.cancellation_points,
        ),
        (semantics.vjp_rule.as_str(), expectation.vjp_rule),
        (semantics.jvp_rule.as_str(), expectation.jvp_rule),
    ]
    .into_iter()
    .all(|(evidence_value, resolution_value)| evidence_value == resolution_value);
    if !identity_matches || !signature_matches || !semantic_fields_match {
        return Err(format!(
            "evidence identity or semantics do not match the resolution for {}",
            expectation.operation_id
        ));
    }
    Ok(())
}

fn reject_symlinked_path(
    root: &Path,
    relative: &Path,
    expect_directory: bool,
) -> Result<(), String> {
    let mut current = root.to_path_buf();
    let components = relative.components().collect::<Vec<_>>();
    for (index, component) in components.iter().enumerate() {
        let std::path::Component::Normal(component) = component else {
            return Err("evidence path has a non-normal component".to_owned());
        };
        current.push(component);
        let metadata = fs::symlink_metadata(&current)
            .map_err(|error| format!("evidence path component is unavailable: {error}"))?;
        if metadata.file_type().is_symlink() {
            return Err("evidence path contains a symbolic link".to_owned());
        }
        let last = index + 1 == components.len();
        if (!last || expect_directory) && !metadata.is_dir() {
            return Err("evidence parent component is not a directory".to_owned());
        }
        if last && !expect_directory && !metadata.is_file() {
            return Err("evidence leaf is not a regular file".to_owned());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn evidence_value() -> Value {
        json!({
            "schema_version": 1,
            "resolution_module": "module_01",
            "operation_id": "COMFY-TENSOR-OP-000000000001",
            "baseline_overload_id": "COMFY-TENSOR-OP-000000000001:blocked",
            "baseline_fixture_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
            "overload_id": "zed.native.test.v1",
            "owner_task_id": "owner-task",
            "ordered_parameters": [],
            "output_types": ["Tensor<F32>"],
            "semantics": {
                "rust_signature": "fn test() -> Tensor",
                "mutation_rule": "out of place",
                "alias_rule": "fresh output",
                "shape_rule": "preserves shape",
                "dtype_rule": "F32",
                "accumulation_dtype": "F32",
                "layout_rule": "contiguous",
                "device_rule": "CPU",
                "numeric_rule": "identity",
                "tolerance": "bit exact",
                "determinism": "fixed traversal",
                "cancellation_points": "before publication",
                "vjp_rule": "identity",
                "jvp_rule": "identity"
            }
        })
    }

    fn parse(value: Value) -> ResolutionEvidence {
        serde_json::from_value(value).expect("test evidence must deserialize")
    }

    fn valid_profile() -> Value {
        json!({
            "dependency": "torchvision",
            "version": "0.27",
            "profile": "reviewed-source-cpu-profile",
            "fingerprint_sha256": "2222222222222222222222222222222222222222222222222222222222222222"
        })
    }

    fn valid_observation(id: &str) -> Value {
        json!({
            "id": id,
            "case": "forward",
            "provenance": "source_derived",
            "inputs": {"shape": [1], "values": [2.0]},
            "expected": {"shape": [1], "values": [2.0]},
            "tolerance": {"mode": "bit_exact"}
        })
    }

    #[test]
    fn historical_evidence_without_observations_remains_compatible() {
        let evidence = parse(evidence_value());
        assert_eq!(
            validate_source_observations(
                &evidence,
                "3333333333333333333333333333333333333333333333333333333333333333"
            ),
            Ok(())
        );
    }

    #[test]
    fn structured_source_observations_are_strict_and_bounded() {
        let mut value = evidence_value();
        value["source_profile"] = valid_profile();
        value["source_observations"] = json!([valid_observation("forward_vector")]);
        let evidence = parse(value.clone());
        assert_eq!(
            validate_source_observations(
                &evidence,
                "3333333333333333333333333333333333333333333333333333333333333333"
            ),
            Ok(())
        );

        value["source_observations"] = json!([]);
        assert!(
            validate_source_observations(
                &parse(value.clone()),
                "3333333333333333333333333333333333333333333333333333333333333333"
            )
            .is_err()
        );

        value["source_observations"] =
            json!([valid_observation("same"), valid_observation("same")]);
        assert!(
            validate_source_observations(
                &parse(value.clone()),
                "3333333333333333333333333333333333333333333333333333333333333333"
            )
            .is_err()
        );

        value["source_observations"] = Value::Array(
            (0..=MAX_SOURCE_OBSERVATIONS)
                .map(|index| valid_observation(&format!("case_{index}")))
                .collect(),
        );
        assert!(
            validate_source_observations(
                &parse(value.clone()),
                "3333333333333333333333333333333333333333333333333333333333333333"
            )
            .is_err()
        );

        value["source_observations"] = json!([{
            "id": "empty_input",
            "case": "forward",
            "provenance": "source_derived",
            "inputs": {},
            "expected": {"value": 1},
            "tolerance": {"mode": "bit_exact"}
        }]);
        assert!(
            validate_source_observations(
                &parse(value.clone()),
                "3333333333333333333333333333333333333333333333333333333333333333"
            )
            .is_err()
        );

        value["source_observations"] = json!([valid_observation("forward_vector")]);
        value["source_profile"]["fingerprint_sha256"] =
            json!("3333333333333333333333333333333333333333333333333333333333333333");
        assert!(
            validate_source_observations(
                &parse(value),
                "3333333333333333333333333333333333333333333333333333333333333333"
            )
            .is_err()
        );
    }

    #[test]
    fn source_observation_unknown_fields_and_invalid_provenance_are_rejected() {
        let mut value = evidence_value();
        value["source_profile"] = valid_profile();
        let mut observation = valid_observation("forward_vector");
        observation["extra"] = json!(true);
        value["source_observations"] = json!([observation]);
        assert!(serde_json::from_value::<ResolutionEvidence>(value).is_err());

        let mut value = evidence_value();
        value["source_profile"] = valid_profile();
        let mut observation = valid_observation("forward_vector");
        observation["provenance"] = json!("independently_analytical");
        value["source_observations"] = json!([observation]);
        let evidence = parse(value);
        assert!(
            validate_source_observations(
                &evidence,
                "3333333333333333333333333333333333333333333333333333333333333333"
            )
            .is_err()
        );
    }
}
