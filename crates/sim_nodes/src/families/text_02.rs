use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeDynamicInputDescriptor, NativeEffectClass, NativeInputDescriptor, NativeInputRequirement,
    NativeNode, NativeNodeBinding, NativeNodeBindingsFactory, NativeNodeContext,
    NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure, NativeNodeFailureKind,
    NativeNodeOutcome, NativeNodePresentation, NativeOutputDescriptor, NativePortCardinality,
    NativePrimitive, NativeTextFormatError, NativeTextFormatter, NativeTextRegex,
    NativeTextRegexError, NativeTextRegexFlags, NativeValue, built_in_source_schema,
    native_value_type_for_output_schema, native_value_types_for_input_schema,
};
use sim_types::CancellationToken;
use futures::future::BoxFuture;
use std::{collections::BTreeMap, sync::Arc};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "RegexReplace",
    "ReplaceText",
    "StringCompare",
    "StringConcatenate",
    "StringContains",
    "StringFormat",
    "StringLength",
    "StringReplace",
    "StringSubstring",
    "StringTrim",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

const CATEGORY: &str = "text";
const FORMAT_INPUT_NAMES: &[&str] = &[
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
    "t", "u", "v", "w", "x", "y", "z",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TextKind {
    RegexReplace,
    ReplaceText,
    Compare,
    Concatenate,
    Contains,
    Format,
    Length,
    Replace,
    Substring,
    Trim,
}

impl TextKind {
    const fn class_type(self) -> &'static str {
        match self {
            Self::RegexReplace => "RegexReplace",
            Self::ReplaceText => "ReplaceText",
            Self::Compare => "StringCompare",
            Self::Concatenate => "StringConcatenate",
            Self::Contains => "StringContains",
            Self::Format => "StringFormat",
            Self::Length => "StringLength",
            Self::Replace => "StringReplace",
            Self::Substring => "StringSubstring",
            Self::Trim => "StringTrim",
        }
    }

    const fn feature_id(self) -> &'static str {
        match self {
            Self::RegexReplace => "COMFY-NODE-0531",
            Self::ReplaceText => "COMFY-NODE-0537",
            Self::Compare => "COMFY-NODE-0641",
            Self::Concatenate => "COMFY-NODE-0642",
            Self::Contains => "COMFY-NODE-0643",
            Self::Format => "COMFY-NODE-0644",
            Self::Length => "COMFY-NODE-0645",
            Self::Replace => "COMFY-NODE-0646",
            Self::Substring => "COMFY-NODE-0647",
            Self::Trim => "COMFY-NODE-0648",
        }
    }

    const fn implementation_version(self) -> &'static str {
        match self {
            Self::ReplaceText => "source-3b27465f-v1",
            Self::Format => NativeTextFormatter::SEMANTIC_VERSION,
            _ => "source-bb019631-v1",
        }
    }

    const fn display_name(self) -> &'static str {
        match self {
            Self::RegexReplace => "Replace Text (Regex)",
            Self::ReplaceText => "Replace Text (DEPRECATED)",
            Self::Compare => "Compare Text",
            Self::Concatenate => "Concatenate Text",
            Self::Contains => "Contains Text",
            Self::Format => "Format Text",
            Self::Length => "Text Length",
            Self::Replace => "Replace Text",
            Self::Substring => "Substring",
            Self::Trim => "Trim Text",
        }
    }

    const fn description(self) -> &'static str {
        match self {
            Self::RegexReplace => "Find and replace text using regex patterns.",
            Self::ReplaceText => "Replace text in all texts.",
            Self::Format => {
                "Same as Python's string format method. Supports all of Python's format options and features."
            }
            _ => "",
        }
    }

    fn input_names(self) -> &'static [&'static str] {
        match self {
            Self::RegexReplace => &[
                "string",
                "regex_pattern",
                "replace",
                "case_insensitive",
                "multiline",
                "dotall",
                "count",
            ],
            Self::ReplaceText => &["texts", "find", "replace"],
            Self::Compare => &["string_a", "string_b", "mode", "case_sensitive"],
            Self::Concatenate => &["string_a", "string_b", "delimiter"],
            Self::Contains => &["string", "substring", "case_sensitive"],
            Self::Format => &["f_string"],
            Self::Length => &["string"],
            Self::Replace => &["string", "find", "replace"],
            Self::Substring => &["string", "start", "end"],
            Self::Trim => &["string", "mode"],
        }
    }

    const fn output_name(self) -> &'static str {
        match self {
            Self::ReplaceText => "texts",
            Self::Compare => "boolean",
            Self::Contains => "contains",
            Self::Length => "length",
            _ => "string",
        }
    }

    fn search_aliases(self) -> Vec<String> {
        let aliases: &[&str] = match self {
            Self::RegexReplace => &["regex replace", "regex", "pattern replace", "substitution"],
            Self::Compare => &[
                "compare",
                "text match",
                "string equals",
                "starts with",
                "ends with",
            ],
            Self::Concatenate => &[
                "concatenate",
                "text concat",
                "join text",
                "merge text",
                "combine strings",
                "string concat",
                "append text",
                "combine text",
            ],
            Self::Contains => &["contains", "text includes", "string includes"],
            Self::Format => &["string", "format"],
            Self::Length => &["character count", "text size", "string length"],
            Self::Replace => &["replace", "find and replace", "substitute", "swap text"],
            Self::Substring => &["substring", "extract text", "text portion"],
            Self::Trim => &[
                "trim",
                "clean whitespace",
                "remove whitespace",
                "remove spaces",
                "strip",
            ],
            Self::ReplaceText => &[],
        };
        aliases.iter().map(|alias| (*alias).to_owned()).collect()
    }
}

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    [
        TextKind::RegexReplace,
        TextKind::ReplaceText,
        TextKind::Compare,
        TextKind::Concatenate,
        TextKind::Contains,
        TextKind::Format,
        TextKind::Length,
        TextKind::Replace,
        TextKind::Substring,
        TextKind::Trim,
    ]
    .into_iter()
    .map(native_binding)
    .collect()
}

fn native_binding(kind: TextKind) -> Result<NativeNodeBinding, NativeNodeContractError> {
    let catalog_schema = built_in_source_schema(kind.class_type())
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let input_names = kind
        .input_names()
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    let dynamic_schema = catalog_schema.dynamic_inputs.clone();
    let source_schema = catalog_schema
        .bind_execution_ports(
            &input_names,
            &dynamic_schema,
            &[kind.output_name().to_owned()],
        )
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let inputs = catalog_schema
        .inputs
        .iter()
        .map(|input| {
            Ok(NativeInputDescriptor {
                name: input.schema.name.clone(),
                accepted_types: native_value_types_for_input_schema(&input.schema).map_err(
                    |error| NativeNodeContractError::InvalidSourceSchema(error.to_string()),
                )?,
                required: input.requirement == NativeInputRequirement::Required,
                hidden: input.requirement == NativeInputRequirement::Hidden,
                lazy: false,
                cardinality: NativePortCardinality::Scalar,
                allows_literal: true,
            })
        })
        .collect::<Result<Vec<_>, NativeNodeContractError>>()?;
    let dynamic_inputs = dynamic_schema
        .iter()
        .map(|dynamic| {
            Ok(NativeDynamicInputDescriptor {
                name_template: dynamic.identity.clone(),
                start_index: dynamic.start_index,
                minimum_count: dynamic.minimum_count,
                maximum_count: dynamic.maximum_count,
                input: NativeInputDescriptor {
                    name: dynamic.input.name.clone(),
                    accepted_types: native_value_types_for_input_schema(&dynamic.input).map_err(
                        |error| NativeNodeContractError::InvalidSourceSchema(error.to_string()),
                    )?,
                    required: true,
                    hidden: false,
                    lazy: false,
                    cardinality: NativePortCardinality::Scalar,
                    allows_literal: false,
                },
            })
        })
        .collect::<Result<Vec<_>, NativeNodeContractError>>()?;
    let produced_type =
        native_value_type_for_output_schema(source_schema.outputs.first().ok_or_else(|| {
            NativeNodeContractError::InvalidSourceSchema("missing output".into())
        })?)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    Ok(NativeNodeBinding::Executable {
        feature_id: kind.feature_id().to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: kind.class_type().to_owned(),
            implementation_version: kind.implementation_version().to_owned(),
            source_schema: Some(source_schema),
            inputs,
            dynamic_inputs,
            outputs: vec![NativeOutputDescriptor {
                name: kind.output_name().to_owned(),
                produced_type,
                is_list: false,
            }],
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        },
        presentation: NativeNodePresentation {
            display_name: kind.display_name().to_owned(),
            category: CATEGORY.to_owned(),
            description: kind.description().to_owned(),
            output_names: vec![kind.output_name().to_owned()],
            search_aliases: kind.search_aliases(),
            is_deprecated: kind == TextKind::ReplaceText,
            is_experimental: kind == TextKind::ReplaceText,
        },
        node: Arc::new(TextNode { kind }),
    })
}

#[derive(Debug)]
struct TextNode {
    kind: TextKind,
}

impl NativeNode for TextNode {
    fn class_type(&self) -> &str {
        self.kind.class_type()
    }

    fn implementation_version(&self) -> &str {
        self.kind.implementation_version()
    }

    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        validate_inputs(self.kind, inputs)?;
        Ok(format!(
            "{}-{}",
            self.kind.class_type(),
            self.kind.implementation_version()
        ))
    }

    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check_cancellation(&context.cancellation, self.kind.class_type())?;
        validate_inputs(self.kind, inputs)?;
        Ok(NativeCacheDependencies::default())
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check_cancellation(&context.cancellation, self.kind.class_type())?;
            validate_inputs(self.kind, &inputs)?;
            let output = execute_text(self.kind, &inputs, &context.cancellation)?;
            check_cancellation(&context.cancellation, self.kind.class_type())?;
            let outcome = NativeNodeOutcome::Values {
                outputs: vec![output],
                ui: None,
                effects: Vec::new(),
            };
            outcome
                .validate()
                .map_err(|error| invalid_inputs(error.to_string()))?;
            Ok(outcome)
        })
    }
}

fn validate_inputs(
    kind: TextKind,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(), NativeNodeFailure> {
    if kind == TextKind::Format {
        return validate_format_inputs(inputs);
    }
    let required_count = if kind == TextKind::RegexReplace {
        3
    } else {
        kind.input_names().len()
    };
    if inputs.len() < required_count
        || inputs.len() > kind.input_names().len()
        || inputs
            .keys()
            .any(|name| !kind.input_names().contains(&name.as_str()))
        || !kind
            .input_names()
            .iter()
            .take(required_count)
            .all(|name| inputs.contains_key(*name))
    {
        return Err(invalid_inputs(format!(
            "{} received a missing or unknown input",
            kind.class_type()
        )));
    }
    for (name, value) in inputs {
        value
            .validate()
            .map_err(|error| invalid_inputs(error.to_string()))?;
        let expected = input_type(kind, name)?;
        if !expected.accepts(value) {
            return Err(invalid_inputs(format!(
                "{} input {name} has the wrong type",
                kind.class_type()
            )));
        }
    }
    if kind == TextKind::RegexReplace {
        let count = optional_integer(inputs, "count", 0)?;
        if !(0..=100).contains(&count) {
            return Err(invalid_inputs("count must be between 0 and 100"));
        }
    }
    Ok(())
}

fn validate_format_inputs(inputs: &BTreeMap<String, NativeValue>) -> Result<(), NativeNodeFailure> {
    required_string(inputs, "f_string")?;
    if inputs.len() > FORMAT_INPUT_NAMES.len() + 1
        || inputs
            .keys()
            .any(|name| name != "f_string" && !FORMAT_INPUT_NAMES.contains(&name.as_str()))
    {
        return Err(invalid_inputs("StringFormat received an unknown input"));
    }
    for (name, value) in inputs {
        value
            .validate()
            .map_err(|error| invalid_inputs(error.to_string()))?;
        if name != "f_string" && matches!(value, NativeValue::List { .. }) {
            return Err(invalid_inputs(format!(
                "StringFormat input {name} must be scalar"
            )));
        }
    }
    Ok(())
}

fn input_type(kind: TextKind, name: &str) -> Result<crate::NativeTypeUnion, NativeNodeFailure> {
    let catalog_schema = built_in_source_schema(kind.class_type())
        .map_err(|error| invalid_inputs(error.to_string()))?;
    let schema = catalog_schema
        .inputs
        .iter()
        .find(|input| input.schema.name == name)
        .ok_or_else(|| invalid_inputs(format!("unknown input {name}")))?;
    native_value_types_for_input_schema(&schema.schema)
        .map_err(|error| invalid_inputs(error.to_string()))
}

fn execute_text(
    kind: TextKind,
    inputs: &BTreeMap<String, NativeValue>,
    cancellation: &CancellationToken,
) -> Result<NativeValue, NativeNodeFailure> {
    match kind {
        TextKind::RegexReplace => regex_replace(inputs, cancellation).map(string_value),
        TextKind::ReplaceText => Ok(string_value(required_string(inputs, "texts")?.replace(
            required_string(inputs, "find")?,
            required_string(inputs, "replace")?,
        ))),
        TextKind::Compare => Ok(boolean_value(compare_strings(inputs)?)),
        TextKind::Concatenate => Ok(string_value(
            [
                required_string(inputs, "string_a")?,
                required_string(inputs, "string_b")?,
            ]
            .join(required_string(inputs, "delimiter")?),
        )),
        TextKind::Contains => Ok(boolean_value(contains_string(inputs)?)),
        TextKind::Format => format_string(inputs, cancellation).map(string_value),
        TextKind::Length => {
            let length = i64::try_from(required_string(inputs, "string")?.chars().count())
                .map_err(|_| invalid_inputs("string length exceeds INT range"))?;
            Ok(integer_value(length))
        }
        TextKind::Replace => Ok(string_value(required_string(inputs, "string")?.replace(
            required_string(inputs, "find")?,
            required_string(inputs, "replace")?,
        ))),
        TextKind::Substring => Ok(string_value(python_substring(
            required_string(inputs, "string")?,
            required_integer(inputs, "start")?,
            required_integer(inputs, "end")?,
        ))),
        TextKind::Trim => Ok(string_value(trim_string(
            required_string(inputs, "string")?,
            required_string(inputs, "mode")?,
        ))),
    }
}

fn regex_replace(
    inputs: &BTreeMap<String, NativeValue>,
    cancellation: &CancellationToken,
) -> Result<String, NativeNodeFailure> {
    let regex = NativeTextRegex::checked(
        required_string(inputs, "regex_pattern")?,
        NativeTextRegexFlags {
            case_insensitive: optional_boolean(inputs, "case_insensitive", true)?,
            multi_line: optional_boolean(inputs, "multiline", false)?,
            dot_matches_new_line: optional_boolean(inputs, "dotall", false)?,
        },
    )
    .map_err(|error| regex_failure("RegexReplace", error))?;
    let count = usize::try_from(optional_integer(inputs, "count", 0)?)
        .map_err(|_| invalid_inputs("count must be non-negative"))?;
    regex
        .replace(
            required_string(inputs, "string")?,
            required_string(inputs, "replace")?,
            count,
            cancellation,
        )
        .map_err(|error| regex_failure("RegexReplace", error))
}

fn compare_strings(inputs: &BTreeMap<String, NativeValue>) -> Result<bool, NativeNodeFailure> {
    let string_a = required_string(inputs, "string_a")?;
    let string_b = required_string(inputs, "string_b")?;
    let (left, right) = if required_boolean(inputs, "case_sensitive")? {
        (string_a.to_owned(), string_b.to_owned())
    } else {
        (string_a.to_lowercase(), string_b.to_lowercase())
    };
    match required_string(inputs, "mode")? {
        "Equal" => Ok(left == right),
        "Starts With" => Ok(left.starts_with(&right)),
        "Ends With" => Ok(left.ends_with(&right)),
        mode => Err(invalid_inputs(format!(
            "unsupported comparison mode {mode}"
        ))),
    }
}

fn contains_string(inputs: &BTreeMap<String, NativeValue>) -> Result<bool, NativeNodeFailure> {
    let string = required_string(inputs, "string")?;
    let substring = required_string(inputs, "substring")?;
    if required_boolean(inputs, "case_sensitive")? {
        Ok(string.contains(substring))
    } else {
        Ok(string.to_lowercase().contains(&substring.to_lowercase()))
    }
}

fn format_string(
    inputs: &BTreeMap<String, NativeValue>,
    cancellation: &CancellationToken,
) -> Result<String, NativeNodeFailure> {
    let values = inputs
        .iter()
        .filter(|(name, _)| name.as_str() != "f_string")
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    NativeTextFormatter::format(required_string(inputs, "f_string")?, &values, cancellation)
        .map_err(|error| format_failure("StringFormat", error))
}

fn python_substring(value: &str, start: i64, end: i64) -> String {
    let characters = value.chars().collect::<Vec<_>>();
    let start = python_slice_index(start, characters.len());
    let end = python_slice_index(end, characters.len());
    if start >= end {
        String::new()
    } else {
        characters[start..end].iter().collect()
    }
}

fn python_slice_index(index: i64, length: usize) -> usize {
    let length = length as i128;
    let index = index as i128;
    if index < 0 {
        (length + index).clamp(0, length) as usize
    } else {
        index.min(length) as usize
    }
}

fn trim_string<'a>(value: &'a str, mode: &str) -> &'a str {
    match mode {
        "Both" => value.trim_matches(python_whitespace),
        "Left" => value.trim_start_matches(python_whitespace),
        "Right" => value.trim_end_matches(python_whitespace),
        _ => value,
    }
}

fn python_whitespace(character: char) -> bool {
    character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
}

fn required_string<'a>(
    inputs: &'a BTreeMap<String, NativeValue>,
    name: &str,
) -> Result<&'a str, NativeNodeFailure> {
    match inputs.get(name) {
        Some(NativeValue::Primitive {
            value: NativePrimitive::String(value),
        }) => Ok(value),
        _ => Err(invalid_inputs(format!("{name} must be a STRING"))),
    }
}

fn required_boolean(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
) -> Result<bool, NativeNodeFailure> {
    match inputs.get(name) {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Boolean(value),
        }) => Ok(*value),
        _ => Err(invalid_inputs(format!("{name} must be a BOOLEAN"))),
    }
}

fn optional_boolean(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
    default: bool,
) -> Result<bool, NativeNodeFailure> {
    if inputs.contains_key(name) {
        required_boolean(inputs, name)
    } else {
        Ok(default)
    }
}

fn required_integer(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
) -> Result<i64, NativeNodeFailure> {
    match inputs.get(name) {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => Ok(*value),
        Some(NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }) => i64::try_from(*value).map_err(|_| invalid_inputs(format!("{name} is too large"))),
        _ => Err(invalid_inputs(format!("{name} must be an INT"))),
    }
}

fn optional_integer(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
    default: i64,
) -> Result<i64, NativeNodeFailure> {
    if inputs.contains_key(name) {
        required_integer(inputs, name)
    } else {
        Ok(default)
    }
}

fn string_value(value: impl Into<String>) -> NativeValue {
    NativeValue::Primitive {
        value: NativePrimitive::String(value.into()),
    }
}

fn boolean_value(value: bool) -> NativeValue {
    NativeValue::Primitive {
        value: NativePrimitive::Boolean(value),
    }
}

fn integer_value(value: i64) -> NativeValue {
    NativeValue::Primitive {
        value: NativePrimitive::Integer(value),
    }
}

fn check_cancellation(
    cancellation: &CancellationToken,
    class_type: &str,
) -> Result<(), NativeNodeFailure> {
    cancellation
        .check()
        .map_err(|_| interrupted_failure(class_type))
}

fn regex_failure(class_type: &str, error: NativeTextRegexError) -> NativeNodeFailure {
    if error == NativeTextRegexError::Cancelled {
        interrupted_failure(class_type)
    } else {
        NativeNodeFailure {
            code: "native_text_regex_failed".to_owned(),
            message: format!("{class_type} failed: {error}"),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        }
    }
}

fn format_failure(class_type: &str, error: NativeTextFormatError) -> NativeNodeFailure {
    if error == NativeTextFormatError::Cancelled {
        interrupted_failure(class_type)
    } else {
        NativeNodeFailure {
            code: "native_text_format_failed".to_owned(),
            message: format!("{class_type} failed: {error}"),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        }
    }
}

fn invalid_inputs(message: impl Into<String>) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_node_inputs".to_owned(),
        message: message.into(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

fn interrupted_failure(class_type: &str) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "execution_interrupted".to_owned(),
        message: format!("{class_type} execution was interrupted"),
        kind: NativeNodeFailureKind::Interrupted,
        retryable: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        NativeHandleStore, NativeHandleStoreError, NativeHandleStoreIdentity, NativeHandleType,
        NativeOpaqueHandle, NativeResolvedPayload, NativeStoredPayload, NodeRegistry,
    };
    use sim_tensor::CpuWorkspaceAuthority;
    use sim_types::{AttemptId, NodeId, PromptId};
    use serde_json::Value;
    use std::error::Error;
    use uuid::Uuid;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/nodes/text-comfy-node-0531/fixture.json"
    ));

    #[derive(Debug)]
    struct RejectingStore {
        identity: NativeHandleStoreIdentity,
        attempt_id: AttemptId,
    }

    impl NativeHandleStore for RejectingStore {
        fn identity(&self) -> NativeHandleStoreIdentity {
            self.identity
        }

        fn attempt_id(&self) -> AttemptId {
            self.attempt_id
        }

        fn resolve(
            &self,
            handle: &NativeOpaqueHandle,
            _expected_type: &NativeHandleType,
            _cancellation: &CancellationToken,
        ) -> Result<NativeResolvedPayload, NativeHandleStoreError> {
            Err(NativeHandleStoreError::Missing(
                handle.identifier().to_owned(),
            ))
        }

        fn publish(
            &self,
            _payload: NativeStoredPayload,
            _cancellation: &CancellationToken,
        ) -> Result<NativeOpaqueHandle, NativeHandleStoreError> {
            Err(NativeHandleStoreError::Rejected(
                "text nodes do not publish handles".to_owned(),
            ))
        }

        fn revoke(
            &self,
            handle: &NativeOpaqueHandle,
            _cancellation: &CancellationToken,
        ) -> Result<(), NativeHandleStoreError> {
            Err(NativeHandleStoreError::Missing(
                handle.identifier().to_owned(),
            ))
        }
    }

    fn context(cancellation: CancellationToken) -> Result<NativeNodeContext, Box<dyn Error>> {
        let attempt_id = AttemptId(Uuid::from_u128(0x47501));
        let (_backend, workspace) = CpuWorkspaceAuthority::create_backend(1)?;
        Ok(NativeNodeContext::new(
            PromptId(Uuid::from_u128(0x47502)),
            attempt_id,
            NodeId("text-family-part-two-test".to_owned()),
            cancellation,
            workspace.authorize_workspace(0)?,
            Arc::new(RejectingStore {
                identity: NativeHandleStoreIdentity::new(
                    Uuid::from_u128(0x47503),
                    Uuid::from_u128(0x47504),
                )?,
                attempt_id,
            }),
        )?)
    }

    fn executable(class_type: &str) -> Result<Arc<dyn NativeNode>, Box<dyn Error>> {
        native_node_bindings()?
            .into_iter()
            .find_map(|binding| match binding {
                NativeNodeBinding::Executable {
                    descriptor, node, ..
                } if descriptor.class_type == class_type => Some(node),
                _ => None,
            })
            .ok_or_else(|| format!("{class_type} executable binding is absent").into())
    }

    fn execute(
        class_type: &str,
        inputs: BTreeMap<String, NativeValue>,
    ) -> Result<NativeValue, Box<dyn Error>> {
        let outcome = futures::executor::block_on(
            executable(class_type)?.execute(context(CancellationToken::default())?, inputs),
        )?;
        let NativeNodeOutcome::Values {
            mut outputs,
            ui,
            effects,
        } = outcome
        else {
            return Err("text node did not return values".into());
        };
        assert!(ui.is_none());
        assert!(effects.is_empty());
        outputs
            .pop()
            .filter(|_| outputs.is_empty())
            .ok_or_else(|| "text node returned the wrong output count".into())
    }

    fn string(value: &str) -> NativeValue {
        string_value(value)
    }

    fn integer(value: i64) -> NativeValue {
        integer_value(value)
    }

    fn boolean(value: bool) -> NativeValue {
        boolean_value(value)
    }

    fn output_string(value: NativeValue) -> Result<String, Box<dyn Error>> {
        match value {
            NativeValue::Primitive {
                value: NativePrimitive::String(value),
            } => Ok(value),
            _ => Err("output was not a string".into()),
        }
    }

    #[test]
    fn source_fixture_and_all_exact_schemas_are_registered() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        assert_eq!(
            fixture.get("stable_task_id").and_then(Value::as_str),
            Some("comfy-parity-native-nodes-text-comfy-node-0531")
        );
        let bindings = native_node_bindings()?;
        assert_eq!(bindings.len(), NODE_DESCRIPTOR_IDS.len());
        let registry = NodeRegistry::built_in()?;
        for (binding, class_type) in bindings.iter().zip(NODE_DESCRIPTOR_IDS) {
            assert_eq!(binding.descriptor().class_type, *class_type);
            assert_eq!(binding.descriptor().effect, NativeEffectClass::Pure);
            assert_eq!(binding.descriptor().cache, NativeCachePolicy::InputIdentity);
            binding.validate()?;
            registry.validate_native_binding(binding)?;
        }
        let format = bindings
            .iter()
            .find(|binding| binding.descriptor().class_type == "StringFormat")
            .ok_or("StringFormat descriptor is absent")?;
        assert_eq!(format.descriptor().inputs.len(), 1);
        assert_eq!(format.descriptor().dynamic_inputs.len(), 1);
        assert_eq!(
            format.descriptor().dynamic_inputs[0].name_template,
            "{name}"
        );
        assert_eq!(format.descriptor().dynamic_inputs[0].minimum_count, 0);
        assert_eq!(format.descriptor().dynamic_inputs[0].maximum_count, 26);
        let regex = bindings
            .iter()
            .find(|binding| binding.descriptor().class_type == "RegexReplace")
            .ok_or("RegexReplace descriptor is absent")?;
        assert!(
            regex.descriptor().inputs[..3]
                .iter()
                .all(|input| input.required)
        );
        assert!(
            regex.descriptor().inputs[3..]
                .iter()
                .all(|input| !input.required)
        );
        Ok(())
    }

    #[test]
    fn plain_text_operations_match_python_string_behavior() -> Result<(), Box<dyn Error>> {
        assert_eq!(
            output_string(execute(
                "StringConcatenate",
                BTreeMap::from([
                    ("string_a".to_owned(), string("alpha")),
                    ("string_b".to_owned(), string("βeta")),
                    ("delimiter".to_owned(), string(" → ")),
                ]),
            )?)?,
            "alpha → βeta"
        );
        assert_eq!(
            execute(
                "StringLength",
                BTreeMap::from([("string".to_owned(), string("A😀é"))]),
            )?,
            integer(3)
        );
        assert_eq!(
            output_string(execute(
                "StringSubstring",
                BTreeMap::from([
                    ("string".to_owned(), string("A😀éZ")),
                    ("start".to_owned(), integer(-3)),
                    ("end".to_owned(), integer(-1)),
                ]),
            )?)?,
            "😀é"
        );
        assert_eq!(
            output_string(execute(
                "StringReplace",
                BTreeMap::from([
                    ("string".to_owned(), string("😀a")),
                    ("find".to_owned(), string("")),
                    ("replace".to_owned(), string("-")),
                ]),
            )?)?,
            "-😀-a-"
        );
        assert_eq!(
            output_string(execute(
                "StringTrim",
                BTreeMap::from([
                    ("string".to_owned(), string("\u{2003}text \n")),
                    ("mode".to_owned(), string("Both")),
                ]),
            )?)?,
            "text"
        );
        assert_eq!(trim_string("\u{1c}text\u{1f}", "Both"), "text");
        Ok(())
    }

    #[test]
    fn comparisons_contains_and_deprecated_replace_are_exact() -> Result<(), Box<dyn Error>> {
        assert_eq!(
            execute(
                "StringCompare",
                BTreeMap::from([
                    ("string_a".to_owned(), string("CAFÉ noir")),
                    ("string_b".to_owned(), string("café")),
                    ("mode".to_owned(), string("Starts With")),
                    ("case_sensitive".to_owned(), boolean(false)),
                ]),
            )?,
            boolean(true)
        );
        assert_eq!(
            execute(
                "StringContains",
                BTreeMap::from([
                    ("string".to_owned(), string("One TWO three")),
                    ("substring".to_owned(), string("two")),
                    ("case_sensitive".to_owned(), boolean(false)),
                ]),
            )?,
            boolean(true)
        );
        assert_eq!(
            output_string(execute(
                "ReplaceText",
                BTreeMap::from([
                    ("texts".to_owned(), string("one one")),
                    ("find".to_owned(), string("one")),
                    ("replace".to_owned(), string("two")),
                ]),
            )?)?,
            "two two"
        );
        Ok(())
    }

    #[test]
    fn regex_replace_and_format_delegate_to_canonical_owners() -> Result<(), Box<dyn Error>> {
        let regex_inputs = BTreeMap::from([
            ("string".to_owned(), string("A-1 b-2 C-3")),
            ("regex_pattern".to_owned(), string(r"([a-z])-(\d)")),
            ("replace".to_owned(), string(r"\2<\1>")),
            ("case_insensitive".to_owned(), boolean(true)),
            ("multiline".to_owned(), boolean(false)),
            ("dotall".to_owned(), boolean(false)),
            ("count".to_owned(), integer(2)),
        ]);
        assert_eq!(
            output_string(execute("RegexReplace", regex_inputs)?)?,
            "1<A> 2<b> C-3"
        );
        assert_eq!(
            output_string(execute(
                "RegexReplace",
                BTreeMap::from([
                    ("string".to_owned(), string("ONE two")),
                    ("regex_pattern".to_owned(), string("one")),
                    ("replace".to_owned(), string("1")),
                ]),
            )?)?,
            "1 two"
        );
        assert_eq!(
            output_string(execute(
                "StringFormat",
                BTreeMap::from([
                    ("f_string".to_owned(), string("{a!r:>8} / {b:04d}")),
                    ("a".to_owned(), string("é")),
                    ("b".to_owned(), integer(7)),
                ]),
            )?)?,
            "     'é' / 0007"
        );
        Ok(())
    }

    #[test]
    fn validation_cancellation_cache_and_recovery_fail_closed() -> Result<(), Box<dyn Error>> {
        let regex = executable("RegexReplace")?;
        let invalid = BTreeMap::from([
            ("string".to_owned(), string("text")),
            ("regex_pattern".to_owned(), string("(")),
            ("replace".to_owned(), string("x")),
        ]);
        let failure = futures::executor::block_on(
            regex.execute(context(CancellationToken::default())?, invalid),
        )
        .expect_err("invalid regex must fail");
        assert_eq!(failure.code, "native_text_regex_failed");

        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let failure = futures::executor::block_on(regex.execute(
            context(cancellation)?,
            BTreeMap::from([
                ("string".to_owned(), string("text")),
                ("regex_pattern".to_owned(), string("text")),
                ("replace".to_owned(), string("fresh")),
            ]),
        ))
        .expect_err("cancelled regex must fail");
        assert_eq!(failure.kind, NativeNodeFailureKind::Interrupted);

        let format = executable("StringFormat")?;
        let missing = BTreeMap::from([("f_string".to_owned(), string("{a}"))]);
        let failure = futures::executor::block_on(
            format.execute(context(CancellationToken::default())?, missing),
        )
        .expect_err("missing format field must fail");
        assert_eq!(failure.code, "native_text_format_failed");
        let fresh = BTreeMap::from([
            ("f_string".to_owned(), string("{a}")),
            ("a".to_owned(), string("recovered")),
        ]);
        assert_eq!(
            format.cache_change_token(&fresh)?,
            "StringFormat-source-bb019631-text-v4"
        );
        assert_eq!(output_string(execute("StringFormat", fresh)?)?, "recovered");
        Ok(())
    }

    #[test]
    fn source_v4_string_format_successes_use_selected_canonical_node() -> Result<(), Box<dyn Error>>
    {
        for (template, value, expected) in [
            ("{a:*>08d}", integer(7), "*******7"),
            ("{a:05s}", string("x"), "x0000"),
            ("{a:05c}", integer(65), "0000A"),
            ("{a:٠٨d}", integer(7), "       7"),
            ("{a:٠٨.٣s}", string("abcdef"), "abc     "),
            (
                "{a:.١٧f}",
                NativeValue::Primitive {
                    value: NativePrimitive::Number(0.1),
                },
                "0.10000000000000001",
            ),
            ("{a:f}", integer(7), "7.000000"),
            ("{a:0=8,d}", integer(23), "0,000,023"),
            ("{a:,}", integer(1234), "1,234"),
            ("{a!s:05s}", integer(7), "70000"),
        ] {
            assert_eq!(
                output_string(execute(
                    "StringFormat",
                    BTreeMap::from([
                        ("f_string".to_owned(), string(template)),
                        ("a".to_owned(), value),
                    ])
                )?)?,
                expected,
                "original source template {template}"
            );
        }
        Ok(())
    }

    #[test]
    fn source_v4_string_format_error_phases_cancel_and_recovery_use_selected_node()
    -> Result<(), Box<dyn Error>> {
        let node = executable("StringFormat")?;
        for (template, value, expected) in [
            (
                "{b!q}",
                integer(7),
                NativeTextFormatError::MissingField("b".to_owned()),
            ),
            (
                "{0!q}",
                integer(7),
                NativeTextFormatError::InvalidFieldAccess("0".to_owned()),
            ),
            (
                "{a!s:{b!q}}",
                integer(7),
                NativeTextFormatError::MissingField("b".to_owned()),
            ),
            (
                "{a!q:{b!qq}}",
                integer(7),
                NativeTextFormatError::InvalidTemplate(
                    "conversion must be exactly !s, !r, or !a".to_owned(),
                ),
            ),
            (
                "{a!qq:{b}}",
                integer(7),
                NativeTextFormatError::InvalidTemplate(
                    "conversion must be exactly !s, !r, or !a".to_owned(),
                ),
            ),
            (
                "{a:.2q}",
                integer(7),
                NativeTextFormatError::InvalidTemplate(
                    "unsupported integer format type".to_owned(),
                ),
            ),
            (
                "{a[1]!q}",
                string("x"),
                NativeTextFormatError::InvalidFieldAccess("a[1]".to_owned()),
            ),
            (
                "{a.unknown!q}",
                integer(7),
                NativeTextFormatError::InvalidFieldAccess("a.unknown".to_owned()),
            ),
        ] {
            let failure = futures::executor::block_on(node.execute(
                context(CancellationToken::default())?,
                BTreeMap::from([
                    ("f_string".to_owned(), string(template)),
                    ("a".to_owned(), value),
                ]),
            ))
            .expect_err("source-proved error must not publish node outputs");
            assert_eq!(failure.code, "native_text_format_failed");
            assert_eq!(failure.kind, NativeNodeFailureKind::Failure);
            assert!(!failure.retryable);
            assert_eq!(
                failure.message,
                format!("StringFormat failed: {expected}"),
                "original source template {template}"
            );
        }
        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        let failure = futures::executor::block_on(node.execute(
            context(cancellation)?,
            BTreeMap::from([
                ("f_string".to_owned(), string("{b!qq}")),
                ("a".to_owned(), integer(7)),
            ]),
        ))
        .expect_err("cancelled formatter must not publish outputs");
        assert_eq!(failure.kind, NativeNodeFailureKind::Interrupted);
        let fresh = BTreeMap::from([
            ("f_string".to_owned(), string("{a:05s}")),
            ("a".to_owned(), string("fresh")),
        ]);
        assert_eq!(
            node.cache_change_token(&fresh)?,
            "StringFormat-source-bb019631-text-v4"
        );
        assert_eq!(output_string(execute("StringFormat", fresh)?)?, "fresh");
        Ok(())
    }
}
