use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeInputDescriptor, NativeNode, NativeNodeBinding,
    NativeNodeBindingsFactory, NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor,
    NativeNodeFailure, NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation,
    NativeOutputDescriptor, NativePortCardinality, NativePrimitive, NativePrimitiveType,
    NativeTextFormatError, NativeTextFormatter, NativeTextRegex, NativeTextRegexError,
    NativeTextRegexFlags, NativeTypeUnion, NativeValue, NativeValueType, built_in_source_schema,
};
use sim_types::CancellationToken;
use futures::future::BoxFuture;
use serde_json::{Map, Value};
use std::{collections::BTreeMap, sync::Arc};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "AddTextPrefix",
    "AddTextSuffix",
    "BuildJsonPromptIdeogram",
    "CaseConverter",
    "ConvertArrayToString",
    "ConvertDictionaryToString",
    "JsonExtractString",
    "MergeTextLists",
    "RegexExtract",
    "RegexMatch",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

const CATEGORY: &str = "text";
const MAX_JSON_INDENT: i64 = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TextKind {
    AddPrefix,
    AddSuffix,
    BuildIdeogramPrompt,
    CaseConverter,
    ArrayToString,
    DictionaryToString,
    JsonExtractString,
    MergeTextLists,
    RegexExtract,
    RegexMatch,
}

impl TextKind {
    const fn class_type(self) -> &'static str {
        match self {
            Self::AddPrefix => "AddTextPrefix",
            Self::AddSuffix => "AddTextSuffix",
            Self::BuildIdeogramPrompt => "BuildJsonPromptIdeogram",
            Self::CaseConverter => "CaseConverter",
            Self::ArrayToString => "ConvertArrayToString",
            Self::DictionaryToString => "ConvertDictionaryToString",
            Self::JsonExtractString => "JsonExtractString",
            Self::MergeTextLists => "MergeTextLists",
            Self::RegexExtract => "RegexExtract",
            Self::RegexMatch => "RegexMatch",
        }
    }

    const fn feature_id(self) -> &'static str {
        match self {
            Self::AddPrefix => "COMFY-NODE-0002",
            Self::AddSuffix => "COMFY-NODE-0003",
            Self::BuildIdeogramPrompt => "COMFY-NODE-0030",
            Self::CaseConverter => "COMFY-NODE-0046",
            Self::ArrayToString => "COMFY-NODE-0110",
            Self::DictionaryToString => "COMFY-NODE-0111",
            Self::JsonExtractString => "COMFY-NODE-0276",
            Self::MergeTextLists => "COMFY-NODE-0407",
            Self::RegexExtract => "COMFY-NODE-0529",
            Self::RegexMatch => "COMFY-NODE-0530",
        }
    }

    const fn implementation_version(self) -> &'static str {
        match self {
            Self::AddPrefix | Self::AddSuffix | Self::MergeTextLists => "source-3b27465f-v1",
            Self::BuildIdeogramPrompt => "source-4808ede6-v1",
            Self::JsonExtractString => NativeTextFormatter::SEMANTIC_VERSION,
            _ => "source-bb019631-v1",
        }
    }

    const fn display_name(self) -> &'static str {
        match self {
            Self::AddPrefix => "Add Text Prefix (DEPRECATED)",
            Self::AddSuffix => "Add Text Suffix (DEPRECATED)",
            Self::BuildIdeogramPrompt => "Build JSON Prompt (Ideogram)",
            Self::CaseConverter => "Convert Text Case",
            Self::ArrayToString => "Convert Array to String",
            Self::DictionaryToString => "Convert Dictionary to String",
            Self::JsonExtractString => "Extract Text from JSON",
            Self::MergeTextLists => "Merge Text Lists (DEPRECATED)",
            Self::RegexExtract => "Extract Text",
            Self::RegexMatch => "Match Text",
        }
    }

    const fn description(self) -> &'static str {
        match self {
            Self::AddPrefix => "Add a prefix to all texts.",
            Self::AddSuffix => "Add a suffix to all texts.",
            Self::BuildIdeogramPrompt => "Build a JSON prompt for the Ideogram 4 model.",
            Self::MergeTextLists => "Concatenate multiple text lists into one.",
            _ => "",
        }
    }

    fn input_names(self) -> &'static [&'static str] {
        match self {
            Self::AddPrefix => &["texts", "prefix"],
            Self::AddSuffix => &["texts", "suffix"],
            Self::BuildIdeogramPrompt => &[
                "element",
                "high_level_description",
                "background",
                "style",
                "aesthetics",
                "lighting",
                "medium",
                "color_palette",
            ],
            Self::CaseConverter => &["string", "mode"],
            Self::ArrayToString => &["array", "indent"],
            Self::DictionaryToString => &["dictionary", "indent"],
            Self::JsonExtractString => &["json_string", "key"],
            Self::MergeTextLists => &["texts"],
            Self::RegexExtract => &[
                "string",
                "regex_pattern",
                "mode",
                "case_insensitive",
                "multiline",
                "dotall",
                "group_index",
            ],
            Self::RegexMatch => &[
                "string",
                "regex_pattern",
                "case_insensitive",
                "multiline",
                "dotall",
            ],
        }
    }

    const fn output_name(self) -> &'static str {
        match self {
            Self::AddPrefix | Self::AddSuffix | Self::MergeTextLists => "texts",
            Self::BuildIdeogramPrompt => "prompt",
            Self::RegexMatch => "matches",
            _ => "string",
        }
    }

    fn search_aliases(self) -> Vec<String> {
        let aliases: &[&str] = match self {
            Self::CaseConverter => &[
                "case converter",
                "text case",
                "uppercase",
                "lowercase",
                "capitalize",
            ],
            Self::ArrayToString => &[
                "json",
                "list to json",
                "stringify",
                "serialize",
                "list to string",
                "array to json",
            ],
            Self::DictionaryToString => &[
                "json",
                "dict to json",
                "stringify",
                "serialize",
                "dict to string",
            ],
            Self::JsonExtractString => &[
                "json",
                "extract json",
                "parse json",
                "json value",
                "read json",
            ],
            Self::RegexExtract => &[
                "regex extract",
                "regex",
                "pattern extract",
                "text parser",
                "parse text",
            ],
            Self::RegexMatch => &[
                "regex match",
                "regex",
                "pattern match",
                "text contains",
                "string match",
            ],
            _ => &[],
        };
        aliases.iter().map(|alias| (*alias).to_owned()).collect()
    }
}

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    [
        TextKind::AddPrefix,
        TextKind::AddSuffix,
        TextKind::BuildIdeogramPrompt,
        TextKind::CaseConverter,
        TextKind::ArrayToString,
        TextKind::DictionaryToString,
        TextKind::JsonExtractString,
        TextKind::MergeTextLists,
        TextKind::RegexExtract,
        TextKind::RegexMatch,
    ]
    .into_iter()
    .map(native_binding)
    .collect()
}

fn native_binding(kind: TextKind) -> Result<NativeNodeBinding, NativeNodeContractError> {
    let input_names = kind
        .input_names()
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    let output_names = vec![kind.output_name().to_owned()];
    let source_schema = built_in_source_schema(kind.class_type())
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
        .bind_execution_ports(&input_names, &[], &output_names)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let inputs = kind
        .input_names()
        .iter()
        .map(|name| input_descriptor(kind, name))
        .collect::<Result<Vec<_>, _>>()?;
    let output_type = match kind {
        TextKind::BuildIdeogramPrompt => NativeValueType::NamedPreservedUnknown("DICT".to_owned()),
        TextKind::RegexMatch => NativeValueType::Primitive(NativePrimitiveType::Boolean),
        _ => NativeValueType::Primitive(NativePrimitiveType::String),
    };
    Ok(NativeNodeBinding::Executable {
        feature_id: kind.feature_id().to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: kind.class_type().to_owned(),
            implementation_version: kind.implementation_version().to_owned(),
            source_schema: Some(source_schema),
            inputs,
            dynamic_inputs: Vec::new(),
            outputs: vec![NativeOutputDescriptor {
                name: kind.output_name().to_owned(),
                produced_type: output_type,
                is_list: kind == TextKind::MergeTextLists,
            }],
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        },
        presentation: NativeNodePresentation {
            display_name: kind.display_name().to_owned(),
            category: CATEGORY.to_owned(),
            description: kind.description().to_owned(),
            output_names,
            search_aliases: kind.search_aliases(),
            is_deprecated: matches!(
                kind,
                TextKind::AddPrefix | TextKind::AddSuffix | TextKind::MergeTextLists
            ),
            is_experimental: matches!(
                kind,
                TextKind::AddPrefix
                    | TextKind::AddSuffix
                    | TextKind::BuildIdeogramPrompt
                    | TextKind::MergeTextLists
            ),
        },
        node: Arc::new(TextNode { kind }),
    })
}

fn input_descriptor(
    kind: TextKind,
    name: &str,
) -> Result<NativeInputDescriptor, NativeNodeContractError> {
    let value_type = match (kind, name) {
        (TextKind::BuildIdeogramPrompt, "element") | (TextKind::ArrayToString, "array") => {
            NativeValueType::NamedPreservedUnknown("ARRAY".to_owned())
        }
        (TextKind::BuildIdeogramPrompt, "style") => {
            NativeValueType::NamedPreservedUnknown("COMFY_DYNAMICCOMBO_V3".to_owned())
        }
        (TextKind::BuildIdeogramPrompt, "color_palette") => {
            NativeValueType::NamedPreservedUnknown("COLORS".to_owned())
        }
        (TextKind::DictionaryToString, "dictionary") => {
            NativeValueType::NamedPreservedUnknown("DICT".to_owned())
        }
        (_, "case_insensitive" | "multiline" | "dotall") => {
            NativeValueType::Primitive(NativePrimitiveType::Boolean)
        }
        (_, "indent" | "group_index") => NativeValueType::Primitive(NativePrimitiveType::Integer),
        _ => NativeValueType::Primitive(NativePrimitiveType::String),
    };
    Ok(NativeInputDescriptor {
        name: name.to_owned(),
        accepted_types: NativeTypeUnion::new([value_type])?,
        required: true,
        hidden: false,
        lazy: false,
        cardinality: if kind == TextKind::MergeTextLists && name == "texts" {
            NativePortCardinality::List
        } else {
            NativePortCardinality::Scalar
        },
        allows_literal: !matches!(
            (kind, name),
            (TextKind::BuildIdeogramPrompt, "element" | "color_palette")
                | (TextKind::ArrayToString, "array")
                | (TextKind::DictionaryToString, "dictionary")
        ),
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
        let mut token = format!(
            "{}-{}",
            self.kind.class_type(),
            self.kind.implementation_version()
        );
        if self.kind == TextKind::RegexExtract {
            token.push_str("-optional-captures-v2");
        }
        Ok(token)
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
    if inputs.len() != kind.input_names().len()
        || !kind
            .input_names()
            .iter()
            .all(|name| inputs.contains_key(*name))
    {
        return Err(invalid_inputs(format!(
            "{} requires exactly {}",
            kind.class_type(),
            kind.input_names().join(", ")
        )));
    }
    for name in kind.input_names() {
        let value = inputs
            .get(*name)
            .ok_or_else(|| invalid_inputs(format!("missing {name}")))?;
        let expected =
            input_descriptor(kind, name).map_err(|error| invalid_inputs(error.to_string()))?;
        value
            .validate()
            .map_err(|error| invalid_inputs(error.to_string()))?;
        let type_matches = match (expected.cardinality, value) {
            (NativePortCardinality::List, NativeValue::List { values }) => values
                .iter()
                .all(|value| expected.accepted_types.accepts(value)),
            (NativePortCardinality::Scalar, NativeValue::List { .. }) => false,
            (NativePortCardinality::Scalar, value) => expected.accepted_types.accepts(value),
            (NativePortCardinality::List, _) => false,
            (NativePortCardinality::Mapped, _) => false,
        };
        if !type_matches {
            return Err(invalid_inputs(format!(
                "{} input {name} has the wrong type or cardinality",
                kind.class_type()
            )));
        }
    }
    match kind {
        TextKind::BuildIdeogramPrompt => validate_prompt_inputs(inputs)?,
        TextKind::ArrayToString | TextKind::DictionaryToString => {
            let indent = required_integer(inputs, "indent")?;
            if !(0..=MAX_JSON_INDENT).contains(&indent) {
                return Err(invalid_inputs("indent must be between 0 and 8"));
            }
        }
        TextKind::RegexExtract => {
            let group_index = required_integer(inputs, "group_index")?;
            if !(0..=100).contains(&group_index) {
                return Err(invalid_inputs("group_index must be between 0 and 100"));
            }
        }
        _ => {}
    }
    Ok(())
}

fn execute_text(
    kind: TextKind,
    inputs: &BTreeMap<String, NativeValue>,
    cancellation: &CancellationToken,
) -> Result<NativeValue, NativeNodeFailure> {
    match kind {
        TextKind::AddPrefix => Ok(string_value(format!(
            "{}{}",
            required_string(inputs, "prefix")?,
            required_string(inputs, "texts")?
        ))),
        TextKind::AddSuffix => Ok(string_value(format!(
            "{}{}",
            required_string(inputs, "texts")?,
            required_string(inputs, "suffix")?
        ))),
        TextKind::BuildIdeogramPrompt => build_ideogram_prompt(inputs),
        TextKind::CaseConverter => Ok(string_value(convert_case(
            required_string(inputs, "string")?,
            required_string(inputs, "mode")?,
        ))),
        TextKind::ArrayToString => {
            let value = preserved_value(inputs, "array", "ARRAY")?;
            Ok(string_value(dump_json(
                value,
                required_integer(inputs, "indent")? as usize,
                cancellation,
                kind.class_type(),
            )?))
        }
        TextKind::DictionaryToString => {
            let value = preserved_value(inputs, "dictionary", "DICT")?;
            Ok(string_value(dump_json(
                value,
                required_integer(inputs, "indent")? as usize,
                cancellation,
                kind.class_type(),
            )?))
        }
        TextKind::JsonExtractString => Ok(string_value(json_extract(
            required_string(inputs, "json_string")?,
            required_string(inputs, "key")?,
            cancellation,
        )?)),
        TextKind::MergeTextLists => inputs
            .get("texts")
            .cloned()
            .ok_or_else(|| invalid_inputs("missing texts")),
        TextKind::RegexExtract => Ok(match regex_extract(inputs, cancellation)? {
            Some(value) => string_value(value),
            None => NativeValue::Primitive {
                value: NativePrimitive::Null,
            },
        }),
        TextKind::RegexMatch => Ok(boolean_value(regex_match(inputs, cancellation)?)),
    }
}

fn build_ideogram_prompt(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeValue, NativeNodeFailure> {
    let elements = preserved_value(inputs, "element", "ARRAY")?
        .as_array()
        .cloned()
        .unwrap_or_default();
    let style = preserved_value(inputs, "style", "COMFY_DYNAMICCOMBO_V3")?;
    let kind = style
        .as_object()
        .and_then(|style| style.get("style"))
        .and_then(Value::as_str)
        .unwrap_or("none");
    let photo = style
        .as_object()
        .and_then(|style| style.get("photo"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let art_style = style
        .as_object()
        .and_then(|style| style.get("art_style"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let palette = normalize_palette(preserved_value(inputs, "color_palette", "COLORS")?);
    let high_level_description = required_string(inputs, "high_level_description")?;
    let background = required_string(inputs, "background")?;
    let aesthetics = required_string(inputs, "aesthetics")?;
    let lighting = required_string(inputs, "lighting")?;
    let medium = required_string(inputs, "medium")?;

    let mut caption = Map::new();
    if !high_level_description.trim().is_empty() {
        caption.insert(
            "high_level_description".to_owned(),
            Value::String(high_level_description.to_owned()),
        );
    }
    if kind != "none" {
        let mut style_description = Map::new();
        style_description.insert(
            "aesthetics".to_owned(),
            Value::String(aesthetics.to_owned()),
        );
        style_description.insert("lighting".to_owned(), Value::String(lighting.to_owned()));
        if kind == "photo" {
            style_description.insert("photo".to_owned(), Value::String(photo.to_owned()));
            style_description.insert("medium".to_owned(), Value::String(medium.to_owned()));
        } else {
            style_description.insert("medium".to_owned(), Value::String(medium.to_owned()));
            style_description.insert("art_style".to_owned(), Value::String(art_style.to_owned()));
        }
        if !palette.is_empty() {
            style_description.insert(
                "color_palette".to_owned(),
                Value::Array(palette.into_iter().map(Value::String).collect()),
            );
        }
        caption.insert(
            "style_description".to_owned(),
            Value::Object(style_description),
        );
    }
    caption.insert(
        "compositional_deconstruction".to_owned(),
        Value::Object(Map::from_iter([
            (
                "background".to_owned(),
                Value::String(background.to_owned()),
            ),
            ("elements".to_owned(), Value::Array(elements)),
        ])),
    );
    Ok(NativeValue::PreservedUnknown {
        type_name: "DICT".to_owned(),
        value: Value::Object(caption),
    })
}

fn validate_prompt_inputs(inputs: &BTreeMap<String, NativeValue>) -> Result<(), NativeNodeFailure> {
    let element = preserved_value(inputs, "element", "ARRAY")?;
    if !element.is_array() {
        return Err(invalid_inputs("element must contain an ARRAY value"));
    }
    let style = preserved_value(inputs, "style", "COMFY_DYNAMICCOMBO_V3")?;
    if !style.is_object() {
        return Err(invalid_inputs("style must contain a dynamic-combo object"));
    }
    let palette = preserved_value(inputs, "color_palette", "COLORS")?;
    if !palette.is_array() && !palette.is_object() {
        return Err(invalid_inputs(
            "color_palette must contain an array or object",
        ));
    }
    Ok(())
}

fn normalize_palette(value: &Value) -> Vec<String> {
    let values: Vec<&Value> = match value {
        Value::Array(values) => values.iter().collect(),
        Value::Object(values) => values.values().collect(),
        _ => Vec::new(),
    };
    values
        .into_iter()
        .filter_map(Value::as_str)
        .filter(|color| !color.is_empty())
        .map(|color| color.to_uppercase())
        .collect()
}

fn convert_case(value: &str, mode: &str) -> String {
    match mode {
        "UPPERCASE" => value.to_uppercase(),
        "lowercase" => value.to_lowercase(),
        "Capitalize" => python_capitalize(value),
        "Title Case" => python_title(value),
        _ => value.to_owned(),
    }
}

fn python_capitalize(value: &str) -> String {
    let mut characters = value.chars();
    let Some(first) = characters.next() else {
        return String::new();
    };
    first
        .to_uppercase()
        .chain(characters.flat_map(char::to_lowercase))
        .collect()
}

fn python_title(value: &str) -> String {
    let mut previous_cased = false;
    let mut output = String::new();
    for character in value.chars() {
        if previous_cased {
            output.extend(character.to_lowercase());
        } else {
            output.extend(character.to_uppercase());
        }
        previous_cased = character.is_lowercase() || character.is_uppercase();
    }
    output
}

fn json_extract(
    json_string: &str,
    key: &str,
    cancellation: &CancellationToken,
) -> Result<String, NativeNodeFailure> {
    check_cancellation(cancellation, "JsonExtractString")?;
    let Ok(Value::Object(mut object)) = serde_json::from_str::<Value>(json_string) else {
        return Ok(String::new());
    };
    let value = match object.remove(key) {
        None | Some(Value::Null) => return Ok(String::new()),
        Some(Value::String(value)) => string_value(value),
        Some(value) => NativeValue::PreservedUnknown {
            type_name: "JSON".to_owned(),
            value,
        },
    };
    NativeTextFormatter::format(
        "{value}",
        &BTreeMap::from([("value".to_owned(), value)]),
        cancellation,
    )
    .map_err(|error| match error {
        NativeTextFormatError::Cancelled => interrupted_failure("JsonExtractString"),
        error => NativeNodeFailure {
            code: "native_text_format_failed".to_owned(),
            message: format!("JsonExtractString failed: {error}"),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        },
    })
}

fn dump_json(
    value: &Value,
    indent: usize,
    cancellation: &CancellationToken,
    class_type: &str,
) -> Result<String, NativeNodeFailure> {
    let mut output = String::new();
    write_json_value(value, indent, 0, &mut output, cancellation, class_type)?;
    Ok(output)
}

fn write_json_value(
    value: &Value,
    indent: usize,
    depth: usize,
    output: &mut String,
    cancellation: &CancellationToken,
    class_type: &str,
) -> Result<(), NativeNodeFailure> {
    check_cancellation(cancellation, class_type)?;
    match value {
        Value::Array(values) => {
            output.push('[');
            write_json_sequence(
                values.iter(),
                indent,
                depth,
                output,
                cancellation,
                class_type,
            )?;
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            let entries = values.iter().collect::<Vec<_>>();
            for (index, (key, value)) in entries.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                    if indent == 0 {
                        output.push(' ');
                    }
                }
                if indent > 0 {
                    output.push('\n');
                    output.push_str(&" ".repeat((depth + 1) * indent));
                }
                output.push_str(
                    &serde_json::to_string(key)
                        .map_err(|error| invalid_inputs(error.to_string()))?,
                );
                output.push_str(": ");
                write_json_value(value, indent, depth + 1, output, cancellation, class_type)?;
            }
            if indent > 0 && !entries.is_empty() {
                output.push('\n');
                output.push_str(&" ".repeat(depth * indent));
            }
            output.push('}');
        }
        _ => output.push_str(
            &serde_json::to_string(value).map_err(|error| invalid_inputs(error.to_string()))?,
        ),
    }
    Ok(())
}

fn write_json_sequence<'a>(
    values: impl Iterator<Item = &'a Value>,
    indent: usize,
    depth: usize,
    output: &mut String,
    cancellation: &CancellationToken,
    class_type: &str,
) -> Result<(), NativeNodeFailure> {
    let values = values.collect::<Vec<_>>();
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            output.push(',');
            if indent == 0 {
                output.push(' ');
            }
        }
        if indent > 0 {
            output.push('\n');
            output.push_str(&" ".repeat((depth + 1) * indent));
        }
        write_json_value(value, indent, depth + 1, output, cancellation, class_type)?;
    }
    if indent > 0 && !values.is_empty() {
        output.push('\n');
        output.push_str(&" ".repeat(depth * indent));
    }
    Ok(())
}

fn regex_match(
    inputs: &BTreeMap<String, NativeValue>,
    cancellation: &CancellationToken,
) -> Result<bool, NativeNodeFailure> {
    let pattern = required_string(inputs, "regex_pattern")?;
    let flags = regex_flags(inputs)?;
    let regex = match NativeTextRegex::checked(pattern, flags) {
        Ok(regex) => regex,
        Err(NativeTextRegexError::InvalidPattern(_)) => return Ok(false),
        Err(error) => return Err(regex_failure("RegexMatch", error)),
    };
    regex
        .is_match(required_string(inputs, "string")?, cancellation)
        .map_err(|error| regex_failure("RegexMatch", error))
}

fn regex_extract(
    inputs: &BTreeMap<String, NativeValue>,
    cancellation: &CancellationToken,
) -> Result<Option<String>, NativeNodeFailure> {
    let regex = match NativeTextRegex::checked(
        required_string(inputs, "regex_pattern")?,
        regex_flags(inputs)?,
    ) {
        Ok(regex) => regex,
        Err(NativeTextRegexError::InvalidPattern(_)) => return Ok(Some(String::new())),
        Err(error) => return Err(regex_failure("RegexExtract", error)),
    };
    let rows = regex
        .capture_rows(required_string(inputs, "string")?, cancellation)
        .map_err(|error| regex_failure("RegexExtract", error))?;
    let mode = required_string(inputs, "mode")?;
    let group_index = required_integer(inputs, "group_index")? as usize;
    match mode {
        "First Match" => Ok(Some(
            rows.rows()
                .first()
                .and_then(|row| row.first())
                .and_then(Option::as_deref)
                .unwrap_or_default()
                .to_owned(),
        )),
        "All Matches" => {
            let capture_index = usize::from(rows.capture_count() > 1);
            join_capture_rows(rows.rows(), capture_index, false).map(Some)
        }
        "First Group" => {
            if group_index >= rows.capture_count() {
                return Ok(Some(String::new()));
            }
            match rows.rows().first() {
                Some(row) => row.get(group_index).cloned().ok_or_else(|| {
                    invalid_inputs("regex match did not produce the expected capture")
                }),
                None => Ok(Some(String::new())),
            }
        }
        "All Groups" => {
            if rows.capture_count() <= 1 || group_index >= rows.capture_count() {
                return Ok(Some(String::new()));
            }
            join_capture_rows(rows.rows(), group_index, true).map(Some)
        }
        _ => Ok(Some(String::new())),
    }
}

fn join_capture_rows(
    rows: &[Vec<Option<String>>],
    capture_index: usize,
    require_group: bool,
) -> Result<String, NativeNodeFailure> {
    let mut values = Vec::new();
    for row in rows {
        let capture = row.get(capture_index).ok_or_else(|| {
            invalid_inputs(if require_group {
                "regex group did not participate in a match"
            } else {
                "regex match did not produce the expected capture"
            })
        })?;
        let value = if require_group {
            capture
                .as_deref()
                .ok_or_else(|| invalid_inputs("regex group did not participate in a match"))?
        } else {
            // Python findall fills unmatched captures with empty strings; match.group does not.
            capture.as_deref().unwrap_or_default()
        };
        values.push(value);
    }
    Ok(values.join("\n"))
}

fn regex_flags(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeTextRegexFlags, NativeNodeFailure> {
    Ok(NativeTextRegexFlags {
        case_insensitive: required_boolean(inputs, "case_insensitive")?,
        multi_line: required_boolean(inputs, "multiline")?,
        dot_matches_new_line: required_boolean(inputs, "dotall")?,
    })
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

fn preserved_value<'a>(
    inputs: &'a BTreeMap<String, NativeValue>,
    name: &str,
    type_name: &str,
) -> Result<&'a Value, NativeNodeFailure> {
    match inputs.get(name) {
        Some(NativeValue::PreservedUnknown {
            type_name: actual,
            value,
        }) if actual == type_name => Ok(value),
        _ => Err(invalid_inputs(format!("{name} must be {type_name}"))),
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
    use serde_json::json;
    use std::error::Error;
    use uuid::Uuid;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/nodes/text-comfy-node-0002/fixture.json"
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
        let attempt_id = AttemptId(Uuid::from_u128(0x46901));
        let (_backend, workspace) = CpuWorkspaceAuthority::create_backend(1)?;
        Ok(NativeNodeContext::new(
            PromptId(Uuid::from_u128(0x46902)),
            attempt_id,
            NodeId("text-family-test".to_owned()),
            cancellation,
            workspace.authorize_workspace(0)?,
            Arc::new(RejectingStore {
                identity: NativeHandleStoreIdentity::new(
                    Uuid::from_u128(0x46903),
                    Uuid::from_u128(0x46904),
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
        NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }
    }

    fn boolean(value: bool) -> NativeValue {
        boolean_value(value)
    }

    fn preserved(type_name: &str, value: Value) -> NativeValue {
        NativeValue::PreservedUnknown {
            type_name: type_name.to_owned(),
            value,
        }
    }

    fn output_string(value: NativeValue) -> Result<String, Box<dyn Error>> {
        match value {
            NativeValue::Primitive {
                value: NativePrimitive::String(value),
            } => Ok(value),
            _ => Err("output was not a string".into()),
        }
    }

    fn regex_inputs(
        text: &str,
        pattern: &str,
        mode: Option<&str>,
    ) -> BTreeMap<String, NativeValue> {
        let mut inputs = BTreeMap::from([
            ("string".to_owned(), string(text)),
            ("regex_pattern".to_owned(), string(pattern)),
            ("case_insensitive".to_owned(), boolean(true)),
            ("multiline".to_owned(), boolean(false)),
            ("dotall".to_owned(), boolean(false)),
        ]);
        if let Some(mode) = mode {
            inputs.insert("mode".to_owned(), string(mode));
            inputs.insert("group_index".to_owned(), integer(1));
        }
        inputs
    }

    #[test]
    fn source_fixture_and_all_exact_schemas_are_registered() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        assert_eq!(
            fixture.get("stable_task_id").and_then(Value::as_str),
            Some("comfy-parity-native-nodes-text-comfy-node-0002")
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
        Ok(())
    }

    #[test]
    fn list_transforms_and_python_case_modes_are_source_exact() -> Result<(), Box<dyn Error>> {
        assert_eq!(
            output_string(execute(
                "AddTextPrefix",
                BTreeMap::from([
                    ("texts".to_owned(), string("line\n二")),
                    ("prefix".to_owned(), string("→ ")),
                ]),
            )?)?,
            "→ line\n二"
        );
        assert_eq!(
            output_string(execute(
                "AddTextSuffix",
                BTreeMap::from([
                    ("texts".to_owned(), string("caption")),
                    ("suffix".to_owned(), string("!")),
                ]),
            )?)?,
            "caption!"
        );
        let merged = execute(
            "MergeTextLists",
            BTreeMap::from([(
                "texts".to_owned(),
                NativeValue::List {
                    values: vec![string("one"), string("two")],
                },
            )]),
        )?;
        assert_eq!(
            merged,
            NativeValue::List {
                values: vec![string("one"), string("two")]
            }
        );
        for (mode, expected) in [
            ("UPPERCASE", "THEY'RE CAFÉ"),
            ("lowercase", "they're café"),
            ("Capitalize", "They're café"),
            ("Title Case", "They'Re Café"),
        ] {
            assert_eq!(
                output_string(execute(
                    "CaseConverter",
                    BTreeMap::from([
                        ("string".to_owned(), string("THEY'RE café")),
                        ("mode".to_owned(), string(mode)),
                    ]),
                )?)?,
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn json_extract_original_source_repr_matrix_is_exact() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sim_test_support/fixtures/nodes/text-comfy-node-0002/json-extract-source-contract.json"
        )))?;
        assert_eq!(
            fixture["source"]["sha256"],
            "bb01963178f28efc6e3a9578aecb89f53f8265e89c0518da92ec21c4955e85f1"
        );
        let cases = fixture["ordinary_cases"]
            .as_array()
            .ok_or("missing source cases")?;
        assert_eq!(cases.len(), 63);
        assert_eq!(
            fixture["capability_cases"].as_array().map(Vec::len),
            Some(16)
        );
        for case in cases {
            let inputs = BTreeMap::from([
                (
                    "json_string".to_owned(),
                    string(case["json_string"].as_str().ok_or("missing JSON string")?),
                ),
                (
                    "key".to_owned(),
                    string(case["key"].as_str().ok_or("missing key")?),
                ),
            ]);
            assert_eq!(
                output_string(execute("JsonExtractString", inputs).map_err(|error| {
                    format!("original source case {} failed: {error}", case["id"])
                })?)?,
                case["output"].as_str().ok_or("missing source output")?,
                "source case {}",
                case["id"],
            );
        }
        let node = executable("JsonExtractString")?;
        let mut raw_admission_gates = 0;
        for case in fixture["capability_cases"]
            .as_array()
            .ok_or("missing capability cases")?
        {
            if !case["id"]
                .as_str()
                .is_some_and(|identifier| identifier.starts_with("c1-control-"))
            {
                continue;
            }
            raw_admission_gates += 1;
            let source_json = case["json_string"]
                .as_str()
                .ok_or("missing raw source JSON")?;
            let key = case["key"].as_str().ok_or("missing source key")?;
            assert_eq!(
                json_extract(source_json, key, &CancellationToken::default())?,
                case["output"].as_str().ok_or("missing raw source output")?
            );
            let error = futures::executor::block_on(node.execute(
                context(CancellationToken::default())?,
                BTreeMap::from([
                    ("json_string".to_owned(), string(source_json)),
                    ("key".to_owned(), string(key)),
                ]),
            ))
            .expect_err("literal DEL/C1 input is a canonical admission gate");
            assert_eq!(error.code, "invalid_node_inputs");
            assert_eq!(error.kind, NativeNodeFailureKind::Failure);
            assert!(error.message.contains("primitive string"));
        }
        assert_eq!(raw_admission_gates, 4);
        Ok(())
    }

    #[test]
    fn json_extract_source_exponent_capability_cases_now_match() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sim_test_support/fixtures/nodes/text-comfy-node-0002/json-extract-source-contract.json"
        )))?;
        let cases = fixture["capability_cases"]
            .as_array()
            .ok_or("missing source capability cases")?;
        let mut matched = 0;
        for case in cases {
            if !matches!(
                case["id"].as_str(),
                Some("large-exponent" | "small-exponent")
            ) {
                continue;
            }
            matched += 1;
            let inputs = BTreeMap::from([
                (
                    "json_string".to_owned(),
                    string(case["json_string"].as_str().ok_or("missing source JSON")?),
                ),
                (
                    "key".to_owned(),
                    string(case["key"].as_str().ok_or("missing source key")?),
                ),
            ]);
            assert_eq!(
                output_string(execute("JsonExtractString", inputs)?)?,
                case["output"].as_str().ok_or("missing original output")?,
                "original exponent case {}",
                case["id"]
            );
        }
        assert_eq!(matched, 2);
        Ok(())
    }

    #[test]
    fn json_extract_cancelled_bounded_and_recovery_paths_are_atomic() -> Result<(), Box<dyn Error>>
    {
        let inputs = BTreeMap::from([
            (
                "json_string".to_owned(),
                string(r#"{"value":["line\nbreak","don't"]}"#),
            ),
            ("key".to_owned(), string("value")),
        ]);
        let node = executable("JsonExtractString")?;
        assert_eq!(
            node.cache_change_token(&inputs)?,
            "JsonExtractString-source-bb019631-text-v4"
        );
        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        let error =
            futures::executor::block_on(node.execute(context(cancellation)?, inputs.clone()))
                .expect_err("cancelled extraction must not publish");
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);

        let oversized = serde_json::to_string(
            &json!({"value": "x".repeat(crate::NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES + 1)}),
        )?;
        let error = futures::executor::block_on(node.execute(
            context(CancellationToken::default())?,
            BTreeMap::from([
                ("json_string".to_owned(), string(&oversized)),
                ("key".to_owned(), string("value")),
            ]),
        ))
        .expect_err("node input admission must not publish oversized text");
        assert_eq!(error.code, "invalid_node_inputs");
        let error = json_extract(&oversized, "value", &CancellationToken::default())
            .expect_err("canonical formatter result bounds must fail");
        assert_eq!(error.code, "native_text_format_failed");
        assert!(error.message.contains("result exceeded"));

        let expanding = serde_json::to_string(&json!({"value": vec!["x"; 220_000]}))?;
        let error = futures::executor::block_on(node.execute(
            context(CancellationToken::default())?,
            BTreeMap::from([
                ("json_string".to_owned(), string(&expanding)),
                ("key".to_owned(), string("value")),
            ]),
        ))
        .expect_err("expanded text must respect canonical output admission");
        assert_eq!(error.code, "invalid_node_inputs");
        assert!(error.message.contains("primitive string"));
        assert_eq!(
            output_string(execute("JsonExtractString", inputs)?)?,
            "['line\\nbreak', \"don't\"]"
        );
        Ok(())
    }

    #[test]
    fn json_nodes_preserve_python_text_and_prompt_shapes() -> Result<(), Box<dyn Error>> {
        assert_eq!(
            output_string(execute(
                "JsonExtractString",
                BTreeMap::from([
                    (
                        "json_string".to_owned(),
                        string(r#"{"name":"café","flag":true,"none":null}"#),
                    ),
                    ("key".to_owned(), string("flag")),
                ]),
            )?)?,
            "True"
        );
        assert_eq!(
            output_string(execute(
                "ConvertArrayToString",
                BTreeMap::from([
                    ("array".to_owned(), preserved("ARRAY", json!(["café", 2]))),
                    ("indent".to_owned(), integer(0)),
                ]),
            )?)?,
            r#"["café", 2]"#
        );
        assert_eq!(
            output_string(execute(
                "ConvertDictionaryToString",
                BTreeMap::from([
                    (
                        "dictionary".to_owned(),
                        preserved("DICT", json!({"caption": "café", "count": 2})),
                    ),
                    ("indent".to_owned(), integer(2)),
                ]),
            )?)?,
            "{\n  \"caption\": \"café\",\n  \"count\": 2\n}"
        );
        let prompt = execute(
            "BuildJsonPromptIdeogram",
            BTreeMap::from([
                (
                    "element".to_owned(),
                    preserved("ARRAY", json!([{"description": "subject"}])),
                ),
                ("high_level_description".to_owned(), string("A portrait")),
                ("background".to_owned(), string("studio")),
                (
                    "style".to_owned(),
                    preserved(
                        "COMFY_DYNAMICCOMBO_V3",
                        json!({"style": "photo", "photo": "35mm"}),
                    ),
                ),
                ("aesthetics".to_owned(), string("cinematic")),
                ("lighting".to_owned(), string("rim light")),
                ("medium".to_owned(), string("photograph")),
                (
                    "color_palette".to_owned(),
                    preserved("COLORS", json!(["#aa00ff", "", 7])),
                ),
            ]),
        )?;
        let NativeValue::PreservedUnknown { type_name, value } = prompt else {
            return Err("prompt was not a DICT".into());
        };
        assert_eq!(type_name, "DICT");
        assert_eq!(
            value.pointer("/style_description/color_palette"),
            Some(&json!(["#AA00FF"]))
        );
        assert_eq!(
            value.pointer("/compositional_deconstruction/background"),
            Some(&json!("studio"))
        );
        Ok(())
    }

    #[test]
    fn regex_search_findall_finditer_flags_and_fallbacks_are_exact() -> Result<(), Box<dyn Error>> {
        assert_eq!(
            execute(
                "RegexMatch",
                regex_inputs("go go!", r"(?P<word>\w+)\s+(?P=word)(?=!)", None),
            )?,
            boolean(true)
        );
        assert_eq!(
            output_string(execute(
                "RegexExtract",
                regex_inputs("A-1 b-2", r"([a-z])-(\d)", Some("All Matches")),
            )?)?,
            "A\nb"
        );
        assert_eq!(
            output_string(execute(
                "RegexExtract",
                regex_inputs("A-1 b-2", r"([a-z])-(\d)", Some("All Groups")),
            )?)?,
            "A\nb"
        );
        let mut all_group_zero = regex_inputs("A-1 b-2", r"([a-z])-(\d)", Some("All Groups"));
        all_group_zero.insert("group_index".to_owned(), integer(0));
        assert_eq!(
            output_string(execute("RegexExtract", all_group_zero)?)?,
            "A-1\nb-2"
        );
        let mut flagged = regex_inputs("start\nCAFÉ\nend", r"^café$", None);
        flagged.insert("multiline".to_owned(), boolean(true));
        assert_eq!(execute("RegexMatch", flagged)?, boolean(true));
        let mut dotall = regex_inputs("A\nB", r"a.b", None);
        dotall.insert("dotall".to_owned(), boolean(true));
        assert_eq!(execute("RegexMatch", dotall)?, boolean(true));
        assert_eq!(
            execute("RegexMatch", regex_inputs("text", "(", None))?,
            boolean(false)
        );
        assert_eq!(
            output_string(execute(
                "RegexExtract",
                regex_inputs("text", "(", Some("First Match")),
            )?)?,
            ""
        );
        Ok(())
    }

    #[test]
    fn regex_findall_keeps_unmatched_optional_captures_and_tuple_positions()
    -> Result<(), Box<dyn Error>> {
        for (text, pattern, expected) in [
            ("b ab b", r"(a)?b", "\na\n"),
            ("b ab b", r"(a)?(b)", "\na\n"),
            ("b ab b", r"(?P<letter>a)?b", "\na\n"),
            ("b ab b", r"(a*)b", "\na\n"),
            ("b a b", r"(a)|(b)", "\na\n"),
            ("b b", r"(a)?b", "\n"),
            ("c", r"(a)?b", ""),
            ("b ab b", "b", "b\nb\nb"),
        ] {
            let mut inputs = regex_inputs(text, pattern, Some("All Matches"));
            inputs.insert("group_index".to_owned(), integer(100));
            assert_eq!(
                output_string(execute("RegexExtract", inputs)?)?,
                expected,
                "text={text:?}, pattern={pattern:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn regex_finditer_still_rejects_unmatched_groups_and_keeps_group_zero()
    -> Result<(), Box<dyn Error>> {
        let node = executable("RegexExtract")?;
        for pattern in [r"(a)?b", r"(a)?(b)"] {
            let failure = futures::executor::block_on(node.execute(
                context(CancellationToken::default())?,
                regex_inputs("b ab b", pattern, Some("All Groups")),
            ))
            .expect_err("unmatched match.group values must not be silently filled");
            assert_eq!(failure.code, "invalid_node_inputs");
            assert_eq!(
                failure.message,
                "regex group did not participate in a match"
            );
        }
        let mut whole_match = regex_inputs("b ab b", r"(a)?b", Some("All Groups"));
        whole_match.insert("group_index".to_owned(), integer(0));
        assert_eq!(
            output_string(execute("RegexExtract", whole_match)?)?,
            "b\nab\nb"
        );
        assert_eq!(
            output_string(execute(
                "RegexExtract",
                regex_inputs("b ab b", r"(a*)b", Some("All Groups")),
            )?)?,
            "\na\n"
        );
        Ok(())
    }

    #[test]
    fn regex_first_group_preserves_null_empty_missing_and_selected_capture_values()
    -> Result<(), Box<dyn Error>> {
        for (text, pattern, group_index, expected) in [
            ("b", r"(a)?b", 1, None),
            ("b ab", r"(a)?b", 1, None),
            ("b", r"(?P<letter>a)?b", 1, None),
            ("b", r"(a)?(b)", 1, None),
            ("ab", r"(a)?b", 1, Some("a")),
            ("ab b", r"(a)?b", 1, Some("a")),
            ("b", r"(a*)b", 1, Some("")),
            ("c", r"(a)?b", 1, Some("")),
            ("b", r"(a)?b", 2, Some("")),
            ("b", r"(a)?b", 0, Some("b")),
            ("b", r"(a)?(b)", 2, Some("b")),
            ("b", "b", 1, Some("")),
            ("b", "b", 0, Some("b")),
            ("b", "(", 1, Some("")),
        ] {
            let mut inputs = regex_inputs(text, pattern, Some("First Group"));
            inputs.insert("group_index".to_owned(), integer(group_index));
            let actual = execute("RegexExtract", inputs)?;
            let expected = match expected {
                Some(value) => string(value),
                None => NativeValue::Primitive {
                    value: NativePrimitive::Null,
                },
            };
            assert_eq!(
                actual, expected,
                "text={text:?}, pattern={pattern:?}, group={group_index}"
            );
            let encoded = serde_json::to_vec(&actual)?;
            assert_eq!(serde_json::from_slice::<NativeValue>(&encoded)?, actual);
        }
        Ok(())
    }

    #[test]
    fn regex_nullable_first_group_keeps_string_descriptor_and_precancellation()
    -> Result<(), Box<dyn Error>> {
        let bindings = native_node_bindings()?;
        let binding = bindings
            .iter()
            .find(|binding| binding.descriptor().class_type == "RegexExtract")
            .ok_or("RegexExtract binding is absent")?;
        assert_eq!(
            binding
                .descriptor()
                .outputs
                .first()
                .map(|output| &output.produced_type),
            Some(&NativeValueType::Primitive(NativePrimitiveType::String))
        );
        binding.validate()?;
        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        let failure = futures::executor::block_on(executable("RegexExtract")?.execute(
            context(cancellation)?,
            regex_inputs("b", r"(a)?b", Some("First Group")),
        ))
        .expect_err("pre-cancelled nullable extraction must not produce an output");
        assert_eq!(failure.kind, NativeNodeFailureKind::Interrupted);
        assert_eq!(failure.code, "execution_interrupted");
        Ok(())
    }

    #[test]
    fn regex_optional_capture_cache_revision_is_isolated_to_extract() -> Result<(), Box<dyn Error>>
    {
        for kind in [
            TextKind::AddPrefix,
            TextKind::AddSuffix,
            TextKind::BuildIdeogramPrompt,
            TextKind::CaseConverter,
            TextKind::ArrayToString,
            TextKind::DictionaryToString,
            TextKind::JsonExtractString,
            TextKind::MergeTextLists,
            TextKind::RegexExtract,
            TextKind::RegexMatch,
        ] {
            let inputs = kind
                .input_names()
                .iter()
                .map(|name| {
                    let value = match *name {
                        "texts" if kind == TextKind::MergeTextLists => NativeValue::List {
                            values: vec![string("text")],
                        },
                        "element" | "array" => preserved("ARRAY", json!([])),
                        "dictionary" => preserved("DICT", json!({})),
                        "style" => preserved("COMFY_DYNAMICCOMBO_V3", json!({})),
                        "color_palette" => preserved("COLORS", json!([])),
                        "indent" => integer(0),
                        "group_index" => integer(1),
                        "case_insensitive" | "multiline" | "dotall" => boolean(false),
                        "regex_pattern" => string(r"(a)?b"),
                        "mode" if kind == TextKind::RegexExtract => string("First Group"),
                        "mode" => string("lowercase"),
                        "json_string" => string("{}"),
                        _ => string("b"),
                    };
                    ((*name).to_owned(), value)
                })
                .collect::<BTreeMap<_, _>>();
            let node = executable(kind.class_type())?;
            let legacy = format!("{}-{}", kind.class_type(), kind.implementation_version());
            let actual = node.cache_change_token(&inputs)?;
            assert_eq!(node.implementation_version(), kind.implementation_version());
            if kind == TextKind::RegexExtract {
                assert_ne!(actual, legacy);
                assert_eq!(actual, format!("{legacy}-optional-captures-v2"));
            } else {
                assert_eq!(actual, legacy);
            }
        }
        Ok(())
    }

    #[test]
    fn validation_limits_cancellation_cache_and_recovery_fail_closed() -> Result<(), Box<dyn Error>>
    {
        let node = executable("RegexMatch")?;
        let inputs = regex_inputs(&format!("{}z", "x".repeat(512)), r"^(x+x+)+(?>y)$", None);
        let failure = futures::executor::block_on(
            node.execute(context(CancellationToken::default())?, inputs.clone()),
        )
        .expect_err("bounded backtracking must fail");
        assert_eq!(failure.code, "native_text_regex_failed");

        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let failure = futures::executor::block_on(
            node.execute(context(cancellation)?, regex_inputs("text", "text", None)),
        )
        .expect_err("cancelled regex must fail");
        assert_eq!(failure.kind, NativeNodeFailureKind::Interrupted);

        assert_eq!(
            node.cache_change_token(&inputs)?,
            "RegexMatch-source-bb019631-v1"
        );
        let invalid = regex_inputs("text", "text", None)
            .into_iter()
            .filter(|(name, _)| name != "dotall")
            .collect();
        assert!(node.cache_change_token(&invalid).is_err());
        assert_eq!(
            execute("RegexMatch", regex_inputs("fresh", "fresh", None))?,
            boolean(true)
        );
        Ok(())
    }
}
