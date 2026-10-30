use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

pub const NODE_DESCRIPTOR_SCHEMA_VERSION: u16 = 1;
pub const NATIVE_SCHEMA_METADATA_VERSION: u16 = 2;

const MAX_SCHEMA_DEPTH: usize = 16;
const MAX_SCHEMA_ITEMS: usize = 4_096;
const MAX_AUTOGROW_NAMES: usize = 100;
const MAX_SCHEMA_TEXT_BYTES: usize = 256 * 1024;
const MAX_SCHEMA_TOTAL_BYTES: usize = 2 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NativeSchemaError {
    DepthExceeded,
    ItemCountExceeded,
    TextTooLarge,
    TotalBytesExceeded,
    InvalidDecimal,
    InvalidDigest,
    InvalidFieldName,
    DuplicateField(String),
    InvalidSourceType,
    InvalidMetadata(&'static str),
}

impl fmt::Display for NativeSchemaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DepthExceeded => formatter.write_str("schema value exceeds its depth bound"),
            Self::ItemCountExceeded => {
                formatter.write_str("schema value exceeds its item-count bound")
            }
            Self::TextTooLarge => formatter.write_str("schema text exceeds its byte bound"),
            Self::TotalBytesExceeded => {
                formatter.write_str("schema value exceeds its total byte bound")
            }
            Self::InvalidDecimal => {
                formatter.write_str("schema decimal is not finite or canonical")
            }
            Self::InvalidDigest => formatter.write_str("schema expression digest is invalid"),
            Self::InvalidFieldName => formatter.write_str("schema field name is invalid"),
            Self::DuplicateField(name) => write!(formatter, "duplicate schema field `{name}`"),
            Self::InvalidSourceType => formatter.write_str("schema source type is invalid"),
            Self::InvalidMetadata(field) => {
                write!(formatter, "schema metadata `{field}` is invalid")
            }
        }
    }
}

impl Error for NativeSchemaError {}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeSchemaValue {
    Null,
    Boolean { value: bool },
    SignedInteger { value: i64 },
    UnsignedInteger { value: u64 },
    FiniteDecimal { value: String },
    String { value: String },
    List { values: Vec<NativeSchemaValue> },
    Object { fields: Vec<NativeSchemaField> },
    PreservedExpression { source: String, sha256: String },
}

impl NativeSchemaValue {
    pub fn validate(&self) -> Result<(), NativeSchemaError> {
        let mut total_bytes = 0usize;
        let mut total_items = 0usize;
        self.validate_bounded(0, &mut total_bytes, &mut total_items)
    }

    fn validate_bounded(
        &self,
        depth: usize,
        total_bytes: &mut usize,
        total_items: &mut usize,
    ) -> Result<(), NativeSchemaError> {
        if depth > MAX_SCHEMA_DEPTH {
            return Err(NativeSchemaError::DepthExceeded);
        }
        *total_items = total_items
            .checked_add(1)
            .ok_or(NativeSchemaError::ItemCountExceeded)?;
        if *total_items > MAX_SCHEMA_ITEMS {
            return Err(NativeSchemaError::ItemCountExceeded);
        }
        match self {
            Self::Null
            | Self::Boolean { .. }
            | Self::SignedInteger { .. }
            | Self::UnsignedInteger { .. } => {}
            Self::FiniteDecimal { value } => {
                validate_text(value, total_bytes)?;
                if !valid_finite_decimal(value) {
                    return Err(NativeSchemaError::InvalidDecimal);
                }
            }
            Self::String { value } => validate_text(value, total_bytes)?,
            Self::List { values } => {
                for value in values {
                    value.validate_bounded(depth + 1, total_bytes, total_items)?;
                }
            }
            Self::Object { fields } => {
                let mut names = BTreeSet::new();
                for field in fields {
                    validate_name(&field.name)?;
                    validate_text(&field.name, total_bytes)?;
                    if !names.insert(field.name.as_str()) {
                        return Err(NativeSchemaError::DuplicateField(field.name.clone()));
                    }
                    field
                        .value
                        .validate_bounded(depth + 1, total_bytes, total_items)?;
                }
            }
            Self::PreservedExpression { source, sha256 } => {
                validate_text(source, total_bytes)?;
                if !valid_sha256(sha256) {
                    return Err(NativeSchemaError::InvalidDigest);
                }
                validate_text(sha256, total_bytes)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeSchemaField {
    pub name: String,
    pub value: NativeSchemaValue,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeSchemaProvenance {
    SourceV1,
    SourceV3,
    CompatibilityV1,
    Plugin,
    Synthetic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeUploadKind {
    Image,
    Audio,
    Video,
    Model,
    Artifact,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeInputRequirement {
    Required,
    Optional,
    Hidden,
    Preserved,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeInputSchemaMetadata {
    pub name: String,
    pub source_type_names: Vec<String>,
    pub default: Option<NativeSchemaValue>,
    pub minimum: Option<NativeSchemaValue>,
    pub maximum: Option<NativeSchemaValue>,
    pub step: Option<NativeSchemaValue>,
    pub choices: Vec<NativeSchemaValue>,
    pub display_name: Option<String>,
    pub tooltip: Option<String>,
    pub multiline: bool,
    pub socketless: bool,
    pub widget_type: Option<String>,
    pub force_input: bool,
    pub raw_link: bool,
    pub advanced: bool,
    pub upload: Option<NativeUploadKind>,
    pub extra: Vec<NativeSchemaField>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeStructuredInputField {
    pub path: Vec<String>,
    pub schema: NativeInputSchemaMetadata,
    pub required: bool,
    pub lazy: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeStructuredInputOption {
    pub selector: String,
    pub fields: Vec<NativeStructuredInputField>,
}

impl NativeInputSchemaMetadata {
    pub fn compatibility(name: impl Into<String>, source_type_name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            source_type_names: vec![source_type_name.into()],
            default: None,
            minimum: None,
            maximum: None,
            step: None,
            choices: Vec::new(),
            display_name: None,
            tooltip: None,
            multiline: false,
            socketless: false,
            widget_type: None,
            force_input: false,
            raw_link: false,
            advanced: false,
            upload: None,
            extra: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<(), NativeSchemaError> {
        validate_name(&self.name)?;
        validate_source_types(&self.source_type_names)?;
        validate_optional_values([
            self.default.as_ref(),
            self.minimum.as_ref(),
            self.maximum.as_ref(),
            self.step.as_ref(),
        ])?;
        validate_values(&self.choices)?;
        validate_optional_text(self.display_name.as_deref())?;
        validate_optional_text(self.tooltip.as_deref())?;
        validate_optional_name(self.widget_type.as_deref())?;
        validate_fields(&self.extra)?;
        if self.socketless && self.force_input {
            return Err(NativeSchemaError::InvalidMetadata("socketless"));
        }
        Ok(())
    }

    pub fn structured_options(
        &self,
    ) -> Result<Vec<NativeStructuredInputOption>, NativeSchemaError> {
        if !self
            .source_type_names
            .iter()
            .any(|source_type| source_type == "COMFY_DYNAMICCOMBO_V3")
        {
            return Ok(Vec::new());
        }
        let mut options = Vec::with_capacity(self.choices.len());
        let mut selectors = BTreeSet::new();
        for choice in &self.choices {
            let NativeSchemaValue::PreservedExpression { source, .. } = choice else {
                return Err(NativeSchemaError::InvalidMetadata(
                    "structured_input_option",
                ));
            };
            let expression: JsonValue = serde_json::from_str(source)
                .map_err(|_| NativeSchemaError::InvalidMetadata("structured_input_option"))?;
            let option = structured_option_from_expression(&expression)?;
            if !selectors.insert(option.selector.clone()) {
                return Err(NativeSchemaError::DuplicateField(option.selector));
            }
            options.push(option);
        }
        Ok(options)
    }
}

fn structured_option_from_expression(
    expression: &JsonValue,
) -> Result<NativeStructuredInputOption, NativeSchemaError> {
    let object = expression
        .as_object()
        .ok_or(NativeSchemaError::InvalidMetadata(
            "structured_input_option",
        ))?;
    if object.get("kind").and_then(JsonValue::as_str) != Some("call")
        || !object
            .get("name")
            .and_then(JsonValue::as_str)
            .is_some_and(|name| name.ends_with("DynamicCombo.Option"))
    {
        return Err(NativeSchemaError::InvalidMetadata(
            "structured_input_option",
        ));
    }
    let arguments = object
        .get("arguments")
        .and_then(JsonValue::as_array)
        .ok_or(NativeSchemaError::InvalidMetadata(
            "structured_input_option",
        ))?;
    if arguments.len() > 2 {
        return Err(NativeSchemaError::InvalidMetadata(
            "structured_input_option",
        ));
    }
    let mut selector = arguments.first();
    let mut inputs = arguments.get(1);
    if let Some(keywords) = object.get("keywords") {
        let keywords = keywords
            .as_array()
            .ok_or(NativeSchemaError::InvalidMetadata(
                "structured_input_option",
            ))?;
        for keyword in keywords {
            let value = keyword
                .get("value")
                .ok_or(NativeSchemaError::InvalidMetadata(
                    "structured_input_option",
                ))?;
            match keyword.get("name").and_then(JsonValue::as_str) {
                Some("key") if selector.is_none() => selector = Some(value),
                Some("inputs") if inputs.is_none() => inputs = Some(value),
                _ => {
                    return Err(NativeSchemaError::InvalidMetadata(
                        "structured_input_option",
                    ));
                }
            }
        }
    }
    let selector = structured_option_selector(selector.ok_or(
        NativeSchemaError::InvalidMetadata("structured_input_option"),
    )?)?;
    let items = inputs
        .and_then(|value| value.get("items"))
        .and_then(JsonValue::as_array)
        .ok_or(NativeSchemaError::InvalidMetadata(
            "structured_input_option",
        ))?;
    let mut fields = BTreeMap::<Vec<String>, NativeStructuredInputField>::new();
    for item in items {
        collect_structured_input_fields(item, &[], &mut fields)?;
    }
    Ok(NativeStructuredInputOption {
        selector,
        fields: fields.into_values().collect(),
    })
}

fn structured_option_selector(value: &JsonValue) -> Result<String, NativeSchemaError> {
    match value.get("kind").and_then(JsonValue::as_str) {
        Some("literal") => value
            .get("value")
            .and_then(JsonValue::as_str)
            .map(ToOwned::to_owned)
            .ok_or(NativeSchemaError::InvalidMetadata(
                "structured_input_selector",
            )),
        Some("attribute") | Some("name") => {
            let name = value.get("name").and_then(JsonValue::as_str).ok_or(
                NativeSchemaError::InvalidMetadata("structured_input_selector"),
            )?;
            let member = name
                .rsplit('.')
                .next()
                .ok_or(NativeSchemaError::InvalidMetadata(
                    "structured_input_selector",
                ))?;
            let selector = member.to_ascii_lowercase().replace('_', " ");
            if selector.is_empty() {
                Err(NativeSchemaError::InvalidMetadata(
                    "structured_input_selector",
                ))
            } else {
                Ok(selector)
            }
        }
        _ => Err(NativeSchemaError::InvalidMetadata(
            "structured_input_selector",
        )),
    }
}

fn collect_structured_input_fields(
    expression: &JsonValue,
    prefix: &[String],
    fields: &mut BTreeMap<Vec<String>, NativeStructuredInputField>,
) -> Result<(), NativeSchemaError> {
    let object = expression
        .as_object()
        .ok_or(NativeSchemaError::InvalidMetadata("structured_input_field"))?;
    if object.get("kind").and_then(JsonValue::as_str) != Some("call") {
        return Err(NativeSchemaError::InvalidMetadata("structured_input_field"));
    }
    let arguments = object
        .get("arguments")
        .and_then(JsonValue::as_array)
        .ok_or(NativeSchemaError::InvalidMetadata("structured_input_field"))?;
    let name = arguments
        .first()
        .and_then(|value| value.get("value"))
        .and_then(JsonValue::as_str)
        .ok_or(NativeSchemaError::InvalidMetadata("structured_input_field"))?;
    validate_name(name)?;
    let mut path = prefix.to_vec();
    path.push(name.to_owned());
    if object
        .get("name")
        .and_then(JsonValue::as_str)
        .is_some_and(|constructor| constructor.ends_with("Autogrow.Input"))
    {
        return collect_named_autogrow_fields(object, &path, fields);
    }
    let schema = structured_field_schema(object, name)?;
    let required = !call_keyword_boolean(object, "optional").unwrap_or(false);
    let lazy = call_keyword_boolean(object, "lazy").unwrap_or(false);
    let field = NativeStructuredInputField {
        path: path.clone(),
        schema,
        required,
        lazy,
    };
    if let Some(existing) = fields.get(&path) {
        if existing != &field {
            return Err(NativeSchemaError::InvalidMetadata("structured_input_field"));
        }
    } else {
        fields.insert(path.clone(), field);
    }
    Ok(())
}

fn collect_named_autogrow_fields(
    call: &serde_json::Map<String, JsonValue>,
    prefix: &[String],
    fields: &mut BTreeMap<Vec<String>, NativeStructuredInputField>,
) -> Result<(), NativeSchemaError> {
    let invalid = || NativeSchemaError::InvalidMetadata("structured_autogrow_input");
    let template = call_keyword_value(call, "template")
        .and_then(JsonValue::as_object)
        .ok_or_else(invalid)?;
    let template_kind = template
        .get("name")
        .and_then(JsonValue::as_str)
        .ok_or_else(invalid)?;
    let input = template
        .get("arguments")
        .and_then(JsonValue::as_array)
        .filter(|arguments| arguments.len() == 1)
        .and_then(|arguments| arguments[0].as_object())
        .ok_or_else(invalid)?;
    let minimum = match call_keyword_value(template, "min") {
        Some(value) => value
            .get("value")
            .and_then(JsonValue::as_u64)
            .ok_or_else(invalid)?,
        None => 1,
    };
    let names = if template_kind.ends_with("Autogrow.TemplateNames") {
        let values = call_keyword_value(template, "names")
            .and_then(|value| value.get("items"))
            .and_then(JsonValue::as_array)
            .filter(|values| !values.is_empty() && values.len() <= MAX_SCHEMA_ITEMS)
            .ok_or_else(invalid)?;
        values
            .iter()
            .take(MAX_AUTOGROW_NAMES)
            .map(|value| {
                if value.get("kind").and_then(JsonValue::as_str) != Some("literal") {
                    return Err(invalid());
                }
                value
                    .get("value")
                    .and_then(JsonValue::as_str)
                    .map(ToOwned::to_owned)
                    .ok_or_else(invalid)
            })
            .collect::<Result<Vec<_>, _>>()?
    } else if template_kind.ends_with("Autogrow.TemplatePrefix") {
        let prefix = call_keyword_value(template, "prefix")
            .and_then(|value| value.get("value"))
            .and_then(JsonValue::as_str)
            .ok_or_else(invalid)?;
        let maximum = match call_keyword_value(template, "max") {
            Some(value) => value
                .get("value")
                .and_then(JsonValue::as_u64)
                .ok_or_else(invalid)?,
            None => 10,
        };
        if maximum == 0 || maximum > MAX_AUTOGROW_NAMES as u64 {
            return Err(invalid());
        }
        (0..maximum)
            .map(|index| format!("{prefix}{index}"))
            .collect::<Vec<_>>()
    } else {
        return Err(invalid());
    };
    if minimum > names.len() as u64 {
        return Err(invalid());
    }
    let group_optional = call_keyword_boolean(call, "optional").unwrap_or(false);
    let input_optional = call_keyword_boolean(input, "optional").unwrap_or(false);
    let lazy = call_keyword_boolean(input, "lazy").unwrap_or(false);
    let mut seen = BTreeSet::new();
    for (index, name) in names.iter().enumerate() {
        validate_name(name)?;
        if !seen.insert(name) {
            return Err(NativeSchemaError::DuplicateField(name.to_owned()));
        }
        let mut path = prefix.to_vec();
        path.push(name.to_owned());
        let schema = structured_field_schema(input, name)?;
        let field = NativeStructuredInputField {
            path: path.clone(),
            schema,
            required: !group_optional && !input_optional && index < minimum as usize,
            lazy,
        };
        if fields.insert(path, field).is_some() {
            return Err(NativeSchemaError::DuplicateField(name.to_owned()));
        }
    }
    Ok(())
}

fn structured_field_schema(
    call: &serde_json::Map<String, JsonValue>,
    name: &str,
) -> Result<NativeInputSchemaMetadata, NativeSchemaError> {
    let constructor = call
        .get("name")
        .and_then(JsonValue::as_str)
        .ok_or(NativeSchemaError::InvalidMetadata("structured_input_field"))?;
    let arguments = call
        .get("arguments")
        .and_then(JsonValue::as_array)
        .ok_or(NativeSchemaError::InvalidMetadata("structured_input_field"))?;
    let source_type_names = if constructor.ends_with("MultiType.Input") {
        arguments
            .get(1)
            .and_then(|value| value.get("items"))
            .and_then(JsonValue::as_array)
            .ok_or(NativeSchemaError::InvalidMetadata("structured_input_type"))?
            .iter()
            .map(structured_source_type)
            .collect::<Result<Vec<_>, _>>()?
    } else {
        vec![structured_constructor_source_type(constructor)?]
    };
    let mut schema = NativeInputSchemaMetadata::compatibility(name, "STRING");
    schema.source_type_names = source_type_names;
    schema.default = call_keyword_value(call, "default").map(ast_schema_value);
    schema.minimum = call_keyword_value(call, "min").map(ast_schema_value);
    schema.maximum = call_keyword_value(call, "max").map(ast_schema_value);
    schema.step = call_keyword_value(call, "step").map(ast_schema_value);
    schema.choices = call_keyword_value(call, "options")
        .and_then(|value| value.get("items"))
        .and_then(JsonValue::as_array)
        .map(|values| values.iter().map(ast_schema_value).collect())
        .unwrap_or_default();
    schema.display_name = call_keyword_string(call, "display_name");
    schema.tooltip = call_keyword_string(call, "tooltip");
    schema.multiline = call_keyword_boolean(call, "multiline").unwrap_or(false);
    schema.socketless = call_keyword_boolean(call, "socketless").unwrap_or(false);
    schema.force_input = call_keyword_boolean(call, "force_input")
        .or_else(|| call_keyword_boolean(call, "forceInput"))
        .unwrap_or(false);
    schema.raw_link = call_keyword_boolean(call, "raw_link")
        .or_else(|| call_keyword_boolean(call, "rawLink"))
        .unwrap_or(false);
    schema.advanced = call_keyword_boolean(call, "advanced").unwrap_or(false);
    schema.validate()?;
    Ok(schema)
}

fn structured_constructor_source_type(constructor: &str) -> Result<String, NativeSchemaError> {
    let type_name = constructor
        .strip_suffix(".Input")
        .and_then(|value| value.rsplit('.').next())
        .ok_or(NativeSchemaError::InvalidMetadata("structured_input_type"))?;
    let source_type = match type_name {
        "Boolean" => "BOOLEAN",
        "Combo" => "COMBO",
        "DynamicCombo" => "COMFY_DYNAMICCOMBO_V3",
        "Float" => "FLOAT",
        "Int" => "INT",
        "MatchType" => "COMFY_MATCHTYPE_V3",
        "String" => "STRING",
        value => return Ok(value.to_ascii_uppercase()),
    };
    Ok(source_type.to_owned())
}

fn structured_source_type(value: &JsonValue) -> Result<String, NativeSchemaError> {
    match value.get("kind").and_then(JsonValue::as_str) {
        Some("attribute") | Some("name") => value
            .get("name")
            .and_then(JsonValue::as_str)
            .and_then(|name| name.rsplit('.').next())
            .map(|name| name.to_ascii_uppercase())
            .ok_or(NativeSchemaError::InvalidMetadata("structured_input_type")),
        _ => Err(NativeSchemaError::InvalidMetadata("structured_input_type")),
    }
}

fn call_keyword_value<'a>(
    call: &'a serde_json::Map<String, JsonValue>,
    name: &str,
) -> Option<&'a JsonValue> {
    call.get("keywords")
        .and_then(JsonValue::as_array)?
        .iter()
        .find(|keyword| keyword.get("name").and_then(JsonValue::as_str) == Some(name))?
        .get("value")
}

fn call_keyword_boolean(call: &serde_json::Map<String, JsonValue>, name: &str) -> Option<bool> {
    call_keyword_value(call, name)?.get("value")?.as_bool()
}

fn call_keyword_string(call: &serde_json::Map<String, JsonValue>, name: &str) -> Option<String> {
    call_keyword_value(call, name)?
        .get("value")?
        .as_str()
        .map(ToOwned::to_owned)
}

fn ast_schema_value(expression: &JsonValue) -> NativeSchemaValue {
    if expression.get("kind").and_then(JsonValue::as_str) == Some("literal")
        && let Some(value) = expression.get("value")
    {
        if value.is_null() {
            return NativeSchemaValue::Null;
        }
        if let Some(value) = value.as_bool() {
            return NativeSchemaValue::Boolean { value };
        }
        if let Some(value) = value.as_i64() {
            return NativeSchemaValue::SignedInteger { value };
        }
        if let Some(value) = value.as_u64() {
            return NativeSchemaValue::UnsignedInteger { value };
        }
        if let Some(value) = value.as_f64().filter(|value| value.is_finite()) {
            return NativeSchemaValue::FiniteDecimal {
                value: value.to_string(),
            };
        }
        if let Some(value) = value.as_str() {
            return NativeSchemaValue::String {
                value: value.to_owned(),
            };
        }
    }
    if expression.get("kind").and_then(JsonValue::as_str) == Some("list")
        && let Some(values) = expression.get("items").and_then(JsonValue::as_array)
    {
        return NativeSchemaValue::List {
            values: values.iter().map(ast_schema_value).collect(),
        };
    }
    let source = expression.to_string();
    let sha256 = format!("{:x}", Sha256::digest(source.as_bytes()));
    NativeSchemaValue::PreservedExpression { source, sha256 }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeDynamicSchemaMetadata {
    pub identity: String,
    pub prefix: Option<String>,
    pub names: Vec<String>,
    pub start_index: u32,
    pub minimum_count: u32,
    pub maximum_count: u32,
    pub input: Box<NativeInputSchemaMetadata>,
    pub extra: Vec<NativeSchemaField>,
}

impl NativeDynamicSchemaMetadata {
    pub fn compatibility(
        identity: impl Into<String>,
        start_index: u32,
        minimum_count: u32,
        maximum_count: u32,
        input: NativeInputSchemaMetadata,
    ) -> Self {
        Self {
            identity: identity.into(),
            prefix: None,
            names: Vec::new(),
            start_index,
            minimum_count,
            maximum_count,
            input: Box::new(input),
            extra: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<(), NativeSchemaError> {
        validate_name(&self.identity)?;
        validate_optional_name(self.prefix.as_deref())?;
        let mut names = BTreeSet::new();
        for name in &self.names {
            validate_name(name)?;
            if !names.insert(name.as_str()) {
                return Err(NativeSchemaError::DuplicateField(name.clone()));
            }
        }
        if self.minimum_count > self.maximum_count
            || self.start_index.checked_add(self.maximum_count).is_none()
        {
            return Err(NativeSchemaError::InvalidMetadata("dynamic_count"));
        }
        self.input.validate()?;
        validate_fields(&self.extra)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeOutputSchemaMetadata {
    pub name: String,
    pub source_type_name: String,
    pub display_name: Option<String>,
    pub tooltip: Option<String>,
    pub choices: Vec<NativeSchemaValue>,
    pub match_template: Option<String>,
    pub extra: Vec<NativeSchemaField>,
}

impl NativeOutputSchemaMetadata {
    pub fn compatibility(name: impl Into<String>, source_type_name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            source_type_name: source_type_name.into(),
            display_name: None,
            tooltip: None,
            choices: Vec::new(),
            match_template: None,
            extra: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<(), NativeSchemaError> {
        validate_name(&self.name)?;
        validate_name(&self.source_type_name)?;
        validate_optional_text(self.display_name.as_deref())?;
        validate_optional_text(self.tooltip.as_deref())?;
        validate_values(&self.choices)?;
        validate_optional_name(self.match_template.as_deref())?;
        validate_fields(&self.extra)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeNodeSchemaMetadata {
    pub schema_version: u16,
    pub provenance: NativeSchemaProvenance,
    pub feature_id: Option<String>,
    pub definition_sha256: Option<String>,
    pub has_intermediate_output: bool,
    pub development_only: bool,
    pub api_node: bool,
    pub not_idempotent: bool,
    pub enable_expand: bool,
    pub accept_all_inputs: bool,
    pub essentials_category: Option<String>,
    pub price_badge: Option<NativeSchemaValue>,
    pub extra: Vec<NativeSchemaField>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeDescriptorSchemaMetadata {
    pub node: NativeNodeSchemaMetadata,
    pub inputs: Vec<NativeInputSchemaMetadata>,
    pub dynamic_inputs: Vec<NativeDynamicSchemaMetadata>,
    pub outputs: Vec<NativeOutputSchemaMetadata>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogNodeInputSchemaMetadata {
    #[serde(flatten)]
    pub schema: NativeInputSchemaMetadata,
    pub requirement: NativeInputRequirement,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogNodeOutputSchemaMetadata {
    pub source_name: Option<String>,
    pub source_type_name: String,
    pub display_name: Option<String>,
    pub tooltip: Option<String>,
    pub choices: Vec<NativeSchemaValue>,
    pub match_template: Option<String>,
    pub extra: Vec<NativeSchemaField>,
    pub ordinal: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeSourcePresentationMetadata {
    pub is_deprecated: bool,
    pub is_experimental: bool,
    pub display_name: Option<String>,
    pub description: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogNodeSchemaMetadata {
    pub schema_version: u16,
    pub provenance: NativeSchemaProvenance,
    pub catalog_sha256: String,
    pub definition_sha256: String,
    pub inputs: Vec<CatalogNodeInputSchemaMetadata>,
    pub dynamic_inputs: Vec<NativeDynamicSchemaMetadata>,
    pub outputs: Vec<CatalogNodeOutputSchemaMetadata>,
    pub unresolved_inputs: Vec<NativeSchemaValue>,
    pub unresolved_outputs: Vec<NativeSchemaValue>,
    pub hidden: Vec<NativeSchemaValue>,
    pub node: NativeNodeSchemaMetadata,
    pub presentation: NativeSourcePresentationMetadata,
}

impl CatalogNodeSchemaMetadata {
    pub fn validate(&self) -> Result<(), NativeSchemaError> {
        if self.schema_version != NATIVE_SCHEMA_METADATA_VERSION
            || !valid_sha256(&self.catalog_sha256)
            || !valid_sha256(&self.definition_sha256)
        {
            return Err(NativeSchemaError::InvalidMetadata("catalog_schema"));
        }
        if self.inputs.len() > MAX_SCHEMA_ITEMS
            || self.dynamic_inputs.len() > MAX_SCHEMA_ITEMS
            || self.outputs.len() > MAX_SCHEMA_ITEMS
            || self.unresolved_inputs.len() > MAX_SCHEMA_ITEMS
            || self.unresolved_outputs.len() > MAX_SCHEMA_ITEMS
            || self.hidden.len() > MAX_SCHEMA_ITEMS
        {
            return Err(NativeSchemaError::ItemCountExceeded);
        }
        self.node.validate()?;
        let mut names = BTreeSet::new();
        for input in &self.inputs {
            input.schema.validate()?;
            if !names.insert(input.schema.name.as_str()) {
                return Err(NativeSchemaError::DuplicateField(input.schema.name.clone()));
            }
        }
        for dynamic_input in &self.dynamic_inputs {
            dynamic_input.validate()?;
        }
        for (ordinal, output) in self.outputs.iter().enumerate() {
            if output.ordinal != ordinal {
                return Err(NativeSchemaError::InvalidMetadata("output_ordinal"));
            }
            validate_optional_name(output.source_name.as_deref())?;
            validate_name(&output.source_type_name)?;
            validate_optional_text(output.display_name.as_deref())?;
            validate_optional_text(output.tooltip.as_deref())?;
            validate_optional_name(output.match_template.as_deref())?;
            validate_values(&output.choices)?;
            validate_fields(&output.extra)?;
        }
        validate_values(&self.unresolved_inputs)?;
        validate_values(&self.unresolved_outputs)?;
        validate_values(&self.hidden)?;
        validate_optional_text(self.presentation.display_name.as_deref())?;
        validate_optional_text(self.presentation.description.as_deref())?;
        let mut total_bytes = 0usize;
        let mut total_items = 0usize;
        accumulate_text(&self.catalog_sha256, &mut total_bytes, &mut total_items)?;
        accumulate_text(&self.definition_sha256, &mut total_bytes, &mut total_items)?;
        accumulate_node_metadata(&self.node, &mut total_bytes, &mut total_items)?;
        for input in &self.inputs {
            accumulate_input_metadata(&input.schema, &mut total_bytes, &mut total_items)?;
        }
        for dynamic_input in &self.dynamic_inputs {
            accumulate_dynamic_metadata(dynamic_input, &mut total_bytes, &mut total_items)?;
        }
        for output in &self.outputs {
            accumulate_catalog_output_metadata(output, &mut total_bytes, &mut total_items)?;
        }
        for value in self
            .unresolved_inputs
            .iter()
            .chain(&self.unresolved_outputs)
            .chain(&self.hidden)
        {
            value.validate_bounded(0, &mut total_bytes, &mut total_items)?;
        }
        accumulate_optional_text(
            self.presentation.display_name.as_deref(),
            &mut total_bytes,
            &mut total_items,
        )?;
        accumulate_optional_text(
            self.presentation.description.as_deref(),
            &mut total_bytes,
            &mut total_items,
        )?;
        Ok(())
    }

    pub fn bind_execution_ports(
        &self,
        input_names: &[String],
        dynamic_inputs: &[NativeDynamicSchemaMetadata],
        output_names: &[String],
    ) -> Result<NativeDescriptorSchemaMetadata, NativeSchemaError> {
        self.validate()?;
        if !self.unresolved_inputs.is_empty()
            || !self.unresolved_outputs.is_empty()
            || self.inputs.len() != input_names.len()
            || self.dynamic_inputs != dynamic_inputs
            || self.outputs.len() != output_names.len()
            || !self
                .inputs
                .iter()
                .zip(input_names)
                .all(|(schema, name)| schema.schema.name == *name)
        {
            return Err(NativeSchemaError::InvalidMetadata("execution_ports"));
        }
        let outputs = self
            .outputs
            .iter()
            .zip(output_names)
            .map(|(schema, name)| NativeOutputSchemaMetadata {
                name: name.clone(),
                source_type_name: schema.source_type_name.clone(),
                display_name: schema.display_name.clone(),
                tooltip: schema.tooltip.clone(),
                choices: schema.choices.clone(),
                match_template: schema.match_template.clone(),
                extra: schema.extra.clone(),
            })
            .collect();
        let value = NativeDescriptorSchemaMetadata {
            node: self.node.clone(),
            inputs: self
                .inputs
                .iter()
                .map(|input| input.schema.clone())
                .collect(),
            dynamic_inputs: dynamic_inputs.to_vec(),
            outputs,
        };
        value.validate()?;
        Ok(value)
    }
}

fn accumulate_catalog_output_metadata(
    metadata: &CatalogNodeOutputSchemaMetadata,
    total_bytes: &mut usize,
    total_items: &mut usize,
) -> Result<(), NativeSchemaError> {
    accumulate_optional_text(metadata.source_name.as_deref(), total_bytes, total_items)?;
    accumulate_text(&metadata.source_type_name, total_bytes, total_items)?;
    accumulate_optional_text(metadata.display_name.as_deref(), total_bytes, total_items)?;
    accumulate_optional_text(metadata.tooltip.as_deref(), total_bytes, total_items)?;
    accumulate_optional_text(metadata.match_template.as_deref(), total_bytes, total_items)?;
    for choice in &metadata.choices {
        choice.validate_bounded(0, total_bytes, total_items)?;
    }
    accumulate_fields(&metadata.extra, total_bytes, total_items)
}

impl NativeDescriptorSchemaMetadata {
    pub fn compatibility(
        provenance: NativeSchemaProvenance,
        inputs: impl IntoIterator<Item = (String, String)>,
        dynamic_inputs: impl IntoIterator<Item = NativeDynamicSchemaMetadata>,
        outputs: impl IntoIterator<Item = (String, String)>,
    ) -> Self {
        Self {
            node: NativeNodeSchemaMetadata::compatibility(provenance),
            inputs: inputs
                .into_iter()
                .map(|(name, source_type)| {
                    NativeInputSchemaMetadata::compatibility(name, source_type)
                })
                .collect(),
            dynamic_inputs: dynamic_inputs.into_iter().collect(),
            outputs: outputs
                .into_iter()
                .map(|(name, source_type)| {
                    NativeOutputSchemaMetadata::compatibility(name, source_type)
                })
                .collect(),
        }
    }

    pub fn validate(&self) -> Result<(), NativeSchemaError> {
        if self.inputs.len() > MAX_SCHEMA_ITEMS
            || self.dynamic_inputs.len() > MAX_SCHEMA_ITEMS
            || self.outputs.len() > MAX_SCHEMA_ITEMS
        {
            return Err(NativeSchemaError::ItemCountExceeded);
        }
        self.node.validate()?;
        for input in &self.inputs {
            input.validate()?;
        }
        for dynamic_input in &self.dynamic_inputs {
            dynamic_input.validate()?;
        }
        for output in &self.outputs {
            output.validate()?;
        }
        let mut total_bytes = 0usize;
        let mut total_items = 0usize;
        accumulate_node_metadata(&self.node, &mut total_bytes, &mut total_items)?;
        for input in &self.inputs {
            accumulate_input_metadata(input, &mut total_bytes, &mut total_items)?;
        }
        for dynamic_input in &self.dynamic_inputs {
            accumulate_dynamic_metadata(dynamic_input, &mut total_bytes, &mut total_items)?;
        }
        for output in &self.outputs {
            accumulate_output_metadata(output, &mut total_bytes, &mut total_items)?;
        }
        Ok(())
    }

    pub fn synthetic(
        input_names: impl IntoIterator<Item = String>,
        dynamic_inputs: impl IntoIterator<Item = NativeDynamicSchemaMetadata>,
        output_names: impl IntoIterator<Item = String>,
    ) -> Self {
        Self::compatibility(
            NativeSchemaProvenance::Synthetic,
            input_names.into_iter().map(|name| (name, "ANY".to_owned())),
            dynamic_inputs,
            output_names
                .into_iter()
                .map(|name| (name, "ANY".to_owned())),
        )
    }
}

fn accumulate_node_metadata(
    metadata: &NativeNodeSchemaMetadata,
    total_bytes: &mut usize,
    total_items: &mut usize,
) -> Result<(), NativeSchemaError> {
    accumulate_optional_text(metadata.feature_id.as_deref(), total_bytes, total_items)?;
    accumulate_optional_text(
        metadata.definition_sha256.as_deref(),
        total_bytes,
        total_items,
    )?;
    accumulate_optional_text(
        metadata.essentials_category.as_deref(),
        total_bytes,
        total_items,
    )?;
    if let Some(value) = &metadata.price_badge {
        value.validate_bounded(0, total_bytes, total_items)?;
    }
    accumulate_fields(&metadata.extra, total_bytes, total_items)
}

fn accumulate_input_metadata(
    metadata: &NativeInputSchemaMetadata,
    total_bytes: &mut usize,
    total_items: &mut usize,
) -> Result<(), NativeSchemaError> {
    accumulate_text(&metadata.name, total_bytes, total_items)?;
    for source_type in &metadata.source_type_names {
        accumulate_text(source_type, total_bytes, total_items)?;
    }
    for value in [
        metadata.default.as_ref(),
        metadata.minimum.as_ref(),
        metadata.maximum.as_ref(),
        metadata.step.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        value.validate_bounded(0, total_bytes, total_items)?;
    }
    for choice in &metadata.choices {
        choice.validate_bounded(0, total_bytes, total_items)?;
    }
    accumulate_optional_text(metadata.display_name.as_deref(), total_bytes, total_items)?;
    accumulate_optional_text(metadata.tooltip.as_deref(), total_bytes, total_items)?;
    accumulate_optional_text(metadata.widget_type.as_deref(), total_bytes, total_items)?;
    accumulate_fields(&metadata.extra, total_bytes, total_items)
}

fn accumulate_dynamic_metadata(
    metadata: &NativeDynamicSchemaMetadata,
    total_bytes: &mut usize,
    total_items: &mut usize,
) -> Result<(), NativeSchemaError> {
    accumulate_text(&metadata.identity, total_bytes, total_items)?;
    accumulate_optional_text(metadata.prefix.as_deref(), total_bytes, total_items)?;
    for name in &metadata.names {
        accumulate_text(name, total_bytes, total_items)?;
    }
    accumulate_input_metadata(&metadata.input, total_bytes, total_items)?;
    accumulate_fields(&metadata.extra, total_bytes, total_items)
}

fn accumulate_output_metadata(
    metadata: &NativeOutputSchemaMetadata,
    total_bytes: &mut usize,
    total_items: &mut usize,
) -> Result<(), NativeSchemaError> {
    accumulate_text(&metadata.name, total_bytes, total_items)?;
    accumulate_text(&metadata.source_type_name, total_bytes, total_items)?;
    accumulate_optional_text(metadata.display_name.as_deref(), total_bytes, total_items)?;
    accumulate_optional_text(metadata.tooltip.as_deref(), total_bytes, total_items)?;
    accumulate_optional_text(metadata.match_template.as_deref(), total_bytes, total_items)?;
    for choice in &metadata.choices {
        choice.validate_bounded(0, total_bytes, total_items)?;
    }
    accumulate_fields(&metadata.extra, total_bytes, total_items)
}

fn accumulate_fields(
    fields: &[NativeSchemaField],
    total_bytes: &mut usize,
    total_items: &mut usize,
) -> Result<(), NativeSchemaError> {
    for field in fields {
        accumulate_text(&field.name, total_bytes, total_items)?;
        field.value.validate_bounded(0, total_bytes, total_items)?;
    }
    Ok(())
}

fn accumulate_optional_text(
    value: Option<&str>,
    total_bytes: &mut usize,
    total_items: &mut usize,
) -> Result<(), NativeSchemaError> {
    if let Some(value) = value {
        accumulate_text(value, total_bytes, total_items)?;
    }
    Ok(())
}

fn accumulate_text(
    value: &str,
    total_bytes: &mut usize,
    total_items: &mut usize,
) -> Result<(), NativeSchemaError> {
    *total_items = total_items
        .checked_add(1)
        .ok_or(NativeSchemaError::ItemCountExceeded)?;
    if *total_items > MAX_SCHEMA_ITEMS {
        return Err(NativeSchemaError::ItemCountExceeded);
    }
    validate_text(value, total_bytes)
}

impl NativeNodeSchemaMetadata {
    pub fn compatibility(provenance: NativeSchemaProvenance) -> Self {
        Self {
            schema_version: NATIVE_SCHEMA_METADATA_VERSION,
            provenance,
            feature_id: None,
            definition_sha256: None,
            has_intermediate_output: false,
            development_only: false,
            api_node: false,
            not_idempotent: false,
            enable_expand: false,
            accept_all_inputs: false,
            essentials_category: None,
            price_badge: None,
            extra: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<(), NativeSchemaError> {
        if self.schema_version != NATIVE_SCHEMA_METADATA_VERSION {
            return Err(NativeSchemaError::InvalidMetadata("schema_version"));
        }
        validate_optional_name(self.feature_id.as_deref())?;
        if let Some(digest) = &self.definition_sha256
            && !valid_sha256(digest)
        {
            return Err(NativeSchemaError::InvalidDigest);
        }
        validate_optional_text(self.essentials_category.as_deref())?;
        if let Some(value) = &self.price_badge {
            value.validate()?;
        }
        validate_fields(&self.extra)
    }
}

fn validate_values(values: &[NativeSchemaValue]) -> Result<(), NativeSchemaError> {
    if values.len() > MAX_SCHEMA_ITEMS {
        return Err(NativeSchemaError::ItemCountExceeded);
    }
    for value in values {
        value.validate()?;
    }
    Ok(())
}

fn validate_optional_values<'a>(
    values: impl IntoIterator<Item = Option<&'a NativeSchemaValue>>,
) -> Result<(), NativeSchemaError> {
    for value in values.into_iter().flatten() {
        value.validate()?;
    }
    Ok(())
}

fn validate_fields(fields: &[NativeSchemaField]) -> Result<(), NativeSchemaError> {
    let mut names = BTreeSet::new();
    for field in fields {
        validate_name(&field.name)?;
        if !names.insert(field.name.as_str()) {
            return Err(NativeSchemaError::DuplicateField(field.name.clone()));
        }
        field.value.validate()?;
    }
    Ok(())
}

fn validate_source_types(values: &[String]) -> Result<(), NativeSchemaError> {
    if values.is_empty() || values.len() > 64 {
        return Err(NativeSchemaError::InvalidSourceType);
    }
    let mut unique = BTreeSet::new();
    for value in values {
        validate_name(value).map_err(|_| NativeSchemaError::InvalidSourceType)?;
        if !unique.insert(value.as_str()) {
            return Err(NativeSchemaError::InvalidSourceType);
        }
    }
    Ok(())
}

fn validate_optional_text(value: Option<&str>) -> Result<(), NativeSchemaError> {
    if let Some(value) = value {
        let mut total = 0;
        validate_text(value, &mut total)?;
    }
    Ok(())
}

fn validate_optional_name(value: Option<&str>) -> Result<(), NativeSchemaError> {
    if let Some(value) = value {
        validate_name(value)?;
    }
    Ok(())
}

fn validate_name(value: &str) -> Result<(), NativeSchemaError> {
    if value.is_empty()
        || value.len() > MAX_SCHEMA_TEXT_BYTES
        || value.chars().any(|character| character.is_control())
    {
        return Err(NativeSchemaError::InvalidFieldName);
    }
    Ok(())
}

fn validate_text(value: &str, total_bytes: &mut usize) -> Result<(), NativeSchemaError> {
    if value.len() > MAX_SCHEMA_TEXT_BYTES || value.contains('\0') {
        return Err(NativeSchemaError::TextTooLarge);
    }
    *total_bytes = total_bytes
        .checked_add(value.len())
        .ok_or(NativeSchemaError::TotalBytesExceeded)?;
    if *total_bytes > MAX_SCHEMA_TOTAL_BYTES {
        return Err(NativeSchemaError::TotalBytesExceeded);
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_finite_decimal(value: &str) -> bool {
    if value.is_empty()
        || value.bytes().any(|byte| byte.is_ascii_whitespace())
        || matches!(
            value,
            "NaN" | "nan" | "inf" | "-inf" | "+inf" | "Infinity" | "-Infinity" | "+Infinity"
        )
    {
        return false;
    }
    let bytes = value.as_bytes();
    let mut index = usize::from(matches!(bytes.first(), Some(b'-') | Some(b'+')));
    let integer_start = index;
    while matches!(bytes.get(index), Some(byte) if byte.is_ascii_digit()) {
        index += 1;
    }
    let integer_digits = index - integer_start;
    let mut fractional_digits = 0;
    if bytes.get(index) == Some(&b'.') {
        index += 1;
        let fractional_start = index;
        while matches!(bytes.get(index), Some(byte) if byte.is_ascii_digit()) {
            index += 1;
        }
        fractional_digits = index - fractional_start;
    }
    if integer_digits == 0 && fractional_digits == 0 {
        return false;
    }
    if matches!(bytes.get(index), Some(b'e') | Some(b'E')) {
        index += 1;
        if matches!(bytes.get(index), Some(b'-') | Some(b'+')) {
            index += 1;
        }
        let exponent_start = index;
        while matches!(bytes.get(index), Some(byte) if byte.is_ascii_digit()) {
            index += 1;
        }
        if exponent_start == index {
            return false;
        }
    }
    index == bytes.len()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortDescriptor {
    pub name: String,
    pub type_name: String,
    pub required: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeDescriptor {
    pub type_name: String,
    pub display_name: String,
    pub inputs: Vec<PortDescriptor>,
    pub outputs: Vec<PortDescriptor>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CatalogNodeSource {
    Registered,
    Inactive,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CatalogNodeStatus {
    DescriptorOnly,
    ProviderRequired,
    Inactive,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogNodeDescriptor {
    pub schema_version: u16,
    pub source: CatalogNodeSource,
    pub node_identifier: String,
    pub class_name: String,
    pub display_name: String,
    pub category: String,
    pub product: String,
    pub classification: String,
    pub availability: String,
    pub evidence_level: String,
    pub confidence: String,
    pub schema_api: Option<String>,
    pub schema_source: String,
    pub inputs: String,
    pub outputs: String,
    pub input_is_list: String,
    pub output_is_list: String,
    pub lazy_inputs: String,
    pub output_node: bool,
    pub execution_function: String,
    pub validation: String,
    pub caching: String,
    pub change_detection: String,
    pub execution_blocking: String,
    pub error_behavior: String,
    pub source_file: String,
    pub source_symbol: String,
    pub source_line: Option<u32>,
    pub test_evidence: String,
    pub registration_evidence: String,
    pub inactive_reason: Option<String>,
    pub zed_status: Option<String>,
    pub parity_gap: Option<String>,
    pub feature_id: String,
    pub catalog_status: CatalogNodeStatus,
}

impl CatalogNodeDescriptor {
    pub fn is_registered(&self) -> bool {
        self.source == CatalogNodeSource::Registered
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn native_schema_values_preserve_integer_decimal_and_field_order() -> Result<(), Box<dyn Error>>
    {
        let value = NativeSchemaValue::Object {
            fields: vec![
                NativeSchemaField {
                    name: "maximum".to_owned(),
                    value: NativeSchemaValue::UnsignedInteger { value: u64::MAX },
                },
                NativeSchemaField {
                    name: "step".to_owned(),
                    value: NativeSchemaValue::FiniteDecimal {
                        value: "1.25e-3".to_owned(),
                    },
                },
            ],
        };
        value.validate()?;
        let encoded = serde_json::to_string(&value)?;
        let decoded: NativeSchemaValue = serde_json::from_str(&encoded)?;
        assert_eq!(decoded, value);
        assert!(encoded.find("maximum") < encoded.find("step"));
        Ok(())
    }

    #[test]
    fn native_schema_values_reject_non_finite_and_malformed_data() {
        for value in ["NaN", "inf", "1e", ".", " 1.0"] {
            assert_eq!(
                NativeSchemaValue::FiniteDecimal {
                    value: value.to_owned(),
                }
                .validate(),
                Err(NativeSchemaError::InvalidDecimal)
            );
        }
        assert!(matches!(
            NativeSchemaValue::Object {
                fields: vec![
                    NativeSchemaField {
                        name: "choice".to_owned(),
                        value: NativeSchemaValue::Null,
                    },
                    NativeSchemaField {
                        name: "choice".to_owned(),
                        value: NativeSchemaValue::Null,
                    },
                ],
            }
            .validate(),
            Err(NativeSchemaError::DuplicateField(name)) if name == "choice"
        ));
        assert_eq!(
            NativeSchemaValue::PreservedExpression {
                source: "2 ** 31 - 1".to_owned(),
                sha256: "invalid".to_owned(),
            }
            .validate(),
            Err(NativeSchemaError::InvalidDigest)
        );
    }

    #[test]
    fn native_schema_metadata_validates_cross_field_contracts() -> Result<(), Box<dyn Error>> {
        let mut input = NativeInputSchemaMetadata::compatibility("seed", "INT");
        input.minimum = Some(NativeSchemaValue::SignedInteger { value: 0 });
        input.maximum = Some(NativeSchemaValue::UnsignedInteger { value: u64::MAX });
        input.choices = vec![NativeSchemaValue::String {
            value: "first".to_owned(),
        }];
        input.validate()?;
        input.socketless = true;
        input.force_input = true;
        assert_eq!(
            input.validate(),
            Err(NativeSchemaError::InvalidMetadata("socketless"))
        );

        let dynamic = NativeDynamicSchemaMetadata::compatibility(
            "images",
            0,
            1,
            50,
            NativeInputSchemaMetadata::compatibility("image", "IMAGE"),
        );
        dynamic.validate()?;
        let invalid = NativeDynamicSchemaMetadata::compatibility(
            "images",
            u32::MAX,
            1,
            2,
            NativeInputSchemaMetadata::compatibility("image", "IMAGE"),
        );
        assert_eq!(
            invalid.validate(),
            Err(NativeSchemaError::InvalidMetadata("dynamic_count"))
        );
        Ok(())
    }

    #[test]
    fn native_schema_metadata_enforces_one_aggregate_budget() {
        let mut metadata = NativeDescriptorSchemaMetadata::compatibility(
            NativeSchemaProvenance::Synthetic,
            [("value".to_owned(), "STRING".to_owned())],
            std::iter::empty(),
            [("value".to_owned(), "STRING".to_owned())],
        );
        metadata.inputs[0].choices = (0..9)
            .map(|_| NativeSchemaValue::String {
                value: "x".repeat(250_000),
            })
            .collect();
        assert_eq!(
            metadata.validate(),
            Err(NativeSchemaError::TotalBytesExceeded)
        );

        let preserved_expressions = NativeSchemaValue::List {
            values: (0..4_095)
                .map(|_| NativeSchemaValue::PreservedExpression {
                    source: "x".repeat(450),
                    sha256: "a".repeat(64),
                })
                .collect(),
        };
        assert_eq!(
            preserved_expressions.validate(),
            Err(NativeSchemaError::TotalBytesExceeded)
        );
    }

    #[test]
    fn dynamic_combo_options_recover_source_declared_multitype_fields() -> Result<(), Box<dyn Error>>
    {
        let expression = serde_json::json!({
            "arguments": [
                {"kind": "attribute", "name": "ResizeType.MATCH_SIZE"},
                {"kind": "list", "items": [
                    {
                        "arguments": [
                            {"kind": "literal", "value": "match"},
                            {"kind": "list", "items": [
                                {"kind": "attribute", "name": "io.Image"},
                                {"kind": "attribute", "name": "io.Mask"}
                            ]}
                        ],
                        "keywords": [
                            {"name": "tooltip", "value": {"kind": "literal", "value": "reference"}}
                        ],
                        "kind": "call",
                        "name": "io.MultiType.Input"
                    },
                    {
                        "arguments": [{"kind": "literal", "value": "crop"}],
                        "keywords": [
                            {"name": "options", "value": {"kind": "list", "items": [
                                {"kind": "literal", "value": "disabled"},
                                {"kind": "literal", "value": "center"}
                            ]}},
                            {"name": "default", "value": {"kind": "literal", "value": "center"}}
                        ],
                        "kind": "call",
                        "name": "io.Combo.Input"
                    }
                ]}
            ],
            "keywords": [],
            "kind": "call",
            "name": "io.DynamicCombo.Option"
        });
        let source = serde_json::to_string(&expression)?;
        let mut schema =
            NativeInputSchemaMetadata::compatibility("resize_type", "COMFY_DYNAMICCOMBO_V3");
        schema.choices.push(NativeSchemaValue::PreservedExpression {
            sha256: format!("{:x}", Sha256::digest(source.as_bytes())),
            source,
        });
        let options = schema.structured_options()?;
        assert_eq!(options.len(), 1);
        assert_eq!(options[0].selector, "match size");
        assert_eq!(options[0].fields.len(), 2);
        assert_eq!(options[0].fields[0].path, ["crop"]);
        assert_eq!(options[0].fields[1].path, ["match"]);
        assert_eq!(
            options[0].fields[1].schema.source_type_names,
            ["IMAGE", "MASK"]
        );
        assert!(options[0].fields.iter().all(|field| field.required));
        Ok(())
    }

    #[test]
    fn resize_image_mask_catalog_retains_the_source_declared_match_union()
    -> Result<(), Box<dyn Error>> {
        let schema = crate::built_in_source_schema("ResizeImageMaskNode")?;
        let resize_type = schema
            .inputs
            .iter()
            .find(|input| input.schema.name == "resize_type")
            .ok_or("ResizeImageMaskNode has no resize_type input")?;
        let match_size = resize_type
            .schema
            .structured_options()?
            .into_iter()
            .find(|option| option.selector == "match size")
            .ok_or("ResizeImageMaskNode has no match size option")?;
        let match_input = match_size
            .fields
            .iter()
            .find(|field| field.path.as_slice() == ["match"])
            .ok_or("match size has no match input")?;
        assert_eq!(match_input.schema.source_type_names, ["IMAGE", "MASK"]);
        assert!(match_input.required);
        Ok(())
    }

    #[test]
    fn openrouter_dynamic_options_expand_named_autogrow_inputs() -> Result<(), Box<dyn Error>> {
        let schema = crate::built_in_source_schema("OpenRouterLLMNode")?;
        let model = schema
            .inputs
            .iter()
            .find(|input| input.schema.name == "model")
            .ok_or("OpenRouter model input is absent")?;
        let options = model.schema.structured_options()?;
        let claude = options
            .iter()
            .find(|option| option.selector == "anthropic/claude-opus-4.7")
            .ok_or("pinned Claude option is absent")?;
        assert!(claude.fields.iter().all(|field| field.path != ["images"]));
        let images = claude
            .fields
            .iter()
            .filter(|field| field.path.first().is_some_and(|part| part == "images"))
            .collect::<Vec<_>>();
        assert_eq!(images.len(), 20);
        assert!(images.iter().all(|field| !field.required));
        assert_eq!(images[0].schema.source_type_names, ["IMAGE"]);
        assert!(
            images
                .iter()
                .any(|field| field.path == ["images", "image_20"])
        );

        let gemini = options
            .iter()
            .find(|option| option.selector == "google/gemini-3.5-flash")
            .ok_or("pinned Gemini option is absent")?;
        assert_eq!(
            gemini
                .fields
                .iter()
                .filter(|field| field.path.first().is_some_and(|part| part == "videos"))
                .count(),
            4
        );

        let grok = crate::built_in_source_schema("GrokVideoReferenceNode")?;
        let grok_model = grok
            .inputs
            .iter()
            .find(|input| input.schema.name == "model")
            .ok_or("Grok model input is absent")?;
        let option = grok_model
            .schema
            .structured_options()?
            .into_iter()
            .find(|option| option.selector == "grok-imagine-video")
            .ok_or("pinned Grok option is absent")?;
        let references = option
            .fields
            .iter()
            .filter(|field| {
                field
                    .path
                    .first()
                    .is_some_and(|part| part == "reference_images")
            })
            .collect::<Vec<_>>();
        assert_eq!(references.len(), 7);
        assert_eq!(references[0].path, ["reference_images", "reference_0"]);
        assert!(references[0].required);
        assert!(references.iter().skip(1).all(|field| !field.required));
        assert_eq!(references[0].schema.source_type_names, ["IMAGE"]);
        Ok(())
    }

    #[test]
    fn dynamic_options_bind_source_positional_and_keyword_arguments() -> Result<(), Box<dyn Error>>
    {
        let key = serde_json::json!({"kind":"literal","value":"on"});
        let inputs = serde_json::json!({"kind":"list","items":[]});
        let expression = |arguments: JsonValue, keywords: JsonValue| {
            serde_json::json!({
                "kind":"call","name":"io.DynamicCombo.Option","arguments":arguments,"keywords":keywords,
            })
        };
        let expected = NativeStructuredInputOption {
            selector: "on".to_owned(),
            fields: Vec::new(),
        };
        for value in [
            expression(serde_json::json!([key, inputs]), serde_json::json!([])),
            expression(
                serde_json::json!([key]),
                serde_json::json!([{"name":"inputs","value":inputs}]),
            ),
            expression(
                serde_json::json!([]),
                serde_json::json!([{"name":"inputs","value":inputs},{"name":"key","value":key}]),
            ),
        ] {
            assert_eq!(structured_option_from_expression(&value)?, expected);
        }
        for value in [
            expression(
                serde_json::json!([key, inputs, inputs]),
                serde_json::json!([]),
            ),
            expression(
                serde_json::json!([key, inputs]),
                serde_json::json!([{"name":"key","value":key}]),
            ),
            expression(
                serde_json::json!([key]),
                serde_json::json!([{"name":"inputs","value":inputs},{"name":"inputs","value":inputs}]),
            ),
            expression(
                serde_json::json!([]),
                serde_json::json!([{"name":"key","value":key},{"name":"unknown","value":inputs}]),
            ),
            expression(serde_json::json!([key]), serde_json::json!([])),
        ] {
            assert!(structured_option_from_expression(&value).is_err());
        }
        Ok(())
    }

    #[test]
    fn text_generation_dynamic_options_preserve_source_fields_and_unsigned_seed()
    -> Result<(), Box<dyn Error>> {
        for name in ["TextGenerate", "TextGenerateLTX2Prompt"] {
            let schema = crate::built_in_source_schema(name)?;
            let options = schema
                .inputs
                .iter()
                .find(|input| input.schema.name == "sampling_mode")
                .ok_or("sampling mode absent")?
                .schema
                .structured_options()?;
            assert_eq!(options.len(), 2);
            let enabled = options
                .iter()
                .find(|option| option.selector == "on")
                .ok_or("on option absent")?;
            let disabled = options
                .iter()
                .find(|option| option.selector == "off")
                .ok_or("off option absent")?;
            assert_eq!(enabled.fields.len(), 7);
            assert!(disabled.fields.is_empty());
            let seed = enabled
                .fields
                .iter()
                .find(|field| field.path == ["seed"])
                .ok_or("seed absent")?;
            assert!(seed.required);
            assert_eq!(
                seed.schema.maximum,
                Some(NativeSchemaValue::UnsignedInteger { value: u64::MAX })
            );
            let presence = enabled
                .fields
                .iter()
                .find(|field| field.path == ["presence_penalty"])
                .ok_or("presence penalty absent")?;
            assert!(!presence.required);
        }
        Ok(())
    }
}
