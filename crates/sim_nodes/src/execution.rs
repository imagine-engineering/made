use crate::{
    NativeDescriptorSchemaMetadata, NativeInputSchemaMetadata, NativeSchemaValue,
    NativeStoredPayload, NativeStoredPayloadError,
};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_media::{
    MetadataWritePolicy, NativeVideoBitDepth, NativeVideoCodec, NativeVideoCrf, NativeVideoPayload,
    NativeVideoPixelFormat, PngError, PngLimits, encode_api_png_frame_with_context,
    encode_api_png_mask_with_context, encode_api_png_source_batch_with_context,
    encode_png_frame_with_policy_and_context,
};
use sim_model::clip::ClipError;
use sim_model::vae::NativeVaeRecipeError;
use sim_model::{
    AudioEncoderError, ClipVisionError, FrameInterpolationError, LossMap, ModelFamilyError,
    NativeBackgroundRemovalError, NativeDepthAnything3Error, NativeGligenError,
    NativeLatentUpscaleModelError, NativeLoraApplicationMode, NativeLoraModelResource,
    NativeModelPayload, NativeModelPayloadError, NativeModelResourceRole, NativeMogeError,
    NativePhotoMakerError, NativeStyleModelError, NativeVisionModelError,
    conditioning::ConditioningSet,
};
use sim_sampler::{NativeDiffusionPayload, NativeSamplingPolicy, profile_for_model};
use sim_tensor::{
    CpuBackend, DType, DeviceId, ExecutionContext, ImageTensor, MAX_SHADER_OUTPUTS,
    MAX_SHADER_PASSES, NativeLatentBundle, NativeShaderError, NativeShaderExecutor,
    NativeShaderRequest, NativeShaderResult, NativeTrainingComputeDType, NativeTrainingLoss,
    NativeTrainingOptimizer, RetryRngPolicy, RngAlgorithm, RngError, RngProfileVersion, RngStream,
    RngStreamAddress, RngTransaction, ScratchBindingIdentity, ScratchReservation, StreamId, Tensor,
    TensorError,
};
use sim_types::{ApiPrompt, AttemptId, CancellationToken, NodeId, PromptId};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    sync::Arc,
};
use thiserror::Error;
use uuid::Uuid;

pub const NATIVE_NODE_CONTRACT_SCHEMA_VERSION: u16 = 2;
pub const LEGACY_NATIVE_NODE_CONTRACT_SCHEMA_VERSION: u16 = 1;
pub const NATIVE_OPAQUE_HANDLE_SCHEMA_VERSION: u16 = 1;
pub const NATIVE_STRUCTURED_VALUE_SCHEMA_VERSION: u16 = 1;
pub const NATIVE_TEXT_GENERATION_RNG_PHASE: &str = "native-text-generation";
pub const NATIVE_TEXT_GENERATION_MAX_NEW_TOKENS: usize = 32_768;

const MAX_IDENTIFIER_BYTES: usize = 4_096;
const MAX_TEXT_BYTES: usize = 1024 * 1024;
const MAX_UNKNOWN_BYTES: usize = 1024 * 1024;
const MAX_VALUE_DEPTH: usize = 32;
const MAX_LIST_VALUES: usize = 1_000_000;
const MAX_PORTS: usize = 65_536;
const MAX_TYPE_UNION_MEMBERS: usize = 128;
const MAX_NATIVE_ASSET_FOLDER_CATEGORY_BYTES: usize = 128;
const MAX_NATIVE_ASSET_NAME_BYTES: usize = 16 * 1024;
const MAX_NATIVE_ASSET_NAME_SELECTIONS: usize = 3;
const MAX_NATIVE_ASSET_NAME_LIST_ENTRIES: usize = 100_000;
const MAX_NATIVE_MODEL_FOLDER_CATEGORY_BYTES: usize = 128;
const MAX_NATIVE_MODEL_SOURCE_NAME_BYTES: usize = 16 * 1024;
const MAX_NATIVE_MODEL_SOURCE_NAMES: usize = 4;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativePrimitiveType {
    Null,
    Boolean,
    Integer,
    Number,
    String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum NativePrimitive {
    Null,
    Boolean(bool),
    Integer(i64),
    UnsignedInteger(u64),
    Number(f64),
    String(String),
}

impl NativePrimitive {
    pub const fn primitive_type(&self) -> NativePrimitiveType {
        match self {
            Self::Null => NativePrimitiveType::Null,
            Self::Boolean(_) => NativePrimitiveType::Boolean,
            Self::Integer(_) | Self::UnsignedInteger(_) => NativePrimitiveType::Integer,
            Self::Number(_) => NativePrimitiveType::Number,
            Self::String(_) => NativePrimitiveType::String,
        }
    }

    fn validate(&self) -> Result<(), NativeNodeContractError> {
        match self {
            Self::Number(value) if !value.is_finite() => {
                Err(NativeNodeContractError::NonFiniteNumber)
            }
            Self::String(value) => validate_workflow_text("primitive string", value),
            _ => Ok(()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeHandleKind {
    Tensor,
    Model,
    Clip,
    Vae,
    ControlNet,
    Conditioning,
    Latent,
    Image,
    Mask,
    Audio,
    Video,
    ThreeD,
    Artifact,
    ProviderTask,
    StructuredCompute,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeHandleType {
    pub kind: NativeHandleKind,
    pub type_id: String,
}

impl NativeHandleType {
    pub fn new(
        kind: NativeHandleKind,
        type_id: impl Into<String>,
    ) -> Result<Self, NativeNodeContractError> {
        let handle_type = Self {
            kind,
            type_id: type_id.into(),
        };
        handle_type.validate()?;
        Ok(handle_type)
    }

    pub fn validate(&self) -> Result<(), NativeNodeContractError> {
        if self.type_id.is_empty()
            || self.type_id.len() > MAX_IDENTIFIER_BYTES
            || !self.type_id.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b':' | b'@')
            })
        {
            return Err(NativeNodeContractError::InvalidHandleType);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeHandleStoreIdentity {
    pub store_id: Uuid,
    pub generation_id: Uuid,
}

impl NativeHandleStoreIdentity {
    pub fn new(store_id: Uuid, generation_id: Uuid) -> Result<Self, NativeNodeContractError> {
        let identity = Self {
            store_id,
            generation_id,
        };
        identity.validate()?;
        Ok(identity)
    }

    pub fn validate(&self) -> Result<(), NativeNodeContractError> {
        if self.store_id.is_nil() || self.generation_id.is_nil() {
            return Err(NativeNodeContractError::InvalidHandleStoreIdentity);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeOpaqueHandle {
    schema_version: u16,
    handle_type: NativeHandleType,
    store_identity: NativeHandleStoreIdentity,
    identifier: String,
    generation: u64,
    digest_sha256: Option<String>,
}

impl NativeOpaqueHandle {
    pub fn new(
        handle_type: NativeHandleType,
        store_identity: NativeHandleStoreIdentity,
        identifier: impl Into<String>,
        generation: u64,
        digest_sha256: Option<String>,
    ) -> Result<Self, NativeNodeContractError> {
        let handle = Self {
            schema_version: NATIVE_OPAQUE_HANDLE_SCHEMA_VERSION,
            handle_type,
            store_identity,
            identifier: identifier.into(),
            generation,
            digest_sha256,
        };
        handle.validate()?;
        Ok(handle)
    }

    pub fn handle_type(&self) -> &NativeHandleType {
        &self.handle_type
    }

    pub const fn store_identity(&self) -> NativeHandleStoreIdentity {
        self.store_identity
    }

    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub fn digest_sha256(&self) -> Option<&str> {
        self.digest_sha256.as_deref()
    }

    pub fn validate(&self) -> Result<(), NativeNodeContractError> {
        if self.schema_version != NATIVE_OPAQUE_HANDLE_SCHEMA_VERSION {
            return Err(NativeNodeContractError::UnsupportedHandleSchema(
                self.schema_version,
            ));
        }
        self.handle_type.validate()?;
        self.store_identity.validate()?;
        validate_identifier("opaque handle identifier", &self.identifier)?;
        if self.generation == 0 {
            return Err(NativeNodeContractError::InvalidHandleGeneration);
        }
        if let Some(digest) = &self.digest_sha256
            && !valid_sha256(digest)
        {
            return Err(NativeNodeContractError::InvalidDigest);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NativeValue {
    Primitive { value: NativePrimitive },
    Handle { value: NativeOpaqueHandle },
    List { values: Vec<NativeValue> },
    PreservedUnknown { type_name: String, value: Value },
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeStructuredValue {
    type_name: String,
    fields: BTreeMap<String, NativeValue>,
}

impl NativeStructuredValue {
    const MARKER: &'static str = "zed.native-structured-value@1";

    pub fn checked(
        type_name: impl Into<String>,
        fields: BTreeMap<String, NativeValue>,
    ) -> Result<Self, NativeNodeContractError> {
        let value = Self {
            type_name: type_name.into(),
            fields,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn type_name(&self) -> &str {
        &self.type_name
    }

    pub fn fields(&self) -> &BTreeMap<String, NativeValue> {
        &self.fields
    }

    pub fn get(&self, name: &str) -> Option<&NativeValue> {
        self.fields.get(name)
    }

    pub fn validate(&self) -> Result<(), NativeNodeContractError> {
        validate_identifier("structured value type", &self.type_name)?;
        if self.fields.len() > MAX_LIST_VALUES {
            return Err(NativeNodeContractError::TooManyListValues);
        }
        for (name, value) in &self.fields {
            validate_identifier("structured value field", name)?;
            value.validate()?;
        }
        Ok(())
    }

    pub fn into_native_value(self) -> NativeValue {
        let fields = self
            .fields
            .into_iter()
            .map(|(name, value)| NativeValue::List {
                values: vec![
                    NativeValue::Primitive {
                        value: NativePrimitive::String(name),
                    },
                    value,
                ],
            })
            .collect();
        NativeValue::List {
            values: vec![
                NativeValue::Primitive {
                    value: NativePrimitive::String(Self::MARKER.to_owned()),
                },
                NativeValue::Primitive {
                    value: NativePrimitive::String(self.type_name),
                },
                NativeValue::List { values: fields },
            ],
        }
    }

    pub fn from_native_value(value: &NativeValue) -> Result<Option<Self>, NativeNodeContractError> {
        let NativeValue::List { values } = value else {
            return Ok(None);
        };
        let marker = match values.first() {
            Some(NativeValue::Primitive {
                value: NativePrimitive::String(marker),
            }) => marker,
            _ => return Ok(None),
        };
        if marker != Self::MARKER {
            return Ok(None);
        }
        if values.len() != 3 {
            return Err(NativeNodeContractError::InvalidStructuredValue);
        }
        let type_name = match values.get(1) {
            Some(NativeValue::Primitive {
                value: NativePrimitive::String(type_name),
            }) => type_name.clone(),
            _ => return Err(NativeNodeContractError::InvalidStructuredValue),
        };
        let field_values = match values.get(2) {
            Some(NativeValue::List { values }) => values,
            _ => return Err(NativeNodeContractError::InvalidStructuredValue),
        };
        let mut fields = BTreeMap::new();
        for field in field_values {
            let NativeValue::List { values } = field else {
                return Err(NativeNodeContractError::InvalidStructuredValue);
            };
            let name = match values.as_slice() {
                [
                    NativeValue::Primitive {
                        value: NativePrimitive::String(name),
                    },
                    _,
                ] => name.clone(),
                _ => return Err(NativeNodeContractError::InvalidStructuredValue),
            };
            let value = values
                .get(1)
                .cloned()
                .ok_or(NativeNodeContractError::InvalidStructuredValue)?;
            if fields.insert(name, value).is_some() {
                return Err(NativeNodeContractError::InvalidStructuredValue);
            }
        }
        Self::checked(type_name, fields).map(Some)
    }

    pub fn into_runtime_value(self) -> Result<NativeValue, NativeNodeContractError> {
        if let Some(value) = self.as_json_value()? {
            return Ok(NativeValue::PreservedUnknown {
                type_name: self.type_name,
                value,
            });
        }
        Ok(self.into_native_value())
    }

    fn as_json_value(&self) -> Result<Option<Value>, NativeNodeContractError> {
        let mut fields = serde_json::Map::new();
        for (name, value) in &self.fields {
            let Some(value) = native_value_as_json(value)? else {
                return Ok(None);
            };
            fields.insert(name.clone(), value);
        }
        Ok(Some(Value::Object(fields)))
    }
}

fn native_value_as_json(value: &NativeValue) -> Result<Option<Value>, NativeNodeContractError> {
    Ok(Some(match value {
        NativeValue::Primitive { value } => match value {
            NativePrimitive::Null => Value::Null,
            NativePrimitive::Boolean(value) => Value::Bool(*value),
            NativePrimitive::Integer(value) => Value::Number((*value).into()),
            NativePrimitive::UnsignedInteger(value) => Value::Number((*value).into()),
            NativePrimitive::Number(value) => {
                let Some(value) = serde_json::Number::from_f64(*value) else {
                    return Err(NativeNodeContractError::InvalidStructuredValue);
                };
                Value::Number(value)
            }
            NativePrimitive::String(value) => Value::String(value.clone()),
        },
        NativeValue::Handle { .. } => return Ok(None),
        NativeValue::List { values } => {
            if let Some(structured) = NativeStructuredValue::from_native_value(value)? {
                let Some(value) = structured.as_json_value()? else {
                    return Ok(None);
                };
                value
            } else {
                let mut result = Vec::with_capacity(values.len());
                for value in values {
                    let Some(value) = native_value_as_json(value)? else {
                        return Ok(None);
                    };
                    result.push(value);
                }
                Value::Array(result)
            }
        }
        NativeValue::PreservedUnknown { value, .. } => value.clone(),
    }))
}

impl NativeValue {
    pub fn validate(&self) -> Result<(), NativeNodeContractError> {
        self.validate_at_depth(0)
    }

    fn validate_at_depth(&self, depth: usize) -> Result<(), NativeNodeContractError> {
        if depth > MAX_VALUE_DEPTH {
            return Err(NativeNodeContractError::ValueNestingTooDeep);
        }
        match self {
            Self::Primitive { value } => value.validate(),
            Self::Handle { value } => value.validate(),
            Self::List { values } => {
                if values.len() > MAX_LIST_VALUES {
                    return Err(NativeNodeContractError::TooManyListValues);
                }
                for value in values {
                    value.validate_at_depth(depth.saturating_add(1))?;
                }
                Ok(())
            }
            Self::PreservedUnknown { type_name, value } => {
                validate_identifier("preserved unknown type", type_name)?;
                let encoded = serde_json::to_vec(value)
                    .map_err(NativeNodeContractError::EncodePreservedUnknown)?;
                if encoded.len() > MAX_UNKNOWN_BYTES {
                    return Err(NativeNodeContractError::PreservedUnknownTooLarge);
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum NativeValueType {
    Any,
    Primitive(NativePrimitiveType),
    Handle(NativeHandleType),
    PreservedUnknown,
    NamedPreservedUnknown(String),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NativeTypeUnion(Vec<NativeValueType>);

impl NativeTypeUnion {
    pub fn new(
        members: impl IntoIterator<Item = NativeValueType>,
    ) -> Result<Self, NativeNodeContractError> {
        let members = members.into_iter().collect::<Vec<_>>();
        let value = Self(members);
        value.validate()?;
        Ok(value)
    }

    pub fn members(&self) -> &[NativeValueType] {
        &self.0
    }

    pub fn accepts(&self, value: &NativeValue) -> bool {
        self.0.iter().any(|member| match (member, value) {
            (NativeValueType::Any, _) => true,
            (NativeValueType::Primitive(expected), NativeValue::Primitive { value }) => {
                *expected == value.primitive_type()
            }
            (NativeValueType::Handle(expected), NativeValue::Handle { value }) => {
                crate::native_handle_type_accepts(expected, value.handle_type())
            }
            (NativeValueType::PreservedUnknown, NativeValue::PreservedUnknown { .. }) => true,
            (
                NativeValueType::NamedPreservedUnknown(expected),
                NativeValue::PreservedUnknown { type_name, .. },
            ) => expected == type_name,
            (NativeValueType::PreservedUnknown, value) => {
                NativeStructuredValue::from_native_value(value)
                    .ok()
                    .flatten()
                    .is_some()
            }
            (NativeValueType::NamedPreservedUnknown(expected), value) => {
                NativeStructuredValue::from_native_value(value)
                    .ok()
                    .flatten()
                    .is_some_and(|structured| structured.type_name == *expected)
            }
            _ => false,
        })
    }

    pub fn validate(&self) -> Result<(), NativeNodeContractError> {
        if self.0.is_empty() || self.0.len() > MAX_TYPE_UNION_MEMBERS {
            return Err(NativeNodeContractError::InvalidTypeUnion);
        }
        if !self.0.windows(2).all(|pair| pair[0] < pair[1]) {
            return Err(NativeNodeContractError::InvalidTypeUnion);
        }
        if self.0.contains(&NativeValueType::Any) && self.0.len() != 1 {
            return Err(NativeNodeContractError::InvalidTypeUnion);
        }
        for member in &self.0 {
            if let NativeValueType::NamedPreservedUnknown(type_name) = member {
                validate_identifier("preserved unknown value type", type_name)?;
            }
        }
        Ok(())
    }
}

pub fn native_value_matches_input_schema(
    value: &NativeValue,
    schema: &NativeInputSchemaMetadata,
) -> bool {
    match value {
        NativeValue::List { values } => values
            .iter()
            .all(|value| native_value_matches_input_schema(value, schema)),
        _ => {
            let Some(schema_value) = native_value_as_schema_value(value) else {
                return schema.choices.is_empty()
                    && schema.minimum.is_none()
                    && schema.maximum.is_none();
            };
            if !schema.choices.is_empty()
                && !schema
                    .choices
                    .iter()
                    .any(|choice| schema_values_equal(&schema_value, choice))
            {
                return false;
            }
            schema.minimum.as_ref().is_none_or(|minimum| {
                // Preserved source expressions are provenance, not executable numeric bounds.
                matches!(minimum, NativeSchemaValue::PreservedExpression { .. })
                    || schema_number_at_least(&schema_value, minimum)
            }) && schema.maximum.as_ref().is_none_or(|maximum| {
                matches!(maximum, NativeSchemaValue::PreservedExpression { .. })
                    || schema_number_at_most(&schema_value, maximum)
            })
        }
    }
}

pub fn native_value_matches_source_input_schema(
    value: &NativeValue,
    class_type: &str,
    source_schema: &NativeDescriptorSchemaMetadata,
    input_schema: &NativeInputSchemaMetadata,
) -> bool {
    if class_type != "LoadLatent" || input_schema.name != "latent" {
        return native_value_matches_input_schema(value, input_schema);
    }
    // The original custom validator checks file existence and deliberately skips
    // UI choice membership. File authority stays with the checked asset service.
    if source_schema.node.definition_sha256.as_deref()
        != Some("18373aaddf2c80f458364c5859841ad2a47c651e6c69f873f95f3a3fe7133788")
        || !matches!(input_schema.choices.as_slice(),
            [NativeSchemaValue::PreservedExpression { sha256, .. }]
                if sha256 == "e6f894958f20ca497e3e68bdc90f605d70c2c2f40f4f21804fd84e093237cf4b")
    {
        return false;
    }
    let Ok(catalog) = crate::built_in_source_schema("LoadLatent") else {
        return false;
    };
    let Ok(expected) =
        catalog.bind_execution_ports(&["latent".to_owned()], &[], &["latent".to_owned()])
    else {
        return false;
    };
    if source_schema != &expected || expected.inputs.first() != Some(input_schema) {
        return false;
    }
    source_asset_filename_matches(value)
}

fn source_asset_filename_matches(value: &NativeValue) -> bool {
    match value {
        NativeValue::List { values } => values.iter().all(source_asset_filename_matches),
        NativeValue::PreservedUnknown { type_name, value } => {
            type_name == "COMBO" && value.is_string()
        }
        _ => false,
    }
}

fn schema_values_equal(left: &NativeSchemaValue, right: &NativeSchemaValue) -> bool {
    left == right
        || compare_schema_numbers(left, right)
            .is_some_and(|ordering| ordering == std::cmp::Ordering::Equal)
}

fn native_value_as_schema_value(value: &NativeValue) -> Option<NativeSchemaValue> {
    let NativeValue::Primitive { value } = value else {
        return None;
    };
    match value {
        NativePrimitive::Null => Some(NativeSchemaValue::Null),
        NativePrimitive::Boolean(value) => Some(NativeSchemaValue::Boolean { value: *value }),
        NativePrimitive::Integer(value) => Some(NativeSchemaValue::SignedInteger { value: *value }),
        NativePrimitive::UnsignedInteger(value) => {
            Some(NativeSchemaValue::UnsignedInteger { value: *value })
        }
        NativePrimitive::Number(value) if value.is_finite() => {
            Some(NativeSchemaValue::FiniteDecimal {
                value: value.to_string(),
            })
        }
        NativePrimitive::Number(_) => None,
        NativePrimitive::String(value) => Some(NativeSchemaValue::String {
            value: value.clone(),
        }),
    }
}

fn native_value_from_schema_value(value: &NativeSchemaValue) -> Option<NativeValue> {
    Some(match value {
        NativeSchemaValue::Null => NativeValue::Primitive {
            value: NativePrimitive::Null,
        },
        NativeSchemaValue::Boolean { value } => NativeValue::Primitive {
            value: NativePrimitive::Boolean(*value),
        },
        NativeSchemaValue::SignedInteger { value } => NativeValue::Primitive {
            value: NativePrimitive::Integer(*value),
        },
        NativeSchemaValue::UnsignedInteger { value } => NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(*value),
        },
        NativeSchemaValue::FiniteDecimal { value } => NativeValue::Primitive {
            value: NativePrimitive::Number(
                value
                    .parse::<f64>()
                    .ok()
                    .filter(|value| value.is_finite())?,
            ),
        },
        NativeSchemaValue::String { value } => NativeValue::Primitive {
            value: NativePrimitive::String(value.clone()),
        },
        NativeSchemaValue::List { values } => NativeValue::List {
            values: values
                .iter()
                .map(native_value_from_schema_value)
                .collect::<Option<Vec<_>>>()?,
        },
        NativeSchemaValue::Object { .. } | NativeSchemaValue::PreservedExpression { .. } => {
            return None;
        }
    })
}

fn native_type_union_accepts_value(union: &NativeTypeUnion, value: &NativeValue) -> bool {
    union.accepts(value)
        || union.members() == [NativeValueType::Any]
        || matches!(
            value,
            NativeValue::Primitive {
                value: NativePrimitive::Integer(_) | NativePrimitive::UnsignedInteger(_)
            }
        ) && union
            .members()
            .contains(&NativeValueType::Primitive(NativePrimitiveType::Number))
}

fn schema_number_at_least(value: &NativeSchemaValue, minimum: &NativeSchemaValue) -> bool {
    compare_schema_numbers(value, minimum).is_some_and(|ordering| !ordering.is_lt())
}

fn schema_number_at_most(value: &NativeSchemaValue, maximum: &NativeSchemaValue) -> bool {
    compare_schema_numbers(value, maximum).is_some_and(|ordering| !ordering.is_gt())
}

fn compare_schema_numbers(
    left: &NativeSchemaValue,
    right: &NativeSchemaValue,
) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering;
    match (left, right) {
        (
            NativeSchemaValue::SignedInteger { value: left },
            NativeSchemaValue::SignedInteger { value: right },
        ) => Some(left.cmp(right)),
        (
            NativeSchemaValue::UnsignedInteger { value: left },
            NativeSchemaValue::UnsignedInteger { value: right },
        ) => Some(left.cmp(right)),
        (
            NativeSchemaValue::SignedInteger { value: left },
            NativeSchemaValue::UnsignedInteger { value: right },
        ) => {
            if *left < 0 {
                Some(Ordering::Less)
            } else {
                Some((*left as u64).cmp(right))
            }
        }
        (
            NativeSchemaValue::UnsignedInteger { value: left },
            NativeSchemaValue::SignedInteger { value: right },
        ) => {
            if *right < 0 {
                Some(Ordering::Greater)
            } else {
                Some(left.cmp(&(*right as u64)))
            }
        }
        (NativeSchemaValue::FiniteDecimal { value: left }, right) => {
            compare_finite_decimal(left, right)
        }
        (left, NativeSchemaValue::FiniteDecimal { value: right }) => {
            compare_finite_decimal(right, left).map(Ordering::reverse)
        }
        _ => None,
    }
}

fn compare_finite_decimal(value: &str, other: &NativeSchemaValue) -> Option<std::cmp::Ordering> {
    const MAX_EXACT_F64_INTEGER: u64 = 1 << 53;
    let value = value
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())?;
    let other = match other {
        NativeSchemaValue::SignedInteger { value } => {
            if value.unsigned_abs() > MAX_EXACT_F64_INTEGER {
                return None;
            }
            *value as f64
        }
        NativeSchemaValue::UnsignedInteger { value } => {
            if *value > MAX_EXACT_F64_INTEGER {
                return None;
            }
            *value as f64
        }
        NativeSchemaValue::FiniteDecimal { value } => value
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite())?,
        _ => return None,
    };
    value.partial_cmp(&other)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativePortCardinality {
    Scalar,
    List,
    Mapped,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeInputDescriptor {
    pub name: String,
    pub accepted_types: NativeTypeUnion,
    pub required: bool,
    pub hidden: bool,
    pub lazy: bool,
    pub cardinality: NativePortCardinality,
    pub allows_literal: bool,
}

impl NativeInputDescriptor {
    fn validate(&self) -> Result<(), NativeNodeContractError> {
        validate_identifier("input name", &self.name)?;
        self.accepted_types.validate()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeDynamicInputDescriptor {
    pub name_template: String,
    pub start_index: u32,
    pub minimum_count: u32,
    pub maximum_count: u32,
    pub input: NativeInputDescriptor,
}

impl NativeDynamicInputDescriptor {
    fn validate(&self) -> Result<(), NativeNodeContractError> {
        validate_text(
            "dynamic input name template",
            &self.name_template,
            MAX_IDENTIFIER_BYTES,
            false,
        )?;
        let indexed = self.name_template.matches("{index}").count() == 1
            && !self.name_template.contains("{name}");
        let named = self.name_template == "{name}";
        if !(indexed || named) || self.minimum_count > self.maximum_count || self.maximum_count == 0
        {
            return Err(NativeNodeContractError::InvalidDynamicInput);
        }
        if indexed {
            self.start_index
                .checked_add(self.maximum_count)
                .ok_or(NativeNodeContractError::InvalidDynamicInput)?;
        }
        self.input.validate()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeOutputDescriptor {
    pub name: String,
    pub produced_type: NativeValueType,
    pub is_list: bool,
}

impl NativeOutputDescriptor {
    fn validate(&self) -> Result<(), NativeNodeContractError> {
        validate_identifier("output name", &self.name)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeEffectClass {
    Pure,
    ReadsArtifact,
    WritesArtifact,
    Provider,
    ExclusiveDevice,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeCachePolicy {
    InputIdentity,
    Never,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeNodeDescriptor {
    pub schema_version: u16,
    pub class_type: String,
    pub implementation_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_schema: Option<NativeDescriptorSchemaMetadata>,
    pub inputs: Vec<NativeInputDescriptor>,
    pub dynamic_inputs: Vec<NativeDynamicInputDescriptor>,
    pub outputs: Vec<NativeOutputDescriptor>,
    pub output_node: bool,
    pub effect: NativeEffectClass,
    pub cache: NativeCachePolicy,
}

impl NativeNodeDescriptor {
    pub fn validate(&self) -> Result<(), NativeNodeContractError> {
        match (self.schema_version, self.source_schema.as_ref()) {
            (LEGACY_NATIVE_NODE_CONTRACT_SCHEMA_VERSION, None) => {}
            (NATIVE_NODE_CONTRACT_SCHEMA_VERSION, Some(source_schema)) => source_schema
                .validate()
                .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?,
            (LEGACY_NATIVE_NODE_CONTRACT_SCHEMA_VERSION, Some(_))
            | (NATIVE_NODE_CONTRACT_SCHEMA_VERSION, None) => {
                return Err(NativeNodeContractError::InvalidSourceSchema(
                    "descriptor version and source metadata do not agree".to_owned(),
                ));
            }
            _ => {
                return Err(NativeNodeContractError::UnsupportedContractSchema(
                    self.schema_version,
                ));
            }
        }
        validate_identifier("node class type", &self.class_type)?;
        validate_identifier("node implementation version", &self.implementation_version)?;
        if self.inputs.len() > MAX_PORTS
            || self.dynamic_inputs.len() > MAX_PORTS
            || self.outputs.len() > MAX_PORTS
        {
            return Err(NativeNodeContractError::InvalidPortCount);
        }
        if let Some(source_schema) = &self.source_schema {
            if source_schema.inputs.len() != self.inputs.len()
                || source_schema.dynamic_inputs.len() != self.dynamic_inputs.len()
                || source_schema.outputs.len() != self.outputs.len()
                || !source_schema
                    .inputs
                    .iter()
                    .zip(&self.inputs)
                    .all(|(schema, input)| schema.name == input.name)
                || !source_schema
                    .outputs
                    .iter()
                    .zip(&self.outputs)
                    .all(|(schema, output)| schema.name == output.name)
                || !source_schema
                    .dynamic_inputs
                    .iter()
                    .zip(&self.dynamic_inputs)
                    .all(|(schema, input)| {
                        schema.identity == input.name_template
                            && schema.start_index == input.start_index
                            && schema.minimum_count == input.minimum_count
                            && schema.maximum_count == input.maximum_count
                            && schema.input.name == input.input.name
                            && (input.name_template != "{name}"
                                || (schema.names.len() >= input.minimum_count as usize
                                    && schema.names.len() <= input.maximum_count as usize))
                    })
            {
                return Err(NativeNodeContractError::InvalidSourceSchema(
                    "schema port order does not match the execution descriptor".to_owned(),
                ));
            }
        }
        let mut input_names = BTreeSet::new();
        for input in &self.inputs {
            input.validate()?;
            if !input_names.insert(input.name.as_str()) {
                return Err(NativeNodeContractError::DuplicatePort(input.name.clone()));
            }
        }
        if let Some(source_schema) = &self.source_schema {
            if matches!(
                source_schema.node.provenance,
                crate::NativeSchemaProvenance::SourceV1 | crate::NativeSchemaProvenance::SourceV3
            ) {
                for (input, schema) in self.inputs.iter().zip(&source_schema.inputs) {
                    let expected =
                        crate::native_value_types_for_input_schema(schema).map_err(|error| {
                            NativeNodeContractError::InvalidSourceSchema(error.to_string())
                        })?;
                    if input.accepted_types != expected {
                        return Err(NativeNodeContractError::InvalidSourceSchema(format!(
                            "execution type for input `{}` does not match its source schema",
                            input.name
                        )));
                    }
                }
                for (input, schema) in self
                    .dynamic_inputs
                    .iter()
                    .zip(&source_schema.dynamic_inputs)
                {
                    let expected = crate::native_value_types_for_input_schema(&schema.input)
                        .map_err(|error| {
                            NativeNodeContractError::InvalidSourceSchema(error.to_string())
                        })?;
                    if input.input.accepted_types != expected {
                        return Err(NativeNodeContractError::InvalidSourceSchema(format!(
                            "execution type for dynamic input `{}` does not match its source schema",
                            input.name_template
                        )));
                    }
                }
                for (output, schema) in self.outputs.iter().zip(&source_schema.outputs) {
                    let expected =
                        crate::native_value_type_for_output_schema(schema).map_err(|error| {
                            NativeNodeContractError::InvalidSourceSchema(error.to_string())
                        })?;
                    if output.produced_type != expected {
                        return Err(NativeNodeContractError::InvalidSourceSchema(format!(
                            "execution type for output `{}` does not match its source schema",
                            output.name
                        )));
                    }
                }
            }
            for (input, schema) in self.inputs.iter().zip(&source_schema.inputs) {
                if let Some(default) = schema
                    .default
                    .as_ref()
                    .and_then(native_value_from_schema_value)
                    && (!native_type_union_accepts_value(&input.accepted_types, &default)
                        || !native_value_matches_input_schema(&default, schema))
                {
                    return Err(NativeNodeContractError::InvalidSourceSchema(format!(
                        "default for input `{}` is incompatible with its execution type or constraints",
                        input.name
                    )));
                }
            }
        }
        let mut templates = BTreeSet::new();
        for input in &self.dynamic_inputs {
            input.validate()?;
            if !templates.insert(input.name_template.as_str()) {
                return Err(NativeNodeContractError::DuplicatePort(
                    input.name_template.clone(),
                ));
            }
        }
        let mut output_names = BTreeSet::new();
        for output in &self.outputs {
            output.validate()?;
            if !output_names.insert(output.name.as_str()) {
                return Err(NativeNodeContractError::DuplicatePort(output.name.clone()));
            }
        }
        Ok(())
    }

    pub fn validate_exact_schema_v2(&self) -> Result<(), NativeNodeContractError> {
        self.validate()?;
        if self.schema_version != NATIVE_NODE_CONTRACT_SCHEMA_VERSION
            || self.source_schema.is_none()
        {
            return Err(NativeNodeContractError::InvalidSourceSchema(
                "an exact schema-v2 descriptor is required".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeNodePresentation {
    pub display_name: String,
    pub category: String,
    pub description: String,
    pub output_names: Vec<String>,
    pub search_aliases: Vec<String>,
    pub is_deprecated: bool,
    pub is_experimental: bool,
}

impl NativeNodePresentation {
    pub fn validate(&self) -> Result<(), NativeNodeContractError> {
        validate_text(
            "node display name",
            &self.display_name,
            MAX_IDENTIFIER_BYTES,
            false,
        )?;
        validate_text("node category", &self.category, MAX_IDENTIFIER_BYTES, true)?;
        validate_workflow_text("node description", &self.description)?;
        if self.output_names.len() > MAX_PORTS {
            return Err(NativeNodeContractError::InvalidPresentationOutputs);
        }
        let mut output_names = BTreeSet::new();
        for output_name in &self.output_names {
            validate_identifier("node presentation output name", output_name)?;
            if !output_names.insert(output_name.as_str()) {
                return Err(NativeNodeContractError::InvalidPresentationOutputs);
            }
        }
        let mut aliases = BTreeSet::new();
        for alias in &self.search_aliases {
            validate_text("node search alias", alias, MAX_IDENTIFIER_BYTES, false)?;
            if !aliases.insert(alias.as_str()) {
                return Err(NativeNodeContractError::DuplicateSearchAlias(alias.clone()));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum NativeHandleStoreError {
    #[error("native handle operation was cancelled")]
    Cancelled,
    #[error("native handle belongs to a different store")]
    WrongStore,
    #[error("native handle belongs to a different store generation")]
    WrongGeneration,
    #[error("native handle type `{actual}` does not match `{expected}`")]
    WrongType { expected: String, actual: String },
    #[error("native handle `{0}` is absent")]
    Missing(String),
    #[error("native handle digest does not match the stored object")]
    DigestMismatch,
    #[error("native handle store rejected the operation: {0}")]
    Rejected(String),
    #[error("native handle contract is invalid: {0}")]
    InvalidHandle(#[from] NativeNodeContractError),
    #[error("native stored payload is invalid: {0}")]
    InvalidPayload(#[source] Box<NativeStoredPayloadError>),
}

impl From<NativeStoredPayloadError> for NativeHandleStoreError {
    fn from(error: NativeStoredPayloadError) -> Self {
        Self::InvalidPayload(Box::new(error))
    }
}

pub trait NativeResolvedPayloadRetention: fmt::Debug + Send + Sync {}

#[derive(Clone)]
pub struct NativeResolvedPayload {
    payload: Arc<NativeStoredPayload>,
    _retention: Arc<dyn NativeResolvedPayloadRetention>,
}

impl NativeResolvedPayload {
    pub fn checked(
        payload: Arc<NativeStoredPayload>,
        retention: Arc<dyn NativeResolvedPayloadRetention>,
    ) -> Result<Self, NativeStoredPayloadError> {
        payload.validate()?;
        Ok(Self {
            payload,
            _retention: retention,
        })
    }
}

impl AsRef<NativeStoredPayload> for NativeResolvedPayload {
    fn as_ref(&self) -> &NativeStoredPayload {
        self.payload.as_ref()
    }
}

impl std::ops::Deref for NativeResolvedPayload {
    type Target = NativeStoredPayload;

    fn deref(&self) -> &Self::Target {
        self.payload.as_ref()
    }
}

impl fmt::Debug for NativeResolvedPayload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeResolvedPayload")
            .finish_non_exhaustive()
    }
}

pub trait NativeHandleStore: Send + Sync + fmt::Debug {
    fn identity(&self) -> NativeHandleStoreIdentity;
    fn attempt_id(&self) -> AttemptId;

    fn resolve(
        &self,
        handle: &NativeOpaqueHandle,
        expected_type: &NativeHandleType,
        cancellation: &CancellationToken,
    ) -> Result<NativeResolvedPayload, NativeHandleStoreError>;

    fn publish(
        &self,
        payload: NativeStoredPayload,
        cancellation: &CancellationToken,
    ) -> Result<NativeOpaqueHandle, NativeHandleStoreError>;

    fn revoke(
        &self,
        handle: &NativeOpaqueHandle,
        cancellation: &CancellationToken,
    ) -> Result<(), NativeHandleStoreError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeNodeServiceIdentity {
    service_id: Uuid,
    attempt_id: AttemptId,
    node_id: NodeId,
}

impl NativeNodeServiceIdentity {
    pub fn checked(
        service_id: Uuid,
        attempt_id: AttemptId,
        node_id: NodeId,
    ) -> Result<Self, NativeNodeContractError> {
        if service_id.is_nil() || attempt_id.0.is_nil() {
            return Err(NativeNodeContractError::InvalidNodeServiceIdentity);
        }
        validate_identifier("native node service node ID", &node_id.0)?;
        Ok(Self {
            service_id,
            attempt_id,
            node_id,
        })
    }

    pub const fn service_id(&self) -> Uuid {
        self.service_id
    }

    pub const fn attempt_id(&self) -> AttemptId {
        self.attempt_id
    }

    pub const fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    fn matches(&self, attempt_id: AttemptId, node_id: &NodeId) -> bool {
        self.attempt_id == attempt_id && &self.node_id == node_id
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeAssetReference {
    service_id: Uuid,
    reference_id: Uuid,
    source_type_id: String,
    byte_length: u64,
    sha256: String,
}

impl NativeAssetReference {
    pub fn checked(
        service_id: Uuid,
        reference_id: Uuid,
        source_type_id: impl Into<String>,
        byte_length: u64,
        sha256: impl Into<String>,
    ) -> Result<Self, NativeAssetServiceError> {
        let source_type_id = source_type_id.into();
        let sha256 = sha256.into();
        if service_id.is_nil()
            || reference_id.is_nil()
            || byte_length == 0
            || byte_length > 2 * 1024 * 1024 * 1024
            || validate_identifier("native asset source type", &source_type_id).is_err()
            || !valid_sha256(&sha256)
        {
            return Err(NativeAssetServiceError::InvalidReference);
        }
        Ok(Self {
            service_id,
            reference_id,
            source_type_id,
            byte_length,
            sha256,
        })
    }

    pub const fn service_id(&self) -> Uuid {
        self.service_id
    }

    pub const fn reference_id(&self) -> Uuid {
        self.reference_id
    }

    pub fn source_type_id(&self) -> &str {
        &self.source_type_id
    }

    pub const fn byte_length(&self) -> u64 {
        self.byte_length
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeAssetReadRequest {
    reference: NativeAssetReference,
    maximum_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeAssetNameListRequest {
    folder_category: String,
}

impl NativeAssetNameListRequest {
    pub fn checked(folder_category: impl Into<String>) -> Result<Self, NativeAssetServiceError> {
        let folder_category = folder_category.into();
        validate_native_asset_folder_category(&folder_category)?;
        Ok(Self { folder_category })
    }

    pub fn folder_category(&self) -> &str {
        &self.folder_category
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeAssetNameList {
    folder_category: String,
    names: Vec<String>,
}

impl NativeAssetNameList {
    pub fn checked(
        folder_category: impl Into<String>,
        names: Vec<String>,
    ) -> Result<Self, NativeAssetServiceError> {
        let folder_category = folder_category.into();
        validate_native_asset_folder_category(&folder_category)?;
        if names.len() > MAX_NATIVE_ASSET_NAME_LIST_ENTRIES {
            return Err(NativeAssetServiceError::TooLarge);
        }
        let mut previous = None;
        for name in &names {
            validate_native_asset_name(name)?;
            if previous.is_some_and(|previous: &String| previous >= name) {
                return Err(NativeAssetServiceError::InvalidRequest);
            }
            previous = Some(name);
        }
        Ok(Self {
            folder_category,
            names,
        })
    }

    pub fn folder_category(&self) -> &str {
        &self.folder_category
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeAssetNameResolutionRequest {
    folder_category: String,
    names: Vec<String>,
    source_type_id: String,
}

impl NativeAssetNameResolutionRequest {
    pub fn checked(
        folder_category: impl Into<String>,
        names: Vec<String>,
        source_type_id: impl Into<String>,
    ) -> Result<Self, NativeAssetServiceError> {
        let folder_category = folder_category.into();
        let source_type_id = source_type_id.into();
        validate_native_asset_folder_category(&folder_category)?;
        if names.is_empty() || names.len() > MAX_NATIVE_ASSET_NAME_SELECTIONS {
            return Err(NativeAssetServiceError::InvalidRequest);
        }
        for name in &names {
            validate_native_asset_name(name)?;
        }
        if validate_identifier("native asset source type", &source_type_id).is_err() {
            return Err(NativeAssetServiceError::InvalidRequest);
        }
        Ok(Self {
            folder_category,
            names,
            source_type_id,
        })
    }

    pub fn folder_category(&self) -> &str {
        &self.folder_category
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn source_type_id(&self) -> &str {
        &self.source_type_id
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeNamedAssetReference {
    name: String,
    reference: NativeAssetReference,
}

impl NativeNamedAssetReference {
    pub fn checked(
        name: impl Into<String>,
        reference: NativeAssetReference,
    ) -> Result<Self, NativeAssetServiceError> {
        let name = name.into();
        validate_native_asset_name(&name)?;
        Ok(Self { name, reference })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn reference(&self) -> &NativeAssetReference {
        &self.reference
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeAssetNameResolution {
    folder_category: String,
    assets: Vec<NativeNamedAssetReference>,
}

impl NativeAssetNameResolution {
    pub fn checked(
        folder_category: impl Into<String>,
        assets: Vec<NativeNamedAssetReference>,
    ) -> Result<Self, NativeAssetServiceError> {
        let folder_category = folder_category.into();
        validate_native_asset_folder_category(&folder_category)?;
        if assets.is_empty() || assets.len() > MAX_NATIVE_ASSET_NAME_SELECTIONS {
            return Err(NativeAssetServiceError::InvalidRequest);
        }
        Ok(Self {
            folder_category,
            assets,
        })
    }

    pub fn folder_category(&self) -> &str {
        &self.folder_category
    }

    pub fn assets(&self) -> &[NativeNamedAssetReference] {
        &self.assets
    }
}

fn validate_native_asset_folder_category(
    folder_category: &str,
) -> Result<(), NativeAssetServiceError> {
    if folder_category.is_empty()
        || folder_category.len() > MAX_NATIVE_ASSET_FOLDER_CATEGORY_BYTES
        || !folder_category
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    {
        return Err(NativeAssetServiceError::InvalidFolderCategory);
    }
    Ok(())
}

fn validate_native_asset_name(name: &str) -> Result<(), NativeAssetServiceError> {
    if name.is_empty()
        || name.len() > MAX_NATIVE_ASSET_NAME_BYTES
        || name.starts_with('/')
        || name.ends_with('/')
        || name.contains('\\')
        || name.chars().any(char::is_control)
        || name
            .split('/')
            .any(|component| component.is_empty() || matches!(component, "." | ".."))
    {
        return Err(NativeAssetServiceError::InvalidAssetName);
    }
    Ok(())
}

impl NativeAssetReadRequest {
    pub fn checked(
        reference: NativeAssetReference,
        maximum_bytes: u64,
    ) -> Result<Self, NativeAssetServiceError> {
        if maximum_bytes == 0
            || maximum_bytes > 2 * 1024 * 1024 * 1024
            || reference.byte_length() > maximum_bytes
        {
            return Err(NativeAssetServiceError::InvalidRequest);
        }
        Ok(Self {
            reference,
            maximum_bytes,
        })
    }

    pub const fn reference(&self) -> &NativeAssetReference {
        &self.reference
    }

    pub const fn maximum_bytes(&self) -> u64 {
        self.maximum_bytes
    }
}

#[derive(Clone, Debug)]
pub struct NativeResolvedAsset {
    reference: NativeAssetReference,
    bytes: Arc<[u8]>,
    byte_length: u64,
    sha256: String,
}

impl NativeResolvedAsset {
    pub fn checked(
        reference: NativeAssetReference,
        bytes: Arc<[u8]>,
        sha256: impl Into<String>,
    ) -> Result<Self, NativeAssetServiceError> {
        let sha256 = sha256.into();
        let byte_length =
            u64::try_from(bytes.len()).map_err(|_| NativeAssetServiceError::TooLarge)?;
        if byte_length > 2 * 1024 * 1024 * 1024
            || byte_length != reference.byte_length()
            || !valid_sha256(&sha256)
            || sha256 != reference.sha256()
            || sha256 != format!("{:x}", Sha256::digest(&bytes))
        {
            return Err(NativeAssetServiceError::DigestMismatch);
        }
        Ok(Self {
            reference,
            bytes,
            byte_length,
            sha256,
        })
    }

    pub const fn reference(&self) -> &NativeAssetReference {
        &self.reference
    }

    pub fn bytes(&self) -> &Arc<[u8]> {
        &self.bytes
    }

    pub const fn byte_length(&self) -> u64 {
        self.byte_length
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum NativeAssetServiceError {
    #[error("native asset service is unavailable")]
    Unavailable,
    #[error("native asset operation was cancelled")]
    Cancelled,
    #[error("native asset reference is invalid")]
    InvalidReference,
    #[error("native asset read request is invalid")]
    InvalidRequest,
    #[error("native asset folder category is invalid or unsupported")]
    InvalidFolderCategory,
    #[error("native asset name is invalid")]
    InvalidAssetName,
    #[error("native asset permission was denied")]
    PermissionDenied,
    #[error("native asset is missing")]
    Missing,
    #[error("native asset exceeds the authorized byte limit")]
    TooLarge,
    #[error("native asset digest changed or did not match")]
    DigestMismatch,
    #[error("native asset changed during the verified read")]
    ChangedDuringRead,
    #[error("native asset service rejected the operation")]
    Rejected,
}

pub trait NativeAssetResolver: Send + Sync + fmt::Debug {
    fn identity(&self) -> &NativeNodeServiceIdentity;

    fn list_names(
        &self,
        request: &NativeAssetNameListRequest,
        cancellation: &CancellationToken,
    ) -> Result<NativeAssetNameList, NativeAssetServiceError>;

    fn resolve_names(
        &self,
        request: &NativeAssetNameResolutionRequest,
        cancellation: &CancellationToken,
    ) -> Result<NativeAssetNameResolution, NativeAssetServiceError>;

    fn read_verified(
        &self,
        request: &NativeAssetReadRequest,
        cancellation: &CancellationToken,
    ) -> Result<NativeResolvedAsset, NativeAssetServiceError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeOutputNamespace {
    Output,
    Temporary,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeOutputShape {
    File,
    Image { width: u32, height: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeOutputMediaKind {
    Image,
    Animation,
    Video,
    Audio,
    ThreeD,
    Text,
    Json,
    Binary,
}

impl NativeOutputMediaKind {
    pub fn accepts(self, extension: &str, media_type: &str) -> bool {
        matches!(
            (self, extension, media_type),
            (Self::Image, "png", "image/png")
                | (Self::Image, "jpg" | "jpeg", "image/jpeg")
                | (Self::Image, "webp", "image/webp")
                | (Self::Animation, "gif", "image/gif")
                | (Self::Video, "webm", "video/webm")
                | (Self::Video, "mp4", "video/mp4")
                | (Self::Audio, "wav", "audio/wav")
                | (Self::Audio, "mp3", "audio/mpeg")
                | (Self::ThreeD, "glb", "model/gltf-binary")
                | (Self::ThreeD, "gltf", "model/gltf+json")
                | (Self::Text, "txt", "text/plain")
                | (Self::Json, "json", "application/json")
                | (Self::Binary, _, "application/octet-stream")
        )
    }
}

#[derive(Clone, Debug)]
pub struct NativeOutputEffectRequest {
    namespace: NativeOutputNamespace,
    filename_prefix: String,
    extension: String,
    batch_index: u32,
    shape: NativeOutputShape,
    media_kind: NativeOutputMediaKind,
    media_type: String,
    content: Arc<[u8]>,
    request_digest_sha256: String,
}

impl NativeOutputEffectRequest {
    pub fn checked(
        namespace: NativeOutputNamespace,
        filename_prefix: impl Into<String>,
        extension: impl Into<String>,
        batch_index: u32,
        shape: NativeOutputShape,
        content: Arc<[u8]>,
        maximum_bytes: u64,
    ) -> Result<Self, NativeEffectServiceError> {
        let extension = extension.into();
        let (media_kind, media_type) = default_output_media(&extension);
        Self::checked_media(
            namespace,
            filename_prefix,
            extension,
            media_type,
            media_kind,
            batch_index,
            shape,
            content,
            maximum_bytes,
        )
    }

    pub fn checked_media(
        namespace: NativeOutputNamespace,
        filename_prefix: impl Into<String>,
        extension: impl Into<String>,
        media_type: impl Into<String>,
        media_kind: NativeOutputMediaKind,
        batch_index: u32,
        shape: NativeOutputShape,
        content: Arc<[u8]>,
        maximum_bytes: u64,
    ) -> Result<Self, NativeEffectServiceError> {
        let filename_prefix = filename_prefix.into();
        let extension = extension.into();
        let media_type = media_type.into();
        if !is_safe_output_filename_prefix(&filename_prefix)
            || extension.is_empty()
            || extension.len() > 32
            || !extension
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
            || !media_kind.accepts(&extension, &media_type)
            || content.is_empty()
            || u64::try_from(content.len()).map_or(true, |length| {
                length > maximum_bytes || length > 2 * 1024 * 1024 * 1024
            })
            || matches!(shape, NativeOutputShape::Image { width: 0, .. })
            || matches!(shape, NativeOutputShape::Image { height: 0, .. })
        {
            return Err(NativeEffectServiceError::InvalidRequest);
        }
        let request_digest_sha256 = output_request_digest(
            namespace,
            &filename_prefix,
            &extension,
            batch_index,
            shape,
            media_kind,
            &media_type,
            &content,
        );
        Ok(Self {
            namespace,
            filename_prefix,
            extension,
            batch_index,
            shape,
            media_kind,
            media_type,
            content,
            request_digest_sha256,
        })
    }

    pub const fn namespace(&self) -> NativeOutputNamespace {
        self.namespace
    }

    pub fn filename_prefix(&self) -> &str {
        &self.filename_prefix
    }

    pub fn extension(&self) -> &str {
        &self.extension
    }

    pub const fn batch_index(&self) -> u32 {
        self.batch_index
    }

    pub const fn shape(&self) -> NativeOutputShape {
        self.shape
    }

    pub const fn media_kind(&self) -> NativeOutputMediaKind {
        self.media_kind
    }

    pub fn media_type(&self) -> &str {
        &self.media_type
    }

    pub fn content(&self) -> &Arc<[u8]> {
        &self.content
    }

    pub fn request_digest_sha256(&self) -> &str {
        &self.request_digest_sha256
    }
}

fn default_output_media(extension: &str) -> (NativeOutputMediaKind, String) {
    let (kind, media_type) = match extension {
        "png" => (NativeOutputMediaKind::Image, "image/png"),
        "jpg" | "jpeg" => (NativeOutputMediaKind::Image, "image/jpeg"),
        "webp" => (NativeOutputMediaKind::Image, "image/webp"),
        "gif" => (NativeOutputMediaKind::Animation, "image/gif"),
        "webm" => (NativeOutputMediaKind::Video, "video/webm"),
        "mp4" => (NativeOutputMediaKind::Video, "video/mp4"),
        "wav" => (NativeOutputMediaKind::Audio, "audio/wav"),
        "mp3" => (NativeOutputMediaKind::Audio, "audio/mpeg"),
        "glb" => (NativeOutputMediaKind::ThreeD, "model/gltf-binary"),
        "gltf" => (NativeOutputMediaKind::ThreeD, "model/gltf+json"),
        "json" => (NativeOutputMediaKind::Json, "application/json"),
        "txt" => (NativeOutputMediaKind::Text, "text/plain"),
        _ => (NativeOutputMediaKind::Binary, "application/octet-stream"),
    };
    (kind, media_type.to_owned())
}

fn is_safe_output_filename_prefix(filename_prefix: &str) -> bool {
    if filename_prefix.is_empty()
        || filename_prefix.len() > MAX_IDENTIFIER_BYTES
        || filename_prefix
            .bytes()
            .any(|byte| byte.is_ascii_control() || matches!(byte, b'\\' | b':'))
    {
        return false;
    }
    filename_prefix
        .split('/')
        .all(|segment| !segment.is_empty() && !matches!(segment, "." | ".."))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativePreparedEffectKind {
    Output,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum NativeEffectServiceError {
    #[error("native prepared-effect service is unavailable")]
    Unavailable,
    #[error("native prepared-effect operation was cancelled")]
    Cancelled,
    #[error("native output effect request is invalid")]
    InvalidRequest,
    #[error("native prepared-effect service rejected the operation")]
    Rejected,
    #[error("native effect ticket is duplicated or belongs to another node attempt")]
    InvalidTicket,
}

#[derive(Debug, Error)]
pub enum NativeImagePreviewError {
    #[error("native image preview compute session is invalid: {0}")]
    Contract(#[from] NativeNodeContractError),
    #[error("native image preview tensor is invalid: {0}")]
    Tensor(#[from] TensorError),
    #[error("native image preview PNG encoding failed: {0}")]
    Png(#[from] PngError),
    #[error("native image preview effect preparation failed: {0}")]
    Effect(#[from] NativeEffectServiceError),
    #[error("native image preview dimensions or batch index exceeded the output contract")]
    DimensionOverflow,
}

#[derive(Clone, Debug)]
pub struct NativePreparedImagePreview {
    effects: Vec<NativePreparedEffectRequest>,
    ui: Value,
}

impl NativePreparedImagePreview {
    pub fn effects(&self) -> &[NativePreparedEffectRequest] {
        &self.effects
    }

    pub fn ui(&self) -> &Value {
        &self.ui
    }

    pub fn into_parts(self) -> (Vec<NativePreparedEffectRequest>, Value) {
        (self.effects, self.ui)
    }
}

pub trait NativePreparedEffectService: Send + Sync + fmt::Debug {
    fn identity(&self) -> &NativeNodeServiceIdentity;
    fn maximum_output_bytes(&self) -> u64;

    fn metadata_enabled(&self) -> Result<bool, NativeEffectServiceError> {
        Err(NativeEffectServiceError::Unavailable)
    }

    fn prepare_output(
        &self,
        request: NativeOutputEffectRequest,
        cancellation: &CancellationToken,
    ) -> Result<NativePreparedEffectRequest, NativeEffectServiceError>;

    fn rollback_prepared(
        &self,
        request: &NativePreparedEffectRequest,
    ) -> Result<(), NativeEffectServiceError>;

    fn rollback_all_prepared(&self) -> Result<(), NativeEffectServiceError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeLtxvPreprocessServiceIdentity {
    configuration_sha256: String,
}

impl NativeLtxvPreprocessServiceIdentity {
    pub fn checked(
        configuration_sha256: impl Into<String>,
    ) -> Result<Self, NativeNodeContractError> {
        let configuration_sha256 = configuration_sha256.into();
        if !valid_sha256(&configuration_sha256) {
            return Err(NativeNodeContractError::InvalidNodeServiceIdentity);
        }
        Ok(Self {
            configuration_sha256,
        })
    }

    pub fn configuration_sha256(&self) -> &str {
        &self.configuration_sha256
    }
}

#[derive(Debug, Error)]
pub enum NativeLtxvPreprocessServiceError {
    #[error("native LTXV preprocessing service is unavailable")]
    Unavailable,
    #[error("native LTXV preprocessing request was cancelled")]
    Cancelled,
    #[error("native LTXV preprocessing service is busy")]
    Busy,
    #[error("native LTXV preprocessing request is invalid")]
    InvalidRequest,
    #[error("native LTXV preprocessing exhausted its reviewed resources")]
    ResourceExhausted,
    #[error("native LTXV preprocessing failed: {0}")]
    Execution(String),
}

pub trait NativeLtxvPreprocessService: Send + Sync + fmt::Debug {
    fn identity(&self) -> &NativeLtxvPreprocessServiceIdentity;

    fn preprocess_image(
        &self,
        image: &ImageTensor,
        compression: u8,
        context: &ExecutionContext<'_>,
    ) -> BoxFuture<'static, Result<ImageTensor, NativeLtxvPreprocessServiceError>>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeWebmEncodeServiceIdentity {
    configuration_sha256: String,
}

impl NativeWebmEncodeServiceIdentity {
    pub fn checked(
        configuration_sha256: impl Into<String>,
    ) -> Result<Self, NativeNodeContractError> {
        let configuration_sha256 = configuration_sha256.into();
        if !valid_sha256(&configuration_sha256) {
            return Err(NativeNodeContractError::InvalidNodeServiceIdentity);
        }
        Ok(Self {
            configuration_sha256,
        })
    }

    pub fn configuration_sha256(&self) -> &str {
        &self.configuration_sha256
    }
}

#[derive(Debug, Error)]
pub enum NativeWebmEncodeServiceError {
    #[error("native WebM encoding service is unavailable")]
    Unavailable,
    #[error("native WebM encoding request was cancelled")]
    Cancelled,
    #[error("native WebM encoding service is busy")]
    Busy,
    #[error("native WebM encoding request is invalid")]
    InvalidRequest,
    #[error("native WebM encoding exhausted its reviewed resources")]
    ResourceExhausted,
    #[error("native WebM encoder returned an invalid result projection")]
    InvalidProjection,
    #[error("native WebM encoding failed: {0}")]
    Execution(String),
}

#[derive(Clone, Debug)]
pub struct NativeWebmEncodeRequest {
    images: ImageTensor,
    codec: NativeVideoCodec,
    frame_rate: (u64, u64),
    crf: NativeVideoCrf,
    metadata: Vec<(String, String)>,
}

impl NativeWebmEncodeRequest {
    pub fn checked(
        images: ImageTensor,
        codec: NativeVideoCodec,
        frame_rate: (u64, u64),
        crf: NativeVideoCrf,
        metadata: Vec<(String, String)>,
    ) -> Result<Self, NativeWebmEncodeServiceError> {
        let (frame_count, height, width, channels) = images
            .dimensions()
            .map_err(|_| NativeWebmEncodeServiceError::InvalidRequest)?;
        if codec == NativeVideoCodec::H264
            || frame_count == 0
            || height == 0
            || width == 0
            || !matches!(channels, 3 | 4)
            || frame_rate.0 == 0
            || frame_rate.1 == 0
            || frame_rate.0 > i32::MAX as u64
            || frame_rate.1 > i32::MAX as u64
            || gcd_u64(frame_rate.0, frame_rate.1) != 1
            || metadata
                .iter()
                .any(|(key, value)| key.is_empty() || key.contains('\0') || value.contains('\0'))
        {
            return Err(NativeWebmEncodeServiceError::InvalidRequest);
        }
        Ok(Self {
            images,
            codec,
            frame_rate,
            crf,
            metadata,
        })
    }

    pub fn images(&self) -> &ImageTensor {
        &self.images
    }

    pub const fn codec(&self) -> NativeVideoCodec {
        self.codec
    }

    pub const fn frame_rate(&self) -> (u64, u64) {
        self.frame_rate
    }

    pub const fn crf(&self) -> NativeVideoCrf {
        self.crf
    }

    pub fn metadata(&self) -> &[(String, String)] {
        &self.metadata
    }

    pub fn into_parts(
        self,
    ) -> (
        ImageTensor,
        NativeVideoCodec,
        (u64, u64),
        NativeVideoCrf,
        Vec<(String, String)>,
    ) {
        (
            self.images,
            self.codec,
            self.frame_rate,
            self.crf,
            self.metadata,
        )
    }
}

#[derive(Debug)]
pub struct NativeEncodedWebm {
    bytes: Tensor,
    content_sha256: [u8; 32],
    codec: NativeVideoCodec,
    dimensions: (u64, u64),
    frame_rate: (u64, u64),
    frame_count: u64,
    pixel_format: NativeVideoPixelFormat,
    bit_depth: NativeVideoBitDepth,
    has_alpha: bool,
}

impl NativeEncodedWebm {
    #[allow(clippy::too_many_arguments)]
    pub fn checked(
        bytes: Tensor,
        content_sha256: [u8; 32],
        codec: NativeVideoCodec,
        dimensions: (u64, u64),
        frame_rate: (u64, u64),
        frame_count: u64,
        pixel_format: NativeVideoPixelFormat,
        bit_depth: NativeVideoBitDepth,
        has_alpha: bool,
    ) -> Result<Self, NativeWebmEncodeServiceError> {
        let descriptor = bytes.descriptor();
        let actual_content_sha256: [u8; 32] = Sha256::digest(
            bytes
                .contiguous_bytes()
                .map_err(|_| NativeWebmEncodeServiceError::InvalidProjection)?,
        )
        .into();
        let projection_is_valid = descriptor.dtype() == DType::U8
            && descriptor.device() == DeviceId::CPU
            && descriptor.shape().len() == 1
            && descriptor.shape().first().copied().unwrap_or(0) > 0
            && descriptor
                .is_contiguous()
                .map_err(|_| NativeWebmEncodeServiceError::InvalidProjection)?
            && dimensions.0 > 0
            && dimensions.1 > 0
            && frame_rate.0 > 0
            && frame_rate.1 > 0
            && gcd_u64(frame_rate.0, frame_rate.1) == 1
            && frame_count > 0
            && actual_content_sha256 == content_sha256
            && match codec {
                NativeVideoCodec::Vp9 => {
                    bit_depth == NativeVideoBitDepth::Eight
                        && matches!(
                            (pixel_format, has_alpha),
                            (NativeVideoPixelFormat::Yuv420p, false)
                                | (NativeVideoPixelFormat::Yuva420p, true)
                        )
                }
                NativeVideoCodec::Av1 => {
                    bit_depth == NativeVideoBitDepth::Ten
                        && pixel_format == NativeVideoPixelFormat::Yuv420p10le
                        && !has_alpha
                }
                NativeVideoCodec::H264 => false,
            };
        if !projection_is_valid {
            return Err(NativeWebmEncodeServiceError::InvalidProjection);
        }
        Ok(Self {
            bytes,
            content_sha256,
            codec,
            dimensions,
            frame_rate,
            frame_count,
            pixel_format,
            bit_depth,
            has_alpha,
        })
    }

    pub fn bytes(&self) -> &Tensor {
        &self.bytes
    }

    pub const fn content_sha256(&self) -> [u8; 32] {
        self.content_sha256
    }

    pub const fn codec(&self) -> NativeVideoCodec {
        self.codec
    }

    pub const fn dimensions(&self) -> (u64, u64) {
        self.dimensions
    }

    pub const fn frame_rate(&self) -> (u64, u64) {
        self.frame_rate
    }

    pub const fn frame_count(&self) -> u64 {
        self.frame_count
    }

    pub const fn pixel_format(&self) -> NativeVideoPixelFormat {
        self.pixel_format
    }

    pub const fn bit_depth(&self) -> NativeVideoBitDepth {
        self.bit_depth
    }

    pub const fn has_alpha(&self) -> bool {
        self.has_alpha
    }

    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        Tensor,
        [u8; 32],
        NativeVideoCodec,
        (u64, u64),
        (u64, u64),
        u64,
        NativeVideoPixelFormat,
        NativeVideoBitDepth,
        bool,
    ) {
        (
            self.bytes,
            self.content_sha256,
            self.codec,
            self.dimensions,
            self.frame_rate,
            self.frame_count,
            self.pixel_format,
            self.bit_depth,
            self.has_alpha,
        )
    }
}

pub trait NativeWebmEncodeService: Send + Sync + fmt::Debug {
    fn identity(&self) -> &NativeWebmEncodeServiceIdentity;

    #[allow(clippy::too_many_arguments)]
    fn encode_webm(
        &self,
        request: NativeWebmEncodeRequest,
        context: &ExecutionContext<'_>,
    ) -> BoxFuture<'static, Result<NativeEncodedWebm, NativeWebmEncodeServiceError>>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeComponentH264Mp4BackingServiceIdentity {
    configuration_sha256: String,
}

impl NativeComponentH264Mp4BackingServiceIdentity {
    pub fn checked(
        configuration_sha256: impl Into<String>,
    ) -> Result<Self, NativeNodeContractError> {
        let configuration_sha256 = configuration_sha256.into();
        if !valid_sha256(&configuration_sha256) {
            return Err(NativeNodeContractError::InvalidNodeServiceIdentity);
        }
        Ok(Self {
            configuration_sha256,
        })
    }

    pub fn configuration_sha256(&self) -> &str {
        &self.configuration_sha256
    }
}

#[derive(Debug, Error)]
pub enum NativeComponentH264Mp4BackingServiceError {
    #[error("native component H.264 MP4 backing service is unavailable")]
    Unavailable,
    #[error("native component H.264 MP4 backing request was cancelled")]
    Cancelled,
    #[error("native component H.264 MP4 backing service is busy")]
    Busy,
    #[error("native component H.264 MP4 backing request is invalid")]
    InvalidRequest,
    #[error("native component H.264 MP4 backing exhausted its reviewed resources")]
    ResourceExhausted,
    #[error("native component H.264 MP4 backing returned an invalid result projection")]
    InvalidProjection,
    #[error("native component H.264 MP4 backing failed: {0}")]
    Execution(String),
}

#[derive(Clone, Debug)]
pub struct NativeComponentH264Mp4BackingRequest {
    video: Arc<NativeVideoPayload>,
}

impl NativeComponentH264Mp4BackingRequest {
    pub fn checked(
        video: Arc<NativeVideoPayload>,
    ) -> Result<Self, NativeComponentH264Mp4BackingServiceError> {
        video
            .validate()
            .map_err(|_| NativeComponentH264Mp4BackingServiceError::InvalidRequest)?;
        let components = video
            .components()
            .ok_or(NativeComponentH264Mp4BackingServiceError::InvalidRequest)?;
        let [frame_count, height, width, channels] = components.frames().descriptor().shape()
        else {
            return Err(NativeComponentH264Mp4BackingServiceError::InvalidRequest);
        };
        let descriptor = components.frames().descriptor();
        let valid = *frame_count > 0
            && *height > 0
            && *width > 0
            && matches!(*channels, 3 | 4)
            && descriptor.dtype() == DType::F32
            && descriptor.device() == DeviceId::CPU
            && descriptor
                .is_contiguous()
                .map_err(|_| NativeComponentH264Mp4BackingServiceError::InvalidRequest)?
            && components.audio().is_none();
        if !valid {
            return Err(NativeComponentH264Mp4BackingServiceError::InvalidRequest);
        }
        Ok(Self { video })
    }

    pub fn video(&self) -> &Arc<NativeVideoPayload> {
        &self.video
    }

    pub fn into_video(self) -> Arc<NativeVideoPayload> {
        self.video
    }
}

pub trait NativeComponentH264Mp4BackingService: Send + Sync + fmt::Debug {
    fn identity(&self) -> &NativeComponentH264Mp4BackingServiceIdentity;

    fn encode_backing(
        &self,
        request: NativeComponentH264Mp4BackingRequest,
        context: &ExecutionContext<'_>,
    ) -> BoxFuture<'static, Result<NativeVideoPayload, NativeComponentH264Mp4BackingServiceError>>;
}

#[derive(Debug, Error)]
pub enum NativeShaderServiceError {
    #[error("native shader execution service is unavailable")]
    Unavailable,
    #[error("native shader compute session is invalid: {0}")]
    Contract(#[from] NativeNodeContractError),
    #[error("native shader execution failed: {0}")]
    Shader(#[from] NativeShaderError),
    #[error("native shader executor returned an invalid result projection")]
    InvalidProjection,
}

#[derive(Debug, Error)]
pub enum NativeShaderPreviewError {
    #[error("native shader execution failed: {0}")]
    Shader(#[from] NativeShaderServiceError),
    #[error("native shader preview preparation failed: {0}")]
    Preview(#[from] NativeImagePreviewError),
    #[error("native shader preview effect rollback failed: {0}")]
    Effect(#[from] NativeEffectServiceError),
}

#[derive(Clone, Debug)]
pub struct NativePreparedShaderResult {
    shader: NativeShaderResult,
    effects: Vec<NativePreparedEffectRequest>,
    ui: Value,
}

impl NativePreparedShaderResult {
    pub fn shader(&self) -> &NativeShaderResult {
        &self.shader
    }

    pub fn effects(&self) -> &[NativePreparedEffectRequest] {
        &self.effects
    }

    pub fn ui(&self) -> &Value {
        &self.ui
    }

    pub fn into_parts(self) -> (NativeShaderResult, Vec<NativePreparedEffectRequest>, Value) {
        (self.shader, self.effects, self.ui)
    }
}

#[derive(Clone, Debug)]
pub struct NativeNodeComputeSession {
    identity: NativeNodeServiceIdentity,
    backend: Arc<CpuBackend>,
    stream: StreamId,
    scratch_binding: ScratchBindingIdentity,
}

impl NativeNodeComputeSession {
    pub fn checked(
        identity: NativeNodeServiceIdentity,
        backend: Arc<CpuBackend>,
        stream: StreamId,
        scratch: &ScratchReservation,
    ) -> Result<Self, NativeNodeContractError> {
        backend
            .validate_scratch_reservation(scratch)
            .map_err(|_| NativeNodeContractError::InvalidComputeSession)?;
        Ok(Self {
            identity,
            backend,
            stream,
            scratch_binding: scratch.binding_identity(),
        })
    }

    pub const fn identity(&self) -> &NativeNodeServiceIdentity {
        &self.identity
    }

    pub fn backend(&self) -> &CpuBackend {
        &self.backend
    }

    pub const fn stream(&self) -> StreamId {
        self.stream
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeMultimodalTextGenerationRequest {
    clip: NativeOpaqueHandle,
    image: Option<NativeOpaqueHandle>,
    video: Option<NativeOpaqueHandle>,
    audio: Option<NativeOpaqueHandle>,
    formatted_prompt: String,
    maximum_new_tokens: usize,
    do_sample: bool,
    temperature_bits: u32,
    top_k: usize,
    top_p_bits: u32,
    minimum_p_bits: u32,
    repetition_penalty_bits: u32,
    presence_penalty_bits: u32,
    seed: u64,
    use_default_template: bool,
    thinking: bool,
}

impl NativeMultimodalTextGenerationRequest {
    #[allow(clippy::too_many_arguments)]
    pub fn checked(
        clip: NativeOpaqueHandle,
        image: Option<NativeOpaqueHandle>,
        video: Option<NativeOpaqueHandle>,
        audio: Option<NativeOpaqueHandle>,
        formatted_prompt: impl Into<String>,
        maximum_new_tokens: usize,
        do_sample: bool,
        temperature_bits: u32,
        top_k: usize,
        top_p_bits: u32,
        minimum_p_bits: u32,
        repetition_penalty_bits: u32,
        presence_penalty_bits: u32,
        seed: u64,
        use_default_template: bool,
        thinking: bool,
    ) -> Result<Self, NativeTextGenerationServiceError> {
        let request = Self {
            clip,
            image,
            video,
            audio,
            formatted_prompt: formatted_prompt.into(),
            maximum_new_tokens,
            do_sample,
            temperature_bits,
            top_k,
            top_p_bits,
            minimum_p_bits,
            repetition_penalty_bits,
            presence_penalty_bits,
            seed,
            use_default_template,
            thinking,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn clip(&self) -> &NativeOpaqueHandle {
        &self.clip
    }

    pub fn image(&self) -> Option<&NativeOpaqueHandle> {
        self.image.as_ref()
    }

    pub fn video(&self) -> Option<&NativeOpaqueHandle> {
        self.video.as_ref()
    }

    pub fn audio(&self) -> Option<&NativeOpaqueHandle> {
        self.audio.as_ref()
    }

    pub fn formatted_prompt(&self) -> &str {
        &self.formatted_prompt
    }

    pub const fn maximum_new_tokens(&self) -> usize {
        self.maximum_new_tokens
    }

    pub const fn do_sample(&self) -> bool {
        self.do_sample
    }

    pub const fn temperature_bits(&self) -> u32 {
        self.temperature_bits
    }

    pub const fn top_k(&self) -> usize {
        self.top_k
    }

    pub const fn top_p_bits(&self) -> u32 {
        self.top_p_bits
    }

    pub const fn minimum_p_bits(&self) -> u32 {
        self.minimum_p_bits
    }

    pub const fn repetition_penalty_bits(&self) -> u32 {
        self.repetition_penalty_bits
    }

    pub const fn presence_penalty_bits(&self) -> u32 {
        self.presence_penalty_bits
    }

    pub const fn seed(&self) -> u64 {
        self.seed
    }

    pub const fn use_default_template(&self) -> bool {
        self.use_default_template
    }

    pub const fn thinking(&self) -> bool {
        self.thinking
    }

    pub fn validate(&self) -> Result<(), NativeTextGenerationServiceError> {
        validate_workflow_text("native text generation prompt", &self.formatted_prompt)
            .map_err(|_| NativeTextGenerationServiceError::InvalidRequest)?;
        if !(1..=NATIVE_TEXT_GENERATION_MAX_NEW_TOKENS).contains(&self.maximum_new_tokens)
            || self.top_k > 1_000
        {
            return Err(NativeTextGenerationServiceError::InvalidRequest);
        }
        let temperature = f32::from_bits(self.temperature_bits);
        let top_p = f32::from_bits(self.top_p_bits);
        let minimum_p = f32::from_bits(self.minimum_p_bits);
        let repetition_penalty = f32::from_bits(self.repetition_penalty_bits);
        let presence_penalty = f32::from_bits(self.presence_penalty_bits);
        if !temperature.is_finite()
            || !(0.0..=2.0).contains(&temperature)
            || !top_p.is_finite()
            || !(0.0..=1.0).contains(&top_p)
            || !minimum_p.is_finite()
            || !(0.0..=1.0).contains(&minimum_p)
            || !repetition_penalty.is_finite()
            || !(0.0..=5.0).contains(&repetition_penalty)
            || !presence_penalty.is_finite()
            || !(0.0..=5.0).contains(&presence_penalty)
            || (self.do_sample && temperature < 0.01)
        {
            return Err(NativeTextGenerationServiceError::InvalidRequest);
        }

        let expected_clip = NativeHandleType::new(NativeHandleKind::Clip, "CLIP")
            .map_err(|_| NativeTextGenerationServiceError::InvalidRequest)?;
        let expected_image = NativeHandleType::new(NativeHandleKind::Image, "IMAGE")
            .map_err(|_| NativeTextGenerationServiceError::InvalidRequest)?;
        let expected_audio = NativeHandleType::new(NativeHandleKind::Audio, "AUDIO")
            .map_err(|_| NativeTextGenerationServiceError::InvalidRequest)?;
        if self.clip.handle_type() != &expected_clip
            || self
                .image
                .as_ref()
                .is_some_and(|handle| handle.handle_type() != &expected_image)
            || self
                .video
                .as_ref()
                .is_some_and(|handle| handle.handle_type() != &expected_image)
            || self
                .audio
                .as_ref()
                .is_some_and(|handle| handle.handle_type() != &expected_audio)
        {
            return Err(NativeTextGenerationServiceError::InvalidRequest);
        }
        let store_identity = self.clip.store_identity();
        if self
            .image
            .iter()
            .chain(self.video.iter())
            .chain(self.audio.iter())
            .any(|handle| handle.store_identity() != store_identity)
        {
            return Err(NativeTextGenerationServiceError::InvalidRequest);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeTextGenerationServiceOutput {
    text: String,
    generated_token_count: usize,
}

impl NativeTextGenerationServiceOutput {
    pub fn checked(
        text: impl Into<String>,
        generated_token_count: usize,
        maximum_new_tokens: usize,
    ) -> Result<Self, NativeTextGenerationServiceError> {
        let output = Self {
            text: text.into(),
            generated_token_count,
        };
        validate_workflow_text("native text generation result", &output.text)
            .map_err(|_| NativeTextGenerationServiceError::InvalidOutput)?;
        if generated_token_count > maximum_new_tokens {
            return Err(NativeTextGenerationServiceError::InvalidOutput);
        }
        Ok(output)
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub const fn generated_token_count(&self) -> usize {
        self.generated_token_count
    }

    pub fn into_text(self) -> String {
        self.text
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum NativeTextGenerationServiceError {
    #[error("native text generation service is unavailable")]
    Unavailable,
    #[error("native text generation request was cancelled")]
    Cancelled,
    #[error("native text generation request is invalid")]
    InvalidRequest,
    #[error("native text generation service returned an invalid output")]
    InvalidOutput,
    #[error("native text generation failed: {0}")]
    Execution(String),
}

pub trait NativeTextGenerationService: Send + Sync + fmt::Debug {
    fn identity(&self) -> &NativeNodeServiceIdentity;

    fn generate(
        &self,
        context: &NativeNodeContext,
        request: &NativeMultimodalTextGenerationRequest,
    ) -> Result<NativeTextGenerationServiceOutput, NativeTextGenerationServiceError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeModelTrainingAlgorithm {
    LoRa,
    LoHa,
    LoKr,
    Oft,
}

impl NativeModelTrainingAlgorithm {
    pub const fn source_name(self) -> &'static str {
        match self {
            Self::LoRa => "LoRA",
            Self::LoHa => "LoHa",
            Self::LoKr => "LoKr",
            Self::Oft => "OFT",
        }
    }
}

#[derive(Clone)]
pub struct NativeModelTrainingRequest {
    model: Arc<NativeModelPayload>,
    sampling_policy: Arc<NativeSamplingPolicy>,
    latents: Vec<Arc<NativeLatentBundle>>,
    positive: Vec<Arc<ConditioningSet>>,
    existing_lora: Option<Arc<NativeLoraModelResource>>,
    batch_size: u64,
    gradient_accumulation_steps: u64,
    steps: u64,
    learning_rate: f32,
    rank: u32,
    optimizer: NativeTrainingOptimizer,
    loss: NativeTrainingLoss,
    seed: u64,
    training_dtype: NativeTrainingComputeDType,
    lora_dtype: DType,
    quantized_backward: bool,
    algorithm: NativeModelTrainingAlgorithm,
    gradient_checkpointing: bool,
    checkpoint_depth: u8,
    offloading: bool,
    bucket_mode: bool,
    application_mode: NativeLoraApplicationMode,
}

impl NativeModelTrainingRequest {
    #[allow(clippy::too_many_arguments)]
    pub fn checked(
        model: Arc<NativeModelPayload>,
        latents: Vec<Arc<NativeLatentBundle>>,
        positive: Vec<Arc<ConditioningSet>>,
        existing_lora: Option<Arc<NativeLoraModelResource>>,
        batch_size: u64,
        gradient_accumulation_steps: u64,
        steps: u64,
        learning_rate: f32,
        rank: u32,
        optimizer: NativeTrainingOptimizer,
        loss: NativeTrainingLoss,
        seed: u64,
        training_dtype: NativeTrainingComputeDType,
        lora_dtype: DType,
        quantized_backward: bool,
        algorithm: NativeModelTrainingAlgorithm,
        gradient_checkpointing: bool,
        checkpoint_depth: u8,
        offloading: bool,
        bucket_mode: bool,
        application_mode: NativeLoraApplicationMode,
    ) -> Result<Self, NativeModelTrainingServiceError> {
        let sampling_policy = Arc::new(
            profile_for_model(&model)
                .map_err(|error| NativeModelTrainingServiceError::Execution(error.to_string()))?,
        );
        let request = Self {
            model,
            sampling_policy,
            latents,
            positive,
            existing_lora,
            batch_size,
            gradient_accumulation_steps,
            steps,
            learning_rate,
            rank,
            optimizer,
            loss,
            seed,
            training_dtype,
            lora_dtype,
            quantized_backward,
            algorithm,
            gradient_checkpointing,
            checkpoint_depth,
            offloading,
            bucket_mode,
            application_mode,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> Result<(), NativeModelTrainingServiceError> {
        self.model.validate()?;
        self.sampling_policy
            .validate()
            .map_err(|error| NativeModelTrainingServiceError::Execution(error.to_string()))?;
        let model = self
            .model
            .native_family_model_resource()
            .ok_or(NativeModelTrainingServiceError::IncompatibleModel)?;
        let total_microsteps = self
            .steps
            .checked_mul(self.gradient_accumulation_steps)
            .ok_or(NativeModelTrainingServiceError::InvalidDataset)?;
        let completed_steps = self
            .existing_lora
            .as_ref()
            .map_or(0, |existing| existing.completed_steps())
            .checked_add(self.steps)
            .ok_or(NativeModelTrainingServiceError::InvalidDataset)?;
        if self.latents.is_empty()
            || self.positive.is_empty()
            || self.latents.len() > 1_000_000
            || self.positive.len() > 1_000_000
            || !(1..=10_000).contains(&self.batch_size)
            || !(1..=1_024).contains(&self.gradient_accumulation_steps)
            || !(1..=100_000).contains(&self.steps)
            || !self.learning_rate.is_finite()
            || !(1.0e-7..=1.0).contains(&self.learning_rate)
            || !(1..=128).contains(&self.rank)
            || !matches!(self.lora_dtype, DType::F32 | DType::Bf16)
            || !(1..=5).contains(&self.checkpoint_depth)
            || total_microsteps > 100_000
            || completed_steps > 1_000_000_000
            || (!self.bucket_mode
                && self.positive.len() != 1
                && self.positive.len() != self.latents.len())
        {
            return Err(NativeModelTrainingServiceError::InvalidDataset);
        }
        for latent in &self.latents {
            latent
                .validate_retained()
                .map_err(|_| NativeModelTrainingServiceError::InvalidDataset)?;
            if latent.samples().tensor().is_none() {
                return Err(NativeModelTrainingServiceError::InvalidDataset);
            }
        }
        for conditioning in &self.positive {
            conditioning
                .validate()
                .map_err(|_| NativeModelTrainingServiceError::InvalidDataset)?;
            if conditioning.identity() != model.conditioning_identity() {
                return Err(NativeModelTrainingServiceError::IncompatibleConditioning);
            }
        }
        if let Some(existing) = &self.existing_lora {
            existing.validate()?;
            if existing.target_model_sha256() != self.model.identity().execution_sha256()
                || existing.rank() != self.rank
                || existing.dtype() != self.lora_dtype
            {
                return Err(NativeModelTrainingServiceError::IncompatibleAdapter);
            }
        }
        Ok(())
    }

    pub fn model(&self) -> &Arc<NativeModelPayload> {
        &self.model
    }
    pub fn sampling_policy(&self) -> &Arc<NativeSamplingPolicy> {
        &self.sampling_policy
    }
    pub fn with_sampling_policy(
        mut self,
        sampling_policy: Arc<NativeSamplingPolicy>,
    ) -> Result<Self, NativeModelTrainingServiceError> {
        self.sampling_policy = sampling_policy;
        self.validate()?;
        Ok(self)
    }
    pub fn latents(&self) -> &[Arc<NativeLatentBundle>] {
        &self.latents
    }
    pub fn positive(&self) -> &[Arc<ConditioningSet>] {
        &self.positive
    }
    pub fn existing_lora(&self) -> Option<&Arc<NativeLoraModelResource>> {
        self.existing_lora.as_ref()
    }
    pub const fn batch_size(&self) -> u64 {
        self.batch_size
    }
    pub const fn gradient_accumulation_steps(&self) -> u64 {
        self.gradient_accumulation_steps
    }
    pub const fn steps(&self) -> u64 {
        self.steps
    }
    pub const fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
    pub const fn rank(&self) -> u32 {
        self.rank
    }
    pub const fn optimizer(&self) -> NativeTrainingOptimizer {
        self.optimizer
    }
    pub const fn loss(&self) -> NativeTrainingLoss {
        self.loss
    }
    pub const fn seed(&self) -> u64 {
        self.seed
    }
    pub const fn training_dtype(&self) -> NativeTrainingComputeDType {
        self.training_dtype
    }
    pub const fn lora_dtype(&self) -> DType {
        self.lora_dtype
    }
    pub const fn quantized_backward(&self) -> bool {
        self.quantized_backward
    }
    pub const fn algorithm(&self) -> NativeModelTrainingAlgorithm {
        self.algorithm
    }
    pub const fn gradient_checkpointing(&self) -> bool {
        self.gradient_checkpointing
    }
    pub const fn checkpoint_depth(&self) -> u8 {
        self.checkpoint_depth
    }
    pub const fn offloading(&self) -> bool {
        self.offloading
    }
    pub const fn bucket_mode(&self) -> bool {
        self.bucket_mode
    }
    pub const fn application_mode(&self) -> NativeLoraApplicationMode {
        self.application_mode
    }
}

#[derive(Clone, Debug)]
pub struct NativeModelTrainingOutput {
    lora: Arc<NativeLoraModelResource>,
    losses: Arc<LossMap>,
    steps: u64,
    application_mode: NativeLoraApplicationMode,
}

impl NativeModelTrainingOutput {
    pub fn checked(
        lora: Arc<NativeLoraModelResource>,
        losses: Arc<LossMap>,
        steps: u64,
        application_mode: NativeLoraApplicationMode,
    ) -> Result<Self, NativeModelTrainingServiceError> {
        lora.validate()?;
        losses.validate()?;
        if steps == 0 || steps != lora.completed_steps() {
            return Err(NativeModelTrainingServiceError::InvalidOutput);
        }
        Ok(Self {
            lora,
            losses,
            steps,
            application_mode,
        })
    }

    pub fn lora(&self) -> &Arc<NativeLoraModelResource> {
        &self.lora
    }
    pub fn losses(&self) -> &Arc<LossMap> {
        &self.losses
    }
    pub const fn steps(&self) -> u64 {
        self.steps
    }
    pub const fn application_mode(&self) -> NativeLoraApplicationMode {
        self.application_mode
    }
}

#[derive(Debug, Error)]
pub enum NativeModelTrainingServiceError {
    #[error("native model training service is unavailable")]
    Unavailable,
    #[error("native model training was cancelled")]
    Cancelled,
    #[error("native model training dataset is invalid")]
    InvalidDataset,
    #[error("native model is not compatible with native LoRA training")]
    IncompatibleModel,
    #[error("native training conditioning is incompatible with the retained model")]
    IncompatibleConditioning,
    #[error("existing native LoRA state is incompatible with this training request")]
    IncompatibleAdapter,
    #[error("native model training exceeded its memory budget")]
    OutOfMemory,
    #[error("native model training produced a non-finite loss")]
    NonFiniteLoss,
    #[error("native model training checkpoint failed")]
    CheckpointFailure,
    #[error("native model training service returned invalid output")]
    InvalidOutput,
    #[error("native model training failed: {0}")]
    Execution(String),
    #[error(transparent)]
    Payload(#[from] NativeModelPayloadError),
}

pub trait NativeModelTrainingService: Send + Sync + fmt::Debug {
    fn configuration_sha256(&self) -> &str;

    fn train(
        &self,
        context: &NativeNodeContext,
        request: &NativeModelTrainingRequest,
    ) -> Result<NativeModelTrainingOutput, NativeModelTrainingServiceError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeModelResourceRequest {
    role: NativeModelResourceRole,
    folder_category: String,
    source_names: Vec<String>,
}

const NATIVE_MODEL_RESOURCE_PREFLIGHT_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeModelResourcePreflight {
    schema_version: u16,
    role: NativeModelResourceRole,
    folder_category: String,
    source_names: Vec<String>,
    ordered_source_identity_sha256: String,
    ordered_model_identity_sha256: Vec<String>,
    manifest_sha256: String,
}

impl NativeModelResourcePreflight {
    pub fn checked(
        request: &NativeModelResourceRequest,
        ordered_source_identity_sha256: impl Into<String>,
        ordered_model_identity_sha256: Vec<String>,
        manifest_sha256: impl Into<String>,
    ) -> Result<Self, NativeModelResourceServiceError> {
        let evidence = Self {
            schema_version: NATIVE_MODEL_RESOURCE_PREFLIGHT_SCHEMA_VERSION,
            role: request.role,
            folder_category: request.folder_category.clone(),
            source_names: request.source_names.clone(),
            ordered_source_identity_sha256: ordered_source_identity_sha256.into(),
            ordered_model_identity_sha256,
            manifest_sha256: manifest_sha256.into(),
        };
        evidence.validate_for(request)?;
        Ok(evidence)
    }

    pub fn validate_for(
        &self,
        request: &NativeModelResourceRequest,
    ) -> Result<(), NativeModelResourceServiceError> {
        request.validate_recipe()?;
        if self.schema_version != NATIVE_MODEL_RESOURCE_PREFLIGHT_SCHEMA_VERSION
            || self.role != request.role
            || self.folder_category != request.folder_category
            || self.source_names != request.source_names
            || self.ordered_model_identity_sha256.len() != request.source_names.len()
            || !valid_sha256(&self.ordered_source_identity_sha256)
            || !valid_sha256(&self.manifest_sha256)
            || self
                .ordered_model_identity_sha256
                .iter()
                .any(|digest| !valid_sha256(digest))
        {
            return Err(NativeModelResourceServiceError::SourceChanged);
        }
        Ok(())
    }

    pub const fn role(&self) -> NativeModelResourceRole {
        self.role
    }

    pub fn folder_category(&self) -> &str {
        &self.folder_category
    }

    pub fn source_names(&self) -> &[String] {
        &self.source_names
    }

    pub fn ordered_source_identity_sha256(&self) -> &str {
        &self.ordered_source_identity_sha256
    }

    pub fn ordered_model_identity_sha256(&self) -> &[String] {
        &self.ordered_model_identity_sha256
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }
}

impl NativeModelResourceRequest {
    pub fn checked(
        role: NativeModelResourceRole,
        folder_category: impl Into<String>,
        source_names: Vec<String>,
    ) -> Result<Self, NativeModelResourceServiceError> {
        let request = Self {
            role,
            folder_category: folder_category.into(),
            source_names,
        };
        request.validate()?;
        Ok(request)
    }

    pub const fn role(&self) -> NativeModelResourceRole {
        self.role
    }

    pub fn folder_category(&self) -> &str {
        &self.folder_category
    }

    pub fn source_names(&self) -> &[String] {
        &self.source_names
    }

    pub fn validate(&self) -> Result<(), NativeModelResourceServiceError> {
        if self.folder_category.is_empty()
            || self.folder_category.len() > MAX_NATIVE_MODEL_FOLDER_CATEGORY_BYTES
            || !self
                .folder_category
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
            || self.source_names.is_empty()
            || self.source_names.len() > MAX_NATIVE_MODEL_SOURCE_NAMES
            || self.source_names.iter().collect::<BTreeSet<_>>().len() != self.source_names.len()
            || self.source_names.iter().any(|name| {
                name.is_empty()
                    || name.len() > MAX_NATIVE_MODEL_SOURCE_NAME_BYTES
                    || name.chars().any(char::is_control)
                    || name.starts_with('/')
                    || name.contains(['\\', ':'])
                    || name
                        .split('/')
                        .any(|component| component.is_empty() || matches!(component, "." | ".."))
            })
        {
            return Err(NativeModelResourceServiceError::InvalidRequest);
        }
        Ok(())
    }

    pub fn validate_recipe(&self) -> Result<(), NativeModelResourceServiceError> {
        self.validate()?;
        let source_count = self.source_names.len();
        let matches = match self.role {
            NativeModelResourceRole::Model => {
                source_count == 1
                    && matches!(
                        self.folder_category.as_str(),
                        "checkpoints" | "diffusion_models"
                    )
            }
            NativeModelResourceRole::Clip => {
                (self.folder_category == "checkpoints" && source_count == 1)
                    || (self.folder_category == "text_encoders" && (1..=4).contains(&source_count))
            }
            NativeModelResourceRole::Vae => {
                source_count == 1 && matches!(self.folder_category.as_str(), "checkpoints" | "vae")
            }
            NativeModelResourceRole::AudioEncoder => {
                source_count == 1 && self.folder_category == "audio_encoders"
            }
            NativeModelResourceRole::ClipVision => {
                source_count == 1 && self.folder_category == "clip_vision"
            }
            NativeModelResourceRole::StyleModel => {
                source_count == 1 && self.folder_category == "style_models"
            }
            NativeModelResourceRole::Photomaker => {
                source_count == 1 && self.folder_category == "photomaker"
            }
            NativeModelResourceRole::Gligen => {
                source_count == 1 && self.folder_category == "gligen"
            }
            NativeModelResourceRole::LatentUpscaleModel => {
                source_count == 1 && self.folder_category == "latent_upscale_models"
            }
            NativeModelResourceRole::BackgroundRemoval => {
                source_count == 1 && self.folder_category == "background_removal"
            }
            NativeModelResourceRole::FrameInterpolation => {
                source_count == 1 && self.folder_category == "frame_interpolation"
            }
            NativeModelResourceRole::Da3Model | NativeModelResourceRole::MogeModel => {
                source_count == 1 && self.folder_category == "geometry_estimation"
            }
            NativeModelResourceRole::OpticalFlow => {
                source_count == 1 && self.folder_category == "optical_flow"
            }
            NativeModelResourceRole::ControlNet => {
                source_count == 1 && self.folder_category == "controlnet"
            }
            NativeModelResourceRole::FaceDetection => {
                source_count == 1 && self.folder_category == "face_detection"
            }
            NativeModelResourceRole::Hooks => source_count == 1 && self.folder_category == "hooks",
            NativeModelResourceRole::HookKeyframes => {
                source_count == 1 && self.folder_category == "hook_keyframes"
            }
            NativeModelResourceRole::LatentOperation => {
                source_count == 1 && self.folder_category == "latent_operations"
            }
            NativeModelResourceRole::LoraModel => {
                source_count == 1 && self.folder_category == "loras"
            }
            NativeModelResourceRole::ModelPatch => {
                source_count == 1 && self.folder_category == "model_patches"
            }
            NativeModelResourceRole::UpscaleModel => {
                source_count == 1 && self.folder_category == "upscale_models"
            }
        };
        if matches {
            Ok(())
        } else {
            Err(NativeModelResourceServiceError::InvalidRequest)
        }
    }
}

#[derive(Debug, Error)]
pub enum NativeModelResourceServiceError {
    #[error("native model resource service is unavailable")]
    Unavailable,
    #[error("native model resource operation was cancelled")]
    Cancelled,
    #[error("native model resource request is invalid or unsupported")]
    InvalidRequest,
    #[error("native model resource service identity is stale")]
    StaleIdentity,
    #[error("native model source is unavailable")]
    SourceUnavailable,
    #[error("native model source changed during loading")]
    SourceChanged,
    #[error("native model source or response is malformed: {0}")]
    MalformedSource(&'static str),
    #[error("native model resource memory limit was exceeded")]
    OutOfMemory,
    #[error(transparent)]
    StyleModel(#[from] Box<NativeStyleModelError>),
    #[error(transparent)]
    PhotoMaker(#[from] Box<NativePhotoMakerError>),
    #[error(transparent)]
    Gligen(#[from] NativeGligenError),
    #[error(transparent)]
    AudioEncoder(#[from] AudioEncoderError),
    #[error(transparent)]
    BackgroundRemoval(#[from] Box<NativeBackgroundRemovalError>),
    #[error(transparent)]
    Clip(#[from] Box<ClipError>),
    #[error(transparent)]
    ClipVision(#[from] Box<ClipVisionError>),
    #[error(transparent)]
    DepthAnything3(#[from] Box<NativeDepthAnything3Error>),
    #[error(transparent)]
    FrameInterpolation(#[from] FrameInterpolationError),
    #[error(transparent)]
    LatentUpscale(#[from] NativeLatentUpscaleModelError),
    #[error(transparent)]
    ModelFamily(#[from] Box<ModelFamilyError>),
    #[error(transparent)]
    Moge(#[from] Box<NativeMogeError>),
    #[error(transparent)]
    VisionModel(#[from] Box<NativeVisionModelError>),
    #[error(transparent)]
    VaeRecipe(#[from] Box<NativeVaeRecipeError>),
    #[error(transparent)]
    Payload(#[from] Box<NativeModelPayloadError>),
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

impl From<NativeDepthAnything3Error> for NativeModelResourceServiceError {
    fn from(error: NativeDepthAnything3Error) -> Self {
        Self::DepthAnything3(Box::new(error))
    }
}

impl From<NativeVaeRecipeError> for NativeModelResourceServiceError {
    fn from(error: NativeVaeRecipeError) -> Self {
        Self::VaeRecipe(Box::new(error))
    }
}

impl From<NativeStyleModelError> for NativeModelResourceServiceError {
    fn from(error: NativeStyleModelError) -> Self {
        Self::StyleModel(Box::new(error))
    }
}

impl From<NativePhotoMakerError> for NativeModelResourceServiceError {
    fn from(error: NativePhotoMakerError) -> Self {
        Self::PhotoMaker(Box::new(error))
    }
}

impl From<NativeBackgroundRemovalError> for NativeModelResourceServiceError {
    fn from(error: NativeBackgroundRemovalError) -> Self {
        Self::BackgroundRemoval(Box::new(error))
    }
}

impl From<ClipError> for NativeModelResourceServiceError {
    fn from(error: ClipError) -> Self {
        Self::Clip(Box::new(error))
    }
}

impl From<ClipVisionError> for NativeModelResourceServiceError {
    fn from(error: ClipVisionError) -> Self {
        Self::ClipVision(Box::new(error))
    }
}

impl From<ModelFamilyError> for NativeModelResourceServiceError {
    fn from(error: ModelFamilyError) -> Self {
        Self::ModelFamily(Box::new(error))
    }
}

impl From<NativeMogeError> for NativeModelResourceServiceError {
    fn from(error: NativeMogeError) -> Self {
        Self::Moge(Box::new(error))
    }
}

impl From<NativeVisionModelError> for NativeModelResourceServiceError {
    fn from(error: NativeVisionModelError) -> Self {
        Self::VisionModel(Box::new(error))
    }
}

impl From<NativeModelPayloadError> for NativeModelResourceServiceError {
    fn from(error: NativeModelPayloadError) -> Self {
        Self::Payload(Box::new(error))
    }
}

pub trait NativeModelResourceService: Send + Sync + fmt::Debug {
    fn identity(&self) -> &NativeNodeServiceIdentity;

    fn preflight(
        &self,
        _context: &NativeNodeContext,
        _request: &NativeModelResourceRequest,
    ) -> Result<NativeModelResourcePreflight, NativeModelResourceServiceError> {
        Err(NativeModelResourceServiceError::Unavailable)
    }

    fn load(
        &self,
        context: &NativeNodeContext,
        request: &NativeModelResourceRequest,
    ) -> Result<Arc<NativeModelPayload>, NativeModelResourceServiceError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeModelCompileBackend {
    Inductor,
    CudaGraphs,
}

impl NativeModelCompileBackend {
    pub const fn source_name(self) -> &'static str {
        match self {
            Self::Inductor => "inductor",
            Self::CudaGraphs => "cudagraphs",
        }
    }

    pub fn from_source_name(value: &str) -> Result<Self, NativeModelCompileServiceError> {
        match value {
            "inductor" => Ok(Self::Inductor),
            "cudagraphs" => Ok(Self::CudaGraphs),
            _ => Err(NativeModelCompileServiceError::UnsupportedBackend),
        }
    }
}

#[derive(Clone)]
pub struct NativeModelCompileRequest {
    model: Arc<NativeDiffusionPayload>,
    backend: NativeModelCompileBackend,
}

impl NativeModelCompileRequest {
    pub fn checked(
        model: Arc<NativeDiffusionPayload>,
        backend: NativeModelCompileBackend,
    ) -> Result<Self, NativeModelCompileServiceError> {
        let request = Self { model, backend };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> Result<(), NativeModelCompileServiceError> {
        self.model
            .validate()
            .map_err(|_| NativeModelCompileServiceError::InvalidRequest)?;
        if self.model.model_resources().is_none() {
            return Err(NativeModelCompileServiceError::InvalidRequest);
        }
        Ok(())
    }

    pub fn model(&self) -> &Arc<NativeDiffusionPayload> {
        &self.model
    }

    pub const fn backend(&self) -> NativeModelCompileBackend {
        self.backend
    }
}

#[derive(Clone)]
pub struct NativeModelCompileOutput {
    model: Arc<NativeDiffusionPayload>,
    compiled_model: Arc<dyn NativeCompiledModelInvocation>,
    source_model_sha256: String,
    policy_sha256: String,
}

#[derive(Debug, Error)]
pub enum NativeCompiledModelInvocationError {
    #[error("native compiled model invocation was cancelled")]
    Cancelled,
    #[error("native compiled model invocation guard changed")]
    GuardMismatch,
    #[error("native compiled model invocation failed: {0}")]
    Operation(String),
    #[error("native compiled model wrapper is invalid")]
    Invalid,
}

pub trait NativeCompiledModelInvocation: Send + Sync + fmt::Debug {
    fn backend(&self) -> NativeModelCompileBackend;

    fn source_model_sha256(&self) -> &str;

    fn policy_sha256(&self) -> &str;

    fn resident_bytes(&self) -> usize;

    fn validate(&self) -> Result<(), NativeModelCompileServiceError>;

    fn invoke(
        &self,
        cancellation: &CancellationToken,
        typed_inputs: &BTreeMap<String, String>,
        operation: &mut dyn FnMut() -> Result<(), String>,
    ) -> Result<(), NativeCompiledModelInvocationError>;
}

impl NativeModelCompileOutput {
    pub fn checked(
        request: &NativeModelCompileRequest,
        model: Arc<NativeDiffusionPayload>,
        compiled_model: Arc<dyn NativeCompiledModelInvocation>,
        source_model_sha256: impl Into<String>,
        policy_sha256: impl Into<String>,
    ) -> Result<Self, NativeModelCompileServiceError> {
        let output = Self {
            model,
            compiled_model,
            source_model_sha256: source_model_sha256.into(),
            policy_sha256: policy_sha256.into(),
        };
        output.validate_for(request)?;
        Ok(output)
    }

    pub fn validate_for(
        &self,
        request: &NativeModelCompileRequest,
    ) -> Result<(), NativeModelCompileServiceError> {
        request.validate()?;
        self.model
            .validate()
            .map_err(|_| NativeModelCompileServiceError::InvalidOutput)?;
        self.compiled_model.validate()?;
        let source_resources = request
            .model()
            .model_resources()
            .ok_or(NativeModelCompileServiceError::InvalidRequest)?;
        let output_resources = self
            .model
            .model_resources()
            .ok_or(NativeModelCompileServiceError::InvalidOutput)?;
        if self.source_model_sha256 != request.model().digest_sha256()
            || self.compiled_model.backend() != request.backend()
            || !valid_sha256(&self.policy_sha256)
            || self.compiled_model.source_model_sha256()
                != source_resources.0.identity().execution_sha256()
            || self.compiled_model.policy_sha256() != self.policy_sha256
            || Arc::ptr_eq(request.model(), &self.model)
            || !Arc::ptr_eq(source_resources.0, output_resources.0)
            || !Arc::ptr_eq(source_resources.1, output_resources.1)
        {
            return Err(NativeModelCompileServiceError::InvalidOutput);
        }
        Ok(())
    }

    pub fn model(&self) -> &Arc<NativeDiffusionPayload> {
        &self.model
    }

    pub fn compiled_model(&self) -> &Arc<dyn NativeCompiledModelInvocation> {
        &self.compiled_model
    }

    pub fn policy_sha256(&self) -> &str {
        &self.policy_sha256
    }
}

#[derive(Debug, Error)]
pub enum NativeModelCompileServiceError {
    #[error("native model compile service is unavailable")]
    Unavailable,
    #[error("native model compile operation was cancelled")]
    Cancelled,
    #[error("native model compile request is invalid")]
    InvalidRequest,
    #[error("native model compile output is invalid")]
    InvalidOutput,
    #[error("native model compile service identity is stale")]
    StaleIdentity,
    #[error("source compile backend is unsupported")]
    UnsupportedBackend,
    #[error("native model compilation failed: {0}")]
    CompileFailure(String),
    #[error("native model compile graph rebind failed: {0}")]
    RebindFailure(String),
}

pub trait NativeModelCompileService: Send + Sync + fmt::Debug {
    fn identity(&self) -> &NativeNodeServiceIdentity;

    fn compile(
        &self,
        context: &NativeNodeContext,
        request: &NativeModelCompileRequest,
    ) -> Result<NativeModelCompileOutput, NativeModelCompileServiceError>;
}

#[derive(Clone, Debug)]
pub struct NativeSdPoseRequest {
    pub model: NativeOpaqueHandle,
    pub vae: NativeOpaqueHandle,
    pub image: NativeOpaqueHandle,
    pub batch_size: usize,
    pub bounding_boxes: Option<NativeOpaqueHandle>,
}

pub trait NativeSdPoseService: Send + Sync + fmt::Debug {
    fn identity(&self) -> &NativeNodeServiceIdentity;

    fn extract(
        &self,
        context: &NativeNodeContext,
        request: &NativeSdPoseRequest,
    ) -> Result<sim_media::NativePoseKeypointPayload, NativeNodeFailure>;
}

#[derive(Clone, Debug, Default)]
pub struct NativeNodeServices {
    assets: Option<Arc<dyn NativeAssetResolver>>,
    effects: Option<Arc<dyn NativePreparedEffectService>>,
    compute: Option<NativeNodeComputeSession>,
    shader: Option<Arc<dyn NativeShaderExecutor>>,
    ltxv_preprocess: Option<Arc<dyn NativeLtxvPreprocessService>>,
    webm_encode: Option<Arc<dyn NativeWebmEncodeService>>,
    component_h264_mp4_backing: Option<Arc<dyn NativeComponentH264Mp4BackingService>>,
    text_generation: Option<Arc<dyn NativeTextGenerationService>>,
    model_resources: Option<Arc<dyn NativeModelResourceService>>,
    model_training: Option<Arc<dyn NativeModelTrainingService>>,
    model_compile: Option<Arc<dyn NativeModelCompileService>>,
    sdpose: Option<Arc<dyn NativeSdPoseService>>,
    provider_execution: Option<NativeProviderExecutionIdentity>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeProviderExecutionIdentity {
    compiled_plan_sha256: String,
}

impl NativeProviderExecutionIdentity {
    pub fn checked(
        compiled_plan_sha256: impl Into<String>,
    ) -> Result<Self, NativeNodeContractError> {
        let compiled_plan_sha256 = compiled_plan_sha256.into();
        if !valid_sha256(&compiled_plan_sha256) {
            return Err(NativeNodeContractError::InvalidNodeContext);
        }
        Ok(Self {
            compiled_plan_sha256,
        })
    }

    pub fn compiled_plan_sha256(&self) -> &str {
        &self.compiled_plan_sha256
    }
}

impl NativeNodeServices {
    pub fn checked(
        assets: Option<Arc<dyn NativeAssetResolver>>,
        effects: Option<Arc<dyn NativePreparedEffectService>>,
        compute: Option<NativeNodeComputeSession>,
    ) -> Result<Self, NativeNodeContractError> {
        for identity in assets
            .as_deref()
            .map(NativeAssetResolver::identity)
            .into_iter()
            .chain(
                effects
                    .as_deref()
                    .map(NativePreparedEffectService::identity),
            )
            .chain(compute.as_ref().map(NativeNodeComputeSession::identity))
        {
            if identity.service_id().is_nil() {
                return Err(NativeNodeContractError::InvalidNodeServiceIdentity);
            }
        }
        Ok(Self {
            assets,
            effects,
            compute,
            shader: None,
            ltxv_preprocess: None,
            webm_encode: None,
            component_h264_mp4_backing: None,
            text_generation: None,
            model_resources: None,
            model_training: None,
            model_compile: None,
            sdpose: None,
            provider_execution: None,
        })
    }

    pub fn with_shader(mut self, shader: Arc<dyn NativeShaderExecutor>) -> Self {
        self.shader = Some(shader);
        self
    }

    pub fn with_ltxv_preprocess(
        mut self,
        service: Arc<dyn NativeLtxvPreprocessService>,
    ) -> Result<Self, NativeNodeContractError> {
        NativeLtxvPreprocessServiceIdentity::checked(
            service.identity().configuration_sha256().to_owned(),
        )?;
        self.ltxv_preprocess = Some(service);
        Ok(self)
    }

    pub fn with_webm_encode(
        mut self,
        service: Arc<dyn NativeWebmEncodeService>,
    ) -> Result<Self, NativeNodeContractError> {
        NativeWebmEncodeServiceIdentity::checked(
            service.identity().configuration_sha256().to_owned(),
        )?;
        self.webm_encode = Some(service);
        Ok(self)
    }

    pub fn with_component_h264_mp4_backing(
        mut self,
        service: Arc<dyn NativeComponentH264Mp4BackingService>,
    ) -> Result<Self, NativeNodeContractError> {
        NativeComponentH264Mp4BackingServiceIdentity::checked(
            service.identity().configuration_sha256().to_owned(),
        )?;
        self.component_h264_mp4_backing = Some(service);
        Ok(self)
    }

    pub fn with_text_generation(
        mut self,
        service: Arc<dyn NativeTextGenerationService>,
    ) -> Result<Self, NativeNodeContractError> {
        if service.identity().service_id().is_nil() {
            return Err(NativeNodeContractError::InvalidNodeServiceIdentity);
        }
        self.text_generation = Some(service);
        Ok(self)
    }

    pub fn with_model_resources(
        mut self,
        service: Arc<dyn NativeModelResourceService>,
    ) -> Result<Self, NativeNodeContractError> {
        if service.identity().service_id().is_nil() {
            return Err(NativeNodeContractError::InvalidNodeServiceIdentity);
        }
        self.model_resources = Some(service);
        Ok(self)
    }

    pub fn with_model_training(
        mut self,
        service: Arc<dyn NativeModelTrainingService>,
    ) -> Result<Self, NativeNodeContractError> {
        if !valid_sha256(service.configuration_sha256()) {
            return Err(NativeNodeContractError::InvalidNodeServiceIdentity);
        }
        self.model_training = Some(service);
        Ok(self)
    }

    pub fn with_model_compile(
        mut self,
        service: Arc<dyn NativeModelCompileService>,
    ) -> Result<Self, NativeNodeContractError> {
        if service.identity().service_id().is_nil() {
            return Err(NativeNodeContractError::InvalidNodeServiceIdentity);
        }
        self.model_compile = Some(service);
        Ok(self)
    }

    pub fn with_sdpose(
        mut self,
        service: Arc<dyn NativeSdPoseService>,
    ) -> Result<Self, NativeNodeContractError> {
        if service.identity().service_id().is_nil() {
            return Err(NativeNodeContractError::InvalidNodeServiceIdentity);
        }
        self.sdpose = Some(service);
        Ok(self)
    }

    pub fn with_provider_execution(
        mut self,
        provider_execution: NativeProviderExecutionIdentity,
    ) -> Self {
        self.provider_execution = Some(provider_execution);
        self
    }
}

#[derive(Clone, Debug)]
pub struct NativeNodeContext {
    pub prompt_id: PromptId,
    pub attempt_id: AttemptId,
    pub node_id: NodeId,
    pub cancellation: CancellationToken,
    pub scratch: ScratchReservation,
    handle_store: Arc<dyn NativeHandleStore>,
    services: NativeNodeServices,
}

impl NativeNodeContext {
    pub fn new(
        prompt_id: PromptId,
        attempt_id: AttemptId,
        node_id: NodeId,
        cancellation: CancellationToken,
        scratch: ScratchReservation,
        handle_store: Arc<dyn NativeHandleStore>,
    ) -> Result<Self, NativeNodeContractError> {
        Self::new_with_services(
            prompt_id,
            attempt_id,
            node_id,
            cancellation,
            scratch,
            handle_store,
            NativeNodeServices::default(),
        )
    }

    pub fn new_with_services(
        prompt_id: PromptId,
        attempt_id: AttemptId,
        node_id: NodeId,
        cancellation: CancellationToken,
        scratch: ScratchReservation,
        handle_store: Arc<dyn NativeHandleStore>,
        services: NativeNodeServices,
    ) -> Result<Self, NativeNodeContractError> {
        let context = Self {
            prompt_id,
            attempt_id,
            node_id,
            cancellation,
            scratch,
            handle_store,
            services,
        };
        context.validate()?;
        Ok(context)
    }

    pub fn handle_store(&self) -> &dyn NativeHandleStore {
        self.handle_store.as_ref()
    }

    pub fn asset_resolver(&self) -> Result<&dyn NativeAssetResolver, NativeAssetServiceError> {
        self.services
            .assets
            .as_deref()
            .ok_or(NativeAssetServiceError::Unavailable)
    }

    pub fn prepared_effects(
        &self,
    ) -> Result<&dyn NativePreparedEffectService, NativeEffectServiceError> {
        self.services
            .effects
            .as_deref()
            .ok_or(NativeEffectServiceError::Unavailable)
    }

    pub fn compute_session(&self) -> Result<&NativeNodeComputeSession, NativeNodeContractError> {
        self.services
            .compute
            .as_ref()
            .ok_or(NativeNodeContractError::InvalidComputeSession)
    }

    pub fn provider_execution(
        &self,
    ) -> Result<&NativeProviderExecutionIdentity, NativeNodeContractError> {
        self.services
            .provider_execution
            .as_ref()
            .ok_or(NativeNodeContractError::InvalidNodeContext)
    }

    pub fn ltxv_preprocess_service(
        &self,
    ) -> Result<&dyn NativeLtxvPreprocessService, NativeLtxvPreprocessServiceError> {
        self.services
            .ltxv_preprocess
            .as_deref()
            .ok_or(NativeLtxvPreprocessServiceError::Unavailable)
    }

    pub fn webm_encode_service(
        &self,
    ) -> Result<&dyn NativeWebmEncodeService, NativeWebmEncodeServiceError> {
        self.services
            .webm_encode
            .as_deref()
            .ok_or(NativeWebmEncodeServiceError::Unavailable)
    }

    pub fn component_h264_mp4_backing_service(
        &self,
    ) -> Result<&dyn NativeComponentH264Mp4BackingService, NativeComponentH264Mp4BackingServiceError>
    {
        self.services
            .component_h264_mp4_backing
            .as_deref()
            .ok_or(NativeComponentH264Mp4BackingServiceError::Unavailable)
    }

    pub fn execute_multimodal_text_generation(
        &self,
        request: &NativeMultimodalTextGenerationRequest,
    ) -> Result<NativeTextGenerationServiceOutput, NativeTextGenerationServiceError> {
        self.cancellation
            .check()
            .map_err(|_| NativeTextGenerationServiceError::Cancelled)?;
        request.validate()?;
        let service = self
            .services
            .text_generation
            .as_deref()
            .ok_or(NativeTextGenerationServiceError::Unavailable)?;
        if !service.identity().matches(self.attempt_id, &self.node_id) {
            return Err(NativeTextGenerationServiceError::InvalidRequest);
        }
        let output = service.generate(self, request)?;
        self.cancellation
            .check()
            .map_err(|_| NativeTextGenerationServiceError::Cancelled)?;
        NativeTextGenerationServiceOutput::checked(
            output.text,
            output.generated_token_count,
            request.maximum_new_tokens,
        )
    }

    pub fn load_native_model_resource(
        &self,
        request: &NativeModelResourceRequest,
    ) -> Result<Arc<NativeModelPayload>, NativeModelResourceServiceError> {
        self.cancellation
            .check()
            .map_err(|_| NativeModelResourceServiceError::Cancelled)?;
        request.validate()?;
        let service = self
            .services
            .model_resources
            .as_deref()
            .ok_or(NativeModelResourceServiceError::Unavailable)?;
        if !service.identity().matches(self.attempt_id, &self.node_id) {
            return Err(NativeModelResourceServiceError::StaleIdentity);
        }
        let payload = service.load(self, request)?;
        self.cancellation
            .check()
            .map_err(|_| NativeModelResourceServiceError::Cancelled)?;
        if !service.identity().matches(self.attempt_id, &self.node_id) {
            return Err(NativeModelResourceServiceError::StaleIdentity);
        }
        payload.validate()?;
        if payload.identity().role() != request.role {
            return Err(NativeModelResourceServiceError::MalformedSource(
                "payload role does not match the checked request",
            ));
        }
        Ok(payload)
    }

    pub fn execute_native_model_training(
        &self,
        request: &NativeModelTrainingRequest,
    ) -> Result<NativeModelTrainingOutput, NativeModelTrainingServiceError> {
        self.cancellation
            .check()
            .map_err(|_| NativeModelTrainingServiceError::Cancelled)?;
        request.validate()?;
        let service = self
            .services
            .model_training
            .as_deref()
            .ok_or(NativeModelTrainingServiceError::Unavailable)?;
        let output = service.train(self, request)?;
        self.cancellation
            .check()
            .map_err(|_| NativeModelTrainingServiceError::Cancelled)?;
        NativeModelTrainingOutput::checked(
            output.lora,
            output.losses,
            output.steps,
            output.application_mode,
        )
    }

    pub fn compile_native_model(
        &self,
        request: &NativeModelCompileRequest,
    ) -> Result<NativeModelCompileOutput, NativeModelCompileServiceError> {
        self.cancellation
            .check()
            .map_err(|_| NativeModelCompileServiceError::Cancelled)?;
        request.validate()?;
        let service = self
            .services
            .model_compile
            .as_deref()
            .ok_or(NativeModelCompileServiceError::Unavailable)?;
        if !service.identity().matches(self.attempt_id, &self.node_id) {
            return Err(NativeModelCompileServiceError::StaleIdentity);
        }
        let output = service.compile(self, request)?;
        self.cancellation
            .check()
            .map_err(|_| NativeModelCompileServiceError::Cancelled)?;
        if !service.identity().matches(self.attempt_id, &self.node_id) {
            return Err(NativeModelCompileServiceError::StaleIdentity);
        }
        output.validate_for(request)?;
        Ok(output)
    }

    pub fn extract_native_sdpose(
        &self,
        request: &NativeSdPoseRequest,
    ) -> Result<sim_media::NativePoseKeypointPayload, NativeNodeFailure> {
        let invalid = |message: String| NativeNodeFailure {
            code: "invalid_native_sdpose_request".to_owned(),
            message,
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        };
        let check = || {
            if self.cancellation.is_cancelled() {
                return Err(NativeNodeFailure {
                    code: "interrupted".to_owned(),
                    message: "SDPose extraction cancelled".to_owned(),
                    kind: NativeNodeFailureKind::Interrupted,
                    retryable: true,
                });
            }
            let service = self
                .services
                .sdpose
                .as_deref()
                .ok_or_else(|| NativeNodeFailure {
                    code: "native_sdpose_unavailable".to_owned(),
                    message: "The native SDPose execution service is unavailable".to_owned(),
                    kind: NativeNodeFailureKind::Failure,
                    retryable: false,
                })?;
            if !service.identity().matches(self.attempt_id, &self.node_id) {
                return Err(NativeNodeFailure {
                    code: "native_sdpose_stale_service".to_owned(),
                    message: "The SDPose service belongs to another execution attempt".to_owned(),
                    kind: NativeNodeFailureKind::Failure,
                    retryable: false,
                });
            }
            Ok(service)
        };
        let service = check()?;
        if !(1..=10_000).contains(&request.batch_size) {
            return Err(invalid(
                "SDPose batch size must be between 1 and 10000".to_owned(),
            ));
        }
        for (handle, kind, type_id) in [
            (&request.model, NativeHandleKind::Model, "MODEL"),
            (&request.vae, NativeHandleKind::Vae, "VAE"),
            (&request.image, NativeHandleKind::Image, "IMAGE"),
        ]
        .into_iter()
        .chain(
            request
                .bounding_boxes
                .as_ref()
                .map(|handle| (handle, NativeHandleKind::StructuredCompute, "BOUNDING_BOX")),
        ) {
            if handle.handle_type().kind != kind || handle.handle_type().type_id != type_id {
                return Err(invalid(format!("SDPose requires a {type_id} handle")));
            }
            self.handle_store()
                .resolve(handle, handle.handle_type(), &self.cancellation)
                .map_err(|error| match check() {
                    Err(error) => error,
                    Ok(_) => invalid(error.to_string()),
                })?;
        }
        let output = service.extract(self, request);
        check()?;
        let output = output?;
        output
            .validate()
            .map_err(|error| invalid(error.to_string()))?;
        Ok(output)
    }

    pub fn preflight_native_model_resource(
        &self,
        request: &NativeModelResourceRequest,
    ) -> Result<NativeModelResourcePreflight, NativeModelResourceServiceError> {
        self.cancellation
            .check()
            .map_err(|_| NativeModelResourceServiceError::Cancelled)?;
        request.validate_recipe()?;
        let service = self
            .services
            .model_resources
            .as_deref()
            .ok_or(NativeModelResourceServiceError::Unavailable)?;
        if !service.identity().matches(self.attempt_id, &self.node_id) {
            return Err(NativeModelResourceServiceError::StaleIdentity);
        }
        let evidence = service.preflight(self, request)?;
        self.cancellation
            .check()
            .map_err(|_| NativeModelResourceServiceError::Cancelled)?;
        if !service.identity().matches(self.attempt_id, &self.node_id) {
            return Err(NativeModelResourceServiceError::StaleIdentity);
        }
        evidence.validate_for(request)?;
        Ok(evidence)
    }

    pub fn execute_shader(
        &self,
        request: &NativeShaderRequest,
    ) -> Result<NativeShaderResult, NativeShaderServiceError> {
        self.cancellation
            .check()
            .map_err(|_| NativeShaderError::Cancelled)?;
        let compute = self.compute_session()?;
        let execution_context = compute.execution_context(self)?;
        let shader = self
            .services
            .shader
            .as_deref()
            .ok_or(NativeShaderServiceError::Unavailable)?;
        let result = shader
            .execute(request, compute.backend(), &execution_context)
            .map_err(NativeShaderServiceError::from)?;
        let (batch, _, _, _) = request
            .images
            .first()
            .ok_or(NativeShaderServiceError::InvalidProjection)?
            .dimensions()
            .map_err(|_| NativeShaderServiceError::InvalidProjection)?;
        if result.outputs.len() != MAX_SHADER_OUTPUTS
            || result.pass_count == 0
            || result.pass_count > MAX_SHADER_PASSES
        {
            return Err(NativeShaderServiceError::InvalidProjection);
        }
        for output in &result.outputs {
            let dimensions = output
                .dimensions()
                .map_err(|_| NativeShaderServiceError::InvalidProjection)?;
            if dimensions
                != (
                    batch,
                    u64::from(request.height),
                    u64::from(request.width),
                    4,
                )
            {
                return Err(NativeShaderServiceError::InvalidProjection);
            }
        }
        Ok(result)
    }

    pub fn execute_shader_with_previews(
        &self,
        request: &NativeShaderRequest,
    ) -> Result<NativePreparedShaderResult, NativeShaderPreviewError> {
        let shader = self.execute_shader(request)?;
        let effects_service = self.prepared_effects()?;
        let mut effects = Vec::new();
        let mut input_images = Vec::new();

        for (image_index, image) in request.images.iter().enumerate() {
            let prefix = format!("GLSLShader_input_{image_index}");
            let preview = match self.prepare_image_preview(image, &prefix) {
                Ok(preview) => preview,
                Err(error) => {
                    for effect in effects.iter().rev() {
                        effects_service.rollback_prepared(effect)?;
                    }
                    return Err(error.into());
                }
            };
            let (preview_effects, ui) = preview.into_parts();
            if let Some(images) = ui.get("images").and_then(Value::as_array) {
                input_images.extend(images.iter().cloned());
            }
            effects.extend(preview_effects);
        }

        let output = shader
            .outputs
            .first()
            .ok_or(NativeShaderServiceError::InvalidProjection)?;
        let preview = match self.prepare_image_preview(output, "GLSLShader_output") {
            Ok(preview) => preview,
            Err(error) => {
                for effect in effects.iter().rev() {
                    effects_service.rollback_prepared(effect)?;
                }
                return Err(error.into());
            }
        };
        let (preview_effects, ui) = preview.into_parts();
        let output_images = ui
            .get("images")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        effects.extend(preview_effects);

        if self.cancellation.is_cancelled() {
            for effect in effects.iter().rev() {
                effects_service.rollback_prepared(effect)?;
            }
            return Err(NativeShaderServiceError::Shader(NativeShaderError::Cancelled).into());
        }
        Ok(NativePreparedShaderResult {
            shader,
            effects,
            ui: json!({"input_images": input_images, "images": output_images}),
        })
    }

    pub fn prepare_provider_image_png(
        &self,
        image: &ImageTensor,
        batch_index: u64,
        maximum_pixels: Option<u64>,
    ) -> Result<Vec<u8>, NativeNodeFailure> {
        let compute = self
            .compute_session()
            .map_err(|error| self.provider_image_preparation_failure(error))?;
        let execution_context = compute
            .execution_context(self)
            .map_err(|error| self.provider_image_preparation_failure(error))?;
        encode_api_png_frame_with_context(
            compute.backend(),
            &execution_context,
            image,
            batch_index,
            maximum_pixels,
            PngLimits::default(),
        )
        .map_err(|error| self.provider_image_preparation_failure(error))
    }

    pub fn prepare_provider_source_image_png(
        &self,
        image: &ImageTensor,
        maximum_pixels: u64,
    ) -> Result<Vec<u8>, NativeNodeFailure> {
        let compute = self
            .compute_session()
            .map_err(|error| self.provider_image_preparation_failure(error))?;
        let execution = compute
            .execution_context(self)
            .map_err(|error| self.provider_image_preparation_failure(error))?;
        encode_api_png_source_batch_with_context(
            compute.backend(),
            &execution,
            image,
            maximum_pixels,
            PngLimits::default(),
        )
        .map_err(|error| self.provider_image_preparation_failure(error))
    }

    pub fn prepare_provider_mask_png(
        &self,
        mask: &Tensor,
        maximum_pixels: u64,
    ) -> Result<Vec<u8>, NativeNodeFailure> {
        let compute = self
            .compute_session()
            .map_err(|error| self.provider_image_preparation_failure(error))?;
        let execution = compute
            .execution_context(self)
            .map_err(|error| self.provider_image_preparation_failure(error))?;
        encode_api_png_mask_with_context(
            compute.backend(),
            &execution,
            mask,
            maximum_pixels,
            PngLimits::default(),
        )
        .map_err(|error| self.provider_image_preparation_failure(error))
    }

    pub async fn prepare_provider_mask_video(
        &self,
        mask: &Tensor,
        frame_rate: (u64, u64),
        source_frame_count: u64,
        maximum_pixels: u64,
    ) -> Result<Vec<u8>, NativeNodeFailure> {
        let failure = |message: String| NativeNodeFailure {
            code: "provider_video_preparation_failed".to_owned(),
            message,
            kind: if self.cancellation.is_cancelled() {
                NativeNodeFailureKind::Interrupted
            } else {
                NativeNodeFailureKind::Failure
            },
            retryable: false,
        };
        self.cancellation
            .check()
            .map_err(|error| failure(error.to_string()))?;
        let count = match mask.descriptor().shape() {
            [_, _] => 1,
            [count, _, _] => *count,
            _ => {
                return Err(failure(
                    "source API mask must have rank two or three".to_owned(),
                ));
            }
        };
        if count != source_frame_count || count == 0 {
            return Err(failure(format!(
                "Custom alpha video frame count ({count}) does not match the source video frame count ({source_frame_count}). The Beeble API requires one mask per source frame."
            )));
        }
        let service = self
            .component_h264_mp4_backing_service()
            .map_err(|error| failure(error.to_string()))?;
        let compute = self
            .compute_session()
            .map_err(|error| failure(error.to_string()))?;
        let execution = compute
            .execution_context(self)
            .map_err(|error| failure(error.to_string()))?;
        let images = sim_media::prepare_api_mask_batch_with_context(
            compute.backend(),
            &execution,
            mask,
            maximum_pixels,
            PngLimits::default(),
        )
        .map_err(|error| failure(error.to_string()))?;
        let video = Arc::new(
            NativeVideoPayload::checked_cancellable(
                images.tensor().clone(),
                frame_rate.0,
                frame_rate.1,
                NativeVideoBitDepth::Eight,
                None,
                None,
                BTreeMap::new(),
                &self.cancellation,
            )
            .map_err(|error| failure(error.to_string()))?,
        );
        let encoded_frame_rate = sim_media::source_rounded_rational_frame_rate(frame_rate)
            .map_err(|error| failure(error.to_string()))?;
        let request = NativeComponentH264Mp4BackingRequest::checked(video.clone())
            .map_err(|error| failure(error.to_string()))?;
        let result = service
            .encode_backing(request, &execution)
            .await
            .map_err(|error| {
                let mut failure = failure(error.to_string());
                if matches!(error, NativeComponentH264Mp4BackingServiceError::Cancelled) {
                    failure.kind = NativeNodeFailureKind::Interrupted;
                }
                failure
            })?;
        self.cancellation
            .check()
            .map_err(|error| failure(error.to_string()))?;
        result
            .validate()
            .map_err(|error| failure(error.to_string()))?;
        let encoded = result
            .encoded()
            .ok_or_else(|| failure("codec returned video components".to_owned()))?;
        if encoded.source_video_sha256() != video.semantic_digest_sha256()
            || encoded.dimensions() != video.dimensions()
            || encoded.frame_count() != source_frame_count
            || encoded.frame_rate() != encoded_frame_rate
            || encoded.bit_depth() != NativeVideoBitDepth::Eight
        {
            return Err(failure("codec returned unrelated alpha video".to_owned()));
        }
        let bytes = encoded
            .bytes()
            .contiguous_bytes()
            .map_err(|error| failure(error.to_string()))?
            .to_vec();
        self.cancellation
            .check()
            .map_err(|error| failure(error.to_string()))?;
        Ok(bytes)
    }

    fn provider_image_preparation_failure(&self, error: impl fmt::Display) -> NativeNodeFailure {
        NativeNodeFailure {
            code: "provider_image_preparation_failed".to_owned(),
            message: error.to_string(),
            kind: if self.cancellation.is_cancelled() {
                NativeNodeFailureKind::Interrupted
            } else {
                NativeNodeFailureKind::Failure
            },
            retryable: false,
        }
    }

    pub fn prepare_image_preview(
        &self,
        image: &ImageTensor,
        filename_prefix: &str,
    ) -> Result<NativePreparedImagePreview, NativeImagePreviewError> {
        self.cancellation
            .check()
            .map_err(|_| NativeImagePreviewError::Effect(NativeEffectServiceError::Cancelled))?;
        let compute = self.compute_session()?;
        let execution_context = compute.execution_context(self)?;
        let effects_service = self.prepared_effects()?;
        let (batch, height, width, channels) = image.dimensions()?;
        let pixels = image.as_f32_slice()?;
        let metadata = BTreeMap::new();
        let mut effects = Vec::new();
        let mut ui_images = Vec::new();
        for batch_index in 0..batch {
            let prepared = (|| {
                self.cancellation.check().map_err(|_| {
                    NativeImagePreviewError::Effect(NativeEffectServiceError::Cancelled)
                })?;
                let encoded = encode_png_frame_with_policy_and_context(
                    compute.backend(),
                    &execution_context,
                    pixels,
                    batch,
                    height,
                    width,
                    channels,
                    batch_index,
                    &metadata,
                    MetadataWritePolicy {
                        metadata_enabled: false,
                    },
                    PngLimits::default(),
                )?;
                let output_index = u32::try_from(batch_index)
                    .map_err(|_| NativeImagePreviewError::DimensionOverflow)?;
                let request = NativeOutputEffectRequest::checked(
                    NativeOutputNamespace::Temporary,
                    filename_prefix,
                    "png",
                    output_index,
                    NativeOutputShape::Image {
                        width: u32::try_from(width)
                            .map_err(|_| NativeImagePreviewError::DimensionOverflow)?,
                        height: u32::try_from(height)
                            .map_err(|_| NativeImagePreviewError::DimensionOverflow)?,
                    },
                    Arc::from(encoded),
                    effects_service.maximum_output_bytes(),
                )?;
                effects_service
                    .prepare_output(request, &self.cancellation)
                    .map_err(NativeImagePreviewError::from)
            })();
            let effect = match prepared {
                Ok(effect) => effect,
                Err(error) => {
                    for effect in effects.iter().rev() {
                        effects_service.rollback_prepared(effect)?;
                    }
                    return Err(error);
                }
            };
            ui_images.push(json!({
                "transaction_id": effect.transaction_id(),
                "batch_index": batch_index,
                "type": "temp",
            }));
            effects.push(effect);
        }
        if self.cancellation.is_cancelled() {
            for effect in effects.iter().rev() {
                effects_service.rollback_prepared(effect)?;
            }
            return Err(NativeImagePreviewError::Effect(
                NativeEffectServiceError::Cancelled,
            ));
        }
        Ok(NativePreparedImagePreview {
            effects,
            ui: json!({"images": ui_images, "animated": [false]}),
        })
    }

    pub fn validate(&self) -> Result<(), NativeNodeContractError> {
        if self.prompt_id.0.is_nil()
            || self.attempt_id.0.is_nil()
            || self.handle_store.attempt_id() != self.attempt_id
        {
            return Err(NativeNodeContractError::InvalidNodeContext);
        }
        validate_identifier("native node context node ID", &self.node_id.0)?;
        self.handle_store.identity().validate()?;
        for identity in self
            .services
            .assets
            .as_deref()
            .map(NativeAssetResolver::identity)
            .into_iter()
            .chain(
                self.services
                    .effects
                    .as_deref()
                    .map(NativePreparedEffectService::identity),
            )
            .chain(
                self.services
                    .compute
                    .as_ref()
                    .map(NativeNodeComputeSession::identity),
            )
            .chain(
                self.services
                    .text_generation
                    .as_deref()
                    .map(NativeTextGenerationService::identity),
            )
            .chain(
                self.services
                    .model_resources
                    .as_deref()
                    .map(NativeModelResourceService::identity),
            )
        {
            if !identity.matches(self.attempt_id, &self.node_id) {
                return Err(NativeNodeContractError::InvalidNodeServiceIdentity);
            }
        }
        if let Some(compute) = &self.services.compute {
            if compute.scratch_binding != self.scratch.binding_identity() {
                return Err(NativeNodeContractError::InvalidComputeSession);
            }
            compute
                .backend
                .validate_scratch_reservation(&self.scratch)
                .map_err(|_| NativeNodeContractError::InvalidComputeSession)?;
        }
        Ok(())
    }
}

pub fn native_text_generation_transaction(
    context: &NativeNodeContext,
    seed: u64,
) -> Result<RngTransaction, RngError> {
    if context.cancellation.is_cancelled() {
        return Err(RngError::Cancelled);
    }
    let address = RngStreamAddress::new(
        context.prompt_id.0.to_string(),
        context.attempt_id.0.to_string(),
        context.node_id.0.clone(),
        0,
        NATIVE_TEXT_GENERATION_RNG_PHASE,
        0,
        0,
        RetryRngPolicy::Replay,
    )?;
    RngStream::new(
        RngProfileVersion::V2,
        RngAlgorithm::Philox4x32_10,
        seed,
        address,
    )?
    .begin(None)
}

impl NativeNodeComputeSession {
    pub fn execution_context<'a>(
        &self,
        context: &'a NativeNodeContext,
    ) -> Result<ExecutionContext<'a>, NativeNodeContractError> {
        if !self.identity.matches(context.attempt_id, &context.node_id)
            || self.scratch_binding != context.scratch.binding_identity()
        {
            return Err(NativeNodeContractError::InvalidComputeSession);
        }
        self.backend
            .validate_scratch_reservation(&context.scratch)
            .map_err(|_| NativeNodeContractError::InvalidComputeSession)?;
        Ok(ExecutionContext {
            stream: self.stream,
            scratch: context.scratch.clone(),
            rng_phase: None,
            cancellation: &context.cancellation,
        })
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeCacheDependencies {
    pub artifact_digests: BTreeMap<String, String>,
    pub plugin_digest: Option<String>,
    pub rng_phase: Option<String>,
}

impl NativeCacheDependencies {
    pub fn validate(&self) -> Result<(), NativeNodeContractError> {
        for (identifier, digest) in &self.artifact_digests {
            validate_identifier("cache artifact identifier", identifier)?;
            if !valid_sha256(digest) {
                return Err(NativeNodeContractError::InvalidDigest);
            }
        }
        if let Some(digest) = &self.plugin_digest
            && !valid_sha256(digest)
        {
            return Err(NativeNodeContractError::InvalidDigest);
        }
        if let Some(phase) = &self.rng_phase {
            validate_identifier("cache RNG phase", phase)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativePreparedEffectRequest {
    service_id: Uuid,
    transaction_id: Uuid,
    kind: NativePreparedEffectKind,
    request_digest_sha256: String,
}

impl NativePreparedEffectRequest {
    pub fn checked(
        service_id: Uuid,
        transaction_id: Uuid,
        kind: NativePreparedEffectKind,
        request_digest_sha256: impl Into<String>,
    ) -> Result<Self, NativeNodeContractError> {
        let request = Self {
            service_id,
            transaction_id,
            kind,
            request_digest_sha256: request_digest_sha256.into(),
        };
        request.validate()?;
        Ok(request)
    }

    pub const fn service_id(&self) -> Uuid {
        self.service_id
    }

    pub const fn transaction_id(&self) -> Uuid {
        self.transaction_id
    }

    pub const fn kind(&self) -> NativePreparedEffectKind {
        self.kind
    }

    pub fn request_digest_sha256(&self) -> &str {
        &self.request_digest_sha256
    }

    pub fn validate(&self) -> Result<(), NativeNodeContractError> {
        if self.service_id.is_nil()
            || self.transaction_id.is_nil()
            || !valid_sha256(&self.request_digest_sha256)
        {
            return Err(NativeNodeContractError::InvalidEffectRequest);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum NativeNodeOutcome {
    Values {
        outputs: Vec<NativeValue>,
        ui: Option<Value>,
        effects: Vec<NativePreparedEffectRequest>,
    },
    Blocked {
        reason: String,
    },
    Expansion {
        prompt: ApiPrompt,
        output_node: NodeId,
    },
}

impl NativeNodeOutcome {
    pub fn validate(&self) -> Result<(), NativeNodeContractError> {
        match self {
            Self::Values {
                outputs,
                ui,
                effects,
            } => {
                if outputs.len() > MAX_PORTS || effects.len() > MAX_PORTS {
                    return Err(NativeNodeContractError::InvalidOutcome);
                }
                for output in outputs {
                    output.validate()?;
                }
                if let Some(ui) = ui {
                    let encoded = serde_json::to_vec(ui)
                        .map_err(NativeNodeContractError::EncodePresentationValue)?;
                    if encoded.len() > MAX_UNKNOWN_BYTES {
                        return Err(NativeNodeContractError::PresentationValueTooLarge);
                    }
                }
                for effect in effects {
                    effect.validate()?;
                }
                Ok(())
            }
            Self::Blocked { reason } => {
                validate_text("blocked reason", reason, MAX_TEXT_BYTES, false)
            }
            Self::Expansion {
                prompt,
                output_node,
            } => {
                validate_identifier("expansion output node", &output_node.0)?;
                if prompt.0.is_empty() || !prompt.0.contains_key(output_node) {
                    return Err(NativeNodeContractError::InvalidExpansion);
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeNodeFailureKind {
    Failure,
    Interrupted,
}

#[derive(Clone, Debug, Error, Eq, PartialEq, Serialize, Deserialize)]
#[error("{code}: {message}")]
#[serde(deny_unknown_fields)]
pub struct NativeNodeFailure {
    pub code: String,
    pub message: String,
    pub kind: NativeNodeFailureKind,
    pub retryable: bool,
}

impl NativeNodeFailure {
    pub fn validate(&self) -> Result<(), NativeNodeContractError> {
        validate_identifier("node failure code", &self.code)?;
        validate_text("node failure message", &self.message, MAX_TEXT_BYTES, false)
    }
}

pub trait NativeNode: Send + Sync {
    fn class_type(&self) -> &str;
    fn implementation_version(&self) -> &str;

    fn implementation_namespace(&self) -> &str {
        "zed.native_rust"
    }

    fn demanded_lazy_inputs(
        &self,
        _context: &NativeNodeContext,
        _available_inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<BTreeSet<String>, NativeNodeFailure> {
        Ok(BTreeSet::new())
    }

    fn cache_change_token(
        &self,
        _inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        Ok("stable".to_owned())
    }

    fn cache_dependencies(
        &self,
        _context: &NativeNodeContext,
        _inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        Ok(NativeCacheDependencies::default())
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>>;
}

pub type NativeNodeBindingsFactory =
    fn() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError>;

#[derive(Clone)]
pub enum NativeNodeBinding {
    Executable {
        feature_id: String,
        descriptor: NativeNodeDescriptor,
        presentation: NativeNodePresentation,
        node: Arc<dyn NativeNode>,
    },
    ProviderRequired {
        feature_id: String,
        descriptor: NativeNodeDescriptor,
        presentation: NativeNodePresentation,
        provider: String,
        reason: String,
    },
    Unavailable {
        feature_id: String,
        descriptor: NativeNodeDescriptor,
        presentation: NativeNodePresentation,
        reason: String,
    },
}

impl fmt::Debug for NativeNodeBinding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeNodeBinding")
            .field("disposition", &self.disposition())
            .field("feature_id", &self.feature_id())
            .field("class_type", &self.descriptor().class_type)
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeNodeBindingDisposition {
    Executable,
    ProviderRequired,
    Unavailable,
}

impl NativeNodeBinding {
    pub fn feature_id(&self) -> &str {
        match self {
            Self::Executable { feature_id, .. }
            | Self::ProviderRequired { feature_id, .. }
            | Self::Unavailable { feature_id, .. } => feature_id,
        }
    }

    pub fn descriptor(&self) -> &NativeNodeDescriptor {
        match self {
            Self::Executable { descriptor, .. }
            | Self::ProviderRequired { descriptor, .. }
            | Self::Unavailable { descriptor, .. } => descriptor,
        }
    }

    pub fn presentation(&self) -> &NativeNodePresentation {
        match self {
            Self::Executable { presentation, .. }
            | Self::ProviderRequired { presentation, .. }
            | Self::Unavailable { presentation, .. } => presentation,
        }
    }

    pub const fn disposition(&self) -> NativeNodeBindingDisposition {
        match self {
            Self::Executable { .. } => NativeNodeBindingDisposition::Executable,
            Self::ProviderRequired { .. } => NativeNodeBindingDisposition::ProviderRequired,
            Self::Unavailable { .. } => NativeNodeBindingDisposition::Unavailable,
        }
    }

    pub fn validate(&self) -> Result<(), NativeNodeContractError> {
        validate_feature_id(self.feature_id())?;
        self.descriptor().validate()?;
        self.presentation().validate()?;
        if self
            .descriptor()
            .outputs
            .iter()
            .map(|output| output.name.as_str())
            .ne(self.presentation().output_names.iter().map(String::as_str))
        {
            return Err(NativeNodeContractError::InvalidPresentationOutputs);
        }
        match self {
            Self::Executable {
                descriptor, node, ..
            } => {
                if descriptor.class_type != node.class_type()
                    || descriptor.implementation_version != node.implementation_version()
                    || node.implementation_namespace().trim().is_empty()
                {
                    return Err(NativeNodeContractError::BindingImplementationMismatch);
                }
                Ok(())
            }
            Self::ProviderRequired {
                provider, reason, ..
            } => {
                validate_identifier("native provider", provider)?;
                validate_text("provider reason", reason, MAX_TEXT_BYTES, false)
            }
            Self::Unavailable { reason, .. } => {
                validate_text("unavailable reason", reason, MAX_TEXT_BYTES, false)
            }
        }
    }
}

pub fn validate_generated_family_bindings(
    bindings: &[NativeNodeBinding],
    descriptor_ids: &[&str],
) -> Result<(), NativeNodeContractError> {
    let expected = descriptor_ids.iter().copied().collect::<BTreeSet<_>>();
    if expected.len() != descriptor_ids.len() {
        return Err(NativeNodeContractError::DuplicateGeneratedDescriptor);
    }
    let mut actual = BTreeSet::new();
    for binding in bindings {
        binding.validate()?;
        if !actual.insert(binding.descriptor().class_type.as_str()) {
            return Err(NativeNodeContractError::DuplicateGeneratedDescriptor);
        }
    }
    if actual != expected {
        return Err(NativeNodeContractError::GeneratedBindingMismatch);
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum NativeNodeContractError {
    #[error("native node contract schema {0} is unsupported")]
    UnsupportedContractSchema(u16),
    #[error("native opaque handle schema {0} is unsupported")]
    UnsupportedHandleSchema(u16),
    #[error("native node contract {field} is invalid")]
    InvalidText { field: &'static str },
    #[error("native node feature ID is invalid")]
    InvalidFeatureId,
    #[error("native opaque handle generation must be nonzero")]
    InvalidHandleGeneration,
    #[error("native opaque handle type identity is invalid")]
    InvalidHandleType,
    #[error("native opaque handle store identity is nil")]
    InvalidHandleStoreIdentity,
    #[error("native node SHA-256 digest is invalid")]
    InvalidDigest,
    #[error("native node number must be finite")]
    NonFiniteNumber,
    #[error("native node value nesting exceeds its limit")]
    ValueNestingTooDeep,
    #[error("native node list exceeds its value limit")]
    TooManyListValues,
    #[error("native preserved unknown value exceeds its byte limit")]
    PreservedUnknownTooLarge,
    #[error("native structured value is invalid")]
    InvalidStructuredValue,
    #[error("native preserved unknown value could not be encoded: {0}")]
    EncodePreservedUnknown(serde_json::Error),
    #[error("native UI presentation value could not be encoded: {0}")]
    EncodePresentationValue(serde_json::Error),
    #[error("native UI presentation value exceeds its byte limit")]
    PresentationValueTooLarge,
    #[error("native type union is empty, duplicated, unsorted, or ambiguous")]
    InvalidTypeUnion,
    #[error("native dynamic input descriptor is invalid")]
    InvalidDynamicInput,
    #[error("native node descriptor has an invalid port count")]
    InvalidPortCount,
    #[error("native node source schema is invalid: {0}")]
    InvalidSourceSchema(String),
    #[error("native node descriptor repeats port `{0}`")]
    DuplicatePort(String),
    #[error("native node presentation repeats search alias `{0}`")]
    DuplicateSearchAlias(String),
    #[error("native node presentation output names do not match its descriptor")]
    InvalidPresentationOutputs,
    #[error("native prepared effect request is invalid")]
    InvalidEffectRequest,
    #[error("native node outcome exceeds limits or is malformed")]
    InvalidOutcome,
    #[error("native node expansion is empty or lacks its output node")]
    InvalidExpansion,
    #[error("native node binding does not match its executable implementation")]
    BindingImplementationMismatch,
    #[error("native node context does not match its attempt-local handle store")]
    InvalidNodeContext,
    #[error("native node service identity does not match its attempt and node")]
    InvalidNodeServiceIdentity,
    #[error("native node compute session does not match its backend and scratch reservation")]
    InvalidComputeSession,
    #[error("native stored object metadata does not match its payload contract")]
    InvalidStoredObject,
    #[error("generated native node descriptors contain a duplicate")]
    DuplicateGeneratedDescriptor,
    #[error("generated native node bindings do not exactly match descriptor IDs")]
    GeneratedBindingMismatch,
}

fn validate_identifier(field: &'static str, value: &str) -> Result<(), NativeNodeContractError> {
    validate_text(field, value, MAX_IDENTIFIER_BYTES, false)
}

fn validate_text(
    field: &'static str,
    value: &str,
    maximum_bytes: usize,
    allow_empty: bool,
) -> Result<(), NativeNodeContractError> {
    if (!allow_empty && value.is_empty())
        || value.len() > maximum_bytes
        || value.contains('\0')
        || value.chars().any(char::is_control)
    {
        return Err(NativeNodeContractError::InvalidText { field });
    }
    Ok(())
}

fn validate_workflow_text(field: &'static str, value: &str) -> Result<(), NativeNodeContractError> {
    if value.len() > MAX_TEXT_BYTES
        || value.contains('\0')
        || value
            .chars()
            .any(|character| character.is_control() && !matches!(character, '\n' | '\r' | '\t'))
    {
        return Err(NativeNodeContractError::InvalidText { field });
    }
    Ok(())
}

fn validate_feature_id(value: &str) -> Result<(), NativeNodeContractError> {
    let suffix = value
        .strip_prefix("COMFY-NODE-")
        .ok_or(NativeNodeContractError::InvalidFeatureId)?;
    let digits = suffix.strip_prefix("INACTIVE-").unwrap_or(suffix);
    if digits.len() != 4 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(NativeNodeContractError::InvalidFeatureId);
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn gcd_u64(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn output_request_digest(
    namespace: NativeOutputNamespace,
    filename_prefix: &str,
    extension: &str,
    batch_index: u32,
    shape: NativeOutputShape,
    media_kind: NativeOutputMediaKind,
    media_type: &str,
    content: &[u8],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"zed.sim.native-output-effect.v2");
    hasher.update([match namespace {
        NativeOutputNamespace::Output => 0,
        NativeOutputNamespace::Temporary => 1,
    }]);
    hasher.update((filename_prefix.len() as u64).to_le_bytes());
    hasher.update(filename_prefix.as_bytes());
    hasher.update((extension.len() as u64).to_le_bytes());
    hasher.update(extension.as_bytes());
    hasher.update(batch_index.to_le_bytes());
    match shape {
        NativeOutputShape::File => hasher.update([0]),
        NativeOutputShape::Image { width, height } => {
            hasher.update([1]);
            hasher.update(width.to_le_bytes());
            hasher.update(height.to_le_bytes());
        }
    }
    hasher.update([match media_kind {
        NativeOutputMediaKind::Image => 0,
        NativeOutputMediaKind::Animation => 1,
        NativeOutputMediaKind::Video => 2,
        NativeOutputMediaKind::Audio => 3,
        NativeOutputMediaKind::ThreeD => 4,
        NativeOutputMediaKind::Text => 5,
        NativeOutputMediaKind::Json => 6,
        NativeOutputMediaKind::Binary => 7,
    }]);
    hasher.update((media_type.len() as u64).to_le_bytes());
    hasher.update(media_type.as_bytes());
    hasher.update((content.len() as u64).to_le_bytes());
    hasher.update(content);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_sampler::{NativeNoisePayload, NativeSamplerPayload};
    use sim_tensor::CpuWorkspaceAuthority;
    use std::sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    };

    #[test]
    fn source_custom_asset_validator_admission_is_pinned_and_typed()
    -> Result<(), Box<dyn std::error::Error>> {
        let source = crate::built_in_source_schema("LoadLatent")?.bind_execution_ports(
            &["latent".to_owned()],
            &[],
            &["latent".to_owned()],
        )?;
        let input = source.inputs.first().ok_or("LoadLatent input missing")?;
        let filename = |name: &str| NativeValue::PreservedUnknown {
            type_name: "COMBO".to_owned(),
            value: json!(name),
        };
        for value in [filename("existing.latent"), filename("existing.txt")] {
            assert!(!native_value_matches_input_schema(&value, input));
            assert!(native_value_matches_source_input_schema(
                &value,
                "LoadLatent",
                &source,
                input
            ));
        }
        for value in [
            NativeValue::Primitive {
                value: NativePrimitive::String("existing.latent".to_owned()),
            },
            NativeValue::PreservedUnknown {
                type_name: "OTHER".to_owned(),
                value: json!("existing.latent"),
            },
            NativeValue::PreservedUnknown {
                type_name: "COMBO".to_owned(),
                value: json!(17),
            },
            NativeValue::List {
                values: vec![
                    filename("existing.latent"),
                    NativeValue::PreservedUnknown {
                        type_name: "COMBO".to_owned(),
                        value: json!(false),
                    },
                ],
            },
        ] {
            assert!(!native_value_matches_source_input_schema(
                &value,
                "LoadLatent",
                &source,
                input
            ));
        }
        let value = filename("existing.txt");
        assert!(native_value_matches_source_input_schema(
            &NativeValue::List {
                values: vec![value.clone(), filename("another.latent")]
            },
            "LoadLatent",
            &source,
            input
        ));
        assert!(!native_value_matches_source_input_schema(
            &value, "Other", &source, input
        ));
        let mut wrong_port = input.clone();
        wrong_port.name = "other".to_owned();
        assert!(!native_value_matches_source_input_schema(
            &value,
            "LoadLatent",
            &source,
            &wrong_port
        ));
        for mutation in 0..7 {
            let mut changed = source.clone();
            match mutation {
                0 => changed.node.feature_id = Some("COMFY-NODE-0368".to_owned()),
                1 => changed.node.definition_sha256 = Some("0".repeat(64)),
                2 => changed.node.provenance = crate::NativeSchemaProvenance::CompatibilityV1,
                3 => changed
                    .inputs
                    .first_mut()
                    .ok_or("input missing")?
                    .choices
                    .clear(),
                4 => {
                    let choices = &mut changed.inputs.first_mut().ok_or("input missing")?.choices;
                    let Some(NativeSchemaValue::PreservedExpression { sha256, .. }) =
                        choices.first_mut()
                    else {
                        return Err("choice expression missing".into());
                    };
                    *sha256 = "0".repeat(64);
                }
                5 => {
                    changed
                        .outputs
                        .first_mut()
                        .ok_or("output missing")?
                        .source_type_name = "OTHER".to_owned()
                }
                _ => changed.node.feature_id = None,
            }
            assert!(!native_value_matches_source_input_schema(
                &value,
                "LoadLatent",
                &changed,
                changed.inputs.first().ok_or("input missing")?
            ));
        }
        let mut finite = input.clone();
        finite.choices = vec![NativeSchemaValue::String {
            value: "allowed".to_owned(),
        }];
        for (text, expected) in [("allowed", true), ("outside", false)] {
            assert_eq!(
                native_value_matches_source_input_schema(
                    &NativeValue::Primitive {
                        value: NativePrimitive::String(text.to_owned())
                    },
                    "FiniteCombo",
                    &source,
                    &finite
                ),
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn native_model_service_errors_fit_the_result_size_budget() {
        assert!(std::mem::size_of::<NativeModelResourceServiceError>() < 128);
    }

    #[derive(Debug)]
    struct TestResolvedPayloadRetention;

    impl NativeResolvedPayloadRetention for TestResolvedPayloadRetention {}

    struct TestHandleStore {
        identity: NativeHandleStoreIdentity,
        attempt_id: AttemptId,
        next_identifier: AtomicU64,
        values: Mutex<BTreeMap<String, Arc<NativeStoredPayload>>>,
    }

    struct TestPreviewEffectService {
        identity: NativeNodeServiceIdentity,
        fail_on_ordinal: Option<u64>,
        next_ordinal: AtomicU64,
        prepared: Mutex<Vec<NativePreparedEffectRequest>>,
        rolled_back: Mutex<Vec<Uuid>>,
    }

    #[derive(Debug)]
    struct TestTextGenerationService {
        identity: NativeNodeServiceIdentity,
        calls: Arc<AtomicU64>,
        cancellation_after_generate: Option<CancellationToken>,
        return_invalid_output: bool,
    }

    impl NativeTextGenerationService for TestTextGenerationService {
        fn identity(&self) -> &NativeNodeServiceIdentity {
            &self.identity
        }

        fn generate(
            &self,
            _context: &NativeNodeContext,
            request: &NativeMultimodalTextGenerationRequest,
        ) -> Result<NativeTextGenerationServiceOutput, NativeTextGenerationServiceError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if let Some(cancellation) = &self.cancellation_after_generate {
                cancellation.cancel();
            }
            if self.return_invalid_output {
                return Ok(NativeTextGenerationServiceOutput {
                    text: "invalid".to_owned(),
                    generated_token_count: request.maximum_new_tokens() + 1,
                });
            }
            NativeTextGenerationServiceOutput::checked(
                format!("generated:{}", request.formatted_prompt()),
                1,
                request.maximum_new_tokens(),
            )
        }
    }

    #[test]
    fn native_model_resource_service_request_is_path_free_bounded_and_object_safe()
    -> Result<(), Box<dyn std::error::Error>> {
        fn assert_object_safe(_: Option<&dyn NativeModelResourceService>) {}

        assert_object_safe(None);
        let request = NativeModelResourceRequest::checked(
            NativeModelResourceRole::StyleModel,
            "style_models",
            vec!["nested/style.safetensors".to_owned()],
        )?;
        assert_eq!(request.role(), NativeModelResourceRole::StyleModel);
        assert_eq!(request.folder_category(), "style_models");
        assert_eq!(request.source_names(), ["nested/style.safetensors"]);

        for names in [
            Vec::new(),
            vec!["a".to_owned(), "a".to_owned()],
            vec![
                "a".to_owned(),
                "b".to_owned(),
                "c".to_owned(),
                "d".to_owned(),
                "e".to_owned(),
            ],
        ] {
            assert!(matches!(
                NativeModelResourceRequest::checked(
                    NativeModelResourceRole::StyleModel,
                    "style_models",
                    names,
                ),
                Err(NativeModelResourceServiceError::InvalidRequest)
            ));
        }
        for name in ["../style.safetensors", "/style.safetensors", "C:\\style.pt"] {
            assert!(matches!(
                NativeModelResourceRequest::checked(
                    NativeModelResourceRole::StyleModel,
                    "style_models",
                    vec![name.to_owned()],
                ),
                Err(NativeModelResourceServiceError::InvalidRequest)
            ));
        }
        Ok(())
    }

    #[test]
    fn output_effect_request_admits_only_safe_nested_relative_prefixes()
    -> Result<(), Box<dyn std::error::Error>> {
        let request = |prefix: &str| {
            NativeOutputEffectRequest::checked(
                NativeOutputNamespace::Output,
                prefix,
                "webm",
                0,
                NativeOutputShape::File,
                Arc::from([1_u8, 2, 3]),
                1024,
            )
        };
        let nested = request("video/ComfyUI")?;
        assert_eq!(nested.filename_prefix(), "video/ComfyUI");
        assert_eq!(nested.media_kind(), NativeOutputMediaKind::Video);
        assert_eq!(nested.media_type(), "video/webm");
        let mp4 = NativeOutputEffectRequest::checked_media(
            NativeOutputNamespace::Output,
            "video/ComfyUI",
            "mp4",
            "video/mp4",
            NativeOutputMediaKind::Video,
            0,
            NativeOutputShape::File,
            Arc::from([1_u8, 2, 3]),
            1024,
        )?;
        assert_eq!(mp4.media_kind(), NativeOutputMediaKind::Video);
        assert_eq!(mp4.media_type(), "video/mp4");
        assert_ne!(mp4.request_digest_sha256(), nested.request_digest_sha256());
        assert_ne!(
            nested.request_digest_sha256(),
            request("video/Other")?.request_digest_sha256()
        );
        assert!(
            NativeOutputEffectRequest::checked_media(
                NativeOutputNamespace::Output,
                "video/ComfyUI",
                "webm",
                "image/png",
                NativeOutputMediaKind::Video,
                0,
                NativeOutputShape::File,
                Arc::from([1_u8, 2, 3]),
                1024,
            )
            .is_err()
        );
        for invalid in [
            "/video/ComfyUI",
            "video/ComfyUI/",
            "video//ComfyUI",
            ".",
            "..",
            "video/./ComfyUI",
            "video/../ComfyUI",
            "video\\ComfyUI",
            "C:/video/ComfyUI",
            "video\0ComfyUI",
            "video\nComfyUI",
        ] {
            assert!(
                request(invalid).is_err(),
                "accepted unsafe prefix {invalid:?}"
            );
        }
        assert!(request(&"a".repeat(MAX_IDENTIFIER_BYTES + 1)).is_err());
        Ok(())
    }

    #[derive(Debug)]
    struct TestShaderExecutor;

    #[derive(Debug)]
    struct TestLtxvPreprocessService {
        identity: NativeLtxvPreprocessServiceIdentity,
        compression: AtomicU64,
    }

    #[derive(Debug)]
    struct TestWebmEncodeService {
        identity: NativeWebmEncodeServiceIdentity,
        backend: Arc<CpuBackend>,
        metadata: Mutex<Vec<(String, String)>>,
    }

    #[derive(Debug)]
    struct TestComponentH264Mp4BackingService {
        identity: NativeComponentH264Mp4BackingServiceIdentity,
        backend: Arc<CpuBackend>,
    }

    impl NativeLtxvPreprocessService for TestLtxvPreprocessService {
        fn identity(&self) -> &NativeLtxvPreprocessServiceIdentity {
            &self.identity
        }

        fn preprocess_image(
            &self,
            image: &ImageTensor,
            compression: u8,
            context: &ExecutionContext<'_>,
        ) -> BoxFuture<'static, Result<ImageTensor, NativeLtxvPreprocessServiceError>> {
            self.compression
                .store(u64::from(compression), Ordering::SeqCst);
            let image = image.clone();
            let cancelled = context.cancellation.is_cancelled();
            Box::pin(async move {
                if cancelled {
                    Err(NativeLtxvPreprocessServiceError::Cancelled)
                } else {
                    Ok(image)
                }
            })
        }
    }

    impl NativeWebmEncodeService for TestWebmEncodeService {
        fn identity(&self) -> &NativeWebmEncodeServiceIdentity {
            &self.identity
        }

        fn encode_webm(
            &self,
            request: NativeWebmEncodeRequest,
            context: &ExecutionContext<'_>,
        ) -> BoxFuture<'static, Result<NativeEncodedWebm, NativeWebmEncodeServiceError>> {
            let (images, codec, frame_rate, _crf, metadata) = request.into_parts();
            *self.metadata.lock().expect("test metadata mutex poisoned") = metadata;
            let result = (|| {
                context
                    .check()
                    .map_err(|_| NativeWebmEncodeServiceError::Cancelled)?;
                let (frame_count, height, width, _) = images
                    .dimensions()
                    .map_err(|error| NativeWebmEncodeServiceError::Execution(error.to_string()))?;
                let encoded = b"HPPPT";
                let descriptor = sim_tensor::TensorDescriptor::contiguous(
                    vec![encoded.len() as u64],
                    DType::U8,
                    DeviceId::CPU,
                    context.stream,
                )
                .map_err(|error| NativeWebmEncodeServiceError::Execution(error.to_string()))?;
                let (bytes, _) = self
                    .backend
                    .upload_bytes(descriptor, encoded, context)
                    .map_err(|error| NativeWebmEncodeServiceError::Execution(error.to_string()))?;
                NativeEncodedWebm::checked(
                    bytes,
                    Sha256::digest(encoded).into(),
                    codec,
                    (width, height),
                    frame_rate,
                    frame_count,
                    NativeVideoPixelFormat::Yuv420p,
                    NativeVideoBitDepth::Eight,
                    false,
                )
            })();
            Box::pin(async move { result })
        }
    }

    impl NativeComponentH264Mp4BackingService for TestComponentH264Mp4BackingService {
        fn identity(&self) -> &NativeComponentH264Mp4BackingServiceIdentity {
            &self.identity
        }

        fn encode_backing(
            &self,
            request: NativeComponentH264Mp4BackingRequest,
            context: &ExecutionContext<'_>,
        ) -> BoxFuture<'static, Result<NativeVideoPayload, NativeComponentH264Mp4BackingServiceError>>
        {
            let video = request.into_video();
            let result = (|| {
                context
                    .check()
                    .map_err(|_| NativeComponentH264Mp4BackingServiceError::Cancelled)?;
                let encoded = match video.bit_depth() {
                    NativeVideoBitDepth::Eight => b"HMP4".as_slice(),
                    NativeVideoBitDepth::Ten => b"H10P4".as_slice(),
                };
                let descriptor = sim_tensor::TensorDescriptor::contiguous(
                    vec![encoded.len() as u64],
                    DType::U8,
                    DeviceId::CPU,
                    context.stream,
                )
                .map_err(|error| {
                    NativeComponentH264Mp4BackingServiceError::Execution(error.to_string())
                })?;
                let (bytes, _) = self
                    .backend
                    .upload_bytes(descriptor, encoded, context)
                    .map_err(|error| {
                        NativeComponentH264Mp4BackingServiceError::Execution(error.to_string())
                    })?;
                let components = video
                    .components()
                    .ok_or(NativeComponentH264Mp4BackingServiceError::InvalidRequest)?;
                let frame_count = components
                    .frames()
                    .descriptor()
                    .shape()
                    .first()
                    .copied()
                    .ok_or(NativeComponentH264Mp4BackingServiceError::InvalidRequest)?;
                NativeVideoPayload::checked_h264_mp4_from_component(
                    video.as_ref(),
                    bytes,
                    Sha256::digest(encoded).into(),
                    video.dimensions(),
                    video.frame_rate(),
                    frame_count,
                )
                .map_err(|_| NativeComponentH264Mp4BackingServiceError::InvalidProjection)
            })();
            Box::pin(async move { result })
        }
    }

    impl NativeShaderExecutor for TestShaderExecutor {
        fn configuration_identity(&self) -> String {
            "test-shader-v1".to_owned()
        }

        fn execute(
            &self,
            request: &NativeShaderRequest,
            backend: &CpuBackend,
            context: &ExecutionContext<'_>,
        ) -> Result<NativeShaderResult, NativeShaderError> {
            context
                .cancellation
                .check()
                .map_err(|_| NativeShaderError::Cancelled)?;
            let batch = request
                .images
                .first()
                .ok_or_else(|| NativeShaderError::Bounds("missing test image".to_owned()))?
                .dimensions()?
                .0;
            let pixel_count = usize::try_from(
                batch
                    .checked_mul(u64::from(request.width))
                    .and_then(|value| value.checked_mul(u64::from(request.height)))
                    .ok_or_else(|| {
                        NativeShaderError::Bounds("test output overflowed".to_owned())
                    })?,
            )
            .map_err(|_| NativeShaderError::Bounds("test output is too large".to_owned()))?;
            let output_values = pixel_count
                .checked_mul(4)
                .ok_or_else(|| NativeShaderError::Bounds("test output overflowed".to_owned()))?;
            let mut values = Vec::new();
            values.try_reserve_exact(output_values).map_err(|error| {
                NativeShaderError::Bounds(format!("test output allocation failed: {error}"))
            })?;
            for _ in 0..pixel_count {
                values.extend_from_slice(&[0.25, 0.5, 0.75, 1.0]);
            }
            let output = ImageTensor::from_f32(
                backend,
                context,
                batch,
                u64::from(request.height),
                u64::from(request.width),
                4,
                &values,
            )?;
            Ok(NativeShaderResult {
                outputs: vec![output.clone(), output.clone(), output.clone(), output],
                pass_count: 1,
            })
        }
    }

    impl fmt::Debug for TestPreviewEffectService {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter
                .debug_struct("TestPreviewEffectService")
                .field("identity", &self.identity)
                .finish_non_exhaustive()
        }
    }

    impl TestPreviewEffectService {
        fn new(identity: NativeNodeServiceIdentity, fail_on_ordinal: Option<u64>) -> Self {
            Self {
                identity,
                fail_on_ordinal,
                next_ordinal: AtomicU64::new(0),
                prepared: Mutex::new(Vec::new()),
                rolled_back: Mutex::new(Vec::new()),
            }
        }
    }

    impl NativePreparedEffectService for TestPreviewEffectService {
        fn identity(&self) -> &NativeNodeServiceIdentity {
            &self.identity
        }

        fn maximum_output_bytes(&self) -> u64 {
            1024 * 1024
        }

        fn prepare_output(
            &self,
            request: NativeOutputEffectRequest,
            cancellation: &CancellationToken,
        ) -> Result<NativePreparedEffectRequest, NativeEffectServiceError> {
            cancellation
                .check()
                .map_err(|_| NativeEffectServiceError::Cancelled)?;
            let ordinal = self.next_ordinal.fetch_add(1, Ordering::AcqRel);
            if self.fail_on_ordinal == Some(ordinal) {
                return Err(NativeEffectServiceError::Rejected);
            }
            let ticket = NativePreparedEffectRequest::checked(
                self.identity.service_id(),
                Uuid::from_u128(0x600 + u128::from(ordinal)),
                NativePreparedEffectKind::Output,
                request.request_digest_sha256(),
            )
            .map_err(|_| NativeEffectServiceError::Rejected)?;
            self.prepared
                .lock()
                .map_err(|_| NativeEffectServiceError::Rejected)?
                .push(ticket.clone());
            Ok(ticket)
        }

        fn rollback_prepared(
            &self,
            request: &NativePreparedEffectRequest,
        ) -> Result<(), NativeEffectServiceError> {
            let mut prepared = self
                .prepared
                .lock()
                .map_err(|_| NativeEffectServiceError::Rejected)?;
            let index = prepared
                .iter()
                .position(|ticket| ticket == request)
                .ok_or(NativeEffectServiceError::InvalidTicket)?;
            prepared.remove(index);
            self.rolled_back
                .lock()
                .map_err(|_| NativeEffectServiceError::Rejected)?
                .push(request.transaction_id());
            Ok(())
        }

        fn rollback_all_prepared(&self) -> Result<(), NativeEffectServiceError> {
            let mut prepared = self
                .prepared
                .lock()
                .map_err(|_| NativeEffectServiceError::Rejected)?;
            let mut rolled_back = self
                .rolled_back
                .lock()
                .map_err(|_| NativeEffectServiceError::Rejected)?;
            rolled_back.extend(prepared.drain(..).map(|ticket| ticket.transaction_id()));
            Ok(())
        }
    }

    impl fmt::Debug for TestHandleStore {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter
                .debug_struct("TestHandleStore")
                .field("identity", &self.identity)
                .field("attempt_id", &self.attempt_id)
                .finish_non_exhaustive()
        }
    }

    impl TestHandleStore {
        fn new(identity: NativeHandleStoreIdentity, attempt_id: AttemptId) -> Self {
            Self {
                identity,
                attempt_id,
                next_identifier: AtomicU64::new(1),
                values: Mutex::new(BTreeMap::new()),
            }
        }

        fn object_count(&self) -> Result<usize, NativeHandleStoreError> {
            self.values
                .lock()
                .map(|values| values.len())
                .map_err(|_| NativeHandleStoreError::Rejected("test store is poisoned".to_owned()))
        }

        fn check_handle(
            &self,
            handle: &NativeOpaqueHandle,
            expected_type: &NativeHandleType,
            cancellation: &CancellationToken,
        ) -> Result<(), NativeHandleStoreError> {
            cancellation
                .check()
                .map_err(|_| NativeHandleStoreError::Cancelled)?;
            handle.validate()?;
            if handle.store_identity.store_id != self.identity.store_id {
                return Err(NativeHandleStoreError::WrongStore);
            }
            if handle.store_identity.generation_id != self.identity.generation_id {
                return Err(NativeHandleStoreError::WrongGeneration);
            }
            if handle.handle_type() != expected_type {
                return Err(NativeHandleStoreError::WrongType {
                    expected: expected_type.type_id.clone(),
                    actual: handle.handle_type().type_id.clone(),
                });
            }
            Ok(())
        }
    }

    impl NativeHandleStore for TestHandleStore {
        fn identity(&self) -> NativeHandleStoreIdentity {
            self.identity
        }

        fn attempt_id(&self) -> AttemptId {
            self.attempt_id
        }

        fn resolve(
            &self,
            handle: &NativeOpaqueHandle,
            expected_type: &NativeHandleType,
            cancellation: &CancellationToken,
        ) -> Result<NativeResolvedPayload, NativeHandleStoreError> {
            self.check_handle(handle, expected_type, cancellation)?;
            let values = self.values.lock().map_err(|_| {
                NativeHandleStoreError::Rejected("test store is poisoned".to_owned())
            })?;
            let value = values
                .get(handle.identifier())
                .ok_or_else(|| NativeHandleStoreError::Missing(handle.identifier().to_owned()))?;
            if value.digest_sha256() != handle.digest_sha256().unwrap_or_default() {
                return Err(NativeHandleStoreError::DigestMismatch);
            }
            Ok(NativeResolvedPayload::checked(
                value.clone(),
                Arc::new(TestResolvedPayloadRetention),
            )?)
        }

        fn publish(
            &self,
            payload: NativeStoredPayload,
            cancellation: &CancellationToken,
        ) -> Result<NativeOpaqueHandle, NativeHandleStoreError> {
            cancellation
                .check()
                .map_err(|_| NativeHandleStoreError::Cancelled)?;
            payload.validate()?;
            let handle_type = payload.handle_type()?;
            let digest_sha256 = payload.digest_sha256();
            let generation = self.next_identifier.fetch_add(1, Ordering::AcqRel);
            let identifier = format!("handle-{generation}");
            let handle = NativeOpaqueHandle::new(
                handle_type,
                self.identity,
                identifier.clone(),
                generation,
                Some(digest_sha256),
            )?;
            self.values
                .lock()
                .map_err(|_| NativeHandleStoreError::Rejected("test store is poisoned".to_owned()))?
                .insert(identifier, Arc::new(payload));
            Ok(handle)
        }

        fn revoke(
            &self,
            handle: &NativeOpaqueHandle,
            cancellation: &CancellationToken,
        ) -> Result<(), NativeHandleStoreError> {
            self.check_handle(handle, handle.handle_type(), cancellation)?;
            let removed = self
                .values
                .lock()
                .map_err(|_| NativeHandleStoreError::Rejected("test store is poisoned".to_owned()))?
                .remove(handle.identifier());
            if removed.is_none() {
                return Err(NativeHandleStoreError::Missing(
                    handle.identifier().to_owned(),
                ));
            }
            Ok(())
        }
    }

    fn model_type() -> Result<NativeHandleType, NativeNodeContractError> {
        NativeHandleType::new(NativeHandleKind::StructuredCompute, "NOISE")
    }

    fn test_payload(value: u64) -> Result<NativeStoredPayload, Box<dyn std::error::Error>> {
        Ok(NativeStoredPayload::Noise(Arc::new(
            NativeNoisePayload::random(value)?,
        )))
    }

    fn store_identity(
        store_id: u128,
        generation_id: u128,
    ) -> Result<NativeHandleStoreIdentity, NativeNodeContractError> {
        NativeHandleStoreIdentity::new(Uuid::from_u128(store_id), Uuid::from_u128(generation_id))
    }

    struct IdentityNode;

    impl NativeNode for IdentityNode {
        fn class_type(&self) -> &str {
            "IdentityModel"
        }

        fn implementation_version(&self) -> &str {
            "1"
        }

        fn execute<'a>(
            &'a self,
            context: NativeNodeContext,
            mut inputs: BTreeMap<String, NativeValue>,
        ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
            Box::pin(async move {
                context
                    .cancellation
                    .check()
                    .map_err(|_| NativeNodeFailure {
                        code: "execution_interrupted".to_owned(),
                        message: "native node execution was interrupted".to_owned(),
                        kind: NativeNodeFailureKind::Interrupted,
                        retryable: true,
                    })?;
                let output = inputs.remove("model").ok_or_else(|| NativeNodeFailure {
                    code: "missing_input".to_owned(),
                    message: "required model input is missing".to_owned(),
                    kind: NativeNodeFailureKind::Failure,
                    retryable: false,
                })?;
                Ok(NativeNodeOutcome::Values {
                    outputs: vec![output],
                    ui: None,
                    effects: Vec::new(),
                })
            })
        }
    }

    fn identity_descriptor() -> Result<NativeNodeDescriptor, NativeNodeContractError> {
        Ok(NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: "IdentityModel".to_owned(),
            implementation_version: "1".to_owned(),
            source_schema: Some(NativeDescriptorSchemaMetadata::synthetic(
                ["model".to_owned()],
                std::iter::empty(),
                ["model".to_owned()],
            )),
            inputs: vec![NativeInputDescriptor {
                name: "model".to_owned(),
                accepted_types: NativeTypeUnion::new([NativeValueType::Handle(model_type()?)])?,
                required: true,
                hidden: false,
                lazy: false,
                cardinality: NativePortCardinality::Scalar,
                allows_literal: false,
            }],
            dynamic_inputs: Vec::new(),
            outputs: vec![NativeOutputDescriptor {
                name: "model".to_owned(),
                produced_type: NativeValueType::Handle(model_type()?),
                is_list: false,
            }],
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        })
    }

    fn identity_binding() -> Result<NativeNodeBinding, NativeNodeContractError> {
        Ok(NativeNodeBinding::Executable {
            feature_id: "COMFY-NODE-0001".to_owned(),
            descriptor: identity_descriptor()?,
            presentation: NativeNodePresentation {
                display_name: "Identity Model".to_owned(),
                category: String::new(),
                description: "Passes one opaque model handle through unchanged.".to_owned(),
                output_names: vec!["model".to_owned()],
                search_aliases: vec!["model identity".to_owned()],
                is_deprecated: true,
                is_experimental: false,
            },
            node: Arc::new(IdentityNode),
        })
    }

    #[test]
    fn typed_values_cover_handles_lists_and_preserved_unknowns()
    -> Result<(), Box<dyn std::error::Error>> {
        let model_type = model_type()?;
        let model = NativeValue::Handle {
            value: NativeOpaqueHandle::new(
                model_type.clone(),
                store_identity(1, 2)?,
                "model-1",
                1,
                Some("a".repeat(64)),
            )?,
        };
        let value = NativeValue::List {
            values: vec![
                model.clone(),
                NativeValue::Primitive {
                    value: NativePrimitive::Integer(7),
                },
                NativeValue::PreservedUnknown {
                    type_name: "future.socket@2".to_owned(),
                    value: serde_json::json!({"future": true}),
                },
            ],
        };
        value.validate()?;
        assert!(NativeTypeUnion::new([NativeValueType::Handle(model_type)])?.accepts(&model));
        let media_union = NativeTypeUnion::new(
            ["FILE_3D", "KSPLAT", "PLY", "SPLAT", "SPZ"]
                .into_iter()
                .map(|type_id| {
                    NativeHandleType::new(NativeHandleKind::ThreeD, type_id)
                        .map(NativeValueType::Handle)
                })
                .collect::<Result<Vec<_>, _>>()?,
        )?;
        assert_eq!(media_union.members().len(), 5);
        assert_eq!(
            serde_json::from_slice::<NativeValue>(&serde_json::to_vec(&value)?)?,
            value
        );
        Ok(())
    }

    #[test]
    fn structured_values_keep_handles_typed_and_plain_fields_json_compatible()
    -> Result<(), Box<dyn std::error::Error>> {
        let model_type = model_type()?;
        let handle = NativeValue::Handle {
            value: NativeOpaqueHandle::new(
                model_type,
                store_identity(7, 8)?,
                "structured-model",
                1,
                Some("b".repeat(64)),
            )?,
        };
        let structured = NativeStructuredValue::checked(
            "COMFY_DYNAMICCOMBO_V3",
            BTreeMap::from([
                (
                    "choice".to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::String("model".to_owned()),
                    },
                ),
                ("model".to_owned(), handle.clone()),
            ]),
        )?;
        let encoded = structured.into_runtime_value()?;
        let decoded = NativeStructuredValue::from_native_value(&encoded)?
            .ok_or("typed structured value was not recognized")?;
        assert_eq!(decoded.type_name(), "COMFY_DYNAMICCOMBO_V3");
        assert_eq!(decoded.get("model"), Some(&handle));
        assert!(
            NativeTypeUnion::new([NativeValueType::NamedPreservedUnknown(
                "COMFY_DYNAMICCOMBO_V3".to_owned(),
            )])?
            .accepts(&encoded)
        );

        let plain = NativeStructuredValue::checked(
            "COMFY_DYNAMICCOMBO_V3",
            BTreeMap::from([(
                "choice".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::String("plain".to_owned()),
                },
            )]),
        )?
        .into_runtime_value()?;
        assert_eq!(
            plain,
            NativeValue::PreservedUnknown {
                type_name: "COMFY_DYNAMICCOMBO_V3".to_owned(),
                value: serde_json::json!({"choice": "plain"}),
            }
        );
        Ok(())
    }

    #[test]
    fn integer_primitives_preserve_full_signed_and_unsigned_ranges()
    -> Result<(), Box<dyn std::error::Error>> {
        for value in [
            NativeValue::Primitive {
                value: NativePrimitive::Integer(i64::MIN),
            },
            NativeValue::Primitive {
                value: NativePrimitive::UnsignedInteger(u64::MAX),
            },
        ] {
            assert_eq!(
                serde_json::from_slice::<NativeValue>(&serde_json::to_vec(&value)?)?,
                value
            );
            assert!(matches!(
                &value,
                NativeValue::Primitive { value }
                    if value.primitive_type() == NativePrimitiveType::Integer
            ));
        }
        Ok(())
    }

    #[test]
    fn primitive_strings_preserve_bounded_multiline_text_without_weakening_identifiers()
    -> Result<(), Box<dyn std::error::Error>> {
        let value = NativeValue::Primitive {
            value: NativePrimitive::String("first\tcolumn\r\nsecond line\u{2028}third".to_owned()),
        };
        value.validate()?;
        assert_eq!(
            serde_json::from_slice::<NativeValue>(&serde_json::to_vec(&value)?)?,
            value
        );
        for invalid in ["nul\0text", "bell\u{0007}text"] {
            assert!(matches!(
                NativeValue::Primitive {
                    value: NativePrimitive::String(invalid.to_owned()),
                }
                .validate(),
                Err(NativeNodeContractError::InvalidText {
                    field: "primitive string"
                })
            ));
        }
        assert!(matches!(
            NativeValue::Primitive {
                value: NativePrimitive::String("a".repeat(MAX_TEXT_BYTES + 1)),
            }
            .validate(),
            Err(NativeNodeContractError::InvalidText {
                field: "primitive string"
            })
        ));
        assert!(matches!(
            validate_identifier("test identifier", "line\nbreak"),
            Err(NativeNodeContractError::InvalidText {
                field: "test identifier"
            })
        ));
        Ok(())
    }

    #[test]
    fn descriptors_and_bindings_reject_ambiguous_or_mismatched_contracts()
    -> Result<(), Box<dyn std::error::Error>> {
        identity_binding()?.validate()?;
        assert!(
            NativeTypeUnion::new([NativeValueType::Handle(model_type()?), NativeValueType::Any,])
                .is_err()
        );
        let mut descriptor = identity_descriptor()?;
        descriptor.implementation_version = "2".to_owned();
        let binding = NativeNodeBinding::Executable {
            feature_id: "COMFY-NODE-0001".to_owned(),
            descriptor,
            presentation: identity_binding()?.presentation().clone(),
            node: Arc::new(IdentityNode),
        };
        assert!(matches!(
            binding.validate(),
            Err(NativeNodeContractError::BindingImplementationMismatch)
        ));
        let mut binding = identity_binding()?;
        if let NativeNodeBinding::Executable { presentation, .. } = &mut binding {
            presentation.output_names = vec!["wrong".to_owned()];
        }
        assert!(matches!(
            binding.validate(),
            Err(NativeNodeContractError::InvalidPresentationOutputs)
        ));
        Ok(())
    }

    #[test]
    fn native_descriptor_v1_decode_preserves_absent_schema_without_inference()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut legacy = identity_descriptor()?;
        legacy.schema_version = LEGACY_NATIVE_NODE_CONTRACT_SCHEMA_VERSION;
        legacy.source_schema = None;
        let encoded = serde_json::to_vec(&legacy)?;
        let decoded: NativeNodeDescriptor = serde_json::from_slice(&encoded)?;
        assert_eq!(decoded, legacy);
        decoded.validate()?;
        assert!(decoded.validate_exact_schema_v2().is_err());
        assert!(!serde_json::to_string(&decoded)?.contains("source_schema"));
        Ok(())
    }

    #[test]
    fn native_descriptor_v2_round_trips_exact_ordered_schema()
    -> Result<(), Box<dyn std::error::Error>> {
        let descriptor = identity_descriptor()?;
        descriptor.validate_exact_schema_v2()?;
        let encoded = serde_json::to_vec(&descriptor)?;
        let decoded: NativeNodeDescriptor = serde_json::from_slice(&encoded)?;
        assert_eq!(decoded, descriptor);

        let mut missing = descriptor.clone();
        missing.source_schema = None;
        assert!(matches!(
            missing.validate(),
            Err(NativeNodeContractError::InvalidSourceSchema(_))
        ));
        let mut mismatched = descriptor;
        mismatched
            .source_schema
            .as_mut()
            .ok_or("v2 source schema is missing")?
            .inputs[0]
            .name = "other".to_owned();
        assert!(matches!(
            mismatched.validate(),
            Err(NativeNodeContractError::InvalidSourceSchema(_))
        ));
        Ok(())
    }

    #[test]
    fn portable_execution_checks_cancellation_and_preserves_handle_identity()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let scratch = authority.authorize_workspace(1024)?;
        let prompt_id = PromptId(Uuid::from_u128(10));
        let attempt_id = AttemptId(Uuid::from_u128(11));
        let handle_store = Arc::new(TestHandleStore::new(store_identity(12, 13)?, attempt_id));
        let mismatched_store = Arc::new(TestHandleStore::new(
            store_identity(12, 13)?,
            AttemptId(Uuid::from_u128(14)),
        ));
        assert!(matches!(
            NativeNodeContext::new(
                prompt_id,
                attempt_id,
                NodeId::from("identity"),
                CancellationToken::default(),
                authority.authorize_workspace(1024)?,
                mismatched_store,
            ),
            Err(NativeNodeContractError::InvalidNodeContext)
        ));
        let handle = handle_store.publish(test_payload(6)?, &CancellationToken::default())?;
        let model = NativeValue::Handle { value: handle };
        let context = NativeNodeContext::new(
            prompt_id,
            attempt_id,
            NodeId::from("identity"),
            CancellationToken::default(),
            scratch,
            handle_store.clone(),
        )?;
        let outcome = futures::executor::block_on(IdentityNode.execute(
            context,
            BTreeMap::from([("model".to_owned(), model.clone())]),
        ))?;
        assert_eq!(
            outcome,
            NativeNodeOutcome::Values {
                outputs: vec![model.clone()],
                ui: None,
                effects: Vec::new(),
            }
        );
        outcome.validate()?;

        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        let interrupted = futures::executor::block_on(IdentityNode.execute(
            NativeNodeContext::new(
                prompt_id,
                attempt_id,
                NodeId::from("identity"),
                cancellation,
                authority.authorize_workspace(1024)?,
                handle_store.clone(),
            )?,
            BTreeMap::from([("model".to_owned(), model)]),
        ))
        .expect_err("pre-cancelled execution must not publish an output");
        assert_eq!(interrupted.kind, NativeNodeFailureKind::Interrupted);
        assert_eq!(handle_store.object_count()?, 1);
        drop(backend);
        Ok(())
    }

    #[test]
    fn text_generation_rng_is_attempt_addressed_replayable_and_cancellable()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let prompt_id = PromptId(Uuid::from_u128(20));
        let attempt_id = AttemptId(Uuid::from_u128(21));
        let handle_store = Arc::new(TestHandleStore::new(store_identity(22, 23)?, attempt_id));
        let make_context = |cancellation| -> Result<NativeNodeContext, Box<dyn std::error::Error>> {
            Ok(NativeNodeContext::new(
                prompt_id,
                attempt_id,
                NodeId::from("text-generation"),
                cancellation,
                authority.authorize_workspace(1024)?,
                handle_store.clone(),
            )?)
        };

        let first =
            native_text_generation_transaction(&make_context(CancellationToken::default())?, 42)?;
        let replay =
            native_text_generation_transaction(&make_context(CancellationToken::default())?, 42)?;
        let different_seed =
            native_text_generation_transaction(&make_context(CancellationToken::default())?, 43)?;
        assert_eq!(first.checkpoint(), replay.checkpoint());
        assert_ne!(first.checkpoint(), different_seed.checkpoint());
        assert_eq!(NATIVE_TEXT_GENERATION_RNG_PHASE, "native-text-generation");

        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        assert!(matches!(
            native_text_generation_transaction(&make_context(cancellation)?, 42),
            Err(RngError::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn text_generation_service_is_attempt_bound_and_validates_typed_projection()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let prompt_id = PromptId(Uuid::from_u128(0x701));
        let attempt_id = AttemptId(Uuid::from_u128(0x702));
        let node_id = NodeId::from("text-generation");
        let handle_store_identity = store_identity(0x703, 0x704)?;
        let handle_store = Arc::new(TestHandleStore::new(handle_store_identity, attempt_id));
        let handle = |kind, type_id, identifier| {
            NativeOpaqueHandle::new(
                NativeHandleType::new(kind, type_id)?,
                handle_store_identity,
                identifier,
                1,
                Some("a".repeat(64)),
            )
        };
        let request = NativeMultimodalTextGenerationRequest::checked(
            handle(NativeHandleKind::Clip, "CLIP", "clip")?,
            Some(handle(NativeHandleKind::Image, "IMAGE", "image")?),
            None,
            Some(handle(NativeHandleKind::Audio, "AUDIO", "audio")?),
            "hello",
            8,
            true,
            0.7_f32.to_bits(),
            64,
            0.95_f32.to_bits(),
            0.05_f32.to_bits(),
            1.05_f32.to_bits(),
            0.0_f32.to_bits(),
            42,
            true,
            false,
        )?;
        let identity = NativeNodeServiceIdentity::checked(
            Uuid::from_u128(0x705),
            attempt_id,
            node_id.clone(),
        )?;
        let calls = Arc::new(AtomicU64::new(0));
        let services = NativeNodeServices::checked(None, None, None)?.with_text_generation(
            Arc::new(TestTextGenerationService {
                identity,
                calls: calls.clone(),
                cancellation_after_generate: None,
                return_invalid_output: false,
            }),
        )?;
        let context = NativeNodeContext::new_with_services(
            prompt_id,
            attempt_id,
            node_id.clone(),
            CancellationToken::default(),
            authority.authorize_workspace(1024)?,
            handle_store.clone(),
            services,
        )?;
        let output = context.execute_multimodal_text_generation(&request)?;
        assert_eq!(output.text(), "generated:hello");
        assert_eq!(output.generated_token_count(), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        let foreign = NativeOpaqueHandle::new(
            NativeHandleType::new(NativeHandleKind::Image, "IMAGE")?,
            store_identity(0x706, 0x707)?,
            "foreign-image",
            1,
            Some("b".repeat(64)),
        )?;
        assert!(matches!(
            NativeMultimodalTextGenerationRequest::checked(
                request.clip().clone(),
                Some(foreign),
                None,
                None,
                "hello",
                8,
                false,
                1.0_f32.to_bits(),
                50,
                1.0_f32.to_bits(),
                0.0_f32.to_bits(),
                1.0_f32.to_bits(),
                0.0_f32.to_bits(),
                0,
                true,
                false,
            ),
            Err(NativeTextGenerationServiceError::InvalidRequest)
        ));

        let mismatched_services = NativeNodeServices::checked(None, None, None)?
            .with_text_generation(Arc::new(TestTextGenerationService {
                identity: NativeNodeServiceIdentity::checked(
                    Uuid::from_u128(0x708),
                    AttemptId(Uuid::from_u128(0x709)),
                    node_id,
                )?,
                calls,
                cancellation_after_generate: None,
                return_invalid_output: false,
            }))?;
        assert!(matches!(
            NativeNodeContext::new_with_services(
                prompt_id,
                attempt_id,
                NodeId::from("text-generation"),
                CancellationToken::default(),
                authority.authorize_workspace(1024)?,
                handle_store,
                mismatched_services,
            ),
            Err(NativeNodeContractError::InvalidNodeServiceIdentity)
        ));

        let unavailable_context = NativeNodeContext::new_with_services(
            prompt_id,
            attempt_id,
            NodeId::from("text-generation"),
            CancellationToken::default(),
            authority.authorize_workspace(1024)?,
            Arc::new(TestHandleStore::new(handle_store_identity, attempt_id)),
            NativeNodeServices::checked(None, None, None)?,
        )?;
        assert_eq!(
            unavailable_context.execute_multimodal_text_generation(&request),
            Err(NativeTextGenerationServiceError::Unavailable)
        );

        let invalid_calls = Arc::new(AtomicU64::new(0));
        let invalid_services = NativeNodeServices::checked(None, None, None)?
            .with_text_generation(Arc::new(TestTextGenerationService {
                identity: NativeNodeServiceIdentity::checked(
                    Uuid::from_u128(0x70a),
                    attempt_id,
                    NodeId::from("text-generation"),
                )?,
                calls: invalid_calls.clone(),
                cancellation_after_generate: None,
                return_invalid_output: true,
            }))?;
        let invalid_context = NativeNodeContext::new_with_services(
            prompt_id,
            attempt_id,
            NodeId::from("text-generation"),
            CancellationToken::default(),
            authority.authorize_workspace(1024)?,
            Arc::new(TestHandleStore::new(handle_store_identity, attempt_id)),
            invalid_services,
        )?;
        assert_eq!(
            invalid_context.execute_multimodal_text_generation(&request),
            Err(NativeTextGenerationServiceError::InvalidOutput)
        );
        assert_eq!(invalid_calls.load(Ordering::SeqCst), 1);

        let cancellation = CancellationToken::default();
        let cancelled_calls = Arc::new(AtomicU64::new(0));
        let cancelled_services = NativeNodeServices::checked(None, None, None)?
            .with_text_generation(Arc::new(TestTextGenerationService {
                identity: NativeNodeServiceIdentity::checked(
                    Uuid::from_u128(0x70b),
                    attempt_id,
                    NodeId::from("text-generation"),
                )?,
                calls: cancelled_calls.clone(),
                cancellation_after_generate: Some(cancellation.clone()),
                return_invalid_output: false,
            }))?;
        let cancelled_context = NativeNodeContext::new_with_services(
            prompt_id,
            attempt_id,
            NodeId::from("text-generation"),
            cancellation,
            authority.authorize_workspace(1024)?,
            Arc::new(TestHandleStore::new(handle_store_identity, attempt_id)),
            cancelled_services,
        )?;
        assert_eq!(
            cancelled_context.execute_multimodal_text_generation(&request),
            Err(NativeTextGenerationServiceError::Cancelled)
        );
        assert_eq!(cancelled_calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn compute_session_requires_the_contexts_exact_backend_and_scratch_binding()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let backend = Arc::new(backend);
        let (foreign_backend, foreign_authority) =
            CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let foreign_backend = Arc::new(foreign_backend);
        let scratch = authority.authorize_workspace(1024)?;
        let prompt_id = PromptId(Uuid::from_u128(0x501));
        let attempt_id = AttemptId(Uuid::from_u128(0x502));
        let node_id = NodeId::from("compute");
        let identity = NativeNodeServiceIdentity::checked(
            Uuid::from_u128(0x503),
            attempt_id,
            node_id.clone(),
        )?;
        let compute = NativeNodeComputeSession::checked(
            identity.clone(),
            backend,
            StreamId::DEFAULT,
            &scratch,
        )?;
        assert!(matches!(
            NativeNodeComputeSession::checked(
                identity,
                foreign_backend,
                StreamId::DEFAULT,
                &scratch,
            ),
            Err(NativeNodeContractError::InvalidComputeSession)
        ));
        let store = Arc::new(TestHandleStore::new(
            store_identity(0x504, 0x505)?,
            attempt_id,
        ));
        let context = NativeNodeContext::new_with_services(
            prompt_id,
            attempt_id,
            node_id,
            CancellationToken::default(),
            scratch,
            store,
            NativeNodeServices::checked(None, None, Some(compute.clone()))?,
        )?;
        assert_eq!(
            compute.execution_context(&context)?.stream,
            StreamId::DEFAULT
        );
        assert!(matches!(
            NativeNodeContext::new_with_services(
                prompt_id,
                attempt_id,
                NodeId::from("compute"),
                CancellationToken::default(),
                foreign_authority.authorize_workspace(1024)?,
                Arc::new(TestHandleStore::new(
                    store_identity(0x506, 0x507)?,
                    attempt_id,
                )),
                NativeNodeServices::checked(None, None, Some(compute))?,
            ),
            Err(NativeNodeContractError::InvalidComputeSession)
        ));
        Ok(())
    }

    #[test]
    fn ltxv_preprocess_service_requires_checked_identity_and_preserves_context_authority()
    -> Result<(), Box<dyn std::error::Error>> {
        assert!(NativeLtxvPreprocessServiceIdentity::checked("not-a-digest").is_err());
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let backend = Arc::new(backend);
        let scratch = authority.authorize_workspace(1024 * 1024)?;
        let prompt_id = PromptId(Uuid::from_u128(0x521));
        let attempt_id = AttemptId(Uuid::from_u128(0x522));
        let node_id = NodeId::from("ltxv-preprocess");
        let compute = NativeNodeComputeSession::checked(
            NativeNodeServiceIdentity::checked(
                Uuid::from_u128(0x523),
                attempt_id,
                node_id.clone(),
            )?,
            backend.clone(),
            StreamId::DEFAULT,
            &scratch,
        )?;
        let service = Arc::new(TestLtxvPreprocessService {
            identity: NativeLtxvPreprocessServiceIdentity::checked("a".repeat(64))?,
            compression: AtomicU64::new(0),
        });
        let services = NativeNodeServices::checked(None, None, Some(compute.clone()))?
            .with_ltxv_preprocess(service.clone())?;
        let context = NativeNodeContext::new_with_services(
            prompt_id,
            attempt_id,
            node_id,
            CancellationToken::default(),
            scratch,
            Arc::new(TestHandleStore::new(
                store_identity(0x524, 0x525)?,
                attempt_id,
            )),
            services,
        )?;
        let image = ImageTensor::from_f32(
            &backend,
            &compute.execution_context(&context)?,
            1,
            1,
            1,
            1,
            &[0.5],
        )?;
        let returned =
            futures::executor::block_on(context.ltxv_preprocess_service()?.preprocess_image(
                &image,
                35,
                &compute.execution_context(&context)?,
            ))?;
        assert_eq!(returned.tensor().storage_id(), image.tensor().storage_id());
        assert_eq!(service.compression.load(Ordering::SeqCst), 35);
        assert_eq!(
            context
                .ltxv_preprocess_service()?
                .identity()
                .configuration_sha256(),
            "a".repeat(64)
        );

        let unavailable = NativeNodeContext::new(
            prompt_id,
            attempt_id,
            NodeId::from("ltxv-unavailable"),
            CancellationToken::default(),
            authority.authorize_workspace(1024)?,
            Arc::new(TestHandleStore::new(
                store_identity(0x526, 0x527)?,
                attempt_id,
            )),
        )?;
        assert!(matches!(
            unavailable.ltxv_preprocess_service(),
            Err(NativeLtxvPreprocessServiceError::Unavailable)
        ));
        Ok(())
    }

    #[test]
    fn webm_encode_service_requires_checked_identity_and_validates_portable_contract()
    -> Result<(), Box<dyn std::error::Error>> {
        assert!(NativeWebmEncodeServiceIdentity::checked("not-a-digest").is_err());
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let backend = Arc::new(backend);
        let scratch = authority.authorize_workspace(1024 * 1024)?;
        let prompt_id = PromptId(Uuid::from_u128(0x531));
        let attempt_id = AttemptId(Uuid::from_u128(0x532));
        let node_id = NodeId::from("webm-encode");
        let compute = NativeNodeComputeSession::checked(
            NativeNodeServiceIdentity::checked(
                Uuid::from_u128(0x533),
                attempt_id,
                node_id.clone(),
            )?,
            backend.clone(),
            StreamId::DEFAULT,
            &scratch,
        )?;
        let service = Arc::new(TestWebmEncodeService {
            identity: NativeWebmEncodeServiceIdentity::checked("b".repeat(64))?,
            backend: backend.clone(),
            metadata: Mutex::new(Vec::new()),
        });
        let services = NativeNodeServices::checked(None, None, Some(compute.clone()))?
            .with_webm_encode(service.clone())?;
        let context = NativeNodeContext::new_with_services(
            prompt_id,
            attempt_id,
            node_id,
            CancellationToken::default(),
            scratch,
            Arc::new(TestHandleStore::new(
                store_identity(0x534, 0x535)?,
                attempt_id,
            )),
            services,
        )?;
        let execution = compute.execution_context(&context)?;
        let image = ImageTensor::from_f32(&backend, &execution, 1, 1, 1, 3, &[0.25, 0.5, 0.75])?;
        let input_storage = image.tensor().storage_id();
        let metadata = vec![
            ("prompt".to_owned(), "first".to_owned()),
            ("workflow".to_owned(), "{}".to_owned()),
            ("prompt".to_owned(), "last".to_owned()),
        ];
        let request = NativeWebmEncodeRequest::checked(
            image,
            NativeVideoCodec::Vp9,
            (125, 2_997),
            NativeVideoCrf::checked(31.5)?,
            metadata.clone(),
        )?;
        assert_eq!(request.images().tensor().storage_id(), input_storage);
        assert_eq!(request.crf().bits(), 31.5_f64.to_bits());
        let result = futures::executor::block_on(
            context
                .webm_encode_service()?
                .encode_webm(request, &execution),
        )?;
        assert_eq!(result.bytes().contiguous_bytes()?, b"HPPPT");
        assert_eq!(result.codec(), NativeVideoCodec::Vp9);
        assert_eq!(result.pixel_format(), NativeVideoPixelFormat::Yuv420p);
        assert_eq!(result.bit_depth(), NativeVideoBitDepth::Eight);
        assert!(!result.has_alpha());
        assert!(matches!(
            NativeEncodedWebm::checked(
                result.bytes().clone(),
                [0; 32],
                NativeVideoCodec::Vp9,
                (1, 1),
                (125, 2_997),
                1,
                NativeVideoPixelFormat::Yuv420p,
                NativeVideoBitDepth::Eight,
                false,
            ),
            Err(NativeWebmEncodeServiceError::InvalidProjection)
        ));
        assert!(matches!(
            NativeEncodedWebm::checked(
                result.bytes().clone(),
                result.content_sha256(),
                NativeVideoCodec::Av1,
                (1, 1),
                (125, 2_997),
                1,
                NativeVideoPixelFormat::Yuv420p10le,
                NativeVideoBitDepth::Ten,
                true,
            ),
            Err(NativeWebmEncodeServiceError::InvalidProjection)
        ));
        assert_eq!(
            *service
                .metadata
                .lock()
                .expect("test metadata mutex poisoned"),
            metadata
        );

        let invalid_codec_image =
            ImageTensor::from_f32(&backend, &execution, 1, 1, 1, 3, &[0.0, 0.0, 0.0])?;
        let unreduced_rate_image = invalid_codec_image.clone();
        for invalid in [
            NativeWebmEncodeRequest::checked(
                invalid_codec_image,
                NativeVideoCodec::H264,
                (1, 1),
                NativeVideoCrf::checked(1.0)?,
                Vec::new(),
            ),
            NativeWebmEncodeRequest::checked(
                unreduced_rate_image,
                NativeVideoCodec::Vp9,
                (2, 2),
                NativeVideoCrf::checked(1.0)?,
                Vec::new(),
            ),
        ] {
            assert!(matches!(
                invalid,
                Err(NativeWebmEncodeServiceError::InvalidRequest)
            ));
        }

        let unavailable = NativeNodeContext::new(
            prompt_id,
            attempt_id,
            NodeId::from("webm-unavailable"),
            CancellationToken::default(),
            authority.authorize_workspace(1024)?,
            Arc::new(TestHandleStore::new(
                store_identity(0x536, 0x537)?,
                attempt_id,
            )),
        )?;
        assert!(matches!(
            unavailable.webm_encode_service(),
            Err(NativeWebmEncodeServiceError::Unavailable)
        ));
        Ok(())
    }

    #[test]
    fn component_h264_mp4_backing_service_requires_checked_video_and_portable_contract()
    -> Result<(), Box<dyn std::error::Error>> {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NativeComponentH264Mp4BackingRequest>();
        assert_send_sync::<NativeVideoPayload>();
        assert!(NativeComponentH264Mp4BackingServiceIdentity::checked("not-a-digest").is_err());
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let backend = Arc::new(backend);
        let scratch = authority.authorize_workspace(1024 * 1024)?;
        let prompt_id = PromptId(Uuid::from_u128(0x541));
        let attempt_id = AttemptId(Uuid::from_u128(0x542));
        let node_id = NodeId::from("component-h264-mp4-backing");
        let compute = NativeNodeComputeSession::checked(
            NativeNodeServiceIdentity::checked(
                Uuid::from_u128(0x543),
                attempt_id,
                node_id.clone(),
            )?,
            backend.clone(),
            StreamId::DEFAULT,
            &scratch,
        )?;
        let service = Arc::new(TestComponentH264Mp4BackingService {
            identity: NativeComponentH264Mp4BackingServiceIdentity::checked("c".repeat(64))?,
            backend: backend.clone(),
        });
        let services = NativeNodeServices::checked(None, None, Some(compute.clone()))?
            .with_component_h264_mp4_backing(service)?;
        let context = NativeNodeContext::new_with_services(
            prompt_id,
            attempt_id,
            node_id,
            CancellationToken::default(),
            scratch,
            Arc::new(TestHandleStore::new(
                store_identity(0x544, 0x545)?,
                attempt_id,
            )),
            services,
        )?;
        let execution = compute.execution_context(&context)?;
        let frames =
            ImageTensor::from_f32(&backend, &execution, 1, 1, 1, 4, &[0.25, 0.5, 0.75, 0.125])?;
        let frame_storage = frames.tensor().storage_id();
        let video = Arc::new(NativeVideoPayload::checked(
            frames.tensor().clone(),
            2_997,
            100,
            NativeVideoBitDepth::Eight,
            None,
            None,
            BTreeMap::from([("prompt".to_owned(), "source-only".to_owned())]),
        )?);
        let request = NativeComponentH264Mp4BackingRequest::checked(video.clone())?;
        assert_eq!(
            request
                .video()
                .components()
                .ok_or(NativeComponentH264Mp4BackingServiceError::InvalidRequest)?
                .frames()
                .storage_id(),
            frame_storage
        );
        let result = futures::executor::block_on(
            context
                .component_h264_mp4_backing_service()?
                .encode_backing(request, &execution),
        )?;
        let encoded = result
            .encoded()
            .ok_or(NativeComponentH264Mp4BackingServiceError::InvalidProjection)?;
        assert_eq!(encoded.bytes().contiguous_bytes()?, b"HMP4");
        assert_eq!(
            encoded.source_video_sha256(),
            video.semantic_digest_sha256()
        );
        assert_eq!(encoded.container(), sim_media::NativeVideoContainer::Mp4);
        assert_eq!(encoded.codec(), NativeVideoCodec::H264);
        assert_eq!(encoded.pixel_format(), NativeVideoPixelFormat::Yuv420p);
        assert_eq!(encoded.bit_depth(), NativeVideoBitDepth::Eight);
        assert!(!encoded.has_alpha());
        assert!(
            NativeVideoPayload::checked_h264_mp4_from_component(
                video.as_ref(),
                encoded.bytes().clone(),
                [0; 32],
                encoded.dimensions(),
                encoded.frame_rate(),
                encoded.frame_count(),
            )
            .is_err()
        );

        let ten_bit = Arc::new(NativeVideoPayload::checked(
            frames.tensor().clone(),
            2_997,
            100,
            NativeVideoBitDepth::Ten,
            None,
            None,
            BTreeMap::new(),
        )?);
        let ten_bit_request = NativeComponentH264Mp4BackingRequest::checked(ten_bit.clone())?;
        let ten_bit_result = futures::executor::block_on(
            context
                .component_h264_mp4_backing_service()?
                .encode_backing(ten_bit_request, &execution),
        )?;
        let ten_bit_encoded = ten_bit_result
            .encoded()
            .ok_or(NativeComponentH264Mp4BackingServiceError::InvalidProjection)?;
        assert_eq!(ten_bit_encoded.bytes().contiguous_bytes()?, b"H10P4");
        assert_eq!(ten_bit_encoded.bit_depth(), NativeVideoBitDepth::Ten);
        assert_eq!(
            ten_bit_encoded.pixel_format(),
            NativeVideoPixelFormat::Yuv420p10le
        );
        assert_eq!(
            ten_bit_encoded.source_video_sha256(),
            ten_bit.semantic_digest_sha256()
        );
        let unavailable = NativeNodeContext::new(
            prompt_id,
            attempt_id,
            NodeId::from("component-h264-mp4-unavailable"),
            CancellationToken::default(),
            authority.authorize_workspace(1024)?,
            Arc::new(TestHandleStore::new(
                store_identity(0x546, 0x547)?,
                attempt_id,
            )),
        )?;
        assert!(matches!(
            unavailable.component_h264_mp4_backing_service(),
            Err(NativeComponentH264Mp4BackingServiceError::Unavailable)
        ));
        Ok(())
    }

    #[test]
    fn shader_service_uses_the_attempts_exact_compute_session()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let backend = Arc::new(backend);
        let scratch = authority.authorize_workspace(1024 * 1024)?;
        let prompt_id = PromptId(Uuid::from_u128(0x551));
        let attempt_id = AttemptId(Uuid::from_u128(0x552));
        let node_id = NodeId::from("shader");
        let identity = NativeNodeServiceIdentity::checked(
            Uuid::from_u128(0x553),
            attempt_id,
            node_id.clone(),
        )?;
        let compute = NativeNodeComputeSession::checked(
            identity.clone(),
            backend.clone(),
            StreamId::DEFAULT,
            &scratch,
        )?;
        let effects = Arc::new(TestPreviewEffectService::new(identity, None));
        let services = NativeNodeServices::checked(None, Some(effects), Some(compute))?
            .with_shader(Arc::new(TestShaderExecutor));
        let context = NativeNodeContext::new_with_services(
            prompt_id,
            attempt_id,
            node_id,
            CancellationToken::default(),
            scratch,
            Arc::new(TestHandleStore::new(
                store_identity(0x554, 0x555)?,
                attempt_id,
            )),
            services,
        )?;
        let request = NativeShaderRequest {
            fragment_source: "#version 300 es\nvoid main() {}".to_owned(),
            images: vec![ImageTensor::from_f32(
                &backend,
                &context.compute_session()?.execution_context(&context)?,
                1,
                2,
                2,
                3,
                &[0.0; 12],
            )?],
            floats: Vec::new(),
            ints: Vec::new(),
            bools: Vec::new(),
            curves: Vec::new(),
            width: 2,
            height: 2,
        };
        let result = context.execute_shader(&request)?;
        assert!(
            result.outputs[0]
                .as_f32_slice()?
                .chunks_exact(4)
                .all(|pixel| pixel == [0.25, 0.5, 0.75, 1.0])
        );
        let prepared = context.execute_shader_with_previews(&request)?;
        assert_eq!(prepared.effects().len(), 2);
        assert_eq!(
            prepared.ui()["input_images"].as_array().map(Vec::len),
            Some(1)
        );
        assert_eq!(prepared.ui()["images"].as_array().map(Vec::len), Some(1));
        assert_eq!(prepared.shader().outputs.len(), MAX_SHADER_OUTPUTS);

        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        let scratch = authority.authorize_workspace(1024)?;
        let identity = NativeNodeServiceIdentity::checked(
            Uuid::from_u128(0x556),
            attempt_id,
            NodeId::from("shader"),
        )?;
        let compute = NativeNodeComputeSession::checked(
            identity,
            backend.clone(),
            StreamId::DEFAULT,
            &scratch,
        )?;
        let cancelled = NativeNodeContext::new_with_services(
            prompt_id,
            attempt_id,
            NodeId::from("shader"),
            cancellation,
            scratch,
            Arc::new(TestHandleStore::new(
                store_identity(0x557, 0x558)?,
                attempt_id,
            )),
            NativeNodeServices::checked(None, None, Some(compute))?
                .with_shader(Arc::new(TestShaderExecutor)),
        )?;
        assert!(matches!(
            cancelled.execute_shader(&request),
            Err(NativeShaderServiceError::Shader(
                NativeShaderError::Cancelled
            ))
        ));

        let scratch = authority.authorize_workspace(1024 * 1024)?;
        let node_id = NodeId::from("shader-preview-failure");
        let identity = NativeNodeServiceIdentity::checked(
            Uuid::from_u128(0x559),
            attempt_id,
            node_id.clone(),
        )?;
        let compute = NativeNodeComputeSession::checked(
            identity.clone(),
            backend,
            StreamId::DEFAULT,
            &scratch,
        )?;
        let failing_effects = Arc::new(TestPreviewEffectService::new(identity, Some(1)));
        let failed = NativeNodeContext::new_with_services(
            prompt_id,
            attempt_id,
            node_id,
            CancellationToken::default(),
            scratch,
            Arc::new(TestHandleStore::new(
                store_identity(0x55a, 0x55b)?,
                attempt_id,
            )),
            NativeNodeServices::checked(None, Some(failing_effects.clone()), Some(compute))?
                .with_shader(Arc::new(TestShaderExecutor)),
        )?;
        assert!(matches!(
            failed.execute_shader_with_previews(&request),
            Err(NativeShaderPreviewError::Preview(
                NativeImagePreviewError::Effect(NativeEffectServiceError::Rejected)
            ))
        ));
        assert!(
            failing_effects
                .prepared
                .lock()
                .map_err(|_| "shader preview prepared state is poisoned")?
                .is_empty()
        );
        assert_eq!(
            failing_effects
                .rolled_back
                .lock()
                .map_err(|_| "shader preview rollback state is poisoned")?
                .len(),
            1
        );
        Ok(())
    }

    #[test]
    fn image_preview_prepares_batched_pngs_and_rolls_back_partial_failure()
    -> Result<(), Box<dyn std::error::Error>> {
        let prompt_id = PromptId(Uuid::from_u128(0x581));
        let attempt_id = AttemptId(Uuid::from_u128(0x582));
        let node_id = NodeId::from("preview");
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let backend = Arc::new(backend);
        let scratch = authority.authorize_workspace(1024 * 1024)?;
        let identity = NativeNodeServiceIdentity::checked(
            Uuid::from_u128(0x583),
            attempt_id,
            node_id.clone(),
        )?;
        let compute = NativeNodeComputeSession::checked(
            identity.clone(),
            backend.clone(),
            StreamId::DEFAULT,
            &scratch,
        )?;
        let effects = Arc::new(TestPreviewEffectService::new(identity, None));
        let context = NativeNodeContext::new_with_services(
            prompt_id,
            attempt_id,
            node_id.clone(),
            CancellationToken::default(),
            scratch,
            Arc::new(TestHandleStore::new(
                store_identity(0x584, 0x585)?,
                attempt_id,
            )),
            NativeNodeServices::checked(None, Some(effects), Some(compute))?,
        )?;
        let execution_context = context.compute_session()?.execution_context(&context)?;
        let image = ImageTensor::from_f32(
            &backend,
            &execution_context,
            2,
            1,
            1,
            3,
            &[0.0, 0.5, 1.0, 1.0, 0.5, 0.0],
        )?;
        let preview = context.prepare_image_preview(&image, "preview")?;
        assert_eq!(preview.effects().len(), 2);
        assert_eq!(preview.ui()["images"].as_array().map(Vec::len), Some(2));
        assert_eq!(preview.ui()["animated"], json!([false]));

        let scratch = authority.authorize_workspace(1024 * 1024)?;
        let identity = NativeNodeServiceIdentity::checked(
            Uuid::from_u128(0x586),
            attempt_id,
            node_id.clone(),
        )?;
        let compute = NativeNodeComputeSession::checked(
            identity.clone(),
            backend,
            StreamId::DEFAULT,
            &scratch,
        )?;
        let failing_effects = Arc::new(TestPreviewEffectService::new(identity, Some(1)));
        let context = NativeNodeContext::new_with_services(
            prompt_id,
            attempt_id,
            node_id,
            CancellationToken::default(),
            scratch,
            Arc::new(TestHandleStore::new(
                store_identity(0x587, 0x588)?,
                attempt_id,
            )),
            NativeNodeServices::checked(None, Some(failing_effects.clone()), Some(compute))?,
        )?;
        assert!(matches!(
            context.prepare_image_preview(&image, "preview"),
            Err(NativeImagePreviewError::Effect(
                NativeEffectServiceError::Rejected
            ))
        ));
        assert!(
            failing_effects
                .prepared
                .lock()
                .map_err(|_| "preview prepared state is poisoned")?
                .is_empty()
        );
        assert_eq!(
            failing_effects
                .rolled_back
                .lock()
                .map_err(|_| "preview rollback state is poisoned")?
                .len(),
            1
        );
        Ok(())
    }

    #[test]
    fn attempt_local_handle_store_rejects_foreign_identity_type_and_cancellation()
    -> Result<(), Box<dyn std::error::Error>> {
        let attempt_id = AttemptId(Uuid::from_u128(20));
        assert!(matches!(
            NativeHandleStoreIdentity::new(Uuid::nil(), Uuid::from_u128(22)),
            Err(NativeNodeContractError::InvalidHandleStoreIdentity)
        ));
        let identity = store_identity(21, 22)?;
        let store = TestHandleStore::new(identity, attempt_id);
        let model_type = model_type()?;
        let handle = store.publish(test_payload(7)?, &CancellationToken::default())?;
        let resolved = store.resolve(&handle, &model_type, &CancellationToken::default())?;
        resolved.validate()?;

        let wrong_store = NativeOpaqueHandle::new(
            model_type.clone(),
            store_identity(23, 22)?,
            handle.identifier(),
            handle.generation(),
            None,
        )?;
        assert!(matches!(
            store.resolve(&wrong_store, &model_type, &CancellationToken::default()),
            Err(NativeHandleStoreError::WrongStore)
        ));
        let wrong_generation = NativeOpaqueHandle::new(
            model_type.clone(),
            store_identity(21, 24)?,
            handle.identifier(),
            handle.generation(),
            None,
        )?;
        assert!(matches!(
            store.resolve(
                &wrong_generation,
                &model_type,
                &CancellationToken::default()
            ),
            Err(NativeHandleStoreError::WrongGeneration)
        ));
        let forged_digest = NativeOpaqueHandle::new(
            model_type.clone(),
            identity,
            handle.identifier(),
            handle.generation(),
            Some("c".repeat(64)),
        )?;
        assert!(matches!(
            store.resolve(&forged_digest, &model_type, &CancellationToken::default()),
            Err(NativeHandleStoreError::DigestMismatch)
        ));
        let image_type = NativeHandleType::new(NativeHandleKind::Image, "IMAGE")?;
        assert!(matches!(
            store.resolve(&handle, &image_type, &CancellationToken::default()),
            Err(NativeHandleStoreError::WrongType { .. })
        ));

        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        let before = store.object_count()?;
        assert!(matches!(
            store.publish(test_payload(8)?, &cancellation),
            Err(NativeHandleStoreError::Cancelled)
        ));
        assert_eq!(store.object_count()?, before);
        Ok(())
    }

    #[test]
    fn persisted_sampler_handles_distinguish_parameters_and_reject_stale_generations()
    -> Result<(), Box<dyn std::error::Error>> {
        let attempt_id = AttemptId(Uuid::from_u128(40));
        let first_identity = store_identity(41, 42)?;
        let first_store = TestHandleStore::new(first_identity, attempt_id);
        let lms_four = Arc::new(NativeSamplerPayload::lms(4)?);
        let lms_five = Arc::new(NativeSamplerPayload::lms(5)?);
        assert_ne!(
            lms_four.semantic_digest_sha256(),
            lms_five.semantic_digest_sha256()
        );
        let first_handle = first_store.publish(
            NativeStoredPayload::Sampler(lms_four.clone()),
            &CancellationToken::default(),
        )?;
        let persisted = serde_json::to_vec(&first_handle)?;
        let recovered_handle: NativeOpaqueHandle = serde_json::from_slice(&persisted)?;
        let sampler_type = NativeHandleType::new(NativeHandleKind::StructuredCompute, "SAMPLER")?;
        let recovered = first_store.resolve(
            &recovered_handle,
            &sampler_type,
            &CancellationToken::default(),
        )?;
        assert_eq!(recovered.digest_sha256(), lms_four.semantic_digest_sha256());

        let replacement_store = TestHandleStore::new(store_identity(41, 43)?, attempt_id);
        assert!(matches!(
            replacement_store.resolve(
                &recovered_handle,
                &sampler_type,
                &CancellationToken::default()
            ),
            Err(NativeHandleStoreError::WrongGeneration)
        ));
        assert_eq!(replacement_store.object_count()?, 0);
        let fresh_handle = replacement_store.publish(
            NativeStoredPayload::Sampler(lms_five),
            &CancellationToken::default(),
        )?;
        assert_ne!(first_handle.digest_sha256(), fresh_handle.digest_sha256());
        assert_eq!(replacement_store.object_count()?, 1);
        Ok(())
    }

    #[test]
    fn generated_binding_validation_is_exact_and_collision_free()
    -> Result<(), Box<dyn std::error::Error>> {
        let binding = identity_binding()?;
        validate_generated_family_bindings(std::slice::from_ref(&binding), &["IdentityModel"])?;
        assert!(matches!(
            validate_generated_family_bindings(&[binding.clone(), binding], &["IdentityModel"]),
            Err(NativeNodeContractError::DuplicateGeneratedDescriptor)
        ));
        assert!(matches!(
            validate_generated_family_bindings(&[], &["IdentityModel"]),
            Err(NativeNodeContractError::GeneratedBindingMismatch)
        ));
        Ok(())
    }

    #[test]
    fn stored_payload_derives_exact_metadata_without_an_untyped_escape()
    -> Result<(), Box<dyn std::error::Error>> {
        let payload = test_payload(9)?;
        payload.validate()?;
        assert_eq!(payload.handle_type()?, model_type()?);
        assert_eq!(payload.digest_sha256().len(), 64);
        assert!(payload.resident_bytes()? > 0);
        let different = test_payload(10)?;
        different.validate()?;
        assert_ne!(payload.digest_sha256(), different.digest_sha256());
        let provider = crate::NativeProviderPayload::checked(
            NativeHandleType::new(NativeHandleKind::ProviderTask, "MODEL_TASK_ID")?,
            "signed.provider",
            "a".repeat(64),
            Vec::new(),
        )?;
        assert_eq!(provider.signed_namespace(), "signed.provider");
        assert_ne!(provider.identity_digest_sha256(), "a".repeat(64));
        assert!(matches!(
            crate::NativeProviderPayload::checked(
                NativeHandleType::new(NativeHandleKind::Image, "IMAGE")?,
                "signed.provider",
                "b".repeat(64),
                Vec::new(),
            ),
            Err(NativeStoredPayloadError::InvalidProviderPayload)
        ));
        Ok(())
    }

    #[test]
    fn native_model_resource_service_recipes_are_exact() -> Result<(), Box<dyn std::error::Error>> {
        for (role, category, count) in [
            (NativeModelResourceRole::Model, "checkpoints", 1),
            (NativeModelResourceRole::Model, "diffusion_models", 1),
            (NativeModelResourceRole::Clip, "checkpoints", 1),
            (NativeModelResourceRole::Clip, "text_encoders", 4),
            (NativeModelResourceRole::Vae, "vae", 1),
            (NativeModelResourceRole::AudioEncoder, "audio_encoders", 1),
            (NativeModelResourceRole::ClipVision, "clip_vision", 1),
            (NativeModelResourceRole::StyleModel, "style_models", 1),
            (NativeModelResourceRole::Photomaker, "photomaker", 1),
            (NativeModelResourceRole::Gligen, "gligen", 1),
            (
                NativeModelResourceRole::LatentUpscaleModel,
                "latent_upscale_models",
                1,
            ),
            (
                NativeModelResourceRole::BackgroundRemoval,
                "background_removal",
                1,
            ),
            (
                NativeModelResourceRole::FrameInterpolation,
                "frame_interpolation",
                1,
            ),
            (NativeModelResourceRole::Da3Model, "geometry_estimation", 1),
            (NativeModelResourceRole::MogeModel, "geometry_estimation", 1),
            (NativeModelResourceRole::OpticalFlow, "optical_flow", 1),
        ] {
            let names = (0..count).map(|index| format!("source-{index}")).collect();
            NativeModelResourceRequest::checked(role, category, names)?.validate_recipe()?;
        }
        assert!(matches!(
            NativeModelResourceRequest::checked(
                NativeModelResourceRole::Clip,
                "text_encoders",
                vec!["a".into(), "b".into(), "c".into(), "d".into()]
            )?
            .validate_recipe(),
            Ok(())
        ));
        assert!(matches!(
            NativeModelResourceRequest::checked(
                NativeModelResourceRole::Clip,
                "checkpoints",
                vec!["a".into(), "b".into()]
            )?
            .validate_recipe(),
            Err(NativeModelResourceServiceError::InvalidRequest)
        ));
        for (role, category) in [
            (NativeModelResourceRole::ControlNet, "controlnet"),
            (NativeModelResourceRole::FaceDetection, "face_detection"),
            (NativeModelResourceRole::Hooks, "hooks"),
            (NativeModelResourceRole::HookKeyframes, "hook_keyframes"),
            (
                NativeModelResourceRole::LatentOperation,
                "latent_operations",
            ),
            (NativeModelResourceRole::LoraModel, "loras"),
            (NativeModelResourceRole::ModelPatch, "model_patches"),
            (NativeModelResourceRole::UpscaleModel, "upscale_models"),
        ] {
            NativeModelResourceRequest::checked(role, category, vec!["owner-retained".into()])?
                .validate_recipe()?;
            assert!(matches!(
                NativeModelResourceRequest::checked(
                    role,
                    "checkpoints",
                    vec!["owner-retained".into()],
                )?
                .validate_recipe(),
                Err(NativeModelResourceServiceError::InvalidRequest)
            ));
            assert!(matches!(
                NativeModelResourceRequest::checked(
                    role,
                    category,
                    vec!["one".into(), "two".into()],
                )?
                .validate_recipe(),
                Err(NativeModelResourceServiceError::InvalidRequest)
            ));
        }
        Ok(())
    }
}
