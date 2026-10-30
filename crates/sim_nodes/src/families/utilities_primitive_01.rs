use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeHandleKind, NativeHandleStoreError, NativeHandleType,
    NativeInputDescriptor, NativeNode, NativeNodeBinding, NativeNodeBindingsFactory,
    NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure,
    NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation, NativeOutputDescriptor,
    NativePortCardinality, NativePrimitive, NativePrimitiveType, NativeStoredPayload,
    NativeTypeUnion, NativeValue, NativeValueType, built_in_source_schema,
};
use sim_media::{NativeBoundingBox, NativeBoundingBoxPayload};
use futures::future::BoxFuture;
use std::{collections::BTreeMap, sync::Arc};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "PrimitiveBoolean",
    "PrimitiveBoundingBox",
    "PrimitiveFloat",
    "PrimitiveInt",
    "PrimitiveString",
    "PrimitiveStringMultiline",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

const CATEGORY: &str = "utilities/primitive";
const MAX_RESOLUTION: u64 = 16_384;
const SOURCE_INTEGER_MAXIMUM: u64 = i64::MAX as u64;
const SOURCE_FLOAT_MAXIMUM: f64 = i64::MAX as f64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PrimitiveKind {
    Boolean,
    BoundingBox,
    Float,
    Int,
    String,
    StringMultiline,
}

impl PrimitiveKind {
    const ALL: [Self; 6] = [
        Self::Boolean,
        Self::BoundingBox,
        Self::Float,
        Self::Int,
        Self::String,
        Self::StringMultiline,
    ];

    const fn class_type(self) -> &'static str {
        match self {
            Self::Boolean => "PrimitiveBoolean",
            Self::BoundingBox => "PrimitiveBoundingBox",
            Self::Float => "PrimitiveFloat",
            Self::Int => "PrimitiveInt",
            Self::String => "PrimitiveString",
            Self::StringMultiline => "PrimitiveStringMultiline",
        }
    }

    const fn feature_id(self) -> &'static str {
        match self {
            Self::Boolean => "COMFY-NODE-0494",
            Self::BoundingBox => "COMFY-NODE-0495",
            Self::Float => "COMFY-NODE-0496",
            Self::Int => "COMFY-NODE-0497",
            Self::String => "COMFY-NODE-0498",
            Self::StringMultiline => "COMFY-NODE-0499",
        }
    }

    const fn implementation_version(self) -> &'static str {
        match self {
            Self::BoundingBox => "source-a57638bf-v1",
            _ => "source-a64a4efa-v1",
        }
    }

    const fn display_name(self) -> &'static str {
        match self {
            Self::Boolean => "Boolean",
            Self::BoundingBox => "Bounding Box",
            Self::Float => "Float",
            Self::Int => "Int",
            Self::String => "Text String (DEPRECATED)",
            Self::StringMultiline => "Input Text",
        }
    }

    const fn output_name(self) -> &'static str {
        match self {
            Self::BoundingBox => "bounding_box",
            _ => "value",
        }
    }

    fn input_names(self) -> Vec<String> {
        match self {
            Self::BoundingBox => ["x", "y", "width", "height"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            _ => vec!["value".to_owned()],
        }
    }

    fn search_aliases(self) -> Vec<String> {
        match self {
            Self::String => ["text", "string", "text box", "prompt"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            Self::StringMultiline => [
                "text",
                "string",
                "text multiline",
                "string multiline",
                "text box",
                "prompt",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            _ => Vec::new(),
        }
    }
}

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    PrimitiveKind::ALL.into_iter().map(native_binding).collect()
}

fn native_binding(kind: PrimitiveKind) -> Result<NativeNodeBinding, NativeNodeContractError> {
    let input_names = kind.input_names();
    let source_schema = built_in_source_schema(kind.class_type())
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
        .bind_execution_ports(&input_names, &[], &[kind.output_name().to_owned()])
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let inputs = match kind {
        PrimitiveKind::Boolean => vec![input(
            "value",
            NativeValueType::Primitive(NativePrimitiveType::Boolean),
        )?],
        PrimitiveKind::BoundingBox => ["x", "y", "width", "height"]
            .into_iter()
            .map(|name| {
                input(
                    name,
                    NativeValueType::Primitive(NativePrimitiveType::Integer),
                )
            })
            .collect::<Result<Vec<_>, _>>()?,
        PrimitiveKind::Float => vec![input(
            "value",
            NativeValueType::Primitive(NativePrimitiveType::Number),
        )?],
        PrimitiveKind::Int => vec![input(
            "value",
            NativeValueType::Primitive(NativePrimitiveType::Integer),
        )?],
        PrimitiveKind::String | PrimitiveKind::StringMultiline => vec![input(
            "value",
            NativeValueType::Primitive(NativePrimitiveType::String),
        )?],
    };
    let produced_type = match kind {
        PrimitiveKind::Boolean => NativeValueType::Primitive(NativePrimitiveType::Boolean),
        PrimitiveKind::BoundingBox => NativeValueType::Handle(bounding_box_type()?),
        PrimitiveKind::Float => NativeValueType::Primitive(NativePrimitiveType::Number),
        PrimitiveKind::Int => NativeValueType::Primitive(NativePrimitiveType::Integer),
        PrimitiveKind::String | PrimitiveKind::StringMultiline => {
            NativeValueType::Primitive(NativePrimitiveType::String)
        }
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
            description: String::new(),
            output_names: vec![kind.output_name().to_owned()],
            search_aliases: kind.search_aliases(),
            is_deprecated: kind == PrimitiveKind::String,
            is_experimental: false,
        },
        node: Arc::new(PrimitiveNode { kind }),
    })
}

fn input(
    name: &str,
    value_type: NativeValueType,
) -> Result<NativeInputDescriptor, NativeNodeContractError> {
    Ok(NativeInputDescriptor {
        name: name.to_owned(),
        accepted_types: NativeTypeUnion::new([value_type])?,
        required: true,
        hidden: false,
        lazy: false,
        cardinality: NativePortCardinality::Scalar,
        allows_literal: true,
    })
}

fn bounding_box_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::StructuredCompute, "BOUNDING_BOX")
}

#[derive(Debug)]
struct PrimitiveNode {
    kind: PrimitiveKind,
}

impl NativeNode for PrimitiveNode {
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
        check_cancellation(context, self.kind.class_type())?;
        validate_inputs(self.kind, inputs)?;
        Ok(NativeCacheDependencies::default())
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check_cancellation(&context, self.kind.class_type())?;
            let outcome = match self.kind {
                PrimitiveKind::BoundingBox => execute_bounding_box(&context, &inputs)?,
                _ => {
                    let output = primitive_output(self.kind, &inputs)?;
                    check_cancellation(&context, self.kind.class_type())?;
                    value_outcome(output)?
                }
            };
            Ok(outcome)
        })
    }
}

fn validate_inputs(
    kind: PrimitiveKind,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(), NativeNodeFailure> {
    match kind {
        PrimitiveKind::BoundingBox => {
            bounding_box_values(inputs)?;
        }
        _ => {
            primitive_output(kind, inputs)?;
        }
    }
    Ok(())
}

fn primitive_output(
    kind: PrimitiveKind,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeValue, NativeNodeFailure> {
    if inputs.len() != 1 {
        return Err(invalid_inputs(format!(
            "{} requires exactly value",
            kind.class_type()
        )));
    }
    let value = inputs
        .get("value")
        .ok_or_else(|| invalid_inputs(format!("{} requires value", kind.class_type())))?;
    value
        .validate()
        .map_err(|error| invalid_inputs(error.to_string()))?;
    let primitive = match (kind, value) {
        (
            PrimitiveKind::Boolean,
            NativeValue::Primitive {
                value: NativePrimitive::Boolean(value),
            },
        ) => NativePrimitive::Boolean(*value),
        (PrimitiveKind::Float, value) => float_value(value)?,
        (
            PrimitiveKind::Int,
            NativeValue::Primitive {
                value: value @ (NativePrimitive::Integer(_) | NativePrimitive::UnsignedInteger(_)),
            },
        ) => {
            validate_source_integer(value, "value")?;
            value.clone()
        }
        (
            PrimitiveKind::String | PrimitiveKind::StringMultiline,
            NativeValue::Primitive {
                value: NativePrimitive::String(value),
            },
        ) => NativePrimitive::String(value.clone()),
        (PrimitiveKind::Boolean, _) => {
            return Err(invalid_inputs("PrimitiveBoolean value must be a BOOLEAN"));
        }
        (PrimitiveKind::Int, _) => {
            return Err(invalid_inputs("PrimitiveInt value must be an INT"));
        }
        (PrimitiveKind::String, _) => {
            return Err(invalid_inputs("PrimitiveString value must be a STRING"));
        }
        (PrimitiveKind::StringMultiline, _) => {
            return Err(invalid_inputs(
                "PrimitiveStringMultiline value must be a STRING",
            ));
        }
        (PrimitiveKind::BoundingBox, _) => {
            return Err(invalid_inputs(
                "PrimitiveBoundingBox requires structured inputs",
            ));
        }
    };
    Ok(NativeValue::Primitive { value: primitive })
}

fn float_value(value: &NativeValue) -> Result<NativePrimitive, NativeNodeFailure> {
    let (numeric, primitive) = match value {
        NativeValue::Primitive {
            value: NativePrimitive::Number(value),
        } => (*value, NativePrimitive::Number(*value)),
        NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        } => {
            validate_source_integer(&NativePrimitive::Integer(*value), "value")?;
            (*value as f64, NativePrimitive::Integer(*value))
        }
        NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        } => {
            validate_source_integer(&NativePrimitive::UnsignedInteger(*value), "value")?;
            (*value as f64, NativePrimitive::UnsignedInteger(*value))
        }
        _ => return Err(invalid_inputs("PrimitiveFloat value must be a FLOAT")),
    };
    if !numeric.is_finite() || !(-SOURCE_FLOAT_MAXIMUM..=SOURCE_FLOAT_MAXIMUM).contains(&numeric) {
        return Err(invalid_inputs(
            "PrimitiveFloat value must be finite and within source sys.maxsize bounds",
        ));
    }
    Ok(primitive)
}

fn validate_source_integer(value: &NativePrimitive, name: &str) -> Result<(), NativeNodeFailure> {
    match value {
        NativePrimitive::Integer(value) if *value != i64::MIN => Ok(()),
        NativePrimitive::UnsignedInteger(value) if *value <= SOURCE_INTEGER_MAXIMUM => Ok(()),
        NativePrimitive::Integer(_) | NativePrimitive::UnsignedInteger(_) => Err(invalid_inputs(
            format!("{name} must be between -sys.maxsize and sys.maxsize"),
        )),
        _ => Err(invalid_inputs(format!("{name} must be an INT"))),
    }
}

fn bounding_box_values(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(u64, u64, u64, u64), NativeNodeFailure> {
    if inputs.len() != 4 {
        return Err(invalid_inputs(
            "PrimitiveBoundingBox requires exactly x, y, width, and height",
        ));
    }
    Ok((
        bounded_unsigned(inputs.get("x"), "x", 0, MAX_RESOLUTION)?,
        bounded_unsigned(inputs.get("y"), "y", 0, MAX_RESOLUTION)?,
        bounded_unsigned(inputs.get("width"), "width", 1, MAX_RESOLUTION)?,
        bounded_unsigned(inputs.get("height"), "height", 1, MAX_RESOLUTION)?,
    ))
}

fn bounded_unsigned(
    value: Option<&NativeValue>,
    name: &str,
    minimum: u64,
    maximum: u64,
) -> Result<u64, NativeNodeFailure> {
    let value = match value {
        Some(NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }) => *value,
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => u64::try_from(*value)
            .map_err(|_| invalid_inputs(format!("{name} must be non-negative")))?,
        _ => return Err(invalid_inputs(format!("{name} must be an INT"))),
    };
    if !(minimum..=maximum).contains(&value) {
        return Err(invalid_inputs(format!(
            "{name} must be between {minimum} and {maximum}"
        )));
    }
    Ok(value)
}

fn execute_bounding_box(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let (x, y, width, height) = bounding_box_values(inputs)?;
    let bounding_box =
        NativeBoundingBox::checked(x as f64, y as f64, width as f64, height as f64, None, None)
            .map_err(native_failure)?;
    let payload =
        NativeBoundingBoxPayload::checked(vec![vec![bounding_box]]).map_err(native_failure)?;
    check_cancellation(context, PrimitiveKind::BoundingBox.class_type())?;
    let handle = context
        .handle_store()
        .publish(
            NativeStoredPayload::BoundingBox(Arc::new(payload)),
            &context.cancellation,
        )
        .map_err(handle_failure)?;
    value_outcome(NativeValue::Handle { value: handle })
}

fn value_outcome(output: NativeValue) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let outcome = NativeNodeOutcome::Values {
        outputs: vec![output],
        ui: None,
        effects: Vec::new(),
    };
    outcome
        .validate()
        .map_err(|error| invalid_inputs(error.to_string()))?;
    Ok(outcome)
}

fn check_cancellation(
    context: &NativeNodeContext,
    class_type: &str,
) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|_| interrupted_failure(class_type))
}

fn handle_failure(error: NativeHandleStoreError) -> NativeNodeFailure {
    if matches!(error, NativeHandleStoreError::Cancelled) {
        interrupted_failure(PrimitiveKind::BoundingBox.class_type())
    } else {
        NativeNodeFailure {
            code: "native_bounding_box_store_failed".to_owned(),
            message: format!("PrimitiveBoundingBox could not publish BOUNDING_BOX: {error}"),
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

fn native_failure(error: impl std::fmt::Display) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "native_primitive_failed".to_owned(),
        message: error.to_string(),
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
        NativeHandleStore, NativeHandleStoreIdentity, NativeOpaqueHandle, NativeResolvedPayload,
        NativeResolvedPayloadRetention, NodeRegistry,
    };
    use sim_tensor::CpuWorkspaceAuthority;
    use sim_types::{AttemptId, CancellationToken, NodeId, PromptId};
    use serde_json::Value;
    use std::sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    };
    use uuid::Uuid;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/nodes/utilities-primitive-comfy-node-0494/fixture.json"
    ));

    #[derive(Debug)]
    struct TestRetention;

    impl NativeResolvedPayloadRetention for TestRetention {}

    #[derive(Debug)]
    struct TestStore {
        identity: NativeHandleStoreIdentity,
        attempt_id: AttemptId,
        next_generation: AtomicU64,
        values: Mutex<BTreeMap<String, Arc<NativeStoredPayload>>>,
    }

    impl TestStore {
        fn new(identity: u128) -> Result<Arc<Self>, NativeNodeContractError> {
            Ok(Arc::new(Self {
                identity: NativeHandleStoreIdentity::new(
                    Uuid::from_u128(identity),
                    Uuid::from_u128(identity + 1),
                )?,
                attempt_id: AttemptId(Uuid::from_u128(0x47503)),
                next_generation: AtomicU64::new(1),
                values: Mutex::new(BTreeMap::new()),
            }))
        }

        fn count(&self) -> Result<usize, NativeHandleStoreError> {
            self.values
                .lock()
                .map(|values| values.len())
                .map_err(|_| NativeHandleStoreError::Rejected("test store is poisoned".to_owned()))
        }
    }

    impl NativeHandleStore for TestStore {
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
            cancellation
                .check()
                .map_err(|_| NativeHandleStoreError::Cancelled)?;
            if handle.store_identity() != self.identity {
                return Err(NativeHandleStoreError::WrongStore);
            }
            if handle.handle_type() != expected_type {
                return Err(NativeHandleStoreError::WrongType {
                    expected: expected_type.type_id.clone(),
                    actual: handle.handle_type().type_id.clone(),
                });
            }
            let values = self.values.lock().map_err(|_| {
                NativeHandleStoreError::Rejected("test store is poisoned".to_owned())
            })?;
            let payload = values
                .get(handle.identifier())
                .ok_or_else(|| NativeHandleStoreError::Missing(handle.identifier().to_owned()))?;
            if payload.digest_sha256() != handle.digest_sha256().unwrap_or_default() {
                return Err(NativeHandleStoreError::DigestMismatch);
            }
            Ok(NativeResolvedPayload::checked(
                payload.clone(),
                Arc::new(TestRetention),
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
            let generation = self.next_generation.fetch_add(1, Ordering::AcqRel);
            let identifier = format!("primitive-{generation}");
            let handle = NativeOpaqueHandle::new(
                payload.handle_type()?,
                self.identity,
                identifier.clone(),
                generation,
                Some(payload.digest_sha256()),
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
            cancellation
                .check()
                .map_err(|_| NativeHandleStoreError::Cancelled)?;
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

    struct Harness {
        store: Arc<TestStore>,
    }

    impl Harness {
        fn new(identity: u128) -> Result<Self, NativeNodeContractError> {
            Ok(Self {
                store: TestStore::new(identity)?,
            })
        }

        fn context(
            &self,
            cancellation: CancellationToken,
        ) -> Result<NativeNodeContext, Box<dyn std::error::Error>> {
            let (_backend, authority) = CpuWorkspaceAuthority::create_backend(1)?;
            Ok(NativeNodeContext::new(
                PromptId(Uuid::from_u128(0x47504)),
                self.store.attempt_id,
                NodeId("utilities-primitive-test".to_owned()),
                cancellation,
                authority.authorize_workspace(0)?,
                self.store.clone(),
            )?)
        }
    }

    fn executable(class_type: &str) -> Result<Arc<dyn NativeNode>, Box<dyn std::error::Error>> {
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

    fn one_input(value: NativePrimitive) -> BTreeMap<String, NativeValue> {
        BTreeMap::from([("value".to_owned(), NativeValue::Primitive { value })])
    }

    fn output_value(outcome: NativeNodeOutcome) -> Result<NativeValue, Box<dyn std::error::Error>> {
        let NativeNodeOutcome::Values {
            mut outputs,
            ui,
            effects,
        } = outcome
        else {
            return Err("primitive node did not return values".into());
        };
        assert!(ui.is_none());
        assert!(effects.is_empty());
        if outputs.len() != 1 {
            return Err("primitive node output count changed".into());
        }
        outputs
            .pop()
            .ok_or_else(|| "primitive node output is absent".into())
    }

    #[test]
    fn source_fixture_and_descriptors_are_exact() -> Result<(), Box<dyn std::error::Error>> {
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        assert_eq!(
            fixture["stable_task_id"],
            "comfy-parity-native-nodes-utilities-primitive-comfy-node-0494"
        );
        assert_eq!(
            fixture["sources"][0]["sha256"],
            "a64a4efa55d36016d01587fefac37f98c70e42b1d9757b7b308b720ac0d966b6"
        );
        assert_eq!(
            fixture["sources"][1]["sha256"],
            "a57638bf58f93698b8b5ed44e53d31ac3d4ac608cc5c49d38c3768fdf722e448"
        );
        assert_eq!(
            fixture["sources"][0]["symbols"].as_array().map(Vec::len),
            Some(5)
        );
        assert_eq!(
            fixture["sources"][1]["symbols"].as_array().map(Vec::len),
            Some(1)
        );
        let bindings = native_node_bindings()?;
        assert_eq!(bindings.len(), NODE_DESCRIPTOR_IDS.len());
        let registry = NodeRegistry::built_in()?;
        for (binding, class_type) in bindings.iter().zip(NODE_DESCRIPTOR_IDS) {
            assert_eq!(binding.descriptor().class_type, *class_type);
            binding.validate()?;
            registry.validate_native_binding(binding)?;
            assert_eq!(binding.descriptor().effect, NativeEffectClass::Pure);
            assert_eq!(binding.descriptor().cache, NativeCachePolicy::InputIdentity);
        }
        Ok(())
    }

    #[test]
    fn scalar_primitives_preserve_source_values_and_numeric_boundaries()
    -> Result<(), Box<dyn std::error::Error>> {
        let harness = Harness::new(0x47510)?;
        let cases = [
            (
                PrimitiveKind::Boolean,
                NativePrimitive::Boolean(true),
                NativePrimitive::Boolean(true),
            ),
            (
                PrimitiveKind::Int,
                NativePrimitive::Integer(i64::MAX),
                NativePrimitive::Integer(i64::MAX),
            ),
            (
                PrimitiveKind::Float,
                NativePrimitive::Integer(7),
                NativePrimitive::Integer(7),
            ),
            (
                PrimitiveKind::Float,
                NativePrimitive::Number(-0.25),
                NativePrimitive::Number(-0.25),
            ),
            (
                PrimitiveKind::String,
                NativePrimitive::String("deprecated identity".to_owned()),
                NativePrimitive::String("deprecated identity".to_owned()),
            ),
        ];
        for (kind, input, expected) in cases {
            let outcome = futures::executor::block_on(executable(kind.class_type())?.execute(
                harness.context(CancellationToken::default())?,
                one_input(input),
            ))?;
            assert_eq!(
                output_value(outcome)?,
                NativeValue::Primitive { value: expected }
            );
        }
        assert_eq!(harness.store.count()?, 0);
        assert!(
            primitive_output(
                PrimitiveKind::Int,
                &one_input(NativePrimitive::Integer(i64::MIN))
            )
            .is_err()
        );
        assert!(
            primitive_output(
                PrimitiveKind::Float,
                &one_input(NativePrimitive::Number(f64::MAX))
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn multiline_string_preserves_bounded_workflow_text_exactly()
    -> Result<(), Box<dyn std::error::Error>> {
        let harness = Harness::new(0x47520)?;
        for text in ["", "first\tcolumn\r\nsecond line\u{2028}third", "雪🙂"] {
            let outcome = futures::executor::block_on(
                executable(PrimitiveKind::StringMultiline.class_type())?.execute(
                    harness.context(CancellationToken::default())?,
                    one_input(NativePrimitive::String(text.to_owned())),
                ),
            )?;
            assert_eq!(
                output_value(outcome)?,
                NativeValue::Primitive {
                    value: NativePrimitive::String(text.to_owned())
                }
            );
        }
        for invalid in ["nul\0text", "bell\u{0007}text"] {
            assert!(
                primitive_output(
                    PrimitiveKind::StringMultiline,
                    &one_input(NativePrimitive::String(invalid.to_owned()))
                )
                .is_err()
            );
        }
        assert!(
            primitive_output(
                PrimitiveKind::StringMultiline,
                &one_input(NativePrimitive::String("a".repeat(1024 * 1024 + 1)))
            )
            .is_err()
        );
        Ok(())
    }

    fn bounding_inputs(x: u64, y: u64, width: u64, height: u64) -> BTreeMap<String, NativeValue> {
        [("x", x), ("y", y), ("width", width), ("height", height)]
            .into_iter()
            .map(|(name, value)| {
                (
                    name.to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::UnsignedInteger(value),
                    },
                )
            })
            .collect()
    }

    #[test]
    fn bounding_box_uses_canonical_payload_and_restart_recovery()
    -> Result<(), Box<dyn std::error::Error>> {
        let first = Harness::new(0x47530)?;
        let outcome = futures::executor::block_on(
            executable(PrimitiveKind::BoundingBox.class_type())?.execute(
                first.context(CancellationToken::default())?,
                bounding_inputs(3, 4, 512, 256),
            ),
        )?;
        let NativeValue::Handle { value: handle } = output_value(outcome)? else {
            return Err("bounding box output is not a handle".into());
        };
        let resolved = first.store.resolve(
            &handle,
            &bounding_box_type()?,
            &CancellationToken::default(),
        )?;
        let NativeStoredPayload::BoundingBox(payload) = resolved.as_ref() else {
            return Err("bounding box output did not use its canonical payload".into());
        };
        let [frame] = payload.frames() else {
            return Err("bounding box payload frame count changed".into());
        };
        let [bounding_box] = frame.as_ref() else {
            return Err("bounding box payload item count changed".into());
        };
        assert_eq!(
            (
                bounding_box.x(),
                bounding_box.y(),
                bounding_box.width(),
                bounding_box.height(),
            ),
            (3.0, 4.0, 512.0, 256.0)
        );
        assert!(bounding_box.label().is_none());
        assert!(bounding_box.score().is_none());

        let restarted = Harness::new(0x47540)?;
        assert!(matches!(
            restarted.store.resolve(
                &handle,
                &bounding_box_type()?,
                &CancellationToken::default()
            ),
            Err(NativeHandleStoreError::WrongStore)
        ));
        let fresh = futures::executor::block_on(
            executable(PrimitiveKind::BoundingBox.class_type())?.execute(
                restarted.context(CancellationToken::default())?,
                bounding_inputs(0, MAX_RESOLUTION, 1, MAX_RESOLUTION),
            ),
        )?;
        assert!(matches!(output_value(fresh)?, NativeValue::Handle { .. }));
        assert_eq!(restarted.store.count()?, 1);
        assert!(bounding_box_values(&bounding_inputs(0, 0, 0, 1)).is_err());
        assert!(bounding_box_values(&bounding_inputs(MAX_RESOLUTION + 1, 0, 1, 1)).is_err());
        Ok(())
    }

    #[test]
    fn cancellation_and_input_shape_fail_before_publication()
    -> Result<(), Box<dyn std::error::Error>> {
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let harness = Harness::new(0x47550)?;
        let failure = futures::executor::block_on(
            executable(PrimitiveKind::BoundingBox.class_type())?
                .execute(harness.context(cancellation)?, bounding_inputs(0, 0, 1, 1)),
        )
        .expect_err("cancelled bounding box must fail");
        assert_eq!(failure.kind, NativeNodeFailureKind::Interrupted);
        assert_eq!(harness.store.count()?, 0);
        assert!(primitive_output(PrimitiveKind::Boolean, &BTreeMap::new()).is_err());
        assert!(
            primitive_output(
                PrimitiveKind::Boolean,
                &BTreeMap::from([
                    (
                        "value".to_owned(),
                        NativeValue::Primitive {
                            value: NativePrimitive::Boolean(true),
                        },
                    ),
                    (
                        "extra".to_owned(),
                        NativeValue::Primitive {
                            value: NativePrimitive::Boolean(false),
                        },
                    ),
                ])
            )
            .is_err()
        );
        Ok(())
    }
}
