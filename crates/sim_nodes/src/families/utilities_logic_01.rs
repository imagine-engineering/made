use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeDynamicInputDescriptor, NativeEffectClass, NativeInputDescriptor, NativeNode,
    NativeNodeBinding, NativeNodeBindingsFactory, NativeNodeContext, NativeNodeContractError,
    NativeNodeDescriptor, NativeNodeFailure, NativeNodeFailureKind, NativeNodeOutcome,
    NativeNodePresentation, NativeOutputDescriptor, NativePortCardinality, NativePrimitive,
    NativePrimitiveType, NativeTypeUnion, NativeValue, NativeValueType, built_in_source_schema,
};
use futures::future::BoxFuture;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "ComfyAndNode",
    "ComfyNotNode",
    "ComfyOrNode",
    "ComfySwitchNode",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

const CATEGORY: &str = "utilities/logic";
const IMPLEMENTATION_VERSION: &str = "source-2cb1ce14-v1";
const MAX_AUTOGROW_INPUTS: usize = 65_536;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LogicKind {
    And,
    Not,
    Or,
    Switch,
}

impl LogicKind {
    const fn class_type(self) -> &'static str {
        match self {
            Self::And => "ComfyAndNode",
            Self::Not => "ComfyNotNode",
            Self::Or => "ComfyOrNode",
            Self::Switch => "ComfySwitchNode",
        }
    }

    const fn feature_id(self) -> &'static str {
        match self {
            Self::And => "COMFY-NODE-0082",
            Self::Not => "COMFY-NODE-0084",
            Self::Or => "COMFY-NODE-0086",
            Self::Switch => "COMFY-NODE-0087",
        }
    }

    const fn display_name(self) -> &'static str {
        match self {
            Self::And => "And",
            Self::Not => "Not",
            Self::Or => "Or",
            Self::Switch => "If/Else Switch",
        }
    }

    const fn description(self) -> &'static str {
        match self {
            Self::And => {
                "Logical AND operation. Returns true if all of the values are truthy. Uses Python's rules for truthiness."
            }
            Self::Not => {
                "Logical NOT operation. Returns true if the value is falsy. Uses Python's rules for truthiness."
            }
            Self::Or => {
                "Logical OR operation. Returns true if any of the values are truthy. Uses Python's rules for truthiness."
            }
            Self::Switch => "",
        }
    }

    fn search_aliases(self) -> Vec<String> {
        match self {
            Self::And => vec!["all".to_owned(), "every".to_owned()],
            Self::Not => vec![
                "invert".to_owned(),
                "toggle".to_owned(),
                "negate".to_owned(),
                "flip boolean".to_owned(),
            ],
            Self::Or => vec!["any".to_owned(), "some".to_owned()],
            Self::Switch => vec![
                "if".to_owned(),
                "then".to_owned(),
                "switch".to_owned(),
                "conditional".to_owned(),
                "branch".to_owned(),
            ],
        }
    }
}

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    [LogicKind::And, LogicKind::Not, LogicKind::Or, LogicKind::Switch]
        .into_iter()
        .map(native_binding)
        .collect()
}

fn native_binding(kind: LogicKind) -> Result<NativeNodeBinding, NativeNodeContractError> {
    let (input_names, output_name) = match kind {
        LogicKind::And | LogicKind::Or => (Vec::new(), "result"),
        LogicKind::Not => (vec!["value".to_owned()], "result"),
        LogicKind::Switch => (
            vec![
                "switch".to_owned(),
                "on_false".to_owned(),
                "on_true".to_owned(),
            ],
            "output",
        ),
    };
    let catalog_schema = built_in_source_schema(kind.class_type())
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let dynamic_schema = catalog_schema.dynamic_inputs.clone();
    let source_schema = catalog_schema
        .bind_execution_ports(&input_names, &dynamic_schema, &[output_name.to_owned()])
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let inputs = match kind {
        LogicKind::And | LogicKind::Or => Vec::new(),
        LogicKind::Not => vec![input("value", NativeValueType::Any, false, false)?],
        LogicKind::Switch => vec![
            input(
                "switch",
                NativeValueType::Primitive(NativePrimitiveType::Boolean),
                false,
                true,
            )?,
            input("on_false", NativeValueType::Any, true, false)?,
            input("on_true", NativeValueType::Any, true, false)?,
        ],
    };
    let dynamic_inputs = if matches!(kind, LogicKind::And | LogicKind::Or) {
        vec![NativeDynamicInputDescriptor {
            name_template: "value{index}".to_owned(),
            start_index: 1,
            minimum_count: 1,
            maximum_count: MAX_AUTOGROW_INPUTS as u32,
            input: input("value", NativeValueType::Any, false, false)?,
        }]
    } else {
        Vec::new()
    };
    let produced_type = if kind == LogicKind::Switch {
        NativeValueType::Any
    } else {
        NativeValueType::Primitive(NativePrimitiveType::Boolean)
    };
    Ok(NativeNodeBinding::Executable {
        feature_id: kind.feature_id().to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: kind.class_type().to_owned(),
            implementation_version: IMPLEMENTATION_VERSION.to_owned(),
            source_schema: Some(source_schema),
            inputs,
            dynamic_inputs,
            outputs: vec![NativeOutputDescriptor {
                name: output_name.to_owned(),
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
            output_names: vec![output_name.to_owned()],
            search_aliases: kind.search_aliases(),
            is_deprecated: false,
            is_experimental: kind == LogicKind::Switch,
        },
        node: Arc::new(LogicNode { kind }),
    })
}

fn input(
    name: &str,
    value_type: NativeValueType,
    lazy: bool,
    allows_literal: bool,
) -> Result<NativeInputDescriptor, NativeNodeContractError> {
    Ok(NativeInputDescriptor {
        name: name.to_owned(),
        accepted_types: NativeTypeUnion::new([value_type])?,
        required: true,
        hidden: false,
        lazy,
        cardinality: NativePortCardinality::Scalar,
        allows_literal,
    })
}

#[derive(Debug)]
struct LogicNode {
    kind: LogicKind,
}

impl NativeNode for LogicNode {
    fn class_type(&self) -> &str {
        self.kind.class_type()
    }

    fn implementation_version(&self) -> &str {
        IMPLEMENTATION_VERSION
    }

    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        available_inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<BTreeSet<String>, NativeNodeFailure> {
        check_cancellation(context, self.kind.class_type())?;
        if self.kind != LogicKind::Switch {
            return Ok(BTreeSet::new());
        }
        validate_switch_keys(available_inputs)?;
        let selected = selected_switch_input(available_inputs)?;
        Ok(if available_inputs.contains_key(selected) {
            BTreeSet::new()
        } else {
            BTreeSet::from([selected.to_owned()])
        })
    }

    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        validate_inputs(self.kind, inputs)?;
        Ok(format!(
            "{}-{IMPLEMENTATION_VERSION}",
            self.kind.class_type()
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
            validate_inputs(self.kind, &inputs)?;
            let output = match self.kind {
                LogicKind::And => boolean_value(inputs.values().all(python_truthy)),
                LogicKind::Not => {
                    let value = inputs
                        .get("value")
                        .ok_or_else(|| invalid_inputs("ComfyNotNode requires value"))?;
                    boolean_value(!python_truthy(value))
                }
                LogicKind::Or => boolean_value(inputs.values().any(python_truthy)),
                LogicKind::Switch => {
                    let selected = selected_switch_input(&inputs)?;
                    inputs
                        .get(selected)
                        .cloned()
                        .ok_or_else(|| invalid_inputs(format!("missing selected input {selected}")))?
                }
            };
            check_cancellation(&context, self.kind.class_type())?;
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
    kind: LogicKind,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(), NativeNodeFailure> {
    match kind {
        LogicKind::And | LogicKind::Or => validate_autogrow_inputs(inputs),
        LogicKind::Not => {
            if inputs.len() == 1 && inputs.contains_key("value") {
                Ok(())
            } else {
                Err(invalid_inputs("ComfyNotNode requires exactly value"))
            }
        }
        LogicKind::Switch => {
            validate_switch_keys(inputs)?;
            let selected = selected_switch_input(inputs)?;
            if inputs.contains_key(selected) {
                Ok(())
            } else {
                Err(invalid_inputs(format!("missing selected input {selected}")))
            }
        }
    }
}

fn validate_autogrow_inputs(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(), NativeNodeFailure> {
    if inputs.is_empty() || inputs.len() > MAX_AUTOGROW_INPUTS {
        return Err(invalid_inputs(format!(
            "logic autogrow requires between 1 and {MAX_AUTOGROW_INPUTS} values"
        )));
    }
    let mut indices = inputs
        .keys()
        .map(|name| {
            name.strip_prefix("value")
                .and_then(|suffix| suffix.parse::<usize>().ok())
                .filter(|index| (1..=MAX_AUTOGROW_INPUTS).contains(index))
                .ok_or_else(|| invalid_inputs(format!("invalid autogrow input {name}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    indices.sort_unstable();
    if !indices
        .iter()
        .copied()
        .eq(1..=indices.len())
    {
        return Err(invalid_inputs(
            "logic autogrow inputs must be contiguous from value1",
        ));
    }
    Ok(())
}

fn validate_switch_keys(inputs: &BTreeMap<String, NativeValue>) -> Result<(), NativeNodeFailure> {
    if inputs
        .keys()
        .any(|name| !matches!(name.as_str(), "switch" | "on_false" | "on_true"))
    {
        return Err(invalid_inputs("ComfySwitchNode received an unknown input"));
    }
    selected_switch_input(inputs)?;
    Ok(())
}

fn selected_switch_input(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<&'static str, NativeNodeFailure> {
    match inputs.get("switch") {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Boolean(true),
        }) => Ok("on_true"),
        Some(NativeValue::Primitive {
            value: NativePrimitive::Boolean(false),
        }) => Ok("on_false"),
        _ => Err(invalid_inputs("switch must be a BOOLEAN")),
    }
}

fn python_truthy(value: &NativeValue) -> bool {
    match value {
        NativeValue::Primitive { value } => match value {
            NativePrimitive::Null => false,
            NativePrimitive::Boolean(value) => *value,
            NativePrimitive::Integer(value) => *value != 0,
            NativePrimitive::UnsignedInteger(value) => *value != 0,
            NativePrimitive::Number(value) => *value != 0.0,
            NativePrimitive::String(value) => !value.is_empty(),
        },
        NativeValue::Handle { .. } => true,
        NativeValue::List { values } => !values.is_empty(),
        NativeValue::PreservedUnknown { value, .. } => json_truthy(value),
    }
}

fn json_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                value != 0
            } else if let Some(value) = value.as_u64() {
                value != 0
            } else {
                value.as_f64().is_some_and(|value| value != 0.0)
            }
        }
        Value::String(value) => !value.is_empty(),
        Value::Array(values) => !values.is_empty(),
        Value::Object(values) => !values.is_empty(),
    }
}

fn boolean_value(value: bool) -> NativeValue {
    NativeValue::Primitive {
        value: NativePrimitive::Boolean(value),
    }
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
    use sim_types::{AttemptId, CancellationToken, NodeId, PromptId};
    use serde_json::json;
    use std::error::Error;
    use uuid::Uuid;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/nodes/utilities-logic-comfy-node-0082/fixture.json"
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
                "logic nodes do not publish handles".to_owned(),
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

    fn context(
        attempt_id: AttemptId,
        cancellation: CancellationToken,
    ) -> Result<NativeNodeContext, Box<dyn Error>> {
        let (_backend, workspace) = CpuWorkspaceAuthority::create_backend(1)?;
        let store = Arc::new(RejectingStore {
            identity: NativeHandleStoreIdentity::new(
                Uuid::from_u128(0x47110),
                Uuid::from_u128(0x47111),
            )?,
            attempt_id,
        });
        Ok(NativeNodeContext::new(
            PromptId(Uuid::from_u128(0x47112)),
            attempt_id,
            NodeId("utilities-logic-test".to_owned()),
            cancellation,
            workspace.authorize_workspace(0)?,
            store,
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
        attempt: u128,
    ) -> Result<NativeValue, Box<dyn Error>> {
        let outcome = futures::executor::block_on(executable(class_type)?.execute(
            context(
                AttemptId(Uuid::from_u128(attempt)),
                CancellationToken::default(),
            )?,
            inputs,
        ))?;
        let NativeNodeOutcome::Values {
            mut outputs,
            ui,
            effects,
        } = outcome
        else {
            return Err("logic node did not produce values".into());
        };
        assert!(ui.is_none());
        assert!(effects.is_empty());
        if outputs.len() != 1 {
            return Err("logic node produced the wrong output count".into());
        }
        outputs
            .pop()
            .ok_or_else(|| "logic node output is absent".into())
    }

    fn preserved(value: Value) -> NativeValue {
        NativeValue::PreservedUnknown {
            type_name: "JSON".to_owned(),
            value,
        }
    }

    #[test]
    fn source_fixture_and_exact_schemas_are_registered() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        assert_eq!(
            fixture
                .pointer("/stable_task_id")
                .and_then(Value::as_str),
            Some("comfy-parity-native-nodes-utilities-logic-comfy-node-0082")
        );
        assert_eq!(
            fixture.pointer("/source/sha256").and_then(Value::as_str),
            Some("2cb1ce149ce616c0b12a67962ed7b799eb398fbcdeb20dcec4c5210eb6af6df1")
        );
        assert_eq!(
            fixture.pointer("/source/byte_length").and_then(Value::as_u64),
            Some(12_488)
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

        let and = bindings
            .iter()
            .find(|binding| binding.descriptor().class_type == "ComfyAndNode")
            .ok_or("ComfyAndNode binding is absent")?;
        assert!(and.descriptor().inputs.is_empty());
        assert_eq!(and.descriptor().dynamic_inputs.len(), 1);
        let dynamic = and
            .descriptor()
            .dynamic_inputs
            .first()
            .ok_or("ComfyAndNode dynamic input is absent")?;
        assert_eq!(dynamic.name_template, "value{index}");
        assert_eq!(dynamic.start_index, 1);
        assert_eq!(dynamic.minimum_count, 1);
        assert_eq!(dynamic.maximum_count, 65_536);
        assert_eq!(dynamic.input.accepted_types.members(), [NativeValueType::Any]);

        let switch = bindings
            .iter()
            .find(|binding| binding.descriptor().class_type == "ComfySwitchNode")
            .ok_or("ComfySwitchNode binding is absent")?;
        assert_eq!(
            switch
                .descriptor()
                .inputs
                .iter()
                .map(|input| (input.name.as_str(), input.lazy))
                .collect::<Vec<_>>(),
            [("switch", false), ("on_false", true), ("on_true", true)]
        );
        assert_eq!(
            switch
                .descriptor()
                .outputs
                .first()
                .ok_or("switch output descriptor is missing")?
                .produced_type,
            NativeValueType::Any
        );
        assert!(switch.presentation().is_experimental);
        Ok(())
    }

    #[test]
    fn python_truthiness_and_logic_results_match_source() -> Result<(), Box<dyn Error>> {
        let falsy = [
            NativeValue::Primitive {
                value: NativePrimitive::Null,
            },
            boolean_value(false),
            NativeValue::Primitive {
                value: NativePrimitive::Integer(0),
            },
            NativeValue::Primitive {
                value: NativePrimitive::Number(-0.0),
            },
            NativeValue::Primitive {
                value: NativePrimitive::String(String::new()),
            },
            NativeValue::List { values: Vec::new() },
            preserved(json!({})),
        ];
        assert!(falsy.iter().all(|value| !python_truthy(value)));

        let truthy = [
            boolean_value(true),
            NativeValue::Primitive {
                value: NativePrimitive::Integer(-1),
            },
            NativeValue::Primitive {
                value: NativePrimitive::Number(0.5),
            },
            NativeValue::Primitive {
                value: NativePrimitive::String("text".to_owned()),
            },
            NativeValue::List {
                values: vec![boolean_value(false)],
            },
            preserved(json!({"value": null})),
        ];
        assert!(truthy.iter().all(python_truthy));

        assert_eq!(
            execute(
                "ComfyAndNode",
                BTreeMap::from([
                    ("value1".to_owned(), boolean_value(true)),
                    (
                        "value2".to_owned(),
                        NativeValue::Primitive {
                            value: NativePrimitive::String("ready".to_owned()),
                        },
                    ),
                ]),
                0x47120,
            )?,
            boolean_value(true)
        );
        assert_eq!(
            execute(
                "ComfyAndNode",
                BTreeMap::from([
                    ("value1".to_owned(), boolean_value(true)),
                    ("value2".to_owned(), preserved(json!([]))),
                ]),
                0x47121,
            )?,
            boolean_value(false)
        );
        assert_eq!(
            execute(
                "ComfyOrNode",
                BTreeMap::from([
                    ("value1".to_owned(), boolean_value(false)),
                    ("value2".to_owned(), preserved(json!([null]))),
                ]),
                0x47122,
            )?,
            boolean_value(true)
        );
        assert_eq!(
            execute(
                "ComfyNotNode",
                BTreeMap::from([("value".to_owned(), preserved(json!({})))]),
                0x47123,
            )?,
            boolean_value(true)
        );
        Ok(())
    }

    #[test]
    fn switch_demands_only_selected_branch_and_preserves_value() -> Result<(), Box<dyn Error>> {
        let node = executable("ComfySwitchNode")?;
        let attempt_id = AttemptId(Uuid::from_u128(0x47130));
        let false_inputs = BTreeMap::from([("switch".to_owned(), boolean_value(false))]);
        assert_eq!(
            node.demanded_lazy_inputs(
                &context(attempt_id, CancellationToken::default())?,
                &false_inputs,
            )?,
            BTreeSet::from(["on_false".to_owned()])
        );
        let true_inputs = BTreeMap::from([("switch".to_owned(), boolean_value(true))]);
        assert_eq!(
            node.demanded_lazy_inputs(
                &context(attempt_id, CancellationToken::default())?,
                &true_inputs,
            )?,
            BTreeSet::from(["on_true".to_owned()])
        );

        let false_value = preserved(json!({"branch": "false"}));
        assert_eq!(
            execute(
                "ComfySwitchNode",
                BTreeMap::from([
                    ("switch".to_owned(), boolean_value(false)),
                    ("on_false".to_owned(), false_value.clone()),
                ]),
                0x47131,
            )?,
            false_value
        );
        let true_value = NativeValue::List {
            values: vec![boolean_value(false)],
        };
        assert_eq!(
            execute(
                "ComfySwitchNode",
                BTreeMap::from([
                    ("switch".to_owned(), boolean_value(true)),
                    ("on_false".to_owned(), boolean_value(false)),
                    ("on_true".to_owned(), true_value.clone()),
                ]),
                0x47132,
            )?,
            true_value
        );
        Ok(())
    }

    #[test]
    fn validation_cache_cancellation_and_fresh_attempt_recovery_are_exact()
    -> Result<(), Box<dyn Error>> {
        let and = executable("ComfyAndNode")?;
        assert!(and.cache_change_token(&BTreeMap::new()).is_err());
        assert!(
            and.cache_change_token(&BTreeMap::from([(
                "value2".to_owned(),
                boolean_value(true),
            )]))
            .is_err()
        );
        let values = BTreeMap::from([("value1".to_owned(), boolean_value(true))]);
        assert_eq!(
            and.cache_change_token(&values)?,
            "ComfyAndNode-source-2cb1ce14-v1"
        );
        let attempt_id = AttemptId(Uuid::from_u128(0x47140));
        assert_eq!(
            and.cache_dependencies(
                &context(attempt_id, CancellationToken::default())?,
                &values,
            )?,
            NativeCacheDependencies::default()
        );

        let switch = executable("ComfySwitchNode")?;
        assert!(
            switch
                .cache_change_token(&BTreeMap::from([(
                    "switch".to_owned(),
                    boolean_value(true),
                )]))
                .is_err()
        );
        assert!(
            switch
                .cache_change_token(&BTreeMap::from([
                    ("switch".to_owned(), boolean_value(false)),
                    ("on_false".to_owned(), boolean_value(false)),
                    ("extra".to_owned(), boolean_value(true)),
                ]))
                .is_err()
        );

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let error = futures::executor::block_on(and.execute(
            context(attempt_id, cancelled)?,
            values.clone(),
        ))
        .expect_err("cancelled logic execution unexpectedly succeeded");
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);

        assert_eq!(
            execute("ComfyAndNode", values.clone(), 0x47141)?,
            boolean_value(true)
        );
        assert_eq!(
            execute("ComfyAndNode", values, 0x47142)?,
            boolean_value(true)
        );
        Ok(())
    }
}
