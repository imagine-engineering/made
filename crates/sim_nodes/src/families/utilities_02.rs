use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeInputDescriptor, NativeNode, NativeNodeBinding,
    NativeNodeBindingsFactory, NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor,
    NativeNodeFailure, NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation,
    NativeOutputDescriptor, NativePortCardinality, NativePrimitive, NativePrimitiveType,
    NativeTypeUnion, NativeValue, NativeValueType, built_in_source_schema,
};
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &["SeedNode"];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

const FEATURE_ID: &str = "COMFY-NODE-0610";
const CLASS_TYPE: &str = "SeedNode";
const IMPLEMENTATION_VERSION: &str = "source-f1566b44-v1";
const CACHE_CHANGE_TOKEN: &str = "seed-node-identity-source-f1566b44-v1";

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    let source_schema = built_in_source_schema(CLASS_TYPE)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
        .bind_execution_ports(&["seed".to_owned()], &[], &["seed".to_owned()])
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let integer = NativeValueType::Primitive(NativePrimitiveType::Integer);
    Ok(vec![NativeNodeBinding::Executable {
        feature_id: FEATURE_ID.to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: CLASS_TYPE.to_owned(),
            implementation_version: IMPLEMENTATION_VERSION.to_owned(),
            source_schema: Some(source_schema),
            inputs: vec![NativeInputDescriptor {
                name: "seed".to_owned(),
                accepted_types: NativeTypeUnion::new([integer.clone()])?,
                required: true,
                hidden: false,
                lazy: false,
                cardinality: NativePortCardinality::Scalar,
                allows_literal: true,
            }],
            dynamic_inputs: Vec::new(),
            outputs: vec![NativeOutputDescriptor {
                name: "seed".to_owned(),
                produced_type: integer,
                is_list: false,
            }],
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        },
        presentation: NativeNodePresentation {
            display_name: "Seed".to_owned(),
            category: "utilities".to_owned(),
            description: String::new(),
            output_names: vec!["seed".to_owned()],
            search_aliases: vec!["seed".to_owned(), "random".to_owned()],
            is_deprecated: false,
            is_experimental: false,
        },
        node: Arc::new(SeedNode),
    }])
}

#[derive(Debug)]
struct SeedNode;

impl NativeNode for SeedNode {
    fn class_type(&self) -> &str {
        CLASS_TYPE
    }

    fn implementation_version(&self) -> &str {
        IMPLEMENTATION_VERSION
    }

    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        available_inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<BTreeSet<String>, NativeNodeFailure> {
        check_cancellation(context)?;
        seed(available_inputs)?;
        Ok(BTreeSet::new())
    }

    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        seed(inputs)?;
        Ok(CACHE_CHANGE_TOKEN.to_owned())
    }

    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check_cancellation(context)?;
        seed(inputs)?;
        Ok(NativeCacheDependencies::default())
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check_cancellation(&context)?;
            let seed = seed(&inputs)?.clone();
            check_cancellation(&context)?;
            let outcome = NativeNodeOutcome::Values {
                outputs: vec![NativeValue::Primitive { value: seed }],
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

fn seed(inputs: &BTreeMap<String, NativeValue>) -> Result<&NativePrimitive, NativeNodeFailure> {
    if inputs.len() != 1 {
        return Err(invalid_inputs("SeedNode requires exactly one seed input"));
    }
    let Some(NativeValue::Primitive { value }) = inputs.get("seed") else {
        return Err(invalid_inputs("seed must be an integer"));
    };
    match value {
        NativePrimitive::Integer(number) if *number >= 0 => Ok(value),
        NativePrimitive::UnsignedInteger(number) if *number <= i64::MAX as u64 => Ok(value),
        NativePrimitive::Integer(_) | NativePrimitive::UnsignedInteger(_) => Err(invalid_inputs(
            "seed must be between 0 and the source sys.maxsize value 9223372036854775807",
        )),
        _ => Err(invalid_inputs("seed must be an integer")),
    }
}

fn check_cancellation(context: &NativeNodeContext) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|_| interrupted_failure())
}

fn invalid_inputs(message: impl Into<String>) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_node_inputs".to_owned(),
        message: message.into(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

fn interrupted_failure() -> NativeNodeFailure {
    NativeNodeFailure {
        code: "execution_interrupted".to_owned(),
        message: "SeedNode execution was interrupted".to_owned(),
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
    use serde_json::{Value, json};
    use std::{
        error::Error,
        sync::atomic::{AtomicUsize, Ordering},
    };
    use uuid::Uuid;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/nodes/utilities-comfy-node-0610/fixture.json"
    ));

    #[derive(Debug)]
    struct InertStore {
        identity: NativeHandleStoreIdentity,
        attempt_id: AttemptId,
        calls: AtomicUsize,
    }

    impl NativeHandleStore for InertStore {
        fn identity(&self) -> NativeHandleStoreIdentity {
            self.identity
        }
        fn attempt_id(&self) -> AttemptId {
            self.attempt_id
        }

        fn resolve(
            &self,
            _handle: &NativeOpaqueHandle,
            _expected_type: &NativeHandleType,
            _cancellation: &CancellationToken,
        ) -> Result<NativeResolvedPayload, NativeHandleStoreError> {
            self.calls.fetch_add(1, Ordering::AcqRel);
            Err(NativeHandleStoreError::Rejected(
                "SeedNode must not resolve handles".to_owned(),
            ))
        }

        fn publish(
            &self,
            _payload: NativeStoredPayload,
            _cancellation: &CancellationToken,
        ) -> Result<NativeOpaqueHandle, NativeHandleStoreError> {
            self.calls.fetch_add(1, Ordering::AcqRel);
            Err(NativeHandleStoreError::Rejected(
                "SeedNode must not publish handles".to_owned(),
            ))
        }

        fn revoke(
            &self,
            _handle: &NativeOpaqueHandle,
            _cancellation: &CancellationToken,
        ) -> Result<(), NativeHandleStoreError> {
            self.calls.fetch_add(1, Ordering::AcqRel);
            Err(NativeHandleStoreError::Rejected(
                "SeedNode must not revoke handles".to_owned(),
            ))
        }
    }

    struct Harness {
        store: Arc<InertStore>,
    }

    impl Harness {
        fn new(generation: u128) -> Result<Self, Box<dyn Error>> {
            let attempt_id = AttemptId(Uuid::from_u128(0x47001));
            Ok(Self {
                store: Arc::new(InertStore {
                    identity: NativeHandleStoreIdentity::new(
                        Uuid::from_u128(0x47002),
                        Uuid::from_u128(generation),
                    )?,
                    attempt_id,
                    calls: AtomicUsize::new(0),
                }),
            })
        }

        fn context(
            &self,
            cancellation: CancellationToken,
        ) -> Result<NativeNodeContext, Box<dyn Error>> {
            let (_backend, authority) = CpuWorkspaceAuthority::create_backend(1)?;
            Ok(NativeNodeContext::new(
                PromptId(Uuid::from_u128(0x47003)),
                self.store.attempt_id,
                NodeId("seed-node-test".to_owned()),
                cancellation,
                authority.authorize_workspace(0)?,
                self.store.clone(),
            )?)
        }
    }

    fn inputs(value: NativePrimitive) -> BTreeMap<String, NativeValue> {
        BTreeMap::from([("seed".to_owned(), NativeValue::Primitive { value })])
    }

    fn executable() -> Result<
        (
            NativeNodeDescriptor,
            NativeNodePresentation,
            Arc<dyn NativeNode>,
        ),
        Box<dyn Error>,
    > {
        let mut bindings = native_node_bindings()?;
        let binding = bindings.pop().ok_or("SeedNode binding is absent")?;
        if !bindings.is_empty() {
            return Err("SeedNode family emitted extra bindings".into());
        }
        match binding {
            NativeNodeBinding::Executable {
                descriptor,
                presentation,
                node,
                ..
            } => Ok((descriptor, presentation, node)),
            _ => Err("SeedNode binding is not executable".into()),
        }
    }

    fn output(outcome: NativeNodeOutcome) -> Result<NativePrimitive, Box<dyn Error>> {
        let NativeNodeOutcome::Values {
            outputs,
            ui,
            effects,
        } = outcome
        else {
            return Err("SeedNode did not return values".into());
        };
        assert!(ui.is_none());
        assert!(effects.is_empty());
        let [NativeValue::Primitive { value }] = outputs.as_slice() else {
            return Err("SeedNode output changed type or cardinality".into());
        };
        Ok(value.clone())
    }

    #[test]
    fn source_fixture_schema_and_registry_are_exact() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        assert_eq!(fixture["feature_id"], FEATURE_ID);
        assert_eq!(
            fixture["source"]["sha256"],
            "f1566b448ca9e4b4a1cbd15767301635905cf36a44a17f586c18b848e23e5932"
        );
        let (descriptor, presentation, node) = executable()?;
        descriptor.validate_exact_schema_v2()?;
        assert_eq!(descriptor.class_type, CLASS_TYPE);
        assert_eq!(descriptor.inputs.len(), 1);
        assert_eq!(
            descriptor.inputs[0].cardinality,
            NativePortCardinality::Scalar
        );
        assert!(descriptor.inputs[0].allows_literal);
        assert_eq!(descriptor.outputs.len(), 1);
        assert!(!descriptor.outputs[0].is_list);
        assert_eq!(descriptor.effect, NativeEffectClass::Pure);
        assert_eq!(descriptor.cache, NativeCachePolicy::InputIdentity);
        let schema = descriptor
            .source_schema
            .as_ref()
            .ok_or("source schema is absent")?;
        assert_eq!(
            schema.inputs[0].minimum,
            Some(crate::NativeSchemaValue::UnsignedInteger { value: 0 })
        );
        assert!(matches!(
            schema.inputs[0].maximum,
            Some(crate::NativeSchemaValue::PreservedExpression { .. })
        ));
        assert!(
            schema.inputs[0]
                .extra
                .iter()
                .any(|entry| entry.name == "control_after_generate")
        );
        assert_eq!(presentation.display_name, "Seed");
        assert_eq!(presentation.category, "utilities");
        assert_eq!(presentation.output_names, ["seed"]);
        assert_eq!(presentation.search_aliases, ["seed", "random"]);
        assert_eq!(node.class_type(), CLASS_TYPE);
        let binding = native_node_bindings()?.remove(0);
        binding.validate()?;
        NodeRegistry::built_in()?.validate_native_binding(&binding)?;
        Ok(())
    }

    #[test]
    fn identity_success_boundaries_cache_and_effects_are_exact() -> Result<(), Box<dyn Error>> {
        let (_, _, node) = executable()?;
        let harness = Harness::new(0x47004)?;
        for value in [
            NativePrimitive::Integer(0),
            NativePrimitive::UnsignedInteger(0),
            NativePrimitive::Integer(i64::MAX),
            NativePrimitive::UnsignedInteger(i64::MAX as u64),
            NativePrimitive::UnsignedInteger(0x0123_4567_89ab_cdef),
        ] {
            let values = inputs(value.clone());
            let context = harness.context(CancellationToken::default())?;
            assert!(node.demanded_lazy_inputs(&context, &values)?.is_empty());
            assert_eq!(node.cache_change_token(&values)?, CACHE_CHANGE_TOKEN);
            assert_eq!(
                node.cache_dependencies(&context, &values)?,
                NativeCacheDependencies::default()
            );
            assert_eq!(
                output(futures::executor::block_on(node.execute(context, values))?)?,
                value
            );
        }
        assert_eq!(harness.store.calls.load(Ordering::Acquire), 0);
        Ok(())
    }

    #[test]
    fn invalid_inputs_and_cancellation_fail_without_side_effects() -> Result<(), Box<dyn Error>> {
        let (_, _, node) = executable()?;
        let harness = Harness::new(0x47005)?;
        for values in [
            inputs(NativePrimitive::Integer(-1)),
            inputs(NativePrimitive::UnsignedInteger(i64::MAX as u64 + 1)),
            inputs(NativePrimitive::String("1".to_owned())),
            BTreeMap::new(),
            BTreeMap::from([
                (
                    "seed".to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::Integer(1),
                    },
                ),
                (
                    "extra".to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::Integer(2),
                    },
                ),
            ]),
        ] {
            let error = futures::executor::block_on(
                node.execute(harness.context(CancellationToken::default())?, values),
            )
            .expect_err("invalid seed must fail");
            assert_eq!(error.code, "invalid_node_inputs");
        }
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let error = futures::executor::block_on(node.execute(
            harness.context(cancellation)?,
            inputs(NativePrimitive::Integer(1)),
        ))
        .expect_err("cancelled SeedNode must fail");
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
        assert_eq!(harness.store.calls.load(Ordering::Acquire), 0);
        Ok(())
    }

    #[test]
    fn persistence_and_worker_restart_recovery_are_lossless() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        let persisted = fixture["persistence"].clone();
        assert_eq!(
            serde_json::from_slice::<Value>(&serde_json::to_vec(&persisted)?)?,
            persisted
        );
        assert_eq!(
            persisted["node"],
            json!({
                "class_type": "SeedNode",
                "inputs": { "seed": 81985529216486895_u64 },
                "unknown_data": { "preserve": true, "source_extension": "nodes_seed" }
            })
        );
        let (_, _, node) = executable()?;
        let before = Harness::new(0x47006)?;
        let after = Harness::new(0x47007)?;
        let value = NativePrimitive::UnsignedInteger(81_985_529_216_486_895);
        let before_output = output(futures::executor::block_on(node.execute(
            before.context(CancellationToken::default())?,
            inputs(value.clone()),
        ))?)?;
        let wire = serde_json::to_vec(&NativeValue::Primitive {
            value: before_output,
        })?;
        let restored: NativeValue = serde_json::from_slice(&wire)?;
        let NativeValue::Primitive { value: restored } = restored else {
            return Err("restored seed changed type".into());
        };
        let after_output = output(futures::executor::block_on(node.execute(
            after.context(CancellationToken::default())?,
            inputs(restored),
        ))?)?;
        assert_eq!(after_output, value);
        assert_eq!(before.store.calls.load(Ordering::Acquire), 0);
        assert_eq!(after.store.calls.load(Ordering::Acquire), 0);
        Ok(())
    }
}
