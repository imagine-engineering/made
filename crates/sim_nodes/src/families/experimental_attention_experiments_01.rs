use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeHandleKind, NativeHandleType, NativeInputDescriptor,
    NativeModelCompileRequest, NativeNode, NativeNodeBinding, NativeNodeBindingsFactory,
    NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure,
    NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation, NativeOpaqueHandle,
    NativeOutputDescriptor, NativePortCardinality, NativePrimitive, NativePrimitiveType,
    NativeStoredModelPayload, NativeStoredPayload, NativeTypeUnion, NativeValue, NativeValueType,
    built_in_source_schema,
};
use sim_model::patches::{StateKeyMatch, StateKeySelector};
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "CLIPAttentionMultiply",
    "UNetCrossAttentionMultiply",
    "UNetSelfAttentionMultiply",
    "UNetTemporalAttentionMultiply",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

#[derive(Clone, Copy, Debug)]
enum AttentionMultiply {
    Clip,
    Cross,
    SelfAttention,
    Temporal,
}

impl AttentionMultiply {
    fn input_name(self) -> &'static str {
        if matches!(self, Self::Clip) {
            "clip"
        } else {
            "model"
        }
    }
    fn handle_type(self) -> Result<NativeHandleType, NativeNodeContractError> {
        if matches!(self, Self::Clip) {
            NativeHandleType::new(NativeHandleKind::Clip, "CLIP")
        } else {
            NativeHandleType::new(NativeHandleKind::Model, "MODEL")
        }
    }
    fn parameters(self) -> [&'static str; 4] {
        if matches!(self, Self::Temporal) {
            [
                "self_structural",
                "self_temporal",
                "cross_structural",
                "cross_temporal",
            ]
        } else {
            ["q", "k", "v", "out"]
        }
    }
    fn inputs(
        self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<(&NativeOpaqueHandle, Vec<(StateKeySelector, f32)>), NativeNodeFailure> {
        if inputs.len() != 5 {
            return Err(invalid(
                "Attention multiply requires exactly five scalar inputs",
            ));
        }
        let Some(NativeValue::Handle { value: handle }) = inputs.get(self.input_name()) else {
            return Err(invalid("Missing model or CLIP handle"));
        };
        if handle.handle_type() != &self.handle_type().map_err(invalid)? {
            return Err(invalid("Model or CLIP handle has the wrong type"));
        }
        let mut rules = Vec::new();
        for (index, parameter) in self.parameters().into_iter().enumerate() {
            let scale = match inputs.get(parameter) {
                Some(NativeValue::Primitive {
                    value: NativePrimitive::Number(value),
                }) => *value,
                Some(NativeValue::Primitive {
                    value: NativePrimitive::Integer(value),
                }) => *value as f64,
                Some(NativeValue::Primitive {
                    value: NativePrimitive::UnsignedInteger(value),
                }) => *value as f64,
                _ => return Err(invalid(format!("{parameter} must be a scalar number"))),
            };
            if !scale.is_finite() || !(0.0..=10.0).contains(&scale) {
                return Err(invalid(format!("{parameter} must be between 0 and 10")));
            }
            let prefix = match self {
                Self::Clip => format!("self_attn.{parameter}_proj"),
                Self::SelfAttention | Self::Cross => {
                    let attention = if matches!(self, Self::SelfAttention) {
                        "attn1"
                    } else {
                        "attn2"
                    };
                    let projection = if parameter == "out" {
                        "out.0"
                    } else {
                        parameter
                    };
                    format!("{attention}.to_{projection}")
                }
                Self::Temporal => format!("{}.to_out.0", if index < 2 { "attn1" } else { "attn2" }),
            };
            let temporal = matches!(self, Self::Temporal) && index % 2 == 1;
            let structural = matches!(self, Self::Temporal) && !temporal;
            let selector = StateKeySelector::checked(
                vec![
                    StateKeyMatch::Suffix(format!("{prefix}.bias")),
                    StateKeyMatch::Suffix(format!("{prefix}.weight")),
                ],
                if structural {
                    vec![".time_stack.".to_owned()]
                } else {
                    Vec::new()
                },
            )
            .map_err(invalid)?
            .with_required_contains(if temporal {
                vec![".time_stack.".to_owned()]
            } else {
                Vec::new()
            })
            .map_err(invalid)?;
            rules.push((selector, scale as f32));
        }
        Ok((handle, rules))
    }
}

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    [
        AttentionMultiply::Clip,
        AttentionMultiply::Cross,
        AttentionMultiply::SelfAttention,
        AttentionMultiply::Temporal,
    ]
    .into_iter()
    .map(|node| {
        let feature_id = match node {
            AttentionMultiply::Clip => "COMFY-NODE-0057",
            AttentionMultiply::Cross => "COMFY-NODE-0704",
            AttentionMultiply::SelfAttention => "COMFY-NODE-0706",
            AttentionMultiply::Temporal => "COMFY-NODE-0707",
        };
        let input_names = std::iter::once(node.input_name())
            .chain(node.parameters())
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let source_schema = built_in_source_schema(node.class_type())
            .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
            .bind_execution_ports(&input_names, &[], &[node.input_name().to_owned()])
            .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
        let inputs = input_names
            .into_iter()
            .enumerate()
            .map(|(index, name)| {
                Ok(NativeInputDescriptor {
                    name,
                    accepted_types: if index == 0 {
                        NativeTypeUnion::new([NativeValueType::Handle(node.handle_type()?)])?
                    } else {
                        NativeTypeUnion::new([NativeValueType::Primitive(
                            NativePrimitiveType::Number,
                        )])?
                    },
                    required: true,
                    hidden: false,
                    lazy: false,
                    cardinality: NativePortCardinality::Scalar,
                    allows_literal: index != 0,
                })
            })
            .collect::<Result<Vec<_>, NativeNodeContractError>>()?;
        Ok(NativeNodeBinding::Executable {
            feature_id: feature_id.to_owned(),
            descriptor: NativeNodeDescriptor {
                schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
                class_type: node.class_type().to_owned(),
                implementation_version: node.implementation_version().to_owned(),
                source_schema: Some(source_schema),
                inputs,
                dynamic_inputs: Vec::new(),
                outputs: vec![NativeOutputDescriptor {
                    name: node.input_name().to_owned(),
                    produced_type: NativeValueType::Handle(node.handle_type()?),
                    is_list: false,
                }],
                output_node: false,
                effect: NativeEffectClass::Pure,
                cache: NativeCachePolicy::InputIdentity,
            },
            presentation: NativeNodePresentation {
                display_name: node.class_type().to_owned(),
                category: "experimental/attention_experiments".to_owned(),
                description: String::new(),
                output_names: vec![node.input_name().to_owned()],
                search_aliases: if matches!(node, AttentionMultiply::Clip) {
                    vec![
                        "clip attention scale".to_owned(),
                        "text encoder attention".to_owned(),
                    ]
                } else {
                    Vec::new()
                },
                is_deprecated: false,
                is_experimental: true,
            },
            node: Arc::new(node),
        })
    })
    .collect()
}

impl NativeNode for AttentionMultiply {
    fn class_type(&self) -> &str {
        match self {
            Self::Clip => "CLIPAttentionMultiply",
            Self::Cross => "UNetCrossAttentionMultiply",
            Self::SelfAttention => "UNetSelfAttentionMultiply",
            Self::Temporal => "UNetTemporalAttentionMultiply",
        }
    }
    fn implementation_version(&self) -> &str {
        "source-40b6aefb-attention-multiply-v1"
    }
    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<BTreeSet<String>, NativeNodeFailure> {
        check_cancellation(context)?;
        self.inputs(inputs)?;
        Ok(BTreeSet::new())
    }
    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        self.inputs(inputs)?;
        Ok("native-attention-scale-matching-v1".to_owned())
    }
    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check_cancellation(context)?;
        let (handle, _) = self.inputs(inputs)?;
        let resolved = context
            .handle_store()
            .resolve(
                handle,
                &self.handle_type().map_err(invalid)?,
                &context.cancellation,
            )
            .map_err(|error| failure(context, error))?;
        let NativeStoredPayload::Model(model) = resolved.as_ref() else {
            return Err(invalid("Handle does not contain a canonical model payload"));
        };
        Ok(NativeCacheDependencies {
            artifact_digests: BTreeMap::from([
                (
                    "attention-model-execution".to_owned(),
                    model
                        .model_payload()
                        .identity()
                        .execution_sha256()
                        .to_owned(),
                ),
                (
                    "attention-stored-model".to_owned(),
                    model.digest_sha256().to_owned(),
                ),
            ]),
            ..Default::default()
        })
    }
    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check_cancellation(&context)?;
            let (handle, rules) = self.inputs(&inputs)?;
            let resolved = context
                .handle_store()
                .resolve(
                    handle,
                    &self.handle_type().map_err(invalid)?,
                    &context.cancellation,
                )
                .map_err(|error| failure(&context, error))?;
            let NativeStoredPayload::Model(model) = resolved.as_ref() else {
                return Err(invalid("Handle does not contain a canonical model payload"));
            };
            let compute = context
                .compute_session()
                .map_err(|error| failure(&context, error))?;
            let execution = compute
                .execution_context(&context)
                .map_err(|error| failure(&context, error))?;
            let transformed = if let Some(diffusion) = model.diffusion() {
                let diffusion = Arc::new(
                    diffusion
                        .scale_matching_state(&rules, compute.backend(), &execution)
                        .map_err(|error| failure(&context, error))?,
                );
                if let Some(compiled) = model.compiled_model() {
                    let request = NativeModelCompileRequest::checked(diffusion, compiled.backend())
                        .map_err(|error| failure(&context, error))?;
                    let output = context
                        .compile_native_model(&request)
                        .map_err(|error| failure(&context, error))?;
                    if output.policy_sha256() != compiled.policy_sha256() {
                        return Err(invalid(
                            "Compiled attention transform changed the compilation policy",
                        ));
                    }
                    NativeStoredModelPayload::native_compiled_diffusion(
                        output.model().clone(),
                        output.compiled_model().clone(),
                    )
                } else {
                    NativeStoredModelPayload::native_diffusion(diffusion)
                }
            } else {
                let model = model
                    .model_payload()
                    .scale_matching_state(&rules, compute.backend(), &execution)
                    .map_err(|error| failure(&context, error))?;
                NativeStoredModelPayload::model_resource(Arc::new(model))
            }
            .map_err(|error| failure(&context, error))?;
            check_cancellation(&context)?;
            let handle = context
                .handle_store()
                .publish(
                    NativeStoredPayload::Model(Arc::new(transformed)),
                    &context.cancellation,
                )
                .map_err(|error| failure(&context, error))?;
            let outcome = NativeNodeOutcome::Values {
                outputs: vec![NativeValue::Handle {
                    value: handle.clone(),
                }],
                ui: None,
                effects: Vec::new(),
            };
            if let Err(error) =
                check_cancellation(&context).and_then(|()| outcome.validate().map_err(invalid))
            {
                context
                    .handle_store()
                    .revoke(&handle, &sim_types::CancellationToken::default())
                    .map_err(|error| failure(&context, error))?;
                return Err(error);
            }
            Ok(outcome)
        })
    }
}

fn invalid(error: impl std::fmt::Display) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_attention_inputs".to_owned(),
        message: error.to_string(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}
fn check_cancellation(context: &NativeNodeContext) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|error| failure(context, error))
}
fn failure(context: &NativeNodeContext, error: impl std::fmt::Display) -> NativeNodeFailure {
    if context.cancellation.is_cancelled() {
        NativeNodeFailure {
            code: "interrupted".to_owned(),
            message: "Attention transform cancelled".to_owned(),
            kind: NativeNodeFailureKind::Interrupted,
            retryable: true,
        }
    } else {
        NativeNodeFailure {
            code: "native_attention_transform_failed".to_owned(),
            message: error.to_string(),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NativeHandleStoreIdentity, NodeRegistry};
    use sim_model::{PatchPayload, patches::scale_matching_operations};
    use std::error::Error;

    fn inputs(node: AttentionMultiply) -> Result<BTreeMap<String, NativeValue>, Box<dyn Error>> {
        let handle = NativeOpaqueHandle::new(
            node.handle_type()?,
            NativeHandleStoreIdentity::new(uuid::Uuid::from_u128(595), uuid::Uuid::from_u128(1))?,
            "model",
            1,
            Some("a".repeat(64)),
        )?;
        let mut inputs = BTreeMap::from([(
            node.input_name().to_owned(),
            NativeValue::Handle { value: handle },
        )]);
        for (index, name) in node.parameters().into_iter().enumerate() {
            inputs.insert(
                name.to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Number(index as f64 + 1.0),
                },
            );
        }
        Ok(inputs)
    }

    #[test]
    fn attention_rows_preserve_exact_source_schemas_and_single_registration()
    -> Result<(), Box<dyn Error>> {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sim_test_support/fixtures/nodes/experimental-attention-experiments-comfy-node-0057/fixture.json"
        )))?;
        let registry = NodeRegistry::built_in()?;
        let bindings = native_node_bindings()?;
        assert_eq!(bindings.len(), 4);
        for (binding, row) in bindings
            .into_iter()
            .zip(fixture["nodes"].as_array().ok_or("fixture rows absent")?)
        {
            registry.validate_native_binding(&binding)?;
            let NativeNodeBinding::Executable {
                feature_id,
                descriptor,
                presentation,
                node,
            } = binding
            else {
                return Err("attention row is not executable".into());
            };
            descriptor.validate_exact_schema_v2()?;
            assert_eq!(feature_id, row["feature_id"]);
            assert_eq!(descriptor.class_type, row["class_type"]);
            assert_eq!(
                serde_json::to_value(built_in_source_schema(&descriptor.class_type)?)?,
                row["schema"]
            );
            assert!(presentation.is_experimental && !presentation.is_deprecated);
            assert!(descriptor.inputs.iter().all(|input| input.required
                && !input.lazy
                && !input.hidden
                && input.cardinality == NativePortCardinality::Scalar));
            assert_eq!(descriptor.cache, NativeCachePolicy::InputIdentity);
            assert_eq!(descriptor.effect, NativeEffectClass::Pure);
            assert!(!descriptor.output_node);
            assert!(descriptor.outputs.iter().all(|output| !output.is_list));
            assert_eq!(node.class_type(), descriptor.class_type);
            let restored: NativeNodeDescriptor =
                serde_json::from_value(serde_json::to_value(&descriptor)?)?;
            assert_eq!(restored, descriptor);
            assert!(registry.descriptor(&descriptor.class_type).is_some());
        }
        Ok(())
    }

    #[test]
    fn attention_selectors_match_source_suffixes_and_temporal_intersection()
    -> Result<(), Box<dyn Error>> {
        let shapes = [
            "x.attn1.to_q.bias",
            "x.attn1.to_k.weight",
            "x.attn1.to_v.weight",
            "x.attn1.to_out.0.bias",
            "x.attn2.to_out.0.weight",
            "x.time_stack.attn1.to_out.0.weight",
            "x.time_stack.attn2.to_out.0.bias",
            "text.self_attn.q_proj.weight",
            "text.self_attn.k_proj.bias",
            "text.self_attn.v_proj.weight",
            "text.self_attn.out_proj.bias",
            "x.attn1.to_q.weight.extra",
        ]
        .into_iter()
        .map(|key| (key.to_owned(), vec![1]))
        .collect::<BTreeMap<_, _>>();
        for node in [
            AttentionMultiply::Clip,
            AttentionMultiply::Cross,
            AttentionMultiply::SelfAttention,
            AttentionMultiply::Temporal,
        ] {
            let inputs = inputs(node)?;
            let (_, rules) = node.inputs(&inputs)?;
            let operations = scale_matching_operations(&rules, &shapes)?;
            let expected: Vec<(&str, f32)> = match node {
                AttentionMultiply::Clip => vec![
                    ("text.self_attn.k_proj.bias", 2.0),
                    ("text.self_attn.out_proj.bias", 4.0),
                    ("text.self_attn.q_proj.weight", 1.0),
                    ("text.self_attn.v_proj.weight", 3.0),
                ],
                AttentionMultiply::Cross => vec![
                    ("x.attn2.to_out.0.weight", 4.0),
                    ("x.time_stack.attn2.to_out.0.bias", 4.0),
                ],
                AttentionMultiply::SelfAttention => vec![
                    ("x.attn1.to_k.weight", 2.0),
                    ("x.attn1.to_out.0.bias", 4.0),
                    ("x.attn1.to_q.bias", 1.0),
                    ("x.attn1.to_v.weight", 3.0),
                    ("x.time_stack.attn1.to_out.0.weight", 4.0),
                ],
                AttentionMultiply::Temporal => vec![
                    ("x.attn1.to_out.0.bias", 1.0),
                    ("x.attn2.to_out.0.weight", 3.0),
                    ("x.time_stack.attn1.to_out.0.weight", 2.0),
                    ("x.time_stack.attn2.to_out.0.bias", 4.0),
                ],
            };
            assert_eq!(
                operations
                    .iter()
                    .map(|operation| (operation.target_key.as_str(), operation.strength_model))
                    .collect::<Vec<_>>(),
                expected
            );
            assert!(operations.iter().all(|operation| operation.strength == 0.0
                && matches!(operation.payload, PatchPayload::ScaleOnly)));
            assert!(scale_matching_operations(&rules, &BTreeMap::new())?.is_empty());
            for bad in [
                NativePrimitive::Number(-0.01),
                NativePrimitive::Number(10.01),
                NativePrimitive::Number(f64::NAN),
                NativePrimitive::String("1".to_owned()),
            ] {
                let mut invalid_inputs = inputs.clone();
                invalid_inputs.insert(
                    node.parameters()[0].to_owned(),
                    NativeValue::Primitive { value: bad },
                );
                assert!(node.inputs(&invalid_inputs).is_err());
            }
            for boundary in [0.0, 10.0] {
                let mut boundary_inputs = inputs.clone();
                boundary_inputs.insert(
                    node.parameters()[0].to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::Number(boundary),
                    },
                );
                node.inputs(&boundary_inputs)?;
            }
            let mut list_inputs = inputs.clone();
            list_inputs.insert(
                node.parameters()[0].to_owned(),
                NativeValue::List { values: Vec::new() },
            );
            assert!(node.inputs(&list_inputs).is_err());
        }
        Ok(())
    }
}
