use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeHandleKind, NativeHandleType, NativeInputDescriptor, NativeNode,
    NativeNodeBinding, NativeNodeBindingsFactory, NativeNodeContext, NativeNodeContractError,
    NativeNodeDescriptor, NativeNodeFailure, NativeNodeFailureKind, NativeNodeOutcome,
    NativeNodePresentation, NativeOutputDescriptor, NativePortCardinality, NativePrimitive,
    NativePrimitiveType, NativeStoredPayload, NativeTypeUnion, NativeValue, NativeValueType,
    built_in_source_schema,
};
use sim_sampler::{NativeSamplerPayload, vp_sigmas};
use sim_tensor::{NativeTensorPayload, NativeTensorRole};
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &["VOIDSampler", "VPScheduler"];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

#[derive(Clone, Copy, Debug)]
enum SamplingNode {
    Void,
    Vp,
}

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    [SamplingNode::Void, SamplingNode::Vp]
        .into_iter()
        .map(|node| {
            let (feature_id, names, output_name, output_type, output_kind, category) = match node {
                SamplingNode::Void => (
                    "COMFY-NODE-0743",
                    Vec::new(),
                    "sampler",
                    "SAMPLER",
                    NativeHandleKind::StructuredCompute,
                    "model/sampling/samplers",
                ),
                SamplingNode::Vp => (
                    "COMFY-NODE-0748",
                    vec!["steps", "beta_d", "beta_min", "eps_s"],
                    "sigmas",
                    "SIGMAS",
                    NativeHandleKind::Tensor,
                    "model/sampling/schedulers",
                ),
            };
            let input_names = names
                .iter()
                .map(|name| (*name).to_owned())
                .collect::<Vec<_>>();
            let schema = built_in_source_schema(node.class_type())
                .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
                .bind_execution_ports(&input_names, &[], &[output_name.to_owned()])
                .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
            let inputs = names
                .into_iter()
                .map(|name| {
                    Ok(NativeInputDescriptor {
                        name: name.to_owned(),
                        accepted_types: NativeTypeUnion::new([NativeValueType::Primitive(
                            if name == "steps" {
                                NativePrimitiveType::Integer
                            } else {
                                NativePrimitiveType::Number
                            },
                        )])?,
                        required: true,
                        hidden: false,
                        lazy: false,
                        cardinality: NativePortCardinality::Scalar,
                        allows_literal: true,
                    })
                })
                .collect::<Result<Vec<_>, NativeNodeContractError>>()?;
            Ok(NativeNodeBinding::Executable {
                feature_id: feature_id.to_owned(),
                descriptor: NativeNodeDescriptor {
                    schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
                    class_type: node.class_type().to_owned(),
                    implementation_version: node.implementation_version().to_owned(),
                    source_schema: Some(schema),
                    inputs,
                    dynamic_inputs: Vec::new(),
                    outputs: vec![NativeOutputDescriptor {
                        name: output_name.to_owned(),
                        produced_type: NativeValueType::Handle(NativeHandleType::new(
                            output_kind,
                            output_type,
                        )?),
                        is_list: false,
                    }],
                    output_node: false,
                    effect: NativeEffectClass::Pure,
                    cache: NativeCachePolicy::InputIdentity,
                },
                presentation: NativeNodePresentation {
                    display_name: node.class_type().to_owned(),
                    category: category.to_owned(),
                    description: String::new(),
                    output_names: vec![output_name.to_owned()],
                    search_aliases: Vec::new(),
                    is_deprecated: false,
                    is_experimental: false,
                },
                node: Arc::new(node),
            })
        })
        .collect()
}

impl SamplingNode {
    fn parameters(
        self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<Option<(u64, f64, f64, f64)>, NativeNodeFailure> {
        if matches!(self, Self::Void) {
            if !inputs.is_empty() {
                return Err(invalid("VOIDSampler accepts no inputs"));
            }
            return Ok(None);
        }
        if inputs.len() != 4 {
            return Err(invalid("VPScheduler requires four scalar inputs"));
        }
        let steps = match inputs.get("steps") {
            Some(NativeValue::Primitive {
                value: NativePrimitive::Integer(value),
            }) => u64::try_from(*value).ok(),
            Some(NativeValue::Primitive {
                value: NativePrimitive::UnsignedInteger(value),
            }) => Some(*value),
            _ => None,
        }
        .filter(|value| (1..=10000).contains(value))
        .ok_or_else(|| invalid("steps must be an integer from 1 through 10000"))?;
        Ok(Some((
            steps,
            number(inputs, "beta_d", 5000.0)?,
            number(inputs, "beta_min", 5000.0)?,
            number(inputs, "eps_s", 1.0)?,
        )))
    }
}

impl NativeNode for SamplingNode {
    fn class_type(&self) -> &str {
        match self {
            Self::Void => "VOIDSampler",
            Self::Vp => "VPScheduler",
        }
    }
    fn implementation_version(&self) -> &str {
        match self {
            Self::Void => "source-b80cd0b5-void-v1",
            Self::Vp => "source-cc5f944e-vp-v1",
        }
    }
    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<BTreeSet<String>, NativeNodeFailure> {
        check(context)?;
        self.parameters(inputs)?;
        Ok(BTreeSet::new())
    }
    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        self.parameters(inputs)?;
        Ok(self.implementation_version().to_owned())
    }
    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check(context)?;
        self.parameters(inputs)?;
        Ok(NativeCacheDependencies::default())
    }
    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check(&context)?;
            let payload = match self.parameters(&inputs)? {
                None => NativeStoredPayload::Sampler(Arc::new(
                    NativeSamplerPayload::void_ddim().map_err(invalid)?,
                )),
                Some((steps, beta_d, beta_min, eps_s)) => {
                    let compute = context.compute_session().map_err(invalid)?;
                    let execution = compute.execution_context(&context).map_err(invalid)?;
                    let sigmas = vp_sigmas(
                        compute.backend(),
                        steps,
                        beta_d,
                        beta_min,
                        eps_s,
                        &execution,
                    )
                    .map_err(|error| failure(&context, error))?;
                    NativeStoredPayload::Tensor(Arc::new(
                        NativeTensorPayload::from_tensor(NativeTensorRole::Sigmas, sigmas)
                            .map_err(|error| failure(&context, error))?,
                    ))
                }
            };
            check(&context)?;
            let handle = context
                .handle_store()
                .publish(payload, &context.cancellation)
                .map_err(|error| failure(&context, error))?;
            let outcome = NativeNodeOutcome::Values {
                outputs: vec![NativeValue::Handle {
                    value: handle.clone(),
                }],
                ui: None,
                effects: Vec::new(),
            };
            let result = check(&context).and_then(|()| outcome.validate().map_err(invalid));
            if let Err(error) = result {
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

fn number(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
    maximum: f64,
) -> Result<f64, NativeNodeFailure> {
    let value = match inputs.get(name) {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Number(value),
        }) => Some(*value),
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => Some(*value as f64),
        Some(NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }) => Some(*value as f64),
        _ => None,
    };
    value
        .filter(|value| value.is_finite() && (0.0..=maximum).contains(value))
        .ok_or_else(|| {
            invalid(format!(
                "{name} must be a finite number from 0 through {maximum}"
            ))
        })
}
fn invalid(error: impl std::fmt::Display) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_source_sampler_node".to_owned(),
        message: error.to_string(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}
fn failure(context: &NativeNodeContext, error: impl std::fmt::Display) -> NativeNodeFailure {
    if context.cancellation.is_cancelled() {
        NativeNodeFailure {
            code: "interrupted".to_owned(),
            message: "Sampler node cancelled".to_owned(),
            kind: NativeNodeFailureKind::Interrupted,
            retryable: true,
        }
    } else {
        invalid(error)
    }
}
fn check(context: &NativeNodeContext) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|error| failure(context, error))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn void_and_vp_rows_preserve_source_contracts() -> Result<(), Box<dyn Error>> {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sim_test_support/fixtures/nodes/model-sampling-comfy-node-0743/fixture.json"
        )))?;
        let rows = fixture["nodes"].as_array().ok_or("rows absent")?;
        let registry = crate::NodeRegistry::built_in()?;
        let bindings = native_node_bindings()?;
        assert_eq!(rows.len(), 2);
        assert_eq!(bindings.len(), 2);
        for (binding, row) in bindings.iter().zip(rows) {
            registry.validate_native_binding(binding)?;
            let NativeNodeBinding::Executable {
                feature_id,
                descriptor,
                node,
                ..
            } = binding
            else {
                return Err("native implementation absent".into());
            };
            descriptor.validate_exact_schema_v2()?;
            assert_eq!(Some(feature_id.as_str()), row["feature_id"].as_str());
            assert_eq!(
                serde_json::to_value(built_in_source_schema(node.class_type())?)?,
                row["schema"]
            );
            assert_eq!(
                *descriptor,
                serde_json::from_str::<NativeNodeDescriptor>(&serde_json::to_string(descriptor)?)?
            );
            assert_eq!(descriptor.cache, NativeCachePolicy::InputIdentity);
            assert_eq!(descriptor.effect, NativeEffectClass::Pure);
            assert!(!descriptor.output_node);
            assert_eq!(descriptor.outputs.len(), 1);
            assert!(!descriptor.outputs[0].is_list);
            assert!(descriptor.inputs.iter().all(|input| input.required
                && !input.hidden
                && !input.lazy
                && input.cardinality == NativePortCardinality::Scalar));
        }
        Ok(())
    }
}
