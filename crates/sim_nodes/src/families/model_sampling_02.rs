use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeHandleKind, NativeHandleType, NativeInputDescriptor,
    NativeInputRequirement, NativeNode, NativeNodeBinding, NativeNodeBindingsFactory,
    NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure,
    NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation, NativeOutputDescriptor,
    NativePortCardinality, NativePrimitive, NativeStoredPayload, NativeValue, NativeValueType,
    built_in_source_schema, native_value_types_for_input_schema,
};
use sim_sampler::SigmaInterpolationSpacing;
use sim_tensor::{NativeTensorPayload, NativeTensorRole};
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "ExponentialScheduler",
    "ExtendIntermediateSigmas",
    "FlipSigmas",
    "Flux2Scheduler",
    "GITSScheduler",
    "Ideogram4Scheduler",
    "KarrasScheduler",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;
const FEATURES: [&str; 7] = [
    "COMFY-NODE-0169",
    "COMFY-NODE-0170",
    "COMFY-NODE-0173",
    "COMFY-NODE-0177",
    "COMFY-NODE-0208",
    "COMFY-NODE-0237",
    "COMFY-NODE-0278",
];
#[derive(Clone, Copy)]
struct ScheduleNode {
    class_type: &'static str,
}

fn sigmas_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Tensor, "SIGMAS")
}
fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    NODE_DESCRIPTOR_IDS
        .iter()
        .zip(FEATURES)
        .map(|(class_type, feature_id)| {
            let catalog = built_in_source_schema(class_type)
                .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
            let names = catalog
                .inputs
                .iter()
                .map(|input| input.schema.name.clone())
                .collect::<Vec<_>>();
            let source_schema = catalog
                .bind_execution_ports(&names, &[], &["sigmas".to_owned()])
                .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
            let inputs = catalog
                .inputs
                .iter()
                .map(|input| {
                    Ok(NativeInputDescriptor {
                        name: input.schema.name.clone(),
                        accepted_types: native_value_types_for_input_schema(&input.schema)
                            .map_err(|error| {
                                NativeNodeContractError::InvalidSourceSchema(error.to_string())
                            })?,
                        required: input.requirement == NativeInputRequirement::Required,
                        hidden: false,
                        lazy: false,
                        cardinality: NativePortCardinality::Mapped,
                        allows_literal: input.schema.name != "sigmas",
                    })
                })
                .collect::<Result<Vec<_>, NativeNodeContractError>>()?;
            let node = ScheduleNode { class_type };
            Ok(NativeNodeBinding::Executable {
                feature_id: feature_id.to_owned(),
                descriptor: NativeNodeDescriptor {
                    schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
                    class_type: (*class_type).to_owned(),
                    implementation_version: node.implementation_version().to_owned(),
                    source_schema: Some(source_schema),
                    inputs,
                    dynamic_inputs: Vec::new(),
                    outputs: vec![NativeOutputDescriptor {
                        name: "sigmas".to_owned(),
                        produced_type: NativeValueType::Handle(sigmas_type()?),
                        is_list: false,
                    }],
                    output_node: false,
                    effect: NativeEffectClass::Pure,
                    cache: NativeCachePolicy::InputIdentity,
                },
                presentation: NativeNodePresentation {
                    display_name: catalog
                        .presentation
                        .display_name
                        .clone()
                        .unwrap_or_else(|| (*class_type).to_owned()),
                    category: if matches!(*class_type, "FlipSigmas" | "ExtendIntermediateSigmas") {
                        "model/sampling/sigmas"
                    } else {
                        "model/sampling/schedulers"
                    }
                    .to_owned(),
                    description: catalog.presentation.description.clone().unwrap_or_default(),
                    output_names: vec!["sigmas".to_owned()],
                    search_aliases: if *class_type == "ExtendIntermediateSigmas" {
                        vec!["interpolate sigmas".to_owned()]
                    } else {
                        Vec::new()
                    },
                    is_deprecated: catalog.presentation.is_deprecated,
                    is_experimental: catalog.presentation.is_experimental,
                },
                node: Arc::new(node),
            })
        })
        .collect()
}
impl ScheduleNode {
    fn validate_inputs(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<(), NativeNodeFailure> {
        let catalog = built_in_source_schema(self.class_type).map_err(invalid)?;
        if inputs.len() != catalog.inputs.len()
            || inputs.keys().any(|name| {
                !catalog
                    .inputs
                    .iter()
                    .any(|input| input.schema.name == *name)
            })
        {
            return Err(invalid("Missing or unknown schedule input"));
        }
        for (name, value) in inputs {
            value.validate().map_err(invalid)?;
            if let NativeValue::List { values } = value {
                if values.is_empty() {
                    return Err(invalid("Schedule input lists must not be empty"));
                }
                for value in values {
                    self.validate_value(name, value)?;
                }
            } else {
                self.validate_value(name, value)?;
            }
        }
        Ok(())
    }
    fn validate_value(&self, name: &str, value: &NativeValue) -> Result<(), NativeNodeFailure> {
        match name {
            "sigmas" => match value {
                NativeValue::Handle { value }
                    if value.handle_type() == &sigmas_type().map_err(invalid)? => {}
                _ => return Err(invalid("SIGMAS handle required")),
            },
            "spacing" => {
                spacing(value)?;
            }
            "steps" => {
                let (minimum, maximum) = match self.class_type {
                    "ExtendIntermediateSigmas" => (1, 100),
                    "Flux2Scheduler" => (1, 4096),
                    "GITSScheduler" => (2, 1000),
                    "Ideogram4Scheduler" => (1, 200),
                    _ => (1, 10000),
                };
                let value = integer_value(value)?;
                if !(minimum..=maximum).contains(&value) {
                    return Err(invalid("Schedule steps exceed source bounds"));
                }
            }
            "width" | "height" => {
                let (minimum, maximum) = if self.class_type == "Flux2Scheduler" {
                    (16, 16384)
                } else {
                    (256, 8192)
                };
                if !(minimum..=maximum).contains(&integer_value(value)?) {
                    return Err(invalid("Schedule dimensions exceed source bounds"));
                }
            }
            _ => {
                let (minimum, maximum) = match name {
                    "sigma_min" | "sigma_max" => (0.0, 5000.0),
                    "rho" => (0.0, 100.0),
                    "start_at_sigma" => (-1.0, 20000.0),
                    "end_at_sigma" => (0.0, 20000.0),
                    "coeff" => (0.8, 1.5),
                    "denoise" => (0.0, 1.0),
                    "mu" => (-10.0, 10.0),
                    "std" => (0.1, 5.0),
                    _ => return Err(invalid("Unknown schedule field")),
                };
                let value = number_value(value)?;
                if !value.is_finite() || !(minimum..=maximum).contains(&value) {
                    return Err(invalid("Schedule parameter exceeds source bounds"));
                }
            }
        }
        Ok(())
    }
}
impl NativeNode for ScheduleNode {
    fn class_type(&self) -> &str {
        self.class_type
    }
    fn implementation_version(&self) -> &str {
        "source-schedule-part-two-v1"
    }
    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<BTreeSet<String>, NativeNodeFailure> {
        check(context)?;
        self.validate_inputs(inputs)?;
        Ok(BTreeSet::new())
    }
    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        self.validate_inputs(inputs)?;
        Ok(self.implementation_version().to_owned())
    }
    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check(context)?;
        self.validate_inputs(inputs)?;
        let mut artifact_digests = BTreeMap::new();
        if let Some(value) = inputs.get("sigmas") {
            let values = match value {
                NativeValue::List { values } => values.as_slice(),
                value => std::slice::from_ref(value),
            };
            for (index, value) in values.iter().enumerate() {
                let NativeValue::Handle { value } = value else {
                    return Err(invalid("SIGMAS handle required"));
                };
                let payload = context
                    .handle_store()
                    .resolve(
                        value,
                        &sigmas_type().map_err(invalid)?,
                        &context.cancellation,
                    )
                    .map_err(|error| failure(context, error))?;
                artifact_digests.insert(format!("sigmas.{index}"), payload.digest_sha256());
            }
        }
        Ok(NativeCacheDependencies {
            artifact_digests,
            ..Default::default()
        })
    }
    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check(&context)?;
            self.validate_inputs(&inputs)?;
            let source = if let Some(NativeValue::Handle { value }) = inputs.get("sigmas") {
                let payload = context
                    .handle_store()
                    .resolve(
                        value,
                        &sigmas_type().map_err(invalid)?,
                        &context.cancellation,
                    )
                    .map_err(|error| failure(&context, error))?;
                let NativeStoredPayload::Tensor(tensor) = payload.as_ref() else {
                    return Err(invalid("SIGMAS payload required"));
                };
                if self.class_type == "FlipSigmas"
                    && tensor
                        .tensor()
                        .descriptor()
                        .element_count()
                        .map_err(invalid)?
                        == 0
                {
                    check(&context)?;
                    return outcome(value.clone());
                }
                Some(payload)
            } else {
                None
            };
            let compute = context.compute_session().map_err(invalid)?;
            let execution = compute.execution_context(&context).map_err(invalid)?;
            let backend = compute.backend();
            let scalar = |name: &str| {
                inputs
                    .get(name)
                    .ok_or_else(|| invalid(format!("Missing {name}")))
            };
            let integer = |name: &str| integer_value(scalar(name)?);
            let number = |name: &str| number_value(scalar(name)?);
            let tensor = match self.class_type {
                "KarrasScheduler" => {
                    sim_sampler::generated_karras_sim_model_0206::karras_sigmas(
                        backend,
                        integer("steps")?,
                        number("sigma_min")?,
                        number("sigma_max")?,
                        number("rho")?,
                        &execution,
                    )
                }
                "ExponentialScheduler" => {
                    sim_sampler::generated_exponential_sim_model_0205::exponential_sigmas(
                        backend,
                        integer("steps")?,
                        number("sigma_min")?,
                        number("sigma_max")?,
                        &execution,
                    )
                }
                "Flux2Scheduler" => sim_sampler::flux2_sigmas(
                    backend,
                    integer("steps")?,
                    integer("width")?,
                    integer("height")?,
                    &execution,
                ),
                "Ideogram4Scheduler" => sim_sampler::ideogram4_sigmas(
                    backend,
                    integer("steps")?,
                    integer("width")?,
                    integer("height")?,
                    number("mu")?,
                    number("std")?,
                    &execution,
                ),
                "GITSScheduler" => sim_sampler::gits_sigmas(
                    backend,
                    number("coeff")?,
                    integer("steps")?,
                    number("denoise")?,
                    &execution,
                ),
                _ => {
                    let stored = source.ok_or_else(|| invalid("SIGMAS payload required"))?;
                    let NativeStoredPayload::Tensor(source) = stored.as_ref() else {
                        return Err(invalid("SIGMAS payload required"));
                    };
                    let sigmas = sim_tensor::generated_native_diffusion::tensor_to_f32(
                        backend,
                        source.tensor(),
                        &execution,
                    )
                    .map_err(|error| failure(&context, error))?;
                    if self.class_type == "FlipSigmas" {
                        sim_sampler::flip_sigmas(backend, &sigmas, &execution)
                    } else {
                        sim_sampler::extend_intermediate_sigmas(
                            backend,
                            &sigmas,
                            integer("steps")?,
                            number("start_at_sigma")?,
                            number("end_at_sigma")?,
                            spacing(scalar("spacing")?)?,
                            &execution,
                        )
                    }
                }
            }
            .map_err(|error| failure(&context, error))?;
            check(&context)?;
            let payload = NativeTensorPayload::from_tensor(NativeTensorRole::Sigmas, tensor)
                .map_err(|error| failure(&context, error))?;
            let handle = context
                .handle_store()
                .publish(
                    NativeStoredPayload::Tensor(Arc::new(payload)),
                    &context.cancellation,
                )
                .map_err(|error| failure(&context, error))?;
            let result = check(&context).and_then(|()| outcome(handle.clone()));
            if result.is_err() {
                context
                    .handle_store()
                    .revoke(&handle, &sim_types::CancellationToken::default())
                    .map_err(|error| failure(&context, error))?;
            }
            result
        })
    }
}
fn outcome(value: crate::NativeOpaqueHandle) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let outcome = NativeNodeOutcome::Values {
        outputs: vec![NativeValue::Handle { value }],
        ui: None,
        effects: Vec::new(),
    };
    outcome.validate().map_err(invalid)?;
    Ok(outcome)
}
fn integer_value(value: &NativeValue) -> Result<u64, NativeNodeFailure> {
    match value {
        NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        } => u64::try_from(*value).map_err(invalid),
        NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        } => Ok(*value),
        _ => Err(invalid("Integer schedule parameter required")),
    }
}
fn number_value(value: &NativeValue) -> Result<f64, NativeNodeFailure> {
    match value {
        NativeValue::Primitive {
            value: NativePrimitive::Number(value),
        } => Ok(*value),
        NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        } => Ok(*value as f64),
        NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        } => Ok(*value as f64),
        _ => Err(invalid("Numeric schedule parameter required")),
    }
}
fn spacing(value: &NativeValue) -> Result<SigmaInterpolationSpacing, NativeNodeFailure> {
    match value {
        NativeValue::Primitive {
            value: NativePrimitive::String(value),
        } => match value.as_str() {
            "linear" => Ok(SigmaInterpolationSpacing::Linear),
            "cosine" => Ok(SigmaInterpolationSpacing::Cosine),
            "sine" => Ok(SigmaInterpolationSpacing::Sine),
            _ => Err(invalid("Unknown sigma spacing")),
        },
        _ => Err(invalid("Sigma spacing name required")),
    }
}
fn invalid(error: impl std::fmt::Display) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_schedule_inputs".to_owned(),
        message: error.to_string(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}
fn failure(context: &NativeNodeContext, error: impl std::fmt::Display) -> NativeNodeFailure {
    NativeNodeFailure {
        code: if context.cancellation.is_cancelled() {
            "interrupted"
        } else {
            "schedule_failed"
        }
        .to_owned(),
        message: error.to_string(),
        kind: if context.cancellation.is_cancelled() {
            NativeNodeFailureKind::Interrupted
        } else {
            NativeNodeFailureKind::Failure
        },
        retryable: context.cancellation.is_cancelled(),
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
    fn parameterized_schedule_rows_preserve_exact_source_contracts() -> Result<(), Box<dyn Error>> {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sim_test_support/fixtures/nodes/model-sampling-comfy-node-0139/fixture.json"
        )))?;
        let rows = fixture
            .get("nodes")
            .and_then(serde_json::Value::as_array)
            .ok_or("source rows absent")?;
        assert_eq!(rows.len(), 10);
        let bindings = native_node_bindings()?;
        crate::validate_generated_family_bindings(&bindings, NODE_DESCRIPTOR_IDS)?;
        let registry = crate::NodeRegistry::built_in()?;
        assert_eq!(bindings.len(), 7);
        for binding in bindings {
            registry.validate_native_binding(&binding)?;
            let descriptor = binding.descriptor();
            descriptor.validate_exact_schema_v2()?;
            let row = rows
                .iter()
                .find(|row| {
                    row.get("class_type").and_then(serde_json::Value::as_str)
                        == Some(descriptor.class_type.as_str())
                })
                .ok_or("assigned source row absent")?;
            assert_eq!(
                row.get("feature_id").and_then(serde_json::Value::as_str),
                Some(binding.feature_id())
            );
            assert_eq!(
                serde_json::to_value(built_in_source_schema(&descriptor.class_type)?)?,
                *row.get("schema").ok_or("source schema absent")?
            );
            assert_eq!(descriptor.cache, NativeCachePolicy::InputIdentity);
            assert_eq!(descriptor.effect, NativeEffectClass::Pure);
            assert!(!descriptor.output_node);
            assert!(descriptor.inputs.iter().all(|input| input.required
                && !input.hidden
                && !input.lazy
                && input.cardinality == NativePortCardinality::Mapped));
            assert_eq!(descriptor.outputs.len(), 1);
            assert!(descriptor.outputs.iter().all(|output| !output.is_list));
            assert_eq!(
                *descriptor,
                serde_json::from_str::<NativeNodeDescriptor>(&serde_json::to_string(descriptor)?)?
            );
        }
        Ok(())
    }
}
