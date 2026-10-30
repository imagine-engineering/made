use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeHandleKind, NativeHandleType, NativeInputDescriptor, NativeNode,
    NativeNodeBinding, NativeNodeBindingsFactory, NativeNodeContext, NativeNodeContractError,
    NativeNodeDescriptor, NativeNodeFailure, NativeNodeFailureKind, NativeNodeOutcome,
    NativeNodePresentation, NativeOpaqueHandle, NativeOutputDescriptor, NativePortCardinality,
    NativePrimitive, NativePrimitiveType, NativeStoredModelPayload, NativeStoredPayload,
    NativeTypeUnion, NativeValue, NativeValueType, built_in_source_schema,
};
use sim_sampler::generated_seeds_2_sim_model_0199::{Seeds2Options, Seeds2SolverType};
use sim_sampler::{
    NativeGuidanceModifier, NativeGuidanceModifierChain, NativeSaSolverParameters,
    NativeSamplerAlgorithm, NativeSamplerParameters, NativeSamplerPayload, SamplingProfile,
};
use sim_tensor::generated_indexing_masking_01::narrow_function_exact_native;
use sim_tensor::generated_native_diffusion::{tensor_from_f32, tensor_to_f32};
use sim_tensor::{NativeTensorPayload, NativeTensorRole};
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "SamplerLMS",
    "SamplerSASolver",
    "SamplerSEEDS2",
    "SamplingPercentToSigma",
    "SDTurboScheduler",
    "SetFirstSigma",
    "SplitSigmas",
    "SplitSigmasDenoise",
    "VideoLinearCFGGuidance",
    "VideoTriangleCFGGuidance",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

#[derive(Clone, Copy, Debug)]
enum SamplingNode {
    Lms,
    Sa,
    Seeds,
    Percent,
    Turbo,
    First,
    Split,
    SplitDenoise,
    Linear,
    Triangle,
}
impl SamplingNode {
    fn inputs(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::Lms => &[("order", "INT")],
            Self::Sa => &[
                ("model", "MODEL"),
                ("eta", "FLOAT"),
                ("sde_start_percent", "FLOAT"),
                ("sde_end_percent", "FLOAT"),
                ("s_noise", "FLOAT"),
                ("predictor_order", "INT"),
                ("corrector_order", "INT"),
                ("use_pece", "BOOLEAN"),
                ("simple_order_2", "BOOLEAN"),
            ],
            Self::Seeds => &[
                ("solver_type", "STRING"),
                ("eta", "FLOAT"),
                ("s_noise", "FLOAT"),
                ("r", "FLOAT"),
            ],
            Self::Percent => &[
                ("model", "MODEL"),
                ("sampling_percent", "FLOAT"),
                ("return_actual_sigma", "BOOLEAN"),
            ],
            Self::Turbo => &[("model", "MODEL"), ("steps", "INT"), ("denoise", "FLOAT")],
            Self::First => &[("sigmas", "SIGMAS"), ("sigma", "FLOAT")],
            Self::Split => &[("sigmas", "SIGMAS"), ("step", "INT")],
            Self::SplitDenoise => &[("sigmas", "SIGMAS"), ("denoise", "FLOAT")],
            Self::Linear | Self::Triangle => &[("model", "MODEL"), ("min_cfg", "FLOAT")],
        }
    }
    fn outputs(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::Lms | Self::Sa | Self::Seeds => &[("sampler", "SAMPLER")],
            Self::Percent => &[("sigma_value", "FLOAT")],
            Self::Turbo | Self::First => &[("sigmas", "SIGMAS")],
            Self::Split | Self::SplitDenoise => {
                &[("high_sigmas", "SIGMAS"), ("low_sigmas", "SIGMAS")]
            }
            Self::Linear | Self::Triangle => &[("model", "MODEL")],
        }
    }
    fn validate_inputs(
        self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<(), NativeNodeFailure> {
        if inputs.len() != self.inputs().len() {
            return Err(invalid("Unexpected sampling input count"));
        }
        for (name, kind) in self.inputs() {
            match *kind {
                "MODEL" | "SIGMAS" => {
                    input_handle(inputs, name, kind)?;
                }
                "BOOLEAN" => {
                    boolean(inputs, name)?;
                }
                "STRING" => {
                    string(inputs, name)?;
                }
                "INT" => {
                    integer(inputs, name, 0, 10000)?;
                }
                _ => {
                    number(inputs, name, 0.0, 20000.0)?;
                }
            }
        }
        match self {
            Self::Lms => {
                integer(inputs, "order", 1, 100)?;
            }
            Self::Sa => {
                self.sa_parameters(inputs)?;
            }
            Self::Seeds => {
                self.seeds_parameters(inputs)?;
            }
            Self::Percent => {
                number(inputs, "sampling_percent", 0.0, 1.0)?;
            }
            Self::Turbo => {
                integer(inputs, "steps", 1, 10)?;
                number(inputs, "denoise", 0.0, 1.0)?;
            }
            Self::First => {
                number(inputs, "sigma", 0.0, 20000.0)?;
            }
            Self::Split => {
                integer(inputs, "step", 0, 10000)?;
            }
            Self::SplitDenoise => {
                number(inputs, "denoise", 0.0, 1.0)?;
            }
            Self::Linear | Self::Triangle => {
                number(inputs, "min_cfg", 0.0, 100.0)?;
            }
        }
        Ok(())
    }
    fn sa_parameters(
        self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeSaSolverParameters, NativeNodeFailure> {
        Ok(NativeSaSolverParameters {
            eta: number(inputs, "eta", 0.0, 10.0)? as f32,
            sde_start_percent: number(inputs, "sde_start_percent", 0.0, 1.0)? as f32,
            sde_end_percent: number(inputs, "sde_end_percent", 0.0, 1.0)? as f32,
            noise_scale: number(inputs, "s_noise", 0.0, 100.0)? as f32,
            predictor_order: integer(inputs, "predictor_order", 1, 6)? as u8,
            corrector_order: integer(inputs, "corrector_order", 0, 6)? as u8,
            use_pece: boolean(inputs, "use_pece")?,
            simple_order_2: boolean(inputs, "simple_order_2")?,
        })
    }
    fn seeds_parameters(
        self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<Seeds2Options, NativeNodeFailure> {
        Ok(Seeds2Options {
            eta: number(inputs, "eta", 0.0, 100.0)? as f32,
            noise_scale: number(inputs, "s_noise", 0.0, 100.0)? as f32,
            intermediate_step_ratio: number(inputs, "r", 0.01, 1.0)? as f32,
            solver_type: match string(inputs, "solver_type")? {
                "phi_1" => Seeds2SolverType::Phi1,
                "phi_2" => Seeds2SolverType::Phi2,
                _ => return Err(invalid("Unknown SEEDS2 solver type")),
            },
        })
    }
}
fn value_type(kind: &str) -> Result<NativeValueType, NativeNodeContractError> {
    Ok(match kind {
        "INT" => NativeValueType::Primitive(NativePrimitiveType::Integer),
        "FLOAT" => NativeValueType::Primitive(NativePrimitiveType::Number),
        "BOOLEAN" => NativeValueType::Primitive(NativePrimitiveType::Boolean),
        "STRING" => NativeValueType::Primitive(NativePrimitiveType::String),
        _ => NativeValueType::Handle(handle_type(kind)?),
    })
}
fn handle_type(kind: &str) -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(
        match kind {
            "MODEL" => NativeHandleKind::Model,
            "SIGMAS" => NativeHandleKind::Tensor,
            _ => NativeHandleKind::StructuredCompute,
        },
        kind,
    )
}
fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    [
        SamplingNode::Lms,
        SamplingNode::Sa,
        SamplingNode::Seeds,
        SamplingNode::Percent,
        SamplingNode::Turbo,
        SamplingNode::First,
        SamplingNode::Split,
        SamplingNode::SplitDenoise,
        SamplingNode::Linear,
        SamplingNode::Triangle,
    ]
    .into_iter()
    .map(|node| {
        let feature_id = match node {
            SamplingNode::Lms => "COMFY-NODE-0582",
            SamplingNode::Sa => "COMFY-NODE-0583",
            SamplingNode::Seeds => "COMFY-NODE-0584",
            SamplingNode::Percent => "COMFY-NODE-0585",
            SamplingNode::Turbo => "COMFY-NODE-0609",
            SamplingNode::First => "COMFY-NODE-0616",
            SamplingNode::Split => "COMFY-NODE-0633",
            SamplingNode::SplitDenoise => "COMFY-NODE-0634",
            SamplingNode::Linear => "COMFY-NODE-0726",
            SamplingNode::Triangle => "COMFY-NODE-0727",
        };
        let category = match node {
            SamplingNode::Lms | SamplingNode::Sa | SamplingNode::Seeds => "model/sampling/samplers",
            SamplingNode::Turbo => "model/sampling/schedulers",
            SamplingNode::Linear | SamplingNode::Triangle => "model/sampling/guiders",
            _ => "model/sampling/sigmas",
        };
        let portable_schema = built_in_source_schema(node.class_type())
            .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
        let description = portable_schema
            .presentation
            .description
            .clone()
            .unwrap_or_default();
        let source_schema = portable_schema
            .bind_execution_ports(
                &node
                    .inputs()
                    .iter()
                    .map(|(name, _)| (*name).to_owned())
                    .collect::<Vec<_>>(),
                &[],
                &node
                    .outputs()
                    .iter()
                    .map(|(name, _)| (*name).to_owned())
                    .collect::<Vec<_>>(),
            )
            .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
        let inputs = node
            .inputs()
            .iter()
            .map(|(name, kind)| {
                Ok(NativeInputDescriptor {
                    name: (*name).to_owned(),
                    accepted_types: NativeTypeUnion::new([value_type(kind)?])?,
                    required: true,
                    hidden: false,
                    lazy: false,
                    cardinality: NativePortCardinality::Scalar,
                    allows_literal: !matches!(*kind, "MODEL" | "SIGMAS"),
                })
            })
            .collect::<Result<Vec<_>, NativeNodeContractError>>()?;
        let outputs = node
            .outputs()
            .iter()
            .map(|(name, kind)| {
                Ok(NativeOutputDescriptor {
                    name: (*name).to_owned(),
                    produced_type: value_type(kind)?,
                    is_list: false,
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
                outputs,
                output_node: false,
                effect: NativeEffectClass::Pure,
                cache: NativeCachePolicy::InputIdentity,
            },
            presentation: NativeNodePresentation {
                display_name: match node {
                    SamplingNode::Linear => "Video Linear CFG Guidance",
                    SamplingNode::Triangle => "Video Triangle CFG Guidance",
                    _ => node.class_type(),
                }
                .to_owned(),
                category: category.to_owned(),
                description,
                output_names: node
                    .outputs()
                    .iter()
                    .map(|(name, _)| (*name).to_owned())
                    .collect(),
                search_aliases: match node {
                    SamplingNode::Sa => vec!["sde".to_owned()],
                    SamplingNode::Seeds => vec!["sde".to_owned(), "exp heun".to_owned()],
                    _ => Vec::new(),
                },
                is_deprecated: false,
                is_experimental: false,
            },
            node: Arc::new(node),
        })
    })
    .collect()
}
impl NativeNode for SamplingNode {
    fn class_type(&self) -> &str {
        match self {
            Self::Lms => "SamplerLMS",
            Self::Sa => "SamplerSASolver",
            Self::Seeds => "SamplerSEEDS2",
            Self::Percent => "SamplingPercentToSigma",
            Self::Turbo => "SDTurboScheduler",
            Self::First => "SetFirstSigma",
            Self::Split => "SplitSigmas",
            Self::SplitDenoise => "SplitSigmasDenoise",
            Self::Linear => "VideoLinearCFGGuidance",
            Self::Triangle => "VideoTriangleCFGGuidance",
        }
    }
    fn implementation_version(&self) -> &str {
        match self {
            Self::Linear | Self::Triangle => "source-video-cfg-v1",
            _ => "source-c3bd7c80-sampling-v1",
        }
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
        for (name, value) in inputs {
            if let NativeValue::Handle { value: handle } = value {
                let payload = context
                    .handle_store()
                    .resolve(handle, handle.handle_type(), &context.cancellation)
                    .map_err(|error| failure(context, error))?;
                artifact_digests.insert(name.clone(), payload.digest_sha256());
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
            let mut payloads = Vec::new();
            match self {
                Self::Lms => payloads.push(NativeStoredPayload::Sampler(Arc::new(
                    NativeSamplerPayload::lms(integer(&inputs, "order", 1, 100)? as u8)
                        .map_err(invalid)?,
                ))),
                Self::Seeds => payloads.push(NativeStoredPayload::Sampler(Arc::new(
                    NativeSamplerPayload::checked(
                        NativeSamplerAlgorithm::Seeds2,
                        NativeSamplerParameters::Seeds2(self.seeds_parameters(&inputs)?),
                    )
                    .map_err(invalid)?,
                ))),
                Self::Sa | Self::Percent | Self::Turbo | Self::Linear | Self::Triangle => {
                    let handle = input_handle(&inputs, "model", "MODEL")?;
                    let stored = context
                        .handle_store()
                        .resolve(handle, handle.handle_type(), &context.cancellation)
                        .map_err(|error| failure(&context, error))?;
                    let NativeStoredPayload::Model(model) = stored.as_ref() else {
                        return Err(invalid("Expected canonical MODEL"));
                    };
                    let diffusion = model
                        .diffusion()
                        .ok_or_else(|| invalid("MODEL has no canonical sampling policy"))?;
                    let profile = diffusion.sampling_policy();
                    match self {
                        Self::Sa => {
                            let parameters = self.sa_parameters(&inputs)?;
                            let start_sigma = profile
                                .source_sigma_for_sampling_percent(
                                    number(&inputs, "sde_start_percent", 0.0, 1.0)?,
                                    false,
                                )
                                .map_err(invalid)?;
                            let end_sigma = profile
                                .source_sigma_for_sampling_percent(
                                    number(&inputs, "sde_end_percent", 0.0, 1.0)?,
                                    false,
                                )
                                .map_err(invalid)?;
                            payloads.push(NativeStoredPayload::Sampler(Arc::new(
                                NativeSamplerPayload::checked(
                                    NativeSamplerAlgorithm::SaSolver,
                                    NativeSamplerParameters::SaSolverInterval {
                                        parameters,
                                        start_sigma: start_sigma as f32,
                                        end_sigma: end_sigma as f32,
                                    },
                                )
                                .map_err(invalid)?,
                            )));
                        }
                        Self::Percent => {
                            let percent = number(&inputs, "sampling_percent", 0.0, 1.0)?;
                            let actual = boolean(&inputs, "return_actual_sigma")?;
                            let value = profile
                                .source_sigma_for_sampling_percent(percent, actual)
                                .map_err(invalid)?;
                            check(&context)?;
                            return Ok(NativeNodeOutcome::Values {
                                outputs: vec![NativeValue::Primitive {
                                    value: NativePrimitive::Number(value),
                                }],
                                ui: None,
                                effects: Vec::new(),
                            });
                        }
                        Self::Turbo => {
                            let steps = integer(&inputs, "steps", 1, 10)?;
                            let denoise = number(&inputs, "denoise", 0.0, 1.0)?;
                            let start = 10 - (10.0 * denoise).trunc() as usize;
                            let end = (start + steps as usize).min(10);
                            let mut values = [0.0; 11];
                            for (offset, index) in (start..end).enumerate() {
                                check(&context)?;
                                let time = (999 - index * 100) as f32;
                                let value = profile.sigma_for_model_time(time).map_err(invalid)?;
                                *values.get_mut(offset).ok_or_else(|| {
                                    invalid("Turbo schedule index exceeded capacity")
                                })? = value;
                            }
                            let count = end - start + 1;
                            let compute = context.compute_session().map_err(invalid)?;
                            let execution = compute.execution_context(&context).map_err(invalid)?;
                            let sigmas = tensor_from_f32(
                                compute.backend(),
                                &[count as u64],
                                values
                                    .get(..count)
                                    .ok_or_else(|| invalid("Turbo schedule exceeds capacity"))?,
                                &execution,
                            )
                            .map_err(|error| failure(&context, error))?;
                            payloads.push(sigma_payload(sigmas)?);
                        }
                        Self::Linear | Self::Triangle => {
                            let minimum_scale = number(&inputs, "min_cfg", 0.0, 100.0)? as f32;
                            let modifier = if matches!(self, Self::Linear) {
                                NativeGuidanceModifier::VideoLinearCfg { minimum_scale }
                            } else {
                                NativeGuidanceModifier::VideoTriangleCfg { minimum_scale }
                            };
                            let mut modifiers = diffusion
                                .guidance_modifiers()
                                .modifiers()
                                .iter()
                                .copied()
                                .filter(|modifier| match modifier {
                                    NativeGuidanceModifier::AdaptiveProjectedGuidance {
                                        ..
                                    }
                                    | NativeGuidanceModifier::LatentOperation { .. } => true,
                                    NativeGuidanceModifier::CfgOverride { .. }
                                    | NativeGuidanceModifier::VideoLinearCfg { .. }
                                    | NativeGuidanceModifier::VideoTriangleCfg { .. } => false,
                                })
                                .collect::<Vec<_>>();
                            modifiers.push(modifier);
                            let modifiers = Arc::new(
                                NativeGuidanceModifierChain::checked(modifiers).map_err(invalid)?,
                            );
                            let changed = Arc::new(
                                diffusion
                                    .as_ref()
                                    .clone()
                                    .with_guidance_modifiers(modifiers)
                                    .map_err(invalid)?,
                            );
                            let changed = match model.compiled_model() {
                                Some(compiled) => {
                                    NativeStoredModelPayload::native_compiled_diffusion(
                                        changed,
                                        compiled.clone(),
                                    )
                                }
                                None => NativeStoredModelPayload::native_diffusion(changed),
                            }
                            .map_err(invalid)?;
                            payloads.push(NativeStoredPayload::Model(Arc::new(changed)));
                        }
                        _ => return Err(invalid("Unexpected model sampling operation")),
                    }
                }
                Self::First | Self::Split | Self::SplitDenoise => {
                    let handle = input_handle(&inputs, "sigmas", "SIGMAS")?;
                    let stored = context
                        .handle_store()
                        .resolve(handle, handle.handle_type(), &context.cancellation)
                        .map_err(|error| failure(&context, error))?;
                    let NativeStoredPayload::Tensor(sigmas) = stored.as_ref() else {
                        return Err(invalid("Expected canonical SIGMAS"));
                    };
                    let tensor = sigmas.tensor();
                    let [count] = tensor.descriptor().shape() else {
                        return Err(invalid("SIGMAS must be a vector"));
                    };
                    if matches!(self, Self::First) {
                        let compute = context.compute_session().map_err(invalid)?;
                        let execution = compute.execution_context(&context).map_err(invalid)?;
                        let mut values = tensor_to_f32(compute.backend(), tensor, &execution)
                            .map_err(|error| failure(&context, error))?;
                        let first = values.first_mut().ok_or_else(|| {
                            invalid("Cannot replace the first sigma of an empty vector")
                        })?;
                        *first = number(&inputs, "sigma", 0.0, 20000.0)? as f32;
                        let changed =
                            tensor_from_f32(compute.backend(), &[*count], &values, &execution)
                                .map_err(|error| failure(&context, error))?;
                        payloads.push(sigma_payload(changed)?);
                    } else {
                        let (high_count, low_start) = if matches!(self, Self::Split) {
                            let step = integer(&inputs, "step", 0, 10000)?;
                            ((step + 1).min(*count), step.min(*count))
                        } else {
                            let steps = count.saturating_sub(1);
                            let total = (steps as f64 * number(&inputs, "denoise", 0.0, 1.0)?)
                                .round_ties_even() as u64;
                            (
                                if total == 0 {
                                    0
                                } else {
                                    count.saturating_sub(total)
                                },
                                count.saturating_sub(total + 1),
                            )
                        };
                        for (start, length) in [(0, high_count), (low_start, count - low_start)] {
                            let view = narrow_function_exact_native(
                                tensor,
                                0,
                                i64::try_from(start).map_err(invalid)?,
                                length,
                                &context.cancellation,
                            )
                            .map_err(|error| failure(&context, error))?;
                            payloads.push(sigma_payload(view)?);
                        }
                    }
                }
            }
            let mut handles = Vec::new();
            let result = (|| {
                for payload in payloads {
                    check(&context)?;
                    handles.push(
                        context
                            .handle_store()
                            .publish(payload, &context.cancellation)
                            .map_err(|error| failure(&context, error))?,
                    );
                }
                check(&context)?;
                let outcome = NativeNodeOutcome::Values {
                    outputs: handles
                        .iter()
                        .cloned()
                        .map(|value| NativeValue::Handle { value })
                        .collect(),
                    ui: None,
                    effects: Vec::new(),
                };
                outcome.validate().map_err(invalid)?;
                Ok(outcome)
            })();
            if result.is_err() {
                for handle in &handles {
                    context
                        .handle_store()
                        .revoke(handle, &sim_types::CancellationToken::default())
                        .map_err(|error| failure(&context, error))?;
                }
            }
            result
        })
    }
}
fn sigma_payload(tensor: sim_tensor::Tensor) -> Result<NativeStoredPayload, NativeNodeFailure> {
    Ok(NativeStoredPayload::Tensor(Arc::new(
        NativeTensorPayload::from_tensor(NativeTensorRole::Sigmas, tensor).map_err(invalid)?,
    )))
}
fn input_handle<'a>(
    inputs: &'a BTreeMap<String, NativeValue>,
    name: &str,
    kind: &str,
) -> Result<&'a NativeOpaqueHandle, NativeNodeFailure> {
    match inputs.get(name) {
        Some(NativeValue::Handle { value })
            if value.handle_type() == &handle_type(kind).map_err(invalid)? =>
        {
            Ok(value)
        }
        _ => Err(invalid(format!("{name} must be a scalar {kind}"))),
    }
}
fn integer(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
    minimum: u64,
    maximum: u64,
) -> Result<u64, NativeNodeFailure> {
    let value = match inputs.get(name) {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => u64::try_from(*value).ok(),
        Some(NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }) => Some(*value),
        _ => None,
    };
    value
        .filter(|value| (minimum..=maximum).contains(value))
        .ok_or_else(|| {
            invalid(format!(
                "{name} must be an integer from {minimum} through {maximum}"
            ))
        })
}
fn number(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
    minimum: f64,
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
        .filter(|value| value.is_finite() && (minimum..=maximum).contains(value))
        .ok_or_else(|| {
            invalid(format!(
                "{name} must be finite from {minimum} through {maximum}"
            ))
        })
}
fn boolean(inputs: &BTreeMap<String, NativeValue>, name: &str) -> Result<bool, NativeNodeFailure> {
    match inputs.get(name) {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Boolean(value),
        }) => Ok(*value),
        _ => Err(invalid(format!("{name} must be a boolean"))),
    }
}
fn string<'a>(
    inputs: &'a BTreeMap<String, NativeValue>,
    name: &str,
) -> Result<&'a str, NativeNodeFailure> {
    match inputs.get(name) {
        Some(NativeValue::Primitive {
            value: NativePrimitive::String(value),
        }) => Ok(value),
        _ => Err(invalid(format!("{name} must be a string"))),
    }
}
fn invalid(error: impl std::fmt::Display) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_source_sampling_node".to_owned(),
        message: error.to_string(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}
fn failure(context: &NativeNodeContext, error: impl std::fmt::Display) -> NativeNodeFailure {
    if context.cancellation.is_cancelled() {
        NativeNodeFailure {
            code: "interrupted".to_owned(),
            message: "Sampling node cancelled".to_owned(),
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
    fn sampling_part_five_rows_preserve_source_contracts() -> Result<(), Box<dyn Error>> {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sim_test_support/fixtures/nodes/model-sampling-comfy-node-0582/fixture.json"
        )))?;
        let rows = fixture["nodes"].as_array().ok_or("rows absent")?;
        let registry = crate::NodeRegistry::built_in()?;
        let bindings = native_node_bindings()?;
        assert_eq!(rows.len(), 10);
        assert_eq!(bindings.len(), 10);
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
            assert_eq!(
                descriptor.outputs.len(),
                row["schema"]["outputs"]
                    .as_array()
                    .ok_or("source outputs absent")?
                    .len()
            );
            assert!(descriptor.outputs.iter().all(|output| !output.is_list));
            assert!(descriptor.inputs.iter().all(|input| input.required
                && !input.hidden
                && !input.lazy
                && input.cardinality == NativePortCardinality::Scalar));
        }
        Ok(())
    }
}
