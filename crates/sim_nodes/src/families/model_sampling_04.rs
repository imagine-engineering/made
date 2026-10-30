use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCachePolicy, NativeEffectClass, NativeHandleKind,
    NativeHandleType, NativeInputDescriptor, NativeNode, NativeNodeBinding,
    NativeNodeBindingsFactory, NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor,
    NativeNodeFailure, NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation,
    NativeOutputDescriptor, NativePortCardinality, NativePrimitive, NativePrimitiveType,
    NativeStoredPayload, NativeTypeUnion, NativeValue, NativeValueType, built_in_source_schema,
};
use sim_sampler::NativeSamplerPayload;
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "SamplerDPMAdaptative",
    "SamplerDPMPP_2M_SDE",
    "SamplerDPMPP_2S_Ancestral",
    "SamplerDPMPP_3M_SDE",
    "SamplerDPMPP_SDE",
    "SamplerER_SDE",
    "SamplerEulerAncestral",
    "SamplerEulerAncestralCFGPP",
    "SamplerLCM",
    "SamplerLCMUpscale",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

#[derive(Clone, Copy, Debug)]
struct SamplingNode {
    feature_id: &'static str,
    class_type: &'static str,
    inputs: &'static [(&'static str, &'static str)],
}
const NODES: &[SamplingNode] = &[
    SamplingNode {
        feature_id: "COMFY-NODE-0571",
        class_type: "SamplerDPMAdaptative",
        inputs: &[
            ("order", "INT"),
            ("rtol", "FLOAT"),
            ("atol", "FLOAT"),
            ("h_init", "FLOAT"),
            ("pcoeff", "FLOAT"),
            ("icoeff", "FLOAT"),
            ("dcoeff", "FLOAT"),
            ("accept_safety", "FLOAT"),
            ("eta", "FLOAT"),
            ("s_noise", "FLOAT"),
        ],
    },
    SamplingNode {
        feature_id: "COMFY-NODE-0572",
        class_type: "SamplerDPMPP_2M_SDE",
        inputs: &[
            ("solver_type", "COMBO"),
            ("eta", "FLOAT"),
            ("s_noise", "FLOAT"),
            ("noise_device", "COMBO"),
        ],
    },
    SamplingNode {
        feature_id: "COMFY-NODE-0573",
        class_type: "SamplerDPMPP_2S_Ancestral",
        inputs: &[("eta", "FLOAT"), ("s_noise", "FLOAT")],
    },
    SamplingNode {
        feature_id: "COMFY-NODE-0574",
        class_type: "SamplerDPMPP_3M_SDE",
        inputs: &[
            ("eta", "FLOAT"),
            ("s_noise", "FLOAT"),
            ("noise_device", "COMBO"),
        ],
    },
    SamplingNode {
        feature_id: "COMFY-NODE-0575",
        class_type: "SamplerDPMPP_SDE",
        inputs: &[
            ("eta", "FLOAT"),
            ("s_noise", "FLOAT"),
            ("r", "FLOAT"),
            ("noise_device", "COMBO"),
        ],
    },
    SamplingNode {
        feature_id: "COMFY-NODE-0576",
        class_type: "SamplerER_SDE",
        inputs: &[
            ("solver_type", "COMBO"),
            ("max_stage", "INT"),
            ("eta", "FLOAT"),
            ("s_noise", "FLOAT"),
        ],
    },
    SamplingNode {
        feature_id: "COMFY-NODE-0577",
        class_type: "SamplerEulerAncestral",
        inputs: &[("eta", "FLOAT"), ("s_noise", "FLOAT")],
    },
    SamplingNode {
        feature_id: "COMFY-NODE-0578",
        class_type: "SamplerEulerAncestralCFGPP",
        inputs: &[("eta", "FLOAT"), ("s_noise", "FLOAT")],
    },
    SamplingNode {
        feature_id: "COMFY-NODE-0580",
        class_type: "SamplerLCM",
        inputs: &[
            ("s_noise", "FLOAT"),
            ("s_noise_end", "FLOAT"),
            ("noise_clip_std", "FLOAT"),
        ],
    },
    SamplingNode {
        feature_id: "COMFY-NODE-0581",
        class_type: "SamplerLCMUpscale",
        inputs: &[
            ("scale_ratio", "FLOAT"),
            ("scale_steps", "INT"),
            ("upscale_method", "COMBO"),
        ],
    },
];
fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    NODES
        .iter()
        .map(|node| {
            let portable_schema = built_in_source_schema(node.class_type)
                .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
            let description = portable_schema
                .presentation
                .description
                .clone()
                .unwrap_or_default();
            let display_name = portable_schema
                .presentation
                .display_name
                .clone()
                .unwrap_or_else(|| node.class_type.to_owned());
            let source_schema = portable_schema
                .bind_execution_ports(
                    &node
                        .inputs
                        .iter()
                        .map(|(name, _)| (*name).to_owned())
                        .collect::<Vec<_>>(),
                    &[],
                    &["sampler".to_owned()],
                )
                .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
            let inputs = node
                .inputs
                .iter()
                .map(|(name, kind)| {
                    let primitive = match *kind {
                        "INT" => NativePrimitiveType::Integer,
                        "FLOAT" => NativePrimitiveType::Number,
                        _ => NativePrimitiveType::String,
                    };
                    Ok(NativeInputDescriptor {
                        name: (*name).to_owned(),
                        accepted_types: NativeTypeUnion::new([NativeValueType::Primitive(
                            primitive,
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
                feature_id: node.feature_id.to_owned(),
                descriptor: NativeNodeDescriptor {
                    schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
                    class_type: node.class_type.to_owned(),
                    implementation_version: node.implementation_version().to_owned(),
                    source_schema: Some(source_schema),
                    inputs,
                    dynamic_inputs: Vec::new(),
                    outputs: vec![NativeOutputDescriptor {
                        name: "sampler".to_owned(),
                        produced_type: NativeValueType::Handle(NativeHandleType::new(
                            NativeHandleKind::StructuredCompute,
                            "SAMPLER",
                        )?),
                        is_list: false,
                    }],
                    output_node: false,
                    effect: NativeEffectClass::Pure,
                    cache: NativeCachePolicy::InputIdentity,
                },
                presentation: NativeNodePresentation {
                    display_name,
                    category: "model/sampling/samplers".to_owned(),
                    description,
                    output_names: vec!["sampler".to_owned()],
                    search_aliases: Vec::new(),
                    is_deprecated: false,
                    is_experimental: false,
                },
                node: Arc::new(*node),
            })
        })
        .collect()
}
impl SamplingNode {
    fn payload(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeSamplerPayload, NativeNodeFailure> {
        if inputs.len() != self.inputs.len() {
            return Err(invalid("Unexpected sampling input count"));
        }
        sampler_payload(self.class_type, inputs)
    }
}
impl NativeNode for SamplingNode {
    fn class_type(&self) -> &str {
        self.class_type
    }
    fn implementation_version(&self) -> &str {
        "source-part-four-sampler-v1"
    }
    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<BTreeSet<String>, NativeNodeFailure> {
        check(context)?;
        self.payload(inputs)?;
        Ok(BTreeSet::new())
    }
    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        self.payload(inputs)?;
        Ok(self.implementation_version().to_owned())
    }
    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check(&context)?;
            let payload = self.payload(&inputs)?;
            check(&context)?;
            let handle = context
                .handle_store()
                .publish(
                    NativeStoredPayload::Sampler(Arc::new(payload)),
                    &context.cancellation,
                )
                .map_err(|error| failure(&context, error))?;
            let result = (|| {
                check(&context)?;
                let outcome = NativeNodeOutcome::Values {
                    outputs: vec![NativeValue::Handle {
                        value: handle.clone(),
                    }],
                    ui: None,
                    effects: Vec::new(),
                };
                outcome.validate().map_err(invalid)?;
                Ok(outcome)
            })();
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
fn sampler_payload(
    name: &str,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeSamplerPayload, NativeNodeFailure> {
    use sim_sampler::NativeSamplerAlgorithm as Algorithm;
    use sim_sampler::NativeSamplerParameters as Parameters;
    let scalar =
        |name: &str, maximum: f64| number(inputs, name, 0.0, maximum).map(|value| value as f32);
    let gpu = || -> Result<bool, NativeNodeFailure> {
        match string(inputs, "noise_device")? {
            "gpu" => Ok(true),
            "cpu" => Ok(false),
            _ => Err(invalid("noise_device must be gpu or cpu")),
        }
    };
    let (algorithm, parameters) = match name {
        "SamplerDPMAdaptative" => (
            Algorithm::DpmAdaptive,
            Parameters::DpmAdaptive(
                sim_sampler::generated_dpm_adaptive_sim_model_0164::DpmAdaptiveOptions {
                    order: integer(inputs, "order", 2, 3)? as u8,
                    relative_tolerance: scalar("rtol", 100.0)?,
                    absolute_tolerance: scalar("atol", 100.0)?,
                    initial_step_size: scalar("h_init", 100.0)?,
                    proportional_coefficient: number(inputs, "pcoeff", 0.0, 100.0)?,
                    integral_coefficient: number(inputs, "icoeff", 0.0, 100.0)?,
                    derivative_coefficient: number(inputs, "dcoeff", 0.0, 100.0)?,
                    acceptance_safety: number(inputs, "accept_safety", 0.0, 100.0)?,
                    eta: scalar("eta", 100.0)?,
                    noise_scale: scalar("s_noise", 100.0)?,
                    ..Default::default()
                },
            ),
        ),
        "SamplerDPMPP_2M_SDE" => {
            use sim_sampler::generated_dpmpp_2m_sde_sim_model_0168::{
                Dpmpp2mSdeOptions, Dpmpp2mSdeSolverType,
            };
            let solver_type = match string(inputs, "solver_type")? {
                "midpoint" => Dpmpp2mSdeSolverType::Midpoint,
                "heun" => Dpmpp2mSdeSolverType::Heun,
                _ => return Err(invalid("solver_type must be midpoint or heun")),
            };
            let eta = scalar("eta", 100.0)?;
            let noise_scale = scalar("s_noise", 100.0)?;
            if gpu()? {
                (
                    Algorithm::Dpmpp2mSdeGpu,
                    Parameters::Dpmpp2mSdeGpu(
                        sim_sampler::generated_dpmpp_2m_sde_gpu_sim_model_0169::Dpmpp2mSdeGpuOptions {
                            eta,
                            noise_scale,
                            solver_type,
                        },
                    ),
                )
            } else {
                (
                    Algorithm::Dpmpp2mSde,
                    Parameters::Dpmpp2mSde(
                        Dpmpp2mSdeOptions::new_with_solver_type(eta, noise_scale, solver_type)
                            .map_err(invalid)?,
                    ),
                )
            }
        }
        "SamplerDPMPP_2S_Ancestral" => (
            Algorithm::Dpmpp2sAncestral,
            Parameters::Dpmpp2sAncestral(
                sim_sampler::generated_dpmpp_2s_ancestral_sim_model_0172::Dpmpp2sAncestralOptions {
                    eta: scalar("eta", 100.0)?,
                    noise_scale: scalar("s_noise", 100.0)?,
                },
            ),
        ),
        "SamplerDPMPP_3M_SDE" => {
            let eta = scalar("eta", 100.0)?;
            let noise_scale = scalar("s_noise", 100.0)?;
            if gpu()? {
                (Algorithm::Dpmpp3mSdeGpu, Parameters::EtaNoise { eta, noise_scale })
            } else {
                (
                    Algorithm::Dpmpp3mSde,
                    Parameters::Dpmpp3mSde(
                        sim_sampler::generated_dpmpp_3m_sde_sim_model_0174::Dpmpp3mSdeOptions::new(
                            eta, noise_scale,
                        ).map_err(invalid)?,
                    ),
                )
            }
        }
        "SamplerDPMPP_SDE" => (
            if gpu()? { Algorithm::DpmppSdeGpu } else { Algorithm::DpmppSde },
            Parameters::DpmppSde(
                sim_sampler::generated_dpmpp_sde_sim_model_0176::DpmppSdeOptions {
                    eta: scalar("eta", 100.0)?,
                    noise_scale: scalar("s_noise", 100.0)?,
                    r: scalar("r", 100.0)?,
                },
            ),
        ),
        "SamplerER_SDE" => {
            use sim_sampler::NativeErSdeSolver;
            let solver = match string(inputs, "solver_type")? {
                "ER-SDE" => NativeErSdeSolver::ErSde,
                "Reverse-time SDE" => NativeErSdeSolver::ReverseTimeSde,
                "ODE" => NativeErSdeSolver::Ode,
                _ => return Err(invalid("Unknown ER-SDE solver type")),
            };
            let mut eta = scalar("eta", 100.0)?;
            let mut noise_scale = scalar("s_noise", 100.0)?;
            if solver == NativeErSdeSolver::Ode
                || (solver == NativeErSdeSolver::ReverseTimeSde && eta == 0.0)
            {
                eta = 0.0;
                noise_scale = 0.0;
            }
            (Algorithm::ErSde, Parameters::ErSde {
                solver, eta, noise_scale, max_stage: integer(inputs, "max_stage", 1, 3)? as u8,
            })
        }
        "SamplerEulerAncestral" => (
            Algorithm::EulerAncestral,
            Parameters::EulerAncestral(
                sim_sampler::generated_euler_ancestral_sim_model_0180::EulerAncestralOptions::new(
                    scalar("eta", 100.0)?, scalar("s_noise", 100.0)?,
                ).map_err(invalid)?,
            ),
        ),
        "SamplerEulerAncestralCFGPP" => (
            Algorithm::EulerAncestralCfgPp,
            Parameters::EulerAncestralCfgPp(
                sim_sampler::generated_euler_ancestral_cfg_pp_sim_model_0181::EulerAncestralCfgPpOptions {
                    eta: scalar("eta", 1.0)?, noise_scale: scalar("s_noise", 10.0)?,
                },
            ),
        ),
        "SamplerLCM" => (
            Algorithm::Lcm,
            Parameters::Lcm(sim_sampler::generated_lcm_sim_model_0191::LcmOptions {
                noise_scale_start: scalar("s_noise", 64.0)?,
                noise_scale_end: Some(scalar("s_noise_end", 64.0)?),
                noise_clip_standard_deviations: scalar("noise_clip_std", 10.0)?,
            }),
        ),
        "SamplerLCMUpscale" => {
            use sim_sampler::generated_lcm_upscale::{LcmUpscaleMethod, LcmUpscaleOptions};
            let scale_steps = match inputs.get("scale_steps") {
                Some(NativeValue::Primitive { value: NativePrimitive::Integer(-1) }) => None,
                _ => Some(integer(inputs, "scale_steps", 0, 1000)? as u16),
            };
            let method = match string(inputs, "upscale_method")? {
                "bislerp" => LcmUpscaleMethod::Bislerp,
                "nearest-exact" => LcmUpscaleMethod::NearestExact,
                "bilinear" => LcmUpscaleMethod::Bilinear,
                "area" => LcmUpscaleMethod::Area,
                "bicubic" => LcmUpscaleMethod::Bicubic,
                _ => return Err(invalid("Unknown LCM upscale method")),
            };
            (
                Algorithm::LcmUpscale,
                Parameters::LcmUpscale(LcmUpscaleOptions::new(
                    number(inputs, "scale_ratio", 0.1, 20.0)? as f32, scale_steps, method,
                ).map_err(invalid)?),
            )
        }
        _ => return Err(invalid("Unknown part-four sampler")),
    };
    NativeSamplerPayload::checked(algorithm, parameters).map_err(invalid)
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
    fn sampling_part_four_rows_preserve_source_contracts() -> Result<(), Box<dyn Error>> {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sim_test_support/fixtures/nodes/model-sampling-comfy-node-0571/fixture.json"
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
