use serde_json::Value;
use sim_nodes::{
    NativeHandleKind, NativeHandleStore, NativeHandleType, NativeMultimodalTextGenerationRequest,
    NativeNodeContext, NativeNodeFailureKind, NativeNodeOutcome, NativeNodeServiceIdentity,
    NativeNodeServices, NativeOpaqueHandle, NativePrimitive, NativeTextGenerationService,
    NativeTextGenerationServiceError, NativeTextGenerationServiceOutput, NativeValue,
};
use sim_runtime::{NativeHandleStoreGeneration, generated_native_node_registry_projection};
use sim_tensor::{CancellationToken, CpuWorkspaceAuthority};
use sim_types::{AttemptId, NodeId, PromptId};
use std::{
    collections::BTreeMap,
    error::Error,
    sync::{Arc, Mutex},
};
use uuid::Uuid;

const FIXTURE: &str = include_str!("../fixtures/nodes/text-comfy-node-0649/fixture.json");
const MULTILINE_OUTPUT: &str = "first line\nsecond\tcolumn\r\nthird line";

fn context(
    authority: &CpuWorkspaceAuthority,
    store: Arc<dyn NativeHandleStore>,
    name: &str,
    cancellation: CancellationToken,
    services: NativeNodeServices,
) -> Result<NativeNodeContext, Box<dyn Error>> {
    Ok(NativeNodeContext::new_with_services(
        PromptId(Uuid::from_u128(0x67801)),
        store.attempt_id(),
        NodeId(name.to_owned()),
        cancellation,
        authority.authorize_workspace(1024)?,
        store,
        services,
    )?)
}
fn primitive(value: &Value) -> Result<NativeValue, Box<dyn Error>> {
    let value = match value {
        Value::String(value) => NativePrimitive::String(value.clone()),
        Value::Bool(value) => NativePrimitive::Boolean(*value),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                NativePrimitive::Integer(value)
            } else if let Some(value) = value.as_u64() {
                NativePrimitive::UnsignedInteger(value)
            } else {
                NativePrimitive::Number(value.as_f64().ok_or("source number absent")?)
            }
        }
        _ => return Err("source scalar absent".into()),
    };
    Ok(NativeValue::Primitive { value })
}
fn output(outcome: NativeNodeOutcome) -> Result<NativeValue, Box<dyn Error>> {
    let NativeNodeOutcome::Values {
        mut outputs,
        ui,
        effects,
    } = outcome
    else {
        return Err("ordinary values absent".into());
    };
    assert!(ui.is_none() && effects.is_empty());
    assert_eq!(outputs.len(), 1);
    outputs.pop().ok_or_else(|| "single output absent".into())
}

#[test]
fn task678_registered_text_transforms_preserve_unicode_and_transport_boundaries()
-> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let registry = generated_native_node_registry_projection(None)?;
    let (_backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let generation = NativeHandleStoreGeneration::new()?;
    let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x67802)));
    let mut successful = 0;
    let mut rejected = 0;
    for case in fixture["source_transforms"]["cases"]
        .as_array()
        .ok_or("source cases absent")?
    {
        let name = case["class_type"].as_str().ok_or("source class absent")?;
        let node = registry.node(name).ok_or("registered node absent")?;
        let inputs = case["inputs"]
            .as_object()
            .ok_or("source inputs absent")?
            .iter()
            .map(|(name, value)| Ok((name.clone(), primitive(value)?)))
            .collect::<Result<BTreeMap<_, _>, Box<dyn Error>>>()?;
        let persisted = serde_json::to_string(&inputs)?;
        let invocation = context(
            &authority,
            store.clone(),
            name,
            CancellationToken::default(),
            NativeNodeServices::checked(None, None, None)?,
        )?;
        let result = smol::block_on(node.execute(invocation, inputs.clone()));
        let source_text = case["inputs"]["texts"]
            .as_str()
            .ok_or("source text absent")?;
        let forbidden = source_text
            .chars()
            .any(|character| character.is_control() && !matches!(character, '\n' | '\r' | '\t'));
        if forbidden {
            assert!(matches!(result,Err(ref error) if error.kind==NativeNodeFailureKind::Failure));
            assert!(node.cache_change_token(&inputs).is_err());
            rejected += 1;
        } else {
            assert_eq!(output(result?)?, primitive(&case["output"])?);
            let recovered: BTreeMap<String, NativeValue> = serde_json::from_str(&persisted)?;
            assert_eq!(
                node.cache_change_token(&inputs)?,
                node.cache_change_token(&recovered)?
            );
            let retry = context(
                &authority,
                store.clone(),
                name,
                CancellationToken::default(),
                NativeNodeServices::checked(None, None, None)?,
            )?;
            assert_eq!(
                output(smol::block_on(node.execute(retry, recovered))?)?,
                primitive(&case["output"])?
            );
            successful += 1;
        }
        assert_eq!(serde_json::to_string(&inputs)?, persisted);
    }
    assert_eq!((successful, rejected), (48, 16));
    Ok(())
}

#[derive(Debug)]
struct CaptureGeneration {
    identity: NativeNodeServiceIdentity,
    requests: Arc<Mutex<Vec<NativeMultimodalTextGenerationRequest>>>,
    cancel_after: bool,
    fail: bool,
    text: String,
}
impl NativeTextGenerationService for CaptureGeneration {
    fn identity(&self) -> &NativeNodeServiceIdentity {
        &self.identity
    }
    fn generate(
        &self,
        context: &NativeNodeContext,
        request: &NativeMultimodalTextGenerationRequest,
    ) -> Result<NativeTextGenerationServiceOutput, NativeTextGenerationServiceError> {
        self.requests
            .lock()
            .map_err(|_| {
                NativeTextGenerationServiceError::Execution("capture mutex poisoned".to_owned())
            })?
            .push(request.clone());
        if self.cancel_after {
            context.cancellation.cancel();
        }
        if self.fail {
            return Err(NativeTextGenerationServiceError::Execution(
                "injected generation failure".to_owned(),
            ));
        }
        NativeTextGenerationServiceOutput::checked(
            self.text.clone(),
            1,
            request.maximum_new_tokens(),
        )
    }
}
fn generation_inputs(
    case: &Value,
    store: &dyn NativeHandleStore,
) -> Result<BTreeMap<String, NativeValue>, Box<dyn Error>> {
    let mut inputs = BTreeMap::new();
    inputs.insert(
        "clip".to_owned(),
        NativeValue::Handle {
            value: NativeOpaqueHandle::new(
                NativeHandleType::new(NativeHandleKind::Clip, "CLIP")?,
                store.identity(),
                "source-clip-reference",
                1,
                Some("a".repeat(64)),
            )?,
        },
    );
    for (name, value) in case["inputs"].as_object().ok_or("source inputs absent")? {
        if value.is_null() {
            continue;
        }
        let value = if name == "sampling_mode" {
            NativeValue::PreservedUnknown {
                type_name: "COMFY_DYNAMICCOMBO_V3".to_owned(),
                value: value.clone(),
            }
        } else if ["image", "video", "audio"].contains(&name.as_str()) {
            NativeValue::Handle {
                value: NativeOpaqueHandle::new(
                    NativeHandleType::new(
                        if name == "audio" {
                            NativeHandleKind::Audio
                        } else {
                            NativeHandleKind::Image
                        },
                        if name == "audio" { "AUDIO" } else { "IMAGE" },
                    )?,
                    store.identity(),
                    name,
                    1,
                    Some("b".repeat(64)),
                )?,
            }
        } else {
            primitive(value)?
        };
        inputs.insert(name.clone(), value);
    }
    Ok(inputs)
}
fn compare_request(
    request: &NativeMultimodalTextGenerationRequest,
    case: &Value,
) -> Result<(), Box<dyn Error>> {
    let calls = case["calls"].as_array().ok_or("source calls absent")?;
    let tokenize = calls.first().ok_or("tokenize call absent")?;
    let generate = calls.get(1).ok_or("generate call absent")?;
    assert_eq!(
        request.formatted_prompt(),
        tokenize["prompt"].as_str().ok_or("source prompt absent")?
    );
    assert_eq!(
        request.use_default_template(),
        !tokenize["skip_template"]
            .as_bool()
            .ok_or("source template absent")?
    );
    assert_eq!(
        request.thinking(),
        tokenize["thinking"].as_bool().ok_or("thinking absent")?
    );
    assert_eq!(request.image().is_some(), !tokenize["image"].is_null());
    assert_eq!(request.video().is_some(), !tokenize["video"].is_null());
    assert_eq!(request.audio().is_some(), !tokenize["audio"].is_null());
    assert_eq!(
        request.do_sample(),
        generate["do_sample"]
            .as_bool()
            .ok_or("source sampling absent")?
    );
    assert_eq!(
        request.maximum_new_tokens() as u64,
        generate["max_length"]
            .as_u64()
            .ok_or("source length absent")?
    );
    assert_eq!(
        request.top_k() as u64,
        generate["top_k"].as_u64().ok_or("source top k absent")?
    );
    assert_eq!(request.seed(), generate["seed"].as_u64().unwrap_or(0));
    for (bits, name) in [
        (request.temperature_bits(), "temperature"),
        (request.top_p_bits(), "top_p"),
        (request.minimum_p_bits(), "min_p"),
        (request.repetition_penalty_bits(), "repetition_penalty"),
        (request.presence_penalty_bits(), "presence_penalty"),
    ] {
        assert_eq!(
            bits,
            (generate[name].as_f64().ok_or("source parameter absent")? as f32).to_bits()
        );
    }
    Ok(())
}
#[test]
fn task678_registered_generation_projects_source_parameters_and_multiline_results()
-> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let registry = generated_native_node_registry_projection(None)?;
    let (_backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let generation = NativeHandleStoreGeneration::new()?;
    let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x67803)));
    let mut count = 0;
    for case in fixture["source_generation"]["cases"]
        .as_array()
        .ok_or("source cases absent")?
    {
        let name = case["class_type"].as_str().ok_or("source class absent")?;
        let node = registry.node(name).ok_or("registered node absent")?;
        let requests = Arc::new(Mutex::new(Vec::new()));
        let service = Arc::new(CaptureGeneration {
            identity: NativeNodeServiceIdentity::checked(
                Uuid::from_u128(0x67804),
                store.attempt_id(),
                NodeId(name.to_owned()),
            )?,
            requests: requests.clone(),
            cancel_after: false,
            fail: false,
            text: MULTILINE_OUTPUT.to_owned(),
        });
        let inputs = generation_inputs(case, store.as_ref())?;
        let token = node.cache_change_token(&inputs)?;
        let persisted = serde_json::to_string(&inputs)?;
        for _ in 0..2 {
            let recovered: BTreeMap<String, NativeValue> = serde_json::from_str(&persisted)?;
            assert_eq!(node.cache_change_token(&recovered)?, token);
            let invocation = context(
                &authority,
                store.clone(),
                name,
                CancellationToken::default(),
                NativeNodeServices::checked(None, None, None)?
                    .with_text_generation(service.clone())?,
            )?;
            assert_eq!(
                output(smol::block_on(node.execute(invocation, recovered))?)?,
                NativeValue::Primitive {
                    value: NativePrimitive::String(MULTILINE_OUTPUT.to_owned())
                }
            );
        }
        let captured = requests.lock().map_err(|_| "capture mutex poisoned")?;
        assert_eq!(captured.len(), 2);
        for request in captured.iter() {
            compare_request(request, case)?;
        }
        assert_eq!(serde_json::to_string(&inputs)?, persisted);
        count += 1;
    }
    assert_eq!(count, 24);
    Ok(())
}

#[test]
fn task678_text_nodes_reject_invalid_inputs_and_pre_cancel_without_outputs()
-> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let registry = generated_native_node_registry_projection(None)?;
    let (_backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let generation = NativeHandleStoreGeneration::new()?;
    let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x67805)));
    for row in fixture["nodes"].as_array().ok_or("node rows absent")? {
        let name = row["class_type"].as_str().ok_or("class absent")?;
        let node = registry.node(name).ok_or("registered node absent")?;
        let inputs = if name.starts_with("TextGenerate") {
            let case = fixture["source_generation"]["cases"]
                .as_array()
                .ok_or("generation cases absent")?
                .iter()
                .find(|case| case["class_type"] == name)
                .ok_or("generation case absent")?;
            generation_inputs(case, store.as_ref())?
        } else {
            let mut inputs = BTreeMap::from([(
                "texts".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::String("hello\nworld".to_owned()),
                },
            )]);
            if name == "TruncateText" {
                inputs.insert(
                    "max_length".to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::Integer(5),
                    },
                );
            }
            inputs
        };
        let cancelled = CancellationToken::default();
        assert!(cancelled.cancel());
        let invocation = context(
            &authority,
            store.clone(),
            name,
            cancelled,
            NativeNodeServices::checked(None, None, None)?,
        )?;
        assert!(
            matches!(node.demanded_lazy_inputs(&invocation,&inputs),Err(error) if error.kind==NativeNodeFailureKind::Interrupted)
        );
        assert!(
            matches!(smol::block_on(node.execute(invocation,inputs.clone())),Err(error) if error.kind==NativeNodeFailureKind::Interrupted)
        );
        let mut invalid_cases = vec![BTreeMap::new()];
        let mut extra = inputs.clone();
        extra.insert(
            "unexpected".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::Boolean(true),
            },
        );
        invalid_cases.push(extra);
        let text_name = if name.starts_with("TextGenerate") {
            "prompt"
        } else {
            "texts"
        };
        for value in [
            NativeValue::Primitive {
                value: NativePrimitive::String("bad\0value".to_owned()),
            },
            NativeValue::Primitive {
                value: NativePrimitive::String("a".repeat(1024 * 1024 + 1)),
            },
            NativeValue::Primitive {
                value: NativePrimitive::Integer(7),
            },
            NativeValue::List { values: vec![] },
        ] {
            let mut invalid = inputs.clone();
            invalid.insert(text_name.to_owned(), value);
            invalid_cases.push(invalid);
        }
        if inputs.contains_key("max_length") {
            for value in [0, if name == "TruncateText" { 10001 } else { 32769 }, -1] {
                let mut invalid = inputs.clone();
                invalid.insert(
                    "max_length".to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::Integer(value),
                    },
                );
                invalid_cases.push(invalid);
            }
        }
        if name.starts_with("TextGenerate") {
            for (field, value) in [
                ("temperature", serde_json::json!(2.01)),
                ("top_k", serde_json::json!(1001)),
                ("top_p", serde_json::json!(-0.01)),
                ("min_p", serde_json::json!(1.01)),
                ("repetition_penalty", serde_json::json!(5.01)),
                ("presence_penalty", serde_json::json!(-0.01)),
                ("seed", serde_json::json!(-1)),
                ("sampling_mode", serde_json::json!("invalid")),
                ("unexpected", serde_json::json!(1)),
            ] {
                let mut fields = serde_json::json!({"sampling_mode":"off"});
                fields[field] = value;
                let mut invalid = inputs.clone();
                invalid.insert(
                    "sampling_mode".to_owned(),
                    NativeValue::PreservedUnknown {
                        type_name: "COMFY_DYNAMICCOMBO_V3".to_owned(),
                        value: fields,
                    },
                );
                invalid_cases.push(invalid);
            }
            for field in ["thinking", "use_default_template"] {
                let mut invalid = inputs.clone();
                invalid.insert(
                    field.to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::String("true".to_owned()),
                    },
                );
                invalid_cases.push(invalid);
            }
        }
        for invalid in invalid_cases {
            assert!(
                node.cache_change_token(&invalid).is_err(),
                "invalid cache inputs admitted for {name}"
            );
            let invocation = context(
                &authority,
                store.clone(),
                name,
                CancellationToken::default(),
                NativeNodeServices::checked(None, None, None)?,
            )?;
            assert!(
                matches!(smol::block_on(node.execute(invocation,invalid)),Err(error) if error.kind==NativeNodeFailureKind::Failure)
            );
        }
    }
    Ok(())
}

#[test]
fn task678_generation_failure_cancellation_and_retry_are_atomic() -> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let registry = generated_native_node_registry_projection(None)?;
    let (_backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let generation = NativeHandleStoreGeneration::new()?;
    let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x67806)));
    for name in ["TextGenerate", "TextGenerateLTX2Prompt"] {
        let node = registry.node(name).ok_or("registered node absent")?;
        let case = fixture["source_generation"]["cases"]
            .as_array()
            .ok_or("generation cases absent")?
            .iter()
            .find(|case| case["class_type"] == name)
            .ok_or("source case absent")?;
        let inputs = generation_inputs(case, store.as_ref())?;
        let invocation = context(
            &authority,
            store.clone(),
            name,
            CancellationToken::default(),
            NativeNodeServices::checked(None, None, None)?,
        )?;
        assert!(
            matches!(smol::block_on(node.execute(invocation,inputs.clone())),Err(error) if error.message.contains("unavailable") && error.kind==NativeNodeFailureKind::Failure)
        );
        for (cancel_after, fail, text) in [
            (true, false, MULTILINE_OUTPUT.to_owned()),
            (false, true, MULTILINE_OUTPUT.to_owned()),
            (false, false, "invalid\0output".to_owned()),
            (false, false, "x".repeat(1024 * 1024 + 1)),
        ] {
            let requests = Arc::new(Mutex::new(Vec::new()));
            let identity = NativeNodeServiceIdentity::checked(
                Uuid::from_u128(0x67807),
                store.attempt_id(),
                NodeId(name.to_owned()),
            )?;
            let service = Arc::new(CaptureGeneration {
                identity: identity.clone(),
                requests: requests.clone(),
                cancel_after,
                fail,
                text,
            });
            let invocation = context(
                &authority,
                store.clone(),
                name,
                CancellationToken::default(),
                NativeNodeServices::checked(None, None, None)?.with_text_generation(service)?,
            )?;
            let error = smol::block_on(node.execute(invocation, inputs.clone()))
                .expect_err("injected failure must not return an output");
            assert_eq!(
                error.kind,
                if cancel_after {
                    NativeNodeFailureKind::Interrupted
                } else {
                    NativeNodeFailureKind::Failure
                }
            );
            assert_eq!(
                requests.lock().map_err(|_| "capture mutex poisoned")?.len(),
                1
            );
            let service = Arc::new(CaptureGeneration {
                identity,
                requests: requests.clone(),
                cancel_after: false,
                fail: false,
                text: MULTILINE_OUTPUT.to_owned(),
            });
            let retry = context(
                &authority,
                store.clone(),
                name,
                CancellationToken::default(),
                NativeNodeServices::checked(None, None, None)?.with_text_generation(service)?,
            )?;
            assert_eq!(
                output(smol::block_on(node.execute(retry, inputs.clone()))?)?,
                NativeValue::Primitive {
                    value: NativePrimitive::String(MULTILINE_OUTPUT.to_owned())
                }
            );
            assert_eq!(
                requests.lock().map_err(|_| "capture mutex poisoned")?.len(),
                2
            );
        }
    }
    Ok(())
}

use sim_model::{
    DecoderActivation, DecoderArchitecture, DecoderAttentionWeights, DecoderLayerKind,
    DecoderLayerWeights, DecoderRopeConfiguration, DecoderTextConfiguration, DecoderTextWeights,
    Gemma4AudioBlockWeights, Gemma4AudioConfiguration, Gemma4AudioFeedForwardWeights,
    Gemma4AudioProfile, Gemma4AudioWeights, Gemma4ClippedLinearWeights, Gemma4DecoderConfiguration,
    Gemma4LayerInputWeights, Gemma4PerLayerWeights, Gemma4VisionBlockWeights,
    Gemma4VisionConfiguration, Gemma4VisionWeights, GemmaTokenizer, NativeDecoderTextEncoder,
    NativeGemma4AudioEncoder, NativeGemma4VisionEncoder, NativeGemmaMultimodal, NativeModelPayload,
    NativePromptTokenizer, NativeTokenizerFamily, RopeScaling, TokenizerConfiguration,
};
use sim_tensor::{
    CpuBackend, DType, DeviceId, ExecutionContext, StreamId, Tensor, TensorDescriptor,
};

fn tensor(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, context.stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn filled_tensor(
    backend: &CpuBackend,
    shape: &[u64],
    value: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn Error>> {
    let elements = shape.iter().try_fold(1_usize, |total, dimension| {
        total.checked_mul(usize::try_from(*dimension).ok()?)
    });
    tensor(
        backend,
        shape,
        &vec![value; elements.ok_or("fixture tensor shape overflowed")?],
        context,
    )
}

fn matrix_tensor(
    backend: &CpuBackend,
    output: usize,
    input: usize,
    scale: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn Error>> {
    let count = output.checked_mul(input).ok_or("matrix size overflowed")?;
    let values = (0..count)
        .map(|index| {
            let row = index / input;
            let column = index % input;
            if row % input == column {
                scale
            } else {
                ((index * 11 % 17) as f32 - 8.0) * scale * 0.0125
            }
        })
        .collect::<Vec<_>>();
    tensor(
        backend,
        &[u64::try_from(output)?, u64::try_from(input)?],
        &values,
        context,
    )
}

fn reduced_gemma4_clipped_linear(
    backend: &CpuBackend,
    input: usize,
    output: usize,
    scale: f32,
    minimum: &Tensor,
    maximum: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Gemma4ClippedLinearWeights, Box<dyn Error>> {
    let mut values = vec![0.0_f32; input * output];
    for row in 0..output {
        for column in 0..input {
            values[row * input + column] =
                scale * (1.0 + ((row * input + column) % 7) as f32 * 0.125);
        }
    }
    Ok(Gemma4ClippedLinearWeights {
        weight: tensor(
            backend,
            &[u64::try_from(output)?, u64::try_from(input)?],
            &values,
            context,
        )?,
        input_minimum: minimum.clone(),
        input_maximum: maximum.clone(),
        output_minimum: minimum.clone(),
        output_maximum: maximum.clone(),
    })
}

fn reduced_gemma4_vision_weights(
    backend: &CpuBackend,
    configuration: &Gemma4VisionConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Gemma4VisionWeights, Box<dyn Error>> {
    let hidden = configuration.hidden_size;
    let intermediate = configuration.intermediate_size;
    let patch_width = 3 * configuration.patch_size * configuration.patch_size;
    let minimum = filled_tensor(backend, &[1], -8.0, context)?;
    let maximum = filled_tensor(backend, &[1], 8.0, context)?;
    let mut patch_values = vec![0.0_f32; hidden * patch_width];
    for output in 0..hidden {
        for source in 0..patch_width {
            patch_values[output * patch_width + source] =
                0.0005 * (1.0 + ((source + output * 3) % 11) as f32);
        }
    }
    let mut position_values = vec![0.0_f32; 2 * configuration.position_embeddings * hidden];
    for axis in 0..2 {
        for position in 0..configuration.position_embeddings {
            for feature in 0..hidden {
                position_values
                    [(axis * configuration.position_embeddings + position) * hidden + feature] =
                    axis as f32 * 0.003 + position as f32 * 0.0002 + feature as f32 * 0.0001;
            }
        }
    }
    let mut blocks = Vec::new();
    for layer in 0..configuration.layer_count {
        let scale = 0.01 + layer as f32 * 0.001;
        blocks.push(Gemma4VisionBlockWeights {
            input_normalization_weight: filled_tensor(
                backend,
                &[u64::try_from(hidden)?],
                1.0,
                context,
            )?,
            query: reduced_gemma4_clipped_linear(
                backend, hidden, hidden, scale, &minimum, &maximum, context,
            )?,
            key: reduced_gemma4_clipped_linear(
                backend,
                hidden,
                hidden,
                scale * 1.1,
                &minimum,
                &maximum,
                context,
            )?,
            value: reduced_gemma4_clipped_linear(
                backend,
                hidden,
                hidden,
                scale * 1.2,
                &minimum,
                &maximum,
                context,
            )?,
            query_normalization_weight: filled_tensor(
                backend,
                &[u64::try_from(configuration.head_dimension)?],
                1.0,
                context,
            )?,
            key_normalization_weight: filled_tensor(
                backend,
                &[u64::try_from(configuration.head_dimension)?],
                1.0,
                context,
            )?,
            attention_output: reduced_gemma4_clipped_linear(
                backend,
                hidden,
                hidden,
                scale * 0.7,
                &minimum,
                &maximum,
                context,
            )?,
            post_attention_normalization_weight: filled_tensor(
                backend,
                &[u64::try_from(hidden)?],
                1.0,
                context,
            )?,
            pre_feed_forward_normalization_weight: filled_tensor(
                backend,
                &[u64::try_from(hidden)?],
                1.0,
                context,
            )?,
            feed_forward_gate: reduced_gemma4_clipped_linear(
                backend,
                hidden,
                intermediate,
                scale * 0.6,
                &minimum,
                &maximum,
                context,
            )?,
            feed_forward_up: reduced_gemma4_clipped_linear(
                backend,
                hidden,
                intermediate,
                scale * 0.5,
                &minimum,
                &maximum,
                context,
            )?,
            feed_forward_down: reduced_gemma4_clipped_linear(
                backend,
                intermediate,
                hidden,
                scale * 0.4,
                &minimum,
                &maximum,
                context,
            )?,
            post_feed_forward_normalization_weight: filled_tensor(
                backend,
                &[u64::try_from(hidden)?],
                1.0,
                context,
            )?,
        });
    }
    Ok(Gemma4VisionWeights {
        patch_projection_weight: tensor(
            backend,
            &[u64::try_from(hidden)?, u64::try_from(patch_width)?],
            &patch_values,
            context,
        )?,
        position_embedding: tensor(
            backend,
            &[
                2,
                u64::try_from(configuration.position_embeddings)?,
                u64::try_from(hidden)?,
            ],
            &position_values,
            context,
        )?,
        blocks,
        projector_weight: filled_tensor(
            backend,
            &[
                u64::try_from(configuration.output_hidden_size)?,
                u64::try_from(hidden)?,
            ],
            0.075,
            context,
        )?,
    })
}

fn reduced_gemma4_audio_feed_forward(
    backend: &CpuBackend,
    configuration: &Gemma4AudioConfiguration,
    scale: f32,
    minimum: &Tensor,
    maximum: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Gemma4AudioFeedForwardWeights, Box<dyn Error>> {
    Ok(Gemma4AudioFeedForwardWeights {
        pre_normalization_weight: filled_tensor(
            backend,
            &[u64::try_from(configuration.hidden_size)?],
            1.0,
            context,
        )?,
        first: reduced_gemma4_clipped_linear(
            backend,
            configuration.hidden_size,
            configuration.intermediate_size,
            scale,
            minimum,
            maximum,
            context,
        )?,
        second: reduced_gemma4_clipped_linear(
            backend,
            configuration.intermediate_size,
            configuration.hidden_size,
            scale * 0.7,
            minimum,
            maximum,
            context,
        )?,
        post_normalization_weight: filled_tensor(
            backend,
            &[u64::try_from(configuration.hidden_size)?],
            1.0,
            context,
        )?,
    })
}

fn reduced_gemma4_audio_weights(
    backend: &CpuBackend,
    configuration: &Gemma4AudioConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Gemma4AudioWeights, Box<dyn Error>> {
    let minimum = filled_tensor(backend, &[1], -8.0, context)?;
    let maximum = filled_tensor(backend, &[1], 8.0, context)?;
    let hidden = configuration.hidden_size;
    let mut blocks = Vec::new();
    for index in 0..configuration.layer_count {
        let scale = 0.01 + index as f32 * 0.001;
        blocks.push(Gemma4AudioBlockWeights {
            feed_forward_one: reduced_gemma4_audio_feed_forward(
                backend,
                configuration,
                scale,
                &minimum,
                &maximum,
                context,
            )?,
            query: reduced_gemma4_clipped_linear(
                backend, hidden, hidden, scale, &minimum, &maximum, context,
            )?,
            key: reduced_gemma4_clipped_linear(
                backend,
                hidden,
                hidden,
                scale * 1.1,
                &minimum,
                &maximum,
                context,
            )?,
            value: reduced_gemma4_clipped_linear(
                backend,
                hidden,
                hidden,
                scale * 1.2,
                &minimum,
                &maximum,
                context,
            )?,
            attention_output: reduced_gemma4_clipped_linear(
                backend,
                hidden,
                hidden,
                scale * 0.8,
                &minimum,
                &maximum,
                context,
            )?,
            attention_scale: filled_tensor(
                backend,
                &[u64::try_from(hidden / configuration.attention_heads)?],
                0.1,
                context,
            )?,
            relative_key_projection_weight: filled_tensor(
                backend,
                &[u64::try_from(hidden)?, u64::try_from(hidden)?],
                0.02,
                context,
            )?,
            pre_attention_normalization_weight: filled_tensor(
                backend,
                &[u64::try_from(hidden)?],
                1.0,
                context,
            )?,
            post_attention_normalization_weight: filled_tensor(
                backend,
                &[u64::try_from(hidden)?],
                1.0,
                context,
            )?,
            convolution_pre_normalization_weight: filled_tensor(
                backend,
                &[u64::try_from(hidden)?],
                1.0,
                context,
            )?,
            convolution_start: reduced_gemma4_clipped_linear(
                backend,
                hidden,
                hidden * 2,
                scale * 0.9,
                &minimum,
                &maximum,
                context,
            )?,
            depthwise_convolution_weight: filled_tensor(
                backend,
                &[
                    u64::try_from(hidden)?,
                    1,
                    u64::try_from(configuration.convolution_kernel_size)?,
                ],
                0.2,
                context,
            )?,
            convolution_normalization_weight: filled_tensor(
                backend,
                &[u64::try_from(hidden)?],
                1.0,
                context,
            )?,
            convolution_end: reduced_gemma4_clipped_linear(
                backend,
                hidden,
                hidden,
                scale * 0.6,
                &minimum,
                &maximum,
                context,
            )?,
            feed_forward_two: reduced_gemma4_audio_feed_forward(
                backend,
                configuration,
                scale * 0.5,
                &minimum,
                &maximum,
                context,
            )?,
            output_normalization_weight: filled_tensor(
                backend,
                &[u64::try_from(hidden)?],
                1.0,
                context,
            )?,
        });
    }
    Ok(Gemma4AudioWeights {
        first_convolution_weight: filled_tensor(
            backend,
            &[
                u64::try_from(configuration.first_convolution_channels)?,
                1,
                3,
                3,
            ],
            0.04,
            context,
        )?,
        first_convolution_normalization_weight: filled_tensor(
            backend,
            &[u64::try_from(configuration.first_convolution_channels)?],
            1.0,
            context,
        )?,
        second_convolution_weight: filled_tensor(
            backend,
            &[
                u64::try_from(configuration.second_convolution_channels)?,
                u64::try_from(configuration.first_convolution_channels)?,
                3,
                3,
            ],
            0.03,
            context,
        )?,
        second_convolution_normalization_weight: filled_tensor(
            backend,
            &[u64::try_from(configuration.second_convolution_channels)?],
            1.0,
            context,
        )?,
        subsample_projection_weight: filled_tensor(
            backend,
            &[
                u64::try_from(hidden)?,
                u64::try_from(configuration.mel_bins.div_ceil(2).div_ceil(2))?
                    * u64::try_from(configuration.second_convolution_channels)?,
            ],
            0.02,
            context,
        )?,
        blocks,
        encoder_output_weight: filled_tensor(
            backend,
            &[
                u64::try_from(configuration.encoder_output_size)?,
                u64::try_from(hidden)?,
            ],
            0.025,
            context,
        )?,
        encoder_output_bias: filled_tensor(
            backend,
            &[u64::try_from(configuration.encoder_output_size)?],
            0.01,
            context,
        )?,
        projector_weight: filled_tensor(
            backend,
            &[
                u64::try_from(configuration.output_hidden_size)?,
                u64::try_from(configuration.encoder_output_size)?,
            ],
            0.03,
            context,
        )?,
    })
}

fn gemma4_prompt_tokenizer(
    hidden_size: usize,
) -> Result<Arc<NativePromptTokenizer>, Box<dyn Error>> {
    let mut tokenizer: Value = serde_json::from_str(include_str!(
        "../fixtures/text_generation/gemma_multimodal/tokenizer/gemma4-tokenizer.json"
    ))?;
    let added = tokenizer["added_tokens"]
        .as_array()
        .ok_or("added tokens absent")?
        .clone();
    let vocabulary = tokenizer["model"]["vocab"]
        .as_array_mut()
        .ok_or("tokenizer vocabulary absent")?;
    for index in vocabulary.len()..262_144 {
        vocabulary.push(serde_json::json!([
            format!("[fixture-token-{index}]"),
            -100.0
        ]));
    }
    *vocabulary.get_mut(1).ok_or("fixture end token absent")? =
        serde_json::json!(["[fixture-token-1]", 0.0]);
    for token in added {
        let index = usize::try_from(token["id"].as_u64().ok_or("token ID absent")?)?;
        *vocabulary
            .get_mut(index)
            .ok_or("token ID outside vocabulary")? = serde_json::json!([token["content"], -100.0]);
    }
    let family = GemmaTokenizer::gemma4_from_tokenizer_json(
        &serde_json::to_string(&tokenizer)?,
        &CancellationToken::default(),
    )?;
    Ok(Arc::new(NativePromptTokenizer::checked(
        NativeTokenizerFamily::Gemma(family),
        TokenizerConfiguration {
            maximum_length: 131_072,
            minimum_length: Some(1),
            minimum_padding: None,
            pad_to_maximum_length: false,
            pad_left: true,
            start_token: Some(2),
            end_token: None,
            pad_token: 0,
            maximum_word_length: 8,
            disable_weights: true,
            embedding_width: Some(hidden_size),
        },
        BTreeMap::new(),
    )?))
}

fn reduced_gemma4_decoder_configuration() -> DecoderTextConfiguration {
    DecoderTextConfiguration {
        architecture: DecoderArchitecture::Gemma,
        dtype: DType::F32,
        device: DeviceId::CPU,
        vocabulary_size: 262_144,
        maximum_tokens: 8192,
        hidden_size: 4,
        feed_forward_size: 8,
        layer_kinds: (0_usize..10)
            .map(|index| {
                if (index + 1).is_multiple_of(5) {
                    DecoderLayerKind::FullAttention
                } else {
                    DecoderLayerKind::SlidingAttention
                }
            })
            .collect(),
        attention_heads: 2,
        key_value_heads: 1,
        head_dimension: 2,
        query_key_norm: true,
        qwen35_linear: None,
        gemma3: None,
        gemma4: Some(Gemma4DecoderConfiguration {
            source_profile: None,
            local_rope: DecoderRopeConfiguration {
                theta: 10_000.0,
                rotary_dimension: 2,
                interleaved_sections: Vec::new(),
                scaling: RopeScaling::None,
            },
            global_head_dimension: 4,
            global_rotary_pairs: 1,
            sliding_layers_per_cycle: 4,
            hidden_size_per_layer_input: 2,
            shared_key_value_layers: 5,
            double_wide_mlp: true,
        }),
        normalization_epsilon_bits: 1.0e-6_f32.to_bits(),
        rope: DecoderRopeConfiguration {
            theta: 1_000_000.0,
            rotary_dimension: 2,
            interleaved_sections: Vec::new(),
            scaling: RopeScaling::None,
        },
        sliding_window: Some(512),
        activation: DecoderActivation::GeluTanh,
        embedding_scale_bits: 2.0_f32.to_bits(),
        residual_scale_bits: 1.0_f32.to_bits(),
        norm_weight_offset_bits: 0.0_f32.to_bits(),
        logits_soft_cap_bits: Some(30.0_f32.to_bits()),
        tied_output_head: true,
        stop_tokens: vec![1, 50, 106],
    }
}

fn reduced_gemma4_decoder(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    scale_offset: f32,
) -> Result<Arc<NativeDecoderTextEncoder>, Box<dyn Error>> {
    let configuration = reduced_gemma4_decoder_configuration();
    let gemma4 = configuration.gemma4.as_ref().ok_or("Gemma4 config")?;
    let first_shared = configuration.layer_kinds.len() - gemma4.shared_key_value_layers;
    let mut layers = Vec::new();
    for (index, kind) in configuration.layer_kinds.iter().copied().enumerate() {
        let head_dimension = if kind == DecoderLayerKind::FullAttention {
            gemma4.global_head_dimension
        } else {
            configuration.head_dimension
        };
        let query_width = configuration.attention_heads * head_dimension;
        let key_value_width = configuration.key_value_heads * head_dimension;
        let feed_forward = if gemma4.double_wide_mlp && index >= first_shared {
            configuration.feed_forward_size * 2
        } else {
            configuration.feed_forward_size
        };
        let scale = scale_offset + 0.01 + index as f32 * 0.001;
        layers.push(DecoderLayerWeights {
            attention_norm_weight: filled_tensor(backend, &[4], 0.3, context)?,
            attention: DecoderAttentionWeights::DotProduct {
                query_weight: matrix_tensor(backend, query_width, 4, scale, context)?,
                key_weight: matrix_tensor(backend, key_value_width, 4, scale + 0.01, context)?,
                value_weight: matrix_tensor(backend, key_value_width, 4, scale + 0.02, context)?,
                query_norm_weight: Some(filled_tensor(
                    backend,
                    &[u64::try_from(head_dimension)?],
                    0.8,
                    context,
                )?),
                key_norm_weight: Some(filled_tensor(
                    backend,
                    &[u64::try_from(head_dimension)?],
                    1.1,
                    context,
                )?),
                output_weight: matrix_tensor(backend, 4, query_width, scale + 0.03, context)?,
            },
            feed_forward_norm_weight: filled_tensor(backend, &[4], 0.25, context)?,
            feed_forward_gate_weight: matrix_tensor(
                backend,
                feed_forward,
                4,
                scale + 0.04,
                context,
            )?,
            feed_forward_up_weight: matrix_tensor(backend, feed_forward, 4, scale + 0.05, context)?,
            feed_forward_down_weight: matrix_tensor(
                backend,
                4,
                feed_forward,
                scale + 0.06,
                context,
            )?,
            post_attention_norm_weight: Some(filled_tensor(backend, &[4], 0.2, context)?),
            post_feed_forward_norm_weight: Some(filled_tensor(backend, &[4], 0.22, context)?),
            attention_sink: None,
            gemma4_layer_input: Some(Gemma4LayerInputWeights {
                gate_weight: matrix_tensor(backend, 2, 4, scale + 0.07, context)?,
                projection_weight: matrix_tensor(backend, 4, 2, scale + 0.08, context)?,
                post_norm_weight: filled_tensor(backend, &[4], 0.4, context)?,
                layer_scalar: tensor(backend, &[1], &[0.97 + index as f32 * 0.001], context)?,
            }),
        });
    }
    let total_per_layer = configuration.layer_kinds.len() * gemma4.hidden_size_per_layer_input;
    Ok(Arc::new(NativeDecoderTextEncoder::new(
        configuration,
        DecoderTextWeights {
            token_embedding: filled_tensor(backend, &[262_144, 4], 0.001, context)?,
            layers,
            final_norm_weight: filled_tensor(backend, &[4], 0.35, context)?,
            output_head_weight: None,
            gemma4_per_layer: Some(Gemma4PerLayerWeights {
                token_embedding: filled_tensor(
                    backend,
                    &[262_144, u64::try_from(total_per_layer)?],
                    0.002 + scale_offset,
                    context,
                )?,
                model_projection_weight: matrix_tensor(backend, total_per_layer, 4, 0.03, context)?,
                projection_norm_weight: tensor(backend, &[2], &[0.9, 1.1], context)?,
            }),
        },
    )?))
}

fn reduced_gemma_payload(
    backend: &CpuBackend,
    execution: &ExecutionContext<'_>,
) -> Result<sim_nodes::NativeStoredPayload, Box<dyn Error>> {
    let tokenizer = gemma4_prompt_tokenizer(4)?;
    let decoder = reduced_gemma4_decoder(backend, execution, 0.0)?;
    let vision_configuration = Gemma4VisionConfiguration::reduced_fixture(
        sim_model::Gemma4VisionProfile::E4B,
        4,
        8,
        1,
        1,
        4,
        16,
        64,
        3,
        4,
    );
    let vision = Arc::new(NativeGemma4VisionEncoder::new(
        vision_configuration.clone(),
        reduced_gemma4_vision_weights(backend, &vision_configuration, execution)?,
    )?);
    let audio_configuration = Gemma4AudioConfiguration::reduced_fixture(
        Gemma4AudioProfile::E4B,
        128,
        2,
        2,
        4,
        8,
        1,
        2,
        3,
        2,
        3,
        4,
        4,
    )?;
    let audio = Arc::new(NativeGemma4AudioEncoder::new(
        audio_configuration.clone(),
        reduced_gemma4_audio_weights(backend, &audio_configuration, execution)?,
        execution.cancellation,
    )?);
    let resource = Arc::new(NativeGemmaMultimodal::reduced_gemma4_fixture(
        tokenizer,
        decoder,
        vision,
        Some(audio),
        execution.cancellation,
    )?);
    assert!(!resource.is_source_exact_profile());
    assert!(NativeModelPayload::gemma_multimodal_clip(resource.clone()).is_err());
    let payload = Arc::new(NativeModelPayload::gemma_multimodal_clip_test_fixture(
        resource,
    )?);
    payload.validate()?;
    Ok(sim_nodes::NativeStoredPayload::Model(Arc::new(
        sim_nodes::NativeStoredModelPayload::model_resource(payload)?,
    )))
}

#[test]
fn task678_engine_text_nodes_execute_canonical_gemma_inference() -> Result<(), Box<dyn Error>> {
    use sim_nodes::{NativeNodeComputeSession, NativeStoredPayload};
    use sim_runtime::{
        AttemptState, CompiledNode, CompiledPlan, InputBinding, NativeImageExecutor,
        NativeRuntimeTextGenerationService,
    };
    const LIMIT: u64 = 256 * 1024 * 1024;
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let cancellation = CancellationToken::default();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(LIMIT)?,
        &cancellation,
    );
    let model = reduced_gemma_payload(&backend, &execution)?;
    let model_digest = model.digest_sha256();
    let image =
        sim_tensor::ImageTensor::from_f32(&backend, &execution, 1, 1, 1, 3, &[0.2, 0.5, 0.9])?;
    let image_payload = NativeStoredPayload::Tensor(Arc::new(
        sim_tensor::NativeTensorPayload::from_image(sim_tensor::NativeTensorRole::Image, image)?,
    ));
    let waveform_values = (0..640)
        .map(|index| (index as f32 * 0.017).sin() * 0.25)
        .collect::<Vec<_>>();
    let waveform = tensor(&backend, &[1, 1, 640], &waveform_values, &execution)?;
    let audio_payload = NativeStoredPayload::Audio(Arc::new(
        sim_media::NativeAudioPayload::checked(waveform, 16000)?,
    ));
    for name in ["TextGenerate", "TextGenerateLTX2Prompt"] {
        for media in ["none", "image", "video", "audio", "sampled"] {
            let sampled = media == "sampled";
            let executor = NativeImageExecutor::new_with_generated_registry(
                sim_types::ProfileId(Uuid::from_u128(0x67810)),
                BTreeMap::new(),
                true,
                backend.clone(),
            )?;
            let attempt = AttemptId(Uuid::from_u128(0x67811));
            let store = executor
                .handle_store_generation_for_test()
                .handle_store_for_attempt(attempt);
            let clip = store.publish(model.clone(), &cancellation)?;
            let visual = store.publish(image_payload.clone(), &cancellation)?;
            let audio = store.publish(audio_payload.clone(), &cancellation)?;
            let case = fixture["source_generation"]["cases"]
                .as_array()
                .ok_or("generation cases absent")?
                .iter()
                .find(|case| {
                    case["class_type"] == name
                        && case["inputs"]["image"].is_null() == (media != "image")
                        && case["inputs"]["sampling_mode"]["sampling_mode"]
                            == if sampled { "on" } else { "off" }
                        && case["inputs"]["use_default_template"] == (media != "none")
                })
                .ok_or("source inference case absent")?;
            let mut inputs = generation_inputs(case, store.as_ref())?;
            inputs.insert(
                "clip".to_owned(),
                NativeValue::Handle {
                    value: clip.clone(),
                },
            );
            for field in ["image", "video", "audio"] {
                inputs.remove(field);
            }
            let (image, video, audio) = match media {
                "image" => (Some(visual.clone()), None, None),
                "video" => (None, Some(visual.clone()), None),
                "audio" => (None, None, Some(audio)),
                _ => (None, None, None),
            };
            for (field, handle) in [("image", &image), ("video", &video), ("audio", &audio)] {
                if let Some(handle) = handle {
                    inputs.insert(
                        field.to_owned(),
                        NativeValue::Handle {
                            value: handle.clone(),
                        },
                    );
                }
            }
            inputs.insert(
                "max_length".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Integer(1),
                },
            );
            if sampled {
                let Some(NativeValue::PreservedUnknown { value, .. }) =
                    inputs.get_mut("sampling_mode")
                else {
                    return Err("sampling fields absent".into());
                };
                value["seed"] = serde_json::json!(u64::MAX);
                value["top_k"] = serde_json::json!(1);
            }
            let expected_prompt = case["calls"]
                .as_array()
                .and_then(|calls| calls.first())
                .and_then(|call| call["prompt"].as_str())
                .ok_or("source prompt absent")?;
            let request = NativeMultimodalTextGenerationRequest::checked(
                clip,
                image,
                video,
                audio,
                expected_prompt,
                1,
                sampled,
                if sampled { 0.7_f32 } else { 1.0_f32 }.to_bits(),
                if sampled { 1 } else { 50 },
                if sampled { 0.95_f32 } else { 1.0_f32 }.to_bits(),
                if sampled { 0.05_f32 } else { 0.0_f32 }.to_bits(),
                if sampled { 1.05_f32 } else { 1.0_f32 }.to_bits(),
                0.0_f32.to_bits(),
                if sampled { u64::MAX } else { 0 },
                media != "none",
                true,
            )?;
            let node_id = NodeId("text".to_owned());
            let identity = NativeNodeServiceIdentity::checked(
                Uuid::from_u128(0x67812),
                attempt,
                node_id.clone(),
            )?;
            let scratch = authority.authorize_workspace(LIMIT)?;
            let compute = NativeNodeComputeSession::checked(
                identity.clone(),
                backend.clone(),
                StreamId::DEFAULT,
                &scratch,
            )?;
            let direct = NativeNodeContext::new_with_services(
                PromptId(Uuid::from_u128(0x67813)),
                attempt,
                node_id.clone(),
                cancellation.clone(),
                scratch,
                store.clone(),
                NativeNodeServices::checked(None, None, Some(compute))?.with_text_generation(
                    Arc::new(NativeRuntimeTextGenerationService::new(identity)),
                )?,
            )?;
            let mut descriptor = registry
                .descriptor(name)
                .ok_or("descriptor absent")?
                .clone();
            descriptor.output_node = true;
            flatten_sampling_inputs(&mut inputs)?;
            let mut plan = CompiledPlan {
                prompt_id: PromptId(Uuid::from_u128(0x67813)),
                client_id: None,
                prompt_number: None,
                extra_data: BTreeMap::new(),
                unknown: BTreeMap::new(),
                nodes: BTreeMap::from([(
                    node_id.clone(),
                    CompiledNode {
                        id: node_id.clone(),
                        class_type: name.to_owned(),
                        descriptor,
                        inputs: inputs
                            .into_iter()
                            .map(|(name, value)| (name, InputBinding::Literal { value }))
                            .collect(),
                        unknown: BTreeMap::new(),
                    },
                )]),
                topological_order: vec![node_id.clone()],
                static_required_nodes: std::collections::BTreeSet::from([node_id.clone()]),
                output_nodes: vec![node_id.clone()],
                provider_execution: None,
                persistence_unknown_fields: BTreeMap::new(),
            };
            plan.validate_integrity()?;
            let expected = direct
                .execute_multimodal_text_generation(&request)
                .map_err(|error| format!("{name}/{media} canonical generation: {error}"))?;
            assert_eq!(expected.generated_token_count(), 1, "{name}/{media}");
            store.revoke(request.clip(), &cancellation)?;
            assert!(matches!(
                direct.execute_multimodal_text_generation(&request),
                Err(NativeTextGenerationServiceError::Execution(_))
            ));
            let replacement = store.publish(model.clone(), &cancellation)?;
            plan.nodes
                .get_mut(&node_id)
                .ok_or("compiled text node absent")?
                .inputs
                .insert(
                    "clip".to_owned(),
                    InputBinding::Literal {
                        value: NativeValue::Handle { value: replacement },
                    },
                );
            plan.validate_integrity()?;

            let result = executor.execute_blocking(
                &plan,
                attempt,
                cancellation.clone(),
                0,
                authority.authorize_workspace(LIMIT)?,
            )?;
            assert_eq!(
                result.report.state,
                AttemptState::Succeeded,
                "{name}/{media}: {:?}",
                result.report.error
            );
            let actual = result
                .report
                .outputs
                .get(&node_id)
                .and_then(|outputs| outputs.first())
                .ok_or("engine text absent")?;
            assert_eq!(
                actual,
                &NativeValue::Primitive {
                    value: NativePrimitive::String(expected.into_text())
                },
                "{name}/{media}"
            );
            assert_eq!(model.digest_sha256(), model_digest);
        }
    }
    Ok(())
}

#[test]
fn task678_compiled_text_plans_preserve_dynamic_source_fields() -> Result<(), Box<dyn Error>> {
    use sim_runtime::{CompiledNode, CompiledPlan, InputBinding};
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let registry = generated_native_node_registry_projection(None)?;
    let generation = NativeHandleStoreGeneration::new()?;
    let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x67820)));
    for case in fixture["source_generation"]["cases"]
        .as_array()
        .ok_or("source cases absent")?
    {
        let name = case["class_type"].as_str().ok_or("source class absent")?;
        let mut inputs = generation_inputs(case, store.as_ref())?;
        flatten_sampling_inputs(&mut inputs)?;
        let mut descriptor = registry
            .descriptor(name)
            .ok_or("descriptor absent")?
            .clone();
        descriptor.output_node = true;
        let node_id = NodeId("text".to_owned());
        let mut plan = CompiledPlan {
            prompt_id: PromptId(Uuid::from_u128(0x67821)),
            client_id: None,
            prompt_number: None,
            extra_data: BTreeMap::new(),
            unknown: BTreeMap::new(),
            nodes: BTreeMap::from([(
                node_id.clone(),
                CompiledNode {
                    id: node_id.clone(),
                    class_type: name.to_owned(),
                    descriptor,
                    inputs: inputs
                        .into_iter()
                        .map(|(name, value)| (name, InputBinding::Literal { value }))
                        .collect(),
                    unknown: BTreeMap::new(),
                },
            )]),
            topological_order: vec![node_id.clone()],
            static_required_nodes: std::collections::BTreeSet::from([node_id.clone()]),
            output_nodes: vec![node_id],
            provider_execution: None,
            persistence_unknown_fields: BTreeMap::new(),
        };
        plan.validate_integrity()?;
        let restored: CompiledPlan = serde_json::from_str(&serde_json::to_string(&plan)?)?;
        restored.validate_integrity()?;
        let compiled = plan
            .nodes
            .get_mut(&NodeId("text".to_owned()))
            .ok_or("compiled node absent")?;
        let InputBinding::Literal { value: prompt } = compiled
            .inputs
            .get_mut("prompt")
            .ok_or("compiled prompt absent")?
        else {
            return Err("literal prompt absent".into());
        };
        *prompt = NativeValue::List {
            values: vec![prompt.clone(), prompt.clone()],
        };
        if let Some(InputBinding::Literal { value }) = compiled.inputs.get_mut("sampling_mode.seed")
        {
            *value = NativeValue::List {
                values: vec![primitive(&serde_json::json!(u64::MAX))?],
            };
        }
        plan.validate_integrity()?;
        let restored: CompiledPlan = serde_json::from_str(&serde_json::to_string(&plan)?)?;
        restored.validate_integrity()?;
        let mut mapped = generation_inputs(case, store.as_ref())?;
        let prompt = mapped.get_mut("prompt").ok_or("prompt absent")?;
        *prompt = NativeValue::List {
            values: vec![prompt.clone(), prompt.clone()],
        };
        let node = registry.node(name).ok_or("registered node absent")?;
        node.cache_change_token(&mapped)?;
        let mut invalid = mapped.clone();
        invalid.insert(
            "prompt".to_owned(),
            NativeValue::List {
                values: vec![
                    primitive(&serde_json::json!("valid"))?,
                    primitive(&serde_json::json!(7))?,
                ],
            },
        );
        assert!(node.cache_change_token(&invalid).is_err());
        let sampling = mapped.remove("sampling_mode").ok_or("sampling absent")?;
        let NativeValue::PreservedUnknown { value, .. } = sampling else {
            return Err("sampling fields absent".into());
        };
        let mut fields = value
            .as_object()
            .ok_or("sampling object absent")?
            .iter()
            .map(|(name, value)| Ok((name.clone(), primitive(value)?)))
            .collect::<Result<BTreeMap<_, _>, Box<dyn Error>>>()?;
        if fields.contains_key("seed") {
            fields.insert(
                "seed".to_owned(),
                NativeValue::List {
                    values: vec![
                        primitive(&serde_json::json!(0))?,
                        primitive(&serde_json::json!(u64::MAX))?,
                    ],
                },
            );
        }
        mapped.insert(
            "sampling_mode".to_owned(),
            sim_nodes::NativeStructuredValue::checked("COMFY_DYNAMICCOMBO_V3", fields.clone())?
                .into_runtime_value()?,
        );
        node.cache_change_token(&mapped)?;
        mapped.insert(
            "sampling_mode".to_owned(),
            sim_nodes::NativeStructuredValue::checked("COMFY_DYNAMICCOMBO_V3", fields)?
                .into_native_value(),
        );
        node.cache_change_token(&mapped)?;
    }
    Ok(())
}
fn flatten_sampling_inputs(
    inputs: &mut BTreeMap<String, NativeValue>,
) -> Result<(), Box<dyn Error>> {
    let Some(NativeValue::PreservedUnknown { type_name, value }) = inputs.remove("sampling_mode")
    else {
        return Err("sampling choice absent".into());
    };
    let mut fields = value.as_object().ok_or("sampling fields absent")?.clone();
    let selector = fields
        .remove("sampling_mode")
        .ok_or("sampling selector absent")?;
    inputs.insert(
        "sampling_mode".to_owned(),
        NativeValue::PreservedUnknown {
            type_name,
            value: selector,
        },
    );
    for (name, value) in fields {
        inputs.insert(format!("sampling_mode.{name}"), primitive(&value)?);
    }
    Ok(())
}

#[test]
fn task678_text_inputs_preserve_source_repeat_last_mapping() -> Result<(), Box<dyn Error>> {
    use sim_runtime::{CompiledNode, CompiledPlan, InputBinding, NativeImageExecutor};
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let executor = NativeImageExecutor::new_with_generated_registry(
        sim_types::ProfileId(Uuid::from_u128(0x67830)),
        BTreeMap::new(),
        true,
        Arc::new(backend),
    )?;
    let mut descriptor = registry
        .descriptor("TruncateText")
        .ok_or("descriptor absent")?
        .clone();
    descriptor.output_node = true;
    let node_id = NodeId("text".to_owned());
    let inputs = BTreeMap::from([
        (
            "texts".to_owned(),
            NativeValue::List {
                values: ["ABCDE", "é👩\u{200d}💻末", "last"]
                    .into_iter()
                    .map(|value| primitive(&serde_json::json!(value)))
                    .collect::<Result<Vec<_>, _>>()?,
            },
        ),
        (
            "max_length".to_owned(),
            NativeValue::List {
                values: [1, 3]
                    .into_iter()
                    .map(|value| primitive(&serde_json::json!(value)))
                    .collect::<Result<Vec<_>, _>>()?,
            },
        ),
    ]);
    let plan = CompiledPlan {
        prompt_id: PromptId(Uuid::from_u128(0x67831)),
        client_id: None,
        prompt_number: None,
        extra_data: BTreeMap::new(),
        unknown: BTreeMap::new(),
        nodes: BTreeMap::from([(
            node_id.clone(),
            CompiledNode {
                id: node_id.clone(),
                class_type: "TruncateText".to_owned(),
                descriptor,
                inputs: inputs
                    .into_iter()
                    .map(|(name, value)| (name, InputBinding::Literal { value }))
                    .collect(),
                unknown: BTreeMap::new(),
            },
        )]),
        topological_order: vec![node_id.clone()],
        static_required_nodes: std::collections::BTreeSet::from([node_id.clone()]),
        output_nodes: vec![node_id.clone()],
        provider_execution: None,
        persistence_unknown_fields: BTreeMap::new(),
    };
    plan.validate_integrity()?;
    let result = executor.execute_blocking(
        &plan,
        AttemptId(Uuid::from_u128(0x67832)),
        CancellationToken::default(),
        0,
        authority.authorize_workspace(1024 * 1024)?,
    )?;
    assert_eq!(
        result.report.state,
        sim_runtime::AttemptState::Succeeded,
        "{:?}",
        result.report.error
    );
    assert_eq!(
        result.report.outputs.get(&node_id),
        Some(&vec![NativeValue::List {
            values: ["A", "é👩\u{200d}", "las"]
                .into_iter()
                .map(|value| primitive(&serde_json::json!(value)))
                .collect::<Result<Vec<_>, _>>()?
        }])
    );
    Ok(())
}
