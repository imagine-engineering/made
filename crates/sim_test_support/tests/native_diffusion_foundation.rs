use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_media::{PngLimits, encode_png_frame};
use sim_model::clip::{
    NativeClipProfile, NativeClipResource, NativeTokenizer, native_clip_transform_fixture,
};
use sim_model::generated_native_diffusion::{Sd1Tokenizer, empty_sd15_latent, encode_sd15_prompt};
use sim_model::model_family::{
    NativeFamilyDenoiserContext, NativeFamilyDenoiserInvocation, NativeFamilyModelResource,
};
use sim_model::{
    AdapterTensor, LatentFormatIdentity, ModelFamilyError, NativeFamilyBuildOptions,
    NativeFamilyModel, PatchGraph, PatchPayload, PatchTensor, PatchValueTransform,
    SemanticPatchOperation, build_model_family,
    conditioning::{
        ConditioningEntry, ConditioningEntryOptions, ConditioningIdentity, ConditioningSet,
        ConditioningValue,
    },
    generated_auraflow_sim_model_0064 as aura, generated_qwenimage_sim_model_0113 as qwen,
    map_model_weights,
    patches::{
        ModelStateExportKind, PatchAdapterError, PatchKeyMap, PatchKeyMapping, PatchMergeKind,
        PatchMergeMode, PatchMergeRatio, PatchMergeRule, StateKeyMatch, StateKeySelector,
        export_safetensors_exact,
    },
};
use sim_model::{ModelTokenizerDescriptor, NativeModelPayload};
use sim_nodes::{NativeStoredModelPayload, NativeStoredPayload, NativeValue};
use sim_runtime::{
    AttemptState, InputBinding, NativeDiffusionBundle, NativeDiffusionProvider,
    NativeImageExecutor, Sd15GuidanceAdapter, compile_native_diffusion_workflow,
};
use sim_sampler::generated_native_diffusion::interpret_prediction_for_profile;
use sim_sampler::generated_native_diffusion::{
    checked_native_diffusion_plan, checked_native_diffusion_plan_for_profile, normal_noise,
    normal_sigmas, normal_sigmas_for_profile, sample_euler, scale_initial_noise, scale_model_input,
    sd15_interpret_prediction, sd15_model_time,
};
use sim_sampler::{
    ExecutionHookGraph, ExecutionHookKind, ExecutionHookOperation, GuidanceOptions,
    HypernetworkActivation, HypernetworkDimensionPatch, HypernetworkLinearLayer, HypernetworkPatch,
    NativeConditioningPayload, NativeDiffusionPayload, NativeFamilyGuidanceDenoiser,
    NativeGuidanceModifier, NativeGuidanceModifierChain, NativeGuiderKind, NativeGuiderPayload,
    NativeLatentTransform, NativePredictionKind, NativeSamplingPolicy, NativeSamplingPolicyKind,
    NoiseRequest, SamplingProfileReplacement, execute_guidance_with_hook_graph, profile_for_model,
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    NativeLatentBundle, NativeLatentMetadata, RngCheckpoint, StreamId, Tensor,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use sim_test_support::{NativeDiffusionFixture, NativeDiffusionFixtureError};
use sim_types::{AttemptId, DeviceKind, NodeId, ProfileId, PromptId};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
use uuid::Uuid;

const MEMORY_LIMIT: u64 = 2 * 1024 * 1024 * 1024;
const SEED: u64 = 0x0123_4567_89ab_cdef;
const FIXTURE_PROMPT_ID: &str = "53494d00-0000-0000-0000-000000003702";
const FIXTURE_KSAMPLER_NODE_ID: &str = "5";
const FAMILY_ARTIFACT_DIGEST: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[test]
fn task406_void_sampler_runs_through_production_ksampler_without_euler_aliasing()
-> Result<(), Box<dyn std::error::Error>> {
    let source_reference = NativeDiffusionFixture::checked_in().source_void_sampler_reference()?;
    let mut workflow: Value =
        serde_json::from_slice(include_bytes!("../fixtures/native_diffusion/workflow.json"))?;
    let sampler = workflow
        .get_mut("nodes")
        .and_then(Value::as_array_mut)
        .and_then(|nodes| {
            nodes
                .iter_mut()
                .find(|node| node.get("id").and_then(Value::as_str) == Some("5"))
        })
        .ok_or("native diffusion workflow omitted KSampler")?;
    let sampler_widgets = sampler
        .get_mut("widgets_values")
        .and_then(Value::as_array_mut)
        .ok_or("native diffusion KSampler omitted widget values")?;
    *sampler_widgets
        .get_mut(3)
        .ok_or("native diffusion KSampler omitted sampler_name")? = json!("void_ddim");

    let provider: Arc<dyn NativeDiffusionProvider> = Arc::new(NativeDiffusionFixture::checked_in());
    let mut plan = compile_native_diffusion_workflow(
        &serde_json::to_vec(&workflow)?,
        &BTreeSet::new(),
        provider.clone(),
    )?;
    // The pinned source noise is bound to this prompt identity.
    plan.prompt_id = PromptId(Uuid::parse_str(FIXTURE_PROMPT_ID)?);
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let backend = Arc::new(backend);
    let executor = NativeImageExecutor::new_with_diffusion_provider(
        ProfileId(Uuid::from_u128(0x406_02)),
        BTreeMap::new(),
        true,
        backend.clone(),
        provider,
    )?;
    let result = executor.execute_blocking(
        &plan,
        AttemptId(Uuid::from_u128(0x406_03)),
        CancellationToken::default(),
        0,
        workspace_authority.authorize_workspace(MEMORY_LIMIT)?,
    )?;
    assert_eq!(
        result.report.state,
        AttemptState::Succeeded,
        "production VOID KSampler failed: {:?}",
        result.report.error
    );
    let sampler_ui = result
        .report
        .ui_outputs
        .get(&sim_types::NodeId("5".to_owned()))
        .ok_or("production VOID KSampler omitted UI evidence")?;
    assert_eq!(
        sampler_ui.get("sampler_algorithm").and_then(Value::as_str),
        Some("void_ddim")
    );
    assert_eq!(
        sampler_ui.get("initial_sha256"),
        sampler_ui.get("noise_sha256"),
        "VOID must enter alpha-space updates from raw noise"
    );
    assert_eq!(
        sampler_ui.get("sampler_progress").and_then(Value::as_str),
        Some("fixed")
    );
    assert_eq!(
        sampler_ui.get("noise_sha256"),
        source_reference.get("noise_sha256"),
        "production VOID must consume the exact pinned source noise input"
    );
    assert_eq!(
        sampler_ui.get("latent_sha256"),
        source_reference.get("latent_sha256"),
        "production VOID must match independently source-verified alpha-space checkpoints"
    );
    assert_eq!(
        sampler_ui.get("denoiser_sha256"),
        source_reference.get("denoiser_sha256"),
        "production VOID guidance must match the independently source-verified denoiser"
    );

    let sampler = workflow
        .get_mut("nodes")
        .and_then(Value::as_array_mut)
        .and_then(|nodes| {
            nodes
                .iter_mut()
                .find(|node| node.get("id").and_then(Value::as_str) == Some("5"))
        })
        .ok_or("native diffusion workflow omitted KSampler")?;
    let sampler_widgets = sampler
        .get_mut("widgets_values")
        .and_then(Value::as_array_mut)
        .ok_or("native diffusion KSampler omitted widget values")?;
    *sampler_widgets
        .get_mut(3)
        .ok_or("native diffusion KSampler omitted sampler_name")? = json!("lcm_upscale");
    let mut lcm_plan = compile_native_diffusion_workflow(
        &serde_json::to_vec(&workflow)?,
        &BTreeSet::new(),
        Arc::new(NativeDiffusionFixture::checked_in()),
    )?;
    lcm_plan.prompt_id = PromptId(Uuid::from_u128(0x406_04));
    let family_cancellation = CancellationToken::default();
    let family_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(MEMORY_LIMIT)?,
        &family_cancellation,
    );
    let family = family_fixture_model(&backend, &family_context, false, true)?;
    let positive_identity = family_identity(&family, "task406-positive")?;
    let negative_identity = family_identity(&family, "task406-negative")?;
    let positive = family_conditioning(
        positive_identity,
        tensor_from_f32(
            &backend,
            &[1, 2, 2_048],
            &family_patterned_values("positive", 4_096),
            &family_context,
        )?,
        "positive",
        &family_cancellation,
    )?;
    let negative = family_conditioning(
        negative_identity,
        tensor_from_f32(
            &backend,
            &[1, 2, 2_048],
            &family_patterned_values("negative.bias", 4_096),
            &family_context,
        )?,
        "negative",
        &family_cancellation,
    )?;
    let diffusion = Arc::new(family_diffusion(family, &backend, &family_context)?);
    let latent = Arc::new(NativeLatentBundle::single(
        tensor_from_f32(
            &backend,
            &[1, 4, 3, 3],
            &(0..36)
                .map(|index| (index as f32 - 18.0) * 0.025)
                .collect::<Vec<_>>(),
            &family_context,
        )?,
        None,
        None,
        NativeLatentMetadata::default(),
        &family_context,
    )?);
    let lcm_attempt = AttemptId(Uuid::from_u128(0x406_05));
    let handle_store = executor
        .handle_store_generation_for_test()
        .handle_store_for_attempt(lcm_attempt);
    let cancellation = CancellationToken::default();
    let model_handle = handle_store.publish(
        NativeStoredPayload::Model(Arc::new(NativeStoredModelPayload::native_diffusion(
            diffusion,
        )?)),
        &cancellation,
    )?;
    let positive_handle =
        handle_store.publish(NativeStoredPayload::Conditioning(positive), &cancellation)?;
    let negative_handle =
        handle_store.publish(NativeStoredPayload::Conditioning(negative), &cancellation)?;
    let latent_handle = handle_store.publish(NativeStoredPayload::Latent(latent), &cancellation)?;
    let sampler_id = sim_types::NodeId("5".to_owned());
    let mut lcm_node = lcm_plan
        .nodes
        .remove(&sampler_id)
        .ok_or("compiled workflow omitted KSampler")?;
    lcm_node.descriptor.output_node = true;
    for (name, handle) in [
        ("model", model_handle),
        ("positive", positive_handle),
        ("negative", negative_handle),
        ("latent_image", latent_handle),
    ] {
        lcm_node.inputs.insert(
            name.to_owned(),
            InputBinding::Literal {
                value: NativeValue::Handle { value: handle },
            },
        );
    }
    lcm_plan.nodes = BTreeMap::from([(sampler_id.clone(), lcm_node)]);
    lcm_plan.topological_order = vec![sampler_id.clone()];
    lcm_plan.static_required_nodes = BTreeSet::from([sampler_id.clone()]);
    lcm_plan.output_nodes = vec![sampler_id.clone()];
    lcm_plan.provider_execution = None;
    lcm_plan.validate_integrity()?;
    let lcm = executor.execute_blocking(
        &lcm_plan,
        lcm_attempt,
        cancellation,
        0,
        workspace_authority.authorize_workspace(MEMORY_LIMIT)?,
    )?;
    assert_eq!(
        lcm.report.state,
        AttemptState::Succeeded,
        "production multi-step LCM-upscale KSampler failed: {:?}",
        lcm.report.error
    );
    let lcm_ui = lcm
        .report
        .ui_outputs
        .get(&sampler_id)
        .ok_or("production LCM-upscale KSampler omitted UI evidence")?;
    assert_eq!(
        lcm_ui.get("sampler_algorithm").and_then(Value::as_str),
        Some("lcm_upscale")
    );
    assert_eq!(lcm_ui.get("final_shape"), Some(&json!([1, 4, 6, 6])));
    assert_eq!(
        lcm_ui
            .get("latent_sha256")
            .and_then(Value::as_array)
            .map(Vec::len),
        Some(5)
    );
    Ok(())
}

fn source_golden_safetensors(header: &str, values: &[f32]) -> Vec<u8> {
    let mut header = header.as_bytes().to_vec();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut output = Vec::with_capacity(8 + header.len() + std::mem::size_of_val(values));
    output.extend_from_slice(&(header.len() as u64).to_le_bytes());
    output.extend_from_slice(&header);
    for value in values {
        output.extend_from_slice(&value.to_le_bytes());
    }
    output
}

fn family_patterned_values(key: &str, elements: usize) -> Vec<f32> {
    let normalization = key.contains("norm_") || key.ends_with("txt_norm.weight");
    let bias = key.ends_with(".bias");
    let aura_mlp =
        key.contains(".mlpC.") || key.contains(".mlpX.") || key.contains("single_layers.0.mlp.");
    let scale = if aura_mlp { 0.5 } else { 0.000_75 };
    (0..elements)
        .map(|index| {
            if normalization {
                0.95 + (index % 7) as f32 * 0.01
            } else if bias {
                ((index % 11) as f32 - 5.0) * 0.002
            } else {
                ((index % 17) as f32 - 8.0) * scale
            }
        })
        .collect()
}

fn family_tensor(
    backend: &CpuBackend,
    shape: &[u64],
    key: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let elements = shape.iter().try_fold(1_usize, |total, dimension| {
        total.checked_mul(usize::try_from(*dimension).ok()?)
    });
    let elements = elements.ok_or("family fixture tensor shape overflow")?;
    Ok(tensor_from_f32(
        backend,
        shape,
        &family_patterned_values(key, elements),
        context,
    )?)
}

fn family_weight_shape(
    key: &str,
    qwen_model: bool,
) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
    let shape = if qwen_model {
        match key {
            "native.img_in.weight" => vec![128, 64],
            "native.img_in.bias"
            | "native.txt_in.bias"
            | "native.time_text_embed.timestep_embedder.linear_1.bias"
            | "native.time_text_embed.timestep_embedder.linear_2.bias" => vec![128],
            "native.txt_norm.weight" => vec![3_584],
            "native.txt_in.weight" => vec![128, 3_584],
            "native.time_text_embed.timestep_embedder.linear_1.weight" => vec![128, 256],
            "native.time_text_embed.timestep_embedder.linear_2.weight" => vec![128, 128],
            key if key.ends_with("img_mod.1.weight") || key.ends_with("txt_mod.1.weight") => {
                vec![768, 128]
            }
            key if key.ends_with("img_mod.1.bias") || key.ends_with("txt_mod.1.bias") => vec![768],
            key if key.contains("attn.norm_") => vec![128],
            key if key.contains("attn.") && key.ends_with(".weight") => vec![128, 128],
            key if key.contains("attn.") && key.ends_with(".bias") => vec![128],
            key if key.ends_with("mlp.net.0.proj.weight") => vec![512, 128],
            key if key.ends_with("mlp.net.0.proj.bias") => vec![512],
            key if key.ends_with("mlp.net.2.weight") => vec![128, 512],
            key if key.ends_with("mlp.net.2.bias") => vec![128],
            "native.norm_out.linear.weight" => vec![256, 128],
            "native.norm_out.linear.bias" => vec![256],
            "native.proj_out.weight" => vec![64, 128],
            "native.proj_out.bias" => vec![64],
            _ => return Err(format!("missing Qwen fixture shape for {key}").into()),
        }
    } else {
        match key {
            "native.init_x_linear.weight" => vec![2, 16],
            "native.init_x_linear.bias" => vec![2],
            "native.positional_encoding" => vec![1, 16, 2],
            "native.register_tokens" => vec![1, 8, 2],
            "native.cond_seq_linear.weight" => vec![2, 2_048],
            "native.t_embedder.mlp.0.weight" => vec![2, 256],
            "native.t_embedder.mlp.0.bias" | "native.t_embedder.mlp.2.bias" => vec![2],
            "native.t_embedder.mlp.2.weight" => vec![2, 2],
            "native.double_layers.0.modC.1.weight"
            | "native.double_layers.0.modX.1.weight"
            | "native.single_layers.0.modCX.1.weight" => vec![12, 2],
            "native.double_layers.0.attn.w1q.weight"
            | "native.double_layers.0.attn.w1k.weight"
            | "native.double_layers.0.attn.w1v.weight"
            | "native.double_layers.0.attn.w1o.weight"
            | "native.double_layers.0.attn.w2q.weight"
            | "native.double_layers.0.attn.w2k.weight"
            | "native.double_layers.0.attn.w2v.weight"
            | "native.double_layers.0.attn.w2o.weight"
            | "native.single_layers.0.attn.w1q.weight"
            | "native.single_layers.0.attn.w1k.weight"
            | "native.single_layers.0.attn.w1v.weight"
            | "native.single_layers.0.attn.w1o.weight" => vec![2, 2],
            key if key.ends_with("c_fc1.weight") || key.ends_with("c_fc2.weight") => vec![256, 2],
            key if key.ends_with("c_proj.weight") => vec![2, 256],
            "native.modF.1.weight" => vec![4, 2],
            "native.final_linear.weight" => vec![16, 2],
            _ => return Err(format!("missing Aura fixture shape for {key}").into()),
        }
    };
    Ok(shape)
}

fn family_fixture_model(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    qwen_model: bool,
    source_bound: bool,
) -> Result<NativeFamilyModel, Box<dyn std::error::Error>> {
    if source_bound {
        return task377::bound_model(backend, context, qwen_model, MEMORY_LIMIT);
    }
    let (definition, keys): (_, &[&str]) = if qwen_model {
        (&qwen::MODEL_FAMILY, qwen::DENOISER_INVOCATION_REQUIRED_KEYS)
    } else {
        (&aura::MODEL_FAMILY, aura::DENOISER_INVOCATION_REQUIRED_KEYS)
    };
    let mut source = BTreeMap::new();
    for key in keys {
        let shape = family_weight_shape(key, qwen_model)?;
        source.insert(
            key.replacen("native.", "model.diffusion_model.", 1),
            family_tensor(backend, &shape, key, context)?,
        );
    }
    if qwen_model {
        for key in [
            "native.__reference_method__",
            "native.__additional_timestep_condition__",
        ] {
            source.insert(
                key.replacen("native.", "model.diffusion_model.", 1),
                tensor_from_f32(backend, &[1], &[0.0], context)?,
            );
        }
    }
    let options = NativeFamilyBuildOptions {
        dtype: DType::F32,
        device: DeviceKind::Cpu,
        activation_elements: 1,
        memory_budget_bytes: MEMORY_LIMIT,
        allow_unexpected_weights: true,
    };
    Ok(build_model_family(
        definition,
        map_model_weights(definition, FAMILY_ARTIFACT_DIGEST, source)?,
        options,
    )?)
}

fn family_identity(
    model: &NativeFamilyModel,
    namespace: &str,
) -> Result<ConditioningIdentity, Box<dyn std::error::Error>> {
    Ok(ConditioningIdentity::new(
        namespace,
        model.identity()?,
        LatentFormatIdentity::new(
            model.profile().latent_feature_id,
            model.profile().latent_identifier,
        )?,
    )?)
}

fn family_conditioning(
    identity: ConditioningIdentity,
    tensor: Tensor,
    name: &str,
    cancellation: &CancellationToken,
) -> Result<Arc<ConditioningSet>, Box<dyn std::error::Error>> {
    Ok(Arc::new(ConditioningSet::checked(
        identity,
        vec![ConditioningEntry::checked(
            name,
            ConditioningValue::cross_attention(tensor)?,
            ConditioningEntryOptions::default(),
        )?],
        cancellation,
    )?))
}

fn family_diffusion(
    model: NativeFamilyModel,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<NativeDiffusionPayload, Box<dyn std::error::Error>> {
    let model = Arc::new(model);
    let graph = Arc::new(PatchGraph::checked_semantic(
        model.weights().base_artifact_digest(),
        Vec::new(),
    )?);
    let identity = family_identity(&model, "native-family-model")?;
    let resource = Arc::new(NativeFamilyModelResource::materialize(
        model,
        graph.clone(),
        identity,
        backend,
        context,
    )?);
    let payload = Arc::new(NativeModelPayload::native_family_model(resource)?);
    let conditioning = Arc::new(NativeConditioningPayload::checked_family(
        &payload, graph, None,
    )?);
    Ok(NativeDiffusionPayload::model(payload, conditioning)?)
}

fn tensor_f32_sha256(
    backend: &CpuBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut digest = Sha256::new();
    for value in tensor_to_f32(backend, tensor, context)?.iter() {
        digest.update(value.to_le_bytes());
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn family_oracle() -> Result<Value, Box<dyn std::error::Error>> {
    Ok(serde_json::from_str(include_str!(
        "../fixtures/models/native-family-model-resource-foundation/provenance.json"
    ))?)
}

fn decode_hex_nibble(value: u8) -> Result<u8, Box<dyn std::error::Error>> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err(format!("invalid family oracle hex nibble {value}").into()),
    }
}

fn family_oracle_raw(
    oracle: &Value,
    family: &str,
    branch: &str,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if family == "aura" {
        let source: Value = serde_json::from_str(include_str!(
            "../fixtures/models/auraflow-comfy-model-0064/guidance-source-oracle.json"
        ))?;
        assert_eq!(
            source["generator_sha256"],
            format!(
                "{:x}",
                Sha256::digest(include_bytes!(
                    "../fixtures/models/auraflow-comfy-model-0064/generate_guidance_source_oracle.py"
                ))
            )
        );
        assert_eq!(
            source["model_generator_sha256"],
            format!(
                "{:x}",
                Sha256::digest(include_bytes!(
                    "../fixtures/models/auraflow-comfy-model-0064/generate_source_oracle.py"
                ))
            )
        );
        assert_eq!(source["absolute_tolerance"], json!(1e-6));
        return source["branches"][branch]
            .as_array()
            .ok_or_else(|| format!("source AuraFlow branch {branch} is missing").into())
            .and_then(|values| {
                values
                    .iter()
                    .map(|value| {
                        value
                            .as_f64()
                            .map(|value| value as f32)
                            .ok_or_else(|| "source AuraFlow value is invalid".into())
                    })
                    .collect()
            });
    }
    let encoded = oracle
        .pointer(&format!("/families/{family}/{branch}_raw_f32_le_hex"))
        .and_then(Value::as_str)
        .ok_or_else(|| format!("family oracle is missing {family}.{branch} raw values"))?;
    if encoded.len() % 8 != 0 {
        return Err("family oracle raw values are not complete F32 words".into());
    }
    encoded
        .as_bytes()
        .chunks_exact(8)
        .map(|word| {
            let mut bytes = [0_u8; 4];
            for (index, pair) in word.chunks_exact(2).enumerate() {
                bytes[index] = (decode_hex_nibble(pair[0])? << 4) | decode_hex_nibble(pair[1])?;
            }
            Ok(f32::from_le_bytes(bytes))
        })
        .collect()
}

fn assert_tensor_close(
    backend: &CpuBackend,
    actual: &Tensor,
    expected: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let actual = tensor_to_f32(backend, actual, context)?;
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*actual - *expected).abs() <= 1.0e-6,
            "family resource oracle diverged at {index}: {actual} != {expected}"
        );
    }
    Ok(())
}

mod task377 {
    include!("native_family_execution_projection.rs");

    pub(super) fn bound_model(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        qwen_model: bool,
        budget: u64,
    ) -> Result<NativeFamilyModel, Box<dyn Error>> {
        let registry =
            ModelFamilyRegistry::checked_registrations(GENERATED_MODEL_FAMILY_REGISTRATIONS)?;
        let probe = if qwen_model {
            qwen_probe()?
        } else {
            aura_probe()?
        };
        bind(
            &registry,
            &probe,
            projection_state(backend, context, qwen_model)?,
            context,
            budget,
        )
        .map_err(Into::into)
    }

    pub(super) fn assert_raw_oracle(
        model: &NativeFamilyModel,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        qwen_model: bool,
    ) -> Result<(), Box<dyn Error>> {
        task376::assert_bound_model_oracle(model, backend, context, qwen_model)
    }
}

#[test]
fn family_model_resource_transport_guidance_and_reconstruction_are_canonical()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(MEMORY_LIMIT)?,
        &cancellation,
    );

    let legacy = Arc::new(family_fixture_model(&backend, &context, false, false)?);
    let legacy_patch = Arc::new(PatchGraph::checked_semantic(
        FAMILY_ARTIFACT_DIGEST,
        Vec::new(),
    )?);
    assert!(matches!(
        NativeFamilyModelResource::materialize(
            legacy.clone(),
            legacy_patch,
            family_identity(&legacy, "native-family-model")?,
            &backend,
            &context,
        ),
        Err(ModelFamilyError::FamilyResourceSourceBindingMissing)
    ));

    for qwen_model in [false, true] {
        let base = Arc::new(family_fixture_model(&backend, &context, qwen_model, true)?);
        let patch_graph = Arc::new(PatchGraph::checked_semantic(
            FAMILY_ARTIFACT_DIGEST,
            Vec::new(),
        )?);
        let identity = family_identity(&base, "native-family-model")?;
        let resource = Arc::new(NativeFamilyModelResource::materialize(
            base,
            patch_graph.clone(),
            identity,
            &backend,
            &context,
        )?);
        let reconstructed = resource.reconstruct_in_memory(&backend, &context)?;
        assert_eq!(
            reconstructed.semantic_digest_sha256(),
            resource.semantic_digest_sha256()
        );
        assert_eq!(reconstructed.resident_bytes()?, resource.resident_bytes()?);
        assert!(!Arc::ptr_eq(reconstructed.model(), resource.model()));

        let model = Arc::new(NativeModelPayload::native_family_model(resource.clone())?);
        let execution = Arc::new(NativeConditioningPayload::checked_family(
            &model,
            patch_graph,
            None,
        )?);
        let diffusion = NativeDiffusionPayload::model(model.clone(), execution)?;
        diffusion.validate()?;
        assert!(
            diffusion
                .model_resources()
                .is_some_and(|(stored, _)| Arc::ptr_eq(stored, &model))
        );
        assert!(model.resident_parts()?.resident_bytes()? > 0);
        assert!(diffusion.resident_parts()?.resident_bytes()? > 0);
    }

    let base = Arc::new(family_fixture_model(&backend, &context, false, true)?);
    let patch_graph = Arc::new(PatchGraph::checked_semantic(
        FAMILY_ARTIFACT_DIGEST,
        Vec::new(),
    )?);
    let resource = Arc::new(NativeFamilyModelResource::materialize(
        base.clone(),
        patch_graph,
        family_identity(&base, "native-family-model")?,
        &backend,
        &context,
    )?);
    let model = Arc::new(NativeModelPayload::native_family_model(resource)?);
    let latent = tensor_from_f32(
        &backend,
        &[1, 4, 3, 3],
        &(0..36)
            .map(|index| (index as f32 - 18.0) * 0.025)
            .collect::<Vec<_>>(),
        &context,
    )?;
    let positive = family_conditioning(
        family_identity(&base, "aura-positive")?,
        tensor_from_f32(
            &backend,
            &[1, 2, 2_048],
            &family_patterned_values("positive", 2 * 2_048),
            &context,
        )?,
        "positive",
        &cancellation,
    )?;
    let negative = family_conditioning(
        family_identity(&base, "aura-negative")?,
        tensor_from_f32(
            &backend,
            &[1, 2, 2_048],
            &family_patterned_values("negative.bias", 2 * 2_048),
            &context,
        )?,
        "negative",
        &cancellation,
    )?;
    let profile = profile_for_model(&model)?;
    let plan =
        checked_native_diffusion_plan_for_profile(&profile, "euler", "normal", 7, 4, 2.0, 1.0)?;
    let guider = NativeGuiderPayload::cfg(model.clone(), positive.clone(), negative.clone(), 2.0)?;
    let mut denoiser = NativeFamilyGuidanceDenoiser::checked(&model, &backend)?;
    let guided = guider.execute(
        &backend,
        &latent,
        0.5,
        &profile,
        &plan,
        GuidanceOptions::default(),
        &mut denoiser,
        &mut [],
        &context,
    )?;
    assert_eq!(guided.guided().descriptor(), latent.descriptor());
    assert!(
        tensor_to_f32(&backend, guided.guided(), &context)?
            .iter()
            .all(|value| value.is_finite())
    );
    let basic = NativeGuiderPayload::basic(model.clone(), positive.clone())?;
    let dual_cfg = NativeGuiderPayload::dual_cfg(
        model.clone(),
        positive.clone(),
        negative.clone(),
        negative.clone(),
        3.0,
        2.0,
        true,
    )?;
    let dual_model = NativeGuiderPayload::dual_model(
        model.clone(),
        model.clone(),
        positive.clone(),
        negative.clone(),
        2.0,
    )?;
    assert_eq!(basic.kind(), NativeGuiderKind::Basic);
    assert_eq!(guider.kind(), NativeGuiderKind::Cfg);
    assert_eq!(dual_cfg.kind(), NativeGuiderKind::DualCfg);
    assert_eq!(dual_model.kind(), NativeGuiderKind::DualModel);
    let guider_digests = [
        basic.semantic_digest_sha256(),
        guider.semantic_digest_sha256(),
        dual_cfg.semantic_digest_sha256(),
        dual_model.semantic_digest_sha256(),
    ]
    .into_iter()
    .collect::<BTreeSet<_>>();
    assert_eq!(guider_digests.len(), 4);
    assert!(
        [basic, guider.clone(), dual_cfg, dual_model]
            .into_iter()
            .all(|payload| payload.validate().is_ok() && payload.resident_bytes() > 0)
    );

    let hypernetwork_layer = HypernetworkLinearLayer {
        input_dimension: 2,
        output_dimension: 2,
        weights: vec![0.125, 0.0, 0.0, -0.125],
        bias: vec![0.25, -0.25],
        activation: HypernetworkActivation::Tanh,
        layer_norm_scale: None,
        layer_norm_bias: None,
        dropout_probability: 0.0,
    };
    let hook_graph = ExecutionHookGraph::checked(
        93,
        vec![
            ExecutionHookOperation {
                identifier: "family-attention".to_owned(),
                order: 1,
                kind: ExecutionHookKind::Attention,
                scale: 1.125,
                bias: -0.0625,
                start_percent: 0.0,
                end_percent: 1.0,
                rng_address: 11,
                rng_amplitude: 0.0,
                hypernetwork: None,
                context_window: None,
                sampling_profile: None,
            },
            ExecutionHookOperation {
                identifier: "family-hypernetwork".to_owned(),
                order: 2,
                kind: ExecutionHookKind::Hypernetwork,
                scale: 1.0,
                bias: 0.0,
                start_percent: 0.0,
                end_percent: 1.0,
                rng_address: 12,
                rng_amplitude: 0.0,
                hypernetwork: Some(HypernetworkPatch {
                    strength: 0.5,
                    dimensions: BTreeMap::from([(
                        2,
                        HypernetworkDimensionPatch {
                            key_layers: vec![hypernetwork_layer.clone()],
                            value_layers: vec![hypernetwork_layer],
                        },
                    )]),
                }),
                context_window: None,
                sampling_profile: None,
            },
            ExecutionHookOperation {
                identifier: "family-output-block".to_owned(),
                order: 3,
                kind: ExecutionHookKind::OutputBlock,
                scale: 0.875,
                bias: 0.03125,
                start_percent: 0.0,
                end_percent: 1.0,
                rng_address: 13,
                rng_amplitude: 0.0,
                hypernetwork: None,
                context_window: None,
                sampling_profile: None,
            },
        ],
    )?;
    let mut hooked_denoiser = NativeFamilyGuidanceDenoiser::checked(&model, &backend)?;
    let hooked = execute_guidance_with_hook_graph(
        &backend,
        &latent,
        0.5,
        &profile,
        &plan,
        &positive,
        &negative,
        GuidanceOptions::default(),
        &mut hooked_denoiser,
        &hook_graph,
        &context,
    )?;
    assert_ne!(
        tensor_to_f32(&backend, hooked.guided(), &context)?.to_vec(),
        tensor_to_f32(&backend, guided.guided(), &context)?.to_vec(),
        "serialized hooks must reach the retained model's actual attention K/V and output sites"
    );
    let incompatible_rope = ExecutionHookGraph::checked(
        93,
        vec![ExecutionHookOperation {
            identifier: "aura-has-no-rotary-site".to_owned(),
            order: 1,
            kind: ExecutionHookKind::Rope,
            scale: 1.0,
            bias: 0.0,
            start_percent: 0.0,
            end_percent: 1.0,
            rng_address: 14,
            rng_amplitude: 0.0,
            hypernetwork: None,
            context_window: None,
            sampling_profile: None,
        }],
    )?;
    let mut incompatible_denoiser = NativeFamilyGuidanceDenoiser::checked(&model, &backend)?;
    assert!(
        execute_guidance_with_hook_graph(
            &backend,
            &latent,
            0.5,
            &profile,
            &plan,
            &positive,
            &negative,
            GuidanceOptions::default(),
            &mut incompatible_denoiser,
            &incompatible_rope,
            &context,
        )
        .is_err()
    );
    let replacement_graph = ExecutionHookGraph::checked(
        93,
        vec![ExecutionHookOperation {
            identifier: "family-sampling-replacement".to_owned(),
            order: 1,
            kind: ExecutionHookKind::SamplingProfile,
            scale: 1.0,
            bias: 0.0,
            start_percent: 0.0,
            end_percent: 1.0,
            rng_address: 15,
            rng_amplitude: 0.0,
            hypernetwork: None,
            context_window: None,
            sampling_profile: Some(SamplingProfileReplacement::DiscreteVPrediction),
        }],
    )?;
    let mut replacement_denoiser = NativeFamilyGuidanceDenoiser::checked(&model, &backend)?;
    let replacement = execute_guidance_with_hook_graph(
        &backend,
        &latent,
        0.5,
        &profile,
        &plan,
        &positive,
        &negative,
        GuidanceOptions::default(),
        &mut replacement_denoiser,
        &replacement_graph,
        &context,
    )?;
    assert_ne!(
        tensor_to_f32(&backend, replacement.guided(), &context)?.to_vec(),
        tensor_to_f32(&backend, guided.guided(), &context)?.to_vec(),
        "sampling replacement must change model-input, time, and prediction equations before CFG"
    );

    let attached_policy = Arc::new(NativeSamplingPolicy::checked_with_latent(
        NativeSamplingPolicyKind::ContinuousEdm {
            prediction: NativePredictionKind::Edm,
            sigma_min: 0.002,
            sigma_max: 120.0,
            sigma_data: 0.5,
        },
        NativeLatentTransform::PlaygroundV25,
    )?);
    let attached_modifiers = Arc::new(NativeGuidanceModifierChain::checked(vec![
        NativeGuidanceModifier::AdaptiveProjectedGuidance {
            eta: 0.75,
            norm_threshold: 5.0,
            momentum: 0.25,
        },
        NativeGuidanceModifier::VideoLinearCfg { minimum_scale: 1.0 },
    ])?);
    let family_resource = model
        .native_family_model_resource()
        .ok_or("fixture lost retained family MODEL")?;
    let attached_conditioning = Arc::new(NativeConditioningPayload::checked_family(
        &model,
        family_resource.patch_graph().clone(),
        None,
    )?);
    let base_payload = NativeDiffusionPayload::model(model.clone(), attached_conditioning)?;
    let attached = base_payload
        .clone()
        .with_sampling_policy(attached_policy.clone())?
        .with_guidance_modifiers(attached_modifiers.clone())?;
    attached.validate()?;
    assert_ne!(attached.digest_sha256(), base_payload.digest_sha256());
    assert_eq!(
        attached.sampling_policy().semantic_digest_sha256(),
        attached_policy.semantic_digest_sha256()
    );
    assert_eq!(
        attached.guidance_modifiers().semantic_digest_sha256(),
        attached_modifiers.semantic_digest_sha256()
    );
    assert!(attached.resident_bytes()? > base_payload.resident_bytes()?);

    let wrong_identity = ConditioningIdentity::new(
        "wrong-family",
        sim_model::ModelFamilyIdentity::new("COMFY-MODEL-9999", "wrong", "v1")?,
        family_identity(&base, "temporary")?.latent_format().clone(),
    )?;
    let wrong = family_conditioning(
        wrong_identity,
        tensor_from_f32(&backend, &[1, 2, 2_048], &[0.0; 4_096], &context)?,
        "wrong",
        &cancellation,
    )?;
    assert!(NativeGuiderPayload::basic(model, wrong).is_err());
    Ok(())
}

#[test]
fn family_model_resources_match_source_guidance_oracles() -> Result<(), Box<dyn std::error::Error>>
{
    let cancellation = CancellationToken::default();
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(MEMORY_LIMIT)?,
        &cancellation,
    );
    let oracle = family_oracle()?;
    let expected_generator_sha256 = oracle
        .get("generator_sha256")
        .and_then(Value::as_str)
        .ok_or("family oracle is missing its generator digest")?;
    let actual_generator_sha256 = format!(
        "{:x}",
        Sha256::digest(include_bytes!(
            "../fixtures/models/native-family-model-resource-foundation/generate_oracle.py"
        ))
    );
    assert_eq!(actual_generator_sha256, expected_generator_sha256);

    for (family, qwen_model) in [("aura", false), ("qwen", true)] {
        let base = Arc::new(task377::bound_model(
            &backend,
            &context,
            qwen_model,
            MEMORY_LIMIT,
        )?);
        task377::assert_raw_oracle(&base, &backend, &context, qwen_model)?;
        let patch_graph = Arc::new(PatchGraph::checked_semantic(
            FAMILY_ARTIFACT_DIGEST,
            Vec::new(),
        )?);
        let resource = Arc::new(NativeFamilyModelResource::materialize(
            base.clone(),
            patch_graph,
            family_identity(&base, "native-family-model")?,
            &backend,
            &context,
        )?);
        let model = Arc::new(NativeModelPayload::native_family_model(resource.clone())?);
        let (latent_shape, conditioning_width, latent_values) = if qwen_model {
            (
                vec![1, 16, 1, 3, 3],
                3_584,
                (0..144)
                    .map(|index| (index as f32 - 72.0) * 0.005)
                    .collect::<Vec<_>>(),
            )
        } else {
            (
                vec![1, 4, 3, 3],
                2_048,
                (0..36)
                    .map(|index| (index as f32 - 18.0) * 0.025)
                    .collect::<Vec<_>>(),
            )
        };
        let latent = tensor_from_f32(&backend, &latent_shape, &latent_values, &context)?;
        let positive = family_conditioning(
            family_identity(&base, &format!("{family}-positive"))?,
            tensor_from_f32(
                &backend,
                &[1, 2, conditioning_width],
                &family_patterned_values("positive", 2 * conditioning_width as usize),
                &context,
            )?,
            "positive",
            &cancellation,
        )?;
        let negative = family_conditioning(
            family_identity(&base, &format!("{family}-negative"))?,
            tensor_from_f32(
                &backend,
                &[1, 2, conditioning_width],
                &family_patterned_values("negative.bias", 2 * conditioning_width as usize),
                &context,
            )?,
            "negative",
            &cancellation,
        )?;
        let model_time = tensor_from_f32(&backend, &[1], &[0.5], &context)?;
        let mut raw = Vec::new();
        for conditioning in [&positive, &negative] {
            let resolved = conditioning.resolve(latent.descriptor(), &backend, &context)?;
            let entry = resolved
                .first()
                .ok_or("family conditioning did not resolve")?;
            let family_context =
                NativeFamilyDenoiserContext::checked(conditioning.identity(), &context)?;
            raw.push(resource.invoke_denoiser(
                &backend,
                NativeFamilyDenoiserInvocation {
                    scaled_latent: &latent,
                    model_time: &model_time,
                    conditioning: entry,
                    attention_mask: None,
                    reference_latents: &[],
                    additional_timestep_condition: None,
                },
                &family_context,
            )?);
        }
        let expected_positive_raw = family_oracle_raw(&oracle, family, "positive")?;
        let expected_negative_raw = family_oracle_raw(&oracle, family, "negative")?;
        assert_tensor_close(&backend, &raw[0], &expected_positive_raw, &context)?;
        assert_tensor_close(&backend, &raw[1], &expected_negative_raw, &context)?;
        assert_ne!(
            tensor_f32_sha256(&backend, &raw[0], &context)?,
            tensor_f32_sha256(&backend, &raw[1], &context)?
        );
        let resolved_positive = positive.resolve(latent.descriptor(), &backend, &context)?;
        let positive_entry = resolved_positive
            .first()
            .ok_or("family positive conditioning did not resolve")?;
        let positive_context = NativeFamilyDenoiserContext::checked(positive.identity(), &context)?;
        let changed_time = tensor_from_f32(&backend, &[1], &[0.25], &context)?;
        let changed_time_raw = resource.invoke_denoiser(
            &backend,
            NativeFamilyDenoiserInvocation {
                scaled_latent: &latent,
                model_time: &changed_time,
                conditioning: positive_entry,
                attention_mask: None,
                reference_latents: &[],
                additional_timestep_condition: None,
            },
            &positive_context,
        )?;
        assert_ne!(
            tensor_f32_sha256(&backend, &changed_time_raw, &context)?,
            tensor_f32_sha256(&backend, &raw[0], &context)?
        );
        let mut changed_latent_values = latent_values.clone();
        let first = changed_latent_values
            .first_mut()
            .ok_or("family latent fixture is empty")?;
        *first += 0.125;
        let changed_latent =
            tensor_from_f32(&backend, &latent_shape, &changed_latent_values, &context)?;
        let changed_latent_raw = resource.invoke_denoiser(
            &backend,
            NativeFamilyDenoiserInvocation {
                scaled_latent: &changed_latent,
                model_time: &model_time,
                conditioning: positive_entry,
                attention_mask: None,
                reference_latents: &[],
                additional_timestep_condition: None,
            },
            &positive_context,
        )?;
        assert_ne!(
            tensor_f32_sha256(&backend, &changed_latent_raw, &context)?,
            tensor_f32_sha256(&backend, &raw[0], &context)?
        );

        let profile = profile_for_model(&model)?;
        let positive_interpreted =
            interpret_prediction_for_profile(&profile, &backend, &raw[0], &latent, 0.5, &context)?;
        let negative_interpreted =
            interpret_prediction_for_profile(&profile, &backend, &raw[1], &latent, 0.5, &context)?;
        let expected_positive_raw_tensor =
            tensor_from_f32(&backend, &latent_shape, &expected_positive_raw, &context)?;
        let expected_negative_raw_tensor =
            tensor_from_f32(&backend, &latent_shape, &expected_negative_raw, &context)?;
        let expected_positive_interpreted = interpret_prediction_for_profile(
            &profile,
            &backend,
            &expected_positive_raw_tensor,
            &latent,
            0.5,
            &context,
        )?;
        let expected_negative_interpreted = interpret_prediction_for_profile(
            &profile,
            &backend,
            &expected_negative_raw_tensor,
            &latent,
            0.5,
            &context,
        )?;
        assert_tensor_close(
            &backend,
            &positive_interpreted,
            &tensor_to_f32(&backend, &expected_positive_interpreted, &context)?,
            &context,
        )?;
        assert_tensor_close(
            &backend,
            &negative_interpreted,
            &tensor_to_f32(&backend, &expected_negative_interpreted, &context)?,
            &context,
        )?;

        let plan =
            checked_native_diffusion_plan_for_profile(&profile, "euler", "normal", 7, 4, 2.0, 1.0)?;
        let guider =
            NativeGuiderPayload::cfg(model.clone(), positive.clone(), negative.clone(), 2.0)?;
        let mut denoiser = NativeFamilyGuidanceDenoiser::checked(&model, &backend)?;
        let guided = guider.execute(
            &backend,
            &latent,
            0.5,
            &profile,
            &plan,
            GuidanceOptions::default(),
            &mut denoiser,
            &mut [],
            &context,
        )?;
        let expected_positive = tensor_to_f32(&backend, &expected_positive_interpreted, &context)?;
        let expected_negative = tensor_to_f32(&backend, &expected_negative_interpreted, &context)?;
        let expected_guided = expected_positive
            .iter()
            .zip(expected_negative.iter())
            .map(|(positive, negative)| *negative + (*positive - *negative) * 2.0)
            .collect::<Vec<_>>();
        assert_tensor_close(&backend, guided.guided(), &expected_guided, &context)?;

        let unit_plan =
            checked_native_diffusion_plan_for_profile(&profile, "euler", "normal", 7, 4, 1.0, 1.0)?;
        let unit_guider = NativeGuiderPayload::cfg(model.clone(), positive, negative, 1.0)?;
        let mut unit_denoiser = NativeFamilyGuidanceDenoiser::checked(&model, &backend)?;
        let unit_guided = unit_guider.execute(
            &backend,
            &latent,
            0.5,
            &profile,
            &unit_plan,
            GuidanceOptions::default(),
            &mut unit_denoiser,
            &mut [],
            &context,
        )?;
        assert_tensor_close(
            &backend,
            unit_guided.guided(),
            &tensor_to_f32(&backend, &expected_positive_interpreted, &context)?,
            &context,
        )?;

        let sigmas = normal_sigmas_for_profile(&profile, &backend, &context, 4, 1.0)?;
        let expected_sigma_bits = oracle
            .pointer(&format!("/families/{family}/normal_sigmas_bits"))
            .and_then(Value::as_array)
            .ok_or("family oracle has no normal sigma bits")?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or("family oracle sigma bit is invalid")
            })
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(
            sigmas
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected_sigma_bits
        );

        let reconstructed = resource.reconstruct_in_memory(&backend, &context)?;
        assert_eq!(
            reconstructed.semantic_digest_sha256(),
            resource.semantic_digest_sha256()
        );
        assert_eq!(reconstructed.resident_bytes()?, resource.resident_bytes()?);
    }
    Ok(())
}

#[test]
fn family_model_resource_patch_budget_and_cancellation_fail_atomically()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(MEMORY_LIMIT)?,
        &cancellation,
    );
    let base = Arc::new(task377::bound_model(
        &backend,
        &context,
        false,
        MEMORY_LIMIT,
    )?);
    let identity = family_identity(&base, "native-family-model")?;
    let empty = NativeFamilyModelResource::materialize(
        base.clone(),
        Arc::new(PatchGraph::checked_semantic(
            FAMILY_ARTIFACT_DIGEST,
            Vec::new(),
        )?),
        identity.clone(),
        &backend,
        &context,
    )?;

    let patch_graph = Arc::new(PatchGraph::checked_semantic(
        FAMILY_ARTIFACT_DIGEST,
        vec![SemanticPatchOperation {
            identifier: "fixture-aura-input-bias".to_owned(),
            target_key: "native.init_x_linear.bias".to_owned(),
            expected_shape: vec![2],
            strength: 1.0,
            strength_model: 1.0,
            slices: Vec::new(),
            transform: PatchValueTransform::default(),
            payload: PatchPayload::DenseDiff {
                tensor: PatchTensor::checked(vec![2], vec![0.01, -0.01])?,
                pad_weight: false,
            },
        }],
    )?);
    let patched = NativeFamilyModelResource::materialize(
        base.clone(),
        patch_graph,
        identity.clone(),
        &backend,
        &context,
    )?;
    assert_ne!(
        patched.semantic_digest_sha256(),
        empty.semantic_digest_sha256()
    );
    assert_ne!(patched.mapped_state_sha256(), empty.mapped_state_sha256());
    assert!(
        patched.resident_tensor_allocations()?.len() > empty.resident_tensor_allocations()?.len()
    );

    let wrong_patch = Arc::new(PatchGraph::checked_semantic("f".repeat(64), Vec::new())?);
    assert!(matches!(
        NativeFamilyModelResource::materialize(
            base.clone(),
            wrong_patch,
            identity.clone(),
            &backend,
            &context,
        ),
        Err(ModelFamilyError::FamilyResourcePatchMismatch)
    ));
    assert!(matches!(
        NativeFamilyModelResource::materialize(
            base.clone(),
            Arc::new(PatchGraph::checked_semantic(
                FAMILY_ARTIFACT_DIGEST,
                Vec::new(),
            )?),
            family_identity(&base, "wrong-resource-namespace")?,
            &backend,
            &context,
        ),
        Err(ModelFamilyError::FamilyResourceConditioningMismatch)
    ));

    let required = empty.resident_bytes()?;
    let low_budget_base = Arc::new(task377::bound_model(
        &backend,
        &context,
        false,
        required.saturating_sub(1),
    )?);
    assert!(matches!(
        NativeFamilyModelResource::materialize(
            low_budget_base,
            Arc::new(PatchGraph::checked_semantic(
                FAMILY_ARTIFACT_DIGEST,
                Vec::new(),
            )?),
            identity.clone(),
            &backend,
            &context,
        ),
        Err(ModelFamilyError::OutOfMemory { .. })
    ));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(MEMORY_LIMIT)?,
        &cancelled,
    );
    assert!(matches!(
        NativeFamilyModelResource::materialize(
            base,
            Arc::new(PatchGraph::checked_semantic(
                FAMILY_ARTIFACT_DIGEST,
                Vec::new(),
            )?),
            identity,
            &backend,
            &cancelled_context,
        ),
        Err(ModelFamilyError::Cancelled(_))
    ));
    Ok(())
}

#[test]
fn native_model_transform_is_immutable_atomic_and_exports_exact_state()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let backend = Arc::new(backend);
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(MEMORY_LIMIT)?,
        &cancellation,
    );
    let target = family_diffusion(
        family_fixture_model(&backend, &context, false, true)?,
        &backend,
        &context,
    )?;
    let source = family_diffusion(
        family_fixture_model(&backend, &context, false, true)?,
        &backend,
        &context,
    )?;
    let key = "native.init_x_linear.bias";
    let model_state = target
        .model_payload()
        .keyed_transform_state(&cancellation)?;
    let expected_model_keys = aura::DENOISER_INVOCATION_REQUIRED_KEYS
        .iter()
        .map(|key| (*key).to_owned())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        model_state.keys().cloned().collect::<BTreeSet<_>>(),
        expected_model_keys,
        "MODEL keyed-state enumeration must be complete and exact"
    );
    let selector = StateKeySelector::checked(
        vec![
            StateKeyMatch::Contains("init_x_linear".to_owned()),
            StateKeyMatch::Suffix(".bias".to_owned()),
        ],
        vec!["final".to_owned()],
    )?;
    let original_digest = target.digest_sha256().to_owned();
    let original_values = tensor_to_f32(
        &backend,
        target
            .model_payload()
            .keyed_transform_state(&cancellation)?
            .get(key)
            .ok_or("missing transform fixture key")?,
        &context,
    )?;
    let source = source.scale_selected_state(&selector, 2.0, &backend, &context)?;
    assert_eq!(target.digest_sha256(), original_digest);
    let source_resource = source
        .model_payload()
        .native_family_model_resource()
        .ok_or("scaled source lost family resource")?;
    assert!(
        source_resource
            .patch_graph()
            .semantic_operations()
            .iter()
            .all(|operation| {
                matches!(operation.payload, PatchPayload::ScaleOnly) && operation.strength == 0.0
            })
    );

    let simple = PatchMergeRule::checked(
        PatchMergeKind::Model,
        key,
        key,
        PatchMergeMode::Interpolate { ratio: 0.25 },
        Vec::new(),
    )?;
    let merged = target.merge_from(&source, &simple, &backend, &context)?;
    let merged_values = tensor_to_f32(
        &backend,
        merged
            .model_payload()
            .keyed_transform_state(&cancellation)?
            .get(key)
            .ok_or("merged transform fixture key disappeared")?,
        &context,
    )?;
    for (actual, original) in merged_values.iter().zip(original_values.iter()) {
        assert!((actual - original * 1.75).abs() <= f32::EPSILON);
    }
    assert_eq!(target.digest_sha256(), original_digest);

    let subtract = PatchMergeRule::checked(
        PatchMergeKind::Model,
        key,
        key,
        PatchMergeMode::Subtract { multiplier: 0.5 },
        Vec::new(),
    )?;
    let subtracted = target.merge_from(&source, &subtract, &backend, &context)?;
    let subtracted_values = tensor_to_f32(
        &backend,
        subtracted
            .model_payload()
            .keyed_transform_state(&cancellation)?
            .get(key)
            .ok_or("subtracted transform fixture key disappeared")?,
        &context,
    )?;
    for (actual, original) in subtracted_values.iter().zip(original_values.iter()) {
        assert!((actual + original * 0.5).abs() <= f32::EPSILON);
    }

    let bypass = PatchMergeRule::checked(
        PatchMergeKind::Model,
        key,
        key,
        PatchMergeMode::Interpolate { ratio: 1.0 },
        Vec::new(),
    )?;
    let bypassed = target.merge_from(&source, &bypass, &backend, &context)?;
    let bypass_graph = bypassed
        .model_payload()
        .native_family_model_resource()
        .ok_or("bypassed transform lost family resource")?
        .patch_graph();
    assert!(
        bypass_graph
            .semantic_operations()
            .iter()
            .all(|operation| { matches!(operation.payload, PatchPayload::ScaleOnly) })
    );

    let blocks = PatchMergeRule::checked(
        PatchMergeKind::Model,
        "native.",
        "native.",
        PatchMergeMode::Interpolate { ratio: 0.1 },
        vec![
            PatchMergeRatio::checked("double_layers.", 0.2)?,
            PatchMergeRatio::checked("double_layers.0.attn.", 0.8)?,
        ],
    )?;
    let mapping = blocks
        .map("native.double_layers.0.attn.w1q.weight")?
        .ok_or("longest-prefix fixture did not map")?;
    assert_eq!(mapping.model_strength.to_bits(), 0.8_f32.to_bits());

    let metadata = BTreeMap::from([
        ("format".to_owned(), "pt".to_owned()),
        ("task".to_owned(), "404".to_owned()),
    ]);
    let first = merged.export_safetensors(metadata.clone(), MEMORY_LIMIT, &cancellation)?;
    let second = merged.export_safetensors(metadata, MEMORY_LIMIT, &cancellation)?;
    assert_eq!(first, second);
    let header_len = usize::try_from(u64::from_le_bytes(
        first
            .get(..8)
            .ok_or("safetensors header length is missing")?
            .try_into()?,
    ))?;
    let header: Value = serde_json::from_slice(
        first
            .get(8..8 + header_len)
            .ok_or("safetensors header is truncated")?,
    )?;
    assert_eq!(header["__metadata__"]["task"], "404");
    assert!(header.get(key).is_some());
    assert!(
        merged
            .export_safetensors(BTreeMap::new(), 8, &cancellation)
            .is_err()
    );
    let bounded_state = BTreeMap::from([(
        "bounded.weight".to_owned(),
        tensor_from_f32(&backend, &[2], &[1.0, 2.0], &context)?,
    )]);
    let model_golden = source_golden_safetensors(
        r#"{"__metadata__":{"format":"pt","task":"404"},"bounded.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
        &[1.0, 2.0],
    );
    assert_eq!(
        export_safetensors_exact(
            ModelStateExportKind::Model,
            &bounded_state,
            BTreeMap::from([
                ("format".to_owned(), "pt".to_owned()),
                ("task".to_owned(), "404".to_owned()),
            ]),
            MEMORY_LIMIT,
            &cancellation,
        )?,
        model_golden
    );
    assert!(matches!(
        export_safetensors_exact(
            ModelStateExportKind::Model,
            &bounded_state,
            BTreeMap::from([("oversized".to_owned(), "x".repeat(4096))]),
            64,
            &cancellation,
        ),
        Err(PatchAdapterError::ExportCapacity { available: 64, .. })
    ));

    let missing = PatchMergeRule::checked(
        PatchMergeKind::Model,
        key,
        "native.unavailable",
        PatchMergeMode::Add,
        Vec::new(),
    )?;
    assert!(
        target
            .merge_from(&source, &missing, &backend, &context)
            .is_err()
    );
    assert_eq!(target.digest_sha256(), original_digest);
    let qwen = family_diffusion(
        family_fixture_model(&backend, &context, true, true)?,
        &backend,
        &context,
    )?;
    assert!(
        target
            .merge_from(&qwen, &simple, &backend, &context)
            .is_err()
    );

    let fixture = NativeDiffusionFixture::checked_in();
    let sd15 = fixture.load_bundle_with_context(backend.clone(), &context)?;
    let legacy_clip = NativeDiffusionPayload::clip(Arc::new(NativeModelPayload::sd1_clip(
        Arc::new(fixture.tokenizer()?),
        sd15.clip().clone(),
    )?))?;
    assert!(
        legacy_clip
            .scale_selected_state(&selector, 2.0, &backend, &context)
            .is_err()
    );

    assert!(matches!(
        legacy_clip.scale_matching_state(&[(selector.clone(), 2.0)], &backend, &context),
        Err(sim_sampler::NativeDiffusionPayloadError::Model(
            sim_model::NativeModelPayloadError::UnsupportedTransformBacking {
                role: sim_model::NativeModelResourceRole::Clip,
                ..
            }
        ))
    ));

    let target_shape = target
        .model_payload()
        .keyed_transform_state(&cancellation)?
        .get(key)
        .ok_or("LoRA target disappeared")?
        .descriptor()
        .shape()
        .to_vec();
    let lora_difference = tensor_from_f32(
        &backend,
        &target_shape,
        &vec![1.0; original_values.len()],
        &context,
    )?;
    let lora_map = PatchKeyMap::checked(vec![PatchKeyMapping::direct("task404", key)])?;
    let lora_tensors = BTreeMap::from([(
        "task404.diff".to_owned(),
        AdapterTensor::Dense(lora_difference),
    )]);
    let lora_applied = target.apply_lora(&lora_map, &lora_tensors, 1.0, 1.0, &backend, &context)?;
    let lora_values = tensor_to_f32(
        &backend,
        lora_applied
            .model_payload()
            .keyed_transform_state(&cancellation)?
            .get(key)
            .ok_or("LoRA output target disappeared")?,
        &context,
    )?;
    for (actual, original) in lora_values.iter().zip(original_values.iter()) {
        assert!((actual - (original + 1.0)).abs() <= f32::EPSILON);
    }
    let lora_bypassed =
        target.apply_lora(&lora_map, &lora_tensors, 0.0, 2.0, &backend, &context)?;
    assert!(
        lora_bypassed
            .model_payload()
            .native_family_model_resource()
            .ok_or("LoRA bypass lost family resource")?
            .patch_graph()
            .semantic_operations()
            .iter()
            .all(|operation| matches!(operation.payload, PatchPayload::ScaleOnly))
    );

    let lora_export_state = BTreeMap::from([(
        "diffusion_model.native.diff".to_owned(),
        tensor_from_f32(&backend, &[2], &[1.0, 2.0], &context)?,
    )]);
    let lora_bytes = NativeDiffusionPayload::export_lora_safetensors(
        &lora_export_state,
        MEMORY_LIMIT,
        &cancellation,
    )?;
    let source_golden = source_golden_safetensors(
        r#"{"diffusion_model.native.diff":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
        &[1.0, 2.0],
    );
    assert_eq!(lora_bytes, source_golden);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(MEMORY_LIMIT)?,
        &cancelled,
    );
    assert!(
        target
            .scale_selected_state(&selector, 2.0, &backend, &cancelled_context)
            .is_err()
    );
    assert_eq!(target.digest_sha256(), original_digest);
    Ok(())
}

#[test]
fn native_clip_transport_preserves_schedule_state_and_restart_identity()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(64 * 1024 * 1024)?,
        &cancellation,
    );
    let resource = Arc::new(NativeClipResource::checked(
        NativeClipProfile::Sd3,
        true,
        Vec::new(),
        Vec::new(),
        &backend,
        &context,
    )?);
    let outputs = resource.execute(
        &backend,
        "unused by the all-missing SD3 source path",
        &context,
    )?;
    assert_eq!(outputs.len(), 1);
    assert_eq!(
        outputs[0].conditioning().descriptor().shape(),
        &[1, 77, 4_096]
    );
    assert_eq!(
        outputs[0]
            .pooled()
            .ok_or("SD3 pooled fallback is missing")?
            .descriptor()
            .shape(),
        &[1, 2_048]
    );
    assert!(resource.schedule_enabled());
    let restarted = resource.restart(&backend, &context)?;
    assert_eq!(
        restarted.semantic_digest_sha256(),
        resource.semantic_digest_sha256()
    );

    let model_payload = Arc::new(NativeModelPayload::native_clip(resource.clone())?);
    let diffusion = NativeDiffusionPayload::clip(model_payload.clone())?;
    diffusion.validate()?;
    assert!(
        diffusion
            .clip_payload()
            .is_some_and(|stored| Arc::ptr_eq(stored, &model_payload))
    );
    assert!(
        diffusion
            .native_clip_resource()
            .is_some_and(|stored| Arc::ptr_eq(stored, &resource))
    );
    assert!(diffusion.resident_bytes()? > 0);

    let clip_target = NativeDiffusionPayload::clip(Arc::new(NativeModelPayload::native_clip(
        Arc::new(native_clip_transform_fixture(&backend, &context)?),
    )?))?;
    let clip_source = NativeDiffusionPayload::clip(Arc::new(NativeModelPayload::native_clip(
        Arc::new(native_clip_transform_fixture(&backend, &context)?),
    )?))?;
    let clip_state = clip_target
        .model_payload()
        .keyed_transform_state(&cancellation)?;
    let expected_clip_keys = [
        "final_layer_norm.bias",
        "final_layer_norm.weight",
        "layers.0.feed_forward_1.bias",
        "layers.0.feed_forward_1.weight",
        "layers.0.feed_forward_2.bias",
        "layers.0.feed_forward_2.weight",
        "layers.0.key.bias",
        "layers.0.key.weight",
        "layers.0.layer_norm_1.bias",
        "layers.0.layer_norm_1.weight",
        "layers.0.layer_norm_2.bias",
        "layers.0.layer_norm_2.weight",
        "layers.0.output.bias",
        "layers.0.output.weight",
        "layers.0.query.bias",
        "layers.0.query.weight",
        "layers.0.value.bias",
        "layers.0.value.weight",
        "position_embedding.weight",
        "token_embedding.weight",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();
    assert_eq!(
        clip_state.keys().cloned().collect::<Vec<_>>(),
        expected_clip_keys,
        "CLIP keyed-state enumeration must preserve the exact complete ordered key set"
    );
    let clip_selector = StateKeySelector::checked(
        vec![
            StateKeyMatch::Suffix(".bias".to_owned()),
            StateKeyMatch::Contains("query".to_owned()),
        ],
        vec!["layer_norm".to_owned()],
    )?;
    let selected_clip_keys = clip_selector.select(&clip_state.keys().cloned().collect())?;
    assert!(
        selected_clip_keys
            .first()
            .is_some_and(|key| key.ends_with(".bias"))
    );
    let first_contains_only = selected_clip_keys
        .iter()
        .position(|key| key == "layers.0.query.weight")
        .ok_or("ordered CLIP contains selector did not append query weight")?;
    assert!(
        selected_clip_keys[..first_contains_only]
            .iter()
            .all(|key| key.ends_with(".bias"))
    );
    assert!(
        selected_clip_keys
            .iter()
            .all(|key| !key.contains("layer_norm"))
    );
    let scaled_clip = clip_target.scale_selected_state(&clip_selector, 2.0, &backend, &context)?;
    let scaled_component = scaled_clip
        .native_clip_resource()
        .and_then(|resource| resource.components().first())
        .ok_or("scaled CLIP component is missing")?;
    assert!(scaled_component.patch_graph().semantic_operations().len() > 2);
    assert!(
        scaled_component
            .patch_graph()
            .semantic_operations()
            .iter()
            .all(|operation| matches!(operation.payload, PatchPayload::ScaleOnly))
    );
    let clip_merge_rule = PatchMergeRule::checked(
        PatchMergeKind::Clip,
        "layers.0.query.bias",
        "layers.0.query.bias",
        PatchMergeMode::Add,
        Vec::new(),
    )?;
    let merged_clip = scaled_clip.merge_from(&clip_source, &clip_merge_rule, &backend, &context)?;
    let merged_component = merged_clip
        .native_clip_resource()
        .and_then(|resource| resource.components().first())
        .ok_or("merged CLIP component is missing")?;
    assert!(
        merged_component.patch_graph().semantic_operations().len()
            > scaled_component.patch_graph().semantic_operations().len()
    );
    assert!(
        merged_component
            .patch_graph()
            .semantic_operations()
            .last()
            .is_some_and(|operation| operation.identifier.starts_with("merge:"))
    );
    let clip_export = merged_clip.export_safetensors(
        BTreeMap::from([("role".to_owned(), "clip".to_owned())]),
        MEMORY_LIMIT,
        &cancellation,
    )?;
    let clip_header_length = usize::try_from(u64::from_le_bytes(
        clip_export
            .get(..8)
            .ok_or("CLIP export header length is missing")?
            .try_into()?,
    ))?;
    let clip_header: Value = serde_json::from_slice(
        clip_export
            .get(8..8 + clip_header_length)
            .ok_or("CLIP export header is truncated")?,
    )?;
    assert_eq!(clip_header["__metadata__"]["role"], "clip");
    assert!(clip_header.get("layers.0.query.bias").is_some());
    let clip_golden_state = BTreeMap::from([(
        "layers.0.query.bias".to_owned(),
        tensor_from_f32(&backend, &[2], &[-1.0, 0.5], &context)?,
    )]);
    assert_eq!(
        export_safetensors_exact(
            ModelStateExportKind::Clip,
            &clip_golden_state,
            BTreeMap::from([("role".to_owned(), "clip".to_owned())]),
            MEMORY_LIMIT,
            &cancellation,
        )?,
        source_golden_safetensors(
            r#"{"__metadata__":{"role":"clip"},"layers.0.query.bias":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
            &[-1.0, 0.5],
        )
    );

    let clip_lora_map = PatchKeyMap::checked(vec![PatchKeyMapping::direct(
        "clip_task404",
        "layers.0.query.bias",
    )])?;
    let clip_lora_tensor = tensor_from_f32(&backend, &[2], &[1.0, 2.0], &context)?;
    let clip_lora = clip_target.apply_lora(
        &clip_lora_map,
        &BTreeMap::from([(
            "clip_task404.diff".to_owned(),
            AdapterTensor::Dense(clip_lora_tensor),
        )]),
        1.0,
        1.0,
        &backend,
        &context,
    )?;
    assert!(
        clip_lora
            .native_clip_resource()
            .and_then(|resource| resource.components().first())
            .and_then(|component| component.patch_graph().semantic_operations().last())
            .is_some_and(|operation| operation.identifier == "lora:clip_task404")
    );
    Ok(())
}

#[test]
fn native_vae_transport_preserves_the_existing_image_resource()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let backend = Arc::new(backend);
    let context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(MEMORY_LIMIT)?,
        &cancellation,
    );
    let fixture = NativeDiffusionFixture::checked_in();
    let bundle = fixture.load_bundle_with_context(backend.clone(), &context)?;
    let model_payload = Arc::new(NativeModelPayload::native_vae(bundle.vae().clone())?);
    let diffusion = NativeDiffusionPayload::vae(model_payload.clone())?;
    diffusion.validate()?;
    assert!(
        diffusion
            .vae_payload()
            .is_some_and(|stored| Arc::ptr_eq(stored, &model_payload))
    );
    assert!(model_payload.vae().is_some());
    assert!(model_payload.structured_vae().is_none());
    let metadata = BTreeMap::from([("format".to_owned(), "pt".to_owned())]);
    let first = diffusion.export_safetensors(metadata.clone(), MEMORY_LIMIT, &cancellation)?;
    let second = diffusion.export_safetensors(metadata, MEMORY_LIMIT, &cancellation)?;
    assert_eq!(first, second);
    let header_len = usize::try_from(u64::from_le_bytes(
        first
            .get(..8)
            .ok_or("VAE safetensors header length is missing")?
            .try_into()?,
    ))?;
    let header: Value = serde_json::from_slice(
        first
            .get(8..8 + header_len)
            .ok_or("VAE safetensors header is truncated")?,
    )?;
    assert_eq!(header["__metadata__"]["format"], "pt");
    assert!(header.as_object().is_some_and(|header| header.len() > 1));
    let vae_golden_state = BTreeMap::from([(
        "decoder.conv.weight".to_owned(),
        tensor_from_f32(&backend, &[2], &[0.25, -0.75], &context)?,
    )]);
    assert_eq!(
        export_safetensors_exact(
            ModelStateExportKind::Vae,
            &vae_golden_state,
            BTreeMap::from([("format".to_owned(), "pt".to_owned())]),
            MEMORY_LIMIT,
            &cancellation,
        )?,
        source_golden_safetensors(
            r#"{"__metadata__":{"format":"pt"},"decoder.conv.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
            &[0.25, -0.75],
        )
    );
    Ok(())
}

#[test]
fn native_diffusion_fixture_catalog_and_provenance_are_exact()
-> Result<(), Box<dyn std::error::Error>> {
    let workspace = workspace()?;
    let fixture = NativeDiffusionFixture::checked_in();
    let catalog: Value = serde_json::from_slice(&fs::read(
        workspace.join(".agents/specs/comfy-parity/catalogs/native-diffusion-fixture.json"),
    )?)?;
    let required = catalog
        .get("required_checkpoints")
        .and_then(Value::as_array)
        .ok_or("native diffusion catalog has no required checkpoints")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or("native diffusion checkpoint name is not a string")
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    let actual = fs::read_dir(fixture.root())?
        .map(|entry| {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                return Err(std::io::Error::other(format!(
                    "native diffusion fixture contains non-file entry {:?}",
                    entry.path()
                )));
            }
            entry.file_name().into_string().map_err(|name| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("native diffusion fixture name is not UTF-8: {name:?}"),
                )
            })
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    assert_eq!(actual, required);

    assert_eq!(
        catalog
            .pointer("/source_family/feature_id")
            .and_then(Value::as_str),
        Some("COMFY-MODEL-0117")
    );
    assert_eq!(
        catalog
            .pointer("/algorithm_features/latent")
            .and_then(Value::as_str),
        Some("COMFY-MODEL-0045")
    );
    assert_eq!(
        catalog
            .pointer("/algorithm_features/sampler")
            .and_then(Value::as_str),
        Some("COMFY-MODEL-0179")
    );
    assert_eq!(
        catalog
            .pointer("/algorithm_features/scheduler")
            .and_then(Value::as_str),
        Some("COMFY-MODEL-0209")
    );
    assert_eq!(
        digest(&fixture.root().join("vocab.json"))?,
        catalog
            .pointer("/tokenizer/vocab_sha256")
            .and_then(Value::as_str)
            .ok_or("native diffusion catalog has no vocabulary digest")?
    );
    assert_eq!(
        digest(&fixture.root().join("merges.txt"))?,
        catalog
            .pointer("/tokenizer/merges_sha256")
            .and_then(Value::as_str)
            .ok_or("native diffusion catalog has no merges digest")?
    );

    let provenance: Value = serde_json::from_slice(&fixture.read("oracle-provenance.json")?)?;
    assert_eq!(
        provenance
            .get("production_dependency")
            .and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(
        provenance.get("oracle_kind").and_then(Value::as_str),
        Some("pinned-comfyui-source-equation-fixture")
    );
    let sources = provenance
        .get("sources")
        .and_then(Value::as_object)
        .ok_or("native diffusion provenance has no sources")?;
    for (source, expected_digest) in sources {
        assert_eq!(
            digest(&workspace.join(source))?,
            expected_digest
                .as_str()
                .ok_or("native diffusion source digest is not a string")?,
            "stale native diffusion source provenance for {source}"
        );
    }
    assert_eq!(
        digest(&workspace.join("projects/comfy/ComfyUI/comfy/sd1_clip_config.json"))?,
        catalog
            .pointer("/clip/source_config_sha256")
            .and_then(Value::as_str)
            .ok_or("native diffusion catalog has no CLIP config digest")?
    );
    Ok(())
}

#[test]
fn native_diffusion_fixture_rejects_tampering_before_weight_parsing()
-> Result<(), Box<dyn std::error::Error>> {
    let checked_in = NativeDiffusionFixture::checked_in();
    let cancellation = CancellationToken::default();

    let invalid_key_directory = tempfile::tempdir()?;
    copy_fixture_admission_files(&checked_in, invalid_key_directory.path())?;
    copy_model_with_replacement(
        &checked_in,
        invalid_key_directory.path(),
        b"model.diffusion_model.input_blocks.0.0.weight",
        b"model.diffusion_model.input_blocks.0.0.weighx",
    )?;
    let invalid_key_error = NativeDiffusionFixture::at(invalid_key_directory.path())
        .load_model(MEMORY_LIMIT, &cancellation)
        .expect_err("tampered weight key must fail model admission");
    assert!(
        matches!(
            &invalid_key_error,
            NativeDiffusionFixtureError::ModelDigestMismatch { .. }
        ),
        "unexpected invalid-key error: {invalid_key_error:?}"
    );

    let invalid_shape_directory = tempfile::tempdir()?;
    copy_fixture_admission_files(&checked_in, invalid_shape_directory.path())?;
    copy_model_with_replacement(
        &checked_in,
        invalid_shape_directory.path(),
        b"\"shape\":[32,4,3,3]",
        b"\"shape\":[16,8,3,3]",
    )?;
    let invalid_shape_error = NativeDiffusionFixture::at(invalid_shape_directory.path())
        .load_model(MEMORY_LIMIT, &cancellation)
        .expect_err("tampered weight shape must fail model admission");
    assert!(
        matches!(
            &invalid_shape_error,
            NativeDiffusionFixtureError::ModelDigestMismatch { .. }
        ),
        "unexpected invalid-shape error: {invalid_shape_error:?}"
    );
    Ok(())
}

#[test]
fn canonical_clip_load_is_failure_atomic_and_workspace_bounded()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = NativeDiffusionFixture::checked_in();
    let cancellation = CancellationToken::default();
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let backend = Arc::new(backend);
    let workspace = workspace_authority.authorize_workspace(1024 * 1024)?;
    let context = backend.execution_context(StreamId::DEFAULT, workspace.clone(), &cancellation);
    assert!(
        fixture
            .load_clip_with_context(backend.clone(), &context)
            .is_err()
    );
    assert_eq!(workspace.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    Ok(())
}

#[test]
fn native_diffusion_fixture_matches_all_checkpoints() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = NativeDiffusionFixture::checked_in();
    let source_reference = fixture.source_sampler_reference()?;
    let cancellation = CancellationToken::default();
    let tokenizer = fixture.tokenizer()?;
    let positive_tokens = encode_sd15_prompt(&tokenizer, "a test", &cancellation)?;
    let negative_tokens = encode_sd15_prompt(&tokenizer, "", &cancellation)?;
    assert_eq!(&positive_tokens[..4], &[49_406, 320, 1_628, 49_407]);
    assert_eq!(negative_tokens[0], 49_406);
    assert!(negative_tokens[1..].iter().all(|token| *token == 49_407));

    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let backend = Arc::new(backend);
    let workspace = workspace_authority.authorize_workspace(MEMORY_LIMIT)?;
    let context = backend.execution_context(StreamId::DEFAULT, workspace.clone(), &cancellation);
    let cache_identities = fixture.cache_identities(context.cancellation)?;
    assert_eq!(
        cache_identities.tokenizer_digest(),
        tokenizer.identity().digest()
    );
    let bundle = fixture.load_bundle_with_context(backend.clone(), &context)?;
    assert_eq!(&cache_identities, bundle.cache_identities());
    let model = bundle.model().clone();
    let model_digest = cache_identities.model_digest().to_owned();
    let vocabulary = String::from_utf8(fixture.read("vocab.json")?)?;
    let merges = String::from_utf8(fixture.read("merges.txt")?)?;
    let wrong_descriptor = Sd1Tokenizer::from_json_and_merges(
        ModelTokenizerDescriptor::checked("comfy.sd2.tokenizer")?,
        &vocabulary,
        &merges,
    )?;
    assert!(
        NativeDiffusionBundle::new_with_vae(
            "sd15-tiny-v1",
            model_digest.clone(),
            model.clone(),
            Arc::new(wrong_descriptor),
            bundle.clip().clone(),
            bundle.vae().clone(),
        )
        .is_err()
    );
    let mut alternate_vocabulary = serde_json::from_str::<BTreeMap<String, u32>>(&vocabulary)?;
    let ordinary_keys = alternate_vocabulary
        .iter()
        .filter(|(_, token)| **token < sim_model::clip::SD1_START_TOKEN)
        .take(2)
        .map(|(piece, _)| piece.clone())
        .collect::<Vec<_>>();
    let [first_key, second_key] = ordinary_keys.as_slice() else {
        return Err("SD1 vocabulary has fewer than two ordinary tokens".into());
    };
    let first_value = *alternate_vocabulary
        .get(first_key)
        .ok_or("first ordinary SD1 token disappeared")?;
    let second_value = *alternate_vocabulary
        .get(second_key)
        .ok_or("second ordinary SD1 token disappeared")?;
    alternate_vocabulary.insert(first_key.clone(), second_value);
    alternate_vocabulary.insert(second_key.clone(), first_value);
    let alternate_vocabulary = serde_json::to_string(&alternate_vocabulary)?;
    let alternate_tokenizer = Sd1Tokenizer::from_json_and_merges(
        ModelTokenizerDescriptor::checked("comfy.sd1.tokenizer")?,
        &alternate_vocabulary,
        &merges,
    )?;
    assert_ne!(
        alternate_tokenizer.identity().digest(),
        cache_identities.tokenizer_digest()
    );
    assert!(
        NativeDiffusionBundle::new_with_vae(
            "sd15-tiny-v1",
            model_digest,
            model.clone(),
            Arc::new(alternate_tokenizer),
            bundle.clip().clone(),
            bundle.vae().clone(),
        )
        .is_err()
    );
    let (canonical_positive_tokens, positive) = bundle.encode_text("a test", &context)?;
    let (canonical_negative_tokens, negative) = bundle.encode_text("", &context)?;
    assert_eq!(canonical_positive_tokens, positive_tokens);
    assert_eq!(canonical_negative_tokens, negative_tokens);
    assert_tensor_file(
        fixture.root().join("clip-conditioning.safetensors"),
        "positive",
        &positive,
        &backend,
        &context,
    )?;
    assert_tensor_file(
        fixture.root().join("clip-conditioning.safetensors"),
        "negative",
        &negative,
        &backend,
        &context,
    )?;

    let sigmas = normal_sigmas(&backend, &context, 4, 1.0)?;
    let sigma_bytes = fs::read(fixture.root().join("normal-sigmas.f64le"))?;
    let expected_sigmas = sigma_bytes
        .chunks_exact(8)
        .map(|chunk| {
            let encoded: [u8; 8] = chunk.try_into().map_err(|_| "invalid sigma fixture")?;
            Ok(f64::from_le_bytes(encoded))
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    assert_eq!(
        expected_sigmas,
        sigmas
            .iter()
            .map(|value| f64::from(*value))
            .collect::<Vec<_>>()
    );

    let stream = NoiseRequest::native_diffusion(FIXTURE_PROMPT_ID, FIXTURE_KSAMPLER_NODE_ID)?
        .stream(SEED, DeviceId::CPU)?;
    let noise = normal_noise(&backend, &[1, 4, 4, 4], &stream, &context)?;
    let expected_checkpoint: RngCheckpoint = serde_json::from_slice(&fs::read(
        fixture.root().join("rng-state-before-noise.bin"),
    )?)?;
    assert_eq!(noise.before, expected_checkpoint);
    assert_tensor_file(
        fixture.root().join("initial-noise.safetensors"),
        "noise",
        &noise.noise,
        &backend,
        &context,
    )?;
    let latent = empty_sd15_latent(&backend, 1, 32, 32, &context)?;
    let initial = scale_initial_noise(&backend, &noise.noise, &latent, sigmas[0], &context)?;
    let plan = checked_native_diffusion_plan("euler", "normal", SEED, 4, 7.0, 1.0)?;
    let mut guidance = Sd15GuidanceAdapter::checked(&model, &positive, &negative, &context)?;
    let trace = sample_euler(
        &backend,
        initial,
        &sigmas,
        &context,
        |latent, sigma, _step| {
            let model_input = scale_model_input(&backend, latent, sigma, &context)
                .map_err(|error| error.to_string())?;
            let prediction = guidance
                .execute(&backend, &model_input, sigma, &plan, &context)
                .map_err(|error| error.to_string())?;
            sd15_interpret_prediction(&backend, prediction.guided(), latent, sigma, &context)
                .map_err(|error| error.to_string())
        },
    )?;
    assert_eq!(trace.denoiser_evaluations.len(), 4);
    assert_eq!(trace.latents.len(), 5);
    let hashes = |tensors: &[sim_tensor::Tensor]| -> Result<Value, Box<dyn std::error::Error>> {
        let hashes = tensors
            .iter()
            .map(|tensor| {
                Ok(format!(
                    "{:x}",
                    Sha256::digest(f32_bytes(&tensor_to_f32(&backend, tensor, &context)?))
                ))
            })
            .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
        Ok(serde_json::to_value(hashes)?)
    };
    assert_eq!(
        hashes(&trace.denoiser_evaluations)?,
        source_reference["denoiser_sha256"]
    );
    assert_eq!(hashes(&trace.latents)?, source_reference["latent_sha256"]);
    for (index, denoised) in trace.denoiser_evaluations.iter().enumerate() {
        assert_tensor_file(
            fixture
                .root()
                .join(format!("denoiser-eval-{index:03}.safetensors")),
            "denoised",
            denoised,
            &backend,
            &context,
        )?;
    }
    for (index, latent) in trace.latents.iter().enumerate() {
        assert_tensor_file(
            fixture
                .root()
                .join(format!("latent-step-{index:03}.safetensors")),
            "latent",
            latent,
            &backend,
            &context,
        )?;
    }
    let decoded = bundle.vae().decode(
        backend.as_ref(),
        trace.latents.last().ok_or("missing final latent")?,
        &context,
    )?;
    let decoded_values = tensor_to_f32(&backend, &decoded, &context)?;
    let expected_decoded = fs::read(fixture.root().join("vae-decoded.f32le"))?;
    assert_eq!(
        decoded_values.len().checked_mul(4),
        Some(expected_decoded.len())
    );
    for (index, (actual, bytes)) in decoded_values
        .iter()
        .zip(expected_decoded.chunks_exact(4))
        .enumerate()
    {
        let expected = f64::from(f32::from_le_bytes(bytes.try_into()?));
        let error = (f64::from(*actual) - expected).abs();
        assert!(
            error <= 2e-6 + 2e-5 * expected.abs(),
            "decoded[{index}]: native {actual}, recorded {expected}, error {error}"
        );
    }
    drop(decoded);
    let bhwc = nchw_to_bhwc(&decoded_values, 3, 32, 32)?;
    let png = encode_png_frame(
        &bhwc,
        1,
        32,
        32,
        3,
        0,
        &BTreeMap::new(),
        PngLimits::default(),
    )?;
    assert_eq!(png, fs::read(fixture.root().join("output.png"))?);

    let workspace_before_cancel = workspace.in_use_bytes();
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context =
        backend.execution_context(StreamId::DEFAULT, workspace.clone(), &cancelled);
    assert!(bundle.encode_text("a test", &cancelled_context).is_err());
    assert!(
        fixture
            .load_model(1024, &CancellationToken::default())
            .is_err()
    );
    assert_eq!(workspace.in_use_bytes(), workspace_before_cancel);
    Ok(())
}

#[test]
fn native_diffusion_guidance_adapter_preserves_cfg_and_failure_boundaries()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let fixture = NativeDiffusionFixture::checked_in();
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let backend = Arc::new(backend);
    let workspace = workspace_authority.authorize_workspace(MEMORY_LIMIT)?;
    let context = backend.execution_context(StreamId::DEFAULT, workspace, &cancellation);
    let bundle = fixture.load_bundle_with_context(backend.clone(), &context)?;
    let model = bundle.model();
    let (_, positive) = bundle.encode_text("a test", &context)?;
    let (_, negative) = bundle.encode_text("", &context)?;
    let latent = tensor_from_f32(&backend, &[1, 4, 4, 4], &[0.0; 64], &context)?;
    let sigma = normal_sigmas(&backend, &context, 4, 1.0)?[0];
    let model_time = sd15_model_time(sigma)?;

    let scale_one = checked_native_diffusion_plan("euler", "normal", SEED, 4, 1.0, 1.0)?;
    let mut adapter = Sd15GuidanceAdapter::checked(model.as_ref(), &positive, &negative, &context)?;
    let guided = adapter.execute(&backend, &latent, sigma, &scale_one, &context)?;
    let conditional = model.denoise_at_model_time(&latent, model_time, &positive, &context)?;
    assert!(guided.unconditional_skipped());
    assert_eq!(guided.denoiser_evaluations(), 1);
    assert_eq!(guided.guided().descriptor(), latent.descriptor());
    let guided_values = tensor_to_f32(&backend, guided.guided(), &context)?;
    let conditional_values = tensor_to_f32(&backend, &conditional, &context)?;
    assert_eq!(&guided_values[..], &conditional_values[..]);
    drop(conditional_values);
    drop(guided_values);

    let scale_zero = checked_native_diffusion_plan("euler", "normal", SEED, 4, 0.0, 1.0)?;
    let guided = adapter.execute(&backend, &latent, sigma, &scale_zero, &context)?;
    let unconditional = model.denoise_at_model_time(&latent, model_time, &negative, &context)?;
    assert!(!guided.unconditional_skipped());
    assert_eq!(guided.denoiser_evaluations(), 2);
    let guided_values = tensor_to_f32(&backend, guided.guided(), &context)?;
    let unconditional_values = tensor_to_f32(&backend, &unconditional, &context)?;
    assert_eq!(&guided_values[..], &unconditional_values[..]);
    drop(unconditional_values);
    drop(guided_values);

    let wrong_shape = tensor_from_f32(&backend, &[1, 77, 31], &[0.0; 77 * 31], &context)?;
    let mut wrong = Sd15GuidanceAdapter::checked(&model, &wrong_shape, &negative, &context)?;
    assert!(matches!(
        wrong.execute(&backend, &latent, sigma, &scale_zero, &context),
        Err(sim_runtime::NativeImageRuntimeError::Execution(message))
            if message.contains("SD15 conditioning")
    ));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context =
        backend.execution_context(StreamId::DEFAULT, context.scratch.clone(), &cancelled);
    assert!(matches!(
        Sd15GuidanceAdapter::checked(&model, &positive, &negative, &cancelled_context),
        Err(sim_runtime::NativeImageRuntimeError::Cancelled)
    ));
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);

    let constrained_workspace = workspace_authority.authorize_workspace(64)?;
    let constrained_context = backend.execution_context(
        StreamId::DEFAULT,
        constrained_workspace.clone(),
        &cancellation,
    );
    let mut constrained =
        Sd15GuidanceAdapter::checked(&model, &positive, &negative, &constrained_context)?;
    let constrained_error = constrained
        .execute(&backend, &latent, sigma, &scale_zero, &constrained_context)
        .expect_err("undersized workspace authorization must reject guidance");
    assert!(
        matches!(
            &constrained_error,
            sim_runtime::NativeImageRuntimeError::ResourceExhausted(message)
                if message.contains("workspace request of")
                    && message.contains("64-byte authorization")
        ),
        "unexpected constrained-workspace error: {constrained_error:?}"
    );
    assert_eq!(constrained_workspace.in_use_bytes(), 0);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

fn assert_tensor_file(
    path: impl AsRef<Path>,
    name: &str,
    actual: &sim_tensor::Tensor,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let header_length_bytes: [u8; 8] = bytes
        .get(..8)
        .ok_or("missing safetensors header")?
        .try_into()?;
    let header_length = usize::try_from(u64::from_le_bytes(header_length_bytes))?;
    let data_start = 8_usize
        .checked_add(header_length)
        .ok_or("safetensors offset overflow")?;
    let header: Value = serde_json::from_slice(
        bytes
            .get(8..data_start)
            .ok_or("truncated safetensors header")?,
    )?;
    let descriptor = header.get(name).ok_or("missing tensor descriptor")?;
    assert_eq!(descriptor.get("dtype").and_then(Value::as_str), Some("F32"));
    assert_eq!(
        descriptor.get("shape"),
        Some(&serde_json::to_value(actual.descriptor().shape())?)
    );
    let offsets = descriptor
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or("missing offsets")?;
    let start = usize::try_from(
        offsets
            .first()
            .and_then(Value::as_u64)
            .ok_or("missing start")?,
    )?;
    let end = usize::try_from(
        offsets
            .get(1)
            .and_then(Value::as_u64)
            .ok_or("missing end")?,
    )?;
    let expected = bytes
        .get(data_start + start..data_start + end)
        .ok_or("truncated tensor data")?;
    let actual_values = tensor_to_f32(backend, actual, context)?;
    if name == "noise" {
        assert_eq!(f32_bytes(&actual_values), expected);
    } else {
        assert_eq!(actual_values.len().checked_mul(4), Some(expected.len()));
        for (index, (actual, bytes)) in actual_values
            .iter()
            .zip(expected.chunks_exact(4))
            .enumerate()
        {
            let expected = f64::from(f32::from_le_bytes(bytes.try_into()?));
            let error = (f64::from(*actual) - expected).abs();
            assert!(
                error <= 2e-6 + 2e-5 * expected.abs(),
                "{name}[{index}]: native {actual}, recorded {expected}, error {error}"
            );
        }
    }
    Ok(())
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn nchw_to_bhwc(
    values: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if values.len() != channels * height * width {
        return Err("decoded tensor shape is invalid".into());
    }
    let mut output = vec![0.0; values.len()];
    for y in 0..height {
        for x in 0..width {
            for channel in 0..channels {
                output[(y * width + x) * channels + channel] =
                    values[(channel * height + y) * width + x];
            }
        }
    }
    Ok(output)
}

fn workspace() -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?
        .to_path_buf())
}

fn digest(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn copy_fixture_admission_files(
    fixture: &NativeDiffusionFixture,
    destination: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::copy(
        fixture.root().join("sd15-detector-projection.json"),
        destination.join("sd15-detector-projection.json"),
    )?;
    Ok(())
}

fn copy_model_with_replacement(
    fixture: &NativeDiffusionFixture,
    destination: &Path,
    needle: &[u8],
    replacement: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    if needle.len() != replacement.len() {
        return Err("native diffusion test replacement changes safetensors length".into());
    }
    let mut bytes = fixture.read("model.safetensors")?;
    let matches = bytes
        .windows(needle.len())
        .enumerate()
        .filter_map(|(index, candidate)| (candidate == needle).then_some(index))
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!(
            "native diffusion test replacement expected one match, found {}",
            matches.len()
        )
        .into());
    }
    let start = matches[0];
    let end = start
        .checked_add(replacement.len())
        .ok_or("native diffusion test replacement overflowed")?;
    bytes
        .get_mut(start..end)
        .ok_or("native diffusion test replacement is out of bounds")?
        .copy_from_slice(replacement);
    fs::write(destination.join("model.safetensors"), bytes)?;
    Ok(())
}

#[test]
fn attention_matching_scaling_preserves_existing_patches_and_clip_source_keys()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(MEMORY_LIMIT)?,
        &cancellation,
    );
    let resource = native_clip_transform_fixture(&backend, &context)?;
    let component = resource
        .components()
        .first()
        .ok_or("fixture component missing")?;
    let graph = component
        .patch_graph()
        .append_semantic([SemanticPatchOperation {
            identifier: "attention-preexisting-dense-patch".to_owned(),
            target_key: "layers.0.query.weight".to_owned(),
            expected_shape: vec![2, 2],
            strength: 1.0,
            strength_model: 1.0,
            slices: Vec::new(),
            transform: PatchValueTransform::default(),
            payload: PatchPayload::DenseDiff {
                tensor: PatchTensor::checked(vec![2, 2], vec![0.25, -0.5, 1.0, 2.0])?,
                pad_weight: false,
            },
        }])?;
    let resource = resource.rebind_component_patch_graphs(vec![graph], &backend, &context)?;
    let clip = NativeModelPayload::native_clip(Arc::new(resource))?;
    let selector = StateKeySelector::checked(
        vec![
            StateKeyMatch::Suffix("self_attn.q_proj.weight".to_owned()),
            StateKeyMatch::Suffix("self_attn.q_proj.bias".to_owned()),
        ],
        Vec::new(),
    )?;
    let original_digest = clip.identity().digest_sha256().to_owned();
    let original = clip
        .native_clip_resource()
        .ok_or("CLIP fixture resource missing")?;
    let component = original
        .components()
        .first()
        .ok_or("CLIP fixture component missing")?;
    let previous_operations = component.patch_graph().semantic_operations().to_vec();
    let scaled = clip.scale_matching_state(&[(selector.clone(), 2.0)], &backend, &context)?;
    let repeated = scaled.scale_matching_state(&[(selector.clone(), 3.0)], &backend, &context)?;
    assert_eq!(clip.identity().digest_sha256(), original_digest);
    let resource = repeated
        .native_clip_resource()
        .ok_or("scaled CLIP resource missing")?;
    let repeated_component = resource
        .components()
        .first()
        .ok_or("scaled CLIP component missing")?;
    let operations = repeated_component.patch_graph().semantic_operations();
    assert_eq!(operations.len(), previous_operations.len() + 4);
    assert_eq!(
        serde_json::to_value(&operations[..previous_operations.len()])?,
        serde_json::to_value(&previous_operations)?
    );
    for (operation, scale) in operations
        .iter()
        .skip(previous_operations.len())
        .zip([2.0, 2.0, 3.0, 3.0])
    {
        assert_eq!(operation.strength, 0.0);
        assert_eq!(operation.strength_model, scale);
        assert!(matches!(operation.payload, PatchPayload::ScaleOnly));
        assert!(operation.target_key.starts_with("layers.0.query."));
    }
    assert_ne!(
        clip.identity().execution_sha256(),
        scaled.identity().execution_sha256()
    );
    assert_ne!(
        scaled.identity().execution_sha256(),
        repeated.identity().execution_sha256()
    );
    let restarted = resource.restart(&backend, &context)?;
    assert_eq!(
        restarted.semantic_digest_sha256(),
        resource.semantic_digest_sha256()
    );
    let base_weights = component
        .keyed_state()
        .get("layers.0.query.weight")
        .ok_or("query weight missing")?;
    let original_applied = component.patch_graph().apply_single_tensor(
        &backend,
        "layers.0.query.weight",
        base_weights,
        sim_model::PatchComputeBoundary::Configured(DType::F32),
        &context,
    )?;
    let repeated_applied = repeated_component.patch_graph().apply_single_tensor(
        &backend,
        "layers.0.query.weight",
        base_weights,
        sim_model::PatchComputeBoundary::Configured(DType::F32),
        &context,
    )?;
    assert_eq!(
        &*tensor_to_f32(&backend, &repeated_applied, &context)?,
        tensor_to_f32(&backend, &original_applied, &context)?
            .into_iter()
            .map(|value| value * 2.0 * 3.0)
            .collect::<Vec<_>>()
    );
    let zeroed = clip.scale_matching_state(&[(selector.clone(), 0.0)], &backend, &context)?;
    let zeroed_component = zeroed
        .native_clip_resource()
        .and_then(|resource| resource.components().first())
        .ok_or("zeroed CLIP component missing")?;
    let zeroed_tensor = zeroed_component.patch_graph().apply_single_tensor(
        &backend,
        "layers.0.query.weight",
        base_weights,
        sim_model::PatchComputeBoundary::Configured(DType::F32),
        &context,
    )?;
    assert!(
        tensor_to_f32(&backend, &zeroed_tensor, &context)?
            .iter()
            .all(|value| *value == 0.0)
    );
    let no_match = StateKeySelector::checked(
        vec![StateKeyMatch::Suffix("missing.attention.weight".to_owned())],
        Vec::new(),
    )?;
    assert_eq!(
        clip.scale_matching_state(&[(no_match, 2.0)], &backend, &context)?
            .identity()
            .digest_sha256(),
        original_digest
    );
    assert!(
        clip.scale_matching_state(&[(selector.clone(), f32::NAN)], &backend, &context)
            .is_err()
    );
    cancellation.cancel();
    assert!(
        clip.scale_matching_state(&[(selector, 2.0)], &backend, &context)
            .is_err()
    );
    assert_eq!(clip.identity().digest_sha256(), original_digest);
    Ok(())
}

#[test]
fn attention_matching_model_clone_and_selector_identity_are_atomic()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(MEMORY_LIMIT)?,
        &cancellation,
    );
    let model = family_diffusion(
        family_fixture_model(&backend, &context, false, true)?,
        &backend,
        &context,
    )?;
    let selector = StateKeySelector::checked(
        vec![StateKeyMatch::Suffix("init_x_linear.bias".to_owned())],
        Vec::new(),
    )?;
    let legacy = serde_json::json!({"ordered_matches":[{"match":"suffix","value":"init_x_linear.bias"}],"excluded_contains":[]});
    assert_eq!(serde_json::to_value(&selector)?, legacy);
    let restored: StateKeySelector = serde_json::from_value(legacy)?;
    assert_eq!(selector.identity()?, restored.identity()?);
    let temporal = selector
        .clone()
        .with_required_contains(vec![".time_stack.".to_owned()])?;
    assert_ne!(selector.identity()?, temporal.identity()?);
    let original = model.digest_sha256().to_owned();
    let scaled = model.scale_matching_state(&[(selector.clone(), 2.0)], &backend, &context)?;
    let repeated = scaled.scale_matching_state(&[(selector.clone(), 3.0)], &backend, &context)?;
    let resource = repeated
        .model_payload()
        .native_family_model_resource()
        .ok_or("MODEL family lost")?;
    assert_eq!(resource.patch_graph().semantic_operations().len(), 2);
    assert_eq!(
        resource
            .patch_graph()
            .semantic_operations()
            .iter()
            .map(|operation| operation.strength_model)
            .collect::<Vec<_>>(),
        [2.0, 3.0]
    );
    assert_eq!(model.digest_sha256(), original);
    assert_eq!(
        model
            .scale_matching_state(&[(temporal, 2.0)], &backend, &context)?
            .digest_sha256(),
        original
    );
    assert!(
        model
            .scale_matching_state(&[(selector.clone(), f32::INFINITY)], &backend, &context)
            .is_err()
    );
    let constrained = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1)?,
        &cancellation,
    );
    assert!(
        model
            .scale_matching_state(&[(selector.clone(), 2.0)], &backend, &constrained)
            .is_err()
    );
    assert_eq!(model.digest_sha256(), original);
    cancellation.cancel();
    assert!(
        model
            .scale_matching_state(&[(selector, 2.0)], &backend, &context)
            .is_err()
    );
    Ok(())
}

#[test]
fn attention_model_nodes_preserve_compiled_wrappers_and_source_no_match_clones()
-> Result<(), Box<dyn std::error::Error>> {
    use sim_nodes::{
        NativeModelCompileBackend, NativeModelCompileRequest, NativeNodeComputeSession,
        NativeNodeContext, NativeNodeServiceIdentity, NativeNodeServices,
    };
    use sim_runtime::{NativeHandleStoreGeneration, NativeNodeOutcome, NativePrimitive};
    let cancellation = CancellationToken::default();
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let backend = Arc::new(backend);
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(MEMORY_LIMIT)?,
        &cancellation,
    );
    let model = Arc::new(family_diffusion(
        family_fixture_model(&backend, &execution, false, true)?,
        &backend,
        &execution,
    )?);
    let registry = sim_runtime::generated_native_node_registry_projection(None)?;
    for (index, class_type) in [
        "UNetSelfAttentionMultiply",
        "UNetCrossAttentionMultiply",
        "UNetTemporalAttentionMultiply",
    ]
    .into_iter()
    .enumerate()
    {
        let attempt = AttemptId(uuid::Uuid::from_u128(0x595100 + index as u128));
        let generation = NativeHandleStoreGeneration::with_capacities(8, MEMORY_LIMIT as usize)?;
        let store = generation.handle_store_for_attempt(attempt);
        let node_id = NodeId(class_type.to_owned());
        let identity = NativeNodeServiceIdentity::checked(
            uuid::Uuid::from_u128(0x595200),
            attempt,
            node_id.clone(),
        )?;
        let scratch = authority.authorize_workspace(MEMORY_LIMIT)?;
        let compute = NativeNodeComputeSession::checked(
            identity.clone(),
            backend.clone(),
            StreamId::DEFAULT,
            &scratch,
        )?;
        let service =
            sim_runtime::native_execution_controller::NativeRuntimeModelCompileService::new(
                identity, None,
            );
        let context = NativeNodeContext::new_with_services(
            PromptId(uuid::Uuid::from_u128(0x595300)),
            attempt,
            node_id,
            cancellation.clone(),
            scratch,
            store.clone(),
            NativeNodeServices::checked(None, None, Some(compute))?
                .with_model_compile(Arc::new(service))?,
        )?;
        let compiled = context.compile_native_model(&NativeModelCompileRequest::checked(
            model.clone(),
            NativeModelCompileBackend::Inductor,
        )?)?;
        let policy = compiled.policy_sha256().to_owned();
        let payload = NativeStoredPayload::Model(Arc::new(
            NativeStoredModelPayload::native_compiled_diffusion(
                compiled.model().clone(),
                compiled.compiled_model().clone(),
            )?,
        ));
        let original_digest = payload.digest_sha256();
        let handle = store.publish(payload, &cancellation)?;
        let mut inputs =
            BTreeMap::from([("model".to_owned(), NativeValue::Handle { value: handle })]);
        let parameters = if class_type == "UNetTemporalAttentionMultiply" {
            [
                "self_structural",
                "self_temporal",
                "cross_structural",
                "cross_temporal",
            ]
        } else {
            ["q", "k", "v", "out"]
        };
        for parameter in parameters {
            inputs.insert(
                parameter.to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Number(2.0),
                },
            );
        }
        let node = registry
            .node(class_type)
            .ok_or("attention MODEL node missing")?;
        let result = smol::block_on(node.execute(context, inputs.clone()))?;
        let NativeNodeOutcome::Values {
            outputs,
            ui,
            effects,
        } = result
        else {
            return Err("attention MODEL values absent".into());
        };
        assert!(ui.is_none() && effects.is_empty());
        let Some(NativeValue::Handle { value: handle }) = outputs.first() else {
            return Err("attention MODEL handle absent".into());
        };
        assert_eq!(handle.digest_sha256(), Some(original_digest.as_str()));
        let resolved = store.resolve(handle, handle.handle_type(), &cancellation)?;
        let NativeStoredPayload::Model(payload) = resolved.as_ref() else {
            return Err("attention MODEL payload absent".into());
        };
        let wrapper = payload.compiled_model().ok_or("compiled wrapper lost")?;
        wrapper.validate()?;
        assert_eq!(wrapper.backend(), NativeModelCompileBackend::Inductor);
        assert_eq!(wrapper.policy_sha256(), policy);
        assert_eq!(
            payload
                .diffusion()
                .ok_or("diffusion wrapper lost")?
                .digest_sha256(),
            model.digest_sha256()
        );
        let executor = NativeImageExecutor::new_with_generated_registry(
            ProfileId(uuid::Uuid::from_u128(0x595400)),
            BTreeMap::new(),
            true,
            backend.clone(),
        )?;
        let engine_store = executor
            .handle_store_generation_for_test()
            .handle_store_for_attempt(attempt);
        let engine_handle =
            engine_store.publish(NativeStoredPayload::Model(payload.clone()), &cancellation)?;
        inputs.insert(
            "model".to_owned(),
            NativeValue::Handle {
                value: engine_handle,
            },
        );
        let mut descriptor = registry
            .descriptor(class_type)
            .ok_or("attention engine descriptor missing")?
            .clone();
        descriptor.output_node = true;
        let node_id = NodeId(class_type.to_owned());
        let plan = sim_runtime::CompiledPlan {
            prompt_id: PromptId(uuid::Uuid::from_u128(0x595500 + index as u128)),
            client_id: None,
            prompt_number: None,
            extra_data: BTreeMap::new(),
            unknown: BTreeMap::new(),
            nodes: BTreeMap::from([(
                node_id.clone(),
                sim_runtime::CompiledNode {
                    id: node_id.clone(),
                    class_type: class_type.to_owned(),
                    descriptor,
                    inputs: inputs
                        .into_iter()
                        .map(|(name, value)| (name, InputBinding::Literal { value }))
                        .collect(),
                    unknown: BTreeMap::new(),
                },
            )]),
            topological_order: vec![node_id.clone()],
            static_required_nodes: BTreeSet::from([node_id.clone()]),
            output_nodes: vec![node_id.clone()],
            provider_execution: None,
            persistence_unknown_fields: BTreeMap::new(),
        };
        plan.validate_integrity()?;
        let execution = executor.execute_blocking(
            &plan,
            attempt,
            cancellation.clone(),
            0,
            authority.authorize_workspace(MEMORY_LIMIT)?,
        )?;
        assert_eq!(
            execution.report.state,
            AttemptState::Succeeded,
            "{class_type}: {:?}",
            execution.report.error
        );
        let Some(NativeValue::Handle { value: handle }) = execution
            .report
            .outputs
            .get(&node_id)
            .and_then(|outputs| outputs.first())
        else {
            return Err("attention engine did not publish its MODEL".into());
        };
        assert_eq!(handle.digest_sha256(), Some(original_digest.as_str()));
    }
    Ok(())
}
