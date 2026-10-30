use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelFamilyError, ModelProbe, ModelStateTransaction,
    ModelStore, ParserLimits, SD2_CLIP_TARGET, SD2_COMMON_MAPPING, SD2_CONDITIONING,
    SD2_DIFFUSERS_STATE_PLAN, SD2_FORWARD_PROGRAM, SD2_LATENT_FORMAT, SD2_MEMORY_USAGE_FACTOR,
    SD2_PREFIXED_STATE_PLAN, SD2_TRANSFORMER_DEPTH, SD2_UNCLIP_H_CONFIGURATION,
    SD2_UNCLIP_L_CONFIGURATION, SD2_V_PREDICTION_THRESHOLD, Sd2ConditioningFact, Sd2Layout,
    Sd2ModelType, Sd2Variant, lotus_task_embedding, sd2_common_mapping,
    sd2_configuration_for_probe, sd2_state_plan_for_layout, sd2_weight_statistic_request_for_probe,
};
use sim_model::{
    model_family::ModelWeightStatisticObservation, model_store::ModelWeightStatisticError,
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceAuthority, DType, ExecutionContext, StreamId, Tensor, TensorBackend,
    TensorDescriptor, TensorError,
};
use sim_types::CancellationToken;
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
};

const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[test]
fn val_model_detection_001_sd2_lotus_unclip_profiles_and_precedence_are_exact()
-> Result<(), Box<dyn std::error::Error>> {
    let cases = [
        (None, Sd2Variant::Sd20, Sd2ModelType::Eps, None),
        (Some(4), Sd2Variant::LotusD, Sd2ModelType::ImgToImg, None),
        (
            Some(1_536),
            Sd2Variant::Sd21UnclipL,
            Sd2ModelType::Eps,
            Some(SD2_UNCLIP_L_CONFIGURATION),
        ),
        (
            Some(2_048),
            Sd2Variant::Sd21UnclipH,
            Sd2ModelType::Eps,
            Some(SD2_UNCLIP_H_CONFIGURATION),
        ),
    ];
    for layout in [Sd2Layout::PrefixedNative, Sd2Layout::Diffusers] {
        for (adm, variant, model_type, unclip) in cases {
            let configuration =
                sd2_configuration_for_probe(&standard_probe(layout, adm, false), None)?;
            assert_eq!(configuration.variant, variant);
            assert_eq!(configuration.layout, layout);
            assert_eq!(configuration.model_type, model_type);
            assert_eq!(configuration.unclip, unclip);
            assert_eq!(configuration.input_channels, 4);
            assert_eq!(configuration.output_channels, 4);
            assert_eq!(configuration.model_channels, 320);
            assert_eq!(configuration.context_dimension, 1_024);
            assert_eq!(configuration.attention_head_channels, 64);
            assert!(configuration.uses_linear_transformer_projection);
            assert!(!configuration.uses_temporal_attention);
            assert_eq!(configuration.memory_usage_factor, SD2_MEMORY_USAGE_FACTOR);
            assert_eq!(configuration.latent_format.feature_id, "COMFY-MODEL-0045");
            assert!(std::ptr::eq(configuration.clip_target, &SD2_CLIP_TARGET));
        }
    }

    let inpaint = sd2_configuration_for_probe(&standard_native_probe(None, false, 9), None)?;
    assert_eq!(inpaint.variant, Sd2Variant::Sd20);
    assert_eq!(inpaint.model_type, Sd2ModelType::Eps);
    assert_eq!(inpaint.input_channels, 9);
    assert!(
        inpaint
            .conditioning
            .contains(&Sd2ConditioningFact::InpaintLatentAndMask)
    );
    assert!(
        sd2_weight_statistic_request_for_probe(&standard_native_probe(Some(4), true, 4))?.is_none(),
        "Lotus precedence must bypass the inherited SD2 statistic"
    );
    assert_eq!(SD2_TRANSFORMER_DEPTH, [1, 1, 1, 1, 1, 1, 0, 0]);
    Ok(())
}

#[test]
fn val_model_detection_001_sd2_rejects_partial_mixed_cross_family_and_misleading_state() {
    let mut partial = standard_native_probe(None, false, 4);
    partial
        .tensor_shapes
        .remove("model.diffusion_model.time_embed.0.weight");
    assert!(matches!(
        sd2_configuration_for_probe(&partial, None),
        Err(ModelFamilyError::ModelLayoutSelection(_))
    ));

    let mut mixed = standard_native_probe(None, false, 4);
    mixed
        .tensor_shapes
        .extend(standard_diffusers_probe(None, false).tensor_shapes);
    assert!(matches!(
        sd2_configuration_for_probe(&mixed, None),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("ambiguously")
    ));

    let mut wrong_context = standard_native_probe(None, false, 4);
    wrong_context.tensor_shapes.insert(
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight".to_owned(),
        vec![320, 768],
    );
    assert_invalid(wrong_context, "context_dim");

    let mut wrong_depth = standard_diffusers_probe(None, false);
    wrong_depth
        .tensor_shapes
        .remove("down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k.weight");
    wrong_depth
        .tensor_shapes
        .remove("down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q.weight");
    assert_invalid(wrong_depth, "transformer_depth");

    assert_invalid(
        standard_native_probe(Some(512), false, 4),
        "unsupported ADM dimension",
    );
    assert_invalid(
        standard_native_probe(Some(4), false, 9),
        "unsupported ADM dimension",
    );

    let mut metadata_spoof = standard_native_probe(None, false, 4);
    metadata_spoof
        .metadata
        .insert("model_type".to_owned(), "v_prediction".to_owned());
    metadata_spoof
        .metadata
        .insert("population_standard_deviation".to_owned(), "999".to_owned());
    let configuration =
        sd2_configuration_for_probe(&metadata_spoof, None).expect("metadata cannot select V");
    assert_eq!(configuration.model_type, Sd2ModelType::Eps);

    let mut bad_diffusers = standard_diffusers_probe(None, false);
    bad_diffusers
        .tensor_shapes
        .insert("conv_in.weight".to_owned(), vec![320, 9, 3, 3]);
    assert_invalid(bad_diffusers, "exactly four input channels");
}

#[test]
fn val_model_family_row_001_sd2_native_diffusers_and_clip_plans_are_transactional()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );

    let native = ModelStateTransaction::new(&backend, &context).execute(
        &SD2_PREFIXED_STATE_PLAN.compile()?,
        DIGEST,
        &native_mapping_source(&backend, &context)?,
    )?;
    let denoiser = native.component("denoiser").ok_or("missing denoiser")?;
    for key in [
        "native.input_blocks.0.0.weight",
        "native.time_embed.0.weight",
        "native.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",
        "native.middle_block.1.transformer_blocks.0.attn2.to_q.weight",
        "native.out.2.weight",
    ] {
        assert!(denoiser.contains_key(key), "missing {key}");
    }
    let text = native
        .component("text_encoder")
        .ok_or("missing text encoder")?;
    for projection in ["q", "k", "v"] {
        let key = format!(
            "clip_h.transformer.text_model.encoder.layers.0.self_attn.{projection}_proj.weight"
        );
        assert_eq!(text.get(&key).ok_or(key)?.descriptor().shape(), &[2, 2]);
    }
    assert!(text.contains_key("clip_h.transformer.text_model.encoder.layers.0.layer_norm1.weight"));
    assert!(
        text.contains_key("clip_h.transformer.text_model.embeddings.position_embedding.weight")
    );
    assert!(text.contains_key("clip_h.transformer.text_projection.weight"));
    assert_eq!(
        native.component("vision_encoder").map(BTreeMap::len),
        Some(1)
    );
    assert_eq!(native.component("vae").map(BTreeMap::len), Some(1));

    let diffusers = ModelStateTransaction::new(&backend, &context).execute(
        &SD2_DIFFUSERS_STATE_PLAN.compile()?,
        DIGEST,
        &diffusers_mapping_source(&backend, &context)?,
    )?;
    let denoiser = diffusers
        .component("denoiser")
        .ok_or("missing Diffusers denoiser")?;
    for key in [
        "native.input_blocks.0.0.weight",
        "native.time_embed.0.weight",
        "native.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",
        "native.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
        "native.middle_block.1.transformer_blocks.0.attn2.to_q.weight",
        "native.output_blocks.11.1.transformer_blocks.0.norm1.bias",
        "native.out.2.weight",
    ] {
        assert!(denoiser.contains_key(key), "missing {key}");
    }
    assert!(denoiser.contains_key("native.diffusers.down_blocks.0.resnets.0.conv1.weight"));
    assert_eq!(
        sd2_state_plan_for_layout(Sd2Layout::PrefixedNative).encoded_plan,
        SD2_PREFIXED_STATE_PLAN.encoded_plan
    );
    assert_eq!(
        sd2_state_plan_for_layout(Sd2Layout::Diffusers).encoded_plan,
        SD2_DIFFUSERS_STATE_PLAN.encoded_plan
    );
    Ok(())
}

#[test]
fn val_memory_001_val_cancel_001_sd2_loaded_statistic_oom_and_cancellation_are_exact()
-> Result<(), Box<dyn std::error::Error>> {
    let probe = standard_native_probe(None, true, 4);
    let request = sd2_weight_statistic_request_for_probe(&probe)?
        .ok_or("canonical SD2 statistic request missing")?;
    assert_eq!(
        request.tensor_name(),
        "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm1.bias"
    );
    assert_eq!(SD2_V_PREDICTION_THRESHOLD, 0.09);
    let high = observe_statistic(&[-0.15, -0.05, 0.05, 0.15], 16)?;
    let configuration = sd2_configuration_for_probe(&probe, Some(&high))?;
    assert_eq!(configuration.model_type, Sd2ModelType::VPrediction);
    assert!(matches!(
        sd2_configuration_for_probe(&probe, None),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("observation is required")
    ));

    let low = observe_statistic(&[-0.03, -0.01, 0.01, 0.03], 16)?;
    assert_eq!(
        sd2_configuration_for_probe(&probe, Some(&low))?.model_type,
        Sd2ModelType::Eps
    );

    let oom = observe_statistic_result(&[-0.15, -0.05, 0.05, 0.15], 15)?;
    assert!(matches!(
        oom,
        Err(ModelWeightStatisticError::Tensor(
            TensorError::WorkspaceAuthorizationExceeded { .. }
        ))
    ));

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(512 * 1024)?,
        &cancellation,
    );
    let source = native_mapping_source(&backend, &context)?;
    let baseline = backend.memory_snapshot().current_bytes;
    cancellation.cancel();
    assert!(matches!(
        ModelStateTransaction::new(&backend, &context).execute(
            &SD2_PREFIXED_STATE_PLAN.compile()?,
            DIGEST,
            &source,
        ),
        Err(ModelFamilyError::Cancelled(_))
    ));
    assert_eq!(context.scratch.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);
    Ok(())
}

#[test]
fn val_ownership_001_sd2_has_one_adapter_and_canonical_foundational_owners()
-> Result<(), Box<dyn std::error::Error>> {
    assert!(std::ptr::eq(sd2_common_mapping(), &SD2_COMMON_MAPPING));
    assert_eq!(sd2_common_mapping().components.len(), 4);
    assert_eq!(sd2_common_mapping().component_state_schemas.len(), 4);
    assert_eq!(sd2_common_mapping().forward_program, SD2_FORWARD_PROGRAM);
    assert_eq!(
        sd2_common_mapping().latent_format.feature_id,
        "COMFY-MODEL-0045"
    );
    assert_eq!(SD2_LATENT_FORMAT.feature_id, "COMFY-MODEL-0045");
    assert_eq!(SD2_MEMORY_USAGE_FACTOR, 1.0);
    assert!(SD2_CONDITIONING.contains(&Sd2ConditioningFact::CrossAttention));
    let task = lotus_task_embedding();
    assert_eq!(task[0].to_bits(), 1.0_f32.sin().to_bits());
    assert_eq!(task[1].to_bits(), 0.0_f32.sin().to_bits());
    assert_eq!(task[2].to_bits(), 1.0_f32.cos().to_bits());
    assert_eq!(task[3].to_bits(), 0.0_f32.cos().to_bits());

    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repository_root = crate_root
        .parent()
        .and_then(Path::parent)
        .ok_or("sim_model crate is not inside the repository")?;
    let test_path = crate_root.join("tests/sd2_family_adapter.rs");
    let adapter_path = crate_root.join("src/sd2_family.rs");
    let foundation_path = crate_root.join("src/model_family.rs");
    let latent_path = crate_root.join("src/latent_formats/sd15_sim_model_0045.rs");
    let files = rust_files(repository_root)?;
    assert_eq!(
        files_containing(&files, "pub struct Sd2Configuration", &[&test_path])?,
        vec![adapter_path.clone()]
    );
    assert_eq!(
        latent_owner(&files, "COMFY-MODEL-0045", &test_path)?,
        vec![latent_path]
    );
    let transaction_declaration = ["pub struct Model", "StateTransaction"].concat();
    let transaction_owners = files_containing(&files, &transaction_declaration, &[&test_path])?
        .into_iter()
        .filter(|path| path.components().any(|part| part.as_os_str() == "src"))
        .collect::<Vec<_>>();
    assert_eq!(transaction_owners, vec![foundation_path]);
    let adapter = fs::read_to_string(adapter_path)?;
    for forbidden in [
        ["struct Model", "StateTransaction"].concat(),
        ["struct Patch", "Graph"].concat(),
        ["fn estimate_model", "_memory"].concat(),
        ["struct Cancellation", "Token"].concat(),
    ] {
        assert!(!adapter.contains(&forbidden));
    }
    Ok(())
}

fn standard_probe(layout: Sd2Layout, adm: Option<u64>, statistic: bool) -> ModelProbe {
    match layout {
        Sd2Layout::PrefixedNative => standard_native_probe(adm, statistic, 4),
        Sd2Layout::Diffusers => standard_diffusers_probe(adm, statistic),
    }
}

fn standard_native_probe(adm: Option<u64>, statistic: bool, input_channels: u64) -> ModelProbe {
    let prefix = "model.diffusion_model.";
    let mut tensors = BTreeMap::from([
        (
            format!("{prefix}input_blocks.0.0.weight"),
            vec![320, input_channels, 3, 3],
        ),
        (format!("{prefix}time_embed.0.weight"), vec![1_280, 320]),
        (format!("{prefix}out.2.weight"), vec![4, 320, 3, 3]),
        (
            format!("{prefix}middle_block.1.proj_in.weight"),
            vec![1_280, 1_280],
        ),
        (
            format!("{prefix}middle_block.1.transformer_blocks.0.attn2.to_k.weight"),
            vec![1_280, 1_024],
        ),
        (
            format!("{prefix}middle_block.1.transformer_blocks.0.attn2.to_q.weight"),
            vec![1_280, 1_280],
        ),
    ]);
    for index in [3, 6, 9] {
        tensors.insert(format!("{prefix}input_blocks.{index}.0.op.weight"), vec![1]);
    }
    for (index, channels, attention) in [
        (1, 320, true),
        (2, 320, true),
        (4, 640, true),
        (5, 640, true),
        (7, 1_280, true),
        (8, 1_280, true),
        (10, 1_280, false),
        (11, 1_280, false),
    ] {
        tensors.insert(
            format!("{prefix}input_blocks.{index}.0.in_layers.0.weight"),
            vec![channels],
        );
        tensors.insert(
            format!("{prefix}input_blocks.{index}.0.out_layers.3.weight"),
            vec![channels, channels, 3, 3],
        );
        if attention {
            tensors.insert(
                format!("{prefix}input_blocks.{index}.1.proj_in.weight"),
                vec![channels, channels],
            );
            tensors.insert(
                format!("{prefix}input_blocks.{index}.1.transformer_blocks.0.attn2.to_k.weight"),
                vec![channels, 1_024],
            );
            tensors.insert(
                format!("{prefix}input_blocks.{index}.1.transformer_blocks.0.attn1.to_q.weight"),
                vec![channels, channels],
            );
        }
    }
    for index in 0..12 {
        let channels = if index < 3 {
            320
        } else if index < 6 {
            640
        } else {
            1_280
        };
        tensors.insert(
            format!("{prefix}output_blocks.{index}.0.in_layers.0.weight"),
            vec![channels],
        );
        if index >= 3 {
            tensors.insert(
                format!("{prefix}output_blocks.{index}.1.proj_in.weight"),
                vec![channels, channels],
            );
            tensors.insert(
                format!("{prefix}output_blocks.{index}.1.transformer_blocks.0.attn2.to_k.weight"),
                vec![channels, 1_024],
            );
        }
    }
    if let Some(adm) = adm {
        tensors.insert(format!("{prefix}label_emb.0.0.weight"), vec![1_280, adm]);
    }
    if statistic {
        tensors.insert(
            format!("{prefix}output_blocks.11.1.transformer_blocks.0.norm1.bias"),
            vec![1_280],
        );
    }
    ModelProbe {
        tensor_shapes: tensors,
        metadata: BTreeMap::new(),
    }
}

fn standard_diffusers_probe(adm: Option<u64>, statistic: bool) -> ModelProbe {
    let mut tensors = BTreeMap::from([
        ("conv_in.weight".to_owned(), vec![320, 4, 3, 3]),
        (
            "time_embedding.linear_1.weight".to_owned(),
            vec![1_280, 320],
        ),
        ("conv_out.weight".to_owned(), vec![4, 320, 3, 3]),
        (
            "mid_block.attentions.0.transformer_blocks.0.attn2.to_q.weight".to_owned(),
            vec![1_280, 1_280],
        ),
    ]);
    for block in 0..4 {
        for residual in 0..2 {
            tensors.insert(
                format!("down_blocks.{block}.resnets.{residual}.conv1.weight"),
                vec![320, 320, 3, 3],
            );
        }
        if block < 3 {
            for attention in 0..2 {
                tensors.insert(
                    format!(
                        "down_blocks.{block}.attentions.{attention}.transformer_blocks.0.attn2.to_k.weight"
                    ),
                    vec![320, 1_024],
                );
                tensors.insert(
                    format!(
                        "down_blocks.{block}.attentions.{attention}.transformer_blocks.0.attn1.to_q.weight"
                    ),
                    vec![320, 320],
                );
            }
        }
    }
    if let Some(adm) = adm {
        tensors.insert(
            "class_embedding.linear_1.weight".to_owned(),
            vec![1_280, adm],
        );
    }
    if statistic {
        tensors.insert(
            "up_blocks.3.attentions.2.transformer_blocks.0.norm1.bias".to_owned(),
            vec![1_280],
        );
    }
    ModelProbe {
        tensor_shapes: tensors,
        metadata: BTreeMap::new(),
    }
}

fn native_mapping_source(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let keys = [
        ("model.diffusion_model.input_blocks.0.0.weight", vec![2, 2]),
        ("model.diffusion_model.time_embed.0.weight", vec![2, 2]),
        (
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",
            vec![2, 2],
        ),
        (
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
            vec![2, 2],
        ),
        (
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_q.weight",
            vec![2, 2],
        ),
        ("model.diffusion_model.out.2.weight", vec![2, 2]),
        (
            "cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight",
            vec![6, 2],
        ),
        (
            "cond_stage_model.model.transformer.resblocks.0.attn.in_proj_bias",
            vec![6],
        ),
        (
            "cond_stage_model.model.transformer.resblocks.0.ln_1.weight",
            vec![2],
        ),
        ("cond_stage_model.model.positional_embedding", vec![2, 2]),
        ("cond_stage_model.model.text_projection", vec![2, 2]),
        ("embedder.model.visual.proj.weight", vec![2, 2]),
        ("first_stage_model.decoder.weight", vec![2, 2]),
    ];
    keys.into_iter()
        .enumerate()
        .map(|(index, (key, shape))| {
            Ok((
                key.to_owned(),
                tensor(backend, context, &shape, index as f32 + 1.0)?,
            ))
        })
        .collect()
}

fn diffusers_mapping_source(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let keys = [
        "conv_in.weight",
        "time_embedding.linear_1.weight",
        "class_embedding.linear_1.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight",
        "down_blocks.0.resnets.0.conv1.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_q.weight",
        "up_blocks.3.attentions.2.transformer_blocks.0.norm1.bias",
        "conv_out.weight",
        "text_encoder.text_model.embeddings.token_embedding.weight",
        "image_encoder.visual_projection.weight",
        "vae.decoder.weight",
    ];
    keys.into_iter()
        .enumerate()
        .map(|(index, key)| {
            Ok((
                key.to_owned(),
                tensor(backend, context, &[2, 2], index as f32 + 1.0)?,
            ))
        })
        .collect()
}

fn tensor(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    shape: &[u64],
    value: f32,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let elements = usize::try_from(shape.iter().product::<u64>())?;
    Ok(backend
        .upload_f32(
            TensorDescriptor::contiguous(
                shape.to_vec(),
                DType::F32,
                backend.device(),
                StreamId::DEFAULT,
            )?,
            &vec![value; elements],
            context,
        )?
        .0)
}

fn observe_statistic(
    values: &[f32],
    workspace: usize,
) -> Result<ModelWeightStatisticObservation, Box<dyn std::error::Error>> {
    match observe_statistic_result(values, workspace)? {
        Ok(mut observations) => observations
            .pop()
            .ok_or_else(|| "missing statistic observation".into()),
        Err(error) => Err(Box::new(error)),
    }
}

fn observe_statistic_result(
    values: &[f32],
    workspace: usize,
) -> Result<
    Result<Vec<ModelWeightStatisticObservation>, ModelWeightStatisticError>,
    Box<dyn std::error::Error>,
> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("sd2-statistic.safetensors");
    let tensor_name = "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm1.bias";
    write_safetensors(&path, tensor_name, values)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "sd2-statistic",
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("sd2-statistic", "sd2-statistic.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    let request =
        sim_model::model_family::ModelWeightStatisticRequest::population_standard_deviation(
            tensor_name,
            sim_types::DeviceKind::Cpu,
        )?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(u64::try_from(workspace)?)?,
        &cancellation,
    );
    let result = store.observe_weight_statistics_with_context(
        &backend,
        &index,
        &loaded,
        &[request],
        &context,
    );
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(result)
}

fn write_safetensors(
    path: &Path,
    tensor_name: &str,
    values: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let data = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let header = serde_json::to_vec(&serde_json::json!({
        tensor_name: {
            "dtype": "F32",
            "shape": [values.len()],
            "data_offsets": [0, data.len()]
        }
    }))?;
    let mut file = fs::File::create(path)?;
    file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
    file.write_all(&header)?;
    file.write_all(&data)?;
    Ok(())
}

fn assert_invalid(probe: ModelProbe, expected: &str) {
    assert!(matches!(
        sd2_configuration_for_probe(&probe, None),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains(expected)
    ));
}

fn rust_files(root: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
    fn visit(path: &Path, output: &mut Vec<PathBuf>) -> Result<(), std::io::Error> {
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| matches!(name, ".git" | "target" | "projects"))
        {
            return Ok(());
        }
        if path.is_dir() {
            for entry in fs::read_dir(path)? {
                visit(&entry?.path(), output)?;
            }
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
            output.push(path.to_path_buf());
        }
        Ok(())
    }
    let mut output = Vec::new();
    visit(root, &mut output)?;
    output.sort();
    Ok(output)
}

fn files_containing(
    files: &[PathBuf],
    needle: &str,
    excluded: &[&Path],
) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut matches = files
        .iter()
        .filter(|path| !excluded.iter().any(|excluded| path == excluded))
        .filter_map(|path| {
            fs::read_to_string(path)
                .ok()
                .filter(|source| source.contains(needle))
                .map(|_| path.clone())
        })
        .collect::<Vec<_>>();
    matches.sort();
    Ok(matches)
}

fn latent_owner(
    files: &[PathBuf],
    feature_id: &str,
    excluded: &Path,
) -> Result<Vec<PathBuf>, std::io::Error> {
    Ok(files_containing(
        files,
        "pub const LATENT_FORMAT: LatentFormatDefinition",
        &[excluded],
    )?
    .into_iter()
    .filter(|path| {
        fs::read_to_string(path)
            .is_ok_and(|source| source.contains(&format!("feature_id: \"{feature_id}\"")))
    })
    .collect())
}
