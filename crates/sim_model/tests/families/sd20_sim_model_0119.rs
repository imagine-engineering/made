use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyRegistration, ModelFamilyRegistry, ModelProbe, ModelStateTransaction, ModelStore,
    NativeFamilyBuildOptions, ParserLimits, PatchApplication, PatchGraph, PatchKind,
    PatchOperation, PatchTarget, SD2_CLIP_TARGET, SD2_CONDITIONING, SD2_V_PREDICTION_THRESHOLD,
    Sd2ConditioningFact, Sd2Layout, Sd2ModelType, Sd2Variant, build_model_family_for_probe,
    describe_model_family, generated_sd20_sim_model_0119 as sd20,
    generated_sd21uncliph_sim_model_0120 as unclip_h,
    generated_sd21unclipl_sim_model_0121 as unclip_l,
    model_family::ModelWeightStatisticObservation,
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceAuthority, DType, ExecutionContext, StreamId, Tensor, TensorBackend,
    generated_sim_operator_indirection_01::{
        tensor_from_f32_with_context_exact_native, tensor_to_f32_with_context_exact_native,
    },
};
use sim_types::{CancellationToken, DeviceKind};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::Write, path::Path};

const ARTIFACT_DIGEST: &str =
    "1191191191191191191191191191191191191191191191191191191191191191";

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9119",
    identifier: "SD20AmbiguousFixture",
    ..sd20::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    sd20::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 9_119,
        source_architecture: "model_base.SD20AmbiguousFixture",
        ..sd20::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_sd20_source_configuration_detection_and_provenance()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(sd20::MODEL_FAMILY_IDENTIFIER, "SD20");
    assert_eq!(sd20::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0119");
    assert_eq!(sd20::MODEL_FAMILY_SOURCE_ORDINAL, 4);
    assert_eq!(sd20::MODEL_FAMILY_REGISTRATION.source_ordinal, 4);
    assert_eq!(
        sd20::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.BaseModel"
    );
    assert_eq!(sd20::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 1.0);
    let descriptor = describe_model_family(&sd20::MODEL_FAMILY)?;
    assert_eq!(descriptor.architecture_version, "sd20-native-v1");
    assert_eq!(descriptor.latent_format, "SD15");
    assert_eq!(
        descriptor.supported_dtypes,
        ["float16", "bfloat16", "float32"]
    );
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(descriptor.component_graph.len(), 4);

    for (probe, layout) in [
        (standard_native_probe(None, false, 4), Sd2Layout::PrefixedNative),
        (standard_diffusers_probe(None, false), Sd2Layout::Diffusers),
    ] {
        let configuration = sd20::configuration_for_probe(&probe, None)?;
        assert_eq!(configuration.variant, Sd2Variant::Sd20);
        assert_eq!(configuration.layout, layout);
        assert_eq!(configuration.model_type, Sd2ModelType::Eps);
        assert_eq!(configuration.input_channels, 4);
        assert_eq!(configuration.output_channels, 4);
        assert_eq!(configuration.model_channels, 320);
        assert_eq!(configuration.context_dimension, 1_024);
        assert_eq!(configuration.adm_in_channels, None);
        assert_eq!(configuration.attention_head_channels, 64);
        assert!(configuration.uses_linear_transformer_projection);
        assert!(!configuration.uses_temporal_attention);
        assert_eq!(configuration.memory_usage_factor, 1.0);
        assert!(std::ptr::eq(configuration.clip_target, &SD2_CLIP_TARGET));
        assert_eq!(configuration.latent_format.feature_id, "COMFY-MODEL-0045");
        assert!(
            configuration
                .conditioning
                .contains(&Sd2ConditioningFact::CrossAttention)
        );
        assert!(
            configuration
                .conditioning
                .contains(&Sd2ConditioningFact::InpaintLatentAndMask)
        );
        assert_eq!(configuration.conditioning, SD2_CONDITIONING);
    }
    let inpaint =
        sd20::configuration_for_probe(&standard_native_probe(None, false, 9), None)?;
    assert_eq!(inpaint.input_channels, 9);
    assert_eq!(inpaint.model_type, Sd2ModelType::Eps);

    let statistic_probe = standard_native_probe(None, true, 4);
    let request = sd20::weight_statistic_request_for_probe(&statistic_probe)?
        .ok_or("SD20 loaded-weight statistic request is missing")?;
    assert_eq!(
        request.tensor_name(),
        "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm1.bias"
    );
    assert_eq!(SD2_V_PREDICTION_THRESHOLD, 0.09);
    let high = observe_statistic(&[-0.15, -0.05, 0.05, 0.15])?;
    assert_eq!(
        sd20::configuration_for_probe(&statistic_probe, Some(&high))?.model_type,
        Sd2ModelType::VPrediction
    );
    let low = observe_statistic(&[-0.03, -0.01, 0.01, 0.03])?;
    assert_eq!(
        sd20::configuration_for_probe(&statistic_probe, Some(&low))?.model_type,
        Sd2ModelType::Eps
    );
    assert!(matches!(
        sd20::configuration_for_probe(&statistic_probe, None),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("observation is required")
    ));

    let registry = registry()?;
    for (probe, feature, score) in [
        (execution_probe(None), sd20::MODEL_FAMILY_FEATURE_ID, 1_200),
        (
            execution_probe(Some(unclip_l::SOURCE_ADM_IN_CHANNELS)),
            unclip_l::MODEL_FAMILY_FEATURE_ID,
            1_600,
        ),
        (
            execution_probe(Some(unclip_h::SOURCE_ADM_IN_CHANNELS)),
            unclip_h::MODEL_FAMILY_FEATURE_ID,
            1_600,
        ),
    ] {
        let resolved = registry.resolve(&probe)?;
        assert_eq!(resolved.detection().identity.feature_id(), feature);
        assert_eq!(resolved.detection().score, score);
        let clip = &resolved.clip_target().candidates()[0];
        assert_eq!(
            clip.tokenizer().identifier(),
            "sim.text_encoders.sd2_clip.SD2Tokenizer"
        );
        assert_eq!(
            clip.clip_model().target().as_str(),
            "sim.text_encoders.sd2_clip.SD2ClipModel"
        );
    }
    let store_probe = probe_through_model_store(None)?;
    assert_eq!(
        registry.resolve(&store_probe)?.detection().identity.feature_id(),
        sd20::MODEL_FAMILY_FEATURE_ID
    );

    let mut partial = execution_probe(None);
    partial
        .tensor_shapes
        .remove("model.diffusion_model.time_embed.0.weight");
    assert!(matches!(
        registry.resolve(&partial),
        Err(ModelFamilyError::ModelLayoutSelection(_))
    ));
    let mut malformed = execution_probe(None);
    malformed.tensor_shapes.insert(
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight"
            .to_owned(),
        vec![2, 768],
    );
    assert!(matches!(
        registry.detect(&malformed),
        Err(ModelFamilyError::NoDetectionMatch)
    ));
    let mut misleading = execution_probe(None);
    misleading
        .metadata
        .insert("model_family".to_owned(), "SD21UnclipH".to_owned());
    misleading
        .metadata
        .insert("model_type".to_owned(), "v_prediction".to_owned());
    assert_eq!(
        registry.resolve(&misleading)?.detection().identity.feature_id(),
        sd20::MODEL_FAMILY_FEATURE_ID
    );
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(
            &execution_probe(None)
        ),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_200, .. })
    ));

    validate_provenance_and_catalog(
        sd20::MODEL_FAMILY_FIXTURE,
        sd20::MODEL_FAMILY_IDENTIFIER,
        sd20::MODEL_FAMILY_FEATURE_ID,
        sd20::MODEL_FAMILY_SOURCE_ORDINAL,
        sd20::MODEL_FAMILY_PROJECTION_SHA256,
        "model_base.BaseModel",
        None,
        None,
    )?;
    let source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/families/sd20_sim_model_0119.rs"),
    )?;
    for delegation in [
        "SD2_COMPONENT_STATE_SCHEMAS",
        "SD2_FORWARD_PROGRAM",
        "SD2_LAYOUT_SIGNATURES",
        "SD2_PREFIXED_STATE_PLAN",
        "sd2_configuration_for_probe",
        "sd2_weight_statistic_request_for_probe",
    ] {
        assert!(source.contains(delegation), "missing shared owner {delegation}");
    }
    for forbidden in [
        "struct ModelStore",
        "struct ModelStateTransaction",
        "struct PatchGraph",
        "struct CancellationToken",
        "unsafe ",
        "std::process",
        "Command::new",
        "python",
    ] {
        assert!(!source.contains(forbidden), "row contains {forbidden}");
    }
    Ok(())
}

#[test]
fn val_model_family_row_001_sd20_mapping_forward_patch_dtype_memory_and_cancellation()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = registry()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(8 * 1024 * 1024)?,
        &cancellation,
    );

    for dtype in [DType::F32, DType::F16, DType::Bf16] {
        let source = native_source(&backend, &context, None, dtype)?;
        let probe = probe_from_source(&source);
        let resolved = registry.resolve(&probe)?;
        let components = resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &source,
        )?;
        let denoiser = components.component("denoiser").ok_or("missing denoiser")?;
        for key in sim_model::SD2_MODEL_REQUIRED_KEYS {
            assert!(denoiser.contains_key(*key), "missing {key}");
        }
        let text = components
            .component("text_encoder")
            .ok_or("missing text encoder")?;
        for projection in ["q", "k", "v"] {
            let key = format!(
                "clip_h.transformer.text_model.encoder.layers.0.self_attn.{projection}_proj.weight"
            );
            assert_eq!(text.get(&key).ok_or(key)?.descriptor().shape(), &[2, 2]);
        }
        assert_eq!(components.component("vision_encoder").map(BTreeMap::len), Some(1));
        assert_eq!(components.component("vae").map(BTreeMap::len), Some(1));

        let weights = resolved.map_primary_weights(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &source,
        )?;
        let model = build_model_family_for_probe(
            &registry,
            &probe,
            weights.clone(),
            options(dtype, DeviceKind::Cpu, 2 * 1024 * 1024),
        )?;
        assert!(model.memory_estimate().total_bytes > 0);
        if dtype == DType::F32 {
            let input = tensor(&backend, &context, &[1, 2], &[1.0, 2.0], dtype)?;
            let checkpoints = model.forward_checkpoints(&backend, &input, &context)?;
            assert_checkpoint(
                &backend,
                &context,
                &checkpoints,
                "latent_prediction",
                &[0.9934323, 0.6237125],
            )?;
            let patch = PatchGraph::checked(
                ARTIFACT_DIGEST,
                vec![PatchOperation {
                    identifier: "sd20-test-delta".to_owned(),
                    kind: PatchKind::Lora,
                    scale: 1.0,
                    targets: vec![PatchTarget {
                        key: "native.time_embed.0.weight".to_owned(),
                        expected_shape: vec![2, 2],
                        values: vec![1.0, 0.0, 0.0, 0.0],
                        application: PatchApplication::Add,
                    }],
                }],
            )?;
            let patched = model.with_weights(patch.apply(&backend, model.weights(), &context)?)?;
            assert_checkpoint(
                &backend,
                &context,
                &patched.forward_checkpoints(&backend, &input, &context)?,
                "latent_prediction",
                &[0.9999032, 0.9934323],
            )?;
        }
        assert!(matches!(
            build_model_family_for_probe(
                &registry,
                &probe,
                weights.clone(),
                options(dtype, DeviceKind::Cpu, 1),
            ),
            Err(ModelFamilyError::OutOfMemory { .. })
        ));
        assert!(matches!(
            build_model_family_for_probe(
                &registry,
                &probe,
                weights,
                options(dtype, DeviceKind::Metal, 2 * 1024 * 1024),
            ),
            Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
        ));
    }

    let source = native_source(&backend, &context, None, DType::F32)?;
    let probe = probe_from_source(&source);
    let resolved = registry.resolve(&probe)?;
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &cancelled,
    );
    assert!(matches!(
        resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &cancelled_context),
            ARTIFACT_DIGEST,
            &source,
        ),
        Err(ModelFamilyError::Cancelled(_))
    ));
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn generated_sd20_sim_model_0119_writes_validation_artifact()
-> Result<(), Box<dyn std::error::Error>> {
    super::write_model_family_row_artifact(
        sd20::MODEL_FAMILY_FIXTURE,
        sd20::MODEL_FAMILY_FEATURE_ID,
        sd20::MODEL_FAMILY_IDENTIFIER,
        sd20::MODEL_FAMILY_SOURCE_ORDINAL,
        "sd20_comfy_model_0119",
        &[
            "source-provenance-catalog-registration-descriptor",
            "native-diffusers-inpaint-and-loaded-statistic-configuration",
            "sd20-unclip-l-unclip-h-registry-precedence",
            "transactional-denoiser-openclip-vision-vae-mapping",
            "forward-patch-f16-bf16-f32-memory-device-cancellation",
            "partial-malformed-ambiguous-misleading-and-shared-owner-delegation",
        ],
    )?;
    Ok(())
}

pub(crate) fn registry() -> Result<ModelFamilyRegistry, ModelFamilyError> {
    ModelFamilyRegistry::checked_registrations(&[
        sd20::MODEL_FAMILY_REGISTRATION,
        unclip_l::MODEL_FAMILY_REGISTRATION,
        unclip_h::MODEL_FAMILY_REGISTRATION,
    ])
}

pub(crate) fn exercise_registered_runtime(
    adm: u64,
    expected_feature_id: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(32 * 1024 * 1024)?,
        &cancellation,
    );
    let source = native_source(&backend, &context, Some(adm), DType::F32)?;
    let probe = probe_from_source(&source);
    let resolved = registry.resolve(&probe)?;
    assert_eq!(
        resolved.detection().identity.feature_id(),
        expected_feature_id
    );
    let components = resolved.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    assert!(
        components
            .component("denoiser")
            .is_some_and(|component| component.contains_key("native.label_emb.0.0.weight"))
    );
    assert!(components.component("vision_encoder").is_some());
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    let model = build_model_family_for_probe(
        &registry,
        &probe,
        weights.clone(),
        options(DType::F32, DeviceKind::Cpu, 16 * 1024 * 1024),
    )?;
    let input = tensor(
        &backend,
        &context,
        &[1, 2],
        &[1.0, 2.0],
        DType::F32,
    )?;
    assert_checkpoint(
        &backend,
        &context,
        &model.forward_checkpoints(&backend, &input, &context)?,
        "latent_prediction",
        &[0.9934323, 0.6237125],
    )?;
    assert!(matches!(
        build_model_family_for_probe(
            &registry,
            &probe,
            weights,
            options(DType::F32, DeviceKind::Cpu, 1),
        ),
        Err(ModelFamilyError::OutOfMemory { .. })
    ));
    Ok(())
}

pub(crate) fn standard_native_probe(
    adm: Option<u64>,
    statistic: bool,
    input_channels: u64,
) -> ModelProbe {
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
                format!(
                    "{prefix}input_blocks.{index}.1.transformer_blocks.0.attn2.to_k.weight"
                ),
                vec![channels, 1_024],
            );
            tensors.insert(
                format!(
                    "{prefix}input_blocks.{index}.1.transformer_blocks.0.attn1.to_q.weight"
                ),
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
                format!(
                    "{prefix}output_blocks.{index}.1.transformer_blocks.0.attn2.to_k.weight"
                ),
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

pub(crate) fn standard_diffusers_probe(adm: Option<u64>, statistic: bool) -> ModelProbe {
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

pub(crate) fn execution_probe(adm: Option<u64>) -> ModelProbe {
    let mut tensor_shapes = BTreeMap::from([
        (
            "model.diffusion_model.input_blocks.0.0.weight".to_owned(),
            vec![320, 4, 3, 3],
        ),
        (
            "model.diffusion_model.time_embed.0.weight".to_owned(),
            vec![2, 2],
        ),
        (
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"
                .to_owned(),
            vec![2, 2],
        ),
        (
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight"
                .to_owned(),
            vec![2, 1_024],
        ),
        (
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_q.weight"
                .to_owned(),
            vec![2, 2],
        ),
        (
            "model.diffusion_model.out.2.weight".to_owned(),
            vec![4, 320, 3, 3],
        ),
    ]);
    if let Some(adm) = adm {
        tensor_shapes.insert(
            "model.diffusion_model.label_emb.0.0.weight".to_owned(),
            vec![1_280, adm],
        );
    }
    ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    }
}

pub(crate) fn validate_provenance_and_catalog(
    fixture: &str,
    symbol: &str,
    feature_id: &str,
    ordinal: u16,
    catalog_projection: &str,
    architecture: &str,
    adm: Option<u64>,
    timestep_dimension: Option<u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture_root = repository
        .join("crates/sim_test_support/fixtures/models")
        .join(fixture);
    let provenance: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_root.join("provenance.json"))?)?;
    assert_eq!(provenance["feature_id"], feature_id);
    assert_eq!(provenance["source_symbol"], symbol);
    assert_eq!(provenance["source_ordinal"], ordinal);
    assert_eq!(provenance["source_architecture"], architecture);
    assert_eq!(provenance["catalog_projection_sha256"], catalog_projection);
    let projection = provenance["source_projection"]
        .as_str()
        .ok_or("source projection is not text")?;
    assert_eq!(
        sha256(projection.as_bytes()),
        provenance["source_projection_sha256"]
    );
    for source in provenance["source_files"]
        .as_array()
        .ok_or("source files are not an array")?
    {
        let path = source["path"].as_str().ok_or("source path is not text")?;
        assert_eq!(sha256(&std::fs::read(repository.join(path))?), source["sha256"]);
    }
    let catalog: serde_json::Value = serde_json::from_slice(&std::fs::read(
        repository.join("crates/sim_model/catalog/model-families-v1.json"),
    )?)?;
    let row = catalog["models"]
        .as_array()
        .ok_or("catalog models are not an array")?
        .iter()
        .find(|row| row["feature_id"] == feature_id)
        .ok_or("catalog row is missing")?;
    assert_eq!(row["source_ordinal"], ordinal);
    assert_eq!(row["static"]["unet_config"]["value"]["context_dim"], 1_024);
    assert_eq!(row["static"]["unet_config"]["value"]["model_channels"], 320);
    match adm {
        Some(adm) => assert_eq!(row["static"]["unet_config"]["value"]["adm_in_channels"], adm),
        None => assert!(row["static"]["unet_config"]["value"]["adm_in_channels"].is_null()),
    }
    if let Some(timestep_dimension) = timestep_dimension {
        let projection = provenance["source_projection"]
            .as_str()
            .ok_or("source projection is not text")?;
        assert!(projection.contains(&format!("noise_aug_config.timestep_dim={timestep_dimension}")));
    }
    assert_eq!(sha256(&serde_json::to_vec(row)?), catalog_projection);
    Ok(())
}

fn native_source(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    adm: Option<u64>,
    dtype: DType,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let mut specifications = vec![
        (
            "model.diffusion_model.input_blocks.0.0.weight",
            vec![320, 4, 3, 3],
            Vec::new(),
        ),
        (
            "model.diffusion_model.time_embed.0.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "model.diffusion_model.time_embed.0.bias",
            vec![2],
            vec![0.0, 0.0],
        ),
        (
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",
            vec![2, 2],
            vec![2.0, 0.0, 0.0, 0.5],
        ),
        (
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
            vec![2, 1_024],
            Vec::new(),
        ),
        (
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_q.weight",
            vec![2, 2],
            vec![1.0, 1.0, 1.0, -1.0],
        ),
        (
            "model.diffusion_model.out.2.weight",
            vec![4, 320, 3, 3],
            Vec::new(),
        ),
        (
            "cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight",
            vec![6, 2],
            Vec::new(),
        ),
        (
            "cond_stage_model.model.transformer.resblocks.0.ln_1.weight",
            vec![2],
            Vec::new(),
        ),
        (
            "cond_stage_model.model.positional_embedding",
            vec![2, 2],
            Vec::new(),
        ),
        (
            "cond_stage_model.model.text_projection",
            vec![2, 2],
            Vec::new(),
        ),
        (
            "embedder.model.visual.proj.weight",
            vec![2, 2],
            Vec::new(),
        ),
        ("first_stage_model.decoder.weight", vec![2, 2], Vec::new()),
    ];
    if let Some(adm) = adm {
        specifications.push((
            "model.diffusion_model.label_emb.0.0.weight",
            vec![1_280, adm],
            Vec::new(),
        ));
    }
    specifications
        .into_iter()
        .enumerate()
        .map(|(index, (key, shape, values))| {
            let elements = usize::try_from(shape.iter().product::<u64>())?;
            let values = if values.is_empty() {
                vec![index as f32 / 100.0; elements]
            } else {
                values
            };
            Ok((
                key.to_owned(),
                tensor(backend, context, &shape, &values, dtype)?,
            ))
        })
        .collect()
}

fn probe_from_source(source: &BTreeMap<String, Tensor>) -> ModelProbe {
    ModelProbe {
        tensor_shapes: source
            .iter()
            .map(|(key, tensor)| (key.clone(), tensor.descriptor().shape().to_vec()))
            .collect(),
        metadata: BTreeMap::new(),
    }
}

pub(crate) fn probe_through_model_store(
    adm: Option<u64>,
) -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("sd2-row.safetensors");
    write_probe_safetensors(&path, &execution_probe(adm).tensor_shapes)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "sd2-row",
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("sd2-row", "sd2-row.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

pub(crate) fn observe_statistic(
    values: &[f32],
) -> Result<ModelWeightStatisticObservation, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("sd2-statistic.safetensors");
    let tensor_name =
        "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm1.bias";
    write_single_safetensors(&path, tensor_name, values)?;
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
    let request = sim_model::model_family::ModelWeightStatisticRequest::population_standard_deviation(
        tensor_name,
        DeviceKind::Cpu,
    )?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(16)?,
        &cancellation,
    );
    let mut observations = store.observe_weight_statistics_with_context(
        &backend,
        &index,
        &loaded,
        &[request],
        &context,
    )?;
    assert_eq!(context.scratch.in_use_bytes(), 0);
    observations
        .pop()
        .ok_or_else(|| "missing statistic observation".into())
}

fn write_probe_safetensors(
    path: &Path,
    shapes: &BTreeMap<String, Vec<u64>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut offset = 0_usize;
    let mut header = serde_json::Map::new();
    for (key, shape) in shapes {
        let elements = usize::try_from(shape.iter().product::<u64>())?;
        let bytes = elements.checked_mul(4).ok_or("safetensors size overflow")?;
        header.insert(
            key.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [offset, offset + bytes]
            }),
        );
        offset += bytes;
    }
    let header = serde_json::to_vec(&header)?;
    let mut file = std::fs::File::create(path)?;
    file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
    file.write_all(&header)?;
    file.write_all(&vec![0_u8; offset])?;
    Ok(())
}

fn write_single_safetensors(
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
    let mut file = std::fs::File::create(path)?;
    file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
    file.write_all(&header)?;
    file.write_all(&data)?;
    Ok(())
}

fn tensor(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    shape: &[u64],
    values: &[f32],
    dtype: DType,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(tensor_from_f32_with_context_exact_native(
        backend,
        shape,
        values,
        dtype,
        backend.device(),
        context,
    )?)
}

fn options(dtype: DType, device: DeviceKind, budget: u64) -> NativeFamilyBuildOptions {
    NativeFamilyBuildOptions {
        dtype,
        device,
        activation_elements: 2,
        memory_budget_bytes: budget,
        allow_unexpected_weights: false,
    }
}

fn assert_checkpoint(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    checkpoints: &[sim_model::ModelForwardCheckpoint],
    name: &str,
    expected: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = checkpoints
        .iter()
        .find(|checkpoint| checkpoint.name == name)
        .ok_or("checkpoint is missing")?;
    let actual = tensor_to_f32_with_context_exact_native(backend, &checkpoint.tensor, context)?;
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-5, "{actual} != {expected}");
    }
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
